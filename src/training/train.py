"""Main training entry point for XGBoost and LightGBM models.

Features:
- Config loading from YAML (configs/models.yml) with CLI override support.
- Fast Parquet caching and incremental loading via `data_setup.load_data()`.
- Model building via `model_builders.build_model()` with dynamic `scale_pos_weight`.
- Full evaluation (ROC-AUC, PR-AUC, Gini, Log Loss, Accuracy, F1, etc.).
- MLflow experiment tracking and model registration connected to SQLite (mlflow.db).
- Incremental warm-start training support (`--init-model` / `--warm-start`).
- Export and validation of the best model to ONNX format.

Usage:
    uv run python -m training.train --model lightgbm
    uv run python -m training.train --model xgboost --parameters '{"n_estimators": 200, "learning_rate": 0.05}'
    uv run python -m training.train --model lightgbm --init-model models/lightgbm.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.lightgbm
import mlflow.onnx
import mlflow.xgboost
import numpy as np
import onnx
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from training.data_setup import DataBundle, load_data
from training.logging_config import configure_logging, get_logger, log_event, timed
from training.model_builders import SUPPORTED_MODELS, build_model, load_config
from training.onnx_export import export_and_validate

load_dotenv()
log = get_logger("train")

DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)


def generate_experiment_name(prefix: str = "home_credit") -> str:
    """Generate an experiment name formatted like: home_credit_year_month_day_hour:minute:second.

    Example: home_credit_26_10_9_5:19:44
    """
    now = datetime.now(tz=UTC).astimezone()
    return f"{prefix}_{now.strftime('%y')}_{now.month}_{now.day}_{now.hour}:{now.minute:02d}:{now.second:02d}"


def resolve_init_model_path(path: str | Path | None, model_type: str) -> str | None:
    """Resolve warm-start / init-model checkpoint path for the given model framework.

    If an ONNX file or extension-less name is provided, resolves to the corresponding
    native booster checkpoint (.txt for LightGBM, .json for XGBoost).
    """
    if not path:
        return None

    p = Path(path)
    native_ext = ".txt" if model_type == "lightgbm" else ".json"

    # If an .onnx file path was passed, check for the companion native checkpoint
    if p.suffix == ".onnx" or not p.suffix:
        candidate = p.with_suffix(native_ext)
        if candidate.is_file():
            log_event(
                log,
                "warm_start.resolved_native_checkpoint",
                original=str(p),
                resolved=str(candidate),
            )
            return str(candidate)
        elif p.suffix == ".onnx":
            log.warning(
                f"ONNX files ({p}) cannot be used directly for {model_type} warm-start training. "
                f"Native checkpoint '{candidate}' was not found. Proceeding with fresh training."
            )
            return None

    if p.is_file():
        return str(p)

    log.warning(
        f"Specified init_model path '{p}' does not exist. Proceeding with fresh training."
    )
    return None


def parse_parameters_arg(param_str: str | None) -> dict[str, Any]:
    """Parse parameters passed via CLI as a JSON string or comma-separated key=value pairs.

    Examples:
        '{"n_estimators": 200, "learning_rate": 0.05}'
        'n_estimators=200,learning_rate=0.05,max_depth=6'
    """
    if not param_str:
        return {}

    param_str = param_str.strip()
    if not param_str:
        return {}

    # Try JSON format first
    if (param_str.startswith("{") and param_str.endswith("}")) or (
        param_str.startswith('"') and param_str.endswith('"')
    ):
        try:
            return json.loads(param_str)
        except json.JSONDecodeError as e:
            log.warning(
                f"Could not parse JSON parameters string '{param_str}': {e}. Falling back to key=value parser."
            )

    # Parse comma-separated key=value pairs
    params: dict[str, Any] = {}
    for item in param_str.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, val = item.split("=", 1)
        key, val = key.strip(), val.strip()

        # Type conversion
        if val.lower() == "true":
            parsed_val: Any = True
        elif val.lower() == "false":
            parsed_val = False
        elif val.lower() in ("none", "null"):
            parsed_val = None
        else:
            try:
                parsed_val = int(val)
            except ValueError:
                try:
                    parsed_val = float(val)
                except ValueError:
                    parsed_val = val
        params[key] = parsed_val

    return params


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    prefix: str = "val_",
) -> dict[str, float]:
    """Compute comprehensive classification evaluation metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (y_prob >= threshold).astype(int)

    auc = float(roc_auc_score(y_true, y_prob))
    gini = float(2 * auc - 1)
    pr_auc = float(average_precision_score(y_true, y_prob))
    loss = float(log_loss(y_true, y_prob))
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    return {
        f"{prefix}roc_auc": round(auc, 6),
        f"{prefix}gini": round(gini, 6),
        f"{prefix}pr_auc": round(pr_auc, 6),
        f"{prefix}log_loss": round(loss, 6),
        f"{prefix}accuracy": round(acc, 6),
        f"{prefix}f1": round(f1, 6),
        f"{prefix}precision": round(prec, 6),
        f"{prefix}recall": round(rec, 6),
    }


def get_feature_importances(model: Any, feature_names: list[str]) -> pd.DataFrame:
    """Extract and sort feature importances from a trained model."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None or len(importances) != len(feature_names):
        return pd.DataFrame(columns=["feature", "importance"])

    df_imp = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        )
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )
    return df_imp


def train(
    model_type: str,
    parameters: dict[str, Any] | None = None,
    config_path: str = "configs/models.yml",
    output_dir: str = "models",
    experiment_name: str | None = None,
    run_name: str | None = None,
    data_bundle: DataBundle | None = None,
    scale_features: bool | None = None,
    tracking_uri: str | None = None,
    force_refresh: bool = False,
    init_model_path: str | Path | None = None,
) -> tuple[Any, dict[str, float], Path]:
    """Train, evaluate, log to MLflow as a new experiment, register model, and export to ONNX.

    Args:
        model_type: 'xgboost' or 'lightgbm'.
        parameters: Optional hyperparameter overrides.
        config_path: Path to YAML config file.
        output_dir: Root directory to save models.
        experiment_name: Optional explicit MLflow experiment name (defaults to timestamp format).
        run_name: Optional MLflow run name.
        data_bundle: Optional pre-loaded DataBundle.
        scale_features: Whether to scale features with StandardScaler.
        tracking_uri: Optional MLflow tracking URI (defaults to sqlite:///mlflow.db).
        force_refresh: Whether to bypass parquet cache and re-query PostgreSQL.
        init_model_path: Optional path to existing model checkpoint for warm-start training.

    Returns:
        Tuple of (trained_model, metrics_dict, onnx_model_path).
    """
    model_type = model_type.strip().lower()
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"Invalid model '{model_type}'. Supported models: {', '.join(SUPPORTED_MODELS)}"
        )

    # Set MLflow tracking URI
    mlflow_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(mlflow_uri)

    # 1. Load configuration
    cfg = load_config(config_path)
    training_cfg = cfg.get("training", {})
    model_defaults = cfg.get(model_type, {}).copy()

    # Merge hyperparameters
    merged_model_params = model_defaults
    if parameters:
        merged_model_params.update(parameters)

    # Training settings
    val_size = float(training_cfg.get("val_size", 0.2))
    random_state = int(training_cfg.get("random_state", 42))
    stratify = bool(training_cfg.get("stratify", True))
    should_scale = (
        scale_features
        if scale_features is not None
        else bool(training_cfg.get("scale_features", True))
    )

    # MLflow experiment naming: generate a fresh timestamped experiment if not specified
    mlflow_exp = experiment_name or generate_experiment_name(prefix="home_credit")

    out_dir = Path(output_dir or training_cfg.get("models_dir", "models"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve warm-start checkpoint path if provided
    resolved_init_model = resolve_init_model_path(
        init_model_path, model_type=model_type
    )

    log_event(
        log,
        "training.start",
        model_type=model_type,
        experiment=mlflow_exp,
        tracking_uri=mlflow_uri,
        scale_features=should_scale,
        force_refresh=force_refresh,
        warm_start=bool(resolved_init_model),
        params=merged_model_params,
    )

    # 2. Load and preprocess Data (using instant Parquet cache unless force_refresh=True)
    if data_bundle is None:
        data_bundle = load_data(
            scale_features=should_scale, force_refresh=force_refresh
        )

    X_train_full = data_bundle.X_train
    y_train_full = data_bundle.y_train

    # 3. Stratified Train / Validation Split
    with timed(
        log, "train_val_split", val_size=val_size, random_state=random_state
    ) as ctx:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=val_size,
            random_state=random_state,
            stratify=y_train_full if stratify else None,
        )
        ctx.update(
            train_samples=len(X_tr), val_samples=len(X_val), features=X_tr.shape[1]
        )

    # 4. Instantiate Model with resolved scale_pos_weight based on train targets
    model = build_model(
        model_name=model_type,
        params=merged_model_params,
        y=y_tr,
        config_path=config_path,
    )

    # 5. Fit Model with evaluation set (and optional warm-start incremental training)
    with timed(
        log,
        "fit_model",
        model_type=model_type,
        n_train=len(X_tr),
        warm_start=bool(resolved_init_model),
    ) as ctx:
        if model_type == "xgboost":
            fit_kwargs: dict[str, Any] = {
                "eval_set": [(X_tr, y_tr), (X_val, y_val)],
                "verbose": False,
            }
            if resolved_init_model:
                fit_kwargs["xgb_model"] = str(resolved_init_model)
            model.fit(X_tr, y_tr, **fit_kwargs)

        elif model_type == "lightgbm":
            fit_kwargs = {
                "eval_set": [(X_tr, y_tr), (X_val, y_val)],
                "eval_names": ["train", "val"],
            }
            if resolved_init_model:
                fit_kwargs["init_model"] = str(resolved_init_model)
            model.fit(X_tr, y_tr, **fit_kwargs)

    # 6. Evaluation
    with timed(log, "evaluate_model") as ctx:
        val_probs = model.predict_proba(X_val)[:, 1]
        train_probs = model.predict_proba(X_tr)[:, 1]

        val_metrics = compute_metrics(y_val, val_probs, prefix="val_")
        train_metrics = compute_metrics(y_tr, train_probs, prefix="train_")
        all_metrics = {**train_metrics, **val_metrics}
        ctx.update(**all_metrics)

    # 7. Extract Feature Importance
    feat_imp = get_feature_importances(model, list(X_train_full.columns))

    # 8. Export to ONNX and Validate
    sample_df = X_val.iloc[:100].copy()
    onnx_filename = f"{model_type}.onnx"
    onnx_path = export_and_validate(
        model=model,
        model_type=model_type,
        sample_df=sample_df,
        output_dir=out_dir,
        filename=onnx_filename,
    )

    # Save native booster checkpoint for subsequent warm-start incremental training
    if model_type == "lightgbm" and hasattr(model, "booster_"):
        native_model_path = out_dir / f"{model_type}.txt"
        model.booster_.save_model(str(native_model_path))
    elif model_type == "xgboost" and hasattr(model, "save_model"):
        native_model_path = out_dir / f"{model_type}.json"
        model.save_model(str(native_model_path))

    # Also save metadata JSON
    meta_path = out_dir / f"{model_type}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": model_type,
                "experiment_name": mlflow_exp,
                "metrics": all_metrics,
                "parameters": getattr(
                    model, "get_params", lambda: merged_model_params
                )(),
                "n_features": X_train_full.shape[1],
                "feature_names": list(X_train_full.columns),
                "onnx_path": str(onnx_path),
            },
            f,
            indent=2,
            default=str,
        )

    # 9. Log everything to MLflow under the new experiment and register model
    with timed(log, "mlflow_logging", experiment=mlflow_exp):
        mlflow.set_experiment(mlflow_exp)
        run_timestamp = datetime.now(tz=UTC).astimezone().strftime("%H%M%S")
        with mlflow.start_run(
            run_name=run_name or f"{model_type}_{run_timestamp}"
        ) as run:
            # Log params
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("val_size", val_size)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("scale_features", should_scale)
            mlflow.log_param("n_features", X_train_full.shape[1])

            model_params = getattr(model, "get_params", lambda: merged_model_params)()
            for k, v in model_params.items():
                mlflow.log_param(k, str(v) if isinstance(v, (list, dict)) else v)

            # Log metrics
            mlflow.log_metrics(all_metrics)

            # Log tags
            mlflow.set_tags(
                {
                    "model_type": model_type,
                    "experiment_name": mlflow_exp,
                    "best_metric": "val_roc_auc",
                    "val_roc_auc": str(all_metrics.get("val_roc_auc")),
                }
            )

            # Log native model and register in MLflow Model Registry
            registered_name = f"home_credit_{model_type}"
            if model_type == "xgboost":
                mlflow.xgboost.log_model(
                    xgb_model=model,
                    name="model",
                    registered_model_name=registered_name,
                )
            elif model_type == "lightgbm":
                mlflow.lightgbm.log_model(
                    lgb_model=model,
                    name="model",
                    registered_model_name=registered_name,
                    skops_trusted_types=["lightgbm.basic._InnerPredictor"],
                )

            # Log ONNX model and register in MLflow Model Registry
            onnx_proto = onnx.load_model(str(onnx_path))
            mlflow.onnx.log_model(
                onnx_model=onnx_proto,
                name="onnx_model",
                registered_model_name=f"home_credit_{model_type}_onnx",
            )

            # Log artifacts (standalone ONNX, metadata, feature importance CSV)
            mlflow.log_artifact(str(onnx_path), artifact_path="model_files")
            mlflow.log_artifact(str(meta_path), artifact_path="metadata")

            imp_csv = out_dir / f"{model_type}_feature_importance.csv"
            feat_imp.to_csv(imp_csv, index=False)
            mlflow.log_artifact(str(imp_csv), artifact_path="explainability")

            log_event(
                log,
                "mlflow_run.completed",
                run_id=run.info.run_id,
                experiment_id=run.info.experiment_id,
                experiment_name=mlflow_exp,
                registered_model=registered_name,
                val_roc_auc=all_metrics.get("val_roc_auc"),
            )

    print("\n" + "=" * 60)
    print(f"Training Complete for Model: {model_type.upper()}")
    print(f"MLflow Experiment: {mlflow_exp}")
    print(f"Registered Models: {registered_name}, home_credit_{model_type}_onnx")
    print("=" * 60)
    print(f"  Train ROC-AUC : {all_metrics.get('train_roc_auc'):.5f}")
    print(f"  Val   ROC-AUC : {all_metrics.get('val_roc_auc'):.5f}")
    print(f"  Val   PR-AUC  : {all_metrics.get('val_pr_auc'):.5f}")
    print(f"  Val   Gini    : {all_metrics.get('val_gini'):.5f}")
    print(f"  Val   LogLoss : {all_metrics.get('val_log_loss'):.5f}")
    print(f"  ONNX Export   : {onnx_path}")
    print(f"  Top 5 Features:\n{feat_imp.head(5).to_string(index=False)}")
    print("=" * 60 + "\n")

    return model, all_metrics, onnx_path


def main() -> None:
    """CLI parser and entry point."""
    parser = argparse.ArgumentParser(
        description="Home Credit Default Risk - Model Training and ONNX Export Pipeline"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        choices=SUPPORTED_MODELS,
        help="Machine learning model architecture to train ('xgboost' or 'lightgbm').",
    )
    parser.add_argument(
        "--parameters",
        "-p",
        type=str,
        default=None,
        help="Hyperparameter overrides as JSON string or comma-separated key=value pairs.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/models.yml",
        help="Path to YAML configuration file (default: configs/models.yml).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="models",
        help="Directory to save exported ONNX models and artifacts (default: models).",
    )
    parser.add_argument(
        "--experiment-name",
        "-e",
        type=str,
        default=None,
        help="Custom MLflow experiment name (defaults to 'home_credit_YY_MM_DD_H:M:S').",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional custom name for the MLflow run.",
    )
    parser.add_argument(
        "--scale-features",
        dest="scale_features",
        action="store_true",
        default=None,
        help="Enable StandardScaler feature scaling on numerical features.",
    )
    parser.add_argument(
        "--no-scale-features",
        dest="scale_features",
        action="store_false",
        help="Disable feature scaling.",
    )
    parser.add_argument(
        "--force-refresh",
        "--refresh-data",
        dest="force_refresh",
        action="store_true",
        default=False,
        help="Force re-querying PostgreSQL and re-building parquet cache.",
    )
    parser.add_argument(
        "--init-model",
        "--warm-start",
        dest="init_model_path",
        type=str,
        default=None,
        help="Path to previous model checkpoint for incremental continuation training.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: from .env or sqlite:///mlflow.db).",
    )
    parser.add_argument(
        "--log-format",
        type=str,
        choices=["console", "json"],
        default=None,
        help="Logging output format ('console' or 'json').",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging level.",
    )

    args, unknown = parser.parse_known_args()

    configure_logging(log_format=args.log_format, log_level=args.log_level)

    cli_params = parse_parameters_arg(args.parameters)
    for i in range(0, len(unknown), 2):
        flag = unknown[i]
        if flag.startswith("--") and i + 1 < len(unknown):
            key = flag.lstrip("-").replace("-", "_")
            val = unknown[i + 1]
            try:
                cli_params[key] = int(val)
            except ValueError:
                try:
                    cli_params[key] = float(val)
                except ValueError:
                    cli_params[key] = val

    try:
        train(
            model_type=args.model,
            parameters=cli_params,
            config_path=args.config,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
            scale_features=args.scale_features,
            tracking_uri=args.tracking_uri,
            force_refresh=args.force_refresh,
            init_model_path=args.init_model_path,
        )
    except Exception:
        log.exception("Training failed with error")
        sys.exit(1)


if __name__ == "__main__":
    main()
