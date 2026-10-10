"""Model builder factory for XGBoost and LightGBM models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from training.logging_config import get_logger, log_event

log = get_logger("model_builders")

SUPPORTED_MODELS = ("xgboost", "lightgbm")
DEFAULT_CONFIG_PATH = "configs/models.yml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load configuration from a YAML file (e.g. configs/models.yml)."""
    path = Path(config_path)
    if not path.is_file():
        # Fallback check relative to repo root
        root = Path(__file__).resolve().parents[2]
        candidates = [
            root / config_path,
            root / "configs/models.yml",
            root / "configs/model.yml",
        ]
        for cand in candidates:
            if cand.is_file():
                path = cand
                break
        else:
            log.warning(
                f"Configuration file '{config_path}' not found. Returning empty dict."
            )
            return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
            return content or {}
    except (OSError, yaml.YAMLError) as e:
        log.error(f"Failed to read configuration file '{path}': {e}")
        return {}


def calculate_scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    """Calculate scale_pos_weight to balance class weights in imbalanced binary classification.

    Formula: scale_pos_weight = count(negative) / count(positive)
    """
    y_arr = np.asarray(y).astype(int)
    pos_cnt = int(np.sum(y_arr == 1))
    neg_cnt = int(np.sum(y_arr == 0))
    if pos_cnt == 0:
        log.warning(
            "No positive samples in y when calculating scale_pos_weight. Defaulting to 1.0."
        )
        return 1.0

    weight = float(neg_cnt / pos_cnt)
    log_event(
        log,
        "calculate_scale_pos_weight",
        neg_count=neg_cnt,
        pos_count=pos_cnt,
        scale_pos_weight=round(weight, 4),
    )
    return weight


def _resolve_params(
    model_section: str,
    params: dict[str, Any] | None = None,
    y: pd.Series | np.ndarray | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load default parameters directly from YAML config and resolve scale_pos_weight."""
    cfg = load_config(config_path)
    base_params = cfg.get(model_section, {}).copy()

    if params:
        base_params.update(params)

    # Resolve scale_pos_weight if set to 'auto' or dynamically needed
    scale_weight = base_params.get("scale_pos_weight")
    if scale_weight in ("auto", "balanced", None) or isinstance(scale_weight, str):
        if y is not None:
            base_params["scale_pos_weight"] = calculate_scale_pos_weight(y)
        else:
            base_params["scale_pos_weight"] = 1.0
    elif isinstance(scale_weight, (int, float)):
        base_params["scale_pos_weight"] = float(scale_weight)

    return base_params


def build_xgboost(
    params: dict[str, Any] | None = None,
    y: pd.Series | np.ndarray | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> xgb.XGBClassifier:
    """Build and return an XGBoost classifier configured from configs/models.yml.

    Args:
        params: Optional dictionary of hyperparameters to override defaults.
        y: Optional target array to calculate scale_pos_weight automatically.
        config_path: Path to configuration YAML.

    Returns:
        Configured xgb.XGBClassifier instance.
    """
    resolved_params = _resolve_params(
        "xgboost", params=params, y=y, config_path=config_path
    )
    log_event(log, "build_model", model_type="xgboost", params=resolved_params)
    return xgb.XGBClassifier(**resolved_params)


def build_lightgbm(
    params: dict[str, Any] | None = None,
    y: pd.Series | np.ndarray | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> lgb.LGBMClassifier:
    """Build and return a LightGBM classifier configured from configs/models.yml.

    Args:
        params: Optional dictionary of hyperparameters to override defaults.
        y: Optional target array to calculate scale_pos_weight automatically.
        config_path: Path to configuration YAML.

    Returns:
        Configured lgb.LGBMClassifier instance.
    """
    resolved_params = _resolve_params(
        "lightgbm", params=params, y=y, config_path=config_path
    )
    log_event(log, "build_model", model_type="lightgbm", params=resolved_params)
    return lgb.LGBMClassifier(**resolved_params)


def build_model(
    model_name: str,
    params: dict[str, Any] | None = None,
    y: pd.Series | np.ndarray | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> xgb.XGBClassifier | lgb.LGBMClassifier:
    """Factory function to build a model by name using configs/models.yml defaults.

    Args:
        model_name: Either 'xgboost' or 'lightgbm'.
        params: Optional dictionary of hyperparameters.
        y: Optional target series to compute scale_pos_weight.
        config_path: Path to YAML configuration file.

    Returns:
        Initialized model instance.

    Raises:
        ValueError: If model_name is not supported.
    """
    normalized_name = model_name.strip().lower()
    if normalized_name == "xgboost":
        return build_xgboost(params=params, y=y, config_path=config_path)
    elif normalized_name == "lightgbm":
        return build_lightgbm(params=params, y=y, config_path=config_path)
    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models: {', '.join(SUPPORTED_MODELS)}"
        )
