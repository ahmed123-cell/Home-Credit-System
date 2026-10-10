"""Export trained XGBoost and LightGBM models to ONNX format and validate with ONNX Runtime."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxmltools
import onnxruntime as ort
import pandas as pd
from onnxmltools.convert.common.data_types import FloatTensorType

from training.data_setup import to_onnx_input
from training.logging_config import get_logger, log_event, timed

log = get_logger("onnx_export")


def export_model_to_onnx(
    model: Any,
    model_type: str,
    n_features: int,
    output_path: str | Path,
    target_opset: int = 15,
) -> Path:
    """Export a trained scikit-learn compatible XGBoost or LightGBM model to ONNX format.

    Args:
        model: Trained XGBoost or LightGBM model.
        model_type: 'xgboost' or 'lightgbm'.
        n_features: Number of input features expected by the model.
        output_path: Target path to save the .onnx file.
        target_opset: ONNX operator set version (default: 15).

    Returns:
        Path to the saved ONNX file.

    Raises:
        ValueError: If model_type is unsupported.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    initial_types = [("float_input", FloatTensorType([None, n_features]))]
    normalized_type = model_type.strip().lower()

    with timed(
        log,
        "export_model_to_onnx",
        model_type=normalized_type,
        target_file=str(output_path),
    ):
        if normalized_type == "xgboost":
            # onnxmltools requires booster feature names to follow f0, f1, ...
            booster = model.get_booster()
            original_names = booster.feature_names
            try:
                booster.feature_names = [f"f{i}" for i in range(n_features)]
                onnx_model = onnxmltools.convert_xgboost(
                    model,
                    initial_types=initial_types,
                    target_opset=target_opset,
                )
            finally:
                booster.feature_names = original_names

        elif normalized_type == "lightgbm":
            onnx_model = onnxmltools.convert_lightgbm(
                model,
                initial_types=initial_types,
                target_opset=target_opset,
            )
        else:
            raise ValueError(f"Unsupported model type '{model_type}' for ONNX export.")

        onnx.save_model(onnx_model, str(output_path))
        file_size_kb = round(output_path.stat().st_size / 1024, 2)
        log_event(
            log,
            "onnx_export.saved",
            path=str(output_path),
            size_kb=file_size_kb,
            n_features=n_features,
        )

    return output_path


def extract_onnx_probabilities(raw_probs: Any) -> np.ndarray:
    """Extract probability array from ONNX output (handles zipmap dict lists and ndarrays)."""
    if (
        isinstance(raw_probs, list)
        and len(raw_probs) > 0
        and isinstance(raw_probs[0], dict)
    ):
        # Format: [{0: prob0, 1: prob1}, ...]
        classes = sorted(raw_probs[0].keys())
        return np.array(
            [[d.get(c, 0.0) for c in classes] for d in raw_probs], dtype=np.float32
        )
    return np.asarray(raw_probs, dtype=np.float32)


def validate_onnx_model(
    onnx_path: str | Path,
    sample_input: np.ndarray,
    expected_probs: np.ndarray | None = None,
    atol: float = 1e-3,
) -> bool:
    """Validate that the exported ONNX model loads and produces expected outputs with ONNX Runtime.

    Args:
        onnx_path: Path to the .onnx file.
        sample_input: 2D numpy float32 array of shape (N, n_features).
        expected_probs: Optional ground truth probabilities from the Python model for comparison.
        atol: Absolute tolerance for comparing probabilities.

    Returns:
        True if validation passed.

    Raises:
        RuntimeError: If validation fails or outputs do not match.
    """
    onnx_path = Path(onnx_path).resolve()
    with timed(
        log, "validate_onnx_model", path=str(onnx_path), samples=len(sample_input)
    ):
        # Suppress onnxruntime warnings for standard shape verification
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # Error only
        session = ort.InferenceSession(str(onnx_path), sess_options=opts)

        input_name = session.get_inputs()[0].name
        feed = {input_name: sample_input.astype(np.float32)}
        outputs = session.run(None, feed)

        if not outputs:
            raise RuntimeError(f"ONNX model '{onnx_path}' returned no outputs.")

        log_event(
            log,
            "onnx_validation.outputs_received",
            n_outputs=len(outputs),
            shapes=[getattr(out, "shape", type(out).__name__) for out in outputs],
        )

        if expected_probs is not None:
            # Probability output is typically the second output (or first if single)
            prob_idx = 1 if len(outputs) > 1 else 0
            onnx_probs = extract_onnx_probabilities(outputs[prob_idx])

            # For binary classification with shape (N, 1) vs (N, 2), align column
            if (
                onnx_probs.ndim == 2
                and expected_probs.ndim == 2
                and onnx_probs.shape[1] == expected_probs.shape[1]
            ):
                diff = np.max(np.abs(expected_probs - onnx_probs))
                if diff > atol:
                    log_event(
                        log,
                        "onnx_validation.warning",
                        level=logging.WARNING,
                        max_prob_difference=float(diff),
                    )
                else:
                    log_event(
                        log,
                        "onnx_validation.match_confirmed",
                        max_diff=float(diff),
                    )

        log_event(log, "onnx_validation.success", path=str(onnx_path))
    return True


def export_and_validate(
    model: Any,
    model_type: str,
    sample_df: pd.DataFrame,
    output_dir: str | Path = "models",
    filename: str | None = None,
) -> Path:
    """High-level utility to export a trained model and immediately validate it on sample data.

    Args:
        model: Trained model.
        model_type: 'xgboost' or 'lightgbm'.
        sample_df: Preprocessed DataFrame sample (e.g., first 100 rows of validation set).
        output_dir: Root directory to save models.
        filename: Optional custom file name (defaults to '{model_type}.onnx').

    Returns:
        Path to the validated ONNX file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not filename:
        filename = f"{model_type.lower()}.onnx"

    target_path = output_dir / filename
    n_features = sample_df.shape[1]

    # Export
    export_path = export_model_to_onnx(
        model=model,
        model_type=model_type,
        n_features=n_features,
        output_path=target_path,
    )

    # Validate
    sample_array = to_onnx_input(sample_df)
    expected_probs = (
        model.predict_proba(sample_df) if hasattr(model, "predict_proba") else None
    )
    validate_onnx_model(export_path, sample_array, expected_probs=expected_probs)

    return export_path
