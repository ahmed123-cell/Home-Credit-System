"""FastAPI Inference Service for Home Credit Default Risk.

Features:
- Lifespan context manager to load the ONNX model and metadata ONCE at startup.
- Single prediction endpoint (`POST /predict`).
- Batch prediction endpoint (`POST /predict/batch`).
- Health and metadata endpoint (`GET /health`).
- High-precision latency measurement per request and per sample.

Run with:
    uv run uvicorn app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------------
# Lifespan: Load Model and Metadata ONCE at Application Startup
# ----------------------------------------------------------------------------
MODEL_DIR = Path(os.getenv("MODELS_DIR", "models"))
DEFAULT_MODEL_PATH = MODEL_DIR / "lightgbm.onnx"
DEFAULT_META_PATH = MODEL_DIR / "lightgbm_meta.json"


def load_model_resources() -> dict[str, Any]:
    """Load ONNX model session and metadata once into memory."""
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    meta_path = Path(os.getenv("MODEL_META_PATH", str(DEFAULT_META_PATH)))

    # Fallback to xgboost if lightgbm is missing
    if not model_path.is_file():
        alt_model = MODEL_DIR / "xgboost.onnx"
        alt_meta = MODEL_DIR / "xgboost_meta.json"
        if alt_model.is_file():
            model_path = alt_model
            meta_path = alt_meta
        else:
            raise FileNotFoundError(
                f"No ONNX model found at '{model_path}' or '{alt_model}'. Train a model first with `python -m training.train`."
            )

    # Load metadata
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    feature_names = meta.get("feature_names", [])

    # Configure high-performance ONNX Runtime Session
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, os.cpu_count() or 4)
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3  # Error only

    session = ort.InferenceSession(str(model_path), sess_options=opts)
    input_name = session.get_inputs()[0].name
    expected_dim = (
        session.get_inputs()[0].shape[1]
        if len(session.get_inputs()[0].shape) > 1
        else len(feature_names)
    )

    return {
        "session": session,
        "input_name": input_name,
        "feature_names": feature_names,
        "n_features": expected_dim or len(feature_names),
        "model_type": meta.get("model_type", "unknown"),
        "model_path": str(model_path),
        "metrics": meta.get("metrics", {}),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager that loads the model once at startup and releases resources on shutdown."""
    resources = load_model_resources()
    app.state.session = resources["session"]
    app.state.input_name = resources["input_name"]
    app.state.feature_names = resources["feature_names"]
    app.state.n_features = resources["n_features"]
    app.state.model_type = resources["model_type"]
    app.state.model_path = resources["model_path"]
    app.state.metrics = resources["metrics"]
    app.state.startup_time = time.time()

    print(
        f"[Startup] Model loaded successfully: {resources['model_type'].upper()} from {resources['model_path']} "
        f"({resources['n_features']} features)"
    )

    yield

    # Teardown / cleanup
    print("[Shutdown] Releasing model resources.")
    app.state.session = None


# ----------------------------------------------------------------------------
# FastAPI Application & Timing Middleware
# ----------------------------------------------------------------------------
app = FastAPI(
    title="Home Credit Default Risk Inference API",
    description="High-performance real-time credit default prediction powered by ONNX Runtime and FastAPI.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Measure total request-response roundtrip latency and attach X-Process-Time-Ms header."""
    start_time = time.perf_counter()
    response: Response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.3f}"
    return response


# ----------------------------------------------------------------------------
# Pydantic Schemas
# ----------------------------------------------------------------------------
class SinglePredictionRequest(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="Feature dictionary mapping feature names to numerical/categorical values.",
        examples=[
            {
                "amt_credit": 406597.5,
                "amt_income_total": 202500.0,
                "ext_source_mean": 0.45,
                "ext_source_weighted": 0.52,
                "credit_term": 16.4,
                "age_years": 42.5,
            }
        ],
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Classification decision threshold (default: 0.5).",
    )


class SinglePredictionResponse(BaseModel):
    probability: float = Field(
        ..., description="Predicted probability of default (0.0 to 1.0)."
    )
    prediction: int = Field(
        ..., description="Binary prediction: 1 = Default, 0 = Non-Default."
    )
    risk_tier: str = Field(
        ..., description="Qualitative risk tier (e.g. Low, Medium, High)."
    )
    inference_time_ms: float = Field(
        ..., description="ONNX model inference execution time in milliseconds."
    )
    total_time_ms: float = Field(
        ..., description="Total endpoint execution time in milliseconds."
    )
    model_type: str = Field(
        ..., description="Type of model used (e.g. lightgbm, xgboost)."
    )


class BatchPredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of feature dictionaries for batch prediction.",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Classification decision threshold (default: 0.5).",
    )


class BatchPredictionItem(BaseModel):
    index: int
    probability: float
    prediction: int
    risk_tier: str


class BatchPredictionResponse(BaseModel):
    predictions: list[BatchPredictionItem]
    batch_size: int
    total_inference_time_ms: float = Field(
        ..., description="Total batch ONNX inference time in milliseconds."
    )
    avg_per_sample_ms: float = Field(
        ..., description="Average inference time per sample in milliseconds."
    )
    total_time_ms: float = Field(
        ..., description="Total endpoint execution time in milliseconds."
    )
    model_type: str


# ----------------------------------------------------------------------------
# Inference Helpers
# ----------------------------------------------------------------------------
def _vectorize_input(
    records: list[dict[str, Any]], feature_names: list[str], n_features: int
) -> np.ndarray:
    """Transform list of feature dicts into a contiguous float32 numpy array."""
    batch_size = len(records)
    arr = np.full((batch_size, n_features), np.nan, dtype=np.float32)

    if feature_names:
        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        for row_idx, record in enumerate(records):
            for k, v in record.items():
                col_idx = name_to_idx.get(k)
                if col_idx is not None and v is not None:
                    try:
                        arr[row_idx, col_idx] = float(v)
                    except (ValueError, TypeError):
                        # Categorical or non-numeric values passed directly are assigned hash/code or NaN
                        arr[row_idx, col_idx] = np.nan
    else:
        # If no explicit feature names stored, populate sequentially from record values
        for row_idx, record in enumerate(records):
            vals = list(record.values())[:n_features]
            for col_idx, v in enumerate(vals):
                if v is not None:
                    try:
                        arr[row_idx, col_idx] = float(v)
                    except (ValueError, TypeError):
                        arr[row_idx, col_idx] = np.nan

    return arr


def _extract_probabilities(outputs: list[Any]) -> np.ndarray:
    """Extract default probabilities from ONNX Runtime outputs."""
    if not outputs:
        raise ValueError("ONNX model produced no outputs.")

    prob_out = outputs[1] if len(outputs) > 1 else outputs[0]

    # Handle zipmap list of dicts (e.g. [{0: p0, 1: p1}])
    if (
        isinstance(prob_out, list)
        and len(prob_out) > 0
        and isinstance(prob_out[0], dict)
    ):
        return np.array([d.get(1, d.get(1.0, 0.0)) for d in prob_out], dtype=np.float32)

    arr = np.asarray(prob_out, dtype=np.float32)
    # If 2D array of [prob_0, prob_1], return column 1 (positive class)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 1]
    # If 1D probabilities
    return arr.ravel()


def _get_risk_tier(prob: float, threshold: float) -> str:
    """Determine qualitative risk tier based on default probability."""
    if prob < 0.2:
        return "Very Low Risk"
    elif prob < threshold:
        return "Low Risk"
    elif prob < 0.7:
        return "Moderate Risk"
    else:
        return "High Risk"


# ----------------------------------------------------------------------------
# API Endpoints
# ----------------------------------------------------------------------------
@app.get("/health", tags=["Monitoring"])
@app.get("/", tags=["Monitoring"])
def health_check(request: Request) -> dict[str, Any]:
    """Health check endpoint returning model status and performance metrics."""
    if getattr(request.app.state, "session", None) is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    uptime_s = time.time() - request.app.state.startup_time
    return {
        "status": "healthy",
        "model_type": request.app.state.model_type,
        "n_features": request.app.state.n_features,
        "uptime_seconds": round(uptime_s, 2),
        "validation_metrics": request.app.state.metrics.get("val_roc_auc"),
    }


@app.post(
    "/predict",
    response_model=SinglePredictionResponse,
    tags=["Inference"],
    summary="Single Application Default Prediction",
)
def predict_single(
    req: SinglePredictionRequest, request: Request
) -> SinglePredictionResponse:
    """Predict default probability and risk tier for a single applicant."""
    total_start = time.perf_counter()
    session = request.app.state.session
    if session is None:
        raise HTTPException(
            status_code=503, detail="Inference session is not initialized."
        )

    # 1. Prepare input array
    input_data = _vectorize_input(
        records=[req.features],
        feature_names=request.app.state.feature_names,
        n_features=request.app.state.n_features,
    )

    # 2. Run ONNX model inference
    inf_start = time.perf_counter()
    outputs = session.run(None, {request.app.state.input_name: input_data})
    inf_duration_ms = (time.perf_counter() - inf_start) * 1000.0

    # 3. Extract probability
    probs = _extract_probabilities(outputs)
    prob = float(probs[0])
    prediction = int(prob >= req.threshold)
    risk_tier = _get_risk_tier(prob, req.threshold)

    total_duration_ms = (time.perf_counter() - total_start) * 1000.0

    return SinglePredictionResponse(
        probability=round(prob, 6),
        prediction=prediction,
        risk_tier=risk_tier,
        inference_time_ms=round(inf_duration_ms, 3),
        total_time_ms=round(total_duration_ms, 3),
        model_type=request.app.state.model_type,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Inference"],
    summary="Multi / Batch Application Default Prediction",
)
def predict_batch(
    req: BatchPredictionRequest, request: Request
) -> BatchPredictionResponse:
    """Predict default probability and risk tier for multiple applicants in a single vectorized batch."""
    total_start = time.perf_counter()
    session = request.app.state.session
    if session is None:
        raise HTTPException(
            status_code=503, detail="Inference session is not initialized."
        )

    batch_size = len(req.records)

    # 1. Vectorize batch input
    input_data = _vectorize_input(
        records=req.records,
        feature_names=request.app.state.feature_names,
        n_features=request.app.state.n_features,
    )

    # 2. Run vectorized batch inference
    inf_start = time.perf_counter()
    outputs = session.run(None, {request.app.state.input_name: input_data})
    inf_duration_ms = (time.perf_counter() - inf_start) * 1000.0

    # 3. Extract probabilities
    probs = _extract_probabilities(outputs)

    items: list[BatchPredictionItem] = []
    for idx, p in enumerate(probs):
        prob_val = float(p)
        items.append(
            BatchPredictionItem(
                index=idx,
                probability=round(prob_val, 6),
                prediction=int(prob_val >= req.threshold),
                risk_tier=_get_risk_tier(prob_val, req.threshold),
            )
        )

    total_duration_ms = (time.perf_counter() - total_start) * 1000.0
    avg_per_sample_ms = inf_duration_ms / batch_size if batch_size > 0 else 0.0

    return BatchPredictionResponse(
        predictions=items,
        batch_size=batch_size,
        total_inference_time_ms=round(inf_duration_ms, 3),
        avg_per_sample_ms=round(avg_per_sample_ms, 4),
        total_time_ms=round(total_duration_ms, 3),
        model_type=request.app.state.model_type,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
