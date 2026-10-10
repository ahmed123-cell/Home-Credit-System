"""FastAPI Inference Service for Home Credit Default Risk.

What changed compared with the first version
- Raw applicant input (text categories, friendly values) is turned into the exact
  205-column matrix the model was trained on, by reusing the training code:
  mart-level features (ratios, ext_source_*), `_engineer`, and the fitted
  `Preprocessor` (category levels + optional scaler) saved by train.py.
- History counts default to 0 and `has_*` flags are derived, as in the dbt mart.
- New endpoints: GET /schema (category levels for the UI) and GET /ui (serves UI.html).

Run from the project root so `training` is importable:
    PYTHONPATH=src uv run uvicorn app:app --host 0.0.0.0 --port 8000
(if app.py lives in a subfolder, add that folder to PYTHONPATH too, or use --app-dir)
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from training.data_setup import (  # noqa: E402  (needs PYTHONPATH=src)
    ADDRESS_MISMATCH_COLS,
    BUREAU_ENQUIRY_COLS,
    CONTACT_FLAG_COLS,
    EXT_COLS,
    _engineer,
    _ratio,
    to_onnx_input,
)

# ----------------------------------------------------------------------------
# Feature bookkeeping (mirrors dbt mart + data_setup._engineer)
# ----------------------------------------------------------------------------
# Columns computed by data_setup._engineer: never trust client-sent values.
ENGINEERED = {
    "id_publish_birth_ratio", "registration_birth_ratio", "phone_change_birth_ratio",
    "car_age_employed_ratio", "children_ratio", "income_per_child", "credit_minus_goods",
    "annuity_vs_prev_annuity", "bureau_credit_income_ratio", "ext_source_weighted",
    "ext_source_std", "ext_source_2_x_age", "bureau_enquiries_total", "social_def_obs_30",
    "social_def_obs_60", "contact_flags_cnt", "address_mismatch_cnt", "region_rating_diff",
    "prev_approval_rate", "bureau_active_share",
}

# Columns computed in the dbt mart SQL (marts/mart_application_features.sql).
MART_DERIVED = {
    "age_years", "employed_years", "employed_birth_ratio", "credit_income_ratio",
    "annuity_income_ratio", "credit_term", "credit_goods_ratio", "income_per_person",
    "ext_source_mean", "ext_source_min", "ext_source_max", "ext_source_prod",
    "ext_source_nonnull_cnt", "bureau_debt_income_ratio", "credit_prev_credit_ratio",
    "has_bureau", "has_prev", "has_inst", "has_pos", "has_cc",
}

# History counts: zero-filled in the mart when the customer has no history.
HISTORY_GROUPS: dict[str, list[str]] = {
    "has_bureau": ["bureau_cnt", "bureau_active_cnt", "bureau_closed_cnt",
                   "bureau_credit_type_nunique", "bureau_cnt_last_2y", "bureau_overdue_cnt"],
    "has_prev": ["prev_cnt", "prev_approved_cnt", "prev_refused_cnt",
                 "prev_canceled_cnt", "prev_cnt_last_1y"],
    "has_inst": ["inst_cnt", "inst_prev_nunique"],
    "has_pos": ["pos_rows", "pos_prev_nunique", "pos_active_rows", "pos_completed_rows"],
    "has_cc": ["cc_rows", "cc_prev_nunique"],
}

# Raw columns the derived-feature code reads, even if the model dropped them.
EXTRA_INPUTS = {
    "days_id_publish", "days_birth", "days_employed", "days_registration",
    "days_last_phone_change", "own_car_age", "cnt_children", "cnt_fam_members",
    "amt_income_total", "amt_credit", "amt_goods_price", "amt_annuity",
    "prev_amt_annuity_mean", "prev_amt_credit_mean", "bureau_credit_sum", "bureau_debt_sum",
    "def_30_cnt_social_circle", "obs_30_cnt_social_circle",
    "def_60_cnt_social_circle", "obs_60_cnt_social_circle",
    "region_rating_client_w_city", "region_rating_client",
    "flag_document_3", "document_cnt", "bureau_debt_credit_ratio", "prev_refused_share",
    *EXT_COLS, *BUREAU_ENQUIRY_COLS, *CONTACT_FLAG_COLS, *ADDRESS_MISMATCH_COLS,
    *[c for cols in HISTORY_GROUPS.values() for c in cols],
}


def _add_mart_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate the SQL features that data_setup._engineer expects to already exist."""
    df["age_years"] = -df["days_birth"] / 365.25
    df["employed_years"] = -df["days_employed"] / 365.25
    df["employed_birth_ratio"] = _ratio(df["days_employed"], df["days_birth"])
    df["credit_income_ratio"] = _ratio(df["amt_credit"], df["amt_income_total"])
    df["annuity_income_ratio"] = _ratio(df["amt_annuity"], df["amt_income_total"])
    df["credit_term"] = _ratio(df["amt_credit"], df["amt_annuity"])
    df["credit_goods_ratio"] = _ratio(df["amt_credit"], df["amt_goods_price"])
    df["income_per_person"] = _ratio(df["amt_income_total"], df["cnt_fam_members"])

    ext = df[EXT_COLS]
    df["ext_source_mean"] = ext.mean(axis=1)            # NaN-aware, NaN if all missing
    df["ext_source_min"] = ext.min(axis=1)
    df["ext_source_max"] = ext.max(axis=1)
    df["ext_source_prod"] = ext.prod(axis=1, skipna=False)
    df["ext_source_nonnull_cnt"] = ext.notna().sum(axis=1)

    # Only flag_document_3 is a model input, so fall back to it when no count is given.
    df["document_cnt"] = df["document_cnt"].fillna(df["flag_document_3"].fillna(0))

    # History: counts -> 0 when absent, has_* flag derived from them.
    for flag, cols in HISTORY_GROUPS.items():
        df[cols] = df[cols].fillna(0)
        df[flag] = (df[cols].sum(axis=1) > 0).astype("int8")

    df["bureau_debt_credit_ratio"] = df["bureau_debt_credit_ratio"].fillna(
        _ratio(df["bureau_debt_sum"], df["bureau_credit_sum"])
    )
    df["prev_refused_share"] = df["prev_refused_share"].fillna(
        _ratio(df["prev_refused_cnt"], df["prev_cnt"])
    )
    df["bureau_debt_income_ratio"] = _ratio(df["bureau_debt_sum"], df["amt_income_total"])
    df["credit_prev_credit_ratio"] = _ratio(df["amt_credit"], df["prev_amt_credit_mean"])
    return df


def build_model_input(records: list[dict[str, Any]], prep: Any) -> np.ndarray:
    """Raw applicant dicts -> float32 matrix in the model's training column order."""
    df = pd.DataFrame(records)
    df = df.drop(columns=[c for c in df.columns if c in ENGINEERED or c in MART_DERIVED])

    needed = (set(prep.feature_names) - ENGINEERED - MART_DERIVED) | EXTRA_INPUTS
    df = df.reindex(columns=sorted(set(df.columns) | needed))  # add missing columns in one go

    cat_cols = set(prep.cat_levels)
    for c in df.columns:
        if c in cat_cols:
            df[c] = df[c].astype("object").replace({"": np.nan})
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = _add_mart_features(df)
    df = _engineer(df)                       # same function used for training
    X = prep.transform(df, name="request")   # category levels + scaler fitted on train
    return to_onnx_input(X[prep.feature_names])


# ----------------------------------------------------------------------------
# Lifespan: load model, metadata and preprocessor ONCE
# ----------------------------------------------------------------------------
MODEL_DIR = Path(os.getenv("MODELS_DIR", "models"))
DEFAULT_MODEL_PATH = MODEL_DIR / "lightgbm.onnx"
DEFAULT_META_PATH = MODEL_DIR / "lightgbm_meta.json"


def load_model_resources() -> dict[str, Any]:
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    meta_path = Path(os.getenv("MODEL_META_PATH", str(DEFAULT_META_PATH)))

    if not model_path.is_file():
        alt_model = MODEL_DIR / "xgboost.onnx"
        alt_meta = MODEL_DIR / "xgboost_meta.json"
        if alt_model.is_file():
            model_path, meta_path = alt_model, alt_meta
        else:
            raise FileNotFoundError(
                f"No ONNX model found at '{model_path}' or '{alt_model}'. "
                "Train a model first with `python -m training.train`."
            )

    meta: dict[str, Any] = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    prep_path = Path(
        os.getenv("PREPROCESSOR_PATH", str(model_path.parent / f"{model_path.stem}_preprocessor.joblib"))
    )
    if not prep_path.is_file():
        raise FileNotFoundError(
            f"Preprocessor not found at '{prep_path}'. Retrain with the updated train.py so it "
            "saves <model>_preprocessor.joblib next to the ONNX file."
        )
    prep = joblib.load(prep_path)

    if meta.get("feature_names") and list(meta["feature_names"]) != list(prep.feature_names):
        raise RuntimeError("Feature names in the meta JSON and the preprocessor differ. Retrain to resync them.")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, os.cpu_count() or 4)
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3

    session = ort.InferenceSession(str(model_path), sess_options=opts)
    inp = session.get_inputs()[0]
    onnx_dim = inp.shape[1] if len(inp.shape) > 1 else None
    if isinstance(onnx_dim, int) and onnx_dim != len(prep.feature_names):
        raise RuntimeError(f"ONNX model expects {onnx_dim} features but the preprocessor produces {len(prep.feature_names)}.")

    return {
        "session": session,
        "input_name": inp.name,
        "preprocessor": prep,
        "n_features": len(prep.feature_names),
        "model_type": meta.get("model_type", model_path.stem),
        "model_path": str(model_path),
        "metrics": meta.get("metrics", {}),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    res = load_model_resources()
    app.state.session = res["session"]
    app.state.input_name = res["input_name"]
    app.state.preprocessor = res["preprocessor"]
    app.state.n_features = res["n_features"]
    app.state.model_type = res["model_type"]
    app.state.model_path = res["model_path"]
    app.state.metrics = res["metrics"]
    app.state.startup_time = time.time()
    print(
        f"[Startup] {res['model_type'].upper()} loaded from {res['model_path']} "
        f"({res['n_features']} features, scaler={'on' if res['preprocessor'].scaler is not None else 'off'})"
    )
    yield
    print("[Shutdown] Releasing model resources.")
    app.state.session = None


app = FastAPI(
    title="Home Credit Default Risk Inference API",
    description="Real-time credit default scoring powered by ONNX Runtime and FastAPI.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time-Ms"],
)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.perf_counter()
    response: Response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000.0:.3f}"
    return response


# ----------------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------------
class SinglePredictionRequest(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="Raw applicant fields (text categories allowed). Missing fields are treated as unknown.",
        examples=[{
            "name_contract_type": "Cash loans", "code_gender": "F",
            "amt_credit": 406597.5, "amt_income_total": 202500.0, "amt_annuity": 24700.5,
            "days_birth": -15500, "ext_source_2": 0.55,
        }],
    )
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Decision threshold.")


class SinglePredictionResponse(BaseModel):
    probability: float = Field(..., description="Default risk score (0-1). Not calibrated: the model used class weighting.")
    prediction: int = Field(..., description="1 = likely default, 0 = likely repay.")
    risk_tier: str
    preprocess_time_ms: float
    inference_time_ms: float
    total_time_ms: float
    model_type: str


class BatchPredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class BatchPredictionItem(BaseModel):
    index: int
    probability: float
    prediction: int
    risk_tier: str


class BatchPredictionResponse(BaseModel):
    predictions: list[BatchPredictionItem]
    batch_size: int
    preprocess_time_ms: float
    total_inference_time_ms: float
    avg_per_sample_ms: float
    total_time_ms: float
    model_type: str


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _extract_probabilities(outputs: list[Any]) -> np.ndarray:
    if not outputs:
        raise ValueError("ONNX model produced no outputs.")
    prob_out = outputs[1] if len(outputs) > 1 else outputs[0]
    if isinstance(prob_out, list) and prob_out and isinstance(prob_out[0], dict):
        return np.array([d.get(1, d.get(1.0, 0.0)) for d in prob_out], dtype=np.float32)
    arr = np.asarray(prob_out, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 1]
    return arr.ravel()


def _get_risk_tier(prob: float, threshold: float) -> str:
    if prob < 0.2:
        return "Very Low Risk"
    if prob < threshold:
        return "Low Risk"
    if prob < 0.7:
        return "Moderate Risk"
    return "High Risk"


def _score(request: Request, records: list[dict[str, Any]]) -> tuple[np.ndarray, float, float]:
    """Return (probabilities, preprocess_ms, inference_ms)."""
    state = request.app.state
    if state.session is None:
        raise HTTPException(status_code=503, detail="Inference session is not initialized.")

    t0 = time.perf_counter()
    try:
        X = build_model_input(records, state.preprocessor)
    except Exception as exc:  # bad payloads should be a 422, not a 500
        raise HTTPException(status_code=422, detail=f"Could not prepare input: {exc}") from exc
    pre_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    outputs = state.session.run(None, {state.input_name: X})
    inf_ms = (time.perf_counter() - t1) * 1000.0
    return _extract_probabilities(outputs), pre_ms, inf_ms


def _find_ui() -> Path | None:
    candidates = [os.getenv("UI_PATH"), Path(__file__).with_name("UI.html"), Path("UI.html"), Path("ui/UI.html")]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    return None


# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------
@app.get("/health", tags=["Monitoring"])
@app.get("/", tags=["Monitoring"])
def health_check(request: Request) -> dict[str, Any]:
    if getattr(request.app.state, "session", None) is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {
        "status": "healthy",
        "model_type": request.app.state.model_type,
        "n_features": request.app.state.n_features,
        "uptime_seconds": round(time.time() - request.app.state.startup_time, 2),
        "validation_metrics": request.app.state.metrics.get("val_roc_auc"),
    }


@app.get("/schema", tags=["Monitoring"])
def schema(request: Request) -> dict[str, Any]:
    """Category levels the model knows, so a UI can offer exactly valid choices."""
    prep = request.app.state.preprocessor
    return {
        "categories": prep.cat_levels,
        "n_features": request.app.state.n_features,
        "scaled": prep.scaler is not None,
        "model_type": request.app.state.model_type,
    }


@app.get("/ui", include_in_schema=False)
def serve_ui() -> FileResponse:
    path = _find_ui()
    if path is None:
        raise HTTPException(status_code=404, detail="UI.html not found. Put it next to app.py or set UI_PATH.")
    return FileResponse(path, media_type="text/html")


@app.post("/predict", response_model=SinglePredictionResponse, tags=["Inference"])
def predict_single(req: SinglePredictionRequest, request: Request) -> SinglePredictionResponse:
    total_start = time.perf_counter()
    probs, pre_ms, inf_ms = _score(request, [req.features])
    prob = float(probs[0])
    return SinglePredictionResponse(
        probability=round(prob, 6),
        prediction=int(prob >= req.threshold),
        risk_tier=_get_risk_tier(prob, req.threshold),
        preprocess_time_ms=round(pre_ms, 3),
        inference_time_ms=round(inf_ms, 3),
        total_time_ms=round((time.perf_counter() - total_start) * 1000.0, 3),
        model_type=request.app.state.model_type,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(req: BatchPredictionRequest, request: Request) -> BatchPredictionResponse:
    total_start = time.perf_counter()
    probs, pre_ms, inf_ms = _score(request, req.records)
    items = [
        BatchPredictionItem(
            index=i,
            probability=round(float(p), 6),
            prediction=int(float(p) >= req.threshold),
            risk_tier=_get_risk_tier(float(p), req.threshold),
        )
        for i, p in enumerate(probs)
    ]
    n = len(req.records)
    return BatchPredictionResponse(
        predictions=items,
        batch_size=n,
        preprocess_time_ms=round(pre_ms, 3),
        total_inference_time_ms=round(inf_ms, 3),
        avg_per_sample_ms=round(inf_ms / n, 4),
        total_time_ms=round((time.perf_counter() - total_start) * 1000.0, 3),
        model_type=request.app.state.model_type,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)