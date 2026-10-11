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
import math
import os
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
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

from training.data_setup import (
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
    "id_publish_birth_ratio",
    "registration_birth_ratio",
    "phone_change_birth_ratio",
    "car_age_employed_ratio",
    "children_ratio",
    "income_per_child",
    "credit_minus_goods",
    "annuity_vs_prev_annuity",
    "bureau_credit_income_ratio",
    "ext_source_weighted",
    "ext_source_std",
    "ext_source_2_x_age",
    "bureau_enquiries_total",
    "social_def_obs_30",
    "social_def_obs_60",
    "contact_flags_cnt",
    "address_mismatch_cnt",
    "region_rating_diff",
    "prev_approval_rate",
    "bureau_active_share",
}

# Columns computed in the dbt mart SQL (marts/mart_application_features.sql).
MART_DERIVED = {
    "age_years",
    "employed_years",
    "employed_birth_ratio",
    "credit_income_ratio",
    "annuity_income_ratio",
    "credit_term",
    "credit_goods_ratio",
    "income_per_person",
    "ext_source_mean",
    "ext_source_min",
    "ext_source_max",
    "ext_source_prod",
    "ext_source_nonnull_cnt",
    "bureau_debt_income_ratio",
    "credit_prev_credit_ratio",
    "has_bureau",
    "has_prev",
    "has_inst",
    "has_pos",
    "has_cc",
}

# History counts: zero-filled in the mart when the customer has no history.
HISTORY_GROUPS: dict[str, list[str]] = {
    "has_bureau": [
        "bureau_cnt",
        "bureau_active_cnt",
        "bureau_closed_cnt",
        "bureau_credit_type_nunique",
        "bureau_cnt_last_2y",
        "bureau_overdue_cnt",
    ],
    "has_prev": [
        "prev_cnt",
        "prev_approved_cnt",
        "prev_refused_cnt",
        "prev_canceled_cnt",
        "prev_cnt_last_1y",
    ],
    "has_inst": ["inst_cnt", "inst_prev_nunique"],
    "has_pos": [
        "pos_rows",
        "pos_prev_nunique",
        "pos_active_rows",
        "pos_completed_rows",
    ],
    "has_cc": ["cc_rows", "cc_prev_nunique"],
}

# Raw columns the derived-feature code reads, even if the model dropped them.
EXTRA_INPUTS = {
    "days_id_publish",
    "days_birth",
    "days_employed",
    "days_registration",
    "days_last_phone_change",
    "own_car_age",
    "cnt_children",
    "cnt_fam_members",
    "amt_income_total",
    "amt_credit",
    "amt_goods_price",
    "amt_annuity",
    "prev_amt_annuity_mean",
    "prev_amt_credit_mean",
    "bureau_credit_sum",
    "bureau_debt_sum",
    "def_30_cnt_social_circle",
    "obs_30_cnt_social_circle",
    "def_60_cnt_social_circle",
    "obs_60_cnt_social_circle",
    "region_rating_client_w_city",
    "region_rating_client",
    "flag_document_3",
    "document_cnt",
    "bureau_debt_credit_ratio",
    "prev_refused_share",
    *EXT_COLS,
    *BUREAU_ENQUIRY_COLS,
    *CONTACT_FLAG_COLS,
    *ADDRESS_MISMATCH_COLS,
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
    df["ext_source_mean"] = ext.mean(axis=1)  # NaN-aware, NaN if all missing
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
    df["bureau_debt_income_ratio"] = _ratio(
        df["bureau_debt_sum"], df["amt_income_total"]
    )
    df["credit_prev_credit_ratio"] = _ratio(
        df["amt_credit"], df["prev_amt_credit_mean"]
    )
    return df


def build_model_input(records: list[dict[str, Any]], prep: Any) -> np.ndarray:
    """Raw applicant dicts -> float32 matrix in the model's training column order."""
    df = pd.DataFrame(records)
    df = df.drop(
        columns=[c for c in df.columns if c in ENGINEERED or c in MART_DERIVED]
    )

    needed = (set(prep.feature_names) - ENGINEERED - MART_DERIVED) | EXTRA_INPUTS
    df = df.reindex(
        columns=sorted(set(df.columns) | needed)
    )  # add missing columns in one go

    cat_cols = set(prep.cat_levels)
    # Convert every column first and build the frame ONCE. Assigning ~230 columns one at a
    # time leaves one memory block per column, which is what triggered pandas'
    # "DataFrame is highly fragmented" PerformanceWarning.
    df = pd.DataFrame(
        {
            c: (
                df[c].astype("object").replace({"": np.nan})
                if c in cat_cols
                else pd.to_numeric(df[c], errors="coerce")
            )
            for c in df.columns
        }
    )

    df = _add_mart_features(df)
    df = _engineer(df)  # same function used for training
    X = prep.transform(df, name="request")  # category levels + scaler fitted on train
    return to_onnx_input(X[prep.feature_names])


# ----------------------------------------------------------------------------
# Input validation: readable messages, one per problem
# ----------------------------------------------------------------------------
def _years(v: float) -> float:
    return -v / 365.25


def _months(v: float) -> float:
    return -v / 30.4


def _same(v: float) -> float:
    return v


@dataclass(frozen=True)
class Rule:
    """Allowed range for one numeric input, expressed in the units a person would type."""

    label: str
    lo: float | None = None
    hi: float | None = None
    unit: str = ""
    integer: bool = False
    lo_exclusive: bool = False
    to_display: Callable[[float], float] = (
        _same  # model units -> units shown to the user
    )


def _count(label: str, hi: float | None = None) -> Rule:
    return Rule(label, 0, hi, integer=True)


def _amount(label: str) -> Rule:
    return Rule(label, 0)


def _positive(label: str) -> Rule:
    return Rule(label, 0, lo_exclusive=True)


RULES: dict[str, Rule] = {
    "days_birth": Rule("Age", 18, 100, "years", to_display=_years),
    "days_employed": Rule("Time in current job", 0, 70, "years", to_display=_years),
    "days_registration": Rule(
        "Registration last changed", 0, 100, "years ago", to_display=_years
    ),
    "days_id_publish": Rule("ID issued", 0, 100, "years ago", to_display=_years),
    "days_last_phone_change": Rule(
        "Phone number last changed", 0, 600, "months ago", to_display=_months
    ),
    "amt_income_total": _positive("Annual income"),
    "amt_credit": _positive("Loan amount"),
    "amt_annuity": _positive("Annuity (per installment)"),
    "amt_goods_price": _amount("Price of goods"),
    "cnt_children": _count("Children", 30),
    "cnt_fam_members": Rule("Family members", 1, 30, integer=True),
    "own_car_age": Rule("Car age", 0, 80, "years"),
    "hour_appr_process_start": _count("Application hour", 23),
    "region_rating_client": Rule("Region rating", 1, 3, integer=True),
    "region_rating_client_w_city": Rule("Region and city rating", 1, 3, integer=True),
    "region_population_relative": Rule("Region population", 0, 1),
    "ext_source_1": Rule("External score 1", 0, 1),
    "ext_source_2": Rule("External score 2", 0, 1),
    "ext_source_3": Rule("External score 3", 0, 1),
    "amt_req_credit_bureau_hour": _count("Bureau enquiries, last hour"),
    "amt_req_credit_bureau_day": _count("Bureau enquiries, last day"),
    "amt_req_credit_bureau_week": _count("Bureau enquiries, last week"),
    "amt_req_credit_bureau_mon": _count("Bureau enquiries, last month"),
    "amt_req_credit_bureau_qrt": _count("Bureau enquiries, last quarter"),
    "amt_req_credit_bureau_year": _count("Bureau enquiries, last year"),
    "obs_30_cnt_social_circle": _count("Social circle observed, 30 days"),
    "def_30_cnt_social_circle": _count("Social circle defaulted, 30 days"),
    "obs_60_cnt_social_circle": _count("Social circle observed, 60 days"),
    "def_60_cnt_social_circle": _count("Social circle defaulted, 60 days"),
    "bureau_cnt": _count("Credits reported"),
    "bureau_active_cnt": _count("Active credits"),
    "bureau_closed_cnt": _count("Closed credits"),
    "bureau_cnt_last_2y": _count("Credits opened in the last 2 years"),
    "bureau_overdue_cnt": _count("Credits ever overdue"),
    "bureau_credit_sum": _amount("Total credit amount"),
    "bureau_debt_sum": _amount("Current debt"),
    "bureau_day_overdue_max": Rule("Longest overdue", 0, unit="days"),
    "prev_cnt": _count("Earlier applications"),
    "prev_approved_cnt": _count("Approved applications"),
    "prev_refused_cnt": _count("Refused applications"),
    "prev_canceled_cnt": _count("Canceled applications"),
    "prev_amt_credit_mean": _amount("Average credit"),
    "prev_amt_annuity_mean": _amount("Average annuity"),
    "inst_cnt": _count("Installments paid"),
    "inst_late_share": Rule("Share paid late", 0, 1),
    "inst_dpd_mean": Rule("Average days late", 0),
    "inst_dpd_max": Rule("Longest delay", 0, unit="days"),
    "pos_rows": _count("Cash loan months on record"),
    "cc_rows": _count("Credit card months on record"),
    "cc_utilization_mean": Rule("Average card utilization", 0),
}

# 0/1 inputs (checkboxes in the UI)
FLAG_LABELS = {
    "flag_phone": "Home phone",
    "flag_work_phone": "Work phone",
    "flag_emp_phone": "Employer phone",
    "flag_cont_mobile": "Mobile phone",
    "flag_email": "Email provided",
    "flag_document_3": "Document 3 provided",
    "reg_region_not_live_region": "Permanent vs contact address (region)",
    "reg_region_not_work_region": "Permanent address vs work (region)",
    "live_region_not_work_region": "Contact address vs work (region)",
    "reg_city_not_live_city": "Permanent vs contact address (city)",
    "reg_city_not_work_city": "Permanent address vs work (city)",
    "live_city_not_work_city": "Contact address vs work (city)",
}

CAT_LABELS = {
    "name_contract_type": "Loan type",
    "code_gender": "Gender",
    "flag_own_car": "Owns a car",
    "flag_own_realty": "Owns property",
    "name_type_suite": "Accompanied by",
    "name_income_type": "Income source",
    "name_education_type": "Education",
    "name_family_status": "Family status",
    "name_housing_type": "Housing",
    "occupation_type": "Occupation",
    "weekday_appr_process_start": "Application weekday",
    "organization_type": "Employer type",
}

REQUIRED = [
    "days_birth",
    "amt_income_total",
    "amt_credit",
    "amt_annuity",
    "name_contract_type",
]

# A block of history is only meaningful if its count is at least 1.
HISTORY_DEPENDENTS = {
    "bureau_cnt": [
        "bureau_active_cnt",
        "bureau_closed_cnt",
        "bureau_cnt_last_2y",
        "bureau_overdue_cnt",
        "bureau_credit_sum",
        "bureau_debt_sum",
        "bureau_day_overdue_max",
    ],
    "prev_cnt": [
        "prev_approved_cnt",
        "prev_refused_cnt",
        "prev_canceled_cnt",
        "prev_amt_credit_mean",
        "prev_amt_annuity_mean",
    ],
    "inst_cnt": ["inst_late_share", "inst_dpd_mean", "inst_dpd_max"],
    "cc_rows": ["cc_utilization_mean"],
}


def _label(key: str) -> str:
    if key in RULES:
        return RULES[key].label
    return (
        CAT_LABELS.get(key)
        or FLAG_LABELS.get(key)
        or key.replace("_", " ").capitalize()
    )


def _num(x: float) -> str:
    return f"{x:,.2f}".rstrip("0").rstrip(".")


def _present(v: Any) -> bool:
    return v is not None and v != "" and not (isinstance(v, float) and math.isnan(v))


def _to_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return float(v)
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _range_message(rule: Rule, shown: float) -> str:
    u = f" {rule.unit}" if rule.unit else ""
    if rule.lo is not None and rule.hi is not None:
        base = f"{rule.label} must be between {_num(rule.lo)} and {_num(rule.hi)}{u}"
    elif rule.lo_exclusive:
        base = f"{rule.label} must be greater than {_num(rule.lo or 0)}{u}"
    elif rule.lo == 0:
        base = f"{rule.label} can't be negative"
    elif rule.lo is not None:
        base = f"{rule.label} must be at least {_num(rule.lo)}{u}"
    else:
        base = f"{rule.label} must be at most {_num(rule.hi or 0)}{u}"
    return f"{base} (you entered {_num(shown)})."


def validate_record(
    rec: dict[str, Any], prep: Any, row: int | None = None
) -> list[dict[str, Any]]:
    """Return one readable error per problem. An empty list means the record is acceptable."""
    errors: list[dict[str, Any]] = []
    bad: set[str] = set()

    def fail(key: str, message: str) -> None:
        bad.add(key)
        errors.append(
            {
                "field": key,
                "label": _label(key),
                "message": message,
                "row": None if row is None else row + 1,
            }
        )

    # 1. required fields
    for key in REQUIRED:
        if not _present(rec.get(key)):
            fail(key, f"{_label(key)} is required.")

    # 2. numbers: type and range, in the units a person would type
    numeric_keys = (set(prep.feature_names) | EXTRA_INPUTS) - set(prep.cat_levels)
    for key in sorted(numeric_keys & set(rec)):
        v = rec[key]
        if not _present(v) or key in bad:
            continue
        x = _to_float(v)
        if x is None:
            fail(key, f"{_label(key)} must be a number (you entered {str(v)[:40]!r}).")
            continue
        if key in FLAG_LABELS:
            if x not in (0.0, 1.0):
                fail(key, f"{_label(key)} must be Yes or No (0 or 1).")
            continue
        rule = RULES.get(key)
        if rule is None:
            continue
        shown = rule.to_display(x)
        if rule.integer and abs(shown - round(shown)) > 1e-9:
            fail(
                key, f"{rule.label} must be a whole number (you entered {_num(shown)})."
            )
        elif (
            rule.lo is not None
            and (shown <= rule.lo if rule.lo_exclusive else shown < rule.lo)
        ) or (rule.hi is not None and shown > rule.hi):
            fail(key, _range_message(rule, shown))

    # 3. categories: the value must be one the model knows
    for key, levels in prep.cat_levels.items():
        v = rec.get(key)
        if not _present(v) or key in bad:
            continue
        if str(v) not in levels:
            sample = ", ".join(levels[:8]) + (", …" if len(levels) > 8 else "")
            fail(
                key,
                f"{_label(key)} '{str(v)[:40]}' isn't recognised. Choose one of: {sample}.",
            )

    # 4. relationships between fields
    def val(key: str) -> float | None:
        return (
            None
            if key in bad or not _present(rec.get(key))
            else _to_float(rec.get(key))
        )

    emp, age = val("days_employed"), val("days_birth")
    if emp is not None and age is not None and _years(emp) > _years(age):
        fail(
            "days_employed",
            f"Time in current job ({_num(_years(emp))} years) can't be longer than the "
            f"applicant's age ({_num(_years(age))} years).",
        )

    cnt = val("bureau_cnt")
    for key in ("bureau_active_cnt", "bureau_closed_cnt"):
        n = val(key)
        if cnt is not None and n is not None and n > cnt:
            fail(
                key,
                f"{_label(key)} ({_num(n)}) can't be more than credits reported ({_num(cnt)}).",
            )

    prev = val("prev_cnt")
    parts = [
        val(k) for k in ("prev_approved_cnt", "prev_refused_cnt", "prev_canceled_cnt")
    ]
    if prev is not None and sum(p or 0 for p in parts) > prev:
        fail(
            "prev_cnt",
            f"Approved, refused and canceled applications add up to {_num(sum(p or 0 for p in parts))}, "
            f"more than the {_num(prev)} applications reported.",
        )

    for count_key, dependents in HISTORY_DEPENDENTS.items():
        if count_key in bad:
            continue
        n = val(count_key)
        if (n is None or n < 1) and any(_present(rec.get(d)) for d in dependents):
            fail(
                count_key,
                f"{_label(count_key)} must be 1 or more, because other details in this block are filled in.",
            )

    return errors


def _reject(errors: list[dict[str, Any]]) -> None:
    shown = errors[:50]
    first = "; ".join(
        (f"Row {e['row']}: " if e["row"] else "") + e["message"].rstrip(".")
        for e in shown[:3]
    )
    more = f" (and {len(errors) - 3} more)" if len(errors) > 3 else ""
    raise HTTPException(
        status_code=422,
        detail={
            "message": f"{len(errors)} input{'s' if len(errors) != 1 else ''} need fixing: {first}{more}",
            "errors": shown,
        },
    )


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
        os.getenv(
            "PREPROCESSOR_PATH",
            str(model_path.parent / f"{model_path.stem}_preprocessor.joblib"),
        )
    )
    if not prep_path.is_file():
        raise FileNotFoundError(
            f"Preprocessor not found at '{prep_path}'. Retrain with the updated train.py so it "
            "saves <model>_preprocessor.joblib next to the ONNX file."
        )
    prep = joblib.load(prep_path)

    if meta.get("feature_names") and list(meta["feature_names"]) != list(
        prep.feature_names
    ):
        raise RuntimeError(
            "Feature names in the meta JSON and the preprocessor differ. Retrain to resync them."
        )

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, os.cpu_count() or 4)
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3

    session = ort.InferenceSession(str(model_path), sess_options=opts)
    inp = session.get_inputs()[0]
    onnx_dim = inp.shape[1] if len(inp.shape) > 1 else None
    if isinstance(onnx_dim, int) and onnx_dim != len(prep.feature_names):
        raise RuntimeError(
            f"ONNX model expects {onnx_dim} features but the preprocessor produces {len(prep.feature_names)}."
        )

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
    response.headers["X-Process-Time-Ms"] = (
        f"{(time.perf_counter() - start) * 1000.0:.3f}"
    )
    return response


# ----------------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------------
class SinglePredictionRequest(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="Raw applicant fields (text categories allowed). Missing fields are treated as unknown.",
        examples=[
            {
                "name_contract_type": "Cash loans",
                "code_gender": "F",
                "amt_credit": 406597.5,
                "amt_income_total": 202500.0,
                "amt_annuity": 24700.5,
                "days_birth": -15500,
                "ext_source_2": 0.55,
            }
        ],
    )
    threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Decision threshold."
    )


class SinglePredictionResponse(BaseModel):
    probability: float = Field(
        ...,
        description="Default risk score (0-1). Not calibrated: the model used class weighting.",
    )
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


def _score(
    request: Request, records: list[dict[str, Any]]
) -> tuple[np.ndarray, float, float]:
    """Return (probabilities, preprocess_ms, inference_ms)."""
    state = request.app.state
    if state.session is None:
        raise HTTPException(
            status_code=503, detail="Inference session is not initialized."
        )

    errors = [
        e
        for i, rec in enumerate(records)
        for e in validate_record(
            rec, state.preprocessor, i if len(records) > 1 else None
        )
    ]
    if errors:
        _reject(errors)

    t0 = time.perf_counter()
    try:
        X = build_model_input(records, state.preprocessor)
    except Exception as exc:  # bad payloads should be a 422, not a 500
        raise HTTPException(
            status_code=422, detail=f"Could not prepare input: {exc}"
        ) from exc
    pre_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    outputs = state.session.run(None, {state.input_name: X})
    inf_ms = (time.perf_counter() - t1) * 1000.0
    return _extract_probabilities(outputs), pre_ms, inf_ms


def _find_ui() -> Path | None:
    candidates = [
        os.getenv("UI_PATH"),
        Path(__file__).with_name("UI.html"),
        Path("UI.html"),
        Path("ui/UI.html"),
    ]
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
        raise HTTPException(
            status_code=404,
            detail="UI.html not found. Put it next to app.py or set UI_PATH.",
        )
    return FileResponse(path, media_type="text/html")


@app.post("/predict", response_model=SinglePredictionResponse, tags=["Inference"])
def predict_single(
    req: SinglePredictionRequest, request: Request
) -> SinglePredictionResponse:
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
def predict_batch(
    req: BatchPredictionRequest, request: Request
) -> BatchPredictionResponse:
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
