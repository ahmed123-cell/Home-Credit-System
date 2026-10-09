"""Load mart_train / mart_test from PostgreSQL and prepare model-ready data.

Steps
1. Read the two dbt views.
2. Clean types: Decimal -> float, bool -> int, inf -> NaN.
3. Engineer extra features (ratios, aggregates of flags, null-aware score blends).
4. Learn which columns to drop on TRAIN only, then apply to both sets.
5. Convert text columns to pandas `category` with train-defined levels.

Logging
    Structured: every record has `ts`, `level`, `logger`, `event` plus event-specific fields.
    LOG_FORMAT=json (default) | console      LOG_LEVEL=INFO (default) | DEBUG | ...

Usage:
    from data_setup import load_data
    data = load_data()
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Iterator

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
log = logging.getLogger("data_setup")

ID_COL = "sk_id_curr"
TARGET_COL = "target"
DATASET_COL = "dataset"

MISSING_THRESHOLD = 0.90      # drop columns with more than 90% missing (train)
DOMINANT_THRESHOLD = 0.999    # drop columns where one value covers >= 99.9% of rows
KEEP_DOCUMENT_FLAGS = {"flag_document_3"}  # the only informative document flag

EXT_COLS = ["ext_source_1", "ext_source_2", "ext_source_3"]
EXT_WEIGHTS = np.array([2.0, 3.0, 4.0])

BUREAU_ENQUIRY_COLS = [
    "amt_req_credit_bureau_hour",
    "amt_req_credit_bureau_day",
    "amt_req_credit_bureau_week",
    "amt_req_credit_bureau_mon",
    "amt_req_credit_bureau_qrt",
    "amt_req_credit_bureau_year",
]
CONTACT_FLAG_COLS = [
    "flag_emp_phone", "flag_work_phone", "flag_cont_mobile", "flag_phone", "flag_email",
]
ADDRESS_MISMATCH_COLS = [
    "reg_region_not_live_region", "reg_region_not_work_region", "live_region_not_work_region",
    "reg_city_not_live_city", "reg_city_not_work_city", "live_city_not_work_city",
]


# ----------------------------------------------------------------------------
# Structured logging
# ----------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    """One JSON object per line: ts, level, logger, event, <fields>, exception."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-friendly variant: `time LEVEL logger event key=value ...`."""

    def format(self, record: logging.LogRecord) -> str:
        fields = " ".join(f"{k}={v}" for k, v in getattr(record, "fields", {}).items())
        line = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} {record.name} {record.getMessage()} {fields}".rstrip()
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def configure_logging() -> None:
    """Call once from an entry point (never at import time, so libraries stay quiet)."""
    handler = logging.StreamHandler(sys.stdout)
    use_console = os.getenv("LOG_FORMAT", "json").lower() == "console"
    handler.setFormatter(ConsoleFormatter() if use_console else JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    # keep SQLAlchemy / urllib noise out unless explicitly requested
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def _log(event: str, level: int = logging.INFO, **fields: Any) -> None:
    log.log(level, event, extra={"fields": fields})


@contextmanager
def timed(event: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Log `<event>.started`, then `<event>.completed` (or `.failed`) with duration_s.

    The body can add result fields: `with timed("x") as ctx: ctx["rows"] = 10`.
    """
    result: dict[str, Any] = {}
    start = time.perf_counter()
    _log(f"{event}.started", **fields)
    try:
        yield result
    except Exception:
        log.exception(
            f"{event}.failed",
            extra={"fields": {**fields, "duration_s": round(time.perf_counter() - start, 2)}},
        )
        raise
    _log(f"{event}.completed", duration_s=round(time.perf_counter() - start, 2), **fields, **result)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def _engine():
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set (check your .env file)")
    # SQLAlchemy needs the driver name to use psycopg 3
    return create_engine(dsn.replace("postgresql://", "postgresql+psycopg://", 1))


def read_mart(view: str) -> pd.DataFrame:
    with timed("read_mart", view=view) as ctx:
        with _engine().connect() as conn:
            df = pd.read_sql(f"select * from marts.{view}", conn)
        ctx.update(rows=len(df), columns=df.shape[1], memory_mb=round(df.memory_usage(deep=True).sum() / 1e6))
    return df


# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------
def _clean_types(df: pd.DataFrame) -> pd.DataFrame:
    """Postgres avg()/sum() on bigint return numeric -> Decimal objects in pandas."""
    with timed("clean_types") as ctx:
        decimal_cols = []
        for c in df.select_dtypes("object").columns:
            sample = df[c].dropna()
            if len(sample) and isinstance(sample.iloc[0], Decimal):
                df[c] = df[c].astype("float64")
                decimal_cols.append(c)

        bool_cols = df.select_dtypes("bool").columns
        df[bool_cols] = df[bool_cols].astype("int8")

        num_cols = df.select_dtypes("number").columns
        inf_count = int(np.isinf(df[num_cols].to_numpy(dtype="float64", na_value=np.nan)).sum())
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

        ctx.update(decimal_cols_converted=len(decimal_cols), bool_cols_converted=len(bool_cols), inf_replaced=inf_count)
    return df


def _ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    """a / b with division by zero -> NaN."""
    return a / b.where(b != 0)


# ----------------------------------------------------------------------------
# Feature engineering (row-wise only, so it can never leak across rows or sets)
# ----------------------------------------------------------------------------
def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    with timed("engineer_features") as ctx:
        new: dict[str, pd.Series] = {}

        # Timeline ratios relative to age
        new["id_publish_birth_ratio"] = _ratio(df["days_id_publish"], df["days_birth"])
        new["registration_birth_ratio"] = _ratio(df["days_registration"], df["days_birth"])
        new["phone_change_birth_ratio"] = _ratio(df["days_last_phone_change"], df["days_birth"])
        new["car_age_employed_ratio"] = _ratio(df["own_car_age"], df["employed_years"])

        # Household and money
        new["children_ratio"] = _ratio(df["cnt_children"], df["cnt_fam_members"])
        new["income_per_child"] = df["amt_income_total"] / (df["cnt_children"] + 1)
        new["credit_minus_goods"] = df["amt_credit"] - df["amt_goods_price"]
        new["annuity_vs_prev_annuity"] = _ratio(df["amt_annuity"], df["prev_amt_annuity_mean"])
        new["bureau_credit_income_ratio"] = _ratio(df["bureau_credit_sum"], df["amt_income_total"])

        # External scores: null-aware weighted blend, spread, and interaction with age
        ext = df[EXT_COLS]
        weights = ext.notna().to_numpy() * EXT_WEIGHTS
        weighted_sum = (ext.fillna(0).to_numpy() * weights).sum(axis=1)
        weight_total = weights.sum(axis=1)
        new["ext_source_weighted"] = pd.Series(
            np.where(weight_total > 0, weighted_sum / np.where(weight_total > 0, weight_total, 1), np.nan),
            index=df.index,
        )
        new["ext_source_std"] = ext.std(axis=1)
        new["ext_source_2_x_age"] = df["ext_source_2"] * df["age_years"]

        # Credit bureau enquiries and social circle
        new["bureau_enquiries_total"] = df[BUREAU_ENQUIRY_COLS].sum(axis=1, min_count=1)
        new["social_def_obs_30"] = _ratio(df["def_30_cnt_social_circle"], df["obs_30_cnt_social_circle"])
        new["social_def_obs_60"] = _ratio(df["def_60_cnt_social_circle"], df["obs_60_cnt_social_circle"])

        # Flag aggregates
        new["contact_flags_cnt"] = df[CONTACT_FLAG_COLS].sum(axis=1)
        new["address_mismatch_cnt"] = df[ADDRESS_MISMATCH_COLS].sum(axis=1)
        new["region_rating_diff"] = df["region_rating_client_w_city"] - df["region_rating_client"]

        # History shares (counts were zero-filled in the mart, so 0 denominators give NaN)
        new["prev_approval_rate"] = _ratio(df["prev_approved_cnt"], df["prev_cnt"])
        new["bureau_active_share"] = _ratio(df["bureau_active_cnt"], df["bureau_cnt"])

        out = pd.concat([df, pd.DataFrame(new)], axis=1).replace([np.inf, -np.inf], np.nan)
        ctx.update(features_added=len(new), columns_after=out.shape[1])
        _log("engineer_features.names", level=logging.DEBUG, names=list(new))
    return out


# ----------------------------------------------------------------------------
# Fit on train, apply to both
# ----------------------------------------------------------------------------
@dataclass
class Preprocessor:
    drop_cols: list[str]
    cat_levels: dict[str, list[str]]
    feature_names: list[str]

    @classmethod
    def fit(cls, X: pd.DataFrame) -> "Preprocessor":
        with timed("preprocessor.fit", rows=len(X), columns=X.shape[1]) as ctx:
            cols = set(X.columns)

            # 1. Known redundant columns
            explicit = {
                c for c in cols
                if c.startswith("flag_document_") and c not in KEEP_DOCUMENT_FLAGS
            }
            # building features come as _avg / _mode / _medi triples; keep _avg
            explicit |= {
                c for c in cols
                if c.endswith(("_mode", "_medi")) and f"{c.rsplit('_', 1)[0]}_avg" in cols
            }

            # 2. Mostly missing
            missing = X.isna().mean()
            too_missing = set(missing[missing > MISSING_THRESHOLD].index)

            # 3. Constant or almost constant
            dominant = {
                c for c in X.columns
                if X[c].value_counts(normalize=True, dropna=False).iloc[0] >= DOMINANT_THRESHOLD
            }

            drop = sorted(explicit | too_missing | dominant)
            kept = X.drop(columns=drop)
            cat_cols = kept.select_dtypes(exclude="number").columns
            cat_levels = {c: sorted(kept[c].dropna().unique().tolist()) for c in cat_cols}

            ctx.update(
                dropped_total=len(drop),
                dropped_redundant=len(explicit),
                dropped_mostly_missing=len(too_missing),
                dropped_near_constant=len(dominant),
                features_kept=kept.shape[1],
                categorical_cols=len(cat_levels),
            )
            _log("preprocessor.dropped_columns", level=logging.DEBUG, columns=drop)
            _log(
                "preprocessor.missing_summary",
                level=logging.DEBUG,
                worst_missing={k: round(float(v), 3) for k, v in missing.nlargest(10).items()},
            )
        return cls(drop_cols=drop, cat_levels=cat_levels, feature_names=list(kept.columns))

    def transform(self, X: pd.DataFrame, name: str = "data") -> pd.DataFrame:
        with timed("preprocessor.transform", dataset=name, rows=len(X)):
            X = X[self.feature_names].copy()
            for c, levels in self.cat_levels.items():
                before = int(X[c].notna().sum())
                X[c] = pd.Categorical(X[c], categories=levels)  # unseen levels -> NaN
                unseen = before - int(X[c].notna().sum())
                if unseen:
                    _log("preprocessor.unseen_categories", level=logging.WARNING,
                         dataset=name, column=c, rows_set_to_nan=unseen)
        return X


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
@dataclass
class DataBundle:
    X_train: pd.DataFrame
    y_train: pd.Series
    ids_train: pd.Series
    X_test: pd.DataFrame
    ids_test: pd.Series
    cat_cols: list[str]
    preprocessor: Preprocessor


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    ids = df[ID_COL]
    y = df[TARGET_COL].astype("int8") if df[TARGET_COL].notna().all() else None
    X = df.drop(columns=[ID_COL, TARGET_COL, DATASET_COL])
    return X, ids, y


def load_data() -> DataBundle:
    with timed("load_data") as ctx:
        train = _engineer(_clean_types(read_mart("mart_train")))
        test = _engineer(_clean_types(read_mart("mart_test")))

        X_train, ids_train, y_train = _split(train)
        X_test, ids_test, _ = _split(test)

        prep = Preprocessor.fit(X_train)
        X_train = prep.transform(X_train, name="train")
        X_test = prep.transform(X_test, name="test")

        if list(X_train.columns) != list(X_test.columns):
            raise AssertionError("train and test feature columns differ after preprocessing")

        ctx.update(
            train_rows=len(X_train),
            test_rows=len(X_test),
            n_features=X_train.shape[1],
            default_rate=round(float(y_train.mean()), 6),
            train_memory_mb=round(X_train.memory_usage(deep=True).sum() / 1e6),
        )
        _log("load_data.dtypes", level=logging.DEBUG,
             counts=X_train.dtypes.astype(str).value_counts().to_dict())
        _log("load_data.categorical_columns", level=logging.DEBUG, columns=list(prep.cat_levels))

    return DataBundle(
        X_train=X_train,
        y_train=y_train,
        ids_train=ids_train.reset_index(drop=True),
        X_test=X_test,
        ids_test=ids_test.reset_index(drop=True),
        cat_cols=list(prep.cat_levels),
        preprocessor=prep,
    )


if __name__ == "__main__":
    configure_logging()
    load_data()