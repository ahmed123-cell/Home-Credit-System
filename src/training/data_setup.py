"""Load mart_train / mart_test from PostgreSQL or fast local cache and prepare model-ready data.

Pipeline Steps:
1. Check for fast local parquet cache (`data/processed/*.parquet`).
2. If cache missing or `force_refresh=True`:
   a. Read the dbt incremental tables (`marts.mart_train` and `marts.mart_test`).
   b. Clean types: Decimal -> float, bool -> int, inf -> NaN.
   c. Engineer extra domain features (ratios, aggregates of flags, null-aware score blends).
   d. Cache pre-engineered data to Parquet for instant loading (< 1s) on subsequent runs.
3. Learn column drops, categorical encodings, and numerical scaling on TRAIN only.
4. Apply identically to both train and test sets.

Usage:
    from training.data_setup import load_data
    bundle = load_data()
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
import joblib

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

from training.logging_config import configure_logging, get_logger, log_event, timed

load_dotenv()
log = get_logger("data_setup")

ID_COL = "sk_id_curr"
TARGET_COL = "target"
DATASET_COL = "dataset"

CACHE_DIR = Path("data/processed")

MISSING_THRESHOLD = 0.90  # drop columns with more than 90% missing (train)
DOMINANT_THRESHOLD = 0.999  # drop columns where one value covers >= 99.9% of rows
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
    "flag_emp_phone",
    "flag_work_phone",
    "flag_cont_mobile",
    "flag_phone",
    "flag_email",
]
ADDRESS_MISMATCH_COLS = [
    "reg_region_not_live_region",
    "reg_region_not_work_region",
    "live_region_not_work_region",
    "reg_city_not_live_city",
    "reg_city_not_work_city",
    "live_city_not_work_city",
]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def _engine():
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set (check your .env file)")
    return create_engine(dsn.replace("postgresql://", "postgresql+psycopg://", 1))


def read_mart(view: str) -> pd.DataFrame:
    """Read a mart view or table from the marts schema in PostgreSQL."""
    with timed(log, "read_mart", view=view) as ctx:
        with _engine().connect() as conn:
            df = pd.read_sql(f"select * from marts.{view}", conn)
        ctx.update(
            rows=len(df),
            columns=df.shape[1],
            memory_mb=round(df.memory_usage(deep=True).sum() / 1e6),
        )
    return df


# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------
def _clean_types(df: pd.DataFrame) -> pd.DataFrame:
    """Postgres avg()/sum() on bigint return numeric -> Decimal objects in pandas."""
    with timed(log, "clean_types") as ctx:
        decimal_cols = []
        obj_cols = df.select_dtypes(include=["object", "string"]).columns
        for c in obj_cols:
            sample = df[c].dropna()
            if len(sample) and isinstance(sample.iloc[0], Decimal):
                df[c] = df[c].astype("float64")
                decimal_cols.append(c)

        bool_cols = df.select_dtypes("bool").columns
        df[bool_cols] = df[bool_cols].astype("int8")

        num_cols = df.select_dtypes("number").columns
        inf_count = int(
            np.isinf(df[num_cols].to_numpy(dtype="float64", na_value=np.nan)).sum()
        )
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

        ctx.update(
            decimal_cols_converted=len(decimal_cols),
            bool_cols_converted=len(bool_cols),
            inf_replaced=inf_count,
        )
    return df


def _ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    """a / b with division by zero -> NaN."""
    return a / b.where(b != 0)


# ----------------------------------------------------------------------------
# Feature engineering (row-wise only, so it can never leak across rows or sets)
# ----------------------------------------------------------------------------
def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Compute row-wise engineered domain features."""
    with timed(log, "engineer_features") as ctx:
        new: dict[str, pd.Series] = {}

        # Timeline ratios relative to age
        new["id_publish_birth_ratio"] = _ratio(df["days_id_publish"], df["days_birth"])
        new["registration_birth_ratio"] = _ratio(
            df["days_registration"], df["days_birth"]
        )
        new["phone_change_birth_ratio"] = _ratio(
            df["days_last_phone_change"], df["days_birth"]
        )
        new["car_age_employed_ratio"] = _ratio(df["own_car_age"], df["employed_years"])

        # Household and money
        new["children_ratio"] = _ratio(df["cnt_children"], df["cnt_fam_members"])
        new["income_per_child"] = df["amt_income_total"] / (df["cnt_children"] + 1)
        new["credit_minus_goods"] = df["amt_credit"] - df["amt_goods_price"]
        new["annuity_vs_prev_annuity"] = _ratio(
            df["amt_annuity"], df["prev_amt_annuity_mean"]
        )
        new["bureau_credit_income_ratio"] = _ratio(
            df["bureau_credit_sum"], df["amt_income_total"]
        )

        # External scores: null-aware weighted blend, spread, and interaction with age
        ext = df[EXT_COLS]
        weights = ext.notna().to_numpy() * EXT_WEIGHTS
        weighted_sum = (ext.fillna(0).to_numpy() * weights).sum(axis=1)
        weight_total = weights.sum(axis=1)
        new["ext_source_weighted"] = pd.Series(
            np.where(
                weight_total > 0,
                weighted_sum / np.where(weight_total > 0, weight_total, 1),
                np.nan,
            ),
            index=df.index,
        )
        new["ext_source_std"] = ext.std(axis=1)
        new["ext_source_2_x_age"] = df["ext_source_2"] * df["age_years"]

        # Credit bureau enquiries and social circle
        new["bureau_enquiries_total"] = df[BUREAU_ENQUIRY_COLS].sum(axis=1, min_count=1)
        new["social_def_obs_30"] = _ratio(
            df["def_30_cnt_social_circle"], df["obs_30_cnt_social_circle"]
        )
        new["social_def_obs_60"] = _ratio(
            df["def_60_cnt_social_circle"], df["obs_60_cnt_social_circle"]
        )

        # Flag aggregates
        new["contact_flags_cnt"] = df[CONTACT_FLAG_COLS].sum(axis=1)
        new["address_mismatch_cnt"] = df[ADDRESS_MISMATCH_COLS].sum(axis=1)
        new["region_rating_diff"] = (
            df["region_rating_client_w_city"] - df["region_rating_client"]
        )

        # History shares (counts were zero-filled in the mart, so 0 denominators give NaN)
        new["prev_approval_rate"] = _ratio(df["prev_approved_cnt"], df["prev_cnt"])
        new["bureau_active_share"] = _ratio(df["bureau_active_cnt"], df["bureau_cnt"])

        out = pd.concat([df, pd.DataFrame(new)], axis=1).replace(
            [np.inf, -np.inf], np.nan
        )
        ctx.update(features_added=len(new), columns_after=out.shape[1])
        log_event(log, "engineer_features.names", level=logging.DEBUG, names=list(new))
    return out


# ----------------------------------------------------------------------------
# Fit on train, apply to both
# ----------------------------------------------------------------------------
@dataclass
class Preprocessor:
    """Preprocesses features learned from the training distribution, including scaling."""

    drop_cols: list[str]
    cat_levels: dict[str, list[str]]
    feature_names: list[str]
    numeric_cols: list[str] = field(default_factory=list)
    scaler: StandardScaler | None = None
    scale_features: bool = True

    @classmethod
    def fit(cls, X: pd.DataFrame, scale_features: bool = True) -> Preprocessor:
        """Learn column filtering, categorical encodings, and numerical scaling on train data."""
        with timed(log, "preprocessor.fit", rows=len(X), columns=X.shape[1]) as ctx:
            cols = set(X.columns)

            # 1. Known redundant columns
            explicit = {
                c
                for c in cols
                if c.startswith("flag_document_") and c not in KEEP_DOCUMENT_FLAGS
            }
            # building features come as _avg / _mode / _medi triples; keep _avg
            explicit |= {
                c
                for c in cols
                if c.endswith(("_mode", "_medi"))
                and f"{c.rsplit('_', 1)[0]}_avg" in cols
            }

            # 2. Mostly missing
            missing = X.isna().mean()
            too_missing = set(missing[missing > MISSING_THRESHOLD].index)

            # 3. Constant or almost constant
            dominant = {
                c
                for c in X.columns
                if X[c].value_counts(normalize=True, dropna=False).iloc[0]
                >= DOMINANT_THRESHOLD
            }

            drop = sorted(explicit | too_missing | dominant)
            kept = X.drop(columns=drop)
            cat_cols = kept.select_dtypes(exclude="number").columns
            cat_levels = {
                c: sorted(kept[c].dropna().unique().tolist()) for c in cat_cols
            }
            numeric_cols = [c for c in kept.columns if c not in cat_levels]

            scaler = None
            if scale_features and numeric_cols:
                with timed(log, "preprocessor.fit_scaler", n_numeric=len(numeric_cols)):
                    scaler = StandardScaler()
                    scaler.fit(kept[numeric_cols])

            ctx.update(
                dropped_total=len(drop),
                dropped_redundant=len(explicit),
                dropped_mostly_missing=len(too_missing),
                dropped_near_constant=len(dominant),
                features_kept=kept.shape[1],
                categorical_cols=len(cat_levels),
                numeric_cols_scaled=len(numeric_cols) if scaler else 0,
            )
            log_event(
                log, "preprocessor.dropped_columns", level=logging.DEBUG, columns=drop
            )
            log_event(
                log,
                "preprocessor.missing_summary",
                level=logging.DEBUG,
                worst_missing={
                    k: round(float(v), 3) for k, v in missing.nlargest(10).items()
                },
            )
        return cls(
            drop_cols=drop,
            cat_levels=cat_levels,
            feature_names=list(kept.columns),
            numeric_cols=numeric_cols,
            scaler=scaler,
            scale_features=scale_features,
        )

    def transform(self, X: pd.DataFrame, name: str = "data") -> pd.DataFrame:
        """Apply column selection, categorical encoding, and feature scaling to a dataset."""
        with timed(log, "preprocessor.transform", dataset=name, rows=len(X)):
            X = X[self.feature_names].copy()
            for c, levels in self.cat_levels.items():
                before = int(X[c].notna().sum())
                X[c] = pd.Categorical(X[c], categories=levels)  # unseen levels -> NaN
                unseen = before - int(X[c].notna().sum())
                if unseen:
                    log_event(
                        log,
                        "preprocessor.unseen_categories",
                        level=logging.WARNING,
                        dataset=name,
                        column=c,
                        rows_set_to_nan=unseen,
                    )

            if self.scaler is not None and self.numeric_cols:
                scaled_values = self.scaler.transform(X[self.numeric_cols])
                X[self.numeric_cols] = scaled_values

        return X

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "Preprocessor":
        return joblib.load(path)


# ----------------------------------------------------------------------------
# Public API & Caching
# ----------------------------------------------------------------------------
@dataclass
class DataBundle:
    """Container holding preprocessed training and testing data."""

    X_train: pd.DataFrame
    y_train: pd.Series
    ids_train: pd.Series
    X_test: pd.DataFrame
    ids_test: pd.Series
    cat_cols: list[str]
    preprocessor: Preprocessor


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    ids = df[ID_COL]
    y = (
        df[TARGET_COL].astype("int8")
        if TARGET_COL in df.columns and df[TARGET_COL].notna().all()
        else None
    )
    cols_to_drop = [c for c in [ID_COL, TARGET_COL, DATASET_COL] if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    return X, ids, y


def _load_or_create_cached_marts(
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load engineered train and test marts from fast Parquet cache or PostgreSQL."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_cache = CACHE_DIR / "engineered_mart_train.parquet"
    test_cache = CACHE_DIR / "engineered_mart_test.parquet"

    if not force_refresh and train_cache.is_file() and test_cache.is_file():
        with timed(log, "load_from_parquet_cache"):
            train = pd.read_parquet(train_cache)
            test = pd.read_parquet(test_cache)
            log_event(
                log,
                "parquet_cache.loaded",
                train_rows=len(train),
                test_rows=len(test),
                cache_path=str(CACHE_DIR),
            )
            return train, test

    # Re-fetch and engineer from PostgreSQL
    with timed(log, "fetch_and_cache_from_postgres"):
        train = _engineer(_clean_types(read_mart("mart_train")))
        test = _engineer(_clean_types(read_mart("mart_test")))

        # Save to parquet cache for subsequent runs (< 1s load)
        train.to_parquet(train_cache, index=False)
        test.to_parquet(test_cache, index=False)
        log_event(
            log, "parquet_cache.saved", train_rows=len(train), test_rows=len(test)
        )

    return train, test


def load_data(scale_features: bool = True, force_refresh: bool = False) -> DataBundle:
    """Load, clean, engineer, preprocess, and scale data from cache or PostgreSQL marts.

    Args:
        scale_features: Whether to fit and apply StandardScaler to numerical features.
        force_refresh: Whether to bypass parquet cache and re-query PostgreSQL.

    Returns:
        DataBundle with X_train, y_train, ids_train, X_test, ids_test, cat_cols, preprocessor.
    """
    with timed(
        log, "load_data", scale_features=scale_features, force_refresh=force_refresh
    ) as ctx:
        train, test = _load_or_create_cached_marts(force_refresh=force_refresh)

        X_train, ids_train, y_train = _split(train)
        X_test, ids_test, _ = _split(test)

        prep = Preprocessor.fit(X_train, scale_features=scale_features)
        X_train = prep.transform(X_train, name="train")
        X_test = prep.transform(X_test, name="test")

        if list(X_train.columns) != list(X_test.columns):
            raise AssertionError(
                "train and test feature columns differ after preprocessing"
            )

        ctx.update(
            train_rows=len(X_train),
            test_rows=len(X_test),
            n_features=X_train.shape[1],
            default_rate=round(float(y_train.mean()), 6)
            if y_train is not None
            else 0.0,
            train_memory_mb=round(X_train.memory_usage(deep=True).sum() / 1e6),
        )
        log_event(
            log,
            "load_data.dtypes",
            level=logging.DEBUG,
            counts=X_train.dtypes.astype(str).value_counts().to_dict(),
        )
        log_event(
            log,
            "load_data.categorical_columns",
            level=logging.DEBUG,
            columns=list(prep.cat_levels),
        )

    return DataBundle(
        X_train=X_train,
        y_train=y_train,
        ids_train=ids_train.reset_index(drop=True),
        X_test=X_test,
        ids_test=ids_test.reset_index(drop=True),
        cat_cols=list(prep.cat_levels),
        preprocessor=prep,
    )


def to_onnx_input(df: pd.DataFrame) -> np.ndarray:
    """Convert a preprocessed DataFrame into a 2D float32 numpy array for ONNX Runtime.

    Categorical columns are converted to integer codes (with NaN represented as -1.0).
    """
    df_onnx = df.copy()
    for col in df_onnx.columns:
        if isinstance(df_onnx[col].dtype, pd.CategoricalDtype):
            df_onnx[col] = df_onnx[col].cat.codes.astype(np.float32)
        else:
            df_onnx[col] = df_onnx[col].astype(np.float32)
    return df_onnx.to_numpy(dtype=np.float32)


if __name__ == "__main__":
    configure_logging()
    load_data()
