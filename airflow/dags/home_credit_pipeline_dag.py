"""Home Credit Default Risk End-to-End Orchestration DAG.

Orchestrates the entire machine learning data lifecycle:
1. Verifies connectivity to the PostgreSQL data warehouse.
2. Loads raw CSV data into PostgreSQL (`raw` schema).
3. Executes dbt models in dependency order: Staging -> Intermediate -> Marts.
4. Executes dbt data quality test suite.
5. Trains LightGBM model, logs runs to MLflow, and exports to ONNX.
6. Validates the generated ONNX model, metadata, and preprocessor artifacts.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from airflow import DAG

try:
    from airflow.providers.standard.operators.bash import BashOperator
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:
    from airflow.operators.bash import BashOperator
    from airflow.operators.python import PythonOperator

log = logging.getLogger("airflow.home_credit")

PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/opt/airflow/project"))
DBT_DIR = PROJECT_DIR / "dbt"
MODELS_DIR = PROJECT_DIR / "models"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def test_warehouse_connection() -> None:
    """Validate connection to the PostgreSQL data warehouse."""
    import psycopg

    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://home_credit:home_credit_password@postgres:5432/home_credit",
    )
    log.info("Testing connection to database at %s", db_url.split("@")[-1])
    with psycopg.connect(db_url) as conn, conn.cursor() as cur:
        cur.execute("SELECT 1;")
        result = cur.fetchone()
        assert result and result[0] == 1, "Database query returned unexpected result"
    log.info("PostgreSQL database connection successful.")


def validate_artifacts() -> None:
    """Verify that all required model artifacts exist and can be loaded."""
    import json

    import joblib
    import onnxruntime as ort

    onnx_path = MODELS_DIR / "lightgbm.onnx"
    prep_path = MODELS_DIR / "lightgbm_preprocessor.joblib"
    meta_path = MODELS_DIR / "lightgbm_meta.json"

    for path in (onnx_path, prep_path, meta_path):
        if not path.is_file():
            raise FileNotFoundError(f"Expected artifact not found: {path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    prep = joblib.load(prep_path)
    session = ort.InferenceSession(str(onnx_path))
    input_shape = session.get_inputs()[0].shape

    log.info("Artifact validation succeeded:")
    log.info(" - Model type: %s", meta.get("model_type"))
    log.info(" - Features: %d", len(prep.feature_names))
    log.info(" - ONNX input tensor: %s", input_shape)
    log.info(" - Validation ROC-AUC: %s", meta.get("metrics", {}).get("val_roc_auc"))


with DAG(
    dag_id="home_credit_pipeline",
    default_args=default_args,
    description="End-to-End Home Credit ingestion, dbt transformation, and model retraining",
    schedule=None,  # Trigger on-demand or weekly: "@weekly"
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    tags=["home-credit", "mlops", "dbt", "lightgbm"],
) as dag:
    # 1. Connectivity Check
    check_db = PythonOperator(
        task_id="check_db_connection",
        python_callable=test_warehouse_connection,
    )

    # 2. Raw Ingestion
    load_raw = BashOperator(
        task_id="load_raw_data",
        bash_command=f"cd {PROJECT_DIR} && python -m ingestion.load_raw",
    )

    # 3. dbt Staging Layer
    dbt_staging = BashOperator(
        task_id="dbt_run_staging",
        bash_command=f"dbt run --select staging --project-dir {DBT_DIR} --profiles-dir {DBT_DIR}",
    )

    # 4. dbt Intermediate Layer
    dbt_intermediate = BashOperator(
        task_id="dbt_run_intermediate",
        bash_command=f"dbt run --select intermediate --project-dir {DBT_DIR} --profiles-dir {DBT_DIR}",
    )

    # 5. dbt Marts Layer
    dbt_marts = BashOperator(
        task_id="dbt_run_marts",
        bash_command=f"dbt run --select marts --project-dir {DBT_DIR} --profiles-dir {DBT_DIR}",
    )

    # 6. dbt Data Quality Tests
    dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command=f"dbt test --project-dir {DBT_DIR} --profiles-dir {DBT_DIR}",
    )

    # 7. Model Training & MLflow Logging
    train_model = BashOperator(
        task_id="train_lightgbm_model",
        bash_command=f"cd {PROJECT_DIR} && python -m training.train --model lightgbm",
    )

    # 8. Artifact Validation
    validate_model = PythonOperator(
        task_id="validate_onnx_artifacts",
        python_callable=validate_artifacts,
    )

    # Task dependency graph
    (
        check_db
        >> load_raw
        >> dbt_staging
        >> dbt_intermediate
        >> dbt_marts
        >> dbt_test
        >> train_model
        >> validate_model
    )
