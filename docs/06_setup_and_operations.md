# 06. Setup & Operations Guide: How to Run the Whole Project

This guide provides step-by-step instructions for installing, configuring, running, and operating the complete **Home Credit Default Risk System** across different environments (macOS, Windows with WSL2, and native Linux).

---

## 1. System Requirements & Prerequisites

- **Docker & Docker Compose**: Docker 24.0+ and Docker Compose v2.20+ (Docker Desktop on Windows/macOS or Docker Engine on Linux).
- **RAM**: Minimum 8 GB (16 GB recommended when running the full Airflow orchestration profile).
- **Disk Space**: At least 15 GB free disk space (to accommodate PostgreSQL volumes, models, and Docker images).
- **Local Dev Tools (Optional)**: Python 3.12 and [Astral `uv`](https://github.com/astral-sh/uv) if running outside Docker.

---

## 2. Configuration & Environment Variables

Clone the repository and prepare your environment configuration:

```bash
git clone https://github.com/<your-username>/home-credit-system.git
cd home-credit-system

# Create .env from the provided example template
cp .env.example .env
```

### Key Variables in [`.env`]:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `POSTGRES_USER` | `home_credit` | PostgreSQL warehouse username |
| `POSTGRES_PASSWORD` | `home_credit_password` | PostgreSQL warehouse password |
| `POSTGRES_DB` | `home_credit` | Data warehouse database name |
| `POSTGRES_PORT` | `5432` | Exposed PostgreSQL port |
| `APP_PORT` | `8000` | Port for FastAPI & Web UI |
| `MLFLOW_PORT` | `5000` | Port for MLflow tracking server |
| `AIRFLOW_UID` | `1000` | Container UID matching your host user to ensure volume write permissions |

---

## 3. Running with Docker Compose (Recommended)

The system supports two execution profiles:

### Mode 1: Core Service Stack (Lightweight)
Launches the PostgreSQL data warehouse, the MLflow tracking server, and the real-time FastAPI inference engine with the Web UI:

```bash
docker compose up -d
```

**Services Started:**
- **PostgreSQL Data Warehouse**: `localhost:5432`
- **MLflow Tracking Server**: [http://localhost:5000](http://localhost:5000)
- **FastAPI Inference Service**: [http://localhost:8000](http://localhost:8000)
- **Interactive Scoring Dashboard**: [http://localhost:8000/ui](http://localhost:8000/ui)
- **Interactive Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Mode 2: Full Airflow Orchestration Stack
Launches the entire system including the Apache Airflow 3 CeleryExecutor cluster for end-to-end automated retraining and ETL:

```bash
docker compose --profile airflow up -d
```

**Additional Services Started:**
- **Airflow Web UI**: [http://localhost:8080](http://localhost:8080)
  - **Username**: `admin`
  - **Password**: `admin`
- **Airflow Scheduler & Worker**: Automatically polls and executes pipelines.
- **Airflow PostgreSQL & Redis Broker**: Metadata and task queuing backends.

---

## 4. Triggering the End-to-End Pipeline via Airflow

Once the Airflow profile is running:
1. Open the Airflow Web UI at [http://localhost:8080](http://localhost:8080).
2. Log in with `admin` / `admin`.
3. Locate the `home_credit_pipeline` DAG and ensure it is **unpaused** (toggle switch on).
4. Click **Trigger DAG** (the Play icon) in the top-right corner.

The pipeline will execute the 8 stages sequentially:
1. `check_db_connection` &rarr; Verifies PostgreSQL connectivity.
2. `load_raw_data` &rarr; Ingests CSV files into the `raw` schema.
3. `dbt_run_staging` &rarr; Builds clean staging views.
4. `dbt_run_intermediate` &rarr; Builds feature aggregations.
5. `dbt_run_marts` &rarr; Materializes 205+ features in `mart_application_features`.
6. `dbt_test` &rarr; Validates all 69 data quality assertions.
7. `train_lightgbm_model` &rarr; Retrains the LightGBM model, logs to MLflow, and exports to ONNX.
8. `validate_onnx_artifacts` &rarr; Validates the generated model artifact.

---

## 5. Local Development Workflow (Without Docker)

If you prefer developing directly on your host machine using Python:

### 1. Install Dependencies with `uv`
```bash
# Install uv if not already present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.12 and all dependencies (including dev tools)
uv sync --group dev
```

### 2. Start PostgreSQL Warehouse
```bash
docker compose up -d postgres
```

### 3. Run Ingestion & dbt Warehouse Transformations
```bash
# Ingest raw CSV data into PostgreSQL
uv run python -m ingestion.load_raw

# Run dbt transformations
uv run dbt run --project-dir dbt --profiles-dir dbt

# Run the 69 dbt data quality tests
uv run dbt test --project-dir dbt --profiles-dir dbt
```

### 4. Retrain the Model
```bash
uv run python -m training.train --model lightgbm
```

### 5. Launch the FastAPI Application
```bash
uv run uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Open [http://localhost:8000/ui](http://localhost:8000/ui) in your browser.

### 6. Run Code Quality Checks & Tests
```bash
# Format code
uv run ruff format .

# Run linter
uv run ruff check .

# Run unit and integration tests with coverage
uv run pytest tests/ -v --cov=src --cov=app
```

---

## 6. Operational Troubleshooting & FAQs

### Q1: Permission denied on `airflow/logs`
**Cause**: The Airflow worker needs write access to the mounted `./airflow/logs` directory.  
**Fix**: Ensure `AIRFLOW_UID=1000` is present in your `.env` file (matches your host user ID) and run:
```bash
chmod -R 777 airflow/logs
docker compose --profile airflow up -d
```

### Q2: MLflow rejects requests with 403 Forbidden
**Cause**: MLflow 3.x host header validation blocks requests from internal Docker hostnames.  
**Fix**: Ensure `--allowed-hosts "*"` is set in the `mlflow` service definition in `docker-compose.yml`.

### Q3: Rebuilding images after modifying code or dependencies
If you update dependencies in `pyproject.toml` or `Dockerfile.airflow`:
```bash
# Rebuild the core app image
docker compose build app

# Rebuild the airflow image
docker compose --profile airflow build airflow-worker
```

### Q4: Resetting the data warehouse completely
```bash
# Stop all containers and remove persistent volumes
docker compose --profile airflow down -v

# Restart fresh
docker compose up -d
```
