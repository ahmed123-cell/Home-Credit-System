# Home Credit Default Risk System

An end-to-end production-grade machine learning system for credit default risk scoring, featuring:
- **Raw Data Ingestion**: Automated ingestion of the Home Credit dataset into PostgreSQL.
- **Analytics Engineering (dbt)**: Robust feature marts (`mart_train`, `mart_test`, `mart_application_features`) with incremental materialization.
- **Model Training & Experimentation**: Automated LightGBM & XGBoost training pipeline with hyperparameter configs, automated class-imbalance weighting, and MLflow tracking.
- **Model Optimization & Export**: Fast ONNX export for low-latency scoring.
- **Real-Time API & Interactive UI**: High-performance FastAPI server with vectorized batch predictions and a modern web dashboard.
- **Multi-Platform Containerization**: Runs out-of-the-box on **any device** (macOS Apple Silicon/Intel, Windows WSL2/native, Linux x86_64/ARM64) via Docker and Docker Compose.

---

## 🚀 Quickstart with Docker Compose (Recommended)

Start the entire stack (PostgreSQL 17, MLflow, and FastAPI + Web UI) with a single command:

```bash
docker compose up -d
```

### Accessing the Services

| Service | URL | Description |
| :--- | :--- | :--- |
| **Interactive Web UI** | [http://localhost:8000/ui](http://localhost:8000/ui) | Real-time credit risk scoring interface |
| **API Documentation** | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive Swagger API docs |
| **API Health Check** | [http://localhost:8000/health](http://localhost:8000/health) | Model status, uptime, and validation metrics |
| **MLflow UI** | [http://localhost:5000](http://localhost:5000) | Experiment runs, metrics, and registered models |
| **PostgreSQL Database** | `localhost:5432` | Data warehouse (DB: `home_credit`) |

To stop the services:
```bash
docker compose down
```

---

## 🌪️ Running Apache Airflow (Pipeline Orchestration)

To spin up the complete **Apache Airflow orchestration cluster** alongside the core stack:

```bash
docker compose --profile airflow up -d
```

### Accessing Airflow UI
- **URL**: [http://localhost:8080](http://localhost:8080)
- **Username**: `admin`
- **Password**: `admin`

### Automated End-to-End DAG (`home_credit_pipeline`)
The DAG orchestrates the complete machine learning lifecycle:
1. `check_db_connection`: Validates PostgreSQL data warehouse connectivity.
2. `load_raw_data`: Ingests CSVs into PostgreSQL `raw` schema.
3. `dbt_run_staging`: Builds dbt staging models.
4. `dbt_run_intermediate`: Builds incremental intermediate aggregations.
5. `dbt_run_marts`: Materializes training and testing feature marts.
6. `dbt_test`: Runs data quality and integrity tests.
7. `train_lightgbm_model`: Retrains LightGBM model with dynamic class weights and logs run to MLflow.
8. `validate_onnx_artifacts`: Verifies output ONNX model and preprocessor artifacts.

To stop the Airflow cluster:
```bash
docker compose --profile airflow down
```

---

## 🛠️ Running Pipeline Tasks with Docker

All CLI tools and dependencies are packaged inside the container. You can execute tasks without installing Python or packages locally:

### 1. Run Tests
```bash
docker compose run --rm app uv run pytest -v
```

### 2. Run Data Ingestion
```bash
docker compose run --rm app uv run python -m ingestion.load_raw
```

### 3. Run dbt Transformations
```bash
docker compose run --rm app uv run dbt run --project-dir dbt --profiles-dir dbt
```

### 4. Train a Model (LightGBM or XGBoost)
```bash
docker compose run --rm app uv run python -m training.train --model lightgbm
```

---

## 🐳 Standalone Docker Container

If you only need the inference server and interactive UI without the database or MLflow:

```bash
# Build the image
docker build -t home-credit-app .

# Run the container
docker run -p 8000:8000 -v $(pwd)/models:/app/models:ro home-credit-app
```

---

## 💻 Local Development Setup (Without Docker)

If developing directly on your host machine:

### 1. Install `uv` & Synchronize Dependencies
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

### 3. Run the Inference Application
```bash
uv run uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📊 API Usage Examples

### Single Applicant Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "name_contract_type": "Cash loans",
      "amt_credit": 500000.0,
      "amt_income_total": 150000.0,
      "amt_annuity": 25000.0,
      "days_birth": -12775,
      "days_employed": -1825,
      "ext_source_2": 0.5
    },
    "threshold": 0.5
  }'
```

### Batch Prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "name_contract_type": "Cash loans",
        "amt_credit": 400000.0,
        "amt_income_total": 180000.0,
        "amt_annuity": 20000.0,
        "days_birth": -14000,
        "ext_source_2": 0.6
      },
      {
        "name_contract_type": "Cash loans",
        "amt_credit": 800000.0,
        "amt_income_total": 90000.0,
        "amt_annuity": 45000.0,
        "days_birth": -9000,
        "ext_source_2": 0.2
      }
    ],
    "threshold": 0.5
  }'
```
