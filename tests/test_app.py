"""Tests for FastAPI inference application."""

from fastapi.testclient import TestClient

from app import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_type" in data
        assert "n_features" in data
        assert data["n_features"] > 0
        assert "X-Process-Time-Ms" in response.headers


def test_single_prediction():
    with TestClient(app) as client:
        payload = {
            "features": {
                "amt_credit": 500000.0,
                "amt_income_total": 150000.0,
                "ext_source_mean": 0.5,
                "ext_source_weighted": 0.45,
                "credit_term": 20.0,
                "age_years": 35.0,
            },
            "threshold": 0.5,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "probability" in data
        assert 0.0 <= data["probability"] <= 1.0
        assert data["prediction"] in (0, 1)
        assert "risk_tier" in data
        assert data["inference_time_ms"] > 0.0
        assert data["total_time_ms"] > 0.0
        assert "X-Process-Time-Ms" in response.headers


def test_batch_prediction():
    with TestClient(app) as client:
        payload = {
            "records": [
                {
                    "amt_credit": 400000.0,
                    "amt_income_total": 180000.0,
                    "ext_source_mean": 0.6,
                },
                {
                    "amt_credit": 800000.0,
                    "amt_income_total": 90000.0,
                    "ext_source_mean": 0.2,
                },
                {
                    "amt_credit": 250000.0,
                    "amt_income_total": 220000.0,
                    "ext_source_mean": 0.7,
                },
            ],
            "threshold": 0.5,
        }
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["batch_size"] == 3
        assert len(data["predictions"]) == 3
        assert data["total_inference_time_ms"] > 0.0
        assert data["avg_per_sample_ms"] > 0.0
        assert "X-Process-Time-Ms" in response.headers
