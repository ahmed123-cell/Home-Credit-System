"""Tests for FastAPI inference application."""

import pytest
from fastapi.testclient import TestClient

from app import app

VALID = {
    "name_contract_type": "Cash loans",
    "amt_credit": 500000.0,
    "amt_income_total": 150000.0,
    "amt_annuity": 25000.0,
    "days_birth": -35 * 365.25,  # the API takes days; the UI converts from years
    "days_employed": -5 * 365.25,
    "ext_source_2": 0.5,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs lifespan once: loads model + preprocessor
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["n_features"] > 0
    assert "X-Process-Time-Ms" in response.headers


def test_schema_endpoint(client):
    data = client.get("/schema").json()
    assert "name_contract_type" in data["categories"]
    assert data["n_features"] > 0


def test_single_prediction(client):
    response = client.post("/predict", json={"features": VALID, "threshold": 0.5})
    assert response.status_code == 200, response.text
    data = response.json()
    assert 0.0 <= data["probability"] <= 1.0
    assert data["prediction"] in (0, 1)
    assert data["risk_tier"]
    assert data["inference_time_ms"] > 0.0
    assert data["preprocess_time_ms"] > 0.0
    assert data["total_time_ms"] > 0.0
    assert "X-Process-Time-Ms" in response.headers


def test_derived_fields_sent_by_client_are_ignored(client):
    base = client.post("/predict", json={"features": VALID}).json()["probability"]
    tampered = {**VALID, "ext_source_mean": 0.99, "age_years": 5, "credit_term": 1}
    assert (
        client.post("/predict", json={"features": tampered}).json()["probability"]
        == base
    )


def test_batch_prediction(client):
    records = [
        VALID,
        {
            **VALID,
            "amt_credit": 800000.0,
            "amt_income_total": 90000.0,
            "ext_source_2": 0.2,
        },
        {
            **VALID,
            "amt_credit": 250000.0,
            "amt_income_total": 220000.0,
            "ext_source_2": 0.7,
        },
    ]
    response = client.post(
        "/predict/batch", json={"records": records, "threshold": 0.5}
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["batch_size"] == 3
    assert len(data["predictions"]) == 3
    assert data["total_inference_time_ms"] > 0.0
    assert data["avg_per_sample_ms"] > 0.0
    assert "X-Process-Time-Ms" in response.headers


# ---------------- validation ----------------
def _errors(response):
    assert response.status_code == 422
    return response.json()["detail"]["errors"]


@pytest.mark.parametrize(
    "change, field, text",
    [
        (
            {"days_birth": 30 * 365.25},
            "days_birth",
            "Age must be between 18 and 100 years",
        ),  # age -30
        ({"amt_income_total": 0}, "amt_income_total", "greater than 0"),
        ({"amt_credit": "lots"}, "amt_credit", "must be a number"),
        ({"ext_source_1": 1.7}, "ext_source_1", "between 0 and 1"),
        ({"cnt_children": 2.5}, "cnt_children", "whole number"),
        (
            {"name_contract_type": "Magic loans"},
            "name_contract_type",
            "isn't recognised",
        ),
        (
            {"bureau_cnt": 2, "bureau_active_cnt": 5},
            "bureau_active_cnt",
            "can't be more than",
        ),
    ],
)
def test_invalid_input_gives_readable_error(client, change, field, text):
    errors = _errors(client.post("/predict", json={"features": {**VALID, **change}}))
    match = [e for e in errors if e["field"] == field]
    assert match and text in match[0]["message"]


def test_missing_required_fields(client):
    errors = _errors(client.post("/predict", json={"features": {"amt_credit": 1000}}))
    assert {"days_birth", "amt_income_total", "amt_annuity", "name_contract_type"} <= {
        e["field"] for e in errors
    }


def test_batch_error_names_the_row(client):
    records = [VALID, {**VALID, "ext_source_2": 5}, VALID]
    errors = _errors(client.post("/predict/batch", json={"records": records}))
    assert errors[0]["row"] == 2
