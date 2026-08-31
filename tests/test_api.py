"""Integration tests for the FastAPI inference service."""

from fastapi.testclient import TestClient

from bankmarketing.api.main import app
from bankmarketing.api.schemas import PredictionRequest
from bankmarketing.api.service import prepare_features
from bankmarketing.features.build_features import FEATURE_COLUMNS

CLIENT = TestClient(app)


def sample_payload() -> dict:
    """Return one realistic model request."""
    return {
        "age": 42,
        "job": "management",
        "marital": "married",
        "education": "tertiary",
        "default": "no",
        "balance": 1850,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "duration": 320,
        "campaign": 2,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown",
    }


def test_health_reports_loaded_model() -> None:
    """Check model readiness and artifact identity."""
    response = CLIENT.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_loaded": True,
        "model_artifact": "bank_marketing_model.joblib",
    }


def test_predict_returns_probability_and_label() -> None:
    """Check the complete prediction path with the validated artifact."""
    response = CLIENT.post("/predict", json=sample_payload())

    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] in [0, 1]
    assert 0.0 <= result["subscription_probability"] <= 1.0
    assert result["prediction"] == int(
        result["subscription_probability"] >= result["threshold"]
    )
    assert result["threshold"] == 0.5


def test_predict_normalizes_categorical_values() -> None:
    """Check that API strings receive training-time normalization."""
    payload = sample_payload()
    payload["job"] = " Management "

    response = CLIENT.post("/predict", json=payload)

    assert response.status_code == 200


def test_predict_rejects_invalid_domain_value() -> None:
    """Check categorical domain validation."""
    payload = sample_payload()
    payload["month"] = "invalid"

    response = CLIENT.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_missing_field() -> None:
    """Check required-field validation."""
    payload = sample_payload()
    del payload["duration"]

    response = CLIENT.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_unexpected_field() -> None:
    """Check that undeclared inputs cannot silently reach the model."""
    payload = sample_payload()
    payload["id"] = 123

    response = CLIENT.post("/predict", json=payload)

    assert response.status_code == 422


def test_request_builds_exact_model_feature_schema() -> None:
    """Check reuse of the established feature engineering contract."""
    request = PredictionRequest.model_validate(sample_payload())
    features = prepare_features(request)

    assert list(features.columns) == FEATURE_COLUMNS
    assert features.shape == (1, 24)
