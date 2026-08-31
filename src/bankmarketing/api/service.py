"""Model loading and prediction services."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from bankmarketing.api.schemas import PredictionRequest, PredictionResponse
from bankmarketing.features.build_features import (
    FEATURE_COLUMNS,
    build_model_dataframe,
    validate_feature_dataframe,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "bank_marketing_model.joblib"
MODEL_PATH_ENV = "BANK_MARKETING_MODEL_PATH"
PREDICTION_THRESHOLD = 0.5


def get_model_path() -> Path:
    """Resolve the configured model artifact path."""
    configured_path = os.getenv(MODEL_PATH_ENV)
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return DEFAULT_MODEL_PATH


@lru_cache(maxsize=1)
def load_model(model_path: str) -> Any:
    """Load and cache the validated sklearn pipeline."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    model = joblib.load(path)
    if not hasattr(model, "predict_proba"):
        raise TypeError("Model artifact does not support probability prediction.")

    model_features = list(getattr(model, "feature_names_in_", []))
    if model_features != FEATURE_COLUMNS:
        raise ValueError(
            "Model feature schema does not match the application feature schema."
        )
    return model


def get_model() -> Any:
    """Return the cached model for the active configuration."""
    return load_model(str(get_model_path()))


def prepare_features(request: PredictionRequest) -> pd.DataFrame:
    """Build the engineered dataframe expected by the model pipeline."""
    raw_dataframe = pd.DataFrame([request.model_dump(mode="json")])
    features = build_model_dataframe(raw_dataframe, include_target=False)
    validate_feature_dataframe(features, include_target=False)
    return features


def predict_subscription(request: PredictionRequest) -> PredictionResponse:
    """Predict the probability of term-deposit subscription."""
    model = get_model()
    features = prepare_features(request)
    probabilities = np.asarray(model.predict_proba(features))
    classes = np.asarray(model.classes_)

    positive_indices = np.flatnonzero(classes == 1)
    if len(positive_indices) != 1:
        raise ValueError(
            "Model artifact does not define one positive class labelled 1."
        )

    probability = float(probabilities[0, positive_indices[0]])
    prediction = int(probability >= PREDICTION_THRESHOLD)

    return PredictionResponse(
        prediction=prediction,
        subscription_probability=probability,
        threshold=PREDICTION_THRESHOLD,
    )
