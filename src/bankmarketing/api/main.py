"""FastAPI application for Bank Marketing model inference."""

import logging

from fastapi import FastAPI, HTTPException, status

from bankmarketing.api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from bankmarketing.api.service import get_model, get_model_path, predict_subscription

LOGGER = logging.getLogger(__name__)

app = FastAPI(
    title="Bank Marketing Inference API",
    summary="Predict term-deposit subscription probability.",
    description=(
        "Serves the validated Kaggle competition model. The `duration` input "
        "is only available after a marketing call, so this model is not a "
        "pre-contact customer targeting system."
    ),
    version="1.0.0",
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Operations"],
)
def health() -> HealthResponse:
    """Confirm that the service can load the model artifact."""
    try:
        get_model()
    except Exception as error:
        LOGGER.exception("Model readiness check failed.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is unavailable.",
        ) from error

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_artifact=get_model_path().name,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Return a binary prediction and positive-class probability."""
    try:
        return predict_subscription(request)
    except FileNotFoundError as error:
        LOGGER.exception("Model artifact is unavailable.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is unavailable.",
        ) from error
    except Exception as error:
        LOGGER.exception("Inference failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed.",
        ) from error
