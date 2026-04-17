"""
API routes for the Fake News Detection service.
"""
import logging
from fastapi import APIRouter, HTTPException

from backend.app.models.schemas import (
    PredictTextRequest,
    PredictionResponse,
    HealthResponse,
)
from backend.app.services import ml_service
from backend.app.services.text_processing import clean_text

logger = logging.getLogger(__name__)

router = APIRouter()


# ──────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
async def health_check():
    return HealthResponse(
        status="ok",
        is_model_loaded=ml_service.is_model_loaded(),
    )


# ──────────────────────────────────────────────────────────────
# Predict — raw text (with grey-area agent trigger)
# ──────────────────────────────────────────────────────────────
@router.post(
    "/predict/text",
    response_model=PredictionResponse,
    summary="Predict from raw text",
    tags=["Prediction"],
)
async def predict_text(body: PredictTextRequest):
    """
    Run the ML pipeline on the submitted article.
    """
    try:
        # ML Prediction
        result = ml_service.predict_news(title=body.title, text=body.text)

        return PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            label=result["label"],
        )

    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


