"""
Pydantic request / response schemas for the prediction API.
"""
from typing import Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
# Requests
# ──────────────────────────────────────────────────────────────
class PredictTextRequest(BaseModel):
    """Request body for text-based prediction."""
    title: str = Field(
        ...,
        min_length=1,
        description="Headline / title of the news article",
        examples=["Breaking: Government announces new economic policy"],
    )
    text: str = Field(
        ...,
        min_length=1,
        description="Full body text of the news article",
        examples=["The finance minister introduced a new policy today after discussions in parliament..."],
    )





# ──────────────────────────────────────────────────────────────
# Responses
# ──────────────────────────────────────────────────────────────



class PredictionResponse(BaseModel):
    """Unified prediction response (ML result + optional fact-check)."""
    prediction: str = Field(
        ...,
        description="Human-readable label: 'Real' or 'Fake'",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence (probability of the predicted class)",
    )
    label: int = Field(
        ...,
        description="Numeric label: 0 = Fake, 1 = Real",
    )



class HealthResponse(BaseModel):
    """Health-check response."""
    status: str = "ok"
    is_model_loaded: bool = False
