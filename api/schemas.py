"""Pydantic schemas for the production API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded RGB image")


class PredictResponse(BaseModel):
    predicted_grade: int
    severity_label: str
    confidence: float
    continuous_prediction: float
    gradcam_heatmap: str | None = None


class ExplainRequest(BaseModel):
    image: str
    method: str = Field(default="gradcam", pattern="^(gradcam|gradcam\\+\\+|eigencam|scorecam|layercam|ablationcam|lime|shap)$")


class AttentionBreakdownResponse(BaseModel):
    macula_pct: float
    vessels_pct: float
    optic_disc_pct: float
    background_pct: float


class ExplainResponse(BaseModel):
    heatmap_overlay: str
    landmark_overlay: str
    multiclass_grid: str | None = None
    xai_reliability_score: float
    clinical_flags: list[str]
    attention_breakdown: AttentionBreakdownResponse


class MethodInfo(BaseModel):
    method: str
    description: str
    avg_latency_ms: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
