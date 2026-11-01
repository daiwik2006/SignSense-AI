"""Pydantic schemas for the API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app_env: str
    model_ready: bool


class ModelInfoResponse(BaseModel):
    model_path: str
    model_ready: bool
    num_classes: Optional[int] = None
    labels: List[str] = Field(default_factory=list)
    feature_dim: Optional[int] = None
    confidence_threshold: Optional[float] = None


class FrameMessage(BaseModel):
    """A frame received over WebSocket (base64 JPEG variant)."""

    type: str = Field(pattern="^frame$")
    image: str  # data URI or raw base64 JPEG


class PredictionResponse(BaseModel):
    type: str = Field(pattern="^prediction$")
    label: Optional[str] = None
    confidence: float = 0.0
    probabilities: Dict[str, float] = Field(default_factory=dict)
    accepted_label: Optional[str] = None
    fps: float = 0.0
    annotated: str = ""  # base64 JPEG with landmarks drawn