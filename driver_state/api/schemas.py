"""Pydantic request/response schemas for the driver-state API."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..signals import N_FEATURES


class HealthResponse(BaseModel):
    status: str
    serving_model: str
    n_features: int
    window_len: int


class WindowRequest(BaseModel):
    # window: [window_len, N_FEATURES] raw signals (see /schema for column order)
    window: list[list[float]] = Field(..., description="[T, features] raw signals")


class DriverStateResponse(BaseModel):
    fatigue_class: int
    fatigue_state: str
    fatigue_score: float
    confidence: float
    probabilities: dict[str, float]
    model: str


class PolicyRequest(WindowRequest):
    is_night: bool = False
    time_on_task_min: float | None = None
    stress: float | None = Field(None, ge=0, le=1)


class PolicyResponse(BaseModel):
    driver_state: DriverStateResponse
    policy: dict


class SchemaResponse(BaseModel):
    feature_columns: list[str]
    window_len: int
    n_features: int = N_FEATURES
