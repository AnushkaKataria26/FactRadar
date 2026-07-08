"""
schemas.py — Pydantic request/response models for the fake news detection API.

Uses Pydantic v2 field validators to enforce input constraints that
go beyond simple type annotations (e.g., rejecting whitespace-only strings).
"""

from typing import Literal, Optional

from pydantic import BaseModel, field_validator


class PredictRequest(BaseModel):
    """Schema for POST /predict request body.

    Attributes
    ----------
    title : str
        Optional article title. Default empty string. Not used by the
        current v0.1 baseline model (trained on text column only), but
        included for forward compatibility with future models.
    text : str
        Required article body text. Must contain at least one non-whitespace
        character — enforced via a Pydantic validator, not just a type
        annotation, since a whitespace-only string passes a naive str check.
    """

    title: str = ""
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        """Reject empty or whitespace-only text."""
        if not v.strip():
            raise ValueError(
                "text field must contain at least one non-whitespace character"
            )
        return v


class PredictResponse(BaseModel):
    """Schema for POST /predict response body."""

    predicted_label: Literal["fake", "real"]
    confidence: float
    model_version: str
    warning: Optional[str] = None


class HealthResponse(BaseModel):
    """Schema for GET /health response body."""

    status: str
    model_loaded: bool


class ModelVersionResponse(BaseModel):
    """Schema for GET /model/version response body."""

    model_version: str
    trained_at: str
    metrics: dict


class TokenWeight(BaseModel):
    token: str
    weight: float


class ExplainResponse(BaseModel):
    """Schema for POST /explain response body."""

    predicted_label: Literal["fake", "real"]
    confidence: float
    top_contributing_tokens: list[TokenWeight]
    model_version: str
    warning: Optional[str] = None
