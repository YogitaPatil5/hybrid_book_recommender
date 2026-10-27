"""Shared response envelopes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str = Field(..., examples=["validation_error"])
    message: str = Field(..., examples=["Query must not be empty."])
    details: dict[str, Any] | None = None
    request_id: str | None = None


class ErrorResponse(BaseModel):
    """Every non-2xx JSON response uses this envelope."""

    error: ErrorBody
