"""
Domain exception hierarchy.

Every error the service raises intentionally derives from ``BookMindError``
and carries an HTTP status, a stable machine-readable ``code`` and a
user-safe ``message``. Internal details go to logs, never to clients.
"""

from __future__ import annotations

from typing import Any


class BookMindError(Exception):
    """Base class for all application errors."""

    status_code: int = 500
    code: str = "internal_error"
    default_message: str = "An unexpected error occurred."

    def __init__(self, message: str | None = None, *, details: dict[str, Any] | None = None):
        self.message = message or self.default_message
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


class ConfigurationError(BookMindError):
    code = "configuration_error"
    default_message = "The service is misconfigured."


class ArtifactNotFoundError(BookMindError):
    status_code = 503
    code = "artifacts_missing"
    default_message = "Search index is not built yet. Run the data pipeline first."


class ArtifactIntegrityError(BookMindError):
    status_code = 503
    code = "artifacts_corrupt"
    default_message = "Search artifacts are inconsistent. Rebuild the index."


class ServiceUnavailableError(BookMindError):
    status_code = 503
    code = "service_unavailable"
    default_message = "This feature is temporarily unavailable."


class ValidationError(BookMindError):
    status_code = 422
    code = "validation_error"
    default_message = "The request is invalid."


class BookNotFoundError(BookMindError):
    status_code = 404
    code = "book_not_found"
    default_message = "The requested book was not found in the catalogue."


class EmbeddingError(BookMindError):
    status_code = 500
    code = "embedding_error"
    default_message = "Failed to process the search query."


class DataValidationError(BookMindError):
    """Raised by the offline pipeline when an input dataset is unusable."""

    status_code = 400
    code = "data_validation_error"
    default_message = "Input data failed validation."
