"""Pydantic models: the public contract of the API."""

from bookmind.schemas.book import Book, BuyLinks, PopularBook
from bookmind.schemas.common import ErrorBody, ErrorResponse
from bookmind.schemas.search import (
    AutocompleteResponse,
    GenresResponse,
    HealthResponse,
    ReadinessResponse,
    SearchResponse,
    SimilarResponse,
    StatsResponse,
)

__all__ = [
    "AutocompleteResponse",
    "Book",
    "BuyLinks",
    "ErrorBody",
    "ErrorResponse",
    "GenresResponse",
    "HealthResponse",
    "PopularBook",
    "ReadinessResponse",
    "SearchResponse",
    "SimilarResponse",
    "StatsResponse",
]
