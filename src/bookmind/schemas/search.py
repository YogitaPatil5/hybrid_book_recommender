"""Search / recommendation request & response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from bookmind.schemas.book import Book


class SearchResponse(BaseModel):
    query: str
    genre: str
    count: int = Field(..., ge=0)
    took_ms: float = Field(..., ge=0)
    results: list[Book]


class SimilarResponse(BaseModel):
    source_title: str = Field(..., description="Catalogue title that matched the request")
    engine: Literal["semantic", "knn"]
    count: int = Field(..., ge=0)
    took_ms: float = Field(..., ge=0)
    results: list[Book]


class AutocompleteResponse(BaseModel):
    query: str
    suggestions: list[str]
    mode: Literal["match", "popular"] = Field(
        "match", description="'popular' when the query is empty (discovery suggestions)"
    )
    label: str | None = Field(None, description="Heading for the dropdown, e.g. 'Popular books'")


class GenresResponse(BaseModel):
    genres: list[str]


class StatsResponse(BaseModel):
    total_books: int = Field(..., ge=0)
    total_genres: int = Field(..., ge=0)
    total_ratings: int = Field(..., ge=0)
    total_users: int = Field(..., ge=0)
    embedding_model: str | None = None
    index_built_at: str | None = None
    knn_available: bool
    knn_books: int = Field(0, ge=0)
    knn_mode: str | None = None
    knn_trained_at: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    env: str


class ComponentStatus(BaseModel):
    ready: bool
    detail: str | None = None


class ReadinessResponse(BaseModel):
    status: Literal["ready", "degraded", "unavailable"]
    components: dict[str, ComponentStatus]
