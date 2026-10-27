"""
/api/v1 — JSON endpoints.

Endpoints are plain ``def`` (not ``async def``) on purpose: embedding and
FAISS search are CPU-bound, so FastAPI runs them in its threadpool instead
of blocking the event loop.

Validation is two-layered: FastAPI ``Query`` enforces hard type/range
bounds (documented in OpenAPI), then the engine applies the configurable
limits from settings (``BOOKMIND_MAX_TOP_K`` etc.).
"""

from __future__ import annotations

import time
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query

from bookmind.api.deps import get_app_settings, get_registry, get_semantic_engine
from bookmind.core.config import Settings
from bookmind.core.exceptions import ValidationError
from bookmind.schemas.common import ErrorResponse
from bookmind.schemas.search import (
    AutocompleteResponse,
    GenresResponse,
    SearchResponse,
    SimilarResponse,
    StatsResponse,
)
from bookmind.services.recommendation import autocomplete as merged_autocomplete
from bookmind.services.recommendation import catalogue_stats, find_similar, popular_titles
from bookmind.services.registry import ServiceRegistry
from bookmind.services.semantic_search import SemanticSearchEngine

ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    404: {"model": ErrorResponse, "description": "Book not found"},
    422: {"model": ErrorResponse, "description": "Invalid parameters"},
    429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    503: {"model": ErrorResponse, "description": "Index not built / engine unavailable"},
}

router = APIRouter(prefix="/api/v1", tags=["books"], responses=ERROR_RESPONSES)

Engine = Annotated[SemanticSearchEngine, Depends(get_semantic_engine)]
Registry = Annotated[ServiceRegistry, Depends(get_registry)]
AppSettings = Annotated[Settings, Depends(get_app_settings)]


@router.get("/search", response_model=SearchResponse, summary="Semantic search or genre browsing")
def search_books(
    engine: Engine,
    settings: AppSettings,
    q: Annotated[
        str,
        Query(
            max_length=5000,
            description="Title, author or free-text description. Optional when a genre is "
            "given: an empty query returns the most-read books in that genre.",
            examples=["a slow-burn mystery set in winter"],
        ),
    ] = "",
    genre: Annotated[str | None, Query(max_length=100, description="Genre filter")] = None,
    top_k: Annotated[int | None, Query(ge=1, le=500, description="Number of results")] = None,
) -> SearchResponse:
    started = time.perf_counter()
    k = top_k or settings.default_top_k
    if q.strip():
        results = engine.search(q, genre=genre, top_k=k)
    elif genre and genre.strip().lower() != "all":
        results = engine.browse(genre, top_k=k)
    else:
        raise ValidationError(
            "Provide a search query (q), a genre, or both.", details={"field": "q"}
        )
    return SearchResponse(
        query=" ".join(q.split()),
        genre=genre or "All",
        count=len(results),
        took_ms=round((time.perf_counter() - started) * 1000, 2),
        results=results,
    )


@router.get("/books/similar", response_model=SimilarResponse, summary="Books similar to a title")
def similar_books(
    registry: Registry,
    title: Annotated[str, Query(min_length=1, max_length=5000, examples=["The Hobbit"])],
    top_k: Annotated[int, Query(ge=1, le=500)] = 5,
    engine: Annotated[
        Literal["auto", "semantic", "knn"],
        Query(description="auto = k-NN if the title is known to it, else semantic"),
    ] = "auto",
) -> SimilarResponse:
    started = time.perf_counter()
    source, used, results = find_similar(registry, title, top_k=top_k, engine=engine)
    return SimilarResponse(
        source_title=source,
        engine=used,
        count=len(results),
        took_ms=round((time.perf_counter() - started) * 1000, 2),
        results=results,
    )


@router.get("/autocomplete", response_model=AutocompleteResponse, summary="Title suggestions")
def autocomplete(
    registry: Registry,
    settings: AppSettings,
    q: Annotated[str, Query(max_length=300)] = "",
    limit: Annotated[int | None, Query(ge=1, le=50)] = None,
    genre: Annotated[
        str | None, Query(max_length=100, description="Only used when q is empty")
    ] = None,
) -> AutocompleteResponse:
    """
    Title suggestions from both catalogues (semantic + collaborative filtering).

    Empty ``q`` returns *discovery* suggestions — the most popular titles
    (optionally within ``genre``) — so users who don't know what to search for
    still get a useful dropdown.
    """
    n = limit or settings.autocomplete_limit
    if not q.strip():
        titles, label = popular_titles(registry, n, genre)
        return AutocompleteResponse(query=q, suggestions=titles, mode="popular", label=label)
    return AutocompleteResponse(query=q, suggestions=merged_autocomplete(registry, q, n))


@router.get("/genres", response_model=GenresResponse, summary="Available genres")
def list_genres(engine: Engine) -> GenresResponse:
    return GenresResponse(genres=engine.list_genres())


@router.get("/stats", response_model=StatsResponse, summary="Catalogue statistics")
def stats(registry: Registry) -> StatsResponse:
    return catalogue_stats(registry)


# ── Backwards compatibility with the v1 Flask app's autocomplete ─────────────
legacy_router = APIRouter(tags=["legacy"])


@legacy_router.get(
    "/api/search", response_model=list[str], deprecated=True, summary="Legacy autocomplete"
)
def legacy_autocomplete(
    registry: Registry,
    settings: AppSettings,
    q: Annotated[str, Query(max_length=300)] = "",
) -> list[str]:
    if registry.semantic is None and registry.knn is None:
        return []
    return merged_autocomplete(registry, q, settings.autocomplete_limit)
