"""
HTML pages.

User-facing errors (bad input, unknown title, index not built) are shown
inline on the page with the proper HTTP status, rather than a stack trace
or a blank page.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, Query, Request
from fastapi.responses import HTMLResponse

from bookmind.api.deps import get_app_settings, get_registry
from bookmind.core.config import Settings
from bookmind.core.exceptions import BookMindError
from bookmind.schemas.book import Book
from bookmind.services.recommendation import catalogue_stats, find_similar, popular_books
from bookmind.services.registry import ServiceRegistry
from bookmind.services.validation import ALL_GENRES

log = logging.getLogger(__name__)

router = APIRouter(include_in_schema=False)

Registry = Annotated[ServiceRegistry, Depends(get_registry)]
AppSettings = Annotated[Settings, Depends(get_app_settings)]


def _render(
    request: Request, template: str, context: dict[str, Any], status_code: int = 200
) -> HTMLResponse:
    templates = request.app.state.templates
    registry: ServiceRegistry = request.app.state.registry
    base = {
        "search_ready": registry.semantic is not None,
        "knn_available": registry.knn is not None,
    }
    response: HTMLResponse = templates.TemplateResponse(
        request, template, {**base, **context}, status_code=status_code
    )
    return response


@router.get("/", response_class=HTMLResponse, name="home")
def home(request: Request, registry: Registry, settings: AppSettings) -> HTMLResponse:
    return _render(
        request,
        "index.html",
        {
            "popular_books": popular_books(registry, settings.popular_books_count),
            "stats": catalogue_stats(registry),
        },
    )


@router.get("/search", response_class=HTMLResponse, name="search")
def search_page(
    request: Request,
    registry: Registry,
    settings: AppSettings,
    q: Annotated[str, Query(max_length=5000)] = "",
    genre: Annotated[str, Query(max_length=100)] = ALL_GENRES,
) -> HTMLResponse:
    query = q.strip()
    results: list[Book] = []
    error: str | None = None
    status = 200
    genres = registry.semantic.list_genres() if registry.semantic else []

    if registry.semantic is None:
        error, status = "Search is not available yet — the search index hasn't been built.", 503
    elif query or (genre and genre.strip().lower() != ALL_GENRES.lower()):
        try:
            if query:
                results = registry.semantic.search(query, genre=genre, top_k=settings.default_top_k)
            else:  # genre-only browsing: no text needed
                results = registry.semantic.browse(genre, top_k=settings.default_top_k)
            if not results:
                error = (
                    "No books matched your search. Try a different title, genre, or description."
                )
        except BookMindError as exc:
            error, status = exc.message, exc.status_code

    return _render(
        request,
        "search.html",
        {
            "query": query,
            "selected_genre": genre or ALL_GENRES,
            "all_genres": genres,
            "results": results,
            "error": error,
        },
        status_code=status,
    )


def _recommend(request: Request, registry: Registry, title: str) -> HTMLResponse:
    recommendations: list[Book] = []
    error: str | None = None
    engine_used: str | None = None
    source = title.strip()
    status = 200
    if source:
        try:
            source, engine_used, recommendations = find_similar(registry, source, top_k=5)
        except BookMindError as exc:
            error, status = exc.message, exc.status_code
    return _render(
        request,
        "recommend.html",
        {
            "recommendations": recommendations,
            "searched_book": source,
            "engine_used": engine_used,
            "error": error,
        },
        status_code=status,
    )


@router.get("/recommend", response_class=HTMLResponse, name="recommend")
def recommend_get(
    request: Request, registry: Registry, q: Annotated[str, Query(max_length=5000)] = ""
) -> HTMLResponse:
    return _recommend(request, registry, q)


@router.post("/recommend", response_class=HTMLResponse, name="recommend_post")
def recommend_post(
    request: Request,
    registry: Registry,
    book_name: Annotated[str, Form(max_length=5000)] = "",
) -> HTMLResponse:
    if not book_name.strip():
        return _render(
            request,
            "recommend.html",
            {
                "recommendations": [],
                "searched_book": "",
                "engine_used": None,
                "error": "Please enter a book title.",
            },
            status_code=422,
        )
    return _recommend(request, registry, book_name)


@router.get("/about", response_class=HTMLResponse, name="about")
def about(request: Request, registry: Registry) -> HTMLResponse:
    return _render(request, "about.html", {"stats": catalogue_stats(registry)})
