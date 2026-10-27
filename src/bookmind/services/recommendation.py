"""Use-cases shared by the JSON API and the HTML pages."""

from __future__ import annotations

import logging
from typing import Literal

from bookmind.core.exceptions import BookNotFoundError, ServiceUnavailableError
from bookmind.schemas.book import Book, PopularBook
from bookmind.schemas.search import StatsResponse
from bookmind.services.registry import ServiceRegistry
from bookmind.services.validation import ALL_GENRES

log = logging.getLogger(__name__)

EngineChoice = Literal["auto", "semantic", "knn"]


def find_similar(
    registry: ServiceRegistry, title: str, top_k: int = 5, engine: EngineChoice = "auto"
) -> tuple[str, Literal["semantic", "knn"], list[Book]]:
    """
    'More like this'.

    ``auto`` prefers collaborative filtering (a different, behavioural signal)
    when the legacy model knows the title, otherwise falls back to semantic
    similarity over the much larger catalogue.
    """
    if engine == "knn" and registry.knn is None:
        raise ServiceUnavailableError(
            "The collaborative-filtering engine is not available.",
            details={"component": "knn", "reason": registry.errors.get("knn")},
        )

    if engine in ("auto", "knn") and registry.knn is not None:
        try:
            source, books = registry.knn.recommend(title, top_k=top_k)
            return source, "knn", books
        except BookNotFoundError:
            if engine == "knn":
                raise
            log.info("Title not in k-NN catalogue; falling back to semantic similarity")

    source, books = registry.require_semantic().similar(title, top_k=top_k)
    return source, "semantic", books


def popular_books(registry: ServiceRegistry, n: int) -> list[PopularBook]:
    if registry.knn is not None:
        return registry.knn.popular(n)
    if registry.semantic is not None:
        return registry.semantic.popular(n)
    return []


def autocomplete(registry: ServiceRegistry, query: str, limit: int) -> list[str]:
    """Merge title suggestions from both engines, prefix matches first, de-duplicated."""
    if registry.semantic is None and registry.knn is None:
        registry.require_semantic()  # raises a clean 503
    needle = " ".join(str(query).split()).lower()
    seen: set[str] = set()
    starts: list[str] = []
    contains: list[str] = []
    for engine in (registry.semantic, registry.knn):
        if engine is None:
            continue
        for title in engine.autocomplete(query, limit=limit):
            key = title.lower()
            if key in seen:
                continue
            seen.add(key)
            (starts if key.startswith(needle) else contains).append(title)
    # Prefix matches from either catalogue rank above substring matches.
    return (sorted(starts, key=str.lower) + contains)[:limit]


def popular_titles(
    registry: ServiceRegistry, limit: int, genre: str | None = None
) -> tuple[list[str], str]:
    """
    Discovery suggestions for an empty search box.

    With a specific genre: the most-read titles in that genre. Otherwise the
    most-read titles from both catalogues, interleaved (k-NN first, because
    those titles also get "readers who liked this" recommendations).
    """
    if registry.semantic is None and registry.knn is None:
        registry.require_semantic()  # raises a clean 503
    limit = max(1, min(int(limit), 50))
    if genre and genre.strip().lower() != ALL_GENRES.lower():
        engine = registry.require_semantic()
        books = engine.browse(genre, top_k=limit)
        canonical = books[0].genre if books else genre.strip()
        return [b.title for b in books], f"Popular in {canonical}"

    sources = [
        [b.title for b in engine.popular(limit)]
        for engine in (registry.knn, registry.semantic)
        if engine is not None
    ]
    seen: set[str] = set()
    out: list[str] = []
    for i in range(limit):
        for titles in sources:
            if i < len(titles) and titles[i].lower() not in seen:
                seen.add(titles[i].lower())
                out.append(titles[i])
    return out[:limit], "Popular books"


def catalogue_stats(registry: ServiceRegistry) -> StatsResponse:
    sem, knn = registry.semantic, registry.knn
    manifest = sem.manifest if sem else None
    return StatsResponse(
        total_books=sem.size if sem else 0,
        total_genres=len(sem.genres) if sem else 0,
        total_ratings=knn.total_ratings if knn else 0,
        total_users=knn.total_users if knn else 0,
        embedding_model=manifest.embedding_model if manifest else None,
        index_built_at=manifest.built_at if manifest else None,
        knn_available=knn is not None,
        knn_books=len(knn.titles) if knn else 0,
        knn_mode=(knn.manifest or {}).get("mode") if knn else None,
        knn_trained_at=(knn.manifest or {}).get("trained_at") if knn else None,
    )
