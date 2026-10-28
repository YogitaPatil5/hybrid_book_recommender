"""
Semantic book search over a FAISS inner-product index.

Capabilities
------------
* ``search``        free text / title / "vibe" query, optional genre filter
* ``similar``       "more like this" for a catalogue title (uses stored vectors)
* ``autocomplete``  title suggestions (prefix matches first, then substring)
* ``popular``       most-rated books, used by the home page when k-NN is absent

Design notes
------------
* Genre filtering uses FAISS ``IDSelectorBatch`` so the filtered search is
  exact (the old version over-fetched ``top_k*5`` and could return too few).
* Results are de-duplicated by (title, author) — the dataset has multiple
  editions of the same book.
* Exact title matches are pinned to the top of free-text results.
* All public methods validate their input; the engine never trusts callers.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import faiss
import numpy as np
import pandas as pd

from bookmind.core.exceptions import ArtifactIntegrityError, BookNotFoundError, ValidationError
from bookmind.ml.embedder import CachedEncoder, Encoder, HashingEncoder, build_encoder
from bookmind.schemas.book import Book, BuyLinks, PopularBook
from bookmind.services.artifacts import Manifest, load_artifacts
from bookmind.services.buy_links import get_buy_links
from bookmind.services.validation import ALL_GENRES, clean_text, validate_genre, validate_top_k

if TYPE_CHECKING:
    from bookmind.core.config import Settings

log = logging.getLogger(__name__)

DESCRIPTION_PREVIEW_CHARS = 280
_DEDUP_OVERFETCH = 3


def _clean_str(value: Any, default: str = "") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    text = str(value).strip()
    return default if not text or text.lower() == "nan" else text


def _clean_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def _clean_int(value: Any) -> int | None:
    number = _clean_float(value)
    return None if number is None else int(number)


def truncate(text: str, limit: int = DESCRIPTION_PREVIEW_CHARS) -> str:
    """Trim on a word boundary and add an ellipsis only when shortened."""
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",.;:- ")
    return f"{cut}…"


@dataclass(frozen=True)
class _Record:
    title: str
    author: str
    genre: str
    description: str
    image_url: str | None
    avg_rating: float | None
    published_year: int | None
    ratings_count: int | None

    @property
    def dedup_key(self) -> tuple[str, str]:
        return self.title.lower(), self.author.lower()


class SemanticSearchEngine:
    """Thread-safe (read-only) semantic search engine."""

    def __init__(
        self,
        index: Any,
        meta: pd.DataFrame,
        encoder: Encoder,
        manifest: Manifest | None = None,
        *,
        max_top_k: int = 50,
        max_query_length: int = 300,
        cache_size: int = 2048,
    ):
        if index.ntotal != len(meta):
            raise ArtifactIntegrityError("Index and metadata sizes differ.")
        if encoder.dim != index.d:
            raise ArtifactIntegrityError(
                f"Encoder dim {encoder.dim} does not match index dim {index.d}. "
                "Rebuild the index with the same embedding model."
            )
        self.index = index
        self.manifest = manifest
        self.encoder = (
            encoder
            if isinstance(encoder, CachedEncoder)
            else CachedEncoder(encoder, maxsize=cache_size)
        )
        self.max_top_k = max_top_k
        self.max_query_length = max_query_length

        # Many web threads x OpenMP threads oversubscribes CPUs; flat search
        # over a few thousand vectors is fastest single-threaded anyway.
        faiss.omp_set_num_threads(1)

        self._records: list[_Record] = [
            self._make_record({str(k): v for k, v in row.items()})
            for row in meta.to_dict("records")
        ]
        self._titles_lower: list[str] = [r.title.lower() for r in self._records]
        self._title_positions: dict[str, list[int]] = defaultdict(list)
        genre_positions: dict[str, list[int]] = defaultdict(list)
        for pos, rec in enumerate(self._records):
            self._title_positions[rec.title.lower()].append(pos)
            genre_positions[rec.genre].append(pos)

        self.genres: list[str] = sorted(genre_positions)
        self._genre_ids = {g: np.asarray(ids, dtype=np.int64) for g, ids in genre_positions.items()}
        self.titles: list[str] = sorted({r.title for r in self._records}, key=str.lower)
        self._titles_sorted_lower = [t.lower() for t in self.titles]
        log.info(
            "Semantic engine ready",
            extra={"books": self.size, "genres": len(self.genres), "encoder": self.encoder.name},
        )

    # ── Construction ─────────────────────────────────────────────────────────
    @classmethod
    def from_artifacts(
        cls, settings: Settings, encoder: Encoder | None = None
    ) -> SemanticSearchEngine:
        index, meta, manifest = load_artifacts(settings.artifacts_dir)
        if encoder is None:
            # Always query with the SAME encoder that built the index.
            if manifest.embedding_backend == "hashing":
                encoder = HashingEncoder(dim=manifest.dim)
            else:  # pragma: no cover - requires model download
                encoder = build_encoder(settings, model_name=manifest.embedding_model)
        return cls(
            index,
            meta,
            encoder,
            manifest,
            max_top_k=settings.max_top_k,
            max_query_length=settings.max_query_length,
            cache_size=settings.embedding_cache_size,
        )

    @staticmethod
    def _make_record(row: dict[str, Any]) -> _Record:
        thumb = _clean_str(row.get("thumbnail"))
        if thumb.startswith("http://"):
            # Upgrade to https: avoids mixed-content blocking and satisfies the CSP.
            thumb = "https://" + thumb[len("http://") :]
        rating = _clean_float(row.get("average_rating"))
        if rating is not None and not 0 <= rating <= 5:
            rating = None
        year = _clean_int(row.get("published_year"))
        if year is not None and not 0 < year <= 2100:
            year = None
        count = _clean_int(row.get("ratings_count"))
        return _Record(
            title=_clean_str(row.get("title"), "Untitled"),
            author=_clean_str(row.get("authors"), "Unknown"),
            genre=_clean_str(row.get("genre"), "Other"),
            description=_clean_str(row.get("description")),
            image_url=thumb if thumb.startswith("https://") else None,
            avg_rating=rating,
            published_year=year,
            ratings_count=count if count is not None and count >= 0 else None,
        )

    # ── Introspection ────────────────────────────────────────────────────────
    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    def list_genres(self) -> list[str]:
        return list(self.genres)

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _to_book(self, pos: int, score: float | None) -> Book:
        rec = self._records[pos]
        return Book(
            title=rec.title,
            author=rec.author,
            genre=rec.genre,
            description=truncate(rec.description),
            image_url=rec.image_url,
            avg_rating=rec.avg_rating,
            published_year=rec.published_year,
            score=None if score is None else round(float(score), 4),
            buy_links=BuyLinks.model_validate(get_buy_links(rec.title, rec.author)),
        )

    def _knn(self, vec: np.ndarray, k: int, genre: str) -> list[tuple[int, float]]:
        k = min(k, self.size)
        if k <= 0:
            return []
        if genre == ALL_GENRES:
            scores, ids = self.index.search(vec, k)
        else:
            subset = self._genre_ids.get(genre)
            if subset is None or subset.size == 0:
                return []
            k = min(k, int(subset.size))
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(subset))  # type: ignore[call-arg]
            scores, ids = self.index.search(vec, k, params=params)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def _collect(
        self,
        hits: list[tuple[int, float]],
        top_k: int,
        *,
        exclude_titles: set[str] | None = None,
        pinned: list[int] | None = None,
        scores_visible: bool = True,
    ) -> list[Book]:
        seen: set[tuple[str, str]] = set()
        out: list[Book] = []
        ordered = [(p, None) for p in (pinned or [])] + hits
        scores = dict(hits)
        for pos, _ in ordered:
            rec = self._records[pos]
            if exclude_titles and rec.title.lower() in exclude_titles:
                continue
            if rec.dedup_key in seen:
                continue
            seen.add(rec.dedup_key)
            out.append(self._to_book(pos, scores.get(pos) if scores_visible else None))
            if len(out) >= top_k:
                break
        return out

    def resolve_title(self, title: str) -> int:
        """Map a user-supplied title to a catalogue row (exact > prefix > substring)."""
        needle = title.lower()
        exact = self._title_positions.get(needle)
        if exact:
            return self._best_edition(exact)
        for matcher in (str.startswith, str.__contains__):
            candidates = [pos for pos, t in enumerate(self._titles_lower) if matcher(t, needle)]
            if candidates:
                # Prefer the shortest title (closest to what was typed), then most rated.
                shortest = min(len(self._titles_lower[p]) for p in candidates)
                best = [p for p in candidates if len(self._titles_lower[p]) == shortest]
                return self._best_edition(best)
        raise BookNotFoundError(
            f"'{title[:80]}' was not found in the catalogue.",
            details={"suggestions": self.autocomplete(title[:20], limit=5)},
        )

    def _best_edition(self, positions: list[int]) -> int:
        return max(positions, key=lambda p: (self._records[p].ratings_count or 0, -p))

    # ── Public API ───────────────────────────────────────────────────────────
    def search(self, query: str, genre: str | None = None, top_k: int = 12) -> list[Book]:
        """Free-text semantic search with optional genre filter."""
        query = clean_text(query, field="query", max_length=self.max_query_length)
        genre = validate_genre(genre, self.genres)
        top_k = validate_top_k(top_k, max_top_k=self.max_top_k)

        started = time.perf_counter()
        vec = self.encoder.encode_one(query)
        hits = self._knn(vec, top_k * _DEDUP_OVERFETCH, genre)

        pinned = [
            p
            for p in self._title_positions.get(query.lower(), [])
            if genre == ALL_GENRES or self._records[p].genre == genre
        ]
        results = self._collect(hits, top_k, pinned=pinned[:1])
        log.info(
            "search",
            extra={
                "query_len": len(query),
                "genre": genre,
                "top_k": top_k,
                "results": len(results),
                "ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return results

    def similar(self, title: str, top_k: int = 5) -> tuple[str, list[Book]]:
        """Books semantically closest to ``title`` (excluding all its editions)."""
        title = clean_text(title, field="title", max_length=self.max_query_length)
        top_k = validate_top_k(top_k, max_top_k=self.max_top_k)

        pos = self.resolve_title(title)
        source = self._records[pos]
        vec = np.asarray(self.index.reconstruct(pos), dtype=np.float32).reshape(1, -1)
        same_count = len(self._title_positions[source.title.lower()])
        hits = self._knn(vec, (top_k + same_count) * _DEDUP_OVERFETCH, ALL_GENRES)
        results = self._collect(hits, top_k, exclude_titles={source.title.lower()})
        return source.title, results

    def autocomplete(self, prefix: str, limit: int = 10) -> list[str]:
        """Unique title suggestions; prefix matches rank before substring matches."""
        if not isinstance(prefix, str):
            return []
        needle = " ".join(prefix.split()).lower()[: self.max_query_length]
        if not needle:
            return []
        limit = max(1, min(int(limit), 50))
        starts: list[str] = []
        contains: list[str] = []
        for title, lower in zip(self.titles, self._titles_sorted_lower):
            if lower.startswith(needle):
                starts.append(title)
                if len(starts) >= limit:
                    break
            elif needle in lower and len(contains) < limit:
                contains.append(title)
        return (starts + contains)[:limit]

    def browse(self, genre: str, top_k: int = 12) -> list[Book]:
        """
        Genre-only browsing (no text query): the most-read books in ``genre``,
        ties broken by average rating, one entry per title/author.
        """
        canonical = validate_genre(genre, self.genres)
        if canonical == ALL_GENRES:
            raise ValidationError(
                "Choose a specific genre to browse, or enter a search query.",
                details={"field": "genre"},
            )
        top_k = validate_top_k(top_k, max_top_k=self.max_top_k)
        positions = self._genre_ids.get(canonical, np.empty(0, dtype=np.int64)).tolist()

        def rank(p: int) -> tuple[int, float]:
            rec = self._records[p]
            return rec.ratings_count or 0, rec.avg_rating or 0.0

        ordered = sorted(positions, key=rank, reverse=True)
        return self._collect([(p, 0.0) for p in ordered], top_k, scores_visible=False)

    def popular(self, n: int = 50) -> list[PopularBook]:
        """Most-rated books (ties broken by rating), one per title."""
        n = max(1, min(int(n), 500))

        def rank(p: int) -> tuple[int, float]:
            rec = self._records[p]
            return rec.ratings_count or 0, rec.avg_rating or 0.0

        out: list[PopularBook] = []
        seen: set[str] = set()
        for pos in sorted(range(self.size), key=rank, reverse=True):
            rec = self._records[pos]
            if rec.title.lower() in seen:
                continue
            seen.add(rec.title.lower())
            out.append(
                PopularBook(
                    title=rec.title,
                    author=rec.author,
                    image_url=rec.image_url,
                    avg_rating=None if rec.avg_rating is None else round(rec.avg_rating, 1),
                    num_ratings=rec.ratings_count,
                    genre=rec.genre,
                    buy_links=BuyLinks.model_validate(get_buy_links(rec.title, rec.author)),
                )
            )
            if len(out) >= n:
                break
        return out
