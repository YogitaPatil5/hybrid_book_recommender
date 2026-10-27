"""
Collaborative-filtering recommender ("readers who liked X also liked Y").

Loads the four artifacts produced by ``bookmind-pipeline train-knn`` (or the
original ``train_model.py`` — same format): ``model_knn.pkl``,
``book_names.pkl``, ``final_rating.pkl``, ``book_pivot.pkl``.

When ``knn_manifest.json`` is present (written by ``train-knn``), file
checksums are verified before unpickling and a scikit-learn version mismatch
is logged — pickled sklearn models are only guaranteed to load correctly with
the version that produced them.

Security note: pickle can execute code on load, so only ever place files you
created yourself in ``artifacts/``. The semantic engine does not use pickle.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bookmind.core.config import KNN_ARTIFACTS
from bookmind.core.exceptions import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    BookNotFoundError,
)
from bookmind.schemas.book import Book, BuyLinks, PopularBook
from bookmind.services.artifacts import sha256_file
from bookmind.services.buy_links import get_buy_links
from bookmind.services.validation import clean_text, validate_top_k

log = logging.getLogger(__name__)

REQUIRED_RATING_COLUMNS = frozenset({"title", "rating", "user_id"})
KNN_MANIFEST = "knn_manifest.json"


def knn_artifacts_present(artifacts_dir: Path) -> bool:
    return all((artifacts_dir / f"{name}.pkl").exists() for name in KNN_ARTIFACTS)


def verify_knn_manifest(artifacts_dir: Path) -> dict[str, Any] | None:
    """Verify checksums from knn_manifest.json (if present) before unpickling."""
    path = artifacts_dir / KNN_MANIFEST
    if not path.exists():
        log.warning(
            "No %s found — k-NN files are loaded without checksum verification", KNN_MANIFEST
        )
        return None
    try:
        manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactIntegrityError(f"{KNN_MANIFEST} is not valid JSON: {exc}") from exc
    for filename, expected in (manifest.get("checksums") or {}).items():
        file_path = artifacts_dir / filename
        if not file_path.exists() or sha256_file(file_path) != expected:
            raise ArtifactIntegrityError(
                f"Checksum mismatch for {filename}. Re-run `bookmind-pipeline train-knn`."
            )
    import sklearn

    trained_with = manifest.get("sklearn_version")
    if trained_with and trained_with != sklearn.__version__:
        log.warning(
            "k-NN model was trained with scikit-learn %s but %s is installed; "
            "retrain with `bookmind-pipeline train-knn` if results look wrong",
            trained_with,
            sklearn.__version__,
        )
    return manifest


def _load_pickle(path: Path) -> Any:
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)  # noqa: S301 - see module docstring
    except Exception as exc:
        raise ArtifactIntegrityError(f"Could not unpickle {path.name}: {exc}") from exc


class KnnRecommender:
    """Thin, validated wrapper around the legacy sklearn NearestNeighbors model."""

    def __init__(
        self,
        model: Any,
        book_pivot: pd.DataFrame,
        final_rating: pd.DataFrame,
        *,
        max_top_k: int = 50,
        max_query_length: int = 300,
        manifest: dict[str, Any] | None = None,
    ):
        if not hasattr(model, "kneighbors"):
            raise ArtifactIntegrityError("model_knn does not expose kneighbors().")
        if not isinstance(book_pivot, pd.DataFrame) or book_pivot.empty:
            raise ArtifactIntegrityError("book_pivot must be a non-empty DataFrame.")
        if not isinstance(final_rating, pd.DataFrame):
            raise ArtifactIntegrityError("final_rating must be a DataFrame.")
        missing = REQUIRED_RATING_COLUMNS - set(final_rating.columns)
        if missing:
            raise ArtifactIntegrityError(f"final_rating missing columns: {sorted(missing)}")

        self.model = model
        self.book_pivot = book_pivot
        self.final_rating = final_rating
        self.max_top_k = max_top_k
        self.max_query_length = max_query_length
        self.manifest = manifest

        self._title_to_row = {str(t): i for i, t in enumerate(book_pivot.index)}
        self._lower_to_title = {str(t).lower(): str(t) for t in book_pivot.index}
        self.titles: list[str] = sorted(self._title_to_row, key=str.lower)

        # Precompute per-title author / cover once (old code filtered the full
        # ratings frame on every call: O(n) per lookup).
        first = final_rating.drop_duplicates("title").set_index("title")
        self._authors: dict[str, str] = (
            first["author"].astype(str).to_dict() if "author" in first.columns else {}
        )
        self._images: dict[str, str] = (
            first["image_url"].astype(str).to_dict() if "image_url" in first.columns else {}
        )
        log.info("k-NN recommender ready", extra={"books": len(self.titles)})

    @classmethod
    def from_artifacts(
        cls, artifacts_dir: Path, *, max_top_k: int = 50, max_query_length: int = 300
    ) -> KnnRecommender:
        if not knn_artifacts_present(artifacts_dir):
            raise ArtifactNotFoundError(
                "Legacy k-NN artifacts not found.", details={"required": list(KNN_ARTIFACTS)}
            )
        manifest = verify_knn_manifest(artifacts_dir)
        loaded = {name: _load_pickle(artifacts_dir / f"{name}.pkl") for name in KNN_ARTIFACTS}
        return cls(
            loaded["model_knn"],
            loaded["book_pivot"],
            loaded["final_rating"],
            max_top_k=max_top_k,
            max_query_length=max_query_length,
            manifest=manifest,
        )

    # ── Lookups ──────────────────────────────────────────────────────────────
    @property
    def total_ratings(self) -> int:
        return len(self.final_rating)

    @property
    def total_users(self) -> int:
        return int(self.final_rating["user_id"].nunique())

    def author(self, title: str) -> str:
        value = self._authors.get(title, "Unknown")
        return "Unknown" if value.lower() in {"", "nan", "none"} else value

    def image(self, title: str) -> str | None:
        url = self._images.get(title, "").strip()
        if url.startswith("http://"):
            url = "https://" + url[len("http://") :]
        return url if url.startswith("https://") else None

    def _book(self, title: str, score: float | None = None) -> Book:
        author = self.author(title)
        return Book(
            title=title,
            author=author,
            genre="Other",
            image_url=self.image(title),
            score=score,
            buy_links=BuyLinks.model_validate(get_buy_links(title, author)),
        )

    # ── Public API ───────────────────────────────────────────────────────────
    def autocomplete(self, prefix: str, limit: int = 10) -> list[str]:
        """Title suggestions (prefix matches first, then substring)."""
        if not isinstance(prefix, str):
            return []
        needle = " ".join(prefix.split()).lower()[: self.max_query_length]
        if not needle:
            return []
        limit = max(1, min(int(limit), 50))
        starts = [t for t in self.titles if t.lower().startswith(needle)]
        contains = [t for t in self.titles if needle in t.lower() and t not in starts]
        return (starts + contains)[:limit]

    def recommend(self, title: str, top_k: int = 5) -> tuple[str, list[Book]]:
        title = clean_text(title, field="title", max_length=self.max_query_length)
        top_k = validate_top_k(top_k, max_top_k=self.max_top_k)

        canonical = self._lower_to_title.get(title.lower())
        if canonical is None:
            raise BookNotFoundError(
                f"'{title[:80]}' is not in the collaborative-filtering catalogue.",
                details={"suggestions": [t for t in self.titles if title.lower() in t.lower()][:5]},
            )
        row = self._title_to_row[canonical]
        n_neighbors = min(top_k + 1, len(self.book_pivot))
        try:
            distances, suggestions = self.model.kneighbors(
                self.book_pivot.iloc[row, :].to_numpy().reshape(1, -1), n_neighbors=n_neighbors
            )
        except Exception as exc:
            raise ArtifactIntegrityError(f"k-NN model failed: {exc}") from exc

        results: list[Book] = []
        for dist, idx in zip(np.ravel(distances), np.ravel(suggestions)):
            neighbour = str(self.book_pivot.index[int(idx)])
            if neighbour == canonical:
                continue
            results.append(self._book(neighbour, score=round(1.0 - float(dist), 4)))
            if len(results) >= top_k:
                break
        return canonical, results

    def popular(self, n: int = 50) -> list[PopularBook]:
        """
        Most-read books. ``avg_rating`` uses explicit ratings only (Book-Crossing
        0 = implicit, not a zero score) and is converted from /10 to /5 so it is
        on the same scale as the semantic catalogue.
        """
        n = max(1, min(int(n), 500))
        ratings = self.final_rating
        explicit = ratings[ratings["rating"] > 0].groupby("title")["rating"].mean() / 2
        counts = (
            pd.DataFrame({"num_ratings": ratings.groupby("title")["user_id"].nunique()})
            .assign(avg_rating=explicit)
            .sort_values("num_ratings", ascending=False)
            .head(n)
        )
        out: list[PopularBook] = []
        for title, row in counts.iterrows():
            t = str(title)
            author = self.author(t)
            out.append(
                PopularBook(
                    title=t,
                    author=author,
                    image_url=self.image(t),
                    num_ratings=int(row["num_ratings"]),
                    avg_rating=(
                        None if pd.isna(row["avg_rating"]) else round(float(row["avg_rating"]), 1)
                    ),
                    buy_links=BuyLinks.model_validate(get_buy_links(t, author)),
                )
            )
        return out
