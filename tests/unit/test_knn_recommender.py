from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bookmind.core.config import Settings
from bookmind.core.exceptions import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    BookNotFoundError,
)
from bookmind.core.exceptions import ValidationError as BMValidationError
from bookmind.services.knn_recommender import KnnRecommender, knn_artifacts_present


@pytest.fixture
def knn(knn_settings: Settings) -> KnnRecommender:
    return KnnRecommender.from_artifacts(knn_settings.artifacts_dir)


def test_presence_check(tmp_path: Path, knn_settings: Settings) -> None:
    assert knn_artifacts_present(knn_settings.artifacts_dir)
    assert not knn_artifacts_present(tmp_path)


def test_recommend_excludes_source(knn: KnnRecommender) -> None:
    source, books = knn.recommend("dune", top_k=3)
    assert source == "Dune"  # canonicalised casing
    assert len(books) == 3
    assert "Dune" not in {b.title for b in books}
    assert all(b.author.startswith("Author of") for b in books)
    assert all(b.image_url and b.image_url.startswith("https://") for b in books)
    assert all(b.score is not None and -1.0 <= b.score <= 1.0 for b in books)


def test_top_k_capped_by_catalogue(knn: KnnRecommender) -> None:
    _, books = knn.recommend("Dune", top_k=50)
    assert len(books) == len(knn.titles) - 1


def test_unknown_title(knn: KnnRecommender) -> None:
    with pytest.raises(BookNotFoundError) as exc:
        knn.recommend("Hob")
    assert exc.value.details["suggestions"] == ["The Hobbit"]


def test_input_validation(knn: KnnRecommender) -> None:
    with pytest.raises(BMValidationError):
        knn.recommend("  ")
    with pytest.raises(BMValidationError):
        knn.recommend("Dune", top_k=0)


def test_popular_and_stats(knn: KnnRecommender) -> None:
    books = knn.popular(3)
    assert len(books) == 3
    counts = [b.num_ratings for b in books]
    assert counts == sorted(counts, reverse=True)  # type: ignore[type-var]
    assert knn.total_ratings > 0
    assert 0 < knn.total_users <= 20


def test_missing_artifacts(tmp_path: Path) -> None:
    with pytest.raises(ArtifactNotFoundError):
        KnnRecommender.from_artifacts(tmp_path)


def test_corrupt_pickle(knn_settings: Settings) -> None:
    (knn_settings.artifacts_dir / "model_knn.pkl").write_bytes(b"not a pickle")
    with pytest.raises(ArtifactIntegrityError, match="unpickle"):
        KnnRecommender.from_artifacts(knn_settings.artifacts_dir)


class _NoKneighbors:
    pass


class _Broken:
    def kneighbors(self, *a: object, **k: object) -> None:
        raise RuntimeError("boom")


def _pivot() -> pd.DataFrame:
    return pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=["A", "B"])


def _ratings(**drop: bool) -> pd.DataFrame:
    df = pd.DataFrame({"title": ["A", "B"], "rating": [5, 3], "user_id": [1, 2]})
    return df.drop(columns=list(drop)) if drop else df


@pytest.mark.parametrize(
    "model,pivot,ratings,match",
    [
        (_NoKneighbors(), _pivot(), _ratings(), "kneighbors"),
        (_Broken(), pd.DataFrame(), _ratings(), "non-empty"),
        (_Broken(), _pivot(), "not a frame", "DataFrame"),
        (_Broken(), _pivot(), _ratings(user_id=True), "missing columns"),
    ],
)
def test_constructor_validation(model: object, pivot: object, ratings: object, match: str) -> None:
    with pytest.raises(ArtifactIntegrityError, match=match):
        KnnRecommender(model, pivot, ratings)  # type: ignore[arg-type]


def test_model_failure_wrapped() -> None:
    rec = KnnRecommender(_Broken(), _pivot(), _ratings())
    with pytest.raises(ArtifactIntegrityError, match="k-NN model failed"):
        rec.recommend("A")


def test_missing_optional_columns_default() -> None:
    rec = KnnRecommender(_Broken(), _pivot(), _ratings())
    assert rec.author("A") == "Unknown"
    assert rec.image("A") is None
