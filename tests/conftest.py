"""Shared pytest fixtures.

The whole suite runs offline: the hashing encoder replaces the downloadable
transformer model, and the keyword classifier replaces Ollama. Everything
else (FAISS, parquet, FastAPI, middleware) is the real production code.
"""

from __future__ import annotations

import pickle
import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from bookmind.core.config import Settings
from bookmind.main import create_app
from bookmind.pipeline.build_index import build_index
from bookmind.pipeline.classify import KeywordClassifier, run_classify
from bookmind.pipeline.prepare import run_prepare
from bookmind.services.semantic_search import SemanticSearchEngine

FIXTURES = Path(__file__).parent / "fixtures"
RAW_SAMPLE = FIXTURES / "books_raw_sample.csv"


def make_settings(tmp: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "env": "test",
        "log_to_file": False,
        "log_level": "WARNING",
        "data_dir": tmp / "data",
        "artifacts_dir": tmp / "artifacts",
        "log_dir": tmp / "logs",
        "embedding_backend": "hashing",
        "rate_limit_enabled": False,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[call-arg]


@pytest.fixture(scope="session")
def built_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the real pipeline once per session: prepare → classify → build-index."""
    root = tmp_path_factory.mktemp("built")
    data = root / "data"
    data.mkdir()
    shutil.copy(RAW_SAMPLE, data / "books_raw.csv")
    settings = make_settings(root)
    run_prepare(data / "books_raw.csv", data / "books.csv")
    run_classify(data / "books.csv", data / "books_genres.csv", KeywordClassifier())
    from bookmind.ml.embedder import HashingEncoder

    build_index(data / "books_genres.csv", settings.artifacts_dir, HashingEncoder(384))
    return root


@pytest.fixture
def workdir(tmp_path: Path, built_dir: Path) -> Path:
    """A private, writable copy of the built artifacts for one test."""
    shutil.copytree(built_dir / "artifacts", tmp_path / "artifacts")
    shutil.copytree(built_dir / "data", tmp_path / "data")
    return tmp_path


@pytest.fixture
def settings(workdir: Path) -> Settings:
    return make_settings(workdir)


@pytest.fixture
def engine(settings: Settings) -> SemanticSearchEngine:
    return SemanticSearchEngine.from_artifacts(settings)


def write_knn_artifacts(artifacts_dir: Path) -> None:
    """Create small but real legacy k-NN pickles (sklearn NearestNeighbors)."""
    titles = ["The Hobbit", "Dune", "Foundation", "Emma", "Dracula", "Ulysses"]
    rng = np.random.default_rng(7)
    ratings = []
    for user in range(1, 21):
        for title in titles:
            if rng.random() < 0.7:
                ratings.append(
                    {
                        "user_id": user,
                        "title": title,
                        "rating": int(rng.integers(1, 11)),
                        "author": f"Author of {title}",
                        "image_url": f"http://images.example.com/{title.replace(' ', '_')}.jpg",
                    }
                )
    final_rating = pd.DataFrame(ratings)
    pivot = final_rating.pivot_table(columns="user_id", index="title", values="rating").fillna(0)
    model = NearestNeighbors(algorithm="brute", metric="cosine").fit(csr_matrix(pivot.values))
    objects = {
        "model_knn": model,
        "book_names": pivot.index,
        "final_rating": final_rating,
        "book_pivot": pivot,
    }
    for name, obj in objects.items():
        with (artifacts_dir / f"{name}.pkl").open("wb") as fh:
            pickle.dump(obj, fh)


@pytest.fixture
def knn_settings(settings: Settings) -> Settings:
    write_knn_artifacts(settings.artifacts_dir)
    return settings


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture
def knn_client(knn_settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(knn_settings)) as c:
        yield c


@pytest.fixture
def empty_client(tmp_path: Path) -> Iterator[TestClient]:
    """App started with NO artifacts → degraded mode."""
    with TestClient(create_app(make_settings(tmp_path)), raise_server_exceptions=False) as c:
        yield c
