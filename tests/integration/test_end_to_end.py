"""Full flow: raw CSV → CLI pipeline → running app → search results."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bookmind.main import create_app
from bookmind.pipeline import cli
from tests.conftest import RAW_SAMPLE, make_settings

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_pipeline_to_api(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.data_dir.mkdir(parents=True)
    shutil.copy(RAW_SAMPLE, settings.data_dir / "books_raw.csv")

    code = cli.main(["all", "--classifier", "keyword", "--encoder", "hashing"], settings=settings)
    assert code == 0

    with TestClient(create_app(settings)) as c:
        assert c.get("/health/ready").json()["status"] == "ready"
        top = c.get("/api/v1/search", params={"q": "mathematician galactic empire"}).json()
        assert top["results"][0]["title"] == "Foundation"
        assert top["results"][0]["genre"] == "Science Fiction"
        assert c.get("/api/v1/books/similar", params={"title": "Dune"}).status_code == 200


def test_rebuild_is_picked_up_after_restart(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.data_dir.mkdir(parents=True)
    shutil.copy(RAW_SAMPLE, settings.data_dir / "books_raw.csv")
    assert (
        cli.main(["all", "--classifier", "keyword", "--encoder", "hashing"], settings=settings) == 0
    )
    first = (settings.artifacts_dir / "manifest.json").read_text()
    assert cli.main(["build-index", "--backend", "hashing"], settings=settings) == 0
    second = (settings.artifacts_dir / "manifest.json").read_text()
    assert first != second  # new built_at timestamp
    with TestClient(create_app(settings)) as c:
        assert c.get("/api/v1/stats").json()["total_books"] == 30
