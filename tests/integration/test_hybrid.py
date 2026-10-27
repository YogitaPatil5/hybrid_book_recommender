"""Full hybrid flow: download → all (semantic + k-NN) → app serving both engines."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bookmind.main import create_app
from bookmind.pipeline import cli
from tests.bx_fixture import cluster_of, write_bx_fixture
from tests.conftest import RAW_SAMPLE, make_settings

pytestmark = [pytest.mark.integration, pytest.mark.slow]

KNN_ARGS = ["--min-user-ratings", "3", "--min-book-ratings", "3"]


@pytest.fixture
def hybrid_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    settings = make_settings(tmp_path)

    def fake_download(data_dir: Path, *, which, force=False):  # type: ignore[no-untyped-def]
        data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(RAW_SAMPLE, data_dir / "books_raw.csv")
        write_bx_fixture(data_dir / "bx")
        return {}

    monkeypatch.setattr(cli, "download_datasets", fake_download)
    assert cli.main(["download"], settings=settings) == 0
    code = cli.main(
        ["all", "--classifier", "keyword", "--encoder", "hashing", *KNN_ARGS], settings=settings
    )
    assert code == 0
    return settings


def test_all_trains_both_engines(hybrid_settings) -> None:  # type: ignore[no-untyped-def]
    art = hybrid_settings.artifacts_dir
    assert (art / "manifest.json").exists() and (art / "knn_manifest.json").exists()

    with TestClient(create_app(hybrid_settings)) as c:
        ready = c.get("/health/ready").json()
        assert ready["status"] == "ready"
        assert ready["components"]["knn"]["ready"] is True

        stats = c.get("/api/v1/stats").json()
        assert stats["knn_available"] and stats["knn_books"] == 16
        assert stats["knn_mode"] == "binary" and stats["knn_trained_at"]

        # k-NN title -> collaborative filtering
        r = c.get("/api/v1/books/similar", params={"title": "wizard hollow", "top_k": 3}).json()
        assert r["engine"] == "knn" and r["source_title"] == "Wizard Hollow"
        assert all(cluster_of(b["title"]) == "fantasy" for b in r["results"])

        # semantic-only title -> falls back to embeddings
        assert (
            c.get("/api/v1/books/similar", params={"title": "Dune"}).json()["engine"] == "semantic"
        )

        # autocomplete merges both catalogues
        sugg = c.get("/api/v1/autocomplete", params={"q": "d"}).json()["suggestions"]
        assert "Dune" in sugg and "Dragon Gate" in sugg

        about = c.get("/about").text
        assert "Collaborative Filtering" in about and "16 books" in about


def test_all_skip_knn(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.data_dir.mkdir(parents=True)
    shutil.copy(RAW_SAMPLE, settings.data_dir / "books_raw.csv")
    write_bx_fixture(settings.data_dir / "bx")
    args = ["all", "--classifier", "keyword", "--encoder", "hashing", "--skip-knn"]
    assert cli.main(args, settings=settings) == 0
    assert not (settings.artifacts_dir / "model_knn.pkl").exists()


def test_all_without_bx_data_skips_knn(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.data_dir.mkdir(parents=True)
    shutil.copy(RAW_SAMPLE, settings.data_dir / "books_raw.csv")
    assert (
        cli.main(["all", "--classifier", "keyword", "--encoder", "hashing"], settings=settings) == 0
    )
    assert not (settings.artifacts_dir / "model_knn.pkl").exists()


def test_train_knn_with_evaluation(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    write_bx_fixture(settings.data_dir / "bx")
    assert cli.main(["train-knn", "--evaluate", "--k", "3", *KNN_ARGS], settings=settings) == 0
    report = json.loads((settings.artifacts_dir / "knn_eval_binary.json").read_text())
    assert report["hit_rate"] > report["popularity_hit_rate"]


def test_evaluate_compare_modes(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    write_bx_fixture(settings.data_dir / "bx")
    assert cli.main(["evaluate-knn", "--compare-modes", *KNN_ARGS], settings=settings) == 0
    for mode in ("binary", "weighted", "explicit"):
        assert (settings.artifacts_dir / f"knn_eval_{mode}.json").exists()


def test_compare_modes_tolerates_a_failing_mode(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    write_bx_fixture(settings.data_dir / "bx")
    # explicit mode drops implicit rows -> nothing survives these thresholds; others succeed
    args = ["evaluate-knn", "--compare-modes", "--min-user-ratings", "4", "--min-book-ratings", "3"]
    assert cli.main(args, settings=settings) == 0


def test_train_knn_missing_data_returns_error(tmp_path: Path) -> None:
    assert cli.main(["train-knn"], settings=make_settings(tmp_path)) == cli.EXIT_ERROR


def test_custom_bx_dir(tmp_path: Path) -> None:
    bx = write_bx_fixture(tmp_path / "elsewhere")
    settings = make_settings(tmp_path)
    assert cli.main(["train-knn", "--bx-dir", str(bx), *KNN_ARGS], settings=settings) == 0


def test_download_only_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def fake(data_dir: Path, *, which, force=False):  # type: ignore[no-untyped-def]
        seen.update(which=list(which), force=force)
        return {}

    monkeypatch.setattr(cli, "download_datasets", fake)
    assert cli.main(["download", "--only", "knn", "--force"], settings=make_settings(tmp_path)) == 0
    assert seen == {"which": ["knn"], "force": True}
