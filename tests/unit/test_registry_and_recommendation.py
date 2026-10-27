from __future__ import annotations

from pathlib import Path

import pytest

from bookmind.core.config import Settings
from bookmind.core.exceptions import BookNotFoundError, ServiceUnavailableError
from bookmind.services import registry as registry_mod
from bookmind.services.recommendation import catalogue_stats, find_similar, popular_books
from bookmind.services.registry import ServiceRegistry
from tests.conftest import make_settings


def test_loads_semantic_without_knn(settings: Settings) -> None:
    reg = ServiceRegistry.load(settings)
    assert reg.semantic is not None
    assert reg.knn is None
    assert reg.knn_status == "absent"


def test_loads_both(knn_settings: Settings) -> None:
    reg = ServiceRegistry.load(knn_settings)
    assert reg.knn is not None and reg.knn_status == "loaded"


def test_knn_disabled_by_config(knn_settings: Settings) -> None:
    reg = ServiceRegistry.load(knn_settings.model_copy(update={"enable_knn": False}))
    assert reg.knn is None and reg.knn_status == "disabled"


def test_corrupt_knn_marks_failed(knn_settings: Settings) -> None:
    (knn_settings.artifacts_dir / "book_pivot.pkl").write_bytes(b"junk")
    reg = ServiceRegistry.load(knn_settings)
    assert reg.knn_status == "failed"
    assert reg.semantic is not None  # failure is isolated


def test_missing_index_is_degraded_not_crash(tmp_path: Path) -> None:
    reg = ServiceRegistry.load(make_settings(tmp_path))
    assert reg.semantic is None
    assert "semantic" in reg.errors
    with pytest.raises(ServiceUnavailableError) as exc:
        reg.require_semantic()
    assert exc.value.status_code == 503


def test_unexpected_errors_are_contained(
    settings: Settings, knn_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: object, **k: object) -> None:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(registry_mod.SemanticSearchEngine, "from_artifacts", boom)
    monkeypatch.setattr(registry_mod.KnnRecommender, "from_artifacts", boom)
    reg = ServiceRegistry.load(knn_settings)
    assert reg.semantic is None and reg.knn is None
    assert reg.knn_status == "failed"


class TestFindSimilar:
    def test_auto_uses_knn_when_title_known(self, knn_settings: Settings) -> None:
        reg = ServiceRegistry.load(knn_settings)
        source, engine, books = find_similar(reg, "Dune", top_k=2)
        assert (source, engine, len(books)) == ("Dune", "knn", 2)

    def test_auto_falls_back_to_semantic(self, knn_settings: Settings) -> None:
        reg = ServiceRegistry.load(knn_settings)
        _, engine, _ = find_similar(reg, "Atomic Habits")
        assert engine == "semantic"

    def test_explicit_semantic(self, knn_settings: Settings) -> None:
        reg = ServiceRegistry.load(knn_settings)
        assert find_similar(reg, "Dune", engine="semantic")[1] == "semantic"

    def test_explicit_knn_unknown_title_raises(self, knn_settings: Settings) -> None:
        reg = ServiceRegistry.load(knn_settings)
        with pytest.raises(BookNotFoundError):
            find_similar(reg, "Atomic Habits", engine="knn")

    def test_explicit_knn_when_unavailable(self, settings: Settings) -> None:
        reg = ServiceRegistry.load(settings)
        with pytest.raises(ServiceUnavailableError):
            find_similar(reg, "Dune", engine="knn")


def test_popular_prefers_knn(knn_settings: Settings, settings: Settings) -> None:
    reg = ServiceRegistry.load(knn_settings)
    assert all(b.genre is None for b in popular_books(reg, 3))  # k-NN rows have no genre
    reg_sem = ServiceRegistry.load(settings.model_copy(update={"enable_knn": False}))
    assert all(b.genre for b in popular_books(reg_sem, 3))


def test_popular_empty_when_nothing_loaded(tmp_path: Path) -> None:
    assert popular_books(ServiceRegistry.load(make_settings(tmp_path)), 5) == []


def test_stats(knn_settings: Settings, tmp_path: Path) -> None:
    stats = catalogue_stats(ServiceRegistry.load(knn_settings))
    assert stats.total_books == 30 and stats.knn_available and stats.total_users > 0
    assert stats.embedding_model == "hashing-384"
    empty = catalogue_stats(ServiceRegistry.load(make_settings(tmp_path / "x")))
    assert empty.total_books == 0 and not empty.knn_available
