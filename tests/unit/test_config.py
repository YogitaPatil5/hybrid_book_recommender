from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from bookmind.core.config import Environment, Settings, get_settings


def _settings(**kw: object) -> Settings:
    return Settings(_env_file=None, **kw)  # type: ignore[call-arg]


def test_defaults_are_sane() -> None:
    s = _settings()
    assert s.env is Environment.DEVELOPMENT
    assert s.debug is False
    assert s.default_top_k <= s.max_top_k
    assert s.artifacts_dir.is_absolute()


def test_derived_paths(tmp_path: Path) -> None:
    s = _settings(artifacts_dir=tmp_path)
    assert s.index_path == tmp_path / "book_index.faiss"
    assert s.meta_path == tmp_path / "book_meta.parquet"
    assert s.manifest_path == tmp_path / "manifest.json"


def test_env_vars_are_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOOKMIND_MAX_TOP_K", "77")
    monkeypatch.setenv("BOOKMIND_ENV", "staging")
    s = _settings()
    assert s.max_top_k == 77
    assert s.env is Environment.STAGING


def test_debug_forbidden_in_production() -> None:
    with pytest.raises(ValidationError, match="debug must be false"):
        _settings(env="production", debug=True)


def test_default_top_k_cannot_exceed_max() -> None:
    with pytest.raises(ValidationError, match="default_top_k"):
        _settings(default_top_k=60, max_top_k=50)


@pytest.mark.parametrize("value", ["60", "abc/minute", "0/minute", "10/week", "-1/second"])
def test_invalid_rate_limit(value: str) -> None:
    with pytest.raises(ValidationError):
        _settings(rate_limit=value)


@pytest.mark.parametrize("field,value", [("port", 0), ("port", 70000), ("max_top_k", 0)])
def test_out_of_range_numbers(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        _settings(**{field: value})


def test_invalid_log_level() -> None:
    with pytest.raises(ValidationError):
        _settings(log_level="LOUD")


def test_is_production_flag() -> None:
    assert _settings(env="production").is_production
    assert not _settings(env="test").is_production


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()
    get_settings.cache_clear()
