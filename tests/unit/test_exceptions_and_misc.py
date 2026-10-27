from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import bookmind.main as main_mod
from bookmind.core import exceptions as exc
from bookmind.main import create_app
from bookmind.middleware import rate_limit
from tests.conftest import make_settings


@pytest.mark.parametrize(
    "cls,status,code",
    [
        (exc.BookMindError, 500, "internal_error"),
        (exc.ConfigurationError, 500, "configuration_error"),
        (exc.ArtifactNotFoundError, 503, "artifacts_missing"),
        (exc.ArtifactIntegrityError, 503, "artifacts_corrupt"),
        (exc.ServiceUnavailableError, 503, "service_unavailable"),
        (exc.ValidationError, 422, "validation_error"),
        (exc.BookNotFoundError, 404, "book_not_found"),
        (exc.EmbeddingError, 500, "embedding_error"),
        (exc.DataValidationError, 400, "data_validation_error"),
    ],
)
def test_exception_contract(cls: type[exc.BookMindError], status: int, code: str) -> None:
    err = cls()
    assert err.status_code == status and err.code == code
    assert err.message == cls.default_message
    assert err.to_dict() == {"code": code, "message": cls.default_message}
    assert isinstance(err, exc.BookMindError)


def test_exception_details() -> None:
    err = exc.ValidationError("bad", details={"field": "q"})
    assert err.to_dict() == {
        "code": "validation_error",
        "message": "bad",
        "details": {"field": "q"},
    }
    assert str(err) == "bad"


def test_rate_limit_fails_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_hit(self: object, *a: object) -> bool:
        raise ConnectionError("redis down")

    monkeypatch.setattr(rate_limit.MovingWindowRateLimiter, "hit", broken_hit)
    app = create_app(make_settings(tmp_path, rate_limit_enabled=True, rate_limit="1/minute"))
    with TestClient(app) as c:
        assert all(c.get("/api/v1/stats").status_code == 200 for _ in range(3))


def test_module_level_app_is_lazy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main_mod, "get_settings", lambda: make_settings(tmp_path))
    monkeypatch.delitem(main_mod.__dict__, "app", raising=False)
    app = main_mod.app
    assert app is main_mod.app  # cached after first access
    with pytest.raises(AttributeError):
        _ = main_mod.does_not_exist
