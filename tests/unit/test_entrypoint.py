"""`python -m bookmind` launcher and offline-first model loading."""

from __future__ import annotations

import socket
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import bookmind.__main__ as entry
from bookmind.core.exceptions import ConfigurationError
from bookmind.ml.embedder import SentenceTransformerEncoder
from tests.conftest import make_settings


@pytest.fixture
def busy_port():  # type: ignore[no-untyped-def]
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    yield sock.getsockname()[1]
    sock.close()


class TestPorts:
    def test_busy_port_detected(self, busy_port: int) -> None:
        assert not entry.port_is_free("127.0.0.1", busy_port)

    def test_find_free_port_skips_busy(self, busy_port: int) -> None:
        port = entry.find_free_port("127.0.0.1", busy_port)
        assert port != busy_port and entry.port_is_free("127.0.0.1", port)

    def test_no_free_port(self, busy_port: int) -> None:
        with pytest.raises(SystemExit, match="No free port"):
            entry.find_free_port("127.0.0.1", busy_port, attempts=1)


@pytest.fixture
def launched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):  # type: ignore[no-untyped-def]
    calls: dict[str, Any] = {}

    def run(app: str, **kwargs: Any) -> None:
        calls.update(kwargs, app=app)

    monkeypatch.setattr(entry.uvicorn, "run", run)
    monkeypatch.setattr(entry, "get_settings", lambda: make_settings(tmp_path, host="127.0.0.1"))
    return calls


class TestMain:
    def test_defaults(self, launched: dict[str, Any]) -> None:
        entry.main([])
        assert launched["app"] == "bookmind.main:create_app" and launched["factory"] is True
        assert launched["access_log"] is False and launched["reload"] is False

    def test_busy_port_falls_back_in_development(
        self, launched: dict[str, Any], busy_port: int, caplog: pytest.LogCaptureFixture
    ) -> None:
        entry.main(["--port", str(busy_port)])
        assert launched["port"] != busy_port
        assert any("is busy" in r.getMessage() for r in caplog.records)

    def test_production_keeps_configured_port(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, busy_port: int
    ) -> None:
        calls: dict[str, Any] = {}
        monkeypatch.setattr(entry.uvicorn, "run", lambda app, **kw: calls.update(kw))
        monkeypatch.setattr(
            entry,
            "get_settings",
            lambda: make_settings(tmp_path, env="production", host="127.0.0.1"),
        )
        entry.main(["--port", str(busy_port)])
        assert calls["port"] == busy_port  # fail fast in production, no silent change

    def test_open_browser_and_reload(
        self, launched: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opened: list[tuple[Any, ...]] = []

        class FakeTimer:
            def __init__(self, delay: float, fn: Any, args: tuple[Any, ...] = ()) -> None:
                opened.append(args)

            def start(self) -> None:
                pass

        monkeypatch.setattr(entry.threading, "Timer", FakeTimer)
        entry.main(["--open", "--reload", "--host", "127.0.0.1"])
        assert opened and opened[0][0].startswith("http://localhost:")
        assert launched["reload"] is True and launched["host"] == "127.0.0.1"


class TestOfflineFirstModelLoading:
    def _install_fake(
        self, monkeypatch: pytest.MonkeyPatch, *, cached: bool, online: bool, new_api: bool = True
    ) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []

        class FakeST:
            def __init__(
                self, name: str, device: str | None = None, local_files_only: bool = False
            ) -> None:
                calls.append({"name": name, "device": device, "local_files_only": local_files_only})
                if local_files_only and not cached:
                    raise OSError("not in cache")
                if not local_files_only and not online:
                    raise ConnectionError("offline")

            def get_sentence_embedding_dimension(self) -> int:
                return 16

        if new_api:
            FakeST.get_embedding_dimension = lambda self: 16  # type: ignore[attr-defined]
        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = FakeST  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sentence_transformers", module)
        return calls

    def test_cached_model_loads_without_network(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._install_fake(monkeypatch, cached=True, online=False)
        enc = SentenceTransformerEncoder("mini")
        assert calls == [{"name": "mini", "device": "cpu", "local_files_only": True}]
        assert enc.dim == 16 and enc.name == "mini"

    def test_first_run_downloads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._install_fake(monkeypatch, cached=False, online=True)
        SentenceTransformerEncoder("mini")
        assert [c["local_files_only"] for c in calls] == [True, False]

    def test_not_cached_and_offline_gives_clear_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._install_fake(monkeypatch, cached=False, online=False)
        with pytest.raises(ConfigurationError, match="first download"):
            SentenceTransformerEncoder("mini")

    def test_device_is_explicit_on_both_paths(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Hosts that report a GPU (e.g. Hugging Face ZeroGPU) must not get "auto" -> CUDA.
        calls = self._install_fake(monkeypatch, cached=False, online=True)
        SentenceTransformerEncoder("mini", device="cpu")
        assert [c["device"] for c in calls] == ["cpu", "cpu"]

    def test_older_library_api_supported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._install_fake(monkeypatch, cached=True, online=False, new_api=False)
        assert SentenceTransformerEncoder("mini").dim == 16
