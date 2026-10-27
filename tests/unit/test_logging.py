from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from bookmind.core.logging import (
    JsonFormatter,
    RequestIdFilter,
    TextFormatter,
    request_id_ctx,
    setup_logging,
)
from tests.conftest import make_settings


def _record(msg: str = "hello", **extra: object) -> logging.LogRecord:
    record = logging.LogRecord("bookmind.test", logging.INFO, __file__, 10, msg, (), None)
    for k, v in extra.items():
        setattr(record, k, v)
    RequestIdFilter().filter(record)
    return record


def test_json_formatter_includes_request_id_and_extras() -> None:
    token = request_id_ctx.set("req-12345678")
    try:
        payload = json.loads(JsonFormatter().format(_record(user="u1", count=3)))
    finally:
        request_id_ctx.reset(token)
    assert payload["message"] == "hello"
    assert payload["request_id"] == "req-12345678"
    assert payload["user"] == "u1" and payload["count"] == 3
    assert payload["level"] == "INFO"


def test_json_formatter_exception() -> None:
    try:
        raise ValueError("bad")
    except ValueError:
        import sys

        record = _record()
        record.exc_info = sys.exc_info()
    payload = json.loads(JsonFormatter().format(record))
    assert "ValueError: bad" in payload["exc_info"]


def test_text_formatter_appends_extras() -> None:
    line = TextFormatter().format(_record(route="/x"))
    assert "rid=-" in line and line.endswith("route=/x")


def test_text_formatter_without_extras() -> None:
    assert TextFormatter().format(_record()).endswith("hello")


def test_setup_is_idempotent(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, log_to_file=True, log_level="INFO")
    setup_logging(settings)
    setup_logging(settings)
    ours = [h for h in logging.getLogger().handlers if getattr(h, "_bookmind_handler", False)]
    assert len(ours) == 2  # console + file, not duplicated
    logging.getLogger("bookmind.test").info("to file", extra={"k": "v"})
    for h in ours:
        h.flush()
    line = (tmp_path / "logs" / "bookmind.log").read_text().strip().splitlines()[-1]
    assert json.loads(line)["k"] == "v"
    setup_logging(make_settings(tmp_path))  # restore quiet test config


def test_file_logging_failure_is_tolerated(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    setup_logging(make_settings(tmp_path, log_to_file=True, log_dir=blocker / "logs"))
    assert any("File logging disabled" in r.message for r in caplog.records)
    setup_logging(make_settings(tmp_path))


def test_json_console_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(make_settings(tmp_path, log_json=True, log_level="INFO"))
    logging.getLogger("bookmind.test").info("json line")
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(out)["message"] == "json line"
    setup_logging(make_settings(tmp_path))
