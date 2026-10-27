"""
Structured logging.

* Human-readable console output in development, one-JSON-object-per-line in
  production (``BOOKMIND_LOG_JSON=true``) so log shippers (CloudWatch, ELK,
  Loki, Datadog) can index fields.
* Every record carries the current ``request_id`` (set by middleware) so a
  single request can be traced across log lines.
* Optional size-based rotating file handler.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bookmind.core.config import Settings

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

_STANDARD_ATTRS = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()
    | {"message", "asctime", "request_id"}
)
_CONFIGURED_MARKER = "_bookmind_handler"


class RequestIdFilter(logging.Filter):
    """Inject the active request id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON documents."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
            "module": record.module,
            "line": record.lineno,
        }
        # Anything passed via `extra={...}` becomes a top-level field.
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


TEXT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | rid=%(request_id)s | %(message)s"


class TextFormatter(logging.Formatter):
    """Human-readable format that also shows ``extra={...}`` fields as k=v."""

    def __init__(self) -> None:
        super().__init__(TEXT_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _STANDARD_ATTRS and not k.startswith("_")
        }
        if not extras:
            return base
        rendered = " ".join(f"{k}={v}" for k, v in extras.items())
        head, sep, tail = base.partition("\n")  # keep tracebacks after the extras
        return f"{head} | {rendered}{sep}{tail}"


def setup_logging(settings: Settings) -> None:
    """Configure the root logger. Safe to call more than once."""
    root = logging.getLogger()
    root.setLevel(settings.log_level)

    # Remove handlers we installed previously (idempotency for tests / reloads).
    for handler in list(root.handlers):
        if getattr(handler, _CONFIGURED_MARKER, False):
            root.removeHandler(handler)
            handler.close()

    formatter: logging.Formatter = JsonFormatter() if settings.log_json else TextFormatter()
    request_filter = RequestIdFilter()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.addFilter(request_filter)
    setattr(console, _CONFIGURED_MARKER, True)
    root.addHandler(console)

    if settings.log_to_file:
        try:
            settings.log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                settings.log_dir / "bookmind.log",
                maxBytes=settings.log_max_bytes,
                backupCount=settings.log_backup_count,
                encoding="utf-8",
            )
            # Files are always JSON: easiest to grep/parse after the fact.
            file_handler.setFormatter(JsonFormatter())
            file_handler.addFilter(request_filter)
            setattr(file_handler, _CONFIGURED_MARKER, True)
            root.addHandler(file_handler)
        except OSError as exc:  # read-only FS in some containers
            root.warning("File logging disabled: %s", exc)

    # Tame noisy third-party loggers.
    for noisy in (
        "sentence_transformers",
        "urllib3",
        "httpx",
        "httpx2",
        "httpcore",
        "huggingface_hub",
        "transformers",
        "faiss",
        "multipart",
    ):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
