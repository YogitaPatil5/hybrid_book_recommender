"""
Per-client rate limiting (pure ASGI middleware, built on the ``limits`` library).

* Moving-window algorithm (no burst at window boundaries).
* Keyed by client IP. Behind a reverse proxy, run uvicorn/gunicorn with
  ``--proxy-headers --forwarded-allow-ips=<proxy>`` so the real client IP is used.
* Storage: ``memory://`` (per worker) or ``redis://host:6379`` (shared across
  workers/replicas) via ``BOOKMIND_RATE_LIMIT_STORAGE_URI``.
* Health probes, metrics and static assets are never throttled.
* Storage outages fail *open* (requests allowed, error logged) so a Redis
  blip can't take the whole service down.
"""

from __future__ import annotations

import logging
import math
import time

from limits import parse
from limits.storage import storage_from_string
from limits.strategies import MovingWindowRateLimiter
from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from bookmind.api.errors import error_response

log = logging.getLogger(__name__)

EXEMPT_PREFIXES = ("/health", "/metrics", "/static", "/favicon.ico")


class RateLimitMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        limit: str,
        storage_uri: str = "memory://",
        enabled: bool = True,
        exempt_prefixes: tuple[str, ...] = EXEMPT_PREFIXES,
    ):
        self.app = app
        self.enabled = enabled
        self.item = parse(limit)
        self.limiter = MovingWindowRateLimiter(storage_from_string(storage_uri))
        self.exempt_prefixes = exempt_prefixes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            not self.enabled
            or scope["type"] != "http"
            or scope["path"].startswith(self.exempt_prefixes)
        ):
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        key = client[0] if client else "anonymous"
        try:
            allowed = self.limiter.hit(self.item, "bookmind", key)
        except Exception:  # storage outage → fail open
            log.exception("Rate-limit storage error; allowing request")
            allowed = True

        if allowed:
            await self.app(scope, receive, send)
            return

        try:
            reset_at = self.limiter.get_window_stats(self.item, "bookmind", key).reset_time
            retry_after = max(1, math.ceil(reset_at - time.time()))
        except Exception:  # pragma: no cover - storage outage mid-request
            retry_after = 60
        log.warning("Rate limit exceeded", extra={"client": key, "limit": str(self.item)})
        request = Request(scope, receive)
        response = error_response(
            request,
            429,
            "rate_limited",
            "Too many requests. Please slow down and try again shortly.",
            {"limit": str(self.item), "retry_after_seconds": retry_after},
            headers={"Retry-After": str(retry_after)},
        )
        await response(scope, receive, send)
