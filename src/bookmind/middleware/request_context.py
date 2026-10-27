"""
Per-request middleware.

* Accepts an inbound ``X-Request-ID`` (if well-formed) or generates one,
  binds it to the logging context and echoes it on the response.
* Emits one structured access-log line per request with latency.
* Records Prometheus metrics keyed by the *route template* (``/api/v1/search``),
  never the raw path, to keep metric cardinality bounded.
* Adds standard security headers (CSP, nosniff, frame-deny, referrer policy).
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable

from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Match
from starlette.types import ASGIApp

from bookmind.core.logging import request_id_ctx

log = logging.getLogger("bookmind.access")

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")

HTTP_REQUESTS = Counter(
    "bookmind_http_requests_total", "HTTP requests", ["method", "route", "status"]
)
HTTP_LATENCY = Histogram(
    "bookmind_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)

CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com",
        "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com data:",
        "img-src 'self' https: data:",
        "connect-src 'self'",
        "frame-ancestors 'none'",
        "base-uri 'self'",
        "form-action 'self'",
    ]
)
# Swagger UI loads assets from jsDelivr; relax CSP only on the docs pages.
DOCS_CSP = (
    "default-src 'self'; img-src 'self' https: data:; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net"
)
DOCS_PATHS = ("/docs", "/redoc")


def _route_template(request: Request) -> str:
    """Route template set by the router during dispatch (bounded label set)."""
    route = request.scope.get("route")
    path = getattr(route, "path_format", None) or getattr(route, "path", None)
    if path:
        return str(path)
    # Fallback for frameworks/versions that don't populate scope["route"].
    for candidate in request.app.router.routes:
        match, _ = candidate.matches(request.scope)
        if match is Match.FULL and getattr(candidate, "path", None):
            return str(candidate.path)
    return "unmatched"


class RequestContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, *, enable_hsts: bool = False):
        super().__init__(app)
        self.enable_hsts = enable_hsts

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        inbound = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = inbound if _VALID_REQUEST_ID.match(inbound) else uuid.uuid4().hex
        token = request_id_ctx.set(request_id)
        request.state.request_id = request_id

        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            log.exception("Unhandled error escaped the exception handlers")
            raise
        finally:
            elapsed = time.perf_counter() - started
            route = _route_template(request)
            if route == "unmatched" and request.url.path.startswith("/static/"):
                route = "/static"
            HTTP_REQUESTS.labels(request.method, route, str(status)).inc()
            HTTP_LATENCY.labels(request.method, route).observe(elapsed)
            if route not in {"/health/live", "/metrics", "/static"}:
                log.info(
                    "%s %s -> %s",
                    request.method,
                    request.url.path,
                    status,
                    extra={
                        "method": request.method,
                        "path": request.url.path,
                        "route": route,
                        "status": status,
                        "duration_ms": round(elapsed * 1000, 2),
                        "client": request.client.host if request.client else None,
                    },
                )
            request_id_ctx.reset(token)

        response.headers[REQUEST_ID_HEADER] = request_id
        self._security_headers(request, response)
        return response

    def _security_headers(self, request: Request, response: Response) -> None:
        headers = response.headers
        is_docs = request.url.path.startswith(DOCS_PATHS)
        headers.setdefault(
            "Content-Security-Policy", DOCS_CSP if is_docs else CONTENT_SECURITY_POLICY
        )
        headers.setdefault("X-Content-Type-Options", "nosniff")
        headers.setdefault("X-Frame-Options", "DENY")
        headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        if self.enable_hsts:
            headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
