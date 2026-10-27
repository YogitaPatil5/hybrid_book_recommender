"""
Global exception handlers.

Guarantees:
* API clients always receive the ``ErrorResponse`` envelope with a stable
  ``code`` and the ``request_id`` for support/tracing.
* Browser users get a friendly HTML error page instead of raw JSON.
* Stack traces and internal messages are logged, never returned.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from bookmind.core.exceptions import BookMindError

log = logging.getLogger(__name__)

_JSON_PREFIXES = ("/api", "/health", "/metrics")


def wants_json(request: Request) -> bool:
    if request.url.path.startswith(_JSON_PREFIXES):
        return True
    accept = request.headers.get("accept", "")
    return "application/json" in accept and "text/html" not in accept


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> Response:
    if wants_json(request):
        body: dict[str, Any] = {
            "error": {"code": code, "message": message, "request_id": _request_id(request)}
        }
        if details:
            body["error"]["details"] = details
        return JSONResponse(status_code=status_code, content=body, headers=headers)

    templates = request.app.state.templates
    registry = getattr(request.app.state, "registry", None)
    return templates.TemplateResponse(  # type: ignore[no-any-return]
        request,
        "error.html",
        {
            "status_code": status_code,
            "message": message,
            "request_id": _request_id(request),
            "search_ready": bool(registry and registry.semantic is not None),
            "knn_available": bool(registry and registry.knn is not None),
        },
        status_code=status_code,
        headers=headers,
    )


async def bookmind_error_handler(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, BookMindError)
    level = logging.ERROR if exc.status_code >= 500 else logging.INFO
    log.log(level, "%s: %s", exc.code, exc.message, extra={"code": exc.code})
    return error_response(request, exc.status_code, exc.code, exc.message, exc.details or None)


async def validation_error_handler(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, RequestValidationError)
    errors = [
        {
            "field": ".".join(str(p) for p in err.get("loc", ()) if p not in ("query", "body")),
            "message": err.get("msg", "invalid"),
        }
        for err in exc.errors()
    ]
    log.info("Request validation failed", extra={"errors": errors})
    return error_response(
        request, 422, "validation_error", "The request parameters are invalid.", {"errors": errors}
    )


async def http_error_handler(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, StarletteHTTPException)
    code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
    message = str(exc.detail) if exc.detail else "Request failed."
    if exc.status_code == 404:
        message = "The page or resource you requested does not exist."
    return error_response(
        request, exc.status_code, code, message, headers=getattr(exc, "headers", None)
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> Response:
    log.exception("Unhandled exception: %s", type(exc).__name__)
    return error_response(
        request, 500, "internal_error", "Something went wrong on our side. Please try again."
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(BookMindError, bookmind_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
