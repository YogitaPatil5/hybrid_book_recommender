"""
Application factory.

    uvicorn bookmind.main:app                      # dev
    gunicorn -c gunicorn.conf.py bookmind.main:app # prod

``create_app`` accepts explicit ``settings`` / ``registry`` so tests (and
other entry points) can inject fakes without monkey-patching globals.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.trustedhost import TrustedHostMiddleware

from bookmind import __version__
from bookmind.api import health
from bookmind.api.errors import register_exception_handlers
from bookmind.api.v1 import routes as api_v1
from bookmind.core.config import Settings, get_settings
from bookmind.core.logging import setup_logging
from bookmind.middleware.rate_limit import EXEMPT_PREFIXES, RateLimitMiddleware
from bookmind.middleware.request_context import RequestContextMiddleware
from bookmind.services.registry import ServiceRegistry
from bookmind.web import routes as web

log = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"


def create_app(
    settings: Settings | None = None, registry: ServiceRegistry | None = None
) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if getattr(app.state, "registry", None) is None:
            log.info("Loading engines …", extra={"artifacts_dir": str(settings.artifacts_dir)})
            app.state.registry = ServiceRegistry.load(settings)
        reg: ServiceRegistry = app.state.registry
        log.info(
            "%s v%s started",
            settings.app_name,
            __version__,
            extra={
                "env": settings.env.value,
                "semantic_ready": reg.semantic is not None,
                "knn_status": reg.knn_status,
            },
        )
        yield
        log.info("Shutting down")

    show_docs = not settings.is_production or settings.debug or settings.enable_docs
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="Semantic book search & recommendation API.",
        debug=settings.debug,
        lifespan=lifespan,
        docs_url="/docs" if show_docs else None,
        redoc_url="/redoc" if show_docs else None,
        openapi_url="/openapi.json" if show_docs else None,
    )
    app.state.settings = settings
    app.state.registry = registry
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.state.templates.env.globals["app_version"] = __version__

    register_exception_handlers(app)

    # Middleware: the LAST added runs FIRST (outermost).
    app.add_middleware(
        RateLimitMiddleware,
        limit=settings.rate_limit,
        storage_uri=settings.rate_limit_storage_uri,
        enabled=settings.rate_limit_enabled,
        exempt_prefixes=(*EXEMPT_PREFIXES, *settings.mounted_app_paths),
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_hosts)
    app.add_middleware(
        RequestContextMiddleware,
        enable_hsts=settings.is_production,
        frame_ancestors=tuple(settings.frame_ancestors),
        csp_exempt_prefixes=tuple(settings.mounted_app_paths),
    )

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.include_router(health.router)
    app.include_router(api_v1.router)
    app.include_router(api_v1.legacy_router)
    app.include_router(web.router)
    return app


def __getattr__(name: str) -> FastAPI:
    """Lazily build ``app`` so importing this module has no side effects."""
    if name == "app":
        application = create_app()
        globals()["app"] = application
        return application
    raise AttributeError(name)
