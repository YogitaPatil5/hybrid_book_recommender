"""Liveness, readiness and Prometheus metrics endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from bookmind import __version__
from bookmind.api.deps import get_app_settings, get_registry
from bookmind.core.config import Settings
from bookmind.schemas.search import ComponentStatus, HealthResponse, ReadinessResponse
from bookmind.services.registry import ServiceRegistry

router = APIRouter(tags=["ops"])


@router.get("/health/live", response_model=HealthResponse, summary="Liveness probe")
def live(settings: Annotated[Settings, Depends(get_app_settings)]) -> HealthResponse:
    """The process is up. Never touches dependencies (safe for k8s liveness)."""
    return HealthResponse(version=__version__, env=settings.env.value)


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse}},
    summary="Readiness probe",
)
def ready(registry: Annotated[ServiceRegistry, Depends(get_registry)]) -> JSONResponse:
    """
    ``ready``      — semantic search loaded (k-NN loaded or intentionally absent)
    ``degraded``   — semantic OK, but k-NN artifacts exist and failed to load
    ``unavailable``— semantic search not loaded → 503 so the LB stops routing here
    """
    knn_detail = registry.errors.get("knn")
    knn_failed = registry.knn_status == "failed"
    components = {
        "semantic": ComponentStatus(
            ready=registry.semantic is not None, detail=registry.errors.get("semantic")
        ),
        "knn": ComponentStatus(ready=registry.knn is not None, detail=knn_detail),
    }
    if registry.semantic is None:
        status, code = "unavailable", 503
    elif knn_failed:
        status, code = "degraded", 200
    else:
        status, code = "ready", 200
    body = ReadinessResponse(status=status, components=components)
    return JSONResponse(status_code=code, content=body.model_dump())


@router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
