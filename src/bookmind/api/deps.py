"""FastAPI dependencies (dependency-injection seams; overridable in tests)."""

from __future__ import annotations

from fastapi import Request

from bookmind.core.config import Settings
from bookmind.services.registry import ServiceRegistry
from bookmind.services.semantic_search import SemanticSearchEngine


def get_registry(request: Request) -> ServiceRegistry:
    registry: ServiceRegistry = request.app.state.registry
    return registry


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_semantic_engine(request: Request) -> SemanticSearchEngine:
    return get_registry(request).require_semantic()
