"""
Service registry: owns the long-lived engines for the process lifetime.

Engines are loaded once at startup (FastAPI lifespan). If an engine fails
to load, the app still starts in *degraded* mode: pages render a helpful
message, the API returns 503 with a clear error code, and ``/health/ready``
reports exactly which component is down — instead of the old behaviour of
crashing on import.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from bookmind.core.exceptions import BookMindError, ServiceUnavailableError
from bookmind.services.knn_recommender import KnnRecommender, knn_artifacts_present
from bookmind.services.semantic_search import SemanticSearchEngine

if TYPE_CHECKING:
    from bookmind.core.config import Settings
    from bookmind.ml.embedder import Encoder

log = logging.getLogger(__name__)


@dataclass
class ServiceRegistry:
    settings: Settings
    semantic: SemanticSearchEngine | None = None
    knn: KnnRecommender | None = None
    errors: dict[str, str] = field(default_factory=dict)
    knn_status: Literal["loaded", "absent", "disabled", "failed"] = "absent"

    @classmethod
    def load(cls, settings: Settings, encoder: Encoder | None = None) -> ServiceRegistry:
        registry = cls(settings=settings)
        registry._load_semantic(encoder)
        registry._load_knn()
        return registry

    def _load_semantic(self, encoder: Encoder | None) -> None:
        try:
            self.semantic = SemanticSearchEngine.from_artifacts(self.settings, encoder=encoder)
        except BookMindError as exc:
            self.errors["semantic"] = exc.message
            log.error("Semantic engine unavailable: %s", exc.message, extra={"code": exc.code})
        except Exception as exc:  # defensive: never crash the process on load
            self.errors["semantic"] = "Unexpected error while loading the search index."
            log.exception("Semantic engine failed to load: %s", exc)

    def _load_knn(self) -> None:
        if not self.settings.enable_knn:
            self.errors["knn"] = "Disabled by configuration (BOOKMIND_ENABLE_KNN=false)."
            self.knn_status = "disabled"
            return
        if not knn_artifacts_present(self.settings.artifacts_dir):
            self.errors["knn"] = "Legacy k-NN artifacts not present (optional)."
            self.knn_status = "absent"
            log.info("Legacy k-NN artifacts not found — /recommend uses semantic similarity.")
            return
        try:
            self.knn = KnnRecommender.from_artifacts(
                self.settings.artifacts_dir,
                max_top_k=self.settings.max_top_k,
                max_query_length=self.settings.max_query_length,
            )
            self.knn_status = "loaded"
        except BookMindError as exc:
            self.knn_status = "failed"
            self.errors["knn"] = exc.message
            log.error("k-NN recommender unavailable: %s", exc.message)
        except Exception as exc:
            self.knn_status = "failed"
            self.errors["knn"] = "Unexpected error while loading k-NN artifacts."
            log.exception("k-NN recommender failed to load: %s", exc)

    # ── Accessors that raise a clean 503 when a component is missing ─────────
    def require_semantic(self) -> SemanticSearchEngine:
        if self.semantic is None:
            raise ServiceUnavailableError(
                "Search is unavailable: the index has not been built. "
                "Run `bookmind-pipeline all` and restart the service.",
                details={"component": "semantic", "reason": self.errors.get("semantic")},
            )
        return self.semantic
