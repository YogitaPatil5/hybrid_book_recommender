"""
Centralised, validated application configuration.

All settings come from environment variables prefixed with ``BOOKMIND_``
(or a local ``.env`` file). Invalid values fail fast at startup with a
clear message instead of surfacing as obscure runtime errors.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Canonical artifact file names, shared by the pipeline and the service.
INDEX_FILENAME = "book_index.faiss"
META_FILENAME = "book_meta.parquet"
MANIFEST_FILENAME = "manifest.json"
KNN_ARTIFACTS = ("model_knn", "book_names", "final_rating", "book_pivot")

GENRE_TAXONOMY: tuple[str, ...] = (
    "Fantasy",
    "Science Fiction",
    "Mystery",
    "Thriller",
    "Romance",
    "Horror",
    "Historical Fiction",
    "Literary Fiction",
    "Young Adult",
    "Children's",
    "Biography & Memoir",
    "Self-Help",
    "History",
    "Science & Nature",
    "Business & Economics",
    "Philosophy & Religion",
    "Poetry",
    "True Crime",
    "Humor",
    "Other",
)


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class Settings(BaseSettings):
    """Application settings (12-factor style)."""

    model_config = SettingsConfigDict(
        env_prefix="BOOKMIND_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Runtime ──────────────────────────────────────────────────────────────
    app_name: str = "BookMind Pro"
    env: Environment = Environment.DEVELOPMENT
    debug: bool = False
    enable_docs: bool = False  # show /docs in production (e.g. a public demo)
    host: str = "0.0.0.0"  # noqa: S104 - bound inside a container; fronted by a proxy
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=2, ge=1, le=64)

    # ── Paths ────────────────────────────────────────────────────────────────
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    log_dir: Path = Path("logs")

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_json: bool = False
    log_to_file: bool = True
    log_max_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    log_backup_count: int = Field(default=5, ge=0, le=100)

    # ── ML ───────────────────────────────────────────────────────────────────
    embedding_backend: Literal["sentence-transformers", "hashing"] = "sentence-transformers"
    embedding_model: str = Field(default="all-MiniLM-L6-v2", min_length=1)
    embedding_dim: int = Field(default=384, ge=8, le=4096, description="hashing backend only")
    embedding_cache_size: int = Field(default=2048, ge=0, le=1_000_000)
    enable_knn: bool = True

    # ── API limits & security ────────────────────────────────────────────────
    max_query_length: int = Field(default=300, ge=10, le=5000)
    max_top_k: int = Field(default=50, ge=1, le=500)
    default_top_k: int = Field(default=12, ge=1, le=500)
    autocomplete_limit: int = Field(default=10, ge=1, le=50)
    popular_books_count: int = Field(default=50, ge=1, le=500)
    rate_limit: str = "60/minute"
    rate_limit_enabled: bool = True
    rate_limit_storage_uri: str = Field(
        default="memory://", description="Use redis://host:6379 to share limits across workers"
    )
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8000"])
    trusted_hosts: list[str] = Field(default_factory=lambda: ["*"])

    # ── Pipeline ─────────────────────────────────────────────────────────────
    ollama_model: str = "llama3.2"
    ollama_host: str = "http://localhost:11434"
    embed_batch_size: int = Field(default=64, ge=1, le=4096)

    # ── Validators ───────────────────────────────────────────────────────────
    @field_validator("rate_limit")
    @classmethod
    def _validate_rate_limit(cls, v: str) -> str:
        parts = v.split("/")
        if len(parts) != 2 or not parts[0].strip().isdigit() or int(parts[0]) <= 0:
            raise ValueError("rate_limit must look like '<count>/<period>', e.g. '60/minute'")
        if parts[1].strip() not in {"second", "minute", "hour", "day"}:
            raise ValueError("rate_limit period must be one of second|minute|hour|day")
        return v

    @model_validator(mode="after")
    def _cross_field_checks(self) -> Settings:
        if self.default_top_k > self.max_top_k:
            raise ValueError("default_top_k cannot exceed max_top_k")
        if self.env is Environment.PRODUCTION and self.debug:
            raise ValueError("debug must be false when env=production")
        # Resolve paths once so every component agrees on absolute locations.
        self.data_dir = self.data_dir.expanduser().resolve()
        self.artifacts_dir = self.artifacts_dir.expanduser().resolve()
        self.log_dir = self.log_dir.expanduser().resolve()
        return self

    # ── Derived paths ────────────────────────────────────────────────────────
    @property
    def index_path(self) -> Path:
        return self.artifacts_dir / INDEX_FILENAME

    @property
    def meta_path(self) -> Path:
        return self.artifacts_dir / META_FILENAME

    @property
    def manifest_path(self) -> Path:
        return self.artifacts_dir / MANIFEST_FILENAME

    @property
    def is_production(self) -> bool:
        return self.env is Environment.PRODUCTION


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
