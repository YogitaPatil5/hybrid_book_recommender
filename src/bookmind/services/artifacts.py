"""
Safe persistence of the semantic-search artifacts.

Replaces the old ``book_meta.pkl`` (pickle can execute arbitrary code on
load) with:

* ``book_index.faiss``   — FAISS vector index
* ``book_meta.parquet``  — catalogue metadata, row-aligned with the index
* ``manifest.json``      — what built the index (model, dim, counts) plus
                           SHA-256 checksums of the two files above

On load, the manifest is validated and checksums verified, so a
half-written or mismatched index is detected at startup rather than as
wrong search results in production.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import faiss
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from bookmind.core.config import INDEX_FILENAME, MANIFEST_FILENAME, META_FILENAME
from bookmind.core.exceptions import ArtifactIntegrityError, ArtifactNotFoundError

log = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = 1
META_COLUMNS: tuple[str, ...] = (
    "isbn13",
    "title",
    "authors",
    "description",
    "genre",
    "thumbnail",
    "average_rating",
    "published_year",
    "ratings_count",
)


class Manifest(BaseModel):
    schema_version: int = Field(MANIFEST_SCHEMA_VERSION, ge=1)
    embedding_backend: Literal["sentence-transformers", "hashing"]
    embedding_model: str
    dim: int = Field(..., ge=1)
    num_books: int = Field(..., ge=1)
    genres: list[str]
    built_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    index_sha256: str
    meta_sha256: str


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write via temp file + rename so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# Public alias for other modules (pipeline) that need atomic writes.
atomic_write_bytes = _atomic_write_bytes


def save_artifacts(
    artifacts_dir: Path,
    index: Any,
    meta: pd.DataFrame,
    *,
    embedding_backend: Literal["sentence-transformers", "hashing"],
    embedding_model: str,
) -> Manifest:
    """Persist index + metadata + manifest atomically-ish (manifest last)."""
    if index.ntotal != len(meta):
        raise ArtifactIntegrityError(
            f"Index has {index.ntotal} vectors but metadata has {len(meta)} rows."
        )
    missing = [c for c in META_COLUMNS if c not in meta.columns]
    if missing:
        raise ArtifactIntegrityError(f"Metadata missing columns: {missing}")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    index_path = artifacts_dir / INDEX_FILENAME
    meta_path = artifacts_dir / META_FILENAME
    manifest_path = artifacts_dir / MANIFEST_FILENAME

    # Remove the old manifest first: if we crash mid-way the service refuses
    # to load inconsistent files instead of trusting stale checksums.
    manifest_path.unlink(missing_ok=True)

    _atomic_write_bytes(index_path, faiss.serialize_index(index).tobytes())
    tmp_meta = meta_path.with_suffix(".parquet.tmp")
    meta.loc[:, list(META_COLUMNS)].reset_index(drop=True).to_parquet(tmp_meta, index=False)
    tmp_meta.replace(meta_path)

    manifest = Manifest(
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        dim=int(index.d),
        num_books=int(index.ntotal),
        genres=sorted(meta["genre"].dropna().astype(str).unique().tolist()),
        index_sha256=sha256_file(index_path),
        meta_sha256=sha256_file(meta_path),
    )
    _atomic_write_bytes(manifest_path, manifest.model_dump_json(indent=2).encode("utf-8"))
    log.info(
        "Saved artifacts",
        extra={"num_books": manifest.num_books, "dim": manifest.dim, "dir": str(artifacts_dir)},
    )
    return manifest


def load_manifest(artifacts_dir: Path) -> Manifest:
    path = artifacts_dir / MANIFEST_FILENAME
    if not path.exists():
        raise ArtifactNotFoundError(details={"missing": MANIFEST_FILENAME})
    try:
        return Manifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ArtifactIntegrityError(f"Manifest is invalid: {exc}") from exc


def load_artifacts(
    artifacts_dir: Path, *, verify_checksums: bool = True
) -> tuple[Any, pd.DataFrame, Manifest]:
    """Load and cross-validate index, metadata and manifest."""
    index_path = artifacts_dir / INDEX_FILENAME
    meta_path = artifacts_dir / META_FILENAME
    missing = [p.name for p in (index_path, meta_path) if not p.exists()]
    if missing:
        raise ArtifactNotFoundError(details={"missing": missing})

    manifest = load_manifest(artifacts_dir)
    if manifest.schema_version > MANIFEST_SCHEMA_VERSION:
        raise ArtifactIntegrityError(
            f"Manifest schema v{manifest.schema_version} is newer than supported "
            f"v{MANIFEST_SCHEMA_VERSION}. Upgrade the service."
        )

    if verify_checksums:
        for path, expected in (
            (index_path, manifest.index_sha256),
            (meta_path, manifest.meta_sha256),
        ):
            actual = sha256_file(path)
            if actual != expected:
                raise ArtifactIntegrityError(
                    f"Checksum mismatch for {path.name}.",
                    details={"expected": expected[:12], "actual": actual[:12]},
                )

    try:
        index = faiss.read_index(str(index_path))
        meta = pd.read_parquet(meta_path)
    except Exception as exc:
        raise ArtifactIntegrityError(f"Failed to read artifacts: {exc}") from exc

    if index.ntotal != len(meta) or index.ntotal != manifest.num_books:
        raise ArtifactIntegrityError(
            "Row count mismatch between index, metadata and manifest.",
            details={"index": index.ntotal, "meta": len(meta), "manifest": manifest.num_books},
        )
    if index.d != manifest.dim:
        raise ArtifactIntegrityError(f"Index dim {index.d} != manifest dim {manifest.dim}.")
    missing_cols = [c for c in META_COLUMNS if c not in meta.columns]
    if missing_cols:
        raise ArtifactIntegrityError(f"Metadata missing columns: {missing_cols}")

    return index, meta.reset_index(drop=True), manifest
