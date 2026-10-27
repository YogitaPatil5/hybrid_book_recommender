"""
Step 3 — embed every book and build the FAISS index + manifest.

What we embed per book::

    "{title}. {title}. Genre: {genre}. {description}"

The title is repeated because it's the strongest signal for search-by-name;
the description carries the meaning for free-text / "vibe" queries.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Literal

import faiss
import numpy as np
import pandas as pd

from bookmind.core.config import GENRE_TAXONOMY
from bookmind.core.exceptions import DataValidationError, EmbeddingError
from bookmind.ml.embedder import Encoder, HashingEncoder
from bookmind.pipeline.io import read_csv
from bookmind.services.artifacts import META_COLUMNS, Manifest, save_artifacts

log = logging.getLogger(__name__)

MAX_EMBED_CHARS = 2000  # transformer models truncate anyway; bound memory/time


def build_text_for_embedding(row: pd.Series) -> str:
    def s(key: str) -> str:
        value = row.get(key, "")
        return "" if pd.isna(value) else str(value).strip()

    title, genre, desc = s("title"), s("genre"), s("description")
    return f"{title}. {title}. Genre: {genre}. {desc}"[:MAX_EMBED_CHARS]


def prepare_metadata(df: pd.DataFrame) -> pd.DataFrame:
    meta = df.copy()
    for col in META_COLUMNS:
        if col not in meta.columns:
            meta[col] = pd.NA
    for col in ("isbn13", "title", "authors", "description", "genre", "thumbnail"):
        meta[col] = meta[col].fillna("").astype(str)
    meta["average_rating"] = pd.to_numeric(meta["average_rating"], errors="coerce").astype(
        "float64"
    )
    meta["published_year"] = pd.to_numeric(meta["published_year"], errors="coerce").astype("Int64")
    meta["ratings_count"] = pd.to_numeric(meta["ratings_count"], errors="coerce").astype("Int64")
    unknown = ~meta["genre"].isin(GENRE_TAXONOMY)
    if unknown.any():
        log.warning("Mapping %d unknown genres to 'Other'", int(unknown.sum()))
        meta.loc[unknown, "genre"] = "Other"
    result: pd.DataFrame = meta.loc[:, list(META_COLUMNS)].reset_index(drop=True)
    return result


def build_index(
    input_csv: Path,
    artifacts_dir: Path,
    encoder: Encoder,
    *,
    batch_size: int = 64,
) -> Manifest:
    df = read_csv(input_csv, required=("title", "description", "genre"))
    df = df[df["title"].fillna("").str.strip() != ""].reset_index(drop=True)
    if df.empty:
        raise DataValidationError("No books with a title to index.")

    meta = prepare_metadata(df)
    texts = [build_text_for_embedding(row) for _, row in meta.iterrows()]

    started = time.perf_counter()
    log.info("Embedding %d books with %s", len(texts), encoder.name)
    embeddings = encoder.encode(texts, batch_size=batch_size)
    if embeddings.shape != (len(texts), encoder.dim):
        raise EmbeddingError(
            f"Encoder returned shape {embeddings.shape}, expected ({len(texts)}, {encoder.dim})"
        )
    if not np.isfinite(embeddings).all():
        raise EmbeddingError("Encoder produced NaN/inf values.")

    index = faiss.IndexFlatIP(encoder.dim)
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))

    backend: Literal["sentence-transformers", "hashing"] = (
        "hashing" if isinstance(encoder, HashingEncoder) else "sentence-transformers"
    )
    manifest = save_artifacts(
        artifacts_dir, index, meta, embedding_backend=backend, embedding_model=encoder.name
    )
    log.info(
        "build-index complete",
        extra={
            "vectors": manifest.num_books,
            "dim": manifest.dim,
            "seconds": round(time.perf_counter() - started, 2),
        },
    )
    return manifest
