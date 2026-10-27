from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import pytest

from bookmind.core.exceptions import ArtifactIntegrityError, ArtifactNotFoundError
from bookmind.services.artifacts import (
    META_COLUMNS,
    load_artifacts,
    load_manifest,
    save_artifacts,
    sha256_file,
)


def _meta(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "isbn13": [str(i) for i in range(n)],
            "title": [f"Book {i}" for i in range(n)],
            "authors": ["A"] * n,
            "description": ["d"] * n,
            "genre": ["Fantasy"] * n,
            "thumbnail": [""] * n,
            "average_rating": [4.0] * n,
            "published_year": pd.array([2000] * n, dtype="Int64"),
            "ratings_count": pd.array([10] * n, dtype="Int64"),
        }
    )


def _index(n: int, d: int = 8) -> faiss.IndexFlatIP:
    idx = faiss.IndexFlatIP(d)
    idx.add(np.random.default_rng(0).random((n, d), dtype=np.float32))
    return idx


def test_roundtrip(tmp_path: Path) -> None:
    manifest = save_artifacts(
        tmp_path, _index(5), _meta(5), embedding_backend="hashing", embedding_model="hashing-8"
    )
    index, meta, loaded = load_artifacts(tmp_path)
    assert index.ntotal == 5
    assert list(meta.columns) == list(META_COLUMNS)
    assert loaded == manifest
    assert loaded.genres == ["Fantasy"]
    assert loaded.index_sha256 == sha256_file(tmp_path / "book_index.faiss")


def test_no_pickle_files_written(tmp_path: Path) -> None:
    save_artifacts(tmp_path, _index(3), _meta(3), embedding_backend="hashing", embedding_model="h")
    assert not list(tmp_path.glob("*.pkl"))


def test_save_rejects_size_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ArtifactIntegrityError, match="vectors"):
        save_artifacts(
            tmp_path, _index(4), _meta(5), embedding_backend="hashing", embedding_model="h"
        )


def test_save_rejects_missing_columns(tmp_path: Path) -> None:
    with pytest.raises(ArtifactIntegrityError, match="missing columns"):
        save_artifacts(
            tmp_path,
            _index(2),
            _meta(2).drop(columns=["genre"]),
            embedding_backend="hashing",
            embedding_model="h",
        )


def test_missing_files(tmp_path: Path) -> None:
    with pytest.raises(ArtifactNotFoundError) as exc:
        load_artifacts(tmp_path)
    assert exc.value.status_code == 503


def test_missing_manifest(tmp_path: Path) -> None:
    save_artifacts(tmp_path, _index(2), _meta(2), embedding_backend="hashing", embedding_model="h")
    (tmp_path / "manifest.json").unlink()
    with pytest.raises(ArtifactNotFoundError):
        load_artifacts(tmp_path)


def test_corrupt_manifest(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{not json")
    with pytest.raises(ArtifactIntegrityError, match="Manifest is invalid"):
        load_manifest(tmp_path)


def test_manifest_schema_validation(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"dim": -1}))
    with pytest.raises(ArtifactIntegrityError):
        load_manifest(tmp_path)


def test_tampered_index_detected(tmp_path: Path) -> None:
    save_artifacts(tmp_path, _index(3), _meta(3), embedding_backend="hashing", embedding_model="h")
    faiss.write_index(_index(3), str(tmp_path / "book_index.faiss"))  # different random? same seed
    (tmp_path / "book_index.faiss").write_bytes(
        (tmp_path / "book_index.faiss").read_bytes() + b"\x00"
    )
    with pytest.raises(ArtifactIntegrityError, match="Checksum mismatch"):
        load_artifacts(tmp_path)


def test_tampered_meta_detected_and_skippable(tmp_path: Path) -> None:
    save_artifacts(tmp_path, _index(3), _meta(3), embedding_backend="hashing", embedding_model="h")
    _meta(3).assign(title="changed").to_parquet(tmp_path / "book_meta.parquet")
    with pytest.raises(ArtifactIntegrityError, match="Checksum"):
        load_artifacts(tmp_path)
    # With verification off the files still load (rows still align).
    _, meta, _ = load_artifacts(tmp_path, verify_checksums=False)
    assert set(meta["title"]) == {"changed"}


def test_row_count_mismatch_detected(tmp_path: Path) -> None:
    save_artifacts(tmp_path, _index(3), _meta(3), embedding_backend="hashing", embedding_model="h")
    _meta(2).to_parquet(tmp_path / "book_meta.parquet")
    with pytest.raises(ArtifactIntegrityError, match="Row count"):
        load_artifacts(tmp_path, verify_checksums=False)


def test_newer_schema_rejected(tmp_path: Path) -> None:
    save_artifacts(tmp_path, _index(2), _meta(2), embedding_backend="hashing", embedding_model="h")
    path = tmp_path / "manifest.json"
    data = json.loads(path.read_text())
    data["schema_version"] = 99
    path.write_text(json.dumps(data))
    with pytest.raises(ArtifactIntegrityError, match="newer"):
        load_artifacts(tmp_path)


def test_unreadable_index(tmp_path: Path) -> None:
    save_artifacts(tmp_path, _index(2), _meta(2), embedding_backend="hashing", embedding_model="h")
    (tmp_path / "book_index.faiss").write_bytes(b"garbage")
    with pytest.raises(ArtifactIntegrityError, match="Failed to read"):
        load_artifacts(tmp_path, verify_checksums=False)


def test_dim_mismatch(tmp_path: Path) -> None:
    save_artifacts(tmp_path, _index(2), _meta(2), embedding_backend="hashing", embedding_model="h")
    path = tmp_path / "manifest.json"
    data = json.loads(path.read_text())
    data["dim"] = 16
    path.write_text(json.dumps(data))
    with pytest.raises(ArtifactIntegrityError, match="dim"):
        load_artifacts(tmp_path)


def test_missing_meta_columns_on_load(tmp_path: Path) -> None:
    save_artifacts(tmp_path, _index(2), _meta(2), embedding_backend="hashing", embedding_model="h")
    _meta(2).drop(columns=["genre"]).to_parquet(tmp_path / "book_meta.parquet")
    with pytest.raises(ArtifactIntegrityError, match="missing columns"):
        load_artifacts(tmp_path, verify_checksums=False)
