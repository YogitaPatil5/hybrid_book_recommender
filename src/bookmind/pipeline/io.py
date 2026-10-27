"""Robust CSV I/O helpers for the pipeline."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import pandas as pd

from bookmind.core.exceptions import DataValidationError

log = logging.getLogger(__name__)


def read_csv(path: Path, *, required: tuple[str, ...] = ()) -> pd.DataFrame:
    """Read a CSV with clear errors for missing files, bad encoding, bad structure."""
    if not path.exists():
        raise DataValidationError(f"Input file not found: {path}", details={"path": str(path)})
    if path.stat().st_size == 0:
        raise DataValidationError(f"Input file is empty: {path}", details={"path": str(path)})
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=True, encoding="utf-8")
    except UnicodeDecodeError:
        log.warning("%s is not UTF-8; retrying as latin-1", path.name)
        df = pd.read_csv(path, dtype=str, keep_default_na=True, encoding="latin-1")
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataValidationError(f"Could not parse {path.name}: {exc}") from exc

    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"{path.name} is missing required columns: {missing}",
            details={"missing": missing, "found": df.columns.tolist()},
        )
    if df.empty:
        raise DataValidationError(f"{path.name} has no data rows.")
    log.info("Loaded %s", path.name, extra={"rows": len(df), "columns": len(df.columns)})
    return df


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write CSV via temp file + rename: an interrupted run never corrupts output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
