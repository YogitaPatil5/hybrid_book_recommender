"""
Step 0 — download both Kaggle datasets with ``kagglehub`` (free).

* ``dylanjcastillo/7k-books-with-metadata``  → ``data/books_raw.csv``
  (titles + descriptions: needed by the semantic search engine)
* ``ra4u12/bookrecommendation`` (Book-Crossing) → ``data/bx/BX-*.csv``
  (user ratings: needed by the collaborative-filtering k-NN engine)

Public datasets usually download without logging in. If Kaggle asks for
credentials, create a free account and either run ``kagglehub.login()`` once
or place ``kaggle.json`` in ``~/.kaggle/``.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path

from bookmind.core.exceptions import ConfigurationError, DataValidationError

log = logging.getLogger(__name__)

SEMANTIC_DATASET = "dylanjcastillo/7k-books-with-metadata"
BX_DATASET = "ra4u12/bookrecommendation"
SEMANTIC_FILE = "books.csv"
BX_REQUIRED = ("BX-Books.csv", "BX-Book-Ratings.csv")
BX_OPTIONAL = ("BX-Users.csv",)

Downloader = Callable[[str], str]


def _default_downloader() -> Downloader:
    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - environment specific
        raise ConfigurationError(
            "kagglehub is not installed: pip install -r requirements-pipeline.txt"
        ) from exc
    fn: Downloader = kagglehub.dataset_download
    return fn


def _find(root: Path, name: str) -> Path | None:
    matches = [p for p in root.rglob("*") if p.is_file() and p.name.lower() == name.lower()]
    return min(matches, key=lambda p: len(p.parts)) if matches else None


def _fetch(dataset: str, downloader: Downloader) -> Path:
    log.info("Downloading Kaggle dataset %s …", dataset)
    try:
        path = Path(downloader(dataset))
    except Exception as exc:
        raise DataValidationError(
            f"Could not download '{dataset}': {exc}. Check your internet connection; if Kaggle "
            'asks for credentials run `python -c "import kagglehub; kagglehub.login()"` once.',
            details={"dataset": dataset},
        ) from exc
    if not path.exists():
        raise DataValidationError(f"Downloader returned a missing path: {path}")
    return path


def download_datasets(
    data_dir: Path,
    *,
    which: Iterable[str] = ("semantic", "knn"),
    downloader: Downloader | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Download and place the datasets; returns {logical name: destination path}."""
    which = set(which)
    unknown = which - {"semantic", "knn"}
    if unknown:
        raise DataValidationError(f"Unknown dataset selection: {sorted(unknown)}")
    downloader = downloader or _default_downloader()
    data_dir.mkdir(parents=True, exist_ok=True)
    placed: dict[str, Path] = {}

    if "semantic" in which:
        dest = data_dir / "books_raw.csv"
        if dest.exists() and not force:
            log.info("%s already exists — skipping (use --force to re-download)", dest)
        else:
            src = _find(_fetch(SEMANTIC_DATASET, downloader), SEMANTIC_FILE)
            if src is None:
                raise DataValidationError(f"'{SEMANTIC_FILE}' not found in {SEMANTIC_DATASET}.")
            shutil.copy2(src, dest)
        placed["books_raw"] = dest

    if "knn" in which:
        bx_dir = data_dir / "bx"
        bx_dir.mkdir(parents=True, exist_ok=True)
        if all((bx_dir / f).exists() for f in BX_REQUIRED) and not force:
            log.info("Book-Crossing files already in %s — skipping", bx_dir)
        else:
            root = _fetch(BX_DATASET, downloader)
            for name in BX_REQUIRED + BX_OPTIONAL:
                src = _find(root, name)
                if src is None:
                    if name in BX_REQUIRED:
                        found = sorted(p.name for p in root.rglob("*.csv"))
                        raise DataValidationError(
                            f"'{name}' not found in {BX_DATASET}.", details={"found": found}
                        )
                    continue
                shutil.copy2(src, bx_dir / name)
        placed["bx_dir"] = bx_dir

    log.info("download complete", extra={k: str(v) for k, v in placed.items()})
    return placed
