"""
Step 1 — normalise a raw books CSV (e.g. Kaggle "7k-books-with-metadata")
into the pipeline schema.

Validation & cleaning
---------------------
* required columns: title, authors, description
* strips HTML tags / control characters, collapses whitespace
* drops rows with missing title or a description shorter than ``min_description_chars``
* coerces numbers: rating ∈ [0, 5], year ∈ [1000, current+1], ratings_count ≥ 0
* keeps only http(s) thumbnails
* de-duplicates by ISBN-13, then by (title, authors)
* assigns a stable ``book_id`` (ISBN-13 when valid, otherwise a content hash)
"""

from __future__ import annotations

import hashlib
import html
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bookmind.core.exceptions import DataValidationError
from bookmind.pipeline.io import read_csv, write_csv_atomic

log = logging.getLogger(__name__)

REQUIRED_SOURCE_COLUMNS = ("title", "authors", "description")
OUTPUT_COLUMNS = (
    "book_id",
    "isbn13",
    "title",
    "authors",
    "description",
    "categories",
    "thumbnail",
    "average_rating",
    "published_year",
    "ratings_count",
)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_ISBN13_RE = re.compile(r"^\d{13}$")


@dataclass(frozen=True)
class PrepareReport:
    input_rows: int
    output_rows: int
    dropped_missing: int
    dropped_short: int
    dropped_duplicates: int


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    if value is pd.NA:
        return ""
    text = html.unescape(str(value))
    text = _TAG_RE.sub(" ", text)
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    return _WS_RE.sub(" ", text).strip()


def normalise_authors(value: object) -> str:
    text = clean_text(value)
    if not text:
        return "Unknown"
    # Kaggle data separates multiple authors with ';'
    parts = [p.strip() for p in text.split(";") if p.strip()]
    return ", ".join(parts) if parts else "Unknown"


def normalise_isbn13(value: object) -> str:
    digits = re.sub(r"\D", "", clean_text(value).split(".")[0])
    return digits if _ISBN13_RE.match(digits) else ""


def _make_book_id(row: pd.Series) -> str:
    if row["isbn13"]:
        return str(row["isbn13"])
    key = f"{row['title'].lower()}|{row['authors'].lower()}"
    return "h" + hashlib.sha1(key.encode("utf-8"), usedforsecurity=False).hexdigest()[:15]


def _numeric(series: pd.Series, lo: float, hi: float) -> pd.Series:
    nums = pd.to_numeric(series, errors="coerce")
    result: pd.Series = nums.where((nums >= lo) & (nums <= hi))
    return result


def prepare_dataframe(
    df: pd.DataFrame, *, min_description_chars: int = 30
) -> tuple[pd.DataFrame, PrepareReport]:
    missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Source data missing required columns: {missing}")
    if min_description_chars < 0:
        raise DataValidationError("min_description_chars must be >= 0")

    n_input = len(df)
    out = pd.DataFrame(index=df.index)
    out["title"] = df["title"].map(clean_text)
    out["authors"] = df["authors"].map(normalise_authors)
    out["description"] = df["description"].map(clean_text)
    out["categories"] = df["categories"].map(clean_text) if "categories" in df else ""
    out["isbn13"] = df["isbn13"].map(normalise_isbn13) if "isbn13" in df else ""

    if "thumbnail" in df:
        thumbs = df["thumbnail"].map(clean_text)
        thumbs = thumbs.str.replace(r"^http://", "https://", regex=True)
        out["thumbnail"] = thumbs.where(thumbs.str.match(r"^https://"), "")
    else:
        out["thumbnail"] = ""

    year_max = datetime.now(timezone.utc).year + 1
    out["average_rating"] = (
        _numeric(df["average_rating"], 0, 5) if "average_rating" in df else float("nan")
    )
    out["published_year"] = (
        _numeric(df["published_year"], 1000, year_max).astype("Int64")
        if "published_year" in df
        else pd.NA
    )
    out["ratings_count"] = (
        _numeric(df["ratings_count"], 0, 1e10).astype("Int64") if "ratings_count" in df else pd.NA
    )

    has_required = (out["title"] != "") & (out["description"] != "")
    dropped_missing = int((~has_required).sum())
    out = out[has_required]

    long_enough = out["description"].str.len() >= min_description_chars
    dropped_short = int((~long_enough).sum())
    out = out[long_enough]

    before = len(out)
    with_isbn = out[out["isbn13"] != ""].drop_duplicates("isbn13", keep="first")
    out = pd.concat([with_isbn, out[out["isbn13"] == ""]])
    dedup_key = out["title"].str.lower() + "|" + out["authors"].str.lower()
    out = out.loc[~dedup_key.duplicated(keep="first")].sort_index()
    dropped_dupes = before - len(out)

    if out.empty:
        raise DataValidationError("No usable rows remain after cleaning.")

    out = out.reset_index(drop=True)
    out["book_id"] = out.apply(_make_book_id, axis=1)
    if out["book_id"].duplicated().any():  # pragma: no cover - guarded by dedup above
        raise DataValidationError("Duplicate book_id values generated.")

    report = PrepareReport(n_input, len(out), dropped_missing, dropped_short, dropped_dupes)
    return out.loc[:, list(OUTPUT_COLUMNS)], report


def run_prepare(
    input_csv: Path, output_csv: Path, *, min_description_chars: int = 30
) -> PrepareReport:
    raw = read_csv(input_csv, required=REQUIRED_SOURCE_COLUMNS)
    prepared, report = prepare_dataframe(raw, min_description_chars=min_description_chars)
    write_csv_atomic(prepared, output_csv)
    log.info(
        "prepare complete → %s",
        output_csv,
        extra={
            "input_rows": report.input_rows,
            "output_rows": report.output_rows,
            "dropped_missing": report.dropped_missing,
            "dropped_short": report.dropped_short,
            "dropped_duplicates": report.dropped_duplicates,
        },
    )
    return report
