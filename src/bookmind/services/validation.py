"""
Input sanitisation & validation shared by the API, web pages and engines.

Rules are applied in one place so a query is treated identically whether
it arrives through ``/api/v1/search``, the HTML form, or a direct call.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from bookmind.core.exceptions import ValidationError

_WHITESPACE_RE = re.compile(r"\s+")
ALL_GENRES = "All"


def clean_text(value: object, *, field: str = "query", max_length: int = 300) -> str:
    """
    Normalise free text: NFKC-normalise, strip control characters, collapse
    whitespace, enforce non-empty and a maximum length.
    """
    if value is None:
        raise ValidationError(f"{field} is required.", details={"field": field})
    if not isinstance(value, str):
        raise ValidationError(f"{field} must be a string.", details={"field": field})

    text = unicodedata.normalize("NFKC", value)
    text = "".join(ch for ch in text if ch.isprintable() or ch.isspace())
    text = _WHITESPACE_RE.sub(" ", text).strip()

    if not text:
        raise ValidationError(f"{field} must not be empty.", details={"field": field})
    if len(text) > max_length:
        raise ValidationError(
            f"{field} must be at most {max_length} characters.",
            details={"field": field, "max_length": max_length, "length": len(text)},
        )
    return text


def validate_top_k(value: object, *, max_top_k: int) -> int:
    """Ensure ``top_k`` is an int in ``[1, max_top_k]``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError("top_k must be an integer.", details={"field": "top_k"})
    if not 1 <= value <= max_top_k:
        raise ValidationError(
            f"top_k must be between 1 and {max_top_k}.",
            details={"field": "top_k", "min": 1, "max": max_top_k},
        )
    return value


def validate_genre(value: str | None, allowed: Iterable[str]) -> str:
    """Return a canonical genre name or ``"All"``; reject unknown genres."""
    if value is None:
        return ALL_GENRES
    genre = value.strip()
    if not genre or genre.lower() == ALL_GENRES.lower():
        return ALL_GENRES
    lookup = {g.lower(): g for g in allowed}
    canonical = lookup.get(genre.lower())
    if canonical is None:
        raise ValidationError(
            f"Unknown genre '{genre[:50]}'.",
            details={"field": "genre", "allowed": sorted(lookup.values())},
        )
    return canonical
