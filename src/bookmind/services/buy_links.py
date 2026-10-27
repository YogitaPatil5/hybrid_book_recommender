"""
"Buy this book" links.

Builds retailer **search-result** URLs (always resolve to a live page). It
deliberately does NOT guess product IDs (ASINs / Flipkart PIDs): those
need each retailer's affiliate API, and a wrong guess sends users to the
wrong book. Swap these functions for real API lookups later without
changing any call sites.
"""

from __future__ import annotations

from urllib.parse import quote_plus

from bookmind.core.exceptions import ValidationError

_UNKNOWN_AUTHORS = frozenset({"", "unknown", "nan", "none", "null", "n/a"})
ALLOWED_AMAZON_DOMAINS = frozenset({"amazon.in", "amazon.com", "amazon.co.uk", "amazon.de"})
MAX_QUERY_CHARS = 200


def build_query(title: str, author: str | None = None) -> str:
    """Return a clean ``"title author"`` search string."""
    if not isinstance(title, str) or not title.strip():
        raise ValidationError("title is required to build buy links.", details={"field": "title"})
    clean_title = " ".join(title.split())
    clean_author = " ".join(str(author).split()) if author is not None else ""
    if clean_author.lower() in _UNKNOWN_AUTHORS:
        clean_author = ""
    query = f"{clean_title} {clean_author}".strip()
    return query[:MAX_QUERY_CHARS]


def amazon_link(title: str, author: str | None = None, domain: str = "amazon.in") -> str:
    if domain not in ALLOWED_AMAZON_DOMAINS:
        raise ValidationError(f"Unsupported Amazon domain '{domain}'.", details={"field": "domain"})
    return f"https://www.{domain}/s?k={quote_plus(build_query(title, author))}&i=stripbooks-intl"


def flipkart_link(title: str, author: str | None = None) -> str:
    return f"https://www.flipkart.com/search?q={quote_plus(build_query(title, author))}"


def google_books_link(title: str, author: str | None = None) -> str:
    return f"https://www.google.com/search?tbm=bks&q={quote_plus(build_query(title, author))}"


def get_buy_links(title: str, author: str | None = None) -> dict[str, str]:
    """Retailer → URL mapping. Templates use ``book.buy_links.amazon`` etc."""
    return {
        "amazon": amazon_link(title, author),
        "flipkart": flipkart_link(title, author),
        "google_books": google_books_link(title, author),
    }
