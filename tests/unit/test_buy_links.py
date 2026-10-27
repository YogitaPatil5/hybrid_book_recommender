from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest

from bookmind.core.exceptions import ValidationError
from bookmind.schemas.book import BuyLinks
from bookmind.services.buy_links import (
    MAX_QUERY_CHARS,
    amazon_link,
    build_query,
    flipkart_link,
    get_buy_links,
    google_books_link,
)


def test_query_includes_author() -> None:
    assert build_query("Dune", "Frank Herbert") == "Dune Frank Herbert"


@pytest.mark.parametrize("author", [None, "", "Unknown", "nan", "NaN", "None", "  "])
def test_unknown_author_is_dropped(author: str | None) -> None:
    assert build_query("Dune", author) == "Dune"


def test_whitespace_is_collapsed() -> None:
    assert build_query("  The   Hobbit ", " J.R.R.  Tolkien ") == "The Hobbit J.R.R. Tolkien"


@pytest.mark.parametrize("title", ["", "   ", None, 42])
def test_title_required(title: object) -> None:
    with pytest.raises(ValidationError):
        build_query(title)  # type: ignore[arg-type]


def test_query_is_capped() -> None:
    assert len(build_query("x" * 1000)) == MAX_QUERY_CHARS


def test_special_characters_are_url_encoded() -> None:
    url = amazon_link("Rock & Roll #1?", "A/B")
    params = parse_qs(urlparse(url).query)
    assert params["k"] == ["Rock & Roll #1? A/B"]
    assert params["i"] == ["stripbooks-intl"]


def test_amazon_domain_whitelist() -> None:
    assert amazon_link("Dune", domain="amazon.com").startswith("https://www.amazon.com/")
    with pytest.raises(ValidationError):
        amazon_link("Dune", domain="evil.example.com")


def test_flipkart_and_google() -> None:
    assert flipkart_link("Dune").startswith("https://www.flipkart.com/search?q=Dune")
    assert "tbm=bks" in google_books_link("Dune")


def test_get_buy_links_validates_as_schema() -> None:
    links = get_buy_links("Dune", "Frank Herbert")
    assert set(links) == {"amazon", "flipkart", "google_books"}
    BuyLinks.model_validate(links)  # all are valid https URLs
