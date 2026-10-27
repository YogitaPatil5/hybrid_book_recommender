"""Server-rendered HTML pages."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


def test_home(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "Trending &amp; Popular" in r.text
    assert "Harry Potter" in r.text
    assert "via.placeholder.com" not in r.text  # dead service removed
    assert "Search index not built" not in r.text


def test_home_with_knn(knn_client: TestClient) -> None:
    r = knn_client.get("/")
    assert "Author of" in r.text and "Ratings" in r.text


def test_search_page_empty_state(client: TestClient) -> None:
    r = client.get("/search")
    assert r.status_code == 200 and "Three Ways to Search" in r.text
    assert '<option value="Fantasy"' in r.text


def test_search_results(client: TestClient) -> None:
    r = client.get("/search", params={"q": "haunted hotel ghosts"})
    assert r.status_code == 200 and "The Shining" in r.text
    assert "book found" in r.text or "books found" in r.text


def test_search_genre_selected(client: TestClient) -> None:
    r = client.get("/search", params={"q": "murder", "genre": "Mystery"})
    assert '<option value="Mystery" selected' in r.text


def test_search_validation_error_rendered(client: TestClient) -> None:
    r = client.get("/search", params={"q": "x", "genre": "Cooking"})
    assert r.status_code == 422 and "Unknown genre" in r.text


def test_output_is_html_escaped(client: TestClient) -> None:
    r = client.get("/search", params={"q": '<script>alert("x")</script>'})
    assert '<script>alert("x")</script>' not in r.text
    assert "&lt;script&gt;" in r.text


def test_no_inline_event_handlers(client: TestClient) -> None:
    for path in ("/", "/search?q=dragon", "/recommend?q=Dune", "/about"):
        html = client.get(path).text
        assert "onclick=" not in html and "onerror=" not in html


def test_recommend_get_prefill(client: TestClient) -> None:
    r = client.get("/recommend", params={"q": "Dune"})
    assert r.status_code == 200 and "Because you liked" in r.text
    assert "most similar themes" in r.text


def test_recommend_post(client: TestClient) -> None:
    r = client.post("/recommend", data={"book_name": "The Hobbit"})
    assert r.status_code == 200 and r.text.count('class="result-card"') == 5


def test_recommend_post_empty(client: TestClient) -> None:
    r = client.post("/recommend", data={"book_name": "   "})
    assert r.status_code == 422 and "Please enter a book title" in r.text


def test_recommend_unknown_title(client: TestClient) -> None:
    r = client.post("/recommend", data={"book_name": "zzzzzz"})
    assert r.status_code == 404 and "not found" in r.text


def test_recommend_knn_wording(knn_client: TestClient) -> None:
    r = knn_client.get("/recommend", params={"q": "Dune"})
    assert "Readers who rated this book" in r.text


def test_about(client: TestClient) -> None:
    r = client.get("/about")
    assert r.status_code == 200 and "hashing-384" in r.text


def test_static_assets(client: TestClient) -> None:
    for path in (
        "/static/css/style.css",
        "/static/js/main.js",
        "/static/js/autocomplete.js",
        "/static/img/no-cover.svg",
    ):
        assert client.get(path).status_code == 200


def test_404_page_is_html(client: TestClient) -> None:
    r = client.get("/does-not-exist", headers={"accept": "text/html"})
    assert r.status_code == 404 and "<!DOCTYPE html>" in r.text and "Reference:" in r.text


def test_405(client: TestClient) -> None:
    assert client.delete("/search").status_code == 405


class TestDegraded:
    def test_pages_render_with_banner(self, empty_client: TestClient) -> None:
        for path in ("/", "/about", "/recommend"):
            r = empty_client.get(path)
            assert r.status_code == 200 and "Search index not built" in r.text

    def test_search_503(self, empty_client: TestClient) -> None:
        r = empty_client.get("/search", params={"q": "x"})
        assert r.status_code == 503 and "isn't been built" not in r.text

    def test_recommend_503(self, empty_client: TestClient) -> None:
        assert empty_client.get("/recommend", params={"q": "Dune"}).status_code == 503


def test_all_cover_images_are_https_for_csp(client: TestClient) -> None:
    """Regression: http:// covers were blocked by the CSP (img-src https:)."""
    import re

    html = client.get("/").text + client.get("/search", params={"q": "story"}).text
    srcs = re.findall(r'<img[^>]+src="([^"]+)"', html)
    assert srcs
    assert all(s.startswith(("https://", "http://testserver/static/")) for s in srcs)


def test_search_input_not_required(client: TestClient) -> None:
    """Regression: users must be able to browse a genre without typing a query."""
    html = client.get("/search").text
    tag = html[html.index('id="bookInput"') - 200 : html.index('id="bookInput"') + 400]
    assert " required" not in tag


def test_genre_only_browsing_page(client: TestClient) -> None:
    r = client.get("/search", params={"q": "", "genre": "Mystery"})
    assert r.status_code == 200
    assert "Most popular in <em>Mystery</em>" in r.text
    assert r.text.count('class="result-card"') >= 1


def test_empty_query_all_genres_shows_empty_state(client: TestClient) -> None:
    r = client.get("/search", params={"q": "", "genre": "All"})
    assert r.status_code == 200 and "Three Ways to Search" in r.text


class TestThemes:
    def test_theme_toggle_and_early_script_present(self, client: TestClient) -> None:
        html = client.get("/").text
        assert 'id="themeToggle"' in html
        head = html[: html.index("</head>")]
        # theme.js must load in <head>, without defer, before the stylesheet paints
        assert "js/theme.js" in head and 'theme.js" defer' not in head
        assert '<meta name="color-scheme" content="light dark"' in head

    def test_theme_assets_served(self, client: TestClient) -> None:
        js = client.get("/static/js/theme.js")
        assert js.status_code == 200 and "prefers-color-scheme" in js.text
        css = client.get("/static/css/style.css").text
        assert ':root[data-theme="dark"]' in css

    def test_toggle_present_on_every_page(self, client: TestClient) -> None:
        for path in ("/", "/search", "/recommend", "/about", "/does-not-exist"):
            assert 'id="themeToggle"' in client.get(path, headers={"accept": "text/html"}).text

    def test_csp_still_blocks_inline_scripts(self, client: TestClient) -> None:
        html = client.get("/").text
        assert "<script>" not in html  # all JS is external, so script-src 'self' holds
