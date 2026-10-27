"""Contract tests for the /api/v1 JSON API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


def _error(resp) -> dict:  # type: ignore[no-untyped-def,type-arg]
    body = resp.json()
    assert set(body) == {"error"}
    assert {"code", "message", "request_id"} <= set(body["error"])
    return body["error"]  # type: ignore[no-any-return]


class TestSearch:
    def test_success_shape(self, client: TestClient) -> None:
        r = client.get("/api/v1/search", params={"q": "wizard dragon", "top_k": 3})
        assert r.status_code == 200
        body = r.json()
        assert body["query"] == "wizard dragon" and body["genre"] == "All"
        assert body["count"] == len(body["results"]) == 3
        assert body["took_ms"] >= 0
        book = body["results"][0]
        assert {
            "title",
            "author",
            "genre",
            "description",
            "image_url",
            "avg_rating",
            "published_year",
            "score",
            "buy_links",
        } <= set(book)
        assert set(book["buy_links"]) == {"amazon", "flipkart", "google_books"}

    def test_default_top_k(self, client: TestClient) -> None:
        assert client.get("/api/v1/search?q=book").json()["count"] == 12

    def test_genre_filter(self, client: TestClient) -> None:
        r = client.get("/api/v1/search", params={"q": "story", "genre": "Fantasy"})
        assert {b["genre"] for b in r.json()["results"]} == {"Fantasy"}

    def test_query_whitespace_normalised(self, client: TestClient) -> None:
        assert (
            client.get("/api/v1/search", params={"q": "  dune   desert "}).json()["query"]
            == "dune desert"
        )

    @pytest.mark.parametrize(
        "params,field",
        [
            ({"q": "x", "top_k": 0}, "top_k"),
            ({"q": "x", "top_k": "abc"}, "top_k"),
            ({"q": "x", "top_k": 501}, "top_k"),
        ],
    )
    def test_request_validation(self, client: TestClient, params: dict, field: str) -> None:  # type: ignore[type-arg]
        r = client.get("/api/v1/search", params=params)
        assert r.status_code == 422
        err = _error(r)
        assert err["code"] == "validation_error"
        assert any(e["field"] == field for e in err["details"]["errors"])

    def test_engine_level_limits(self, client: TestClient) -> None:
        r = client.get("/api/v1/search", params={"q": "x", "top_k": 100})  # > max_top_k=50
        assert r.status_code == 422 and "between 1 and 50" in _error(r)["message"]
        r = client.get("/api/v1/search", params={"q": "x" * 301})
        assert r.status_code == 422 and _error(r)["details"]["max_length"] == 300

    @pytest.mark.parametrize("params", [{}, {"q": ""}, {"q": "   "}, {"q": "", "genre": "All"}])
    def test_query_or_genre_required(self, client: TestClient, params: dict) -> None:  # type: ignore[type-arg]
        r = client.get("/api/v1/search", params=params)
        assert r.status_code == 422
        err = _error(r)
        assert err["code"] == "validation_error" and err["details"]["field"] == "q"
        assert "query (q), a genre" in err["message"]

    def test_genre_only_browsing(self, client: TestClient) -> None:
        r = client.get("/api/v1/search", params={"genre": "Fantasy", "top_k": 5})
        assert r.status_code == 200
        body = r.json()
        assert body["query"] == "" and body["genre"] == "Fantasy" and body["count"] >= 1
        assert {b["genre"] for b in body["results"]} == {"Fantasy"}
        assert all(b["score"] is None for b in body["results"])

    def test_genre_only_unknown_genre(self, client: TestClient) -> None:
        r = client.get("/api/v1/search", params={"genre": "Cooking"})
        assert r.status_code == 422 and "Unknown genre" in _error(r)["message"]

    def test_unknown_genre(self, client: TestClient) -> None:
        r = client.get("/api/v1/search", params={"q": "x", "genre": "Cooking"})
        assert r.status_code == 422
        assert "Fantasy" in _error(r)["details"]["allowed"]

    def test_injection_strings_are_harmless(self, client: TestClient) -> None:
        for q in [
            "<script>alert(1)</script>",
            "'; DROP TABLE books; --",
            "../../etc/passwd",
            "\u0000‮",
        ]:
            r = client.get("/api/v1/search", params={"q": q})
            assert r.status_code in (200, 422)


class TestSimilar:
    def test_semantic(self, client: TestClient) -> None:
        r = client.get("/api/v1/books/similar", params={"title": "the hobbit", "top_k": 4})
        body = r.json()
        assert r.status_code == 200
        assert body["source_title"] == "The Hobbit" and body["engine"] == "semantic"
        assert body["count"] == 4
        assert "The Hobbit" not in {b["title"] for b in body["results"]}

    def test_not_found(self, client: TestClient) -> None:
        r = client.get("/api/v1/books/similar", params={"title": "qwertyuiop"})
        assert r.status_code == 404 and _error(r)["code"] == "book_not_found"

    def test_knn_unavailable(self, client: TestClient) -> None:
        r = client.get("/api/v1/books/similar", params={"title": "Dune", "engine": "knn"})
        assert r.status_code == 503 and _error(r)["details"]["component"] == "knn"

    def test_invalid_engine(self, client: TestClient) -> None:
        assert (
            client.get("/api/v1/books/similar", params={"title": "Dune", "engine": "x"}).status_code
            == 422
        )

    def test_knn_engine(self, knn_client: TestClient) -> None:
        r = knn_client.get("/api/v1/books/similar", params={"title": "Dune", "engine": "auto"})
        assert r.json()["engine"] == "knn"


class TestMisc:
    def test_autocomplete(self, client: TestClient) -> None:
        r = client.get("/api/v1/autocomplete", params={"q": "harry"})
        assert r.json() == {
            "query": "harry",
            "suggestions": ["Harry Potter and the Sorcerer's Stone"],
            "mode": "match",
            "label": None,
        }

    def test_empty_query_returns_popular_titles(self, client: TestClient) -> None:
        body = client.get("/api/v1/autocomplete").json()
        assert body["mode"] == "popular" and body["label"] == "Popular books"
        assert body["suggestions"][0] == "Harry Potter and the Sorcerer's Stone"  # most-read
        assert len(body["suggestions"]) == 10

    def test_empty_query_popular_within_genre(self, client: TestClient) -> None:
        body = client.get("/api/v1/autocomplete", params={"genre": "fantasy", "limit": 3}).json()
        assert body["label"] == "Popular in Fantasy"
        assert body["suggestions"][0] == "The Hobbit"
        assert len(body["suggestions"]) <= 3

    def test_empty_query_genre_all_means_overall(self, client: TestClient) -> None:
        body = client.get("/api/v1/autocomplete", params={"genre": "All"}).json()
        assert body["label"] == "Popular books"

    def test_empty_query_unknown_genre(self, client: TestClient) -> None:
        r = client.get("/api/v1/autocomplete", params={"genre": "Cooking"})
        assert r.status_code == 422

    def test_genre_ignored_when_typing(self, client: TestClient) -> None:
        body = client.get("/api/v1/autocomplete", params={"q": "dune", "genre": "Poetry"}).json()
        assert body["mode"] == "match" and body["suggestions"] == ["Dune"]

    def test_popular_merges_knn_titles_first(self, knn_client: TestClient) -> None:
        body = knn_client.get("/api/v1/autocomplete", params={"limit": 6}).json()
        titles = body["suggestions"]
        assert len(titles) == 6 and len({t.lower() for t in titles}) == 6
        knn_titles = {"The Hobbit", "Dune", "Foundation", "Emma", "Dracula", "Ulysses"}
        assert titles[0] in knn_titles  # interleaved, k-NN catalogue first
        assert any(t not in knn_titles for t in titles)  # semantic titles too

    def test_autocomplete_limit(self, client: TestClient) -> None:
        assert len(client.get("/api/v1/autocomplete?q=the&limit=2").json()["suggestions"]) == 2
        assert client.get("/api/v1/autocomplete?q=the&limit=999").status_code == 422

    def test_genres(self, client: TestClient) -> None:
        genres = client.get("/api/v1/genres").json()["genres"]
        assert genres == sorted(genres) and "Fantasy" in genres

    def test_stats(self, knn_client: TestClient) -> None:
        body = knn_client.get("/api/v1/stats").json()
        assert body["total_books"] == 30 and body["knn_available"] is True

    def test_legacy_autocomplete(self, client: TestClient) -> None:
        assert client.get("/api/search?q=dune").json() == ["Dune"]

    def test_openapi_documents_endpoints(self, client: TestClient) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        assert {
            "/api/v1/search",
            "/api/v1/books/similar",
            "/api/v1/autocomplete",
            "/api/v1/genres",
            "/api/v1/stats",
            "/health/live",
            "/health/ready",
        } <= set(paths)


class TestDegradedMode:
    @pytest.mark.parametrize(
        "url",
        [
            "/api/v1/search?q=x",
            "/api/v1/genres",
            "/api/v1/autocomplete?q=x",
            "/api/v1/books/similar?title=x",
        ],
    )
    def test_503_when_index_missing(self, empty_client: TestClient, url: str) -> None:
        r = empty_client.get(url)
        assert r.status_code == 503
        assert _error(r)["code"] == "service_unavailable"

    def test_stats_and_legacy_still_work(self, empty_client: TestClient) -> None:
        assert empty_client.get("/api/v1/stats").json()["total_books"] == 0
        assert empty_client.get("/api/search?q=x").json() == []
