from __future__ import annotations

import faiss
import numpy as np
import pandas as pd
import pytest

from bookmind.core.exceptions import ArtifactIntegrityError, BookNotFoundError, ValidationError
from bookmind.ml.embedder import HashingEncoder
from bookmind.services.semantic_search import (
    SemanticSearchEngine,
    _clean_float,
    _clean_int,
    _clean_str,
    truncate,
)


class TestHelpers:
    @pytest.mark.parametrize(
        "value,expected", [(None, ""), (float("nan"), ""), ("nan", ""), ("  x ", "x"), (5, "5")]
    )
    def test_clean_str(self, value: object, expected: str) -> None:
        assert _clean_str(value) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [("4.5", 4.5), (None, None), ("abc", None), (float("nan"), None), (float("inf"), None)],
    )
    def test_clean_float(self, value: object, expected: float | None) -> None:
        assert _clean_float(value) == expected

    def test_clean_int(self) -> None:
        assert _clean_int("1999.0") == 1999
        assert _clean_int(None) is None

    def test_truncate_short_text_untouched(self) -> None:
        assert truncate("short", 10) == "short"

    def test_truncate_on_word_boundary(self) -> None:
        out = truncate("one two three four five", 12)
        assert out == "one two…"
        assert len(out) <= 13


class TestSearch:
    def test_returns_relevant_book_first(self, engine: SemanticSearchEngine) -> None:
        results = engine.search("wizard dragon quest kingdom", top_k=3)
        assert results[0].title in {"The Hobbit", "The Lord of the Rings"}
        assert all(r.score is not None for r in results)

    def test_scores_are_sorted_desc(self, engine: SemanticSearchEngine) -> None:
        scores = [r.score for r in engine.search("murder mystery island", top_k=8)]
        assert scores == sorted(scores, reverse=True)  # type: ignore[type-var]

    def test_respects_top_k(self, engine: SemanticSearchEngine) -> None:
        assert len(engine.search("book", top_k=4)) == 4

    def test_top_k_larger_than_catalogue(self, settings) -> None:  # type: ignore[no-untyped-def]
        eng = SemanticSearchEngine.from_artifacts(settings)
        eng.max_top_k = 500
        assert len(eng.search("book", top_k=500)) <= eng.size

    def test_genre_filter_is_exact(self, engine: SemanticSearchEngine) -> None:
        results = engine.search("anything at all", genre="Mystery", top_k=10)
        assert results
        assert {r.genre for r in results} == {"Mystery"}

    def test_genre_is_case_insensitive(self, engine: SemanticSearchEngine) -> None:
        assert engine.search("x", genre="mystery", top_k=1)[0].genre == "Mystery"

    def test_unknown_genre_rejected(self, engine: SemanticSearchEngine) -> None:
        with pytest.raises(ValidationError, match="Unknown genre"):
            engine.search("x", genre="Cooking")

    def test_exact_title_is_pinned_first(self, engine: SemanticSearchEngine) -> None:
        assert engine.search("dune", top_k=5)[0].title == "Dune"

    def test_results_are_deduplicated_across_editions(self, engine: SemanticSearchEngine) -> None:
        results = engine.search("the hobbit dwarves dragon", top_k=10)
        keys = [(r.title.lower(), r.author.lower()) for r in results]
        assert len(keys) == len(set(keys))

    @pytest.mark.parametrize("query", ["", "   "])
    def test_empty_query_rejected(self, engine: SemanticSearchEngine, query: str) -> None:
        with pytest.raises(ValidationError):
            engine.search(query)

    def test_long_query_rejected(self, engine: SemanticSearchEngine) -> None:
        with pytest.raises(ValidationError, match="at most"):
            engine.search("x" * (engine.max_query_length + 1))

    @pytest.mark.parametrize("top_k", [0, -3, 10_000])
    def test_invalid_top_k(self, engine: SemanticSearchEngine, top_k: int) -> None:
        with pytest.raises(ValidationError):
            engine.search("dragon", top_k=top_k)

    def test_book_fields_are_clean(self, engine: SemanticSearchEngine) -> None:
        for book in engine.search("love passion moors heathcliff", top_k=30):
            assert book.author and book.author.lower() != "nan"
            assert book.image_url is None or book.image_url.startswith("https://")
            assert book.avg_rating is None or 0 <= book.avg_rating <= 5

    def test_uses_query_cache(self, engine: SemanticSearchEngine) -> None:
        engine.search("dragon")
        engine.search("dragon")
        assert engine.encoder.hits >= 1


class TestSimilar:
    def test_excludes_source_and_its_editions(self, engine: SemanticSearchEngine) -> None:
        source, results = engine.similar("The Hobbit", top_k=5)
        assert source == "The Hobbit"
        assert "the hobbit" not in {r.title.lower() for r in results}
        assert len(results) == 5

    def test_case_insensitive_and_prefix_match(self, engine: SemanticSearchEngine) -> None:
        assert engine.similar("harry potter", top_k=2)[0].startswith("Harry Potter")

    def test_substring_match(self, engine: SemanticSearchEngine) -> None:
        assert engine.similar("Gatsby", top_k=1)[0] == "The Great Gatsby"

    def test_not_found_includes_suggestions(self, engine: SemanticSearchEngine) -> None:
        with pytest.raises(BookNotFoundError) as exc:
            engine.similar("Zzzz unknown book")
        assert "suggestions" in exc.value.details
        assert exc.value.status_code == 404

    def test_prefers_most_rated_edition(self, engine: SemanticSearchEngine) -> None:
        pos = engine.resolve_title("the hobbit")
        assert engine._records[pos].ratings_count == 3_200_000


class TestAutocomplete:
    def test_prefix_before_substring(self, engine: SemanticSearchEngine) -> None:
        out = engine.autocomplete("ha")
        assert out[0].lower().startswith("ha")
        assert "The Handmaid's Tale" in out

    def test_limit(self, engine: SemanticSearchEngine) -> None:
        assert len(engine.autocomplete("the", limit=3)) == 3

    def test_limit_clamped(self, engine: SemanticSearchEngine) -> None:
        assert len(engine.autocomplete("e", limit=10_000)) <= 50

    @pytest.mark.parametrize("value", ["", "   ", None, 5])
    def test_bad_input_returns_empty(self, engine: SemanticSearchEngine, value: object) -> None:
        assert engine.autocomplete(value) == []  # type: ignore[arg-type]

    def test_unique_titles(self, engine: SemanticSearchEngine) -> None:
        out = engine.autocomplete("hobbit")
        assert out.count("The Hobbit") == 1


class TestPopularAndMeta:
    def test_popular_sorted_by_ratings_count(self, engine: SemanticSearchEngine) -> None:
        books = engine.popular(5)
        counts = [b.num_ratings or 0 for b in books]
        assert counts == sorted(counts, reverse=True)
        assert books[0].title == "Harry Potter and the Sorcerer's Stone"

    def test_popular_unique_titles(self, engine: SemanticSearchEngine) -> None:
        titles = [b.title for b in engine.popular(100)]
        assert len(titles) == len(set(titles))

    def test_list_genres_is_copy(self, engine: SemanticSearchEngine) -> None:
        genres = engine.list_genres()
        genres.append("X")
        assert "X" not in engine.list_genres()

    def test_size(self, engine: SemanticSearchEngine) -> None:
        assert engine.size == 30


class TestConstruction:
    def _meta(self, n: int) -> pd.DataFrame:
        return pd.DataFrame({"title": [f"t{i}" for i in range(n)], "genre": ["Other"] * n})

    def test_dim_mismatch_rejected(self) -> None:
        idx = faiss.IndexFlatIP(16)
        idx.add(np.ones((2, 16), dtype=np.float32))
        with pytest.raises(ArtifactIntegrityError, match="dim"):
            SemanticSearchEngine(idx, self._meta(2), HashingEncoder(32))

    def test_size_mismatch_rejected(self) -> None:
        idx = faiss.IndexFlatIP(32)
        idx.add(np.ones((3, 32), dtype=np.float32))
        with pytest.raises(ArtifactIntegrityError):
            SemanticSearchEngine(idx, self._meta(2), HashingEncoder(32))

    def test_bad_metadata_values_are_sanitised(self) -> None:
        idx = faiss.IndexFlatIP(32)
        idx.add(HashingEncoder(32).encode(["a", "b"]))
        meta = pd.DataFrame(
            {
                "title": ["Valid", None],
                "authors": [float("nan"), "nan"],
                "genre": [None, "Other"],
                "thumbnail": ["javascript:alert(1)", "https://ok/img.png"],
                "average_rating": [9.9, "bad"],
                "published_year": [5000, 1990],
                "ratings_count": [-5, 10],
            }
        )
        eng = SemanticSearchEngine(idx, meta, HashingEncoder(32))
        rec0, rec1 = eng._records
        assert rec0.author == "Unknown" and rec0.genre == "Other"
        assert rec0.image_url is None and rec0.avg_rating is None
        assert rec0.published_year is None and rec0.ratings_count is None
        assert rec1.title == "Untitled" and rec1.image_url == "https://ok/img.png"

    def test_empty_genre_subset_returns_nothing(self, engine: SemanticSearchEngine) -> None:
        vec = engine.encoder.encode_one("x")
        assert engine._knn(vec, 5, "Nonexistent") == []
        assert engine._knn(vec, 0, "All") == []


class TestBrowse:
    def test_browse_genre_sorted_by_popularity(self, engine: SemanticSearchEngine) -> None:
        books = engine.browse("fantasy", top_k=10)  # case-insensitive
        assert books and {b.genre for b in books} == {"Fantasy"}
        assert books[0].title == "The Hobbit"  # most ratings in the sample
        assert all(b.score is None for b in books)

    def test_browse_respects_top_k(self, engine: SemanticSearchEngine) -> None:
        assert len(engine.browse("Fantasy", top_k=1)) == 1

    @pytest.mark.parametrize("genre", ["All", "", None])
    def test_browse_requires_specific_genre(
        self, engine: SemanticSearchEngine, genre: str | None
    ) -> None:
        with pytest.raises(ValidationError, match="specific genre"):
            engine.browse(genre)  # type: ignore[arg-type]

    def test_browse_unknown_genre(self, engine: SemanticSearchEngine) -> None:
        with pytest.raises(ValidationError, match="Unknown genre"):
            engine.browse("Cooking")

    def test_browse_invalid_top_k(self, engine: SemanticSearchEngine) -> None:
        with pytest.raises(ValidationError):
            engine.browse("Fantasy", top_k=0)
