from __future__ import annotations

import pytest

from bookmind.core.exceptions import ValidationError
from bookmind.services.validation import clean_text, validate_genre, validate_top_k

GENRES = ["Fantasy", "Mystery", "Science Fiction"]


class TestCleanText:
    def test_strips_and_collapses_whitespace(self) -> None:
        assert clean_text("  a   slow\tburn\n mystery  ") == "a slow burn mystery"

    def test_removes_control_characters(self) -> None:
        assert clean_text("dra\x00gon\x07") == "dragon"

    def test_nfkc_normalisation(self) -> None:
        assert clean_text("\uff46\uff41\uff4e\uff54\uff41\uff53\uff59") == "fantasy"  # full-width

    @pytest.mark.parametrize("value", ["", "   ", "\n\t"])
    def test_rejects_empty(self, value: str) -> None:
        with pytest.raises(ValidationError, match="must not be empty"):
            clean_text(value)

    def test_rejects_none(self) -> None:
        with pytest.raises(ValidationError, match="required"):
            clean_text(None)

    def test_rejects_non_string(self) -> None:
        with pytest.raises(ValidationError, match="must be a string"):
            clean_text(123)

    def test_rejects_too_long(self) -> None:
        with pytest.raises(ValidationError) as exc:
            clean_text("x" * 11, max_length=10)
        assert exc.value.details["max_length"] == 10
        assert exc.value.status_code == 422

    def test_custom_field_name_in_message(self) -> None:
        with pytest.raises(ValidationError, match="title"):
            clean_text("", field="title")


class TestTopK:
    @pytest.mark.parametrize("value", [1, 25, 50])
    def test_accepts_valid(self, value: int) -> None:
        assert validate_top_k(value, max_top_k=50) == value

    @pytest.mark.parametrize("value", [0, -1, 51])
    def test_rejects_out_of_range(self, value: int) -> None:
        with pytest.raises(ValidationError, match="between 1 and 50"):
            validate_top_k(value, max_top_k=50)

    @pytest.mark.parametrize("value", ["5", 5.0, True, None])
    def test_rejects_non_int(self, value: object) -> None:
        with pytest.raises(ValidationError, match="integer"):
            validate_top_k(value, max_top_k=50)


class TestGenre:
    @pytest.mark.parametrize("value", [None, "", "All", "all", "  ALL "])
    def test_all_variants(self, value: str | None) -> None:
        assert validate_genre(value, GENRES) == "All"

    def test_case_insensitive_canonicalisation(self) -> None:
        assert validate_genre("science fiction", GENRES) == "Science Fiction"

    def test_unknown_genre(self) -> None:
        with pytest.raises(ValidationError) as exc:
            validate_genre("Cooking", GENRES)
        assert exc.value.details["allowed"] == sorted(GENRES)
