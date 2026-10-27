from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bookmind.core.exceptions import DataValidationError
from bookmind.pipeline.io import read_csv, write_csv_atomic
from bookmind.pipeline.prepare import (
    OUTPUT_COLUMNS,
    clean_text,
    normalise_authors,
    normalise_isbn13,
    prepare_dataframe,
    run_prepare,
)
from tests.conftest import RAW_SAMPLE


def _raw(**overrides: list[object]) -> pd.DataFrame:
    base: dict[str, list[object]] = {
        "title": ["Dune", "Emma"],
        "authors": ["Frank Herbert", "Jane Austen"],
        "description": [
            "A desert planet epic about spice and prophecy.",
            "A matchmaking comedy of manners in Regency England.",
        ],
    }
    base.update(overrides)
    return pd.DataFrame(base)


class TestCleaners:
    def test_clean_text_html_entities_and_whitespace(self) -> None:
        assert clean_text("<p>Hi&nbsp;&amp;\n bye</p>") == "Hi & bye"

    @pytest.mark.parametrize("value", [None, float("nan"), pd.NA])
    def test_clean_text_missing(self, value: object) -> None:
        assert clean_text(value) == ""

    def test_authors_semicolon_split(self) -> None:
        assert normalise_authors("A;B; ;C") == "A, B, C"
        assert normalise_authors(None) == "Unknown"
        assert normalise_authors(";;") == "Unknown"

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("9780261102217", "9780261102217"),
            ("9780261102217.0", "9780261102217"),
            ("978-0-26-110221-7", "9780261102217"),
            ("123", ""),
            ("bad", ""),
            (None, ""),
        ],
    )
    def test_isbn13(self, value: object, expected: str) -> None:
        assert normalise_isbn13(value) == expected


class TestPrepareDataframe:
    def test_minimal_columns(self) -> None:
        out, report = prepare_dataframe(_raw())
        assert list(out.columns) == list(OUTPUT_COLUMNS)
        assert report.output_rows == 2
        assert out["book_id"].str.startswith("h").all()  # no ISBN → content hash

    def test_missing_required_columns(self) -> None:
        with pytest.raises(DataValidationError, match="missing required"):
            prepare_dataframe(pd.DataFrame({"title": ["x"]}))

    def test_negative_min_description(self) -> None:
        with pytest.raises(DataValidationError):
            prepare_dataframe(_raw(), min_description_chars=-1)

    def test_all_rows_dropped(self) -> None:
        with pytest.raises(DataValidationError, match="No usable rows"):
            prepare_dataframe(_raw(description=["short", None]))

    def test_dedup_by_title_author(self) -> None:
        df = _raw(title=["Dune", "dune"], authors=["Frank Herbert", "frank herbert"])
        out, report = prepare_dataframe(df)
        assert len(out) == 1 and report.dropped_duplicates == 1

    def test_book_ids_are_stable(self) -> None:
        a, _ = prepare_dataframe(_raw())
        b, _ = prepare_dataframe(_raw())
        assert a["book_id"].tolist() == b["book_id"].tolist()

    def test_full_sample_cleaning(self) -> None:
        out, report = prepare_dataframe(read_csv(RAW_SAMPLE))
        assert report.input_rows == 35
        assert report.dropped_missing == 2  # no description, no title
        assert report.dropped_short == 1  # "Too short."
        # duplicate ISBN + a second edition of "The Hobbit" (same title & author)
        assert report.dropped_duplicates == 2
        assert report.output_rows == 30
        wh = out[out["title"] == "Wuthering Heights"].iloc[0]
        assert wh["authors"] == "Emily Bronte, Anne Bronte"
        assert "<" not in wh["description"] and "&amp;" not in wh["description"]
        assert wh["thumbnail"] == ""  # javascript: URL rejected
        assert pd.isna(wh["average_rating"])  # 7.5 out of range
        assert pd.isna(wh["published_year"])  # year 3000 rejected
        assert pd.isna(wh["ratings_count"])  # negative rejected
        assert out["book_id"].is_unique
        assert (out["published_year"].dropna() >= 1000).all()


def test_run_prepare_writes_file(tmp_path: Path) -> None:
    out_csv = tmp_path / "sub" / "books.csv"
    report = run_prepare(RAW_SAMPLE, out_csv)
    assert out_csv.exists()
    assert len(pd.read_csv(out_csv)) == report.output_rows


class TestIO:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="not found"):
            read_csv(tmp_path / "nope.csv")

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "e.csv"
        p.write_text("")
        with pytest.raises(DataValidationError, match="empty"):
            read_csv(p)

    def test_header_only(self, tmp_path: Path) -> None:
        p = tmp_path / "h.csv"
        p.write_text("title,authors\n")
        with pytest.raises(DataValidationError, match="no data rows"):
            read_csv(p)

    def test_required_columns_case_insensitive(self, tmp_path: Path) -> None:
        p = tmp_path / "c.csv"
        p.write_text(" Title ,AUTHORS\nx,y\n")
        assert list(read_csv(p, required=("title", "authors")).columns) == ["title", "authors"]

    def test_missing_required(self, tmp_path: Path) -> None:
        p = tmp_path / "c.csv"
        p.write_text("a\n1\n")
        with pytest.raises(DataValidationError) as exc:
            read_csv(p, required=("title",))
        assert exc.value.details["missing"] == ["title"]

    def test_latin1_fallback(self, tmp_path: Path) -> None:
        p = tmp_path / "l.csv"
        p.write_bytes("title\nCaf\xe9\n".encode("latin-1"))
        assert read_csv(p)["title"].iloc[0] == "Café"

    def test_parser_error(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.csv"
        p.write_text('a,b\n"unterminated,1\n')
        with pytest.raises(DataValidationError, match="Could not parse"):
            read_csv(p)

    def test_atomic_write_leaves_no_temp_files(self, tmp_path: Path) -> None:
        write_csv_atomic(pd.DataFrame({"a": [1]}), tmp_path / "o.csv")
        assert [p.name for p in tmp_path.iterdir()] == ["o.csv"]

    def test_atomic_write_cleans_up_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail(*a: object, **k: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(pd.DataFrame, "to_csv", fail)
        with pytest.raises(OSError):
            write_csv_atomic(pd.DataFrame({"a": [1]}), tmp_path / "o.csv")
        assert list(tmp_path.iterdir()) == []


def test_http_thumbnails_upgraded_to_https() -> None:
    df = _raw(thumbnail=["http://books.google.com/x", "ftp://bad"])
    out, _ = prepare_dataframe(df)
    assert out["thumbnail"].tolist() == ["https://books.google.com/x", ""]
