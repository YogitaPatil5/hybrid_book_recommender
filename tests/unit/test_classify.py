from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from bookmind.core.config import GENRE_TAXONOMY
from bookmind.core.exceptions import ConfigurationError, DataValidationError
from bookmind.pipeline.classify import (
    KeywordClassifier,
    OllamaClassifier,
    parse_genre,
    run_classify,
)
from bookmind.pipeline.prepare import run_prepare
from tests.conftest import RAW_SAMPLE


class TestParseGenre:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Fantasy", "Fantasy"),
            ("fantasy", "Fantasy"),
            ("  Science Fiction.\n", "Science Fiction"),
            ('"Mystery"', "Mystery"),
            ("Genre: Horror", "Horror"),
            ("**Romance**", "Romance"),
            ("Historical Fiction novel", "Historical Fiction"),
            ("I think this is Literary Fiction because...", "Literary Fiction"),
            ("Children's", "Children's"),
            ("cooking", "Other"),
            ("", "Other"),
            (None, "Other"),
        ],
    )
    def test_parse(self, raw: str | None, expected: str) -> None:
        assert parse_genre(raw) == expected

    def test_longest_match_wins(self) -> None:
        # "History" is a substring of nothing here, but "Historical Fiction" must beat "History"
        assert parse_genre("This is historical fiction about history") == "Historical Fiction"

    def test_word_boundaries(self) -> None:
        assert parse_genre("prehistory") == "Other"


class TestKeywordClassifier:
    clf = KeywordClassifier()

    @pytest.mark.parametrize(
        "categories,expected",
        [
            ("Juvenile Fiction", "Children's"),
            ("Young Adult Fiction", "Young Adult"),
            ("Biography & Autobiography", "Biography & Memoir"),
            ("Business & Economics", "Business & Economics"),
            ("Religion", "Philosophy & Religion"),
            ("True Crime", "True Crime"),
            ("History", "History"),
        ],
    )
    def test_category_rules(self, categories: str, expected: str) -> None:
        assert self.clf.classify("t", "d", categories) == expected

    def test_keywords_from_description(self) -> None:
        assert (
            self.clf.classify("X", "A starship crew meets an alien on a far planet")
            == "Science Fiction"
        )

    def test_title_weighted_more(self) -> None:
        assert self.clf.classify("Ghost House", "a love story with a heart") == "Horror"

    def test_word_boundary_regression(self) -> None:
        # "teen" must not match "Nineteen"
        assert self.clf.classify("Nineteen Eighty-Four", "a dystopian regime") == "Science Fiction"

    def test_generic_fiction(self) -> None:
        assert self.clf.classify("X", "nothing specific here", "Fiction") == "Literary Fiction"

    def test_fallback_other(self) -> None:
        assert self.clf.classify("X", "zzz") == "Other"

    def test_always_in_taxonomy(self) -> None:
        assert self.clf.classify("", "", "") in GENRE_TAXONOMY


class FakeOllama:
    def __init__(self, replies: list[Any], models: list[str] | None = None):
        self.replies = list(replies)
        self.calls = 0
        self.models = models if models is not None else ["llama3.2:latest"]

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        assert kwargs["options"] == {"temperature": 0}
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return {"message": {"content": reply}}

    def list(self) -> dict[str, Any]:
        return {"models": [{"model": m} for m in self.models]}


class TestOllamaClassifier:
    def _make(self, client: Any, **kw: Any) -> OllamaClassifier:
        return OllamaClassifier(client=client, sleep=lambda s: None, **kw)

    def test_success(self) -> None:
        clf = self._make(FakeOllama(["Mystery"]))
        assert clf.classify("t", "d") == "Mystery"
        assert clf.name == "ollama:llama3.2"

    def test_retries_then_succeeds(self) -> None:
        fake = FakeOllama([TimeoutError("slow"), ConnectionError("x"), "Fantasy"])
        delays: list[float] = []
        clf = OllamaClassifier(client=fake, sleep=delays.append, backoff_seconds=0.5)
        assert clf.classify("t", "d") == "Fantasy"
        assert fake.calls == 3
        assert delays == [0.5, 1.0]  # exponential backoff

    def test_falls_back_after_max_retries(self) -> None:
        fake = FakeOllama([RuntimeError("down")] * 3)
        clf = self._make(fake, max_retries=3)
        assert clf.classify("Haunted", "ghost story", "Horror") == "Horror"  # keyword fallback
        assert clf.failures == 1

    def test_malformed_response_is_retried(self) -> None:
        class Weird(FakeOllama):
            def chat(self, **kwargs: Any) -> dict[str, Any]:
                self.calls += 1
                return (
                    {"unexpected": True} if self.calls == 1 else {"message": {"content": "Poetry"}}
                )

        assert self._make(Weird([])).classify("t", "d") == "Poetry"

    def test_invalid_retries_config(self) -> None:
        with pytest.raises(ConfigurationError):
            OllamaClassifier(client=FakeOllama([]), max_retries=0)

    def test_check_available_ok(self) -> None:
        self._make(FakeOllama([])).check_available()

    def test_check_available_model_missing(self) -> None:
        with pytest.raises(ConfigurationError, match="ollama pull"):
            self._make(FakeOllama([], models=["mistral:latest"])).check_available()

    def test_check_available_server_down(self) -> None:
        class Down(FakeOllama):
            def list(self) -> dict[str, Any]:
                raise ConnectionError("refused")

        with pytest.raises(ConfigurationError, match="Cannot reach"):
            self._make(Down([])).check_available()

    def test_check_available_object_response(self) -> None:
        class Obj:
            def __init__(self, model: str):
                self.model = model

        class Listing:
            models = [Obj("llama3.2:3b")]

        class ObjClient(FakeOllama):
            def list(self) -> Any:  # type: ignore[override]
                return Listing()

        self._make(ObjClient([])).check_available()


class TestRunClassify:
    @pytest.fixture
    def prepared(self, tmp_path: Path) -> Path:
        out = tmp_path / "books.csv"
        run_prepare(RAW_SAMPLE, out)
        return out

    def test_classifies_everything(self, prepared: Path, tmp_path: Path) -> None:
        out = tmp_path / "genres.csv"
        report = run_classify(prepared, out, KeywordClassifier(), checkpoint_every=7)
        df = pd.read_csv(out)
        assert report.classified == report.total == len(df) == 30
        assert set(df["genre"]) <= set(GENRE_TAXONOMY)
        assert sum(report.genre_counts.values()) == 30

    def test_resume_skips_done_books(self, prepared: Path, tmp_path: Path) -> None:
        out = tmp_path / "genres.csv"
        run_classify(prepared, out, KeywordClassifier())

        class Counting(KeywordClassifier):
            calls = 0

            def classify(self, *a: Any, **k: Any) -> str:
                Counting.calls += 1
                return super().classify(*a, **k)

        report = run_classify(prepared, out, Counting())
        assert report.already_done == 30 and report.classified == 0 and Counting.calls == 0

    def test_invalid_checkpoint_rows_are_redone(self, prepared: Path, tmp_path: Path) -> None:
        out = tmp_path / "genres.csv"
        run_classify(prepared, out, KeywordClassifier())
        df = pd.read_csv(out)
        df.loc[0, "genre"] = "NotAGenre"
        df.to_csv(out, index=False)
        report = run_classify(prepared, out, KeywordClassifier())
        assert report.classified == 1

    def test_interrupt_saves_progress(self, prepared: Path, tmp_path: Path) -> None:
        out = tmp_path / "genres.csv"

        class Interrupting(KeywordClassifier):
            n = 0

            def classify(self, *a: Any, **k: Any) -> str:
                Interrupting.n += 1
                if Interrupting.n == 6:
                    raise KeyboardInterrupt
                return super().classify(*a, **k)

        with pytest.raises(KeyboardInterrupt):
            run_classify(prepared, out, Interrupting(), checkpoint_every=100)
        assert len(pd.read_csv(out)) == 5
        report = run_classify(prepared, out, KeywordClassifier())
        assert report.already_done == 5 and report.classified == 25

    def test_unknown_labels_never_persisted(self, prepared: Path, tmp_path: Path) -> None:
        class Rogue:
            name = "rogue"

            def classify(self, *a: Any, **k: Any) -> str:
                return "Cookbooks"

        out = tmp_path / "genres.csv"
        run_classify(prepared, out, Rogue())
        assert set(pd.read_csv(out)["genre"]) == {"Other"}

    def test_duplicate_ids_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "dup.csv"
        pd.DataFrame(
            {"book_id": ["1", "1"], "title": ["a", "b"], "description": ["x", "y"]}
        ).to_csv(p, index=False)
        with pytest.raises(DataValidationError, match="duplicate book_id"):
            run_classify(p, tmp_path / "o.csv", KeywordClassifier())

    def test_bad_checkpoint_every(self, prepared: Path, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError):
            run_classify(prepared, tmp_path / "o.csv", KeywordClassifier(), checkpoint_every=0)

    def test_progress_callback(self, prepared: Path, tmp_path: Path) -> None:
        seen: list[int] = []

        def progress(items: Any, total: int) -> Any:
            seen.append(total)
            return items

        run_classify(prepared, tmp_path / "o.csv", KeywordClassifier(), progress=progress)
        assert seen == [30]
