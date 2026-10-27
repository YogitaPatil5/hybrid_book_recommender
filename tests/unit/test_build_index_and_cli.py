from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bookmind.core.exceptions import DataValidationError, EmbeddingError
from bookmind.ml.embedder import HashingEncoder
from bookmind.pipeline import cli
from bookmind.pipeline.build_index import build_index, build_text_for_embedding, prepare_metadata
from bookmind.services.artifacts import load_artifacts
from tests.conftest import RAW_SAMPLE, make_settings


def _genres_csv(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    p = tmp_path / "genres.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_embedding_text_repeats_title() -> None:
    text = build_text_for_embedding(
        pd.Series({"title": "Dune", "genre": "Sci", "description": "Spice"})
    )
    assert text == "Dune. Dune. Genre: Sci. Spice"


def test_embedding_text_handles_nan_and_caps_length() -> None:
    row = pd.Series({"title": "T", "genre": float("nan"), "description": "x" * 5000})
    text = build_text_for_embedding(row)
    assert "nan" not in text and len(text) == 2000


def test_prepare_metadata_fills_and_coerces() -> None:
    meta = prepare_metadata(
        pd.DataFrame(
            {"title": ["a"], "description": ["d"], "genre": ["Bogus"], "average_rating": ["x"]}
        )
    )
    assert meta.loc[0, "genre"] == "Other"
    assert pd.isna(meta.loc[0, "average_rating"])
    assert meta.loc[0, "authors"] == ""


def test_build_index_roundtrip(tmp_path: Path) -> None:
    src = _genres_csv(
        tmp_path,
        [
            {"title": "A", "description": "dragons", "genre": "Fantasy"},
            {"title": "", "description": "no title", "genre": "Fantasy"},
            {"title": "B", "description": "murder", "genre": "Mystery"},
        ],
    )
    manifest = build_index(src, tmp_path / "art", HashingEncoder(32))
    assert manifest.num_books == 2 and manifest.embedding_backend == "hashing"
    index, meta, _ = load_artifacts(tmp_path / "art")
    assert index.ntotal == 2 and meta["title"].tolist() == ["A", "B"]


def test_build_index_no_rows(tmp_path: Path) -> None:
    src = _genres_csv(tmp_path, [{"title": " ", "description": "x", "genre": "Other"}])
    with pytest.raises(DataValidationError, match="No books"):
        build_index(src, tmp_path / "art", HashingEncoder(32))


class _BadShape(HashingEncoder):
    def encode(self, texts, batch_size=64):  # type: ignore[no-untyped-def]
        return np.zeros((1, self.dim), dtype=np.float32)


class _NaN(HashingEncoder):
    def encode(self, texts, batch_size=64):  # type: ignore[no-untyped-def]
        return np.full((len(texts), self.dim), np.nan, dtype=np.float32)


@pytest.mark.parametrize("encoder,match", [(_BadShape(32), "shape"), (_NaN(32), "NaN")])
def test_build_index_rejects_bad_embeddings(
    tmp_path: Path, encoder: HashingEncoder, match: str
) -> None:
    src = _genres_csv(
        tmp_path,
        [
            {"title": "A", "description": "d", "genre": "Other"},
            {"title": "B", "description": "d", "genre": "Other"},
        ],
    )
    with pytest.raises(EmbeddingError, match=match):
        build_index(src, tmp_path / "art", encoder)


class TestCli:
    @pytest.fixture
    def data_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "data"
        d.mkdir()
        shutil.copy(RAW_SAMPLE, d / "books_raw.csv")
        return d

    def _run(self, tmp_path: Path, *args: str) -> int:
        settings = make_settings(tmp_path)
        return cli.main(
            ["--data-dir", str(tmp_path / "data"), "--artifacts-dir", str(tmp_path / "art"), *args],
            settings=settings,
        )

    def test_all_steps(self, tmp_path: Path, data_dir: Path) -> None:
        code = self._run(tmp_path, "all", "--classifier", "keyword", "--encoder", "hashing")
        assert code == cli.EXIT_OK
        assert (tmp_path / "art" / "manifest.json").exists()

    def test_individual_steps(self, tmp_path: Path, data_dir: Path) -> None:
        assert self._run(tmp_path, "prepare") == 0
        assert self._run(tmp_path, "classify", "--backend", "keyword") == 0
        assert self._run(tmp_path, "-v", "build-index", "--backend", "hashing") == 0

    def test_prepare_custom_input(self, tmp_path: Path, data_dir: Path) -> None:
        assert self._run(tmp_path, "prepare", "--input", str(RAW_SAMPLE)) == 0

    def test_missing_input_returns_error_code(self, tmp_path: Path) -> None:
        assert self._run(tmp_path, "prepare") == cli.EXIT_ERROR

    def test_ollama_unreachable_returns_error(
        self, tmp_path: Path, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert self._run(tmp_path, "prepare") == 0
        from bookmind.core.exceptions import ConfigurationError

        def down(self: object) -> None:
            raise ConfigurationError("Cannot reach the Ollama server")

        monkeypatch.setattr(cli.OllamaClassifier, "check_available", down)
        assert self._run(tmp_path, "classify") == cli.EXIT_ERROR

    def test_keyboard_interrupt(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def interrupt(*a: object, **k: object) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(cli, "run_prepare", interrupt)
        assert self._run(tmp_path, "prepare") == cli.EXIT_INTERRUPTED

    def test_bad_usage_exits_2(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as exc:
            cli.main(["unknown-cmd"], settings=make_settings(tmp_path))
        assert exc.value.code == 2

    def test_requires_subcommand(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            cli.main([], settings=make_settings(tmp_path))
