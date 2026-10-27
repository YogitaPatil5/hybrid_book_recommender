from __future__ import annotations

from pathlib import Path

import pytest

from bookmind.core.exceptions import DataValidationError
from bookmind.pipeline.download import BX_DATASET, SEMANTIC_DATASET, download_datasets


class FakeKaggle:
    """Stands in for kagglehub.dataset_download: creates a cache folder per dataset."""

    def __init__(self, root: Path, *, bx_files: tuple[str, ...] | None = None):
        self.root = root
        self.calls: list[str] = []
        self.bx_files = bx_files or ("BX-Books.csv", "BX-Book-Ratings.csv", "BX-Users.csv")

    def __call__(self, dataset: str) -> str:
        self.calls.append(dataset)
        folder = self.root / dataset.replace("/", "_") / "versions" / "1"
        folder.mkdir(parents=True, exist_ok=True)
        if dataset == SEMANTIC_DATASET:
            (folder / "books.csv").write_text("title,authors,description\nA,B,C\n")
        else:
            nested = folder / "Books Data"
            nested.mkdir(exist_ok=True)
            for name in self.bx_files:
                (nested / name).write_text('"ISBN"\n"1"\n')
        return str(folder)


def test_downloads_and_places_both(tmp_path: Path) -> None:
    fake = FakeKaggle(tmp_path / "cache")
    placed = download_datasets(tmp_path / "data", downloader=fake)
    assert (tmp_path / "data" / "books_raw.csv").exists()
    for name in ("BX-Books.csv", "BX-Book-Ratings.csv", "BX-Users.csv"):
        assert (tmp_path / "data" / "bx" / name).exists()  # found in nested folder
    assert placed["bx_dir"] == tmp_path / "data" / "bx"
    assert fake.calls == [SEMANTIC_DATASET, BX_DATASET]


def test_skips_existing_unless_forced(tmp_path: Path) -> None:
    fake = FakeKaggle(tmp_path / "cache")
    download_datasets(tmp_path / "data", downloader=fake)
    download_datasets(tmp_path / "data", downloader=fake)
    assert len(fake.calls) == 2
    download_datasets(tmp_path / "data", downloader=fake, force=True)
    assert len(fake.calls) == 4


def test_only_one_dataset(tmp_path: Path) -> None:
    fake = FakeKaggle(tmp_path / "cache")
    placed = download_datasets(tmp_path / "data", which=["knn"], downloader=fake)
    assert set(placed) == {"bx_dir"} and fake.calls == [BX_DATASET]


def test_optional_users_file_may_be_missing(tmp_path: Path) -> None:
    fake = FakeKaggle(tmp_path / "cache", bx_files=("BX-Books.csv", "BX-Book-Ratings.csv"))
    download_datasets(tmp_path / "data", which=["knn"], downloader=fake)
    assert not (tmp_path / "data" / "bx" / "BX-Users.csv").exists()


def test_required_file_missing(tmp_path: Path) -> None:
    fake = FakeKaggle(tmp_path / "cache", bx_files=("BX-Users.csv",))
    with pytest.raises(DataValidationError, match=r"BX-Books\.csv"):
        download_datasets(tmp_path / "data", which=["knn"], downloader=fake)


def test_semantic_file_missing(tmp_path: Path) -> None:
    def empty(dataset: str) -> str:
        d = tmp_path / "empty"
        d.mkdir(exist_ok=True)
        return str(d)

    with pytest.raises(DataValidationError, match=r"books\.csv"):
        download_datasets(tmp_path / "data", which=["semantic"], downloader=empty)


def test_download_failure_has_helpful_message(tmp_path: Path) -> None:
    def offline(dataset: str) -> str:
        raise ConnectionError("no internet")

    with pytest.raises(DataValidationError, match=r"kagglehub\.login"):
        download_datasets(tmp_path / "data", downloader=offline)


def test_missing_path_returned(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="missing path"):
        download_datasets(tmp_path / "data", downloader=lambda d: str(tmp_path / "nope"))


def test_unknown_selection(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="Unknown dataset"):
        download_datasets(tmp_path / "data", which=["movies"], downloader=lambda d: "")
