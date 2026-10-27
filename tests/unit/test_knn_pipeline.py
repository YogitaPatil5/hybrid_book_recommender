from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from bookmind.core.exceptions import ArtifactIntegrityError, DataValidationError
from bookmind.pipeline.knn import (
    KnnConfig,
    build_interactions,
    build_pivot,
    evaluate_knn,
    fit_model,
    interaction_value,
    load_books,
    load_ratings,
    run_evaluation,
    train_knn,
)
from bookmind.services.knn_recommender import KnnRecommender
from tests.bx_fixture import cluster_of, write_bx_fixture

CFG = KnnConfig(min_user_ratings=3, min_book_ratings=3)


@pytest.fixture
def bx(tmp_path: Path) -> Path:
    return write_bx_fixture(tmp_path / "bx")


class TestLoading:
    def test_books_cleaned(self, bx: Path) -> None:
        books = load_books(bx)
        assert "" not in set(books["title"])
        assert "Tom & Jerry" in set(books["title"])  # HTML entity unescaped
        assert set(books["publisher"]) >= {"Pub & Sons"}
        tj = books[books["title"] == "Tom & Jerry"].iloc[0]
        assert pd.isna(tj["year"]) and tj["image_url"] == ""  # year 3000, ftp:// rejected
        assert books["image_url"].str.startswith(("https://", "")).all()
        assert not books["image_url"].str.startswith("http://").any()
        assert books["isbn"].is_unique

    def test_ratings_invalid_rows_dropped(self, bx: Path, caplog: pytest.LogCaptureFixture) -> None:
        ratings = load_ratings(bx)
        assert ratings["rating"].between(0, 10).all()
        assert (ratings["isbn"] != "").all()
        assert any("invalid rating rows" in r.message for r in caplog.records)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="download"):
            load_books(tmp_path)

    def test_missing_columns(self, tmp_path: Path) -> None:
        (tmp_path / "BX-Book-Ratings.csv").write_text('"A";"B"\n"1";"2"\n', encoding="latin-1")
        with pytest.raises(DataValidationError, match="missing columns"):
            load_ratings(tmp_path)

    def test_header_only(self, tmp_path: Path) -> None:
        (tmp_path / "BX-Book-Ratings.csv").write_text('"User-ID";"ISBN";"Book-Rating"\n')
        with pytest.raises(DataValidationError, match="no rows"):
            load_ratings(tmp_path)

    def test_empty_file(self, tmp_path: Path) -> None:
        (tmp_path / "BX-Book-Ratings.csv").write_text("")
        with pytest.raises(DataValidationError, match="Could not parse"):
            load_ratings(tmp_path)


class TestTransform:
    def test_interaction_modes(self) -> None:
        r = pd.Series([0, 5, 10])
        assert interaction_value(r, "binary").tolist() == [1.0, 1.0, 1.0]
        assert interaction_value(r, "weighted").tolist() == [1.0, 5.0, 10.0]
        assert interaction_value(r, "explicit").isna().tolist() == [True, False, False]

    @pytest.mark.parametrize(
        "cfg",
        [
            KnnConfig(mode="nope"),
            KnnConfig(min_user_ratings=-1),  # type: ignore[arg-type]
            KnnConfig(min_book_ratings=0),
        ],
    )
    def test_config_validation(self, cfg: KnnConfig) -> None:
        with pytest.raises(DataValidationError):
            cfg.validate()

    def test_editions_merged_and_titles_disambiguated(self, bx: Path) -> None:
        inter = build_interactions(load_ratings(bx), load_books(bx), CFG)
        titles = set(inter["title"])
        assert "Dragon Gate" in titles  # 2 ISBNs merged into one book
        assert inter[inter["title"] == "Dragon Gate"].duplicated(["user_id"]).sum() == 0
        assert {"Cold Case (Crime Author)", "Cold Case (Other Writer)"} <= titles
        assert (inter["value"] == 1.0).all()  # binary default
        assert inter["image_url"].str.startswith(("https://", "")).all()

    def test_thresholds_filter(self, bx: Path) -> None:
        inter = build_interactions(
            load_ratings(bx), load_books(bx), KnnConfig(min_user_ratings=3, min_book_ratings=5)
        )
        assert inter.groupby("title")["user_id"].nunique().min() >= 5
        assert "Cold Case (Other Writer)" not in set(inter["title"])  # only 4 readers

    def test_explicit_mode_drops_implicit(self, bx: Path) -> None:
        inter = build_interactions(
            load_ratings(bx),
            load_books(bx),
            KnnConfig(mode="explicit", min_user_ratings=1, min_book_ratings=2),
        )
        assert (inter["rating"] > 0).all()

    def test_everything_filtered(self, bx: Path) -> None:
        with pytest.raises(DataValidationError, match="No interactions left"):
            build_interactions(load_ratings(bx), load_books(bx), KnnConfig(min_user_ratings=999))

    def test_pivot_and_model(self, bx: Path) -> None:
        inter = build_interactions(load_ratings(bx), load_books(bx), CFG)
        pivot = build_pivot(inter)
        assert pivot.shape == (inter["title"].nunique(), inter["user_id"].nunique())
        assert set(pivot.to_numpy().ravel()) <= {0.0, 1.0}
        assert fit_model(pivot).n_samples_fit_ == pivot.shape[0]

    def test_model_needs_two_books(self) -> None:
        with pytest.raises(DataValidationError):
            fit_model(pd.DataFrame([[1.0]], index=["only"]))


class TestTrainAndServe:
    def test_artifacts_load_and_recommend_same_cluster(self, bx: Path, tmp_path: Path) -> None:
        art = tmp_path / "art"
        report = train_knn(bx, art, CFG)
        assert report.books == 16  # 15 cluster books + 'Cold Case (Other Writer)'
        assert 60 <= report.active_users <= 75  # users with <=3 books filtered out
        manifest = json.loads((art / "knn_manifest.json").read_text())
        assert set(manifest["checksums"]) == {
            "model_knn.pkl",
            "book_names.pkl",
            "final_rating.pkl",
            "book_pivot.pkl",
        }
        import numpy as np
        import pandas as pd
        import sklearn

        assert manifest["library_versions"] == {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "scipy": __import__("scipy").__version__,
        }
        rec = KnnRecommender.from_artifacts(art)
        assert rec.manifest is not None and rec.manifest["mode"] == "binary"
        for seed in ("Wizard Hollow", "Night Detective", "Mars Colony"):
            source, books = rec.recommend(seed, top_k=3)
            assert all(cluster_of(b.title) == cluster_of(source) for b in books), seed

    def test_popular_avg_rating_on_five_point_scale(self, bx: Path, tmp_path: Path) -> None:
        train_knn(bx, tmp_path / "art", CFG)
        books = KnnRecommender.from_artifacts(tmp_path / "art").popular(5)
        assert all(b.avg_rating is None or 2.5 <= b.avg_rating <= 5 for b in books)

    def test_tampered_artifact_rejected(self, bx: Path, tmp_path: Path) -> None:
        art = tmp_path / "art"
        train_knn(bx, art, CFG)
        (art / "book_pivot.pkl").write_bytes((art / "book_pivot.pkl").read_bytes() + b"x")
        with pytest.raises(ArtifactIntegrityError, match="Checksum mismatch"):
            KnnRecommender.from_artifacts(art)

    def test_bad_manifest_json(self, bx: Path, tmp_path: Path) -> None:
        art = tmp_path / "art"
        train_knn(bx, art, CFG)
        (art / "knn_manifest.json").write_text("{broken")
        with pytest.raises(ArtifactIntegrityError, match="not valid JSON"):
            KnnRecommender.from_artifacts(art)

    def test_sklearn_version_mismatch_warns(
        self, bx: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        art = tmp_path / "art"
        train_knn(bx, art, CFG)
        path = art / "knn_manifest.json"
        data = json.loads(path.read_text())
        data["sklearn_version"] = "0.0.1"
        path.write_text(json.dumps(data))
        with caplog.at_level(logging.WARNING):
            KnnRecommender.from_artifacts(art)
        assert any("scikit-learn 0.0.1" in r.getMessage() for r in caplog.records)

    def test_missing_manifest_still_loads_with_warning(
        self, bx: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        art = tmp_path / "art"
        train_knn(bx, art, CFG)
        (art / "knn_manifest.json").unlink()
        with caplog.at_level(logging.WARNING):
            rec = KnnRecommender.from_artifacts(art)
        assert rec.manifest is None
        assert any("without checksum" in r.getMessage() for r in caplog.records)

    def test_autocomplete(self, bx: Path, tmp_path: Path) -> None:
        train_knn(bx, tmp_path / "art", CFG)
        rec = KnnRecommender.from_artifacts(tmp_path / "art")
        assert rec.autocomplete("star")[0] == "Star Drift"
        assert "Nova Station" in rec.autocomplete("station")
        assert rec.autocomplete("") == [] and rec.autocomplete(None) == []  # type: ignore[arg-type]


class TestEvaluation:
    def test_knn_beats_popularity_on_clustered_data(self, bx: Path) -> None:
        inter = build_interactions(load_ratings(bx), load_books(bx), CFG)
        rep = evaluate_knn(inter, k=3)
        assert rep.queries > 0
        assert rep.hit_rate > rep.popularity_hit_rate
        assert rep.lift_over_popularity > 1
        assert rep.catalogue_coverage > rep.popularity_coverage

    def test_deterministic(self, bx: Path) -> None:
        inter = build_interactions(load_ratings(bx), load_books(bx), CFG)
        assert evaluate_knn(inter) == evaluate_knn(inter)

    @pytest.mark.parametrize("kwargs", [{"k": 0}, {"test_fraction": 0}, {"test_fraction": 1}])
    def test_invalid_params(self, bx: Path, kwargs: dict) -> None:  # type: ignore[type-arg]
        inter = build_interactions(load_ratings(bx), load_books(bx), CFG)
        with pytest.raises(DataValidationError):
            evaluate_knn(inter, **kwargs)

    def test_no_queries(self) -> None:
        inter = pd.DataFrame(
            {"user_id": [1, 2, 3, 4, 5], "title": ["A", "B", "A", "B", "A"], "value": 1.0}
        )
        with pytest.raises(DataValidationError, match="No test users"):
            evaluate_knn(inter)

    def test_run_evaluation_writes_report(self, bx: Path, tmp_path: Path) -> None:
        out = tmp_path / "eval.json"
        rep = run_evaluation(bx, CFG, k=3, output=out)
        assert json.loads(out.read_text())["hit_rate"] == rep.hit_rate
