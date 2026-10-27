# mypy: disable-error-code="no-any-return"
"""
Collaborative-filtering (item-item k-NN) training & evaluation on Book-Crossing.

Replaces the original ``preprocess.py`` + ``train_model.py`` with the same
model family (scikit-learn ``NearestNeighbors``, brute force, cosine) and fixes:

1. **Implicit ratings.** In Book-Crossing a rating of ``0`` means "interacted,
   no explicit score". The original kept those zeros *and* filled missing cells
   with 0, so "read it" and "never saw it" were indistinguishable. Every mode
   here keeps 0 in the matrix strictly for "no interaction":

   * ``binary``   (default) any interaction = 1
   * ``weighted`` explicit score 1-10, implicit interaction = 1
   * ``explicit`` explicit scores only (implicit rows dropped)

   ``binary`` was chosen by measurement: on the real Book-Crossing ratings it
   had the best Hit-Rate@5 of the three at the default thresholds (see
   ``bookmind-pipeline evaluate-knn --compare-modes``).
2. **Grouping key.** Books are grouped by normalised *(title, author)* instead
   of title only — editions of the same book merge, different books that share
   a title (e.g. "Selected Poems") no longer do.
3. **Filter order.** Active users are counted *after* joining with the book
   catalogue, so ratings for unknown ISBNs don't inflate activity counts.
4. **Cleaning.** HTML entities unescaped, ISBNs normalised, invalid years
   dropped, cover URLs upgraded to https.
5. **Validation, logging, atomic writes and a manifest** (row counts,
   thresholds, scikit-learn version, checksums) like the rest of the pipeline.

The output is the same four artifacts the web app's ``KnnRecommender`` loads:
``model_knn.pkl``, ``book_names.pkl``, ``final_rating.pkl``, ``book_pivot.pkl``
(+ ``knn_manifest.json``).
"""

from __future__ import annotations

import html
import json
import logging
import pickle
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from bookmind.core.config import KNN_ARTIFACTS
from bookmind.core.exceptions import DataValidationError
from bookmind.services.artifacts import atomic_write_bytes, sha256_file

log = logging.getLogger(__name__)

RatingMode = Literal["weighted", "explicit", "binary"]
BOOKS_FILE = "BX-Books.csv"
RATINGS_FILE = "BX-Book-Ratings.csv"
KNN_MANIFEST = "knn_manifest.json"
IMPLICIT_WEIGHT = 1.0  # value given to an implicit (0) interaction in weighted mode

_BOOK_COLUMNS = {
    "ISBN": "isbn",
    "Book-Title": "title",
    "Book-Author": "author",
    "Year-Of-Publication": "year",
    "Publisher": "publisher",
    "Image-URL-L": "image_url",
}
_RATING_COLUMNS = {"User-ID": "user_id", "ISBN": "isbn", "Book-Rating": "rating"}
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class KnnConfig:
    mode: RatingMode = "binary"
    min_user_ratings: int = 200  # keep users with MORE than this many interactions
    min_book_ratings: int = 50  # keep books with AT LEAST this many distinct readers
    n_neighbors_hint: int = 6

    def validate(self) -> None:
        if self.mode not in ("weighted", "explicit", "binary"):
            raise DataValidationError(f"Unknown rating mode '{self.mode}'.")
        if self.min_user_ratings < 0 or self.min_book_ratings < 1:
            raise DataValidationError(
                "Thresholds must be min_user_ratings>=0, min_book_ratings>=1."
            )


@dataclass
class KnnTrainReport:
    mode: str
    raw_books: int
    raw_ratings: int
    joined_ratings: int
    active_users: int
    books: int
    interactions: int
    density: float
    min_user_ratings: int
    min_book_ratings: int
    sklearn_version: str = sklearn.__version__
    # Versions that wrote the pickles; deployments pin these so the files load the same way.
    library_versions: dict[str, str] = field(
        default_factory=lambda: {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__,
        }
    )
    trained_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    checksums: dict[str, str] = field(default_factory=dict)


# ── Loading ──────────────────────────────────────────────────────────────────
def _read_bx(path: Path, columns: dict[str, str]) -> pd.DataFrame:
    if not path.exists():
        raise DataValidationError(
            f"{path.name} not found in {path.parent}. Run `bookmind-pipeline download` "
            "or copy the Book-Crossing CSVs there.",
            details={"path": str(path)},
        )
    try:
        df = pd.read_csv(
            path, sep=";", encoding="latin-1", on_bad_lines="skip", dtype=str, low_memory=False
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataValidationError(f"Could not parse {path.name}: {exc}") from exc
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"{path.name} is missing columns {missing}", details={"found": df.columns.tolist()}
        )
    df = df.loc[:, list(columns)].rename(columns=columns)
    if df.empty:
        raise DataValidationError(f"{path.name} has no rows.")
    return df


def _clean(series: pd.Series) -> pd.Series:
    result: pd.Series = (
        series.fillna("")
        .astype(str)
        .map(html.unescape)
        .str.replace(_WS, " ", regex=True)
        .str.strip()
    )
    return result


def _norm_isbn(series: pd.Series) -> pd.Series:
    result: pd.Series = (
        series.fillna("").astype(str).str.upper().str.replace(r"[^0-9X]", "", regex=True)
    )
    return result


def load_books(bx_dir: Path) -> pd.DataFrame:
    books = _read_bx(bx_dir / BOOKS_FILE, _BOOK_COLUMNS)
    books["isbn"] = _norm_isbn(books["isbn"])
    for col in ("title", "author", "publisher"):
        books[col] = _clean(books[col])
    books.loc[books["author"] == "", "author"] = "Unknown"
    year = pd.to_numeric(books["year"], errors="coerce")
    max_year = datetime.now(timezone.utc).year + 1
    books["year"] = year.where((year >= 1000) & (year <= max_year)).astype("Int64")
    url = _clean(books["image_url"]).str.replace(r"^http://", "https://", regex=True)
    books["image_url"] = url.where(url.str.startswith("https://"), "")
    books = books[(books["isbn"] != "") & (books["title"] != "")]
    return books.drop_duplicates("isbn").reset_index(drop=True)


def load_ratings(bx_dir: Path) -> pd.DataFrame:
    ratings = _read_bx(bx_dir / RATINGS_FILE, _RATING_COLUMNS)
    ratings["isbn"] = _norm_isbn(ratings["isbn"])
    ratings["user_id"] = pd.to_numeric(ratings["user_id"], errors="coerce")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    valid = ratings["user_id"].notna() & ratings["rating"].between(0, 10) & (ratings["isbn"] != "")
    dropped = int((~valid).sum())
    if dropped:
        log.warning("Dropped %d invalid rating rows", dropped)
    ratings = ratings[valid].astype({"user_id": "int64", "rating": "int64"})
    return ratings.reset_index(drop=True)


# ── Transform ────────────────────────────────────────────────────────────────
def interaction_value(rating: pd.Series, mode: RatingMode) -> pd.Series:
    """Map raw Book-Crossing ratings (0 = implicit, 1-10 = explicit) to matrix values."""
    r = rating.astype("float64")
    if mode == "binary":
        return pd.Series(np.ones(len(r)), index=r.index)
    if mode == "explicit":
        return r.where(r > 0)  # implicit rows become NaN and are dropped
    return r.where(r > 0, IMPLICIT_WEIGHT)  # weighted


def build_interactions(ratings: pd.DataFrame, books: pd.DataFrame, cfg: KnnConfig) -> pd.DataFrame:
    """Return the cleaned long table used for the pivot (one row per user x book)."""
    cfg.validate()
    df = ratings.merge(books, on="isbn", how="inner")
    df["value"] = interaction_value(df["rating"], cfg.mode)
    df = df[df["value"].notna()]

    # Group editions: same normalised title + author == same book.
    df["item_key"] = df["title"].str.lower() + "|" + df["author"].str.lower()

    user_counts = df.groupby("user_id")["item_key"].nunique()
    active = user_counts[user_counts > cfg.min_user_ratings].index
    df = df[df["user_id"].isin(active)]

    book_counts = df.groupby("item_key")["user_id"].nunique()
    popular = book_counts[book_counts >= cfg.min_book_ratings].index
    df = df[df["item_key"].isin(popular)]
    if df.empty:
        raise DataValidationError(
            "No interactions left after filtering — lower --min-user-ratings / --min-book-ratings.",
            details={"mode": cfg.mode},
        )

    # One row per (user, book): keep the strongest signal across editions.
    df = df.sort_values("value", ascending=False).drop_duplicates(["user_id", "item_key"])

    # Display title must be unique (it is the pivot index the app looks up).
    meta = df.drop_duplicates("item_key")[["item_key", "title", "author"]]
    dup = meta["title"].duplicated(keep=False)
    meta.loc[dup, "title"] = meta.loc[dup, "title"] + " (" + meta.loc[dup, "author"] + ")"
    df = df.drop(columns=["title"]).merge(meta[["item_key", "title"]], on="item_key")

    # Representative cover per book: first non-empty URL.
    covers = (
        df[df["image_url"] != ""].drop_duplicates("item_key").set_index("item_key")["image_url"]
    )
    df["image_url"] = df["item_key"].map(covers).fillna("")
    cols = [
        "user_id",
        "isbn",
        "title",
        "author",
        "year",
        "publisher",
        "image_url",
        "rating",
        "value",
    ]
    return df.loc[:, cols].reset_index(drop=True)


def build_pivot(interactions: pd.DataFrame) -> pd.DataFrame:
    pivot = interactions.pivot_table(
        index="title", columns="user_id", values="value", aggfunc="max", fill_value=0.0
    )
    pivot.columns = pivot.columns.astype("int64")
    return pivot.astype("float64")


def fit_model(pivot: pd.DataFrame) -> NearestNeighbors:
    if pivot.shape[0] < 2:
        raise DataValidationError("Need at least 2 books to train a k-NN model.")
    model = NearestNeighbors(algorithm="brute", metric="cosine")
    model.fit(csr_matrix(pivot.to_numpy()))
    return model


# ── Persist ──────────────────────────────────────────────────────────────────
def save_knn_artifacts(
    artifacts_dir: Path,
    model: NearestNeighbors,
    pivot: pd.DataFrame,
    interactions: pd.DataFrame,
    report: KnnTrainReport,
) -> KnnTrainReport:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / KNN_MANIFEST).unlink(missing_ok=True)
    objects: dict[str, Any] = {
        "model_knn": model,
        "book_names": pivot.index,
        "final_rating": interactions,
        "book_pivot": pivot,
    }
    for name in KNN_ARTIFACTS:
        path = artifacts_dir / f"{name}.pkl"
        atomic_write_bytes(path, pickle.dumps(objects[name], protocol=4))
        report.checksums[path.name] = sha256_file(path)
    atomic_write_bytes(
        artifacts_dir / KNN_MANIFEST, json.dumps(asdict(report), indent=2).encode("utf-8")
    )
    return report


def train_knn(bx_dir: Path, artifacts_dir: Path, cfg: KnnConfig | None = None) -> KnnTrainReport:
    cfg = cfg or KnnConfig()
    cfg.validate()
    started = time.perf_counter()
    books = load_books(bx_dir)
    ratings = load_ratings(bx_dir)
    log.info("Loaded Book-Crossing", extra={"books": len(books), "ratings": len(ratings)})

    joined = int(ratings["isbn"].isin(books["isbn"]).sum())
    interactions = build_interactions(ratings, books, cfg)
    pivot = build_pivot(interactions)
    model = fit_model(pivot)

    report = KnnTrainReport(
        mode=cfg.mode,
        raw_books=len(books),
        raw_ratings=len(ratings),
        joined_ratings=joined,
        active_users=int(pivot.shape[1]),
        books=int(pivot.shape[0]),
        interactions=len(interactions),
        density=round(len(interactions) / float(pivot.shape[0] * pivot.shape[1]), 6),
        min_user_ratings=cfg.min_user_ratings,
        min_book_ratings=cfg.min_book_ratings,
    )
    save_knn_artifacts(artifacts_dir, model, pivot, interactions, report)
    log.info(
        "train-knn complete",
        extra={
            "books": report.books,
            "users": report.active_users,
            "interactions": report.interactions,
            "seconds": round(time.perf_counter() - started, 2),
        },
    )
    return report


# ── Evaluation ───────────────────────────────────────────────────────────────
@dataclass
class KnnEvalReport:
    mode: str
    k: int
    test_users: int
    queries: int
    hit_rate: float
    popularity_hit_rate: float
    lift_over_popularity: float
    catalogue_coverage: float
    popularity_coverage: float
    books: int


def evaluate_knn(
    interactions: pd.DataFrame,
    *,
    k: int = 5,
    test_fraction: float = 0.2,
    seeds_per_user: int = 5,
    random_state: int = 42,
    mode: str = "binary",
) -> KnnEvalReport:
    """
    Offline evaluation with a user-level hold-out (no leakage):

    1. Split users 80/20. Fit the item-item model on *train users only*.
    2. For each test user, take up to ``seeds_per_user`` books they interacted
       with as seeds. A query is a **hit** if any of the user's *other* books
       appears in the top-``k`` neighbours of the seed.
    3. Compare with a popularity baseline (top-``k`` most-read train books).

    Report both hit-rate *and* coverage: a popularity list can score a high
    hit-rate on Book-Crossing while recommending the same handful of books to
    everyone (coverage near 0), which is useless as a "similar books" feature.
    """
    if not 0 < test_fraction < 1:
        raise DataValidationError("test_fraction must be between 0 and 1.")
    if k < 1:
        raise DataValidationError("k must be >= 1.")
    rng = np.random.default_rng(random_state)
    users = interactions["user_id"].unique()
    rng.shuffle(users)
    n_test = max(1, int(len(users) * test_fraction))
    test_users, train_users = set(users[:n_test]), set(users[n_test:])

    train = interactions[interactions["user_id"].isin(train_users)]
    pivot = build_pivot(train)
    model = fit_model(pivot)
    titles = pivot.index.to_numpy()
    row_of = {t: i for i, t in enumerate(titles)}
    popular = train.groupby("title")["user_id"].nunique().sort_values(ascending=False).index

    hits = pop_hits = queries = 0
    recommended: set[str] = set()
    pop_recommended: set[str] = set()
    test = interactions[interactions["user_id"].isin(test_users)]
    n_neighbors = min(k + 1, len(titles))
    for _, books in test.groupby("user_id")["title"]:
        liked = [b for b in books.unique() if b in row_of]
        if len(liked) < 2:
            continue
        for seed in rng.permutation(liked)[:seeds_per_user]:
            others = set(liked) - {seed}
            _, idx = model.kneighbors(
                pivot.iloc[row_of[seed]].to_numpy().reshape(1, -1), n_neighbors=n_neighbors
            )
            recs = [titles[i] for i in idx[0] if titles[i] != seed][:k]
            pop = [t for t in popular if t != seed][:k]
            recommended.update(recs)
            pop_recommended.update(pop)
            hits += bool(others.intersection(recs))
            pop_hits += bool(others.intersection(pop))
            queries += 1

    if queries == 0:
        raise DataValidationError("No test users with >=2 books in the catalogue to evaluate.")
    hit_rate = hits / queries
    pop_rate = pop_hits / queries
    return KnnEvalReport(
        mode=mode,
        k=k,
        test_users=len(test_users),
        queries=queries,
        hit_rate=round(hit_rate, 4),
        popularity_hit_rate=round(pop_rate, 4),
        lift_over_popularity=round(hit_rate / pop_rate, 3) if pop_rate else float("inf"),
        catalogue_coverage=round(len(recommended) / len(titles), 4),
        popularity_coverage=round(len(pop_recommended) / len(titles), 4),
        books=len(titles),
    )


def run_evaluation(
    bx_dir: Path,
    cfg: KnnConfig,
    *,
    k: int = 5,
    output: Path | None = None,
) -> KnnEvalReport:
    books, ratings = load_books(bx_dir), load_ratings(bx_dir)
    report = evaluate_knn(build_interactions(ratings, books, cfg), k=k, mode=cfg.mode)
    if output is not None:
        atomic_write_bytes(output, json.dumps(asdict(report), indent=2).encode("utf-8"))
    log.info("evaluate-knn complete", extra=asdict(report))
    return report
