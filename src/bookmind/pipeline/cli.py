"""
``bookmind-pipeline`` — command-line entry point for the offline pipeline.

    bookmind-pipeline download     [--only semantic|knn] [--force]
    bookmind-pipeline prepare      [--input data/books_raw.csv]
    bookmind-pipeline classify     [--backend ollama|keyword] [--model llama3.2]
    bookmind-pipeline build-index  [--backend sentence-transformers|hashing]
    bookmind-pipeline train-knn    [--mode binary|weighted|explicit] [--evaluate]
    bookmind-pipeline evaluate-knn [--k 5] [--compare-modes]
    bookmind-pipeline all          prepare → classify → build-index → train-knn*
                                   (*train-knn runs when data/bx/ has the Book-Crossing CSVs)

Exit codes: 0 success · 1 data/config error · 2 bad CLI usage · 130 interrupted.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from bookmind.core.config import Settings, get_settings
from bookmind.core.exceptions import BookMindError
from bookmind.core.logging import setup_logging
from bookmind.ml.embedder import Encoder, HashingEncoder, SentenceTransformerEncoder
from bookmind.pipeline.build_index import build_index
from bookmind.pipeline.classify import (
    GenreClassifier,
    KeywordClassifier,
    OllamaClassifier,
    run_classify,
)
from bookmind.pipeline.download import download_datasets
from bookmind.pipeline.knn import BOOKS_FILE, RATINGS_FILE, KnnConfig, run_evaluation, train_knn
from bookmind.pipeline.prepare import run_prepare

log = logging.getLogger("bookmind.pipeline")

EXIT_OK, EXIT_ERROR, EXIT_INTERRUPTED = 0, 1, 130


def _tqdm_progress(items: Iterable[Any], total: int) -> Iterable[Any]:
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        return items
    return tqdm(items, total=total, unit="book")


def _paths(args: argparse.Namespace, settings: Settings) -> dict[str, Path]:
    data = Path(args.data_dir) if args.data_dir else settings.data_dir
    return {
        "raw": Path(args.input) if getattr(args, "input", None) else data / "books_raw.csv",
        "prepared": data / "books.csv",
        "genres": data / "books_genres.csv",
        "artifacts": Path(args.artifacts_dir) if args.artifacts_dir else settings.artifacts_dir,
        "bx": Path(args.bx_dir) if getattr(args, "bx_dir", None) else data / "bx",
    }


def _classifier(args: argparse.Namespace, settings: Settings) -> GenreClassifier:
    if args.classifier == "keyword":
        return KeywordClassifier()
    clf = OllamaClassifier(model=args.model or settings.ollama_model, host=settings.ollama_host)
    clf.check_available()
    return clf


def _encoder(args: argparse.Namespace, settings: Settings) -> Encoder:
    backend = args.encoder or settings.embedding_backend
    if backend == "hashing":
        return HashingEncoder(dim=settings.embedding_dim)
    return SentenceTransformerEncoder(settings.embedding_model)  # pragma: no cover


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bookmind-pipeline",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", help="Override BOOKMIND_DATA_DIR")
    parser.add_argument("--artifacts-dir", help="Override BOOKMIND_ARTIFACTS_DIR")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Clean & validate the raw CSV")
    p.add_argument("--input", help="Raw CSV (default: <data-dir>/books_raw.csv)")
    p.add_argument("--min-description-chars", type=int, default=30)

    c = sub.add_parser("classify", help="Assign one genre per book")
    c.add_argument("--backend", dest="classifier", choices=["ollama", "keyword"], default="ollama")
    c.add_argument("--model", help="Ollama model (default: BOOKMIND_OLLAMA_MODEL)")
    c.add_argument("--checkpoint-every", type=int, default=50)

    b = sub.add_parser("build-index", help="Embed books and build the FAISS index")
    b.add_argument("--backend", dest="encoder", choices=["sentence-transformers", "hashing"])

    d = sub.add_parser("download", help="Download both Kaggle datasets (kagglehub)")
    d.add_argument("--only", choices=["semantic", "knn"], help="Download just one dataset")
    d.add_argument("--force", action="store_true", help="Re-download even if files exist")

    def knn_options(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--bx-dir", help="Folder with BX-*.csv (default: <data-dir>/bx)")
        sp.add_argument("--mode", choices=["binary", "weighted", "explicit"], default="binary")
        sp.add_argument("--min-user-ratings", type=int, default=200)
        sp.add_argument("--min-book-ratings", type=int, default=50)

    t = sub.add_parser("train-knn", help="Train the collaborative-filtering k-NN model")
    knn_options(t)
    t.add_argument("--evaluate", action="store_true", help="Also run evaluate-knn")
    t.add_argument("--k", type=int, default=5)

    e = sub.add_parser("evaluate-knn", help="Offline Hit-Rate@K vs popularity baseline")
    knn_options(e)
    e.add_argument("--k", type=int, default=5)
    e.add_argument("--compare-modes", action="store_true", help="Evaluate all 3 rating modes")

    a = sub.add_parser("all", help="prepare → classify → build-index → train-knn")
    a.add_argument("--input", help="Raw CSV (default: <data-dir>/books_raw.csv)")
    a.add_argument("--min-description-chars", type=int, default=30)
    a.add_argument("--classifier", choices=["ollama", "keyword"], default="ollama")
    a.add_argument("--model")
    a.add_argument("--checkpoint-every", type=int, default=50)
    a.add_argument("--encoder", choices=["sentence-transformers", "hashing"])
    a.add_argument("--skip-knn", action="store_true", help="Do not train the k-NN model")
    knn_options(a)
    return parser


def _knn_config(args: argparse.Namespace, mode: str | None = None) -> KnnConfig:
    return KnnConfig(
        mode=mode or args.mode,  # type: ignore[arg-type]
        min_user_ratings=args.min_user_ratings,
        min_book_ratings=args.min_book_ratings,
    )


def _evaluate(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    modes = ["binary", "weighted", "explicit"] if getattr(args, "compare_modes", False) else [None]
    for mode in modes:
        cfg = _knn_config(args, mode)
        out = paths["artifacts"] / f"knn_eval_{cfg.mode}.json"
        try:
            r = run_evaluation(paths["bx"], cfg, k=args.k, output=out)
        except BookMindError as exc:
            if len(modes) == 1:
                raise
            log.warning("mode=%s skipped: %s", cfg.mode, exc.message)
            continue
        log.info(
            "k-NN [%s] Hit-Rate@%d=%.3f (popularity %.3f) | coverage %.1f%% (popularity %.1f%%)"
            " | %d books, %d queries -> %s",
            r.mode,
            r.k,
            r.hit_rate,
            r.popularity_hit_rate,
            r.catalogue_coverage * 100,
            r.popularity_coverage * 100,
            r.books,
            r.queries,
            out,
        )


def main(argv: Sequence[str] | None = None, settings: Settings | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = settings or get_settings()
    if args.verbose:
        settings = settings.model_copy(update={"log_level": "DEBUG"})
    setup_logging(settings)
    paths = _paths(args, settings)

    try:
        if args.command == "download":
            which = [args.only] if args.only else ["semantic", "knn"]
            data_dir = Path(args.data_dir) if args.data_dir else settings.data_dir
            download_datasets(data_dir, which=which, force=args.force)
        if args.command in ("prepare", "all"):
            run_prepare(
                paths["raw"], paths["prepared"], min_description_chars=args.min_description_chars
            )
        if args.command in ("classify", "all"):
            run_classify(
                paths["prepared"],
                paths["genres"],
                _classifier(args, settings),
                checkpoint_every=args.checkpoint_every,
                progress=_tqdm_progress,
            )
        if args.command in ("build-index", "all"):
            manifest = build_index(
                paths["genres"],
                paths["artifacts"],
                _encoder(args, settings),
                batch_size=settings.embed_batch_size,
            )
            log.info(
                "Index ready: %d books, model=%s", manifest.num_books, manifest.embedding_model
            )
        if args.command == "all" and not args.skip_knn:
            bx_ready = all((paths["bx"] / f).exists() for f in (BOOKS_FILE, RATINGS_FILE))
            if bx_ready:
                args.command = "train-knn"
            else:
                log.info("No Book-Crossing files in %s — skipping train-knn", paths["bx"])
        if args.command == "train-knn":
            report = train_knn(paths["bx"], paths["artifacts"], _knn_config(args))
            log.info(
                "k-NN ready: %d books x %d users (%d interactions, mode=%s)",
                report.books,
                report.active_users,
                report.interactions,
                report.mode,
            )
            if getattr(args, "evaluate", False):
                _evaluate(args, paths)
        if args.command == "evaluate-knn":
            _evaluate(args, paths)
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        return EXIT_INTERRUPTED
    except BookMindError as exc:
        log.error("%s: %s", exc.code, exc.message)
        return EXIT_ERROR
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
