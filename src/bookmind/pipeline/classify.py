"""
Step 2 — assign exactly one genre per book from ``GENRE_TAXONOMY``.

Backends
--------
* ``ollama``  — local LLM (default model ``llama3.2``); no API key, no data leaves
                the machine. Retries with exponential backoff; on persistent
                failure falls back to the keyword classifier for that book.
* ``keyword`` — deterministic rules over Kaggle ``categories`` + title +
                description. Instant, offline, good baseline; used in CI.

Resumable: progress is keyed by ``book_id`` (the old script keyed by title,
which silently merged different books that share a title) and checkpointed
atomically every N books and on Ctrl+C.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from bookmind.core.config import GENRE_TAXONOMY
from bookmind.core.exceptions import ConfigurationError, DataValidationError
from bookmind.pipeline.io import read_csv, write_csv_atomic

log = logging.getLogger(__name__)

FALLBACK_GENRE = "Other"
_GENRE_LOOKUP = {g.lower(): g for g in GENRE_TAXONOMY}

PROMPT_TEMPLATE = """You are a librarian classifying a book into exactly ONE genre.

Genre options: {genres}

Book title: {title}
Description: {description}

Reply with ONLY the single genre name from the list above. No explanation, no punctuation."""


def parse_genre(raw: str | None) -> str:
    """
    Map free-form LLM output onto the taxonomy.

    exact match  >  output starts with a genre  >  longest genre mentioned  >  "Other"
    (longest-first avoids "Fiction" style partial matches picking the wrong genre).
    """
    if not raw:
        return FALLBACK_GENRE
    # Keep inner apostrophes ("Children's") but drop quotes/markdown/punctuation.
    text = re.sub(r"[\"`*.\n\r]+", " ", raw).strip().strip("'").strip().lower()
    text = re.sub(r"^(genre|answer)\s*:\s*", "", text).strip()
    if text in _GENRE_LOOKUP:
        return _GENRE_LOOKUP[text]
    by_length = sorted(_GENRE_LOOKUP, key=len, reverse=True)
    for key in by_length:
        if text.startswith(key):
            return _GENRE_LOOKUP[key]
    for key in by_length:
        if re.search(rf"(?<![a-z]){re.escape(key)}(?![a-z])", text):
            return _GENRE_LOOKUP[key]
    return FALLBACK_GENRE


class GenreClassifier(Protocol):
    name: str

    def classify(self, title: str, description: str, categories: str = "") -> str: ...


# ── Keyword backend ──────────────────────────────────────────────────────────
_CATEGORY_RULES: tuple[tuple[str, str], ...] = (
    (r"juvenile|children", "Children's"),
    (r"young adult", "Young Adult"),
    (r"biography|autobiography|memoir", "Biography & Memoir"),
    (r"self-help|self help|personal growth", "Self-Help"),
    (r"true crime", "True Crime"),
    (r"business|economics|management", "Business & Economics"),
    (r"religion|philosophy|spiritual|bible|christian", "Philosophy & Religion"),
    (r"poetry", "Poetry"),
    (r"humor|comic", "Humor"),
    (r"science fiction|sci-fi", "Science Fiction"),
    (r"fantasy", "Fantasy"),
    (r"horror", "Horror"),
    (r"romance", "Romance"),
    (r"mystery|detective", "Mystery"),
    (r"thriller|suspense", "Thriller"),
    (r"historical fiction", "Historical Fiction"),
    (r"history", "History"),
    (r"science|nature|mathematics|medical|technology", "Science & Nature"),
)

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Fantasy": ("wizard", "dragon", "magic", "sorcer", "elves", "kingdom", "quest", "fae"),
    "Science Fiction": (
        "spaceship",
        "galaxy",
        "alien",
        "planet",
        "robot",
        "dystopi",
        "starship",
        "time travel",
    ),
    "Mystery": ("detective", "murder", "mystery", "investigat", "whodunit", "sleuth", "clue"),
    "Thriller": (
        "thriller",
        "assassin",
        "conspiracy",
        "spy",
        "serial killer",
        "fbi",
        "cia",
        "race against time",
    ),
    "Romance": ("love story", "romance", "falls in love", "passion", "heart", "lovers", "duke"),
    "Horror": ("horror", "haunted", "ghost", "vampire", "demon", "terror", "nightmare", "zombie"),
    "Historical Fiction": ("world war", "civil war", "century", "victorian", "medieval", "tudor"),
    "Young Adult": (
        "teen",
        "high school",
        "sixteen-year-old",
        "seventeen-year-old",
        "coming of age",
    ),
    "Children's": ("picture book", "children", "kids", "bedtime", "ages 4"),
    "Biography & Memoir": ("memoir", "biography", "life of", "autobiography", "my life"),
    "Self-Help": ("self-help", "habits", "productivity", "happiness", "success", "mindset"),
    "History": ("history of", "historian", "empire", "ancient", "revolution"),
    "Science & Nature": (
        "scientist",
        "physics",
        "biology",
        "evolution",
        "universe",
        "nature",
        "climate",
    ),
    "Business & Economics": (
        "business",
        "economy",
        "economics",
        "leadership",
        "market",
        "entrepreneur",
    ),
    "Philosophy & Religion": (
        "philosophy",
        "god",
        "faith",
        "religion",
        "spiritual",
        "buddh",
        "christian",
    ),
    "Poetry": ("poems", "poetry", "verse", "poet"),
    "True Crime": ("true story of a murder", "true crime", "real-life crime"),
    "Humor": ("hilarious", "humor", "humour", "comic", "funny", "laugh"),
    "Literary Fiction": ("novel", "family", "story of", "fiction"),
}


# Word-start boundary (\b prefix) so "teen" doesn't match "nineteen", while
# stems such as "dystopi" still match "dystopian".
_KEYWORD_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    genre: tuple(re.compile(rf"\b{re.escape(w)}") for w in words)
    for genre, words in _KEYWORDS.items()
}


class KeywordClassifier:
    name = "keyword"

    def classify(self, title: str, description: str, categories: str = "") -> str:
        cats = (categories or "").lower()
        for pattern, genre in _CATEGORY_RULES:
            if re.search(pattern, cats):
                return genre
        title_l, desc_l = (title or "").lower(), (description or "").lower()
        scores: dict[str, int] = {}
        for genre, patterns in _KEYWORD_PATTERNS.items():
            score = sum(3 * len(p.findall(title_l)) + len(p.findall(desc_l)) for p in patterns)
            if score:
                scores[genre] = score
        if scores:
            # Literary Fiction is the generic catch-all; prefer any specific genre.
            specific = {g: s for g, s in scores.items() if g != "Literary Fiction"}
            pool = specific or scores
            return max(pool.items(), key=lambda kv: (kv[1], -GENRE_TAXONOMY.index(kv[0])))[0]
        if "fiction" in cats:
            return "Literary Fiction"
        return FALLBACK_GENRE


# ── Ollama backend ───────────────────────────────────────────────────────────
class OllamaClassifier:
    """Local-LLM classifier with retries, timeout and graceful fallback."""

    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
        *,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        client: Any | None = None,
        fallback: GenreClassifier | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if max_retries < 1:
            raise ConfigurationError("max_retries must be >= 1")
        self.model = model
        self.name = f"ollama:{model}"
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.fallback = fallback or KeywordClassifier()
        self._sleep = sleep
        self.failures = 0
        if client is None:
            try:
                import ollama
            except ImportError as exc:  # pragma: no cover - environment specific
                raise ConfigurationError(
                    "The 'ollama' package is not installed: "
                    "pip install -r requirements-pipeline.txt"
                ) from exc
            client = ollama.Client(host=host, timeout=timeout)
        self._client = client

    def check_available(self) -> None:
        """Fail fast if the server is down or the model has not been pulled."""
        try:
            listing = self._client.list()
        except Exception as exc:
            raise ConfigurationError(
                f"Cannot reach the Ollama server ({exc}). Start it with `ollama serve`, "
                "or use --backend keyword."
            ) from exc
        models = (
            listing.get("models", [])
            if isinstance(listing, dict)
            else getattr(listing, "models", [])
        )
        names = {
            str(m.get("model") or m.get("name") if isinstance(m, dict) else getattr(m, "model", ""))
            for m in models
        }
        if not any(n == self.model or n.startswith(f"{self.model}:") for n in names):
            raise ConfigurationError(
                f"Ollama model '{self.model}' is not pulled. Run: ollama pull {self.model}"
            )

    def classify(self, title: str, description: str, categories: str = "") -> str:
        prompt = PROMPT_TEMPLATE.format(
            genres=", ".join(GENRE_TAXONOMY),
            title=title[:300],
            description=(description or "")[:600],
        )
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0},
                )
                content = response["message"]["content"]
                return parse_genre(str(content))
            except Exception as exc:  # network, timeout, malformed response
                last_error = exc
                if attempt < self.max_retries:
                    delay = self.backoff_seconds * (2 ** (attempt - 1))
                    log.warning(
                        "Ollama call failed (attempt %d): %s; retrying in %.1fs",
                        attempt,
                        exc,
                        delay,
                    )
                    self._sleep(delay)
        self.failures += 1
        log.error(
            "Ollama failed for '%s' after %d attempts: %s — using fallback",
            title[:60],
            self.max_retries,
            last_error,
        )
        return self.fallback.classify(title, description, categories)


# ── Runner ───────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ClassifyReport:
    total: int
    already_done: int
    classified: int
    genre_counts: dict[str, int]


def _load_progress(output_csv: Path) -> pd.DataFrame | None:
    if not output_csv.exists():
        return None
    done = read_csv(output_csv, required=("book_id", "genre"))
    invalid = ~done["genre"].isin(GENRE_TAXONOMY)
    if invalid.any():
        log.warning("Discarding %d checkpoint rows with invalid genres", int(invalid.sum()))
        done = done[~invalid]
    deduped: pd.DataFrame = done.drop_duplicates("book_id", keep="last")
    return deduped


def run_classify(
    input_csv: Path,
    output_csv: Path,
    classifier: GenreClassifier,
    *,
    checkpoint_every: int = 50,
    progress: Callable[[Iterable[Any], int], Iterable[Any]] | None = None,
) -> ClassifyReport:
    if checkpoint_every < 1:
        raise DataValidationError("checkpoint_every must be >= 1")
    books = read_csv(input_csv, required=("book_id", "title", "description"))
    if books["book_id"].duplicated().any():
        raise DataValidationError("Input has duplicate book_id values; re-run `prepare`.")

    done = _load_progress(output_csv)
    done_ids = set(done["book_id"]) if done is not None else set()
    if done is not None:
        done = done[done["book_id"].isin(books["book_id"])]
    remaining = books[~books["book_id"].isin(done_ids)]
    log.info(
        "Classifying with %s",
        classifier.name,
        extra={"total": len(books), "already_done": len(done_ids), "remaining": len(remaining)},
    )

    new_rows: list[dict[str, Any]] = []
    genre_by_id: dict[str, str] = (
        dict(zip(done["book_id"], done["genre"])) if done is not None else {}
    )

    def checkpoint() -> None:
        for row in new_rows:
            genre_by_id[row["book_id"]] = row["genre"]
        new_rows.clear()
        merged = books[books["book_id"].isin(genre_by_id)].copy()
        merged["genre"] = merged["book_id"].map(genre_by_id)
        write_csv_atomic(merged, output_csv)

    records = remaining.to_dict("records")
    iterator: Iterable[Any] = progress(records, len(records)) if progress else records
    classified = 0
    try:
        for i, rec in enumerate(iterator, start=1):
            genre = classifier.classify(
                str(rec.get("title") or ""),
                str(rec.get("description") or ""),
                str(rec.get("categories") or "") if pd.notna(rec.get("categories")) else "",
            )
            if genre not in GENRE_TAXONOMY:  # defensive: never persist unknown labels
                genre = FALLBACK_GENRE
            new_rows.append({"book_id": rec["book_id"], "genre": genre})
            classified += 1
            if i % checkpoint_every == 0:
                checkpoint()
    except KeyboardInterrupt:
        log.warning("Interrupted — saving progress (%d new books)", len(new_rows))
        checkpoint()
        raise
    checkpoint()

    final = read_csv(output_csv, required=("book_id", "genre"))
    counts = {str(k): int(v) for k, v in final["genre"].value_counts().items()}
    log.info("classify complete → %s", output_csv, extra={"genres": counts})
    return ClassifyReport(len(books), len(done_ids), classified, counts)
