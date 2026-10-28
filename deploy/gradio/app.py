"""
BookMind Pro on Hugging Face Spaces — the full FastAPI website, on free hardware.

Free Hugging Face accounts can only create *Gradio* Spaces (Docker is paid), and a Gradio
Space simply runs ``python app.py`` and shows whatever answers on port 7860. So this file
serves the **complete BookMind website** (the same pages, light/dark theme, popular-books
dropdown, REST API and /docs as `run.bat` locally) and mounts a small Gradio screen at
``/demo``, which keeps the ZeroGPU hardware happy. No search or recommendation logic
lives here: both front ends call the same service layer.

Run locally (inside the project venv, after building artifacts/):
    python -m pip install gradio
    python deploy/gradio/app.py            # http://127.0.0.1:7860  (Gradio at /demo)
"""

# ruff: noqa: E501, RUF001  (long Markdown table lines and emoji tab labels are intentional)
from __future__ import annotations

import html
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def extract_package(archive: Path, target: Path) -> Path:
    """Unpack bookmind.zip (code + HTML templates + CSS/JS) to a real folder.

    The website needs real files for Jinja2 templates and static assets, which Python
    cannot serve from inside a zip. Members are checked so nothing can be written outside
    ``target`` (zip-slip protection).
    """
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    root = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            destination = (target / member).resolve()
            if root != destination and root not in destination.parents:
                raise RuntimeError(f"Unsafe path in {archive.name}: {member}")
        zf.extractall(target)
    return target


# On the Space everything sits flat next to this file (no folders, so the browser
# uploader accepts it): `bookmind.zip` is the code, and the artifact files
# (manifest.json, book_index.faiss, ...). Locally fall back to src/ and artifacts/.
if (HERE / "bookmind.zip").exists():
    runtime = Path(tempfile.gettempdir()) / "bookmind_runtime"
    sys.path.insert(0, str(extract_package(HERE / "bookmind.zip", runtime)))
elif (HERE / "bookmind").is_dir():
    sys.path.insert(0, str(HERE))
else:
    sys.path.insert(0, str(HERE.parent.parent / "src"))
for candidate in (HERE, HERE / "artifacts", HERE.parent.parent / "artifacts"):
    if (candidate / "manifest.json").exists():
        os.environ.setdefault("BOOKMIND_ARTIFACTS_DIR", str(candidate))
        break

# Settings for the public demo (each can be overridden in the Space's Settings -> Variables).
SPACE_DEFAULTS = {
    "BOOKMIND_ENV": "production",  # HTTPS-only headers, no debug
    "BOOKMIND_ENABLE_DOCS": "true",  # keep /docs for the mentor
    "BOOKMIND_LOG_TO_FILE": "false",  # Space logs are read from the console
    "BOOKMIND_FRAME_ANCESTORS": '["https://huggingface.co"]',  # the Space page embeds us
    "BOOKMIND_MOUNTED_APP_PATHS": '["/demo"]',  # Gradio brings its own scripts
}
for key, value in SPACE_DEFAULTS.items():
    os.environ.setdefault(key, value)

import gradio as gr  # noqa: E402

from bookmind import __version__  # noqa: E402
from bookmind.core.config import get_settings  # noqa: E402
from bookmind.core.exceptions import BookMindError  # noqa: E402
from bookmind.core.logging import setup_logging  # noqa: E402
from bookmind.main import create_app  # noqa: E402
from bookmind.schemas.book import Book, PopularBook  # noqa: E402
from bookmind.services.recommendation import (  # noqa: E402
    catalogue_stats,
    find_similar,
    popular_titles,
)
from bookmind.services.registry import ServiceRegistry  # noqa: E402

try:  # Present on Hugging Face ZeroGPU hardware; harmless no-op elsewhere.
    import spaces
except ImportError:  # pragma: no cover - local runs without the `spaces` package
    spaces = None

settings = get_settings()
setup_logging(settings)
registry = ServiceRegistry.load(settings)  # loads both engines once, at startup

if spaces is not None:
    # ZeroGPU Spaces must declare at least one GPU function at startup. BookMind runs
    # fully on CPU (MiniLM + FAISS take milliseconds), so this one is never called.
    @spaces.GPU
    def _zerogpu_placeholder() -> None:
        return None


def report_zerogpu_startup() -> bool:
    """Send ZeroGPU's startup report (the GPU function list) to Hugging Face.

    `spaces` normally sends it from inside Gradio's `demo.launch()`. This app is served by
    uvicorn instead (so the full website owns "/"), so the report is sent here. Without it
    the Space stops with "No @spaces.GPU function detected during startup".
    Returns True when a report was sent (only on ZeroGPU hardware).
    """
    startup = getattr(getattr(spaces, "zero", None), "startup", None)
    if not callable(startup):
        return False  # not on ZeroGPU (local runs, CPU hardware)
    startup()
    return True


ALL = "All genres"
PLACEHOLDER_COVER = (  # no quotes inside: it is used in HTML attributes and in JS
    "data:image/svg+xml;utf8,"
    "<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%2296%22 height=%22144%22>"
    "<rect width=%2296%22 height=%22144%22 rx=%226%22 fill=%22%2394a3b8%22 "
    "fill-opacity=%220.25%22/><text x=%2248%22 y=%2280%22 font-size=%2234%22 "
    "text-anchor=%22middle%22>📖</text></svg>"
)

CSS = """
.bm-grid {display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;}
.bm-card {display:flex;gap:14px;padding:14px;border-radius:14px;
  border:1px solid var(--border-color-primary);background:var(--block-background-fill);}
.bm-cover {width:84px;height:126px;flex-shrink:0;border-radius:6px;overflow:hidden;
  background-size:cover;background-position:center;}
.bm-cover img {width:100%;height:100%;object-fit:cover;display:block;color:transparent;}
.bm-body {min-width:0;}
.bm-title {font-weight:700;font-size:1.02em;line-height:1.3;margin:0 0 2px;}
.bm-author {color:var(--body-text-color-subdued);font-size:.9em;margin-bottom:6px;}
.bm-meta {display:flex;flex-wrap:wrap;gap:6px;margin-bottom:6px;font-size:.8em;}
.bm-pill {padding:1px 8px;border-radius:999px;border:1px solid var(--border-color-primary);}
.bm-score {color:var(--color-accent);font-weight:600;}
.bm-desc {font-size:.86em;color:var(--body-text-color-subdued);margin:0 0 6px;}
.bm-links a {font-size:.82em;margin-right:10px;}
.bm-head {margin:4px 0 12px;font-size:1.05em;}
.bm-note {padding:12px 14px;border-radius:10px;border:1px solid var(--border-color-primary);}
footer {display:none !important;}
"""


# ── Rendering ────────────────────────────────────────────────────────────────
def _e(value: Any) -> str:
    """HTML-escape any value (book data is untrusted text)."""
    return html.escape(str(value), quote=True)


def _card(book: Book | PopularBook) -> str:
    cover = book.image_url or PLACEHOLDER_COVER
    meta = [f"<span class='bm-pill'>{_e(book.genre)}</span>"]
    if book.avg_rating is not None:
        meta.append(f"<span>★ {book.avg_rating:.1f}</span>")
    year = getattr(book, "published_year", None)
    if year:
        meta.append(f"<span>{_e(year)}</span>")
    score = getattr(book, "score", None)
    if score is not None:
        meta.append(f"<span class='bm-score'>{score * 100:.0f}% match</span>")
    readers = getattr(book, "num_ratings", None)
    if readers:
        meta.append(f"<span>{readers:,} ratings</span>")
    full = getattr(book, "description", "") or ""
    desc = full[:220] + ("…" if len(full) > 220 else "")
    links = book.buy_links
    return (
        "<div class='bm-card'>"
        # Placeholder is the background; a cover that fails to load shows nothing
        # on top of it (alt=""), so no JavaScript onerror handler is needed.
        f"<div class='bm-cover' style='background-image:url(\"{PLACEHOLDER_COVER}\")'>"
        f"<img src='{_e(cover)}' alt='' loading='lazy'></div>"
        "<div class='bm-body'>"
        f"<p class='bm-title'>{_e(book.title)}</p>"
        f"<div class='bm-author'>by {_e(book.author)}</div>"
        f"<div class='bm-meta'>{''.join(meta)}</div>"
        + (f"<p class='bm-desc'>{_e(desc)}</p>" if desc else "")
        + "<div class='bm-links'>"
        f"<a href='{_e(links.amazon)}' target='_blank' rel='noopener'>Amazon</a>"
        f"<a href='{_e(links.flipkart)}' target='_blank' rel='noopener'>Flipkart</a>"
        f"<a href='{_e(links.google_books)}' target='_blank' rel='noopener'>Google Books</a>"
        "</div></div></div>"
    )


def render(books: list[Book] | list[PopularBook], heading: str) -> str:
    if not books:
        return "<div class='bm-note'>No books found. Try different words or another genre.</div>"
    cards = "".join(_card(b) for b in books)
    return f"<div class='bm-head'>{heading}</div><div class='bm-grid'>{cards}</div>"


def render_error(exc: BookMindError) -> str:
    suggestions = (exc.details or {}).get("suggestions") or []
    extra = ""
    if suggestions:
        items = "".join(f"<li>{_e(s)}</li>" for s in suggestions[:5])
        extra = f"<p>Did you mean:</p><ul>{items}</ul>"
    return f"<div class='bm-note'><b>{_e(exc.message)}</b>{extra}</div>"


# ── Actions (each one is a thin call into the service layer) ─────────────────
def search(query: str, genre: str, top_k: float) -> str:
    """Free-text semantic search; with an empty query, browse the chosen genre."""
    try:
        engine = registry.require_semantic()
        k = int(top_k)
        if query and query.strip():
            books = engine.search(query, genre=None if genre == ALL else genre, top_k=k)
            where = "" if genre == ALL else f" in <b>{_e(genre)}</b>"
            return render(books, f"Results for <b>“{_e(query.strip())}”</b>{where}")
        if genre != ALL:
            return render(engine.browse(genre, top_k=k), f"Most popular in <b>{_e(genre)}</b>")
        return render_popular(k)
    except BookMindError as exc:
        return render_error(exc)


def render_popular(k: int = 12) -> str:
    try:
        books = registry.require_semantic().popular(k)
    except BookMindError as exc:
        return render_error(exc)
    return render(books, "Popular books — describe what you want to read, or pick a genre")


def recommend(title: str, top_k: float) -> str:
    """Hybrid 'more like this': collaborative filtering first, semantic fallback."""
    if not title or not title.strip():
        return "<div class='bm-note'>Pick or type a book title first.</div>"
    try:
        matched, engine_used, books = find_similar(registry, title, top_k=int(top_k))
    except BookMindError as exc:
        return render_error(exc)
    if engine_used == "knn":
        head = f"Readers who rated <b>{_e(matched)}</b> also loved these"
    else:
        head = f"Books with the most similar themes to <b>{_e(matched)}</b>"
    return render(books, head)


def about_markdown() -> str:
    if registry.semantic is None:
        return f"⚠️ Search index not loaded: {registry.errors.get('semantic', 'unknown error')}"
    s = catalogue_stats(registry)
    knn = (
        f"**{s.knn_books:,} books × {s.total_users:,} readers** "
        f"({s.total_ratings:,} ratings, `{s.knn_mode}` encoding)"
        if s.knn_available
        else "not loaded (semantic similarity is used instead)"
    )
    return f"""
### How BookMind works
| Engine | What it does | Data |
|---|---|---|
| **Semantic search** | Turns your words and every book into 384-number vectors (`{s.embedding_model}`) and finds the closest ones with **FAISS** | **{s.total_books:,} books**, {s.total_genres} genres |
| **Collaborative filtering** | Item-item **k-NN** (cosine) over who read what: "readers who liked X also liked Y" | {knn} |

**Hybrid:** *Recommend* uses collaborative filtering when the title is in the ratings data, otherwise semantic similarity.

Built with Python · Sentence-Transformers · FAISS · scikit-learn · pandas · Gradio.
The full project also has a FastAPI website and REST API, 500+ automated tests, Docker and CI.
No paid APIs — everything is free and open source. Version {__version__}.
"""


# ── Layout ───────────────────────────────────────────────────────────────────
def build_demo() -> gr.Blocks:
    genres = [ALL, *(registry.semantic.genres if registry.semantic else [])]
    title_choices: list[str] = []
    if registry.semantic is not None or registry.knn is not None:
        title_choices, _ = popular_titles(registry, limit=50)
        if registry.knn is not None:
            title_choices += [t for t in registry.knn.titles if t not in title_choices]

    with gr.Blocks(title="BookMind — AI Book Recommender") as demo:
        gr.Markdown(
            "# 📚 BookMind\n"
            "Find books by **meaning**, browse a genre, or get **“books like this one”** "
            "recommendations — semantic search + collaborative filtering."
        )
        if registry.semantic is None:
            gr.Markdown(f"⚠️ **Search index not loaded:** {registry.errors.get('semantic')}")

        with gr.Tab("🔎 Search"):
            with gr.Row():
                query = gr.Textbox(
                    label="Describe a book (or leave empty and pick a genre)",
                    placeholder="e.g. a slow-burn mystery set in winter",
                    scale=4,
                )
                genre = gr.Dropdown(genres, value=ALL, label="Genre", scale=2)
            with gr.Row():
                count = gr.Slider(3, 24, value=12, step=1, label="Number of results", scale=4)
                go = gr.Button("Search", variant="primary", scale=1)
            gr.Examples(
                [
                    ["a story about forgiveness", ALL],
                    ["space exploration and first contact", ALL],
                    ["", "Fantasy"],
                    ["detective solving a murder in a small town", "Mystery"],
                ],
                inputs=[query, genre],
            )
            results = gr.HTML()
            go.click(search, [query, genre, count], results)
            query.submit(search, [query, genre, count], results)

        with gr.Tab("✨ Recommend"):
            book = gr.Dropdown(
                title_choices,
                value=None,
                label="A book you liked (pick from the list or type a title)",
                allow_custom_value=True,
                filterable=True,
            )
            with gr.Row():
                rec_count = gr.Slider(3, 20, value=8, step=1, label="Number of books", scale=4)
                rec_go = gr.Button("Recommend", variant="primary", scale=1)
            rec_results = gr.HTML()
            rec_go.click(recommend, [book, rec_count], rec_results)

        with gr.Tab("ℹ️ About"):
            gr.Markdown(about_markdown())

        demo.load(render_popular, None, results)
    return demo


demo = build_demo()

# The website (FastAPI) owns "/", "/search", "/recommend", "/api/v1/...", "/docs", "/health/...";
# the Gradio screen lives at "/demo". Both share the engines loaded above.
app = create_app(settings, registry=registry)
app = gr.mount_gradio_app(
    app,
    demo,
    path="/demo",
    theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
    css=CSS,
    # Hugging Face turns on Gradio's server-side rendering (GRADIO_SSR_MODE), which starts a
    # separate Node.js server that grabs port 7860 - the port this website must listen on.
    # The website renders its own HTML, so Gradio's SSR is not needed: keep it off.
    ssr_mode=False,
)

if __name__ == "__main__":
    import uvicorn

    if report_zerogpu_startup():
        print("ZeroGPU startup report sent", flush=True)
    uvicorn.run(
        app,
        host=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),  # noqa: S104
        port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        proxy_headers=True,  # Hugging Face's proxy terminates HTTPS ...
        forwarded_allow_ips="*",  # ... so trust its X-Forwarded-* (https links, visitor IPs)
        log_config=None,  # keep BookMind's logging format
        access_log=False,
    )
