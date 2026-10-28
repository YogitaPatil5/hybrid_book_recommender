# Code review of the original project & changes made

## Issues found in the original code

| # | Severity | Where | Problem | Fix |
|---|---|---|---|---|
| 1 | Critical | `app.py` | `app.run(debug=True, host="0.0.0.0")` exposes the Werkzeug debugger (remote code execution) to the network | Debug is off by default and **forbidden** when `BOOKMIND_ENV=production`; served by gunicorn/uvicorn |
| 2 | High | `app.py` | Search engine is built at import time — a missing index **crashes the whole app** | Engines load in the lifespan via `ServiceRegistry`; app starts in degraded mode, `/health/ready` returns 503 |
| 3 | High | `build_index.py` / `semantic_search.py` | Metadata stored with `pickle` (arbitrary code execution on load); no link between index and metadata versions | Parquet + `manifest.json` with SHA-256 checksums, row counts, dim and model name — validated on load |
| 4 | High | `semantic_search.py` | Query encoder not tied to the encoder that built the index → silent garbage results if the model changes | Encoder chosen from the manifest; dim mismatch is rejected |
| 5 | High | `classify_genres.py` | Resume keyed by **title** → different books with the same title are silently skipped | Stable `book_id` (ISBN-13 or content hash), resume keyed by it, atomic checkpoints, Ctrl+C saves progress |
| 6 | Medium | `semantic_search.py` | Genre filter over-fetched `top_k*5` then filtered → too few or zero results for small genres | Exact filtering inside FAISS with `IDSelectorBatch` |
| 7 | Medium | `classify_genres.py` | Substring match of LLM output (`"Fiction"` matched first genre in list order) | Exact → prefix → longest whole-word match; unknown → `Other` |
| 8 | Medium | everywhere | NaN ratings/authors reach JSON (invalid JSON) and templates (`"nan"`) | All values sanitised into typed Pydantic models |
| 9 | Medium | `app.py` | No input validation (unbounded query length, any `top_k`, any genre) | Two-layer validation (FastAPI `Query` bounds + engine limits from settings) |
| 10 | Medium | templates | `onclick="prefillSearch('{{ title }}')"` — injection risk; inline JS blocks a strict CSP | All JS moved to static files; links use `urlencode`; CSP `script-src 'self'` |
| 11 | Medium | templates / JS | `via.placeholder.com` has shut down → broken images; `onerror` loop | Local `no-cover.svg`; one-shot error handler |
| 12 | Low | `index.html` | Hard-coded stats (742 books, 1,174 readers) | Live stats from the engines |
| 13 | Low | `app.py` | `get_author`/`get_image` scan the whole ratings frame per call (O(n)) | Precomputed dictionaries |
| 14 | Low | `recommend.html` | Auto-submits the form on every page load with `?q=` | `GET /recommend?q=` renders results server-side |
| 15 | Low | `app.py` | `JSON_SORT_KEYS` is deprecated in Flask 3 | n/a (moved to FastAPI) |
| 16 | Low | `prepare_data.py` | No cleaning: HTML in descriptions, invalid ratings/years, duplicates, `javascript:` thumbnails | Full cleaning & validation step with a report |
| 17 | Low | `classify_genres.py` | Ollama errors silently become `"Other"`; no server/model check | Fail-fast availability check, retries with exponential backoff, keyword fallback, failure count |
| 18 | Low | — | No tests, no logging structure, no health checks, no rate limiting | 529 tests, structured logging, probes, metrics, rate limiting |

## Why FastAPI instead of Flask

- Request/response validation and OpenAPI docs come from the same Pydantic models.
- Async server (uvicorn) with CPU-bound endpoints automatically run in a threadpool.
- Dependency injection makes the engines trivially swappable in tests.
- It is the de-facto standard for serving ML models in Python.

The original page structure (Home, Search, Recommend, About) was kept, then redesigned: a professional
slate + indigo palette with light **and** dark themes (see *Later improvements* below).

---

## Review of the original k-NN project (`preprocess.py`, `train_model.py`, `research.ipynb`)

| # | Severity | Problem | Fix in `bookmind-pipeline train-knn` |
|---|---|---|---|
| 1 | High (ML) | Book-Crossing rating `0` = "interacted, no score". The original kept those 0s **and** filled missing cells with 0, so "read it" and "never saw it" looked identical | Interactions encoded explicitly (`binary` default: any interaction = 1; `weighted`, `explicit` available). 0 in the matrix now only means "no interaction". Default chosen by measured Hit-Rate@5 |
| 2 | Medium | Notebook used Euclidean distance (default), `train_model.py` used cosine — results differed | Cosine everywhere |
| 3 | Medium | Books grouped by **title only** — different books sharing a title (e.g. "Selected Poems", 27×) were merged | Grouped by normalised title + author; editions merge, namesakes are disambiguated |
| 4 | Medium | Active users counted **before** joining with the book catalogue, so ratings of unknown ISBNs inflated activity | Counted after the join |
| 5 | Medium | No evaluation of recommendation quality | `evaluate-knn`: user-level hold-out, Hit-Rate@K, popularity baseline, catalogue coverage, JSON report |
| 6 | Low | README said `../data`, code read `./data`; notebook saved to `artifects/` (typo → crash) | One configurable data dir; `download` places files; atomic artifact writes |
| 7 | Low | `BX-Users.csv` loaded but unused; HTML entities (`&amp;`) left in titles/publishers; invalid years kept | Users file not needed; text unescaped; years validated |
| 8 | Low | Pickles loaded with no integrity check; sklearn version drift breaks pickles silently | `knn_manifest.json` with SHA-256 checksums + sklearn version; verified on load |
| 9 | Low | Average rating on the home page included implicit 0s and used a /10 scale | Explicit ratings only, shown on the same /5 scale as the rest of the UI |

---

## Later improvements (after the first working version)

| Area | Before | After | Why |
|---|---|---|---|
| Genre browsing | The search box was `required`; choosing only a genre showed "Please fill out this field" | `q` is optional when a genre is chosen → the most-read books in that genre (`SemanticSearchEngine.browse`) | Users often know the genre, not a title |
| Discovery | Suggestions appeared only after typing a letter | Focusing the empty box opens a **Popular books** dropdown (genre-aware); suggestions from the first letter | Users who don't know what to search for still get a starting point |
| Launch | Needed `HF_HUB_OFFLINE=1` and a manual port change | `run.bat` / `run.sh` / `python -m bookmind --open`: model loads offline-first, a busy port falls back to the next free one, the index is built automatically if missing | One command for a new user |
| UI theme | Single dark theme, Font Awesome toggle icon | Light + dark themes via CSS custom properties, OS preference by default, saved choice in `localStorage`, applied before first paint (`theme.js`), inline-SVG icons | Accessibility (WCAG-AA contrast), user choice, no flash, works without the icon CDN |
| Images | `http://` covers blocked by the Content-Security-Policy | Cover URLs upgraded to `https://` in `prepare`, the semantic engine and k-NN | Keeps the strict CSP and still shows covers |
| Logs | Noisy third-party HTTP logs and a `FutureWarning` from sentence-transformers | Noisy loggers silenced; both old and new embedding-dimension APIs supported | Clean, readable console |
| k-NN | Separate script from the original project | Integrated `train-knn` / `evaluate-knn` with three encodings, manifest + checksums, hybrid fallback | One reproducible pipeline, measured choice of encoding |
| Windows | `pytest.exe` / `bookmind-pipeline.exe` blocked by Device Guard | All docs use `python -m pytest` and `python -m bookmind.pipeline.cli` | Works on locked-down college PCs |
