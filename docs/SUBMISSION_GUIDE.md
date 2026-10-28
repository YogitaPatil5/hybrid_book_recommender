# Submission & Demo Guide

| | |
|---|---|
| **Live app** | https://yogita4-bookmind.hf.space |
| **Source code, CI/CD runs** | https://github.com/YogitaPatil5/hybrid_book_recommender (see the **Actions** tab) |
| **Learner's guide** | `docs/EXPLAINER.html` (open in a browser) |

## A. What to submit

1. **The two links above** — in your report, slides and submission form. The GitHub repository
   contains everything: code, tests, docs, built artifacts and the CI/CD history.
2. *(If a file is required)* a ZIP of the repository: on GitHub **Code → Download ZIP**. It already
   excludes `.venv/`, `data/`, `logs/`, `htmlcov/`, caches and `.env`.
3. Screenshots for the report: home page (light and dark), a search result, a recommendation,
   `/docs`, the green **Actions** runs, and `htmlcov/index.html` (coverage).

## B. Night-before checklist (≈ 30 minutes)

| ✔ | Step | Command / action | Expected |
|---|---|---|---|
| ☐ | Tests | `python -m pytest` | **529 passed** |
| ☐ | Lint & types | `python -m ruff check src tests scripts deploy` · `python -m mypy` | no issues |
| ☐ | Coverage report | `python -m pytest --cov --cov-report=html` | open `htmlcov/index.html` (~99%) |
| ☐ | Local app | `run.bat` | browser opens; `/health/ready` → semantic and knn ready |
| ☐ | Live app | open the live link | home page loads (wait ~1 min if it was asleep) |
| ☐ | CI/CD | GitHub → **Actions** | latest **CI** ✅ and **Deploy** ✅ |
| ☐ | Evaluation numbers | `python -m bookmind.pipeline.cli evaluate-knn --compare-modes` *(optional, needs the datasets)* | Hit-Rate@5 / coverage table (section D) |

> Open the live link a few minutes **before** the demo so it is awake. Keep `run.bat` ready as a
> backup in case the internet is unreliable — the local app works fully offline.

## C. 6-minute demo script

1. **Problem (30 s)** — "Keyword search can't find books by meaning, and pure popularity shows everyone
   the same bestsellers. BookMind combines content-based and collaborative recommendations."
2. **Search by meaning (60 s)** — live app → **Search**: click the empty box → *Popular books*
   dropdown. Type "a story about forgiveness" → results match meaning, not words.
   Clear the box, pick only a genre → most popular books in that genre. Toggle light/dark.
3. **Hybrid recommendations (60 s)** — **Recommend** → start typing *Harry Potter* and **pick a title from the
   dropdown** (exact titles from the ratings data):
   "Readers who rated this book also loved these" (collaborative filtering). Then a title that is only
   in the 7k catalogue → "Books with the most similar themes and content" (semantic fallback). **About** explains both.
4. **Engineering (60 s)** — `/docs` (try `/api/v1/search`), `/health/ready`. Call
   `/api/v1/search?q=test&top_k=1000` to show validation and the error format with `request_id`.
5. **Quality & CI/CD (90 s)** — GitHub **Actions**: CI runs lint, strict typing and 529 tests on four
   Python versions plus a Docker build; **Deploy** runs only after CI is green and updates the live
   site. Mention a real catch: CI failed on GitHub while passing locally (an environment variable
   difference) — fixed before anything reached users.
6. **Evaluation & wrap-up (60 s)** — the Hit-Rate/coverage table and why coverage matters; no paid
   APIs; reproducible from two public datasets; live and automatically deployed.

## D. Key numbers

| | |
|---|---|
| Catalogue (semantic search) | 6,323 books, 20 genres, 384-dim MiniLM vectors, FAISS exact cosine search |
| Collaborative filtering | 637 books × 806 readers, 52,615 interactions (from 1,149,725 valid Book-Crossing ratings) |
| k-NN `binary` (default) | Hit-Rate@5 **0.645**, catalogue coverage **80.7%** |
| k-NN `weighted` | Hit-Rate@5 0.571, coverage 90.4% |
| Popularity baseline | Hit-Rate@5 0.688, coverage **0.9%** |
| Tests | 529 (17 unit + 5 integration files), ~99% coverage, CI on Python 3.10–3.13 |

## E. Likely viva questions

**Q: Does it use an LLM or k-NN?**
Both, in different roles. Search uses sentence embeddings + FAISS (a nearest-neighbour search over
meaning). Collaborative filtering is an item-item k-NN over reader ratings. An LLM can tag genres
offline (Ollama); a keyword classifier does it without one. No LLM runs at request time.

**Q: Why cosine similarity?**
It compares direction, not magnitude. For text vectors direction carries meaning; for ratings it
compares *who* read a book rather than *how many*, so a bestseller and a niche book can still be close.
Vectors are L2-normalised so the inner product FAISS computes equals cosine.

**Q: Why is popularity's hit-rate higher than k-NN's?**
Book-Crossing is dominated by a few bestsellers, so recommending them "hits" often — but it shows the
same 5 books to everyone (0.9% coverage) and can't answer "similar to this book". k-NN reaches 64.5%
with 80.7% coverage; that's why both metrics are reported.

**Q: What was wrong with the original k-NN?**
Book-Crossing uses 0 for "interacted without a score"; the original kept those zeros and also used 0 for
"never read", so the two were indistinguishable. Interactions are now encoded explicitly (binary chosen
by measurement), and editions are grouped by title + author.

**Q: How do you avoid data leakage?**
Readers are split 80/20 *before* training; the model never sees test readers' interactions.

**Q: What does "hybrid" mean here?**
`find_similar` tries collaborative filtering first (behavioural signal). If the title isn't in the
ratings model it falls back to semantic similarity over the full catalogue, so every book — including
new ones with no ratings — gets recommendations (no cold-start gap).

**Q: What makes it production-grade?**
Layered architecture; validation at every boundary; typed errors with stable codes; structured logs
with request IDs; health/readiness probes; Prometheus metrics; rate limiting; security headers;
checksummed, version-pinned artifacts; degraded mode instead of crashing; ~99% coverage with strict
typing; Docker; CI/CD where deployment is gated on green tests.

**Q: How does quality assurance work?**
In layers: unit, integration and end-to-end tests; coverage gate (85%); ruff and strict mypy; model
evaluation against a baseline; data validation with checksums; all run in CI on four Python versions;
deploy only after CI passes; health checks and metrics after release.

**Q: Explain your CI/CD pipeline.**
Every push runs CI on GitHub Actions. When CI succeeds on `main`, the Deploy workflow checks out that
exact commit, builds the Space package with `prepare_hf_space.py` and uploads it using an encrypted
`HF_TOKEN` secret; Hugging Face rebuilds the Space. A failed check stops the release.

**Q: What problems did you hit while deploying?**
Free Hugging Face accounts can't run Docker, so the Space runs the FastAPI site from a Gradio Space;
ZeroGPU required a startup report and kept a GPU the model must not use; Gradio's server-side rendering
took port 7860; the Space description exceeded 60 characters; and CI failed on GitHub because of an
environment variable. Each was diagnosed from logs and fixed — most now have automated checks
(see `docs/REVIEW_AND_CHANGES.md`).

**Q: Is it ready for real customers?**
The engineering practices are in place; operating it for customers would add a staging environment and
rollback, alerting on the metrics, load testing with latency targets, paid hosting with an uptime
guarantee (the free Space sleeps), and monitoring of recommendation quality.

**Q: How would it scale?**
FAISS supports approximate indexes (HNSW/IVF) for millions of vectors; workers are stateless so they
scale horizontally behind a load balancer; Redis shares rate limits; collaborative filtering can move to
matrix factorisation for larger rating sets (see `docs/ROADMAP.md`).

**Q: Why is the LLM not used at request time?**
Latency (seconds), cost or GPU needs, and possible hallucination. Genres change rarely, so tagging once
offline gives the benefit for free; embeddings + FAISS answer in milliseconds, deterministically.

**Q: How does the theme avoid a flash on load?**
A small script in `<head>` (`theme.js`) sets `data-theme` from `localStorage` or the OS preference
before the page paints; every colour is a CSS custom property redefined per theme.

**Q: Limitations / future work?**
Collaborative filtering covers only the 637 Book-Crossing titles (the datasets don't share IDs); search
is by title, genre and description, not author; next steps include ISBN linking across catalogues,
adding authors to the search text, a cross-encoder re-ranker, hybrid BM25 + vector search, and A/B
testing recommendation clicks.
