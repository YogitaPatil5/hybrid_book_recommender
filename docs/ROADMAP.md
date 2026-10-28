# Roadmap

The current version (v2.2.0) indexes 6,323 books for semantic search and 637 books for collaborative
filtering, and runs live at https://yogita4-bookmind.hf.space with CI/CD from GitHub.
The architecture lets the data source change without touching the web layer: every engine loads from
versioned artifacts (`manifest.json`, `knn_manifest.json`), and the pipeline's `prepare` step is the
only place that knows a dataset's column names.

## 0. Done in v2.2

- Hybrid recommender (semantic search + item-item k-NN with fallback), evaluated against a baseline.
- Production practices: validation, typed errors, structured logs, health checks, metrics, rate
  limiting, security headers, checksummed and version-pinned artifacts, degraded mode.
- 529 tests (~99% coverage), strict typing, Docker.
- CI on GitHub Actions (Python 3.10–3.13) and automatic deployment to Hugging Face after green CI.

## 1. Production hardening (before serving real customers)

| Item | Why | How |
|---|---|---|
| Staging environment + rollback | Test a release on a copy; undo a bad release in one step | A second Space (`bookmind-staging`) deployed from a `develop` branch; tag releases (`v2.3.0`) and redeploy a previous tag to roll back |
| Branch protection + pull requests | No unreviewed or untested code on `main` | GitHub → Settings → Branches: require PRs and passing CI |
| Alerting & dashboards | Problems are noticed, not discovered by users | Grafana on the existing Prometheus `/metrics`; alerts on error rate and latency |
| Load testing with targets | Know the capacity; catch slowdowns | Locust or k6 against `/api/v1/search`; define targets such as p95 latency |
| Reliable hosting | The free Space sleeps when idle and has no uptime guarantee | Paid hosting, ≥2 instances behind a load balancer, Redis for shared rate limits |
| Recommendation quality monitoring | Models degrade as data changes | Log clicks; track hit-rate over time; scheduled retraining |
| Browser UI tests in CI | Catch broken pages, not just broken functions | Playwright tests for search, recommend and theme toggle |
| Dependency & image scanning | Known vulnerabilities blocked, not just reported | Dependabot; make `pip-audit` blocking; Trivy on the Docker image |
| Self-hosted fonts/icons | No dependency on external CDNs | Bundle Inter and the icons under `static/` |

## 2. Search quality (small changes)

- **Search by author:** book vectors currently use title, genre and description. Adding the author to
  `build_text_for_embedding` (and rebuilding the index) makes author queries work.
- **Hybrid lexical + vector search** (BM25 + FAISS) for exact title/author matches.
- **Cross-encoder re-ranking** of the top results for higher precision.
- **Link the two catalogues by ISBN** so collaborative filtering can cover more of the 6.3k books.

## 3. Bigger static datasets (no redesign needed)

Candidate sources (check current licences and sizes before use):

| Dataset | What it adds |
|---|---|
| **Goodreads (UCSD Book Graph)** — millions of books and interactions | Descriptions, genres, shelves, reviews |
| **Amazon Reviews — Books** | Ratings + review text for both engines |
| **Open Library data dumps** | Titles, authors, subjects, covers |

What changes:
- `pipeline/prepare.py`: a column mapping for the new source (the output schema stays the same).
- **Embeddings in batches** with checkpointing (classification is already resumable).
- **FAISS index type:** `IndexFlatIP` (exact) is fine for this size; for millions of vectors switch to
  `IndexHNSWFlat` or `IndexIVFPQ` (approximate, much faster and smaller at scale).
- **Metadata store:** move from an in-memory parquet file to PostgreSQL or SQLite so each worker
  doesn't hold the whole catalogue in RAM.
- **Collaborative filtering:** replace brute-force k-NN with matrix factorisation (implicit ALS /
  LightFM) or an approximate-NN index over item vectors.

## 4. Live data (fetch books on demand)

Free public APIs: **Open Library** (`openlibrary.org/search.json`, no key) and **Google Books**
(`googleapis.com/books/v1/volumes`, free quota; an API key raises it). Check their current rate limits.

Recommended design — **local index first, live API as a fallback / enrichment**:

```mermaid
flowchart LR
    Q["User query"] --> L["Local FAISS index"]
    L -->|"good matches"| R["Results"]
    L -->|"few/no matches"| API["Open Library / Google Books"]
    API --> C["Cache (Redis / DB)"]
    C --> R
    C --> J["Scheduled job: embed new books<br/>and add them to the index"]
    J --> L
```

Why not call the API on every request:
- **Latency & reliability** — a network call is far slower than the in-memory index and can fail; the
  app must keep working without it (the same degraded-mode pattern used for the engines).
- **Rate limits** — public APIs throttle; a cache plus a background job stays within limits.
- **Semantic search needs vectors** — new books must be embedded before they're searchable by meaning,
  so a scheduled job adds them to the index rather than doing it mid-request.

Implementation sketch:
1. `services/live_catalog.py` — HTTP client with timeouts, retries with backoff and a circuit breaker,
   returning the same `Book` schema.
2. Cache responses (Redis is already in `docker-compose.yml`) with a TTL.
3. A `sync-live` pipeline command that embeds newly cached books and writes a new index version.
4. A settings flag (e.g. `BOOKMIND_LIVE_FALLBACK`) to switch it on per environment.

## 5. Product features

- User accounts and saved reading lists (PostgreSQL + authentication).
- Personalised "for you" feed from a user's own ratings (the collaborative model supports this).
- Feedback signals (clicks, likes) → offline evaluation and A/B tests.
- Kubernetes with horizontal autoscaling (the app is stateless with health/readiness probes).
