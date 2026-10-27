# BookMind Pro — Architecture

## 1. System overview

```mermaid
flowchart LR
    subgraph Offline["Offline pipeline (bookmind-pipeline)"]
        D["download<br/>kagglehub"] --> P["prepare"] --> C["classify<br/>Ollama LLM / keyword"] --> B["build-index<br/>MiniLM + FAISS"]
        D --> T["train-knn<br/>scikit-learn k-NN"] --> E["evaluate-knn<br/>Hit-Rate@K"]
    end
    B --> A[("artifacts/<br/>faiss index · parquet · manifest<br/>knn pickles · knn manifest")]
    T --> A
    A --> S["FastAPI service"]
    U["Browser / API client"] <--> S
```

## 2. Semantic pipeline (content-based)

```mermaid
flowchart TD
    R["data/books_raw.csv<br/>7k books with descriptions"] --> P["prepare<br/>strip HTML · validate · dedupe · stable book_id"]
    P --> C["classify<br/>1 genre from 20 (LLM, retries, resumable)"]
    C --> E["embed<br/>'title. title. Genre: g. description' → 384-dim vector"]
    E --> F["FAISS IndexFlatIP<br/>(cosine on normalised vectors)"]
    F --> M[("book_index.faiss · book_meta.parquet · manifest.json + SHA-256")]
```

## 3. Collaborative-filtering pipeline

```mermaid
flowchart TD
    BX["BX-Books.csv + BX-Book-Ratings.csv<br/>~1.15M ratings"] --> L["load & clean<br/>normalise ISBN · unescape HTML · validate ratings"]
    L --> J["join ratings with books"]
    J --> V["encode feedback<br/>binary: any interaction = 1<br/>(0 now only means 'never read')"]
    V --> G["group editions by title + author"]
    G --> F["filter: active readers, books with enough readers"]
    F --> PV["pivot: books × readers (sparse)"]
    PV --> K["NearestNeighbors (brute, cosine)"]
    K --> O[("4 pickles + knn_manifest.json<br/>(checksums, sklearn version)")]
    F --> EV["evaluate: 80/20 reader split<br/>Hit-Rate@5 vs popularity, coverage"]
```

## 4. Request lifecycle

```mermaid
sequenceDiagram
    participant U as User
    participant M as Middleware
    participant R as Route
    participant V as Validation
    participant H as Hybrid recommender
    participant KNN as k-NN engine
    participant SEM as Semantic engine

    U->>M: GET /api/v1/books/similar?title=...
    M->>M: request id · rate limit · timer
    M->>R: forward
    R->>V: clean title, check top_k
    R->>H: find_similar(title)
    H->>KNN: title known to ratings model?
    alt yes
        KNN-->>H: neighbours by reader overlap
    else no
        H->>SEM: stored vector → FAISS nearest neighbours
        SEM-->>H: neighbours by meaning
    end
    H-->>R: books + engine used
    R-->>M: JSON
    M->>M: access log · metrics · security headers
    M-->>U: 200 + X-Request-ID
```

## 5. Search, browse & autocomplete (discovery)

```mermaid
flowchart TD
    Q["GET /api/v1/search?q=&genre="] --> HQ{"q given?"}
    HQ -->|yes| CL["clean_text(q) · validate_genre · validate_top_k"]
    CL --> EN["encode q → 384-dim unit vector (LRU cached)"]
    EN --> FS["FAISS search<br/>(IDSelectorBatch = only books of that genre)"]
    FS --> RS["results with similarity score"]
    HQ -->|"no, genre only"| BR["engine.browse(genre)<br/>most-read books in genre, score = null"]
    HQ -->|"neither"| E422["422 validation_error"]

    A["GET /api/v1/autocomplete?q=&genre="] --> AQ{"q empty?"}
    AQ -->|yes| POP["popular_titles: interleave k-NN + semantic<br/>mode = popular"]
    AQ -->|no| PRE["prefix matches first, then substring<br/>merged from both engines"]
```

## 6. Launcher (`run.bat` / `python -m bookmind`)

```mermaid
flowchart TD
    RB["run.bat / run.sh"] --> VE{".venv exists?"}
    VE -->|no| ERR["print setup instructions"]
    VE -->|yes| MF{"artifacts/manifest.json?"}
    MF -->|no| BUILD["pipeline download + all --classifier keyword"]
    MF -->|yes| MAIN["python -m bookmind --open"]
    BUILD --> MAIN
    MAIN --> PF{"port free?"}
    PF -->|yes| UV["uvicorn on that port"]
    PF -->|"no, development"| NXT["find_free_port → next port, log it"] --> UV
    PF -->|"no, production"| FAIL["exit with error (exact port required)"]
    UV --> OPEN["open browser at http://localhost:PORT"]
```

## 7. Startup & resilience

```mermaid
flowchart TD
    S["start"] --> CFG["load & validate settings"]
    CFG --> REG["ServiceRegistry.load"]
    REG --> SE{"semantic artifacts valid?<br/>(files, checksums, dims)"}
    SE -->|yes| S1["semantic ready"]
    SE -->|no| S2["degraded: UI banner, API 503,<br/>/health/ready = unavailable"]
    REG --> KN{"k-NN artifacts?"}
    KN -->|valid| K1["k-NN ready"]
    KN -->|absent| K2["semantic handles 'similar'"]
    KN -->|corrupt| K3["/health/ready = degraded"]
```

## 8. Code layers

```mermaid
flowchart TD
    subgraph HTTP
        MW["middleware/"] --> API["api/v1"] & WEB["web/"] & H["api/health"]
    end
    subgraph Services
        REC["recommendation.py (hybrid)"] --> SS["semantic_search.py"] & KR["knn_recommender.py"]
        REG["registry.py"]
        ART["artifacts.py"]
        VAL["validation.py"]
    end
    subgraph Pipeline
        CLI["cli.py"] --> DL["download"] & PR["prepare"] & CL["classify"] & BI["build_index"] & KN["knn"]
    end
    subgraph Core
        CFG["config"] & EXC["exceptions"] & LOG["logging"]
    end
    API & WEB --> REC
    SS --> EMB["ml/embedder.py"]
    BI --> EMB
    SS --> ART
    KR --> ART
```

| Layer | Responsibility |
|---|---|
| Core | Settings, error types, logging used everywhere |
| ML | Text → vectors (swappable model) |
| Services | Business logic, independent of the web framework |
| HTTP | Translate requests ↔ service calls; cross-cutting concerns in middleware |
| Pipeline | Heavy offline work, kept out of the request path |

## 9. Module map

| Module | Key functions / classes | Purpose |
|---|---|---|
| `__main__.py` | `port_is_free`, `find_free_port`, `main` | One-command launcher |
| `main.py` | `create_app`, `lifespan` | App factory, middleware order, routers |
| `core/config.py` | `Settings`, `get_settings`, `GENRE_TAXONOMY` | Validated configuration |
| `core/exceptions.py` | `BookMindError` + 8 subclasses | Typed errors → HTTP status + code |
| `core/logging.py` | `setup_logging`, `JsonFormatter`, `request_id_ctx` | Structured logs with request IDs |
| `middleware/request_context.py` | `RequestContextMiddleware` | Request ID, access log, metrics, security headers |
| `middleware/rate_limit.py` | `RateLimitMiddleware` | Per-IP moving-window limit, 429 envelope |
| `api/v1/routes.py` | `search`, `similar`, `autocomplete`, `genres`, `stats` | JSON API |
| `api/health.py` | `live`, `ready`, `metrics` | Probes and Prometheus |
| `api/errors.py` | `error_response`, handlers | JSON or HTML error pages |
| `web/routes.py` | `home`, `search_page`, `recommend_*`, `about` | Server-rendered pages |
| `services/semantic_search.py` | `SemanticSearchEngine` (`search`, `similar`, `browse`, `popular`, `autocomplete`) | FAISS engine |
| `services/knn_recommender.py` | `KnnRecommender` (`recommend`, `popular`, `autocomplete`) | Collaborative filtering engine |
| `services/recommendation.py` | `find_similar`, `popular_titles`, `autocomplete`, `catalogue_stats` | Hybrid logic |
| `services/registry.py` | `ServiceRegistry` | Loads engines, degraded mode |
| `services/artifacts.py` | `save_artifacts`, `load_artifacts`, `sha256_file`, `atomic_write_bytes` | Versioned, checksummed files |
| `services/validation.py` | `clean_text`, `validate_top_k`, `validate_genre` | Input validation |
| `ml/embedder.py` | `SentenceTransformerEncoder`, `HashingEncoder`, `CachedEncoder`, `l2_normalize` | Text → vectors |
| `pipeline/*.py` | `download_datasets`, `prepare_dataframe`, `run_classify`, `build_index`, `train_knn`, `evaluate_knn` | Offline pipeline |
| `static/js/*.js` | `theme.js`, `main.js`, `autocomplete.js` | Theme, navigation, type-ahead |

A function-by-function walkthrough with the reasoning behind each design choice is in
[`EXPLAINER.html`](EXPLAINER.html).
