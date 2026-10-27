# syntax=docker/dockerfile:1.7
# ─────────────────────────────────────────────────────────────────────────────
# BookMind Pro — production image (multi-stage, non-root, CPU-only torch)
#
#   docker build -t bookmind:latest .
#   docker build --build-arg PRELOAD_MODEL=true -t bookmind:latest .   # bake model in
# ─────────────────────────────────────────────────────────────────────────────

ARG PYTHON_VERSION=3.12

# ── Stage 1: build wheels / venv ─────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only PyTorch first: avoids pulling ~2 GB of CUDA wheels.
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch

WORKDIR /build
COPY requirements.txt requirements-pipeline.txt ./
# Pipeline extras (ollama client, tqdm) are tiny; include them so the same
# image can run `bookmind-pipeline`.
RUN pip install -r requirements-pipeline.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

ARG PRELOAD_MODEL=false
ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
ENV HF_HOME=/opt/hf-cache
RUN mkdir -p /opt/hf-cache && if [ "$PRELOAD_MODEL" = "true" ]; then \
      python -c "from sentence_transformers import SentenceTransformer as S; S('${EMBEDDING_MODEL}')"; \
    fi

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="bookmind-pro" \
      org.opencontainers.image.description="Semantic book search & recommendation service"

# libgomp is required by faiss / torch OpenMP kernels.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd --system app && useradd --system --gid app --home /app app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=app:app /opt/hf-cache /opt/hf-cache

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf-cache \
    BOOKMIND_ENV=production \
    BOOKMIND_LOG_JSON=true \
    BOOKMIND_LOG_TO_FILE=false \
    BOOKMIND_DATA_DIR=/app/data \
    BOOKMIND_ARTIFACTS_DIR=/app/artifacts \
    BOOKMIND_LOG_DIR=/app/logs

WORKDIR /app
COPY --chown=app:app gunicorn.conf.py ./
RUN mkdir -p data artifacts logs && chown -R app:app /app

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=4).status == 200 else 1)"

CMD ["gunicorn", "-c", "gunicorn.conf.py", "bookmind.main:create_app()"]
