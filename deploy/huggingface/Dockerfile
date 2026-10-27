# ─────────────────────────────────────────────────────────────────────────────
# BookMind Pro — Hugging Face Spaces image (Docker SDK)
#
# Built by Hugging Face from the folder that scripts/prepare_hf_space.py creates.
# Differences from the main Dockerfile:
#   * port 7860 (the Spaces default) and a user with UID 1000 (required by Spaces)
#   * the embedding model is baked into the image → starts fully offline
#   * the pre-built artifacts/ folder is copied in (no pipeline run on the server)
#   * /docs is kept visible for demos (BOOKMIND_ENABLE_DOCS=true)
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# libgomp is required by faiss / torch OpenMP kernels.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 user

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf-cache

WORKDIR /app

# CPU-only PyTorch first: avoids ~2 GB of CUDA wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

# Bake the embedding model into the image (no download at startup).
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('all-MiniLM-L6-v2')"

COPY gunicorn.conf.py ./
COPY artifacts ./artifacts
RUN mkdir -p data logs && chown -R user:user /app

USER user

ENV HF_HUB_OFFLINE=1 \
    BOOKMIND_ENV=production \
    BOOKMIND_ENABLE_DOCS=true \
    BOOKMIND_HOST=0.0.0.0 \
    BOOKMIND_PORT=7860 \
    BOOKMIND_WORKERS=2 \
    BOOKMIND_LOG_JSON=false \
    BOOKMIND_LOG_TO_FILE=false \
    BOOKMIND_ARTIFACTS_DIR=/app/artifacts \
    BOOKMIND_DATA_DIR=/app/data \
    BOOKMIND_LOG_DIR=/app/logs \
    FORWARDED_ALLOW_IPS=*

EXPOSE 7860
CMD ["gunicorn", "-c", "gunicorn.conf.py", "bookmind.main:create_app()"]
