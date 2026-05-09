# Multi-stage build: builder installs deps + warms HF model cache; final
# stage is a slim non-root runtime. The bge-small embedding model is
# pre-downloaded into the image so the first /chat after deploy doesn't
# pay a network-fetch penalty.

FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml ./
COPY src ./src

# Install package + runtime deps. Hatchling reads from pyproject.
RUN pip install --upgrade pip \
 && pip install --no-cache-dir .

# Pre-warm the embedding model so the runtime image doesn't fetch from HF on first request.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"


# ---- runtime ----
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HF_HOME=/app/.cache/huggingface \
    LOG_LEVEL=INFO

# Non-root user for least-privilege.
RUN groupadd -r app && useradd -r -g app -d /app -s /sbin/nologin app

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cache/huggingface /app/.cache/huggingface
COPY src ./src
COPY data ./data

RUN chown -R app:app /app
USER app

EXPOSE 8000

# Single uvicorn worker — Render free tier is 512MB RAM and the embedding
# model + FAISS index are loaded once per process. Multiple workers would OOM.
CMD ["sh", "-c", "uvicorn shl_recommender.api.app:app --host 0.0.0.0 --port ${PORT}"]
