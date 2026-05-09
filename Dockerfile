# Multi-stage build: builder installs deps; final stage is a slim
# non-root runtime. No local ML model: catalog embeddings are
# pre-computed offline (data/catalog_embeddings.npy) and query
# embeddings are fetched from OpenAI at request time. That keeps the
# runtime image to ~200 MB and the cold-start RSS to ~120 MB,
# comfortably under Render's 512 MB free-tier ceiling.

FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Hatchling reads pyproject.toml and follows ``readme = "README.md"``,
# so the README must be inside the build context or metadata generation
# fails before any code is touched.
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip \
 && pip install --no-cache-dir .


# ---- runtime ----
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    LOG_LEVEL=INFO

# Non-root user for least-privilege.
RUN groupadd -r app && useradd -r -g app -d /app -s /sbin/nologin app

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src ./src
COPY data ./data

RUN chown -R app:app /app
USER app

EXPOSE 8000

# Single uvicorn worker. Render free tier is 512 MB RAM; multiple
# workers would each load FAISS + the catalog into memory.
CMD ["sh", "-c", "uvicorn shl_recommender.api.app:app --host 0.0.0.0 --port ${PORT}"]
