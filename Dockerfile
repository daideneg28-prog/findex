# findex web service — multi-stage build (https://docs.astral.sh/uv/guides/integration/docker/)
#
#   docker build -t findex .
#   docker run --rm -p 8000:8000 findex                        # then open http://localhost:8000
#   docker run --rm -p 8000:8000 -e LOG_LEVEL=DEBUG findex
#
# BOOKS controls how many Gutenberg books are baked in. 40 books (25 MB of text) give a
# positional index that needs ~310 MiB of RAM; with the embeddings and the model it still
# fits a 512 MB free-tier instance.

# ---- stage 1: dependencies, the package, and the index -----------------------------------
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0 HF_HOME=/app/hf
WORKDIR /app

# dependencies first: this layer is cached until uv.lock changes
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --extra semantic --no-install-project

COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --extra semantic --no-editable

# the corpus (public-domain books), its index and its embeddings are built into the image;
# the embedding model (~30 MB) is downloaded here once and served offline at run time
ARG BOOKS=40
COPY scripts/download_corpus.py scripts/
RUN .venv/bin/python scripts/download_corpus.py --count ${BOOKS} \
 && .venv/bin/findex index data/gutenberg --out data/index.pkl --executor serial \
 && .venv/bin/findex embed data/index.pkl

# ---- stage 2: slim runtime, no build tools, non-root ----------------------------------
FROM python:3.14-slim-bookworm
RUN useradd --create-home --uid 1000 findex
WORKDIR /app
COPY --from=builder --chown=findex:findex /app/.venv /app/.venv
COPY --from=builder --chown=findex:findex /app/data /app/data
COPY --from=builder --chown=findex:findex /app/hf /app/hf
USER findex

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    INDEX_PATH=/app/data/index.pkl \
    HF_HOME=/app/hf \
    HF_HUB_OFFLINE=1 \
    LOG_LEVEL=INFO \
    PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=2)"

# shell form so $PORT (set by the platform) is expanded; exec so uvicorn gets SIGTERM
CMD exec uvicorn findex.web.app:app --host 0.0.0.0 --port "$PORT" --proxy-headers --forwarded-allow-ips "*"
