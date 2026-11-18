# syntax=docker/dockerfile:1.7
# One image for the `api`, `worker` and `migrate` Compose services (only the command differs).
# Docling and FastEmbed models are baked in at build time; containers run with HF_HUB_OFFLINE=1.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

# ---- builder: third-party packages first, so source edits do not invalidate that layer ----
FROM ${PYTHON_IMAGE} AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_NO_DEV=1
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-install-project
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-editable

# ---- models: download Docling (~505 MB) and FastEmbed BM25 files; the download is build-cached ----
FROM builder AS models
COPY tests/fixtures/example_10k_extract.pdf /tmp/warm.pdf
COPY infra/docker/warm_models.py /tmp/warm_models.py
RUN --mount=type=cache,target=/hf-cache \
    mkdir -p /hf-cache/hf /hf-cache/fastembed /opt/models \
    && HF_HOME=/hf-cache/hf FINRAG_FASTEMBED_CACHE_DIR=/hf-cache/fastembed \
       /app/.venv/bin/python /tmp/warm_models.py /tmp/warm.pdf \
    && cp -r /hf-cache/hf /opt/models/hf \
    && cp -r /hf-cache/fastembed /opt/models/fastembed

# ---- runtime ----
FROM ${PYTHON_IMAGE} AS runtime
# libgomp: torch/onnxruntime OpenMP; libglib2.0: OpenCV (Docling); curl: container healthchecks
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 1000 --create-home --shell /usr/sbin/nologin finrag
WORKDIR /app
COPY --from=builder --chown=finrag:finrag /app/.venv /app/.venv
COPY --from=models --chown=finrag:finrag /opt/models /opt/models
COPY --chown=finrag:finrag alembic.ini ./
COPY --chown=finrag:finrag src/finrag/storage/migrations ./src/finrag/storage/migrations
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/models/hf \
    HF_HUB_OFFLINE=1 \
    FINRAG_FASTEMBED_CACHE_DIR=/opt/models/fastembed
USER finrag
EXPOSE 8000
CMD ["uvicorn", "finrag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
