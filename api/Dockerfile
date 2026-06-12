# Employee Handbook API — multi-stage build.
# Stage 1 installs the (heavy) ML/retrieval stack into a venv; stage 2 copies
# just the venv + code, runs as a non-root user, and keeps model caches on a
# mounted volume so the cross-encoder/embedding models aren't re-downloaded.

# ---- builder ----
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- runtime ----
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    # Keep model downloads on the mounted volume.
    HF_HOME=/model_cache \
    SENTENCE_TRANSFORMERS_HOME=/model_cache
COPY --from=builder /opt/venv /opt/venv

# Non-root user; pre-create cache/data dirs it can write to.
RUN useradd -m -u 1000 appuser && mkdir -p /model_cache /app && chown -R appuser /model_cache /app
WORKDIR /app

# Copy only what the API needs at runtime.
COPY ingestion/ ingestion/
COPY retrieval/ retrieval/
COPY agents/ agents/
COPY api/ api/
COPY observability.py ./

USER appuser
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
