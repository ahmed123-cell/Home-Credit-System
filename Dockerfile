# Multi-stage Dockerfile for Home Credit Default Risk System
# Fully cross-platform: works on Linux, macOS (Apple Silicon / Intel), and Windows (WSL2 / Docker Desktop)

# ============================================================================
# Stage 1: Build & Dependency Resolution
# ============================================================================
FROM python:3.12-slim-bookworm AS builder

# Copy official uv binary (multi-architecture: amd64 / arm64)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Copy dependency manifests first for optimal Docker layer caching
COPY pyproject.toml uv.lock ./

# Install all project dependencies into /app/.venv
RUN uv sync --frozen --no-install-project

# ============================================================================
# Stage 2: Final Production Runtime Image
# ============================================================================
FROM python:3.12-slim-bookworm AS runtime

# Install system runtime dependencies:
# - libgomp1: required by LightGBM and XGBoost for OpenMP threading
# - curl: required for container health checks
# - ca-certificates: secure outbound HTTPS connections
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary into runtime to allow CLI operations (e.g. running dbt or training)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Create a non-privileged system user for secure container execution
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser \
    && mkdir -p /tmp/.cache/uv /app \
    && chown -R appuser:appuser /tmp/.cache/uv /app

# Copy the pre-built virtual environment from builder stage with proper ownership
COPY --chown=appuser:appuser --from=builder /app/.venv /app/.venv

# Set runtime environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app:/app/src" \
    MODELS_DIR="/app/models" \
    PORT=8000 \
    UV_CACHE_DIR="/tmp/.cache/uv"

# Copy application code, configs, models, tests, and UI
COPY --chown=appuser:appuser pyproject.toml README.md ./
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser app.py UI.html ./
COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser dbt/ ./dbt/
COPY --chown=appuser:appuser tests/ ./tests/

# Switch to non-root user
USER appuser

# Expose API and UI port
EXPOSE 8000

# Health check to verify FastAPI inference service is responsive
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: launch the FastAPI inference server with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
