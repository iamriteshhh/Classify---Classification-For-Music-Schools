# ==============================================================================
# CLASSIFY — Song Classification System for Music Schools
# Multi-stage Production Dockerfile
# Base: Python 3.11-slim with libsndfile1 and ffmpeg
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build stage (compile native C/C++ audio wheels & dependencies)
# ------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# Install system build dependencies for audio libraries (librosa, soundfile, scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    python3-dev \
    libsndfile1-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install python dependencies into virtualenv
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------------
# Stage 2: Final runtime image (lean, secure non-root runner)
# ------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runner

WORKDIR /app

# Install runtime shared libraries required for audio decoding & health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=wsgi:app
ENV PORT=5000

# Create dedicated non-root application user and instance directories
RUN useradd -m -u 1000 classify && \
    mkdir -p /app/uploads /app/instance /app/model_artifacts && \
    chown -R classify:classify /app

# Copy application source code and artifacts
COPY --chown=classify:classify app.py config.py wsgi.py ./
COPY --chown=classify:classify classify ./classify
COPY --chown=classify:classify classifier ./classifier
COPY --chown=classify:classify model_artifacts ./model_artifacts
COPY --chown=classify:classify static ./static
COPY --chown=classify:classify templates ./templates
COPY --chown=classify:classify datasets ./datasets

# Switch to non-privileged user for security compliance
USER classify

EXPOSE 5000

# Container liveness health check using /healthz
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/healthz || exit 1

# Production WSGI server via Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--timeout", "120", "wsgi:app"]
