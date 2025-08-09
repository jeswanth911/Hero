# syntax=docker/dockerfile:1

###############################
# --- Builder Stage (optional, for wheels caching) ---
###############################
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install build utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# Install pip-tools for deterministic builds (optional, can be removed)
RUN pip install --upgrade pip pip-tools

# Copy requirements files first to leverage Docker layer caching
COPY requirements.txt .

# Download and cache dependencies as wheels
RUN pip wheel --wheel-dir /wheels -r requirements.txt

###############################
# --- Final Stage ---
###############################
FROM python:3.11-slim

# Set a non-root user for security
RUN useradd -m fastapiuser

# Set working directory
WORKDIR /app

# Install system dependencies (for example libpq-dev for postgres, can add others as needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy environment file
COPY .env .env

# Copy requirements.txt and wheels from builder
COPY requirements.txt .
COPY --from=builder /wheels /wheels

# Install Python dependencies from wheels for speed
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --find-links=/wheels -r requirements.txt

# Copy rest of the application code
COPY . .

# Use a non-root user
USER fastapiuser

# Set environment variables from .env automatically (for uvicorn/gunicorn)
ENV PYTHONUNBUFFERED=1
# (Optional: you may want to source .env in entrypoint for more complex envs.)

# Expose FastAPI default port
EXPOSE 8000

# Healthcheck endpoint (expects /health route in FastAPI app)
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD curl -f http://localhost:8000/health || exit 1

# Start with Gunicorn + Uvicorn workers for production-grade serving
CMD ["gunicorn", "api.routes:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--timeout", "180"]
     
     
