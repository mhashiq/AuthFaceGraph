# ================================================================
# AuthBrain AI Face Analysis Engine — Backend Dockerfile
# Multi-stage: development (hot-reload) + production (optimized)
# ================================================================

# ── Base image ──────────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependencies ────────────────────────────────────────────────
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Development stage ───────────────────────────────────────────
FROM deps AS development

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]

# ── Production stage ─────────────────────────────────────────────
FROM deps AS production

# Create non-root user for security
RUN groupadd -r authbrain && useradd -r -g authbrain authbrain

COPY --chown=authbrain:authbrain . .

USER authbrain

EXPOSE 8000

# Production: multiple workers, no reload, optimized logging
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--loop", "uvloop", \
     "--log-level", "info"]
