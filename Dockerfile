# Neuron base image for MVP Phase 0
# Supports CPU by default; override TORCH_URL for GPU builds.

FROM python:3.11-slim AS base


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WERNICKE_QUANTIZE=1 \
    BROCA_QUANTIZE=1 \
    TORCH_COMPILE=0 \
    OTEL_SAMPLE_RATIO=0 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    HF_HUB_DISABLE_PROGRESS=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps early for cache efficiency
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source
COPY . .

# Default command (placeholder until api/main.py exists)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
