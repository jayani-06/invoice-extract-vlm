FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps needed by Pillow / OCR-adjacent tooling later
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast resolver/installer)
RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
COPY src ./src

# Editable install, dev extras included; VLM/heavy extras are opt-in (see [project.optional-dependencies].vlm)
RUN uv pip install --system -e ".[dev]"

COPY . .

CMD ["bash"]
