# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.9.2-base-ubuntu22.04 AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_NO_DEV=1 \
    UV_PYTHON_INSTALL_DIR=/python \
    UV_PYTHON_PREFERENCE=only-managed \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates build-essential git pkg-config \
        openjdk-17-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

RUN uv python install 3.12

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-project --extra all

COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra all --no-editable

FROM nvidia/cuda:12.9.2-base-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git libopenslide0 libvips42 \
        openjdk-17-jre-headless \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /python /python
COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    SEGMENTEER_MODELS_DIR=/weights \
    TORCH_HOME=/weights/torch \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

WORKDIR /app
RUN python -c "import segmenteer, geojson, rasterio, openslide, cv2, torch; print('Segmenteer runtime imports OK')"
