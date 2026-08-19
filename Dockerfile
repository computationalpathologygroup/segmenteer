FROM nvidia/cuda:12.9.2-base-ubuntu22.04 AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/app/.venv/
ENV UV_NO_DEV=1
ENV UV_PYTHON_INSTALL_DIR=/python
ENV UV_PYTHON_PREFERENCE=only-managed
RUN uv python install 3.12

RUN apt-get update && apt-get install --no-install-recommends --yes \
        ca-certificates \
        build-essential \
        git \
        software-properties-common \
    && add-apt-repository ppa:ubuntugis/ppa && apt-get update && apt-get install --no-install-recommends --yes \
        gdal-bin libgdal-dev

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --extra all
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra all
ENV PATH="/app/.venv/bin:$PATH"


FROM nvidia/cuda:12.9.2-base-ubuntu22.04

RUN apt-get update && apt-get install --no-install-recommends --yes \
        ca-certificates \
        git \
        software-properties-common \
    && add-apt-repository ppa:ubuntugis/ppa && apt-get update && apt-get install --no-install-recommends --yes \
        gdal-bin libgdal-dev

COPY --from=builder /python /python
COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

WORKDIR /app
