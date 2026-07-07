# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=nvidia/cuda:12.6.0-runtime-ubuntu24.04
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
ARG TORCH_VERSION=2.12.0
ARG TORCHVISION_VERSION=0.27.0

FROM ${BASE_IMAGE}

ARG TORCH_INDEX_URL
ARG TORCH_VERSION
ARG TORCHVISION_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl ca-certificates \
        libgl1 libglib2.0-0 libgoogle-perftools4 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get install -y --no-install-recommends \
        python3.13 python3.13-venv python3-pip \
    && ln -sf /usr/bin/python3.13 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.13 /usr/local/bin/python3

COPY pyproject.toml poetry.lock ./

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --break-system-packages poetry \
 && poetry install --only main --no-root \
 && pip install --break-system-packages \
        --index-url "${TORCH_INDEX_URL}" --force-reinstall \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

COPY src/ ./src/

EXPOSE 7860

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860"]
