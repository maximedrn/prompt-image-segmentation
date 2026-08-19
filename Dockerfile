# syntax=docker/dockerfile:1.7

# `python:*-slim` is a glibc image on purpose. PyTorch publishes no
# musllinux wheels on PyPI or on any of its CUDA/ROCm indexes, so an
# Alpine base would mean building torch from source - hours of build
# time and a toolchain layer that undoes the size saving it was chosen
# for. The savings come from the multi-stage split below instead.
#
# The torch CUDA/ROCm wheels vendor their own accelerator libraries as
# pip dependencies, so this stays valid for every profile. Swap in
# `nvidia/cuda:13.3.0-runtime-ubuntu24.04` if a host ever needs the
# distribution copies as well.
ARG BASE_IMAGE=python:3.13-slim
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG TORCH_VERSION=2.13.0
ARG TORCHVISION_VERSION=0.28.0

FROM ${BASE_IMAGE} AS build

ARG TORCH_INDEX_URL
ARG TORCH_VERSION
ARG TORCHVISION_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_DEFAULT_TIMEOUT=600 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_KEYRING_ENABLED=false \
    POETRY_REQUESTS_TIMEOUT=600 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /build

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && python -m venv "${VIRTUAL_ENV}"

COPY pyproject.toml poetry.lock ./

# Torch is installed from its own index in the same layer, so the CPU
# wheels poetry resolves never reach a committed layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip poetry \
    && poetry install --only main --no-root \
    && pip install --index-url "${TORCH_INDEX_URL}" --force-reinstall \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

COPY src/ /build/src/

# Compile the application package to a native extension module. Only the
# resulting .so reaches the runtime stage, so no source of ours is
# interpreted - or shipped - in production. Dependencies stay as wheels:
# Nuitka's standalone mode has a long history of breaking on
# torch + transformers, whose PyTorch detection is metadata-based, and
# bundling them would not shrink the image anyway.
RUN --mount=type=cache,target=/root/.cache/pip pip install nuitka \
    && cd /build/src \
    && python -m nuitka \
        --module app \
        --include-package=app \
        --output-dir=/build/compiled \
        --no-progressbar \
        --assume-yes-for-downloads \
    && rm -rf /build/compiled/app.build

# Removes the cold-start penalty for the dependencies. Without it the
# interpreter re-compiles every .py in torch, transformers and their
# dependencies on each boot, in memory, and throws the result away at
# exit. It is also mandatory rather than merely nice once the container
# runs read-only: a read-only filesystem cannot write .pyc at runtime.
RUN python -m compileall -q -j 0 "${VIRTUAL_ENV}"


FROM ${BASE_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app \
    HF_HOME=/var/cache/huggingface \
    # Keeps the caching allocator from fragmenting, which is worth a few
    # hundred megabytes of headroom against the VRAM budget.
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# opencv-python-headless still links glib; curl backs the health probe.
# The cache directory is created and owned here because Docker seeds a
# fresh named volume from the image, and a missing directory yields a
# root-owned volume the unprivileged user cannot write to.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 curl ca-certificates \
    && groupadd --system segmentation \
    && useradd --system --gid segmentation --create-home segmentation \
    && mkdir -p "${HF_HOME}" \
    && chown segmentation:segmentation "${HF_HOME}"

WORKDIR /app

COPY --from=build --chown=segmentation:segmentation /opt/venv /opt/venv
# The compiled extension only. `find /app -name '*.py'` is empty here,
# which is the acceptance criterion for shipping no interpreted source.
COPY --from=build --chown=segmentation:segmentation \
    /build/compiled/ /app/

USER segmentation

# The image is published straight after this build, so the compiled
# extension gets exercised here rather than in a job that would have to
# pull several gigabytes back to do it. Building the transport touches
# every layer that matters - the venv, the .so, the settings - and loads
# no weights, so it costs a second.
RUN AUTH_MODE=none python -c \
    "from app.interfaces.http import create_app; assert create_app()"

EXPOSE 7860

CMD ["uvicorn", "app.interfaces.http:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "7860"]
