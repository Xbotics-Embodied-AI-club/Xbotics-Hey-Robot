ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ARG BUILDPLATFORM
ARG TARGETPLATFORM
ARG TARGETARCH

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        ffmpeg \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libsndfile1 \
        libportaudio2 \
        portaudio19-dev \
        libusb-1.0-0 \
        libudev1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN python -m pip install --upgrade pip \
    && python -m pip install -e ".[all]"

LABEL org.opencontainers.image.title="hey-robot-runtime" \
      org.opencontainers.image.description="Hey Robot unified runtime image" \
      org.opencontainers.image.source="local" \
      org.opencontainers.image.vendor="hey-robot" \
      org.opencontainers.image.base.name="python:${PYTHON_VERSION}-slim" \
      org.opencontainers.image.buildplatform="${BUILDPLATFORM}" \
      org.opencontainers.image.targetplatform="${TARGETPLATFORM}" \
      org.opencontainers.image.architecture="${TARGETARCH}"

CMD ["hey-robot", "--help"]
