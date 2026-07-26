ARG BASE_IMAGE=hey-robot-base:latest
FROM ${BASE_IMAGE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libusb-1.0-0 \
        libudev1 \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --extra robot

CMD ["hey-robot", "robot", "--help"]
