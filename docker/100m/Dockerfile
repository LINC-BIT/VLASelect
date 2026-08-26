FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# Keep the 100M image as a minimal bootstrap layer only.
# Heavy ML/runtime packages are installed later by dep.sh inside the container.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    git \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libvulkan1 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
CMD ["bash"]
