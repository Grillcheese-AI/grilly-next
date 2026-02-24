# ── Grilly-Next: GPU Compute Container ──────────────────────────────────
#
# Vendor-agnostic Vulkan compute image for A40/A100 deployment.
# Uses nvidia/vulkan base which pre-configures the ICD and driver layer.
#
# Build:
#   docker build -t grilly-next .
#
# Run (requires nvidia-container-toolkit):
#   docker run --gpus all --rm grilly-next
#
# Interactive:
#   docker run --gpus all --rm -it grilly-next bash

FROM nvidia/vulkan:1.3-470 AS base

ENV DEBIAN_FRONTEND=noninteractive

# ── 1. System dependencies ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libvulkan-dev \
    vulkan-tools \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Vulkan headless configuration ───────────────────────────────────
# nvidia/vulkan base sets these, but we reinforce for compute-only use
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
ENV VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json

# ── 3. Python dependencies ─────────────────────────────────────────────
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    pytest

# ── 4. Copy source and build ───────────────────────────────────────────
WORKDIR /app
COPY . .

# Initialize git submodules if needed
RUN git config --global --add safe.directory /app && \
    git submodule update --init --recursive 2>/dev/null || true

# Build C++ extension via pip (scikit-build-core + CMake)
RUN pip3 install --no-cache-dir -e ".[all]" || \
    pip3 install --no-cache-dir -e .

# ── 5. Set Python path for grilly_core ─────────────────────────────────
ENV PYTHONPATH=/app:$PYTHONPATH

# ── 6. Default: run the pre-flight stress test ─────────────────────────
CMD ["python3", "tests/local_vsa_stress_test.py"]
