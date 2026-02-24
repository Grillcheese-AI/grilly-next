#!/bin/bash
set -euo pipefail

# ── Grilly-Next A40/A100 Environment Setup ──────────────────────────────
#
# Sets up a headless Nvidia GPU server for Vulkan compute.
# Handles the "ICD Ghost" problem where Vulkan can't find the GPU driver
# on servers without a display/X11.
#
# Usage:
#   chmod +x scripts/setup_a40_env.sh
#   sudo ./scripts/setup_a40_env.sh
#
# After running, verify with:
#   vulkaninfo --summary | grep -i nvidia
#   python3 -c "import grilly; print(grilly.Compute())"

echo "=== Grilly-Next A40/A100 Environment Setup ==="

# ── 1. Core build dependencies ──────────────────────────────────────────
echo "[1/6] Installing system dependencies..."
apt-get update
apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libvulkan-dev \
    vulkan-tools \
    python3-dev \
    python3-pip \
    python3-venv

# ── 2. Nvidia driver + Vulkan ICD ──────────────────────────────────────
#
# The A40/A100 needs the server (headless) driver variant. The key package
# is libnvidia-gl which provides the Vulkan ICD JSON that tells the Vulkan
# loader where to find the GPU.
echo "[2/6] Setting up Nvidia driver and Vulkan runtime..."

# Check if nvidia-smi already works (driver pre-installed on cloud instances)
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    echo "  Nvidia driver already installed."
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "  Driver version: ${DRIVER_VERSION}"
else
    echo "  Installing Nvidia driver 535 (server variant)..."
    apt-get install -y nvidia-driver-535-server nvidia-utils-535-server \
                       libnvidia-compute-535-server libnvidia-gl-535-server
fi

# ── 3. Fix Vulkan ICD discovery (the headless server problem) ──────────
#
# On headless servers, the Vulkan loader often can't find the ICD JSON.
# The ICD lives in one of these paths depending on driver installation:
#   /usr/share/vulkan/icd.d/nvidia_icd.json
#   /etc/vulkan/icd.d/nvidia_icd.json
echo "[3/6] Configuring Vulkan ICD for headless operation..."

ICD_PATH=""
for candidate in \
    /usr/share/vulkan/icd.d/nvidia_icd.json \
    /etc/vulkan/icd.d/nvidia_icd.json \
    /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json; do
    if [ -f "$candidate" ]; then
        ICD_PATH="$candidate"
        break
    fi
done

if [ -z "$ICD_PATH" ]; then
    echo "  WARNING: Nvidia Vulkan ICD not found. Checking installed packages..."
    dpkg -l | grep -i nvidia || true
    echo "  You may need to install libnvidia-gl-XXX for your driver version."
else
    echo "  Found ICD: ${ICD_PATH}"
fi

# Set environment variables for current session and persist to bashrc
export VK_ICD_FILENAMES="${ICD_PATH:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
export VK_DRIVER_FILES="${VK_ICD_FILENAMES}"

BASHRC="${HOME}/.bashrc"
if ! grep -q "VK_ICD_FILENAMES" "$BASHRC" 2>/dev/null; then
    {
        echo ""
        echo "# Grilly-Next: Vulkan headless ICD configuration"
        echo "export VK_ICD_FILENAMES=${VK_ICD_FILENAMES}"
        echo "export VK_DRIVER_FILES=${VK_ICD_FILENAMES}"
    } >> "$BASHRC"
    echo "  Added VK_ICD_FILENAMES to ${BASHRC}"
fi

# ── 4. User permissions ────────────────────────────────────────────────
echo "[4/6] Checking GPU access permissions..."
CURRENT_USER="${SUDO_USER:-$USER}"
for group in video render; do
    if getent group "$group" &>/dev/null; then
        if ! id -nG "$CURRENT_USER" | grep -qw "$group"; then
            usermod -a -G "$group" "$CURRENT_USER"
            echo "  Added ${CURRENT_USER} to ${group} group"
        fi
    fi
done

# ── 5. Enable GPU persistence mode ────────────────────────────────────
#
# Server GPUs like A40/A100 "sleep" between dispatches without this.
# Persistence mode keeps the driver loaded, eliminating cold-start latency.
echo "[5/6] Enabling GPU persistence mode..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi -pm 1 2>/dev/null || echo "  Could not set persistence mode (may need root)"
fi

# ── 6. Verify installation ─────────────────────────────────────────────
echo "[6/6] Verifying hardware..."
echo ""
echo "--- nvidia-smi ---"
nvidia-smi || echo "nvidia-smi failed"
echo ""
echo "--- Vulkan Summary ---"
vulkaninfo --summary 2>/dev/null | grep -iE "GPU|device|subgroup|driver" || \
    echo "vulkaninfo failed (check VK_ICD_FILENAMES)"
echo ""
echo "--- Subgroup Size ---"
vulkaninfo 2>/dev/null | grep -i "subgroupSize" || true

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Log out and back in (for group membership to take effect)"
echo "  2. pip install -e '.[all]'   # Build grilly_core for this GPU"
echo "  3. python3 tests/local_vsa_stress_test.py   # Validate VSA pipeline"
echo "  4. Load profile: grilly_core.ConfigLoader.load('profiles.json', 'A40_MASSIVE')"
