# grilly

GPU-accelerated neural network framework using Vulkan compute shaders. PyTorch-like API that runs on **any GPU** (AMD, NVIDIA, Intel) — no CUDA dependency.

## Quick Start

```bash
# Install (builds C++ Vulkan backend automatically)
pip install -e .

# Verify
python -c "import grilly; print('Vulkan:', grilly.VULKAN_AVAILABLE)"
python -c "import grilly_core; d = grilly_core.Device(); print('GPU:', d.device_name)"
```

## Features

- **C++ Vulkan backend** — Linear, attention, conv, activations, layernorm compiled to native code
- **Flash Attention 2** — Tiled GPU implementation with online softmax
- **KV Cache** — MLA compression + H2O eviction for efficient inference
- **CubeMind** — Vector Symbolic Architecture with GPU Hamming search (<2ms at 490K vectors)
- **SNN support** — Spiking neural networks (LIF, STDP, Hebbian learning)
- **PyTorch-like API** — `nn.Module`, `functional`, optimizers, DataLoader
- **HuggingFace Bridge** — Load pretrained weights without PyTorch runtime

## Requirements

- Python >= 3.10
- Vulkan SDK + drivers
- CMake >= 3.20, C++17 compiler
- Boost 1.82+, Eigen 5.x (header-only)

## Architecture

```
grilly-next/
├── cpp/           # C++ Vulkan backend (grilly_core extension)
├── src/grilly/    # Python package (nn, functional, optim, utils)
├── shaders/       # 170+ GLSL compute shaders
└── tests/         # Test suite
```

## License

MIT
