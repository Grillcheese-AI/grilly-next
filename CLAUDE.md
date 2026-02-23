# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**grilly-next** is the unified monorepo for the Grilly GPU-accelerated neural network framework. It combines:
- **C++ Vulkan backend** (`cpp/`) — high-performance GPU ops via pybind11 (`grilly_core` extension)
- **Python package** (`src/grilly/`) — PyTorch-like API with `nn.Module`, `functional`, optimizers, etc.
- **GLSL compute shaders** (`shaders/`) — 170+ shaders compiled to SPIR-V

The C++ backend replaces the old Python ctypes Vulkan path for core ops (linear, activations, layernorm, attention, conv). Unported ops (SNN, memory, FFT, etc.) still use the legacy Python backend.

## Common Commands

```bash
# Install (builds C++ extension via scikit-build-core + CMake)
pip install -e .                    # editable install
pip install -e ".[dev]"             # with dev dependencies
pip install -e ".[all]"             # everything (ml + legacy + dev)

# Testing
pytest tests/ -v                                # all tests
pytest tests/ -m "not gpu" -v                   # CPU-only (no Vulkan)
pytest tests/ --cov=src/grilly --cov-report=term # with coverage

# Linting & Formatting
ruff check .
black . --check
mypy src/grilly/

# Shaders
glslc shaders/shader.glsl -o shaders/spv/shader.spv   # compile single
.\scripts\compile_all_shaders.ps1                       # compile all (Windows)

# C++ rebuild (after changing cpp/ sources)
pip install -e .   # scikit-build-core recompiles automatically
# Or manual CMake:
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## Architecture

### Directory Layout

```
grilly-next/
├── CMakeLists.txt              # Top-level CMake (builds grilly_core.pyd)
├── pyproject.toml              # scikit-build-core config + package metadata
├── cpp/                        # C++ Vulkan backend
│   ├── include/grilly/         # Headers
│   ├── src/                    # Implementation
│   └── python/bindings.cpp     # pybind11 bindings → grilly_core module
├── third_party/                # Git submodules (pybind11, BLAKE3, VMA)
├── shaders/                    # ALL GLSL compute shaders + compiled SPV
│   ├── *.glsl
│   └── spv/*.spv
├── src/grilly/                 # Python package
│   ├── backend/                # GPU dispatch layer
│   │   ├── _bridge.py          # Thin wrappers around grilly_core
│   │   ├── compute.py          # VulkanCompute (delegates to C++ or legacy)
│   │   └── ...                 # Legacy Python ops (kept for unported ops)
│   ├── nn/                     # PyTorch-like Module subclasses
│   ├── functional/             # Stateless functional API
│   ├── optim/                  # Optimizers (Adam, AdamW, SGD, etc.)
│   ├── utils/                  # DataLoader, HuggingFaceBridge, etc.
│   └── experimental/           # Unstable features
├── tests/                      # pytest test suite
├── benchmarks/                 # Performance benchmarks
└── examples/                   # Usage examples
```

### Key Patterns

- **Entry point**: `grilly.Compute()` → `VulkanCompute` which uses C++ grilly_core for ported ops
- **Bridge layer**: `backend/_bridge.py` wraps `grilly_core` C++ extension with stable Python API
- **Dual dispatch**: `compute.py` checks `self._native` — True uses C++, False falls back to Python ctypes
- **Data format**: All data is `np.float32` numpy arrays; GPU upload/download is transparent
- **GPU tests auto-skip** when Vulkan is unavailable (pytest fixture in `tests/conftest.py`)

### Build System

- **scikit-build-core** bridges Python packaging and CMake
- `pip install -e .` triggers CMake, which builds `grilly_core.pyd/.so`
- Dependencies: Vulkan SDK, Boost 1.82+ (headers + atomic), Eigen 5, pybind11, BLAKE3, VMA

## Requirements

- Python >= 3.10 (3.12+ recommended)
- Vulkan SDK installed and on PATH
- Boost 1.82+ (set `BOOST_ROOT` if not in default location)
- Eigen 5.x (set `EIGEN_ROOT` if not in default location)
- CMake >= 3.20
- C++17 compiler (MSVC 2019+, GCC 10+, Clang 12+)
- Minimum: 8-10GB VRAM GPU, 32GB RAM
