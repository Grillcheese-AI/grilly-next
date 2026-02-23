# Changelog

## [0.5.0] - 2025-02-23

### Added
- **C++ Vulkan backend** (`grilly_core`): High-performance GPU dispatch via pybind11
  - `linear`, `relu`, `gelu`, `silu`, `tanh`, `layernorm`, `flash_attention2`, `conv2d`, `conv1d`
  - KV Cache with MLA compression and H2O eviction
  - CubeMind VSA encoding + GPU Hamming search
  - OpGraph batched execution with operator fusion
- **Bridge layer** (`backend/_bridge.py`): Stable Python wrappers around `grilly_core`
- **scikit-build-core** build system: `pip install -e .` builds C++ extension automatically
- **Monorepo structure**: C++, Python, shaders, tests all in one repository

### Changed
- `VulkanCompute` now delegates core ops to C++ backend (10-100x faster than ctypes path)
- Moved to `src/` layout for Python package
- Lowered minimum Python to 3.10 (was 3.12)
- Reduced mandatory dependencies to just `numpy` (torch/transformers now optional)

### Deprecated
- Python ctypes Vulkan backend modules (core.py, pipelines.py, base.py, etc.)
  - Still functional for unported ops (SNN, memory, FFT, cells, etc.)
  - Will be removed in v0.6.0 when all ops are ported to C++

### Migration from 0.4.x
- `pip install grilly` continues to work — same package name on PyPI
- `grilly.Compute()` API is unchanged
- If you used internal backend modules directly, import paths changed:
  - `from grilly.backend.fnn import VulkanFNN` → still works but deprecated
  - Prefer `from grilly.backend._bridge import linear` for C++ path
