"""
Grilly - GPU-accelerated neural network framework using Vulkan compute shaders.

C++ backend (grilly_core) handles all GPU dispatch via pybind11.
Python package provides the bridge layer and training orchestration.

Legacy nn/functional/optim layers have been removed — use grilly_core directly
or through backend._bridge for all GPU operations.
"""

from grilly.backend import VULKAN_AVAILABLE
from grilly.backend.compute import VulkanCompute

Compute = VulkanCompute

__all__ = [
    "VULKAN_AVAILABLE",
    "VulkanCompute",
    "Compute",
]

try:
    from grilly._version import version as __version__
except ImportError:
    __version__ = "0.5.0.dev0"
