"""
Grilly backend — C++ Vulkan via grilly_core.

All GPU dispatch goes through the C++ extension. Legacy Python ctypes
backend has been removed (available in the grilly repo for reference).
"""

from ._bridge import NATIVE_AVAILABLE

VULKAN_AVAILABLE = NATIVE_AVAILABLE

from .compute import VulkanCompute

__all__ = [
    "VULKAN_AVAILABLE",
    "NATIVE_AVAILABLE",
    "VulkanCompute",
]
