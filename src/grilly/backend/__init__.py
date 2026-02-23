"""
Vulkan compute backend module for GPU-accelerated neural network operations.

In grilly-next, the primary GPU dispatch path is the C++ grilly_core
extension. Legacy Python ctypes ops are kept for unported operations.
"""

from ._bridge import NATIVE_AVAILABLE

# Backward-compat: VULKAN_AVAILABLE used throughout the codebase
try:
    from .base import VULKAN_AVAILABLE
except ImportError:
    VULKAN_AVAILABLE = NATIVE_AVAILABLE

from .capsule_transformer import (
    CapsuleMemory,
    CapsuleTransformerConfig,
    CognitiveFeatures,
    MemoryType,
)
from .compute import VulkanCompute
from .learning import VulkanLearning
from .lora import VulkanLoRA
from .snn_compute import SNNCompute

__all__ = [
    "VULKAN_AVAILABLE",
    "NATIVE_AVAILABLE",
    "VulkanCompute",
    "SNNCompute",
    "VulkanLearning",
    "VulkanLoRA",
    "CapsuleMemory",
    "CapsuleTransformerConfig",
    "CognitiveFeatures",
    "MemoryType",
]
