"""
VulkanCompute — C++ only GPU dispatch.

All ops route through _bridge.py → grilly_core C++ extension.
No Python ctypes fallback. No legacy modules.
"""

import os
import numpy as np

from . import _bridge


def _default_shader_dir():
    """Resolve default shader directory relative to package installation."""
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(pkg_dir, "..", "..", "shaders", "spv"),  # dev / editable
        os.path.join(pkg_dir, "shaders", "spv"),              # installed wheel
    ]
    for path in candidates:
        norm = os.path.normpath(path)
        if os.path.isdir(norm):
            return norm
    return None


class VulkanCompute:
    """C++ Vulkan compute backend.

    All GPU ops dispatch through grilly_core. No Python fallback.
    """

    def __init__(self, shader_dir: str = None):
        if not _bridge.NATIVE_AVAILABLE:
            raise RuntimeError(
                "grilly_core C++ extension not found. "
                "Reinstall with: pip install -e ."
            )
        if shader_dir is None:
            shader_dir = _default_shader_dir()
        self._device = _bridge.create_device(shader_dir)

    @property
    def device(self):
        return self._device

    def cleanup(self):
        """Clean up Vulkan resources."""
        self._device = None

    # ═══════════════════════════════════════════════════════════════════
    # Core ops
    # ═══════════════════════════════════════════════════════════════════

    def linear(self, x, weight, bias=None, **kwargs):
        return _bridge.linear(self._device, x, weight, bias)

    def activation_relu(self, x, **kwargs):
        return _bridge.relu(self._device, x)

    def activation_gelu(self, x, **kwargs):
        return _bridge.gelu(self._device, x)

    def activation_silu(self, x, **kwargs):
        return _bridge.silu(self._device, x)

    def activation_tanh(self, x, **kwargs):
        return _bridge.tanh_act(self._device, x)

    def layernorm(self, x, gamma, beta, eps=1e-5, **kwargs):
        return _bridge.layernorm(self._device, x, gamma, beta, eps)

    def flash_attention2(self, Q, K, V, mask=None, **kwargs):
        return _bridge.flash_attention2(self._device, Q, K, V, mask, **kwargs)

    def conv2d(self, input, weight, bias=None, stride=(1, 1),
               padding=(0, 0), dilation=(1, 1), groups=1):
        return _bridge.conv2d(self._device, input, weight, bias,
                              stride, padding, dilation, groups)

    def conv1d(self, input, weight, bias=None, stride=1,
               padding=0, dilation=1, groups=1):
        return _bridge.conv1d(self._device, input, weight, bias,
                              stride, padding, dilation, groups)

    # ═══════════════════════════════════════════════════════════════════
    # KV Cache
    # ═══════════════════════════════════════════════════════════════════

    def create_kv_cache(self, **kwargs):
        return _bridge.create_kv_cache(self._device, **kwargs)

    def kv_cache_append(self, kv_cache, new_keys, new_values):
        return _bridge.kv_cache_append(self._device, kv_cache, new_keys, new_values)

    def kv_cache_decode(self, kv_cache):
        return _bridge.kv_cache_decode(self._device, kv_cache)

    def kv_cache_evict_h2o(self, kv_cache, attention_scores=None, num_evict=0):
        return _bridge.kv_cache_evict_h2o(self._device, kv_cache, attention_scores, num_evict)

    # ═══════════════════════════════════════════════════════════════════
    # VSA / CubeMind
    # ═══════════════════════════════════════════════════════════════════

    def hamming_search(self, query, cache):
        return _bridge.hamming_search(self._device, query, cache)

    def swizzle_kv(self, input, wave_size=32, reverse=False):
        return _bridge.swizzle_kv(self._device, input, wave_size, reverse)

    def __del__(self):
        pass
