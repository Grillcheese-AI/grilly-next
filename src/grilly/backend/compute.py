"""
Main VulkanCompute class that composes all operation modules.

In grilly-next, ops that have been ported to C++ delegate to _bridge.py
(which wraps grilly_core). Ops not yet ported still use the legacy Python
backend modules — these are marked with "# TODO: port to C++" and will
be migrated in future releases.
"""

import os
import numpy as np

from . import _bridge

# Legacy Python modules for ops NOT yet ported to C++
from .snn import VulkanSNN
from .memory import VulkanMemory
from .cells import VulkanCells
from .affect import VulkanAffect
from .learning import VulkanLearning
from .fft import VulkanFFT
from .contrastive import VulkanContrastive
from .pooling import VulkanPooling
from .faiss import VulkanFAISS
from .lora import VulkanLoRA
from .core import VulkanCore
from .pipelines import VulkanPipelines
from .fnn import VulkanFNN
from .attention import VulkanAttention
from .normalization import VulkanNormalization


def _default_shader_dir():
    """Resolve default shader directory relative to package installation."""
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # In grilly-next: shaders/ is at repo root, two levels up from src/grilly/
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
    """Complete Vulkan compute backend with GPU dispatch.

    In grilly-next v0.5+, core GPU ops (linear, activations, layernorm,
    attention, conv) are dispatched via the C++ grilly_core extension.
    Unported ops (SNN, memory, cells, FFT, etc.) still use the Python
    ctypes backend for backward compatibility.
    """

    def __init__(self, shader_dir: str = None):
        """Initialize Vulkan compute backend."""
        # ── C++ backend (grilly_core) ──────────────────────────────────
        if _bridge.NATIVE_AVAILABLE:
            if shader_dir is None:
                shader_dir = _default_shader_dir()
            self._device = _bridge.create_device(shader_dir)
            self._native = True
        else:
            self._device = None
            self._native = False

        # ── Legacy Python backend (for unported ops) ───────────────────
        # These still use the old ctypes Vulkan path.
        # TODO: remove once all ops are ported to C++
        self.core = VulkanCore(shader_dir)
        self.pipelines = VulkanPipelines(self.core)

        self.architecture = None
        self.snn = VulkanSNN(self.core, self.pipelines)
        self.faiss = VulkanFAISS(self.core, self.pipelines)
        self.fnn = VulkanFNN(self.core, self.pipelines, self.core.shaders)
        self.attention = VulkanAttention(
            self.core, self.pipelines, self.core.shaders,
            architecture=self.architecture,
        )
        self.memory = VulkanMemory(
            self.core, self.pipelines, self.core.shaders, self.fnn
        )
        self.cells = VulkanCells(self.core, self.pipelines, self.core.shaders)
        self.affect = VulkanAffect(self.core, self.pipelines, self.core.shaders)
        self.learning = VulkanLearning(self.core, self.pipelines, self.core.shaders)
        self._fft = VulkanFFT(self.core, self.pipelines, self.core.shaders)
        self.contrastive = VulkanContrastive(self.core, self.pipelines, self.core.shaders)
        self.pooling = VulkanPooling(self.core, self.pipelines, self.core.shaders)
        self.conv_legacy = VulkanConv(self.core, self.pipelines, self.core.shaders) if False else None
        self.normalization = VulkanNormalization(self.core, self.pipelines, self.core.shaders)
        self.lora = VulkanLoRA(self.core, self.pipelines, self.core.shaders)

        self.device = self.core.device
        self.queue = self.core.queue
        self.shaders = self.core.shaders

    def cleanup(self):
        """Clean up Vulkan resources."""
        for attr in (
            "fnn", "attention", "snn", "memory", "cells", "affect",
            "learning", "_fft", "contrastive", "pooling",
            "normalization", "lora", "faiss",
        ):
            module = getattr(self, attr, None)
            if module is not None and hasattr(module, "clear_weight_cache"):
                try:
                    module.clear_weight_cache()
                except Exception:
                    pass
        if hasattr(self, "fnn") and self.fnn._pool is not None:
            try:
                self.fnn._pool.clear()
                self.fnn._pool = None
            except Exception:
                pass
        if hasattr(self, "pipelines"):
            self.pipelines.cleanup()
        if hasattr(self, "core"):
            self.core.cleanup()

    def set_architecture(self, architecture: str):
        """Set the model architecture for shader selection."""
        self.architecture = architecture
        self.attention.architecture = architecture

        self._create_buffer = self.core._create_buffer
        self._upload_buffer = self.core._upload_buffer
        self._download_buffer = self.core._download_buffer
        self._dispatch_compute = self.core._dispatch_compute
        self.device = self.core.device
        self.queue = self.core.queue
        self.descriptor_pool = self.core.descriptor_pool
        self.shaders = self.core.shaders
        self.create_pipeline = self.pipelines.get_or_create_pipeline
        self.dispatch = self.core._dispatch_compute
        self.get_tiling_info = self.core.get_tiling_info

    # ═══════════════════════════════════════════════════════════════════
    # C++ ACCELERATED OPS (via grilly_core)
    # ═══════════════════════════════════════════════════════════════════

    def linear(self, x, weight, bias=None, **kwargs):
        """GPU-accelerated linear projection."""
        if self._native:
            return _bridge.linear(self._device, x, weight, bias)
        return self.fnn.linear(x, weight, bias, **kwargs)

    def activation_relu(self, x, **kwargs):
        """Apply ReLU activation."""
        if self._native:
            return _bridge.relu(self._device, x)
        return self.fnn.activation_relu(x, **kwargs)

    def activation_gelu(self, x, **kwargs):
        """Apply GELU activation."""
        if self._native:
            return _bridge.gelu(self._device, x)
        return self.fnn.activation_gelu(x, **kwargs)

    def activation_silu(self, x, **kwargs):
        """Apply SiLU (Swish) activation."""
        if self._native:
            return _bridge.silu(self._device, x)
        return self.fnn.activation_silu(x, **kwargs)

    def activation_tanh(self, x, **kwargs):
        """Apply tanh activation."""
        if self._native:
            return _bridge.tanh_act(self._device, x)
        return self.fnn.activation_tanh(x, **kwargs)

    def layernorm(self, x, gamma, beta, eps=1e-5, **kwargs):
        """GPU-accelerated layer normalization."""
        if self._native:
            return _bridge.layernorm(self._device, x, gamma, beta, eps)
        return self.fnn.layernorm(x, gamma, beta, eps=eps, **kwargs)

    def flash_attention2(self, Q, K, V, mask=None, **kwargs):
        """GPU-accelerated Flash Attention 2."""
        if self._native:
            return _bridge.flash_attention2(self._device, Q, K, V, mask, **kwargs)
        return self.attention.flash_attention2(Q, K, V, mask=mask, **kwargs)

    def conv2d(self, input, weight, bias=None, stride=(1, 1),
               padding=(0, 0), dilation=(1, 1), groups=1):
        """GPU Conv2d forward."""
        if self._native:
            return _bridge.conv2d(self._device, input, weight, bias,
                                  stride, padding, dilation, groups)
        return self.fnn.conv2d(input, weight, bias, stride, padding, dilation, groups)

    def conv1d(self, input, weight, bias=None, stride=1,
               padding=0, dilation=1, groups=1):
        """GPU Conv1d forward."""
        if self._native:
            return _bridge.conv1d(self._device, input, weight, bias,
                                  stride, padding, dilation, groups)
        return self.fnn.conv1d(input, weight, bias, stride, padding, dilation, groups)

    # ═══════════════════════════════════════════════════════════════════
    # LEGACY OPS (still on Python ctypes backend)
    # TODO: port to C++
    # ═══════════════════════════════════════════════════════════════════

    def activation_gcu(self, *args, **kwargs):
        return self.fnn.activation_gcu(*args, **kwargs)

    def activation_roswish(self, *args, **kwargs):
        return self.fnn.activation_roswish(*args, **kwargs)

    def activation_swiglu(self, *args, **kwargs):
        return self.fnn.activation_swiglu(*args, **kwargs)

    def activation_tanh_backward(self, *args, **kwargs):
        return self.fnn.activation_tanh_backward(*args, **kwargs)

    def activation_softmax(self, *args, **kwargs):
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")
        return self.fnn.activation_softmax(*args, **kwargs)

    def xavier_init(self, *args, **kwargs):
        return self.fnn.xavier_init(*args, **kwargs)

    def linear_backward(self, *args, **kwargs):
        return self.fnn.linear_backward(*args, **kwargs)

    def cross_entropy_backward(self, *args, **kwargs):
        return self.fnn.cross_entropy_backward(*args, **kwargs)

    def cross_entropy_loss(self, *args, **kwargs):
        return self.fnn.cross_entropy_loss(*args, **kwargs)

    def activation_gelu_backward(self, *args, **kwargs):
        return self.fnn.activation_gelu_backward(*args, **kwargs)

    def activation_gcu_backward(self, *args, **kwargs):
        return self.fnn.activation_gcu_backward(*args, **kwargs)

    def activation_roswish_backward(self, *args, **kwargs):
        return self.fnn.activation_roswish_backward(*args, **kwargs)

    def activation_swiglu_backward(self, *args, **kwargs):
        return self.fnn.activation_swiglu_backward(*args, **kwargs)

    def layernorm_backward(self, *args, **kwargs):
        return self.fnn.layernorm_backward(*args, **kwargs)

    def softmax_backward(self, *args, **kwargs):
        return self.fnn.softmax_backward(*args, **kwargs)

    def residual(self, *args, **kwargs):
        return self.fnn.residual(*args, **kwargs)

    def dropout(self, *args, **kwargs):
        return self.fnn.dropout(*args, **kwargs)

    # SNN operations
    def lif_step(self, *args, **kwargs):
        return self.snn.lif_step(*args, **kwargs)

    def hebbian_learning(self, *args, **kwargs):
        return self.snn.hebbian_learning(*args, **kwargs)

    def stdp_learning(self, *args, **kwargs):
        return self.snn.stdp_learning(*args, **kwargs)

    def gif_neuron_step(self, *args, **kwargs):
        return self.snn.gif_neuron_step(*args, **kwargs)

    # FAISS operations
    def faiss_compute_distances(self, *args, **kwargs):
        return self.faiss.compute_distances(*args, **kwargs)

    def faiss_topk(self, *args, **kwargs):
        return self.faiss.topk(*args, **kwargs)

    # Memory operations
    def memory_read(self, *args, **kwargs):
        return self.memory.memory_read(*args, **kwargs)

    def memory_write(self, *args, **kwargs):
        return self.memory.memory_write(*args, **kwargs)

    def memory_query_pooling(self, *args, **kwargs):
        return self.memory.memory_query_pooling(*args, **kwargs)

    def memory_inject_gate(self, *args, **kwargs):
        return self.memory.memory_inject_gate(*args, **kwargs)

    # Attention operations
    def attention_scores(self, *args, **kwargs):
        return self.attention.attention_scores(*args, **kwargs)

    def attention_mask(self, *args, **kwargs):
        return self.attention.attention_mask(*args, **kwargs)

    def attention_output(self, *args, **kwargs):
        return self.attention.attention_output(*args, **kwargs)

    def attention_concat_heads(self, *args, **kwargs):
        return self.attention.attention_concat_heads(*args, **kwargs)

    # Cell operations
    def place_cell(self, *args, **kwargs):
        return self.cells.place_cell(*args, **kwargs)

    def time_cell(self, *args, **kwargs):
        return self.cells.time_cell(*args, **kwargs)

    # Learning operations
    def adam_update(self, *args, **kwargs):
        return self.learning.adam_update(*args, **kwargs)

    def fisher_info_update(self, *args, **kwargs):
        return self.learning.fisher_info_update(*args, **kwargs)

    def ewc_penalty(self, *args, **kwargs):
        return self.learning.ewc_penalty(*args, **kwargs)

    def natural_gradient(self, *args, **kwargs):
        return self.learning.natural_gradient(*args, **kwargs)

    def nlms_predict(self, *args, **kwargs):
        return self.learning.nlms_predict(*args, **kwargs)

    def nlms_update(self, *args, **kwargs):
        return self.learning.nlms_update(*args, **kwargs)

    def whitening_transform(self, *args, **kwargs):
        return self.learning.whitening_transform(*args, **kwargs)

    def continuous_to_spikes(self, *args, **kwargs):
        return self.learning.continuous_to_spikes(*args, **kwargs)

    def spikes_to_continuous(self, *args, **kwargs):
        return self.learning.spikes_to_continuous(*args, **kwargs)

    def domain_route(self, *args, **kwargs):
        return self.learning.domain_route(*args, **kwargs)

    def embedding_lookup(self, *args, **kwargs):
        return self.learning.embedding_lookup(*args, **kwargs)

    def ssm_fused_math(self, *args, **kwargs):
        return self.learning.ssm_fused_math(*args, **kwargs)

    def ssm_fused_uv(self, *args, **kwargs):
        return self.learning.ssm_fused_uv(*args, **kwargs)

    def ssd_chunk_scan(self, *args, **kwargs):
        return self.learning.ssd_chunk_scan(*args, **kwargs)

    # FFT operations
    def fft(self, *args, **kwargs):
        return self._fft.fft(*args, **kwargs)

    def fft_magnitude(self, *args, **kwargs):
        return self._fft.fft_magnitude(*args, **kwargs)

    def fft_power_spectrum(self, *args, **kwargs):
        return self._fft.fft_power_spectrum(*args, **kwargs)

    def fft_normalize(self, *args, **kwargs):
        return self._fft.fft_normalize(*args, **kwargs)

    # Contrastive learning operations
    def contrastive_loss(self, *args, **kwargs):
        return self.contrastive.contrastive_loss(*args, **kwargs)

    def contrastive_gradient(self, *args, **kwargs):
        return self.contrastive.contrastive_gradient(*args, **kwargs)

    # Buffer management (legacy API)
    def create_buffer(self, data_or_size, usage="storage"):
        from .base import VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
        if isinstance(usage, str):
            usage_flag = (
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT if usage == "storage"
                else VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
            )
        else:
            usage_flag = usage
        if isinstance(data_or_size, (int, np.integer)):
            size = int(data_or_size)
        elif isinstance(data_or_size, bytes):
            size = len(data_or_size)
        elif hasattr(data_or_size, "nbytes"):
            size = data_or_size.nbytes
        elif isinstance(data_or_size, (list, tuple)):
            size = len(data_or_size) * 4
        else:
            raise ValueError(f"Unsupported data type: {type(data_or_size)}")
        return self.core._create_buffer(size, usage_flag)

    def upload_buffer(self, buffer, memory, data):
        return self.core._upload_buffer(buffer, memory, data)

    def read_buffer(self, memory, size=None, dtype=np.float32):
        if size is None:
            raise ValueError("Size must be provided for read_buffer")
        return self.core._download_buffer(memory, size, dtype=dtype)

    def get_tiling_info(self):
        return self.core.get_tiling_info()

    def __del__(self):
        if hasattr(self, "pipelines"):
            self.pipelines.cleanup()
        if hasattr(self, "core"):
            self.core.cleanup()
