"""
Bridge layer: thin wrappers around grilly_core C++ extension.

This module provides backward-compatible Python callables for every op
exposed by the pybind11 bindings in grilly_core.  Higher-level modules
(compute.py, nn/, functional/) import from here instead of touching
grilly_core directly, which keeps the migration surface in one file.
"""

import numpy as np

try:
    import grilly_core as _core

    NATIVE_AVAILABLE = True
except ImportError:
    try:
        from grilly import grilly_core as _core

        NATIVE_AVAILABLE = True
    except ImportError:
        _core = None
        NATIVE_AVAILABLE = False


# ── Device wrapper ────────────────────────────────────────────────────────

def create_device(shader_dir: str = None):
    """Create a GrillyCoreContext (Device) and optionally load shaders."""
    if not NATIVE_AVAILABLE:
        raise RuntimeError(
            "grilly_core C++ extension not found. "
            "Reinstall with: pip install -e ."
        )
    dev = _core.Device()
    if shader_dir is not None:
        dev.load_shaders(shader_dir)
    return dev


# ── Linear ────────────────────────────────────────────────────────────────

def linear(device, x, weight, bias=None):
    """GPU linear projection: output = x @ W^T + bias"""
    return _core.linear(device, x, weight, bias)


def linear_cpu(x, weight, bias=None):
    """CPU linear projection using Eigen (for verification)"""
    return _core.linear_cpu(x, weight, bias)


# ── Activations ───────────────────────────────────────────────────────────

def relu(device, x):
    return _core.relu(device, x)


def gelu(device, x):
    return _core.gelu(device, x)


def silu(device, x):
    return _core.silu(device, x)


def tanh_act(device, x):
    return _core.tanh_act(device, x)


# ── LayerNorm ─────────────────────────────────────────────────────────────

def layernorm(device, x, gamma, beta, eps=1e-5):
    return _core.layernorm(device, x, gamma, beta, eps)


# ── Flash Attention 2 ────────────────────────────────────────────────────

def flash_attention2(device, Q, K, V, mask=None, scale=0.0,
                     tile_size_q=64, tile_size_k=64):
    return _core.flash_attention2(device, Q, K, V, mask, scale,
                                  tile_size_q, tile_size_k)


# ── Conv2d / Conv1d ──────────────────────────────────────────────────────

def conv2d(device, input, weight, bias=None,
           stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    return _core.conv2d(device, input, weight, bias,
                        list(stride), list(padding), list(dilation), groups)


def conv1d(device, input, weight, bias=None,
           stride=1, padding=0, dilation=1, groups=1):
    return _core.conv1d(device, input, weight, bias,
                        stride, padding, dilation, groups)


# ── KV Cache ─────────────────────────────────────────────────────────────

def create_kv_cache(device, **kwargs):
    return _core.create_kv_cache(device, **kwargs)


def kv_cache_append(device, kv_cache, new_keys, new_values):
    return _core.kv_cache_append(device, kv_cache, new_keys, new_values)


def kv_cache_decode(device, kv_cache):
    return _core.kv_cache_decode(device, kv_cache)


def kv_cache_evict_h2o(device, kv_cache, attention_scores=None, num_evict=0):
    return _core.kv_cache_evict_h2o(device, kv_cache, attention_scores, num_evict)


def kv_cache_compact(device, kv_cache):
    return _core.kv_cache_compact(device, kv_cache)


def destroy_kv_cache(device, kv_cache):
    return _core.destroy_kv_cache(device, kv_cache)


def kv_cache_init_eviction_head(device, kv_cache, input_dim, hidden_dim=32, lr=1e-3):
    return _core.kv_cache_init_eviction_head(device, kv_cache, input_dim, hidden_dim, lr)


def kv_cache_train_eviction_head(device, kv_cache, token_features, attention_scores, seq_len):
    return _core.kv_cache_train_eviction_head(
        device, kv_cache, token_features, attention_scores, seq_len)


def kv_cache_evict_speculative(device, kv_cache, hidden_states=None, hidden_dim=64):
    return _core.kv_cache_evict_speculative(device, kv_cache, hidden_states, hidden_dim)


# ── Swizzle ──────────────────────────────────────────────────────────────

def swizzle_kv(device, input, wave_size=32, reverse=False):
    return _core.swizzle_kv(device, input, wave_size, reverse)


# ── Experimental: Fused Attention ────────────────────────────────────────

def fused_attention_cpu(Q, latents, w_up, mask=None, cached_tokens=0,
                        num_heads=8, head_dim=64, latent_dim=16, scale=0.0):
    return _core.fused_attention_cpu(Q, latents, w_up, mask, cached_tokens,
                                     num_heads, head_dim, latent_dim, scale)


# ── CubeMind: VSA ────────────────────────────────────────────────────────

def blake3_role(key, dim, domain="grilly.cubemind"):
    return _core.blake3_role(key, dim, domain)


def vsa_bind(a, b):
    return _core.vsa_bind(a, b)


def vsa_bundle(vectors):
    return _core.vsa_bundle(vectors)


def vsa_bitpack(bipolar):
    return _core.vsa_bitpack(bipolar)


def vsa_encode(roles, fillers, dim):
    return _core.vsa_encode(roles, fillers, dim)


# ── CubeMind: Hamming Search ─────────────────────────────────────────────

def hamming_search(device, query, cache):
    return _core.hamming_search(device, query, cache)


def hamming_search_cpu(query, cache):
    return _core.hamming_search_cpu(query, cache)


# ── CubeMind: HammingSearchBench ─────────────────────────────────────────

HammingSearchBench = _core.HammingSearchBench if NATIVE_AVAILABLE else None


# ── CubeMind: Cube State ─────────────────────────────────────────────────

def cube_solved(size=3):
    return _core.cube_solved(size)


def cube_apply_move(state, size=3, move=0):
    return _core.cube_apply_move(state, size, move)


def cube_random_walk(size=3, num_moves=20, seed=0):
    return _core.cube_random_walk(size, num_moves, seed)


def cube_estimate_distance(state, size=3):
    return _core.cube_estimate_distance(state, size)


def cube_to_vsa(state, size=3, dim=10240):
    return _core.cube_to_vsa(state, size, dim)


# ── CubeMind: VSA Cache ─────────────────────────────────────────────────

VSACache = _core.VSACache if NATIVE_AVAILABLE else None


# ── CubeMind: TextEncoder ────────────────────────────────────────────────

TextEncoder = _core.TextEncoder if NATIVE_AVAILABLE else None


# ── CubeMind: SemanticAssigner (Memoized LSH Projection Cache) ──────────

SemanticAssigner = _core.SemanticAssigner if NATIVE_AVAILABLE else None


# ── Training Pipeline (Producer-Consumer) ────────────────────────────────

ParsedDocument = _core.ParsedDocument if NATIVE_AVAILABLE else None
TrainingPayload = _core.TrainingPayload if NATIVE_AVAILABLE else None
TrainingPipeline = _core.TrainingPipeline if NATIVE_AVAILABLE else None


# ── SystemProfile: Hardware Configuration ─────────────────────────────

SystemProfile = _core.SystemProfile if NATIVE_AVAILABLE else None


# ── OpGraph ──────────────────────────────────────────────────────────────

OpGraph = _core.OpGraph if NATIVE_AVAILABLE else None


# ── KVCache class ────────────────────────────────────────────────────────

KVCache = _core.KVCache if NATIVE_AVAILABLE else None


# ── Autograd: TapeArena + Wengert List Backward Engine ──────────────────

# OpType enum — maps to backward shader dispatch in C++
OpType = _core.OpType if NATIVE_AVAILABLE else None

# TensorRef — lightweight tensor descriptor for the autograd graph
TensorRef = _core.TensorRef if NATIVE_AVAILABLE else None

# TapeContext — records forward pass, runs backward via Wengert list walk
TapeContext = _core.TapeContext if NATIVE_AVAILABLE else None

# AutogradNode — read-only inspection of arena-allocated nodes
AutogradNode = _core.AutogradNode if NATIVE_AVAILABLE else None

AUTOGRAD_AVAILABLE = NATIVE_AVAILABLE and TapeContext is not None


def create_tape_context(device, arena_capacity=64 * 1024 * 1024):
    """Create a TapeContext for recording autograd graphs on the C++ arena.

    Parameters
    ----------
    device : grilly_core.Device
        The Vulkan device context.
    arena_capacity : int
        Arena size in bytes (default 64 MB).

    Returns
    -------
    grilly_core.TapeContext
    """
    if not AUTOGRAD_AVAILABLE:
        raise RuntimeError("C++ autograd not available — grilly_core missing or outdated")
    return _core.TapeContext(device, arena_capacity)


def make_tensor_ref(buffer_id, shape, dtype=0, requires_grad=True):
    """Create a TensorRef descriptor for the autograd graph.

    Parameters
    ----------
    buffer_id : int
        BufferPool buffer ID.
    shape : list[int]
        Tensor dimensions (max 8).
    dtype : int
        0=f32, 1=f16, 2=u32, 3=i32
    requires_grad : bool
        Whether this tensor needs gradients.

    Returns
    -------
    grilly_core.TensorRef
    """
    if not AUTOGRAD_AVAILABLE:
        raise RuntimeError("C++ autograd not available")
    ref = _core.TensorRef()
    ref.buffer_id = buffer_id
    ref.set_shape(shape)
    ref.dtype = dtype
    ref.requires_grad = requires_grad
    return ref
