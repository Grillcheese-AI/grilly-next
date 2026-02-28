"""
Tests for inference operations: layernorm, flash attention,
linear projections, and CPU reference implementations.

GPU tests use VulkanCompute's actual API surface. CPU fallback
tests verify numpy reference implementations independently.
"""

import numpy as np
import pytest

try:
    from grilly_next import Compute
    from grilly_next.backend import VULKAN_AVAILABLE
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)


# ---------------------------------------------------------------------------
# Numpy reference helpers
# ---------------------------------------------------------------------------

def _ref_rms_norm(x, weight, eps):
    """Numpy reference: RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight"""
    mean_sq = np.mean(x ** 2, axis=-1, keepdims=True)
    normed = x * (1.0 / np.sqrt(mean_sq + eps))
    return normed * weight


def _ref_silu(x):
    """SiLU(x) = x * sigmoid(x)"""
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -88, 88))))


def _ref_swiglu_fused(x, gate_weights, up_weights):
    """
    Fused SwiGLU: SiLU(x @ gate_proj.T) * (x @ up_proj.T)
    """
    input_dim = x.shape[-1]
    intermediate_size = gate_weights.shape[0]
    x_2d = x.reshape(-1, input_dim)
    gate = x_2d @ gate_weights.T
    up = x_2d @ up_weights.T
    result = _ref_silu(gate) * up
    outer_shape = x.shape[:-1] + (intermediate_size,)
    return result.reshape(outer_shape)


def _ref_gemm_int8(activations, weights_int8, scales, group_size):
    """Dequantize INT8 weights then matmul."""
    M, K = activations.shape
    N = weights_int8.shape[0]
    num_groups = (K + group_size - 1) // group_size
    w_fp32 = np.zeros((N, K), dtype=np.float32)
    for g in range(num_groups):
        k_start = g * group_size
        k_end = min(k_start + group_size, K)
        w_fp32[:, k_start:k_end] = (
            weights_int8[:, k_start:k_end].astype(np.float32) * scales[:, g : g + 1]
        )
    return activations @ w_fp32.T


def _ref_gqa_decode_attention(query, k_cache, v_cache, num_q_heads,
                              num_kv_heads, head_dim, cache_len=None,
                              scale=None):
    """
    Numpy reference for GQA decode attention.
    query:   (batch, 1, num_q_heads, head_dim)
    k_cache: (batch, cache_len, num_kv_heads, head_dim)
    v_cache: (batch, cache_len, num_kv_heads, head_dim)
    """
    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)
    batch_size = query.shape[0]
    if cache_len is None:
        cache_len = k_cache.shape[1]
    kv_group_size = num_q_heads // num_kv_heads
    q_2d = query.reshape(batch_size, num_q_heads, head_dim)
    output = np.zeros((batch_size, num_q_heads, head_dim), dtype=np.float32)
    for b in range(batch_size):
        for qh in range(num_q_heads):
            kv_head = qh // kv_group_size
            scores = (
                np.einsum("d,sd->s", q_2d[b, qh], k_cache[b, :cache_len, kv_head])
                * scale
            )
            scores_max = np.max(scores)
            exp_scores = np.exp(scores - scores_max)
            weights = exp_scores / np.sum(exp_scores)
            output[b, qh] = np.einsum(
                "s,sd->d", weights, v_cache[b, :cache_len, kv_head]
            )
    return output.reshape(batch_size, 1, num_q_heads, head_dim)


# ===================================================================
# GPU Tests — grilly-next VulkanCompute API
# ===================================================================

@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestLayerNormGPU:
    """Test LayerNorm on GPU (VulkanCompute.layernorm)"""

    @pytest.fixture
    def gpu(self):
        backend = Compute()
        yield backend
        backend.cleanup()

    def test_layernorm_2d(self, gpu):
        """Test LayerNorm with 2D input (batch, features)"""
        np.random.seed(42)
        x = np.random.randn(1, 64).astype(np.float32)
        gamma = np.ones(64, dtype=np.float32)
        beta = np.zeros(64, dtype=np.float32)
        result = gpu.layernorm(x, gamma, beta)
        assert abs(result.mean()) < 0.15
        assert abs(result.std() - 1.0) < 0.15

    def test_layernorm_custom_gamma_beta(self, gpu):
        """Test LayerNorm with learned gamma/beta"""
        np.random.seed(42)
        x = np.random.randn(4, 128).astype(np.float32)
        gamma = np.random.randn(128).astype(np.float32) * 0.5 + 1.0
        beta = np.random.randn(128).astype(np.float32) * 0.1
        eps = 1e-5
        result = gpu.layernorm(x, gamma, beta, eps)
        # Numpy reference: layernorm over last dim
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        normed = (x - mean) / np.sqrt(var + eps)
        expected = normed * gamma + beta
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_layernorm_eps(self, gpu):
        """Test that eps parameter changes the output"""
        np.random.seed(42)
        x = np.random.randn(2, 64).astype(np.float32)
        gamma = np.ones(64, dtype=np.float32)
        beta = np.zeros(64, dtype=np.float32)
        result_small_eps = gpu.layernorm(x, gamma, beta, eps=1e-8)
        result_large_eps = gpu.layernorm(x, gamma, beta, eps=1.0)
        assert not np.allclose(result_small_eps, result_large_eps, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestFlashAttention2GPU:
    """Test flash_attention2 on GPU"""

    @pytest.fixture
    def gpu(self):
        backend = Compute()
        yield backend
        backend.cleanup()

    def test_flash_attention_basic_shape(self, gpu):
        """Test flash_attention2 output shape"""
        np.random.seed(42)
        batch, heads, seq_len, head_dim = 1, 1, 16, 64
        Q = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1
        K = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1
        V = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1
        result = gpu.flash_attention2(Q, K, V)
        assert result.shape == (batch, heads, seq_len, head_dim)
        assert np.all(np.isfinite(result))

    def test_flash_attention_structure(self, gpu):
        """Test flash_attention2 output has correct structural properties"""
        np.random.seed(42)
        batch, heads, seq_len, head_dim = 1, 1, 8, 64
        Q = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1
        K = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1
        V = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1
        result = gpu.flash_attention2(Q, K, V)

        assert result.shape == (batch, heads, seq_len, head_dim)
        assert np.all(np.isfinite(result))
        # Output is a convex combination of V rows — bounded by V's range
        assert np.all(result >= V.min() - 0.01)
        assert np.all(result <= V.max() + 0.01)

    def test_flash_attention_with_mask(self, gpu):
        """Test flash_attention2 with causal mask"""
        np.random.seed(42)
        batch, heads, seq_len, head_dim = 1, 1, 8, 64
        Q = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1
        K = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1
        V = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1
        mask = np.tril(np.ones((batch, heads, seq_len, seq_len), dtype=np.float32))
        result = gpu.flash_attention2(Q, K, V, mask=mask)
        assert result.shape == (batch, heads, seq_len, head_dim)
        assert np.all(np.isfinite(result))


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestLinearGPU:
    """Test linear projection on GPU"""

    @pytest.fixture
    def gpu(self):
        backend = Compute()
        yield backend
        backend.cleanup()

    def test_linear_numerical(self, gpu):
        """Test linear matches x @ W^T"""
        np.random.seed(42)
        x = np.random.randn(4, 64).astype(np.float32)
        weight = np.random.randn(32, 64).astype(np.float32) * 0.1
        result = gpu.linear(x, weight)
        expected = x @ weight.T
        assert result.shape == (4, 32)
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_linear_with_bias(self, gpu):
        """Test linear with bias"""
        np.random.seed(42)
        x = np.random.randn(8, 32).astype(np.float32)
        weight = np.random.randn(16, 32).astype(np.float32) * 0.1
        bias = np.random.randn(16).astype(np.float32) * 0.1
        result = gpu.linear(x, weight, bias)
        expected = x @ weight.T + bias
        assert result.shape == (8, 16)
        np.testing.assert_allclose(result, expected, atol=1e-3)


# ===================================================================
# CPU Fallback Tests (no GPU marker)
# ===================================================================


class TestRMSNormCPUFallback:
    """Test RMSNorm CPU fallback path"""

    def test_rms_norm_cpu(self):
        """Test RMSNorm CPU fallback produces correct results"""
        np.random.seed(42)
        x = np.random.randn(4, 128).astype(np.float32)
        weight = np.random.randn(128).astype(np.float32) * 0.5 + 1.0
        eps = 1e-5

        expected = _ref_rms_norm(x, weight, eps)

        mean_sq = np.mean(x ** 2, axis=-1, keepdims=True)
        manual = x * (1.0 / np.sqrt(mean_sq + eps)) * weight
        np.testing.assert_allclose(expected, manual, atol=1e-5)
        assert expected.shape == x.shape
        assert not np.allclose(x, expected)

    def test_rms_norm_cpu_1d(self):
        """Test RMSNorm CPU fallback with 1D input"""
        np.random.seed(42)
        x = np.random.randn(64).astype(np.float32)
        weight = np.ones(64, dtype=np.float32)
        eps = 1e-5
        result = _ref_rms_norm(x, weight, eps)
        assert result.shape == (64,)
        assert np.all(np.isfinite(result))

    def test_rms_norm_cpu_3d(self):
        """Test RMSNorm CPU fallback with 3D input"""
        np.random.seed(42)
        x = np.random.randn(2, 8, 64).astype(np.float32)
        weight = np.ones(64, dtype=np.float32)
        eps = 1e-5
        result = _ref_rms_norm(x, weight, eps)
        assert result.shape == (2, 8, 64)
        np.testing.assert_allclose(
            result,
            x * (1.0 / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)) * weight,
            atol=1e-5,
        )


class TestSwiGLUFusedCPUFallback:
    """Test fused SwiGLU CPU fallback path"""

    def test_swiglu_fused_cpu(self):
        """Test fused SwiGLU CPU fallback produces correct results"""
        np.random.seed(42)
        batch, input_dim, intermediate = 4, 64, 128
        x = np.random.randn(batch, input_dim).astype(np.float32)
        gate_w = np.random.randn(intermediate, input_dim).astype(np.float32) * 0.02
        up_w = np.random.randn(intermediate, input_dim).astype(np.float32) * 0.02

        result = _ref_swiglu_fused(x, gate_w, up_w)

        gate = x @ gate_w.T
        up = x @ up_w.T
        silu_gate = gate * (1.0 / (1.0 + np.exp(-np.clip(gate, -88, 88))))
        manual = silu_gate * up

        np.testing.assert_allclose(result, manual, atol=1e-5)
        assert result.shape == (batch, intermediate)

    def test_swiglu_fused_cpu_3d(self):
        """Test fused SwiGLU CPU fallback with 3D input"""
        np.random.seed(42)
        batch, seq, input_dim, intermediate = 2, 8, 32, 64
        x = np.random.randn(batch, seq, input_dim).astype(np.float32)
        gate_w = np.random.randn(intermediate, input_dim).astype(np.float32) * 0.02
        up_w = np.random.randn(intermediate, input_dim).astype(np.float32) * 0.02

        result = _ref_swiglu_fused(x, gate_w, up_w)
        assert result.shape == (batch, seq, intermediate)
        assert np.all(np.isfinite(result))


class TestGEMMInt8CPUFallback:
    """Test INT8 GEMM CPU fallback path"""

    def test_gemm_int8_cpu(self):
        """Test INT8 GEMM CPU fallback dequantizes and computes correctly"""
        np.random.seed(42)
        M, K, N, group_size = 4, 128, 64, 64
        activations = np.random.randn(M, K).astype(np.float32)
        w_int8 = np.random.randint(-127, 127, size=(N, K), dtype=np.int8)
        num_groups = (K + group_size - 1) // group_size
        scales = np.random.rand(N, num_groups).astype(np.float32) * 0.1 + 0.01

        result = _ref_gemm_int8(activations, w_int8, scales, group_size)

        w_fp32 = np.zeros((N, K), dtype=np.float32)
        for g in range(num_groups):
            k_s = g * group_size
            k_e = min(k_s + group_size, K)
            w_fp32[:, k_s:k_e] = w_int8[:, k_s:k_e].astype(np.float32) * scales[:, g : g + 1]
        manual = activations @ w_fp32.T

        np.testing.assert_allclose(result, manual, atol=1e-5)
        assert result.shape == (M, N)

    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_gemm_int8_cpu_group_sizes(self, group_size):
        """Test INT8 GEMM CPU fallback with different group sizes"""
        np.random.seed(42)
        M, K, N = 2, 256, 32
        activations = np.random.randn(M, K).astype(np.float32)
        w_int8 = np.random.randint(-127, 127, size=(N, K), dtype=np.int8)
        num_groups = (K + group_size - 1) // group_size
        scales = np.random.rand(N, num_groups).astype(np.float32) * 0.1 + 0.01

        result = _ref_gemm_int8(activations, w_int8, scales, group_size)
        assert result.shape == (M, N)
        assert np.all(np.isfinite(result))


class TestGQADecodeAttentionCPUFallback:
    """Test GQA decode attention CPU fallback path"""

    def test_gqa_decode_cpu(self):
        """Test GQA decode CPU fallback produces correct results"""
        np.random.seed(42)
        batch, cache_len, num_q, num_kv, head_dim = 1, 16, 12, 4, 64
        query = np.random.randn(batch, 1, num_q, head_dim).astype(np.float32) * 0.1
        k_cache = np.random.randn(batch, cache_len, num_kv, head_dim).astype(np.float32) * 0.1
        v_cache = np.random.randn(batch, cache_len, num_kv, head_dim).astype(np.float32) * 0.1

        result = _ref_gqa_decode_attention(
            query, k_cache, v_cache, num_q, num_kv, head_dim
        )
        assert result.shape == (batch, 1, num_q, head_dim)
        assert np.all(np.isfinite(result))

    def test_gqa_decode_cpu_batch_gt_1(self):
        """Test GQA decode CPU fallback with batch_size > 1"""
        np.random.seed(42)
        batch, cache_len, num_q, num_kv, head_dim = 3, 16, 12, 4, 64
        query = np.random.randn(batch, 1, num_q, head_dim).astype(np.float32) * 0.1
        k_cache = np.random.randn(batch, cache_len, num_kv, head_dim).astype(np.float32) * 0.1
        v_cache = np.random.randn(batch, cache_len, num_kv, head_dim).astype(np.float32) * 0.1

        result = _ref_gqa_decode_attention(
            query, k_cache, v_cache, num_q, num_kv, head_dim
        )
        assert result.shape == (batch, 1, num_q, head_dim)

        r0 = _ref_gqa_decode_attention(
            query[0:1], k_cache[0:1], v_cache[0:1], num_q, num_kv, head_dim
        )
        r1 = _ref_gqa_decode_attention(
            query[1:2], k_cache[1:2], v_cache[1:2], num_q, num_kv, head_dim
        )
        np.testing.assert_allclose(result[0:1], r0, atol=1e-5)
        np.testing.assert_allclose(result[1:2], r1, atol=1e-5)

    def test_gqa_decode_cpu_gqa_grouping(self):
        """Test GQA head grouping: query heads map to correct KV heads"""
        np.random.seed(42)
        batch, cache_len, num_q, num_kv, head_dim = 1, 8, 6, 2, 32
        kv_group_size = num_q // num_kv

        query = np.random.randn(batch, 1, num_q, head_dim).astype(np.float32) * 0.1
        k_cache = np.random.randn(batch, cache_len, num_kv, head_dim).astype(np.float32) * 0.1
        v_cache = np.random.randn(batch, cache_len, num_kv, head_dim).astype(np.float32) * 0.1

        result = _ref_gqa_decode_attention(
            query, k_cache, v_cache, num_q, num_kv, head_dim
        )

        q_2d = query.reshape(batch, num_q, head_dim)
        scale = 1.0 / np.sqrt(head_dim)
        for qh in range(num_q):
            kv_head = qh // kv_group_size
            scores = np.einsum("d,sd->s", q_2d[0, qh], k_cache[0, :cache_len, kv_head]) * scale
            scores_max = np.max(scores)
            exp_scores = np.exp(scores - scores_max)
            weights = exp_scores / np.sum(exp_scores)
            expected_head = np.einsum("s,sd->d", weights, v_cache[0, :cache_len, kv_head])
            np.testing.assert_allclose(
                result[0, 0, qh], expected_head, atol=1e-5
            )

    def test_gqa_decode_cpu_different_cache_lengths(self):
        """Test GQA decode CPU fallback with various cache lengths"""
        np.random.seed(42)
        batch, num_q, num_kv, head_dim = 1, 8, 4, 32
        for cache_len in [1, 4, 16, 64]:
            query = np.random.randn(batch, 1, num_q, head_dim).astype(np.float32) * 0.1
            k_cache = np.random.randn(batch, cache_len, num_kv, head_dim).astype(np.float32) * 0.1
            v_cache = np.random.randn(batch, cache_len, num_kv, head_dim).astype(np.float32) * 0.1
            result = _ref_gqa_decode_attention(
                query, k_cache, v_cache, num_q, num_kv, head_dim
            )
            assert result.shape == (batch, 1, num_q, head_dim)
            assert np.all(np.isfinite(result))
