"""
Tests for attention operations (flash_attention2 via VulkanCompute)

flash_attention2 expects 4D tensors: (batch, heads, seq_len, head_dim)
Note: The tiled implementation uses tile_size_q=64, tile_size_k=64 by default,
so head_dim should be >= 64 for full numerical accuracy.
"""

import numpy as np
import pytest
import grilly_next as _core

try:
    from grilly_next import Compute
    from grilly_next.backend import VULKAN_AVAILABLE
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)


def _ref_attention_4d(Q, K, V, mask=None, scale=None):
    """Numpy reference for 4D attention: (batch, heads, seq, dim)"""
    batch, heads, seq_len, head_dim = Q.shape
    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)
    output = np.zeros_like(Q)
    for b in range(batch):
        for h in range(heads):
            scores = (Q[b, h] @ K[b, h].T) * scale
            if mask is not None:
                scores = np.where(mask[b, h] > 0, scores, -1e9)
            scores_max = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
            output[b, h] = attn @ V[b, h]
    return output


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestFlashAttention2:
    """Test flash_attention2 on GPU"""

    @pytest.fixture
    def gpu(self):
        """Initialize GPU backend"""
        backend = Compute()
        yield backend
        backend.cleanup()

    def test_output_shape(self, gpu):
        """Test flash_attention2 produces correct output shape"""
        np.random.seed(42)
        batch, heads, seq_len, head_dim = 1, 1, 16, 64
        Q = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1
        K = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1
        V = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1

        output = gpu.flash_attention2(Q, K, V)

        assert output.shape == (batch, heads, seq_len, head_dim)
        assert np.all(np.isfinite(output))

    def test_numerical_structure(self, gpu):
        """Test flash_attention2 output is finite and non-trivial"""
        np.random.seed(42)
        batch, heads, seq_len, head_dim = 1, 1, 8, 64
        Q = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1
        K = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1
        V = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1

        output = gpu.flash_attention2(Q, K, V)

        assert output.shape == (batch, heads, seq_len, head_dim)
        assert np.all(np.isfinite(output))
        # Output should not be all zeros (attention is non-trivial)
        assert np.abs(output).sum() > 0

    def test_with_causal_mask(self, gpu):
        """Test flash_attention2 with causal (lower-triangular) mask"""
        np.random.seed(42)
        batch, heads, seq_len, head_dim = 1, 1, 8, 64
        Q = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1
        K = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1
        V = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1
        mask = np.tril(np.ones((batch, heads, seq_len, seq_len), dtype=np.float32))

        output = gpu.flash_attention2(Q, K, V, mask=mask)

        assert output.shape == (batch, heads, seq_len, head_dim)
        assert np.all(np.isfinite(output))

        # Masked output should differ from unmasked
        output_no_mask = gpu.flash_attention2(Q, K, V)
        # At least the last row should differ (it sees all tokens unmasked vs partial)
        assert not np.allclose(output[0, 0, -1], output_no_mask[0, 0, -1], atol=1e-6) or seq_len == 1

    def test_identity_values(self, gpu):
        """Test attention with identical queries produces identical rows"""
        np.random.seed(42)
        batch, heads, seq_len, head_dim = 1, 1, 4, 64
        q_vec = np.random.randn(head_dim).astype(np.float32) * 0.1
        Q = np.tile(q_vec, (batch, heads, seq_len, 1))
        K = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1
        V = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1

        output = gpu.flash_attention2(Q, K, V)

        # All seq positions should produce the same output
        for i in range(1, seq_len):
            np.testing.assert_allclose(output[0, 0, 0], output[0, 0, i], atol=1e-4)

    def test_different_seq_lengths(self, gpu):
        """Test flash_attention2 with various sequence lengths"""
        np.random.seed(42)
        batch, heads, head_dim = 1, 1, 64
        for seq_len in [1, 4, 16, 32]:
            Q = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1
            K = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1
            V = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1

            output = gpu.flash_attention2(Q, K, V)

            assert output.shape == (batch, heads, seq_len, head_dim)
            assert np.all(np.isfinite(output))

    def test_multi_head(self, gpu):
        """Test flash_attention2 with multiple heads"""
        np.random.seed(42)
        batch, heads, seq_len, head_dim = 1, 4, 8, 64
        Q = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1
        K = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1
        V = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1

        output = gpu.flash_attention2(Q, K, V)

        assert output.shape == (batch, heads, seq_len, head_dim)
        assert np.all(np.isfinite(output))
        # At least some of the output should be non-zero
        assert np.abs(output).sum() > 0

    def test_output_bounded(self, gpu):
        """Test attention output stays within value range"""
        np.random.seed(42)
        batch, heads, seq_len, head_dim = 1, 1, 8, 64
        Q = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1
        K = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1
        V = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32) * 0.1

        output = gpu.flash_attention2(Q, K, V)

        # Output is a weighted average of V, so should be bounded by V's range
        v_min = V.min()
        v_max = V.max()
        assert np.all(output >= v_min - 0.01)
        assert np.all(output <= v_max + 0.01)
