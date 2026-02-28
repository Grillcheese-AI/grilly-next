"""
Integration tests for Grilly-Next SDK

Tests end-to-end pipelines using the VulkanCompute API and grilly_core
C++ extension. Covers neural network forward passes, VSA encoding +
retrieval, KV cache operations, and multi-op sequences.
"""

import numpy as np
import pytest

try:
    from grilly_next import Compute
    from grilly_next.backend import VULKAN_AVAILABLE
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)

try:
    import grilly_core
    GRILLY_CORE_AVAILABLE = True
except ImportError:
    GRILLY_CORE_AVAILABLE = False


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestNeuralNetPipeline:
    """Integration: linear -> activation -> layernorm pipeline"""

    @pytest.fixture
    def gpu(self):
        backend = Compute()
        yield backend
        backend.cleanup()

    def test_linear_relu_layernorm(self, gpu):
        """Test linear -> ReLU -> layernorm pipeline"""
        np.random.seed(42)
        x = np.random.randn(8, 64).astype(np.float32)
        weight = np.random.randn(32, 64).astype(np.float32) * 0.1
        bias = np.random.randn(32).astype(np.float32) * 0.01
        gamma = np.ones(32, dtype=np.float32)
        beta = np.zeros(32, dtype=np.float32)

        h = gpu.linear(x, weight, bias)
        assert h.shape == (8, 32)

        h = gpu.activation_relu(h)
        assert np.all(h >= 0)

        out = gpu.layernorm(h, gamma, beta)
        assert out.shape == (8, 32)
        assert np.all(np.isfinite(out))

    def test_linear_gelu_linear(self, gpu):
        """Test MLP block: linear -> GELU -> linear"""
        np.random.seed(42)
        x = np.random.randn(4, 128).astype(np.float32)
        w1 = np.random.randn(256, 128).astype(np.float32) * 0.05
        w2 = np.random.randn(128, 256).astype(np.float32) * 0.05

        h = gpu.linear(x, w1)
        assert h.shape == (4, 256)

        h = gpu.activation_gelu(h)

        out = gpu.linear(h, w2)
        assert out.shape == (4, 128)
        assert np.all(np.isfinite(out))

    def test_linear_silu_linear(self, gpu):
        """Test SwiGLU-style block: linear -> SiLU -> linear"""
        np.random.seed(42)
        x = np.random.randn(4, 64).astype(np.float32)
        w_gate = np.random.randn(128, 64).astype(np.float32) * 0.05
        w_down = np.random.randn(64, 128).astype(np.float32) * 0.05

        h = gpu.linear(x, w_gate)
        h = gpu.activation_silu(h)
        out = gpu.linear(h, w_down)
        assert out.shape == (4, 64)
        assert np.all(np.isfinite(out))


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestAttentionPipeline:
    """Integration: QKV projection -> flash_attention2 -> output projection"""

    @pytest.fixture
    def gpu(self):
        backend = Compute()
        yield backend
        backend.cleanup()

    def test_qkv_attention_pipeline(self, gpu):
        """Test full attention: project Q/K/V -> flash_attention2 -> project out"""
        np.random.seed(42)
        batch, heads, seq_len, d_model, head_dim = 1, 1, 8, 64, 64

        x = np.random.randn(seq_len, d_model).astype(np.float32) * 0.1
        w_q = np.random.randn(head_dim, d_model).astype(np.float32) * 0.05
        w_k = np.random.randn(head_dim, d_model).astype(np.float32) * 0.05
        w_v = np.random.randn(head_dim, d_model).astype(np.float32) * 0.05
        w_o = np.random.randn(d_model, head_dim).astype(np.float32) * 0.05

        Q = gpu.linear(x, w_q).reshape(batch, heads, seq_len, head_dim)
        K = gpu.linear(x, w_k).reshape(batch, heads, seq_len, head_dim)
        V = gpu.linear(x, w_v).reshape(batch, heads, seq_len, head_dim)

        attn_out = gpu.flash_attention2(Q, K, V)
        assert attn_out.shape == (batch, heads, seq_len, head_dim)

        # Reshape back to 2D for output projection
        attn_2d = attn_out.reshape(seq_len, head_dim)
        out = gpu.linear(attn_2d, w_o)
        assert out.shape == (seq_len, d_model)
        assert np.all(np.isfinite(out))


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestConvPipeline:
    """Integration: conv -> activation -> layernorm"""

    @pytest.fixture
    def gpu(self):
        backend = Compute()
        yield backend
        backend.cleanup()

    def test_conv1d_relu_pipeline(self, gpu):
        """Test conv1d -> ReLU pipeline"""
        np.random.seed(42)
        x = np.random.randn(1, 3, 32).astype(np.float32)
        weight = np.random.randn(8, 3, 3).astype(np.float32) * 0.1

        h = gpu.conv1d(x, weight, padding=1)
        assert h.shape == (1, 8, 32)

        # Flatten for activation (element-wise)
        h_flat = h.flatten()
        out_flat = gpu.activation_relu(h_flat)
        assert np.all(out_flat >= 0)

    def test_conv2d_gelu_pipeline(self, gpu):
        """Test conv2d -> GELU pipeline"""
        np.random.seed(42)
        x = np.random.randn(1, 1, 8, 8).astype(np.float32)
        weight = np.random.randn(4, 1, 3, 3).astype(np.float32) * 0.1

        h = gpu.conv2d(x, weight, padding=(1, 1))
        assert h.shape == (1, 4, 8, 8)

        h_flat = h.flatten()
        out_flat = gpu.activation_gelu(h_flat)
        assert np.all(np.isfinite(out_flat))


@pytest.mark.gpu
@pytest.mark.skipif(not GRILLY_CORE_AVAILABLE, reason="grilly_core not available")
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestVSAPipeline:
    """Integration: VSA encode -> cache -> search"""

    def test_vsa_encode_shape_and_type(self):
        """Test VSA encode returns correct shape and dtype"""
        dim = 10240
        roles = ["subject", "verb"]
        filler1 = np.array(list(b"cat"), dtype=np.int8)
        filler2 = np.array(list(b"sat"), dtype=np.int8)
        encoded = grilly_core.vsa_encode(roles, [filler1, filler2], dim)
        assert encoded.dtype == np.uint32
        words = (dim + 31) // 32
        assert encoded.shape == (words,)

        # Different fillers should produce different encodings
        filler3 = np.array(list(b"dog"), dtype=np.int8)
        encoded2 = grilly_core.vsa_encode(roles, [filler1, filler3], dim)
        assert not np.array_equal(encoded, encoded2)

    def test_blake3_role_determinism(self):
        """Test blake3_role produces deterministic bitpacked roles"""
        dim = 10240
        r1 = grilly_core.blake3_role("subject", dim)
        r2 = grilly_core.blake3_role("subject", dim)
        np.testing.assert_array_equal(r1, r2)

        r3 = grilly_core.blake3_role("object", dim)
        assert not np.array_equal(r1, r3)

    def test_vsa_bind_self_inverse(self):
        """Test VSA bind (bipolar multiply) is self-inverse"""
        dim = 1024
        # Create bipolar {-1, +1} vectors
        rng = np.random.RandomState(42)
        a = rng.choice([-1, 1], size=dim).astype(np.int8)
        b = rng.choice([-1, 1], size=dim).astype(np.int8)

        bound = grilly_core.vsa_bind(a, b)
        # XOR-like self-inverse: bind(bind(a, b), b) == a
        recovered = grilly_core.vsa_bind(bound, b)
        np.testing.assert_array_equal(recovered, a)

    def test_vsa_bitpack(self):
        """Test bipolar -> bitpacked conversion"""
        dim = 256
        bipolar = np.ones(dim, dtype=np.int8)  # all +1
        packed = grilly_core.vsa_bitpack(bipolar)
        words = (dim + 31) // 32
        assert packed.shape == (words,)
        # All +1 should pack to all 1-bits = 0xFFFFFFFF
        for w in packed:
            assert w == 0xFFFFFFFF


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestMultiOpSequence:
    """Integration: verify GPU state is clean across different op types"""

    @pytest.fixture
    def gpu(self):
        backend = Compute()
        yield backend
        backend.cleanup()

    def test_activation_then_linear(self, gpu):
        """Test running activations then linear preserves correctness"""
        np.random.seed(42)

        # Run activations first
        x = np.random.randn(100).astype(np.float32)
        r = gpu.activation_relu(x)
        g = gpu.activation_gelu(x)
        s = gpu.activation_silu(x)
        t = gpu.activation_tanh(x)
        assert np.all(r >= 0)
        assert np.all(np.isfinite(g))
        assert np.all(np.isfinite(s))
        assert np.all(np.abs(t) <= 1.0 + 1e-5)

        # Now run a linear — should still work correctly
        mat = np.random.randn(4, 32).astype(np.float32)
        w = np.eye(32, dtype=np.float32)
        out = gpu.linear(mat, w)
        np.testing.assert_allclose(out, mat, atol=1e-5)

    def test_layernorm_then_attention(self, gpu):
        """Test layernorm followed by flash_attention2"""
        np.random.seed(42)
        batch, heads, seq_len, dim = 1, 1, 8, 64

        x = np.random.randn(seq_len, dim).astype(np.float32)
        gamma = np.ones(dim, dtype=np.float32)
        beta = np.zeros(dim, dtype=np.float32)

        # Layernorm needs 2D input
        normed = gpu.layernorm(x, gamma, beta)

        # Reshape for 4D attention
        Q = normed.reshape(batch, heads, seq_len, dim)
        K = Q.copy()
        V = Q.copy()
        out = gpu.flash_attention2(Q, K, V)
        assert out.shape == (batch, heads, seq_len, dim)
        assert np.all(np.isfinite(out))

    def test_large_batch_activations(self, gpu):
        """Test processing large batches doesn't corrupt state"""
        n = 10000
        x = np.random.randn(n).astype(np.float32)

        relu_out = gpu.activation_relu(x)
        assert relu_out.shape == (n,)
        assert np.all(relu_out >= 0)

        gelu_out = gpu.activation_gelu(x)
        assert gelu_out.shape == (n,)
        assert np.all(np.isfinite(gelu_out))
