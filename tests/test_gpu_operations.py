"""
Tests for GPU operations (requires Vulkan)
"""

import numpy as np
import pytest

try:
    from grilly_next import Compute
    from grilly_next.backend import VULKAN_AVAILABLE
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestActivations:
    """Test activation functions on GPU"""

    @pytest.fixture
    def gpu(self):
        """Initialize GPU backend"""
        backend = Compute()
        yield backend
        backend.cleanup()

    def test_activation_relu(self, gpu):
        """Test ReLU activation"""
        input_data = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        output = gpu.activation_relu(input_data)
        expected = np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float32)
        np.testing.assert_allclose(output, expected, rtol=1e-5)

    def test_activation_relu_batch(self, gpu):
        """Test ReLU with larger batch"""
        input_data = np.random.randn(1000).astype(np.float32)
        output = gpu.activation_relu(input_data)
        expected = np.maximum(input_data, 0.0)
        np.testing.assert_allclose(output, expected, rtol=1e-5)

    def test_activation_gelu(self, gpu):
        """Test GELU activation"""
        input_data = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        output = gpu.activation_gelu(input_data)
        # GELU(0) should be 0
        assert abs(output[0]) < 1e-5
        # GELU should be positive for positive inputs
        assert output[1] > 0

    def test_activation_gelu_approximation(self, gpu):
        """Test GELU matches tanh approximation"""
        input_data = np.linspace(-3.0, 3.0, 100).astype(np.float32)
        output = gpu.activation_gelu(input_data)
        # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x = input_data
        expected = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
        np.testing.assert_allclose(output, expected, atol=1e-3)

    def test_activation_silu(self, gpu):
        """Test SiLU (Swish) activation"""
        input_data = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
        output = gpu.activation_silu(input_data)
        # SiLU(x) = x * sigmoid(x)
        expected = input_data / (1.0 + np.exp(-input_data))
        np.testing.assert_allclose(output, expected, rtol=1e-5)

    def test_activation_silu_batch(self, gpu):
        """Test SiLU with larger batch"""
        input_data = np.random.randn(500).astype(np.float32)
        output = gpu.activation_silu(input_data)
        expected = input_data / (1.0 + np.exp(-np.clip(input_data, -88, 88)))
        np.testing.assert_allclose(output, expected, rtol=1e-4)

    def test_activation_tanh(self, gpu):
        """Test tanh activation"""
        input_data = np.array([0.0, 1.0, -1.0, 3.0, -3.0], dtype=np.float32)
        output = gpu.activation_tanh(input_data)
        expected = np.tanh(input_data)
        np.testing.assert_allclose(output, expected, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestLinearOp:
    """Test linear (matmul) on GPU"""

    @pytest.fixture
    def gpu(self):
        backend = Compute()
        yield backend
        backend.cleanup()

    def test_linear_basic(self, gpu):
        """Test basic linear: y = x @ W^T"""
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)          # (1, 3)
        weight = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]], dtype=np.float32)      # (2, 3)
        output = gpu.linear(x, weight)
        expected = x @ weight.T  # (1, 2)
        np.testing.assert_allclose(output, expected, atol=1e-5)

    def test_linear_with_bias(self, gpu):
        """Test linear with bias: y = x @ W^T + b"""
        x = np.ones((4, 8), dtype=np.float32)
        weight = np.ones((3, 8), dtype=np.float32) * 0.5
        bias = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        output = gpu.linear(x, weight, bias)
        expected = x @ weight.T + bias
        np.testing.assert_allclose(output, expected, atol=1e-4)

    def test_linear_batch(self, gpu):
        """Test linear with larger batch"""
        np.random.seed(42)
        x = np.random.randn(16, 64).astype(np.float32)
        weight = np.random.randn(32, 64).astype(np.float32) * 0.1
        output = gpu.linear(x, weight)
        expected = x @ weight.T
        assert output.shape == (16, 32)
        np.testing.assert_allclose(output, expected, atol=1e-3)


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestLayerNorm:
    """Test layer normalization on GPU"""

    @pytest.fixture
    def gpu(self):
        backend = Compute()
        yield backend
        backend.cleanup()

    def test_layernorm_basic(self, gpu):
        """Test layernorm normalises to ~mean=0, std=1"""
        input_data = np.random.randn(1, 100).astype(np.float32)
        gamma = np.ones(100, dtype=np.float32)
        beta = np.zeros(100, dtype=np.float32)
        output = gpu.layernorm(input_data, gamma, beta)
        assert abs(output.mean()) < 0.1
        assert abs(output.std() - 1.0) < 0.15

    def test_layernorm_with_scale_shift(self, gpu):
        """Test layernorm with gamma and beta"""
        np.random.seed(42)
        input_data = np.random.randn(1, 64).astype(np.float32)
        gamma = np.ones(64, dtype=np.float32) * 2.0
        beta = np.ones(64, dtype=np.float32) * 0.5
        output = gpu.layernorm(input_data, gamma, beta)
        # After norm with gamma=2, beta=0.5: mean ≈ 0.5, std ≈ 2.0
        assert abs(output.mean() - 0.5) < 0.3
        assert abs(output.std() - 2.0) < 0.3

    def test_layernorm_numerical(self, gpu):
        """Test layernorm matches numpy reference"""
        np.random.seed(42)
        x = np.random.randn(4, 128).astype(np.float32)
        gamma = np.random.randn(128).astype(np.float32) * 0.5 + 1.0
        beta = np.random.randn(128).astype(np.float32) * 0.1
        eps = 1e-5
        output = gpu.layernorm(x, gamma, beta, eps)
        # Numpy reference: layernorm over last dim
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        normed = (x - mean) / np.sqrt(var + eps)
        expected = normed * gamma + beta
        np.testing.assert_allclose(output, expected, atol=1e-3)


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestConvolutions:
    """Test conv1d and conv2d on GPU"""

    @pytest.fixture
    def gpu(self):
        backend = Compute()
        yield backend
        backend.cleanup()

    def test_conv1d_identity(self, gpu):
        """Test conv1d with identity-like kernel"""
        # (batch=1, channels=1, length=5)
        x = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float32)
        # (out_ch=1, in_ch=1, kernel=1)
        weight = np.array([[[1.0]]], dtype=np.float32)
        output = gpu.conv1d(x, weight)
        np.testing.assert_allclose(output, x, atol=1e-5)

    def test_conv1d_with_padding(self, gpu):
        """Test conv1d preserves length with padding"""
        np.random.seed(42)
        x = np.random.randn(1, 3, 16).astype(np.float32)
        weight = np.random.randn(8, 3, 3).astype(np.float32) * 0.1
        output = gpu.conv1d(x, weight, padding=1)
        # With kernel_size=3, padding=1, stride=1: output_len = input_len
        assert output.shape == (1, 8, 16)

    def test_conv2d_identity(self, gpu):
        """Test conv2d with 1x1 identity kernel"""
        x = np.random.randn(1, 1, 4, 4).astype(np.float32)
        weight = np.array([[[[1.0]]]], dtype=np.float32)  # (1,1,1,1)
        output = gpu.conv2d(x, weight)
        np.testing.assert_allclose(output, x, atol=1e-5)

    def test_conv2d_shape(self, gpu):
        """Test conv2d output shape"""
        np.random.seed(42)
        x = np.random.randn(2, 3, 8, 8).astype(np.float32)
        weight = np.random.randn(16, 3, 3, 3).astype(np.float32) * 0.1
        output = gpu.conv2d(x, weight, padding=(1, 1))
        # With 3x3 kernel, padding=1, stride=1: spatial dims preserved
        assert output.shape == (2, 16, 8, 8)
