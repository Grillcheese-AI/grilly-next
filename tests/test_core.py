"""
Tests for core Grilly functionality and initialization
"""

import numpy as np
import pytest

try:
    import grilly_next
    from grilly_next.backend import VULKAN_AVAILABLE

    GRILLY_AVAILABLE = True
except ImportError:
    GRILLY_AVAILABLE = False
    VULKAN_AVAILABLE = False

try:
    import grilly_core

    GRILLY_CORE_AVAILABLE = True
except ImportError:
    GRILLY_CORE_AVAILABLE = False


class TestGrillyImports:
    """Test that grilly can be imported correctly"""

    def test_import_grilly(self):
        """Test basic grilly import"""
        assert GRILLY_AVAILABLE, "grilly package not available"
        assert hasattr(grilly_next, "VULKAN_AVAILABLE")

    def test_import_compute(self):
        """Test Compute class import"""
        from grilly_next import Compute, VulkanCompute

        assert Compute is VulkanCompute

    def test_vulkan_available_flag(self):
        """Test VULKAN_AVAILABLE flag"""
        from grilly_next.backend import VULKAN_AVAILABLE

        assert isinstance(VULKAN_AVAILABLE, bool)

    def test_import_bridge(self):
        """Test bridge module exposes core wrappers"""
        from grilly_next.backend import _bridge

        assert hasattr(_bridge, "NATIVE_AVAILABLE")
        assert hasattr(_bridge, "create_device")
        assert hasattr(_bridge, "linear")
        assert hasattr(_bridge, "relu")
        assert hasattr(_bridge, "gelu")
        assert hasattr(_bridge, "silu")
        assert hasattr(_bridge, "layernorm")
        assert hasattr(_bridge, "flash_attention2")

    def test_import_grilly_core(self):
        """Test grilly_core C++ extension is importable"""
        assert GRILLY_CORE_AVAILABLE, "grilly_core not available"
        assert hasattr(grilly_core, "Device")
        assert hasattr(grilly_core, "linear")
        assert hasattr(grilly_core, "relu")
        assert hasattr(grilly_core, "gelu")


@pytest.mark.gpu
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestComputeInitialization:
    """Test Compute backend initialization"""

    def test_compute_init(self):
        """Test Compute initialization creates a device"""
        from grilly_next import Compute

        backend = Compute()
        assert backend.device is not None
        backend.cleanup()

    def test_compute_cleanup(self):
        """Test Compute cleanup nulls the device"""
        from grilly_next import Compute

        backend = Compute()
        assert backend.device is not None
        backend.cleanup()
        assert backend.device is None

    def test_compute_has_core_ops(self):
        """Test Compute exposes core operation methods"""
        from grilly_next import Compute

        backend = Compute()
        assert callable(backend.linear)
        assert callable(backend.activation_relu)
        assert callable(backend.activation_gelu)
        assert callable(backend.activation_silu)
        assert callable(backend.activation_tanh)
        assert callable(backend.layernorm)
        assert callable(backend.flash_attention2)
        assert callable(backend.conv1d)
        assert callable(backend.conv2d)
        backend.cleanup()

    def test_compute_has_kv_cache_ops(self):
        """Test Compute exposes KV cache methods"""
        from grilly_next import Compute

        backend = Compute()
        assert callable(backend.create_kv_cache)
        assert callable(backend.kv_cache_append)
        assert callable(backend.kv_cache_decode)
        assert callable(backend.kv_cache_evict_h2o)
        backend.cleanup()

    def test_compute_has_vsa_ops(self):
        """Test Compute exposes VSA / CubeMind methods"""
        from grilly_next import Compute

        backend = Compute()
        assert callable(backend.hamming_search)
        assert callable(backend.swizzle_kv)
        backend.cleanup()


@pytest.mark.skipif(not GRILLY_CORE_AVAILABLE, reason="grilly_core not available")
class TestGrillyCoreDirectAccess:
    """Test grilly_core C++ extension classes"""

    def test_vsa_cache_class(self):
        """Test VSACache is accessible from grilly_core"""
        assert hasattr(grilly_core, "VSACache")

    def test_text_encoder_class(self):
        """Test TextEncoder is accessible from grilly_core"""
        assert hasattr(grilly_core, "TextEncoder")

    def test_vsa_encode(self):
        """Test VSA encode function exists"""
        assert hasattr(grilly_core, "vsa_encode")

    def test_blake3_role(self):
        """Test blake3_role function exists"""
        assert hasattr(grilly_core, "blake3_role")

    def test_hippocampal_consolidator_class(self):
        """Test HippocampalConsolidator is accessible"""
        assert hasattr(grilly_core, "HippocampalConsolidator")
        hc = grilly_core.HippocampalConsolidator()
        assert hc.buffer_size == 0
