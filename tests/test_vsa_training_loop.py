"""Tests for VSATrainingLoop: STDP + hippocampal dual-path learning."""

import pathlib

import numpy as np
import pytest

try:
    import grilly_core
    GRILLY_CORE_AVAILABLE = True
except ImportError:
    GRILLY_CORE_AVAILABLE = False

try:
    from grilly_next.training import VSATrainingLoop, TrainingResult
    from grilly_next.backend import VULKAN_AVAILABLE
except ImportError:
    VULKAN_AVAILABLE = False

_SHADER_DIR = str(pathlib.Path(__file__).parent.parent / "shaders" / "spv")


@pytest.mark.gpu
@pytest.mark.skipif(not GRILLY_CORE_AVAILABLE, reason="grilly_core not available")
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestVSATrainingLoopCreation:
    """Test VSATrainingLoop construction."""

    @pytest.fixture
    def device(self):
        dev = grilly_core.Device()
        dev.load_shaders(_SHADER_DIR)
        return dev

    def test_creation_default_params(self, device):
        loop = VSATrainingLoop(device)
        assert loop.K == 4
        assert loop.vsa_dim == 10240
        assert loop.step_count == 0

    def test_creation_custom_K(self, device):
        loop = VSATrainingLoop(device, K=8)
        assert loop.K == 8
        assert loop.model.K == 8

    def test_creation_small_dims(self, device):
        loop = VSATrainingLoop(device, d_model=32, vsa_dim=256, K=2)
        assert loop.vsa_dim == 256


@pytest.mark.gpu
@pytest.mark.skipif(not GRILLY_CORE_AVAILABLE, reason="grilly_core not available")
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestVSATrainingStep:
    """Test single training steps."""

    @pytest.fixture
    def loop(self):
        device = grilly_core.Device()
        device.load_shaders(_SHADER_DIR)
        return VSATrainingLoop(device, d_model=32, vsa_dim=256, K=2, seed=42)

    def _make_states(self, dim=256):
        num_words = (dim + 31) // 32
        rng = np.random.RandomState(42)
        state_t = rng.randint(0, 2**32, size=num_words, dtype=np.uint32)
        state_t1 = rng.randint(0, 2**32, size=num_words, dtype=np.uint32)
        return state_t, state_t1

    def test_step_returns_result(self, loop):
        state_t, state_t1 = self._make_states()
        result = loop.step(state_t, state_t1)
        assert isinstance(result, TrainingResult)
        assert np.isfinite(result.loss)
        assert result.loss >= 0.0
        assert np.isfinite(result.stdp_weight_norm)
        assert result.stdp_weight_norm > 0.0

    def test_step_increments_count(self, loop):
        state_t, state_t1 = self._make_states()
        loop.step(state_t, state_t1)
        assert loop.step_count == 1
        loop.step(state_t, state_t1)
        assert loop.step_count == 2

    def test_no_dream_before_interval(self, loop):
        state_t, state_t1 = self._make_states()
        result = loop.step(state_t, state_t1)
        assert result.dream_report is None

    def test_stdp_weights_change(self, loop):
        state_t, state_t1 = self._make_states()
        w_before = loop.stdp.weight.copy()
        loop.step(state_t, state_t1)
        w_after = loop.stdp.weight
        assert not np.allclose(w_before, w_after), "STDP weights should change after step"


@pytest.mark.gpu
@pytest.mark.skipif(not GRILLY_CORE_AVAILABLE, reason="grilly_core not available")
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestHippocampalConsolidation:
    """Test dream consolidation path."""

    def test_dream_fires_at_interval(self):
        device = grilly_core.Device()
        device.load_shaders(_SHADER_DIR)
        loop = VSATrainingLoop(
            device, d_model=32, vsa_dim=256, K=2, dream_interval=5, seed=42
        )
        rng = np.random.RandomState(42)
        num_words = (256 + 31) // 32

        results = []
        for i in range(6):
            state_t = rng.randint(0, 2**32, size=num_words, dtype=np.uint32)
            state_t1 = rng.randint(0, 2**32, size=num_words, dtype=np.uint32)
            result = loop.step(state_t, state_t1)
            results.append(result)

        # Steps 1-4: no dream
        for i in range(4):
            assert results[i].dream_report is None

        # Step 5: dream fires
        assert results[4].dream_report is not None
        assert results[4].dream_report.episodes_replayed == 5


@pytest.mark.skipif(not GRILLY_CORE_AVAILABLE, reason="grilly_core not available")
class TestBitunpack:
    """Test bitunpack utility (CPU only)."""

    def test_bitunpack_to_bipolar(self):
        device = grilly_core.Device()
        device.load_shaders(_SHADER_DIR)
        loop = VSATrainingLoop(device, d_model=32, vsa_dim=64, K=2)

        # All ones = all bits set
        packed = np.array([0xFFFFFFFF, 0xFFFFFFFF], dtype=np.uint32)
        bipolar = loop._bitunpack_to_bipolar(packed)
        assert bipolar.shape == (64,)
        assert np.all(bipolar == 1.0)

        # All zeros
        packed_zero = np.array([0x00000000, 0x00000000], dtype=np.uint32)
        bipolar_zero = loop._bitunpack_to_bipolar(packed_zero)
        assert np.all(bipolar_zero == -1.0)
