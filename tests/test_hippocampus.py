"""
Tests for HippocampalConsolidator (offline dream consolidation).

These tests use grilly_core directly (C++ extension) and require
a Vulkan device for WorldModel initialization.
"""

import numpy as np
import pytest

try:
    import grilly_core
    GRILLY_CORE_AVAILABLE = True
except ImportError:
    GRILLY_CORE_AVAILABLE = False

pytestmark = pytest.mark.gpu


def make_bitpacked(dim=10240, seed=None):
    """Create a random bitpacked uint32 array for a given VSA dim."""
    rng = np.random.RandomState(seed)
    words = (dim + 31) // 32
    return rng.randint(0, 2**32, size=words, dtype=np.uint32)


@pytest.fixture
def dev():
    if not GRILLY_CORE_AVAILABLE:
        pytest.skip("grilly_core not available")
    try:
        d = grilly_core.Device()
        d.load_shaders("shaders/spv")
        return d
    except Exception as e:
        pytest.skip(f"Vulkan device not available: {e}")


@pytest.fixture
def world_model(dev):
    return grilly_core.WorldModel(dev, dim=10240)


class TestHippocampalConsolidator:
    """Tests for the HippocampalConsolidator C++ class."""

    def test_basic_creation(self):
        """Constructor works with default and custom capacity."""
        if not GRILLY_CORE_AVAILABLE:
            pytest.skip("grilly_core not available")
        hc = grilly_core.HippocampalConsolidator()
        assert hc.buffer_size == 0

        hc2 = grilly_core.HippocampalConsolidator(max_capacity=500)
        assert hc2.buffer_size == 0

    def test_record_episode(self):
        """record_episode increments buffer_size."""
        if not GRILLY_CORE_AVAILABLE:
            pytest.skip("grilly_core not available")
        hc = grilly_core.HippocampalConsolidator(max_capacity=100)
        s0 = make_bitpacked(seed=0)
        s1 = make_bitpacked(seed=1)
        hc.record_episode(s0, s1)
        assert hc.buffer_size == 1

        hc.record_episode(s0, s1)
        assert hc.buffer_size == 2

    def test_fifo_eviction(self):
        """Buffer evicts oldest episodes when at max_capacity."""
        if not GRILLY_CORE_AVAILABLE:
            pytest.skip("grilly_core not available")
        cap = 50
        hc = grilly_core.HippocampalConsolidator(max_capacity=cap)

        s0 = make_bitpacked(seed=0)
        s1 = make_bitpacked(seed=1)
        for _ in range(cap + 20):
            hc.record_episode(s0, s1)

        assert hc.buffer_size == cap

    def test_dream_identical_transitions_extracts_rules(self, world_model):
        """Identical transitions should appear in >5% and be extracted."""
        hc = grilly_core.HippocampalConsolidator(max_capacity=10000)

        # Record 100 identical transitions — delta appears 100% of the time
        s0 = make_bitpacked(seed=42)
        s1 = make_bitpacked(seed=43)
        for _ in range(100):
            hc.record_episode(s0, s1)

        initial_facts = world_model.fact_count
        report = hc.dream(world_model, cycles=64)

        assert report.episodes_replayed == 100
        assert report.new_rules_extracted >= 1
        assert report.synthetic_dreams == 64
        assert world_model.fact_count > initial_facts

    def test_dream_random_transitions_no_rules(self, world_model):
        """Random unique transitions should stay below the 5% threshold."""
        hc = grilly_core.HippocampalConsolidator(max_capacity=10000)

        # Each transition is unique → each delta count = 1, threshold = 5
        for i in range(100):
            s0 = make_bitpacked(seed=i * 2)
            s1 = make_bitpacked(seed=i * 2 + 1)
            hc.record_episode(s0, s1)

        initial_facts = world_model.fact_count
        report = hc.dream(world_model, cycles=32)

        assert report.episodes_replayed == 100
        assert report.new_rules_extracted == 0
        assert world_model.fact_count == initial_facts

    def test_dream_clears_buffer(self, world_model):
        """After dream(), the episodic buffer should be empty."""
        hc = grilly_core.HippocampalConsolidator(max_capacity=10000)

        s0 = make_bitpacked(seed=0)
        s1 = make_bitpacked(seed=1)
        for _ in range(50):
            hc.record_episode(s0, s1)

        assert hc.buffer_size == 50
        hc.dream(world_model, cycles=16)
        assert hc.buffer_size == 0

    def test_dream_empty_buffer(self, world_model):
        """dream() on empty buffer returns zeros and does not crash."""
        hc = grilly_core.HippocampalConsolidator()
        report = hc.dream(world_model, cycles=64)

        assert report.episodes_replayed == 0
        assert report.new_rules_extracted == 0
        assert report.synthetic_dreams == 0

    def test_dream_report_fields(self, world_model):
        """DreamReport has all expected readonly fields."""
        hc = grilly_core.HippocampalConsolidator(max_capacity=1000)

        s0 = make_bitpacked(seed=10)
        s1 = make_bitpacked(seed=11)
        for _ in range(30):
            hc.record_episode(s0, s1)

        report = hc.dream(world_model, cycles=16)

        # Check all fields are accessible and have expected types
        assert isinstance(report.episodes_replayed, int)
        assert isinstance(report.synthetic_dreams, int)
        assert isinstance(report.new_rules_extracted, int)

    def test_buffer_size_property(self):
        """buffer_size is a readonly property that tracks episode count."""
        if not GRILLY_CORE_AVAILABLE:
            pytest.skip("grilly_core not available")
        hc = grilly_core.HippocampalConsolidator(max_capacity=200)
        assert hc.buffer_size == 0

        for i in range(25):
            hc.record_episode(make_bitpacked(seed=i), make_bitpacked(seed=i+100))

        assert hc.buffer_size == 25
