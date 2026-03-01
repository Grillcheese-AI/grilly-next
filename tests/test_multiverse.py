"""Tests for Hilbert Multiverse Generator (Annex H)."""

import pathlib

import numpy as np
import pytest

try:
    import grilly_core
    GRILLY_CORE_AVAILABLE = True
except ImportError:
    GRILLY_CORE_AVAILABLE = False

try:
    from grilly_next.multiverse import (
        CognitiveUniverse,
        MultiverseGenerator,
        MultiverseStep,
    )
    from grilly_next.backend import VULKAN_AVAILABLE
except ImportError:
    VULKAN_AVAILABLE = False

_SHADER_DIR = str(pathlib.Path(__file__).parent.parent / "shaders" / "spv")


class TestCognitiveUniverse:
    """Test CognitiveUniverse enum values match Annex H."""

    def test_phase_offsets(self):
        assert CognitiveUniverse.ANALYTICAL == 0
        assert CognitiveUniverse.EMPATHETIC == 512
        assert CognitiveUniverse.SKEPTICAL == 1024
        assert CognitiveUniverse.POETIC == 2048
        assert CognitiveUniverse.ASSERTIVE == 4096

    def test_enum_iteration(self):
        universes = list(CognitiveUniverse)
        assert len(universes) == 5


class TestPhaseShift:
    """Test circular permutation (Annex H, Section H.2) on CPU."""

    def test_zero_shift_is_identity(self):
        deltas = np.random.randn(4, 256).astype(np.float32)
        shifts = np.zeros(4, dtype=np.uint32)
        result = MultiverseGenerator._apply_phase_shift(deltas, shifts)
        np.testing.assert_array_equal(result, deltas)

    def test_shift_is_circular(self):
        D = 256
        deltas = np.arange(D, dtype=np.float32).reshape(1, D)
        shifts = np.array([3], dtype=np.uint32)
        result = MultiverseGenerator._apply_phase_shift(deltas, shifts)
        # roll left by 3: [3, 4, 5, ..., 255, 0, 1, 2]
        expected = np.roll(deltas, -3, axis=-1)
        np.testing.assert_array_equal(result, expected)

    def test_full_rotation_is_identity(self):
        D = 256
        deltas = np.random.randn(2, D).astype(np.float32)
        shifts = np.array([D, D], dtype=np.uint32)
        result = MultiverseGenerator._apply_phase_shift(deltas, shifts)
        np.testing.assert_array_equal(result, deltas)

    def test_partitioned_shifts(self):
        """Test that different partitions get different shifts."""
        K, D = 8, 64
        deltas = np.ones((K, D), dtype=np.float32)
        # First element is 1.0, rest are 0.0 — makes shifts visible
        deltas[:, :] = 0.0
        deltas[:, 0] = 1.0

        shifts = np.zeros(K, dtype=np.uint32)
        shifts[0:2] = 0     # Analytical
        shifts[2:4] = 4     # Shift by 4
        shifts[4:6] = 8     # Shift by 8
        shifts[6:8] = 16    # Shift by 16

        result = MultiverseGenerator._apply_phase_shift(deltas, shifts)

        # Analytical branches: 1.0 stays at index 0
        assert result[0, 0] == 1.0
        assert result[1, 0] == 1.0

        # Shift-4 branches: 1.0 moves from index 0 to index D-4
        assert result[2, D - 4] == 1.0
        assert result[3, D - 4] == 1.0

        # Shift-8 branches: 1.0 at index D-8
        assert result[4, D - 8] == 1.0

        # Shift-16 branches: 1.0 at index D-16
        assert result[6, D - 16] == 1.0


@pytest.mark.gpu
@pytest.mark.skipif(not GRILLY_CORE_AVAILABLE, reason="grilly_core not available")
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestMultiverseGeneratorCreation:
    """Test MultiverseGenerator construction."""

    @pytest.fixture
    def components(self):
        dev = grilly_core.Device()
        dev.load_shaders(_SHADER_DIR)
        model = grilly_core.VSAHypernetwork(
            dev, d_model=32, vsa_dim=256, K=8, seed=42
        )
        world_model = grilly_core.WorldModel(dev, dim=256)
        vsa_cache = grilly_core.VSACache(dev, dim=256, max_capacity=1000)
        return dev, model, world_model, vsa_cache

    def test_creation(self, components):
        dev, model, world_model, vsa_cache = components
        gen = MultiverseGenerator(
            dev, model, world_model, vsa_cache,
            vsa_dim=256, K=8,
        )
        assert gen.K == 8
        assert gen.vsa_dim == 256

    def test_hallucination_threshold_default(self, components):
        dev, model, world_model, vsa_cache = components
        gen = MultiverseGenerator(
            dev, model, world_model, vsa_cache,
            vsa_dim=256, K=8,
        )
        assert gen.hallucination_threshold == 256 * 0.45

    def test_hallucination_threshold_custom(self, components):
        dev, model, world_model, vsa_cache = components
        gen = MultiverseGenerator(
            dev, model, world_model, vsa_cache,
            vsa_dim=256, K=8, hallucination_threshold=50.0,
        )
        assert gen.hallucination_threshold == 50.0


@pytest.mark.gpu
@pytest.mark.skipif(not GRILLY_CORE_AVAILABLE, reason="grilly_core not available")
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestMultiverseGeneration:
    """Test MultiverseGenerator end-to-end generation."""

    @pytest.fixture
    def generator(self):
        dev = grilly_core.Device()
        dev.load_shaders(_SHADER_DIR)
        model = grilly_core.VSAHypernetwork(
            dev, d_model=32, vsa_dim=256, K=8, seed=42
        )
        world_model = grilly_core.WorldModel(dev, dim=256)
        vsa_cache = grilly_core.VSACache(dev, dim=256, max_capacity=1000)
        gen = MultiverseGenerator(
            dev, model, world_model, vsa_cache,
            vsa_dim=256, K=8,
        )
        return gen

    def test_generate_analytical(self, generator):
        """Test generation in default ANALYTICAL universe."""
        rng = np.random.RandomState(42)
        words = (256 + 31) // 32
        state = rng.randint(0, 2**32, size=words, dtype=np.uint32)

        trajectory = generator.generate_warped(
            state,
            target_universe=CognitiveUniverse.ANALYTICAL,
            max_steps=3,
        )

        assert len(trajectory) > 0
        assert len(trajectory) <= 3
        for step in trajectory:
            assert isinstance(step, MultiverseStep)
            assert step.universe == CognitiveUniverse.ANALYTICAL
            assert step.state.dtype == np.uint32
            assert step.state.shape == (words,)
            assert np.isfinite(step.coherence_score)

    def test_generate_empathetic(self, generator):
        """Test generation in EMPATHETIC universe (phi=512)."""
        rng = np.random.RandomState(42)
        words = (256 + 31) // 32
        state = rng.randint(0, 2**32, size=words, dtype=np.uint32)

        trajectory = generator.generate_warped(
            state,
            target_universe=CognitiveUniverse.EMPATHETIC,
            max_steps=3,
        )

        assert len(trajectory) > 0
        for step in trajectory:
            assert step.universe == CognitiveUniverse.EMPATHETIC

    def test_different_universes_different_trajectories(self, generator):
        """Different cognitive universes should produce different states."""
        rng = np.random.RandomState(42)
        words = (256 + 31) // 32
        state = rng.randint(0, 2**32, size=words, dtype=np.uint32)

        traj_analytical = generator.generate_warped(
            state, CognitiveUniverse.ANALYTICAL, max_steps=2
        )
        traj_skeptical = generator.generate_warped(
            state, CognitiveUniverse.SKEPTICAL, max_steps=2
        )

        # With different phase shifts, final states should differ
        if len(traj_analytical) > 0 and len(traj_skeptical) > 0:
            assert not np.array_equal(
                traj_analytical[-1].state, traj_skeptical[-1].state
            )

    def test_insert_and_decode(self, generator):
        """Test inserting text into cache and decoding it back."""
        rng = np.random.RandomState(42)
        words = (256 + 31) // 32
        vec = rng.randint(0, 2**32, size=words, dtype=np.uint32)

        # Insert with text
        ok = generator.insert_with_text("hello world", vec)
        assert ok

        # Decode should find it
        decoded = generator._decode_from_cache(vec)
        assert decoded == "hello world"

    def test_decode_unknown_returns_uncharted(self, generator):
        """Cache miss should return [Uncharted]."""
        words = (256 + 31) // 32
        unknown = np.zeros(words, dtype=np.uint32)
        decoded = generator._decode_from_cache(unknown)
        # With empty cache, should be Uncharted or Index
        assert "[" in decoded

    def test_query_api(self, generator):
        """Test the query() convenience method."""
        # Insert some facts so cache has entries
        for s, v, c in [("dog", "is", "animal"), ("sun", "is", "star")]:
            state = generator.world_model.encode_triple(s, v, c)
            generator.insert_with_text(f"{s} {v} {c}", state)

        traj = generator.query("is", universe=CognitiveUniverse.ANALYTICAL, max_steps=2)
        assert len(traj) > 0
        assert all(isinstance(s.decoded_text, str) for s in traj)

    def test_query_all_universes(self, generator):
        """Test query_all_universes() returns all 5 universes."""
        results = generator.query_all_universes("test", max_steps=1)
        assert len(results) == 5
        for universe in CognitiveUniverse:
            assert universe in results
            assert isinstance(results[universe], list)

    def test_encode_prompt(self, generator):
        """Test encode_prompt returns a valid bitpacked VSA state."""
        state = generator.encode_prompt("the dog runs fast")
        words = (256 + 31) // 32
        assert state.dtype == np.uint32
        assert state.shape == (words,)

    def test_encode_prompt_different_inputs(self, generator):
        """Different prompts should produce different states."""
        s1 = generator.encode_prompt("hello world")
        s2 = generator.encode_prompt("goodbye moon")
        assert not np.array_equal(s1, s2)

    def test_generate_from_prompt_no_context(self, generator):
        """Test generate_from_prompt without prior context."""
        trajectory = generator.generate_from_prompt(
            "the cat sat on the mat",
            target_universe=CognitiveUniverse.ANALYTICAL,
            max_steps=2,
        )
        assert len(trajectory) > 0
        for step in trajectory:
            assert isinstance(step, MultiverseStep)
            assert step.universe == CognitiveUniverse.ANALYTICAL

    def test_generate_from_prompt_with_context(self, generator):
        """Test XOR context binding produces different trajectory."""
        # First query to get context
        traj1 = generator.generate_from_prompt("hello", max_steps=1)
        assert len(traj1) > 0

        # Second query with context from first
        traj_with_ctx = generator.generate_from_prompt(
            "world", context_state=traj1[-1].state, max_steps=1
        )
        # Same query without context
        traj_no_ctx = generator.generate_from_prompt(
            "world", max_steps=1
        )

        # Context should change the trajectory
        assert len(traj_with_ctx) > 0
        assert len(traj_no_ctx) > 0
        assert not np.array_equal(
            traj_with_ctx[0].state, traj_no_ctx[0].state
        )

    def test_text_encoder_initialized(self, generator):
        """TextEncoder should be initialized with correct dim."""
        assert generator._text_encoder.dim == 256
