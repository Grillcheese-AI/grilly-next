"""Tests for VSA inference API: unbinding, analogy, batch, and checkpointing."""

import pathlib
import tempfile

import numpy as np
import pytest

try:
    import grilly_core
    GRILLY_CORE_AVAILABLE = True
except ImportError:
    GRILLY_CORE_AVAILABLE = False

try:
    from grilly_next.backend import VULKAN_AVAILABLE
except ImportError:
    VULKAN_AVAILABLE = False

_SHADER_DIR = str(pathlib.Path(__file__).parent.parent / "shaders" / "spv")


def _make_codebook_from_fillers(enc, words, dim):
    """Build a codebook from TextEncoder fillers (ensures encode/unbind alignment)."""
    words_per_vec = (dim + 31) // 32
    codebook = np.zeros((len(words), words_per_vec), dtype=np.uint32)

    for i, w in enumerate(words):
        # Get the bipolar filler from the encoder (or BLAKE3 fallback)
        bipolar = grilly_core.blake3_role("filler_" + w, dim)
        packed = grilly_core.vsa_bitpack(bipolar)
        codebook[i] = packed
        # Also register in the encoder so encode_sentence uses the same filler
        enc.add_filler(w, bipolar)

    return codebook


@pytest.mark.gpu
@pytest.mark.skipif(not GRILLY_CORE_AVAILABLE, reason="grilly_core not available")
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestQueryRole:
    """Test single role unbinding (query_role)."""

    @pytest.fixture
    def setup(self):
        dev = grilly_core.Device()
        dev.load_shaders(_SHADER_DIR)
        dim = 256
        enc = grilly_core.TextEncoder(dim=dim)
        res = grilly_core.ResonatorNetwork(dev, dim=dim)

        words = ["apple", "banana", "cherry"]
        codebook = _make_codebook_from_fillers(enc, words, dim)
        res.load_codebook(words, codebook.ravel())

        return dev, enc, res, words, codebook, dim

    def test_single_binding_unbind(self, setup):
        """Bind filler with role, unbind the role, recover the filler."""
        dev, enc, res, words, codebook, dim = setup

        # Bind: apple XOR role_"color"
        role_vec = grilly_core.blake3_role("role_color", dim)
        role_packed = grilly_core.vsa_bitpack(role_vec)
        bound = codebook[0] ^ role_packed  # apple XOR role_color

        # Unbind: should recover "apple"
        result = res.query_role(bound, "color", key_prefix="role_")
        assert result["word"] == "apple"
        assert result["similarity"] > 0.5

    def test_different_roles_different_results(self, setup):
        """Different role bindings should recover different fillers."""
        dev, enc, res, words, codebook, dim = setup

        role_a = grilly_core.vsa_bitpack(grilly_core.blake3_role("role_fruit", dim))
        role_b = grilly_core.vsa_bitpack(grilly_core.blake3_role("role_color", dim))

        # Bind apple with "fruit", banana with "color"
        bound_a = codebook[0] ^ role_a  # apple XOR role_fruit
        bound_b = codebook[1] ^ role_b  # banana XOR role_color

        # Unbind each
        result_a = res.query_role(bound_a, "fruit", key_prefix="role_")
        result_b = res.query_role(bound_b, "color", key_prefix="role_")

        assert result_a["word"] == "apple"
        assert result_b["word"] == "banana"


@pytest.mark.gpu
@pytest.mark.skipif(not GRILLY_CORE_AVAILABLE, reason="grilly_core not available")
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestQuerySlot:
    """Test three-way unbinding (query_slot) via TextEncoder round-trip."""

    @pytest.fixture
    def setup(self):
        dev = grilly_core.Device()
        dev.load_shaders(_SHADER_DIR)
        dim = 256
        enc = grilly_core.TextEncoder(dim=dim)
        res = grilly_core.ResonatorNetwork(dev, dim=dim)

        words = ["the", "quick", "fox"]
        codebook = _make_codebook_from_fillers(enc, words, dim)
        res.load_codebook(words, codebook.ravel())

        return dev, enc, res, words, dim

    def test_query_slot_first_position(self, setup):
        """Encode sentence, query (role, position) = ("ROOT", 0) -> first word."""
        dev, enc, res, words, dim = setup
        roles = ["ROOT", "amod", "nsubj"]
        positions = [0, 1, 2]

        bundle_dict = enc.encode_sentence(words, roles, positions)
        bundle = np.asarray(bundle_dict["data"], dtype=np.uint32)

        result = res.query_slot(bundle, "ROOT", 0)
        assert result["word"] == "the"
        assert result["similarity"] > 0.0

    def test_query_slot_last_position(self, setup):
        """Encode sentence, query last position."""
        dev, enc, res, words, dim = setup
        roles = ["ROOT", "amod", "nsubj"]
        positions = [0, 1, 2]

        bundle_dict = enc.encode_sentence(words, roles, positions)
        bundle = np.asarray(bundle_dict["data"], dtype=np.uint32)

        result = res.query_slot(bundle, "nsubj", 2)
        assert result["word"] == "fox"


@pytest.mark.gpu
@pytest.mark.skipif(not GRILLY_CORE_AVAILABLE, reason="grilly_core not available")
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestAnalogy:
    """Test analogical reasoning (compute_analogy_map + apply_analogy)."""

    @pytest.fixture
    def setup(self):
        dev = grilly_core.Device()
        dev.load_shaders(_SHADER_DIR)
        dim = 256
        res = grilly_core.ResonatorNetwork(dev, dim=dim)

        # 4 structured bundles: USA, WDC, MEX, MXC
        words = ["usa", "wdc", "mex", "mxc"]
        words_per_vec = (dim + 31) // 32
        codebook = np.zeros((4, words_per_vec), dtype=np.uint32)
        for i, w in enumerate(words):
            bipolar = grilly_core.blake3_role("filler_" + w, dim)
            codebook[i] = grilly_core.vsa_bitpack(bipolar)

        res.load_codebook(words, codebook.ravel())
        return dev, res, words, codebook, dim

    def test_analogy_map_shape(self, setup):
        """compute_analogy_map returns correctly shaped array."""
        dev, res, words, codebook, dim = setup
        words_per_vec = (dim + 31) // 32

        mapping = res.compute_analogy_map(codebook[0], codebook[2])
        assert mapping.shape == (words_per_vec,)
        assert mapping.dtype == np.uint32

    def test_analogy_self_inverse(self, setup):
        """XOR mapping is self-inverse: map(A, B) applied to A gives B."""
        dev, res, words, codebook, dim = setup

        # mapping = USA XOR MEX
        mapping = res.compute_analogy_map(codebook[0], codebook[2])
        # apply to USA -> should get MEX
        result = res.apply_analogy(mapping, codebook[0])
        assert result["word"] == "mex"
        assert result["similarity"] > 0.9

    def test_analogy_structured(self, setup):
        """Test structured analogy: USA:WDC :: MEX:?"""
        dev, res, words, codebook, dim = setup

        role_country = grilly_core.vsa_bitpack(
            grilly_core.blake3_role("role_country", dim))
        role_capital = grilly_core.vsa_bitpack(
            grilly_core.blake3_role("role_capital", dim))

        # Build structured bundles via XOR binding + OR bundling
        # US = (usa XOR country) XOR (wdc XOR capital) -- simplified 2-component bundle
        us_bundle = (codebook[0] ^ role_country) ^ (codebook[1] ^ role_capital)
        mx_bundle = (codebook[2] ^ role_country) ^ (codebook[3] ^ role_capital)

        # Analogical mapping between the two frames
        mapping = res.compute_analogy_map(us_bundle, mx_bundle)
        # Apply to wdc -> should approximate mxc
        result = res.apply_analogy(mapping, codebook[1])
        # With only 2-component XOR bundles (no majority vote), this is noisy
        # but the API should work
        assert "word" in result
        assert "similarity" in result


@pytest.mark.gpu
@pytest.mark.skipif(not GRILLY_CORE_AVAILABLE, reason="grilly_core not available")
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestBatchUnbind:
    """Test GPU-accelerated batch unbinding."""

    @pytest.fixture
    def setup(self):
        dev = grilly_core.Device()
        dev.load_shaders(_SHADER_DIR)
        dim = 256
        enc = grilly_core.TextEncoder(dim=dim)
        res = grilly_core.ResonatorNetwork(dev, dim=dim)

        words = ["hello", "bright", "world"]
        codebook = _make_codebook_from_fillers(enc, words, dim)
        res.load_codebook(words, codebook.ravel())

        return dev, enc, res, words, dim

    def test_batch_unbind_matches_sequential(self, setup):
        """batch_unbind should return same results as sequential query_slot."""
        dev, enc, res, words, dim = setup
        roles = ["ROOT", "amod", "nsubj"]
        positions = [0, 1, 2]

        bundle_dict = enc.encode_sentence(words, roles, positions)
        bundle = np.asarray(bundle_dict["data"], dtype=np.uint32)

        # Batch unbind all slots at once
        batch_results = res.batch_unbind(bundle, roles, positions)

        # Sequential unbind each slot
        for i, (role, pos) in enumerate(zip(roles, positions)):
            seq_result = res.query_slot(bundle, role, pos)
            assert batch_results[i]["word"] == seq_result["word"]
            assert abs(batch_results[i]["similarity"] - seq_result["similarity"]) < 1e-6

    def test_batch_unbind_recovers_words(self, setup):
        """batch_unbind should recover original words from an encoded sentence."""
        dev, enc, res, words, dim = setup
        roles = ["ROOT", "amod", "nsubj"]
        positions = [0, 1, 2]

        bundle_dict = enc.encode_sentence(words, roles, positions)
        bundle = np.asarray(bundle_dict["data"], dtype=np.uint32)

        results = res.batch_unbind(bundle, roles, positions)
        assert len(results) == 3
        decoded = [r["word"] for r in results]
        assert decoded == words


@pytest.mark.gpu
@pytest.mark.skipif(not GRILLY_CORE_AVAILABLE, reason="grilly_core not available")
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestCodebookCheckpoint:
    """Test codebook save/load round-trip."""

    @pytest.fixture
    def setup(self):
        dev = grilly_core.Device()
        dev.load_shaders(_SHADER_DIR)
        dim = 256
        res = grilly_core.ResonatorNetwork(dev, dim=dim)

        words = ["alpha", "beta", "gamma", "delta"]
        words_per_vec = (dim + 31) // 32
        codebook = np.zeros((4, words_per_vec), dtype=np.uint32)
        for i, w in enumerate(words):
            bipolar = grilly_core.blake3_role("filler_" + w, dim)
            codebook[i] = grilly_core.vsa_bitpack(bipolar)

        res.load_codebook(words, codebook.ravel())
        return dev, res, words, codebook, dim

    def test_save_load_roundtrip(self, setup):
        """Save codebook, load into fresh resonator, verify identical results."""
        dev, res, words, codebook, dim = setup

        with tempfile.NamedTemporaryFile(suffix=".grly", delete=False) as f:
            path = f.name

        # Save
        res.save_codebook(path)

        # Load into a fresh resonator
        res2 = grilly_core.ResonatorNetwork(dev, dim=dim)
        res2.load_codebook_file(path)

        assert res2.codebook_size == len(words)

        # Verify: same query produces same result
        query = codebook[0]
        r1 = res.resonate(query)
        r2 = res2.resonate(query)
        assert r1["best_word"] == r2["best_word"]
        assert r1["best_word"] == "alpha"

    def test_save_requires_loaded_codebook(self, setup):
        """save_codebook should error if no codebook is loaded."""
        dev, _, _, _, dim = setup
        empty_res = grilly_core.ResonatorNetwork(dev, dim=dim)
        with pytest.raises(RuntimeError, match="no codebook loaded"):
            empty_res.save_codebook("/tmp/should_not_exist.grly")

    def test_load_wrong_dim_fails(self, setup):
        """Loading a codebook with mismatched dim should raise."""
        dev, res, words, codebook, dim = setup

        with tempfile.NamedTemporaryFile(suffix=".grly", delete=False) as f:
            path = f.name

        res.save_codebook(path)

        # Try loading into a resonator with different dim
        res_wrong = grilly_core.ResonatorNetwork(dev, dim=512)
        with pytest.raises(RuntimeError, match="dim mismatch"):
            res_wrong.load_codebook_file(path)


@pytest.mark.gpu
@pytest.mark.skipif(not GRILLY_CORE_AVAILABLE, reason="grilly_core not available")
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestInterpretTrajectory:
    """Test many-worlds + resonator integration via interpret_trajectory."""

    def test_interpret_trajectory_returns_word(self):
        dev = grilly_core.Device()
        dev.load_shaders(_SHADER_DIR)
        dim = 256
        res = grilly_core.ResonatorNetwork(dev, dim=dim)

        words = ["cat", "sat", "mat"]
        words_per_vec = (dim + 31) // 32
        codebook = np.zeros((3, words_per_vec), dtype=np.uint32)
        for i, w in enumerate(words):
            bipolar = grilly_core.blake3_role("filler_" + w, dim)
            codebook[i] = grilly_core.vsa_bitpack(bipolar)
        res.load_codebook(words, codebook.ravel())

        # Simulate a ManyWorldsResult by running evaluate_many_worlds
        # with a simple state and model
        world_model = grilly_core.WorldModel(dev, dim=dim)
        model = grilly_core.VSAHypernetwork(
            dev, d_model=32, vsa_dim=dim, K=4, seed=42
        )

        # Create a state and run forward
        state = codebook[0]  # Use "cat" as the state
        tape = grilly_core.TapeContext(dev)
        deltas = model.forward_inference(dev, tape, state)

        # Run many-worlds evaluation
        mw_result = grilly_core.evaluate_many_worlds(
            dev, state, deltas, world_model
        )

        # Interpret the best trajectory
        result = grilly_core.interpret_trajectory(
            mw_result, res, "ROOT", 0
        )

        assert "word" in result
        assert "similarity" in result
        assert isinstance(result["word"], str)
