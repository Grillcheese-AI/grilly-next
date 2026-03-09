"""Tests for VSABaremetalEngine: load real weights + codebook, run inference."""

import pathlib

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
_MODEL_DIR = pathlib.Path(__file__).parent.parent / "model"
_STUDENT_BIN = _MODEL_DIR / "student" / "cubemind_student.bin"
_CODEBOOK_BIN = _MODEL_DIR / "codebooks" / "vsa_codebook.bin"


def _load_vocabulary(model_dir: pathlib.Path) -> list[str]:
    """Load vocabulary from vocab.txt (one word per line) next to the codebook."""
    vocab_path = model_dir / "codebooks" / "vocab.txt"
    if not vocab_path.exists():
        return []
    with open(vocab_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


@pytest.mark.gpu
@pytest.mark.skipif(not GRILLY_CORE_AVAILABLE, reason="grilly_core not available")
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestVSABaremetalEngine:
    """Integration tests for the two-shader bare-metal VSA inference pipeline."""

    @pytest.fixture
    def device(self):
        dev = grilly_core.Device()
        dev.load_shaders(_SHADER_DIR)
        return dev

    @pytest.fixture
    def vocabulary(self):
        return _load_vocabulary(_MODEL_DIR)

    # ── Loading Tests ────────────────────────────────────────────────────

    @pytest.mark.skipif(not _STUDENT_BIN.exists(), reason="cubemind_student.bin not found")
    def test_load_logic_weights(self, device):
        """Load student logic weights from disk."""
        engine = grilly_core.VSABaremetalEngine(device, state_dim=10240)
        engine.load_logic_weights(str(_STUDENT_BIN))
        assert not engine.ready  # codebook not loaded yet

    @pytest.mark.skipif(not _CODEBOOK_BIN.exists(), reason="vsa_codebook.bin not found")
    def test_load_codebook(self, device, vocabulary):
        """Load codebook from disk with vocabulary."""
        if not vocabulary:
            pytest.skip("vocab.txt not found")

        engine = grilly_core.VSABaremetalEngine(device, state_dim=10240)
        engine.load_codebook(str(_CODEBOOK_BIN), vocabulary)
        assert engine.vocab_size == len(vocabulary)
        assert not engine.ready  # logic weights not loaded yet

    @pytest.mark.skipif(
        not (_STUDENT_BIN.exists() and _CODEBOOK_BIN.exists()),
        reason="model files not found",
    )
    def test_ready_after_both_loads(self, device, vocabulary):
        """Engine reports ready after both weights and codebook are loaded."""
        if not vocabulary:
            pytest.skip("vocab.txt not found")

        engine = grilly_core.VSABaremetalEngine(device, state_dim=10240)
        engine.load_logic_weights(str(_STUDENT_BIN))
        engine.load_codebook(str(_CODEBOOK_BIN), vocabulary)
        assert engine.ready
        assert engine.state_dim == 10240

    # ── Inference Tests ──────────────────────────────────────────────────

    @pytest.mark.skipif(
        not (_STUDENT_BIN.exists() and _CODEBOOK_BIN.exists()),
        reason="model files not found",
    )
    def test_single_step(self, device, vocabulary):
        """Run a single inference step and verify output structure."""
        if not vocabulary:
            pytest.skip("vocab.txt not found")

        engine = grilly_core.VSABaremetalEngine(device, state_dim=10240)
        engine.load_logic_weights(str(_STUDENT_BIN))
        engine.load_codebook(str(_CODEBOOK_BIN), vocabulary)

        # Create a random input state (320 uint32 words for d=10240)
        rng = np.random.default_rng(42)
        input_state = rng.integers(0, 2**32, size=320, dtype=np.uint32)

        result = engine.step(device, input_state)

        assert "word" in result
        assert "distance" in result
        assert "predicted_state" in result
        assert isinstance(result["word"], str)
        assert isinstance(result["distance"], int)
        assert len(result["predicted_state"]) == 320

    @pytest.mark.skipif(
        not (_STUDENT_BIN.exists() and _CODEBOOK_BIN.exists()),
        reason="model files not found",
    )
    def test_generate_tokens(self, device, vocabulary):
        """Run autoregressive generation and verify output."""
        if not vocabulary:
            pytest.skip("vocab.txt not found")

        engine = grilly_core.VSABaremetalEngine(device, state_dim=10240)
        engine.load_logic_weights(str(_STUDENT_BIN))
        engine.load_codebook(str(_CODEBOOK_BIN), vocabulary)

        rng = np.random.default_rng(42)
        input_state = rng.integers(0, 2**32, size=320, dtype=np.uint32)

        words = engine.generate(device, input_state, max_tokens=5)

        assert isinstance(words, list)
        assert len(words) <= 5
        assert all(isinstance(w, str) for w in words)

    @pytest.mark.skipif(
        not (_STUDENT_BIN.exists() and _CODEBOOK_BIN.exists()),
        reason="model files not found",
    )
    def test_get_word(self, device, vocabulary):
        """get_word returns correct vocabulary entries."""
        if not vocabulary:
            pytest.skip("vocab.txt not found")

        engine = grilly_core.VSABaremetalEngine(device, state_dim=10240)
        engine.load_logic_weights(str(_STUDENT_BIN))
        engine.load_codebook(str(_CODEBOOK_BIN), vocabulary)

        assert engine.get_word(0) == vocabulary[0]
        assert engine.get_word(len(vocabulary) - 1) == vocabulary[-1]
        # Out-of-bounds returns <UNK>
        assert engine.get_word(999_999) == "<UNK>"

    # ── Synthetic (No Model Files) Tests ─────────────────────────────────

    def test_synthetic_step_cpu_fallback(self, device):
        """Test the CPU fallback path with synthetic data (no real model files)."""
        dim = 256
        words_per_vec = dim // 32  # 8

        engine = grilly_core.VSABaremetalEngine(device, state_dim=dim)

        # Create synthetic logic weights (single vector for XNOR)
        logic_weights = np.ones(words_per_vec, dtype=np.uint32) * 0xAAAAAAAA

        # Create synthetic codebook: 3 entries
        vocab = ["hello", "world", "<EOS>"]
        codebook = np.zeros((3, words_per_vec), dtype=np.uint32)
        codebook[0] = 0x55555555  # complement of logic weights
        codebook[1] = 0xAAAAAAAA  # same as logic weights
        codebook[2] = 0x00000000

        # Load via array APIs
        engine.load_logic_weights_array = None  # check if array API exists
        # For now, write temp files
        import tempfile, os

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            logic_weights.tofile(f)
            logic_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            codebook.ravel().tofile(f)
            codebook_path = f.name

        try:
            engine.load_logic_weights(logic_path)
            engine.load_codebook(codebook_path, vocab)
            assert engine.ready
            assert engine.vocab_size == 3

            # Run a step with all-ones input
            input_state = np.ones(words_per_vec, dtype=np.uint32) * 0xFFFFFFFF
            result = engine.step(device, input_state)

            assert "word" in result
            assert "distance" in result
            assert isinstance(result["word"], str)
        finally:
            os.unlink(logic_path)
            os.unlink(codebook_path)


@pytest.mark.gpu
@pytest.mark.skipif(not GRILLY_CORE_AVAILABLE, reason="grilly_core not available")
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestVSAInferenceEngine:
    """Tests for the simpler single-shader VSAInferenceEngine."""

    def test_load_and_infer_synthetic(self):
        """Load synthetic weights, run XNOR inference, verify output shape."""
        dev = grilly_core.Device()
        dev.load_shaders(_SHADER_DIR)

        dim = 256
        words_per_vec = dim // 32

        engine = grilly_core.VSAInferenceEngine(dev, state_dim=dim)

        # Synthetic weights
        weights = np.ones(words_per_vec, dtype=np.uint32) * 0xAAAAAAAA

        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            weights.tofile(f)
            path = f.name

        try:
            engine.load_weights(path)
            assert engine.weights_loaded

            input_state = np.ones(words_per_vec, dtype=np.uint32) * 0xFFFFFFFF
            result = engine.infer(dev, input_state)

            assert "data" in result
            assert len(result["data"]) == words_per_vec
        finally:
            os.unlink(path)

    def test_error_without_weights(self):
        """Inference without loaded weights should raise."""
        dev = grilly_core.Device()
        dev.load_shaders(_SHADER_DIR)

        engine = grilly_core.VSAInferenceEngine(dev, state_dim=256)
        assert not engine.weights_loaded

        input_state = np.zeros(8, dtype=np.uint32)
        with pytest.raises(RuntimeError, match="weights not loaded"):
            engine.infer(dev, input_state)
