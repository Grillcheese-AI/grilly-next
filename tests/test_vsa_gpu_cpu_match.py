"""Tests that verify GPU vsa-bmm shader produces bitwise identical output to a CPU reference."""

import os
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


def cpu_binary_mlp(input_packed, w1_packed, w2_packed, state_words, hidden_words):
    """CPU reference: 2-layer binary MLP with XNOR + POPCNT + threshold + final XNOR bind."""
    state_dim = state_words * 32
    hidden_dim = hidden_words * 32

    # Layer 1: input -> hidden
    hidden_out = np.zeros(hidden_words, dtype=np.uint32)
    for h_word in range(hidden_words):
        h_val = 0
        for bit in range(32):
            neuron = h_word * 32 + bit
            w_offset = neuron * state_words
            match_count = 0
            for i in range(state_words):
                xnor = ~(int(input_packed[i]) ^ int(w1_packed[w_offset + i])) & 0xFFFFFFFF
                match_count += bin(xnor).count("1")
            if match_count > state_dim // 2:
                h_val |= 1 << bit
        hidden_out[h_word] = np.uint32(h_val)

    # Layer 2: hidden -> output
    output = np.zeros(state_words, dtype=np.uint32)
    for o_word in range(state_words):
        o_val = 0
        for bit in range(32):
            neuron = o_word * 32 + bit
            w_offset = neuron * hidden_words
            match_count = 0
            for i in range(hidden_words):
                xnor = ~(int(hidden_out[i]) ^ int(w2_packed[w_offset + i])) & 0xFFFFFFFF
                match_count += bin(xnor).count("1")
            if match_count > hidden_dim // 2:
                o_val |= 1 << bit
        # XNOR bind with input
        output[o_word] = np.uint32((~(int(input_packed[o_word]) ^ o_val)) & 0xFFFFFFFF)

    return output


@pytest.mark.gpu
@pytest.mark.skipif(not GRILLY_CORE_AVAILABLE, reason="grilly_core not available")
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestGPUCPUMatch:
    """Verify GPU vsa-bmm shader output is bitwise identical to CPU reference."""

    def test_mlp_output_matches_cpu(self):
        """GPU binary MLP must produce bitwise identical output to the CPU reference."""
        # Use dim=256 for speed (CPU ref is O(dim^2 * hidden_dim) with Python loops)
        dim = 256
        hidden = 128
        state_words = dim // 32  # 8
        hidden_words = hidden // 32  # 4

        # Create random weights with fixed seed for reproducibility
        rng = np.random.default_rng(42)
        w1 = rng.integers(0, 2**32, size=hidden * state_words, dtype=np.uint32)
        w2 = rng.integers(0, 2**32, size=dim * hidden_words, dtype=np.uint32)
        weights = np.concatenate([w1, w2])

        # Random input
        input_packed = rng.integers(0, 2**32, size=state_words, dtype=np.uint32)

        # CPU reference
        cpu_output = cpu_binary_mlp(input_packed, w1, w2, state_words, hidden_words)

        # GPU inference
        dev = grilly_core.Device()
        dev.load_shaders(_SHADER_DIR)

        engine = grilly_core.VSABaremetalEngine(
            dev, state_dim=dim, hidden_dim=hidden
        )

        # Write weights to temp file
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            weights.tofile(f)
            weights_path = f.name

        # Need a dummy codebook for step() to work
        vocab = ["dummy"]
        codebook = rng.integers(0, 2**32, size=state_words, dtype=np.uint32)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            codebook.tofile(f)
            codebook_path = f.name

        try:
            engine.load_logic_weights(weights_path)
            engine.load_codebook(codebook_path, vocab)
            result = engine.step(dev, input_packed)
            gpu_output = np.array(result["predicted_state"], dtype=np.uint32)
        finally:
            os.unlink(weights_path)
            os.unlink(codebook_path)

        np.testing.assert_array_equal(gpu_output, cpu_output)
