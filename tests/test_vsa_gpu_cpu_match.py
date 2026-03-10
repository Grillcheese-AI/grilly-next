"""Tests that verify GPU vsa-bmm-residual shader produces bitwise identical output to a CPU reference."""

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


def cpu_residual_binary_mlp(input_packed, weights_flat, state_words, hidden_words):
    """CPU reference: 4-layer residual binary MLP with XNOR + POPCNT."""
    state_dim = state_words * 32
    hidden_dim = hidden_words * 32

    # Weight offsets (contiguous: w1, w2, w3, w4)
    w1_off = 0
    w2_off = hidden_dim * state_words
    w3_off = w2_off + hidden_dim * hidden_words
    w4_off = w3_off + hidden_dim * hidden_words

    def xnor_popcnt_layer(inp, inp_words, w_data, w_off, out_words, threshold):
        out = np.zeros(out_words, dtype=np.uint32)
        for o_word in range(out_words):
            o_val = 0
            for bit in range(32):
                neuron = o_word * 32 + bit
                mc = 0
                wo = w_off + neuron * inp_words
                for i in range(inp_words):
                    xnor = ~(int(inp[i]) ^ int(w_data[wo + i])) & 0xFFFFFFFF
                    mc += bin(xnor).count("1")
                if mc > threshold:
                    o_val |= (1 << bit)
            out[o_word] = np.uint32(o_val)
        return out

    # Layer 1: input -> hidden
    h1 = xnor_popcnt_layer(input_packed, state_words, weights_flat, w1_off,
                            hidden_words, state_dim // 2)

    # Layer 2: hidden -> hidden + residual XNOR (bipolar multiply)
    h2_raw = xnor_popcnt_layer(h1, hidden_words, weights_flat, w2_off,
                                hidden_words, hidden_dim // 2)
    h2 = np.array([np.uint32((~(int(a) ^ int(b))) & 0xFFFFFFFF) for a, b in zip(h2_raw, h1)], dtype=np.uint32)

    # Layer 3: hidden -> hidden + residual XNOR (bipolar multiply)
    h3_raw = xnor_popcnt_layer(h2, hidden_words, weights_flat, w3_off,
                                hidden_words, hidden_dim // 2)
    h3 = np.array([np.uint32((~(int(a) ^ int(b))) & 0xFFFFFFFF) for a, b in zip(h3_raw, h2)], dtype=np.uint32)

    # Layer 4: hidden -> output + XNOR bind with input
    out = xnor_popcnt_layer(h3, hidden_words, weights_flat, w4_off,
                             state_words, hidden_dim // 2)
    output = np.array([np.uint32((~(int(a) ^ int(b))) & 0xFFFFFFFF)
                       for a, b in zip(input_packed, out)], dtype=np.uint32)
    return output


@pytest.mark.gpu
@pytest.mark.skipif(not GRILLY_CORE_AVAILABLE, reason="grilly_core not available")
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestGPUCPUMatchResidual:
    """Verify GPU vsa-bmm-residual shader output is bitwise identical to CPU reference."""

    def test_residual_mlp_matches_cpu(self):
        """GPU 4-layer residual binary MLP must match CPU reference."""
        dim = 256
        hidden = 128
        sw = dim // 32    # 8
        hw = hidden // 32  # 4

        rng = np.random.default_rng(99)

        # Weights: W1(hidden*sw) + W2(hidden*hw) + W3(hidden*hw) + W4(dim*hw)
        w1 = rng.integers(0, 2**32, size=hidden * sw, dtype=np.uint32)
        w2 = rng.integers(0, 2**32, size=hidden * hw, dtype=np.uint32)
        w3 = rng.integers(0, 2**32, size=hidden * hw, dtype=np.uint32)
        w4 = rng.integers(0, 2**32, size=dim * hw, dtype=np.uint32)
        weights = np.concatenate([w1, w2, w3, w4])

        input_packed = rng.integers(0, 2**32, size=sw, dtype=np.uint32)

        cpu_output = cpu_residual_binary_mlp(input_packed, weights, sw, hw)

        dev = grilly_core.Device()
        dev.load_shaders(_SHADER_DIR)
        engine = grilly_core.VSABaremetalEngine(dev, state_dim=dim, hidden_dim=hidden)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            weights.tofile(f)
            wp = f.name

        vocab = ["dummy"]
        cb = rng.integers(0, 2**32, size=sw, dtype=np.uint32)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            cb.tofile(f)
            cp = f.name

        try:
            engine.load_logic_weights(wp)
            engine.load_codebook(cp, vocab)
            result = engine.step(dev, input_packed)
            gpu_output = np.array(result["predicted_state"], dtype=np.uint32)
        finally:
            os.unlink(wp)
            os.unlink(cp)

        np.testing.assert_array_equal(gpu_output, cpu_output)

    def test_residual_mlp_dim2048(self):
        """GPU matches CPU at production dimensions (dim=2048, hidden=1024)."""
        dim = 2048
        hidden = 1024
        sw = dim // 32    # 64
        hw = hidden // 32  # 32

        rng = np.random.default_rng(42)

        w1 = rng.integers(0, 2**32, size=hidden * sw, dtype=np.uint32)
        w2 = rng.integers(0, 2**32, size=hidden * hw, dtype=np.uint32)
        w3 = rng.integers(0, 2**32, size=hidden * hw, dtype=np.uint32)
        w4 = rng.integers(0, 2**32, size=dim * hw, dtype=np.uint32)
        weights = np.concatenate([w1, w2, w3, w4])

        input_packed = rng.integers(0, 2**32, size=sw, dtype=np.uint32)

        cpu_output = cpu_residual_binary_mlp(input_packed, weights, sw, hw)

        dev = grilly_core.Device()
        dev.load_shaders(_SHADER_DIR)
        engine = grilly_core.VSABaremetalEngine(dev, state_dim=dim, hidden_dim=hidden)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            weights.tofile(f)
            wp = f.name

        vocab = ["dummy"]
        cb = rng.integers(0, 2**32, size=sw, dtype=np.uint32)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            cb.tofile(f)
            cp = f.name

        try:
            engine.load_logic_weights(wp)
            engine.load_codebook(cp, vocab)
            result = engine.step(dev, input_packed)
            gpu_output = np.array(result["predicted_state"], dtype=np.uint32)
        finally:
            os.unlink(wp)
            os.unlink(cp)

        np.testing.assert_array_equal(gpu_output, cpu_output)

