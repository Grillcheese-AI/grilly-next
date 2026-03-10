"""Tests for SelfBinarizingMLP: tanh annealing, gradient flow, binary convergence."""

import numpy as np
import pytest

# We'll import from the script after implementation
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class TestSelfBinarizingMLP:
    """Verify SelfBinarizingMLP produces correct behavior at different sharpness levels."""

    def test_output_shape(self):
        """Forward pass produces correct output shape."""
        from poc_simhash_vsa import SelfBinarizingMLP
        model = SelfBinarizingMLP(dim=64, hidden=32, seed=42)
        x = np.sign(np.random.randn(8, 64)).astype(np.float32)
        x[x == 0] = 1.0
        out = model.forward(x)
        assert out.data.shape == (8, 64)

    def test_smooth_at_low_sharpness(self):
        """At v=1, outputs are smooth (not saturated at +/-1)."""
        from poc_simhash_vsa import SelfBinarizingMLP
        model = SelfBinarizingMLP(dim=64, hidden=32, seed=42)
        model.sharpness = 1.0
        x = np.sign(np.random.randn(16, 64)).astype(np.float32)
        x[x == 0] = 1.0
        out = model.forward(x)
        # At v=1, most outputs should NOT be saturated at exactly +/-1
        abs_vals = np.abs(out.data)
        frac_saturated = np.mean(abs_vals > 0.99)
        assert frac_saturated < 0.5, f"Too many saturated outputs at v=1: {frac_saturated:.2f}"

    def test_near_binary_at_high_sharpness(self):
        """At v=10, outputs approximate sign() — nearly all +/-1."""
        from poc_simhash_vsa import SelfBinarizingMLP
        model = SelfBinarizingMLP(dim=64, hidden=32, seed=42)
        model.sharpness = 10.0
        x = np.sign(np.random.randn(16, 64)).astype(np.float32)
        x[x == 0] = 1.0
        out = model.forward(x)
        abs_vals = np.abs(out.data)
        frac_binary = np.mean(abs_vals > 0.99)
        assert frac_binary > 0.9, f"Not enough binary outputs at v=10: {frac_binary:.2f}"

    def test_gradients_are_finite(self):
        """Backward pass produces finite, non-zero gradients on all weights."""
        from poc_simhash_vsa import SelfBinarizingMLP
        model = SelfBinarizingMLP(dim=64, hidden=32, seed=42)
        model.sharpness = 1.0
        x = np.sign(np.random.randn(8, 64)).astype(np.float32)
        x[x == 0] = 1.0
        target = np.sign(np.random.randn(8, 64)).astype(np.float32)
        target[target == 0] = 1.0

        out = model.forward(x)
        # Simple MSE-like gradient
        grad = (out.data - target) / target.size
        out.backward(grad)

        for name, p in zip(["w1", "w2", "w3", "w4"], model.parameters()):
            assert p.grad is not None, f"{name}.grad is None"
            assert np.all(np.isfinite(p.grad)), f"{name}.grad has non-finite values"
            assert np.any(p.grad != 0), f"{name}.grad is all zeros"

    def test_gradients_no_explosion(self):
        """At v=1, gradient norms stay reasonable (no STE explosion)."""
        from poc_simhash_vsa import SelfBinarizingMLP
        model = SelfBinarizingMLP(dim=256, hidden=128, seed=42)
        model.sharpness = 1.0
        x = np.sign(np.random.randn(32, 256)).astype(np.float32)
        x[x == 0] = 1.0
        target = np.sign(np.random.randn(32, 256)).astype(np.float32)
        target[target == 0] = 1.0

        out = model.forward(x)
        grad = (out.data - target) / target.size
        out.backward(grad)

        for name, p in zip(["w1", "w2", "w3", "w4"], model.parameters()):
            gnorm = np.sqrt(np.sum(p.grad ** 2))
            assert gnorm < 100.0, f"{name} gradient norm exploded: {gnorm:.1f}"

    def test_weight_export_binary(self):
        """sign(W_float) produces valid bipolar weights for GPU export."""
        from poc_simhash_vsa import SelfBinarizingMLP
        model = SelfBinarizingMLP(dim=64, hidden=32, seed=42)
        for p in model.parameters():
            w_bin = np.sign(p.data)
            w_bin[w_bin == 0] = 1.0
            unique = set(np.unique(w_bin))
            assert unique <= {-1.0, 1.0}, f"Export produced non-bipolar values: {unique}"
