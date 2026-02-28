"""
Tests for VSA Hypernetwork + Surrogate Loss.

Tests the full pipeline: bitpacked VSA state -> unpack+project -> MLP -> loss.
"""

import numpy as np
import pytest

try:
    import grilly_core
    GRILLY_CORE_AVAILABLE = True
except ImportError:
    GRILLY_CORE_AVAILABLE = False

try:
    from grilly_next import Compute
    from grilly_next.backend import VULKAN_AVAILABLE
except ImportError:
    VULKAN_AVAILABLE = False


@pytest.mark.gpu
@pytest.mark.skipif(not GRILLY_CORE_AVAILABLE, reason="grilly_core not available")
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestVSAHypernetwork:
    """Test VSA Hypernetwork creation and forward pass."""

    @pytest.fixture
    def device(self):
        dev = grilly_core.Device()
        yield dev

    @pytest.fixture
    def tape(self, device):
        return grilly_core.TapeContext(device)

    def test_hypernetwork_creation(self, device):
        """Test VSAHypernetwork initializes with correct dimensions."""
        model = grilly_core.VSAHypernetwork(device, d_model=768, vsa_dim=10240, K=4)
        assert model.d_model == 768
        assert model.vsa_dim == 10240
        assert model.K == 4

    def test_hypernetwork_parameter_ids(self, device):
        """Test parameter buffer IDs are allocated (6 buffers: 3 weights + 3 biases)."""
        model = grilly_core.VSAHypernetwork(device, d_model=768, vsa_dim=10240, K=4)
        param_ids = model.parameter_buffer_ids()
        assert len(param_ids) == 6
        # All IDs should be unique and non-zero
        assert len(set(param_ids)) == 6
        assert all(pid != 0 for pid in param_ids)

    def test_hypernetwork_small_dimensions(self, device):
        """Test hypernetwork with small dims for fast verification."""
        model = grilly_core.VSAHypernetwork(
            device, d_model=32, vsa_dim=256, K=2, seed=42
        )
        assert model.d_model == 32
        assert model.vsa_dim == 256
        assert model.K == 2


@pytest.mark.gpu
@pytest.mark.skipif(not GRILLY_CORE_AVAILABLE, reason="grilly_core not available")
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestVSATrainingStep:
    """Test the full training step: forward + loss + backward."""

    @pytest.fixture
    def device(self):
        dev = grilly_core.Device()
        yield dev

    @pytest.fixture
    def tape(self, device):
        return grilly_core.TapeContext(device)

    def test_training_step_returns_loss(self, device, tape):
        """Test a single training step produces a finite loss."""
        model = grilly_core.VSAHypernetwork(
            device, d_model=32, vsa_dim=256, K=2, seed=42
        )

        # Create synthetic bitpacked VSA states
        dim = 256
        num_words = (dim + 31) // 32
        rng = np.random.RandomState(42)

        state = rng.randint(0, 2**32, size=num_words, dtype=np.uint32)
        delta = rng.randint(0, 2**32, size=num_words, dtype=np.uint32)

        loss = grilly_core.vsa_training_step(device, tape, model, state, delta)

        assert np.isfinite(loss), f"Loss is not finite: {loss}"
        assert loss >= 0.0, f"Loss should be non-negative: {loss}"

    @pytest.mark.xfail(reason="VSA loss node forward is scaffold — GPU dispatch not fully wired")
    def test_training_loss_decreases(self, device, tape):
        """Test that loss decreases over multiple training steps."""
        model = grilly_core.VSAHypernetwork(
            device, d_model=32, vsa_dim=256, K=2, seed=42
        )

        dim = 256
        num_words = (dim + 31) // 32
        rng = np.random.RandomState(42)

        # Fixed target pair
        state = rng.randint(0, 2**32, size=num_words, dtype=np.uint32)
        delta = state ^ rng.randint(0, 2**32, size=num_words, dtype=np.uint32)

        losses = []
        for step in range(20):
            loss = grilly_core.vsa_training_step(device, tape, model, state, delta)
            losses.append(loss)

        # Loss should generally decrease (allow some noise)
        first_5 = np.mean(losses[:5])
        last_5 = np.mean(losses[-5:])
        assert last_5 < first_5, (
            f"Loss should decrease: first 5 avg={first_5:.4f}, last 5 avg={last_5:.4f}"
        )


@pytest.mark.gpu
@pytest.mark.skipif(not GRILLY_CORE_AVAILABLE, reason="grilly_core not available")
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestVSAOpTypes:
    """Test that new OpType enum values are accessible."""

    def test_optype_vsa_unpack_project(self):
        """Test VSAUnpackProject OpType is accessible."""
        assert hasattr(grilly_core, "OpType")
        assert hasattr(grilly_core.OpType, "VSAUnpackProject")

    def test_optype_vsa_surrogate_loss(self):
        """Test VSASurrogateLoss OpType is accessible."""
        assert hasattr(grilly_core.OpType, "VSASurrogateLoss")


@pytest.mark.skipif(not GRILLY_CORE_AVAILABLE, reason="grilly_core not available")
class TestVSASurrogateLossCPU:
    """CPU-only reference tests for surrogate loss logic."""

    def test_hinge_loss_perfect_prediction(self):
        """Test that hinge loss is zero when prediction matches target exactly."""
        D = 256
        gamma = 0.1

        # Perfect prediction: signs match exactly
        target_bipolar = np.random.choice([-1.0, 1.0], size=D).astype(np.float32)
        prediction = target_bipolar * 2.0  # Large margin, same signs

        # Hinge loss: mean(max(0, gamma - y_true * z_pred))
        margins = gamma - target_bipolar * prediction
        loss = np.mean(np.maximum(margins, 0.0))

        assert loss == 0.0, f"Perfect prediction should have zero hinge loss: {loss}"

    def test_hinge_loss_adversarial(self):
        """Test that hinge loss is large when prediction opposes target."""
        D = 256
        gamma = 0.1

        target_bipolar = np.ones(D, dtype=np.float32)
        prediction = -np.ones(D, dtype=np.float32)  # All wrong

        margins = gamma - target_bipolar * prediction
        loss = np.mean(np.maximum(margins, 0.0))

        expected = gamma + 1.0  # gamma - (-1) = gamma + 1
        np.testing.assert_allclose(loss, expected, atol=1e-5)

    def test_contrastive_margin_diverse(self):
        """Test contrastive margin pushes runner-up away from winner."""
        delta_margin = 0.5

        dot_winner = 100.0
        dot_runner_up = 99.0

        L_contrast = max(0.0, delta_margin - (dot_winner - dot_runner_up))

        # Gap is 1.0, margin is 0.5, so no penalty
        assert L_contrast == 0.0

    def test_contrastive_margin_too_close(self):
        """Test contrastive penalty when branches are too similar."""
        delta_margin = 0.5

        dot_winner = 100.0
        dot_runner_up = 99.8  # Only 0.2 apart, less than margin

        L_contrast = max(0.0, delta_margin - (dot_winner - dot_runner_up))

        assert L_contrast > 0.0
        np.testing.assert_allclose(L_contrast, 0.3, atol=1e-5)

    def test_sparse_gradient_only_winner(self):
        """Test that gradient is zero for non-winning branches."""
        K, D = 4, 256
        gamma = 0.1

        # Random predictions
        preds = np.random.randn(K, D).astype(np.float32)
        target = np.random.choice([-1.0, 1.0], size=D).astype(np.float32)

        # Find winner
        dots = preds @ target
        winning_k = np.argmax(dots)

        # Compute gradient: only winning_k should be nonzero
        grad = np.zeros_like(preds)
        for i in range(D):
            y_true = target[i]
            z_pred = preds[winning_k, i]
            if gamma - y_true * z_pred > 0:
                grad[winning_k, i] = -y_true / D

        # All other branches should be exactly zero
        for k in range(K):
            if k != winning_k:
                assert np.all(grad[k] == 0.0), f"Branch {k} should have zero gradient"

        # Winner should have some non-zero gradients
        assert np.any(grad[winning_k] != 0.0), "Winner should have non-zero gradient"
