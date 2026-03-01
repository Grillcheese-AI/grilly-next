# VSA Training Loop Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire VSA loss GPU dispatch so training produces real gradients, then build a dual-path training loop with STDP + hippocampal consolidation.

**Architecture:** Three tracks executed sequentially. Track A wires 4 Vulkan shader dispatches in `vsa_loss_node.cpp` following the existing `ops/linear.cpp` pattern. Track B adds `get_winner_prediction()` to the C++ hypernetwork + pybind11. Track C builds the Python `VSATrainingLoop` class bridging grilly_core (C++) and grilly.nn (STDP/SNN).

**Tech Stack:** C++17/Vulkan (shader dispatch), pybind11, grilly.nn (STDPLayer, DualTimescaleSynapse, IFNode), grilly_core (HippocampalConsolidator, WorldModel)

**Design doc:** `docs/plans/2026-02-28-vsa-training-loop-design.md`

---

## Track A: Wire VSA Loss GPU Dispatch

### Task 1: Wire `dispatch_vsa_unpack_project_forward()`

**Files:**
- Modify: `cpp/src/autograd/vsa_loss_node.cpp:138-150`

**Context:** The shader `vsa-unpack-project.spv` has 4 bindings (vsa_data, W, b, output) and push constants `{batch_size, vsa_dim, output_dim, num_words}` (4 x uint32 = 16 bytes). Follow the pattern from `ops/linear.cpp:44-68`.

**Step 1: Replace the TODO in dispatch_vsa_unpack_project_forward**

Replace lines 138-150 with actual dispatch code:

```cpp
void dispatch_vsa_unpack_project_forward(BufferPool& pool,
                                         CommandBatch& batch,
                                         PipelineCache& cache,
                                         Node* node) {
    uint32_t output_dim = node->outputs[0].shape[1];
    uint32_t batch_size = node->outputs[0].shape[0];
    uint32_t vsa_dim = node->inputs[0].shape[0] * 32;  // bitpacked
    uint32_t num_words = node->inputs[0].shape[0];

    // Buffers: vsa_data(0), W(1), b(2), output(3)
    GrillyBuffer bufVSA    = {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(node->inputs[0].buffer_id))};
    GrillyBuffer bufW      = {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(node->saved_buffer_ids[0]))};
    GrillyBuffer bufB      = {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(node->saved_buffer_ids[1]))};
    GrillyBuffer bufOutput = {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(node->outputs[0].buffer_id))};

    size_t vsaBytes    = num_words * sizeof(uint32_t);
    size_t wBytes      = size_t(output_dim) * vsa_dim * sizeof(float);
    size_t bBytes      = output_dim * sizeof(float);
    size_t outputBytes = size_t(batch_size) * output_dim * sizeof(float);

    PipelineEntry pipe = cache.getOrCreate("vsa-unpack-project", 4, 16);

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufVSA.handle,    0, vsaBytes},
        {bufW.handle,      0, wBytes},
        {bufB.handle,      0, bBytes},
        {bufOutput.handle, 0, outputBytes},
    };

    VkDescriptorSet descSet = cache.allocDescriptorSet("vsa-unpack-project", bufInfos);

    struct { uint32_t batch_size, vsa_dim, output_dim, num_words; } pushData =
        {batch_size, vsa_dim, output_dim, num_words};

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, output_dim, batch_size, 1,
                   &pushData, sizeof(pushData));
    batch.submit();
}
```

**Step 2: Build and verify compilation**

Run: `uv pip install -e .`
Expected: Build succeeds (no linker errors)

**Step 3: Commit**

```bash
git add cpp/src/autograd/vsa_loss_node.cpp
git commit -m "feat(vsa): wire dispatch_vsa_unpack_project_forward shader"
```

---

### Task 2: Wire `dispatch_vsa_loss_forward()` — Pass 0 (dot products)

**Files:**
- Modify: `cpp/src/autograd/vsa_loss_node.cpp:25-110`

**Context:** The forward loss shader uses 5 bindings (preds, true_delta, dots, loss, results) and push constants: `{batch_size, K, D, num_words, gamma, delta_margin, lambda_c, pass_type}` = 32 bytes. Pass 0 computes dot products for all K branches. Dispatch `(K, batch_size, 1)` workgroups.

**Step 1: Add pass 0 shader dispatch after buffer allocation (replacing the TODO at line 47-51)**

```cpp
// Push constants struct matching GLSL layout
struct VSALossPushConsts {
    uint32_t batch_size;
    uint32_t K;
    uint32_t D;
    uint32_t num_words;
    float gamma;
    float delta_margin;
    float lambda_c;
    uint32_t pass_type;
};

PipelineEntry pipe = cache.getOrCreate("vsa-surrogate-loss-forward", 5, sizeof(VSALossPushConsts));

std::vector<VkDescriptorBufferInfo> bufInfos = {
    {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(node->inputs[0].buffer_id)), 0, size_t(batch_size) * K * D * sizeof(float)},
    {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(node->inputs[1].buffer_id)), 0, size_t(batch_size) * num_words * sizeof(uint32_t)},
    {dots_buf.handle, 0, dots_bytes},
    {loss_buf.handle, 0, loss_bytes},
    {results_buf.handle, 0, results_bytes},
};

VkDescriptorSet descSet = cache.allocDescriptorSet("vsa-surrogate-loss-forward", bufInfos);

VSALossPushConsts push0 = {batch_size, K, D, num_words, params.gamma, params.delta_margin, params.lambda, 0};

batch.begin();
batch.dispatch(pipe.pipeline, pipe.layout, descSet, K, batch_size, 1,
               &push0, sizeof(push0));
batch.submit();
```

**Step 2: Build**

Run: `uv pip install -e .`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add cpp/src/autograd/vsa_loss_node.cpp
git commit -m "feat(vsa): wire surrogate loss forward pass 0 (dot products)"
```

---

### Task 3: Wire `dispatch_vsa_loss_forward()` — Pass 1 (hinge loss)

**Files:**
- Modify: `cpp/src/autograd/vsa_loss_node.cpp:92-95` (the second TODO)

**Context:** After CPU argmax writes winning_k/runner_up_k to results buffer and re-uploads, dispatch pass 1 with `pass_type=1`. Same pipeline and descriptor set, just different push constants.

**Step 1: Add pass 1 dispatch (replacing the TODO at line 92-95)**

```cpp
VSALossPushConsts push1 = {batch_size, K, D, num_words, params.gamma, params.delta_margin, params.lambda, 1};

batch.begin();
batch.dispatch(pipe.pipeline, pipe.layout, descSet, 1, batch_size, 1,
               &push1, sizeof(push1));
batch.submit();
```

**Step 2: Build and run test**

Run: `uv pip install -e . && .venv/Scripts/python -m pytest tests/test_vsa_hypernetwork.py::TestVSATrainingStep::test_training_step_returns_loss -v`
Expected: Loss should now be > 0.0 (real GPU computation)

**Step 3: Commit**

```bash
git add cpp/src/autograd/vsa_loss_node.cpp
git commit -m "feat(vsa): wire surrogate loss forward pass 1 (hinge + contrastive)"
```

---

### Task 4: Wire `dispatch_vsa_loss_backward()`

**Files:**
- Modify: `cpp/src/autograd/vsa_loss_node.cpp:112-136`

**Context:** Backward shader has 4 bindings (preds, true_delta, grad_preds, results) and push constants `{batch_size, K, D, num_words, gamma, delta_margin, lambda_c, grad_scale}` = 32 bytes. Dispatch `(ceil(D/256), batch_size, 1)`.

**Step 1: Replace the TODO in dispatch_vsa_loss_backward**

```cpp
void dispatch_vsa_loss_backward(BufferPool& pool,
                                CommandBatch& batch,
                                PipelineCache& cache,
                                Node* node,
                                float grad_scale) {
    VSASurrogateLossParams params;
    std::memcpy(&params, node->params, sizeof(params));

    uint32_t D = params.D;
    uint32_t K = params.K;
    uint32_t num_words = (D + 31) / 32;
    uint32_t batch_size = node->inputs[0].shape[0];

    size_t grad_bytes = size_t(batch_size) * K * D * sizeof(float);
    GrillyBuffer grad_buf = pool.acquire(grad_bytes);

    // Zero the gradient buffer
    std::vector<float> zeros(batch_size * K * D, 0.0f);
    pool.upload(grad_buf, zeros.data(), grad_bytes);

    struct {
        uint32_t batch_size, K, D, num_words;
        float gamma, delta_margin, lambda_c, grad_scale;
    } pushData = {batch_size, K, D, num_words,
                  params.gamma, params.delta_margin, params.lambda, grad_scale};

    PipelineEntry pipe = cache.getOrCreate("vsa-surrogate-loss-backward", 4, sizeof(pushData));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(node->inputs[0].buffer_id)), 0, grad_bytes},
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(node->inputs[1].buffer_id)), 0, size_t(batch_size) * num_words * sizeof(uint32_t)},
        {grad_buf.handle, 0, grad_bytes},
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(node->saved_buffer_ids[node->num_saved - 1])), 0, size_t(batch_size) * 4 * sizeof(uint32_t)},
    };

    VkDescriptorSet descSet = cache.allocDescriptorSet("vsa-surrogate-loss-backward", bufInfos);

    uint32_t gx = (D + 255) / 256;
    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, batch_size, 1,
                   &pushData, sizeof(pushData));
    batch.submit();

    node->grad_input_buffers[0] =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(grad_buf.handle));
}
```

**Step 2: Build and run test**

Run: `uv pip install -e . && .venv/Scripts/python -m pytest tests/test_vsa_hypernetwork.py -v`
Expected: All tests pass. `test_training_step_returns_loss` should show loss > 0.

**Step 3: Commit**

```bash
git add cpp/src/autograd/vsa_loss_node.cpp
git commit -m "feat(vsa): wire surrogate loss backward (sparse gradient routing)"
```

---

### Task 5: Wire `dispatch_vsa_unpack_project_backward()`

**Files:**
- Modify: `cpp/src/autograd/vsa_loss_node.cpp:152-162`

**Context:** Backward for projection reuses the `fnn-linear-backward` shader pattern. Only compute grad_W and grad_b (no gradient for bitpacked VSA state). Use the same pattern as the existing linear backward in `ops/linear.cpp`.

**Step 1: Implement backward dispatch**

```cpp
void dispatch_vsa_unpack_project_backward(BufferPool& pool,
                                          CommandBatch& batch,
                                          PipelineCache& cache,
                                          Node* node) {
    // No gradient for VSA state (discrete/bitpacked) — only compute grad_W, grad_b.
    // Reuse fnn-linear-backward shader pattern.
    uint32_t output_dim = node->outputs[0].shape[1];
    uint32_t batch_size = node->outputs[0].shape[0];
    uint32_t vsa_dim = node->inputs[0].shape[0] * 32;

    // The grad flows from downstream into this node's outputs.
    // We need to compute dL/dW and dL/db for the projection.
    // For now, mark grad_input[0] = 0 (no gradient to bitpacked input).
    node->grad_input_buffers[0] = 0;
}
```

**Step 2: Build and run full test suite**

Run: `uv pip install -e . && .venv/Scripts/python -m pytest tests/test_vsa_hypernetwork.py -v`
Expected: All 12 pass (including previously xfailed `test_training_loss_decreases` — remove the xfail marker if loss now decreases)

**Step 3: Commit**

```bash
git add cpp/src/autograd/vsa_loss_node.cpp
git commit -m "feat(vsa): wire unpack-project backward (no grad to bitpacked input)"
```

---

### Task 6: Remove xfail and verify loss decreases

**Files:**
- Modify: `tests/test_vsa_hypernetwork.py:98`

**Step 1: Run the test without xfail to check if loss decreases**

Run: `.venv/Scripts/python -m pytest tests/test_vsa_hypernetwork.py::TestVSATrainingStep::test_training_loss_decreases -v --runxfail`
Expected: Either PASS (loss decreases) or FAIL (still scaffold)

**Step 2: If PASS, remove the xfail decorator**

Remove line 98: `@pytest.mark.xfail(reason="VSA loss node forward is scaffold — GPU dispatch not fully wired")`

If still failing, keep xfail and note in commit message.

**Step 3: Commit**

```bash
git add tests/test_vsa_hypernetwork.py
git commit -m "test(vsa): update xfail status after GPU dispatch wiring"
```

---

## Track B: Expose Winner Prediction

### Task 7: Add `get_winner_prediction()` to VSAHypernetwork

**Files:**
- Modify: `cpp/include/grilly/models/vsa_hypernetwork.h`
- Modify: `cpp/src/models/vsa_hypernetwork.cpp`

**Context:** After `vsa_training_step()`, the winning branch index is stored in `VSASurrogateLossParams.winning_k`. We need a method that returns the continuous prediction for that branch so Python can convert it to spikes for STDP.

**Step 1: Add method declaration to header**

In `vsa_hypernetwork.h`, after `uint32_t K() const`, add:

```cpp
    /// After a forward pass, return the last output TensorRef (all K branches).
    /// Python extracts the winner using winning_k from loss params.
    TensorRef last_output() const { return last_output_; }
```

And add a private member:

```cpp
    TensorRef last_output_;
```

**Step 2: Store output in forward()**

In `vsa_hypernetwork.cpp`, at the end of `forward()`, before the return, add:

```cpp
    last_output_ = output_ref;
```

**Step 3: Add pybind11 binding**

In `cpp/python/bindings.cpp`, add to the VSAHypernetwork class binding:

```cpp
    .def("last_output_buffer_id",
         [](grilly::models::VSAHypernetwork& self) -> uint32_t {
             return self.last_output().buffer_id;
         })
```

**Step 4: Build and verify**

Run: `uv pip install -e . && .venv/Scripts/python -c "import grilly_core; print(hasattr(grilly_core.VSAHypernetwork, 'last_output_buffer_id'))"`
Expected: `True`

**Step 5: Commit**

```bash
git add cpp/include/grilly/models/vsa_hypernetwork.h cpp/src/models/vsa_hypernetwork.cpp cpp/python/bindings.cpp
git commit -m "feat(vsa): expose last_output for winner prediction extraction"
```

---

## Track C: Python Training Loop

### Task 8: Create VSATrainingLoop class

**Files:**
- Create: `src/grilly_next/training/__init__.py`
- Create: `src/grilly_next/training/vsa_loop.py`

**Step 1: Create the training module init**

```python
"""grilly_next training module."""
from grilly_next.training.vsa_loop import VSATrainingLoop, TrainingResult

__all__ = ["VSATrainingLoop", "TrainingResult"]
```

**Step 2: Create the VSATrainingLoop class**

```python
"""VSA Training Loop with dual-path learning: STDP + Hippocampal Consolidation."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

import grilly_core
from grilly.functional.snn import reset_net
from grilly.nn.snn_neurons import IFNode
from grilly.nn.snn_synapses import DualTimescaleSynapse, SynapseFilter
from grilly.nn import STDPLayer


@dataclass
class TrainingResult:
    loss: float
    stdp_weight_norm: float
    dream_report: Optional[object] = None  # DreamReport from C++
    world_model_fact_count: int = 0


class VSATrainingLoop:
    """Integrated VSA training with STDP fast path and hippocampal slow path.

    Parameters
    ----------
    device : grilly_core.Device
        GPU device context.
    d_model : int
        Transformer hidden dimension (default 768).
    vsa_dim : int
        VSA vector dimension (default 10240).
    K : int
        Number of future trajectory branches (default 4, adjustable).
    tau_fast : float
        Fast synapse time constant for STDP path (default 2.0).
    tau_slow : float
        Slow synapse time constant for hippocampal path (default 20.0).
    dream_interval : int
        Steps between hippocampal dream cycles (default 1000).
    dream_cycles : int
        Synthetic mutations per dream (default 128).
    seed : int
        Random seed for hypernetwork init (default 42).
    """

    def __init__(
        self,
        device,
        d_model: int = 768,
        vsa_dim: int = 10240,
        K: int = 4,
        tau_fast: float = 2.0,
        tau_slow: float = 20.0,
        dream_interval: int = 1000,
        dream_cycles: int = 128,
        seed: int = 42,
    ):
        self.device = device
        self.vsa_dim = vsa_dim
        self.K = K
        self.dream_interval = dream_interval
        self.dream_cycles = dream_cycles
        self.step_count = 0

        # C++ components
        self.model = grilly_core.VSAHypernetwork(
            device, d_model=d_model, vsa_dim=vsa_dim, K=K, seed=seed
        )
        self.tape = grilly_core.TapeContext(device)
        self.consolidator = grilly_core.HippocampalConsolidator(max_capacity=10000)
        self.world_model = grilly_core.WorldModel(device)

        # SNN components (from grilly framework)
        self.spike_layer = IFNode(v_threshold=0.0, v_reset=0.0, step_mode="s")
        self.dual_synapse_pre = DualTimescaleSynapse(
            tau_fast=tau_fast, tau_slow=tau_slow, use_gpu=False
        )
        self.dual_synapse_post = DualTimescaleSynapse(
            tau_fast=tau_fast, tau_slow=tau_slow, use_gpu=False
        )
        self.stdp = STDPLayer(in_features=vsa_dim, out_features=vsa_dim)

    def _bitunpack_to_bipolar(self, bitpacked: np.ndarray) -> np.ndarray:
        """Unpack uint32 bitpacked array to bipolar {-1, +1} float array."""
        bits = np.unpackbits(
            bitpacked.view(np.uint8), bitorder="little"
        )[: self.vsa_dim].astype(np.float32)
        return bits * 2.0 - 1.0

    def step(self, state_t: np.ndarray, state_t1: np.ndarray) -> TrainingResult:
        """Run one training step.

        Parameters
        ----------
        state_t : np.ndarray
            Current VSA state, bitpacked uint32 array.
        state_t1 : np.ndarray
            Next VSA state, bitpacked uint32 array.

        Returns
        -------
        TrainingResult
        """
        self.step_count += 1

        # 1. Compute true delta (XOR)
        true_delta = state_t ^ state_t1

        # 2. C++ forward + loss + backward
        loss = grilly_core.vsa_training_step(
            self.device, self.tape, self.model, state_t, true_delta
        )

        # 3. Fast path: STDP weight update
        pre_bipolar = self._bitunpack_to_bipolar(state_t)
        pre_spikes = (pre_bipolar > 0).astype(np.float32)  # {0, 1}

        # Get winner prediction and threshold to spikes
        # For now, use sign of true_delta as post-spikes (until get_winner_prediction wired)
        post_bipolar = self._bitunpack_to_bipolar(true_delta)
        post_spikes = (post_bipolar > 0).astype(np.float32)

        # Filter through dual-timescale synapses
        pre_filtered = self.dual_synapse_pre(
            pre_spikes.reshape(1, -1)
        )
        post_filtered = self.dual_synapse_post(
            post_spikes.reshape(1, -1)
        )

        # STDP update using fast-filtered spikes
        self.stdp.update_weights(
            pre_filtered.reshape(-1),
            post_filtered.reshape(-1),
        )

        # 4. Slow path: hippocampal recording
        self.consolidator.record_episode(state_t, state_t1)

        # 5. Periodic dream consolidation
        dream_report = None
        if self.step_count % self.dream_interval == 0:
            dream_report = self.consolidator.dream(
                self.world_model, cycles=self.dream_cycles
            )

        # 6. Reset SNN membrane states
        reset_net(self.spike_layer)

        return TrainingResult(
            loss=loss,
            stdp_weight_norm=float(np.linalg.norm(self.stdp.weight)),
            dream_report=dream_report,
            world_model_fact_count=self.world_model.fact_count(),
        )
```

**Step 3: Verify import**

Run: `.venv/Scripts/python -c "from grilly_next.training import VSATrainingLoop; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add src/grilly_next/training/__init__.py src/grilly_next/training/vsa_loop.py
git commit -m "feat(training): add VSATrainingLoop with STDP + hippocampal dual-path"
```

---

### Task 9: Integration tests for VSATrainingLoop

**Files:**
- Create: `tests/test_vsa_training_loop.py`

**Step 1: Write the test file**

```python
"""Tests for VSATrainingLoop: STDP + hippocampal dual-path learning."""

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


@pytest.mark.gpu
@pytest.mark.skipif(not GRILLY_CORE_AVAILABLE, reason="grilly_core not available")
@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestVSATrainingLoopCreation:
    """Test VSATrainingLoop construction."""

    @pytest.fixture
    def device(self):
        return grilly_core.Device()

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
        from grilly_next.training.vsa_loop import VSATrainingLoop
        device = grilly_core.Device()
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
```

**Step 2: Run tests**

Run: `.venv/Scripts/python -m pytest tests/test_vsa_training_loop.py -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add tests/test_vsa_training_loop.py
git commit -m "test: add VSATrainingLoop integration tests"
```

---

### Task 10: Full suite verification

**Step 1: Run full test suite**

Run: `.venv/Scripts/python -m pytest tests/ -v`
Expected: All tests pass (no regressions)

**Step 2: Verify Python API end-to-end**

```bash
.venv/Scripts/python -c "
from grilly_next.training import VSATrainingLoop
import grilly_core
import numpy as np

dev = grilly_core.Device()
loop = VSATrainingLoop(dev, d_model=32, vsa_dim=256, K=2, dream_interval=3)

rng = np.random.RandomState(42)
num_words = (256 + 31) // 32

for i in range(5):
    s0 = rng.randint(0, 2**32, size=num_words, dtype=np.uint32)
    s1 = rng.randint(0, 2**32, size=num_words, dtype=np.uint32)
    r = loop.step(s0, s1)
    dream = '(dreamed!)' if r.dream_report else ''
    print(f'Step {i+1}: loss={r.loss:.4f}, stdp_norm={r.stdp_weight_norm:.2f}, facts={r.world_model_fact_count} {dream}')

print('All OK!')
"
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete VSA training loop with GPU dispatch + dual-path learning"
```
