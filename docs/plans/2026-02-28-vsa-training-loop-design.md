# VSA Training Loop: GPU Dispatch + STDP + Hippocampal Consolidation

## Abstract

We present a biologically-inspired dual-path learning system for Vector Symbolic Architecture (VSA) state-transition prediction. A hypernetwork generates K parallel future trajectories in continuous space from bitpacked bipolar VSA states. A surrogate hinge loss with contrastive margin selects the winning trajectory via sparse gradient routing (O(2D) backward instead of O(KD)). Training results flow through two complementary learning paths: (1) a **fast path** using Spike-Timing-Dependent Plasticity (STDP) for online synaptic weight adaptation, filtered through a dual-timescale exponential synapse, and (2) a **slow path** using hippocampal episodic buffering with offline dream consolidation that extracts recurring state-transition patterns as permanent WorldModel rules. This mirrors the biological complementary learning systems (CLS) theory: the hippocampus rapidly encodes episodes while the neocortex slowly generalizes.

## Notation

| Symbol | Meaning |
|--------|---------|
| S_t | Bitpacked VSA state at time t, S_t in {0,1}^(D/32) stored as uint32 words |
| D | VSA dimension (default 10240) |
| K | Number of future trajectory branches (adjustable at runtime) |
| delta_t | True transition: S_t XOR S_{t+1} |
| z_k | Continuous prediction from branch k, z_k in R^D |
| k* | Winning branch: k* = argmax_k (z_k . y_true) |
| y_true | Bipolar expansion of delta_t: y_i = 2*bit_i - 1 |
| gamma | Hinge margin (default 0.1) |
| delta_m | Contrastive margin between winner and runner-up (default 0.5) |
| lambda | Contrastive weight (default 0.1) |
| tau_f | Fast synapse time constant (default 2.0) |
| tau_s | Slow synapse time constant (default 20.0) |

## Loss Function

The surrogate loss combines per-element hinge loss on the winning branch with a contrastive margin between winner and runner-up:

```
L_hinge = (1/D) * sum_i max(0, gamma - y_true_i * z_{k*,i})

L_contrast = max(0, delta_m - (dot(z_{k*}, y_true) - dot(z_{k'}, y_true)))

L_total = L_hinge + lambda * L_contrast
```

where k' is the runner-up branch. Gradients are routed only to k* (and k' for the contrastive term), giving O(2D) backward cost independent of K.

## Dual-Path Learning

### Fast Path: STDP

Converts VSA predictions to spike-timing signals for online Hebbian-style learning:

1. Pre-synaptic spikes: s_pre = sign(bitunpack(S_t)) in {-1, +1}^D
2. Post-synaptic spikes: s_post = Heaviside(z_{k*}) in {0, 1}^D
3. Dual-timescale filtering: s_pre^f = DualSynapse(s_pre).fast, s_post^f = DualSynapse(s_post).fast
4. STDP update: W <- W + a_plus * (s_post^f * s_pre^f^T) - a_minus * (s_pre^f * s_post^f^T)

The STDP weight matrix W in R^(D x D) captures fast temporal correlations between current states and predicted transitions. GPU-accelerated via stdp-learning.glsl.

### Slow Path: Hippocampal Consolidation

Episodic memory with offline generalization:

1. Record: buffer <- buffer + (S_t, S_{t+1}) after each step
2. Dream (every N steps):
   a. Compute deltas: for each (S_t, S_{t+1}) in buffer, d = S_t XOR S_{t+1}
   b. Count frequency of each unique delta
   c. Burn deltas appearing in >5% of episodes as WorldModel facts
   d. Generate synthetic mutations (random bit flips) for exploration
   e. Clear buffer
3. WorldModel stores permanent transition rules as bitpacked VSA vectors with GPU Hamming-distance coherence checking (~58us per query).

### Why Dual-Path

| Property | Fast (STDP) | Slow (Hippocampal) |
|----------|-------------|-------------------|
| Timescale | Per-step | Every N steps |
| What it learns | Temporal spike correlations | Recurring transition patterns |
| Storage | Weight matrix W | WorldModel fact vectors |
| Forgetting | Continuous decay | Threshold-based pruning |
| Biological analog | Synaptic plasticity | Sleep consolidation |

## Goal

Wire the VSA surrogate loss GPU dispatch so training produces real gradients, then build an integrated training loop that stores results via dual-path learning: fast online STDP weight adaptation and slow offline hippocampal dream consolidation into WorldModel rules.

## Architecture

Approach A (SNN Gating Layer): The VSA hypernetwork's winning trajectory prediction passes through an SNN spike layer. STDP updates synaptic weights based on temporal correlation between VSA state (pre-spikes) and predicted delta (post-spikes). A DualTimescaleSynapse from the grilly framework filters spikes into fast (STDP) and slow (hippocampal) channels. Periodically, the hippocampal consolidator dreams to extract recurring transition patterns as permanent WorldModel facts.

## Track A: Wire VSA Loss GPU Dispatch

### Context

`vsa_loss_node.cpp` has 4 TODO blocks where Vulkan shader dispatch calls need to replace scaffold code. The GLSL shaders are already compiled to SPIR-V.

### Changes

**`dispatch_vsa_loss_forward()`** — two-pass shader dispatch:
- Pass 0: Bind `vsa-surrogate-loss-forward.spv` with `pass_type=0`. Computes dot products between K predictions and true delta. Dispatch `(K, batch_size, 1)` workgroups of `(256, 1, 1)`.
- CPU argmax: Download dots, find winning_k and runner_up_k (already implemented).
- Pass 1: Bind same shader with `pass_type=1`. Computes hinge + contrastive margin loss. Dispatch `(1, batch_size, 1)` workgroups.

**`dispatch_vsa_loss_backward()`** — sparse gradient routing:
- Bind `vsa-surrogate-loss-backward.spv`. Only winning branch (+ runner-up for contrastive) receives gradients.
- Dispatch `(ceil(D/256), batch_size, 1)` workgroups.

**`dispatch_vsa_unpack_project_forward()`** — fused bitunpack + linear projection:
- Bind `vsa-unpack-project.spv`.
- Dispatch `(output_dim, batch_size, 1)` workgroups.

**`dispatch_vsa_unpack_project_backward()`** — projection gradient:
- Reuse `fnn-linear-backward.glsl` pattern for W_proj/b_proj gradients.
- No gradient for VSA state (discrete/bitpacked).

### Pattern

Follow existing dispatch pattern from `ops/linear.cpp`:
```cpp
auto pipeline = cache.getOrCreate("shaders/spv/shader-name.spv", ...);
batch.bindPipeline(pipeline);
batch.bindBuffers({buf0, buf1, ...});
batch.pushConstants(&push_data, sizeof(push_data));
batch.dispatch(gx, gy, gz);
batch.submit();
batch.waitIdle();
```

### Verification

After wiring, `vsa_training_step()` returns real loss > 0.0, and `test_training_loss_decreases` flips from xfail to pass.

## Track B: STDP + Hippocampal Storage

### Data Flow Per Training Step

```
VSATrainingLoop.step(state_t, state_t1)

1. true_delta = state_t XOR state_t1  (bitpacked)
2. loss = grilly_core.vsa_training_step(device, tape, model, state_t, true_delta)
3. winner_pred = model.get_winner_prediction()

Fast path (STDP):
  pre  = sign(bitunpack(state_t))                  → bipolar spikes
  post = IFNode(winner_pred)                        → binary spikes
  pre_f, post_f = dual_synapse(pre), dual_synapse(post)
  stdp.update_weights(pre_f.fast, post_f.fast)      → GPU via stdp-learning.glsl

Slow path (Hippocampal):
  consolidator.record_episode(state_t, state_t1)    → episodic buffer
  if step % dream_interval == 0:
      report = consolidator.dream(world_model)       → extract rules

4. reset_net(spike_layer)
5. return TrainingResult(loss, report?)
```

### Components Used

From **grilly framework** (grilly>=0.4.5):
- `grilly.nn.STDPLayer(in_features, out_features)` — GPU-accelerated STDP via `stdp-learning.glsl`
- `grilly.nn.snn_synapses.DualTimescaleSynapse(tau_fast, tau_slow)` — fast/slow spike filtering
- `grilly.nn.snn_neurons.IFNode(v_threshold=0.0)` — threshold to binary spikes
- `grilly.functional.snn.reset_net()` — clear membrane states between batches

From **grilly_core** (C++ extension):
- `VSAHypernetwork` — K-branch prediction model
- `vsa_training_step()` — forward + loss + backward
- `HippocampalConsolidator` — episodic buffer + dream cycle
- `WorldModel` — permanent fact storage with GPU coherence checking

### New Files

**`src/grilly_next/training/vsa_loop.py`**:
```python
class VSATrainingLoop:
    def __init__(self, device, d_model=768, vsa_dim=10240, K=4,
                 tau_fast=2.0, tau_slow=20.0,
                 dream_interval=1000, dream_cycles=128):
        ...

    def step(self, state_t, state_t1) -> TrainingResult:
        ...
```

**`tests/test_vsa_training_loop.py`** — integration tests.

### Constructor Parameters

| Param | Default | Description |
|-------|---------|-------------|
| device | required | grilly_core.Device |
| d_model | 768 | Transformer hidden dim |
| vsa_dim | 10240 | VSA vector dimension |
| K | 4 | Number of future trajectory branches (adjustable) |
| tau_fast | 2.0 | Fast synapse time constant (STDP path) |
| tau_slow | 20.0 | Slow synapse time constant (hippocampal path) |
| dream_interval | 1000 | Steps between dream consolidation cycles |
| dream_cycles | 128 | Synthetic mutations per dream |

K is a runtime parameter that flows through: `VSAHypernetwork(K=K)` -> `VSASurrogateLossParams.K = K` -> backward sparse routing.

### Return Value

```python
@dataclass
class TrainingResult:
    loss: float
    stdp_weight_norm: float
    dream_report: Optional[DreamReport]  # None unless dream() ran this step
    world_model_fact_count: int
```

## Track C: Minor C++ Addition

Expose `get_winner_prediction()` on `VSAHypernetwork` so Python can extract the winning branch's continuous output for STDP spike conversion. Add pybind11 binding.

## Integration with TrainingPipeline

The existing `TrainingPipeline` produces `TrainingPayload` with `vsa_state` and `llm_input_tokens`. The consumer loop calls `VSATrainingLoop.step()` with consecutive payloads to form `(state_t, state_t1)` pairs:

```python
pipeline = grilly_core.TrainingPipeline(dim=10240)
loop = VSATrainingLoop(device, K=4)

prev_state = None
while pipeline.pop(payload):
    if prev_state is not None:
        result = loop.step(prev_state, payload.vsa_state)
    prev_state = payload.vsa_state
```
