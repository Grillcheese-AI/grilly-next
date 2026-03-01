# Grilly-Next Architecture Insights

Technical insights collected during the design and implementation of the VSA training loop, many-worlds inference, and dual-path learning system.

---

## VSA Fundamentals

### Bitpacking as Geometry Compression

10240-dimensional bipolar {-1, +1} vectors are stored as 320 `uint32` words. This isn't just memory savings (32x compression) — it transforms algebraic operations into bitwise ones:

- **XOR = Binding**: In bipolar space, binding is element-wise multiply. Since (-1)(-1) = (+1)(+1) = +1 and (-1)(+1) = -1, multiplication maps exactly to XNOR. Bitpacked XOR gives us binding at 32 dimensions per clock cycle.
- **Hamming distance = Dot product proxy**: The Hamming distance between two bitpacked vectors is linearly related to their bipolar dot product: `dot(a, b) = D - 2 * hamming(a, b)`. This means GPU `bitCount(a XOR b)` gives us similarity search without any floating-point arithmetic.
- **Majority-vote bundling**: Superposition of N vectors via element-wise majority vote, implementable as popcount across columns.

### BLAKE3 Deterministic Role Generation

Each symbol in the VSA space gets a deterministic bipolar vector via `blake3Role("filler_" + name, dim)`. This means:
- No learned embeddings needed — any string maps to a unique, quasi-orthogonal vector
- Vectors are reproducible across sessions (same key = same vector)
- The "filler_" prefix matches the CubeMind TextEncoder's key scheme for interoperability

---

## Training Architecture

### Surrogate Loss with Sparse Gradient Routing

The hypernetwork generates K parallel future trajectories, but only the **winning branch** (highest dot product with true delta) receives gradients. This gives O(2D) backward cost instead of O(KD):

```
L_total = L_hinge(winner) + lambda * L_contrast(winner, runner_up)
```

- **Hinge loss**: Per-element `max(0, gamma - y_true * z_pred)` on the winner — pushes predictions to match the true transition delta with margin gamma
- **Contrastive margin**: `max(0, delta_m - (dot_winner - dot_runner_up))` — ensures the winning branch is decisively better, preventing mode collapse where all K branches predict the same thing
- **Sparse routing**: Gradients flow only to k* (and k' for contrastive), so adding more branches (larger K) doesn't increase backward cost

### Dual-Path Learning (Complementary Learning Systems)

Mirrors the biological CLS theory — hippocampus rapidly encodes episodes while neocortex slowly generalizes:

| Property | Fast Path (STDP) | Slow Path (Hippocampal) |
|----------|------------------|------------------------|
| Timescale | Per training step | Every N steps (dream cycle) |
| What it learns | Temporal spike correlations | Recurring state-transition patterns |
| Storage | Weight matrix W (D x D) | WorldModel fact vectors (bitpacked) |
| Forgetting | Continuous exponential decay | Threshold-based pruning (>5% frequency) |
| Biological analog | Synaptic plasticity | Sleep consolidation |
| GPU acceleration | `stdp-learning.glsl` | Hamming search (~29us per lookup) |

**Why both?** STDP captures fine-grained temporal correlations that are too noisy to commit permanently. The hippocampal dream cycle acts as a statistical filter — only transitions that recur across many episodes get burned into the WorldModel as permanent rules. This prevents overfitting to individual episodes while still adapting quickly to new patterns.

### DualTimescaleSynapse as Information Router

The `DualTimescaleSynapse(tau_fast, tau_slow)` from `grilly.nn.snn_synapses` splits spike signals into two temporal channels:

- **Fast channel** (tau=2.0): High-pass filter capturing rapid spike timing correlations. Feeds the STDP weight update rule. Decays quickly so only recent temporal relationships matter.
- **Slow channel** (tau=20.0): Low-pass filter capturing sustained activity patterns. Feeds the hippocampal episodic buffer. Integrates over longer windows to detect stable state transitions.

This is more principled than simply running two parallel systems — the synapse naturally decomposes the neural signal into frequency bands appropriate for each learning path.

---

## Inference Architecture

### Many-Worlds Coherence Check

The existing `WorldModel::check_coherence()` validates **one statement** at a time via two sequential `VSACache::lookup()` calls (~58us total). The new many-worlds shader validates **K hypothetical futures simultaneously** in a single GPU dispatch:

```
For each candidate k (one workgroup per k):
    S_{t+1}^{(k)} = S_t XOR Delta_k        // Apply hypothetical intervention
    For each constraint c:
        hamming_dist = popcount(S_{t+1}^{(k)} XOR constraint_c)
        if dist < threshold: violations++   // Future contradicts known rule
```

This is the inference-time complement to training:
- **Training** learns *which* trajectories are good (via surrogate loss + STDP)
- **Inference** rejects *physically impossible* futures (via constraint checking) before scoring

The `CounterfactualReasoner` already does single-branch "what if" evaluation (erase-insert-forward-validate). The many-worlds module is its **batch generalization** — K parallel universes checked in one dispatch.

### Snap-to-Bipolar: The Continuous-Discrete Bridge

The hypernetwork outputs continuous float32 predictions (R^D), but the VSA domain operates on discrete bitpacked {-1, +1} vectors. `snap_to_bipolar()` bridges this gap with a simple signum threshold:

```cpp
bit = (continuous_value > 0.0f) ? 1 : 0  // +1 or -1 in bipolar
```

This is deliberately the simplest possible quantization. The surrogate loss (with hinge margin gamma=0.1) teaches the network to push predictions far from the 0.0 decision boundary, making the snap robust. More sophisticated approaches (Gumbel-softmax, straight-through estimator) aren't needed because the training loss already handles gradient flow through the discrete boundary.

---

## GPU Compute Patterns

### Subgroup vs Workgroup Reduction

A critical distinction in Vulkan compute shaders that affects correctness:

- **`subgroupAdd(value)`**: Reduces across threads within a single **subgroup** (warp). Subgroup size is hardware-dependent: 32 on NVIDIA, 64 on AMD, 16 on Intel.
- **Workgroup-level reduction**: Requires shared memory + barrier to communicate between subgroups within the same workgroup.

With `local_size_x = 320` (for 10240-bit VSA vectors at 32 bits per thread):
- NVIDIA: 320/32 = 10 subgroups — `subgroupAdd` alone only sums 1/10th of the vector
- AMD: 320/64 = 5 subgroups — `subgroupAdd` alone only sums 1/5th of the vector

**Correct two-stage pattern:**
```glsl
shared uint shared_partial[10]; // max subgroups

// Stage 1: subgroup reduction (hardware-accelerated, single instruction)
uint sg_sum = subgroupAdd(local_dist);
if (subgroupElect()) {
    shared_partial[gl_SubgroupID] = sg_sum;
}
barrier();

// Stage 2: workgroup reduction (thread 0 accumulates from shared memory)
if (gl_LocalInvocationID.x == 0) {
    uint total = 0;
    for (uint s = 0; s < gl_NumSubgroups; s++)
        total += shared_partial[s];
    // Now 'total' has the true sum across all 320 threads
}
```

Without this fix, Hamming distances are systematically **underestimated by 80-90%**, causing the coherence checker to miss constraint violations — the model would hallucinate "legal" futures that actually contradict known rules.

### Vulkan Dispatch Pattern (grilly-next standard)

Every GPU operation follows the same pattern from `ops/linear.cpp`:

```cpp
// 1. Acquire buffers (power-of-2 bucketed, persistently mapped)
GrillyBuffer buf = pool.acquire(bytes);

// 2. Upload via persistent mapping (single memcpy, no vkMap/vkUnmap)
pool.upload(buf, data, bytes);

// 3. Get or create pipeline (cached by shader name)
PipelineEntry pipe = cache.getOrCreate("shader-name", numBindings, pushConstSize);

// 4. Allocate descriptor set (LRU cached to prevent pool exhaustion)
VkDescriptorSet desc = cache.allocDescriptorSet("shader-name", bufferInfos);

// 5. Record and submit dispatch
batch.begin();
batch.dispatch(pipe.pipeline, pipe.layout, desc, gx, gy, gz, &pushData, sizeof(pushData));
batch.submit();  // synchronous: end + submit + wait fence

// 6. Download result and release buffers
pool.download(buf, output, bytes);
pool.release(buf);
```

Key properties:
- **Zero Python crossings**: All Vulkan calls are native C++, no ctypes overhead
- **Single command buffer submission**: `batch.dispatch()` records, `batch.submit()` submits all at once
- **Persistent mapping**: `pool.upload()`/`download()` are bare `memcpy` — no vkMapMemory overhead
- **LRU descriptor sets**: Prevents VkDescriptorPool exhaustion for repeated dispatches

### GrillyCoreContext: The Unified GPU Context

Python sees a single `Device` object, but internally it's `GrillyCoreContext` bundling four components:

```cpp
struct GrillyCoreContext {
    GrillyDevice device;     // Vulkan instance + physical/logical device + queue
    BufferPool pool;         // VMA-backed buffer pool with bucket reuse
    PipelineCache cache;     // Shader pipeline + descriptor set cache
    CommandBatch batch;      // Single-shot command buffer recorder
};
```

All GPU operations accept `GrillyCoreContext&` and access `.pool`, `.cache`, `.batch` directly. The `loadShaders()` method auto-discovers all `.spv` files in a directory, so new shaders just need to be compiled and placed in `shaders/spv/`.

---

## WorldModel Coherence Scoring

The WorldModel maintains two parallel `VSACache` stores:

1. **known_facts_**: Positive knowledge (e.g., "dog is animal")
2. **constraints_**: Auto-generated negations (e.g., "dog is_not animal")

Coherence scoring for any statement:
```
support   = 1.0 - known_facts.querySurprise    // How close to a known truth
violation = 1.0 - constraints.querySurprise     // How close to a known falsehood
score     = support - violation                  // Range: [-1, +1]
coherent  = score > 0.3                          // Threshold-based decision
```

Both lookups use the same 29us Hamming shader, so a full coherence check costs ~58us total regardless of how many facts are stored. The `querySurprise` metric is `minHammingDistance / dim`, normalized to [0, 1].

---

## Hippocampal Dream Consolidation

The `HippocampalConsolidator` implements a four-phase dream cycle that extracts permanent knowledge from episodic experience:

1. **Replay**: For each episode (S_t, S_{t+1}), compute transition delta: `d = S_t XOR S_{t+1}`
2. **Frequency count**: Count occurrences of each unique delta across all buffered episodes
3. **Rule extraction**: Deltas appearing in >5% of episodes are "burned" as permanent WorldModel facts via `add_fact_vec()`
4. **Synthetic exploration**: Generate random bit-flip mutations of frequent deltas for exploration (prevents the model from only consolidating what it already knows)

The 5% threshold is the key design choice — too low and noise gets consolidated as rules, too high and genuine patterns take too long to learn. This mirrors the biological observation that memories replayed more frequently during sleep are more likely to be consolidated into long-term memory.
