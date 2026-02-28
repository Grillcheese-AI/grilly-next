# VSA Hypernetwork + Surrogate Loss Design

**Date:** 2026-02-28
**Status:** Approved

## Summary

Implement a C++ VSA Hypernetwork that generates K parallel future state predictions, scored against bitpacked ground-truth deltas via a custom hinge + contrastive surrogate loss. The entire pipeline runs on the TapeArena with GPU-accelerated forward/backward passes via GLSL compute shaders.

## Pipeline

```
TrainingPayload.vsa_state (BitpackedVec, 10240 bits)
    |
    v
[VSA Unpack + Project] -- TensorRef(dtype=u32) -> unpack to fp32 -> W_proj(10240, 768)
    |
    v
h_t (768 floats)
    |
    v
[VSAHypernetwork MLP]
    Layer 1: linear(h_t, W1, b1) -> 1536
    GELU activation
    Layer 2: linear(h_act, W2, b2) -> K * 10240
    Reshape -> [batch, K, 10240]
    |
    v
predicted_deltas [batch, K, 10240]
    |                          true_delta = S_t XOR S_{t+1} (BitpackedVec)
    v                          |
[VSA Surrogate Loss] <--------+
    |
    v
loss (scalar) -> tape.backward() -> optimizer.step()
```

## Components

### 1. VSA State as TensorRef (dtype=u32)

BitpackedVec is promoted to a first-class autograd tensor:
- Upload `BitpackedVec.data` as a `uint32[]` GPU buffer
- Wrap in `TensorRef{buffer_id, ndim=1, shape=[numWords], dtype=2(u32), requires_grad=false}`
- Store in `Node.saved_buffer_ids[]` for the loss node
- Future: enable `requires_grad=true` for learning which bits to flip

### 2. VSA Unpack + Projection (fused shader)

New `OpType::VSAUnpackProject`:
- Input: TensorRef(dtype=u32) — bitpacked VSA state (320 uint32 words for D=10240)
- Weights: W_proj (10240 x 768), b_proj (768) — learnable
- Output: h_t (batch x 768)
- Forward shader `vsa-unpack-project.glsl`:
  - Each workgroup handles one output dimension
  - Unpack bits inline: `float val = ((data[word] >> bit) & 1) ? 1.0 : -1.0`
  - Dot product with projection row + bias
- Backward: gradient w.r.t. W_proj and b_proj (standard linear backward)
- No gradient w.r.t. VSA state (bitpacked, discrete)

### 3. VSAHypernetwork (C++ class)

```cpp
class VSAHypernetwork {
    uint32_t d_model_;   // 768
    uint32_t vsa_dim_;   // 10240
    uint32_t K_;         // 4 (default, scales to 128)

    // Weights (registered to optimizer)
    TensorRef W_proj, b_proj;   // 10240 x 768 projection
    TensorRef W1, b1;           // 768 x 1536
    TensorRef W2, b2;           // 1536 x (K * 10240)

    TensorRef forward(TapeContext& tape, TensorRef vsa_state);
};
```

Forward builds the Wengert list using existing ops:
1. `record_op(VSAUnpackProject, vsa_state, h_t, {W_proj, b_proj})`
2. `record_op(Linear, h_t, h_mid, {W1, b1})`
3. `record_op(GELU, h_mid, h_act)`
4. `record_op(Linear, h_act, raw_deltas, {W2, b2})`
5. `record_op(Reshape, raw_deltas, predicted_deltas)` (conceptual, shapes tracked in TensorRef)

### 4. VSA Surrogate Loss Node

New `OpType::VSASurrogateLoss`:

**Forward (shader: `vsa-surrogate-loss-forward.glsl`):**
1. For each k in [0, K): compute `dot_k = sum_i(pred[k,i] * unpack(true_delta, i))`
2. Find `winning_k = argmax(dot_k)` and `runner_up_k`
3. Hinge loss on winner: `L_hinge = mean_i(max(0, gamma - y_true_i * z_pred_winning_i))`
4. Contrastive margin: `L_contrast = max(0, delta_margin - (dot_winner - dot_runner_up))`
5. Total: `L = L_hinge + lambda * L_contrast`

**Backward (shader: `vsa-surrogate-loss-backward.glsl`):**
- Winner gradient: `grad[winning_k, i] = (-y_true_i / D) * scale` where hinge active
- Runner-up gradient: small positive push `grad[runner_up_k, i] = (y_true_i / D) * lambda * scale` where contrastive active
- Other K-2 branches: zero gradient (sparse)
- Cost: O(2D) instead of O(K*D)

**Parameters stored in Node.params[64]:**
```cpp
struct VSASurrogateLossParams {
    float gamma;           // Hinge margin (default 0.1)
    float delta_margin;    // Contrastive margin (default 0.5)
    float lambda;          // Contrastive weight (default 0.1)
    uint32_t K;            // Number of futures
    uint32_t D;            // VSA dimension (10240)
    uint32_t winning_k;    // Written by forward, read by backward
    uint32_t runner_up_k;  // Written by forward, read by backward
    float loss_value;      // Written by forward
};
static_assert(sizeof(VSASurrogateLossParams) <= 64);
```

### 5. Training Loop

C++ function exposed via pybind11:

```cpp
void vsa_training_step(
    TapeContext& tape,
    VSAHypernetwork& model,
    TrainingPayload& payload,
    AdamOptimizer& optimizer
) {
    tape.begin();

    // 1. Upload true delta as TensorRef(u32)
    BitpackedVec true_delta = payload.vsa_state ^ next_payload.vsa_state;
    TensorRef target = upload_bitpacked(tape, true_delta);

    // 2. Hypernetwork forward (builds Wengert list)
    TensorRef predicted_deltas = model.forward(tape, payload.vsa_state);

    // 3. Loss forward
    Node* loss_node = tape.record_op(
        OpType::VSASurrogateLoss,
        {predicted_deltas, target},
        {loss_output},
        &loss_params, sizeof(loss_params)
    );

    // 4. Backward + optimize
    tape.backward(loss_node, /*grad_output=*/ones_buffer);
    optimizer.step(tape);  // adam-update.glsl

    tape.end();  // O(1) arena reset
}
```

## New Files

| File | Purpose |
|------|---------|
| `cpp/include/grilly/models/vsa_hypernetwork.h` | Hypernetwork class + weight init |
| `cpp/src/models/vsa_hypernetwork.cpp` | Forward pass (tape recording) |
| `cpp/include/grilly/autograd/vsa_loss_node.h` | VSASurrogateLossParams struct |
| `cpp/src/autograd/vsa_loss_node.cpp` | CPU fallback for loss forward/backward |
| `shaders/vsa-surrogate-loss-forward.glsl` | GPU: find winner, compute hinge + contrastive |
| `shaders/vsa-surrogate-loss-backward.glsl` | GPU: sparse gradient routing |
| `shaders/vsa-unpack-project.glsl` | GPU: fused bitunpack + linear projection |

## Changes to Existing Files

| File | Change |
|------|--------|
| `cpp/include/grilly/autograd/autograd.h` | Add `VSASurrogateLoss`, `VSAUnpackProject` to OpType enum |
| `cpp/src/autograd/autograd.cpp` | Add backward handlers in BackwardEngine dispatch table |
| `cpp/python/bindings.cpp` | Expose VSAHypernetwork, new OpTypes, training step |
| `CMakeLists.txt` | Add new .cpp sources + shader compile rules |

## Constraints

- **K=4 default**, configurable. W2 at K=4 is ~240 MB (fits easily in VRAM).
- **No Python nn dependency** for the training loop — fully C++ on the tape.
- **Backward is sparse**: O(2D) per step, not O(K*D).
- **VSA state is not differentiable** (discrete bitpacked). Projection weights are.

## Success Criteria

1. `VSAHypernetwork::forward()` produces [batch, K, 10240] predictions on GPU
2. `VSASurrogateLossNode` forward computes correct hinge + contrastive loss
3. Backward routes gradients only through winner + runner-up (verified by gradient inspection)
4. Training loop drives loss down on synthetic data (random BitpackedVec pairs)
5. All new code compiles with existing CMake, shaders compile to SPIR-V
6. Pybind11 bindings allow Python orchestration of training
