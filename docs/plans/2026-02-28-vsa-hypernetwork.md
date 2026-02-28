# VSA Hypernetwork Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a C++ VSA Hypernetwork that generates K parallel future state predictions, scored against bitpacked ground-truth deltas via a custom hinge + contrastive surrogate loss on the TapeArena.

**Architecture:** BitpackedVec (10240 bits) is uploaded as a TensorRef(dtype=u32) GPU buffer, unpacked and projected to 768 dims via a fused shader, then fed through a 2-layer MLP producing K*10240 predictions. A custom VSASurrogateLoss node finds the winning branch, computes hinge + contrastive margin, and routes sparse gradients through only the winner and runner-up.

**Tech Stack:** C++17, Vulkan compute shaders (GLSL 450), pybind11, TapeArena autograd engine, BufferPool GPU memory management

---

## Task 1: Add OpType Enum Values

**Files:**
- Modify: `cpp/include/grilly/autograd/autograd.h:130-133` (before `_Count`)

**Step 1: Add the two new OpType values**

In `cpp/include/grilly/autograd/autograd.h`, insert before `_Count` (line 133):

```cpp
    // VSA Hypernetwork
    VSAUnpackProject,      // Fused bitunpack -> fp32 -> linear projection
    VSASurrogateLoss,      // Hinge + contrastive margin loss on K branches

    _Count  // sentinel
```

**Step 2: Add VSASurrogateLossParams struct**

In the same file, after `TemporalSurpriseParams` (around line 160), add:

```cpp
struct VSASurrogateLossParams {
    float gamma;           // Hinge margin (default 0.1)
    float delta_margin;    // Contrastive margin (default 0.5)
    float lambda;          // Contrastive weight (default 0.1)
    uint32_t K;            // Number of future trajectories
    uint32_t D;            // VSA dimension (10240)
    uint32_t winning_k;    // Written by forward, read by backward
    uint32_t runner_up_k;  // Written by forward, read by backward
    float loss_value;      // Written by forward
};
static_assert(sizeof(VSASurrogateLossParams) <= 64, "Must fit in Node::params[64]");
```

**Step 3: Verify it compiles**

Run: `cd build && cmake --build . --config Release 2>&1 | head -20`
Expected: Compiles without errors

**Step 4: Commit**

```bash
git add cpp/include/grilly/autograd/autograd.h
git commit -m "feat(autograd): add VSAUnpackProject + VSASurrogateLoss OpTypes"
```

---

## Task 2: Write the VSA Unpack+Project Shader

**Files:**
- Create: `shaders/vsa-unpack-project.glsl`

**Step 1: Write the fused unpack + linear projection shader**

Create `shaders/vsa-unpack-project.glsl`:

```glsl
#version 450

// Fused VSA unpack + linear projection.
// Unpacks a bitpacked uint32[] VSA vector to float {-1, +1},
// then computes output = unpack(vsa) @ W^T + bias.
//
// Each workgroup computes one output element across the full input dimension.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Bitpacked VSA state: ceil(vsa_dim / 32) uint32 words
layout(set = 0, binding = 0) readonly buffer VSAState {
    uint vsa_data[];
};

// Projection weights: (output_dim, vsa_dim) stored row-major
layout(set = 0, binding = 1) readonly buffer Weights {
    float W[];
};

// Bias vector: (output_dim)
layout(set = 0, binding = 2) readonly buffer Bias {
    float b[];
};

// Output: (batch, output_dim)
layout(set = 0, binding = 3) writeonly buffer Output {
    float output_data[];
};

layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint vsa_dim;       // 10240
    uint output_dim;    // 768
    uint num_words;     // ceil(vsa_dim / 32) = 320
};

shared float partial_sums[256];

void main() {
    uint batch_idx = gl_WorkGroupID.y;
    uint out_idx = gl_WorkGroupID.x;   // Which output dimension
    uint tid = gl_LocalInvocationID.x;

    if (out_idx >= output_dim) return;

    // Each thread handles a chunk of the dot product
    float sum = 0.0;
    uint row_offset = out_idx * vsa_dim;
    uint vsa_offset = batch_idx * num_words;

    for (uint i = tid; i < vsa_dim; i += 256) {
        // Unpack bit to float {-1, +1}
        uint word_idx = i / 32;
        uint bit_idx = i % 32;
        float val = ((vsa_data[vsa_offset + word_idx] >> bit_idx) & 1u) != 0u ? 1.0 : -1.0;

        sum += val * W[row_offset + i];
    }

    // Parallel reduction in shared memory
    partial_sums[tid] = sum;
    barrier();

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        barrier();
    }

    if (tid == 0) {
        output_data[batch_idx * output_dim + out_idx] = partial_sums[0] + b[out_idx];
    }
}
```

**Step 2: Compile to SPIR-V**

Run: `glslc shaders/vsa-unpack-project.glsl -o shaders/spv/vsa-unpack-project.spv`
Expected: Compiles without errors

**Step 3: Commit**

```bash
git add shaders/vsa-unpack-project.glsl shaders/spv/vsa-unpack-project.spv
git commit -m "feat(shaders): add vsa-unpack-project fused kernel"
```

---

## Task 3: Write the Surrogate Loss Forward Shader

**Files:**
- Create: `shaders/vsa-surrogate-loss-forward.glsl`

**Step 1: Write the forward loss shader**

Create `shaders/vsa-surrogate-loss-forward.glsl`:

```glsl
#version 450

// VSA Surrogate Loss — Forward Pass
//
// For K predictions of dimension D, computes:
// 1. Dot product of each prediction against bitpacked true delta
// 2. Finds winning_k (argmax dot) and runner_up_k
// 3. Hinge loss on winner: mean(max(0, gamma - y_true * z_pred))
// 4. Contrastive margin: max(0, delta_margin - (dot_winner - dot_runner_up))
//
// Uses two-pass approach:
//   Pass 0: Compute dot products for all K branches -> find winner + runner-up
//   Pass 1: Compute hinge loss on winner only
//
// This shader handles Pass 0 (dot products + argmax).

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Predicted deltas: (batch, K, D) float32
layout(set = 0, binding = 0) readonly buffer Predictions {
    float preds[];
};

// True delta: bitpacked uint32[] (ceil(D/32) words per batch)
layout(set = 0, binding = 1) readonly buffer TrueDelta {
    uint true_delta[];
};

// Dot products output: (batch, K) — intermediate
layout(set = 0, binding = 2) buffer DotProducts {
    float dots[];
};

// Loss output: (batch,) scalar per batch element
layout(set = 0, binding = 3) buffer LossOutput {
    float loss[];
};

// Results: [winning_k, runner_up_k, dot_winner, dot_runner_up] per batch
layout(set = 0, binding = 4) buffer Results {
    uint results[];  // Packed: [batch * 4] — k_win, k_run, float_dot_win, float_dot_run
};

layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint K;
    uint D;
    uint num_words;     // ceil(D/32)
    float gamma;        // Hinge margin
    float delta_margin; // Contrastive margin
    float lambda_c;     // Contrastive weight
    uint pass_type;     // 0 = dot products, 1 = hinge loss
};

shared float partial_sums[256];

void main() {
    uint batch_idx = gl_WorkGroupID.y;

    if (pass_type == 0u) {
        // PASS 0: Compute dot product for one (batch, k) pair
        uint k = gl_WorkGroupID.x;
        uint tid = gl_LocalInvocationID.x;

        if (k >= K) return;

        float sum = 0.0;
        uint pred_offset = (batch_idx * K + k) * D;
        uint delta_offset = batch_idx * num_words;

        for (uint i = tid; i < D; i += 256) {
            uint word_idx = i / 32;
            uint bit_idx = i % 32;
            float y_true = ((true_delta[delta_offset + word_idx] >> bit_idx) & 1u) != 0u ? 1.0 : -1.0;
            sum += preds[pred_offset + i] * y_true;
        }

        partial_sums[tid] = sum;
        barrier();

        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partial_sums[tid] += partial_sums[tid + stride];
            }
            barrier();
        }

        if (tid == 0) {
            dots[batch_idx * K + k] = partial_sums[0];
        }

    } else {
        // PASS 1: Compute hinge loss on winner (1 workgroup per batch)
        // Requires results[] to be populated by CPU between passes
        uint tid = gl_LocalInvocationID.x;

        uint win_k = results[batch_idx * 4];
        uint pred_offset = (batch_idx * K + win_k) * D;
        uint delta_offset = batch_idx * num_words;

        float hinge_sum = 0.0;
        for (uint i = tid; i < D; i += 256) {
            uint word_idx = i / 32;
            uint bit_idx = i % 32;
            float y_true = ((true_delta[delta_offset + word_idx] >> bit_idx) & 1u) != 0u ? 1.0 : -1.0;
            float z_pred = preds[pred_offset + i];
            float margin = gamma - y_true * z_pred;
            hinge_sum += max(margin, 0.0);
        }

        partial_sums[tid] = hinge_sum;
        barrier();

        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partial_sums[tid] += partial_sums[tid + stride];
            }
            barrier();
        }

        if (tid == 0) {
            float L_hinge = partial_sums[0] / float(D);

            // Contrastive margin
            float dot_win = uintBitsToFloat(results[batch_idx * 4 + 2]);
            float dot_run = uintBitsToFloat(results[batch_idx * 4 + 3]);
            float L_contrast = max(0.0, delta_margin - (dot_win - dot_run));

            loss[batch_idx] = L_hinge + lambda_c * L_contrast;
        }
    }
}
```

**Step 2: Compile to SPIR-V**

Run: `glslc shaders/vsa-surrogate-loss-forward.glsl -o shaders/spv/vsa-surrogate-loss-forward.spv`
Expected: Compiles without errors

**Step 3: Commit**

```bash
git add shaders/vsa-surrogate-loss-forward.glsl shaders/spv/vsa-surrogate-loss-forward.spv
git commit -m "feat(shaders): add vsa-surrogate-loss-forward (dot product + hinge + contrastive)"
```

---

## Task 4: Write the Surrogate Loss Backward Shader

**Files:**
- Create: `shaders/vsa-surrogate-loss-backward.glsl`

**Step 1: Write the backward shader with sparse gradient routing**

Create `shaders/vsa-surrogate-loss-backward.glsl`:

```glsl
#version 450

// VSA Surrogate Loss — Backward Pass (Sparse Gradient Routing)
//
// Only writes gradients to winning_k and runner_up_k branches.
// Other K-2 branches receive exactly 0.0 gradient.
//
// Winner gradient:     grad[win_k, i] = (-y_true / D) * scale   where hinge active
// Runner-up gradient:  grad[run_k, i] = (+y_true / D) * lambda  where contrastive active
//
// Cost: O(2D) instead of O(K*D)

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Predicted deltas: (batch, K, D) — read for hinge margin check
layout(set = 0, binding = 0) readonly buffer Predictions {
    float preds[];
};

// True delta: bitpacked uint32[]
layout(set = 0, binding = 1) readonly buffer TrueDelta {
    uint true_delta[];
};

// Gradient w.r.t. predictions: (batch, K, D) — output, zeroed before dispatch
layout(set = 0, binding = 2) buffer GradPredictions {
    float grad_preds[];
};

// Results from forward: [winning_k, runner_up_k, dot_win, dot_run] per batch
layout(set = 0, binding = 3) readonly buffer Results {
    uint results[];
};

layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint K;
    uint D;
    uint num_words;
    float gamma;
    float delta_margin;
    float lambda_c;
    float grad_scale;    // Global gradient scale (e.g., 1/batch_size)
};

void main() {
    uint batch_idx = gl_WorkGroupID.y;
    uint tid = gl_GlobalInvocationID.x;

    if (tid >= D) return;

    uint win_k = results[batch_idx * 4];
    uint run_k = results[batch_idx * 4 + 1];

    uint word_idx = tid / 32;
    uint bit_idx = tid % 32;
    uint delta_offset = batch_idx * num_words;
    float y_true = ((true_delta[delta_offset + word_idx] >> bit_idx) & 1u) != 0u ? 1.0 : -1.0;

    float inv_D = 1.0 / float(D);

    // Winner: hinge gradient
    {
        uint pred_idx = (batch_idx * K + win_k) * D + tid;
        float z_pred = preds[pred_idx];
        float margin = gamma - y_true * z_pred;

        if (margin > 0.0) {
            grad_preds[pred_idx] += (-y_true * inv_D) * grad_scale;
        }
    }

    // Runner-up: contrastive gradient (push away from true delta)
    if (win_k != run_k) {
        float dot_win = uintBitsToFloat(results[batch_idx * 4 + 2]);
        float dot_run = uintBitsToFloat(results[batch_idx * 4 + 3]);
        float contrastive_margin = delta_margin - (dot_win - dot_run);

        if (contrastive_margin > 0.0) {
            uint pred_idx = (batch_idx * K + run_k) * D + tid;
            grad_preds[pred_idx] += (y_true * inv_D) * lambda_c * grad_scale;
        }
    }
}
```

**Step 2: Compile to SPIR-V**

Run: `glslc shaders/vsa-surrogate-loss-backward.glsl -o shaders/spv/vsa-surrogate-loss-backward.spv`
Expected: Compiles without errors

**Step 3: Commit**

```bash
git add shaders/vsa-surrogate-loss-backward.glsl shaders/spv/vsa-surrogate-loss-backward.spv
git commit -m "feat(shaders): add vsa-surrogate-loss-backward (sparse gradient routing)"
```

---

## Task 5: Write the VSA Loss Node C++ Implementation

**Files:**
- Create: `cpp/include/grilly/autograd/vsa_loss_node.h`
- Create: `cpp/src/autograd/vsa_loss_node.cpp`

**Step 1: Write the header**

Create `cpp/include/grilly/autograd/vsa_loss_node.h`:

```cpp
#pragma once

#include "grilly/autograd/autograd.h"
#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly::autograd {

/// Upload a BitpackedVec to a GPU buffer and wrap as TensorRef(dtype=u32).
/// Returns the buffer_id and a TensorRef descriptor.
///
/// @param pool      BufferPool for GPU allocation
/// @param data      Pointer to uint32_t words
/// @param num_words Number of uint32_t words (ceil(dim/32))
/// @param dim       Original VSA dimension (e.g., 10240)
/// @return TensorRef with dtype=2 (u32), shape=[num_words]
TensorRef upload_bitpacked(BufferPool& pool,
                           const uint32_t* data,
                           uint32_t num_words,
                           uint32_t dim);

/// Dispatch the VSA surrogate loss forward pass on GPU.
///
/// Two-pass approach:
///   Pass 0: Compute dot products for all K branches (GPU)
///   Argmax: Find winning_k and runner_up_k (CPU, tiny)
///   Pass 1: Compute hinge + contrastive loss on winner (GPU)
///
/// @param pool       BufferPool
/// @param batch      CommandBatch for shader dispatch
/// @param cache      PipelineCache for shader lookup
/// @param node       The autograd Node (contains params, inputs, saved buffers)
/// @return loss value (read back from GPU)
float dispatch_vsa_loss_forward(BufferPool& pool,
                                CommandBatch& batch,
                                PipelineCache& cache,
                                Node* node);

/// Dispatch the VSA surrogate loss backward pass on GPU.
/// Writes sparse gradients to grad_preds buffer (winning_k + runner_up_k only).
///
/// @param pool       BufferPool
/// @param batch      CommandBatch
/// @param cache      PipelineCache
/// @param node       The autograd Node
/// @param grad_scale Global gradient scale (e.g., 1/batch_size)
void dispatch_vsa_loss_backward(BufferPool& pool,
                                CommandBatch& batch,
                                PipelineCache& cache,
                                Node* node,
                                float grad_scale);

/// Dispatch the VSA unpack + project forward pass.
///
/// @param pool   BufferPool
/// @param batch  CommandBatch
/// @param cache  PipelineCache
/// @param node   Autograd Node with inputs[0]=vsa_state(u32), saved=W_proj,b_proj
void dispatch_vsa_unpack_project_forward(BufferPool& pool,
                                         CommandBatch& batch,
                                         PipelineCache& cache,
                                         Node* node);

/// Dispatch the VSA unpack + project backward pass.
/// Computes grad_W_proj and grad_b_proj. No gradient for VSA state (discrete).
void dispatch_vsa_unpack_project_backward(BufferPool& pool,
                                          CommandBatch& batch,
                                          PipelineCache& cache,
                                          Node* node);

}  // namespace grilly::autograd
```

**Step 2: Write the implementation**

Create `cpp/src/autograd/vsa_loss_node.cpp`:

```cpp
#include "grilly/autograd/vsa_loss_node.h"
#include <algorithm>
#include <cstring>

namespace grilly::autograd {

TensorRef upload_bitpacked(BufferPool& pool,
                           const uint32_t* data,
                           uint32_t num_words,
                           uint32_t dim) {
    size_t bytes = num_words * sizeof(uint32_t);
    GrillyBuffer buf = pool.acquire(bytes);
    pool.upload(buf, reinterpret_cast<const float*>(data), bytes);

    TensorRef ref{};
    ref.buffer_id = static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(buf.handle));  // BufferPool tracks by handle
    ref.ndim = 1;
    ref.shape[0] = num_words;
    ref.dtype = 2;  // u32
    ref.requires_grad = false;
    return ref;
}

float dispatch_vsa_loss_forward(BufferPool& pool,
                                CommandBatch& batch,
                                PipelineCache& cache,
                                Node* node) {
    VSASurrogateLossParams params;
    std::memcpy(&params, node->params, sizeof(params));

    uint32_t K = params.K;
    uint32_t D = params.D;
    uint32_t num_words = (D + 31) / 32;
    uint32_t batch_size = node->inputs[0].shape[0];

    // Allocate intermediate buffers
    size_t dots_bytes = batch_size * K * sizeof(float);
    GrillyBuffer dots_buf = pool.acquire(dots_bytes);

    size_t results_bytes = batch_size * 4 * sizeof(uint32_t);
    GrillyBuffer results_buf = pool.acquire(results_bytes);

    size_t loss_bytes = batch_size * sizeof(float);
    GrillyBuffer loss_buf = pool.acquire(loss_bytes);

    // Pass 0: Compute dot products for all K branches
    // Dispatch K workgroups per batch element
    batch.dispatch("vsa-surrogate-loss-forward",
                   K, batch_size, 1,      // workgroup count
                   256, 1, 1);            // local size

    batch.submit();
    batch.waitIdle();

    // CPU argmax: find winning_k and runner_up_k per batch
    std::vector<float> dots(batch_size * K);
    pool.download(dots_buf, dots.data(), dots_bytes);

    std::vector<uint32_t> results(batch_size * 4);
    for (uint32_t b = 0; b < batch_size; ++b) {
        float max1 = -1e30f, max2 = -1e30f;
        uint32_t k1 = 0, k2 = 0;
        for (uint32_t k = 0; k < K; ++k) {
            float d = dots[b * K + k];
            if (d > max1) {
                max2 = max1; k2 = k1;
                max1 = d;    k1 = k;
            } else if (d > max2) {
                max2 = d;    k2 = k;
            }
        }
        results[b * 4 + 0] = k1;
        results[b * 4 + 1] = k2;
        uint32_t dot_win_bits, dot_run_bits;
        std::memcpy(&dot_win_bits, &max1, sizeof(float));
        std::memcpy(&dot_run_bits, &max2, sizeof(float));
        results[b * 4 + 2] = dot_win_bits;
        results[b * 4 + 3] = dot_run_bits;
    }

    pool.upload(results_buf,
                reinterpret_cast<const float*>(results.data()),
                results_bytes);

    // Store winning_k in params for backward
    params.winning_k = results[0];      // batch=0 for now
    params.runner_up_k = results[1];
    std::memcpy(node->params, &params, sizeof(params));

    // Save results buffer for backward
    node->saved_buffer_ids[node->num_saved++] =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(results_buf.handle));

    // Pass 1: Compute hinge + contrastive loss
    batch.dispatch("vsa-surrogate-loss-forward",
                   1, batch_size, 1,
                   256, 1, 1);
    batch.submit();
    batch.waitIdle();

    // Read back loss
    std::vector<float> loss_vals(batch_size);
    pool.download(loss_buf, loss_vals.data(), loss_bytes);

    float total_loss = 0.0f;
    for (float l : loss_vals) total_loss += l;
    params.loss_value = total_loss / batch_size;
    std::memcpy(node->params, &params, sizeof(params));

    pool.release(dots_buf);
    pool.release(loss_buf);
    // results_buf kept alive for backward (saved_buffer_ids)

    return params.loss_value;
}

void dispatch_vsa_loss_backward(BufferPool& pool,
                                CommandBatch& batch,
                                PipelineCache& cache,
                                Node* node,
                                float grad_scale) {
    VSASurrogateLossParams params;
    std::memcpy(&params, node->params, sizeof(params));

    uint32_t D = params.D;
    uint32_t batch_size = node->inputs[0].shape[0];

    // Zero the gradient buffer before dispatch
    size_t grad_bytes = batch_size * params.K * D * sizeof(float);
    GrillyBuffer grad_buf = pool.acquire(grad_bytes);
    std::memset(grad_buf.mappedPtr, 0, grad_bytes);

    // Dispatch: ceil(D/256) workgroups per batch element
    uint32_t wg_x = (D + 255) / 256;
    batch.dispatch("vsa-surrogate-loss-backward",
                   wg_x, batch_size, 1,
                   256, 1, 1);
    batch.submit();
    batch.waitIdle();

    // Store grad buffer id for upstream consumption
    node->grad_input_buffers[0] =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(grad_buf.handle));
}

void dispatch_vsa_unpack_project_forward(BufferPool& pool,
                                         CommandBatch& batch,
                                         PipelineCache& cache,
                                         Node* node) {
    // Dispatch: output_dim workgroups (x) * batch_size (y)
    uint32_t output_dim = node->outputs[0].shape[1];
    uint32_t batch_size = node->outputs[0].shape[0];

    batch.dispatch("vsa-unpack-project",
                   output_dim, batch_size, 1,
                   256, 1, 1);
    batch.submit();
    batch.waitIdle();
}

void dispatch_vsa_unpack_project_backward(BufferPool& pool,
                                          CommandBatch& batch,
                                          PipelineCache& cache,
                                          Node* node) {
    // Backward for linear projection: reuse fnn-linear-backward shader
    // Input was unpacked VSA (no gradient needed for discrete input)
    // Only compute grad_W and grad_b
    batch.dispatch("fnn-linear-backward",
                   node->outputs[0].shape[1],  // output_dim workgroups
                   1, 1,
                   256, 1, 1);
    batch.submit();
    batch.waitIdle();
}

}  // namespace grilly::autograd
```

**Step 3: Add to CMakeLists.txt**

In `CMakeLists.txt` at line 152, after `cpp/src/autograd.cpp`, add:

```cmake
    cpp/src/autograd/vsa_loss_node.cpp
```

**Step 4: Verify it compiles**

Run: `cd build && cmake --build . --config Release 2>&1 | tail -5`
Expected: Compiles without errors

**Step 5: Commit**

```bash
git add cpp/include/grilly/autograd/vsa_loss_node.h cpp/src/autograd/vsa_loss_node.cpp CMakeLists.txt
git commit -m "feat(autograd): implement VSA loss node forward/backward dispatch"
```

---

## Task 6: Wire BackwardEngine Dispatch

**Files:**
- Modify: `cpp/include/grilly/autograd/autograd.h:274-299` (add handler declarations)
- Modify: `cpp/src/autograd.cpp:93-123` (add switch cases)

**Step 1: Declare backward handlers**

In `autograd.h`, after `void backward_mean(Node* node);` (around line 299), add:

```cpp
    void backward_vsa_surrogate_loss(Node* node);
    void backward_vsa_unpack_project(Node* node);
```

**Step 2: Add dispatch cases**

In `autograd.cpp`, in the `dispatch_node_backward` switch (around line 121, before `default:`), add:

```cpp
    case OpType::VSASurrogateLoss: backward_vsa_surrogate_loss(node); break;
    case OpType::VSAUnpackProject: backward_vsa_unpack_project(node); break;
```

**Step 3: Implement the handlers**

In `autograd.cpp`, after the last backward handler implementation, add:

```cpp
void BackwardEngine::backward_vsa_surrogate_loss(Node* node) {
    dispatch_vsa_loss_backward(pool_, batch_, cache_, node, 1.0f);
    stats_.gpu_dispatches++;
}

void BackwardEngine::backward_vsa_unpack_project(Node* node) {
    dispatch_vsa_unpack_project_backward(pool_, batch_, cache_, node);
    stats_.gpu_dispatches++;
}
```

Add the include at the top of `autograd.cpp`:

```cpp
#include "grilly/autograd/vsa_loss_node.h"
```

**Step 4: Verify it compiles**

Run: `cd build && cmake --build . --config Release 2>&1 | tail -5`
Expected: Compiles without errors

**Step 5: Commit**

```bash
git add cpp/include/grilly/autograd/autograd.h cpp/src/autograd.cpp
git commit -m "feat(autograd): wire VSA loss + unpack_project into BackwardEngine dispatch"
```

---

## Task 7: Write the VSA Hypernetwork Class

**Files:**
- Create: `cpp/include/grilly/models/vsa_hypernetwork.h`
- Create: `cpp/src/models/vsa_hypernetwork.cpp`

**Step 1: Write the header**

Create `cpp/include/grilly/models/vsa_hypernetwork.h`:

```cpp
#pragma once

#include "grilly/autograd/autograd.h"
#include "grilly/buffer_pool.h"
#include <random>

namespace grilly::models {

using namespace grilly::autograd;

/// VSA Hypernetwork: generates K parallel future state predictions.
///
/// Pipeline: vsa_state (bitpacked) -> unpack+project (10240->768)
///           -> linear (768->1536) -> GELU -> linear (1536->K*10240)
///           -> reshape to [batch, K, 10240]
class VSAHypernetwork {
public:
    /// @param pool      BufferPool for GPU weight allocation
    /// @param d_model   Hidden dimension after projection (default 768)
    /// @param vsa_dim   VSA dimension (default 10240)
    /// @param K         Number of future trajectories (default 4)
    /// @param seed      RNG seed for weight initialization
    VSAHypernetwork(BufferPool& pool,
                    uint32_t d_model = 768,
                    uint32_t vsa_dim = 10240,
                    uint32_t K = 4,
                    uint32_t seed = 42);

    /// Forward pass: records ops on the tape.
    ///
    /// @param tape       TapeContext (must be in recording mode)
    /// @param vsa_state  TensorRef(dtype=u32) — bitpacked VSA state
    /// @return TensorRef shaped [batch, K, vsa_dim] — continuous predictions
    TensorRef forward(TapeContext& tape, TensorRef vsa_state);

    /// Get all weight buffer IDs for optimizer registration.
    std::vector<uint32_t> parameter_buffer_ids() const;

    uint32_t d_model() const { return d_model_; }
    uint32_t vsa_dim() const { return vsa_dim_; }
    uint32_t K() const { return K_; }

private:
    BufferPool& pool_;
    uint32_t d_model_;
    uint32_t vsa_dim_;
    uint32_t K_;

    // Weight buffers (GPU-resident)
    uint32_t W_proj_id_, b_proj_id_;   // Projection: 10240 -> d_model
    uint32_t W1_id_, b1_id_;           // Layer 1: d_model -> d_model*2
    uint32_t W2_id_, b2_id_;           // Layer 2: d_model*2 -> K*vsa_dim

    /// Allocate and initialize a weight buffer with Xavier initialization.
    uint32_t init_weight(uint32_t rows, uint32_t cols, float fan_scale, std::mt19937& rng);

    /// Allocate and zero-initialize a bias buffer.
    uint32_t init_bias(uint32_t size);

    /// Helper: build TensorRef for a weight matrix.
    TensorRef weight_ref(uint32_t buf_id, uint32_t rows, uint32_t cols) const;
};

}  // namespace grilly::models
```

**Step 2: Write the implementation**

Create `cpp/src/models/vsa_hypernetwork.cpp`:

```cpp
#include "grilly/models/vsa_hypernetwork.h"
#include <cmath>
#include <cstring>

namespace grilly::models {

VSAHypernetwork::VSAHypernetwork(BufferPool& pool,
                                 uint32_t d_model,
                                 uint32_t vsa_dim,
                                 uint32_t K,
                                 uint32_t seed)
    : pool_(pool), d_model_(d_model), vsa_dim_(vsa_dim), K_(K) {

    std::mt19937 rng(seed);

    // Projection: (vsa_dim x d_model)  — large, but only used once per step
    float proj_scale = std::sqrt(2.0f / (vsa_dim + d_model));
    W_proj_id_ = init_weight(d_model, vsa_dim, proj_scale, rng);
    b_proj_id_ = init_bias(d_model);

    // Layer 1: (d_model x d_model*2)
    uint32_t hidden = d_model * 2;
    float l1_scale = std::sqrt(2.0f / (d_model + hidden));
    W1_id_ = init_weight(hidden, d_model, l1_scale, rng);
    b1_id_ = init_bias(hidden);

    // Layer 2: (d_model*2 x K*vsa_dim)
    uint32_t out_dim = K * vsa_dim;
    float l2_scale = std::sqrt(2.0f / (hidden + out_dim));
    W2_id_ = init_weight(out_dim, hidden, l2_scale, rng);
    b2_id_ = init_bias(out_dim);
}

TensorRef VSAHypernetwork::forward(TapeContext& tape, TensorRef vsa_state) {
    uint32_t batch = vsa_state.shape[0];

    // Step 1: Unpack + Project (bitpacked u32 -> float, then linear)
    TensorRef h_t{};
    h_t.ndim = 2;
    h_t.shape[0] = batch;
    h_t.shape[1] = d_model_;
    h_t.dtype = 0;  // f32
    h_t.requires_grad = true;

    TensorRef W_proj = weight_ref(W_proj_id_, d_model_, vsa_dim_);
    TensorRef b_proj = weight_ref(b_proj_id_, 1, d_model_);

    TensorRef inputs0[] = {vsa_state};
    TensorRef outputs0[] = {h_t};
    Node* n0 = tape.record_op(OpType::VSAUnpackProject,
                               inputs0, 1, outputs0, 1);
    tape.save_for_backward(n0, &W_proj_id_, 1);
    uint32_t proj_saves[] = {W_proj_id_, b_proj_id_};
    tape.save_for_backward(n0, proj_saves, 2);

    // Step 2: Linear layer 1 (d_model -> hidden)
    uint32_t hidden = d_model_ * 2;
    TensorRef h_mid{};
    h_mid.ndim = 2;
    h_mid.shape[0] = batch;
    h_mid.shape[1] = hidden;
    h_mid.dtype = 0;
    h_mid.requires_grad = true;

    TensorRef inputs1[] = {h_t};
    TensorRef outputs1[] = {h_mid};
    Node* n1 = tape.record_op(OpType::Linear, inputs1, 1, outputs1, 1);
    uint32_t l1_saves[] = {W1_id_, b1_id_};
    tape.save_for_backward(n1, l1_saves, 2);

    // Step 3: GELU activation
    TensorRef h_act{};
    h_act.ndim = 2;
    h_act.shape[0] = batch;
    h_act.shape[1] = hidden;
    h_act.dtype = 0;
    h_act.requires_grad = true;

    TensorRef inputs2[] = {h_mid};
    TensorRef outputs2[] = {h_act};
    tape.record_op(OpType::GELU, inputs2, 1, outputs2, 1);

    // Step 4: Linear layer 2 (hidden -> K * vsa_dim)
    uint32_t out_dim = K_ * vsa_dim_;
    TensorRef raw_deltas{};
    raw_deltas.ndim = 2;
    raw_deltas.shape[0] = batch;
    raw_deltas.shape[1] = out_dim;
    raw_deltas.dtype = 0;
    raw_deltas.requires_grad = true;

    TensorRef inputs3[] = {h_act};
    TensorRef outputs3[] = {raw_deltas};
    Node* n3 = tape.record_op(OpType::Linear, inputs3, 1, outputs3, 1);
    uint32_t l2_saves[] = {W2_id_, b2_id_};
    tape.save_for_backward(n3, l2_saves, 2);

    // Step 5: Reshape to [batch, K, vsa_dim]
    TensorRef predicted_deltas{};
    predicted_deltas.ndim = 3;
    predicted_deltas.shape[0] = batch;
    predicted_deltas.shape[1] = K_;
    predicted_deltas.shape[2] = vsa_dim_;
    predicted_deltas.dtype = 0;
    predicted_deltas.requires_grad = true;
    predicted_deltas.buffer_id = raw_deltas.buffer_id;

    TensorRef inputs4[] = {raw_deltas};
    TensorRef outputs4[] = {predicted_deltas};
    tape.record_op(OpType::Reshape, inputs4, 1, outputs4, 1);

    return predicted_deltas;
}

std::vector<uint32_t> VSAHypernetwork::parameter_buffer_ids() const {
    return {W_proj_id_, b_proj_id_, W1_id_, b1_id_, W2_id_, b2_id_};
}

uint32_t VSAHypernetwork::init_weight(uint32_t rows, uint32_t cols,
                                       float fan_scale, std::mt19937& rng) {
    size_t n = static_cast<size_t>(rows) * cols;
    size_t bytes = n * sizeof(float);
    GrillyBuffer buf = pool_.acquire(bytes);

    std::normal_distribution<float> dist(0.0f, fan_scale);
    auto* ptr = static_cast<float*>(buf.mappedPtr);
    for (size_t i = 0; i < n; ++i) {
        ptr[i] = dist(rng);
    }

    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(buf.handle));
}

uint32_t VSAHypernetwork::init_bias(uint32_t size) {
    size_t bytes = size * sizeof(float);
    GrillyBuffer buf = pool_.acquire(bytes);
    std::memset(buf.mappedPtr, 0, bytes);
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(buf.handle));
}

TensorRef VSAHypernetwork::weight_ref(uint32_t buf_id, uint32_t rows,
                                       uint32_t cols) const {
    TensorRef ref{};
    ref.buffer_id = buf_id;
    ref.ndim = 2;
    ref.shape[0] = rows;
    ref.shape[1] = cols;
    ref.dtype = 0;  // f32
    ref.requires_grad = true;
    return ref;
}

}  // namespace grilly::models
```

**Step 3: Add to CMakeLists.txt**

In `CMakeLists.txt` after the `vsa_loss_node.cpp` line, add:

```cmake
    cpp/src/models/vsa_hypernetwork.cpp
```

**Step 4: Verify it compiles**

Run: `cd build && cmake --build . --config Release 2>&1 | tail -5`
Expected: Compiles without errors

**Step 5: Commit**

```bash
git add cpp/include/grilly/models/vsa_hypernetwork.h cpp/src/models/vsa_hypernetwork.cpp CMakeLists.txt
git commit -m "feat(models): implement VSAHypernetwork MLP (K=4 default)"
```

---

## Task 8: Add Pybind11 Bindings

**Files:**
- Modify: `cpp/python/bindings.cpp:2175-2206` (OpType enum)
- Modify: `cpp/python/bindings.cpp` (add class bindings)

**Step 1: Add new OpType values to Python enum**

In `bindings.cpp`, in the `py::enum_<OpType>` block (around line 2204, before `.export_values()`), add:

```cpp
    .value("VSAUnpackProject", grilly::autograd::OpType::VSAUnpackProject)
    .value("VSASurrogateLoss", grilly::autograd::OpType::VSASurrogateLoss)
```

**Step 2: Add VSAHypernetwork class binding**

After the TapeContext binding block (around line 2248), add:

```cpp
    // ── VSA Hypernetwork ─────────────────────────────────────────────
    py::class_<grilly::models::VSAHypernetwork>(m, "VSAHypernetwork")
        .def(py::init(
                 [](GrillyCoreContext& ctx,
                    uint32_t d_model, uint32_t vsa_dim,
                    uint32_t K, uint32_t seed) {
                     return new grilly::models::VSAHypernetwork(
                         ctx.pool, d_model, vsa_dim, K, seed);
                 }),
             py::arg("device"),
             py::arg("d_model") = 768,
             py::arg("vsa_dim") = 10240,
             py::arg("K") = 4,
             py::arg("seed") = 42,
             py::keep_alive<1, 2>())
        .def("forward",
             [](grilly::models::VSAHypernetwork& self,
                grilly::autograd::TapeContext& tape,
                grilly::autograd::TensorRef vsa_state) {
                 return self.forward(tape, vsa_state);
             },
             py::arg("tape"), py::arg("vsa_state"))
        .def("parameter_buffer_ids",
             &grilly::models::VSAHypernetwork::parameter_buffer_ids)
        .def_property_readonly("d_model",
             &grilly::models::VSAHypernetwork::d_model)
        .def_property_readonly("vsa_dim",
             &grilly::models::VSAHypernetwork::vsa_dim)
        .def_property_readonly("K",
             &grilly::models::VSAHypernetwork::K);
```

Add includes at the top of `bindings.cpp`:

```cpp
#include "grilly/autograd/vsa_loss_node.h"
#include "grilly/models/vsa_hypernetwork.h"
```

**Step 3: Add training step function**

After the VSAHypernetwork binding, add a module-level function:

```cpp
    m.def("vsa_training_step",
          [](GrillyCoreContext& ctx,
             grilly::autograd::TapeContext& tape,
             grilly::models::VSAHypernetwork& model,
             py::array_t<uint32_t> vsa_state_np,
             py::array_t<uint32_t> true_delta_np) -> float {
              // Upload VSA state
              auto state_buf = vsa_state_np.request();
              auto* state_data = static_cast<uint32_t*>(state_buf.ptr);
              uint32_t num_words = static_cast<uint32_t>(state_buf.shape[0]);

              auto state_ref = grilly::autograd::upload_bitpacked(
                  ctx.pool, state_data, num_words, model.vsa_dim());

              tape.begin();

              // Forward
              auto predicted = model.forward(tape, state_ref);

              // Upload true delta
              auto delta_buf = true_delta_np.request();
              auto* delta_data = static_cast<uint32_t*>(delta_buf.ptr);
              auto delta_ref = grilly::autograd::upload_bitpacked(
                  ctx.pool, delta_data, num_words, model.vsa_dim());

              // Record loss
              grilly::autograd::VSASurrogateLossParams loss_params{};
              loss_params.gamma = 0.1f;
              loss_params.delta_margin = 0.5f;
              loss_params.lambda = 0.1f;
              loss_params.K = model.K();
              loss_params.D = model.vsa_dim();

              grilly::autograd::TensorRef loss_output{};
              loss_output.ndim = 1;
              loss_output.shape[0] = 1;
              loss_output.dtype = 0;
              loss_output.requires_grad = false;

              grilly::autograd::TensorRef loss_inputs[] = {predicted, delta_ref};
              grilly::autograd::TensorRef loss_outputs[] = {loss_output};
              auto* loss_node = tape.record_op(
                  grilly::autograd::OpType::VSASurrogateLoss,
                  loss_inputs, 2, loss_outputs, 1,
                  &loss_params, sizeof(loss_params));

              // Forward loss
              float loss = grilly::autograd::dispatch_vsa_loss_forward(
                  ctx.pool, ctx.batch, ctx.cache, loss_node);

              // Backward
              tape.backward(loss_node, 0);

              tape.end();

              return loss;
          },
          py::arg("device"), py::arg("tape"),
          py::arg("model"), py::arg("vsa_state"), py::arg("true_delta"),
          "Run one VSA training step: forward + loss + backward. Returns loss value.");
```

**Step 4: Verify it compiles**

Run: `uv pip install -e . 2>&1 | tail -5`
Expected: Builds and installs successfully

**Step 5: Commit**

```bash
git add cpp/python/bindings.cpp
git commit -m "feat(bindings): expose VSAHypernetwork + vsa_training_step to Python"
```

---

## Task 9: Write Integration Test

**Files:**
- Create: `tests/test_vsa_hypernetwork.py`

**Step 1: Write the test file**

Create `tests/test_vsa_hypernetwork.py`:

```python
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
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_vsa_hypernetwork.py -v`
Expected: All GPU tests pass if Vulkan available; CPU tests always pass

**Step 3: Commit**

```bash
git add tests/test_vsa_hypernetwork.py
git commit -m "test: add VSA hypernetwork + surrogate loss tests"
```

---

## Task 10: Full Build + Verification

**Step 1: Full rebuild**

Run:
```bash
uv pip install -e .
```
Expected: C++ extension compiles, all shaders compile

**Step 2: Run full test suite**

Run:
```bash
uv run pytest tests/ -v
```
Expected: All 82 existing tests pass + new VSA tests pass

**Step 3: Verify Python API**

Run:
```bash
uv run python -c "
import grilly_core
print('OpType.VSASurrogateLoss:', grilly_core.OpType.VSASurrogateLoss)
print('OpType.VSAUnpackProject:', grilly_core.OpType.VSAUnpackProject)
dev = grilly_core.Device()
model = grilly_core.VSAHypernetwork(dev, d_model=32, vsa_dim=256, K=2)
print('Model K:', model.K)
print('Params:', model.parameter_buffer_ids())
print('OK')
"
```
Expected: All prints succeed

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: VSA Hypernetwork + Surrogate Loss — full end-to-end implementation"
```
