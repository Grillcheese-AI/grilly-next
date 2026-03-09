#include "grilly/autograd/autograd.h"
#include "grilly/autograd/vsa_loss_node.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>

namespace grilly {
namespace autograd {

// ═════════════════════════════════════════════════════════════════════════
// BackwardEngine
// ═════════════════════════════════════════════════════════════════════════

BackwardEngine::BackwardEngine(BufferPool& pool, CommandBatch& batch,
                               PipelineCache& cache)
    : pool_(pool), batch_(batch), cache_(cache) {
    clear_grads();
}

void BackwardEngine::backward(TapeArena& tape, Node* loss_node,
                               uint64_t grad_output_buffer) {
    stats_ = {};

    // 1. Seed the loss node's gradient: dL/d(loss) flows in from outside.
    //    For loss nodes that compute their own gradient (like VSASurrogateLoss),
    //    grad_output_buffer may be 0 — the handler allocates internally.
    //    We use a sentinel of 1 to ensure the node gets processed.
    loss_node->grad_output_buffer = (grad_output_buffer != 0)
                                        ? grad_output_buffer : 1ULL;

    // 2. Walk the Wengert list backward: tail → prev → prev → nullptr
    //
    //    Because the allocation order is a valid topological order,
    //    walking backward guarantees that when we process node N,
    //    all nodes that USE node N's output (i.e., nodes allocated AFTER N)
    //    have already been processed and have accumulated their gradient
    //    contributions into the grad_table_.
    Node* current = tape.tail();

    while (current != nullptr) {
        stats_.nodes_visited++;

        // ── Gradient propagation ──────────────────────────────────────
        // If this node's grad_output_buffer hasn't been set directly
        // (e.g., by being the loss node), look up its output buffer(s)
        // in the grad_table_ to see if downstream nodes have accumulated
        // gradients for it.
        if (current->grad_output_buffer == 0) {
            for (uint32_t i = 0; i < current->num_outputs; ++i) {
                uint64_t out_buf = current->outputs[i].buffer_id;
                if (out_buf == 0) continue;
                uint64_t grad = get_grad_buffer(out_buf);
                if (grad != 0) {
                    current->grad_output_buffer = grad;
                    break;
                }
            }
        }

        if (current->grad_output_buffer != 0) {
            stats_.nodes_with_grad++;

            // Dispatch the backward shader for this operation
            dispatch_node_backward(current);

            // Accumulate gradients into input nodes.
            // For each input that requires_grad, find/create a grad buffer
            // and add this node's contribution to it (handles fan-out).
            for (uint32_t i = 0; i < current->num_inputs; ++i) {
                if (!current->inputs[i].requires_grad) continue;
                if (current->grad_input_buffers[i] == 0) continue;

                uint64_t input_buf = current->inputs[i].buffer_id;
                uint64_t& accum = find_or_insert_grad(input_buf);

                if (accum == 0) {
                    // First gradient contribution — just store it
                    accum = current->grad_input_buffers[i];
                } else {
                    // Fan-out: both gradients reference the same buffer.
                    // For the hypernetwork's linear chain (no fan-out),
                    // this branch is not hit.
                    // TODO: dispatch element-wise add shader for fan-out
                }
            }
        }

        current = current->prev_in_tape;
    }

}

void BackwardEngine::dispatch_node_backward(Node* node) {
    switch (node->op) {
        case OpType::Linear:    backward_linear(node); break;
        case OpType::MatMul:    backward_matmul(node); break;
        case OpType::ReLU:      backward_relu(node); break;
        case OpType::GELU:      backward_gelu(node); break;
        case OpType::SiLU:      backward_silu(node); break;
        case OpType::Tanh:      backward_tanh(node); break;
        case OpType::Sigmoid:   backward_sigmoid(node); break;
        case OpType::Softmax:   backward_softmax(node); break;
        case OpType::LayerNorm: backward_layernorm(node); break;
        case OpType::FlashAttention2: backward_attention(node); break;
        case OpType::Conv2d:    backward_conv2d(node); break;
        case OpType::Conv1d:    backward_conv1d(node); break;
        case OpType::Add:       backward_add(node); break;
        case OpType::Sub:       backward_sub(node); break;
        case OpType::Mul:       backward_mul(node); break;
        case OpType::Div:       backward_div(node); break;
        case OpType::CrossEntropy: backward_cross_entropy(node); break;
        case OpType::MSELoss:   backward_mse(node); break;
        case OpType::CubeMindSurprise: backward_cubemind_surprise(node); break;
        case OpType::TemporalSurprise: backward_temporal_surprise(node); break;
        case OpType::Reshape:   backward_reshape(node); break;
        case OpType::Transpose: backward_transpose(node); break;
        case OpType::Sum:       backward_sum(node); break;
        case OpType::Mean:      backward_mean(node); break;
        case OpType::VSASurrogateLoss: backward_vsa_surrogate_loss(node); break;
        case OpType::VSAUnpackProject: backward_vsa_unpack_project(node); break;
        case OpType::VSALoRAExpand: backward_vsa_lora_expand(node); break;
        case OpType::VSACosineBlendLoss: backward_vsa_cosine_blend_loss(node); break;

        default:
            stats_.cpu_fallbacks++;
            break;
    }
}

uint64_t BackwardEngine::get_grad_buffer(uint64_t input_buffer_id) const {
    for (uint32_t i = 0; i < grad_count_; ++i) {
        if (grad_table_[i].buffer_id == input_buffer_id) {
            return grad_table_[i].grad_buffer_id;
        }
    }
    return 0ULL;
}

void BackwardEngine::clear_grads() {
    grad_count_ = 0;
    std::memset(grad_table_, 0, sizeof(grad_table_));
}

uint64_t& BackwardEngine::find_or_insert_grad(uint64_t buffer_id) {
    for (uint32_t i = 0; i < grad_count_; ++i) {
        if (grad_table_[i].buffer_id == buffer_id) {
            return grad_table_[i].grad_buffer_id;
        }
    }
    if (grad_count_ < kMaxGradEntries) {
        grad_table_[grad_count_].buffer_id = buffer_id;
        grad_table_[grad_count_].grad_buffer_id = 0;
        return grad_table_[grad_count_++].grad_buffer_id;
    }
    overflow_slot_ = 0;
    return overflow_slot_;
}

void BackwardEngine::accumulate_grad(Node* target_node, uint32_t input_idx,
                                      uint64_t grad_buffer) {
    if (input_idx < kMaxNodeIO) {
        target_node->grad_input_buffers[input_idx] = grad_buffer;
    }
}

// ═════════════════════════════════════════════════════════════════════════
// Backward Shader Dispatch — Implemented
// ═════════════════════════════════════════════════════════════════════════

void BackwardEngine::backward_linear(Node* node) {
    // Linear: y = x @ W^T + b
    //
    // Node layout (as recorded by VSAHypernetwork::forward):
    //   inputs[0]  = x            (input activation, requires_grad=true)
    //   outputs[0] = y            (output activation)
    //   saved_buffer_ids[0] = W   (weight matrix buffer ID)
    //   saved_buffer_ids[1] = b   (bias vector buffer ID)
    //
    // fnn-linear-backward.glsl bindings:
    //   0: grad_output (dL/dy)   — readonly
    //   1: input_data  (x)       — readonly
    //   2: Weights     (W)       — readonly
    //   3: grad_input  (dL/dx)   — output
    //   4: grad_W      (dL/dW)   — output
    //   5: grad_b      (dL/db)   — output
    //   push: {batch_seq, input_dim, output_dim, pass_type}

    uint32_t batch_seq  = node->outputs[0].shape[0];
    uint32_t output_dim = node->outputs[0].shape[1];
    uint32_t input_dim  = node->inputs[0].shape[1];

    // Check if output_dim * input_dim exceeds maxStorageBufferRange (128 MB).
    // If so, dispatch in chunks to avoid descriptor range violations.
    constexpr size_t kMaxDescriptorRange = 128ULL * 1024 * 1024;  // 128 MB
    size_t w_total_bytes = size_t(output_dim) * input_dim * sizeof(float);
    if (w_total_bytes > kMaxDescriptorRange) {
        backward_linear_chunked(node, batch_seq, output_dim, input_dim);
        return;
    }

    uint64_t grad_out_id = node->grad_output_buffer;
    uint64_t input_id    = node->inputs[0].buffer_id;
    uint64_t W_id        = node->saved_buffer_ids[0];
    uint64_t b_id        = node->saved_buffer_ids[1];

    size_t grad_x_bytes = size_t(batch_seq) * input_dim  * sizeof(float);
    size_t grad_w_bytes = size_t(output_dim) * input_dim  * sizeof(float);
    size_t grad_b_bytes = output_dim * sizeof(float);
    size_t grad_y_bytes = size_t(batch_seq) * output_dim  * sizeof(float);
    size_t x_bytes      = size_t(batch_seq) * input_dim   * sizeof(float);
    size_t w_bytes      = size_t(output_dim) * input_dim  * sizeof(float);

    // Allocate gradient buffers (tracked for release)
    GrillyBuffer grad_x_buf = pool_.acquire(grad_x_bytes);
    backward_bufs_.push_back(grad_x_buf);
    GrillyBuffer grad_w_buf = pool_.acquire(grad_w_bytes);
    backward_bufs_.push_back(grad_w_buf);
    GrillyBuffer grad_b_buf = pool_.acquire(grad_b_bytes);
    backward_bufs_.push_back(grad_b_buf);

    // Zero the output buffers
    std::memset(grad_x_buf.mappedPtr, 0, grad_x_bytes);
    std::memset(grad_w_buf.mappedPtr, 0, grad_w_bytes);
    std::memset(grad_b_buf.mappedPtr, 0, grad_b_bytes);

    uint64_t grad_x_id = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(grad_x_buf.handle));
    uint64_t grad_w_id = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(grad_w_buf.handle));
    uint64_t grad_b_id = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(grad_b_buf.handle));

    PipelineEntry pipe = cache_.getOrCreate(
        "fnn-linear-backward", 6, 4 * sizeof(uint32_t));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(grad_out_id)),
         0, grad_y_bytes},                                           // 0: grad_output
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(input_id)),
         0, x_bytes},                                                // 1: input_data
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(W_id)),
         0, w_bytes},                                                // 2: Weights
        {grad_x_buf.handle, 0, grad_x_bytes},                       // 3: grad_input
        {grad_w_buf.handle, 0, grad_w_bytes},                       // 4: grad_W
        {grad_b_buf.handle, 0, grad_b_bytes},                       // 5: grad_b
    };

    VkDescriptorSet descSet = cache_.allocDescriptorSet(
        "fnn-linear-backward", bufInfos);

    struct { uint32_t batch_seq, input_dim, output_dim, pass_type; } push;
    push.batch_seq = batch_seq;
    push.input_dim = input_dim;
    push.output_dim = output_dim;

    // Pass 0: grad_input = grad_output @ W
    if (node->inputs[0].requires_grad) {
        push.pass_type = 0;
        uint32_t gx = (input_dim + 15) / 16;
        uint32_t gy = (batch_seq + 15) / 16;
        batch_.begin();
        batch_.dispatch(pipe.pipeline, pipe.layout, descSet,
                        gx, gy, 1, &push, sizeof(push));
        batch_.submit();
    }

    // Pass 1: grad_W = grad_output^T @ input_data
    push.pass_type = 1;
    {
        uint32_t gx = (input_dim + 15) / 16;
        uint32_t gy = (output_dim + 15) / 16;
        batch_.begin();
        batch_.dispatch(pipe.pipeline, pipe.layout, descSet,
                        gx, gy, 1, &push, sizeof(push));
        batch_.submit();
    }

    // Pass 2: grad_b = sum(grad_output, dim=0)
    push.pass_type = 2;
    {
        uint32_t gx = (output_dim + 15) / 16;
        batch_.begin();
        batch_.dispatch(pipe.pipeline, pipe.layout, descSet,
                        gx, 1, 1, &push, sizeof(push));
        batch_.submit();
    }

    // Store grad_input for propagation to upstream nodes
    if (node->inputs[0].requires_grad) {
        node->grad_input_buffers[0] = grad_x_id;
    }

    // Store weight/bias gradients in grad_table_ (keyed by buffer ID)
    // so the optimizer can retrieve them via get_grad_buffer(W_id)
    {
        uint64_t& gw = find_or_insert_grad(W_id);
        gw = grad_w_id;
        uint64_t& gb = find_or_insert_grad(b_id);
        gb = grad_b_id;
    }

    stats_.shaders_dispatched += 3;
}

void BackwardEngine::backward_linear_chunked(Node* node, uint32_t batch_seq,
                                               uint32_t output_dim,
                                               uint32_t input_dim) {
    // Chunked backward for large linear layers where W exceeds
    // maxStorageBufferRange (128 MB on NVIDIA). Splits output_dim into
    // chunks and dispatches separate passes with descriptor offsets.
    //
    // For batch_seq=1 (hypernetwork case): grad_output is [output_dim] floats
    // with no stride, so chunking with offsets is straightforward.

    constexpr size_t kMaxRange = 128ULL * 1024 * 1024;
    uint32_t chunk_rows = uint32_t(kMaxRange / (size_t(input_dim) * sizeof(float)));
    if (chunk_rows == 0) chunk_rows = 1;
    uint32_t num_chunks = (output_dim + chunk_rows - 1) / chunk_rows;

    uint64_t grad_out_id = node->grad_output_buffer;
    uint64_t input_id    = node->inputs[0].buffer_id;
    uint64_t W_id        = node->saved_buffer_ids[0];
    uint64_t b_id        = node->saved_buffer_ids[1];

    size_t grad_x_bytes = size_t(batch_seq) * input_dim * sizeof(float);
    size_t grad_w_bytes = size_t(output_dim) * input_dim * sizeof(float);
    size_t grad_b_bytes = output_dim * sizeof(float);
    size_t x_bytes      = size_t(batch_seq) * input_dim * sizeof(float);

    GrillyBuffer grad_x_buf = pool_.acquire(grad_x_bytes);
    backward_bufs_.push_back(grad_x_buf);
    GrillyBuffer grad_w_buf = pool_.acquire(grad_w_bytes);
    backward_bufs_.push_back(grad_w_buf);
    GrillyBuffer grad_b_buf = pool_.acquire(grad_b_bytes);
    backward_bufs_.push_back(grad_b_buf);

    std::memset(grad_x_buf.mappedPtr, 0, grad_x_bytes);
    std::memset(grad_w_buf.mappedPtr, 0, grad_w_bytes);
    std::memset(grad_b_buf.mappedPtr, 0, grad_b_bytes);

    uint64_t grad_x_id = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(grad_x_buf.handle));
    uint64_t grad_w_id = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(grad_w_buf.handle));
    uint64_t grad_b_id = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(grad_b_buf.handle));

    PipelineEntry pipe = cache_.getOrCreate(
        "fnn-linear-backward", 6, 4 * sizeof(uint32_t));

    // Temp buffer for accumulating grad_x across chunks
    GrillyBuffer grad_x_temp{};
    if (node->inputs[0].requires_grad && num_chunks > 1) {
        grad_x_temp = pool_.acquire(grad_x_bytes);
        backward_bufs_.push_back(grad_x_temp);
    }

    // Pass 0: grad_input = grad_output @ W (chunked, accumulate across chunks)
    // For batch_seq=1, grad_output is contiguous [output_dim] — no stride issue.
    if (node->inputs[0].requires_grad) {
        struct { uint32_t batch_seq, input_dim, output_dim, pass_type; } push;
        push.batch_seq = batch_seq;
        push.input_dim = input_dim;
        push.pass_type = 0;

        for (uint32_t c = 0; c < num_chunks; ++c) {
            uint32_t row_start = c * chunk_rows;
            uint32_t rows_this = std::min(chunk_rows, output_dim - row_start);
            push.output_dim = rows_this;

            VkDeviceSize goOff   = VkDeviceSize(row_start) * sizeof(float);
            VkDeviceSize goRange = VkDeviceSize(batch_seq) * rows_this * sizeof(float);
            VkDeviceSize wOff    = VkDeviceSize(row_start) * input_dim * sizeof(float);
            VkDeviceSize wRange  = VkDeviceSize(rows_this) * input_dim * sizeof(float);

            // First chunk writes to grad_x directly; subsequent write to temp
            bool first = (c == 0);
            uint64_t out_id = first ? grad_x_id :
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(grad_x_temp.handle));

            if (!first)
                std::memset(grad_x_temp.mappedPtr, 0, grad_x_bytes);

            std::vector<VkDescriptorBufferInfo> bufInfos = {
                {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(grad_out_id)),
                 goOff, goRange},
                {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(input_id)),
                 0, x_bytes},
                {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(W_id)),
                 wOff, wRange},
                {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(out_id)),
                 0, grad_x_bytes},
                {grad_w_buf.handle, 0, std::min(wRange, grad_w_bytes)},  // unused
                {grad_b_buf.handle, 0, grad_b_bytes},                     // unused
            };

            VkDescriptorSet ds = cache_.allocDescriptorSet(
                "fnn-linear-backward", bufInfos);

            uint32_t gx = (input_dim + 15) / 16;
            uint32_t gy = (batch_seq + 15) / 16;
            batch_.begin();
            batch_.dispatch(pipe.pipeline, pipe.layout, ds,
                            gx, gy, 1, &push, sizeof(push));
            batch_.submit();

            // CPU accumulate: grad_x += temp (batch_seq * input_dim floats)
            if (!first) {
                auto* dst = static_cast<float*>(grad_x_buf.mappedPtr);
                auto* src = static_cast<const float*>(grad_x_temp.mappedPtr);
                size_t n = size_t(batch_seq) * input_dim;
                for (size_t i = 0; i < n; ++i) dst[i] += src[i];
            }
        }
        node->grad_input_buffers[0] = grad_x_id;
    }

    // Passes 1 & 2: grad_W and grad_b (chunked, write to offsets)
    for (uint32_t c = 0; c < num_chunks; ++c) {
        uint32_t row_start = c * chunk_rows;
        uint32_t rows_this = std::min(chunk_rows, output_dim - row_start);

        VkDeviceSize goOff   = VkDeviceSize(row_start) * sizeof(float);
        VkDeviceSize goRange = VkDeviceSize(batch_seq) * rows_this * sizeof(float);
        VkDeviceSize wOff    = VkDeviceSize(row_start) * input_dim * sizeof(float);
        VkDeviceSize wRange  = VkDeviceSize(rows_this) * input_dim * sizeof(float);
        VkDeviceSize bOff    = VkDeviceSize(row_start) * sizeof(float);
        VkDeviceSize bRange  = VkDeviceSize(rows_this) * sizeof(float);

        struct { uint32_t batch_seq, input_dim, output_dim, pass_type; } push;
        push.batch_seq = batch_seq;
        push.input_dim = input_dim;

        std::vector<VkDescriptorBufferInfo> bufInfos = {
            {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(grad_out_id)),
             goOff, goRange},
            {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(node->inputs[0].buffer_id)),
             0, x_bytes},
            {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(W_id)),
             wOff, wRange},
            {grad_x_buf.handle, 0, grad_x_bytes},       // unused in passes 1,2
            {grad_w_buf.handle, wOff, wRange},           // grad_W chunk
            {grad_b_buf.handle, bOff, bRange},           // grad_b chunk
        };

        VkDescriptorSet ds = cache_.allocDescriptorSet(
            "fnn-linear-backward", bufInfos);

        // Pass 1: grad_W[chunk] = grad_output[chunk]^T @ input
        push.output_dim = rows_this;
        push.pass_type = 1;
        {
            uint32_t gx = (input_dim + 15) / 16;
            uint32_t gy = (rows_this + 15) / 16;
            batch_.begin();
            batch_.dispatch(pipe.pipeline, pipe.layout, ds,
                            gx, gy, 1, &push, sizeof(push));
            batch_.submit();
        }

        // Pass 2: grad_b[chunk] = sum(grad_output[chunk], dim=0)
        push.pass_type = 2;
        {
            uint32_t gx = (rows_this + 15) / 16;
            batch_.begin();
            batch_.dispatch(pipe.pipeline, pipe.layout, ds,
                            gx, 1, 1, &push, sizeof(push));
            batch_.submit();
        }
    }

    // Store weight/bias gradients
    {
        uint64_t& gw = find_or_insert_grad(W_id);
        gw = grad_w_id;
        uint64_t& gb = find_or_insert_grad(b_id);
        gb = grad_b_id;
    }

    stats_.shaders_dispatched += num_chunks * 3;
}

void BackwardEngine::backward_gelu(Node* node) {
    // GELU backward: dL/dx = dL/dy * gelu'(x)
    //
    // Node layout:
    //   inputs[0]  = x (pre-activation)
    //   outputs[0] = y = gelu(x)
    //   grad_output_buffer = dL/dy
    //
    // activation-gelu-backward.glsl bindings:
    //   0: grad_output (dL/dy)   — readonly
    //   1: input_data  (x)       — readonly
    //   2: grad_input  (dL/dx)   — output
    //   push: {total_elements}

    uint32_t total = node->inputs[0].numel();
    size_t bytes = total * sizeof(float);

    uint64_t grad_out_id = node->grad_output_buffer;
    uint64_t input_id    = node->inputs[0].buffer_id;

    GrillyBuffer grad_in_buf = pool_.acquire(bytes);
    backward_bufs_.push_back(grad_in_buf);
    std::memset(grad_in_buf.mappedPtr, 0, bytes);

    uint64_t grad_in_id = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(grad_in_buf.handle));

    PipelineEntry pipe = cache_.getOrCreate(
        "activation-gelu-backward", 3, sizeof(uint32_t));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(grad_out_id)),
         0, bytes},                                    // 0: grad_output
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(input_id)),
         0, bytes},                                    // 1: input_data
        {grad_in_buf.handle, 0, bytes},                // 2: grad_input
    };

    VkDescriptorSet descSet = cache_.allocDescriptorSet(
        "activation-gelu-backward", bufInfos);

    uint32_t gx = (total + 255) / 256;
    batch_.begin();
    batch_.dispatch(pipe.pipeline, pipe.layout, descSet,
                    gx, 1, 1, &total, sizeof(total));
    batch_.submit();

    if (node->inputs[0].requires_grad) {
        node->grad_input_buffers[0] = grad_in_id;
    }

    stats_.shaders_dispatched++;
}

// ═════════════════════════════════════════════════════════════════════════
// Backward Shader Dispatch — Stubs (not needed for VSA hypernetwork)
// ═════════════════════════════════════════════════════════════════════════

void BackwardEngine::backward_matmul(Node* node) {
    stats_.shaders_dispatched++;
    if (node->inputs[0].requires_grad)
        node->grad_input_buffers[0] = 1;
    if (node->inputs[1].requires_grad)
        node->grad_input_buffers[1] = 1;
}

void BackwardEngine::backward_relu(Node* node) {
    stats_.shaders_dispatched++;
    if (node->inputs[0].requires_grad)
        node->grad_input_buffers[0] = 1;
}

void BackwardEngine::backward_silu(Node* node) {
    stats_.shaders_dispatched++;
    if (node->inputs[0].requires_grad)
        node->grad_input_buffers[0] = 1;
}

void BackwardEngine::backward_tanh(Node* node) {
    stats_.shaders_dispatched++;
    if (node->inputs[0].requires_grad)
        node->grad_input_buffers[0] = 1;
}

void BackwardEngine::backward_sigmoid(Node* node) {
    stats_.shaders_dispatched++;
    if (node->inputs[0].requires_grad)
        node->grad_input_buffers[0] = 1;
}

void BackwardEngine::backward_softmax(Node* node) {
    stats_.shaders_dispatched++;
    if (node->inputs[0].requires_grad)
        node->grad_input_buffers[0] = 1;
}

void BackwardEngine::backward_layernorm(Node* node) {
    stats_.shaders_dispatched++;
    if (node->inputs[0].requires_grad)
        node->grad_input_buffers[0] = 1;
}

void BackwardEngine::backward_attention(Node* node) {
    stats_.shaders_dispatched++;
    for (uint32_t i = 0; i < node->num_inputs && i < 3; ++i) {
        if (node->inputs[i].requires_grad)
            node->grad_input_buffers[i] = 1;
    }
}

void BackwardEngine::backward_conv2d(Node* node) {
    stats_.shaders_dispatched++;
    if (node->inputs[0].requires_grad)
        node->grad_input_buffers[0] = 1;
    if (node->num_inputs > 1 && node->inputs[1].requires_grad)
        node->grad_input_buffers[1] = 1;
}

void BackwardEngine::backward_conv1d(Node* node) {
    stats_.shaders_dispatched++;
    if (node->inputs[0].requires_grad)
        node->grad_input_buffers[0] = 1;
    if (node->num_inputs > 1 && node->inputs[1].requires_grad)
        node->grad_input_buffers[1] = 1;
}

void BackwardEngine::backward_add(Node* node) {
    for (uint32_t i = 0; i < node->num_inputs; ++i) {
        if (node->inputs[i].requires_grad)
            node->grad_input_buffers[i] = node->grad_output_buffer;
    }
}

void BackwardEngine::backward_sub(Node* node) {
    if (node->inputs[0].requires_grad)
        node->grad_input_buffers[0] = node->grad_output_buffer;
    if (node->num_inputs > 1 && node->inputs[1].requires_grad) {
        node->grad_input_buffers[1] = 1;
        stats_.shaders_dispatched++;
    }
}

void BackwardEngine::backward_mul(Node* node) {
    stats_.shaders_dispatched++;
    if (node->inputs[0].requires_grad)
        node->grad_input_buffers[0] = 1;
    if (node->num_inputs > 1 && node->inputs[1].requires_grad)
        node->grad_input_buffers[1] = 1;
}

void BackwardEngine::backward_div(Node* node) {
    stats_.shaders_dispatched++;
    if (node->inputs[0].requires_grad)
        node->grad_input_buffers[0] = 1;
    if (node->num_inputs > 1 && node->inputs[1].requires_grad)
        node->grad_input_buffers[1] = 1;
}

void BackwardEngine::backward_cross_entropy(Node* node) {
    stats_.shaders_dispatched++;
    if (node->inputs[0].requires_grad)
        node->grad_input_buffers[0] = 1;
}

void BackwardEngine::backward_mse(Node* node) {
    stats_.shaders_dispatched++;
    if (node->inputs[0].requires_grad)
        node->grad_input_buffers[0] = 1;
}

void BackwardEngine::backward_cubemind_surprise(Node* node) {
    float surprise = node->emotion.surprise;
    float alpha = 0.5f;
    if (node->params_size >= sizeof(float))
        std::memcpy(&alpha, node->params, sizeof(float));

    if (node->inputs[0].requires_grad) {
        float multiplier = 1.0f + alpha * surprise;
        std::memcpy(node->params, &multiplier, sizeof(float));
        node->params_size = sizeof(float);
        node->grad_input_buffers[0] = node->grad_output_buffer;
        stats_.shaders_dispatched++;
    }
}

void BackwardEngine::backward_temporal_surprise(Node* node) {
    TemporalSurpriseParams tparams;
    std::memcpy(&tparams, node->params, sizeof(TemporalSurpriseParams));

    if (node->inputs[0].requires_grad) {
        float multiplier = tparams.temporal_multiplier * tparams.alpha;
        if (multiplier > 1.0f) multiplier = 1.0f;
        if (multiplier < -1.0f) multiplier = -1.0f;
        std::memcpy(node->params, &multiplier, sizeof(float));
        node->params_size = sizeof(float);
        node->grad_input_buffers[0] = node->grad_output_buffer;
        stats_.shaders_dispatched++;
    }
}

// Shape ops — no GPU shader needed

void BackwardEngine::backward_reshape(Node* node) {
    if (node->inputs[0].requires_grad) {
        node->grad_input_buffers[0] = node->grad_output_buffer;
    }
}

void BackwardEngine::backward_transpose(Node* node) {
    stats_.shaders_dispatched++;
    if (node->inputs[0].requires_grad)
        node->grad_input_buffers[0] = 1;
}

void BackwardEngine::backward_sum(Node* node) {
    if (node->inputs[0].requires_grad) {
        node->grad_input_buffers[0] = 1;
        stats_.shaders_dispatched++;
    }
}

void BackwardEngine::backward_mean(Node* node) {
    if (node->inputs[0].requires_grad) {
        node->grad_input_buffers[0] = 1;
        stats_.shaders_dispatched++;
    }
}

void BackwardEngine::backward_vsa_surrogate_loss(Node* node) {
    autograd::VSASurrogateLossParams params;
    std::memcpy(&params, node->params, sizeof(params));
    uint32_t batch_size = node->inputs[0].shape[0];
    size_t grad_bytes = size_t(batch_size) * params.K * params.D * sizeof(float);
    GrillyBuffer grad_buf = pool_.acquire(grad_bytes);
    backward_bufs_.push_back(grad_buf);

    std::memset(grad_buf.mappedPtr, 0, grad_bytes);

    dispatch_vsa_loss_backward_with_buf(pool_, batch_, cache_, node, 1.0f, grad_buf);
    stats_.shaders_dispatched++;
}

void BackwardEngine::backward_vsa_unpack_project(Node* node) {
    // UnpackProject: output = unpack(vsa_bitpacked) @ W^T + b
    // VSA state is discrete/bitpacked — no gradient for input.
    // grad_W and grad_b computed via GPU shader that unpacks on-the-fly.

    uint32_t batch_size = node->outputs[0].shape[0];
    uint32_t output_dim = node->outputs[0].shape[1];
    uint32_t num_words  = node->inputs[0].shape[0];
    uint32_t vsa_dim    = num_words * 32;

    uint64_t grad_out_id = node->grad_output_buffer;
    uint64_t vsa_id      = node->inputs[0].buffer_id;
    uint64_t W_id        = node->saved_buffer_ids[0];
    uint64_t b_id        = node->saved_buffer_ids[1];

    size_t grad_out_bytes = size_t(batch_size) * output_dim * sizeof(float);
    size_t vsa_bytes      = size_t(num_words) * sizeof(uint32_t);
    size_t grad_w_bytes   = size_t(output_dim) * vsa_dim * sizeof(float);
    size_t grad_b_bytes   = output_dim * sizeof(float);

    // Allocate gradient buffers
    GrillyBuffer grad_w_buf = pool_.acquire(grad_w_bytes);
    backward_bufs_.push_back(grad_w_buf);
    GrillyBuffer grad_b_buf = pool_.acquire(grad_b_bytes);
    backward_bufs_.push_back(grad_b_buf);

    std::memset(grad_w_buf.mappedPtr, 0, grad_w_bytes);
    std::memset(grad_b_buf.mappedPtr, 0, grad_b_bytes);

    uint64_t grad_w_id = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(grad_w_buf.handle));
    uint64_t grad_b_id = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(grad_b_buf.handle));

    PipelineEntry pipe = cache_.getOrCreate(
        "vsa-unpack-project-backward", 4, 5 * sizeof(uint32_t));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(grad_out_id)),
         0, grad_out_bytes},                                            // 0: grad_output
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(vsa_id)),
         0, vsa_bytes},                                                 // 1: vsa_data
        {grad_w_buf.handle, 0, grad_w_bytes},                          // 2: grad_W
        {grad_b_buf.handle, 0, grad_b_bytes},                          // 3: grad_b
    };

    VkDescriptorSet descSet = cache_.allocDescriptorSet(
        "vsa-unpack-project-backward", bufInfos);

    struct { uint32_t batch_size, vsa_dim, output_dim, num_words, pass_type; } push;
    push.batch_size = batch_size;
    push.vsa_dim    = vsa_dim;
    push.output_dim = output_dim;
    push.num_words  = num_words;

    // Pass 0: grad_W = grad_output^T @ unpack(vsa)
    push.pass_type = 0;
    {
        uint32_t gx = (vsa_dim + 15) / 16;
        uint32_t gy = (output_dim + 15) / 16;
        batch_.begin();
        batch_.dispatch(pipe.pipeline, pipe.layout, descSet,
                        gx, gy, 1, &push, sizeof(push));
        batch_.submit();
        stats_.shaders_dispatched++;
    }

    // Pass 1: grad_b = sum(grad_output, dim=0)
    push.pass_type = 1;
    {
        uint32_t gx = (output_dim + 15) / 16;
        batch_.begin();
        batch_.dispatch(pipe.pipeline, pipe.layout, descSet,
                        gx, 1, 1, &push, sizeof(push));
        batch_.submit();
        stats_.shaders_dispatched++;
    }

    // No gradient for VSA input (discrete/bitpacked)
    node->grad_input_buffers[0] = 0;

    // Store weight/bias gradients in grad_table_ for optimizer
    {
        uint64_t& gw = find_or_insert_grad(W_id);
        gw = grad_w_id;
        uint64_t& gb = find_or_insert_grad(b_id);
        gb = grad_b_id;
    }
}

void BackwardEngine::backward_vsa_lora_expand(Node* node) {
    VSALoRAExpandParams params;
    std::memcpy(&params, node->params, sizeof(params));
    uint32_t batch_size = node->inputs[0].shape[0];

    size_t grad_coeffs_bytes = size_t(batch_size) * params.K * params.rank * sizeof(float);
    GrillyBuffer grad_coeffs_buf = pool_.acquire(grad_coeffs_bytes);
    backward_bufs_.push_back(grad_coeffs_buf);
    std::memset(grad_coeffs_buf.mappedPtr, 0, grad_coeffs_bytes);

    size_t grad_basis_bytes = size_t(params.D) * params.rank * sizeof(float);
    GrillyBuffer grad_basis_buf = pool_.acquire(grad_basis_bytes);
    backward_bufs_.push_back(grad_basis_buf);
    std::memset(grad_basis_buf.mappedPtr, 0, grad_basis_bytes);

    dispatch_lora_expand_backward(pool_, batch_, cache_, node, 1.0f,
                                   grad_coeffs_buf, grad_basis_buf);

    // Store B_basis gradient in grad_table_ for optimizer
    uint64_t basis_id = node->saved_buffer_ids[0];
    uint64_t grad_basis_id = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(grad_basis_buf.handle));
    uint64_t& gb = find_or_insert_grad(basis_id);
    gb = grad_basis_id;

    stats_.shaders_dispatched++;
}

void BackwardEngine::backward_vsa_cosine_blend_loss(Node* node) {
    VSACosineBlendLossParams params;
    std::memcpy(&params, node->params, sizeof(params));
    uint32_t batch_size = node->inputs[0].shape[0];

    size_t grad_bytes = size_t(batch_size) * params.K * params.D * sizeof(float);
    GrillyBuffer grad_buf = pool_.acquire(grad_bytes);
    backward_bufs_.push_back(grad_buf);
    std::memset(grad_buf.mappedPtr, 0, grad_bytes);

    // Moderate gradient amplification: 10x compensates for cosine's
    // intrinsic 1/(||b||*sqrt(D)) scaling without saturating the ±1.0
    // per-element clamp in the backward shader.
    float grad_scale = 10.0f;
    dispatch_cosine_blend_loss_backward(pool_, batch_, cache_, node, grad_scale, grad_buf);
    stats_.shaders_dispatched++;
}

// ═════════════════════════════════════════════════════════════════════════
// TapeContext
// ═════════════════════════════════════════════════════════════════════════

TapeContext::TapeContext(BufferPool& pool, CommandBatch& batch,
                         PipelineCache& cache, size_t arena_capacity)
    : pool_(pool), arena_(arena_capacity), engine_(pool, batch, cache) {}

GrillyBuffer TapeContext::acquire_temp(size_t bytes) {
    GrillyBuffer buf = pool_.acquire(bytes);
    step_bufs_.push_back(buf);
    return buf;
}

void TapeContext::begin() {
    arena_.reset();
    engine_.clear_grads();
    seq_counter_ = 0;
    recording_ = true;
}

Node* TapeContext::record_op(OpType op,
                              const TensorRef* inputs, uint32_t num_inputs,
                              const TensorRef* outputs, uint32_t num_outputs,
                              const void* params, uint32_t params_size) {
    if (!recording_) return nullptr;

    Node* node = arena_.allocate_node<Node>();

    node->op = op;
    node->seq = seq_counter_++;
    node->num_inputs = num_inputs;
    node->num_outputs = num_outputs;
    node->num_saved = 0;
    node->grad_output_buffer = 0;
    node->params_size = 0;
    node->emotion = {0.0f, 0.0f};

    std::memset(node->inputs, 0, sizeof(node->inputs));
    std::memset(node->outputs, 0, sizeof(node->outputs));
    std::memset(node->saved_buffer_ids, 0, sizeof(node->saved_buffer_ids));
    std::memset(node->grad_input_buffers, 0, sizeof(node->grad_input_buffers));

    for (uint32_t i = 0; i < num_inputs && i < kMaxNodeIO; ++i) {
        node->inputs[i] = inputs[i];
    }
    for (uint32_t i = 0; i < num_outputs && i < kMaxNodeIO; ++i) {
        node->outputs[i] = outputs[i];
    }

    if (params && params_size > 0 && params_size <= sizeof(node->params)) {
        std::memcpy(node->params, params, params_size);
        node->params_size = params_size;
    }

    return node;
}

void TapeContext::save_for_backward(Node* node, const uint64_t* buffer_ids,
                                     uint32_t count) {
    if (!node) return;
    for (uint32_t i = 0; i < count && i < kMaxNodeIO; ++i) {
        node->saved_buffer_ids[i] = buffer_ids[i];
    }
    node->num_saved = std::min(count, static_cast<uint32_t>(kMaxNodeIO));
}

void TapeContext::backward(Node* loss_node, uint64_t grad_output_buffer) {
    recording_ = false;
    engine_.backward(arena_, loss_node, grad_output_buffer);
}

uint64_t TapeContext::get_grad_buffer(uint64_t input_buffer_id) const {
    return engine_.get_grad_buffer(input_buffer_id);
}

void TapeContext::end() {
    for (auto& buf : step_bufs_) {
        pool_.release(buf);
    }
    step_bufs_.clear();

    for (auto& buf : engine_.backward_bufs_) {
        pool_.release(buf);
    }
    engine_.backward_bufs_.clear();

    arena_.reset();
    engine_.clear_grads();
    seq_counter_ = 0;
    recording_ = false;
}

}  // namespace autograd
}  // namespace grilly
