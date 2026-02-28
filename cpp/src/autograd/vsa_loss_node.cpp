#include "grilly/autograd/vsa_loss_node.h"
#include <algorithm>
#include <cstring>
#include <vector>

namespace grilly::autograd {

TensorRef upload_bitpacked(BufferPool& pool,
                           const uint32_t* data,
                           uint32_t num_words,
                           uint32_t dim) {
    size_t bytes = num_words * sizeof(uint32_t);
    GrillyBuffer buf = pool.acquire(bytes);
    pool.upload(buf, reinterpret_cast<const float*>(data), bytes);

    TensorRef ref{};
    ref.buffer_id = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(buf.handle));
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

    // TODO: Pass 0 — dispatch vsa-surrogate-loss-forward shader with pass_type=0
    // Bind: predictions(0), true_delta(1), dots(2), loss(3), results(4)
    // Push constants: {batch_size, K, D, num_words, gamma, delta_margin, lambda, 0}
    // Dispatch: (K, batch_size, 1) workgroups of (256, 1, 1)
    // batch.submit(); batch.waitIdle();

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
    params.winning_k = results[0];
    params.runner_up_k = results[1];
    std::memcpy(node->params, &params, sizeof(params));

    // Save results buffer for backward
    node->saved_buffer_ids[node->num_saved++] =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(results_buf.handle));

    // TODO: Pass 1 — dispatch vsa-surrogate-loss-forward shader with pass_type=1
    // Same bindings, push constants with pass_type=1
    // Dispatch: (1, batch_size, 1) workgroups
    // batch.submit(); batch.waitIdle();

    // Read back loss
    std::vector<float> loss_vals(batch_size);
    pool.download(loss_buf, loss_vals.data(), loss_bytes);

    float total_loss = 0.0f;
    for (float l : loss_vals) total_loss += l;
    params.loss_value = total_loss / batch_size;
    std::memcpy(node->params, &params, sizeof(params));

    pool.release(dots_buf);
    pool.release(loss_buf);

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

    // Zero the gradient buffer
    size_t grad_bytes = batch_size * params.K * D * sizeof(float);
    GrillyBuffer grad_buf = pool.acquire(grad_bytes);
    std::memset(grad_buf.mappedPtr, 0, grad_bytes);

    // TODO: dispatch vsa-surrogate-loss-backward shader
    // Bind: predictions(0), true_delta(1), grad_preds(2), results(3)
    // Push constants: {batch_size, K, D, num_words, gamma, delta_margin, lambda, grad_scale}
    // Dispatch: (ceil(D/256), batch_size, 1) workgroups
    // batch.submit(); batch.waitIdle();

    node->grad_input_buffers[0] =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(grad_buf.handle));
}

void dispatch_vsa_unpack_project_forward(BufferPool& pool,
                                         CommandBatch& batch,
                                         PipelineCache& cache,
                                         Node* node) {
    uint32_t output_dim = node->outputs[0].shape[1];
    uint32_t batch_size = node->outputs[0].shape[0];

    // TODO: dispatch vsa-unpack-project shader
    // Bind: vsa_data(0), W(1), b(2), output(3)
    // Push constants: {batch_size, vsa_dim, output_dim, num_words}
    // Dispatch: (output_dim, batch_size, 1) workgroups
    // batch.submit(); batch.waitIdle();
}

void dispatch_vsa_unpack_project_backward(BufferPool& pool,
                                          CommandBatch& batch,
                                          PipelineCache& cache,
                                          Node* node) {
    // Backward for projection: reuse fnn-linear-backward pattern
    // No gradient for VSA state (discrete/bitpacked)
    // Only compute grad_W and grad_b

    // TODO: dispatch fnn-linear-backward shader for W_proj, b_proj
    // batch.submit(); batch.waitIdle();
}

}  // namespace grilly::autograd
