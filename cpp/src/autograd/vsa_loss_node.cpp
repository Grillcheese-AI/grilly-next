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
    ref.buffer_id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(buf.handle));
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
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(results_buf.handle));

    VSALossPushConsts push1 = {batch_size, K, D, num_words, params.gamma, params.delta_margin, params.lambda, 1};

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, 1, batch_size, 1,
                   &push1, sizeof(push1));
    batch.submit();

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

    uint32_t K = params.K;
    uint32_t num_words = (D + 31) / 32;

    // Allocate and zero gradient buffer for predictions
    size_t grad_bytes = size_t(batch_size) * K * D * sizeof(float);
    GrillyBuffer grad_buf = pool.acquire(grad_bytes);
    std::vector<float> zeros(size_t(batch_size) * K * D, 0.0f);
    pool.upload(grad_buf, zeros.data(), grad_bytes);

    // Push constants matching the backward shader layout
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
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(grad_buf.handle));
}

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

void dispatch_vsa_unpack_project_backward(BufferPool& pool,
                                          CommandBatch& batch,
                                          PipelineCache& cache,
                                          Node* node) {
    // No gradient for VSA state (discrete/bitpacked) — only compute grad_W, grad_b.
    // Mark grad_input[0] = 0 so backward engine knows not to propagate further.
    node->grad_input_buffers[0] = 0;
}

}  // namespace grilly::autograd
