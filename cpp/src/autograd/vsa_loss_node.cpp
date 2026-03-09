#include "grilly/autograd/vsa_loss_node.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace {
thread_local std::mt19937 tl_rng{std::random_device{}()};
float gumbel_sample() {
    std::uniform_real_distribution<float> dist(1e-8f, 1.0f - 1e-8f);
    float u = dist(tl_rng);
    return -std::log(-std::log(u));
}
}  // namespace

namespace grilly::autograd {

TensorRef upload_bitpacked(TapeContext& tape,
                           const uint32_t* data,
                           uint32_t num_words,
                           uint32_t dim) {
    size_t bytes = num_words * sizeof(uint32_t);
    GrillyBuffer buf = tape.acquire_temp(bytes);
    tape.pool().upload(buf, reinterpret_cast<const float*>(data), bytes);

    TensorRef ref{};
    ref.buffer_id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(buf.handle));
    ref.ndim = 1;
    ref.shape[0] = num_words;
    ref.dtype = 2;  // u32
    ref.requires_grad = false;
    return ref;
}

float dispatch_vsa_loss_forward(TapeContext& tape,
                                CommandBatch& batch,
                                PipelineCache& cache,
                                Node* node) {
    auto& pool = tape.pool();
    VSASurrogateLossParams params;
    std::memcpy(&params, node->params, sizeof(params));

    uint32_t K = params.K;
    uint32_t D = params.D;
    uint32_t num_words = (D + 31) / 32;
    uint32_t batch_size = node->inputs[0].shape[0];

    // Allocate intermediate buffers (step-scoped, released on tape.end())
    size_t dots_bytes = batch_size * K * sizeof(float);
    GrillyBuffer dots_buf = tape.acquire_temp(dots_bytes);

    size_t results_bytes = batch_size * 4 * sizeof(uint32_t);
    GrillyBuffer results_buf = tape.acquire_temp(results_bytes);

    size_t loss_bytes = batch_size * sizeof(float);
    GrillyBuffer loss_buf = tape.acquire_temp(loss_bytes);

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

    // CPU Gumbel-softmax: stochastic winner selection to prevent branch collapse
    std::vector<float> dots(batch_size * K);
    pool.download(dots_buf, dots.data(), dots_bytes);

    float temperature = params.temperature;

    // Allocate branch weights buffer (batch_size * K floats)
    size_t weights_bytes = batch_size * K * sizeof(float);
    GrillyBuffer weights_buf = tape.acquire_temp(weights_bytes);
    std::vector<float> branch_weights(batch_size * K);

    std::vector<uint32_t> results(batch_size * 4);
    for (uint32_t b = 0; b < batch_size; ++b) {
        // Add Gumbel noise scaled by temperature for stochastic selection
        std::vector<float> adjusted(K);
        for (uint32_t k = 0; k < K; ++k) {
            adjusted[k] = dots[b * K + k] + gumbel_sample() * temperature;
        }

        // Argmax on noisy scores to find top-2
        float max1 = -1e30f, max2 = -1e30f;
        uint32_t k1 = 0, k2 = 0;
        for (uint32_t k = 0; k < K; ++k) {
            if (adjusted[k] > max1) {
                max2 = max1; k2 = k1;
                max1 = adjusted[k]; k1 = k;
            } else if (adjusted[k] > max2) {
                max2 = adjusted[k]; k2 = k;
            }
        }

        // Store winner/runner-up using ORIGINAL (unperturbed) dot products
        float dot_win = dots[b * K + k1];
        float dot_run = dots[b * K + k2];
        results[b * 4 + 0] = k1;
        results[b * 4 + 1] = k2;
        uint32_t dot_win_bits, dot_run_bits;
        std::memcpy(&dot_win_bits, &dot_win, sizeof(float));
        std::memcpy(&dot_run_bits, &dot_run, sizeof(float));
        results[b * 4 + 2] = dot_win_bits;
        results[b * 4 + 3] = dot_run_bits;

        // Compute softmax branch weights for all-K backward routing
        float max_dot = *std::max_element(dots.begin() + b * K,
                                          dots.begin() + (b + 1) * K);
        float sum_exp = 0.0f;
        for (uint32_t k = 0; k < K; ++k) {
            branch_weights[b * K + k] = std::exp((dots[b * K + k] - max_dot) / temperature);
            sum_exp += branch_weights[b * K + k];
        }
        for (uint32_t k = 0; k < K; ++k) {
            branch_weights[b * K + k] /= sum_exp;
        }
    }

    pool.upload(results_buf,
                reinterpret_cast<const float*>(results.data()),
                results_bytes);
    pool.upload(weights_buf, branch_weights.data(), weights_bytes);

    // Store winning_k in params for backward
    params.winning_k = results[0];
    params.runner_up_k = results[1];
    std::memcpy(node->params, &params, sizeof(params));

    // Save results buffer and weights buffer for backward
    node->saved_buffer_ids[node->num_saved++] =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(results_buf.handle));
    node->saved_buffer_ids[node->num_saved++] =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(weights_buf.handle));

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

    // dots_buf, loss_buf, results_buf released by tape.end()

    return params.loss_value;
}

void dispatch_vsa_loss_backward_with_buf(BufferPool& pool,
                                          CommandBatch& batch,
                                          PipelineCache& cache,
                                          Node* node,
                                          float grad_scale,
                                          GrillyBuffer& grad_buf) {
    VSASurrogateLossParams params;
    std::memcpy(&params, node->params, sizeof(params));

    uint32_t D = params.D;
    uint32_t batch_size = node->inputs[0].shape[0];

    uint32_t K = params.K;
    uint32_t num_words = (D + 31) / 32;
    size_t grad_bytes = size_t(batch_size) * K * D * sizeof(float);

    // Push constants matching the backward shader layout
    struct {
        uint32_t batch_size, K, D, num_words;
        float gamma, delta_margin, lambda_c, grad_scale;
        float temperature;
    } pushData = {batch_size, K, D, num_words,
                  params.gamma, params.delta_margin, params.lambda, grad_scale,
                  params.temperature};

    PipelineEntry pipe = cache.getOrCreate("vsa-surrogate-loss-backward", 5, sizeof(pushData));

    // Results at num_saved-2, weights at num_saved-1
    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(node->inputs[0].buffer_id)), 0, grad_bytes},
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(node->inputs[1].buffer_id)), 0, size_t(batch_size) * num_words * sizeof(uint32_t)},
        {grad_buf.handle, 0, grad_bytes},
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(node->saved_buffer_ids[node->num_saved - 2])), 0, size_t(batch_size) * 4 * sizeof(uint32_t)},
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(node->saved_buffer_ids[node->num_saved - 1])), 0, size_t(batch_size) * K * sizeof(float)},
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

// ── Cosine Blend Loss ──────────────────────────────────────────────────

float dispatch_cosine_blend_loss_forward(TapeContext& tape,
                                         CommandBatch& batch,
                                         PipelineCache& cache,
                                         Node* node,
                                         const float* router_weights,
                                         uint32_t batch_size) {
    auto& pool = tape.pool();
    VSACosineBlendLossParams params;
    std::memcpy(&params, node->params, sizeof(params));

    uint32_t K = params.K;
    uint32_t D = params.D;
    uint32_t num_words = (D + 31) / 32;

    // ── Pass 0: Compute oracle dot products (reuse surrogate forward shader) ──
    size_t dots_bytes = batch_size * K * sizeof(float);
    GrillyBuffer dots_buf = tape.acquire_temp(dots_bytes);

    // Use existing vsa-surrogate-loss-forward pass 0 for dot products
    size_t results_bytes = batch_size * 4 * sizeof(uint32_t);
    GrillyBuffer results_buf = tape.acquire_temp(results_bytes);
    size_t loss_bytes = batch_size * sizeof(float);
    GrillyBuffer loss_buf_tmp = tape.acquire_temp(loss_bytes);

    struct VSALossPushConsts {
        uint32_t batch_size, K, D, num_words;
        float gamma, delta_margin, lambda_c;
        uint32_t pass_type;
    };

    PipelineEntry dot_pipe = cache.getOrCreate("vsa-surrogate-loss-forward", 5, sizeof(VSALossPushConsts));

    std::vector<VkDescriptorBufferInfo> dotBufInfos = {
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(node->inputs[0].buffer_id)), 0, size_t(batch_size) * K * D * sizeof(float)},
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(node->inputs[1].buffer_id)), 0, size_t(batch_size) * num_words * sizeof(uint32_t)},
        {dots_buf.handle, 0, dots_bytes},
        {loss_buf_tmp.handle, 0, loss_bytes},
        {results_buf.handle, 0, results_bytes},
    };

    VkDescriptorSet dotDescSet = cache.allocDescriptorSet("vsa-surrogate-loss-forward", dotBufInfos);

    VSALossPushConsts push0 = {batch_size, K, D, num_words, 1.0f, 1.0f, 0.0f, 0};

    batch.begin();
    batch.dispatch(dot_pipe.pipeline, dot_pipe.layout, dotDescSet, K, batch_size, 1,
                   &push0, sizeof(push0));
    batch.submit();

    // Download oracle dot products
    std::vector<float> oracle_dots(batch_size * K);
    pool.download(dots_buf, oracle_dots.data(), dots_bytes);

    // ── Upload router weights ──
    size_t weights_bytes = batch_size * K * sizeof(float);
    GrillyBuffer weights_buf = tape.acquire_temp(weights_bytes);
    pool.upload(weights_buf, router_weights, weights_bytes);

    // ── Pass 1: Cosine blend forward ──
    size_t blended_bytes = size_t(batch_size) * D * sizeof(float);
    GrillyBuffer blended_buf = tape.acquire_temp(blended_bytes);

    size_t loss_info_bytes = batch_size * 3 * sizeof(float);
    GrillyBuffer loss_info_buf = tape.acquire_temp(loss_info_bytes);

    struct { uint32_t batch_size, K, D, num_words; }
    push_blend = {batch_size, K, D, num_words};

    PipelineEntry blend_pipe = cache.getOrCreate("vsa-cosine-blend-forward", 5, sizeof(push_blend));

    std::vector<VkDescriptorBufferInfo> blendBufs = {
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(node->inputs[0].buffer_id)), 0, size_t(batch_size) * K * D * sizeof(float)},
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(node->inputs[1].buffer_id)), 0, size_t(batch_size) * num_words * sizeof(uint32_t)},
        {weights_buf.handle, 0, weights_bytes},
        {blended_buf.handle, 0, blended_bytes},
        {loss_info_buf.handle, 0, loss_info_bytes},
    };

    VkDescriptorSet blendDescSet = cache.allocDescriptorSet("vsa-cosine-blend-forward", blendBufs);

    // Must dispatch with gx=1: the shader does shared-memory reduction within
    // a single workgroup. Multiple workgroups would race on loss_out writes.
    // The strided loop handles D > 256 correctly with one workgroup.
    batch.begin();
    batch.dispatch(blend_pipe.pipeline, blend_pipe.layout, blendDescSet,
                   1, batch_size, 1, &push_blend, sizeof(push_blend));
    batch.submit();

    // Download loss info
    std::vector<float> loss_info(batch_size * 3);
    pool.download(loss_info_buf, loss_info.data(), loss_info_bytes);

    // Average cosine loss across batch
    float cosine_loss = 0.0f;
    for (uint32_t b = 0; b < batch_size; ++b)
        cosine_loss += loss_info[b * 3 + 0];
    cosine_loss /= batch_size;

    // ── CPU: Oracle distillation (KL) + entropy regularization ──
    float tau_o = params.tau_oracle;
    float kl_total = 0.0f;
    float entropy_total = 0.0f;

    for (uint32_t b = 0; b < batch_size; ++b) {
        // Oracle softmax
        float max_d = *std::max_element(oracle_dots.begin() + b * K,
                                        oracle_dots.begin() + (b + 1) * K);
        float sum_exp = 0.0f;
        std::vector<float> oracle_w(K);
        for (uint32_t k = 0; k < K; ++k) {
            oracle_w[k] = std::exp((oracle_dots[b * K + k] - max_d) / tau_o);
            sum_exp += oracle_w[k];
        }
        for (uint32_t k = 0; k < K; ++k) oracle_w[k] /= sum_exp;

        // KL divergence: KL(oracle || router)
        float kl = 0.0f;
        float entropy = 0.0f;
        for (uint32_t k = 0; k < K; ++k) {
            float r_w = router_weights[b * K + k];
            if (oracle_w[k] > 1e-8f && r_w > 1e-8f)
                kl += oracle_w[k] * std::log(oracle_w[k] / r_w);
            if (r_w > 1e-8f)
                entropy -= r_w * std::log(r_w);
        }
        kl_total += kl;
        entropy_total += entropy;
    }
    kl_total /= batch_size;
    entropy_total /= batch_size;

    float total_loss = cosine_loss
                     + params.lambda_distill * kl_total
                     - params.lambda_entropy * entropy_total;

    params.loss_value = total_loss;
    std::memcpy(node->params, &params, sizeof(params));

    // Save buffers for backward: weights, blended, loss_info
    node->saved_buffer_ids[node->num_saved++] =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(weights_buf.handle));
    node->saved_buffer_ids[node->num_saved++] =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(blended_buf.handle));
    node->saved_buffer_ids[node->num_saved++] =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(loss_info_buf.handle));

    return total_loss;
}

void dispatch_cosine_blend_loss_backward(BufferPool& pool,
                                          CommandBatch& batch,
                                          PipelineCache& cache,
                                          Node* node,
                                          float grad_scale,
                                          GrillyBuffer& grad_buf) {
    VSACosineBlendLossParams params;
    std::memcpy(&params, node->params, sizeof(params));

    uint32_t K = params.K;
    uint32_t D = params.D;
    uint32_t num_words = (D + 31) / 32;
    uint32_t batch_size = node->inputs[0].shape[0];

    struct {
        uint32_t batch_size, K, D, num_words;
        float grad_scale;
    } pushData = {batch_size, K, D, num_words, grad_scale};

    PipelineEntry pipe = cache.getOrCreate("vsa-cosine-blend-backward", 6, sizeof(pushData));

    // Bindings: predictions, true_delta, weights, blended, loss_info, grad_preds
    // saved_buffer_ids: [num_saved-3]=weights, [num_saved-2]=blended, [num_saved-1]=loss_info
    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(node->inputs[0].buffer_id)), 0, size_t(batch_size) * K * D * sizeof(float)},
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(node->inputs[1].buffer_id)), 0, size_t(batch_size) * num_words * sizeof(uint32_t)},
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(node->saved_buffer_ids[node->num_saved - 3])), 0, size_t(batch_size) * K * sizeof(float)},
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(node->saved_buffer_ids[node->num_saved - 2])), 0, size_t(batch_size) * D * sizeof(float)},
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(node->saved_buffer_ids[node->num_saved - 1])), 0, batch_size * 3 * sizeof(float)},
        {grad_buf.handle, 0, size_t(batch_size) * K * D * sizeof(float)},
    };

    VkDescriptorSet descSet = cache.allocDescriptorSet("vsa-cosine-blend-backward", bufInfos);

    uint32_t gx = (D + 255) / 256;
    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, batch_size, 1,
                   &pushData, sizeof(pushData));
    batch.submit();

    node->grad_input_buffers[0] =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(grad_buf.handle));
}

// ── LoRA Expand Backward ──────────────────────────────────────────────

void dispatch_lora_expand_backward(BufferPool& pool,
                                    CommandBatch& batch,
                                    PipelineCache& cache,
                                    Node* node,
                                    float grad_scale,
                                    GrillyBuffer& grad_coeffs_buf,
                                    GrillyBuffer& grad_basis_buf) {
    VSALoRAExpandParams params;
    std::memcpy(&params, node->params, sizeof(params));

    uint32_t K = params.K;
    uint32_t D = params.D;
    uint32_t rank = params.rank;
    uint32_t batch_size = node->inputs[0].shape[0];

    // saved_buffer_ids: [0]=B_basis, [1]=coefficients
    uint64_t basis_id = node->saved_buffer_ids[0];
    uint64_t coeffs_id = node->saved_buffer_ids[1];

    struct {
        uint32_t batch_size, K, D, rank;
        float grad_scale;
    } pushData = {batch_size, K, D, rank, grad_scale};

    PipelineEntry pipe = cache.getOrCreate("vsa-lora-expand-backward", 5, sizeof(pushData));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(basis_id)), 0, size_t(D) * rank * sizeof(float)},
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(coeffs_id)), 0, size_t(batch_size) * K * rank * sizeof(float)},
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(node->grad_output_buffer)), 0, size_t(batch_size) * K * D * sizeof(float)},
        {grad_coeffs_buf.handle, 0, size_t(batch_size) * K * rank * sizeof(float)},
        {grad_basis_buf.handle, 0, size_t(D) * rank * sizeof(float)},
    };

    VkDescriptorSet descSet = cache.allocDescriptorSet("vsa-lora-expand-backward", bufInfos);

    uint32_t gx = (std::max(D, rank) + 255) / 256;
    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, batch_size, K,
                   &pushData, sizeof(pushData));
    batch.submit();

    // grad_coeffs flows to Linear2 input, grad_basis flows to B_basis weight
    node->grad_input_buffers[0] =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(grad_coeffs_buf.handle));
}

// ── VSA Unpack + Project ─────────────────────────────────────────────

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
