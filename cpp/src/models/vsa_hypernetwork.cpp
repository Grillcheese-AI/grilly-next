#include "grilly/models/vsa_hypernetwork.h"
#include "grilly/autograd/vsa_loss_node.h"
#include "grilly/ops/linear.h"
#include <cmath>
#include <cstring>
#include <vulkan/vulkan.h>

namespace grilly::models {

VSAHypernetwork::VSAHypernetwork(BufferPool& pool,
                                 CommandBatch& batch,
                                 PipelineCache& cache,
                                 uint32_t d_model,
                                 uint32_t vsa_dim,
                                 uint32_t K,
                                 uint32_t router_hidden,
                                 uint32_t seed)
    : pool_(pool), batch_(batch), cache_(cache),
      d_model_(d_model), vsa_dim_(vsa_dim), K_(K),
      router_hidden_(router_hidden) {

    std::mt19937 rng(seed);

    // Projection: (d_model x vsa_dim)
    float proj_scale = std::sqrt(2.0f / (vsa_dim + d_model));
    W_proj_id_ = init_weight(d_model, vsa_dim, proj_scale, rng, 0);
    b_proj_id_ = init_bias(d_model, 1);

    // Layer 1: (hidden x d_model) where hidden = d_model * 2
    uint32_t hidden = d_model * 2;
    float l1_scale = std::sqrt(2.0f / (d_model + hidden));
    W1_id_ = init_weight(hidden, d_model, l1_scale, rng, 2);
    b1_id_ = init_bias(hidden, 3);

    // Layer 2: (K*vsa_dim x hidden) — full-rank output, one prediction per branch
    uint32_t out_dim = K * vsa_dim;
    float l2_scale = 1.0f / std::sqrt(float(hidden));
    W2_id_ = init_weight(out_dim, hidden, l2_scale, rng, 4);
    b2_id_ = init_bias(out_dim, 5);

    // Router MLP layer 1: (router_hidden x d_model)
    float r1_scale = 1.0f / std::sqrt(float(d_model));
    W_r1_id_ = init_weight(router_hidden, d_model, r1_scale, rng, 6);
    b_r1_id_ = init_bias(router_hidden, 7);

    // Router MLP layer 2: (K x router_hidden)
    float r2_scale = 1.0f / std::sqrt(float(router_hidden));
    W_r2_id_ = init_weight(K, router_hidden, r2_scale, rng, 8);
    b_r2_id_ = init_bias(K, 9);
}

TensorRef VSAHypernetwork::forward(TapeContext& tape, TensorRef vsa_state) {
    uint32_t batch = (vsa_state.ndim >= 2) ? vsa_state.shape[0] : 1;

    // ── Step 1: Unpack + Project (bitpacked u32 → float, then linear) ──
    size_t h_t_bytes = size_t(batch) * d_model_ * sizeof(float);
    GrillyBuffer h_t_buf = tape.acquire_temp(h_t_bytes);

    TensorRef h_t{};
    h_t.ndim = 2;
    h_t.shape[0] = batch;
    h_t.shape[1] = d_model_;
    h_t.dtype = 0;
    h_t.requires_grad = true;
    h_t.buffer_id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(h_t_buf.handle));

    TensorRef inputs0[] = {vsa_state};
    TensorRef outputs0[] = {h_t};
    Node* n0 = tape.record_op(OpType::VSAUnpackProject,
                               inputs0, 1, outputs0, 1);
    uint64_t proj_saves[] = {W_proj_id_, b_proj_id_};
    tape.save_for_backward(n0, proj_saves, 2);

    grilly::autograd::dispatch_vsa_unpack_project_forward(pool_, batch_, cache_, n0);

    // Save h_t for router MLP access in loss computation
    h_t_last_ = h_t;
    h_t_buf_ = h_t_buf;

    // ── Step 2: Linear layer 1 (d_model → hidden) ──
    uint32_t hidden = d_model_ * 2;
    size_t h_mid_bytes = size_t(batch) * hidden * sizeof(float);
    GrillyBuffer h_mid_buf = tape.acquire_temp(h_mid_bytes);

    TensorRef h_mid{};
    h_mid.ndim = 2;
    h_mid.shape[0] = batch;
    h_mid.shape[1] = hidden;
    h_mid.dtype = 0;
    h_mid.requires_grad = true;
    h_mid.buffer_id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(h_mid_buf.handle));

    dispatchLinear(h_t.buffer_id, W1_id_, b1_id_, h_mid.buffer_id,
                   batch, d_model_, hidden);

    TensorRef inputs1[] = {h_t};
    TensorRef outputs1[] = {h_mid};
    Node* n1 = tape.record_op(OpType::Linear, inputs1, 1, outputs1, 1);
    uint64_t l1_saves[] = {W1_id_, b1_id_};
    tape.save_for_backward(n1, l1_saves, 2);

    // ── Step 3: GELU activation ──
    size_t h_act_bytes = h_mid_bytes;
    GrillyBuffer h_act_buf = tape.acquire_temp(h_act_bytes);

    TensorRef h_act{};
    h_act.ndim = 2;
    h_act.shape[0] = batch;
    h_act.shape[1] = hidden;
    h_act.dtype = 0;
    h_act.requires_grad = true;
    h_act.buffer_id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(h_act_buf.handle));

    dispatchGELU(h_mid.buffer_id, h_act.buffer_id, batch * hidden);

    TensorRef inputs2[] = {h_mid};
    TensorRef outputs2[] = {h_act};
    tape.record_op(OpType::GELU, inputs2, 1, outputs2, 1);

    // ── Step 4: Linear layer 2 (hidden → K * vsa_dim) — full-rank predictions ──
    // Dispatched as K per-branch chunks to stay within maxStorageBufferRange.
    uint32_t out_dim = K_ * vsa_dim_;
    size_t pred_bytes = size_t(batch) * out_dim * sizeof(float);
    GrillyBuffer pred_buf = tape.acquire_temp(pred_bytes);

    // Record as 2D [batch, K*vsa_dim] for backward_linear (it reads shape[1] as output_dim).
    // A 3D [batch, K, vsa_dim] shape would make backward read output_dim = K (e.g., 4),
    // producing a grad_W buffer ~10,000x too small → optimizer OOB → VK_ERROR_DEVICE_LOST.
    TensorRef predicted_flat{};
    predicted_flat.ndim = 2;
    predicted_flat.shape[0] = batch;
    predicted_flat.shape[1] = out_dim;
    predicted_flat.dtype = 0;
    predicted_flat.requires_grad = true;
    predicted_flat.buffer_id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pred_buf.handle));

    dispatchLinearChunked(h_act.buffer_id, W2_id_, b2_id_,
                          predicted_flat.buffer_id,
                          batch, hidden, vsa_dim_, K_);

    TensorRef inputs3[] = {h_act};
    TensorRef outputs3[] = {predicted_flat};
    Node* n3 = tape.record_op(OpType::Linear, inputs3, 1, outputs3, 1);
    uint64_t l2_saves[] = {W2_id_, b2_id_};
    tape.save_for_backward(n3, l2_saves, 2);

    // Expose as 3D [batch, K, vsa_dim] for the loss function and readback
    TensorRef predicted_deltas{};
    predicted_deltas.ndim = 3;
    predicted_deltas.shape[0] = batch;
    predicted_deltas.shape[1] = K_;
    predicted_deltas.shape[2] = vsa_dim_;
    predicted_deltas.dtype = 0;
    predicted_deltas.requires_grad = true;
    predicted_deltas.buffer_id = predicted_flat.buffer_id;

    last_output_ = predicted_deltas;
    last_output_buf_ = pred_buf;
    return predicted_deltas;
}

std::vector<float> VSAHypernetwork::readback_output() const {
    if (!last_output_buf_.mappedPtr)
        return {};
    size_t n = size_t(K_) * vsa_dim_;
    std::vector<float> out(n);
    std::memcpy(out.data(), last_output_buf_.mappedPtr, n * sizeof(float));
    return out;
}

std::vector<uint64_t> VSAHypernetwork::parameter_buffer_ids() const {
    return {W_proj_id_, b_proj_id_, W1_id_, b1_id_, W2_id_, b2_id_,
            W_r1_id_, b_r1_id_, W_r2_id_, b_r2_id_};
}

uint64_t VSAHypernetwork::init_weight(uint32_t rows, uint32_t cols,
                                       float fan_scale, std::mt19937& rng,
                                       int slot) {
    size_t n = static_cast<size_t>(rows) * cols;
    size_t bytes = n * sizeof(float);
    GrillyBuffer buf = pool_.acquire(bytes);

    std::normal_distribution<float> dist(0.0f, fan_scale);
    auto* ptr = static_cast<float*>(buf.mappedPtr);
    for (size_t i = 0; i < n; ++i) {
        ptr[i] = dist(rng);
    }

    weight_bufs_[slot] = buf;
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(buf.handle));
}

uint64_t VSAHypernetwork::init_bias(uint32_t size, int slot) {
    size_t bytes = size * sizeof(float);
    GrillyBuffer buf = pool_.acquire(bytes);
    std::memset(buf.mappedPtr, 0, bytes);
    weight_bufs_[slot] = buf;
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(buf.handle));
}

TensorRef VSAHypernetwork::weight_ref(uint64_t buf_id, uint32_t rows,
                                       uint32_t cols) const {
    TensorRef ref{};
    ref.buffer_id = buf_id;
    ref.ndim = 2;
    ref.shape[0] = rows;
    ref.shape[1] = cols;
    ref.dtype = 0;
    ref.requires_grad = true;
    return ref;
}

void VSAHypernetwork::dispatchLinear(uint64_t in_id, uint64_t w_id,
                                      uint64_t b_id, uint64_t out_id,
                                      uint32_t batch_size, uint32_t in_dim,
                                      uint32_t out_dim) {
    size_t inBytes  = size_t(batch_size) * in_dim  * sizeof(float);
    size_t wBytes   = size_t(out_dim)   * in_dim   * sizeof(float);
    size_t bBytes   = out_dim * sizeof(float);
    size_t outBytes = size_t(batch_size) * out_dim * sizeof(float);

    PipelineEntry pipe = cache_.getOrCreate("fnn-linear", 4, 16);

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(in_id)),  0, inBytes},
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(w_id)),   0, wBytes},
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(b_id)),   0, bBytes},
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(out_id)), 0, outBytes},
    };

    VkDescriptorSet descSet = cache_.allocDescriptorSet("fnn-linear", bufInfos);

    grilly::ops::LinearParams push{batch_size, in_dim, out_dim, 1};
    uint32_t gx = (out_dim + 15) / 16;
    uint32_t gy = (batch_size + 15) / 16;

    batch_.begin();
    batch_.dispatch(pipe.pipeline, pipe.layout, descSet, gx, gy, 1,
                    &push, sizeof(push));
    batch_.submit();
}

void VSAHypernetwork::dispatchGELU(uint64_t in_id, uint64_t out_id,
                                    uint32_t total_elements) {
    size_t bytes = size_t(total_elements) * sizeof(float);

    PipelineEntry pipe = cache_.getOrCreate("activation-gelu", 2, sizeof(uint32_t));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(in_id)),  0, bytes},
        {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(out_id)), 0, bytes},
    };

    VkDescriptorSet descSet = cache_.allocDescriptorSet("activation-gelu", bufInfos);

    uint32_t push = total_elements;
    uint32_t gx = (total_elements + 255) / 256;

    batch_.begin();
    batch_.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                    &push, sizeof(push));
    batch_.submit();
}

void VSAHypernetwork::dispatchLinearChunked(uint64_t in_id, uint64_t w_id,
                                              uint64_t b_id, uint64_t out_id,
                                              uint32_t batch_size,
                                              uint32_t in_dim,
                                              uint32_t chunk_out,
                                              uint32_t num_chunks) {
    // Each chunk k computes: out[k*chunk_out : (k+1)*chunk_out] =
    //   input @ W[k*chunk_out*in_dim : (k+1)*chunk_out*in_dim]^T + b[k*chunk_out : (k+1)*chunk_out]
    //
    // This keeps descriptor ranges ≤ chunk_out * in_dim * 4 bytes per chunk
    // (e.g. 10240 * 1536 * 4 = 60 MB), well within maxStorageBufferRange.

    size_t inBytes   = size_t(batch_size) * in_dim * sizeof(float);
    size_t wChunk    = size_t(chunk_out) * in_dim * sizeof(float);
    size_t bChunk    = size_t(chunk_out) * sizeof(float);
    size_t outChunk  = size_t(batch_size) * chunk_out * sizeof(float);

    PipelineEntry pipe = cache_.getOrCreate("fnn-linear", 4, 16);

    grilly::ops::LinearParams push{batch_size, in_dim, chunk_out, 1};
    uint32_t gx = (chunk_out + 15) / 16;
    uint32_t gy = (batch_size + 15) / 16;

    batch_.begin();

    for (uint32_t k = 0; k < num_chunks; ++k) {
        VkDeviceSize wOffset   = VkDeviceSize(k) * wChunk;
        VkDeviceSize bOffset   = VkDeviceSize(k) * bChunk;
        VkDeviceSize outOffset = VkDeviceSize(k) * outChunk;

        std::vector<VkDescriptorBufferInfo> bufInfos = {
            {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(in_id)),  0, inBytes},
            {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(w_id)),   wOffset, wChunk},
            {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(b_id)),   bOffset, bChunk},
            {reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(out_id)), outOffset, outChunk},
        };

        VkDescriptorSet descSet = cache_.allocDescriptorSet("fnn-linear", bufInfos);

        batch_.dispatch(pipe.pipeline, pipe.layout, descSet, gx, gy, 1,
                        &push, sizeof(push));
    }

    batch_.submit();
}

}  // namespace grilly::models
