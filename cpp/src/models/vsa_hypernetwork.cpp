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
                                 uint32_t seed)
    : pool_(pool), batch_(batch), cache_(cache),
      d_model_(d_model), vsa_dim_(vsa_dim), K_(K) {

    std::mt19937 rng(seed);

    // Projection: (d_model x vsa_dim)
    float proj_scale = std::sqrt(2.0f / (vsa_dim + d_model));
    W_proj_id_ = init_weight(d_model, vsa_dim, proj_scale, rng);
    b_proj_id_ = init_bias(d_model);

    // Layer 1: (hidden x d_model) where hidden = d_model * 2
    uint32_t hidden = d_model * 2;
    float l1_scale = std::sqrt(2.0f / (d_model + hidden));
    W1_id_ = init_weight(hidden, d_model, l1_scale, rng);
    b1_id_ = init_bias(hidden);

    // Layer 2: (K*vsa_dim x hidden)
    uint32_t out_dim = K * vsa_dim;
    float l2_scale = std::sqrt(2.0f / (hidden + out_dim));
    W2_id_ = init_weight(out_dim, hidden, l2_scale, rng);
    b2_id_ = init_bias(out_dim);
}

TensorRef VSAHypernetwork::forward(TapeContext& tape, TensorRef vsa_state) {
    // vsa_state is bitpacked: shape=[num_words] for a single state (batch=1),
    // or shape=[batch, num_words] for a batched call. Derive batch accordingly.
    uint32_t batch = (vsa_state.ndim >= 2) ? vsa_state.shape[0] : 1;

    // ── Step 1: Unpack + Project (bitpacked u32 → float, then linear) ──
    // Allocate output buffer first so the dispatch has a valid destination.
    size_t h_t_bytes = size_t(batch) * d_model_ * sizeof(float);
    GrillyBuffer h_t_buf = pool_.acquire(h_t_bytes);

    TensorRef h_t{};
    h_t.ndim = 2;
    h_t.shape[0] = batch;
    h_t.shape[1] = d_model_;
    h_t.dtype = 0;  // f32
    h_t.requires_grad = true;
    h_t.buffer_id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(h_t_buf.handle));

    TensorRef inputs0[] = {vsa_state};
    TensorRef outputs0[] = {h_t};
    Node* n0 = tape.record_op(OpType::VSAUnpackProject,
                               inputs0, 1, outputs0, 1);
    uint64_t proj_saves[] = {W_proj_id_, b_proj_id_};
    tape.save_for_backward(n0, proj_saves, 2);

    // Execute the unpack+project shader (uses n0->outputs[0].buffer_id = h_t_buf)
    grilly::autograd::dispatch_vsa_unpack_project_forward(pool_, batch_, cache_, n0);

    // ── Step 2: Linear layer 1 (d_model → hidden) ──
    uint32_t hidden = d_model_ * 2;
    size_t h_mid_bytes = size_t(batch) * hidden * sizeof(float);
    GrillyBuffer h_mid_buf = pool_.acquire(h_mid_bytes);

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
    GrillyBuffer h_act_buf = pool_.acquire(h_act_bytes);

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

    // ── Step 4: Linear layer 2 (hidden → K * vsa_dim) ──
    uint32_t out_dim = K_ * vsa_dim_;
    size_t raw_bytes = size_t(batch) * out_dim * sizeof(float);
    GrillyBuffer raw_buf = pool_.acquire(raw_bytes);

    TensorRef raw_deltas{};
    raw_deltas.ndim = 2;
    raw_deltas.shape[0] = batch;
    raw_deltas.shape[1] = out_dim;
    raw_deltas.dtype = 0;
    raw_deltas.requires_grad = true;
    raw_deltas.buffer_id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(raw_buf.handle));

    dispatchLinear(h_act.buffer_id, W2_id_, b2_id_, raw_deltas.buffer_id,
                   batch, hidden, out_dim);

    TensorRef inputs3[] = {h_act};
    TensorRef outputs3[] = {raw_deltas};
    Node* n3 = tape.record_op(OpType::Linear, inputs3, 1, outputs3, 1);
    uint64_t l2_saves[] = {W2_id_, b2_id_};
    tape.save_for_backward(n3, l2_saves, 2);

    // ── Step 5: Reshape to [batch, K, vsa_dim] (zero-copy, same buffer) ──
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

    last_output_ = predicted_deltas;
    last_output_buf_ = raw_buf;
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
    return {W_proj_id_, b_proj_id_, W1_id_, b1_id_, W2_id_, b2_id_};
}

uint64_t VSAHypernetwork::init_weight(uint32_t rows, uint32_t cols,
                                       float fan_scale, std::mt19937& rng) {
    size_t n = static_cast<size_t>(rows) * cols;
    size_t bytes = n * sizeof(float);
    GrillyBuffer buf = pool_.acquire(bytes);

    std::normal_distribution<float> dist(0.0f, fan_scale);
    auto* ptr = static_cast<float*>(buf.mappedPtr);
    for (size_t i = 0; i < n; ++i) {
        ptr[i] = dist(rng);
    }

    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(buf.handle));
}

uint64_t VSAHypernetwork::init_bias(uint32_t size) {
    size_t bytes = size * sizeof(float);
    GrillyBuffer buf = pool_.acquire(bytes);
    std::memset(buf.mappedPtr, 0, bytes);
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

}  // namespace grilly::models
