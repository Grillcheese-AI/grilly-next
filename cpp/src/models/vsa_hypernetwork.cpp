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
    uint32_t batch = vsa_state.shape[0];

    // Step 1: Unpack + Project (bitpacked u32 -> float, then linear)
    TensorRef h_t{};
    h_t.ndim = 2;
    h_t.shape[0] = batch;
    h_t.shape[1] = d_model_;
    h_t.dtype = 0;  // f32
    h_t.requires_grad = true;

    TensorRef inputs0[] = {vsa_state};
    TensorRef outputs0[] = {h_t};
    Node* n0 = tape.record_op(OpType::VSAUnpackProject,
                               inputs0, 1, outputs0, 1);
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
    ref.dtype = 0;
    ref.requires_grad = true;
    return ref;
}

}  // namespace grilly::models
