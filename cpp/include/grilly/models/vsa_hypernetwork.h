#pragma once

#include "grilly/autograd/autograd.h"
#include "grilly/buffer_pool.h"
#include <random>
#include <vector>

namespace grilly::models {

using namespace grilly::autograd;

/// VSA Hypernetwork: generates K parallel future state predictions.
///
/// Pipeline: vsa_state (bitpacked) -> unpack+project (10240->768)
///           -> linear (768->1536) -> GELU -> linear (1536->K*10240)
///           -> reshape to [batch, K, 10240]
class VSAHypernetwork {
public:
    VSAHypernetwork(BufferPool& pool,
                    uint32_t d_model = 768,
                    uint32_t vsa_dim = 10240,
                    uint32_t K = 4,
                    uint32_t seed = 42);

    /// Forward pass: records ops on the tape.
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

    // Weight buffer IDs (GPU-resident)
    uint32_t W_proj_id_, b_proj_id_;   // Projection: vsa_dim -> d_model
    uint32_t W1_id_, b1_id_;           // Layer 1: d_model -> d_model*2
    uint32_t W2_id_, b2_id_;           // Layer 2: d_model*2 -> K*vsa_dim

    uint32_t init_weight(uint32_t rows, uint32_t cols, float fan_scale, std::mt19937& rng);
    uint32_t init_bias(uint32_t size);
    TensorRef weight_ref(uint32_t buf_id, uint32_t rows, uint32_t cols) const;
};

}  // namespace grilly::models
