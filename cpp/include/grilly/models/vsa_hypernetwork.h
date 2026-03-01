#pragma once

#include "grilly/autograd/autograd.h"
#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"
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
                    CommandBatch& batch,
                    PipelineCache& cache,
                    uint32_t d_model = 768,
                    uint32_t vsa_dim = 10240,
                    uint32_t K = 4,
                    uint32_t seed = 42);

    /// Forward pass: records ops on the tape AND executes forward shaders.
    /// Each layer allocates its output buffer, dispatches the GPU shader,
    /// then records the op node so backward can route gradients.
    TensorRef forward(TapeContext& tape, TensorRef vsa_state);

    /// Get all weight buffer IDs for optimizer registration.
    std::vector<uint64_t> parameter_buffer_ids() const;

    uint32_t d_model() const { return d_model_; }
    uint32_t vsa_dim() const { return vsa_dim_; }
    uint32_t K() const { return K_; }

    /// After a forward pass, return the last output TensorRef (all K branches).
    /// Python extracts the winner using winning_k from loss params.
    TensorRef last_output() const { return last_output_; }

    /// Read back the last forward output from host-visible mapped memory.
    /// Returns K * vsa_dim float32 values (the continuous deltas before snap).
    std::vector<float> readback_output() const;

private:
    BufferPool& pool_;
    CommandBatch& batch_;
    PipelineCache& cache_;
    uint32_t d_model_;
    uint32_t vsa_dim_;
    uint32_t K_;

    // Weight buffer IDs (GPU-resident, 64-bit VkBuffer handles)
    uint64_t W_proj_id_, b_proj_id_;   // Projection: vsa_dim -> d_model
    uint64_t W1_id_, b1_id_;           // Layer 1: d_model -> d_model*2
    uint64_t W2_id_, b2_id_;           // Layer 2: d_model*2 -> K*vsa_dim

    TensorRef last_output_;
    GrillyBuffer last_output_buf_;  // Retained for readback via mappedPtr

    uint64_t init_weight(uint32_t rows, uint32_t cols, float fan_scale, std::mt19937& rng);
    uint64_t init_bias(uint32_t size);
    TensorRef weight_ref(uint64_t buf_id, uint32_t rows, uint32_t cols) const;

    /// Dispatch fnn-linear shader using pre-allocated GPU buffer IDs.
    void dispatchLinear(uint64_t in_id, uint64_t w_id, uint64_t b_id,
                        uint64_t out_id, uint32_t batch_size,
                        uint32_t in_dim, uint32_t out_dim);

    /// Dispatch activation-gelu shader using pre-allocated GPU buffer IDs.
    void dispatchGELU(uint64_t in_id, uint64_t out_id, uint32_t total_elements);
};

}  // namespace grilly::models
