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
///           -> linear (768->1536) -> GELU -> linear (1536->K*vsa_dim)
///           -> reshape -> [batch, K, 10240]
///
/// Router MLP: h_t -> linear(768->router_hidden) -> ReLU
///             -> linear(router_hidden->K) -> softmax -> branch weights
class VSAHypernetwork {
public:
    VSAHypernetwork(BufferPool& pool,
                    CommandBatch& batch,
                    PipelineCache& cache,
                    uint32_t d_model = 768,
                    uint32_t vsa_dim = 10240,
                    uint32_t K = 16,
                    uint32_t router_hidden = 32,
                    uint32_t seed = 42);

    /// Forward pass: records ops on the tape AND executes forward shaders.
    TensorRef forward(TapeContext& tape, TensorRef vsa_state);

    /// Get all weight buffer IDs for optimizer registration.
    std::vector<uint64_t> parameter_buffer_ids() const;

    uint32_t d_model() const { return d_model_; }
    uint32_t vsa_dim() const { return vsa_dim_; }
    uint32_t K() const { return K_; }
    uint32_t router_hidden() const { return router_hidden_; }
    uint32_t num_weight_bufs() const { return 10; }

    /// After a forward pass, return the last output TensorRef (all K branches).
    TensorRef last_output() const { return last_output_; }

    /// After a forward pass, return h_t for router MLP computation.
    TensorRef h_t() const { return h_t_last_; }

    /// After a forward pass, return h_t GrillyBuffer (with mappedPtr for CPU readback).
    const GrillyBuffer& h_t_buffer() const { return h_t_buf_; }

    /// Read back the last forward output from host-visible mapped memory.
    /// Returns K * vsa_dim float32 values (the continuous deltas before snap).
    std::vector<float> readback_output() const;

    /// Get weight GrillyBuffer objects for save/load.
    /// Order: W_proj, b_proj, W1, b1, W2, b2, W_r1, b_r1, W_r2, b_r2.
    const GrillyBuffer* weight_buffers() const { return weight_bufs_; }

private:
    BufferPool& pool_;
    CommandBatch& batch_;
    PipelineCache& cache_;
    uint32_t d_model_;
    uint32_t vsa_dim_;
    uint32_t K_;
    uint32_t router_hidden_;

    // Weight buffer IDs (GPU-resident, 64-bit VkBuffer handles)
    uint64_t W_proj_id_, b_proj_id_;   // Projection: vsa_dim -> d_model
    uint64_t W1_id_, b1_id_;           // Layer 1: d_model -> d_model*2
    uint64_t W2_id_, b2_id_;           // Layer 2: d_model*2 -> K*vsa_dim
    uint64_t W_r1_id_, b_r1_id_;      // Router layer 1: [router_hidden, d_model]
    uint64_t W_r2_id_, b_r2_id_;      // Router layer 2: [K, router_hidden]

    // Full GrillyBuffer objects for save/load (retains mappedPtr)
    GrillyBuffer weight_bufs_[12];

    TensorRef last_output_;
    TensorRef h_t_last_;               // Saved h_t for router access
    GrillyBuffer h_t_buf_;             // Retained for CPU readback via mappedPtr
    GrillyBuffer last_output_buf_;     // Retained for readback via mappedPtr

    uint64_t init_weight(uint32_t rows, uint32_t cols, float fan_scale, std::mt19937& rng, int slot);
    uint64_t init_bias(uint32_t size, int slot);
    TensorRef weight_ref(uint64_t buf_id, uint32_t rows, uint32_t cols) const;

    /// Dispatch fnn-linear shader using pre-allocated GPU buffer IDs.
    void dispatchLinear(uint64_t in_id, uint64_t w_id, uint64_t b_id,
                        uint64_t out_id, uint32_t batch_size,
                        uint32_t in_dim, uint32_t out_dim);

    /// Dispatch fnn-linear as K separate per-branch chunks to stay within
    /// maxStorageBufferRange (128 MB on many GPUs). Each chunk computes
    /// [batch, chunk_out] from the same input, reading W/b/out at offsets.
    void dispatchLinearChunked(uint64_t in_id, uint64_t w_id, uint64_t b_id,
                               uint64_t out_id, uint32_t batch_size,
                               uint32_t in_dim, uint32_t chunk_out,
                               uint32_t num_chunks);

    /// Dispatch activation-gelu shader using pre-allocated GPU buffer IDs.
    void dispatchGELU(uint64_t in_id, uint64_t out_id, uint32_t total_elements);
};

}  // namespace grilly::models
