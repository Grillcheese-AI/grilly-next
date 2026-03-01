#include "grilly/generation/many_worlds.h"

#include <algorithm>
#include <cstring>

namespace grilly {
namespace generation {

std::vector<uint32_t> snap_to_bipolar(const float* continuous_data,
                                       uint32_t K, uint32_t D) {
    uint32_t words_per_vec = (D + 31) / 32;
    std::vector<uint32_t> bitpacked(K * words_per_vec, 0);

    for (uint32_t k = 0; k < K; ++k) {
        for (uint32_t i = 0; i < D; ++i) {
            // Signum: > 0 is +1 (bit=1), <= 0 is -1 (bit=0)
            if (continuous_data[k * D + i] > 0.0f) {
                uint32_t word_idx = k * words_per_vec + (i / 32);
                uint32_t bit_idx = i % 32;
                bitpacked[word_idx] |= (1u << bit_idx);
            }
        }
    }
    return bitpacked;
}

ManyWorldsResult evaluate_many_worlds(
    CommandBatch& batch,
    BufferPool& pool,
    PipelineCache& cache,
    const uint32_t* current_state,
    const float* continuous_deltas,
    uint32_t K,
    uint32_t D,
    cognitive::WorldModel& world_model) {

    uint32_t words_per_vec = (D + 31) / 32;
    uint32_t num_constraints = world_model.constraint_count();

    // 1. CPU: Snap continuous pre-activations to bitpacked deltas
    std::vector<uint32_t> snapped_deltas =
        snap_to_bipolar(continuous_deltas, K, D);

    ManyWorldsResult result;
    result.K = K;
    result.words_per_vec = words_per_vec;
    result.violation_counts.resize(K, 0);

    // If no constraints loaded, all trajectories are coherent
    if (num_constraints == 0) {
        // Compute future states on CPU (just XOR, practically instant)
        result.future_states.resize(K * words_per_vec);
        for (uint32_t k = 0; k < K; ++k) {
            for (uint32_t w = 0; w < words_per_vec; ++w) {
                result.future_states[k * words_per_vec + w] =
                    current_state[w] ^
                    snapped_deltas[k * words_per_vec + w];
            }
        }
        result.best_k = 0;
        return result;
    }

    // 2. Vulkan resource allocation
    size_t state_bytes = words_per_vec * sizeof(uint32_t);
    size_t deltas_bytes = K * words_per_vec * sizeof(uint32_t);
    size_t scores_bytes = K * sizeof(uint32_t);

    GrillyBuffer buf_state = pool.acquire(state_bytes);
    GrillyBuffer buf_deltas = pool.acquire(deltas_bytes);
    GrillyBuffer buf_scores = pool.acquire(scores_bytes);

    // upload/download take float* — reinterpret_cast is safe since both
    // types are 4 bytes and we're just doing memcpy underneath.
    pool.upload(buf_state,
                reinterpret_cast<const float*>(current_state), state_bytes);
    pool.upload(buf_deltas,
                reinterpret_cast<const float*>(snapped_deltas.data()),
                deltas_bytes);

    // Zero-initialize scores buffer
    std::vector<uint32_t> zero_scores(K, 0);
    pool.upload(buf_scores,
                reinterpret_cast<const float*>(zero_scores.data()),
                scores_bytes);

    // Get the WorldModel constraints GPU buffer directly (no re-upload)
    GrillyBuffer& buf_constraints = world_model.constraints_gpu_buffer();
    size_t constraints_bytes =
        size_t(num_constraints) * words_per_vec * sizeof(uint32_t);

    // 3. Pipeline & descriptor setup
    // 4 bindings, 12 bytes push constants (3 x uint32)
    PipelineEntry pipe =
        cache.getOrCreate("many-worlds-coherence", 4, 12);

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {buf_state.handle, 0, state_bytes},
        {buf_deltas.handle, 0, deltas_bytes},
        {buf_constraints.handle, 0, constraints_bytes},
        {buf_scores.handle, 0, scores_bytes}};

    VkDescriptorSet descSet =
        cache.allocDescriptorSet("many-worlds-coherence", bufInfos);

    // Push constants: words_per_vec, num_constraints, distance_thresh
    // 45% Hamming distance threshold — futures closer than this to a
    // constraint are violations (they "look like" a known falsehood).
    struct {
        uint32_t words_per_vec;
        uint32_t num_constraints;
        uint32_t distance_thresh;
    } pushData = {words_per_vec, num_constraints,
                  static_cast<uint32_t>(D * 0.45f)};

    // 4. Dispatch: K workgroups, 320 threads each
    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, K, 1, 1,
                   &pushData, sizeof(pushData));
    batch.submit();

    // 5. Readback scores
    pool.download(buf_scores,
                  reinterpret_cast<float*>(result.violation_counts.data()),
                  scores_bytes);

    // 6. Compute future states on CPU (XOR is trivial)
    result.future_states.resize(K * words_per_vec);
    for (uint32_t k = 0; k < K; ++k) {
        for (uint32_t w = 0; w < words_per_vec; ++w) {
            result.future_states[k * words_per_vec + w] =
                current_state[w] ^
                snapped_deltas[k * words_per_vec + w];
        }
    }

    // 7. Find best trajectory (fewest violations)
    result.best_k = 0;
    uint32_t min_violations = result.violation_counts[0];
    for (uint32_t k = 1; k < K; ++k) {
        if (result.violation_counts[k] < min_violations) {
            min_violations = result.violation_counts[k];
            result.best_k = k;
        }
    }

    // 8. Cleanup
    pool.release(buf_state);
    pool.release(buf_deltas);
    pool.release(buf_scores);
    // buf_constraints is owned by WorldModel — do NOT release

    return result;
}

}  // namespace generation
}  // namespace grilly
