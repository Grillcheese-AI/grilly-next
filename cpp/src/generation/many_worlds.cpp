#include "grilly/generation/many_worlds.h"
#include "grilly/cubemind/resonator.h"
#include "grilly/cubemind/types.h"

#include <algorithm>
#include <cstring>
#include <vulkan/vulkan.h>

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

    ManyWorldsResult result;
    result.K = K;
    result.words_per_vec = (D + 31) / 32;
    result.violation_counts.resize(K, 0);
    result.future_states.resize(K * result.words_per_vec, 0);

    // Always compute future states: S_{t+1} = S_t XOR Delta_k
    std::vector<uint32_t> packed_deltas = snap_to_bipolar(continuous_deltas, K, D);
    for (uint32_t k = 0; k < K; ++k) {
        for (uint32_t j = 0; j < result.words_per_vec; ++j) {
            result.future_states[k * result.words_per_vec + j] =
                current_state[j] ^ packed_deltas[k * result.words_per_vec + j];
        }
    }

    if (world_model.constraint_count() == 0) {
        // Degenerate case: no constraints, all branches equally coherent
        result.best_k = 0;
        return result;
    }

    uint32_t C = world_model.constraint_count();

    // packed_deltas already computed above (for future states)

    // 2. Prepare buffers
    size_t state_bytes = result.words_per_vec * sizeof(uint32_t);
    size_t deltas_bytes = packed_deltas.size() * sizeof(uint32_t);
    size_t scores_bytes = K * sizeof(uint32_t); // 1 uint32 per trajectory (violations)

    GrillyBuffer buf_state = pool.acquire(state_bytes);
    GrillyBuffer buf_deltas = pool.acquire(deltas_bytes);
    GrillyBuffer buf_scores = pool.acquire(scores_bytes);

    // Ensure scores are zeroed
    std::vector<uint32_t> zero_scores(K, 0);
    pool.upload(buf_scores, reinterpret_cast<const float*>(zero_scores.data()), scores_bytes);

    // 3. Upload data
    pool.upload(buf_state, reinterpret_cast<const float*>(current_state), state_bytes);
    pool.upload(buf_deltas, reinterpret_cast<const float*>(packed_deltas.data()), deltas_bytes);
    
    // We assume world_model already holds its packed constraints in a GPU buffer
    GrillyBuffer buf_constraints = world_model.constraints_gpu_buffer();

    // 4. Get Shader Pipeline
    // many-worlds-coherence.glsl: 
    //   Workgroup size: e.g., 256
    //   Desciptor set: 
    //     binding 0: current_state
    //     binding 1: deltas (array of K)
    //     binding 2: constraints (array of C)
    //     binding 3: out_violation_counts (array of K)
    
    PipelineEntry pipe = cache.getOrCreate("many-worlds-coherence", 4, sizeof(uint32_t) * 4);

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {buf_state.handle, 0, state_bytes},
        {buf_deltas.handle, 0, deltas_bytes},
        {buf_constraints.handle, 0, C * state_bytes},
        {buf_scores.handle, 0, scores_bytes}
    };

    VkDescriptorSet descSet = cache.allocDescriptorSet("many-worlds-coherence", bufInfos);

    uint32_t push[4] = {K, result.words_per_vec, C, 0};

    batch.begin();
    batch.dispatch(pipe.pipeline, pipe.layout, descSet, K, 1, 1, push, sizeof(push));
    batch.submit();

    // 5. Readback violation counts
    std::vector<uint32_t> scores(K);
    pool.download(buf_scores, reinterpret_cast<float*>(scores.data()), scores_bytes);
    result.violation_counts = scores;

    // future_states already computed above (before constraint check)

    // 6. Find best trajectory (fewest violations, ties broken by lowest k)
    result.best_k = 0;
    uint32_t best_violations = result.violation_counts[0];
    for (uint32_t k = 1; k < K; ++k) {
        if (result.violation_counts[k] < best_violations) {
            best_violations = result.violation_counts[k];
            result.best_k = k;
        }
    }

    pool.release(buf_state);
    pool.release(buf_deltas);
    pool.release(buf_scores);

    return result;
}

std::pair<std::string, float> interpret_trajectory(
    const ManyWorldsResult& mw_result,
    grilly::cubemind::ResonatorNetwork& resonator,
    const std::string& dep_role,
    uint32_t position) {

    uint32_t best_k = mw_result.best_k;
    uint32_t words_per_vec = mw_result.words_per_vec;

    grilly::cubemind::BitpackedVec bundle;
    bundle.dim = words_per_vec * 32;
    bundle.data.assign(
        mw_result.future_states.begin() + best_k * words_per_vec,
        mw_result.future_states.begin() + (best_k + 1) * words_per_vec
    );

    return resonator.query_slot(bundle, dep_role, position);
}

}  // namespace generation
}  // namespace grilly
