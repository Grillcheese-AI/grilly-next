#pragma once

#include <cstdint>
#include <vector>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/cognitive/world_model.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace cubemind {
    class ResonatorNetwork;
}
namespace generation {

/// Result of evaluating K counterfactual futures against WorldModel constraints.
struct ManyWorldsResult {
    /// Violation count per trajectory. 0 = fully coherent, >0 = contradictions.
    std::vector<uint32_t> violation_counts;

    /// The K hypothetical future states S_{t+1}^{(k)} = S_t XOR Delta_k.
    /// Stored contiguously: [K * words_per_vec] uint32 words.
    std::vector<uint32_t> future_states;

    /// Index of best trajectory (fewest violations, ties broken by lowest k).
    uint32_t best_k;

    uint32_t K;
    uint32_t words_per_vec;
};

/// Snap continuous float32 predictions to bitpacked bipolar {-1, +1} vectors.
///
/// Signum threshold at 0.0: positive → bit 1 (+1), non-positive → bit 0 (-1).
/// The surrogate loss (with hinge margin gamma) teaches the network to push
/// predictions far from the decision boundary, making this snap robust.
///
/// @param continuous_data  K * D float32 values (row-major: [K][D])
/// @param K                Number of candidate trajectories
/// @param D                VSA dimension (e.g., 10240)
/// @return                 K * words_per_vec uint32 bitpacked words
std::vector<uint32_t> snap_to_bipolar(const float* continuous_data,
                                       uint32_t K, uint32_t D);

/// Evaluate K counterfactual futures against WorldModel constraints in a
/// single GPU dispatch.
///
/// Pipeline:
///   1. CPU: Snap continuous deltas to bitpacked bipolar vectors
///   2. GPU: For each k, compute S_{t+1}^{(k)} = S_t XOR Delta_k,
///           then check Hamming distance to all WorldModel constraints
///   3. CPU: Readback violation counts and compute future states
///
/// @param batch            Vulkan command batch (from GrillyCoreContext)
/// @param pool             Buffer pool (from GrillyCoreContext)
/// @param cache            Pipeline cache (from GrillyCoreContext)
/// @param current_state    Bitpacked current state S_t [words_per_vec uint32s]
/// @param continuous_deltas K * D float32 continuous predictions from hypernetwork
/// @param K                Number of candidate trajectories
/// @param D                VSA dimension (must be multiple of 32)
/// @param world_model      WorldModel containing negative constraints
/// @return                 ManyWorldsResult with violation counts and future states
ManyWorldsResult evaluate_many_worlds(
    CommandBatch& batch,
    BufferPool& pool,
    PipelineCache& cache,
    const uint32_t* current_state,
    const float* continuous_deltas,
    uint32_t K,
    uint32_t D,
    cognitive::WorldModel& world_model);

/// Interpret the best trajectory from many-worlds via resonator unbinding.
/// Returns (word, similarity) for a specific (role, position) slot.
std::pair<std::string, float> interpret_trajectory(
    const ManyWorldsResult& mw_result,
    grilly::cubemind::ResonatorNetwork& resonator,
    const std::string& dep_role,
    uint32_t position);

}  // namespace generation
}  // namespace grilly
