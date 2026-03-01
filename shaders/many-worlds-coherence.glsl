#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable

// Many-Worlds Coherence Check
//
// Each workgroup simulates one parallel future (k).
// 320 threads process the 10240-bit state (32 bits per thread).
//
// Flow:
//   1. APPLY: S_{t+1}^{(k)} = S_t XOR Delta_k
//   2. VERIFY: For each WorldModel constraint, compute Hamming distance
//      to the hypothetical future. If distance < threshold, it's a violation
//      (the future mathematically contradicts a known rule).
//   3. WRITEBACK: Thread 0 commits the violation count per trajectory.

layout(local_size_x = 320) in;

// --- Buffers ---
// The current geometric state S_t (320 words)
layout(set = 0, binding = 0) readonly buffer CurrentState { uint S_t[]; };

// The K generated interventions from the Hypernetwork (K * 320 words)
layout(set = 0, binding = 1) readonly buffer CandidateDeltas { uint Deltas[]; };

// The WorldModel negative constraints (num_constraints * 320 words)
layout(set = 0, binding = 2) readonly buffer WorldModelConstraints { uint Constraints[]; };

// Output buffer for the violation scores of each K trajectory
layout(set = 0, binding = 3) writeonly buffer CoherenceScores { uint Scores[]; };

// --- Push Constants ---
layout(push_constant) uniform PushConsts {
    uint words_per_vec;   // 320
    uint num_constraints; // How many rules to check against
    uint distance_thresh; // Hamming distance threshold for a violation
};

// Shared memory for cross-subgroup reduction.
// Max 10 subgroups (320 threads / 32 subgroup size on NVIDIA).
// AMD with subgroup size 64 uses 5 slots, Intel with 16 uses 20 —
// but 320/16 = 20, so we allocate for worst case.
shared uint shared_dist[20];

void main() {
    uint k = gl_WorkGroupID.x;              // The counterfactual universe (0 to K-1)
    uint word_idx = gl_LocalInvocationID.x; // The 32-bit chunk (0 to 319)

    // 1. APPLY THE ACTION: Calculate the hypothetical future state
    // S_{t+1}^{(k)} = S_t XOR Delta_k
    uint current_word = S_t[word_idx];
    uint delta_word = Deltas[k * words_per_vec + word_idx];
    uint S_t_plus_1 = current_word ^ delta_word;

    // 2. VERIFY THE FUTURE: ARV Constraints Check
    uint total_violations = 0;

    for (uint c = 0; c < num_constraints; c++) {
        uint constraint_word = Constraints[c * words_per_vec + word_idx];

        // Count differing bits between our hypothetical future and the constraint
        uint local_dist = bitCount(S_t_plus_1 ^ constraint_word);

        // Stage 1: Fast subgroup-level reduction (hardware warp instruction)
        uint sg_sum = subgroupAdd(local_dist);

        // Stage 2: Cross-subgroup reduction via shared memory
        if (subgroupElect()) {
            shared_dist[gl_SubgroupID] = sg_sum;
        }
        barrier();

        // Stage 3: Thread 0 accumulates the full workgroup sum
        if (gl_LocalInvocationID.x == 0) {
            uint full_dist = 0;
            for (uint s = 0; s < gl_NumSubgroups; s++) {
                full_dist += shared_dist[s];
            }

            // If the distance is smaller than the threshold, this future
            // mathematically contradicts a known rule (Hallucination)
            if (full_dist < distance_thresh) {
                total_violations++;
            }
        }

        // Sync before next constraint iteration reuses shared memory
        barrier();
    }

    // 3. WRITEBACK: The first thread in the workgroup commits the final score
    if (gl_LocalInvocationID.x == 0) {
        Scores[k] = total_violations;
    }
}
