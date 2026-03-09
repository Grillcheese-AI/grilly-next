#version 460

// ── VSA Codebook Decode via BDA + Atomic ArgMin ─────────────────────
//
// Finds the closest codebook entry to a predicted vector using
// Hamming distance. All 152K+ vocabulary entries are searched in
// parallel — each thread handles one entry.
//
// Instead of writing all distances to memory and sorting on CPU,
// we pack (distance, token_id) into a single uint64 and use
// atomicMin to find the winner in a single GPU sweep. O(1) dispatch.
//
// Both codebook and predicted vector are accessed via BDA pointers
// (push constants), eliminating descriptor set overhead entirely.
//
// Dispatch: workgroups_x = ceil(vocab_size / 256)

#extension GL_EXT_buffer_reference : require
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_EXT_shader_atomic_int64 : require

// BDA pointers for raw VRAM access
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer Codebook {
    uint data[];
};

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer PredVector {
    uint data[];
};

layout(push_constant) uniform PushConstants {
    Codebook codebook_ptr;       // BDA: 152K × 320 uint32 codebook
    PredVector pred_vector_ptr;  // BDA: 320 uint32 predicted vector
    uint vocab_size;             // Number of codebook entries (e.g., 152000)
    uint uints_per_vec;          // Words per vector (e.g., 320 at d=10240)
} pc;

// Output: single uint64 holding packed (distance << 32 | token_id)
// Initialized to 0xFFFFFFFFFFFFFFFF before dispatch.
layout(set = 0, binding = 0) buffer Output {
    uint64_t best_match;
};

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint token_id = gl_GlobalInvocationID.x;
    if (token_id >= pc.vocab_size) return;

    uint hamming_distance = 0;
    uint codebook_offset = token_id * pc.uints_per_vec;

    // Compute Hamming distance: XOR finds differing bits, bitCount sums them
    for (uint i = 0; i < pc.uints_per_vec; i++) {
        uint pred_chunk = pc.pred_vector_ptr.data[i];
        uint dict_chunk = pc.codebook_ptr.data[codebook_offset + i];
        hamming_distance += bitCount(pred_chunk ^ dict_chunk);
    }

    // Pack distance (top 32 bits) and token_id (bottom 32 bits)
    // atomicMin compares the packed uint64, so the entry with the
    // smallest Hamming distance wins automatically.
    uint64_t packed_result = (uint64_t(hamming_distance) << 32) | uint64_t(token_id);
    atomicMin(best_match, packed_result);
}
