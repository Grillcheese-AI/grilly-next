#version 460

// ── BDA Hamming Distance Search ─────────────────────────────────────
//
// Same as hamming-search.glsl but uses Buffer Device Address (BDA)
// instead of descriptor set bindings. The GPU reads data directly
// from 64-bit virtual pointers passed via push constants.
//
// Advantages over descriptor-bound version:
//   - Zero descriptor set allocation overhead
//   - Instant buffer swapping (just change the pointer in push constants)
//   - Critical for growDimensionality() — no GPU queue stall on resize
//
// Dispatch: workgroups_x = ceil(N / 256)

#extension GL_EXT_buffer_reference : require
#extension GL_KHR_shader_subgroup_arithmetic : require

// Define the dynamic pointer structure for BDA access
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer BitpackedArrayIn {
    uint data[];
};

layout(buffer_reference, std430, buffer_reference_align = 4) writeonly buffer BitpackedArrayOut {
    uint data[];
};

// Match the C++ HammingSearchBDAParams struct exactly
layout(push_constant) uniform PushConstants {
    BitpackedArrayIn queryPtr;
    BitpackedArrayIn cachePtr;
    BitpackedArrayOut distPtr;
    uint numEntries;
    uint wordsPerVec;
    uint numQueries;
    uint queryOffset;
} pc;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint entry = gl_GlobalInvocationID.x;
    if (entry >= pc.numEntries) return;

    uint offset = entry * pc.wordsPerVec;
    uint qb     = pc.queryOffset * pc.wordsPerVec;
    uint dist   = 0;

    // Sequential loop: each thread processes its entire entry
    for (uint w = 0; w < pc.wordsPerVec; w++) {
        dist += bitCount(pc.queryPtr.data[qb + w] ^ pc.cachePtr.data[offset + w]);
    }

    pc.distPtr.data[entry] = dist;
}
