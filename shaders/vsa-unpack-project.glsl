#version 450

// Fused VSA unpack + linear projection.
// Unpacks a bitpacked uint32[] VSA vector to float {-1, +1},
// then computes output = unpack(vsa) @ W^T + bias.
//
// Each workgroup computes one output element across the full input dimension.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Bitpacked VSA state: ceil(vsa_dim / 32) uint32 words
layout(set = 0, binding = 0) readonly buffer VSAState {
    uint vsa_data[];
};

// Projection weights: (output_dim, vsa_dim) stored row-major
layout(set = 0, binding = 1) readonly buffer Weights {
    float W[];
};

// Bias vector: (output_dim)
layout(set = 0, binding = 2) readonly buffer Bias {
    float b[];
};

// Output: (batch, output_dim)
layout(set = 0, binding = 3) writeonly buffer Output {
    float output_data[];
};

layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint vsa_dim;       // 10240
    uint output_dim;    // 768
    uint num_words;     // ceil(vsa_dim / 32) = 320
};

shared float partial_sums[256];

void main() {
    uint batch_idx = gl_WorkGroupID.y;
    uint out_idx = gl_WorkGroupID.x;   // Which output dimension
    uint tid = gl_LocalInvocationID.x;

    if (out_idx >= output_dim) return;

    // Each thread handles a chunk of the dot product
    float sum = 0.0;
    uint row_offset = out_idx * vsa_dim;
    uint vsa_offset = batch_idx * num_words;

    for (uint i = tid; i < vsa_dim; i += 256) {
        // Unpack bit to float {-1, +1}
        uint word_idx = i / 32;
        uint bit_idx = i % 32;
        float val = ((vsa_data[vsa_offset + word_idx] >> bit_idx) & 1u) != 0u ? 1.0 : -1.0;

        sum += val * W[row_offset + i];
    }

    // Parallel reduction in shared memory
    partial_sums[tid] = sum;
    barrier();

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        barrier();
    }

    if (tid == 0) {
        output_data[batch_idx * output_dim + out_idx] = partial_sums[0] + b[out_idx];
    }
}
