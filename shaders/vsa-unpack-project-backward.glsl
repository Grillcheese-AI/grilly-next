#version 450

// Backward pass for VSA UnpackProject: output = unpack(vsa) @ W^T + b
//
// Unlike fnn-linear-backward, the input is bitpacked uint32 VSA data,
// so we unpack on-the-fly to {-1, +1} instead of reading float input.
// No grad_input — VSA state is discrete/not differentiable.
//
// Pass 0: grad_W = grad_output^T @ unpack(vsa)
// Pass 1: grad_b = sum(grad_output, dim=0)

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Gradient w.r.t. output (batch, output_dim)
layout(set = 0, binding = 0) readonly buffer GradOutput {
    float grad_output[];
};

// Bitpacked VSA state: ceil(vsa_dim / 32) uint32 words per batch element
layout(set = 0, binding = 1) readonly buffer VSAState {
    uint vsa_data[];
};

// Gradient w.r.t. weights (output_dim, vsa_dim)
layout(set = 0, binding = 2) buffer GradWeights {
    float grad_W[];
};

// Gradient w.r.t. bias (output_dim)
layout(set = 0, binding = 3) buffer GradBias {
    float grad_b[];
};

layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint vsa_dim;       // e.g. 10240
    uint output_dim;    // e.g. 768
    uint num_words;     // ceil(vsa_dim / 32)
    uint pass_type;     // 0 = grad_W, 1 = grad_b
};

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;

    if (pass_type == 0) {
        // grad_W[out_idx, vsa_idx] = sum_b(grad_output[b, out_idx] * unpack(vsa[b], vsa_idx))
        if (row >= output_dim || col >= vsa_dim) return;

        uint word_idx = col / 32;
        uint bit_idx = col % 32;

        float sum = 0.0;
        for (uint b = 0; b < batch_size; b++) {
            float go = grad_output[b * output_dim + row];
            float val = ((vsa_data[b * num_words + word_idx] >> bit_idx) & 1u) != 0u
                        ? 1.0 : -1.0;
            sum += go * val;
        }

        grad_W[row * vsa_dim + col] = sum;

    } else if (pass_type == 1) {
        // grad_b[out_idx] = sum_b(grad_output[b, out_idx])
        uint out_idx = gl_GlobalInvocationID.x;
        if (out_idx >= output_dim) return;

        float sum = 0.0;
        for (uint b = 0; b < batch_size; b++) {
            sum += grad_output[b * output_dim + out_idx];
        }

        grad_b[out_idx] = sum;
    }
}
