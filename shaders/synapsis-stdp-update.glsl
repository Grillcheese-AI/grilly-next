#version 450

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Presynaptic trace (batch_size, in_features)
layout(set = 0, binding = 0) readonly buffer PreTrace {
    float pre_trace[];
};

// Postsynaptic trace (batch_size, out_features)
layout(set = 0, binding = 1) readonly buffer PostTrace {
    float post_trace[];
};

// Weight matrix (out_features, in_features) - read and write
layout(set = 0, binding = 2) buffer Weights {
    float W[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint in_features;
    uint out_features;
    float lr;             // Learning rate
    float weight_min;     // Minimum weight value
    float weight_max;     // Maximum weight value
    float decay;          // Weight decay multiplier (e.g., 0.999)
};

void main() {
    // Each thread handles one weight: W[post_idx][pre_idx]
    uint post_idx = gl_GlobalInvocationID.y;
    uint pre_idx = gl_GlobalInvocationID.x;

    if (post_idx >= out_features || pre_idx >= in_features) {
        return;
    }

    // Compute batch-averaged traces
    float pre_avg = 0.0;
    float post_avg = 0.0;

    for (uint b = 0; b < batch_size; b++) {
        pre_avg += pre_trace[b * in_features + pre_idx];
        post_avg += post_trace[b * out_features + post_idx];
    }

    pre_avg /= float(batch_size);
    post_avg /= float(batch_size);

    // Hebbian update: W = decay * W + lr * post ⊗ pre
    uint weight_idx = post_idx * in_features + pre_idx;
    float new_weight = decay * W[weight_idx] + lr * post_avg * pre_avg;
    new_weight = clamp(new_weight, weight_min, weight_max);

    W[weight_idx] = new_weight;
}
