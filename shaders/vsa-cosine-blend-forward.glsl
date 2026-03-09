#version 450

// Cosine Blend Forward
//
// 1. Blend: blended[i] = sum_k(weights[k] * preds[k, i])
// 2. Dot:   dot_bt = sum_i(blended[i] * bipolar(true_delta[i]))
// 3. Norm:  norm_sq = sum_i(blended[i]^2)
// 4. Loss:  1 - dot_bt / (sqrt(norm_sq) * sqrt(D))
//
// Outputs: blended buffer (saved for backward), loss scalar, dot_bt, norm_sq

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Predictions {
    float preds[];
};

layout(set = 0, binding = 1) readonly buffer TrueDelta {
    uint true_delta[];
};

layout(set = 0, binding = 2) readonly buffer Weights {
    float weights[];
};

layout(set = 0, binding = 3) writeonly buffer Blended {
    float blended[];
};

// [loss, dot_bt, norm_sq] per batch
layout(set = 0, binding = 4) buffer LossOutput {
    float loss_out[];
};

layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint K;
    uint D;
    uint num_words;
};

shared float partial_dot[256];
shared float partial_norm[256];

void main() {
    uint batch_idx = gl_WorkGroupID.y;
    uint tid = gl_GlobalInvocationID.x;

    float local_dot = 0.0;
    float local_norm = 0.0;

    for (uint i = tid; i < D; i += gl_WorkGroupSize.x * gl_NumWorkGroups.x) {
        // Blend predictions
        float b = 0.0;
        for (uint k = 0; k < K; ++k) {
            b += weights[batch_idx * K + k] * preds[(batch_idx * K + k) * D + i];
        }
        blended[batch_idx * D + i] = b;

        // Bipolar true delta
        uint word_idx = i / 32;
        uint bit_idx = i % 32;
        float y_true = ((true_delta[batch_idx * num_words + word_idx] >> bit_idx) & 1u) != 0u ? 1.0 : -1.0;

        local_dot += b * y_true;
        local_norm += b * b;
    }

    // Parallel reduction
    partial_dot[gl_LocalInvocationID.x] = local_dot;
    partial_norm[gl_LocalInvocationID.x] = local_norm;
    barrier();

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (gl_LocalInvocationID.x < stride) {
            partial_dot[gl_LocalInvocationID.x] += partial_dot[gl_LocalInvocationID.x + stride];
            partial_norm[gl_LocalInvocationID.x] += partial_norm[gl_LocalInvocationID.x + stride];
        }
        barrier();
    }

    if (gl_LocalInvocationID.x == 0) {
        float dot_bt = partial_dot[0];
        float norm_sq = partial_norm[0];
        float norm_b = sqrt(max(norm_sq, 1e-8));
        float norm_t = sqrt(float(D));  // bipolar vector norm = sqrt(D)
        float cosine_sim = dot_bt / (norm_b * norm_t);
        float loss = 1.0 - cosine_sim;

        loss_out[batch_idx * 3 + 0] = loss;
        loss_out[batch_idx * 3 + 1] = dot_bt;
        loss_out[batch_idx * 3 + 2] = norm_sq;
    }
}
