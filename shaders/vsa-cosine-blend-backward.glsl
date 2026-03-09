#version 450

// Cosine Blend Backward
//
// Given saved blended[D], dot_bt, norm_sq from forward:
//   grad_blended[i] = -(y_true[i] / (norm_b * sqrt_D) - cosine * blended[i] / norm_sq)
//   grad_pred[k, i] = grad_blended[i] * weights[k]

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

layout(set = 0, binding = 3) readonly buffer Blended {
    float blended[];
};

layout(set = 0, binding = 4) readonly buffer LossInfo {
    float loss_info[];  // [loss, dot_bt, norm_sq] per batch
};

layout(set = 0, binding = 5) buffer GradPredictions {
    float grad_preds[];
};

layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint K;
    uint D;
    uint num_words;
    float grad_scale;
};

void main() {
    uint batch_idx = gl_WorkGroupID.y;
    uint tid = gl_GlobalInvocationID.x;

    if (tid >= D) return;

    float dot_bt = loss_info[batch_idx * 3 + 1];
    float norm_sq = loss_info[batch_idx * 3 + 2];
    float norm_b = sqrt(max(norm_sq, 1e-8));
    float sqrt_D = sqrt(float(D));
    float cosine_sim = dot_bt / (norm_b * sqrt_D);

    // Bipolar true delta
    uint word_idx = tid / 32;
    uint bit_idx = tid % 32;
    float y_true = ((true_delta[batch_idx * num_words + word_idx] >> bit_idx) & 1u) != 0u ? 1.0 : -1.0;

    float b_i = blended[batch_idx * D + tid];

    // d(cosine)/d(blended[i]) = y_true / (norm_b * sqrt_D) - cosine * b_i / norm_sq
    // d(loss)/d(blended[i]) = -d(cosine)/d(blended[i])
    float grad_b = -(y_true / (norm_b * sqrt_D) - cosine_sim * b_i / max(norm_sq, 1e-8));

    // Route gradient to each branch weighted by router weights.
    // Clamp per-element to prevent outlier explosion through backward chain.
    for (uint k = 0; k < K; ++k) {
        float w_k = weights[batch_idx * K + k];
        uint pred_idx = (batch_idx * K + k) * D + tid;
        float g = grad_b * w_k * grad_scale;
        grad_preds[pred_idx] += clamp(g, -1.0, 1.0);
    }
}
