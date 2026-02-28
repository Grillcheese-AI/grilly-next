#version 450

// VSA Surrogate Loss — Backward Pass (Sparse Gradient Routing)
//
// Only writes gradients to winning_k and runner_up_k branches.
// Other K-2 branches receive exactly 0.0 gradient.
//
// Winner gradient:     grad[win_k, i] = (-y_true / D) * scale   where hinge active
// Runner-up gradient:  grad[run_k, i] = (+y_true / D) * lambda  where contrastive active
//
// Cost: O(2D) instead of O(K*D)

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Predicted deltas: (batch, K, D) — read for hinge margin check
layout(set = 0, binding = 0) readonly buffer Predictions {
    float preds[];
};

// True delta: bitpacked uint32[]
layout(set = 0, binding = 1) readonly buffer TrueDelta {
    uint true_delta[];
};

// Gradient w.r.t. predictions: (batch, K, D) — output, zeroed before dispatch
layout(set = 0, binding = 2) buffer GradPredictions {
    float grad_preds[];
};

// Results from forward: [winning_k, runner_up_k, dot_win, dot_run] per batch
layout(set = 0, binding = 3) readonly buffer Results {
    uint results[];
};

layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint K;
    uint D;
    uint num_words;
    float gamma;
    float delta_margin;
    float lambda_c;
    float grad_scale;    // Global gradient scale (e.g., 1/batch_size)
};

void main() {
    uint batch_idx = gl_WorkGroupID.y;
    uint tid = gl_GlobalInvocationID.x;

    if (tid >= D) return;

    uint win_k = results[batch_idx * 4];
    uint run_k = results[batch_idx * 4 + 1];

    uint word_idx = tid / 32;
    uint bit_idx = tid % 32;
    uint delta_offset = batch_idx * num_words;
    float y_true = ((true_delta[delta_offset + word_idx] >> bit_idx) & 1u) != 0u ? 1.0 : -1.0;

    float inv_D = 1.0 / float(D);

    // Winner: hinge gradient
    {
        uint pred_idx = (batch_idx * K + win_k) * D + tid;
        float z_pred = preds[pred_idx];
        float margin = gamma - y_true * z_pred;

        if (margin > 0.0) {
            grad_preds[pred_idx] += (-y_true * inv_D) * grad_scale;
        }
    }

    // Runner-up: contrastive gradient (push away from true delta)
    if (win_k != run_k) {
        float dot_win = uintBitsToFloat(results[batch_idx * 4 + 2]);
        float dot_run = uintBitsToFloat(results[batch_idx * 4 + 3]);
        float contrastive_margin = delta_margin - (dot_win - dot_run);

        if (contrastive_margin > 0.0) {
            uint pred_idx = (batch_idx * K + run_k) * D + tid;
            grad_preds[pred_idx] += (y_true * inv_D) * lambda_c * grad_scale;
        }
    }
}
