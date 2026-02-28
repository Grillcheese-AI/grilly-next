#version 450

// VSA Surrogate Loss — Forward Pass
//
// Two-pass approach:
//   Pass 0: Compute dot products for all K branches -> store in dots[]
//   Pass 1: Compute hinge + contrastive loss using winner/runner-up from results[]

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Predicted deltas: (batch, K, D) float32
layout(set = 0, binding = 0) readonly buffer Predictions {
    float preds[];
};

// True delta: bitpacked uint32[] (ceil(D/32) words per batch)
layout(set = 0, binding = 1) readonly buffer TrueDelta {
    uint true_delta[];
};

// Dot products output: (batch, K) — intermediate
layout(set = 0, binding = 2) buffer DotProducts {
    float dots[];
};

// Loss output: (batch,) scalar per batch element
layout(set = 0, binding = 3) buffer LossOutput {
    float loss[];
};

// Results: [winning_k, runner_up_k, dot_winner_bits, dot_runner_up_bits] per batch
layout(set = 0, binding = 4) buffer Results {
    uint results[];
};

layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint K;
    uint D;
    uint num_words;     // ceil(D/32)
    float gamma;        // Hinge margin
    float delta_margin; // Contrastive margin
    float lambda_c;     // Contrastive weight
    uint pass_type;     // 0 = dot products, 1 = hinge loss
};

shared float partial_sums[256];

void main() {
    uint batch_idx = gl_WorkGroupID.y;

    if (pass_type == 0u) {
        // PASS 0: Compute dot product for one (batch, k) pair
        uint k = gl_WorkGroupID.x;
        uint tid = gl_LocalInvocationID.x;

        if (k >= K) return;

        float sum = 0.0;
        uint pred_offset = (batch_idx * K + k) * D;
        uint delta_offset = batch_idx * num_words;

        for (uint i = tid; i < D; i += 256) {
            uint word_idx = i / 32;
            uint bit_idx = i % 32;
            float y_true = ((true_delta[delta_offset + word_idx] >> bit_idx) & 1u) != 0u ? 1.0 : -1.0;
            sum += preds[pred_offset + i] * y_true;
        }

        partial_sums[tid] = sum;
        barrier();

        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partial_sums[tid] += partial_sums[tid + stride];
            }
            barrier();
        }

        if (tid == 0) {
            dots[batch_idx * K + k] = partial_sums[0];
        }

    } else {
        // PASS 1: Compute hinge loss on winner (1 workgroup per batch)
        uint tid = gl_LocalInvocationID.x;

        uint win_k = results[batch_idx * 4];
        uint pred_offset = (batch_idx * K + win_k) * D;
        uint delta_offset = batch_idx * num_words;

        float hinge_sum = 0.0;
        for (uint i = tid; i < D; i += 256) {
            uint word_idx = i / 32;
            uint bit_idx = i % 32;
            float y_true = ((true_delta[delta_offset + word_idx] >> bit_idx) & 1u) != 0u ? 1.0 : -1.0;
            float z_pred = preds[pred_offset + i];
            float margin = gamma - y_true * z_pred;
            hinge_sum += max(margin, 0.0);
        }

        partial_sums[tid] = hinge_sum;
        barrier();

        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partial_sums[tid] += partial_sums[tid + stride];
            }
            barrier();
        }

        if (tid == 0) {
            float L_hinge = partial_sums[0] / float(D);

            // Contrastive margin
            float dot_win = uintBitsToFloat(results[batch_idx * 4 + 2]);
            float dot_run = uintBitsToFloat(results[batch_idx * 4 + 3]);
            float L_contrast = max(0.0, delta_margin - (dot_win - dot_run));

            loss[batch_idx] = L_hinge + lambda_c * L_contrast;
        }
    }
}
