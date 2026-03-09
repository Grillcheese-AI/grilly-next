#version 450
#extension GL_EXT_shader_atomic_float : require

// LoRA Expand Backward
//
// Given grad_preds [batch, K, D], compute:
//   grad_coeffs [batch, K, rank] = B^T @ grad_preds  (per branch)
//   grad_B [D, rank] += grad_preds^T @ coeffs         (accumulated)

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Basis {
    float B[];
};

layout(set = 0, binding = 1) readonly buffer Coefficients {
    float coeffs[];
};

layout(set = 0, binding = 2) readonly buffer GradPredictions {
    float grad_preds[];
};

layout(set = 0, binding = 3) writeonly buffer GradCoefficients {
    float grad_coeffs[];
};

layout(set = 0, binding = 4) buffer GradBasis {
    float grad_B[];
};

layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint K;
    uint D;
    uint rank;
    float grad_scale;
};

void main() {
    uint batch_idx = gl_WorkGroupID.y;
    uint k = gl_WorkGroupID.z;
    uint tid = gl_GlobalInvocationID.x;

    if (k >= K) return;

    // Phase 1: Compute grad_coeffs[batch, k, r] for one (batch, k) pair
    // grad_c[r] = sum_i(B[i, r] * grad_pred[k, i])
    if (tid < rank) {
        float sum = 0.0;
        uint pred_base = (batch_idx * K + k) * D;
        for (uint i = 0; i < D; ++i) {
            sum += B[i * rank + tid] * grad_preds[pred_base + i];
        }
        grad_coeffs[batch_idx * K * rank + k * rank + tid] = sum * grad_scale;
    }

    // Phase 2: Accumulate grad_B[i, r] for this (batch, k)
    // grad_B[tid, r] += grad_pred[k, tid] * c[k, r]
    if (tid < D) {
        uint coeff_base = batch_idx * K * rank + k * rank;
        uint pred_idx = (batch_idx * K + k) * D + tid;
        float gp = grad_preds[pred_idx] * grad_scale;
        for (uint r = 0; r < rank; ++r) {
            atomicAdd(grad_B[tid * rank + r], gp * coeffs[coeff_base + r]);
        }
    }
}
