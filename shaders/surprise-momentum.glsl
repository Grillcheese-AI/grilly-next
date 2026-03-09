#version 450

// Surprise-Momentum Optimizer Shader
// Replaces AdamW with hippocampal surprise + biological momentum.
//
// Per-element:
//   1. Instant surprise: prediction error from hippocampal recall mismatch
//   2. Biological momentum: S_bar = α * instant + (1-α) * S_bar_prev
//   3. Adaptive LR: η_eff = η_base * (1 + S_bar)  (high surprise → faster learning)
//   4. Effective update: η_eff * (grad + λ * recalled_grad_weighted)
//
// Uses: push constants for hyperparams, 6 buffer bindings.
// Compatible with existing adamw-update.glsl buffer layout conventions.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Current gradients (flattened param)
layout(set = 0, binding = 0) buffer Gradients {
    float grad[];
};

// CA3-recalled gradient (weighted average from hippocampal retrieval)
layout(set = 0, binding = 1) readonly buffer RecalledGrad {
    float recalled_grad[];
};

// Accumulated surprise S_bar (persistent across steps, like AdamW moment)
layout(set = 0, binding = 2) buffer SurpriseAccum {
    float s_bar[];
};

// Weight parameters (updated in-place)
layout(set = 0, binding = 3) buffer Weights {
    float W[];
};

// Per-element update output (for telemetry / consolidation priority)
layout(set = 0, binding = 4) buffer UpdateOutput {
    float delta[];
};

layout(push_constant) uniform PushConsts {
    uint   total_elems;        // number of elements
    float  eta_base;           // base learning rate
    float  alpha_momentum;     // surprise EMA decay (0.9 typical)
    float  lambda_recall;      // hippocampal recall blend weight (0.3 typical)
    float  surprise_floor;     // minimum surprise to avoid zero LR (0.01)
    float  weight_decay;       // L2 regularisation
    float  clip_max;           // gradient clipping threshold
    uint   clear_grad;         // whether to zero gradients after use
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= total_elems) {
        return;
    }

    float g = grad[idx];
    float r = recalled_grad[idx];

    // NaN/Inf guard: if gradient is corrupted, skip this element entirely.
    // NaN propagation from a single bad forward/backward pass can corrupt
    // ALL weights via clamp(NaN) undefined behavior on AMD GPUs.
    if (isinf(g) || isnan(g) || isinf(r) || isnan(r)) {
        delta[idx] = 0.0;
        if (clear_grad != 0u) {
            grad[idx] = 0.0;
        }
        return;
    }

    // 1. Effective gradient: blend current with hippocampal recall
    float g_eff = g + lambda_recall * r;

    // 2. Instant surprise: absolute prediction error between current and recalled
    float instant_pe = abs(g - r);

    // 3. Biological momentum: EMA of surprise
    float s_prev = s_bar[idx];
    // Guard against NaN in accumulated surprise (can happen if previous step
    // wrote NaN before we added this guard)
    if (isnan(s_prev) || isinf(s_prev)) s_prev = 0.0;
    float s_new = alpha_momentum * instant_pe + (1.0 - alpha_momentum) * s_prev;
    // Cap surprise to prevent runaway LR amplification
    s_new = min(s_new, 5.0);
    s_bar[idx] = s_new;

    // 4. Adaptive learning rate: surprise amplifies LR
    //    Floor ensures we never have zero effective LR
    float adaptive_lr = eta_base * (1.0 + max(s_new, surprise_floor));

    // 5. Compute weight update
    float d = adaptive_lr * g_eff;

    // L2 weight decay (decoupled, like AdamW)
    float w = W[idx];
    d += eta_base * weight_decay * w;

    // Clip to prevent explosion
    d = clamp(d, -clip_max, clip_max);

    // 6. Apply update (with weight clipping to bound predictions)
    float w_new = w - d;
    // Final NaN guard on the weight itself
    if (isnan(w_new) || isinf(w_new)) w_new = 0.0;
    W[idx] = clamp(w_new, -10.0, 10.0);
    delta[idx] = d;

    // Optionally clear gradient
    if (clear_grad != 0u) {
        grad[idx] = 0.0;
    }
}
