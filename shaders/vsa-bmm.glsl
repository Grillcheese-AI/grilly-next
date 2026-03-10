#version 460

// ── VSA Binary Matrix Multiplication (2-Layer MLP, parameterized) ────
//
// True binary neural network forward pass using XNOR + POPCNT.
// Trained with Quantization-Aware Training (STE binarization).
//
// Dimensions are driven by push constants (state_words, hidden_words)
// so the same shader works for any power-of-2 VSA dimension.
//
// Layer 1: Input (state_words*32 bits) -> Hidden (hidden_words*32 bits)
//   Each neuron: bitCount(XNOR(input, w1)) across state_words input words.
//   Threshold at 50% (state_words * 16 bits).
//
// Layer 2: Hidden (hidden_words*32 bits) -> Output (state_words*32 bits)
//   Each neuron: bitCount(XNOR(hidden, w2)) across hidden_words words.
//   Threshold at 50% (hidden_words * 16 bits).
//
// Final step: XNOR-bind output with original input (VSA temporal binding).
//
// Dispatch: 1 workgroup of 64 threads (single dispatch)

#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer WeightMatrix {
    uint data[];
};

layout(push_constant) uniform PushConstants {
    WeightMatrix w1_ptr;   // BDA: Layer 1 weights (input -> hidden)
    WeightMatrix w2_ptr;   // BDA: Layer 2 weights (hidden -> output)
    uint state_words;      // Input/output dim in uint32 words (dim / 32)
    uint hidden_words;     // Hidden dim in uint32 words (hidden_dim / 32)
} pc;

layout(set = 0, binding = 0) readonly buffer InputState  { uint current_state[];   };
layout(set = 0, binding = 1) writeonly buffer OutputState { uint predicted_target[]; };

// Shared memory sized for max expected dims
shared uint hidden_state[256];   // max hidden_words = 256 (hidden_dim up to 8192)
shared uint local_input[512];    // max state_words  = 512 (state_dim  up to 16384)

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint tid = gl_LocalInvocationID.x;

    // Compute words-per-thread dynamically (ceil division by 64 threads)
    uint state_wpt  = (pc.state_words  + 63) / 64;
    uint hidden_wpt = (pc.hidden_words + 63) / 64;

    // Thresholds at 50% of total bits
    uint half_state_dim  = pc.state_words  * 16;
    uint half_hidden_dim = pc.hidden_words * 16;

    // 1. Load input state into shared memory
    //    64 threads × state_wpt words each
    for (uint i = 0; i < state_wpt; i++) {
        uint idx = tid * state_wpt + i;
        if (idx < pc.state_words) {
            local_input[idx] = current_state[idx];
        }
    }
    barrier();

    // ── LAYER 1: Input -> Hidden ─────────────────────────────────────
    // hidden_words uints / 64 threads = hidden_wpt uints per thread
    uint base_neuron = tid * (hidden_wpt * 32);

    for (uint u = 0; u < hidden_wpt; u++) {
        uint h_idx = tid * hidden_wpt + u;
        if (h_idx >= pc.hidden_words) break;

        uint h_out = 0;
        for (uint bit_idx = 0; bit_idx < 32; bit_idx++) {
            uint neuron = base_neuron + (u * 32) + bit_idx;
            uint match_count = 0;
            uint w_offset = neuron * pc.state_words;

            // Binary dot product: XNOR + POPCNT across state_words input words
            for (uint i = 0; i < pc.state_words; i++) {
                match_count += bitCount(~(local_input[i] ^ pc.w1_ptr.data[w_offset + i]));
            }

            // Binary activation: threshold at 50%
            if (match_count > half_state_dim) h_out |= (1u << bit_idx);
        }
        hidden_state[h_idx] = h_out;
    }
    barrier();

    // ── LAYER 2: Hidden -> Output ────────────────────────────────────
    // state_words uints / 64 threads = state_wpt uints per thread
    for (uint u = 0; u < state_wpt; u++) {
        uint out_uint_idx = tid * state_wpt + u;
        if (out_uint_idx >= pc.state_words) break;

        uint out_val = 0;
        uint base_neuron2 = out_uint_idx * 32;

        for (uint bit_idx = 0; bit_idx < 32; bit_idx++) {
            uint neuron = base_neuron2 + bit_idx;
            uint match_count = 0;
            uint w_offset = neuron * pc.hidden_words;

            // Binary dot product against hidden state
            for (uint i = 0; i < pc.hidden_words; i++) {
                match_count += bitCount(~(hidden_state[i] ^ pc.w2_ptr.data[w_offset + i]));
            }

            // Binary activation: threshold at 50%
            if (match_count > half_hidden_dim) out_val |= (1u << bit_idx);
        }

        // VSA temporal binding: XNOR predicted transformation with input
        predicted_target[out_uint_idx] = ~(local_input[out_uint_idx] ^ out_val);
    }
}
