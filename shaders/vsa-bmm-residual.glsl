#version 460

// ── VSA Residual Binary MLP (4-Layer, parameterized) ─────────────────
//
// 4-layer binary MLP with residual XOR skip connections:
//   h1 = threshold(XNOR(input, W1))                       (input -> hidden)
//   h2 = threshold(XNOR(h1, W2)) XOR h1                   (hidden -> hidden, residual)
//   h3 = threshold(XNOR(h2, W3)) XOR h2                   (hidden -> hidden, residual)
//   out = threshold(XNOR(h3, W4))                          (hidden -> output)
//   predicted = XNOR(out, input)                            (temporal binding)
//
// Weight layout (contiguous in BDA buffer):
//   W1: hidden_dim x state_words   (offset 0)
//   W2: hidden_dim x hidden_words  (offset W1_size)
//   W3: hidden_dim x hidden_words  (offset W1_size + W2_size)
//   W4: state_dim x hidden_words   (offset W1_size + 2*W2_size)
//
// Dispatch: 1 workgroup of 64 threads

#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer WeightMatrix {
    uint data[];
};

layout(push_constant) uniform PushConstants {
    WeightMatrix weights_ptr;
    uint state_words;
    uint hidden_words;
    uint num_layers;   // currently always 4
    uint _pad;
} pc;

layout(set = 0, binding = 0) readonly buffer InputState  { uint current_state[];   };
layout(set = 0, binding = 1) writeonly buffer OutputState { uint predicted_target[]; };

shared uint local_input[512];
shared uint hidden_a[256];    // ping buffer for hidden state
shared uint hidden_b[256];    // pong buffer for hidden state

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint state_wpt = (pc.state_words + 63) / 64;
    uint hidden_wpt = (pc.hidden_words + 63) / 64;
    uint half_state = pc.state_words * 16;
    uint half_hidden = pc.hidden_words * 16;

    // Compute weight offsets
    uint w1_off = 0;
    uint w2_off = pc.hidden_words * 32 * pc.state_words;   // hidden_dim * state_words
    uint w3_off = w2_off + pc.hidden_words * 32 * pc.hidden_words;
    uint w4_off = w3_off + pc.hidden_words * 32 * pc.hidden_words;

    // ── Load input ────────────────────────────────────────────────────
    for (uint i = 0; i < state_wpt; i++) {
        uint idx = tid * state_wpt + i;
        if (idx < pc.state_words)
            local_input[idx] = current_state[idx];
    }
    barrier();

    // ── LAYER 1: input(state_words) -> hidden_a(hidden_words) ────────
    for (uint u = 0; u < hidden_wpt; u++) {
        uint h_idx = tid * hidden_wpt + u;
        if (h_idx >= pc.hidden_words) break;
        uint h_out = 0;
        for (uint bit = 0; bit < 32; bit++) {
            uint neuron = h_idx * 32 + bit;
            uint mc = 0;
            uint wo = w1_off + neuron * pc.state_words;
            for (uint i = 0; i < pc.state_words; i++)
                mc += bitCount(~(local_input[i] ^ pc.weights_ptr.data[wo + i]));
            if (mc > half_state) h_out |= (1u << bit);
        }
        hidden_a[h_idx] = h_out;
    }
    barrier();

    // ── LAYER 2: hidden_a -> hidden_b, then XOR with hidden_a (residual)
    for (uint u = 0; u < hidden_wpt; u++) {
        uint h_idx = tid * hidden_wpt + u;
        if (h_idx >= pc.hidden_words) break;
        uint h_out = 0;
        for (uint bit = 0; bit < 32; bit++) {
            uint neuron = h_idx * 32 + bit;
            uint mc = 0;
            uint wo = w2_off + neuron * pc.hidden_words;
            for (uint i = 0; i < pc.hidden_words; i++)
                mc += bitCount(~(hidden_a[i] ^ pc.weights_ptr.data[wo + i]));
            if (mc > half_hidden) h_out |= (1u << bit);
        }
        // Residual: XNOR with pre-layer input (bipolar multiply = XNOR)
        hidden_b[h_idx] = ~(h_out ^ hidden_a[h_idx]);
    }
    barrier();

    // ── LAYER 3: hidden_b -> hidden_a, then XOR with hidden_b (residual)
    for (uint u = 0; u < hidden_wpt; u++) {
        uint h_idx = tid * hidden_wpt + u;
        if (h_idx >= pc.hidden_words) break;
        uint h_out = 0;
        for (uint bit = 0; bit < 32; bit++) {
            uint neuron = h_idx * 32 + bit;
            uint mc = 0;
            uint wo = w3_off + neuron * pc.hidden_words;
            for (uint i = 0; i < pc.hidden_words; i++)
                mc += bitCount(~(hidden_b[i] ^ pc.weights_ptr.data[wo + i]));
            if (mc > half_hidden) h_out |= (1u << bit);
        }
        // Residual: XNOR with pre-layer input (bipolar multiply = XNOR)
        hidden_a[h_idx] = ~(h_out ^ hidden_b[h_idx]);
    }
    barrier();

    // ── LAYER 4: hidden_a(hidden_words) -> output(state_words) ───────
    for (uint u = 0; u < state_wpt; u++) {
        uint out_idx = tid * state_wpt + u;
        if (out_idx >= pc.state_words) break;
        uint o_val = 0;
        for (uint bit = 0; bit < 32; bit++) {
            uint neuron = out_idx * 32 + bit;
            uint mc = 0;
            uint wo = w4_off + neuron * pc.hidden_words;
            for (uint i = 0; i < pc.hidden_words; i++)
                mc += bitCount(~(hidden_a[i] ^ pc.weights_ptr.data[wo + i]));
            if (mc > half_hidden) o_val |= (1u << bit);
        }
        // XNOR bind with original input
        predicted_target[out_idx] = ~(local_input[out_idx] ^ o_val);
    }
}
