#version 460

// ── VSA Binary Neural Network Inference ─────────────────────────────
//
// Executes a binarized MLP forward pass using XNOR + POPCNT.
//
// Standard dot product: y = Σ(w_i * x_i)  [float multiply-add]
// Binary dot product:   y = bitCount(~(w ^ x))  [XNOR + POPCNT]
//
// The binarized weights are loaded via BDA (Buffer Device Address):
// the 64-bit GPU pointer is passed directly through push constants,
// bypassing descriptor set allocation entirely.
//
// Input/output states use standard descriptor bindings (2 buffers),
// while the large weight matrix uses BDA for zero-overhead access.
//
// Dispatch: workgroups_x = ceil(state_dim / 32 / 256)

#extension GL_EXT_buffer_reference : require

// BDA pointer to the packed binary weight matrix
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer PackedWeights {
    uint data[];
};

layout(push_constant) uniform PushConstants {
    PackedWeights projection_matrix;  // 64-bit BDA pointer to binary weights
    uint state_dim;                   // Bipolar dimension (e.g., 10240)
} pc;

// Standard descriptor-bound input/output state buffers
layout(set = 0, binding = 0) readonly buffer InputState  { uint current_state[];   };
layout(set = 0, binding = 1) writeonly buffer OutputState { uint predicted_state[]; };

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint index = gl_GlobalInvocationID.x;
    uint words = pc.state_dim / 32;
    if (index >= words) return;

    uint input_chunk  = current_state[index];
    uint weight_chunk = pc.projection_matrix.data[index];

    // XNOR: bits that match produce 1, bits that differ produce 0.
    // This is the binary equivalent of a dot product between
    // bipolar {-1, +1} vectors where matching signs yield +1.
    //
    // In a full MLP layer, you would accumulate bitCount(xnor_result)
    // across all words in a row and threshold the sum. For this
    // single-layer test, we output the raw XNOR result as the
    // predicted next state — verifying that BDA pointer dereference
    // and the XNOR logic work correctly end-to-end.
    uint xnor_result = ~(input_chunk ^ weight_chunk);

    predicted_state[index] = xnor_result;
}
