# Residual Binary MLP + Permute-Bind Context Encoder

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the shallow 2-layer MLP and roll+bundle context with a 4-layer residual binary MLP and streaming permute-bind context accumulator, improving prediction quality while staying fully in bitpacked VSA space.

**Architecture:** Permute-bind chain for context encoding (unlimited window, O(1) per token, zero extra parameters) + 4-layer residual binary MLP with XOR skip connections for the prediction network.

**Tech Stack:** GLSL compute shaders, C++ Vulkan backend (grilly_core), pybind11, numpy, numba

---

## Context Encoder: Permute-Bind Chain

Replaces roll+majority-vote bundler. Maintains a single state vector updated per token:

```
S_0 = tok_0
S_t = rotate(S_{t-1}, 1) XOR tok_t    for t > 0
```

`rotate` is a circular bit shift by 1 position across the bitpacked uint32 array. Non-commutative (order-dependent), encodes unlimited history with natural exponential recency bias. At inference: context state = 64 uint32 words (256 bytes) carried between tokens.

For training, the pair builder feeds tokens sequentially into the chain, snapshots state at each position to produce (context_state, transform) pairs.

## MLP Architecture: 4-Layer Residual Binary MLP

```
x         = context_state               (2048-bit input)
h1        = sign(W1 @ x)                (1024-bit hidden)
h2        = sign(W2 @ h1) XOR h1        (1024-bit, residual skip)
h3        = sign(W3 @ h2) XOR h2        (1024-bit, residual skip)
transform = sign(W4 @ h3)               (2048-bit output)
predicted = x XOR transform             (unbind)
```

- Layers 1 and 4 change dimension (2048 <-> 1024), no skip connection
- Layers 2 and 3 have residual XOR connections (same dimension, free in bitpacked space)
- Training uses STE sign activation with float weights

## Weight Layout

```
W1: 1024 neurons x 64 words  =  65,536 uint32  (256 KB)
W2: 1024 neurons x 32 words  =  32,768 uint32  (128 KB)
W3: 1024 neurons x 32 words  =  32,768 uint32  (128 KB)
W4: 2048 neurons x 32 words  =  65,536 uint32  (256 KB)
Total: 196,608 uint32 = 768 KB
```

## Shader: vsa-bmm-residual.glsl

4-layer binary MLP with residual XOR skip connections:

```glsl
layout(push_constant) uniform PushConstants {
    WeightMatrix weights_ptr;   // BDA: all weights contiguous
    uint state_words;           // 64 (dim=2048)
    uint hidden_words;          // 32 (hidden=1024)
    uint num_layers;            // 4
    uint _pad;
} pc;
```

Execution:
1. Load input to shared memory (state_words words)
2. Layer 1: XNOR+POPCNT input(state_words) -> hidden(hidden_words), threshold
3. Layer 2: XNOR+POPCNT hidden -> hidden, threshold, then XOR with pre-layer-2 hidden (residual)
4. Layer 3: XNOR+POPCNT hidden -> hidden, threshold, then XOR with pre-layer-3 hidden (residual)
5. Layer 4: XNOR+POPCNT hidden(hidden_words) -> output(state_words), threshold
6. Final XNOR-bind output with original input

Weight offsets computed from push constants:
- W1 starts at: 0
- W2 starts at: hidden_dim * state_words
- W3 starts at: W2 + hidden_dim * hidden_words
- W4 starts at: W3 + hidden_dim * hidden_words

## C++ Changes

- New `VSABMMResidualParams` struct in vsa_inference.h
- New dispatch path in `VSABaremetalEngine::step()` selecting `vsa-bmm-residual` when available
- Falls back to `vsa-bmm` (2-layer) if residual shader not found
- Weight file format: same binary layout, just bigger (768 KB)

## Training Changes (poc_simhash_vsa.py)

1. Replace `_build_all_pairs` with permute-bind chain:
   - For each trajectory, walk tokens sequentially
   - State update: `state = np.roll(state, 1); state = state * tok` (float bipolar)
   - Snapshot (state, transform) at each position

2. New `ResidualBinaryMLP` class:
   - 4 weight matrices with STE sign activation
   - Forward: h1 = sign(W1@x), h2 = sign(W2@h1) * h1, h3 = sign(W3@h2) * h2, out = sign(W4@h3)
   - Backward: manual STE through all 4 layers with residual gradient paths
   - Residual in float space: element-wise multiply (bipolar multiply = XOR in bit space)

3. Per-layer gradient clipping for stability with deeper network

## Inference Demo Changes (vsa_inference_demo.py)

- Replace `context_accumulate` with permute-bind chain:
  ```python
  state = unpack_bipolar(codebook_packed[token_ids[0]])
  for tok_id in token_ids[1:]:
      state = np.roll(state, 1)
      state = state * unpack_bipolar(codebook_packed[tok_id])
  ```
- Carry state between generation steps (no re-encoding from window)

## Success Criteria

- Hamming similarity > 0.90 (up from 0.857 with 2-layer + roll+bundle)
- Inference speed < 1ms per token on RX 6750 XT
- Output shows more coherent token sequences
