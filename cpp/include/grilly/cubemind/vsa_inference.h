#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/cubemind/types.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace cubemind {

// ── VSA Binary Inference Engine ─────────────────────────────────────
//
// Loads binarized MLP weights from disk and executes inference using
// XNOR + POPCNT on Vulkan compute. The weight matrix is stored in a
// persistent BDA-enabled GPU buffer so inference dispatches require
// only a push constant update (zero descriptor set overhead).
//
// Binary format (cubemind_student.bin):
//   Raw array of uint32_t words — bitpacked rows of the weight matrix.
//   Each row is (state_dim / 32) words = 320 words at d=10240.
//
// Dispatch pipeline:
//   1. Upload input state to GPU buffer (1.25 KB at d=10240)
//   2. Push BDA pointer + state_dim via push constants
//   3. Dispatch vsa-inference shader: XNOR + POPCNT per word
//   4. Download predicted state (1.25 KB)

/// Push constants for vsa-inference.glsl (single-layer XNOR test shader).
/// The weight matrix pointer is a BDA 64-bit GPU virtual address.
struct VSAInferenceParams {
    uint64_t projection_matrix_ptr;  // BDA pointer to binary weights
    uint32_t state_dim;              // Bipolar dimension (e.g., 10240)
    uint32_t _pad;                   // Alignment to 16 bytes
};

/// Push constants for vsa-bmm.glsl (2-layer binary matrix multiply).
/// Two BDA pointers for the full student MLP: Layer 1 + Layer 2.
///
/// Weight layout in cubemind_student.bin:
///   w1: hidden_dim neurons × words_per_vec words (e.g., 2048 × 320)
///   w2: state_dim neurons × hidden_words words  (e.g., 10240 × 64)
///
/// w2_ptr = w1_ptr + (hidden_dim × words_per_vec × sizeof(uint32))
struct VSABMMParams {
    uint64_t w1_ptr;       // BDA pointer to Layer 1 weights (input -> hidden)
    uint64_t w2_ptr;       // BDA pointer to Layer 2 weights (hidden -> output)
    uint32_t state_words;  // Input/output dim in uint32 words (e.g., 64 for dim=2048)
    uint32_t hidden_words; // Hidden dim in uint32 words (e.g., 32 for hidden=1024)
};

class VSAInferenceEngine {
public:
    /// @param pool       BufferPool for GPU memory allocation
    /// @param state_dim  VSA bipolar dimension (default 10240)
    VSAInferenceEngine(BufferPool& pool, uint32_t state_dim = 10240);
    ~VSAInferenceEngine();

    VSAInferenceEngine(const VSAInferenceEngine&) = delete;
    VSAInferenceEngine& operator=(const VSAInferenceEngine&) = delete;

    /// Load binarized weights from a binary file into persistent GPU buffer.
    /// The buffer is BDA-enabled so the shader can read via raw pointer.
    void load_weights(const std::string& filepath);

    /// Load binarized weights from a host uint32_t array.
    void load_weights(const uint32_t* data, size_t num_words);

    /// Execute XNOR inference: predicted = XNOR(input_state, weights).
    /// @param batch      CommandBatch for dispatch
    /// @param pipeCache  PipelineCache with vsa-inference shader loaded
    /// @param input      Bitpacked input state vector
    /// @return Bitpacked predicted state vector
    BitpackedVec infer(CommandBatch& batch, PipelineCache& pipeCache,
                       const BitpackedVec& input);

    /// Check if weights are loaded and BDA pointer is valid.
    bool weights_loaded() const { return weights_loaded_; }

    /// Get the BDA 64-bit GPU pointer to the weight buffer.
    uint64_t weights_device_address() const;

    /// Number of uint32 words in the weight buffer.
    size_t weight_words() const { return weight_words_; }

    uint32_t state_dim() const { return state_dim_; }

private:
    BufferPool& pool_;
    uint32_t state_dim_;
    uint32_t words_per_vec_;

    GrillyBuffer weight_buf_;   // Persistent BDA-enabled GPU buffer
    size_t weight_words_ = 0;
    bool weights_loaded_ = false;
};

// ── VSA Decode Push Constants ────────────────────────────────────────
/// Matches vsa-decode.glsl push constant layout.
struct VSADecodeParams {
    uint64_t codebook_ptr;     // BDA pointer to vocab codebook
    uint64_t pred_vector_ptr;  // BDA pointer to predicted vector
    uint32_t vocab_size;       // Number of codebook entries
    uint32_t uints_per_vec;    // Words per vector (dim/32)
};

// ── Bare-Metal VSA Inference Loop ───────────────────────────────────
//
// Two-shader pipeline:
//   1. Transformer (vsa-bmm): 2-layer binary MLP via XNOR + POPCNT
//   2. Decoder (vsa-decode): atomicMin Hamming search -> best token
//
// The codebook and logic weights are persistent BDA buffers in VRAM.
// After the first token, the ~5 MB logic matrix is pinned in L2 cache.

class VSABaremetalEngine {
public:
    /// @param pool        BufferPool for GPU memory allocation
    /// @param state_dim   VSA bipolar dimension (default 10240)
    /// @param hidden_dim  Hidden layer dimension (default 2048)
    VSABaremetalEngine(BufferPool& pool, uint32_t state_dim = 10240,
                       uint32_t hidden_dim = 4096);
    ~VSABaremetalEngine();

    VSABaremetalEngine(const VSABaremetalEngine&) = delete;
    VSABaremetalEngine& operator=(const VSABaremetalEngine&) = delete;

    /// Load the binarized student logic weights from file.
    void load_logic_weights(const std::string& filepath);

    /// Load the VSA codebook from file (vocab_size × words_per_vec uint32s).
    /// Also loads the vocabulary strings for token ID → word mapping.
    void load_codebook(const std::string& codebook_path,
                       const std::vector<std::string>& vocabulary);

    /// Load codebook from a host uint32_t array.
    void load_codebook(const uint32_t* data, size_t num_words,
                       const std::vector<std::string>& vocabulary);

    /// Run the full inference loop: generate up to max_tokens.
    /// @param batch      CommandBatch for shader dispatch
    /// @param pipeCache  PipelineCache with vsa-inference + vsa-decode shaders
    /// @param initial_state  Starting bitpacked state vector
    /// @param max_tokens Maximum tokens to generate
    /// @return Vector of decoded words
    std::vector<std::string> generate(
        CommandBatch& batch, PipelineCache& pipeCache,
        const BitpackedVec& initial_state,
        uint32_t max_tokens = 50);

    /// Single inference step: transform + decode.
    /// Returns (decoded_word, hamming_distance, predicted_state).
    struct StepResult {
        std::string word;
        uint32_t distance;
        BitpackedVec predicted_state;
    };
    StepResult step(CommandBatch& batch, PipelineCache& pipeCache,
                    const BitpackedVec& current_state);

    bool ready() const { return logic_loaded_ && codebook_loaded_; }
    uint32_t vocab_size() const { return vocab_size_; }
    uint32_t state_dim() const { return state_dim_; }
    uint32_t hidden_dim() const { return hidden_dim_; }

    /// Get word by token ID.
    const std::string& get_word(uint32_t token_id) const;

private:
    BufferPool& pool_;
    uint32_t state_dim_;
    uint32_t hidden_dim_;
    uint32_t words_per_vec_;    // state_dim / 32
    uint32_t hidden_words_;     // hidden_dim / 32

    // Logic weights (persistent BDA buffer — contains both layers)
    GrillyBuffer logic_buf_;
    size_t w1_size_bytes_ = 0;  // Offset to w2 = w1_size_bytes_
    bool logic_loaded_ = false;

    // Codebook (persistent BDA buffer)
    GrillyBuffer codebook_buf_;
    std::vector<std::string> vocabulary_;
    uint32_t vocab_size_ = 0;
    bool codebook_loaded_ = false;

    // Helper: load binary file to uint32 vector
    static std::vector<uint32_t> loadBinaryFile(const std::string& path);
};

}  // namespace cubemind
}  // namespace grilly
