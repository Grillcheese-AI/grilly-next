#pragma once

#include <functional>
#include <string>
#include <vector>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"
#include "grilly/cubemind/types.h"
#include "grilly/cubemind/vsa.h"

namespace grilly {
namespace cubemind {

/// Push constants for resonator-bitpacked.glsl (16 bytes)
struct ResonatorParams {
    uint32_t dim;             // Original bipolar dimension (e.g., 10240)
    uint32_t words_per_vec;   // dim / 32 (e.g., 320)
    uint32_t codebook_size;   // Number of entries in the codebook
    uint32_t _pad;            // Alignment padding
};

/// Result from a single resonation step.
struct ResonateResult {
    uint32_t best_index;      // Index of highest-similarity codebook entry
    float best_similarity;    // Similarity score [-1.0, +1.0]
    std::vector<float> all_similarities;  // Full similarity vector (optional)
};

/// ResonatorNetwork: GPU-accelerated VSA generation via Hamming resonance.
///
/// The resonator takes a bitpacked bundle (a superposition of bound
/// word-role-position triples) and "decomposes" it by iteratively:
///   1. Unbinding each position probe (XOR with position role vector)
///   2. Resonating against a word codebook (GPU Hamming similarity)
///   3. Explaining away the found component (subtract from accumulator)
///
/// This replaces autoregressive token generation with parallel VSA
/// decomposition. A 20-word sentence can be decoded in ~0.6ms total
/// (20 × ~29μs per shader dispatch).
///
/// The codebook is uploaded to VRAM once and persists across queries,
/// just like the VSACache's persistent buffer pattern.
///
class ResonatorNetwork {
public:
    /// @param pool   BufferPool for GPU memory allocation
    /// @param batch  CommandBatch for shader dispatch
    /// @param pipeCache  PipelineCache for shader pipeline management
    /// @param dim    VSA bipolar dimension (default 10240)
    ResonatorNetwork(BufferPool& pool, CommandBatch& batch,
                     PipelineCache& pipeCache, uint32_t dim = 10240);

    ~ResonatorNetwork();

    // ── Codebook Management ──────────────────────────────────────────

    /// Load a word codebook into VRAM.
    /// @param words     Vocabulary strings (for index → word mapping)
    /// @param vectors   Bitpacked vectors: words.size() entries, each dim/32 uint32s
    ///                  Flat row-major: vectors[i * words_per_vec + j]
    void load_codebook(const std::vector<std::string>& words,
                       const uint32_t* vectors);

    /// Load codebook from bipolar int8 vectors (bitpacks internally).
    void load_codebook_bipolar(const std::vector<std::string>& words,
                                const int8_t* bipolar_vectors);

    /// Save codebook to a binary file for checkpointing.
    /// Format: [magic=0x47524C59][version=1][count][dim]
    ///         then per entry: [word_len][word_bytes][bitpacked_vector]
    void save_codebook(const std::string& path) const;

    /// Load codebook from a binary checkpoint file and upload to VRAM.
    void load_codebook_file(const std::string& path);

    /// Number of words in the codebook.
    size_t codebook_size() const { return words_.size(); }

    /// Get word string by codebook index.
    const std::string& get_word(uint32_t index) const;

    // ── Resonation (GPU Dispatch) ────────────────────────────────────

    /// Resonate a bitpacked query against the entire codebook.
    /// Returns the index and similarity of the best match,
    /// plus optionally all similarity scores.
    ///
    /// This dispatches the resonator-bitpacked.glsl shader.
    /// At codebook_size=10K, dim=10240: ~29μs per dispatch.
    ResonateResult resonate(const BitpackedVec& query,
                             bool return_all_similarities = false);

    /// CPU fallback resonation (for testing / no-GPU environments).
    ResonateResult resonate_cpu(const BitpackedVec& query) const;

    // ── Sentence Generation (Explaining-Away Loop) ───────────────────

    /// Generate a sentence by decomposing a bitpacked bundle.
    ///
    /// The TextEncoder encodes each token as: word ⊗ role ⊗ position
    /// (three-way binding). To recover the word, we must unbind BOTH
    /// the dependency role and position vectors:
    ///
    ///   probe = bundle ⊗ role_vec ⊗ pos_vec
    ///         = (word ⊗ role ⊗ pos + noise) ⊗ role ⊗ pos
    ///         = word + noise'
    ///
    /// Algorithm for each position i in [0, length):
    ///   1. Threshold accumulator → current bitpacked bundle
    ///   2. Unbind: probe = current XOR role_vec XOR pos_vec
    ///   3. Resonate: find best match in codebook
    ///   4. Explain away: subtract word ⊗ role ⊗ pos from accumulator
    ///
    /// @param sentence_bundle  Bitpacked superposition of bound word-role-pos triples
    /// @param dependency_roles Dependency role labels for each position (e.g., "nsubj", "ROOT")
    /// @param positions        Position indices for each slot
    /// @param explain_away     Enable explaining-away cleanup (default: true)
    /// @return Vector of (word, similarity) pairs
    std::vector<std::pair<std::string, float>> generate_sentence(
        const BitpackedVec& sentence_bundle,
        const std::vector<std::string>& dependency_roles,
        const std::vector<uint32_t>& positions,
        bool explain_away = true);

    /// Simple positional-only decode (legacy, for testing).
    /// Only unbinds position — works when encoding used position-only binding.
    std::vector<std::pair<std::string, float>> generate_positional(
        const BitpackedVec& sentence_bundle,
        uint32_t length,
        bool explain_away = true);

    /// Unbind a single role from a bundle and resonate against codebook.
    /// This is the VSA equivalent of a key→value lookup in a hash table.
    ///
    ///   query = bundle ⊗ role_vec
    ///         = (filler₁⊗role₁ + filler₂⊗role₂ + ...) ⊗ roleᵢ
    ///         = fillerᵢ + noise   (because roleᵢ⊗roleᵢ = identity)
    ///
    /// @param bundle     Bitpacked superposition of bound role-filler pairs
    /// @param role_key   The role to unbind (e.g., "currency", "nsubj")
    /// @param key_prefix Prefix used during encoding (e.g., "role_")
    /// @return (word, similarity) of the best codebook match
    std::pair<std::string, float> query_role(
        const BitpackedVec& bundle,
        const std::string& role_key,
        const std::string& key_prefix);

    /// Query a specific (role, position) slot from a sentence bundle.
    std::pair<std::string, float> query_slot(
        const BitpackedVec& bundle,
        const std::string& dep_role,
        uint32_t position);

    /// Compute the analogical mapping between two bundles.
    /// mapping = bind(inverse(source), target)
    ///         = source XOR target  (since XOR is self-inverse in bipolar)
    ///
    /// Then: bind(mapping, source_filler) ≈ target_filler
    BitpackedVec compute_analogy_map(
        const BitpackedVec& source_bundle,
        const BitpackedVec& target_bundle);

    /// Apply an analogical mapping to a query filler.
    /// "If USD maps to X in the target frame, what is X?"
    std::pair<std::string, float> apply_analogy(
        const BitpackedVec& analogy_map,
        const BitpackedVec& query_filler);

    /// Unbind N roles simultaneously on GPU, then batch-resonate.
    /// Much faster than sequential unbind+resonate for long sentences.
    std::vector<std::pair<std::string, float>> batch_unbind(
        const BitpackedVec& bundle,
        const std::vector<std::string>& role_keys,
        const std::vector<uint32_t>& positions);




    // ── Stats ────────────────────────────────────────────────────────

    uint64_t total_resonations() const { return total_resonations_; }
    double last_resonate_ms() const { return last_resonate_ms_; }

private:
    BufferPool& pool_;
    CommandBatch& batch_;
    PipelineCache& pipeCache_;
    uint32_t dim_;
    uint32_t words_per_vec_;

    // Codebook (persistent in VRAM)
    std::vector<std::string> words_;
    GrillyBuffer codebook_buf_;          // GPU buffer (persistent)
    std::vector<uint32_t> codebook_host_; // Host copy for CPU fallback
    bool codebook_loaded_ = false;

    // Stats
    uint64_t total_resonations_ = 0;
    double last_resonate_ms_ = 0.0;

    // Internal: dispatch the shader and read back similarities
    void dispatch_resonator(const uint32_t* query_packed,
                             float* similarities_out);
};

// ── Hyper-NAR Decoding Loop ─────────────────────────────────────────
//
// Non-autoregressive generation via VSA geometric trajectories.
// Takes a fused multimodal state and iteratively:
//   1. Predicts a structural transition vector (hypernetwork)
//   2. Applies the transition via VSA binding (XOR)
//   3. Resonates the result against the codebook (GPU Hamming search)
//   4. Emits the decoded word
//
// Complexity per token: O(1) MLP + O(1) XOR + O(N) Hamming search
// No quadratic attention. No context window scaling.

/// Callback type for the hypernetwork forward pass.
/// Takes the current bitpacked state and returns a bipolar transformation vector.
using HypernetworkPredictor = std::function<std::vector<int8_t>(
    const BitpackedVec& current_state)>;

/// Execute the Hyper-NAR decoding loop.
///
/// @param fused_prompt_state  Initial state (fused multimodal or text-only)
/// @param resonator           ResonatorNetwork with loaded codebook
/// @param predictor           Hypernetwork forward pass callback
/// @param max_tokens          Maximum tokens to generate
/// @return Vector of decoded words
std::vector<std::string> hyper_nar_decode(
    const BitpackedVec& fused_prompt_state,
    ResonatorNetwork& resonator,
    HypernetworkPredictor predictor,
    uint32_t max_tokens = 50);

}  // namespace cubemind
}  // namespace grilly
