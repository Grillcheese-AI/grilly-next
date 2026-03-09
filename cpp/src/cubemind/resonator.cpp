#include "grilly/cubemind/resonator.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace grilly {
namespace cubemind {

// ── Portable timer ──────────────────────────────────────────────────────

static double now_ms() {
    using Clock = std::chrono::high_resolution_clock;
    auto tp = Clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(tp).count();
}

// ── ResonatorNetwork ────────────────────────────────────────────────────

ResonatorNetwork::ResonatorNetwork(BufferPool& pool, CommandBatch& batch,
                                   PipelineCache& pipeCache, uint32_t dim)
    : pool_(pool), batch_(batch), pipeCache_(pipeCache),
      dim_(dim), words_per_vec_((dim + 31) / 32) {}

ResonatorNetwork::~ResonatorNetwork() {
    if (codebook_loaded_ && codebook_buf_.handle != VK_NULL_HANDLE) {
        pool_.release(codebook_buf_);
    }
}

// ── Codebook Management ─────────────────────────────────────────────────

void ResonatorNetwork::load_codebook(const std::vector<std::string>& words,
                                      const uint32_t* vectors) {
    // Release old codebook if any
    if (codebook_loaded_ && codebook_buf_.handle != VK_NULL_HANDLE) {
        pool_.release(codebook_buf_);
    }

    words_ = words;
    size_t total_words = words.size() * words_per_vec_;
    size_t bytes = total_words * sizeof(uint32_t);

    // Host copy for CPU fallback
    codebook_host_.assign(vectors, vectors + total_words);

    // Upload to VRAM (persistent — stays until codebook is replaced)
    codebook_buf_ = pool_.acquire(bytes);
    pool_.upload(codebook_buf_, reinterpret_cast<const float*>(vectors), bytes);
    codebook_loaded_ = true;
}

void ResonatorNetwork::load_codebook_bipolar(
    const std::vector<std::string>& words,
    const int8_t* bipolar_vectors) {
    // Bitpack each word's bipolar vector and load as packed codebook
    size_t total_words_packed = words.size() * words_per_vec_;
    std::vector<uint32_t> packed(total_words_packed, 0);

    for (size_t w = 0; w < words.size(); ++w) {
        const int8_t* bip = bipolar_vectors + w * dim_;
        uint32_t* out = packed.data() + w * words_per_vec_;

        for (uint32_t j = 0; j < dim_; ++j) {
            if (bip[j] > 0) {
                out[j / 32] |= (1u << (j % 32));
            }
        }
    }

    load_codebook(words, packed.data());
}

const std::string& ResonatorNetwork::get_word(uint32_t index) const {
    static const std::string kUnknown = "<UNK>";
    if (index < words_.size()) return words_[index];
    return kUnknown;
}

// ── Codebook Checkpointing ──────────────────────────────────────────

static constexpr uint32_t kCodebookMagic   = 0x47524C59;  // "GRLY"
static constexpr uint32_t kCodebookVersion = 1;

void ResonatorNetwork::save_codebook(const std::string& path) const {
    if (!codebook_loaded_)
        throw std::runtime_error("save_codebook: no codebook loaded");

    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error("save_codebook: cannot open " + path);

    uint32_t count = static_cast<uint32_t>(words_.size());
    uint32_t dim = dim_;

    // Header
    out.write(reinterpret_cast<const char*>(&kCodebookMagic), 4);
    out.write(reinterpret_cast<const char*>(&kCodebookVersion), 4);
    out.write(reinterpret_cast<const char*>(&count), 4);
    out.write(reinterpret_cast<const char*>(&dim), 4);

    // Entries
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t word_len = static_cast<uint32_t>(words_[i].size());
        out.write(reinterpret_cast<const char*>(&word_len), 4);
        out.write(words_[i].data(), word_len);

        const uint32_t* vec = codebook_host_.data() + i * words_per_vec_;
        out.write(reinterpret_cast<const char*>(vec),
                  words_per_vec_ * sizeof(uint32_t));
    }

    if (!out.good())
        throw std::runtime_error("save_codebook: write error on " + path);
}

void ResonatorNetwork::load_codebook_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("load_codebook_file: cannot open " + path);

    uint32_t magic, version, count, dim;
    in.read(reinterpret_cast<char*>(&magic), 4);
    in.read(reinterpret_cast<char*>(&version), 4);
    in.read(reinterpret_cast<char*>(&count), 4);
    in.read(reinterpret_cast<char*>(&dim), 4);

    if (magic != kCodebookMagic)
        throw std::runtime_error("load_codebook_file: bad magic (not a GRLY codebook)");
    if (version != kCodebookVersion)
        throw std::runtime_error("load_codebook_file: unsupported version " +
                                 std::to_string(version));
    if (dim != dim_)
        throw std::runtime_error("load_codebook_file: dim mismatch (file=" +
                                 std::to_string(dim) + " resonator=" +
                                 std::to_string(dim_) + ")");

    std::vector<std::string> words(count);
    uint32_t file_words_per_vec = (dim + 31) / 32;
    std::vector<uint32_t> vectors(count * file_words_per_vec);

    for (uint32_t i = 0; i < count; ++i) {
        uint32_t word_len;
        in.read(reinterpret_cast<char*>(&word_len), 4);
        words[i].resize(word_len);
        in.read(words[i].data(), word_len);

        uint32_t* vec = vectors.data() + i * file_words_per_vec;
        in.read(reinterpret_cast<char*>(vec),
                file_words_per_vec * sizeof(uint32_t));
    }

    if (!in.good())
        throw std::runtime_error("load_codebook_file: read error on " + path);

    // Upload to VRAM via the existing load_codebook path
    load_codebook(words, vectors.data());
}

// ── GPU Resonation ──────────────────────────────────────────────────────

void ResonatorNetwork::dispatch_resonator(const uint32_t* query_packed,
                                           float* similarities_out) {
    if (!codebook_loaded_) {
        throw std::runtime_error("Codebook not loaded");
    }

    uint32_t cb_size = static_cast<uint32_t>(words_.size());
    size_t queryBytes = size_t(words_per_vec_) * sizeof(uint32_t);
    size_t simBytes = size_t(cb_size) * sizeof(float);
    size_t cbBytes = size_t(cb_size) * words_per_vec_ * sizeof(uint32_t);

    GrillyBuffer bufQuery = pool_.acquire(queryBytes);
    GrillyBuffer bufSim = pool_.acquire(simBytes);

    pool_.upload(bufQuery, reinterpret_cast<const float*>(query_packed),
                 queryBytes);

    PipelineEntry pipe = pipeCache_.getOrCreate(
        "resonator-bitpacked", 3, sizeof(ResonatorParams));

    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {bufQuery.handle, 0, queryBytes},
        {codebook_buf_.handle, 0, cbBytes},
        {bufSim.handle, 0, simBytes},
    };
    VkDescriptorSet descSet = pipeCache_.allocDescriptorSet(
        "resonator-bitpacked", bufInfos);

    ResonatorParams push{dim_, words_per_vec_, cb_size, 0};

    // One workgroup per codebook entry (each WG has 256 threads)
    uint32_t gx = cb_size;

    batch_.begin();
    batch_.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                    &push, sizeof(push));
    batch_.submit();

    pool_.download(bufSim, similarities_out, simBytes);

    pool_.release(bufQuery);
    pool_.release(bufSim);
}

ResonateResult ResonatorNetwork::resonate(const BitpackedVec& query,
                                           bool return_all_similarities) {
    double t0 = now_ms();

    uint32_t cb_size = static_cast<uint32_t>(words_.size());
    std::vector<float> sims(cb_size);

    if (pipeCache_.hasShader("resonator-bitpacked")) {
        dispatch_resonator(query.data.data(), sims.data());
    } else {
        // CPU fallback
        auto cpu_result = resonate_cpu(query);
        last_resonate_ms_ = now_ms() - t0;
        total_resonations_++;
        return cpu_result;
    }

    // Find argmax
    uint32_t best_idx = 0;
    float best_sim = -2.0f;
    for (uint32_t i = 0; i < cb_size; ++i) {
        if (sims[i] > best_sim) {
            best_sim = sims[i];
            best_idx = i;
        }
    }

    last_resonate_ms_ = now_ms() - t0;
    total_resonations_++;

    ResonateResult result;
    result.best_index = best_idx;
    result.best_similarity = best_sim;
    if (return_all_similarities) {
        result.all_similarities = std::move(sims);
    }
    return result;
}

ResonateResult ResonatorNetwork::resonate_cpu(const BitpackedVec& query) const {
    uint32_t cb_size = static_cast<uint32_t>(words_.size());

    uint32_t best_idx = 0;
    float best_sim = -2.0f;
    std::vector<float> all_sims(cb_size);

    for (uint32_t w = 0; w < cb_size; ++w) {
        uint32_t hamming = 0;
        const uint32_t* cb_entry = codebook_host_.data() + w * words_per_vec_;
        for (uint32_t j = 0; j < words_per_vec_; ++j) {
#ifdef _MSC_VER
            hamming += __popcnt(query.data[j] ^ cb_entry[j]);
#else
            hamming += __builtin_popcount(query.data[j] ^ cb_entry[j]);
#endif
        }
        float sim = (float(dim_) - 2.0f * float(hamming)) / float(dim_);
        all_sims[w] = sim;
        if (sim > best_sim) {
            best_sim = sim;
            best_idx = w;
        }
    }

    ResonateResult result;
    result.best_index = best_idx;
    result.best_similarity = best_sim;
    result.all_similarities = std::move(all_sims);
    return result;
}

// ── Sentence Generation (Explaining Away) ────────────────────────────

std::vector<std::pair<std::string, float>> ResonatorNetwork::generate_sentence(
    const BitpackedVec& sentence_bundle,
    const std::vector<std::string>& dependency_roles,
    const std::vector<uint32_t>& positions,
    bool explain_away) {

    uint32_t length = static_cast<uint32_t>(dependency_roles.size());
    std::vector<std::pair<std::string, float>> generated;
    generated.reserve(length);

    // ── Explaining-Away Accumulator ──────────────────────────────────
    //
    // The bundle is a majority-vote superposition of N bound components.
    // Each component was encoded as: word ⊗ role ⊗ position (three-way).
    // After thresholding to {-1,+1}, we lose the vote counts. We use an
    // int16 accumulator to preserve analog magnitudes during subtraction.
    //
    //   accumulator[d] = unpack(bundle)[d]   (initially ±1)
    //   After finding word_i: accumulator -= bipolar(word_i ⊗ role_i ⊗ pos_i)
    //   Query for next position: threshold(accumulator) → BitpackedVec

    // Unpack bundle to int16 accumulator
    std::vector<int16_t> accumulator(dim_);
    for (uint32_t d = 0; d < dim_; ++d) {
        uint32_t word_idx = d / 32;
        uint32_t bit_idx = d % 32;
        bool bit_set = (sentence_bundle.data[word_idx] >> bit_idx) & 1u;
        accumulator[d] = bit_set ? 1 : -1;
    }

    for (uint32_t i = 0; i < length; ++i) {
        // ── 1. Threshold accumulator to get current bundle ──────────
        BitpackedVec current;
        current.dim = dim_;
        current.data.resize(words_per_vec_, 0);

        for (uint32_t d = 0; d < dim_; ++d) {
            if (accumulator[d] > 0) {
                current.data[d / 32] |= (1u << (d % 32));
            }
        }

        // ── 2. Unbind BOTH role AND position (three-way decode) ─────
        //
        // Encoding was: bound = word ⊗ role ⊗ pos
        // Since XOR binding is self-inverse:
        //   probe = current ⊗ role ⊗ pos
        //         = (word ⊗ role ⊗ pos + noise) ⊗ role ⊗ pos
        //         = word + noise'   (role and pos cancel out)
        //
        // We must use the SAME key prefixes as TextEncoder:
        //   role: blake3Role("role_" + dependency_role)
        //   pos:  blake3Role("pos_"  + position_index)

        std::string dep_role = (i < dependency_roles.size())
                                   ? dependency_roles[i] : "UNK";
        uint32_t pos = (i < positions.size())
                           ? positions[i] : i;

        std::vector<int8_t> role_bipolar = blake3Role("role_" + dep_role, dim_);
        std::vector<int8_t> pos_bipolar  = blake3Role("pos_" + std::to_string(pos), dim_);

        BitpackedVec role_packed = vsaBitpack(role_bipolar.data(), dim_);
        BitpackedVec pos_packed  = vsaBitpack(pos_bipolar.data(), dim_);

        BitpackedVec probe;
        probe.dim = dim_;
        probe.data.resize(words_per_vec_);
        for (uint32_t j = 0; j < words_per_vec_; ++j) {
            // XOR both role and position to unbind the three-way binding
            probe.data[j] = current.data[j] ^ role_packed.data[j] ^ pos_packed.data[j];
        }

        // ── 3. Resonate against codebook (29μs GPU shader) ──────────
        ResonateResult result = resonate(probe);

        std::string word = get_word(result.best_index);
        generated.push_back({word, result.best_similarity});

        // ── 4. Explaining Away ──────────────────────────────────────
        //
        // Reconstruct the THREE-WAY bound component: word ⊗ role ⊗ pos
        // Subtract it from the accumulator to clean the bundle for
        // the next position.
        //
        // In bipolar algebra: bound[d] = word[d] * role[d] * pos[d]
        // Subtracting: accumulator[d] -= bound[d]
        //
        // This prevents the winning word from "echoing" and causing
        // repetition or interference in subsequent positions.
        if (explain_away && i + 1 < length) {
            const uint32_t* winner_packed =
                codebook_host_.data() + result.best_index * words_per_vec_;

            for (uint32_t d = 0; d < dim_; ++d) {
                uint32_t w = d / 32;
                uint32_t b = d % 32;

                // Unpack winner word bit → bipolar
                int8_t word_val = ((winner_packed[w] >> b) & 1u) ? 1 : -1;

                // Three-way bound component = word * role * pos
                int8_t bound_val = word_val * role_bipolar[d] * pos_bipolar[d];

                // Subtract from accumulator
                accumulator[d] -= bound_val;
            }
        }
    }

    return generated;
}

std::pair<std::string, float> ResonatorNetwork::query_role(
    const BitpackedVec& bundle,
    const std::string& role_key,
    const std::string& key_prefix) {

    // Generate the same BLAKE3 role vector used during encoding
    std::vector<int8_t> role_bipolar = blake3Role(key_prefix + role_key, dim_);
    BitpackedVec role_packed = vsaBitpack(role_bipolar.data(), dim_);

    // Unbind: probe = bundle XOR role
    BitpackedVec probe;
    probe.dim = dim_;
    probe.data.resize(words_per_vec_);
    for (uint32_t j = 0; j < words_per_vec_; ++j) {
        probe.data[j] = bundle.data[j] ^ role_packed.data[j];
    }

    // Resonate against codebook to find the filler
    ResonateResult result = resonate(probe);
    return {get_word(result.best_index), result.best_similarity};
}

std::pair<std::string, float> ResonatorNetwork::query_slot(
    const BitpackedVec& bundle,
    const std::string& dep_role,
    uint32_t position) {

    std::vector<int8_t> role_bip = blake3Role("role_" + dep_role, dim_);
    std::vector<int8_t> pos_bip  = blake3Role("pos_" + std::to_string(position), dim_);

    BitpackedVec role_pk = vsaBitpack(role_bip.data(), dim_);
    BitpackedVec pos_pk  = vsaBitpack(pos_bip.data(), dim_);

    BitpackedVec probe;
    probe.dim = dim_;
    probe.data.resize(words_per_vec_);
    for (uint32_t j = 0; j < words_per_vec_; ++j) {
        probe.data[j] = bundle.data[j] ^ role_pk.data[j] ^ pos_pk.data[j];
    }

    ResonateResult result = resonate(probe);
    return {get_word(result.best_index), result.best_similarity};
}

BitpackedVec ResonatorNetwork::compute_analogy_map(
    const BitpackedVec& source_bundle,
    const BitpackedVec& target_bundle) {

    BitpackedVec mapping;
    mapping.dim = dim_;
    mapping.data.resize(words_per_vec_);
    for (uint32_t j = 0; j < words_per_vec_; ++j) {
        // In bipolar VSA, inverse = self, so:
        // mapping = source^(-1) ⊗ target = source XOR target
        mapping.data[j] = source_bundle.data[j] ^ target_bundle.data[j];
    }
    return mapping;
}

std::pair<std::string, float> ResonatorNetwork::apply_analogy(
    const BitpackedVec& analogy_map,
    const BitpackedVec& query_filler) {

    BitpackedVec probe;
    probe.dim = dim_;
    probe.data.resize(words_per_vec_);
    for (uint32_t j = 0; j < words_per_vec_; ++j) {
        probe.data[j] = analogy_map.data[j] ^ query_filler.data[j];
    }

    ResonateResult result = resonate(probe);
    return {get_word(result.best_index), result.best_similarity};
}

std::vector<std::pair<std::string, float>> ResonatorNetwork::batch_unbind(
    const BitpackedVec& bundle,
    const std::vector<std::string>& role_keys,
    const std::vector<uint32_t>& positions) {

    uint32_t N = static_cast<uint32_t>(role_keys.size());
    if (N == 0) return {};

    // 1. Build the operator pool (role XOR pos for each slot)
    std::vector<uint32_t> op_pool(N * words_per_vec_);
    for (uint32_t i = 0; i < N; ++i) {
        auto role_bip = blake3Role("role_" + role_keys[i], dim_);
        auto pos_bip  = blake3Role("pos_" + std::to_string(positions[i]), dim_);
        auto role_pk  = vsaBitpack(role_bip.data(), dim_);
        auto pos_pk   = vsaBitpack(pos_bip.data(), dim_);

        for (uint32_t j = 0; j < words_per_vec_; ++j) {
            op_pool[i * words_per_vec_ + j] =
                role_pk.data[j] ^ pos_pk.data[j];
        }
    }

    // Output buffer for unbound probes
    std::vector<uint32_t> hypotheses(N * words_per_vec_, 0);

    // 2. Dispatch vsa-logic-apply.glsl
    if (pipeCache_.hasShader("vsa-logic-apply")) {
        size_t mem_bytes = words_per_vec_ * sizeof(uint32_t);
        size_t ops_bytes = N * words_per_vec_ * sizeof(uint32_t);
        size_t hyp_bytes = N * words_per_vec_ * sizeof(uint32_t);

        GrillyBuffer bufMem = pool_.acquire(mem_bytes);
        GrillyBuffer bufOps = pool_.acquire(ops_bytes);
        GrillyBuffer bufHyp = pool_.acquire(hyp_bytes);

        pool_.upload(bufMem, reinterpret_cast<const float*>(bundle.data.data()), mem_bytes);
        pool_.upload(bufOps, reinterpret_cast<const float*>(op_pool.data()), ops_bytes);

        PipelineEntry pipe = pipeCache_.getOrCreate("vsa-logic-apply", 3, sizeof(VSALogicApplyParams));

        std::vector<VkDescriptorBufferInfo> bufInfos = {
            {bufMem.handle, 0, mem_bytes},
            {bufOps.handle, 0, ops_bytes},
            {bufHyp.handle, 0, hyp_bytes}
        };

        VkDescriptorSet descSet = pipeCache_.allocDescriptorSet("vsa-logic-apply", bufInfos);

        VSALogicApplyParams push{words_per_vec_, N, 0, 0};

        batch_.begin();
        batch_.dispatch(pipe.pipeline, pipe.layout, descSet, N, 1, 1, &push, sizeof(push));
        batch_.submit();

        pool_.download(bufHyp, reinterpret_cast<float*>(hypotheses.data()), hyp_bytes);

        pool_.release(bufMem);
        pool_.release(bufOps);
        pool_.release(bufHyp);
    } else {
        // CPU Fallback: XOR the working memory with each operator
        for (uint32_t i = 0; i < N; ++i) {
            uint32_t p_off = i * words_per_vec_;
            for (uint32_t j = 0; j < words_per_vec_; ++j) {
                hypotheses[p_off + j] = bundle.data[j] ^ op_pool[p_off + j];
            }
        }
    }

    // 3. Batch resonate each probe against codebook
    std::vector<std::pair<std::string, float>> results;
    results.reserve(N);
    for (uint32_t i = 0; i < N; ++i) {
        BitpackedVec probe;
        probe.dim = dim_;
        probe.data.assign(hypotheses.begin() + i * words_per_vec_,
                          hypotheses.begin() + (i + 1) * words_per_vec_);
        auto res = resonate(probe);
        results.push_back({get_word(res.best_index), res.best_similarity});
    }
    return results;
}



std::vector<std::pair<std::string, float>> ResonatorNetwork::generate_positional(
    const BitpackedVec& sentence_bundle,
    uint32_t length,
    bool explain_away) {
    // Legacy: position-only unbinding (no dependency roles).
    // Uses "UNK" as the dependency role for all positions.
    std::vector<std::string> dep_roles(length, "UNK");
    std::vector<uint32_t> positions(length);
    for (uint32_t i = 0; i < length; ++i) {
        positions[i] = i;
    }
    return generate_sentence(sentence_bundle, dep_roles, positions, explain_away);
}

// ── Hyper-NAR Decoding Loop ─────────────────────────────────────────
//
// Non-autoregressive generation via VSA geometric trajectories.
// The entire generative step requires exactly:
//   1. One lightweight MLP forward pass (the hypernetwork predictor)
//   2. One vsaBind operation (bitwise XOR in bipolar domain)
//   3. One resonator.resonate() GPU Hamming search
//
// No context window. No quadratic attention. Just XOR + POPCNT.

std::vector<std::string> hyper_nar_decode(
    const BitpackedVec& fused_prompt_state,
    ResonatorNetwork& resonator,
    HypernetworkPredictor predictor,
    uint32_t max_tokens) {

    std::vector<std::string> generated_sequence;
    uint32_t dim = fused_prompt_state.dim;

    // Initialize the generative state with our fused multimodal prompt
    BitpackedVec current_state = fused_prompt_state;

    for (uint32_t i = 0; i < max_tokens; ++i) {
        // 1. The Student Hypernetwork predicts the structural transition vector
        std::vector<int8_t> transformation = predictor(current_state);

        // 2. Unpack the bitpacked current state back to bipolar {-1, 1}
        std::vector<int8_t> unpacked_state = vsaUnpack(current_state);

        // 3. Execute the structural transition via VSA Binding
        //    (element-wise multiply in bipolar = XOR in bitpacked)
        std::vector<int8_t> target_state =
            vsaBind(unpacked_state.data(), transformation.data(), dim);

        // 4. Pack the newly generated target state
        BitpackedVec packed_target = vsaBitpack(target_state.data(), dim);

        // 5. Resonate: GPU Hamming search against the entire vocabulary
        //    to snap the noisy trajectory to a crisp, readable word
        ResonateResult result = resonator.resonate(packed_target, false);

        // 6. Extract the human-readable string
        std::string decoded_word = resonator.get_word(result.best_index);
        generated_sequence.push_back(decoded_word);

        // 7. Stop generation if the network predicts End-Of-Sequence
        if (decoded_word == "<EOS>") {
            break;
        }

        // 8. Update state for next iteration
        // Use the resolved packed target to prevent drift accumulation
        current_state = packed_target;
    }

    return generated_sequence;
}

}  // namespace cubemind
}  // namespace grilly
