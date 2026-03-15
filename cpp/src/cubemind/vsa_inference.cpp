#include "grilly/cubemind/vsa_inference.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace grilly {
namespace cubemind {

// ── VSAInferenceEngine ──────────────────────────────────────────────

VSAInferenceEngine::VSAInferenceEngine(BufferPool& pool, uint32_t state_dim)
    : pool_(pool)
    , state_dim_(state_dim)
    , words_per_vec_((state_dim + 31) / 32)
    , weight_buf_{} {}

VSAInferenceEngine::~VSAInferenceEngine() {
    if (weight_buf_.handle != VK_NULL_HANDLE) {
        pool_.release(weight_buf_);
    }
}

void VSAInferenceEngine::load_weights(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("Failed to open weight file: " + filepath);

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint32_t> host_weights(file_size / sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(host_weights.data()), file_size);

    load_weights(host_weights.data(), host_weights.size());

    std::cout << "[OK] VSA inference weights loaded: "
              << file_size / 1024 << " KB from " << filepath << std::endl;
}

void VSAInferenceEngine::load_weights(const uint32_t* data, size_t num_words) {
    // Release old buffer if present
    if (weight_buf_.handle != VK_NULL_HANDLE) {
        pool_.release(weight_buf_);
        weight_buf_ = {};
    }

    weight_words_ = num_words;
    size_t bytes = num_words * sizeof(uint32_t);

    // Allocate a persistent buffer — BDA flag is automatically added
    // by BufferPool when VK_KHR_buffer_device_address is enabled.
    weight_buf_ = pool_.acquire(bytes);

    // Upload the binary weights
    pool_.upload(weight_buf_, reinterpret_cast<const float*>(data), bytes);

    weights_loaded_ = true;

    if (weight_buf_.deviceAddress != 0) {
        std::cout << "[OK] BDA pointer: 0x" << std::hex
                  << weight_buf_.deviceAddress << std::dec << std::endl;
    }
}

uint64_t VSAInferenceEngine::weights_device_address() const {
    return weight_buf_.deviceAddress;
}

BitpackedVec VSAInferenceEngine::infer(CommandBatch& batch,
                                        PipelineCache& pipeCache,
                                        const BitpackedVec& input) {
    if (!weights_loaded_)
        throw std::runtime_error("VSAInferenceEngine: weights not loaded");

    const size_t stateBytes = words_per_vec_ * sizeof(uint32_t);

    // Allocate input/output buffers
    GrillyBuffer bufInput = pool_.acquire(stateBytes);
    GrillyBuffer bufOutput = pool_.acquire(stateBytes);

    // Upload input state
    pool_.upload(bufInput, reinterpret_cast<const float*>(input.data.data()),
                 stateBytes);

    if (pipeCache.hasShader("vsa-inference") &&
        weight_buf_.deviceAddress != 0) {
        // BDA path: weight matrix via push constant pointer,
        // input/output via descriptor bindings (2 buffers)
        PipelineEntry pipe = pipeCache.getOrCreate(
            "vsa-inference", 2, sizeof(VSAInferenceParams));

        std::vector<VkDescriptorBufferInfo> bufInfos = {
            {bufInput.handle, 0, stateBytes},
            {bufOutput.handle, 0, stateBytes},
        };
        VkDescriptorSet descSet =
            pipeCache.allocDescriptorSet("vsa-inference", bufInfos);

        VSAInferenceParams push{};
        push.projection_matrix_ptr = weight_buf_.deviceAddress;
        push.state_dim = state_dim_;
        push._pad = 0;

        uint32_t gx = (words_per_vec_ + 255) / 256;

        batch.begin();
        batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                       &push, sizeof(push));
        batch.submit();
    } else {
        // CPU fallback: XNOR directly on host
        if (!weight_buf_.mappedPtr)
            throw std::runtime_error("VSAInferenceEngine: no CPU fallback "
                                     "(weight buffer not host-visible)");

        const uint32_t* weights =
            static_cast<const uint32_t*>(weight_buf_.mappedPtr);

        // Write XNOR result to output buffer
        uint32_t* output = static_cast<uint32_t*>(bufOutput.mappedPtr);
        for (uint32_t w = 0; w < words_per_vec_; ++w) {
            output[w] = ~(input.data[w] ^ weights[w]);
        }
    }

    // Download result
    BitpackedVec result;
    result.dim = state_dim_;
    result.data.resize(words_per_vec_);
    pool_.download(bufOutput, reinterpret_cast<float*>(result.data.data()),
                   stateBytes);

    pool_.release(bufInput);
    pool_.release(bufOutput);

    return result;
}

// ── VSABaremetalEngine ───────────────────────────────────────────────

std::vector<uint32_t> VSABaremetalEngine::loadBinaryFile(
    const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("Failed to open: " + path);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint32_t> buffer(size / sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    return buffer;
}

VSABaremetalEngine::VSABaremetalEngine(BufferPool& pool, uint32_t state_dim,
                                       uint32_t hidden_dim)
    : pool_(pool)
    , state_dim_(state_dim)
    , hidden_dim_(hidden_dim)
    , words_per_vec_((state_dim + 31) / 32)
    , hidden_words_((hidden_dim + 31) / 32)
    , logic_buf_{}
    , codebook_buf_{} {}

VSABaremetalEngine::~VSABaremetalEngine() {
    if (logic_buf_.handle != VK_NULL_HANDLE)
        pool_.release(logic_buf_);
    if (codebook_buf_.handle != VK_NULL_HANDLE)
        pool_.release(codebook_buf_);
}

void VSABaremetalEngine::load_logic_weights(const std::string& filepath) {
    auto host_data = loadBinaryFile(filepath);
    size_t bytes = host_data.size() * sizeof(uint32_t);

    if (logic_buf_.handle != VK_NULL_HANDLE) {
        pool_.release(logic_buf_);
        logic_buf_ = {};
    }

    logic_buf_ = pool_.acquire(bytes);
    pool_.upload(logic_buf_, reinterpret_cast<const float*>(host_data.data()),
                 bytes);

    // w1: hidden_dim neurons × words_per_vec words per neuron
    w1_size_bytes_ = static_cast<size_t>(hidden_dim_) * words_per_vec_ *
                     sizeof(uint32_t);
    logic_buf_bytes_ = bytes;
    logic_loaded_ = true;

    std::cout << "[OK] Logic weights loaded: " << bytes / 1024
              << " KB (w1=" << w1_size_bytes_ / 1024 << " KB, w2="
              << (bytes - w1_size_bytes_) / 1024 << " KB), BDA=0x"
              << std::hex << logic_buf_.deviceAddress
              << std::dec << std::endl;
}

void VSABaremetalEngine::load_codebook(const std::string& codebook_path,
                                        const std::vector<std::string>& vocabulary) {
    auto host_data = loadBinaryFile(codebook_path);
    load_codebook(host_data.data(), host_data.size(), vocabulary);

    std::cout << "[OK] Codebook loaded: "
              << (host_data.size() * sizeof(uint32_t)) / (1024 * 1024)
              << " MB, " << vocab_size_ << " entries" << std::endl;
}

void VSABaremetalEngine::load_codebook(const uint32_t* data, size_t num_words,
                                        const std::vector<std::string>& vocabulary) {
    if (codebook_buf_.handle != VK_NULL_HANDLE) {
        pool_.release(codebook_buf_);
        codebook_buf_ = {};
    }

    size_t bytes = num_words * sizeof(uint32_t);
    codebook_buf_ = pool_.acquire(bytes);
    pool_.upload(codebook_buf_, reinterpret_cast<const float*>(data), bytes);

    vocabulary_ = vocabulary;
    vocab_size_ = static_cast<uint32_t>(vocabulary.size());
    codebook_loaded_ = true;
}

const std::string& VSABaremetalEngine::get_word(uint32_t token_id) const {
    static const std::string unknown = "<UNK>";
    if (token_id < vocabulary_.size())
        return vocabulary_[token_id];
    return unknown;
}

VSABaremetalEngine::StepResult VSABaremetalEngine::step(
    CommandBatch& batch, PipelineCache& pipeCache,
    const BitpackedVec& current_state) {

    if (!ready())
        throw std::runtime_error("VSABaremetalEngine: not ready "
                                 "(load logic weights and codebook first)");

    StepResult result;
    const size_t stateBytes = words_per_vec_ * sizeof(uint32_t);

    // ── STEP 1: TRANSFORMER ────────────────────────────────────────────
    // Execute 2-layer BMM: input -> hidden -> output, then XNOR bind

    GrillyBuffer bufInput = pool_.acquire(stateBytes);
    GrillyBuffer bufOutput = pool_.acquire(stateBytes);

    pool_.upload(bufInput,
                 reinterpret_cast<const float*>(current_state.data.data()),
                 stateBytes);

    // Compute expected weight buffer sizes for safety checks (prevents OOB GPU reads)
    const size_t residual_expected_bytes =
        (size_t(hidden_dim_) * words_per_vec_ +         // W1
         size_t(hidden_dim_) * hidden_words_ +          // W2
         size_t(hidden_dim_) * hidden_words_ +          // W3
         size_t(state_dim_)  * hidden_words_)           // W4
        * sizeof(uint32_t);

    if (pipeCache.hasShader("vsa-bmm-residual") &&
        logic_buf_.deviceAddress != 0 &&
        logic_buf_bytes_ >= residual_expected_bytes) {
        // GPU BDA path: 4-layer residual binary MLP
        PipelineEntry pipe = pipeCache.getOrCreate(
            "vsa-bmm-residual", 2, sizeof(VSABMMResidualParams));

        std::vector<VkDescriptorBufferInfo> bufInfos = {
            {bufInput.handle, 0, stateBytes},
            {bufOutput.handle, 0, stateBytes},
        };
        VkDescriptorSet descSet =
            pipeCache.allocDescriptorSet("vsa-bmm-residual", bufInfos);

        VSABMMResidualParams push{};
        push.weights_ptr = logic_buf_.deviceAddress;
        push.state_words = words_per_vec_;
        push.hidden_words = hidden_words_;
        push.num_layers = 4;
        push._pad = 0;

        batch.begin();
        batch.dispatch(pipe.pipeline, pipe.layout, descSet, 1, 1, 1,
                       &push, sizeof(push));
        batch.submit();
    } else if (pipeCache.hasShader("vsa-bmm") &&
        logic_buf_.deviceAddress != 0) {
        // Fallback: 2-layer binary matrix multiply
        PipelineEntry pipe = pipeCache.getOrCreate(
            "vsa-bmm", 2, sizeof(VSABMMParams));

        std::vector<VkDescriptorBufferInfo> bufInfos = {
            {bufInput.handle, 0, stateBytes},
            {bufOutput.handle, 0, stateBytes},
        };
        VkDescriptorSet descSet =
            pipeCache.allocDescriptorSet("vsa-bmm", bufInfos);

        VSABMMParams push{};
        push.w1_ptr = logic_buf_.deviceAddress;
        push.w2_ptr = logic_buf_.deviceAddress + w1_size_bytes_;
        push.state_words = words_per_vec_;
        push.hidden_words = hidden_words_;

        // Single workgroup of 64 threads
        batch.begin();
        batch.dispatch(pipe.pipeline, pipe.layout, descSet, 1, 1, 1,
                       &push, sizeof(push));
        batch.submit();
    } else if (pipeCache.hasShader("vsa-inference") &&
               logic_buf_.deviceAddress != 0) {
        // Fallback: single-layer XNOR test shader
        PipelineEntry pipe = pipeCache.getOrCreate(
            "vsa-inference", 2, sizeof(VSAInferenceParams));

        std::vector<VkDescriptorBufferInfo> bufInfos = {
            {bufInput.handle, 0, stateBytes},
            {bufOutput.handle, 0, stateBytes},
        };
        VkDescriptorSet descSet =
            pipeCache.allocDescriptorSet("vsa-inference", bufInfos);

        VSAInferenceParams push{};
        push.projection_matrix_ptr = logic_buf_.deviceAddress;
        push.state_dim = state_dim_;
        push._pad = 0;

        uint32_t gx = (words_per_vec_ + 255) / 256;

        batch.begin();
        batch.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                       &push, sizeof(push));
        batch.submit();
    } else {
        // CPU fallback: simple XNOR (single-layer)
        const uint32_t* weights =
            static_cast<const uint32_t*>(logic_buf_.mappedPtr);
        uint32_t* output = static_cast<uint32_t*>(bufOutput.mappedPtr);
        for (uint32_t w = 0; w < words_per_vec_; ++w) {
            output[w] = ~(current_state.data[w] ^ weights[w]);
        }
    }

    // Read back predicted state
    result.predicted_state.dim = state_dim_;
    result.predicted_state.data.resize(words_per_vec_);
    pool_.download(bufOutput,
                   reinterpret_cast<float*>(result.predicted_state.data.data()),
                   stateBytes);

    // ── STEP 2: DECODER (Hamming ArgMin) ─────────────────────────────
    // Find closest codebook entry via atomicMin packed (distance|token_id)

    if (pipeCache.hasShader("vsa-decode") &&
        codebook_buf_.deviceAddress != 0 &&
        bufOutput.deviceAddress != 0) {
        // GPU BDA decode path
        // Allocate output buffer for the single uint64_t atomicMin result
        GrillyBuffer bufResult = pool_.acquire(sizeof(uint64_t));

        // Reset to 0xFFFFFFFFFFFFFFFF so atomicMin works
        uint64_t sentinel = UINT64_MAX;
        pool_.upload(bufResult, reinterpret_cast<const float*>(&sentinel),
                     sizeof(uint64_t));

        PipelineEntry pipe = pipeCache.getOrCreate(
            "vsa-decode", 1, sizeof(VSADecodeParams));

        std::vector<VkDescriptorBufferInfo> decBufInfos = {
            {bufResult.handle, 0, sizeof(uint64_t)},
        };
        VkDescriptorSet decDescSet =
            pipeCache.allocDescriptorSet("vsa-decode", decBufInfos);

        VSADecodeParams decodePush{};
        decodePush.codebook_ptr = codebook_buf_.deviceAddress;
        decodePush.pred_vector_ptr = bufOutput.deviceAddress;
        decodePush.vocab_size = vocab_size_;
        decodePush.uints_per_vec = words_per_vec_;

        uint32_t dgx = (vocab_size_ + 255) / 256;

        batch.begin();
        batch.dispatch(pipe.pipeline, pipe.layout, decDescSet, dgx, 1, 1,
                       &decodePush, sizeof(decodePush));
        batch.submit();

        // Read back the packed result
        uint64_t packed_result = 0;
        pool_.download(bufResult,
                       reinterpret_cast<float*>(&packed_result),
                       sizeof(uint64_t));

        uint32_t token_id = static_cast<uint32_t>(packed_result & 0xFFFFFFFF);
        result.distance = static_cast<uint32_t>(packed_result >> 32);
        result.word = get_word(token_id);

        pool_.release(bufResult);
    } else {
        // CPU fallback: linear Hamming search
        uint32_t best_dist = UINT32_MAX;
        uint32_t best_id = 0;

        const uint32_t* codebook =
            static_cast<const uint32_t*>(codebook_buf_.mappedPtr);

        for (uint32_t t = 0; t < vocab_size_; ++t) {
            uint32_t dist = 0;
            uint32_t offset = t * words_per_vec_;
            for (uint32_t w = 0; w < words_per_vec_; ++w) {
                uint32_t xored = result.predicted_state.data[w] ^
                                 codebook[offset + w];
#ifdef _MSC_VER
                dist += __popcnt(xored);
#else
                dist += __builtin_popcount(xored);
#endif
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_id = t;
            }
        }

        result.distance = best_dist;
        result.word = get_word(best_id);
    }

    pool_.release(bufInput);
    pool_.release(bufOutput);

    return result;
}

std::vector<std::string> VSABaremetalEngine::generate(
    CommandBatch& batch, PipelineCache& pipeCache,
    const BitpackedVec& initial_state,
    uint32_t max_tokens) {

    std::vector<std::string> output;
    BitpackedVec current_state = initial_state;

    for (uint32_t i = 0; i < max_tokens; ++i) {
        auto result = step(batch, pipeCache, current_state);
        output.push_back(result.word);

        if (result.word == "<EOS>")
            break;

        current_state = result.predicted_state;
    }

    return output;
}

}  // namespace cubemind
}  // namespace grilly
