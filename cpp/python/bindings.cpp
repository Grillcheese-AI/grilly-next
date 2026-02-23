#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/device.h"
#include "grilly/op_graph.h"
#include "grilly/ops/activations.h"
#include "grilly/ops/attention.h"
#include "grilly/ops/conv.h"
#include "grilly/ops/kv_cache.h"
#include "grilly/ops/layernorm.h"
#include "grilly/ops/linear.h"
#include "grilly/ops/swizzle.h"
#include "grilly/pipeline_cache.h"
#include "grilly/experimental/paged_latent_pool.h"
#include "grilly/experimental/fused_attention.h"
#include "grilly/cubemind/types.h"
#include "grilly/cubemind/vsa.h"
#include "grilly/cubemind/cube.h"
#include "grilly/cubemind/cache.h"

namespace py = pybind11;

/// Context holds all Vulkan state so Python only sees one object.
/// Internally it owns GrillyDevice -> BufferPool -> PipelineCache -> CommandBatch.
struct GrillyCoreContext {
    grilly::GrillyDevice device;
    grilly::BufferPool pool;
    grilly::PipelineCache cache;
    grilly::CommandBatch batch;
    bool shadersLoaded = false;

    GrillyCoreContext()
        : device(), pool(device), cache(device), batch(device) {}

    /// Load all .spv shaders from a directory into the pipeline cache.
    void loadShaders(const std::string& shaderDir) {
        namespace fs = std::filesystem;
        fs::path dir(shaderDir);

        if (!fs::exists(dir))
            throw std::runtime_error("Shader directory not found: " +
                                     shaderDir);

        int count = 0;
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.path().extension() == ".spv") {
                std::string name = entry.path().stem().string();
                cache.loadSPIRVFile(name, entry.path().string());
                count++;
            }
        }
        shadersLoaded = true;

        // Try to find sibling grilly repo shaders as fallback
        if (count == 0) {
            fs::path grillyShaders =
                fs::path(shaderDir).parent_path().parent_path() / "grilly" /
                "shaders" / "spv";
            if (fs::exists(grillyShaders)) {
                for (const auto& entry :
                     fs::directory_iterator(grillyShaders)) {
                    if (entry.path().extension() == ".spv") {
                        std::string name = entry.path().stem().string();
                        cache.loadSPIRVFile(name, entry.path().string());
                        count++;
                    }
                }
            }
        }
    }
};

/// Persistent GPU cache for Hamming search benchmarking.
/// Bitpacks and uploads cache once; only the query is uploaded per search call.
struct HammingSearchBench {
    GrillyCoreContext* ctx = nullptr;
    std::vector<uint32_t> cachePacked;
    grilly::GrillyBuffer gpuCache{};
    grilly::GrillyBuffer gpuQuery{};   // Persistent query buffer (1.3 KB)
    grilly::GrillyBuffer gpuDist{};    // Persistent distance buffer (N × 4 B)
    VkDescriptorSet descSet = VK_NULL_HANDLE;  // Reused every call
    grilly::PipelineEntry pipe{};              // Cached pipeline
    uint32_t numEntries = 0;
    uint32_t wordsPerVec = 0;
    uint32_t dim = 0;
    bool gpuReady = false;  // True if shader + descriptors are set up

    // Top-1 argmin shader: atomicMin on packed uint64 — reads back 8 bytes
    // Uses SoA-transposed cache for coalesced GPU memory access.
    std::vector<uint32_t> cachePackedSoA;  // Transposed: [word][entry]
    grilly::GrillyBuffer gpuCacheSoA{};    // SoA cache in VRAM
    grilly::GrillyBuffer gpuResult{};      // 8-byte result: (distance<<32)|index
    VkDescriptorSet descSetTop1 = VK_NULL_HANDLE;
    grilly::PipelineEntry pipeTop1{};
    bool top1Ready = false;

    // Subgroup size for dispatch calculations
    uint32_t subgroupSize = 64;    // Default; queried from device
    uint32_t entriesPerWG = 4;     // 256 / subgroupSize

    // GPU timestamp query pool for measuring actual shader execution time
    VkQueryPool tsPool = VK_NULL_HANDLE;
    float timestampPeriod = 0.0f;  // nanoseconds per tick
    double lastGpuMs = 0.0;

    ~HammingSearchBench() {
        if (!ctx) return;
        if (tsPool != VK_NULL_HANDLE)
            vkDestroyQueryPool(ctx->device.device(), tsPool, nullptr);
        auto alloc = ctx->pool.allocator();
        if (gpuCache.handle != VK_NULL_HANDLE)
            vmaDestroyBuffer(alloc, gpuCache.handle, gpuCache.allocation);
        if (gpuQuery.handle != VK_NULL_HANDLE)
            vmaDestroyBuffer(alloc, gpuQuery.handle, gpuQuery.allocation);
        if (gpuDist.handle != VK_NULL_HANDLE)
            vmaDestroyBuffer(alloc, gpuDist.handle, gpuDist.allocation);
        if (gpuResult.handle != VK_NULL_HANDLE)
            vmaDestroyBuffer(alloc, gpuResult.handle, gpuResult.allocation);
        if (gpuCacheSoA.handle != VK_NULL_HANDLE)
            vmaDestroyBuffer(alloc, gpuCacheSoA.handle, gpuCacheSoA.allocation);
    }
};

// ── Helper: extract flat dimensions from numpy array ────────────────────

static std::pair<uint32_t, uint32_t> extractBatchAndLastDim(
    const py::buffer_info& buf) {
    uint32_t lastDim = static_cast<uint32_t>(buf.shape[buf.ndim - 1]);
    uint32_t batch = 1;
    for (int i = 0; i < buf.ndim - 1; ++i)
        batch *= static_cast<uint32_t>(buf.shape[i]);
    if (buf.ndim == 1) batch = 1;
    return {batch, lastDim};
}

PYBIND11_MODULE(grilly_core, m) {
    m.doc() = "grilly C++ Vulkan backend — eliminates Python->C boundary "
              "crossings for GPU dispatch";

    // ── Device info ──
    py::class_<GrillyCoreContext>(m, "Device")
        .def(py::init<>(), "Initialize Vulkan device, buffer pool, and "
                           "pipeline cache")
        .def("load_shaders", &GrillyCoreContext::loadShaders,
             py::arg("shader_dir"),
             "Load all .spv shaders from a directory")
        .def_property_readonly("device_name",
                               [](const GrillyCoreContext& ctx) {
                                   return ctx.device.deviceName();
                               })
        .def_property_readonly("has_cooperative_matrix",
                               [](const GrillyCoreContext& ctx) {
                                   return ctx.device.hasCooperativeMatrix();
                               })
        .def_property_readonly("has_float16",
                               [](const GrillyCoreContext& ctx) {
                                   return ctx.device.hasFloat16();
                               })
        .def("pool_stats",
             [](const GrillyCoreContext& ctx) {
                 auto s = ctx.pool.stats();
                 py::dict d;
                 d["hits"] = s.hits;
                 d["misses"] = s.misses;
                 d["allocations"] = s.allocations;
                 d["total_acquired"] = s.totalAcquired;
                 d["total_released"] = s.totalReleased;
                 return d;
             })
        .def("cache_stats", [](const GrillyCoreContext& ctx) {
            auto s = ctx.cache.cacheStats();
            py::dict d;
            d["hits"] = s.hits;
            d["misses"] = s.misses;
            d["evictions"] = s.evictions;
            d["cached_sets"] = s.cachedSets;
            return d;
        });

    // ═══════════════════════════════════════════════════════════════════════
    // OP GRAPH (batched execution with operator fusion)
    // ═══════════════════════════════════════════════════════════════════════

    py::class_<grilly::OpGraph>(m, "OpGraph")
        .def(py::init<>())
        .def("size", &grilly::OpGraph::size,
             "Number of ops recorded in the graph")
        .def("clear", &grilly::OpGraph::clear,
             "Clear all recorded ops for reuse")
        .def("optimize",
             [](grilly::OpGraph& graph, GrillyCoreContext& ctx) -> py::dict {
                 auto stats = graph.optimize(ctx.cache);
                 py::dict d;
                 d["ops_fused"] = stats.opsFused;
                 d["barriers_eliminated"] = stats.barriersEliminated;
                 d["original_ops"] = stats.originalOps;
                 d["optimized_ops"] = stats.optimizedOps;
                 return d;
             },
             py::arg("device"),
             "Run fusion optimization pass. Returns fusion statistics.")
        .def("execute",
             [](grilly::OpGraph& graph, GrillyCoreContext& ctx) {
                 graph.execute(ctx.batch, ctx.cache);
             },
             py::arg("device"),
             "Execute all recorded ops in a single GPU submission");

    // ═══════════════════════════════════════════════════════════════════════
    // LINEAR
    // ═══════════════════════════════════════════════════════════════════════

    m.def(
        "linear",
        [](GrillyCoreContext& ctx, py::array_t<float> x,
           py::array_t<float> weights,
           std::optional<py::array_t<float>> bias) -> py::array_t<float> {
            auto xBuf = x.request();
            auto wBuf = weights.request();

            if (xBuf.ndim < 1 || xBuf.ndim > 3)
                throw std::runtime_error(
                    "x must be 1D, 2D, or 3D (batch, seq, input_dim)");
            if (wBuf.ndim != 2)
                throw std::runtime_error(
                    "weights must be 2D (output_dim, input_dim)");

            auto [batchSeq, inputDim] = extractBatchAndLastDim(xBuf);
            uint32_t outputDim = static_cast<uint32_t>(wBuf.shape[0]);

            if (static_cast<uint32_t>(wBuf.shape[1]) != inputDim)
                throw std::runtime_error(
                    "Weight input_dim mismatch: " +
                    std::to_string(wBuf.shape[1]) + " vs " +
                    std::to_string(inputDim));

            const float* biasPtr = nullptr;
            uint32_t hasBias = 0;
            if (bias.has_value()) {
                auto bBuf = bias->request();
                if (bBuf.ndim != 1 ||
                    static_cast<uint32_t>(bBuf.shape[0]) != outputDim)
                    throw std::runtime_error("bias must be 1D with size output_dim");
                biasPtr = static_cast<const float*>(bBuf.ptr);
                hasBias = 1;
            }

            grilly::ops::LinearParams p{batchSeq, inputDim, outputDim, hasBias};

            std::vector<py::ssize_t> outShape;
            for (int i = 0; i < xBuf.ndim - 1; ++i)
                outShape.push_back(xBuf.shape[i]);
            if (xBuf.ndim == 1) outShape.push_back(1);
            outShape.push_back(outputDim);

            py::array_t<float> result(outShape);
            auto rBuf = result.request();

            grilly::ops::linear(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(xBuf.ptr),
                static_cast<const float*>(wBuf.ptr), biasPtr,
                static_cast<float*>(rBuf.ptr), p);

            if (xBuf.ndim == 1)
                result = result.reshape({static_cast<py::ssize_t>(outputDim)});

            return result;
        },
        py::arg("device"), py::arg("x"), py::arg("weights"),
        py::arg("bias") = py::none(),
        "GPU linear projection: output = x @ W^T + bias");

    m.def(
        "linear_cpu",
        [](py::array_t<float> x, py::array_t<float> weights,
           std::optional<py::array_t<float>> bias) -> py::array_t<float> {
            auto xBuf = x.request();
            auto wBuf = weights.request();

            auto [batchSeq, inputDim] = extractBatchAndLastDim(xBuf);
            uint32_t outputDim = static_cast<uint32_t>(wBuf.shape[0]);

            const float* biasPtr = nullptr;
            uint32_t hasBias = 0;
            if (bias.has_value()) {
                biasPtr = static_cast<const float*>(bias->request().ptr);
                hasBias = 1;
            }

            grilly::ops::LinearParams p{batchSeq, inputDim, outputDim, hasBias};
            std::vector<float> out = grilly::ops::linearCPU(
                static_cast<const float*>(xBuf.ptr),
                static_cast<const float*>(wBuf.ptr), biasPtr, p);

            std::vector<py::ssize_t> outShape;
            for (int i = 0; i < xBuf.ndim - 1; ++i)
                outShape.push_back(xBuf.shape[i]);
            if (xBuf.ndim == 1) outShape.push_back(1);
            outShape.push_back(outputDim);

            py::array_t<float> result(outShape);
            std::memcpy(result.request().ptr, out.data(),
                        out.size() * sizeof(float));

            if (xBuf.ndim == 1)
                result = result.reshape({static_cast<py::ssize_t>(outputDim)});

            return result;
        },
        py::arg("x"), py::arg("weights"), py::arg("bias") = py::none(),
        "CPU linear projection using Eigen (for verification)");

    // ═══════════════════════════════════════════════════════════════════════
    // ACTIVATIONS
    // ═══════════════════════════════════════════════════════════════════════

    auto defActivation = [&m](const char* name, auto fn) {
        m.def(
            name,
            [fn](GrillyCoreContext& ctx,
                 py::array_t<float> input) -> py::array_t<float> {
                auto inBuf = input.request();
                uint32_t total = 1;
                for (int i = 0; i < inBuf.ndim; ++i)
                    total *= static_cast<uint32_t>(inBuf.shape[i]);

                py::array_t<float> result(input.request().shape);
                auto rBuf = result.request();

                fn(ctx.batch, ctx.pool, ctx.cache,
                   static_cast<const float*>(inBuf.ptr),
                   static_cast<float*>(rBuf.ptr), total);

                return result;
            },
            py::arg("device"), py::arg("input"));
    };

    defActivation("relu", grilly::ops::relu);
    defActivation("gelu", grilly::ops::gelu);
    defActivation("silu", grilly::ops::silu);
    defActivation("tanh_act", grilly::ops::tanh_act);

    // ═══════════════════════════════════════════════════════════════════════
    // LAYERNORM
    // ═══════════════════════════════════════════════════════════════════════

    m.def(
        "layernorm",
        [](GrillyCoreContext& ctx, py::array_t<float> input,
           py::array_t<float> gamma, py::array_t<float> beta,
           float eps) -> py::array_t<float> {
            auto inBuf = input.request();
            auto gBuf = gamma.request();

            if (inBuf.ndim < 2)
                throw std::runtime_error("input must be at least 2D");

            uint32_t features = static_cast<uint32_t>(
                inBuf.shape[inBuf.ndim - 1]);
            uint32_t totalBatch = 1;
            for (int i = 0; i < inBuf.ndim - 1; ++i)
                totalBatch *= static_cast<uint32_t>(inBuf.shape[i]);

            // Treat as (batchSize=1, seqLen=totalBatch, features)
            py::array_t<float> result(inBuf.shape);
            auto rBuf = result.request();

            grilly::ops::layernorm(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(inBuf.ptr),
                static_cast<float*>(rBuf.ptr),
                static_cast<const float*>(gBuf.ptr),
                static_cast<const float*>(beta.request().ptr),
                1, totalBatch, features, eps);

            return result;
        },
        py::arg("device"), py::arg("input"), py::arg("gamma"),
        py::arg("beta"), py::arg("eps") = 1e-5f,
        "GPU LayerNorm: gamma * (x - mean) / sqrt(var + eps) + beta");

    // ═══════════════════════════════════════════════════════════════════════
    // FLASH ATTENTION 2
    // ═══════════════════════════════════════════════════════════════════════

    m.def(
        "flash_attention2",
        [](GrillyCoreContext& ctx,
           py::array_t<float> Q, py::array_t<float> K, py::array_t<float> V,
           std::optional<py::array_t<float>> mask,
           float scale, uint32_t tileSizeQ,
           uint32_t tileSizeK) -> py::array_t<float> {
            auto qBuf = Q.request();

            if (qBuf.ndim != 4)
                throw std::runtime_error(
                    "Q must be 4D (batch, heads, seq_len, head_dim)");

            uint32_t batchSize = static_cast<uint32_t>(qBuf.shape[0]);
            uint32_t numHeads  = static_cast<uint32_t>(qBuf.shape[1]);
            uint32_t seqLen    = static_cast<uint32_t>(qBuf.shape[2]);
            uint32_t headDim   = static_cast<uint32_t>(qBuf.shape[3]);

            const float* maskPtr = nullptr;
            if (mask.has_value()) {
                maskPtr = static_cast<const float*>(mask->request().ptr);
            }

            py::array_t<float> result({
                static_cast<py::ssize_t>(batchSize),
                static_cast<py::ssize_t>(numHeads),
                static_cast<py::ssize_t>(seqLen),
                static_cast<py::ssize_t>(headDim)
            });
            auto rBuf = result.request();

            grilly::ops::flashAttention2(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(qBuf.ptr),
                static_cast<const float*>(K.request().ptr),
                static_cast<const float*>(V.request().ptr),
                maskPtr,
                static_cast<float*>(rBuf.ptr),
                batchSize, seqLen, numHeads, headDim,
                scale, tileSizeQ, tileSizeK);

            return result;
        },
        py::arg("device"), py::arg("Q"), py::arg("K"), py::arg("V"),
        py::arg("mask") = py::none(), py::arg("scale") = 0.0f,
        py::arg("tile_size_q") = 64, py::arg("tile_size_k") = 64,
        "GPU Flash Attention 2 with online softmax tiling");

    // ═══════════════════════════════════════════════════════════════════════
    // CONV2D / CONV1D
    // ═══════════════════════════════════════════════════════════════════════

    m.def(
        "conv2d",
        [](GrillyCoreContext& ctx,
           py::array_t<float> input, py::array_t<float> weight,
           std::optional<py::array_t<float>> bias,
           std::vector<uint32_t> stride, std::vector<uint32_t> padding,
           std::vector<uint32_t> dilation,
           uint32_t groups) -> py::array_t<float> {
            auto inBuf = input.request();
            auto wBuf = weight.request();

            if (inBuf.ndim != 4)
                throw std::runtime_error(
                    "input must be 4D (batch, channels, height, width)");
            if (wBuf.ndim != 4)
                throw std::runtime_error(
                    "weight must be 4D (out_ch, in_ch/groups, kH, kW)");

            uint32_t batchSize  = static_cast<uint32_t>(inBuf.shape[0]);
            uint32_t inChannels = static_cast<uint32_t>(inBuf.shape[1]);
            uint32_t inH        = static_cast<uint32_t>(inBuf.shape[2]);
            uint32_t inW        = static_cast<uint32_t>(inBuf.shape[3]);
            uint32_t outChannels = static_cast<uint32_t>(wBuf.shape[0]);
            uint32_t kH          = static_cast<uint32_t>(wBuf.shape[2]);
            uint32_t kW          = static_cast<uint32_t>(wBuf.shape[3]);

            uint32_t sH = stride.size() >= 1 ? stride[0] : 1;
            uint32_t sW = stride.size() >= 2 ? stride[1] : sH;
            uint32_t pH = padding.size() >= 1 ? padding[0] : 0;
            uint32_t pW = padding.size() >= 2 ? padding[1] : pH;
            uint32_t dH = dilation.size() >= 1 ? dilation[0] : 1;
            uint32_t dW = dilation.size() >= 2 ? dilation[1] : dH;

            uint32_t outH = grilly::ops::convOutputSize(inH, kH, sH, pH, dH);
            uint32_t outW = grilly::ops::convOutputSize(inW, kW, sW, pW, dW);

            const float* biasPtr = nullptr;
            if (bias.has_value())
                biasPtr = static_cast<const float*>(bias->request().ptr);

            py::array_t<float> result({
                static_cast<py::ssize_t>(batchSize),
                static_cast<py::ssize_t>(outChannels),
                static_cast<py::ssize_t>(outH),
                static_cast<py::ssize_t>(outW)
            });
            auto rBuf = result.request();

            grilly::ops::conv2d(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(inBuf.ptr),
                static_cast<const float*>(wBuf.ptr),
                biasPtr,
                static_cast<float*>(rBuf.ptr),
                batchSize, inChannels, inH, inW,
                outChannels, kH, kW,
                sH, sW, pH, pW, dH, dW, groups);

            return result;
        },
        py::arg("device"), py::arg("input"), py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = std::vector<uint32_t>{1, 1},
        py::arg("padding") = std::vector<uint32_t>{0, 0},
        py::arg("dilation") = std::vector<uint32_t>{1, 1},
        py::arg("groups") = 1,
        "GPU Conv2d forward (direct or GEMM path)");

    m.def(
        "conv1d",
        [](GrillyCoreContext& ctx,
           py::array_t<float> input, py::array_t<float> weight,
           std::optional<py::array_t<float>> bias,
           uint32_t stride, uint32_t padding,
           uint32_t dilation, uint32_t groups) -> py::array_t<float> {
            auto inBuf = input.request();
            auto wBuf = weight.request();

            if (inBuf.ndim != 3)
                throw std::runtime_error(
                    "input must be 3D (batch, channels, length)");

            uint32_t batchSize  = static_cast<uint32_t>(inBuf.shape[0]);
            uint32_t inChannels = static_cast<uint32_t>(inBuf.shape[1]);
            uint32_t length     = static_cast<uint32_t>(inBuf.shape[2]);
            uint32_t outChannels = static_cast<uint32_t>(wBuf.shape[0]);
            uint32_t kSize       = static_cast<uint32_t>(wBuf.shape[2]);

            uint32_t outLen = grilly::ops::convOutputSize(
                length, kSize, stride, padding, dilation);

            const float* biasPtr = nullptr;
            if (bias.has_value())
                biasPtr = static_cast<const float*>(bias->request().ptr);

            py::array_t<float> result({
                static_cast<py::ssize_t>(batchSize),
                static_cast<py::ssize_t>(outChannels),
                static_cast<py::ssize_t>(outLen)
            });
            auto rBuf = result.request();

            grilly::ops::conv1d(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(inBuf.ptr),
                static_cast<const float*>(wBuf.ptr),
                biasPtr,
                static_cast<float*>(rBuf.ptr),
                batchSize, inChannels, length,
                outChannels, kSize,
                stride, padding, dilation, groups);

            return result;
        },
        py::arg("device"), py::arg("input"), py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1, py::arg("padding") = 0,
        py::arg("dilation") = 1, py::arg("groups") = 1,
        "GPU Conv1d forward (wrapper around Conv2d)");

    // ═══════════════════════════════════════════════════════════════════════
    // KV CACHE
    // ═══════════════════════════════════════════════════════════════════════

    py::class_<grilly::ops::KVCache>(m, "KVCache")
        .def_readonly("current_len", &grilly::ops::KVCache::currentLen)
        .def("stats", [](const grilly::ops::KVCache& kv) {
            auto s = grilly::ops::kvCacheGetStats(kv);
            py::dict d;
            d["current_tokens"] = s.currentTokens;
            d["max_tokens"] = s.maxTokens;
            d["total_evicted"] = s.totalEvicted;
            d["total_appended"] = s.totalAppended;
            d["compression_ratio"] = s.compressionRatio;
            d["avg_attention_score"] = s.avgAttentionScore;
            return d;
        });

    m.def(
        "create_kv_cache",
        [](GrillyCoreContext& ctx,
           uint32_t maxSeqLen, uint32_t numHeads, uint32_t headDim,
           uint32_t numLayers, uint32_t compressionRatio,
           uint32_t maxCacheTokens, bool useAsymmetricQuant,
           uint32_t valueBits, bool crossLayerSharing,
           bool useH2O, bool useSpeculativeEviction,
           float evictionThreshold) -> grilly::ops::KVCache {
            grilly::ops::KVCacheConfig cfg;
            cfg.maxSeqLen = maxSeqLen;
            cfg.numHeads = numHeads;
            cfg.headDim = headDim;
            cfg.numLayers = numLayers;
            cfg.compressionRatio = compressionRatio;
            cfg.maxCacheTokens = maxCacheTokens;
            cfg.useAsymmetricQuant = useAsymmetricQuant;
            cfg.valueBits = valueBits;
            cfg.crossLayerSharing = crossLayerSharing;
            cfg.useH2O = useH2O;
            cfg.useSpeculativeEviction = useSpeculativeEviction;
            cfg.evictionThreshold = evictionThreshold;

            return grilly::ops::createKVCache(ctx.pool, cfg);
        },
        py::arg("device"),
        py::arg("max_seq_len") = 2048,
        py::arg("num_heads") = 8,
        py::arg("head_dim") = 64,
        py::arg("num_layers") = 12,
        py::arg("compression_ratio") = 4,
        py::arg("max_cache_tokens") = 2048,
        py::arg("use_asymmetric_quant") = false,
        py::arg("value_bits") = 4,
        py::arg("cross_layer_sharing") = false,
        py::arg("use_h2o") = true,
        py::arg("use_speculative_eviction") = false,
        py::arg("eviction_threshold") = 0.1f,
        "Create a KV cache with MLA compression and H2O eviction");

    m.def(
        "kv_cache_append",
        [](GrillyCoreContext& ctx, grilly::ops::KVCache& kvCache,
           py::array_t<float> newKeys,
           py::array_t<float> newValues) {
            auto kBuf = newKeys.request();
            uint32_t numNew = static_cast<uint32_t>(kBuf.shape[0]);

            grilly::ops::kvCacheAppend(
                ctx.batch, ctx.pool, ctx.cache, kvCache,
                static_cast<const float*>(kBuf.ptr),
                static_cast<const float*>(newValues.request().ptr),
                numNew);
        },
        py::arg("device"), py::arg("kv_cache"),
        py::arg("new_keys"), py::arg("new_values"),
        "Append new KV pairs to cache (with MLA compression)");

    m.def(
        "kv_cache_decode",
        [](GrillyCoreContext& ctx,
           const grilly::ops::KVCache& kvCache) -> py::dict {
            const auto& cfg = kvCache.config;
            uint32_t tokens = kvCache.currentLen;

            size_t kvSize = size_t(tokens) * cfg.numHeads * cfg.headDim;
            py::array_t<float> keys({
                static_cast<py::ssize_t>(tokens),
                static_cast<py::ssize_t>(cfg.numHeads),
                static_cast<py::ssize_t>(cfg.headDim)
            });
            py::array_t<float> values({
                static_cast<py::ssize_t>(tokens),
                static_cast<py::ssize_t>(cfg.numHeads),
                static_cast<py::ssize_t>(cfg.headDim)
            });

            grilly::ops::kvCacheDecode(
                ctx.batch, ctx.pool, ctx.cache, kvCache,
                static_cast<float*>(keys.request().ptr),
                static_cast<float*>(values.request().ptr));

            py::dict result;
            result["keys"] = keys;
            result["values"] = values;
            return result;
        },
        py::arg("device"), py::arg("kv_cache"),
        "Decode KV from compressed cache");

    m.def(
        "kv_cache_evict_h2o",
        [](GrillyCoreContext& ctx, grilly::ops::KVCache& kvCache,
           std::optional<py::array_t<float>> attentionScores,
           uint32_t numEvict) {
            const float* scoresPtr = nullptr;
            if (attentionScores.has_value())
                scoresPtr = static_cast<const float*>(
                    attentionScores->request().ptr);

            grilly::ops::kvCacheEvictH2O(
                ctx.batch, ctx.pool, ctx.cache, kvCache,
                scoresPtr, numEvict);
        },
        py::arg("device"), py::arg("kv_cache"),
        py::arg("attention_scores") = py::none(),
        py::arg("num_evict") = 0,
        "Run H2O eviction on KV cache");

    m.def(
        "kv_cache_compact",
        [](GrillyCoreContext& ctx, grilly::ops::KVCache& kvCache) {
            grilly::ops::kvCacheCompact(
                ctx.batch, ctx.pool, ctx.cache, kvCache);
        },
        py::arg("device"), py::arg("kv_cache"),
        "Compact KV cache after eviction");

    m.def(
        "destroy_kv_cache",
        [](GrillyCoreContext& ctx, grilly::ops::KVCache& kvCache) {
            grilly::ops::destroyKVCache(ctx.pool, kvCache);
        },
        py::arg("device"), py::arg("kv_cache"),
        "Destroy KV cache and release GPU buffers");

    m.def(
        "kv_cache_init_eviction_head",
        [](GrillyCoreContext& ctx, grilly::ops::KVCache& kvCache,
           uint32_t inputDim, uint32_t hiddenDim, float lr) {
            grilly::ops::kvCacheInitEvictionHead(ctx.pool, kvCache,
                                                  inputDim, hiddenDim, lr);
        },
        py::arg("device"), py::arg("kv_cache"),
        py::arg("input_dim"), py::arg("hidden_dim") = 32,
        py::arg("lr") = 1e-3f,
        "Initialize trainable eviction head for speculative eviction");

    m.def(
        "kv_cache_train_eviction_head",
        [](GrillyCoreContext& ctx, grilly::ops::KVCache& kvCache,
           py::array_t<float> tokenFeatures,
           py::array_t<float> attentionScores,
           uint32_t seqLen) {
            grilly::ops::kvCacheTrainEvictionHead(
                ctx.batch, ctx.pool, ctx.cache, kvCache,
                static_cast<const float*>(tokenFeatures.request().ptr),
                static_cast<const float*>(attentionScores.request().ptr),
                seqLen);
        },
        py::arg("device"), py::arg("kv_cache"),
        py::arg("token_features"), py::arg("attention_scores"),
        py::arg("seq_len"),
        "Train the eviction head on attention patterns from latest forward pass");

    m.def(
        "kv_cache_evict_speculative",
        [](GrillyCoreContext& ctx, grilly::ops::KVCache& kvCache,
           std::optional<py::array_t<float>> hiddenStates,
           uint32_t hiddenDim) {
            const float* hsPtr = nullptr;
            if (hiddenStates.has_value())
                hsPtr = static_cast<const float*>(
                    hiddenStates->request().ptr);

            grilly::ops::kvCacheEvictSpeculative(
                ctx.batch, ctx.pool, ctx.cache, kvCache,
                hsPtr, hiddenDim);
        },
        py::arg("device"), py::arg("kv_cache"),
        py::arg("hidden_states") = py::none(),
        py::arg("hidden_dim") = 64,
        "Run speculative eviction using trained auxiliary head");

    // ═══════════════════════════════════════════════════════════════════════
    // SUBGROUP-SWIZZLED BLOCK CACHING
    // ═══════════════════════════════════════════════════════════════════════

    m.def(
        "swizzle_kv",
        [](GrillyCoreContext& ctx, py::array_t<float> input,
           uint32_t waveSize, bool reverse) -> py::array_t<float> {
            auto inBuf = input.request();

            if (inBuf.ndim != 4)
                throw std::runtime_error(
                    "input must be 4D (batch, heads, seq_len, head_dim)");

            uint32_t batchSize = static_cast<uint32_t>(inBuf.shape[0]);
            uint32_t numHeads  = static_cast<uint32_t>(inBuf.shape[1]);
            uint32_t seqLen    = static_cast<uint32_t>(inBuf.shape[2]);
            uint32_t headDim   = static_cast<uint32_t>(inBuf.shape[3]);

            // Swizzled output may be padded to wave boundaries
            uint32_t seqPadded = ((seqLen + waveSize - 1) / waveSize) * waveSize;
            uint32_t dimPadded = ((headDim + waveSize - 1) / waveSize) * waveSize;

            py::array_t<float> result;
            if (reverse) {
                // Unswizzle: output is standard shape
                result = py::array_t<float>({
                    static_cast<py::ssize_t>(batchSize),
                    static_cast<py::ssize_t>(numHeads),
                    static_cast<py::ssize_t>(seqLen),
                    static_cast<py::ssize_t>(headDim)
                });
            } else {
                // Swizzle: output is padded flat buffer
                size_t outSize = grilly::ops::swizzledBufferSize(
                    batchSize, numHeads, seqLen, headDim, waveSize);
                result = py::array_t<float>(outSize / sizeof(float));
            }
            auto rBuf = result.request();

            grilly::ops::swizzle(
                ctx.batch, ctx.pool, ctx.cache,
                static_cast<const float*>(inBuf.ptr),
                static_cast<float*>(rBuf.ptr),
                batchSize, numHeads, seqLen, headDim,
                waveSize, reverse);

            return result;
        },
        py::arg("device"), py::arg("input"),
        py::arg("wave_size") = 32, py::arg("reverse") = false,
        "Swizzle/unswizzle KV tensor for AMD Wave32 memory channel alignment.\n"
        "Transforms [B,H,S,D] -> [B,S/W,D/W,W,W] for optimal bandwidth.");

    // ═══════════════════════════════════════════════════════════════════════
    // EXPERIMENTAL: FUSED SUBGROUP-DECOMPRESS + FLASH ATTENTION
    // ═══════════════════════════════════════════════════════════════════════

    m.def(
        "fused_attention_cpu",
        [](py::array_t<float> Q, py::array_t<float> latents,
           py::array_t<float> wUp,
           std::optional<py::array_t<float>> mask,
           uint32_t cachedTokens, uint32_t numHeads,
           uint32_t headDim, uint32_t latentDim,
           float scale) -> py::array_t<float> {
            auto qBuf = Q.request();

            if (qBuf.ndim < 2)
                throw std::runtime_error("Q must be at least 2D");

            // Infer batchSize and seqLen from Q shape
            // Q is (batchSize, numHeads, seqLen, headDim) or flat
            uint32_t batchSize = 1;
            uint32_t seqLen = 1;
            if (qBuf.ndim == 4) {
                batchSize = static_cast<uint32_t>(qBuf.shape[0]);
                seqLen = static_cast<uint32_t>(qBuf.shape[2]);
            } else if (qBuf.ndim == 3) {
                batchSize = static_cast<uint32_t>(qBuf.shape[0]);
                seqLen = static_cast<uint32_t>(qBuf.shape[1]);
            } else {
                // 2D: (seqLen, headDim)
                seqLen = static_cast<uint32_t>(qBuf.shape[0]);
            }

            const float* maskPtr = nullptr;
            if (mask.has_value())
                maskPtr = static_cast<const float*>(mask->request().ptr);

            size_t outSize = size_t(batchSize) * numHeads * seqLen * headDim;
            py::array_t<float> result(outSize);
            auto rBuf = result.request();

            grilly::experimental::fusedAttentionCPU(
                static_cast<const float*>(qBuf.ptr),
                static_cast<const float*>(latents.request().ptr),
                static_cast<const float*>(wUp.request().ptr),
                maskPtr,
                static_cast<float*>(rBuf.ptr),
                batchSize, seqLen, cachedTokens,
                numHeads, headDim, latentDim, scale);

            // Reshape to (batchSize, numHeads, seqLen, headDim)
            result = result.reshape({
                static_cast<py::ssize_t>(batchSize),
                static_cast<py::ssize_t>(numHeads),
                static_cast<py::ssize_t>(seqLen),
                static_cast<py::ssize_t>(headDim)
            });

            return result;
        },
        py::arg("Q"), py::arg("latents"), py::arg("w_up"),
        py::arg("mask") = py::none(),
        py::arg("cached_tokens") = 0,
        py::arg("num_heads") = 8, py::arg("head_dim") = 64,
        py::arg("latent_dim") = 16, py::arg("scale") = 0.0f,
        "CPU reference for fused MLA decompression + attention.\n"
        "Decompresses latents via W_up matmul, then computes standard attention.");

    // ═══════════════════════════════════════════════════════════════════════
    // CUBEMIND: VSA ENCODING
    // ═══════════════════════════════════════════════════════════════════════

    m.def(
        "blake3_role",
        [](const std::string& key, uint32_t dim,
           const std::string& domain) -> py::array_t<int8_t> {
            auto result = grilly::cubemind::blake3Role(key, dim, domain);
            py::array_t<int8_t> arr(dim);
            std::memcpy(arr.request().ptr, result.data(),
                        dim * sizeof(int8_t));
            return arr;
        },
        py::arg("key"), py::arg("dim"),
        py::arg("domain") = "grilly.cubemind",
        "Generate a deterministic bipolar {-1,+1} role vector via BLAKE3");

    m.def(
        "vsa_bind",
        [](py::array_t<int8_t> a, py::array_t<int8_t> b) -> py::array_t<int8_t> {
            auto aBuf = a.request();
            auto bBuf = b.request();
            uint32_t dim = static_cast<uint32_t>(aBuf.shape[0]);
            if (static_cast<uint32_t>(bBuf.shape[0]) != dim)
                throw std::runtime_error("VSA bind: dimension mismatch");

            auto result = grilly::cubemind::vsaBind(
                static_cast<const int8_t*>(aBuf.ptr),
                static_cast<const int8_t*>(bBuf.ptr), dim);

            py::array_t<int8_t> arr(dim);
            std::memcpy(arr.request().ptr, result.data(),
                        dim * sizeof(int8_t));
            return arr;
        },
        py::arg("a"), py::arg("b"),
        "Bipolar binding: element-wise multiply (self-inverse)");

    m.def(
        "vsa_bundle",
        [](std::vector<py::array_t<int8_t>> vectors) -> py::array_t<int8_t> {
            if (vectors.empty())
                throw std::runtime_error("VSA bundle: empty vector list");

            uint32_t dim = static_cast<uint32_t>(vectors[0].request().shape[0]);
            std::vector<const int8_t*> ptrs;
            ptrs.reserve(vectors.size());
            for (auto& v : vectors)
                ptrs.push_back(static_cast<const int8_t*>(v.request().ptr));

            auto result = grilly::cubemind::vsaBundle(ptrs, dim);

            py::array_t<int8_t> arr(dim);
            std::memcpy(arr.request().ptr, result.data(),
                        dim * sizeof(int8_t));
            return arr;
        },
        py::arg("vectors"),
        "Bipolar bundling: majority vote superposition");

    m.def(
        "vsa_bitpack",
        [](py::array_t<int8_t> bipolar) -> py::array_t<uint32_t> {
            auto buf = bipolar.request();
            uint32_t dim = static_cast<uint32_t>(buf.shape[0]);
            auto packed = grilly::cubemind::vsaBitpack(
                static_cast<const int8_t*>(buf.ptr), dim);

            py::array_t<uint32_t> arr(packed.numWords());
            std::memcpy(arr.request().ptr, packed.data.data(),
                        packed.numWords() * sizeof(uint32_t));
            return arr;
        },
        py::arg("bipolar"),
        "Bitpack bipolar {-1,+1} int8 to packed uint32 bits");

    m.def(
        "vsa_encode",
        [](std::vector<std::string> roles,
           std::vector<py::array_t<int8_t>> fillers,
           uint32_t dim) -> py::array_t<uint32_t> {
            if (roles.size() != fillers.size())
                throw std::runtime_error("VSA encode: roles/fillers length mismatch");

            std::vector<const int8_t*> fillerPtrs;
            fillerPtrs.reserve(fillers.size());
            for (auto& f : fillers)
                fillerPtrs.push_back(static_cast<const int8_t*>(f.request().ptr));

            auto packed = grilly::cubemind::vsaEncode(roles, fillerPtrs, dim);

            py::array_t<uint32_t> arr(packed.numWords());
            std::memcpy(arr.request().ptr, packed.data.data(),
                        packed.numWords() * sizeof(uint32_t));
            return arr;
        },
        py::arg("roles"), py::arg("fillers"), py::arg("dim"),
        "Full VSA encode pipeline: BLAKE3 roles + bind + bundle + bitpack");

    // ═══════════════════════════════════════════════════════════════════════
    // CUBEMIND: HAMMING SEARCH
    // ═══════════════════════════════════════════════════════════════════════

    m.def(
        "hamming_search",
        [](GrillyCoreContext& ctx,
           py::array_t<int8_t> query,
           py::array_t<int8_t> cache_data) -> py::array_t<uint32_t> {
            // Accept bipolar int8 arrays, bitpack internally
            auto qBuf = query.request();
            auto cBuf = cache_data.request();

            uint32_t dim = static_cast<uint32_t>(qBuf.shape[0]);
            uint32_t numEntries;
            if (cBuf.ndim == 2) {
                numEntries = static_cast<uint32_t>(cBuf.shape[0]);
                if (static_cast<uint32_t>(cBuf.shape[1]) != dim)
                    throw std::runtime_error("Cache dim mismatch with query");
            } else {
                numEntries = static_cast<uint32_t>(cBuf.size) / dim;
            }

            // Bitpack query
            auto queryPacked = grilly::cubemind::vsaBitpack(
                static_cast<const int8_t*>(qBuf.ptr), dim);

            // Bitpack cache
            uint32_t wordsPerVec = queryPacked.numWords();
            std::vector<uint32_t> cachePacked(size_t(numEntries) * wordsPerVec);
            for (uint32_t i = 0; i < numEntries; ++i) {
                auto packed = grilly::cubemind::vsaBitpack(
                    static_cast<const int8_t*>(cBuf.ptr) + i * dim, dim);
                std::memcpy(cachePacked.data() + size_t(i) * wordsPerVec,
                            packed.data.data(),
                            wordsPerVec * sizeof(uint32_t));
            }

            // GPU Hamming search
            py::array_t<uint32_t> result(numEntries);
            auto rBuf = result.request();

            grilly::cubemind::hammingSearch(
                ctx.batch, ctx.pool, ctx.cache,
                queryPacked.data.data(), cachePacked.data(),
                static_cast<uint32_t*>(rBuf.ptr),
                numEntries, wordsPerVec);

            return result;
        },
        py::arg("device"), py::arg("query"), py::arg("cache"),
        "GPU Hamming search: distances from query to all cache entries.\n"
        "Accepts bipolar int8 arrays, bitpacks internally.");

    m.def(
        "hamming_search_cpu",
        [](py::array_t<int8_t> query,
           py::array_t<int8_t> cache_data) -> py::array_t<uint32_t> {
            auto qBuf = query.request();
            auto cBuf = cache_data.request();

            uint32_t dim = static_cast<uint32_t>(qBuf.shape[0]);
            uint32_t numEntries;
            if (cBuf.ndim == 2) {
                numEntries = static_cast<uint32_t>(cBuf.shape[0]);
            } else {
                numEntries = static_cast<uint32_t>(cBuf.size) / dim;
            }

            auto queryPacked = grilly::cubemind::vsaBitpack(
                static_cast<const int8_t*>(qBuf.ptr), dim);

            uint32_t wordsPerVec = queryPacked.numWords();
            std::vector<uint32_t> cachePacked(size_t(numEntries) * wordsPerVec);
            for (uint32_t i = 0; i < numEntries; ++i) {
                auto packed = grilly::cubemind::vsaBitpack(
                    static_cast<const int8_t*>(cBuf.ptr) + i * dim, dim);
                std::memcpy(cachePacked.data() + size_t(i) * wordsPerVec,
                            packed.data.data(),
                            wordsPerVec * sizeof(uint32_t));
            }

            auto distances = grilly::cubemind::hammingSearchCPU(
                queryPacked.data.data(), cachePacked.data(),
                numEntries, wordsPerVec);

            py::array_t<uint32_t> result(numEntries);
            std::memcpy(result.request().ptr, distances.data(),
                        numEntries * sizeof(uint32_t));
            return result;
        },
        py::arg("query"), py::arg("cache"),
        "CPU reference for Hamming search (for verification)");

    // ═══════════════════════════════════════════════════════════════════════
    // CUBEMIND: PERSISTENT GPU HAMMING SEARCH (for benchmarking)
    // ═══════════════════════════════════════════════════════════════════════
    //
    // HammingSearchBench: bitpack cache once, upload once, search many times.
    // This avoids the per-call PCIe transfer of the entire cache that was
    // making the naive hamming_search() 1000x slower than it should be.

    py::class_<HammingSearchBench>(m, "HammingSearchBench")
        .def(py::init([](GrillyCoreContext& ctx,
                         py::array_t<int8_t> cache_data,
                         uint32_t dim) {
                 auto bench = std::make_unique<HammingSearchBench>();
                 bench->ctx = &ctx;
                 bench->dim = dim;

                 auto cBuf = cache_data.request();
                 bench->numEntries = (cBuf.ndim == 2)
                     ? static_cast<uint32_t>(cBuf.shape[0])
                     : static_cast<uint32_t>(cBuf.size) / dim;

                 // Bitpack cache once on CPU
                 auto probe = grilly::cubemind::vsaBitpack(
                     static_cast<const int8_t*>(cBuf.ptr), dim);
                 bench->wordsPerVec = probe.numWords();

                 bench->cachePacked.resize(
                     size_t(bench->numEntries) * bench->wordsPerVec);
                 std::memcpy(bench->cachePacked.data(),
                             probe.data.data(),
                             bench->wordsPerVec * sizeof(uint32_t));

                 for (uint32_t i = 1; i < bench->numEntries; ++i) {
                     auto packed = grilly::cubemind::vsaBitpack(
                         static_cast<const int8_t*>(cBuf.ptr) + i * dim, dim);
                     std::memcpy(
                         bench->cachePacked.data() + size_t(i) * bench->wordsPerVec,
                         packed.data.data(),
                         bench->wordsPerVec * sizeof(uint32_t));
                 }

                 // ── Pre-allocate ALL GPU resources once ──────────────
                 size_t cacheBytes = size_t(bench->numEntries) *
                                     bench->wordsPerVec * sizeof(uint32_t);
                 size_t queryBytes = size_t(bench->wordsPerVec) * sizeof(uint32_t);
                 size_t distBytes  = size_t(bench->numEntries) * sizeof(uint32_t);

                 // 1. Cache buffer in VRAM — uploaded once via staging, never touched again.
                 //    GPU reads at 288 GB/s (VRAM) instead of 14 GB/s (PCIe).
                 //    At 490K entries: 627 MB → 2.2ms from VRAM vs 45ms from system RAM.
                 bench->gpuCache = ctx.pool.acquireDeviceLocal(cacheBytes);
                 ctx.pool.uploadStaged(bench->gpuCache,
                                       bench->cachePacked.data(),
                                       cacheBytes);

                 // ── VRAM placement diagnostic ──────────────────────
                 {
                     VkMemoryPropertyFlags memFlags;
                     vmaGetMemoryTypeProperties(
                         ctx.pool.allocator(),
                         bench->gpuCache.info.memoryType,
                         &memFlags);
                     bool devLocal = (memFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
                     bool hostVis  = (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
                     std::cout << "[VRAM] Cache: memType="
                               << bench->gpuCache.info.memoryType
                               << " DEVICE_LOCAL=" << devLocal
                               << " HOST_VISIBLE=" << hostVis
                               << " size=" << (cacheBytes / (1024*1024))
                               << " MB" << std::endl;
                     if (!devLocal) {
                         std::cerr << "[WARN] Cache NOT in VRAM!" << std::endl;
                     }
                 }

                 // 2. Query buffer — write-only from CPU (WC is fine)
                 bench->gpuQuery = ctx.pool.acquire(queryBytes);
                 // 3. Distance buffer — READ from CPU, must be cached RAM!
                 //    WC memory reads at ~39 MB/s. Cached reads at ~10 GB/s.
                 bench->gpuDist  = ctx.pool.acquireReadback(distBytes);

                 // Diagnostic: compare memory types
                 {
                     VkMemoryPropertyFlags qFlags, dFlags;
                     vmaGetMemoryTypeProperties(ctx.pool.allocator(),
                         bench->gpuQuery.info.memoryType, &qFlags);
                     vmaGetMemoryTypeProperties(ctx.pool.allocator(),
                         bench->gpuDist.info.memoryType, &dFlags);
                     std::cout << "[MEM] Query: type=" << bench->gpuQuery.info.memoryType
                               << " HOST_CACHED=" << ((qFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0)
                               << std::endl;
                     std::cout << "[MEM] Dist:  type=" << bench->gpuDist.info.memoryType
                               << " HOST_CACHED=" << ((dFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0)
                               << std::endl;
                 }

                 // 3. Pipeline + descriptor set — allocated once, reused every call
                 if (ctx.cache.hasShader("hamming-search")) {
                     bench->pipe = ctx.cache.getOrCreate(
                         "hamming-search", 3,
                         sizeof(grilly::cubemind::HammingSearchParams));

                     std::vector<VkDescriptorBufferInfo> bufInfos = {
                         {bench->gpuQuery.handle, 0, queryBytes},
                         {bench->gpuCache.handle, 0, cacheBytes},
                         {bench->gpuDist.handle,  0, distBytes},
                     };
                     bench->descSet = ctx.cache.allocDescriptorSet(
                         "hamming-search", bufInfos);
                     bench->gpuReady = true;
                 }

                 // ── Top-1 argmin shader: atomicMin on packed uint64 ──
                 // Uses SoA-transposed cache for coalesced GPU memory access.
                 // AoS: [entry][word] — strided reads, ~12% BW utilization
                 // SoA: [word][entry] — coalesced reads, ~80% BW utilization
                 if (ctx.cache.hasShader("hamming-top1")) {
                     bench->pipeTop1 = ctx.cache.getOrCreate(
                         "hamming-top1", 3,  // 3 bindings: query, cache, result
                         sizeof(grilly::cubemind::HammingSearchParams));

                     // Transpose cache: AoS → SoA
                     size_t N = bench->numEntries;
                     size_t W = bench->wordsPerVec;
                     bench->cachePackedSoA.resize(N * W);
                     for (size_t w = 0; w < W; w++) {
                         for (size_t e = 0; e < N; e++) {
                             bench->cachePackedSoA[w * N + e] =
                                 bench->cachePacked[e * W + w];
                         }
                     }

                     // Upload SoA cache to VRAM
                     bench->gpuCacheSoA = ctx.pool.acquireDeviceLocal(cacheBytes);
                     ctx.pool.uploadStaged(bench->gpuCacheSoA,
                                           bench->cachePackedSoA.data(),
                                           cacheBytes);

                     // 8-byte result buffer — host-visible for readback
                     bench->gpuResult = ctx.pool.acquire(sizeof(uint64_t));

                     std::vector<VkDescriptorBufferInfo> top1BufInfos = {
                         {bench->gpuQuery.handle, 0, queryBytes},
                         {bench->gpuCache.handle, 0, cacheBytes},  // AoS (subgroup-strided)
                         {bench->gpuResult.handle, 0, sizeof(uint64_t)},
                     };
                     bench->descSetTop1 = ctx.cache.allocDescriptorSet(
                         "hamming-top1", top1BufInfos);
                     bench->top1Ready = true;
                     std::cout << "[OK] hamming-top1 shader loaded (SoA + atomicMin)"
                               << std::endl;
                 }

                 // Query subgroup size for dispatch calculations
                 {
                     VkPhysicalDeviceSubgroupProperties sgProps{};
                     sgProps.sType =
                         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
                     VkPhysicalDeviceProperties2 props2{};
                     props2.sType =
                         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
                     props2.pNext = &sgProps;
                     vkGetPhysicalDeviceProperties2(
                         ctx.device.physicalDevice(), &props2);
                     bench->subgroupSize = sgProps.subgroupSize;
                     bench->entriesPerWG = 256 / bench->subgroupSize;
                     std::cout << "[OK] Subgroup size: "
                               << bench->subgroupSize
                               << " (entries/WG=" << bench->entriesPerWG
                               << ")" << std::endl;
                 }

                 // Create GPU timestamp query pool (2 timestamps: before + after dispatch)
                 {
                     VkPhysicalDeviceProperties props;
                     vkGetPhysicalDeviceProperties(ctx.device.physicalDevice(), &props);
                     bench->timestampPeriod = props.limits.timestampPeriod;

                     VkQueryPoolCreateInfo qpci{};
                     qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
                     qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
                     qpci.queryCount = 2;
                     vkCreateQueryPool(ctx.device.device(), &qpci, nullptr,
                                       &bench->tsPool);
                     std::cout << "[OK] GPU timestamps: period="
                               << bench->timestampPeriod << "ns" << std::endl;
                 }

                 return bench;
             }),
             py::arg("device"), py::arg("cache"), py::arg("dim") = 10240,
             py::keep_alive<1, 2>())  // Keep GrillyCoreContext alive while bench exists
        .def("search",
             [](HammingSearchBench& bench,
                py::array_t<int8_t> query) -> py::array_t<uint32_t> {
                 auto qBuf = query.request();
                 auto queryPacked = grilly::cubemind::vsaBitpack(
                     static_cast<const int8_t*>(qBuf.ptr), bench.dim);

                 py::array_t<uint32_t> result(bench.numEntries);
                 auto rBuf = result.request();

                 if (bench.gpuReady) {
                     // ── SINGLE-SUBMISSION HOT PATH ─────────────────────
                     // Query + distances in host-visible (memcpy, no staging).
                     // Cache in DEVICE_LOCAL VRAM. One GPU round-trip.
                     const size_t queryBytes = size_t(bench.wordsPerVec) *
                                               sizeof(uint32_t);
                     const size_t distBytes  = size_t(bench.numEntries) *
                                               sizeof(uint32_t);

                     // 1. Upload query via memcpy (1.3 KB, no GPU submit)
                     bench.ctx->pool.upload(
                         bench.gpuQuery,
                         reinterpret_cast<const float*>(queryPacked.data.data()),
                         queryBytes);

                     // 2. Dispatch compute shader
                     // Scalar: one thread per cache entry, workgroup = 256
                     grilly::cubemind::HammingSearchParams push{
                         bench.numEntries, bench.wordsPerVec, 1, 0};
                     uint32_t gx = (bench.numEntries + 255) / 256;

                     bench.ctx->batch.begin();
                     auto cmd = bench.ctx->batch.cmdBuffer();
                     // GPU timestamps: measure actual shader execution
                     vkCmdResetQueryPool(cmd, bench.tsPool, 0, 2);
                     vkCmdWriteTimestamp(cmd,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         bench.tsPool, 0);
                     bench.ctx->batch.dispatch(
                         bench.pipe.pipeline, bench.pipe.layout,
                         bench.descSet, gx, 1, 1,
                         &push, sizeof(push));
                     vkCmdWriteTimestamp(cmd,
                         VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         bench.tsPool, 1);
                     bench.ctx->batch.submit();

                     // Read GPU timestamps
                     uint64_t ts[2] = {0, 0};
                     VkResult tsResult = vkGetQueryPoolResults(
                         bench.ctx->device.device(), bench.tsPool,
                         0, 2, sizeof(ts), ts, sizeof(uint64_t),
                         VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
                     if (bench.lastGpuMs == 0.0) {
                         // First call: print raw values for debugging
                         double diffNs = double(ts[1] - ts[0]) * double(bench.timestampPeriod);
                         std::cout << "[TS] t0=" << ts[0]
                                   << " t1=" << ts[1]
                                   << " diff_ticks=" << (ts[1] - ts[0])
                                   << " diff_ms=" << (diffNs / 1e6)
                                   << " vkResult=" << tsResult
                                   << std::endl;
                     }
                     bench.lastGpuMs = double(ts[1] - ts[0]) *
                                       double(bench.timestampPeriod) / 1e6;

                     // 3. Download distances via memcpy (no staging)
                     bench.ctx->pool.download(
                         bench.gpuDist,
                         reinterpret_cast<float*>(
                             static_cast<uint32_t*>(rBuf.ptr)),
                         distBytes);
                 } else {
                     // CPU fallback
                     auto distances = grilly::cubemind::hammingSearchCPU(
                         queryPacked.data.data(),
                         bench.cachePacked.data(),
                         bench.numEntries, bench.wordsPerVec);
                     std::memcpy(rBuf.ptr, distances.data(),
                                 bench.numEntries * sizeof(uint32_t));
                 }

                 return result;
             },
             py::arg("query"),
             "Zero-alloc GPU Hamming search (pre-built descriptor set)")
        .def("search_top1",
             [](HammingSearchBench& bench,
                py::array_t<int8_t> query) -> py::dict {
                 if (!bench.top1Ready)
                     throw std::runtime_error(
                         "hamming-top1 shader not loaded");

                 auto qBuf = query.request();
                 auto queryPacked = grilly::cubemind::vsaBitpack(
                     static_cast<const int8_t*>(qBuf.ptr), bench.dim);

                 const size_t queryBytes = size_t(bench.wordsPerVec) *
                                           sizeof(uint32_t);

                 // 1. Upload query via memcpy (1.3 KB)
                 bench.ctx->pool.upload(
                     bench.gpuQuery,
                     reinterpret_cast<const float*>(queryPacked.data.data()),
                     queryBytes);

                 // 2. Reset result to max uint64 so atomicMin can find minimum.
                 // 0xFFFFFFFFFFFFFFFF = max distance (0xFFFFFFFF) + max index.
                 uint64_t sentinel = 0xFFFFFFFFFFFFFFFFULL;
                 std::memcpy(bench.gpuResult.mappedPtr, &sentinel,
                             sizeof(uint64_t));

                 // 3. Dispatch top-1 argmin shader
                 // entries_per_wg = 256/subgroupSize (4 for Wave64, 8 for Wave32)
                 grilly::cubemind::HammingSearchParams push{
                     bench.numEntries, bench.wordsPerVec,
                     bench.entriesPerWG, 0};
                 uint32_t gx = (bench.numEntries + bench.entriesPerWG - 1)
                             / bench.entriesPerWG;

                 bench.ctx->batch.begin();
                 auto cmd = bench.ctx->batch.cmdBuffer();
                 vkCmdResetQueryPool(cmd, bench.tsPool, 0, 2);
                 vkCmdWriteTimestamp(cmd,
                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                     bench.tsPool, 0);
                 bench.ctx->batch.dispatch(
                     bench.pipeTop1.pipeline, bench.pipeTop1.layout,
                     bench.descSetTop1, gx, 1, 1,
                     &push, sizeof(push));
                 vkCmdWriteTimestamp(cmd,
                     VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                     bench.tsPool, 1);
                 bench.ctx->batch.submit();

                 // 4. Read GPU timestamps
                 uint64_t ts[2] = {0, 0};
                 vkGetQueryPoolResults(
                     bench.ctx->device.device(), bench.tsPool,
                     0, 2, sizeof(ts), ts, sizeof(uint64_t),
                     VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
                 bench.lastGpuMs = double(ts[1] - ts[0]) *
                                   double(bench.timestampPeriod) / 1e6;

                 // 5. Read back 8 bytes: packed (distance << 32) | index
                 uint64_t packed = 0;
                 std::memcpy(&packed, bench.gpuResult.mappedPtr,
                             sizeof(uint64_t));

                 uint32_t bestDist  = static_cast<uint32_t>(packed >> 32);
                 uint32_t bestIndex = static_cast<uint32_t>(packed & 0xFFFFFFFF);

                 py::dict result;
                 result["index"] = bestIndex;
                 result["distance"] = bestDist;
                 return result;
             },
             py::arg("query"),
             "GPU top-1 Hamming search via atomicMin on packed uint64.\n"
             "Returns dict with 'index' and 'distance'. Reads back 8 bytes\n"
             "instead of N*4, eliminating the PCIe readback bottleneck.")
        .def_property_readonly("gpu_time_ms",
             [](const HammingSearchBench& b) { return b.lastGpuMs; },
             "GPU-side execution time of last search (ms)");

    // ═══════════════════════════════════════════════════════════════════════
    // CUBEMIND: RUBIK'S CUBE STATE GENERATOR
    // ═══════════════════════════════════════════════════════════════════════

    m.def(
        "cube_solved",
        [](uint32_t size) -> py::array_t<uint8_t> {
            auto cs = (size == 2) ? grilly::cubemind::CubeSize::Cube2x2
                                  : grilly::cubemind::CubeSize::Cube3x3;
            auto state = grilly::cubemind::cubeSolved(cs);
            py::array_t<uint8_t> arr(state.facelets.size());
            std::memcpy(arr.request().ptr, state.facelets.data(),
                        state.facelets.size());
            return arr;
        },
        py::arg("size") = 3,
        "Create solved Rubik's cube state (2 or 3)");

    m.def(
        "cube_apply_move",
        [](py::array_t<uint8_t> state, uint32_t size,
           uint8_t move) -> py::array_t<uint8_t> {
            auto buf = state.request();
            grilly::cubemind::CubeState cs;
            cs.size = (size == 2) ? grilly::cubemind::CubeSize::Cube2x2
                                  : grilly::cubemind::CubeSize::Cube3x3;
            cs.facelets.assign(static_cast<uint8_t*>(buf.ptr),
                               static_cast<uint8_t*>(buf.ptr) + buf.size);

            auto result = grilly::cubemind::cubeApplyMove(
                cs, static_cast<grilly::cubemind::CubeMove>(move));

            py::array_t<uint8_t> arr(result.facelets.size());
            std::memcpy(arr.request().ptr, result.facelets.data(),
                        result.facelets.size());
            return arr;
        },
        py::arg("state"), py::arg("size") = 3, py::arg("move") = 0,
        "Apply a move to a cube state");

    m.def(
        "cube_random_walk",
        [](uint32_t size, uint32_t numMoves,
           uint32_t seed) -> py::array_t<uint8_t> {
            auto cs = (size == 2) ? grilly::cubemind::CubeSize::Cube2x2
                                  : grilly::cubemind::CubeSize::Cube3x3;
            auto state = grilly::cubemind::cubeRandomWalk(cs, numMoves, seed);
            py::array_t<uint8_t> arr(state.facelets.size());
            std::memcpy(arr.request().ptr, state.facelets.data(),
                        state.facelets.size());
            return arr;
        },
        py::arg("size") = 3, py::arg("num_moves") = 20,
        py::arg("seed") = 0,
        "Random walk from solved state");

    m.def(
        "cube_estimate_distance",
        [](py::array_t<uint8_t> state, uint32_t size) -> uint32_t {
            auto buf = state.request();
            grilly::cubemind::CubeState cs;
            cs.size = (size == 2) ? grilly::cubemind::CubeSize::Cube2x2
                                  : grilly::cubemind::CubeSize::Cube3x3;
            cs.facelets.assign(static_cast<uint8_t*>(buf.ptr),
                               static_cast<uint8_t*>(buf.ptr) + buf.size);
            return grilly::cubemind::cubeEstimateDistance(cs);
        },
        py::arg("state"), py::arg("size") = 3,
        "Estimate distance from solved via facelet mismatch heuristic");

    m.def(
        "cube_to_vsa",
        [](py::array_t<uint8_t> state, uint32_t size,
           uint32_t dim) -> py::array_t<int8_t> {
            auto buf = state.request();
            grilly::cubemind::CubeState cs;
            cs.size = (size == 2) ? grilly::cubemind::CubeSize::Cube2x2
                                  : grilly::cubemind::CubeSize::Cube3x3;
            cs.facelets.assign(static_cast<uint8_t*>(buf.ptr),
                               static_cast<uint8_t*>(buf.ptr) + buf.size);

            auto result = grilly::cubemind::cubeToVSA(cs, dim);

            py::array_t<int8_t> arr(dim);
            std::memcpy(arr.request().ptr, result.data(),
                        dim * sizeof(int8_t));
            return arr;
        },
        py::arg("state"), py::arg("size") = 3, py::arg("dim") = 10240,
        "Encode cube state as bipolar VSA hypervector");

    // ═══════════════════════════════════════════════════════════════════════
    // CUBEMIND: VSA CACHE
    // ═══════════════════════════════════════════════════════════════════════

    py::class_<grilly::cubemind::VSACache>(m, "VSACache")
        .def(py::init([](GrillyCoreContext& ctx,
                         uint32_t initialCapacity, uint32_t maxCapacity,
                         uint32_t dim, float surpriseThreshold,
                         float utilityDecay) {
                 grilly::cubemind::CacheConfig cfg;
                 cfg.initialCapacity = initialCapacity;
                 cfg.maxCapacity = maxCapacity;
                 cfg.dim = dim;
                 cfg.surpriseThreshold = surpriseThreshold;
                 cfg.utilityDecay = utilityDecay;
                 return std::make_unique<grilly::cubemind::VSACache>(
                     ctx.pool, cfg);
             }),
             py::arg("device"),
             py::arg("initial_capacity") = 1024,
             py::arg("max_capacity") = 500000,
             py::arg("dim") = 10240,
             py::arg("surprise_threshold") = 0.3f,
             py::arg("utility_decay") = 0.99f,
             py::keep_alive<1, 2>())  // Keep GrillyCoreContext alive while cache exists
        .def("lookup",
             [](grilly::cubemind::VSACache& cache,
                GrillyCoreContext& ctx,
                py::array_t<int8_t> query,
                uint32_t topK) -> py::dict {
                 auto buf = query.request();
                 uint32_t dim = static_cast<uint32_t>(buf.shape[0]);
                 auto packed = grilly::cubemind::vsaBitpack(
                     static_cast<const int8_t*>(buf.ptr), dim);

                 auto result = cache.lookup(ctx.batch, ctx.cache, packed, topK);

                 py::dict d;
                 d["indices"] = py::array_t<uint32_t>(
                     result.indices.size(), result.indices.data());
                 d["distances"] = py::array_t<uint32_t>(
                     result.distances.size(), result.distances.data());
                 d["surprise"] = result.querySurprise;
                 return d;
             },
             py::arg("device"), py::arg("query"), py::arg("top_k") = 5)
        .def("insert",
             [](grilly::cubemind::VSACache& cache,
                py::array_t<int8_t> key,
                float surprise, float stress) -> bool {
                 auto buf = key.request();
                 uint32_t dim = static_cast<uint32_t>(buf.shape[0]);
                 auto packed = grilly::cubemind::vsaBitpack(
                     static_cast<const int8_t*>(buf.ptr), dim);
                 grilly::cubemind::EmotionState emo{surprise, stress};
                 return cache.insert(packed, emo);
             },
             py::arg("key"), py::arg("surprise") = 1.0f,
             py::arg("stress") = 0.0f)
        .def("evict", &grilly::cubemind::VSACache::evict, py::arg("count"))
        .def("size", &grilly::cubemind::VSACache::size)
        .def("stats",
             [](const grilly::cubemind::VSACache& cache) -> py::dict {
                 auto s = cache.stats();
                 py::dict d;
                 d["size"] = s.size;
                 d["capacity"] = s.capacity;
                 d["total_inserts"] = s.totalInserts;
                 d["total_evictions"] = s.totalEvictions;
                 d["total_lookups"] = s.totalLookups;
                 d["avg_surprise"] = s.avgSurprise;
                 d["avg_utility"] = s.avgUtility;
                 d["last_lookup_ms"] = s.lastLookupMs;
                 return d;
             });
}
