#pragma once

#include <vulkan/vulkan.h>

#include <boost/container/flat_map.hpp>

#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "grilly/device.h"

namespace grilly {

/// Cached pipeline + layout + descriptor set layout for a single shader.
struct PipelineEntry {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
    VkShaderModule shaderModule = VK_NULL_HANDLE;
};

/// Pipeline and descriptor set cache.
/// Ports backend/pipelines.py:VulkanPipelines to C++.
///
/// Uses boost::container::flat_map for the pipeline cache — for the ~10-50
/// shader entries we expect, flat_map's contiguous memory layout gives better
/// cache locality than std::unordered_map (which scatters nodes across heap).
///
/// Descriptor set caching uses an LRU list (same eviction logic as
/// pipelines.py:164) to prevent VkDescriptorPool exhaustion.
class PipelineCache {
public:
    PipelineCache(GrillyDevice& device, uint32_t maxDescriptorSets = 500,
                  uint32_t maxStorageBuffers = 1000);
    ~PipelineCache();

    PipelineCache(const PipelineCache&) = delete;
    PipelineCache& operator=(const PipelineCache&) = delete;

    /// Load SPIR-V bytecode and register it under `name`.
    void loadSPIRV(const std::string& name, const std::vector<uint8_t>& code);

    /// Load SPIR-V from a file on disk.
    void loadSPIRVFile(const std::string& name, const std::string& path);

    /// Get or create a pipeline for a shader (creates descriptor set layout
    /// with `numBuffers` storage-buffer bindings and optional push constants).
    PipelineEntry getOrCreate(const std::string& name, uint32_t numBuffers,
                              uint32_t pushConstSize = 0);

    /// Allocate (or reuse from LRU cache) a descriptor set for the given
    /// shader, binding the provided buffer infos.
    VkDescriptorSet allocDescriptorSet(
        const std::string& name,
        const std::vector<VkDescriptorBufferInfo>& buffers);

    /// Check if a shader's SPIR-V has been loaded (for fusion availability).
    bool hasShader(const std::string& name) const {
        return spirvCode_.count(name) > 0;
    }

    struct CacheStats {
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t evictions = 0;
        size_t cachedSets = 0;
    };
    CacheStats cacheStats() const;

private:
    VkDescriptorSetLayout createDescLayout(uint32_t numBuffers);
    VkPipelineLayout createPipeLayout(VkDescriptorSetLayout descLayout,
                                       uint32_t pushConstSize);
    VkPipeline createPipeline(const std::vector<uint8_t>& spirv,
                              VkPipelineLayout pipeLayout,
                              VkShaderModule& outModule);

    GrillyDevice& device_;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;

    // Shader SPIR-V storage (name → bytecode)
    std::unordered_map<std::string, std::vector<uint8_t>> spirvCode_;

    // Pipeline cache — flat_map for cache-friendly iteration over ~10-50 entries
    boost::container::flat_map<std::string, PipelineEntry> pipelines_;

    // ── Descriptor set LRU cache ──
    // Key: (shader_name, sorted buffer handles+sizes) → descriptor set
    struct DescCacheKey {
        std::string shaderName;
        std::vector<std::pair<VkBuffer, VkDeviceSize>> bufferBindings;
        bool operator==(const DescCacheKey& o) const;
    };
    struct DescCacheKeyHash {
        size_t operator()(const DescCacheKey& k) const;
    };

    static constexpr size_t kMaxCachedDescSets = 100;

    // LRU list: front = most recently used, back = least recently used
    using LRUList = std::list<DescCacheKey>;
    LRUList lruList_;
    struct DescCacheEntry {
        VkDescriptorSet set;
        LRUList::iterator lruIter;
    };
    std::unordered_map<DescCacheKey, DescCacheEntry, DescCacheKeyHash> descCache_;

    mutable std::mutex mutex_;
    CacheStats stats_{};
};

}  // namespace grilly
