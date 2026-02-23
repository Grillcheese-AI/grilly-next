#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "grilly/device.h"

namespace grilly {

/// A single GPU buffer backed by VMA.
/// Mirrors backend/buffer_pool.py:VMABuffer — but with persistent mapping
/// baked in so upload/download is a single memcpy (no vkMap/vkUnmap).
struct GrillyBuffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo info{};
    size_t size = 0;         ///< Requested size
    size_t bucketSize = 0;   ///< Rounded-up power-of-2 allocation size
    void* mappedPtr = nullptr;  ///< Persistent CPU-visible mapping
};

/// VMA-backed buffer pool with power-of-2 bucket reuse.
/// Ports backend/buffer_pool.py:VMABufferPool to native C++.
///
/// Key difference from Python: VMA is called natively (no ctypes) and buffers
/// use VMA_ALLOCATION_CREATE_MAPPED_BIT for persistent mapping, so upload is
/// a bare memcpy instead of vkMapMemory→copy→vkUnmapMemory.
class BufferPool {
public:
    explicit BufferPool(GrillyDevice& device);
    ~BufferPool();

    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;

    /// Acquire a buffer of at least `size` bytes (bucket-rounded).
    /// Memory is host-visible + persistently mapped.
    GrillyBuffer acquire(size_t size);

    /// Acquire a device-local (VRAM) buffer — NOT host-visible.
    /// Must use uploadStaged() to populate. GPU reads are 20x faster
    /// than host-visible memory on discrete GPUs.
    GrillyBuffer acquireDeviceLocal(size_t size);

    /// Acquire a buffer that prefers device-local (VRAM) memory but
    /// falls back to host-visible if ReBAR/SAM is not available.
    /// On AMD with ReBAR: allocates in VRAM with host-visible mapping
    /// (best of both worlds — VRAM bandwidth + direct CPU access).
    GrillyBuffer acquirePreferDeviceLocal(size_t size);

    /// Acquire a host-visible buffer optimized for CPU reads (readback).
    /// Uses HOST_ACCESS_RANDOM_BIT so the OS maps it into cached system RAM
    /// (L1/L2/L3). Reading 2 MB takes <0.1ms instead of 50ms with WC memory.
    GrillyBuffer acquireReadback(size_t size);

    /// Release buffer back to the pool for reuse.
    void release(GrillyBuffer& buf);

    /// Upload CPU data into a buffer via persistent mapping.
    void upload(GrillyBuffer& buf, const float* data, size_t bytes);

    /// Upload CPU data into a device-local buffer via staging buffer.
    /// Allocates a temporary host-visible staging buffer, copies data in,
    /// issues a vkCmdCopyBuffer, and waits for completion.
    void uploadStaged(GrillyBuffer& deviceBuf, const void* data, size_t bytes);

    /// Download GPU data from a buffer via persistent mapping.
    void download(const GrillyBuffer& buf, float* out, size_t bytes);

    /// Download GPU data from a device-local buffer via staging buffer.
    /// Allocates a temporary host-visible staging buffer, issues a
    /// vkCmdCopyBuffer from device-local to staging, waits, then memcpy.
    void downloadStaged(const GrillyBuffer& deviceBuf, void* out, size_t bytes);

    VmaAllocator allocator() const { return allocator_; }

    struct Stats {
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t allocations = 0;
        uint64_t totalAcquired = 0;
        uint64_t totalReleased = 0;
    };
    Stats stats() const;

private:
    static constexpr int kMinBucketPower = 8;   // 256 B
    static constexpr int kMaxBucketPower = 28;  // 256 MB
    static constexpr size_t kMaxBuffersPerBucket = 32;

    size_t sizeToBucket(size_t size) const;
    GrillyBuffer allocateBuffer(size_t bucketSize);

    GrillyDevice& device_;
    VmaAllocator allocator_ = VK_NULL_HANDLE;

    std::mutex mutex_;
    // bucket size → free buffers
    std::unordered_map<size_t, std::vector<GrillyBuffer>> buckets_;
    Stats stats_{};
};

}  // namespace grilly
