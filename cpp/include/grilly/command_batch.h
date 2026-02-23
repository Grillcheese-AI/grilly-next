#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>

#include "grilly/device.h"

namespace grilly {

/// Batched command buffer recorder with single-submit semantics.
/// Ports backend/core.py:799 (CommandRecorder) to C++ and adds timeline
/// semaphore support for async overlap.
///
/// The Python version waits on a fence after every single dispatch
/// (core.py:743), stalling the CPU. CommandBatch records N dispatches into
/// one command buffer with pipeline barriers between them, then submits once.
/// Timeline semaphores (Vulkan 1.2) let the CPU continue queueing while the
/// GPU catches up — the key piece the Python backend was missing.
class CommandBatch {
public:
    explicit CommandBatch(GrillyDevice& device);
    ~CommandBatch();

    CommandBatch(const CommandBatch&) = delete;
    CommandBatch& operator=(const CommandBatch&) = delete;

    /// Reset and begin recording.
    void begin();

    /// Record a compute dispatch (no submit yet).
    void dispatch(VkPipeline pipeline, VkPipelineLayout layout,
                  VkDescriptorSet descSet,
                  uint32_t gx, uint32_t gy = 1, uint32_t gz = 1,
                  const void* pushData = nullptr, uint32_t pushSize = 0);

    /// Insert a COMPUTE→COMPUTE memory barrier (SHADER_WRITE → SHADER_READ).
    void barrier();

    /// End recording, submit, and wait for completion (synchronous).
    void submit();

    /// End recording and submit with a timeline semaphore signal (async).
    /// The caller is responsible for waiting on the semaphore.
    void submitAsync(VkSemaphore timeline, uint64_t signalValue);

    /// True while between begin() and submit()/submitAsync().
    bool isRecording() const { return recording_; }

    /// Access underlying command buffer (for timestamp queries etc.)
    VkCommandBuffer cmdBuffer() const { return cmd_; }

private:
    GrillyDevice& device_;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    bool recording_ = false;
};

}  // namespace grilly
