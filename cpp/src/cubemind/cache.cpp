#include "grilly/cubemind/cache.h"
#include "grilly/cubemind/vsa.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>
#include <random>

namespace grilly {
namespace cubemind {

// ── VSACache Constructor/Destructor ──────────────────────────────────

VSACache::VSACache(BufferPool& pool, const CacheConfig& config)
    : pool_(pool)
    , config_(config)
    , wordsPerVec_((config.dim + 31) / 32)
    , size_(0)
    , capacity_(config.initialCapacity)
    , gpuKeys_{}
    , gpuDirty_(true)
    , stats_{} {
    // Pre-allocate host storage
    keys_.resize(size_t(capacity_) * wordsPerVec_, 0);
    emotions_.resize(capacity_);
    utilities_.resize(capacity_, 0.0f);
}

VSACache::~VSACache() {
    if (gpuKeys_.handle != VK_NULL_HANDLE) {
        pool_.release(gpuKeys_);
    }
}

// ── GPU Sync ─────────────────────────────────────────────────────────
//
// Upload host-side keys to GPU buffer when dirty. This is called
// before every GPU Hamming search. The GPU buffer is sized to the
// current key array (may grow on insert).

void VSACache::ensureGPUSync() {
    if (!gpuDirty_ || size_ == 0) return;

    size_t keyBytes = size_t(size_) * wordsPerVec_ * sizeof(uint32_t);

    // Release old GPU buffer if it exists and is too small
    if (gpuKeys_.handle != VK_NULL_HANDLE) {
        if (gpuKeys_.size < keyBytes) {
            pool_.release(gpuKeys_);
            gpuKeys_ = {};
        }
    }

    // Acquire host-visible buffer from pool
    if (gpuKeys_.handle == VK_NULL_HANDLE) {
        gpuKeys_ = pool_.acquire(keyBytes);
    }

    pool_.upload(gpuKeys_, reinterpret_cast<const float*>(keys_.data()),
                 keyBytes);
    gpuDirty_ = false;
}

// ── Capacity Growth ──────────────────────────────────────────────────
//
// Double capacity when full, up to maxCapacity. Mirrors the paper's
// "66x growth via surprise-driven expansion" pattern.

void VSACache::growCapacity() {
    uint32_t newCapacity = std::min(capacity_ * 2, config_.maxCapacity);
    if (newCapacity == capacity_) return;  // At max

    keys_.resize(size_t(newCapacity) * wordsPerVec_, 0);
    emotions_.resize(newCapacity);
    utilities_.resize(newCapacity, 0.0f);
    capacity_ = newCapacity;
}

// ── Sanger's GHA Whitening ───────────────────────────────────────────
//
// Streaming extraction of principal components from a saturated bundle.
// The residual error (input minus reconstruction) is mathematically
// guaranteed to be orthogonal to all previously learned components.
//
// Multi-unit Sanger update:
//   Δw_ij = η * y_i * (h_j - Σ_{k=1}^{i} y_k * w_kj)
//
// Complexity: O(M * D) per call, where M = num_principal_components.

std::vector<int8_t> VSACache::computeSangerResidual(
    const std::vector<int8_t>& saturated_bundle) {
    uint32_t D = static_cast<uint32_t>(saturated_bundle.size());

    // Initialize weights if this is the first time or dimension changed
    if (sanger_weights_.empty() || sanger_weights_[0].size() != D) {
        sanger_weights_.assign(num_principal_components_,
                               std::vector<float>(D, 0.01f));
    }

    // 1. Convert bipolar int8 {-1, 1} to float for continuous gradient math
    std::vector<float> x(D);
    for (uint32_t j = 0; j < D; ++j) {
        x[j] = static_cast<float>(saturated_bundle[j]);
    }

    std::vector<float> y(num_principal_components_, 0.0f);
    std::vector<float> reconstruction(D, 0.0f);

    // 2. Sanger's GHA Streaming Loop
    for (uint32_t i = 0; i < num_principal_components_; ++i) {
        // Calculate scalar output y_i = W_i · x
        for (uint32_t j = 0; j < D; ++j) {
            y[i] += sanger_weights_[i][j] * x[j];
        }

        // Apply Sanger's weight update rule
        for (uint32_t j = 0; j < D; ++j) {
            float projection_sum = 0.0f;
            // Subtract projections onto previously learned components
            for (uint32_t k = 0; k <= i; ++k) {
                projection_sum += y[k] * sanger_weights_[k][j];
            }
            // Delta W
            sanger_weights_[i][j] +=
                sanger_learning_rate_ * y[i] * (x[j] - projection_sum);
        }
    }

    // 3. Compute the full reconstruction from the whitened components
    for (uint32_t i = 0; i < num_principal_components_; ++i) {
        for (uint32_t j = 0; j < D; ++j) {
            reconstruction[j] += y[i] * sanger_weights_[i][j];
        }
    }

    // 4. Calculate the Orthogonal Residual Error (e = x - x_hat)
    // Apply signum thresholding to snap it back into the bipolar VSA space
    std::vector<int8_t> residual_error(D);
    for (uint32_t j = 0; j < D; ++j) {
        float e_j = x[j] - reconstruction[j];
        residual_error[j] = (e_j > 0.0f) ? 1 : -1;
    }

    return residual_error;
}

// ── Dimensionality Growth (Algorithmic Neurogenesis) ────────────────
//
// Grow the bipolar dimension of ALL stored vectors by growthDim bits.
// If a saturated_bundle is provided, Sanger's GHA computes the
// mathematically orthogonal residual to use as padding. Otherwise
// falls back to deterministic random noise.
//
// After growth, gpuDirty_ is set so the next GPU sync re-uploads
// the entire cache with the new wider layout.

void VSACache::growDimensionality(uint32_t growthDim,
                                   const BitpackedVec* saturated_bundle) {
    uint32_t oldDim = config_.dim;
    uint32_t oldWordsPerVec = wordsPerVec_;

    // Compute Sanger residual if a saturated bundle is provided
    BitpackedVec packed_residual;
    bool use_sanger = false;
    if (saturated_bundle) {
        std::vector<int8_t> unpacked = vsaUnpack(*saturated_bundle);
        std::vector<int8_t> residual = computeSangerResidual(unpacked);
        packed_residual = vsaBitpack(residual.data(), oldDim);
        use_sanger = true;
    }

    // 1. Expand the mathematical dimensions
    config_.dim += growthDim;
    wordsPerVec_ = (config_.dim + 31) / 32;

    // 2. Allocate a new contiguous block for the expanded keys
    std::vector<uint32_t> newKeys(size_t(capacity_) * wordsPerVec_, 0);

    // 3. Fallback RNG for when no saturated bundle is provided
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist;

    // 4. Pad every existing concept in the memory cache
    for (uint32_t i = 0; i < size_; ++i) {
        uint32_t oldOffset = i * oldWordsPerVec;
        uint32_t newOffset = i * wordsPerVec_;

        // Copy the old historical data
        for (uint32_t w = 0; w < oldWordsPerVec; ++w) {
            newKeys[newOffset + w] = keys_[oldOffset + w];
        }

        // Pad the new dimensions
        uint32_t residual_words_to_copy = wordsPerVec_ - oldWordsPerVec;
        for (uint32_t w = 0; w < residual_words_to_copy; ++w) {
            if (use_sanger) {
                // Modulo wrapping fills growth dim even if growthDim > oldDim
                uint32_t res_idx = w % oldWordsPerVec;
                newKeys[newOffset + oldWordsPerVec + w] =
                    packed_residual.data[res_idx];
            } else {
                newKeys[newOffset + oldWordsPerVec + w] = dist(rng);
            }
        }
    }

    // 5. Replace the old host arrays and flag for a full GPU BDA re-upload
    keys_ = std::move(newKeys);
    gpuDirty_ = true;
}

// ── Top-K Selection ──────────────────────────────────────────────────
//
// Simple partial sort to find k smallest distances. For the cache
// sizes we target (<500K), this is fast enough on CPU. The GPU does
// the expensive Hamming distance computation; top-k is O(n*k).

void VSACache::topKSelect(const uint32_t* distances, uint32_t n,
                            uint32_t k, std::vector<uint32_t>& indices,
                            std::vector<uint32_t>& topDists) const {
    k = std::min(k, n);
    indices.resize(k);
    topDists.resize(k);

    // Create index array and partial sort by distance
    std::vector<uint32_t> allIndices(n);
    std::iota(allIndices.begin(), allIndices.end(), 0);

    std::partial_sort(allIndices.begin(), allIndices.begin() + k,
                       allIndices.end(),
                       [distances](uint32_t a, uint32_t b) {
                           return distances[a] < distances[b];
                       });

    for (uint32_t i = 0; i < k; ++i) {
        indices[i] = allIndices[i];
        topDists[i] = distances[allIndices[i]];
    }
}

// ── GPU Lookup ──────────────────────────────────────────────────────
//
// Pipeline:
//   1. Ensure GPU keys are synced
//   2. Dispatch hamming-search.glsl (or CPU fallback)
//   3. Download distances
//   4. CPU top-k selection
//   5. Return results with emotions and surprise

CacheLookupResult VSACache::lookup(CommandBatch& batch,
                                     PipelineCache& pipeCache,
                                     const BitpackedVec& query,
                                     uint32_t topK) {
    CacheLookupResult result;

    if (size_ == 0) {
        result.querySurprise = 1.0f;  // Everything is novel to empty cache
        return result;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    ensureGPUSync();

    // Compute Hamming distances using persistent GPU cache buffer
    std::vector<uint32_t> distances(size_);
    size_t cacheBytes = size_t(size_) * wordsPerVec_ * sizeof(uint32_t);
    hammingSearchPersistent(batch, pool_, pipeCache,
                             query.data.data(),
                             gpuKeys_, cacheBytes,
                             distances.data(), size_, wordsPerVec_);

    // Top-k selection
    topKSelect(distances.data(), size_, topK,
               result.indices, result.distances);

    // Gather emotions for top-k
    result.emotions.resize(result.indices.size());
    for (size_t i = 0; i < result.indices.size(); ++i) {
        result.emotions[i] = emotions_[result.indices[i]];
    }

    // Surprise = normalized minimum distance
    uint32_t minDist = result.distances.empty() ? config_.dim
                                                 : result.distances[0];
    result.querySurprise = static_cast<float>(minDist) / config_.dim;

    auto t1 = std::chrono::high_resolution_clock::now();
    stats_.lastLookupMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    stats_.totalLookups++;

    return result;
}

CacheLookupResult VSACache::lookupCPU(const BitpackedVec& query,
                                        uint32_t topK) {
    CacheLookupResult result;

    if (size_ == 0) {
        result.querySurprise = 1.0f;
        return result;
    }

    auto distances = hammingSearchCPU(query.data.data(), keys_.data(),
                                       size_, wordsPerVec_);

    topKSelect(distances.data(), size_, topK,
               result.indices, result.distances);

    result.emotions.resize(result.indices.size());
    for (size_t i = 0; i < result.indices.size(); ++i) {
        result.emotions[i] = emotions_[result.indices[i]];
    }

    uint32_t minDist = result.distances.empty() ? config_.dim
                                                 : result.distances[0];
    result.querySurprise = static_cast<float>(minDist) / config_.dim;
    stats_.totalLookups++;

    return result;
}

// ── GPU Capacity Pre-allocation ──────────────────────────────────────
//
// Pre-allocate the GPU buffer for the full current capacity (not just
// current size). This enables incremental GPU sync: after each insert,
// we write just the new entry at the correct offset without reallocating.
// Called by insertGPU() before the first surprise check.

void VSACache::ensureGPUCapacity() {
    size_t capacityBytes = size_t(capacity_) * wordsPerVec_ * sizeof(uint32_t);

    if (gpuKeys_.handle != VK_NULL_HANDLE && gpuKeys_.size >= capacityBytes) {
        return;  // Already big enough
    }

    // Need to (re)allocate
    if (gpuKeys_.handle != VK_NULL_HANDLE) {
        pool_.release(gpuKeys_);
        gpuKeys_ = {};
    }

    gpuKeys_ = pool_.acquire(capacityBytes);

    // Upload existing data into the new buffer
    if (size_ > 0) {
        size_t dataBytes = size_t(size_) * wordsPerVec_ * sizeof(uint32_t);
        pool_.upload(gpuKeys_,
                     reinterpret_cast<const float*>(keys_.data()), dataBytes);
    }
    gpuDirty_ = false;
}

// ── Insert ──────────────────────────────────────────────────────────
//
// Surprise-driven insertion: compute normalized Hamming distance to
// nearest existing entry. If surprise > threshold, insert.
// If cache is full, grow capacity (up to max) or evict.

bool VSACache::insert(const BitpackedVec& key, const EmotionState& emotion) {
    // Check surprise against existing entries
    if (size_ > 0) {
        auto distances = hammingSearchCPU(key.data.data(), keys_.data(),
                                           size_, wordsPerVec_);
        uint32_t minDist = *std::min_element(distances.begin(), distances.end());
        float surprise = static_cast<float>(minDist) / config_.dim;

        if (surprise < config_.surpriseThreshold) {
            return false;  // Not novel enough
        }
    }

    // Grow if needed
    if (size_ >= capacity_) {
        if (capacity_ < config_.maxCapacity) {
            growCapacity();
        } else {
            // At max capacity — evict 1 entry first
            evict(1);
        }
    }

    // Insert at position size_
    std::memcpy(keys_.data() + size_t(size_) * wordsPerVec_,
                key.data.data(),
                wordsPerVec_ * sizeof(uint32_t));
    emotions_[size_] = emotion;
    // Initial utility proportional to surprise — novel entries survive eviction.
    // surprise=1.0 → utility=1.0, surprise=0.1 → utility=0.1
    utilities_[size_] = std::max(emotion.surprise, 0.01f);

    size_++;
    gpuDirty_ = true;
    stats_.totalInserts++;
    stats_.size = size_;

    return true;
}

uint32_t VSACache::insertBatch(const std::vector<BitpackedVec>& keys,
                                const std::vector<EmotionState>& emotions) {
    uint32_t inserted = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (insert(keys[i], emotions[i])) {
            inserted++;
        }
    }
    return inserted;
}

// ── GPU-Accelerated Insert ──────────────────────────────────────────
//
// Uses the GPU hamming-search shader for surprise checking instead of
// the O(n) CPU linear scan. The GPU buffer is maintained incrementally:
// after each successful insert, only the new entry is uploaded (not
// the entire cache), making bulk insertion O(n) total.
//
// Pipeline per insert:
//   1. Ensure GPU buffer is capacity-sized and synced
//   2. Dispatch hamming-search.glsl (same shader as lookup)
//   3. Find minimum distance on CPU (from downloaded distances)
//   4. If novel: insert + incremental GPU upload of new entry
//
// Compared to CPU insert at 93K entries:
//   CPU:  O(n) per insert × n inserts = O(n²) = ~1000+ seconds
//   GPU:  O(1) GPU dispatch per insert × n = O(n) = ~50 seconds

bool VSACache::insertGPU(CommandBatch& batch, PipelineCache& pipeCache,
                          const BitpackedVec& key, const EmotionState& emotion) {
    // Ensure GPU buffer is allocated for full capacity
    ensureGPUCapacity();

    // Re-upload if dirty (happens after evict() or growCapacity())
    if (gpuDirty_) {
        size_t dataBytes = size_t(size_) * wordsPerVec_ * sizeof(uint32_t);
        if (size_ > 0) {
            pool_.upload(gpuKeys_,
                         reinterpret_cast<const float*>(keys_.data()),
                         dataBytes);
        }
        gpuDirty_ = false;
    }

    // GPU surprise check
    if (size_ > 0) {
        std::vector<uint32_t> distances(size_);
        size_t cacheBytes = size_t(size_) * wordsPerVec_ * sizeof(uint32_t);
        hammingSearchPersistent(batch, pool_, pipeCache,
                                 key.data.data(),
                                 gpuKeys_, cacheBytes,
                                 distances.data(), size_, wordsPerVec_);

        uint32_t minDist = *std::min_element(distances.begin(),
                                              distances.end());
        float surprise = static_cast<float>(minDist) / config_.dim;

        if (surprise < config_.surpriseThreshold) {
            return false;  // Not novel enough
        }
    }

    // Grow if needed
    if (size_ >= capacity_) {
        if (capacity_ < config_.maxCapacity) {
            growCapacity();
            ensureGPUCapacity();  // Reallocate + upload existing data
        } else {
            evict(1);
            // Re-upload after eviction (swap-and-pop changes data layout)
            size_t dataBytes = size_t(size_) * wordsPerVec_ * sizeof(uint32_t);
            pool_.upload(gpuKeys_,
                         reinterpret_cast<const float*>(keys_.data()),
                         dataBytes);
            gpuDirty_ = false;
        }
    }

    // Insert at position size_ (host)
    std::memcpy(keys_.data() + size_t(size_) * wordsPerVec_,
                key.data.data(),
                wordsPerVec_ * sizeof(uint32_t));
    emotions_[size_] = emotion;
    utilities_[size_] = std::max(emotion.surprise, 0.01f);

    size_++;
    stats_.totalInserts++;
    stats_.size = size_;

    // Incremental GPU sync: upload just the new entry at its offset
    size_t offset = size_t(size_ - 1) * wordsPerVec_ * sizeof(uint32_t);
    size_t entryBytes = wordsPerVec_ * sizeof(uint32_t);
    pool_.uploadPartial(gpuKeys_, key.data.data(), offset, entryBytes);
    // gpuDirty_ stays false — we maintain sync incrementally

    return true;
}

uint32_t VSACache::insertBatchGPU(CommandBatch& batch, PipelineCache& pipeCache,
                                   const std::vector<BitpackedVec>& keys,
                                   const std::vector<EmotionState>& emotions) {
    uint32_t inserted = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (insertGPU(batch, pipeCache, keys[i], emotions[i])) {
            inserted++;
        }
    }
    return inserted;
}

// ── Unchecked Insert ────────────────────────────────────────────────
//
// Bypass surprise checking entirely. For bulk ingestion of pre-
// deduplicated data where the O(n) per-insert Hamming scan is
// unnecessary. Each insert is O(1) — just a memcpy.

bool VSACache::insertUnchecked(const BitpackedVec& key,
                                const EmotionState& emotion) {
    // Grow if needed
    if (size_ >= capacity_) {
        if (capacity_ < config_.maxCapacity) {
            growCapacity();
        } else {
            return false;  // At max capacity, reject
        }
    }

    std::memcpy(keys_.data() + size_t(size_) * wordsPerVec_,
                key.data.data(),
                wordsPerVec_ * sizeof(uint32_t));
    emotions_[size_] = emotion;
    utilities_[size_] = std::max(emotion.surprise, 0.01f);

    size_++;
    gpuDirty_ = true;
    stats_.totalInserts++;
    stats_.size = size_;

    return true;
}

uint32_t VSACache::insertBatchUnchecked(
    const std::vector<BitpackedVec>& keys,
    const std::vector<EmotionState>& emotions) {
    uint32_t inserted = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (insertUnchecked(keys[i], emotions[i])) {
            inserted++;
        }
    }
    return inserted;
}

// ── Eviction ────────────────────────────────────────────────────────
//
// Remove the n entries with lowest utility. Compact the arrays by
// moving the last entry into each gap (swap-and-pop).

void VSACache::evict(uint32_t count) {
    if (count == 0 || size_ == 0) return;
    count = std::min(count, size_);

    // Find indices of lowest-utility entries
    std::vector<uint32_t> indices(size_);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + count,
                       indices.end(),
                       [this](uint32_t a, uint32_t b) {
                           return utilities_[a] < utilities_[b];
                       });

    // Sort eviction indices in descending order to avoid invalidation
    std::vector<uint32_t> toEvict(indices.begin(), indices.begin() + count);
    std::sort(toEvict.rbegin(), toEvict.rend());

    for (uint32_t idx : toEvict) {
        // Swap with last entry
        uint32_t last = size_ - 1;
        if (idx != last) {
            std::memcpy(keys_.data() + size_t(idx) * wordsPerVec_,
                        keys_.data() + size_t(last) * wordsPerVec_,
                        wordsPerVec_ * sizeof(uint32_t));
            emotions_[idx] = emotions_[last];
            utilities_[idx] = utilities_[last];
        }
        size_--;
    }

    gpuDirty_ = true;
    stats_.totalEvictions += count;
    stats_.size = size_;
}

// ── Utility Update ──────────────────────────────────────────────────
//
// Decay all utilities by the decay factor (recency penalty).
// Boost accessed entries (frequency reward).

void VSACache::updateUtility(const std::vector<uint32_t>& accessedIndices) {
    // Decay all
    for (uint32_t i = 0; i < size_; ++i) {
        utilities_[i] *= config_.utilityDecay;
    }

    // Boost accessed
    for (uint32_t idx : accessedIndices) {
        if (idx < size_) {
            utilities_[idx] += 1.0f;
        }
    }
}

CacheStats VSACache::stats() const {
    CacheStats s = stats_;
    s.size = size_;
    s.capacity = capacity_;

    // Compute averages
    if (size_ > 0) {
        float sumSurprise = 0.0f, sumUtility = 0.0f;
        for (uint32_t i = 0; i < size_; ++i) {
            sumSurprise += emotions_[i].surprise;
            sumUtility += utilities_[i];
        }
        s.avgSurprise = sumSurprise / size_;
        s.avgUtility = sumUtility / size_;
    }

    return s;
}

}  // namespace cubemind
}  // namespace grilly
