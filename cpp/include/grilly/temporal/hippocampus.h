#pragma once

// ── Hippocampal Consolidator ─────────────────────────────────────────
//
// Offline "dream cycle" that generalizes repeated patterns into permanent
// WorldModel rules. The consolidation pipeline:
//
//   1. Record episode pairs (state_t, state_t+1) during online learning
//   2. Periodically run dream():
//      a. XOR all episode pairs -> transition deltas
//      b. SimHash deltas into 256 buckets (8-bit signatures)
//      c. Burn buckets with >2% of episodes as rules (majority vote)
//      d. Generate synthetic mutations (future: GPU dispatch)
//      e. Clear episodic buffer
//
// SimHash replaces exact delta matching, which had ~0 collision probability
// at dim=10240. With M=8 random projections, similar deltas land in the
// same bucket and are consolidated via majority vote.

#include <cstdint>
#include <deque>
#include <mutex>
#include <random>
#include <unordered_map>
#include <vector>

#include "grilly/cognitive/world_model.h"
#include "grilly/cubemind/types.h"

namespace grilly {

/// Summary returned after a dream consolidation cycle.
struct DreamReport {
    uint32_t episodes_replayed;
    uint32_t synthetic_dreams;
    uint32_t new_rules_extracted;
};

/// Hippocampal consolidator: records episodes and dreams to extract rules.
class HippocampalConsolidator {
public:
    struct Episode {
        cubemind::BitpackedVec state_t;
        cubemind::BitpackedVec state_t1;
    };

    explicit HippocampalConsolidator(uint32_t max_capacity = 10000)
        : max_capacity_(max_capacity) {}

    /// Record a state transition for later consolidation.
    void record_episode(const cubemind::BitpackedVec& state_t,
                        const cubemind::BitpackedVec& state_t1) {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        if (episodic_buffer_.size() >= max_capacity_)
            episodic_buffer_.pop_front();
        episodic_buffer_.push_back({state_t, state_t1});
    }

    /// Run offline dream consolidation cycle.
    ///
    /// Phase 1: XOR all episode pairs to compute transition deltas.
    /// Phase 2: SimHash deltas into 256 buckets, burn qualifying buckets.
    /// Phase 3: Generate synthetic mutations (counted only, no GPU dispatch yet).
    /// Phase 4: Clear episodic buffer.
    DreamReport dream(cognitive::WorldModel& wm, uint32_t cycles = 128) {
        std::lock_guard<std::mutex> lock(buffer_mutex_);

        DreamReport report{};
        if (episodic_buffer_.empty())
            return report;

        const uint32_t num_episodes =
            static_cast<uint32_t>(episodic_buffer_.size());
        report.episodes_replayed = num_episodes;

        // Lazy-init SimHash projections on first dream()
        if (episodic_buffer_.front().state_t.data.size() > 0 &&
            simhash_projections_.empty()) {
            init_simhash(
                static_cast<uint32_t>(episodic_buffer_.front().state_t.data.size()));
        }

        // Phase 1 + 2: compute deltas, SimHash into buckets
        std::unordered_map<uint8_t, std::vector<cubemind::BitpackedVec>> buckets;

        for (const auto& ep : episodic_buffer_) {
            cubemind::BitpackedVec delta = ep.state_t ^ ep.state_t1;
            uint8_t sig = compute_simhash(delta);
            buckets[sig].push_back(std::move(delta));
        }

        // Phase 2b: burn qualifying buckets (>2% threshold) via majority vote
        const uint32_t threshold =
            std::max(1u, static_cast<uint32_t>(num_episodes * 0.02));

        for (const auto& [sig, deltas] : buckets) {
            if (deltas.size() >= threshold) {
                auto representative = majority_vote(deltas);
                wm.add_fact_vec(representative);
                report.new_rules_extracted++;
            }
        }

        // Phase 3: synthetic mutations (count only -- GPU dispatch deferred)
        report.synthetic_dreams = 0;
        if (!episodic_buffer_.empty()) {
            std::mt19937 rng(std::random_device{}());
            for (uint32_t c = 0; c < cycles; ++c) {
                std::uniform_int_distribution<size_t> idx_dist(
                    0, episodic_buffer_.size() - 1);
                const auto& base = episodic_buffer_[idx_dist(rng)].state_t1;
                auto mutated = mutate_random_bits(base, 3, rng);
                (void)mutated;  // Future: dispatch to GPU
                report.synthetic_dreams++;
            }
        }

        // Phase 4: clear episodic buffer
        episodic_buffer_.clear();

        return report;
    }

    /// Current number of episodes in the buffer.
    uint32_t buffer_size() const {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        return static_cast<uint32_t>(episodic_buffer_.size());
    }

private:
    uint32_t max_capacity_;
    std::deque<Episode> episodic_buffer_;
    mutable std::mutex buffer_mutex_;

    // SimHash: M=8 random bitpacked projection vectors
    static constexpr uint32_t M_PROJECTIONS = 8;
    std::vector<std::vector<uint32_t>> simhash_projections_;  // M x num_words

    /// Initialize M random bitpacked projection vectors (deterministic seed).
    void init_simhash(uint32_t num_words) {
        std::mt19937 rng(42);  // deterministic for reproducibility
        simhash_projections_.resize(M_PROJECTIONS);
        for (uint32_t m = 0; m < M_PROJECTIONS; ++m) {
            simhash_projections_[m].resize(num_words);
            for (uint32_t w = 0; w < num_words; ++w) {
                simhash_projections_[m][w] = rng();
            }
        }
    }

    /// Compute 8-bit SimHash signature for a bitpacked delta vector.
    /// Each bit = popcount(delta AND proj_m) > D/2.
    uint8_t compute_simhash(const cubemind::BitpackedVec& delta) const {
        uint8_t sig = 0;
        uint32_t half_dim = delta.dim / 2;
        for (uint32_t m = 0; m < M_PROJECTIONS; ++m) {
            uint32_t ones = 0;
            for (size_t w = 0; w < delta.data.size(); ++w) {
                ones += popcount32(delta.data[w] & simhash_projections_[m][w]);
            }
            if (ones > half_dim)
                sig |= (1u << m);
        }
        return sig;
    }

    /// Majority vote across a set of bitpacked vectors.
    /// For each bit position, output the majority value.
    static cubemind::BitpackedVec majority_vote(
        const std::vector<cubemind::BitpackedVec>& vectors) {
        cubemind::BitpackedVec result;
        result.dim = vectors[0].dim;
        uint32_t num_words = static_cast<uint32_t>(vectors[0].data.size());
        result.data.resize(num_words, 0);

        uint32_t half = static_cast<uint32_t>(vectors.size()) / 2;

        for (uint32_t w = 0; w < num_words; ++w) {
            uint32_t word = 0;
            for (uint32_t bit = 0; bit < 32; ++bit) {
                uint32_t count = 0;
                for (const auto& v : vectors) {
                    if (v.data[w] & (1u << bit))
                        count++;
                }
                if (count > half)
                    word |= (1u << bit);
            }
            result.data[w] = word;
        }
        return result;
    }

    /// Portable popcount for uint32.
    static uint32_t popcount32(uint32_t x) {
        x = x - ((x >> 1) & 0x55555555);
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
        return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
    }

    /// Flip num_flips random bits in a bitpacked vector.
    static cubemind::BitpackedVec mutate_random_bits(
        const cubemind::BitpackedVec& base, uint32_t num_flips,
        std::mt19937& rng) {
        cubemind::BitpackedVec result = base;
        if (result.data.empty())
            return result;

        std::uniform_int_distribution<uint32_t> word_dist(
            0, static_cast<uint32_t>(result.data.size()) - 1);
        std::uniform_int_distribution<uint32_t> bit_dist(0, 31);

        for (uint32_t i = 0; i < num_flips; ++i) {
            uint32_t w = word_dist(rng);
            uint32_t b = bit_dist(rng);
            result.data[w] ^= (1u << b);
        }
        return result;
    }
};

}  // namespace grilly
