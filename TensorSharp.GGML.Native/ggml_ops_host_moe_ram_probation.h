// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace tsg {
// Exact, bounded LRU history for second GPU-miss-demand admission. This is
// independent of RAM readiness/residency: a first history observation can still
// be a RAM hit and must not weaken ordinary demand's RAM-hit LRU touch.
// Caller observes each eligible canonical projection once in selected-miss,
// gate/up/down order; no GPU hits, wide scans or individual window fragments.
// Valid observe/erase/clear/stats operations allocate no heap and call no user
// callback. The owner reserves metadata_forecast_bytes before constructing it.
template<std::size_t Capacity = 4096>
class HostMoeRamProbation {
    static_assert(Capacity && !(Capacity & (Capacity - 1)) && Capacity <= 4096,
        "Probation capacity must be a power of two no larger than4096");
public:
    static constexpr std::size_t capacity = Capacity;
    static constexpr std::size_t metadata_forecast_bytes = std::size_t(1) << 20;
    struct Range {
        std::uint64_t source = 0, offset = 0;
        std::size_t bytes = 0;
        bool operator==(const Range& other) const noexcept {
            return source == other.source && offset == other.offset && bytes == other.bytes;
        }
    };
    struct Stats {
        std::size_t resident_keys = 0;
        std::uint64_t observations = 0, repeats = 0, firsts = 0, evictions = 0;
    };
private:
    using Index = std::uint32_t;
    static constexpr Index none = (std::numeric_limits<Index>::max)();
    static constexpr std::size_t bucket_count = 2 * Capacity;
    struct Node {
        Range key;
        Index hash_next = none, previous = none, next = none;
    };
    mutable std::mutex mutex_;
    std::array<Node, Capacity> nodes_{};
    std::array<Index, bucket_count> buckets_{};
    Index free_ = none, newest_ = none, oldest_ = none;
    Stats stats_;

    static std::uint64_t mix(std::uint64_t value) noexcept {
        value ^= value >> 30; value *= UINT64_C(0xbf58476d1ce4e5b9);
        value ^= value >> 27; value *= UINT64_C(0x94d049bb133111eb);
        return value ^ (value >> 31);
    }
    static std::size_t bucket(const Range& key) noexcept {
        auto value = mix(key.source);
        for (const auto part : {key.offset, static_cast<std::uint64_t>(key.bytes)})
            value ^= mix(part) + UINT64_C(0x9e3779b97f4a7c15) + (value << 6) + (value >> 2);
        return static_cast<std::size_t>(value) & (bucket_count - 1);
    }
    static void increment(std::uint64_t& counter) noexcept {
        if (counter != (std::numeric_limits<std::uint64_t>::max)()) ++counter;
    }
    Index find(const Range& key) const noexcept {
        for (auto i = buckets_[bucket(key)]; i != none; i = nodes_[i].hash_next)
            if (nodes_[i].key == key) return i;
        return none;
    }
    void detach_lru(Index i) noexcept {
        const auto before = nodes_[i].previous, after = nodes_[i].next;
        if (before != none) nodes_[before].next = after; else newest_ = after;
        if (after != none) nodes_[after].previous = before; else oldest_ = before;
        nodes_[i].previous = nodes_[i].next = none;
    }
    void touch(Index i) noexcept {
        if (newest_ == i) return;
        detach_lru(i);
        nodes_[i].next = newest_;
        if (newest_ != none) nodes_[newest_].previous = i; else oldest_ = i;
        newest_ = i;
    }
    void remove(Index i) noexcept {
        auto* link = &buckets_[bucket(nodes_[i].key)];
        while (*link != i) link = &nodes_[*link].hash_next;
        *link = nodes_[i].hash_next;
        detach_lru(i);
        nodes_[i].key = {};
        nodes_[i].hash_next = free_; free_ = i;
        --stats_.resident_keys;
    }
    void reset() noexcept {
        buckets_.fill(none);
        for (std::size_t i = 0; i < Capacity; ++i) {
            nodes_[i].key = {};
            nodes_[i].hash_next = i + 1 == Capacity ? none : static_cast<Index>(i + 1);
            nodes_[i].previous = nodes_[i].next = none;
        }
        free_ = 0; newest_ = oldest_ = none; stats_ = {};
    }
public:
    HostMoeRamProbation() noexcept { reset(); }
    HostMoeRamProbation(const HostMoeRamProbation&) = delete;
    HostMoeRamProbation& operator=(const HostMoeRamProbation&) = delete;

    // Return PRIOR membership, then touch/insert MRU. No count is needed for a
    // threshold of two: membership means at least one earlier demand while the
    // exact key remained in this independent LRU history. Invalid logical keys
    // fail before changing history; source is a nonzero registration generation.
    bool observe(const Range& key) {
        if (!key.source || !key.bytes || key.bytes > UINT64_MAX - key.offset)
            throw std::invalid_argument("Invalid RAM probation source/range");
        std::lock_guard<std::mutex> lock(mutex_);
        increment(stats_.observations);
        const auto found = find(key);
        if (found != none) {
            increment(stats_.repeats); touch(found); return true;
        }
        increment(stats_.firsts);
        if (free_ == none) { remove(oldest_); increment(stats_.evictions); }
        const auto i = free_;
        free_ = nodes_[i].hash_next;
        nodes_[i].key = key;
        const auto selected = bucket(key);
        nodes_[i].hash_next = buckets_[selected]; buckets_[selected] = i;
        // A newly allocated node is not yet linked; attach without detach_lru.
        nodes_[i].previous = none; nodes_[i].next = newest_;
        if (newest_ != none) nodes_[newest_].previous = i; else oldest_ = i;
        newest_ = i; ++stats_.resident_keys;
        return false;
    }
    // Erasure is linearizable but has no tombstones: callers must stop old-
    // generation observations before unregistering and never reuse a generation.
    // Other keys and cumulative stats stay; all work is bounded by Capacity.
    void erase_source(std::uint64_t source) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto i = newest_; i != none;) {
            const auto next = nodes_[i].next;
            if (nodes_[i].key.source == source) remove(i);
            i = next;
        }
    }
    // New scope/history generation, including its counters.
    void clear() { std::lock_guard<std::mutex> lock(mutex_); reset(); }
    Stats stats() const { std::lock_guard<std::mutex> lock(mutex_); return stats_; }
#ifdef TSG_HOST_MOE_RAM_PROBATION_TEST_HOOKS
    static std::size_t test_bucket(const Range& key) noexcept { return bucket(key); }
#endif
};
static_assert(sizeof(HostMoeRamProbation<>) <= HostMoeRamProbation<>::metadata_forecast_bytes,
    "RAM probation exceeds its separately reserved metadata forecast");
} // namespace tsg
