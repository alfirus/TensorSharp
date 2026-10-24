// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace tsg {
// Exact registration generations, not mapped addresses or layer-local IDs.
struct HostMoeVictimKey {
    std::array<std::uint64_t, 3> source{};
    std::int32_t expert = -1;
    bool valid() const noexcept { return expert >= 0 && source[0] && source[1] && source[2]; }
    bool operator==(const HostMoeVictimKey& other) const noexcept {
        return expert == other.expert && source == other.source;
    }
};

// Past-only GPU hits in the last Window selected-expert accesses. Misses occupy
// empty ring positions so they age history at precisely the same rate. Storage
// is fixed and can be charged as one sizeof(policy) allocation. No per-route
// allocation, unbounded hash table, probabilistic identity, or future routes.
template<std::size_t Window = 16384, unsigned Threshold = 4>
class HostMoeVictimPolicy {
    static_assert(Window && !(Window & (Window - 1)), "Victim history must be a power of two");
    static_assert(Window <= (std::numeric_limits<std::uint32_t>::max)() / 2 && Threshold,
        "Victim history/count bound is invalid");
    static constexpr std::size_t slots = Window * 2, mask = slots - 1;
    struct Count { HostMoeVictimKey key{}; std::uint32_t value = 0; };
    std::array<HostMoeVictimKey, Window> history_{};
    std::array<Count, slots> counts_{};
    std::size_t position_ = 0, events_ = 0, keys_ = 0;
    static std::size_t hash(const HostMoeVictimKey& key) noexcept {
        std::uint64_t value = static_cast<std::uint32_t>(key.expert);
        for (auto part : key.source) {
            part ^= part >> 30; part *= UINT64_C(0xbf58476d1ce4e5b9);
            part ^= part >> 27; part *= UINT64_C(0x94d049bb133111eb);
            value ^= (part ^ (part >> 31)) + UINT64_C(0x9e3779b97f4a7c15) + (value << 6) + (value >> 2);
        }
        return static_cast<std::size_t>(value) & mask;
    }
    std::size_t find(const HostMoeVictimKey& key) const noexcept {
        auto position = hash(key);
        while (counts_[position].value && !(counts_[position].key == key)) position = (position + 1) & mask;
        return position;
    }
    void decrement(const HostMoeVictimKey& key) noexcept {
        auto hole = find(key);
        if (!counts_[hole].value || --counts_[hole].value) return;
        --keys_;
        // Back-shift deletion retains every colliding key without tombstones.
        auto next = (hole + 1) & mask;
        while (counts_[next].value) {
            const auto home = hash(counts_[next].key);
            if (((next - home) & mask) >= ((next - hole) & mask)) {
                counts_[hole] = counts_[next]; hole = next;
            }
            next = (next + 1) & mask;
        }
        counts_[hole] = {};
    }
public:
    void observe(const HostMoeVictimKey& key, bool gpu_hit) noexcept {
        if (history_[position_].valid()) decrement(history_[position_]);
        auto event = gpu_hit && key.valid() ? key : HostMoeVictimKey{};
        history_[position_] = event;
        position_ = (position_ + 1) & (Window - 1);
        if (events_ < Window) ++events_;
        if (event.valid()) {
            auto& count = counts_[find(event)];
            if (!count.value) { count.key = event; ++keys_; }
            ++count.value;
        }
    }
    unsigned count(const HostMoeVictimKey& key) const noexcept {
        return key.valid() ? counts_[find(key)].value : 0;
    }
    bool admit(const HostMoeVictimKey& key) const noexcept { return count(key) >= Threshold; }
    void clear() noexcept {
        for (auto& event : history_) event = {};
        for (auto& count : counts_) count = {};
        position_ = events_ = keys_ = 0;
    }
    std::size_t events() const noexcept { return events_; }
    std::size_t keys() const noexcept { return keys_; }
    static constexpr std::size_t window() noexcept { return Window; }
    static constexpr unsigned threshold() noexcept { return Threshold; }
};
} // namespace tsg
