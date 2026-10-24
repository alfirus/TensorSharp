// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace tsg {
// One scope-owned work record per exact GLM group. No allocation, source-key
// approximation or future route information is used while compacting payload.
class HostMoeVmmSlotPlan {
public:
    static constexpr std::size_t maximum_slots = 8192;
    struct Move { unsigned from = 0, to = 0; };
private:
    std::array<unsigned, maximum_slots> order_{}, discarded_{};
    std::array<Move, maximum_slots> moves_{};
    std::array<std::uint8_t, maximum_slots> keep_{};
    std::size_t discarded_count_ = 0, move_count_ = 0, kept_hot_ = 0;
public:
    void prepare(std::size_t current, std::size_t target, const int* experts,
        const std::uint64_t* last_used, const std::uint8_t* protected_slots = nullptr) {
        discarded_count_ = move_count_ = kept_hot_ = 0;
        if (!current || !target || current > maximum_slots || target > maximum_slots
            || !experts || !last_used) throw std::invalid_argument("Invalid VMM slot compaction span");
        std::size_t protected_count = 0, hot = 0;
        for (std::size_t i = 0; i != current; ++i) {
            if (experts[i] < -1 || (protected_slots && protected_slots[i] > 1)
                || (protected_slots && protected_slots[i] && experts[i] < 0))
                throw std::invalid_argument("Invalid VMM slot compaction tag");
            order_[i] = static_cast<unsigned>(i); keep_[i] = 0;
            hot += experts[i] >= 0;
            protected_count += protected_slots && protected_slots[i];
        }
        if (protected_count > target) throw std::invalid_argument("VMM target cannot preserve its current union");
        std::sort(order_.begin(), order_.begin() + current, [&](unsigned a, unsigned b) {
            const bool valid_a = experts[a] >= 0, valid_b = experts[b] >= 0;
            if (valid_a != valid_b) return valid_a;
            const bool protected_a = protected_slots && protected_slots[a];
            const bool protected_b = protected_slots && protected_slots[b];
            if (protected_a != protected_b) return protected_a;
            if (last_used[a] != last_used[b]) return last_used[a] > last_used[b];
            // Native LRU discards the first index on an equal timestamp.
            return a > b;
        });
        kept_hot_ = (std::min)(hot, target);
        for (std::size_t i = 0; i != kept_hot_; ++i) keep_[order_[i]] = 1;
        for (std::size_t i = 0; i != current; ++i)
            if (experts[i] >= 0 && !keep_[i]) discarded_[discarded_count_++] = static_cast<unsigned>(i);
        if (target >= current) return;
        std::size_t destination = 0;
        for (std::size_t source = target; source != current; ++source) if (keep_[source]) {
            while (destination < target && keep_[destination]) ++destination;
            if (destination == target) throw std::logic_error("VMM low-prefix holes do not cover survivors");
            moves_[move_count_++] = {static_cast<unsigned>(source), static_cast<unsigned>(destination)};
            keep_[destination++] = 1;
        }
    }
    std::size_t kept_hot() const noexcept { return kept_hot_; }
    std::size_t discarded_count() const noexcept { return discarded_count_; }
    unsigned discarded(std::size_t i) const { return discarded_.at(i); }
    std::size_t move_count() const noexcept { return move_count_; }
    Move move(std::size_t i) const { return moves_.at(i); }
};

// The callback returns the actual three-prefix rounded weight size. Monotonic
// search keeps mandatory scratch/floors inside the existing admitted quota.
template<class WeightBytes>
std::size_t host_moe_vmm_capacity(std::size_t quota, std::size_t scratch,
    std::size_t minimum, std::size_t maximum, WeightBytes weight_bytes) {
    if (!minimum || minimum > maximum || maximum > HostMoeVmmSlotPlan::maximum_slots)
        throw std::invalid_argument("Invalid VMM capacity search bounds");
    if (scratch > quota || weight_bytes(minimum) > quota - scratch) return 0;
    std::size_t low = minimum, high = maximum;
    while (low < high) {
        const auto middle = low + (high - low + 1) / 2;
        if (weight_bytes(middle) <= quota - scratch) low = middle;
        else high = middle - 1;
    }
    return low;
}
} // namespace tsg
