// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tsg {
template<class Upload, class Compute>
bool host_moe_overlap_drain_both(Upload upload, Compute compute) noexcept {
    bool okay = true;
    try { upload(); } catch (...) { okay = false; }
    try { compute(); } catch (...) { okay = false; }
    return okay;
}
// Metadata only: exact selected projection bytes, two slots in the existing
// charged arena. Each phase retains the caller's expert/projection order.
struct HostMoeProjectionRingPlan {
    struct Piece { int slot, part; std::size_t source_offset, expert_offset, staging_offset, bytes; };
    struct Window { std::size_t first, count, bytes; };
    std::vector<Piece> pieces;
    std::vector<Window> windows;
    std::size_t slot_bytes = 0, payload_bytes = 0;
    static constexpr std::size_t max_items = 4096, max_misses = 288;

    static std::size_t add(std::size_t a, std::size_t b) {
        if (b > (std::numeric_limits<std::size_t>::max)() - a)
            throw std::overflow_error("Projection ring extent overflows");
        return a + b;
    }
    static std::size_t mul(std::size_t a, std::size_t b) {
        if (a && b > (std::numeric_limits<std::size_t>::max)() / a)
            throw std::overflow_error("Projection ring extent overflows");
        return a * b;
    }
    HostMoeProjectionRingPlan(const std::vector<std::pair<int, int>>& misses,
        const std::array<std::size_t, 3>& strides, unsigned parts, std::size_t arena,
        const std::array<std::size_t, 3>& source_bytes, int capacity, int experts)
        : slot_bytes(arena / 2) {
        if (!slot_bytes || !(parts & 7) || (parts & ~7u) || misses.empty()
            || misses.size() > max_misses || capacity < 1 || experts < 1)
            throw std::invalid_argument("Invalid projection ring geometry");
        for (int p = 0; p < 3; ++p) {
            if (!strides[p] || source_bytes[p] != mul(strides[p], experts))
                throw std::invalid_argument("Projection ring source extent differs");
            if (parts & (1u << p)) payload_bytes = add(payload_bytes, mul(strides[p], misses.size()));
        }
        const auto windows_needed = payload_bytes / slot_bytes + (payload_bytes % slot_bytes != 0);
        if (windows_needed > max_items || 3 * misses.size() + windows_needed > max_items)
            throw std::length_error("Projection ring metadata exceeds its bound");
        std::vector<int> slots, ids;
        slots.reserve(misses.size()); ids.reserve(misses.size());
        for (const auto& miss : misses) {
            if (miss.first < 0 || miss.first >= capacity || miss.second < 0 || miss.second >= experts)
                throw std::invalid_argument("Projection ring source/target exceeds bounds");
            slots.push_back(miss.first); ids.push_back(miss.second);
        }
        std::sort(slots.begin(), slots.end()); std::sort(ids.begin(), ids.end());
        if (std::adjacent_find(slots.begin(), slots.end()) != slots.end()
            || std::adjacent_find(ids.begin(), ids.end()) != ids.end())
            throw std::invalid_argument("Projection ring repeats a source/target");
        pieces.reserve(3 * misses.size() + windows_needed); windows.reserve(windows_needed);
        for (const auto& miss : misses) for (int part = 0; part < 3; ++part) {
            if (!(parts & (1u << part))) continue;
            for (std::size_t offset = 0; offset < strides[part];) {
                if (windows.empty() || windows.back().bytes == slot_bytes)
                    windows.push_back({pieces.size(), 0, 0});
                auto& window = windows.back();
                const auto count = (std::min)(strides[part] - offset, slot_bytes - window.bytes);
                pieces.push_back({miss.first, part, add(mul(miss.second, strides[part]), offset),
                    offset, window.bytes, count});
                ++window.count; window.bytes += count; offset += count;
            }
        }
    }
};

inline bool host_moe_overlap_spans_disjoint(std::uintptr_t a, std::size_t an,
    std::uintptr_t b, std::size_t bn) noexcept {
    const auto maximum = (std::numeric_limits<std::uintptr_t>::max)();
    return a && b && an && bn && an <= maximum - a && bn <= maximum - b
        && (a + an <= b || b + bn <= a);
}

// IDs refer to immutable tensors; absent sources are zero. The only permitted
// cross-cut dependency is prefix -> suffix. External leaves are intentionally
// absent from this list and are checked against physical storage by the caller.
struct HostMoeOverlapNode {
    std::uintptr_t id = 0;
    std::array<std::uintptr_t, 11> sources{};
};
inline bool host_moe_overlap_cut_valid(const std::vector<HostMoeOverlapNode>& nodes,
    std::size_t cut, std::uintptr_t activation, std::uintptr_t down) noexcept {
    if (!cut || cut >= nodes.size() || nodes.size() > 256
        || nodes[cut - 1].id != activation || nodes[cut].id != down) return false;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (!nodes[i].id) return false;
        for (std::size_t j = 0; j < i; ++j) if (nodes[j].id == nodes[i].id) return false;
        for (const auto source : nodes[i].sources) if (source)
            for (std::size_t j = i; j < nodes.size(); ++j)
                if (source == nodes[j].id) return false;
    }
    return true;
}
} // namespace tsg
