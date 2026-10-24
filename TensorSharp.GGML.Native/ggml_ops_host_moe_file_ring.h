// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tsg {
// Metadata only. Two windows share the existing charged staging arena; every
// source byte belongs to an exact selected expert, in the original route order.
struct HostMoeFileRingPlan {
    struct Piece {
        int slot, part;
        std::size_t source_offset, expert_offset, staging_offset, bytes;
    };
    struct Window { std::size_t first, count, bytes; };
    std::vector<Piece> pieces;
    std::vector<Window> windows;
    std::size_t slot_bytes;

    HostMoeFileRingPlan(const std::vector<std::pair<int, int>>& misses,
        const std::array<std::size_t, 3>& stride, std::size_t arena_bytes)
        : slot_bytes(arena_bytes / 2) {
        if (!slot_bytes) throw std::invalid_argument("Expert ring needs two nonempty slots");
        for (const auto bytes : stride)
            if (!bytes) throw std::invalid_argument("Expert ring has an empty projection");
        std::vector<int> slots, experts;
        for (const auto& miss : misses) {
            if (miss.first < 0 || miss.second < 0)
                throw std::invalid_argument("Expert ring has an invalid miss");
            for (const auto bytes : stride)
                if (static_cast<std::size_t>(miss.second) >= (std::numeric_limits<std::size_t>::max)() / bytes)
                    throw std::overflow_error("Expert ring source extent overflows");
            slots.push_back(miss.first); experts.push_back(miss.second);
        }
        std::sort(slots.begin(), slots.end()); std::sort(experts.begin(), experts.end());
        if (std::adjacent_find(slots.begin(), slots.end()) != slots.end()
            || std::adjacent_find(experts.begin(), experts.end()) != experts.end())
            throw std::invalid_argument("Expert ring repeats a destination or source");
        for (const auto& miss : misses) for (int part = 0; part < 3; ++part) {
            for (std::size_t offset = 0; offset < stride[part];) {
                if (windows.empty() || windows.back().bytes == slot_bytes)
                    windows.push_back({pieces.size(), 0, 0});
                auto& window = windows.back();
                const auto count = (std::min)(stride[part] - offset, slot_bytes - window.bytes);
                pieces.push_back({miss.first, part,
                    static_cast<std::size_t>(miss.second) * stride[part] + offset,
                    offset, window.bytes, count});
                ++window.count; window.bytes += count; offset += count;
            }
        }
    }
};
} // namespace tsg
