// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include "ggml_ops_host_moe_file_ring.h"
#include <cstdint>
#include <cstring>

namespace tsg {
constexpr std::size_t kHostMoeRingCoalesceReadBytes = std::size_t(16) << 20;
constexpr std::size_t kHostMoeRingCoalesceArenaBytes = std::size_t(72) << 20;
constexpr std::size_t kHostMoeRingCoalesceMaxItems = 4096;

inline bool host_moe_ring_coalesce_flag(const char* value) noexcept {
    return value && std::strcmp(value, "1") == 0;
}
inline bool host_moe_ring_coalesce_eligible(bool enabled, bool pooled_split_wide_cuda,
    int selected, std::size_t ram_budget, std::size_t payload, std::size_t arena,
    std::size_t misses) noexcept {
    return enabled && pooled_split_wide_cuda && selected == 8 && ram_budget == 0
        && payload > kHostMoeRingCoalesceArenaBytes && arena == kHostMoeRingCoalesceArenaBytes
        && misses > 0 && misses <= 288;
}

struct HostMoeRingTargetSpan {
    std::uintptr_t data = 0, buffer = 0;
    std::size_t bytes = 0, buffer_bytes = 0;
    bool view = false;
};
// Actual absolute spans, not distinct tensor pointers, establish order-neutral
// targets. Views are deliberately outside this experiment's initial contract.
inline bool host_moe_ring_targets_disjoint(const std::array<HostMoeRingTargetSpan, 3>& targets) noexcept {
    const auto limit = (std::numeric_limits<std::uintptr_t>::max)();
    for (std::size_t p = 0; p < targets.size(); ++p) {
        const auto& t = targets[p];
        if (t.view || !t.data || !t.buffer || !t.bytes || t.data < t.buffer
            || t.data - t.buffer > t.buffer_bytes || t.bytes > t.buffer_bytes - (t.data - t.buffer)
            || t.bytes > limit - t.data || t.buffer_bytes > limit - t.buffer) return false;
        for (std::size_t q = 0; q < p; ++q)
            if (t.data < targets[q].data + targets[q].bytes && targets[q].data < t.data + t.bytes)
                return false;
    }
    return true;
}

// Read and upload partitions are independent. A read-cap cut may cross an
// upload piece because the existing ring consumes only a fully read window.
struct HostMoeCoalescedFileRingPlan {
    using Piece = HostMoeFileRingPlan::Piece;
    struct Read { int part; std::size_t source_offset, staging_offset, bytes; };
    struct Window { std::size_t first_piece, piece_count, first_read, read_count, bytes; };
    std::vector<Piece> pieces;
    std::vector<Read> reads;
    std::vector<Window> windows;
    std::size_t slot_bytes = 0, payload_bytes = 0, coalesced_bytes = 0;
    std::array<std::uint64_t, 3> source_identity{};

    static std::size_t add(std::size_t a, std::size_t b) {
        if (b > (std::numeric_limits<std::size_t>::max)() - a)
            throw std::overflow_error("Coalesced ring byte extent overflows");
        return a + b;
    }
    static std::size_t mul(std::size_t a, std::size_t b) {
        if (a && b > (std::numeric_limits<std::size_t>::max)() / a)
            throw std::overflow_error("Coalesced ring extent overflows");
        return a * b;
    }
    static std::size_t ceil_div(std::size_t a, std::size_t b) { return a / b + (a % b != 0); }
    static bool metadata_fits(std::size_t misses, std::size_t payload, std::size_t slot) noexcept {
        if (!slot || !misses || misses > 288) return false;
        const auto windows = ceil_div(payload, slot);
        // Each added projection/window boundary adds at most one piece. Reads
        // add only cap cuts. Reserve exact bounded upper counts, never payload.
        return windows <= kHostMoeRingCoalesceMaxItems
            && 3 * misses + windows <= kHostMoeRingCoalesceMaxItems
            && 3 * misses + windows + ceil_div(payload, kHostMoeRingCoalesceReadBytes)
                <= kHostMoeRingCoalesceMaxItems;
    }

    HostMoeCoalescedFileRingPlan(const std::vector<std::pair<int, int>>& misses,
        const std::array<std::size_t, 3>& stride, std::size_t arena,
        const std::array<std::size_t, 3>& source_bytes,
        const std::array<std::uint64_t, 3>& identities, int capacity, int experts)
        : slot_bytes(arena / 2), source_identity(identities) {
        if (!slot_bytes || capacity < 1 || experts < 1 || misses.empty() || misses.size() > 288)
            throw std::invalid_argument("Invalid coalesced ring geometry");
        for (int p = 0; p < 3; ++p) {
            if (!stride[p] || !identities[p] || source_bytes[p] != mul(stride[p], experts))
                throw std::invalid_argument("Coalesced ring source generation/extent differs");
            payload_bytes = add(payload_bytes, mul(stride[p], misses.size()));
        }
        if (!metadata_fits(misses.size(), payload_bytes, slot_bytes))
            throw std::length_error("Coalesced ring metadata exceeds bounded route allowance");
        auto ordered = misses;
        std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
        std::vector<int> slots; slots.reserve(ordered.size());
        for (std::size_t i = 0; i < ordered.size(); ++i) {
            const auto& miss = ordered[i];
            if (miss.first < 0 || miss.first >= capacity || miss.second < 0 || miss.second >= experts
                || (i && ordered[i - 1].second == miss.second))
                throw std::invalid_argument("Coalesced ring repeats or exceeds a source/target");
            slots.push_back(miss.first);
            for (int p = 0; p < 3; ++p) {
                const auto source = mul(static_cast<std::size_t>(miss.second), stride[p]);
                if (source > source_bytes[p] || stride[p] > source_bytes[p] - source)
                    throw std::invalid_argument("Coalesced ring expert exceeds its source");
            }
        }
        std::sort(slots.begin(), slots.end());
        if (std::adjacent_find(slots.begin(), slots.end()) != slots.end())
            throw std::invalid_argument("Coalesced ring repeats a target slot");
        const auto count = ceil_div(payload_bytes, slot_bytes);
        pieces.reserve(3 * misses.size() + count);
        reads.reserve(3 * misses.size() + count + ceil_div(payload_bytes, kHostMoeRingCoalesceReadBytes));
        windows.reserve(count);
        for (int p = 0; p < 3; ++p) for (const auto& miss : ordered) {
            for (std::size_t offset = 0; offset < stride[p];) {
                if (windows.empty() || windows.back().bytes == slot_bytes)
                    windows.push_back({pieces.size(), 0, 0, 0, 0});
                auto& window = windows.back();
                const auto bytes = (std::min)(stride[p] - offset, slot_bytes - window.bytes);
                pieces.push_back({miss.first, p, add(mul(miss.second, stride[p]), offset),
                    offset, window.bytes, bytes});
                ++window.piece_count; window.bytes += bytes; offset += bytes;
            }
        }
        for (auto& window : windows) {
            window.first_read = reads.size();
            for (std::size_t first = window.first_piece; first < window.first_piece + window.piece_count;) {
                const auto& base = pieces[first];
                std::size_t bytes = base.bytes, end = first + 1;
                while (end < window.first_piece + window.piece_count) {
                    const auto& next = pieces[end];
                    if (next.part != base.part || next.source_offset != add(base.source_offset, bytes)
                        || next.staging_offset != add(base.staging_offset, bytes)) break;
                    bytes = add(bytes, next.bytes); ++end;
                }
                if (end - first > 1) coalesced_bytes = add(coalesced_bytes, bytes);
                for (std::size_t offset = 0; offset < bytes;) {
                    const auto length = (std::min)(kHostMoeRingCoalesceReadBytes, bytes - offset);
                    reads.push_back({base.part, add(base.source_offset, offset), add(base.staging_offset, offset), length});
                    offset += length;
                }
                first = end;
            }
            window.read_count = reads.size() - window.first_read;
        }
    }
};
// Three bounded arrays plus ordered route metadata stay below 1 MiB on this
// 64-bit build; this is transient metadata, not another arena or budget ticket.
static_assert(kHostMoeRingCoalesceMaxItems * (sizeof(HostMoeCoalescedFileRingPlan::Piece)
    + sizeof(HostMoeCoalescedFileRingPlan::Read) + sizeof(HostMoeCoalescedFileRingPlan::Window))
    + 288 * 16 < (std::size_t(1) << 20), "Coalesced ring metadata allowance exceeds 1 MiB");
} // namespace tsg
