// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>

namespace tsg {

// Diagnostic host wall intervals only. Different stages may nest/overlap;
// neither enqueue time nor a backend wait is a CUDA kernel duration.
enum class HostMoeCacheStage : std::size_t {
    plan, allocation, wide_replace, file_pipeline, upload_enqueue, upload_wait,
    input_bridge, compute, output_bridge, count
};

struct HostMoeCacheTiming {
    static constexpr auto stages = static_cast<std::size_t>(HostMoeCacheStage::count);
    std::array<std::uint64_t, stages> nanoseconds{}, calls{};
    std::array<unsigned, stages> depth{};
    std::uint64_t file_bytes = 0, file_windows = 0, graph_submissions = 0;
    std::uint64_t ring_calls = 0, ring_expanded_calls = 0;
    std::uint64_t ring_arena_peak_bytes = 0, ring_slot_peak_bytes = 0;
    std::uint64_t ring_coalesced_calls = 0, ring_read_tasks_before = 0, ring_read_tasks_after = 0;
    std::uint64_t ring_coalesced_payload_bytes = 0, ring_coalesced_bytes = 0, ring_coalesced_read_cap_bytes = 0;
    // Requested is the process option; remaining fields certify completed
    // source-copy/upload invocations, not physical read issue order or SSD I/O.
    std::uint64_t file_sort_requested = 0, file_sort_calls = 0, file_sort_changed_calls = 0;
    std::uint64_t file_sort_misses = 0, file_sort_bytes = 0;
    std::uint64_t file_sort_inversions_before = 0, file_sort_inversions_after = 0;
    std::uint64_t file_view_requested = 0, file_prefetch_requested = 0;
    std::uint64_t vmm_profile = 0, vmm_capacity = 0, vmm_physical_bytes = 0;
    std::uint64_t vmm_pending_bytes = 0, vmm_quarantine_bytes = 0, vmm_certificate = 0;

    static bool enabled(const char* value) noexcept {
        return value && std::strcmp(value, "1") == 0;
    }
    static void add(std::uint64_t& target, std::uint64_t amount) noexcept {
        const auto maximum = (std::numeric_limits<std::uint64_t>::max)();
        target = amount > maximum - target ? maximum : target + amount;
    }
    void record(HostMoeCacheStage stage, std::int64_t elapsed_ns) noexcept {
        const auto index = static_cast<std::size_t>(stage);
        add(calls[index], 1);
        if (elapsed_ns > 0) add(nanoseconds[index], static_cast<std::uint64_t>(elapsed_ns));
    }
    double milliseconds(HostMoeCacheStage stage) const noexcept {
        return nanoseconds[static_cast<std::size_t>(stage)] / 1e6;
    }
};

// A null owner performs no clock read. Recursive planning regions of the same
// stage count only their outer interval; different stages stay explicitly
// nested rather than being presented as an additive latency breakdown.
template<class Clock = std::chrono::steady_clock>
class HostMoeCacheStageTimer {
    HostMoeCacheTiming* owner_;
    HostMoeCacheStage stage_;
    bool outer_ = false;
    typename Clock::time_point start_{};
public:
    HostMoeCacheStageTimer(HostMoeCacheTiming* owner, HostMoeCacheStage stage) noexcept
        : owner_(owner), stage_(stage) {
        if (owner_) {
            auto& depth = owner_->depth[static_cast<std::size_t>(stage_)];
            outer_ = depth++ == 0;
            if (outer_) start_ = Clock::now();
        }
    }
    ~HostMoeCacheStageTimer() noexcept {
        if (!owner_) return;
        --owner_->depth[static_cast<std::size_t>(stage_)];
        if (outer_) owner_->record(stage_,
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_).count());
    }
    HostMoeCacheStageTimer(const HostMoeCacheStageTimer&) = delete;
    HostMoeCacheStageTimer& operator=(const HostMoeCacheStageTimer&) = delete;
};

} // namespace tsg
