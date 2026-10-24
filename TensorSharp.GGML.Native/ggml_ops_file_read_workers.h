// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace tsg {
struct HostMoeFileReadWorkerPlan {
    unsigned requested = 0, workers = 1, maximum = 1, pool_slots = 1, default_pool_slots = 1;
};

// Auto preserves the old pool exactly. In pipelined mode its submitter consumes
// completions; only pool_slots-1 background threads prepare reads. Explicit1
// selects the existing serial pool, which prepares on the submitter instead.
inline HostMoeFileReadWorkerPlan plan_host_moe_file_read_workers(const char* value, int available) {
    if (available < 1) throw std::invalid_argument("File read workers need positive available parallelism");
    unsigned requested = 0;
    if (value) {
        if (!*value || (value[0] == '0' && value[1]))
            throw std::invalid_argument("TS_HOST_MOE_FILE_READ_WORKERS must be 0 or 1 through 16");
        for (const char* p = value; *p; ++p) {
            if (*p < '0' || *p > '9' || requested > 16 / 10
                || requested * 10 + unsigned(*p - '0') > 16)
                throw std::invalid_argument("TS_HOST_MOE_FILE_READ_WORKERS must be 0 or 1 through 16");
            requested = requested * 10 + unsigned(*p - '0');
        }
    }
    HostMoeFileReadWorkerPlan plan;
    plan.requested = requested;
    plan.default_pool_slots = (std::min)(16u, static_cast<unsigned>(available));
    plan.maximum = plan.default_pool_slots > 1 ? plan.default_pool_slots - 1 : 1;
    plan.workers = requested ? (std::min)(requested, plan.maximum) : plan.maximum;
    plan.pool_slots = requested ? (plan.workers == 1 ? 1 : plan.workers + 1) : plan.default_pool_slots;
    return plan;
}

// Caller-only accounting, updated after the existing synchronous worker join.
// No worker touches this TLS pointer. A failed invocation does not certify its
// partially read tasks or source bytes; only completely joined successes do.
struct HostMoeFileReadActivity {
    unsigned requested = 0, workers = 0, maximum = 0, pool_slots = 0;
    unsigned random_access = 0;
    std::uint64_t calls = 0, tasks = 0, completed_tasks = 0, failed_calls = 0, source_bytes = 0;
    std::uint64_t view_calls = 0, view_bytes = 0, view_fallbacks = 0, view_buffered_bytes = 0;
    std::uint64_t prefetch_calls = 0, prefetch_ranges = 0, prefetch_bytes = 0, prefetch_failures = 0;
    bool valid = true, source_hint_seen = false;
    void add_prefetch(std::uint64_t calls, std::uint64_t ranges, std::uint64_t bytes,
        std::uint64_t failures) noexcept {
        const auto increment = [&](std::uint64_t& target, std::uint64_t amount) {
            if (amount > UINT64_MAX - target) valid = false;
            else target += amount;
        };
        increment(prefetch_calls, calls); increment(prefetch_ranges, ranges);
        increment(prefetch_bytes, bytes); increment(prefetch_failures, failures);
    }
    void add_view(std::uint64_t attempts, std::uint64_t mapped, std::uint64_t fallbacks,
        std::uint64_t buffered) noexcept {
        const auto increment = [&](std::uint64_t& target, std::uint64_t amount) {
            if (amount > (std::numeric_limits<std::uint64_t>::max)() - target) valid = false;
            else target += amount;
        };
        increment(view_calls, attempts); increment(view_bytes, mapped);
        increment(view_fallbacks, fallbacks); increment(view_buffered_bytes, buffered);
    }
    void observe_source_hint(bool hinted, std::size_t read_bytes) noexcept {
        if (!read_bytes || !valid) return;
        const unsigned value = hinted ? 1 : 0;
        if (source_hint_seen && random_access != value) { valid = false; return; }
        random_access = value; source_hint_seen = true;
    }
    void add(const HostMoeFileReadWorkerPlan& plan, std::size_t count,
             std::uint64_t bytes, bool success) noexcept {
        if (!count || !valid) return;
        if (calls && (requested != plan.requested || workers != plan.workers
            || maximum != plan.maximum || pool_slots != plan.pool_slots)) { valid = false; return; }
        requested = plan.requested; workers = plan.workers; maximum = plan.maximum; pool_slots = plan.pool_slots;
        const auto increment = [&](std::uint64_t& target, std::uint64_t amount) {
            if (amount > (std::numeric_limits<std::uint64_t>::max)() - target) valid = false;
            else target += amount;
        };
        increment(calls, 1); increment(tasks, count);
        if (success) { increment(completed_tasks, count); increment(source_bytes, bytes); }
        else increment(failed_calls, 1);
    }
};
inline thread_local HostMoeFileReadActivity* g_host_moe_file_read_activity = nullptr;
} // namespace tsg
