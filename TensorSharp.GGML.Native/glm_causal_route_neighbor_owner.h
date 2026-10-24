// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include "glm_causal_route_neighbor.h"
#include "ggml_ops_shared_cache_budget.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <new>

namespace tsg_glm_neighbor {
inline int option() noexcept {
    static const int value = []() noexcept {
        const char* flag = std::getenv("TS_HOST_MOE_CAUSAL_ROUTE_NEIGHBOR");
        if (!flag || std::strcmp(flag, "0") == 0) return 0;
        return std::strcmp(flag, "1") == 0 ? 1 : -1;
    }();
    return value;
}
struct Source {
    std::array<const void*, 3> pointer{};
    std::array<std::uint64_t, 3> bytes{}, stride{};
    std::array<int, 3> type{};
    std::uint16_t layer = 0;
    bool matches(const std::array<const void*, 3>& other) const noexcept { return pointer == other; }
};
struct CallStats {
    std::uint64_t queries = 0, committed = 0, matches = 0, no_matches = 0;
    std::uint64_t compared = 0, min_lag = 0, history_max = 0, slot_checks = 0, victim_checks = 0;
    std::uint64_t prepare_ns = 0, commit_ns = 0;
    bool timing_valid = true;
};
struct Storage {
    State policy{};
    std::array<Source, max_layers> source{};
    std::array<std::int16_t, 128> source_bucket{};
};
static_assert(sizeof(Storage) <= 1024 * 1024, "Bounded causal-route Host payload must fit profile1");

class Owner {
    Storage* storage_ = nullptr;
    std::shared_ptr<tsg::SharedCacheCharge> charge_;
    std::uint64_t scope_ = 0, pending_epoch_ = 0;
    const void* model_ = nullptr;
    std::size_t declared_ = 0;
    bool requested_ = false, configured_ = false, closing_ = false, closed_ = false, failed_ = false;
    bool active_call_ = false, scalar_call_ = false, query_pending_ = false;
    std::uint64_t tokens_ = 0, rows_ = 0, epochs_ = 0, resets_ = 0, matches_ = 0, no_matches_ = 0;
    std::int64_t sequence_ = 0, expected_head_ = 0, call_head_ = 0;
    int call_rows_ = 0;
    bool have_head_ = false;
    CallStats call_{};
#if defined(TSG_GGML_TEST_HOOKS)
    int fault_ = 0;
    std::int64_t fault_count_ = 0;
    bool fail_allocate_ = false;
#endif
    static std::size_t source_hash(const std::array<const void*, 3>& source) noexcept {
        std::uintptr_t result = 0;
        for (auto pointer : source) {
            const auto value = reinterpret_cast<std::uintptr_t>(pointer);
            result ^= value + std::uintptr_t(0x9e3779b9) + (result << 6) + (result >> 2);
        }
        return static_cast<std::size_t>((result ^ (result >> 17)) & 127);
    }
    static bool counter_room(std::uint64_t value, std::uint64_t add = 1) noexcept {
        return value <= static_cast<std::uint64_t>(INT64_MAX) - add;
    }
public:
    Owner() = default;
    Owner(const Owner&) = delete;
    Owner& operator=(const Owner&) = delete;
    ~Owner() {
        // Public checked Free refuses BEFORE destructor entry if this owner is
        // live. An internal lifetime bug must not invoke the legacy void ACK.
        if (storage_ || charge_) std::terminate();
    }
    static constexpr std::size_t host_bytes() noexcept { return sizeof(Storage); }
    static bool timing_enabled() noexcept {
        static const bool enabled = []() noexcept {
            const char* value = std::getenv("TS_HOST_MOE_EXPERT_CACHE_TIMING");
            return value && std::strcmp(value, "1") == 0;
        }();
        return enabled;
    }
    void record_wall(bool commit, std::int64_t elapsed) noexcept {
        auto& target = commit ? call_.commit_ns : call_.prepare_ns;
        if (elapsed < 0 || !counter_room(target, static_cast<std::uint64_t>(elapsed))) {
            call_.timing_valid = false;
            return;
        }
        target += static_cast<std::uint64_t>(elapsed);
    }
    void request(bool value) noexcept { requested_ = value; }
    bool requested() const noexcept { return requested_; }
    bool configured() const noexcept { return configured_; }
    bool closed() const noexcept { return closed_; }
    bool ready() const noexcept { return requested_ && configured_ && storage_ && charge_
        && !closing_ && !closed_ && !failed_; }
    bool release_ready() const noexcept { return !storage_ && !charge_ && !active_call_ && !query_pending_; }
    bool active_call() const noexcept { return active_call_; }
    bool scalar_call() const noexcept { return scalar_call_; }
    bool query_pending() const noexcept { return query_pending_; }
    std::uint64_t scope() const noexcept { return scope_; }
    const void* model() const noexcept { return model_; }
    Storage* storage() noexcept { return storage_; }
    const Storage* storage() const noexcept { return storage_; }
    const CallStats& call_stats() const noexcept { return call_; }
    std::uint64_t epoch() const noexcept { return storage_ ? storage_->policy.epoch : epochs_; }
    std::size_t history_max() const noexcept {
        std::uint64_t result = 0;
        if (storage_) for (std::size_t i = 0; i < storage_->policy.config.layers; ++i)
            result = std::max(result, std::min<std::uint64_t>(history_rows, storage_->policy.layer[i].completed));
        return static_cast<std::size_t>(result);
    }
    bool configure(const void* model, std::uint64_t scope, int profile, std::size_t expected,
                   const Config& config, const Source* sources) noexcept {
        if (configured_) {
            if (!ready() || model_ != model || scope_ != scope || expected != declared_ || profile != 1
                || !sources || config.layers != storage_->policy.config.layers
                || config.experts != storage_->policy.config.experts || config.routes != storage_->policy.config.routes
                || config.layer_ids != storage_->policy.config.layer_ids) return false;
            for (std::size_t i = 0; i < config.layers; ++i) {
                const auto& old = storage_->source[i]; const auto& next = sources[i];
                if (old.pointer != next.pointer || old.bytes != next.bytes || old.stride != next.stride
                    || old.type != next.type || old.layer != next.layer) return false;
            }
            return true;
        }
        if (!requested_ || closing_ || closed_ || failed_ || !model || !scope || profile != 1
            || expected != host_bytes() || !valid_config(config) || !sources || !release_ready()
            || !tsg::SharedCacheCharge::host_accounting_attached()
            || !tsg::SharedCacheCharge::checked_release_attached()) return false;
        for (std::size_t i = 0; i < config.layers; ++i) {
            if (sources[i].layer != config.layer_ids[i]) return false;
            for (std::size_t p = 0; p < 3; ++p)
                if (!sources[i].pointer[p] || !sources[i].stride[p]
                    || sources[i].stride[p] > UINT64_MAX / config.experts
                    || sources[i].bytes[p] != sources[i].stride[p] * config.experts) return false;
            for (std::size_t j = 0; j < i; ++j) if (sources[i].pointer == sources[j].pointer) return false;
        }
        model_ = model; scope_ = scope; declared_ = expected;
        try {
            charge_ = tsg::SharedCacheCharge::reserve(0, 3, expected);
            if (!charge_ || !charge_->is_accounted() || charge_->reserved_bytes() != expected) { failed_ = true; return false; }
#if defined(TSG_GGML_TEST_HOOKS)
            if (fail_allocate_) { fail_allocate_ = false; failed_ = true; return false; }
#endif
            storage_ = new(std::nothrow) Storage();
            if (!storage_ || !initialize(storage_->policy, config)) { failed_ = true; return false; }
            storage_->source_bucket.fill(-1);
            for (std::size_t i = 0; i < config.layers; ++i) {
                storage_->source[i] = sources[i];
                auto bucket = source_hash(sources[i].pointer);
                while (storage_->source_bucket[bucket] >= 0) bucket = (bucket + 1) & 127;
                storage_->source_bucket[bucket] = static_cast<std::int16_t>(i);
            }
            if (!charge_->commit(expected)) { failed_ = true; return false; }
            configured_ = true; epochs_ = 1;
            return true;
        } catch (...) { failed_ = true; return false; }
    }
    int find_rank(const std::array<const void*, 3>& sources) const noexcept {
        if (!storage_) return -1;
        auto bucket = source_hash(sources);
        for (std::size_t probes = 0; probes < 128; ++probes, bucket = (bucket + 1) & 127) {
            const auto rank = storage_->source_bucket[bucket];
            if (rank < 0) return -1;
            if (storage_->source[static_cast<std::size_t>(rank)].matches(sources)) return rank;
        }
        return -1;
    }
    bool reset() noexcept {
        if (!storage_ || closed_ || closing_ || !counter_room(epochs_) || !counter_room(resets_)) return false;
        if (!reset_history(storage_->policy)) { failed_ = true; return false; }
        if (active_call_ && scalar_call_ && call_.queries) {
            // Maintenance invalidates the whole partial learning token. Keep
            // actual work/row counts, but grant no matched-query credit to a
            // token whose earlier labels/maps have deliberately been cleared.
            call_.matches = 0; call_.no_matches = call_.queries;
            call_.min_lag = call_.history_max = 0;
        }
        ++epochs_; ++resets_;
        return true;
    }
    bool maintenance_reset() noexcept {
        if (!storage_) return !active_call_ && !query_pending_;
        if (!storage_->policy.pending.prepared && !history_max() && !storage_->policy.cycle) return true;
        return reset();
    }
    bool begin_call(bool scalar, std::int64_t head = 0, int rows = 1, std::int64_t sequence = 0) noexcept {
        if (!ready() || active_call_ || query_pending_ || !counter_room(tokens_) || !counter_room(rows_, max_layers)
            || !counter_room(matches_, max_layers) || !counter_room(no_matches_, max_layers)
            || head < 0 || rows < 1 || head > INT64_MAX - rows) return false;
        if (have_head_ && (sequence_ != sequence || expected_head_ != head) && !reset()) return false;
        sequence_ = sequence; call_head_ = head; call_rows_ = rows;
        call_ = {}; active_call_ = true; scalar_call_ = scalar;
        if (scalar) {
            if (!begin_token(storage_->policy)) { active_call_ = false; return false; }
        } else if (!reset()) { active_call_ = false; return false; }
        return true;
    }
    bool prepare_query(std::size_t rank, const std::int32_t* ids, std::size_t count) noexcept {
        if (!ready() || !active_call_ || !scalar_call_ || query_pending_
            || !prepare(storage_->policy, rank, ids, count)) return false;
        query_pending_ = true; pending_epoch_ = storage_->policy.epoch;
        const auto& query = storage_->policy.pending;
        ++call_.queries; call_.compared += query.compared;
        if (query.matched) {
            ++call_.matches;
            call_.min_lag = call_.min_lag ? std::min(call_.min_lag, query.min_completed_lag) : query.min_completed_lag;
        } else ++call_.no_matches;
        return true;
    }
    bool score_slot(std::size_t slot, const std::array<const void*, 3>& source, int expert) noexcept {
        if (!ready() || !query_pending_ || slot >= max_slots) return false;
        ++call_.slot_checks;
        if (expert < 0) { storage_->policy.slot_due[slot] = unknown_due; return true; }
        const int rank = find_rank(source);
        if (rank < 0 || expert >= storage_->policy.config.experts) return false;
        storage_->policy.slot_due[slot] = predicted_due(storage_->policy, static_cast<std::size_t>(rank), static_cast<std::size_t>(expert));
        return true;
    }
    bool prefer_slot(std::size_t a, int expert_a, std::uint64_t age_a,
                     std::size_t b, int expert_b, std::uint64_t age_b) noexcept {
        ++call_.victim_checks;
        return prefer(expert_a < 0, storage_->policy.slot_due[a], age_a, a,
                      expert_b < 0, storage_->policy.slot_due[b], age_b, b);
    }
    bool commit_query(std::size_t rank) noexcept {
        if (!ready() || !active_call_ || !scalar_call_ || !query_pending_
            || !commit(storage_->policy, pending_epoch_, rank)) return false;
        query_pending_ = false; ++call_.committed; ++rows_;
        call_.history_max = history_max();
        return true;
    }
    bool finish_call() noexcept {
        if (!ready() || !active_call_ || query_pending_) return false;
        if (scalar_call_) {
            if (!finish_token(storage_->policy)) return false;
            ++tokens_; matches_ += call_.matches; no_matches_ += call_.no_matches;
        } else if (call_.queries || call_.committed) return false;
        active_call_ = scalar_call_ = false;
        expected_head_ = call_head_ + call_rows_; have_head_ = true;
        return true;
    }
    void fail_call() noexcept {
        failed_ = true; active_call_ = scalar_call_ = query_pending_ = false;
        have_head_ = false;
        if (storage_) {
            if (!abort_token(storage_->policy)) {
                // An exhausted epoch is terminal, but its pending label must
                // still never become a completed historical observation.
                for (auto& layer : storage_->policy.layer) layer = {};
                storage_->policy.pending = {};
                storage_->policy.slot_due.fill(unknown_due);
                storage_->policy.active = storage_->policy.skip_token = false;
                storage_->policy.next_rank = 0;
                storage_->policy.cycle = 0;
            }
            if (counter_room(epochs_)) ++epochs_;
            if (counter_room(resets_)) ++resets_;
        }
    }
    bool recover_reset() noexcept {
        if (!configured_ || !storage_ || closing_ || closed_ || active_call_ || query_pending_) return false;
        failed_ = false;
        have_head_ = false;
        if (!reset()) { failed_ = true; return false; }
        return true;
    }
    bool close() noexcept {
        if (closed_) return release_ready();
        closing_ = true;
        if (active_call_ || query_pending_) return false;
        delete storage_; storage_ = nullptr;
        try {
            if (charge_ && !charge_->release_checked()) return false;
            charge_.reset(); closing_ = false; closed_ = true;
            return true;
        } catch (...) { return false; }
    }
    bool info(std::int64_t* values, int count) const noexcept {
        if (!values || count != 16) return false;
        const std::uint64_t fields[16] = {1, requested_, configured_, closing_, closed_, failed_, declared_,
            storage_ ? host_bytes() : 0, charge_ ? charge_->reserved_bytes() : 0, query_pending_,
            tokens_, rows_, epochs_, resets_, matches_, no_matches_};
        for (std::size_t i = 0; i < 16; ++i) if (fields[i] > INT64_MAX) return false;
        for (std::size_t i = 0; i < 16; ++i) values[i] = static_cast<std::int64_t>(fields[i]);
        return true;
    }
#if defined(TSG_GGML_TEST_HOOKS)
    bool test_fault(int stage) noexcept { if (stage < 0 || stage > 3) return false; fault_ = stage; return true; }
    bool test_fire(int stage) noexcept {
        if (stage <= 0 || stage != fault_ || fault_count_ == INT64_MAX) return false;
        fault_ = 0; ++fault_count_; return true;
    }
    std::int64_t test_failure_count() const noexcept { return fault_count_; }
    void test_fail_allocation() noexcept { fail_allocate_ = true; }
#endif
};

// A null owner or disabled existing cache-timing option performs no clock read.
// These diagnostic intervals are nested host wall spans, not CPU-exclusive time.
class WallTimer {
    Owner* owner_;
    bool commit_;
    std::chrono::steady_clock::time_point start_{};
public:
    explicit WallTimer(Owner* owner, bool commit = false) noexcept
        : owner_(owner && owner->scalar_call() && Owner::timing_enabled() ? owner : nullptr), commit_(commit) {
        if (owner_) start_ = std::chrono::steady_clock::now();
    }
    void finish() noexcept {
        if (!owner_) return;
        owner_->record_wall(commit_, std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start_).count());
        owner_ = nullptr;
    }
    ~WallTimer() noexcept { finish(); }
    WallTimer(const WallTimer&) = delete;
    WallTimer& operator=(const WallTimer&) = delete;
};
} // namespace tsg_glm_neighbor
