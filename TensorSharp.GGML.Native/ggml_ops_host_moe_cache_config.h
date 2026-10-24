// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include "ggml_ops_host_moe_cache_policy.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace tsg {
struct HostMoeCacheSettings {
    std::size_t budget = 0, layers = 0;
    int max_rows = 0;
    // Planned live decode concurrency is independent of the prefill ceiling.
    // Reserve exact-width variants up front so N1 -> N2 does not evict weights.
    int max_decode_rows = 1;
    // Candidate only: old callers retain the existing allocation policy.
    bool capacity_growth = false;
};

// All operations except fast_budget require the enclosing native cache mutex.
// No allocation or environment mutation. An exclusive token prevents a stale
// scope from restoring defaults over a newer owner. Trim may throw: publish
// configuration only after it succeeds, retaining the original token for retry.
class HostMoeCacheConfiguration {
    const HostMoeCacheSettings defaults_;
    HostMoeCacheSettings current_;
    std::atomic<std::size_t> fast_budget_;
    std::uint64_t token_ = 0, next_;
    std::size_t ceiling_ = 0;
    std::uint64_t growth_generation_ = 0;
    // Nonowning dispatch only. Predictor payload/controller is model-owned and
    // must outlive this exclusive scope binding, including failed setup/ACK.
    void* child_cookie_ = nullptr;
    bool (*child_ready_)(void*) = nullptr;
    bool (*child_reset_)(void*) = nullptr;
public:
    explicit HostMoeCacheConfiguration(HostMoeCacheSettings defaults, std::uint64_t next = 1)
        : defaults_(defaults), current_(defaults), fast_budget_(defaults.budget), next_(next) {}
    std::size_t fast_budget() const noexcept { return fast_budget_.load(std::memory_order_relaxed); }
    const HostMoeCacheSettings& settings() const noexcept { return current_; }
    bool owns(std::uint64_t token) const noexcept { return token != 0 && token == token_; }
    bool active() const noexcept { return token_ != 0; }
    std::size_t maximum_budget() const noexcept { return ceiling_; }
    std::uint64_t growth_generation() const noexcept { return growth_generation_; }
    void* child_cookie() const noexcept { return child_cookie_; }
    std::uint64_t child_token(const void* cookie) const noexcept {
        return cookie && child_cookie_ == cookie ? token_ : 0;
    }
    bool bind_child(std::uint64_t token, void* cookie, bool (*ready)(void*), bool (*reset)(void*)) noexcept {
        if (!owns(token) || !cookie || !ready || !reset || (child_cookie_ && child_cookie_ != cookie)) return false;
        child_cookie_ = cookie; child_ready_ = ready; child_reset_ = reset; return true;
    }
    bool detach_child(std::uint64_t token, void* cookie) noexcept {
        if (!owns(token) || child_cookie_ != cookie || !child_ready_ || !child_ready_(cookie)) return false;
        child_cookie_ = nullptr; child_ready_ = child_reset_ = nullptr; return true;
    }
    bool reset_child() noexcept { return !child_cookie_ || (child_reset_ && child_reset_(child_cookie_)); }
    static bool valid(HostMoeCacheSettings value) noexcept {
        return value.budget <= static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)())
            && value.layers > 0 && value.max_rows > 0 && value.max_rows <= kHostMoeCacheMaxGlmPrefillRows
            && value.max_decode_rows > 0 && value.max_decode_rows <= kHostMoeCacheMaxDecodeRows;
    }
    std::uint64_t begin(HostMoeCacheSettings value, bool cache_empty) noexcept {
        if (token_ || child_cookie_ || !next_ || !cache_empty || !valid(value)) return 0;
        token_ = next_++;
        ceiling_ = value.budget;
        current_ = value;
        growth_generation_ = value.capacity_growth ? 1 : 0;
        fast_budget_.store(value.budget, std::memory_order_relaxed);
        return token_;
    }
    template<class Trim> bool update(std::uint64_t token, std::size_t budget, Trim trim) {
        if (!owns(token) || budget > ceiling_) return false;
        trim(budget, current_.layers);
        // A normal admission refusal must not repeat graph planning each token.
        // A later increased ceiling permits exactly one fresh optional attempt.
        if (current_.capacity_growth && budget > current_.budget
            && growth_generation_ != (std::numeric_limits<std::uint64_t>::max)()) ++growth_generation_;
        current_.budget = budget;
        fast_budget_.store(budget, std::memory_order_relaxed);
        return true;
    }
    template<class Release> bool end(std::uint64_t token, Release release) {
        if (!owns(token) || (child_cookie_ && (!child_ready_ || !child_ready_(child_cookie_)))) return false;
        release();
        current_ = defaults_;
        ceiling_ = 0;
        token_ = 0;
        growth_generation_ = 0;
        child_cookie_ = nullptr; child_ready_ = child_reset_ = nullptr;
        fast_budget_.store(defaults_.budget, std::memory_order_relaxed);
        return true;
    }
};
} // namespace tsg
