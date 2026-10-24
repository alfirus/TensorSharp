// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include "ggml_ops_file_source.h"
#include "ggml_ops_host_moe_immutable_cache.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <new>
#if defined(_WIN32) && defined(_MSC_VER)
#include <psapi.h>
#endif

namespace tsg {
#define TSG_MAPPED_RAM_METRICS(X) \
    X(plan_calls) X(plan_nanoseconds) X(admission_calls) X(admission_successes) X(admission_nanoseconds) \
    X(quota_claim_calls) X(quota_claim_wait_nanoseconds) \
    X(quota_read_calls) X(quota_read_nanoseconds) X(quota_write_calls) X(quota_write_nanoseconds) \
    X(quota_expand_writes) X(quota_restore_writes) X(quota_reuse_claims) \
    X(memory_query_calls) X(memory_query_nanoseconds) \
    X(duplicate_calls) X(duplicate_nanoseconds) X(section_calls) X(section_nanoseconds) \
    X(map_calls) X(map_nanoseconds) X(protection_query_calls) X(protection_query_nanoseconds) \
    X(lock_calls) X(lock_nanoseconds) X(page_query_calls) X(page_query_nanoseconds) \
    X(retire_calls) X(retire_nanoseconds) \
    X(guarded_copy_calls) X(guarded_copy_bytes) X(guarded_copy_failures) X(guarded_copy_nanoseconds) \
    X(peak_claim_bytes) X(quota_min_high_water_bytes)
// Optional immutable CPU source storage. VirtualLock is NOT CUDA pinning: a hit
// must guarded-copy into the existing charged staging arena before H2D.
class HostMoeMappedRam {
public:
    static constexpr std::size_t entry_metadata_bytes = 4096;
    static constexpr std::size_t control_metadata_bytes = 65536;
    static constexpr std::size_t max_entries = 4096;
    struct Options {
        bool exclusive_quota = false;
        std::size_t quota_ceiling_bytes = 0;
        std::size_t quota_headroom_bytes = std::size_t(16) << 20;
    };
    struct Geometry {
        bool valid = false;
        std::uint64_t map_offset = 0;
        std::size_t map_bytes = 0, data_delta = 0, lock_delta = 0, pin_bytes = 0, charge_bytes = 0;
    };
    struct Stats {
        std::size_t locked_page_bytes = 0, quarantine_charge_bytes = 0, owned_entries = 0;
        bool quota_restore_pending = false, quota_conflict = false;
        std::uint64_t cleanup_failures = 0;
#define TSG_MAPPED_RAM_STAT(name) std::uint64_t name = 0;
        TSG_MAPPED_RAM_METRICS(TSG_MAPPED_RAM_STAT)
#undef TSG_MAPPED_RAM_STAT
    };
    enum class TestFault { none, allocation, duplicate, section, map, query, lock, locked_query, copy, in_page,
        unlock, unmap, close_section, close_file, restore, quota_set, restore_query };

    static Geometry checked_geometry(std::uint64_t registered_offset, std::uint64_t registered_bytes,
        std::uint64_t file_bytes, std::uint64_t relative, std::size_t extent,
        std::size_t page, std::size_t granularity) noexcept {
        Geometry g;
        const auto max = (std::numeric_limits<std::uint64_t>::max)();
        if (!extent || !page || (page & (page - 1)) || !granularity || granularity % page
            || (granularity & (granularity - 1)) || relative > registered_bytes
            || extent > registered_bytes - relative || registered_offset > max - registered_bytes
            || registered_offset + registered_bytes > file_bytes) return g;
        const auto begin = registered_offset + relative, end = begin + extent;
        if (end > max - (page - 1)) return g;
        const auto lock_begin = begin & ~std::uint64_t(page - 1);
        const auto lock_end = (end + page - 1) & ~std::uint64_t(page - 1);
        const auto map_begin = begin & ~std::uint64_t(granularity - 1);
        // Initial contract refuses EOF padding rather than assuming its accessibility.
        if (lock_end > file_bytes || end - map_begin > SIZE_MAX || begin - map_begin > SIZE_MAX
            || lock_begin - map_begin > SIZE_MAX || lock_end - lock_begin > SIZE_MAX - entry_metadata_bytes) return g;
        g.map_offset = map_begin; g.map_bytes = static_cast<std::size_t>(end - map_begin);
        g.data_delta = static_cast<std::size_t>(begin - map_begin);
        g.lock_delta = static_cast<std::size_t>(lock_begin - map_begin);
        g.pin_bytes = static_cast<std::size_t>(lock_end - lock_begin);
        g.charge_bytes = g.pin_bytes + entry_metadata_bytes; g.valid = true; return g;
    }

private:
    using Clock = std::chrono::steady_clock;
    struct Metrics {
#define TSG_MAPPED_RAM_ATOMIC(name) std::atomic<std::uint64_t> name{0};
        TSG_MAPPED_RAM_METRICS(TSG_MAPPED_RAM_ATOMIC)
#undef TSG_MAPPED_RAM_ATOMIC
    };
    // Process-lifetime monotonic counters survive configure/shrink/finish_zero.
    static Metrics& metrics() noexcept { static Metrics value; return value; }
    static std::uint64_t elapsed(Clock::time_point start) noexcept {
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
    }
    static void peak(std::atomic<std::uint64_t>& value, std::uint64_t next) noexcept {
        auto previous = value.load(std::memory_order_relaxed);
        while (previous < next && !value.compare_exchange_weak(previous, next, std::memory_order_relaxed)) {}
    }
    struct Cost {
        std::atomic<std::uint64_t>& nanoseconds;
        Clock::time_point start = Clock::now();
        Cost(std::atomic<std::uint64_t>& calls, std::atomic<std::uint64_t>& ns) noexcept : nanoseconds(ns) {
            calls.fetch_add(1, std::memory_order_relaxed);
        }
        ~Cost() {
#if defined(_WIN32)
            const auto error = GetLastError();
#endif
            nanoseconds.fetch_add(elapsed(start), std::memory_order_relaxed);
#if defined(_WIN32)
            SetLastError(error);
#endif
        }
    };
    template<class Function>
    static auto measured(std::atomic<std::uint64_t>& calls, std::atomic<std::uint64_t>& ns,
        Function&& function) noexcept -> decltype(function()) {
        Cost cost(calls, ns); return function();
    }
    static std::atomic<int>& fault() noexcept { static std::atomic<int> value{0}; return value; }
    static bool fail(TestFault value) noexcept {
#ifdef TSG_GGML_TEST_HOOKS
        int expected = static_cast<int>(value);
        return fault().compare_exchange_strong(expected, 0);
#else
        (void)value; return false;
#endif
    }
#if defined(_WIN32) && defined(_MSC_VER)
    struct Quota { SIZE_T low = 0, high = 0; DWORD flags = 0; };
    static bool quota(Quota& q) noexcept {
        Cost cost(metrics().quota_read_calls, metrics().quota_read_nanoseconds);
        return GetProcessWorkingSetSizeEx(GetCurrentProcess(), &q.low, &q.high, &q.flags) != FALSE;
    }
    static bool write_quota(const Quota& q, bool restore) noexcept {
        Cost cost(metrics().quota_write_calls, metrics().quota_write_nanoseconds);
        const bool ok = SetProcessWorkingSetSizeEx(GetCurrentProcess(), q.low, q.high, q.flags) != FALSE;
        if (ok) {
            (restore ? metrics().quota_restore_writes : metrics().quota_expand_writes).fetch_add(1, std::memory_order_relaxed);
            peak(metrics().quota_min_high_water_bytes, q.low);
        }
        return ok;
    }
    static bool equal(const Quota& a, const Quota& b) noexcept {
        return a.low == b.low && a.high == b.high && a.flags == b.flags;
    }
    static bool valid_flags(DWORD flags) noexcept {
        return !(flags & ~DWORD(15)) && (flags & 3) != 3 && (flags & 12) != 12;
    }
    using MemoryQuery = BOOL (WINAPI*)(HANDLE, PPROCESS_MEMORY_COUNTERS, DWORD);
    using PageQuery = BOOL (WINAPI*)(HANDLE, PVOID, DWORD);
    static MemoryQuery memory_query() noexcept {
        return reinterpret_cast<MemoryQuery>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "K32GetProcessMemoryInfo"));
    }
    static PageQuery page_query() noexcept {
        return reinterpret_cast<PageQuery>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "K32QueryWorkingSetEx"));
    }
    struct State;
    struct Node {
        enum class Stage { pending, live, retiring, quarantine } stage = Stage::pending;
        std::shared_ptr<State> state;
        std::shared_ptr<void> lifetime;
        Geometry geometry;
        std::size_t slot = max_entries;
        HANDLE file = INVALID_HANDLE_VALUE, section = nullptr;
        void* view = nullptr;
        bool claim = false, lock_attempted = false, counted_locked = false, quarantine_counted = false;
        Node(std::shared_ptr<State> s, const std::shared_ptr<void>& l, const Geometry& g)
            : state(std::move(s)), lifetime(l), geometry(g) {}
    };
    struct State {
        std::mutex mutex;
        Options options;
        Quota original, last;
        std::shared_ptr<void> control_lifetime;
        std::array<Node*, max_entries> nodes{};
        std::size_t claims = 0, locked = 0, possible_locked = 0, entries = 0, quarantines = 0, hint = 0;
        std::uint64_t cleanup_failures = 0;
        bool accepting = true, quota_written = false, conflict = false, restore_failed = false;
        explicit State(const Options& o, const std::shared_ptr<void>& lifetime) : options(o), control_lifetime(lifetime) {}

        bool insert(Node* n) noexcept {
            std::lock_guard<std::mutex> lock(mutex);
            if (!accepting || quarantines || entries == max_entries) return false;
            for (std::size_t i = 0; i < max_entries; ++i) {
                const auto slot = (hint + i) % max_entries;
                if (!nodes[slot]) { nodes[slot] = n; n->slot = slot; hint = (slot + 1) % max_entries; ++entries; return true; }
            }
            return false;
        }
        bool claim(Node& n) noexcept {
            const auto waiting = Clock::now();
            metrics().quota_claim_calls.fetch_add(1, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(mutex);
            metrics().quota_claim_wait_nanoseconds.fetch_add(elapsed(waiting), std::memory_order_relaxed);
            if (!accepting || quarantines || n.geometry.pin_bytes > SIZE_MAX - claims) return false;
            const auto next = claims + n.geometry.pin_bytes;
            if (options.exclusive_quota) {
                Quota current;
                if (!quota(current)) return false;
                if (!equal(current, last)) { conflict = true; accepting = false; return false; }
                PROCESS_MEMORY_COUNTERS memory{}; memory.cb = sizeof(memory);
                auto query = memory_query();
                if (!query || !measured(metrics().memory_query_calls, metrics().memory_query_nanoseconds,
                    [&] { return query(GetCurrentProcess(), &memory, sizeof(memory)); })) return false;
                const auto headroom = options.quota_headroom_bytes;
                const auto other = memory.WorkingSetSize - (std::min)(memory.WorkingSetSize, locked);
                if (next > SIZE_MAX - headroom || other > SIZE_MAX - next - headroom) return false;
                // Keep only an already verified historical entitlement. There is
                // no speculative slack/batching: every increase still needs exact
                // pending Host-backed claims plus the configured headroom.
                Quota wanted{(std::max)({last.low, original.low, next + headroom}),
                    (std::max)({last.high, memory.WorkingSetSize, other + next + headroom}), original.flags};
                wanted.high = (std::max)(wanted.high, wanted.low);
                if (wanted.high > options.quota_ceiling_bytes) return false;
                if (!equal(wanted, last)) {
                    if (fail(TestFault::quota_set) || !write_quota(wanted, false)) return false;
                    last = wanted; quota_written = !equal(last, original);
                    // A successful write already owns quota: retain its claim/token
                    // even if readback fails or reveals an ownership conflict.
                    claims = next; n.claim = true;
                    peak(metrics().peak_claim_bytes, claims);
                    Quota actual;
                    if (!quota(actual)) return false;
                    if (!equal(actual, wanted)) { conflict = true; accepting = false; return false; }
                } else metrics().quota_reuse_claims.fetch_add(1, std::memory_order_relaxed);
            }
            claims = next; n.claim = true; peak(metrics().peak_claim_bytes, claims); return true;
        }
        bool restore_locked() noexcept {
            if (!quota_written) return true;
            Quota current;
            if (!quota(current)) { restore_failed = true; return false; }
            if (!equal(current, last)) { conflict = true; accepting = false; restore_failed = true; return false; }
            if (fail(TestFault::restore) || !write_quota(original, true)) {
                restore_failed = true; return false;
            }
            last = original; // A retry must compare against the successful restoration write.
            Quota restored;
            if (fail(TestFault::restore_query) || !quota(restored) || !equal(restored, original)) {
                // Last already records our successful restoration write. Retain the
                // claim until readback verifies it; retry still compares before writing.
                restore_failed = true; accepting = false; return false;
            }
            last = original; quota_written = restore_failed = false; return true;
        }
        bool release_claim(Node& n) noexcept {
            std::lock_guard<std::mutex> lock(mutex);
            if (!n.claim) return true;
            if (claims < n.geometry.pin_bytes) return false;
            if (claims == n.geometry.pin_bytes && !restore_locked()) return false;
            claims -= n.geometry.pin_bytes; n.claim = false; return true;
        }
        void locked_state(Node& n, bool value) noexcept {
            std::lock_guard<std::mutex> lock(mutex);
            if (n.counted_locked == value) return;
            if (value) locked += n.geometry.pin_bytes; else locked -= n.geometry.pin_bytes;
            n.counted_locked = value;
        }
        void possible_lock(Node& n, bool value) noexcept {
            std::lock_guard<std::mutex> lock(mutex);
            if (n.lock_attempted == value) return;
            if (value) possible_locked += n.geometry.pin_bytes; else possible_locked -= n.geometry.pin_bytes;
            n.lock_attempted = value;
        }
        void failure() noexcept { std::lock_guard<std::mutex> lock(mutex); ++cleanup_failures; }
    };
    static_assert(sizeof(State) + sizeof(Metrics) < control_metadata_bytes, "Broker control metadata exceeds its reserved envelope");

    static bool readonly_interval(const Node& n) noexcept {
        auto begin = reinterpret_cast<std::uintptr_t>(n.view) + n.geometry.lock_delta;
        const auto end = begin + n.geometry.pin_bytes;
        if (end < begin || fail(TestFault::query)) return false;
        while (begin < end) {
            MEMORY_BASIC_INFORMATION info{};
            if (!measured(metrics().protection_query_calls, metrics().protection_query_nanoseconds,
                    [&] { return VirtualQuery(reinterpret_cast<void*>(begin), &info, sizeof(info)); }) || info.State != MEM_COMMIT
                || info.Type != MEM_MAPPED || info.Protect != PAGE_READONLY) return false;
            const auto next = reinterpret_cast<std::uintptr_t>(info.BaseAddress) + info.RegionSize;
            if (next <= begin) return false;
            begin = (std::min)(next, end);
        }
        return true;
    }
    static bool observed_locked(const Node& n) noexcept {
        auto query = page_query();
        if (!query || fail(TestFault::locked_query)) return false;
        std::array<PSAPI_WORKING_SET_EX_INFORMATION, 64> batch{};
        const auto pages = n.geometry.pin_bytes / 4096;
        const auto base = reinterpret_cast<std::uintptr_t>(n.view) + n.geometry.lock_delta;
        for (std::size_t i = 0; i < pages;) {
            const auto count = (std::min)(batch.size(), pages - i);
            for (std::size_t j = 0; j < count; ++j) batch[j].VirtualAddress = reinterpret_cast<void*>(base + (i + j) * 4096);
            if (!measured(metrics().page_query_calls, metrics().page_query_nanoseconds,
                [&] { return query(GetCurrentProcess(), batch.data(), static_cast<DWORD>(count * sizeof(batch[0]))); })) return false;
            for (std::size_t j = 0; j < count; ++j)
                if (!batch[j].VirtualAttributes.Valid || !batch[j].VirtualAttributes.Locked
                    || batch[j].VirtualAttributes.Win32Protection != PAGE_READONLY) return false;
            i += count;
        }
        return true;
    }
    static bool cleanup(Node& n) noexcept {
        Cost cost(metrics().retire_calls, metrics().retire_nanoseconds);
        auto state = n.state;
        if (n.view) {
            if (n.lock_attempted) {
                const bool ok = !fail(TestFault::unlock) && VirtualUnlock(
                    static_cast<std::uint8_t*>(n.view) + n.geometry.lock_delta, n.geometry.pin_bytes);
                if (ok) { state->possible_lock(n, false); state->locked_state(n, false); }
                else if (n.counted_locked) state->failure(); // Unmap may still safely retire the lock.
            }
            if (fail(TestFault::unmap) || !UnmapViewOfFile(n.view)) return false;
            n.view = nullptr; state->possible_lock(n, false); state->locked_state(n, false);
        }
        if (n.section) {
            if (fail(TestFault::close_section) || !CloseHandle(n.section)) return false;
            n.section = nullptr;
        }
        if (n.file != INVALID_HANDLE_VALUE) {
            if (fail(TestFault::close_file) || !CloseHandle(n.file)) return false;
            n.file = INVALID_HANDLE_VALUE;
        }
        return state->release_claim(n);
    }
    static void finish_retire(Node* n) noexcept {
        auto state = n->state; // Keep broker/control ownership alive after deleting the final Node.
        const bool ok = cleanup(*n);
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (ok) {
                if (n->quarantine_counted) --state->quarantines;
                state->nodes[n->slot] = nullptr; --state->entries;
            } else {
                n->stage = Node::Stage::quarantine;
                if (!n->quarantine_counted) { ++state->quarantines; n->quarantine_counted = true; }
                ++state->cleanup_failures;
            }
        }
        if (ok) delete n; // External charge/usage callbacks only after physical/quota cleanup, outside all locks.
    }
    static void retire(Node* n) noexcept {
        {
            std::lock_guard<std::mutex> lock(n->state->mutex);
            n->stage = Node::Stage::retiring;
        }
        finish_retire(n);
    }
    std::shared_ptr<State> state_;
    std::shared_ptr<State> state() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_); return state_;
    }
#endif
    mutable std::mutex mutex_;
    bool configuring_ = false;
    HostMoeMappedRam() = default;
public:
    HostMoeMappedRam(const HostMoeMappedRam&) = delete;
    HostMoeMappedRam& operator=(const HostMoeMappedRam&) = delete;
    static HostMoeMappedRam& instance() { static HostMoeMappedRam value; return value; }

    bool configure(const Options& options, const std::shared_ptr<void>& control_lifetime) noexcept {
#if defined(_WIN32) && defined(_MSC_VER)
        if (!control_lifetime || (options.exclusive_quota != (options.quota_ceiling_bytes != 0))
            || (options.exclusive_quota && (!options.quota_headroom_bytes
                || options.quota_headroom_bytes > options.quota_ceiling_bytes))) return false;
        std::shared_ptr<State> existing;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (configuring_) return false;
            existing = state_;
            if (!existing) configuring_ = true;
        }
        if (existing) {
            std::lock_guard<std::mutex> lock(existing->mutex);
            return existing->accepting && !existing->conflict && !existing->quarantines
                && existing->options.exclusive_quota == options.exclusive_quota
                && existing->options.quota_ceiling_bytes == options.quota_ceiling_bytes
                && existing->options.quota_headroom_bytes == options.quota_headroom_bytes
                && existing->control_lifetime.get() == control_lifetime.get()
                && !existing->control_lifetime.owner_before(control_lifetime)
                && !control_lifetime.owner_before(existing->control_lifetime);
        }
        std::shared_ptr<State> candidate;
        try {
            candidate = std::make_shared<State>(options, control_lifetime);
            SYSTEM_INFO system{}; GetSystemInfo(&system);
            if (system.dwPageSize != 4096 || !system.dwAllocationGranularity || !memory_query() || !page_query()
                || !quota(candidate->original) || !valid_flags(candidate->original.flags)) candidate.reset();
            if (candidate) candidate->last = candidate->original;
        } catch (...) { candidate.reset(); }
        bool result = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            configuring_ = false;
            if (candidate) { state_ = candidate; result = true; }
        }
        return result;
#else
        (void)options; (void)control_lifetime; return false;
#endif
    }
    Geometry plan(const ExpertFileSource& source, std::uint64_t relative, std::size_t extent) const noexcept {
        Cost cost(metrics().plan_calls, metrics().plan_nanoseconds);
#if defined(_WIN32) && defined(_MSC_VER)
        std::uint64_t bytes = 0;
        if (!source.readonly_mapping_file_size(bytes)) return {};
        static const SYSTEM_INFO system = [] { SYSTEM_INFO info{}; GetSystemInfo(&info); return info; }();
        if (system.dwPageSize != 4096) return {};
        return checked_geometry(source.offset(), source.bytes(), bytes, relative, extent,
            system.dwPageSize, system.dwAllocationGranularity);
#else
        (void)source; (void)relative; (void)extent; return {};
#endif
    }
    HostMoeImmutableCache::Storage create(const ExpertFileSource& source,
        const HostMoeImmutableCache::Range& range, const std::shared_ptr<void>& lifetime) noexcept {
        Cost cost(metrics().admission_calls, metrics().admission_nanoseconds);
#if defined(_WIN32) && defined(_MSC_VER)
        auto current = state();
        if (!current || !lifetime || range.source != source.identity()) return {};
        const auto geometry = plan(source, range.offset, range.bytes);
        if (!geometry.valid || fail(TestFault::allocation)) return {};
        auto* node = new (std::nothrow) Node(current, lifetime, geometry);
        if (!node) return {};
        if (!current->insert(node)) { delete node; return {}; }
        std::shared_ptr<Node> owner;
        try { owner = std::shared_ptr<Node>(node, &retire); } catch (...) { return {}; } // Constructor invokes deleter on failure.
        if (!current->claim(*node)) return {};
        if (fail(TestFault::duplicate)) return {};
        node->file = measured(metrics().duplicate_calls, metrics().duplicate_nanoseconds,
            [&] { return source.duplicate_readonly_mapping_file(); });
        if (node->file == INVALID_HANDLE_VALUE || fail(TestFault::section)) return {};
        node->section = measured(metrics().section_calls, metrics().section_nanoseconds,
            [&] { return CreateFileMappingW(node->file, nullptr, PAGE_READONLY, 0, 0, nullptr); });
        if (!node->section || fail(TestFault::map)) return {};
        node->view = measured(metrics().map_calls, metrics().map_nanoseconds,
            [&] { return MapViewOfFile(node->section, FILE_MAP_READ, static_cast<DWORD>(geometry.map_offset >> 32),
                static_cast<DWORD>(geometry.map_offset), geometry.map_bytes); });
        if (!node->view || !readonly_interval(*node)) return {};
        current->possible_lock(*node, true);
        if (fail(TestFault::lock) || !measured(metrics().lock_calls, metrics().lock_nanoseconds,
            [&] { return VirtualLock(static_cast<std::uint8_t*>(node->view) + geometry.lock_delta, geometry.pin_bytes); })) return {};
        current->locked_state(*node, true);
        if (!observed_locked(*node)) return {};
        {
            std::lock_guard<std::mutex> lock(current->mutex);
            if (!current->accepting || current->quarantines) return {};
            node->stage = Node::Stage::live;
        }
        metrics().admission_successes.fetch_add(1, std::memory_order_relaxed);
        return {std::move(owner), static_cast<const std::uint8_t*>(node->view) + geometry.data_delta, range.bytes};
#else
        (void)source; (void)range; (void)lifetime; return {};
#endif
    }
private:
    static bool guarded_copy_impl(void* destination, const void* source, std::size_t count) noexcept {
        if ((!destination || !source) && count) return false;
#if defined(_WIN32) && defined(_MSC_VER)
        // Leaf SEH seam: no C++ unwinding objects in this function. A false result
        // invalidates the entry; caller's exact buffered fallback overwrites all bytes.
        __try {
            if (fail(TestFault::in_page)) RaiseException(EXCEPTION_IN_PAGE_ERROR, 0, 0, nullptr);
            if (fail(TestFault::copy)) { if (count > 1) std::memcpy(destination, source, count / 2); return false; }
            if (count) std::memcpy(destination, source, count);
            return true;
        } __except ((GetExceptionCode() == EXCEPTION_IN_PAGE_ERROR || GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION)
            ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) { return false; }
#else
        (void)destination; (void)source; (void)count; return false;
#endif
    }
public:
    static bool guarded_copy(void* destination, const void* source, std::size_t count) noexcept {
        // Keep clock/atomic bookkeeping outside the leaf SEH function.
        Cost cost(metrics().guarded_copy_calls, metrics().guarded_copy_nanoseconds);
        const bool ok = guarded_copy_impl(destination, source, count);
        if (ok) metrics().guarded_copy_bytes.fetch_add(count, std::memory_order_relaxed);
        else metrics().guarded_copy_failures.fetch_add(1, std::memory_order_relaxed);
        return ok;
    }
    bool retry_cleanup() noexcept {
#if defined(_WIN32) && defined(_MSC_VER)
        auto current = state();
        if (!current) return true;
        for (std::size_t i = 0; i < max_entries; ++i) {
            Node* node = nullptr;
            {
                std::lock_guard<std::mutex> lock(current->mutex);
                if (current->nodes[i] && current->nodes[i]->stage == Node::Stage::quarantine) {
                    node = current->nodes[i]; node->stage = Node::Stage::retiring;
                }
            }
            if (node) finish_retire(node);
        }
        std::lock_guard<std::mutex> lock(current->mutex);
        return current->entries == 0 && current->claims == 0 && current->restore_locked();
#else
        return true;
#endif
    }
    bool finish_zero() noexcept {
#if defined(_WIN32) && defined(_MSC_VER)
        auto current = state();
        if (!current) { std::lock_guard<std::mutex> lock(mutex_); return !configuring_; }
        { std::lock_guard<std::mutex> lock(current->mutex); current->accepting = false; }
        if (!retry_cleanup()) return false;
        std::shared_ptr<State> retired;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (state_ == current) retired = std::move(state_);
        }
        return true; // Local shared owners release control_lifetime outside every mutex.
#else
        return true;
#endif
    }
    Stats stats() const noexcept {
        Stats result;
#define TSG_MAPPED_RAM_SNAPSHOT(name) result.name = metrics().name.load(std::memory_order_relaxed);
        TSG_MAPPED_RAM_METRICS(TSG_MAPPED_RAM_SNAPSHOT)
#undef TSG_MAPPED_RAM_SNAPSHOT
#if defined(_WIN32) && defined(_MSC_VER)
        auto current = state();
        if (!current) return result;
        std::lock_guard<std::mutex> lock(current->mutex);
        // Include uncertain failed-lock envelopes until successful unlock/unmap;
        // this is conservative ownership, never proof of actual resident pages.
        result.locked_page_bytes = current->possible_locked; result.owned_entries = current->entries;
        result.quota_conflict = current->conflict; result.cleanup_failures = current->cleanup_failures;
        result.quota_restore_pending = current->restore_failed || (current->quota_written && !current->accepting);
        for (auto* node : current->nodes)
            if (node && node->stage == Node::Stage::quarantine) result.quarantine_charge_bytes += node->geometry.charge_bytes;
#endif
        return result;
    }
#ifdef TSG_GGML_TEST_HOOKS
    static void test_fail_next(TestFault value) noexcept { fault().store(static_cast<int>(value)); }
    static bool test_storage_locked(const HostMoeImmutableCache::Storage& storage) noexcept {
#if defined(_WIN32) && defined(_MSC_VER)
        return storage.owner && observed_locked(*static_cast<const Node*>(storage.owner.get()));
#else
        (void)storage; return false;
#endif
    }
#endif
};
#undef TSG_MAPPED_RAM_METRICS
}
