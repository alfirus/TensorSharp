// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include "ggml_ops_file_source.h"
#include "ggml_ops_shared_cache_budget.h"
#include <cstdio>
#include <cstring>
#include <new>

namespace tsg {
struct ExpertFileSourceViewPlan {
    std::uint64_t version = 1, total_host_promise = 0, owned_control_bytes = 0,
        pte_forecast_bytes = 0, sources = 0, source_bytes = 0, page_bytes = 0, map_granularity = 0;
};
struct ExpertFileSourceViewStats {
    std::uint64_t version = 1, configured = 0, closing = 0, closed = 0,
        owned_records = 0, live_views = 0, active_readers = 0, mapped_va_bytes = 0,
        owned_control_bytes = 0, held_control_ticket_bytes = 0, pending_records = 0,
        quarantined_records = 0, disabled_records = 0, map_calls = 0, view_close_calls = 0,
        total_host_promise = 0;
};
struct ExpertFileSourceViewPressureStats {
    std::uint64_t version = 1, attempts = 0, completed = 0, range_calls = 0,
        accepted_ranges = 0, unexpected_results = 0, busy_refusals = 0, owned_records = 0;
};
// One independent, bounded owner. Source aliases retain its entire control slab;
// registry erasure cannot release its OS handles or checked accounting ticket.
class ExpertFileSourceViews {
public:
    static constexpr std::size_t maximum_sources = 128;
    enum class TestFault { none, duplicate, section, map, copy_partial, copy_in_page, unmap, close_section, close_file };
    struct TestStats {
        std::uint64_t partial_fired = 0, in_page_fired = 0, cleanup_fired = 0, paused_entries = 0,
            setup_fired = 0, buffered_failures = 0, owned_sections = 0, owned_files = 0, owned_slabs = 0,
            hint_attempts = 0, hint_completed = 0, hint_calls = 0, hint_accepted = 0,
            hint_unexpected = 0, hint_busy = 0, hint_fault_fired = 0;
    };
    static bool plan(int sources, std::uint64_t source_bytes, ExpertFileSourceViewPlan& out) noexcept;
    static bool configure(std::uint64_t token, const ExpertFileSourceViewPlan& plan,
        const std::vector<std::shared_ptr<ExpertFileSource>>& sources) noexcept;
    static bool close(std::uint64_t token) noexcept;
    // No VA, alias, epoch, source, handle or accounting transition. The caller
    // serializes the model/cache lane and resamples physical capacity AFTER.
    // -1 invalid/terminal owner, 0 busy/OS-advisory refusal, 1 all hints accepted.
    static int pressure_hint(std::uint64_t token) noexcept;
    static std::uint64_t scope_token() noexcept;
    static ExpertFileSourceViewPressureStats pressure_stats() noexcept;
    static ExpertFileSourceViewStats stats() noexcept;
    static bool release_ready() noexcept;
    static bool execution_ready() noexcept;
    static void test_fault(TestFault fault) noexcept;
    static void test_pause_copies(bool pause) noexcept;
    static void test_fail_buffered_reads(bool fail) noexcept;
    static bool test_hint_failure_after(int successful_calls) noexcept;
    static TestStats test_stats() noexcept;
    static const char* last_error() noexcept;
};
} // namespace tsg

namespace tsg { namespace source_view_detail {
struct TestState {
    std::atomic<int> fault{0};
    std::atomic<bool> pause{false}, fail_buffered{false};
    std::atomic<std::uint64_t> partial{0}, in_page{0}, cleanup{0}, paused{0}, setup{0}, buffered{0};
    std::atomic<int> hint_remaining{-1};
    std::atomic<std::uint64_t> hint_faults{0};
};
inline TestState& tests() noexcept { static TestState value; return value; }
inline bool fail(ExpertFileSourceViews::TestFault which) noexcept {
#if defined(TSG_GGML_TEST_HOOKS)
    int expected = static_cast<int>(which);
    if (!tests().fault.compare_exchange_strong(expected, 0)) return false;
    if (which == ExpertFileSourceViews::TestFault::duplicate || which == ExpertFileSourceViews::TestFault::section
        || which == ExpertFileSourceViews::TestFault::map) tests().setup.fetch_add(1);
    if (which == ExpertFileSourceViews::TestFault::unmap || which == ExpertFileSourceViews::TestFault::close_section
        || which == ExpertFileSourceViews::TestFault::close_file) tests().cleanup.fetch_add(1);
    return true;
#else
    (void)which; return false;
#endif
}
inline bool add(std::uint64_t a, std::uint64_t b, std::uint64_t& out) noexcept {
    if (a > UINT64_MAX - b) return false;
    out = a + b; return true;
}
inline bool multiply(std::uint64_t a, std::uint64_t b, std::uint64_t& out) noexcept {
    if (b && a > UINT64_MAX / b) return false;
    out = a * b; return true;
}
inline bool align(std::uint64_t value, std::uint64_t page, std::uint64_t& out) noexcept {
    std::uint64_t sum = 0;
    if (!page || (page & (page - 1)) || !add(value, page - 1, sum)) return false;
    out = sum & ~(page - 1); return true;
}
} }

namespace tsg {
struct ExpertFileSourceViewRecord {
    std::weak_ptr<ExpertFileSource> source;
    std::uint64_t identity = 0, bytes = 0;
    std::size_t map_bytes = 0, delta = 0;
    std::atomic<std::uint64_t> readers{0};
    std::atomic<bool> retiring{false}, disabled{false};
#if defined(_WIN32)
    HANDLE file = INVALID_HANDLE_VALUE, section = nullptr;
    const std::uint8_t* base = nullptr;
#endif
    bool pending = false, quarantined = false;
};
namespace source_view_detail {
struct Root { std::array<ExpertFileSourceViewRecord, ExpertFileSourceViews::maximum_sources> records; };
struct Control {
    std::mutex mutex;
    std::uint64_t token = 0;
    bool attempted = false, configured = false, closing = false, closed = false, constructed = false;
    bool sources_detached = false;
    std::atomic<bool> aliases_expired{false};
    ExpertFileSourceViewPlan plan;
    Root* raw = nullptr;
    std::shared_ptr<Root> root;
    std::shared_ptr<SharedCacheCharge> charge;
    std::uint64_t record_count = 0, map_calls = 0, close_calls = 0;
    std::uint64_t hint_attempts = 0, hint_completed = 0, hint_calls = 0,
        hint_accepted = 0, hint_unexpected = 0, hint_busy = 0;
    char error[256]{};
};
inline std::atomic<Control*>& control_slot() noexcept { static std::atomic<Control*> value{nullptr}; return value; }
inline Control& control() {
    // Fixed process-lifetime controller, without a destructor that could drop a
    // quarantined ticket. Default-OFF queries do not construct even this owner.
    alignas(Control) static unsigned char storage[sizeof(Control)];
    static Control* value = [] { auto* created = new (storage) Control;
        control_slot().store(created, std::memory_order_release); return created; }();
    return *value;
}
inline void error(Control& c, const char* text, unsigned long code = 0) noexcept {
    std::snprintf(c.error, sizeof(c.error), "%s (code=%lu)", text, code);
}
inline bool same_plan(const ExpertFileSourceViewPlan& a, const ExpertFileSourceViewPlan& b) noexcept {
    return a.version == b.version && a.total_host_promise == b.total_host_promise
        && a.owned_control_bytes == b.owned_control_bytes && a.pte_forecast_bytes == b.pte_forecast_bytes
        && a.sources == b.sources && a.source_bytes == b.source_bytes
        && a.page_bytes == b.page_bytes && a.map_granularity == b.map_granularity;
}
#if defined(_WIN32) && defined(_MSC_VER)
// Trivial SEH frame: no mutex, shared_ptr, timer or object needing unwinding.
__declspec(noinline) inline bool copy_leaf(void* destination, const void* source, std::size_t bytes,
    ExpertFileSourceViews::TestFault fault) noexcept {
    __try {
        if (fault == ExpertFileSourceViews::TestFault::copy_partial
            || fault == ExpertFileSourceViews::TestFault::copy_in_page) {
            std::memcpy(destination, source, bytes / 2);
            if (fault == ExpertFileSourceViews::TestFault::copy_in_page) RaiseException(EXCEPTION_IN_PAGE_ERROR, 0, 0, nullptr);
            return false;
        }
        std::memcpy(destination, source, bytes); return true;
    } __except ((GetExceptionCode() == EXCEPTION_IN_PAGE_ERROR || GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION)
        ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) { return false; }
}
inline bool open_record(Control& c, ExpertFileSourceViewRecord& r,
    const std::shared_ptr<ExpertFileSource>& source) noexcept {
    std::uint64_t file_bytes = 0;
    if (!source->readonly_mapping_file_size(file_bytes) || source->offset() > file_bytes
        || source->bytes() > file_bytes - source->offset()) { error(c, "Source readonly extent invalid"); return false; }
    const auto begin = source->offset() / c.plan.map_granularity * c.plan.map_granularity;
    const auto delta = source->offset() - begin;
    if (delta > SIZE_MAX || source->bytes() > SIZE_MAX - delta) { error(c, "Source mapping address overflow"); return false; }
    r.source = source; r.identity = source->identity(); r.bytes = source->bytes();
    r.delta = static_cast<std::size_t>(delta); r.map_bytes = r.delta + static_cast<std::size_t>(r.bytes); r.pending = true;
    if (fail(ExpertFileSourceViews::TestFault::duplicate)) { error(c, "Injected duplicate failure"); return false; }
    r.file = source->duplicate_readonly_mapping_file();
    if (r.file == INVALID_HANDLE_VALUE) { error(c, "Duplicate exact readonly source failed", GetLastError()); return false; }
    if (fail(ExpertFileSourceViews::TestFault::section)) { error(c, "Injected section failure"); return false; }
    r.section = CreateFileMappingW(r.file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!r.section) { error(c, "Create readonly section failed", GetLastError()); return false; }
    if (fail(ExpertFileSourceViews::TestFault::map)) { error(c, "Injected mapping failure"); return false; }
    r.base = static_cast<const std::uint8_t*>(MapViewOfFile(r.section, FILE_MAP_READ,
        static_cast<DWORD>(begin >> 32), static_cast<DWORD>(begin), r.map_bytes));
    if (!r.base) { error(c, "Map readonly source failed", GetLastError()); return false; }
    ++c.map_calls;
    MEMORY_BASIC_INFORMATION info{};
    if (VirtualQuery(r.base, &info, sizeof(info)) != sizeof(info) || info.Type != MEM_MAPPED
        || info.AllocationBase != r.base || info.Protect != PAGE_READONLY) {
        error(c, "Mapped source is not readonly"); return false;
    }
    return true;
}
inline bool close_record(Control& c, ExpertFileSourceViewRecord& r) noexcept {
    r.quarantined = false;
    if (r.base) {
        if (fail(ExpertFileSourceViews::TestFault::unmap) || !UnmapViewOfFile(r.base)) {
            r.quarantined = true; error(c, "Source view unmap failed", GetLastError()); return false;
        }
        r.base = nullptr; ++c.close_calls;
    }
    if (r.section) {
        if (fail(ExpertFileSourceViews::TestFault::close_section) || !CloseHandle(r.section)) {
            r.quarantined = true; error(c, "Source section close failed", GetLastError()); return false;
        }
        r.section = nullptr;
    }
    if (r.file != INVALID_HANDLE_VALUE) {
        if (fail(ExpertFileSourceViews::TestFault::close_file) || !CloseHandle(r.file)) {
            r.quarantined = true; error(c, "Source duplicate close failed", GetLastError()); return false;
        }
        r.file = INVALID_HANDLE_VALUE;
    }
    r.pending = false; return true;
}
#endif
} // namespace source_view_detail

inline bool ExpertFileSourceViews::plan(int sources, std::uint64_t source_bytes, ExpertFileSourceViewPlan& out) noexcept {
    out = {}; out.version = 0;
#if defined(_WIN32) && defined(_MSC_VER)
    if (sources <= 0 || sources > static_cast<int>(maximum_sources) || !source_bytes) return false;
    SYSTEM_INFO info{}; GetSystemInfo(&info);
    const std::uint64_t page = info.dwPageSize, granule = info.dwAllocationGranularity;
    if (!page || (page & (page - 1)) || !granule || granule % page || (granule & (granule - 1))) return false;
    std::uint64_t control_bytes = 0, total_pages = 0, source_pages = 0, allowance = 0, pte = 0, total = 0;
    using namespace source_view_detail;
    if (!align(sizeof(Root), page, control_bytes) || !align(source_bytes, page, total_pages)
        || !multiply(static_cast<std::uint64_t>(sources), granule / page + 1, source_pages)
        || !add(total_pages / page, source_pages, total_pages) || !multiply(total_pages, 16, pte)
        || !multiply(static_cast<std::uint64_t>(sources), 65536, allowance) || !add(pte, allowance, pte)
        || !add(pte, 65536, pte) || !align(pte, page, pte) || !add(control_bytes, pte, total)
        || total > INT64_MAX || control_bytes > SIZE_MAX) return false;
    out = {1, total, control_bytes, pte, static_cast<std::uint64_t>(sources), source_bytes, page, granule};
    return true;
#else
    (void)sources; (void)source_bytes; return false;
#endif
}

inline bool ExpertFileSourceViews::configure(std::uint64_t token, const ExpertFileSourceViewPlan& requested,
    const std::vector<std::shared_ptr<ExpertFileSource>>& sources) noexcept {
    using namespace source_view_detail;
    auto& c = control(); std::lock_guard<std::mutex> lock(c.mutex);
#if defined(_WIN32) && defined(_MSC_VER)
    ExpertFileSourceViewPlan calculated;
    if (!token || !plan(static_cast<int>(requested.sources), requested.source_bytes, calculated)
        || !same_plan(requested, calculated) || sources.size() != requested.sources
        || !SharedCacheCharge::host_accounting_attached() || !SharedCacheCharge::checked_release_attached()) {
        error(c, "Source view configuration/checked Host plan invalid"); return false;
    }
    if (c.attempted && (!c.closed || c.token == token || c.raw || c.root || c.charge)) {
        error(c, "Source view scope remains owned or same token is closed"); return false;
    }
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < sources.size(); ++i) {
        if (!sources[i] || sources[i]->source_view_lease() || !add(sum, sources[i]->bytes(), sum)) {
            error(c, "Source view generations invalid"); return false;
        }
        for (std::size_t j = 0; j < i; ++j) if (sources[i]->identity() == sources[j]->identity()) {
            error(c, "Source view repeats a generation"); return false;
        }
    }
    if (sum != requested.source_bytes) { error(c, "Source view total differs from plan"); return false; }
    c.token = token; c.plan = requested; c.attempted = true; c.closed = c.closing = c.configured = false;
    c.sources_detached = false; c.aliases_expired.store(false, std::memory_order_release);
    c.record_count = c.map_calls = c.close_calls = 0; c.error[0] = '\0';
    c.hint_attempts = c.hint_completed = c.hint_calls = c.hint_accepted = c.hint_unexpected = c.hint_busy = 0;
    try {
        c.charge = SharedCacheCharge::reserve(0, 3, static_cast<std::size_t>(requested.owned_control_bytes));
        if (!c.charge || !c.charge->is_accounted() || c.charge->rank() != 0 || c.charge->kind() != 3
            || c.charge->reserved_bytes() != requested.owned_control_bytes) {
            error(c, "Source view control reservation refused"); return false;
        }
        c.raw = static_cast<Root*>(VirtualAlloc(nullptr, static_cast<SIZE_T>(requested.owned_control_bytes),
            MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (!c.raw) { error(c, "Source view control allocation failed", GetLastError()); return false; }
        new (c.raw) Root(); c.constructed = true;
        c.root = std::shared_ptr<Root>(c.raw, [&c](Root*) noexcept {
            // The scope separately owns raw storage. This last-strong-owner
            // notification prevents a weak_ptr::lock race between use_count
            // observation and reset from making the physical slab premature.
            c.aliases_expired.store(true, std::memory_order_release);
        });
        if (!c.charge->commit(static_cast<std::size_t>(requested.owned_control_bytes)) || !c.charge->is_committed()) {
            error(c, "Source view control commit refused"); return false;
        }
        for (std::size_t i = 0; i < sources.size(); ++i) {
            auto& record = c.raw->records[i]; ++c.record_count; // Durable owner BEFORE any OS map step.
            if (!open_record(c, record, sources[i])) return false;
            auto alias = std::shared_ptr<ExpertFileSourceViewRecord>(c.root, &record);
            if (!sources[i]->attach_source_view(alias)) { error(c, "Source view alias publication refused"); return false; }
            record.pending = false;
        }
        c.configured = true; return true;
    } catch (...) { error(c, "Source view configuration exception; owner retained"); return false; }
#else
    (void)token; (void)requested; (void)sources; error(c, "Source views require Windows/MSVC"); return false;
#endif
}

inline int ExpertFileSourceViews::pressure_hint(std::uint64_t token) noexcept {
    using namespace source_view_detail;
    if (!control_slot().load(std::memory_order_acquire)) return 0;
    auto& c = control(); std::lock_guard<std::mutex> lock(c.mutex);
#if defined(_WIN32) && defined(_MSC_VER)
    if (!token || token != c.token) { error(c, "Source view pressure token differs"); return -1; }
    if (!c.attempted || c.closed) return 0;
    auto fatal = [&](const char* text) noexcept {
        c.closing = true; c.configured = false; error(c, text); return -1;
    };
    if (c.closing || !c.configured || !c.raw || !c.constructed || !c.root || c.sources_detached
        || !c.record_count || c.record_count != c.plan.sources || !c.charge || !c.charge->is_committed())
        return fatal("Source view pressure owner is not executable");
    if (c.hint_attempts == UINT64_MAX || c.hint_calls > UINT64_MAX - c.record_count)
        return fatal("Source view pressure counters exhausted");
    ++c.hint_attempts;
    for (std::size_t i = 0; i < c.record_count; ++i) {
        auto& r = c.raw->records[i]; const auto source = r.source.lock();
        if (!source || source->identity() != r.identity || source->bytes() != r.bytes
            || !r.base || !r.section || r.file == INVALID_HANDLE_VALUE || r.pending || r.quarantined)
            return fatal("Source view pressure generation/resource differs");
        const auto alias = source->source_view_lease();
        if (!alias || alias.get() != &r || alias.owner_before(c.root) || c.root.owner_before(alias))
            return fatal("Source view pressure alias generation differs");
        if (r.readers.load(std::memory_order_acquire) || r.retiring.load(std::memory_order_acquire)
            || r.disabled.load(std::memory_order_acquire)) {
            ++c.hint_busy; error(c, "Source view pressure readers/generation not quiescent"); return 0;
        }
    }
    // A captured alias before its reader increment is busy too. Unlike an
    // unmap, a racing read cannot invalidate memory: every VA stays unchanged.
    // Production callers exclude new jobs through the serialized cache lane.
    if (c.root.use_count() != static_cast<long>(c.record_count + 1)) {
        ++c.hint_busy; error(c, "Source view pressure aliases remain borrowed"); return 0;
    }
    bool all_accepted = true;
    for (std::size_t i = 0; i < c.record_count; ++i) {
        const auto& r = c.raw->records[i];
        SetLastError(ERROR_SUCCESS);
        const BOOL unlocked = VirtualUnlock(const_cast<std::uint8_t*>(r.base), r.map_bytes);
        const DWORD result = unlocked ? ERROR_SUCCESS : GetLastError();
        bool accepted = unlocked != 0 || result == ERROR_NOT_LOCKED;
        bool injected_advisory = false;
        ++c.hint_calls; // Actual OS call, not bytes reclaimed or a simulated call.
#if defined(TSG_GGML_TEST_HOOKS)
        auto& test = tests(); const int remaining = test.hint_remaining.load();
        if (accepted && remaining >= 0) {
            if (remaining == 0) {
                test.hint_remaining.store(-1); test.hint_faults.fetch_add(1);
                accepted = false; injected_advisory = true; // Synthetic result after a real call.
            } else test.hint_remaining.store(remaining - 1);
        }
#endif
        if (accepted) ++c.hint_accepted;
        else {
            ++c.hint_unexpected; all_accepted = false;
            error(c, injected_advisory ? "Injected unexpected working-set hint result"
                : "Source view working-set hint not accepted", result);
        }
    }
    if (all_accepted) { ++c.hint_completed; c.error[0] = '\0'; return 1; }
    // An unexpected OS result changes no storage/owner contract. It is only
    // advisory: re-observe physical capacity, never infer released byte credit.
    return 0;
#else
    (void)token; return 0;
#endif
}
inline std::uint64_t ExpertFileSourceViews::scope_token() noexcept {
    if (!source_view_detail::control_slot().load(std::memory_order_acquire)) return 0;
    auto& c = source_view_detail::control(); std::lock_guard<std::mutex> lock(c.mutex);
    return c.attempted && !c.closed ? c.token : 0;
}
inline ExpertFileSourceViewPressureStats ExpertFileSourceViews::pressure_stats() noexcept {
    if (!source_view_detail::control_slot().load(std::memory_order_acquire)) return {};
    auto& c = source_view_detail::control(); std::lock_guard<std::mutex> lock(c.mutex);
    return {1, c.hint_attempts, c.hint_completed, c.hint_calls, c.hint_accepted,
        c.hint_unexpected, c.hint_busy, c.record_count};
}

inline bool ExpertFileSourceViews::close(std::uint64_t token) noexcept {
    using namespace source_view_detail;
    if (!control_slot().load(std::memory_order_acquire)) return true;
    auto& c = control(); std::lock_guard<std::mutex> lock(c.mutex);
    if (!c.attempted || (c.closed && !c.raw && !c.root && !c.charge)) return true;
    if (!token || token != c.token) { error(c, "Source view close token differs"); return false; }
    if (c.closed) return true;
    c.closing = true;
#if defined(_WIN32) && defined(_MSC_VER)
    if (c.raw && c.constructed) {
        for (std::size_t i = 0; i < c.record_count; ++i) c.raw->records[i].retiring.store(true, std::memory_order_release);
        if (!c.sources_detached) {
            for (std::size_t i = 0; i < c.record_count; ++i) {
                auto& record = c.raw->records[i];
                if (auto source = record.source.lock()) {
                    const auto alias = std::shared_ptr<ExpertFileSourceViewRecord>(c.root, &record);
                    if (!source->detach_source_view(alias)) { error(c, "Source alias no longer matches its generation"); return false; }
                }
            }
            c.sources_detached = true;
        }
        // Source aliases are now absent. A reader captured BEFORE incrementing
        // its counter still retains the WHOLE slab and makes this count busy.
        if (c.root && c.root.use_count() != 1) { error(c, "Source view reader aliases remain busy"); return false; }
        c.root.reset();
        // Reset must have retired the strong control lifetime. If a racing
        // weak lock won, retain raw/control/ticket until its final release.
        if (!c.aliases_expired.load(std::memory_order_acquire) && c.record_count) {
            error(c, "Source view alias retirement remains busy"); return false;
        }
        for (std::size_t i = 0; i < c.record_count; ++i)
            if (c.raw->records[i].readers.load(std::memory_order_acquire)) {
                error(c, "Source view readers remain busy"); return false;
            }
        for (std::size_t i = 0; i < c.record_count; ++i) if (!close_record(c, c.raw->records[i])) return false;
        c.raw->~Root(); c.constructed = false;
    }
    if (c.raw) {
        if (!VirtualFree(c.raw, 0, MEM_RELEASE)) { error(c, "Source view control free failed", GetLastError()); return false; }
        c.raw = nullptr; c.record_count = 0;
    }
    if (c.charge) {
        if (!c.charge->release_checked()) { error(c, "Source view control refund acknowledgment retained"); return false; }
        c.charge.reset();
    }
    c.closed = true; return true;
#else
    error(c, "Source views unavailable"); return false;
#endif
}

inline ExpertFileSourceViewStats ExpertFileSourceViews::stats() noexcept {
    using namespace source_view_detail;
    if (!control_slot().load(std::memory_order_acquire)) return {};
    auto& c = control(); std::lock_guard<std::mutex> lock(c.mutex);
    ExpertFileSourceViewStats s;
    s.configured = c.configured; s.closing = c.closing; s.closed = c.closed;
    s.total_host_promise = c.attempted ? c.plan.total_host_promise : 0;
    s.owned_control_bytes = c.raw ? c.plan.owned_control_bytes : 0;
    s.held_control_ticket_bytes = c.charge ? c.charge->reserved_bytes() : 0;
    s.map_calls = c.map_calls; s.view_close_calls = c.close_calls;
    if (c.raw && c.constructed) for (std::size_t i = 0; i < c.record_count; ++i) {
        const auto& r = c.raw->records[i]; ++s.owned_records;
        s.active_readers += r.readers.load(std::memory_order_relaxed);
        s.pending_records += r.pending; s.quarantined_records += r.quarantined;
        s.disabled_records += r.disabled.load(std::memory_order_relaxed);
#if defined(_WIN32)
        if (r.base) { ++s.live_views; s.mapped_va_bytes += r.map_bytes; }
#endif
    }
    if (c.closing && !c.closed && (c.raw || c.charge) && !s.quarantined_records)
        s.quarantined_records = 1; // Includes physical-slab or ticket-only cleanup.
    return s;
}
inline bool ExpertFileSourceViews::release_ready() noexcept {
    if (!source_view_detail::control_slot().load(std::memory_order_acquire)) return true;
    auto& c = source_view_detail::control(); std::lock_guard<std::mutex> lock(c.mutex);
    return !c.attempted || (c.closed && !c.raw && !c.root && !c.charge);
}
inline bool ExpertFileSourceViews::execution_ready() noexcept {
    if (!source_view_detail::control_slot().load(std::memory_order_acquire)) return false;
    auto& c = source_view_detail::control(); std::lock_guard<std::mutex> lock(c.mutex);
    return c.configured && !c.closing && !c.closed;
}
inline const char* ExpertFileSourceViews::last_error() noexcept {
    return source_view_detail::control_slot().load(std::memory_order_acquire) ? source_view_detail::control().error : "";
}
inline void ExpertFileSourceViews::test_fault(TestFault fault) noexcept {
#if defined(TSG_GGML_TEST_HOOKS)
    source_view_detail::tests().fault.store(static_cast<int>(fault));
#else
    (void)fault;
#endif
}
inline void ExpertFileSourceViews::test_pause_copies(bool pause) noexcept {
#if defined(TSG_GGML_TEST_HOOKS)
    source_view_detail::tests().pause.store(pause);
#else
    (void)pause;
#endif
}
inline void ExpertFileSourceViews::test_fail_buffered_reads(bool fail) noexcept {
#if defined(TSG_GGML_TEST_HOOKS)
    source_view_detail::tests().fail_buffered.store(fail);
#else
    (void)fail;
#endif
}
inline bool ExpertFileSourceViews::test_hint_failure_after(int successful_calls) noexcept {
#if defined(TSG_GGML_TEST_HOOKS)
    if (successful_calls < -1 || successful_calls > static_cast<int>(maximum_sources)) return false;
    source_view_detail::tests().hint_remaining.store(successful_calls); return true;
#else
    (void)successful_calls; return false;
#endif
}
inline ExpertFileSourceViews::TestStats ExpertFileSourceViews::test_stats() noexcept {
    auto& t = source_view_detail::tests();
    TestStats result{t.partial.load(), t.in_page.load(), t.cleanup.load(), t.paused.load(), t.setup.load(), t.buffered.load()};
    if (!source_view_detail::control_slot().load(std::memory_order_acquire)) return result;
    auto& c = source_view_detail::control(); std::lock_guard<std::mutex> lock(c.mutex);
    result.owned_slabs = c.raw != nullptr;
    result.hint_attempts = c.hint_attempts; result.hint_completed = c.hint_completed;
    result.hint_calls = c.hint_calls; result.hint_accepted = c.hint_accepted;
    result.hint_unexpected = c.hint_unexpected; result.hint_busy = c.hint_busy;
    result.hint_fault_fired = t.hint_faults.load();
#if defined(_WIN32)
    if (c.raw && c.constructed) for (std::size_t i = 0; i < c.record_count; ++i) {
        result.owned_sections += c.raw->records[i].section != nullptr;
        result.owned_files += c.raw->records[i].file != INVALID_HANDLE_VALUE;
    }
#endif
    return result;
}
inline bool file_source_view_test_buffered_failure_requested() noexcept {
#if defined(TSG_GGML_TEST_HOOKS)
    return source_view_detail::tests().fail_buffered.exchange(false);
#else
    return false;
#endif
}
inline void file_source_view_test_record_buffered_failure() noexcept {
    source_view_detail::tests().buffered.fetch_add(1);
}
inline ExpertFileSourceViewCopy file_source_view_try_copy(const std::shared_ptr<ExpertFileSourceViewRecord>& owner,
    std::uint64_t relative, void* destination, std::size_t bytes) noexcept {
#if defined(_WIN32) && defined(_MSC_VER)
    if (!owner || !bytes) return ExpertFileSourceViewCopy::not_attempted;
    auto& r = *owner;
    if (relative > r.bytes || bytes > r.bytes - relative || !destination) return ExpertFileSourceViewCopy::failed;
    struct Reader {
        ExpertFileSourceViewRecord& r;
        explicit Reader(ExpertFileSourceViewRecord& record) noexcept : r(record) { r.readers.fetch_add(1); }
        ~Reader() { r.readers.fetch_sub(1, std::memory_order_release); }
    } reader(r);
    if (r.retiring.load(std::memory_order_acquire) || r.disabled.load(std::memory_order_relaxed))
        return ExpertFileSourceViewCopy::not_attempted;
#if defined(TSG_GGML_TEST_HOOKS)
    auto& test = source_view_detail::tests();
    if (test.pause.load()) {
        test.paused.fetch_add(1);
        const auto started = GetTickCount64();
        while (test.pause.load() && GetTickCount64() - started < 5000) Sleep(1);
        if (test.pause.load()) { r.disabled.store(true); return ExpertFileSourceViewCopy::failed; }
    }
#endif
    auto fault = ExpertFileSourceViews::TestFault::none;
    if (bytes > 1 && source_view_detail::fail(ExpertFileSourceViews::TestFault::copy_partial)) fault = ExpertFileSourceViews::TestFault::copy_partial;
    else if (bytes > 1 && source_view_detail::fail(ExpertFileSourceViews::TestFault::copy_in_page)) fault = ExpertFileSourceViews::TestFault::copy_in_page;
    const bool ok = source_view_detail::copy_leaf(destination, r.base + r.delta + static_cast<std::size_t>(relative), bytes, fault);
    if (fault == ExpertFileSourceViews::TestFault::copy_partial) source_view_detail::tests().partial.fetch_add(1);
    if (fault == ExpertFileSourceViews::TestFault::copy_in_page) source_view_detail::tests().in_page.fetch_add(1);
    if (ok) return ExpertFileSourceViewCopy::mapped;
    r.disabled.store(true, std::memory_order_release); return ExpertFileSourceViewCopy::failed;
#else
    (void)owner; (void)relative; (void)destination; (void)bytes; return ExpertFileSourceViewCopy::not_attempted;
#endif
}
} // namespace tsg
