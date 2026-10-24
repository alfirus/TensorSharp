// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Generated-file Windows CPU fixture for the real source-view lifecycle.
// Injected copy/cleanup failures are protocol tests, not storage failures.
#ifndef TSG_GGML_TEST_HOOKS
#define TSG_GGML_TEST_HOOKS 1
#endif
#include "../ggml_ops_file_source_view.h"
#include "../ggml_ops_file_source_view_prefetch.h"
#include "../ggml_ops_parallel_read_pipeline.h"
#include "dsv41_engram_io.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <thread>

#if defined(_WIN32) && defined(_MSC_VER)
#include <psapi.h>
namespace {
using Views = tsg::ExpertFileSourceViews;
using Source = tsg::ExpertFileSource;
using Charge = tsg::SharedCacheCharge;
using Plan = tsg::ExpertFileSourceViewPlan;
using Stats = tsg::ExpertFileSourceViewStats;
using Fault = Views::TestFault;
constexpr std::size_t source_offset = 12345;
constexpr std::size_t source_extent = 2 * 65536 + 251;
std::atomic<unsigned> checks{0};
void require(bool value, const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
template<class F> void rejected(F&& action, const char* message) {
    bool threw = false;
    try { action(); } catch (const std::exception&) { threw = true; }
    require(threw, message);
}
template<class F> void until(F&& predicate, const char* message) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error(message);
        std::this_thread::yield();
    }
}
void stage(const char* value) {
    std::fprintf(stderr, "[SOURCE-VIEW-LIFECYCLE-TEST] stage=%s\n", value);
    std::fflush(stderr);
}

// SharedCacheCharge serializes these callbacks. No allocation, native call or
// exception occurs in a callback. A failed acknowledgment may follow an already
// completed ledger refund, and retry must use that same token exactly once.
struct Ledger {
    struct Entry {
        std::uint64_t token = 0;
        std::size_t bytes = 0;
        bool committed = false, refunded = false;
    };
    std::array<Entry, 128> entries{};
    std::size_t pending = 0, committed = 0;
    std::uint64_t next = 0, reserves = 0, commits = 0, releases = 0;
    std::uint64_t last_reserved = 0, failed_ack_token = 0, acknowledged_retry = 0;
    bool failed = false, refuse_reserve = false, refuse_commit = false;
    bool fail_ack = false, refund_before_failed_ack = false;
    Entry* find(std::uint64_t token) noexcept {
        for (auto& entry : entries) if (entry.token == token) return &entry;
        return nullptr;
    }
    std::size_t held() const noexcept {
        std::size_t total = 0;
        for (const auto& entry : entries) if (entry.token) total += entry.bytes;
        return total;
    }
    std::size_t count() const noexcept {
        return static_cast<std::size_t>(std::count_if(entries.begin(), entries.end(),
            [](const Entry& entry) { return entry.token != 0; }));
    }
    bool refund(Entry& entry) noexcept {
        if (entry.refunded) return true;
        auto& amount = entry.committed ? committed : pending;
        if (amount < entry.bytes) { failed = true; return false; }
        amount -= entry.bytes;
        entry.refunded = true;
        return true;
    }
    static std::uint64_t reserve(void* context, int rank, int kind, std::int64_t bytes) noexcept {
        auto& ledger = *static_cast<Ledger*>(context);
        ++ledger.reserves;
        if (rank != 0 || kind != 3 || bytes <= 0) { ledger.failed = true; return 0; }
        if (ledger.refuse_reserve) { ledger.refuse_reserve = false; return 0; }
        const auto amount = static_cast<std::size_t>(bytes);
        if (amount > (8u << 20) || ledger.pending + ledger.committed > (8u << 20) - amount) return 0;
        for (auto& entry : ledger.entries) if (!entry.token) {
            entry = {++ledger.next, amount, false, false};
            ledger.pending += amount;
            ledger.last_reserved = entry.token;
            return entry.token;
        }
        ledger.failed = true;
        return 0;
    }
    static int commit(void* context, std::uint64_t token) noexcept {
        auto& ledger = *static_cast<Ledger*>(context);
        auto* entry = ledger.find(token);
        if (!entry || entry->committed || entry->refunded) { ledger.failed = true; return 0; }
        if (ledger.refuse_commit) { ledger.refuse_commit = false; return 0; }
        if (ledger.pending < entry->bytes) { ledger.failed = true; return 0; }
        ledger.pending -= entry->bytes;
        ledger.committed += entry->bytes;
        entry->committed = true;
        ++ledger.commits;
        return 1;
    }
    static int release_checked(void* context, std::uint64_t token) noexcept {
        auto& ledger = *static_cast<Ledger*>(context);
        auto* entry = ledger.find(token);
        ++ledger.releases;
        if (!entry) { ledger.failed = true; return 0; }
        if (ledger.fail_ack) {
            ledger.fail_ack = false;
            ledger.failed_ack_token = token;
            if (ledger.refund_before_failed_ack && !ledger.refund(*entry)) return 0;
            return 0;
        }
        if (!ledger.refund(*entry)) return 0;
        if (ledger.failed_ack_token == token) ++ledger.acknowledged_retry;
        *entry = {};
        return 1;
    }
    static void release(void* context, std::uint64_t token) noexcept {
        auto& ledger = *static_cast<Ledger*>(context);
        // The production helper is required to keep a failed checked ticket;
        // it must never substitute this void/destructor route.
        ledger.failed = true;
        (void)release_checked(context, token);
    }
};

struct Fixture {
    Ledger ledger;
    bool attached = false;
    std::uint64_t token = 0, next_scope = 100;
    std::uint64_t previous_source_identity = 0;
    std::filesystem::path path;
    std::vector<std::uint8_t> bytes;
    std::vector<std::shared_ptr<Source>> sources;
    explicit Fixture(std::filesystem::path directory) {
        std::filesystem::create_directories(directory);
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = directory / std::filesystem::u8path(u8"source-view-专家-" + std::to_string(stamp) + ".bin");
        bytes.resize(source_offset + source_extent + 65536);
        for (std::size_t i = 0; i < bytes.size(); ++i)
            bytes[i] = static_cast<std::uint8_t>((i * 37 + (i >> 8) * 11 + 17) % 251);
        std::ofstream file(path, std::ios::binary);
        require(bool(file), "Cannot create generated source-view fixture");
        file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        file.close();
        require(bool(file), "Cannot finish generated source-view fixture");
        require(Charge::attach(&ledger, Ledger::reserve, Ledger::commit, Ledger::release,
            false, true, Ledger::release_checked), "Cannot attach checked source-view ledger");
        attached = true;
    }
    void make_sources(std::size_t count = 1) {
        require(token == 0, "Previous scope was not closed");
        sources.clear();
        const auto name = path.u8string();
        for (std::size_t i = 0; i < count; ++i)
            sources.push_back(std::make_shared<Source>(name.c_str(), source_offset + i * 37, source_extent));
        require(sources[0]->identity() != previous_source_identity,
            "A newly registered source reused a prior generation identity");
        previous_source_identity = sources[0]->identity();
    }
    Plan make_plan() const {
        Plan plan;
        require(Views::plan(static_cast<int>(sources.size()), sources.size() * source_extent, plan),
            "Cannot plan generated source views");
        require(plan.version == 1 && plan.sources == sources.size()
            && plan.source_bytes == sources.size() * source_extent && plan.page_bytes > 0
            && plan.map_granularity >= plan.page_bytes && plan.map_granularity % plan.page_bytes == 0
            && plan.owned_control_bytes > 0 && plan.pte_forecast_bytes > 0
            && plan.total_host_promise == plan.owned_control_bytes + plan.pte_forecast_bytes,
            "Source-view Host plan has invalid units or geometry");
        return plan;
    }
    Plan configure() {
        auto plan = make_plan();
        token = ++next_scope;
        require(Views::configure(token, plan, sources), "Cannot configure real generated source views");
        auto stats = Views::stats();
        std::uint64_t exact_map_bytes = 0;
        for (const auto& source : sources)
            exact_map_bytes += source->offset() % plan.map_granularity + source->bytes();
        require(stats.version == 1 && stats.configured == 1 && stats.closing == 0 && stats.closed == 0
            && stats.owned_records == sources.size() && stats.live_views == sources.size()
            && stats.mapped_va_bytes == exact_map_bytes
            && stats.active_readers == 0 && stats.pending_records == 0 && stats.quarantined_records == 0
            && stats.owned_control_bytes == plan.owned_control_bytes
            && stats.held_control_ticket_bytes == plan.owned_control_bytes
            && stats.total_host_promise == plan.total_host_promise
            && ledger.pending == 0 && ledger.committed == plan.owned_control_bytes
            && ledger.held() == plan.owned_control_bytes && ledger.count() == 1 && !Views::release_ready(),
            "Configured source-view ownership differs from checked control ticket");
        return plan;
    }
    void closed() {
        require(token != 0 && Views::close(token), "Checked source-view cleanup failed");
        require(Views::close(token), "Checked source-view cleanup is not idempotent");
        auto stats = Views::stats();
        require(stats.closed == 1 && stats.owned_records == 0 && stats.live_views == 0
            && stats.active_readers == 0 && stats.mapped_va_bytes == 0 && stats.owned_control_bytes == 0
            && stats.held_control_ticket_bytes == 0 && stats.pending_records == 0 && stats.quarantined_records == 0
            && Views::release_ready() && ledger.pending == 0 && ledger.committed == 0
            && ledger.held() == 0 && !ledger.failed,
            "Checked source-view cleanup left physical/control/ticket ownership");
        for (const auto& source : sources) require(!source->source_view_lease(), "Closed source retained a view alias");
        token = 0;
        sources.clear();
    }
    void exact(const std::vector<std::uint8_t>& output, std::size_t prefix,
        std::size_t index, std::size_t relative, std::size_t count) const {
        require(output.size() >= prefix + count, "Invalid fixture output geometry");
        require(std::equal(output.begin() + prefix, output.begin() + prefix + count,
            bytes.begin() + source_offset + index * 37 + relative), "Source view changed exact registered bytes");
        require(std::all_of(output.begin(), output.begin() + prefix, [](std::uint8_t v) { return v == 0xDA; })
            && std::all_of(output.begin() + prefix + count, output.end(), [](std::uint8_t v) { return v == 0xDA; }),
            "Source view overwrote destination guards");
    }
    bool cleanup_noexcept() noexcept {
        tsg::ExpertFileSourceViewPrefetch::test_pause(false);
        tsg::ExpertFileSourceViewPrefetch::test_failure_after(-1);
        Views::test_pause_copies(false);
        Views::test_fail_buffered_reads(false);
        Views::test_fault(Fault::none);
        Views::test_hint_failure_after(-1);
        if (token && !Views::close(token)) return false;
        if (attached && !Charge::detach(&ledger)) return false;
        attached = false;
        sources.clear();
        std::error_code error;
        std::filesystem::remove(path, error);
        return !error;
    }
};

void ranges_and_identity(Fixture& fixture) {
    stage("nonzero-offset-bounds-and-identity");
    Plan rejected_plan;
    require(!Views::plan(0, 1, rejected_plan)
        && !Views::plan(static_cast<int>(Views::maximum_sources) + 1, 1, rejected_plan)
        && !Views::plan(1, 0, rejected_plan), "Invalid source-view plan accepted");
    fixture.make_sources(2);
    require(fixture.sources[0]->identity() != fixture.sources[1]->identity(), "Distinct source generations aliased");
    std::array<std::uint8_t, 17> ordinary{};
    tsg::ExpertFileSourceReadActivity ordinary_activity;
    fixture.sources[0]->read(7, ordinary.data(), ordinary.size(), &ordinary_activity);
    require(ordinary_activity.view_calls == 0 && ordinary_activity.view_bytes == 0
        && ordinary_activity.buffered_bytes == ordinary.size()
        && std::equal(ordinary.begin(), ordinary.end(), fixture.bytes.begin() + source_offset + 7),
        "Unconfigured Source did not retain its original buffered read contract");
    auto plan = fixture.configure();
    const auto before_wrong_token = Views::stats();
    require(!Views::close(fixture.token + 1) && Views::stats().closing == 0
        && Views::stats().live_views == before_wrong_token.live_views
        && Views::stats().held_control_ticket_bytes == before_wrong_token.held_control_ticket_bytes,
        "Foreign close token changed a live source-view scope");
    for (std::size_t index = 0; index < fixture.sources.size(); ++index) {
        const std::array<std::pair<std::size_t, std::size_t>, 5> ranges{{
            {0, 1}, {1, 37}, {65536 - source_offset - 7, 29}, {source_extent - 1, 1}, {0, source_extent}}};
        for (auto range : ranges) {
            std::vector<std::uint8_t> output(range.second + 26, 0xDA), literal(range.second);
            tsg::ExpertFileSourceReadActivity activity;
            fixture.sources[index]->read(range.first, output.data() + 13, range.second, &activity);
            fixture.sources[index]->read_buffered(range.first, literal.data(), literal.size());
            fixture.exact(output, 13, index, range.first, range.second);
            require(std::equal(literal.begin(), literal.end(), output.begin() + 13)
                && activity.view_calls == 1 && activity.view_bytes == range.second
                && activity.view_fallbacks == 0 && activity.buffered_bytes == 0,
                "Mapped exact task was misattributed or differs from authoritative ReadFile");
        }
    }
    std::array<std::uint8_t, 64> sentinel{};
    sentinel.fill(0xA7);
    auto unchanged = sentinel;
    rejected([&] { fixture.sources[0]->read(source_extent, sentinel.data(), 1); }, "Source-view EOF overread accepted");
    rejected([&] { fixture.sources[0]->read(UINT64_MAX - 7, sentinel.data(), 16); }, "Source-view offset overflow accepted");
    rejected([&] { fixture.sources[0]->read(0, nullptr, 1); }, "Null source-view destination accepted");
    require(sentinel == unchanged, "Rejected source-view bounds changed caller memory");
    fixture.sources[0]->read(source_extent, nullptr, 0);
    require(Views::stats().held_control_ticket_bytes == plan.owned_control_bytes,
        "Read-only source service changed its durable charge");
    const auto terminal_token = fixture.token;
    fixture.closed();
    fixture.make_sources(2);
    const auto terminal_plan = fixture.make_plan();
    const auto reserves = fixture.ledger.reserves;
    require(!Views::configure(terminal_token, terminal_plan, fixture.sources)
        && Views::stats().closed == 1 && fixture.ledger.reserves == reserves,
        "A finally closed token recreated native views or a control reservation");
    fixture.sources.clear();
}

void held_alias(Fixture& fixture) {
    stage("captured-alias-after-source-destruction");
    fixture.make_sources(); fixture.configure();
    std::weak_ptr<Source> weak = fixture.sources[0];
    auto alias = fixture.sources[0]->source_view_lease();
    std::weak_ptr<tsg::ExpertFileSourceViewRecord> weak_alias = alias;
    require(bool(alias), "Configured source has no owned view alias");
    fixture.sources.clear();
    require(weak.expired(), "View record unnecessarily retained its original Source");
    HANDLE writer = CreateFileW(fixture.path.c_str(), GENERIC_WRITE, FILE_SHARE_READ,
        nullptr, OPEN_EXISTING, 0, nullptr);
    const DWORD writer_error = GetLastError();
    if (writer != INVALID_HANDLE_VALUE) CloseHandle(writer);
    require(writer == INVALID_HANDLE_VALUE && writer_error == ERROR_SHARING_VIOLATION,
        "View duplicate lost the exact handle's write/delete sharing protection");
    std::vector<std::uint8_t> output(113 + 26, 0xDA);
    require(tsg::file_source_view_try_copy(alias, source_extent - 113, output.data() + 13, 113)
        == tsg::ExpertFileSourceViewCopy::mapped, "Independent captured view did not survive Source destruction");
    fixture.exact(output, 13, 0, source_extent - 113, 113);
    const auto before = Views::stats();
    require(!Views::close(fixture.token) && Views::stats().active_readers == 0
        && Views::stats().live_views == before.live_views
        && Views::stats().held_control_ticket_bytes == before.held_control_ticket_bytes,
        "Close freed a captured pre-reader-increment alias");
    alias.reset();
    fixture.closed();
    require(weak_alias.expired(), "Final checked close left an acquirable weak view alias");
}

void concurrent_busy_close(Fixture& fixture) {
    stage("concurrent-copies-close-busy");
    fixture.make_sources(); fixture.configure();
    constexpr std::size_t count = 8, width = 32771;
    std::array<std::vector<std::uint8_t>, count> outputs;
    for (auto& output : outputs) output.assign(width + 26, 0xDA);
    std::array<tsg::ExpertFileSourceReadActivity, count> activity{};
    std::atomic<int> errors{0};
    struct Workers {
        std::vector<std::thread> values;
        ~Workers() { Views::test_pause_copies(false); for (auto& thread : values) if (thread.joinable()) thread.join(); }
        void join() { Views::test_pause_copies(false); for (auto& thread : values) if (thread.joinable()) thread.join(); }
    } workers;
    const auto initial = Views::test_stats();
    Views::test_pause_copies(true);
    for (std::size_t i = 0; i < count; ++i) workers.values.emplace_back([&, i] {
        try { fixture.sources[0]->read(i * 127, outputs[i].data() + 13, width, &activity[i]); }
        catch (...) { ++errors; }
    });
    until([&] { return Views::stats().active_readers == count
        && Views::test_stats().paused_entries >= initial.paused_entries + count; },
        "Concurrent copies did not reach the bounded production pause seam");
    const auto before = Views::stats();
    const auto release_calls = fixture.ledger.releases;
    require(!Views::close(fixture.token), "Close succeeded with actual active copy leases");
    const auto busy = Views::stats();
    require(busy.live_views == before.live_views && busy.mapped_va_bytes == before.mapped_va_bytes
        && busy.held_control_ticket_bytes == before.held_control_ticket_bytes
        && fixture.ledger.releases == release_calls && !Views::release_ready(),
        "Busy Close unmapped or refunded active copy ownership");
    workers.join();
    require(errors == 0 && Views::stats().active_readers == 0, "Concurrent copies escaped close/join lifetime");
    for (std::size_t i = 0; i < count; ++i) {
        fixture.exact(outputs[i], 13, 0, i * 127, width);
        require(activity[i].view_calls == 1 && activity[i].view_bytes == width
            && activity[i].buffered_bytes == 0, "Already-acquired copy was replaced or partially published");
    }
    fixture.closed();
}

void copy_fault(Fixture& fixture, Fault fault) {
    stage(fault == Fault::copy_partial ? "partial-mapped-copy-full-fallback" : "seh-inpage-full-fallback");
    fixture.make_sources(); fixture.configure();
    const auto before = Views::test_stats();
    Views::test_fault(fault);
    constexpr std::size_t relative = 71, count = 8193;
    std::vector<std::uint8_t> output(count + 26, 0xDA);
    tsg::ExpertFileSourceReadActivity activity;
    fixture.sources[0]->read(relative, output.data() + 13, count, &activity);
    fixture.exact(output, 13, 0, relative, count);
    const auto after = Views::test_stats();
    require(after.partial_fired == before.partial_fired + (fault == Fault::copy_partial)
        && after.in_page_fired == before.in_page_fired + (fault == Fault::copy_in_page)
        && activity.view_calls == 1 && activity.view_bytes == 0 && activity.view_fallbacks == 1
        && activity.buffered_bytes == count && Views::stats().disabled_records == 1,
        "Injected mapped fault was not consumed once followed by a whole-task buffered overwrite");
    tsg::ExpertFileSourceReadActivity second;
    std::fill(output.begin(), output.end(), 0xDA);
    fixture.sources[0]->read(relative, output.data() + 13, count, &second);
    fixture.exact(output, 13, 0, relative, count);
    require(second.view_calls == 0 && second.view_bytes == 0 && second.buffered_bytes == count,
        "A fault-disabled generation retried mapped service");
    fixture.closed();
}

void failed_fallback_join(Fixture& fixture) {
    stage("mapped-fault-buffered-prefix-failure-joined-pipeline");
    fixture.make_sources(); fixture.configure();
    constexpr std::size_t jobs = 16, count = 4099;
    std::array<std::vector<std::uint8_t>, jobs> output;
    std::vector<tsg::ExpertReadTask> tasks;
    tasks.reserve(jobs);
    for (std::size_t i = 0; i < jobs; ++i) {
        output[i].assign(count, 0xDA);
        tasks.push_back({fixture.sources[0].get(), i * 53, output[i].data(), count});
    }
    std::array<bool, jobs> failed{}, consumed{};
    std::atomic<int> active{0};
    const auto before = Views::test_stats();
    Views::test_fault(Fault::copy_partial);
    Views::test_fail_buffered_reads(true);
    tsg_dsv41::engram_io_pool pool(8);
    bool threw = false;
    try {
        pool.run_pipelined(jobs, [&](std::size_t i) {
            ++active;
            struct Done { std::atomic<int>& active; ~Done() { --active; } } done{active};
            try { tasks[i].prepare(); } catch (...) { failed[i] = true; throw; }
        }, [&](std::size_t i) {
            consumed[i] = true;
            require(tasks[i].source_read_bytes == count
                && std::equal(output[i].begin(), output[i].end(), fixture.bytes.begin() + source_offset + i * 53),
                "Pipeline published a partial successful task");
        });
    } catch (const std::exception&) { threw = true; }
    Views::test_fail_buffered_reads(false);
    const auto after = Views::test_stats();
    require(threw && after.partial_fired == before.partial_fired + 1
        && after.buffered_failures == before.buffered_failures + 1
        && active == 0 && Views::stats().active_readers == 0,
        "Failed fallback did not throw after joining every actual preparation worker");
    std::size_t failed_count = 0;
    for (std::size_t i = 0; i < jobs; ++i) if (failed[i]) {
        ++failed_count;
        require(!consumed[i] && tasks[i].source_read_bytes == 0
            && std::equal(output[i].begin(), output[i].begin() + count / 2,
                fixture.bytes.begin() + source_offset + i * 53)
            && !std::equal(output[i].begin(), output[i].end(), fixture.bytes.begin() + source_offset + i * 53),
            "Failed buffered-prefix task was published or was not a genuine partial destination");
    }
    require(failed_count == 1, "Injected fallback did not fail exactly one task");
    std::size_t next_consumed = 0;
    pool.run_pipelined(jobs, [&](std::size_t i) { tasks[i].prepare(); }, [&](std::size_t i) {
        ++next_consumed;
        require(tasks[i].source_read_bytes == count
            && std::equal(output[i].begin(), output[i].end(), fixture.bytes.begin() + source_offset + i * 53),
            "Joined pool could not execute a fresh exact buffered generation");
    });
    require(next_consumed == jobs && Views::stats().active_readers == 0, "Failed pipeline retained a callback or worker");
    fixture.closed();
}

void setup_failure(Fixture& fixture, Fault fault) {
    stage(fault == Fault::duplicate ? "duplicate-failure-owned-attempt" :
        fault == Fault::section ? "section-failure-owned-attempt" : "map-failure-owned-attempt");
    fixture.make_sources();
    auto plan = fixture.make_plan();
    fixture.token = ++fixture.next_scope;
    const auto before = Views::test_stats();
    Views::test_fault(fault);
    require(!Views::configure(fixture.token, plan, fixture.sources)
        && Views::test_stats().setup_fired == before.setup_fired + 1,
        "Injected setup failure was ignored or mistaken for successful source service");
    require(!fixture.sources[0]->source_view_lease() && !Views::release_ready()
        && !Charge::detach(&fixture.ledger), "Failed setup lost its real cleanup/ticket owner");
    fixture.closed();
}

void cleanup_failure(Fixture& fixture, Fault fault) {
    stage(fault == Fault::unmap ? "checked-unmap-retry" :
        fault == Fault::close_section ? "checked-section-retry" : "checked-duplicate-retry");
    fixture.make_sources(); const auto plan = fixture.configure();
    const auto ticket = fixture.ledger.last_reserved;
    const auto before = Views::test_stats();
    Views::test_fault(fault);
    require(!Views::close(fixture.token), "Injected physical cleanup failure was reported clean");
    const auto stats = Views::stats();
    const auto exact = Views::test_stats();
    require(exact.cleanup_fired == before.cleanup_fired + 1 && stats.closed == 0
        && stats.held_control_ticket_bytes == plan.owned_control_bytes
        && fixture.ledger.find(ticket) && fixture.ledger.count() == 1
        && !fixture.sources[0]->source_view_lease() && !Views::release_ready()
        && !Charge::detach(&fixture.ledger), "Physical cleanup failure dropped the exact resource/ticket owner");
    require(exact.owned_files == 1
        && exact.owned_sections == (fault == Fault::close_file ? 0 : 1)
        && stats.live_views == (fault == Fault::unmap ? 1 : 0),
        "Cleanup retry did not retain the exact failed OS resource stage");
    std::array<std::uint8_t, 17> fallback{};
    tsg::ExpertFileSourceReadActivity activity;
    fixture.sources[0]->read(11, fallback.data(), fallback.size(), &activity);
    require(activity.view_calls == 0 && activity.buffered_bytes == fallback.size()
        && std::equal(fallback.begin(), fallback.end(), fixture.bytes.begin() + source_offset + 11)
        && Views::stats().map_calls == stats.map_calls,
        "Closing source attempted to remap or lost authoritative buffered service");
    fixture.closed();
}

void acknowledgment_failure(Fixture& fixture, bool ledger_already_refunded) {
    stage(ledger_already_refunded ? "host-ack-after-ledger-refund-retry" : "host-ack-before-refund-retry");
    fixture.make_sources(); const auto plan = fixture.configure();
    const auto token = fixture.ledger.last_reserved;
    const auto retries = fixture.ledger.acknowledged_retry;
    fixture.ledger.fail_ack = true;
    fixture.ledger.refund_before_failed_ack = ledger_already_refunded;
    require(!Views::close(fixture.token), "Failed checked Host acknowledgment was reported clean");
    const auto stats = Views::stats();
    const auto resources = Views::test_stats();
    require(stats.closed == 0 && stats.live_views == 0 && stats.mapped_va_bytes == 0
        && stats.owned_control_bytes == 0 && resources.owned_sections == 0 && resources.owned_files == 0
        && resources.owned_slabs == 0
        && stats.held_control_ticket_bytes == plan.owned_control_bytes
        && fixture.ledger.failed_ack_token == token && fixture.ledger.find(token)
        && fixture.ledger.held() == plan.owned_control_bytes && !Views::release_ready()
        && !Charge::detach(&fixture.ledger), "Host acknowledgment failure did not retain its exact ticket after physical cleanup");
    require(fixture.ledger.committed == (ledger_already_refunded ? 0 : plan.owned_control_bytes),
        "Fixture did not distinguish post-ledger acknowledgment failure from pre-refund failure");
    fixture.closed();
    require(fixture.ledger.acknowledged_retry == retries + 1, "Checked retry changed token or credited the ledger twice");
}

struct HintIdentity {
    std::uint64_t source = 0;
    tsg::ExpertFileSourceViewRecord* record = nullptr;
    HANDLE file = INVALID_HANDLE_VALUE, section = nullptr;
    const std::uint8_t* base = nullptr;
    std::weak_ptr<tsg::ExpertFileSourceViewRecord> lifetime;
};
std::vector<HintIdentity> hint_identities(const Fixture& fixture) {
    std::vector<HintIdentity> result;
    result.reserve(fixture.sources.size());
    for (const auto& source : fixture.sources) {
        auto alias = source->source_view_lease();
        require(bool(alias), "Cannot snapshot a source without a view alias");
        result.push_back({source->identity(), alias.get(), alias->file, alias->section, alias->base, alias});
    }
    return result; // Weak aliases only: this snapshot must not make a hint busy.
}
void unchanged_hint_ownership(Fixture& fixture, const Plan& plan, const Stats& before,
    const std::vector<HintIdentity>& identities, std::uint64_t reserves,
    std::uint64_t commits, std::uint64_t releases, std::uint64_t ticket) {
    const auto after = Views::stats();
    // All sixteen public fields remain identical. Hint activity has a separate
    // observation and cannot be mistaken for a close/refund or a new mapping.
    require(after.version == before.version && after.configured == before.configured
        && after.closing == before.closing && after.closed == before.closed
        && after.owned_records == before.owned_records && after.live_views == before.live_views
        && after.active_readers == before.active_readers && after.mapped_va_bytes == before.mapped_va_bytes
        && after.owned_control_bytes == before.owned_control_bytes
        && after.held_control_ticket_bytes == before.held_control_ticket_bytes
        && after.pending_records == before.pending_records && after.quarantined_records == before.quarantined_records
        && after.disabled_records == before.disabled_records && after.map_calls == before.map_calls
        && after.view_close_calls == before.view_close_calls && after.total_host_promise == before.total_host_promise,
        "Working-set hint changed a view, source scope or public ownership field");
    require(fixture.ledger.reserves == reserves && fixture.ledger.commits == commits
        && fixture.ledger.releases == releases && fixture.ledger.last_reserved == ticket
        && fixture.ledger.pending == 0 && fixture.ledger.committed == plan.owned_control_bytes
        && fixture.ledger.held() == plan.owned_control_bytes && fixture.ledger.count() == 1
        && fixture.ledger.find(ticket) && !fixture.ledger.failed,
        "Working-set hint reserved, committed, released or changed the original Host ticket");
    for (std::size_t i = 0; i < identities.size(); ++i) {
        const auto& identity = identities[i];
        auto original = identity.lifetime.lock(), current = fixture.sources[i]->source_view_lease();
        require(original && current && !original.owner_before(current) && !current.owner_before(original)
            && fixture.sources[i]->identity() == identity.source && current.get() == identity.record
            && current->identity == identity.source && current->file == identity.file
            && current->section == identity.section && current->base == identity.base
            && !current->retiring.load() && !current->disabled.load(),
            "Working-set hint changed an exact source/handle/VA/alias lifetime");
    }
}

// No static Psapi dependency: the required modern-Windows export is resolved
// explicitly and its absence/query failure fails this required fixture case.
struct WorkingSetPages {
    using Query = BOOL (WINAPI*)(HANDLE, PVOID, DWORD);
    Query query = nullptr;
    std::vector<PSAPI_WORKING_SET_EX_INFORMATION> pages;
    WorkingSetPages(const Fixture& fixture, const Plan& plan) {
        const auto module = GetModuleHandleW(L"kernel32.dll");
        FARPROC address = module ? GetProcAddress(module, "K32QueryWorkingSetEx") : nullptr;
        require(address != nullptr, "Required K32QueryWorkingSetEx is unavailable; WS-hint behavior is unqualified");
        static_assert(sizeof(query) == sizeof(address), "Windows function-pointer representation");
        std::memcpy(&query, &address, sizeof(query));
        const auto page = static_cast<std::size_t>(plan.page_bytes);
        for (const auto& source : fixture.sources) {
            auto alias = source->source_view_lease();
            require(alias && alias->base && alias->bytes > 0 && alias->delta <= SIZE_MAX - alias->bytes,
                "Cannot form exact touched source page ranges");
            const auto first = alias->delta / page * page;
            const auto last = (alias->delta + static_cast<std::size_t>(alias->bytes) - 1) / page * page;
            for (auto offset = first;; offset += page) {
                PSAPI_WORKING_SET_EX_INFORMATION info{};
                info.VirtualAddress = const_cast<std::uint8_t*>(alias->base + offset);
                pages.push_back(info);
                if (offset == last) break;
            }
        }
        require(!pages.empty() && pages.size() <= (std::numeric_limits<DWORD>::max)() / sizeof(pages[0]),
            "Working-set query metadata exceeds its bounded Windows call");
    }
    std::size_t active() {
        for (auto& page : pages) page.VirtualAttributes.Flags = 0;
        require(query(GetCurrentProcess(), pages.data(), static_cast<DWORD>(pages.size() * sizeof(pages[0]))) != FALSE,
            "Required QueryWorkingSetEx failed; no working-set removal qualification");
        return static_cast<std::size_t>(std::count_if(pages.begin(), pages.end(),
            [](const auto& page) { return page.VirtualAttributes.Valid != 0; }));
    }
};
void touch_hint_sources(Fixture& fixture) {
    // Real whole-source guarded copies touch the source VAs. The destination is
    // a short-lived fixture buffer, not a production payload cache.
    std::vector<std::uint8_t> output(source_extent + 26, 0xDA);
    for (std::size_t index = 0; index < fixture.sources.size(); ++index) {
        std::fill(output.begin(), output.end(), static_cast<std::uint8_t>(0xDA));
        tsg::ExpertFileSourceReadActivity activity;
        fixture.sources[index]->read(0, output.data() + 13, source_extent, &activity);
        fixture.exact(output, 13, index, 0, source_extent);
        require(activity.view_calls == 1 && activity.view_bytes == source_extent
            && activity.view_fallbacks == 0 && activity.buffered_bytes == 0,
            "Required touched-page setup did not execute the actual mapped source copy");
    }
}

void working_set_hints(Fixture& fixture, std::size_t count) {
    stage(count == 126 ? "ws-hint-126-touched-ranges-same-ownership" : "ws-hint-repeated-same-alias-lifetime");
    fixture.make_sources(count); const auto plan = fixture.configure();
    const auto identities = hint_identities(fixture);
    const auto before = Views::stats();
    const auto reserves = fixture.ledger.reserves, commits = fixture.ledger.commits;
    const auto releases = fixture.ledger.releases, ticket = fixture.ledger.last_reserved;
    WorkingSetPages pages(fixture, plan);
    fixture.ledger.refuse_reserve = fixture.ledger.refuse_commit = true;
    for (unsigned pass = 0; pass < 2; ++pass) {
        touch_hint_sources(fixture);
        const auto active_before = pages.active();
        require(active_before > 0, "Touched source views had no observed valid PTEs; removal behavior is unqualified");
        const auto hints = Views::test_stats();
        require(Views::pressure_hint(fixture.token) == 1, "Quiescent working-set hint was not fully accepted");
        // Observe BEFORE reading any payload again. This proves only current
        // process PTE Valid bits, never unique RAM/free bytes or future residency.
        const auto active_after = pages.active();
        require(active_after == 0, "Accepted working-set hint left touched source VAs valid");
        const auto after = Views::test_stats();
        require(after.hint_completed == hints.hint_completed + 1
            && after.hint_calls == hints.hint_calls + count
            && after.hint_accepted == hints.hint_accepted + count
            && after.hint_unexpected == hints.hint_unexpected && after.hint_busy == hints.hint_busy
            && Views::execution_ready(), "Working-set hint did not report actual accepted range calls");
        unchanged_hint_ownership(fixture, plan, before, identities, reserves, commits, releases, ticket);
        require(fixture.ledger.refuse_reserve && fixture.ledger.refuse_commit,
            "Working-set hint consumed a new reserve/commit callback");
        std::fprintf(stderr, "[SOURCE-VIEW-WS-HINT-TEST] sources=%zu pass=%u queried=1 sampled_pages=%zu active_before=%zu active_after=%zu exact_ownership=1 scope=process_pte_validity_not_physical_ram_or_future_residency\n",
            count, pass, pages.pages.size(), active_before, active_after);
        // Future exact service is mandatory, even though pages may refault.
        touch_hint_sources(fixture);
    }
    fixture.ledger.refuse_reserve = fixture.ledger.refuse_commit = false;
    fixture.closed();
    for (const auto& identity : identities) require(identity.lifetime.expired(), "Final close retained a hinted alias lifetime");
}

void captured_alias_busy_hint(Fixture& fixture) {
    stage("ws-hint-captured-alias-busy-no-os-call");
    fixture.make_sources(3); const auto plan = fixture.configure();
    const auto identities = hint_identities(fixture);
    const auto before = Views::stats(); const auto hints = Views::test_stats();
    const auto reserves = fixture.ledger.reserves, commits = fixture.ledger.commits;
    const auto releases = fixture.ledger.releases, ticket = fixture.ledger.last_reserved;
    auto held = fixture.sources[1]->source_view_lease();
    require(Views::pressure_hint(fixture.token) == 0, "Borrowed view alias was not an advisory busy hint refusal");
    const auto after = Views::test_stats();
    require(after.hint_busy == hints.hint_busy + 1 && after.hint_calls == hints.hint_calls
        && after.hint_accepted == hints.hint_accepted && Views::execution_ready(),
        "Busy alias hint performed an OS call or disabled healthy service");
    unchanged_hint_ownership(fixture, plan, before, identities, reserves, commits, releases, ticket);
    held.reset();
    require(Views::pressure_hint(fixture.token) == 1, "Released alias did not permit a quiescent hint");
    touch_hint_sources(fixture);
    fixture.closed();
}

void active_copy_busy_hint(Fixture& fixture) {
    stage("ws-hint-active-copy-leases-busy-no-os-call");
    fixture.make_sources(); const auto plan = fixture.configure();
    constexpr std::size_t count = 8, width = 32771;
    std::array<std::vector<std::uint8_t>, count> outputs;
    for (auto& output : outputs) output.assign(width + 26, 0xDA);
    std::array<tsg::ExpertFileSourceReadActivity, count> activity{};
    std::atomic<int> errors{0};
    struct Workers {
        std::vector<std::thread> values;
        ~Workers() { join(); }
        void join() { Views::test_pause_copies(false); for (auto& thread : values) if (thread.joinable()) thread.join(); }
    } workers;
    const auto initial = Views::test_stats();
    Views::test_pause_copies(true);
    for (std::size_t i = 0; i < count; ++i) workers.values.emplace_back([&, i] {
        try { fixture.sources[0]->read(i * 127, outputs[i].data() + 13, width, &activity[i]); }
        catch (...) { ++errors; }
    });
    until([&] { return Views::stats().active_readers == count
        && Views::test_stats().paused_entries >= initial.paused_entries + count; },
        "Hint test did not acquire all actual active copy leases");
    const auto identities = hint_identities(fixture);
    const auto before = Views::stats(); const auto hints = Views::test_stats();
    const auto reserves = fixture.ledger.reserves, commits = fixture.ledger.commits;
    const auto releases = fixture.ledger.releases, ticket = fixture.ledger.last_reserved;
    require(Views::pressure_hint(fixture.token) == 0, "Working-set hint ran while actual source copies were active");
    const auto after = Views::test_stats();
    require(after.hint_busy == hints.hint_busy + 1 && after.hint_calls == hints.hint_calls
        && after.hint_accepted == hints.hint_accepted && Views::execution_ready(),
        "Busy reader hint performed an OS call or changed execution state");
    unchanged_hint_ownership(fixture, plan, before, identities, reserves, commits, releases, ticket);
    workers.join();
    require(errors == 0 && Views::stats().active_readers == 0, "Busy hint lost an already-acquired exact copy");
    for (std::size_t i = 0; i < count; ++i) {
        fixture.exact(outputs[i], 13, 0, i * 127, width);
        require(activity[i].view_calls == 1 && activity[i].view_bytes == width && activity[i].buffered_bytes == 0,
            "Busy hint downgraded or partially published an active mapped copy");
    }
    require(Views::pressure_hint(fixture.token) == 1, "Joined source workers did not permit a fresh hint");
    fixture.closed();
}

void unexpected_hint_advisory(Fixture& fixture) {
    stage("ws-hint-partial-unexpected-result-advisory-still-exact");
    fixture.make_sources(3); const auto plan = fixture.configure();
    const auto identities = hint_identities(fixture);
    const auto before = Views::stats(); const auto hints = Views::test_stats();
    const auto reserves = fixture.ledger.reserves, commits = fixture.ledger.commits;
    const auto releases = fixture.ledger.releases, ticket = fixture.ledger.last_reserved;
    require(Views::test_hint_failure_after(1), "Cannot arm the unexpected-result protocol seam");
    require(Views::pressure_hint(fixture.token) == 0 && Views::execution_ready(),
        "Unexpected OS hint result was not advisory with unchanged executable views");
    const auto after = Views::test_stats();
    const auto calls = after.hint_calls - hints.hint_calls;
    require(after.hint_fault_fired == hints.hint_fault_fired + 1
        && after.hint_unexpected == hints.hint_unexpected + 1 && calls >= 2 && calls <= fixture.sources.size()
        && after.hint_accepted - hints.hint_accepted + 1 == calls
        && after.hint_completed == hints.hint_completed && after.hint_busy == hints.hint_busy,
        "Synthetic unexpected result did not follow an actual accepted range and another real OS call");
    unchanged_hint_ownership(fixture, plan, before, identities, reserves, commits, releases, ticket);
    touch_hint_sources(fixture); // No fallback, resume failure or new mapping.
    require(Views::pressure_hint(fixture.token) == 1 && Views::execution_ready(),
        "Advisory hint result prevented a later legitimate quiescent hint");
    unchanged_hint_ownership(fixture, plan, before, identities, reserves, commits, releases, ticket);
    fixture.closed();
}

void closing_hint_refusal(Fixture& fixture) {
    stage("ws-hint-closing-owner-fatal-no-os-call");
    fixture.make_sources(); fixture.configure();
    auto held = fixture.sources[0]->source_view_lease();
    require(!Views::close(fixture.token), "Cannot retain a real closing owner for the checked hint refusal");
    const auto before = Views::stats(); const auto hints = Views::test_stats();
    const auto releases = fixture.ledger.releases;
    require(Views::pressure_hint(fixture.token) == -1 && !Views::execution_ready(),
        "Closing native owner was confused with an advisory OS hint refusal");
    require(Views::stats().live_views == before.live_views
        && Views::stats().held_control_ticket_bytes == before.held_control_ticket_bytes
        && Views::test_stats().hint_calls == hints.hint_calls && fixture.ledger.releases == releases,
        "Invalid closing-owner hint touched mappings or refunded the retained ticket");
    held.reset(); fixture.closed();
}

using DemandPrefetch = tsg::ExpertFileSourceViewPrefetch;
using PrefetchActivity = tsg::ExpertFileSourceViewPrefetchActivity;

void exact_demand_prefetch(Fixture& fixture) {
    stage("demand-prefetch-real-api-bounded-batches-and-preflight");
    require(DemandPrefetch::available(), "Required Windows PrefetchVirtualMemory unavailable");
    require(DemandPrefetch::parse_option(nullptr) == 0 && DemandPrefetch::parse_option("0") == 0
        && DemandPrefetch::parse_option("1") == 1, "Demand prefetch default/ON option differs");
    for (const char* invalid : {"", "01", " 1", "2", "true"})
        require(DemandPrefetch::parse_option(invalid) == -1, "Malformed demand prefetch option accepted");
    fixture.make_sources(3); const auto plan = fixture.configure();
    const auto identities = hint_identities(fixture); const auto before = Views::stats();
    const auto reserves = fixture.ledger.reserves, commits = fixture.ledger.commits;
    const auto releases = fixture.ledger.releases, ticket = fixture.ledger.last_reserved;
    constexpr std::size_t count = 129, width = 17, prefix = 7;
    std::vector<std::vector<std::uint8_t>> outputs(count, std::vector<std::uint8_t>(prefix + width + 7, 0xDA));
    std::vector<tsg::ExpertReadTask> tasks;
    for (std::size_t i = 0; i < count; ++i)
        tasks.push_back({fixture.sources[i % 3].get(), i * 19, outputs[i].data() + prefix, width});
    PrefetchActivity activity;
    const auto actual_before = DemandPrefetch::test_stats().calls;
    DemandPrefetch::prepare(tasks.data(), tasks.size(), activity);
    require(activity.calls == 3 && activity.ranges == count && activity.bytes == count * width
        && activity.failures == 0 && DemandPrefetch::test_stats().calls == actual_before + 3,
        "Exact ranges did not reach the actual API in bounded64 descriptor batches");
    std::fprintf(stderr, "[SOURCE-VIEW-DEMAND-PREFETCH-TEST] case=batches actual_calls=%llu ranges=%llu bytes=%llu failures=%llu scope=logical_api_ranges_not_residency_or_physical_io\n",
        static_cast<unsigned long long>(activity.calls), static_cast<unsigned long long>(activity.ranges),
        static_cast<unsigned long long>(activity.bytes), static_cast<unsigned long long>(activity.failures));
    for (std::size_t i = 0; i < count; ++i) {
        tasks[i].prepare(); fixture.exact(outputs[i], prefix, i % 3, i * 19, width);
        require(tasks[i].file_view_activity.view_calls == 1 && tasks[i].file_view_activity.view_bytes == width
            && tasks[i].file_view_activity.buffered_bytes == 0, "Prefetch changed exact mapped copy service");
    }
    unchanged_hint_ownership(fixture, plan, before, identities, reserves, commits, releases, ticket);
    const auto calls = DemandPrefetch::test_stats().calls;
    auto invalid = tasks; invalid[1].offset = source_extent;
    PrefetchActivity rejected_activity;
    rejected([&] { DemandPrefetch::prepare(invalid.data(), invalid.size(), rejected_activity); },
        "Invalid later task was prefetched instead of rejecting the complete preflight");
    require(rejected_activity.calls == 0 && DemandPrefetch::test_stats().calls == calls,
        "Invalid range reached an OS call before whole-window preflight");
    invalid = tasks; invalid[0].bytes = DemandPrefetch::maximum_window_bytes + 1;
    rejected([&] { DemandPrefetch::prepare(invalid.data(), invalid.size(), rejected_activity); },
        "Beyond-window demand prefetch accepted");
    require(rejected_activity.calls == 0 && DemandPrefetch::test_stats().calls == calls,
        "Beyond-window request reached the actual API");
    PrefetchActivity empty;
    DemandPrefetch::prepare(nullptr, 0, empty);
    require(empty.calls == 0 && empty.ranges == 0 && empty.bytes == 0 && empty.failures == 0,
        "Empty current demand fabricated prefetch activity");
    fixture.closed();
}

void advisory_demand_prefetch(Fixture& fixture) {
    stage("demand-prefetch-advisory-after-real-call-and-full-fallback");
    fixture.make_sources(); const auto plan = fixture.configure();
    const auto before = Views::stats(); const auto identities = hint_identities(fixture);
    const auto reserves = fixture.ledger.reserves, commits = fixture.ledger.commits;
    const auto releases = fixture.ledger.releases, ticket = fixture.ledger.last_reserved;
    constexpr std::size_t prefix = 13, width = 151, relative = 27;
    std::vector<std::uint8_t> output(prefix + width + 17, 0xDA);
    tsg::ExpertReadTask task{fixture.sources[0].get(), relative, output.data() + prefix, width};
    const auto test_before = DemandPrefetch::test_stats();
    require(DemandPrefetch::test_failure_after(0), "Cannot arm post-real-API advisory result");
    PrefetchActivity activity; DemandPrefetch::prepare(&task, 1, activity);
    const auto test_after = DemandPrefetch::test_stats();
    require(activity.calls == 1 && activity.ranges == 1 && activity.bytes == width && activity.failures == 1
        && test_after.calls == test_before.calls + 1 && test_after.faults == test_before.faults + 1,
        "Advisory fixture did not consume the hook after a real API call");
    unchanged_hint_ownership(fixture, plan, before, identities, reserves, commits, releases, ticket);
    Views::test_fault(Fault::copy_partial);
    task.prepare(); fixture.exact(output, prefix, 0, relative, width);
    require(task.file_view_activity.view_calls == 1 && task.file_view_activity.view_fallbacks == 1
        && task.file_view_activity.view_bytes == 0 && task.file_view_activity.buffered_bytes == width,
        "Demand prefetch bypassed the partial-copy WHOLE authoritative fallback");
    PrefetchActivity disabled; DemandPrefetch::prepare(&task, 1, disabled);
    require(disabled.calls == 0 && disabled.bytes == 0 && DemandPrefetch::test_stats().calls == test_after.calls,
        "Disabled source generation fabricated prefetch or caused extra fault service");
    task.prepare(); fixture.exact(output, prefix, 0, relative, width);
    require(task.file_view_activity.view_calls == 0 && task.file_view_activity.buffered_bytes == width,
        "Disabled mapped generation lost its original buffered service");
    fixture.closed();
}

void held_demand_prefetch(Fixture& fixture) {
    stage("demand-prefetch-real-call-lease-blocks-close-and-pressure");
    fixture.make_sources(); fixture.configure();
    std::vector<std::uint8_t> output(4096, 0xDA);
    tsg::ExpertReadTask task{fixture.sources[0].get(), 19, output.data(), output.size()};
    PrefetchActivity activity; std::exception_ptr failure;
    const auto pauses = DemandPrefetch::test_stats().pauses;
    const auto releases = fixture.ledger.releases;
    struct Worker {
        std::thread thread;
        ~Worker() { DemandPrefetch::test_pause(false); if (thread.joinable()) thread.join(); }
    } worker;
    DemandPrefetch::test_pause(true);
    worker.thread = std::thread([&] { try { DemandPrefetch::prepare(&task, 1, activity); }
        catch (...) { failure = std::current_exception(); } });
    until([&] { return DemandPrefetch::test_stats().pauses == pauses + 1; },
        "Actual prefetch API did not reach the bounded lease pause");
    require(Views::stats().active_readers == 1 && Views::pressure_hint(fixture.token) == 0,
        "Actual prefetch lease was invisible to quiescent pressure hint");
    require(!Views::close(fixture.token) && fixture.ledger.releases == releases
        && Views::stats().live_views == 1, "Close freed or refunded an actual in-flight prefetch owner");
    DemandPrefetch::test_pause(false); worker.thread.join();
    if (failure) std::rethrow_exception(failure);
    require(activity.calls == 1 && activity.ranges == 1 && Views::stats().active_readers == 0,
        "Prefetch lease did not release after the actual OS call");
    fixture.closed();
}

void current_window_demand_prefetch(Fixture& fixture) {
    stage("demand-prefetch-exact-two-slot-worker-pipeline");
    fixture.make_sources(3); const auto plan = fixture.configure();
    const auto before = Views::stats(); const auto identities = hint_identities(fixture);
    const auto reserves = fixture.ledger.reserves, commits = fixture.ledger.commits;
    const auto releases = fixture.ledger.releases, ticket = fixture.ledger.last_reserved;
    constexpr std::size_t windows = 4, parts = 3, width = 8191, guard = 13;
    std::array<std::vector<std::uint8_t>, 2> arena;
    for (auto& slot : arena) slot.assign(guard + parts * width + guard, 0xDA);
    std::vector<tsg::ExpertReadTask> tasks;
    for (std::size_t w = 0; w < windows; ++w) for (std::size_t p = 0; p < parts; ++p)
        tasks.push_back({fixture.sources[p].get(), w * 997 + p * 31,
            arena[w % 2].data() + guard + p * width, width});
    PrefetchActivity total; std::size_t consumed = 0;
    std::mutex prefetch_mutex;
    std::array<std::size_t, 2> expected{0, 1};
    tsg_dsv41::engram_io_pool pool(8);
    tsg::run_bounded_parallel_read_pipeline(pool, std::vector<std::size_t>(windows, parts), 2,
        [&](std::size_t w, std::size_t slot) {
            require(expected[slot] == w, "Future window passed its original slot reuse boundary");
        }, [&](std::size_t w, std::size_t p, std::size_t) {
            auto& task = tasks[w * parts + p];
            PrefetchActivity one; DemandPrefetch::prepare(&task, 1, one);
            { std::lock_guard<std::mutex> lock(prefetch_mutex); total.add(one); }
            task.prepare();
        },
        [&](std::size_t w, std::size_t slot) {
            ++consumed;
            for (std::size_t p = 0; p < parts; ++p) {
                std::vector<std::uint8_t> exact(guard + width + guard, 0xDA);
                std::memcpy(exact.data() + guard, tasks[w * parts + p].destination, width);
                fixture.exact(exact, guard, p, w * 997 + p * 31, width);
            }
            require(std::all_of(arena[slot].begin(), arena[slot].begin() + guard, [](auto x) { return x == 0xDA; })
                && std::all_of(arena[slot].end() - guard, arena[slot].end(), [](auto x) { return x == 0xDA; }),
                "Current-window prefetch/copy overwrote original arena guards");
            expected[slot] += 2;
        });
    require(consumed == windows && total.calls == tasks.size() && total.ranges == tasks.size()
        && total.bytes == windows * parts * width && total.failures == 0,
        "Joined exact-demand windows changed task, range, byte or completion counts");
    std::fprintf(stderr, "[SOURCE-VIEW-DEMAND-PREFETCH-TEST] case=current_worker_windows actual_calls=%llu ranges=%llu bytes=%llu failures=%llu consumed=%zu scope=cpu_window_protocol_not_cuda_or_physical_io\n",
        static_cast<unsigned long long>(total.calls), static_cast<unsigned long long>(total.ranges),
        static_cast<unsigned long long>(total.bytes), static_cast<unsigned long long>(total.failures), consumed);
    unchanged_hint_ownership(fixture, plan, before, identities, reserves, commits, releases, ticket);
    fixture.closed();
}

void failed_worker_demand_prefetch(Fixture& fixture) {
    stage("demand-prefetch-worker-copy-fallback-failure-joined-no-publication");
    fixture.make_sources(3); const auto plan = fixture.configure();
    const auto before = Views::stats();
    const auto source_test_before = Views::test_stats();
    const auto api_calls = DemandPrefetch::test_stats().calls;
    const auto reserves = fixture.ledger.reserves, commits = fixture.ledger.commits;
    const auto releases = fixture.ledger.releases, ticket = fixture.ledger.last_reserved;
    constexpr std::size_t width = 8191;
    std::array<std::vector<std::uint8_t>, 3> output;
    std::vector<tsg::ExpertReadTask> tasks;
    for (std::size_t p = 0; p < output.size(); ++p) {
        output[p].assign(width, 0xDA);
        tasks.push_back({fixture.sources[p].get(), p * 31, output[p].data(), width});
    }
    PrefetchActivity attempted;
    std::mutex prefetch_mutex;
    std::atomic<unsigned> active{0};
    unsigned consumed = 0; bool failed = false, published = false;
    Views::test_fault(Fault::copy_partial); Views::test_fail_buffered_reads(true);
    tsg_dsv41::engram_io_pool pool(8);
    try {
        tsg::run_bounded_parallel_read_pipeline(pool, std::vector<std::size_t>{tasks.size()}, 2,
            [](std::size_t, std::size_t) {},
            [&](std::size_t, std::size_t p, std::size_t) {
                struct Active {
                    std::atomic<unsigned>& value;
                    explicit Active(std::atomic<unsigned>& v) : value(v) { value.fetch_add(1); }
                    ~Active() { value.fetch_sub(1); }
                } activity(active);
                PrefetchActivity one; DemandPrefetch::prepare(&tasks[p], 1, one);
                { std::lock_guard<std::mutex> lock(prefetch_mutex); attempted.add(one); }
                tasks[p].prepare(); // Actual mapped prefix, then actual buffered strict prefix, then throw.
            }, [&](std::size_t, std::size_t) { ++consumed; });
        published = true; // Exactly as production, no joined-success merge occurs on exception.
    } catch (const std::exception&) { failed = true; }
    Views::test_fail_buffered_reads(false); Views::test_fault(Fault::none);
    const auto after_test = Views::test_stats(); const auto after = Views::stats();
    require(failed && !published && consumed == 0 && active.load() == 0 && after.active_readers == 0,
        "Failed per-worker task left work/readers active or published its incomplete window");
    require(after_test.partial_fired == source_test_before.partial_fired + 1
        && after_test.buffered_failures == source_test_before.buffered_failures + 1
        && DemandPrefetch::test_stats().calls > api_calls && attempted.calls > 0
        && attempted.calls == attempted.ranges && attempted.bytes == attempted.calls * width,
        "Worker failure did not follow actual prefetch and actual mapped/buffered prefixes");
    require(after.live_views == before.live_views && after.owned_records == before.owned_records
        && after.owned_control_bytes == before.owned_control_bytes
        && after.held_control_ticket_bytes == before.held_control_ticket_bytes
        && after.map_calls == before.map_calls && after.view_close_calls == before.view_close_calls
        && fixture.ledger.reserves == reserves && fixture.ledger.commits == commits
        && fixture.ledger.releases == releases && fixture.ledger.last_reserved == ticket
        && fixture.ledger.held() == plan.owned_control_bytes && !fixture.ledger.failed,
        "Failed worker-prefetch invocation mutated or refunded exact source-view ownership");
    std::fprintf(stderr, "[SOURCE-VIEW-DEMAND-PREFETCH-TEST] case=worker_failed_join actual_calls=%llu ranges=%llu bytes=%llu consumer_calls=%u active_workers=%u active_readers=%llu partial_mapped=1 partial_buffered=1 joined_success_published=0 scope=injected_protocol_after_real_api_and_source_prefix_not_os_failure\n",
        static_cast<unsigned long long>(attempted.calls), static_cast<unsigned long long>(attempted.ranges),
        static_cast<unsigned long long>(attempted.bytes), consumed, active.load(),
        static_cast<unsigned long long>(after.active_readers));
    fixture.closed();
}
} // namespace

int main(int argc, char** argv) {
    std::filesystem::path directory = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path()
        / "artifacts/perf/glm-memory-tier-20261015/expert-file-source-view-lifecycle";
    if (argc == 3 && std::strcmp(argv[1], "--directory") == 0) directory = std::filesystem::u8path(argv[2]);
    else if (argc != 1) { std::fprintf(stderr, "Usage: expert_file_source_view_test [--directory IGNORED_OUTPUT_DIRECTORY]\n"); return 2; }
    std::unique_ptr<Fixture> fixture;
    try {
        fixture.reset(new Fixture(directory));
        ranges_and_identity(*fixture);
        held_alias(*fixture);
        concurrent_busy_close(*fixture);
        copy_fault(*fixture, Fault::copy_partial);
        copy_fault(*fixture, Fault::copy_in_page);
        failed_fallback_join(*fixture);
        for (auto fault : {Fault::duplicate, Fault::section, Fault::map}) setup_failure(*fixture, fault);
        for (auto fault : {Fault::unmap, Fault::close_section, Fault::close_file}) cleanup_failure(*fixture, fault);
        acknowledgment_failure(*fixture, false);
        acknowledgment_failure(*fixture, true);
        working_set_hints(*fixture, 3);
        working_set_hints(*fixture, 126);
        captured_alias_busy_hint(*fixture);
        active_copy_busy_hint(*fixture);
        unexpected_hint_advisory(*fixture);
        closing_hint_refusal(*fixture);
        exact_demand_prefetch(*fixture);
        advisory_demand_prefetch(*fixture);
        held_demand_prefetch(*fixture);
        current_window_demand_prefetch(*fixture);
        failed_worker_demand_prefetch(*fixture);
        require(fixture->cleanup_noexcept(), "Final generated fixture/checked ledger cleanup failed");
        std::printf("SOURCE-VIEW-LIFECYCLE PASS checks=%u failed=0 skipped=0 scope=generated_file_cpu_lifecycle_not_model_storage_or_performance\n", checks.load());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "SOURCE-VIEW-LIFECYCLE FAIL checks=%u error=%s\n", checks.load(), error.what());
        if (fixture && !fixture->cleanup_noexcept()) {
            std::fprintf(stderr, "SOURCE-VIEW-LIFECYCLE retained failed owner/ledger/source for process teardown\n");
            (void)fixture.release();
        }
        return 1;
    }
}
#else
int main() {
    std::fprintf(stderr, "Source-view lifecycle unavailable: Windows/MSVC SEH required\n");
    return 77;
}
#endif
