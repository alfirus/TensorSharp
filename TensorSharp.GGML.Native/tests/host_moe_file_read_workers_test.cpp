// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "../ggml_ops_file_read_workers.h"
#include "../dsv41_engram_io.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

namespace {
std::size_t checks = 0;
void check(bool value, const char* message) {
    ++checks; if (!value) throw std::runtime_error(message);
}
void geometry(const char* value, int available, unsigned requested, unsigned workers,
              unsigned slots, unsigned maximum, unsigned original_slots) {
    const auto plan = tsg::plan_host_moe_file_read_workers(value, available);
    check(plan.requested == requested && plan.workers == workers && plan.pool_slots == slots
        && plan.maximum == maximum && plan.default_pool_slots == original_slots, "Worker plan geometry differs");
}
void policy() {
    geometry(nullptr, 16, 0, 15, 16, 15, 16);
    geometry("0", 16, 0, 15, 16, 15, 16);
    geometry("8", 16, 8, 8, 9, 15, 16);
    geometry("4", 16, 4, 4, 5, 15, 16);
    geometry("16", 16, 16, 15, 16, 15, 16);
    geometry("1", 16, 1, 1, 1, 15, 16);
    geometry(nullptr, 2, 0, 1, 2, 1, 2);
    geometry("1", 2, 1, 1, 1, 1, 2);
    geometry("8", 4, 8, 3, 4, 3, 4);
    geometry(nullptr, 1, 0, 1, 1, 1, 1);
    geometry("16", 1, 16, 1, 1, 1, 1);
    geometry(nullptr, 128, 0, 15, 16, 15, 16);
    for (const char* bad : {"", "00", "01", "08", "17", "-1", "+8", " 8", "8 ", "1.0", "8x", "999999999999999999999999999999"}) {
        bool refused = false;
        try { (void)tsg::plan_host_moe_file_read_workers(bad, 16); }
        catch (const std::invalid_argument&) { refused = true; }
        check(refused, "Malformed or out-of-range worker flag accepted");
    }
    bool refused = false;
    try { (void)tsg::plan_host_moe_file_read_workers("8", 0); }
    catch (const std::invalid_argument&) { refused = true; }
    check(refused, "Nonpositive available parallelism accepted");
}

void joined_workers() {
    const auto plan = tsg::plan_host_moe_file_read_workers("8", 16);
    tsg_dsv41::engram_io_pool pool(plan.pool_slots);
    const auto caller = std::this_thread::get_id();
    std::atomic<unsigned> active{0}, peak{0};
    std::mutex mutex;
    std::condition_variable changed;
    unsigned arrived = 0;
    std::vector<unsigned> prepared(24), consumed(24);
    pool.run_pipelined(prepared.size(), [&](std::size_t task) {
        if (std::this_thread::get_id() == caller) throw std::runtime_error("Parallel submitter prepared a read");
        const unsigned now = active.fetch_add(1) + 1;
        struct Leave { std::atomic<unsigned>& value; ~Leave() { --value; } } leave{active};
        unsigned old = peak.load();
        while (old < now && !peak.compare_exchange_weak(old, now)) {}
        if (task < 8) {
            std::unique_lock<std::mutex> lock(mutex);
            ++arrived; changed.notify_all();
            if (!changed.wait_for(lock, std::chrono::seconds(10), [&] { return arrived == 8; }))
                throw std::runtime_error("Eight requested preparations could not run concurrently");
        }
        ++prepared[task];
    }, [&](std::size_t task) {
        check(std::this_thread::get_id() == caller && prepared[task] == 1, "Consume escaped caller or preceded preparation");
        ++consumed[task];
    });
    check(peak == 8 && active == 0, "Actual joined preparation width differs from requested eight");
    for (unsigned value : consumed) check(value == 1, "Prepared task was consumed more or less than once");
    bool failed = false;
    try {
        pool.run_pipelined(24, [&](std::size_t task) {
            ++active;
            struct Leave { std::atomic<unsigned>& value; ~Leave() { --value; } } leave{active};
            if (task == 3) throw std::runtime_error("Injected source failure");
        }, [](std::size_t) {});
    } catch (const std::runtime_error&) { failed = true; }
    check(failed && active == 0, "Source failure escaped before all preparations joined");
    unsigned recovered = 0;
    pool.run_pipelined(5, [](std::size_t) {}, [&](std::size_t) { ++recovered; });
    check(recovered == 5, "Joined pool did not recover after source failure");
    const auto serial = tsg::plan_host_moe_file_read_workers("1", 16);
    tsg_dsv41::engram_io_pool one(serial.pool_slots);
    unsigned serial_prepared = 0;
    one.run_pipelined(3, [&](std::size_t) {
        check(std::this_thread::get_id() == caller, "Explicit one-reader mode was not serial"); ++serial_prepared;
    }, [](std::size_t) {});
    check(serial_prepared == 3, "Serial task coverage differs");
}

void activity() {
    const auto plan = tsg::plan_host_moe_file_read_workers("8", 16);
    tsg::HostMoeFileReadActivity hints;
    hints.observe_source_hint(true, 0);
    check(hints.valid && !hints.source_hint_seen && hints.random_access == 0,
        "Empty or cached tasks certified a source handle hint");
    hints.observe_source_hint(false, 37);
    hints.observe_source_hint(true, 0);
    check(hints.valid && hints.source_hint_seen && hints.random_access == 0,
        "A zero-byte hinted task changed actual buffered-read activity");
    hints.observe_source_hint(true, 19);
    check(!hints.valid, "Mixed actual buffered and hinted source handles certified one scope");
    tsg::HostMoeFileReadActivity hinted;
    hinted.observe_source_hint(true, 19);
    hinted.observe_source_hint(true, 37);
    hinted.observe_source_hint(false, 0);
    check(hinted.valid && hinted.source_hint_seen && hinted.random_access == 1,
        "Repeated actual hinted reads lost the successfully opened handle flag");
    hinted.observe_source_hint(false, 1);
    check(!hinted.valid, "Reverse-order mixed source hints were silently certified");
    tsg::HostMoeFileReadActivity value;
    value.add(plan, 24, 123456, true);
    value.add(plan, 3, 98765, false);
    check(value.valid && value.requested == 8 && value.workers == 8 && value.pool_slots == 9
        && value.calls == 2 && value.tasks == 27 && value.completed_tasks == 24
        && value.failed_calls == 1 && value.source_bytes == 123456, "Failed joined invocation certified partial tasks/bytes");
    value.add(plan, 0, 1, true);
    check(value.calls == 2 && value.source_bytes == 123456, "Empty task invocation supplied activity credit");
    tsg::HostMoeFileReadActivity mismatch = value;
    mismatch.add(tsg::plan_host_moe_file_read_workers("4", 16), 1, 1, true);
    check(!mismatch.valid, "Mixed first-use configurations certified one activity scope");
    value.calls = (std::numeric_limits<std::uint64_t>::max)();
    value.add(plan, 1, 1, true);
    check(!value.valid, "Telemetry counter overflow was silently certified");
    tsg::g_host_moe_file_read_activity = &value;
    bool isolated = false;
    std::thread worker([&] { isolated = tsg::g_host_moe_file_read_activity == nullptr; });
    worker.join();
    check(isolated && tsg::g_host_moe_file_read_activity == &value, "Caller activity borrow leaked into worker TLS");
    tsg::g_host_moe_file_read_activity = nullptr;
}
} // namespace

int main() {
    try {
        policy(); joined_workers(); activity();
        std::printf("PASS host MoE file-read worker policy and joined execution: %zu checks\n", checks);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL file-read workers: %s\n", error.what()); return 1;
    }
}
