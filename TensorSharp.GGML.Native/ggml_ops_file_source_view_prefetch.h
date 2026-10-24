// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include "ggml_ops_file_source_view.h"

namespace tsg {
// Submitted logical ranges, not copied bytes, resident pages or physical I/O.
struct ExpertFileSourceViewPrefetchActivity {
    std::uint64_t calls = 0, ranges = 0, bytes = 0, failures = 0;
    void add(const ExpertFileSourceViewPrefetchActivity& value) {
        const auto sum = [](std::uint64_t& target, std::uint64_t amount) {
            if (amount > UINT64_MAX - target) throw std::overflow_error("Source prefetch activity overflow");
            target += amount;
        };
        sum(calls, value.calls); sum(ranges, value.ranges);
        sum(bytes, value.bytes); sum(failures, value.failures);
    }
};

class ExpertFileSourceViewPrefetch {
    struct TestState {
        std::atomic<int> failure_after{-1};
        std::atomic<bool> pause{false};
        std::atomic<std::uint64_t> calls{0}, faults{0}, pauses{0};
    };
    static TestState& tests() noexcept { static TestState state; return state; }
#if defined(_WIN32) && defined(_MSC_VER)
    // Public WIN32_MEMORY_RANGE_ENTRY ABI, also usable when an including
    // project's SDK target predates WIN8. No private backend/context layout.
    struct Range { PVOID address; SIZE_T bytes; };
    static_assert(sizeof(Range) == sizeof(PVOID) + sizeof(SIZE_T)
        && offsetof(Range, bytes) == sizeof(PVOID), "Public Windows range ABI differs");
    using Function = BOOL (WINAPI*)(HANDLE, ULONG_PTR, Range*, ULONG);
    static Function function() noexcept {
        static const auto value = reinterpret_cast<Function>(
            GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory"));
        return value;
    }
#endif
public:
    static constexpr std::size_t maximum_ranges = 64;
    static constexpr std::uint64_t maximum_window_bytes = std::uint64_t(72) << 20;
    struct TestStats { std::uint64_t calls = 0, faults = 0, pauses = 0; };
    static int parse_option(const char* value) noexcept {
        if (!value || std::strcmp(value, "0") == 0) return 0;
        return std::strcmp(value, "1") == 0 ? 1 : -1;
    }
    static int option() noexcept {
        static const int value = parse_option(std::getenv("TS_HOST_MOE_FILE_SOURCE_VIEW_PREFETCH"));
        return value;
    }
    static bool available() noexcept {
#if defined(_WIN32) && defined(_MSC_VER)
        return function() != nullptr;
#else
        return false;
#endif
    }
    // Caller retains every Source through the original joined read operation.
    // Only its current admitted <=72MiB window is eligible, never a whole layer.
    // No API result changes view/fallback state or grants any budget credit.
    static void prepare(const ExpertReadTask* tasks, std::size_t count,
        ExpertFileSourceViewPrefetchActivity& activity) {
        if (!count) return;
        if (!tasks || !available()) throw std::runtime_error("Source-view demand prefetch unavailable");
        std::uint64_t bytes = 0;
        for (std::size_t i = 0; i < count; ++i) {
            const auto& task = tasks[i];
            if (!task.source || task.offset > task.source->bytes()
                || task.bytes > task.source->bytes() - task.offset
                || task.bytes > maximum_window_bytes - bytes || task.direct_bytes
                || task.payload_cache || task.immutable_read)
                throw std::invalid_argument("Source prefetch requires exact uncached current-window ranges");
            bytes += task.bytes;
        }
#if defined(_WIN32) && defined(_MSC_VER)
        for (std::size_t first = 0; first < count;) {
            struct Batch {
                std::array<std::shared_ptr<ExpertFileSourceViewRecord>, maximum_ranges> owners{};
                std::array<Range, maximum_ranges> ranges{};
                std::size_t count = 0;
                std::uint64_t bytes = 0;
                ~Batch() {
                    for (std::size_t i = 0; i < count; ++i)
                        owners[i]->readers.fetch_sub(1, std::memory_order_release);
                }
            } batch;
            for (; first < count && batch.count < maximum_ranges; ++first) {
                const auto& task = tasks[first];
                if (!task.bytes) continue;
                auto owner = task.source->source_view_lease();
                if (!owner || owner->identity != task.source->identity() || owner->bytes != task.source->bytes()
                    || !owner->base || owner->pending || owner->quarantined
                    || owner->delta > owner->map_bytes || owner->bytes > owner->map_bytes - owner->delta)
                    throw std::runtime_error("Source prefetch owner/generation differs");
                // Disabled views retain the original authoritative buffered
                // fallback. Never fault their pages merely to claim activity.
                if (owner->disabled.load(std::memory_order_acquire)) continue;
                owner->readers.fetch_add(1);
                const auto index = batch.count++;
                batch.owners[index] = std::move(owner);
                if (batch.owners[index]->retiring.load(std::memory_order_acquire))
                    throw std::runtime_error("Source prefetch owner is retiring");
                batch.ranges[index] = {const_cast<std::uint8_t*>(batch.owners[index]->base)
                    + batch.owners[index]->delta + static_cast<std::size_t>(task.offset), task.bytes};
                batch.bytes += task.bytes;
            }
            if (!batch.count) continue;
            const BOOL result = function()(GetCurrentProcess(), static_cast<ULONG_PTR>(batch.count), batch.ranges.data(), 0);
            bool accepted = result != 0;
#if defined(TSG_GGML_TEST_HOOKS)
            auto& test = tests(); test.calls.fetch_add(1); // Reached only after an ACTUAL OS call.
            int remaining = test.failure_after.load();
            while (remaining >= 0) {
                if (test.failure_after.compare_exchange_weak(remaining, remaining == 0 ? -1 : remaining - 1)) {
                    if (remaining == 0) {
                        test.faults.fetch_add(1);
                        accepted = false; // Synthetic advisory result, never a claimed OS failure.
                    }
                    break;
                }
            }
            if (test.pause.load()) {
                test.pauses.fetch_add(1);
                const auto started = GetTickCount64();
                while (test.pause.load() && GetTickCount64() - started < 5000) Sleep(1);
                if (test.pause.load()) throw std::runtime_error("Source prefetch test pause timed out");
            }
#endif
            // Advisory: exact source copy still runs unchanged. Checked sums
            // also reject reuse of an already-overflowed diagnostic record.
            activity.add({1, static_cast<std::uint64_t>(batch.count), batch.bytes, accepted ? 0u : 1u});
        }
#else
        (void)activity;
#endif
    }
    static bool test_failure_after(int calls) noexcept {
#if defined(TSG_GGML_TEST_HOOKS)
        if (calls < -1 || calls > 128) return false;
        tests().failure_after.store(calls); return true;
#else
        (void)calls; return false;
#endif
    }
    static void test_pause(bool pause) noexcept {
#if defined(TSG_GGML_TEST_HOOKS)
        tests().pause.store(pause);
#else
        (void)pause;
#endif
    }
    static TestStats test_stats() noexcept {
#if defined(TSG_GGML_TEST_HOOKS)
        auto& value = tests(); return {value.calls.load(), value.faults.load(), value.pauses.load()};
#else
        return {};
#endif
    }
};
} // namespace tsg
