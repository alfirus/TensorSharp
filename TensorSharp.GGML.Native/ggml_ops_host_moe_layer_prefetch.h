// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include "ggml_ops_host_moe_payload_cache.h"
#include <condition_variable>
#include <thread>
#include <vector>
#include <chrono>

namespace tsg {
// Request-owned callbacks. Demand begins before the expert-cache mutex is
// acquired; before_compute runs only after its demand reads have completed.
struct HostMoePrefetchHooks {
    void* context = nullptr;
    void (*demand_begin)(void*) noexcept = nullptr;
    void (*before_compute)(void*) noexcept = nullptr;
};

using HostMoePrefetchLoader = std::function<void(std::uint64_t, void*, std::size_t)>;
// Native adapters never acquire the expert-cache mutex. Capacity and all
// storage, including pending or retired leases, belong to the existing cache.
std::size_t host_moe_prefetch_capacity();
std::uint64_t host_moe_prefetch_hit_bytes();
void host_moe_prefetch_discard();
HostMoePayloadCache::Lease host_moe_prefetch_acquire(std::uint64_t source,
    std::uint64_t base, std::size_t extent, const HostMoePrefetchLoader& loader);

// A speculative read is one complete, immutable expert projection. Pausing
// INSIDE a cache fill would deadlock a demand lookup waiting for that same
// entry. Instead pause between fills and drain the bounded in-flight set.
// No worker ever calls the GPU or changes route IDs or GPU cache residency.
class HostMoeLayerPrefetch {
public:
    struct Item {
        std::uint64_t source = 0, base = 0;
        std::size_t bytes = 0;
        HostMoePrefetchLoader read;
    };
    using Acquire = std::function<HostMoePayloadCache::Lease(std::uint64_t,
        std::uint64_t, std::size_t, const HostMoePrefetchLoader&)>;
    struct Stats {
        std::uint64_t layers = 0, planned_bytes = 0, read_bytes = 0, hit_bytes = 0,
            refused = 0, failures = 0, cancelled_items = 0, demand_wait_ns = 0;
        std::size_t retained_bytes = 0, peak_retained_bytes = 0, in_flight = 0;
    };
    static constexpr std::size_t maximum_item_bytes = std::size_t(16) << 20;
private:
    Acquire acquire_;
    std::mutex mutex_;
    std::condition_variable ready_, drained_;
    std::vector<std::thread> workers_;
    std::vector<Item> jobs_, prepared_;
    std::vector<HostMoePayloadCache::Lease> leases_, prepared_leases_;
    std::size_t cursor_ = 0, in_flight_ = 0, prepared_bytes_ = 0;
    bool stopped_ = false, paused_ = true, replace_ = false;
    Stats stats_;
    HostMoePrefetchHooks hooks_{this,
        [](void* p) noexcept { static_cast<HostMoeLayerPrefetch*>(p)->demand_begin(); },
        [](void* p) noexcept { static_cast<HostMoeLayerPrefetch*>(p)->before_compute(); }};

    void worker() noexcept {
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [&] { return stopped_ || (!paused_ && cursor_ < jobs_.size()); });
            if (stopped_) return;
            const auto index = cursor_++;
            ++in_flight_;
            // jobs_ cannot change until every in-flight fill finishes.
            const auto& item = jobs_[index];
            lock.unlock();
            HostMoePayloadCache::Lease lease;
            std::uint64_t read_bytes = 0;
            bool failed = false;
            try {
                lease = acquire_(item.source, item.base, item.bytes,
                    [&](std::uint64_t offset, void* target, std::size_t count) {
                        // Keep each source read small, but never pause a live
                        // cache entry. Cancellation drains at most one item per
                        // worker before dropping its original source owner.
                        constexpr std::size_t piece = std::size_t(1) << 20;
                        for (std::size_t done = 0; done < count; done += piece) {
                            const auto take = (std::min)(piece, count - done);
                            item.read(offset + done, static_cast<std::uint8_t*>(target) + done, take);
                            read_bytes += take;
                        }
                    });
                if (lease && (lease.bytes != item.bytes || !lease.data))
                    throw std::runtime_error("Invalid prefetched payload lease");
            } catch (...) { failed = true; }
            const bool accepted = lease && !failed;
            lock.lock();
            stats_.read_bytes += read_bytes;
            if (failed) ++stats_.failures;
            else if (!lease) ++stats_.refused;
            if (!stopped_ && accepted) {
                if (lease.cache_hit) stats_.hit_bytes += lease.bytes;
                stats_.retained_bytes += lease.bytes;
                stats_.peak_retained_bytes = (std::max)(stats_.peak_retained_bytes, stats_.retained_bytes);
                leases_[index] = std::move(lease);
            }
            // A refused/failed fill is optional. Stop the remaining scan so a
            // tight budget does not turn it into repeated discarded I/O.
            if (!accepted) {
                stats_.cancelled_items += jobs_.size() - cursor_;
                cursor_ = jobs_.size();
            }
            --in_flight_;
            if (!in_flight_) drained_.notify_all();
        }
    }
public:
    explicit HostMoeLayerPrefetch(Acquire acquire, unsigned threads = 2) : acquire_(std::move(acquire)) {
        if (!acquire_ || threads < 1 || threads > 8) throw std::invalid_argument("Invalid layer prefetch workers");
        try { for (unsigned i = 0; i < threads; ++i) workers_.emplace_back([this] { worker(); }); }
        catch (...) { stop(); throw; }
    }
    ~HostMoeLayerPrefetch() { stop(); }
    HostMoeLayerPrefetch(const HostMoeLayerPrefetch&) = delete;
    HostMoeLayerPrefetch& operator=(const HostMoeLayerPrefetch&) = delete;
    const HostMoePrefetchHooks* hooks() const noexcept { return &hooks_; }

    // Called once before the first chunk of a layer. This only prepares small
    // metadata; the old layer may still be loading. No physical payload is
    // allocated before the first before_compute transition releases it.
    bool prepare(std::vector<Item> items, std::size_t capacity) {
        std::size_t total = 0;
        for (const auto& item : items) {
            if (!item.source || !item.bytes || item.bytes > maximum_item_bytes || !item.read
                || item.base > UINT64_MAX - item.bytes || item.bytes > capacity - total) {
                discard_prepared(); return false;
            }
            total += item.bytes;
        }
        std::vector<HostMoePayloadCache::Lease> leases(items.size());
        std::lock_guard<std::mutex> lock(mutex_);
        prepared_ = std::move(items); prepared_leases_ = std::move(leases);
        prepared_bytes_ = total; replace_ = true;
        return true;
    }
    void discard_prepared() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        prepared_.clear(); prepared_leases_.clear(); prepared_bytes_ = 0; replace_ = true;
    }
    void demand_begin() noexcept {
        const auto start = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(mutex_);
        paused_ = true;
        drained_.wait(lock, [&] { return in_flight_ == 0; });
        stats_.demand_wait_ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count());
    }
    void before_compute() noexcept {
        demand_begin();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopped_) return;
            if (replace_) {
                stats_.cancelled_items += jobs_.size() - cursor_;
                leases_.clear(); jobs_.clear(); // old source/storage owners first
                stats_.retained_bytes = 0;
                jobs_.swap(prepared_); leases_.swap(prepared_leases_); cursor_ = 0;
                stats_.planned_bytes += prepared_bytes_;
                if (!jobs_.empty()) ++stats_.layers;
                prepared_bytes_ = 0; replace_ = false;
            }
            paused_ = false;
        }
        ready_.notify_all();
    }
    void stop() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!stopped_) stats_.cancelled_items += jobs_.size() - cursor_;
            stopped_ = true;
        }
        ready_.notify_all();
        for (auto& worker : workers_) if (worker.joinable()) worker.join();
        std::lock_guard<std::mutex> lock(mutex_);
        leases_.clear(); jobs_.clear(); prepared_leases_.clear(); prepared_.clear();
        stats_.retained_bytes = 0;
    }
    Stats stats() {
        std::lock_guard<std::mutex> lock(mutex_);
        auto result = stats_; result.in_flight = in_flight_; return result;
    }
};
} // namespace tsg
