// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "../ggml_ops_host_moe_layer_prefetch.h"
#include <atomic>
#include <future>
#include <iostream>
#include <stdexcept>

namespace {
std::size_t checks = 0;
void check(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
template<class Predicate> void wait_for(Predicate predicate) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!predicate()) {
        if (std::chrono::steady_clock::now() > end) throw std::runtime_error("Timed out waiting for prefetch worker");
        std::this_thread::yield();
    }
}
struct FakeStorage {
    std::atomic<std::size_t> live{0}, peak{0}, acquired{0};
    std::size_t capacity;
    std::mutex mutex;
    std::vector<std::weak_ptr<const void>> owners;
    explicit FakeStorage(std::size_t limit) : capacity(limit) {}
    tsg::HostMoePayloadCache::Lease acquire(std::uint64_t, std::uint64_t offset,
        std::size_t bytes, const tsg::HostMoePrefetchLoader& read) {
        auto before = live.load();
        do { if (bytes > capacity - before) return {}; }
        while (!live.compare_exchange_weak(before, before + bytes));
        auto previous_peak = peak.load();
        while (previous_peak < before + bytes && !peak.compare_exchange_weak(previous_peak, before + bytes)) {}
        std::shared_ptr<std::uint8_t> allocation;
        try {
            allocation = std::shared_ptr<std::uint8_t>(new std::uint8_t[bytes], [this, bytes](std::uint8_t* p) {
                delete[] p; live.fetch_sub(bytes);
            });
        } catch (...) { live.fetch_sub(bytes); throw; }
        read(offset, allocation.get(), bytes);
        {
            std::lock_guard<std::mutex> lock(mutex);
            owners.push_back(allocation);
        }
        ++acquired;
        return {allocation, allocation.get(), bytes, false, false};
    }
    tsg::HostMoeLayerPrefetch::Acquire callback() {
        return [this](auto source, auto offset, auto bytes, const auto& read) {
            return acquire(source, offset, bytes, read);
        };
    }
};
tsg::HostMoeLayerPrefetch::Item item(std::uint64_t base, std::size_t bytes,
    tsg::HostMoePrefetchLoader loader = {}) {
    if (!loader) loader = [](auto offset, void* destination, auto count) {
        auto* data = static_cast<std::uint8_t*>(destination);
        for (std::size_t i = 0; i < count; ++i) data[i] = static_cast<std::uint8_t>((offset + i) % 251);
    };
    return {1, base, bytes, std::move(loader)};
}

void exact_bytes_and_reuse() {
    constexpr std::size_t bytes = (1u << 20) + 31;
    FakeStorage storage(bytes);
    tsg::HostMoeLayerPrefetch work(storage.callback(), 2);
    std::atomic<int> reads{0};
    auto original = item(123, bytes);
    auto read = original.read;
    original.read = [&](auto offset, void* target, auto count) { ++reads; read(offset, target, count); };
    check(work.prepare({original}, bytes), "valid plan refused");
    check(storage.live == 0, "prepare allocated payload before compute");
    work.hooks()->demand_begin(work.hooks()->context);
    work.hooks()->before_compute(work.hooks()->context);
    wait_for([&] { return work.stats().retained_bytes == bytes; });
    work.demand_begin();
    check(reads == 2, "large extent was not read in bounded pieces");
    std::shared_ptr<const void> owner;
    { std::lock_guard<std::mutex> lock(storage.mutex); owner = storage.owners.back().lock(); }
    const auto* data = static_cast<const std::uint8_t*>(owner.get());
    for (std::size_t i = 0; i < bytes; ++i) check(data[i] == (123 + i) % 251, "prefetch changed source bytes");
    owner.reset();
    check(work.prepare({item(700, bytes)}, bytes), "next layer refused");
    check(storage.live == bytes, "preparing next layer grew physical retention");
    work.before_compute();
    wait_for([&] { return storage.acquired == 2; });
    work.demand_begin();
    check(storage.peak == bytes, "old and new layer payloads overlapped");
    check(work.stats().layers == 2, "layer count differs");
    work.stop();
    check(storage.live == 0, "request stop leaked storage");
}

void demand_priority_and_cancellation() {
    FakeStorage storage(256);
    tsg::HostMoeLayerPrefetch work(storage.callback(), 2);
    std::mutex gate_mutex;
    std::condition_variable gate;
    bool release = false;
    std::atomic<int> entered{0};
    auto blocked = [&](auto, void* destination, auto bytes) {
        ++entered;
        std::unique_lock<std::mutex> lock(gate_mutex);
        gate.wait(lock, [&] { return release; });
        std::memset(destination, 73, bytes);
    };
    check(work.prepare({item(0, 64, blocked), item(64, 64, blocked), item(128, 64), item(192, 64)}, 256), "pause plan refused");
    work.before_compute();
    wait_for([&] { return entered == 2; });
    auto pause = std::async(std::launch::async, [&] { work.demand_begin(); });
    check(pause.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout,
        "demand started before incomplete fills drained");
    { std::lock_guard<std::mutex> lock(gate_mutex); release = true; }
    gate.notify_all();
    check(pause.wait_for(std::chrono::seconds(5)) == std::future_status::ready, "demand pause deadlocked");
    pause.get();
    check(storage.acquired == 2, "background work raced past demand pause");
    check(work.stats().in_flight == 0, "demand pause left a partial cache entry");
    work.discard_prepared();
    work.before_compute();
    check(work.stats().cancelled_items == 2, "replacement did not cancel unstarted work");
    check(storage.live == 0, "replacement did not release old layer");
}

void refusal_failure_and_validation() {
    FakeStorage storage(32);
    tsg::HostMoeLayerPrefetch work(storage.callback(), 1);
    check(!work.prepare({item(0, 65)}, 64), "over-budget metadata accepted");
    check(!work.prepare({item(UINT64_MAX, 1)}, 64), "overflowing extent accepted");
    check(!work.prepare({item(0, tsg::HostMoeLayerPrefetch::maximum_item_bytes + 1)}, SIZE_MAX), "unbounded fill accepted");
    check(work.prepare({item(0, 32), item(32, 32), item(64, 32)}, 96), "refusal plan failed");
    work.before_compute();
    wait_for([&] { return work.stats().refused != 0; });
    work.demand_begin();
    check(storage.acquired == 1, "refused fill performed source I/O");
    check(work.stats().cancelled_items == 1, "refusal did not cancel remaining scan");
    check(work.prepare({item(0, 32, [](auto, void*, auto) { throw std::runtime_error("source failure"); }), item(32, 32)}, 64),
        "failure plan refused");
    work.before_compute();
    wait_for([&] { return work.stats().failures == 1; });
    work.demand_begin();
    check(storage.live == 0, "failed fill leaked storage");
    check(work.stats().retained_bytes == 0, "failed fill was retained");
}

void cancellation_joins_only_in_flight() {
    FakeStorage storage(128);
    std::promise<void> read_started, allow_read;
    auto permitted = allow_read.get_future().share();
    tsg::HostMoeLayerPrefetch work(storage.callback(), 1);
    check(work.prepare({item(0, 64, [&](auto, void* destination, auto bytes) {
        read_started.set_value(); permitted.wait(); std::memset(destination, 41, bytes);
    }), item(64, 64)}, 128), "cancellation plan refused");
    work.before_compute();
    read_started.get_future().wait();
    auto stopping = std::async(std::launch::async, [&] { work.stop(); });
    const bool joined_pending_read = stopping.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout;
    allow_read.set_value();
    check(stopping.wait_for(std::chrono::seconds(5)) == std::future_status::ready, "request cancellation did not join");
    stopping.get();
    check(joined_pending_read, "request stop abandoned a live source read");
    check(storage.acquired == 1, "cancelled request started further reads");
    check(storage.live == 0, "cancelled in-flight payload was published or leaked");
    check(work.stats().cancelled_items == 1, "cancelled item count differs");
}

void source_and_upload_lifetimes() {
    FakeStorage storage(64);
    std::shared_ptr<const void> queued_upload;
    auto source = std::make_shared<std::vector<std::uint8_t>>(64, 19);
    const std::weak_ptr<std::vector<std::uint8_t>> source_lifetime = source;
    {
        tsg::HostMoeLayerPrefetch work(storage.callback(), 1);
        check(work.prepare({item(0, 64, [source](auto offset, void* target, auto bytes) {
            std::memcpy(target, source->data() + offset, bytes);
        })}, 64), "lifetime plan refused");
        source.reset();
        work.before_compute();
        wait_for([&] { return work.stats().retained_bytes == 64; });
        work.demand_begin();
        check(!source_lifetime.expired(), "source destroyed during live request");
        { std::lock_guard<std::mutex> lock(storage.mutex); queued_upload = storage.owners.back().lock(); }
    }
    check(source_lifetime.expired(), "joined request retained its source");
    check(storage.live == 64, "request teardown freed queued upload payload");
    const auto* data = static_cast<const std::uint8_t*>(queued_upload.get());
    for (int i = 0; i < 64; ++i) check(data[i] == 19, "queued upload saw changed bytes");
    queued_upload.reset();
    check(storage.live == 0, "last queued upload lease leaked");
}

void actual_cache_preserves_demand_and_counts_use() {
    tsg::HostMoePayloadCache cache(192);
    const auto read = item(0, 64).read;
    std::uint8_t demand[64]{};
    cache.read(9, 0, demand, 64, true, read);
    tsg::HostMoeLayerPrefetch work([&](auto source, auto offset, auto bytes, const auto& loader) {
        return cache.acquire_cached(source, offset, bytes, loader);
    }, 2);
    check(work.prepare({item(64, 64), item(128, 64)}, 192), "real cache plan refused");
    work.before_compute();
    wait_for([&] { return work.stats().retained_bytes == 128; });
    work.demand_begin();
    std::uint8_t slice[13]{};
    check(cache.read_range(1, 64, 64, 7, slice, 13, false, read), "prefill did not consume prefetched cache slice");
    for (std::size_t i = 0; i < 13; ++i) check(slice[i] == (71 + i) % 251, "cached source slice differs");
    check(cache.stats().prefetch_hit_bytes == 13, "useful prefetch count differs from actual demand slice");
    check(cache.read(1, 128, demand, 64, true, read), "decode promotion missed prefetched bytes");
    check(work.prepare({item(192, 64), item(256, 64)}, 192), "replacement cache plan refused");
    work.before_compute();
    wait_for([&] { return work.stats().refused == 1; });
    work.demand_begin();
    check(cache.stats().reserved_bytes <= 192, "speculative scan exceeded native cache budget");
    check(cache.read(9, 0, demand, 64, true, read), "prefetch evicted an existing demand entry");
    check(cache.read(1, 128, demand, 64, true, read), "prefetch evicted demand-promoted entry");
    work.stop(); cache.discard_prefetched();
    check(cache.stats().reserved_bytes == 128, "request cleanup retained unused speculative payload");
    check(cache.stats().entries == 2, "request cleanup removed demand data");
}

void demand_failure_unwinds_prefetch_after_upload() {
    tsg::HostMoePayloadCache cache(192);
    const auto read = item(0, 64).read;
    std::uint8_t hot[64]{};
    cache.read(9, 0, hot, 64, true, read);
    auto source = std::make_shared<std::vector<std::uint8_t>>(128, 37);
    const std::weak_ptr<std::vector<std::uint8_t>> source_lifetime = source;
    bool drained = false, caught = false;
    std::uint8_t uploaded[64]{};
    try {
        tsg::HostMoeLayerPrefetch work([&](auto identity, auto offset, auto bytes, const auto& loader) {
            return cache.acquire_cached(identity, offset, bytes, loader);
        }, 1);
        struct Cleanup {
            tsg::HostMoeLayerPrefetch& work;
            tsg::HostMoePayloadCache& cache;
            ~Cleanup() { work.stop(); cache.discard_prefetched(); }
        } cleanup{work, cache};
        check(work.prepare({item(0, 64, [source](auto offset, void* data, auto bytes) {
            std::memcpy(data, source->data() + offset, bytes);
        })}, 192), "failure-unwind prefetch plan refused");
        source.reset();
        work.before_compute();
        wait_for([&] { return work.stats().retained_bytes == 64; });
        work.demand_begin();
        std::uint8_t staging[64]{};
        const auto queued = cache.acquire_range(1, 0, 64, 0, staging, 64, false, read);
        check(queued && queued.cache_hit, "failure-unwind demand missed prefetch");
        struct UploadDrain {
            const tsg::HostMoePayloadCache::Lease& lease;
            std::uint8_t* target;
            bool& drained;
            ~UploadDrain() { std::memcpy(target, lease.data, lease.bytes); drained = true; }
        } drain{queued, uploaded, drained};
        // A later ordinary demand source fails after an upload has retained
        // the first immutable payload. Its request destructor must join,
        // release sources and remove only unleased speculative allocations.
        cache.read(8, 0, staging, 64, false,
            [](auto, void*, auto) { throw std::runtime_error("later demand read failed"); });
    } catch (const std::runtime_error&) { caught = true; }
    check(caught && drained, "demand failure skipped queued-upload drain");
    for (auto value : uploaded) check(value == 37, "exception unwind changed queued payload bytes");
    check(source_lifetime.expired(), "failed request retained prefetched source owner");
    check(cache.stats().reserved_bytes == 64 && cache.stats().resident_bytes == 64,
        "failed request leaked speculative payload or removed demand data");
    check(cache.read(9, 0, hot, 64, false, read), "failure cleanup evicted hot demand entry");
    tsg::HostMoeLayerPrefetch recovered([&](auto identity, auto offset, auto bytes, const auto& loader) {
        return cache.acquire_cached(identity, offset, bytes, loader);
    }, 1);
    check(recovered.prepare({item(128, 64)}, 192), "prefetch failed to recover after demand exception");
    recovered.before_compute();
    wait_for([&] { return recovered.stats().retained_bytes == 64; });
    recovered.demand_begin();
    check(cache.read(1, 128, hot, 64, false, read), "recovered demand missed exact prefetched data");
    for (std::size_t i = 0; i < 64; ++i) check(hot[i] == (128 + i) % 251, "recovered demand bytes differ");
    recovered.stop(); cache.discard_prefetched();
    check(cache.stats().reserved_bytes == 64, "recovered prefetch cleanup leaked allocation");
}
}

int main() {
    try {
        exact_bytes_and_reuse(); demand_priority_and_cancellation();
        refusal_failure_and_validation(); cancellation_joins_only_in_flight(); source_and_upload_lifetimes();
        actual_cache_preserves_demand_and_counts_use();
        demand_failure_unwinds_prefetch_after_upload();
        std::cout << "host_moe_layer_prefetch_test: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "host_moe_layer_prefetch_test: " << error.what() << '\n'; return 1;
    }
}
