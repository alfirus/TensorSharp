// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#define TSG_HOST_MOE_PAYLOAD_CACHE_TEST_HOOKS 1
#include "../ggml_ops_host_moe_payload_cache.h"
#include <array>
#include <chrono>
#include <cstdio>
#include <future>
#include <thread>
#include <vector>

static std::atomic<int> checks{0};
static void require(bool value, const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
static auto loader(unsigned seed, std::atomic<int>* calls = nullptr) {
    return [=](std::uint64_t offset, void* destination, std::size_t count) {
        if (calls) ++*calls;
        auto* bytes = static_cast<std::uint8_t*>(destination);
        for (std::size_t i = 0; i < count; ++i) bytes[i] = static_cast<std::uint8_t>(seed + (offset + i) * 17);
    };
}
static void verify(const std::uint8_t* actual, std::size_t count, unsigned seed, std::uint64_t offset) {
    for (std::size_t i = 0; i < count; ++i)
        require(actual[i] == static_cast<std::uint8_t>(seed + (offset + i) * 17), "cached byte changed");
}
template<class Predicate> static void until(Predicate&& predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("concurrency handshake timed out");
        std::this_thread::yield();
    }
}

struct Charges {
    std::atomic<std::size_t> active{0}, peak{0};
    std::atomic<int> commits{0}, reserves{0}, releases{0};
    struct Owner {
        Charges* parent;
        std::size_t bytes;
        ~Owner() { parent->active -= bytes; ++parent->releases; }
    };
    std::shared_ptr<void> reserve(std::size_t bytes) {
        auto result = std::shared_ptr<Owner>(new Owner{this, bytes});
        const auto now = active.fetch_add(bytes) + bytes;
        auto old = peak.load(); while (old < now && !peak.compare_exchange_weak(old, now)) {}
        ++reserves; return result;
    }
    bool commit(const std::shared_ptr<void>& owner, std::size_t bytes) {
        require(std::static_pointer_cast<Owner>(owner)->bytes == bytes, "commit has wrong physical owner");
        require(active >= bytes, "commit happened without reservation");
        ++commits; return true;
    }
};

static void exact_ranges_and_scan_resistance() {
    Charges charges;
    tsg::HostMoePayloadCache cache(128, [&](auto bytes) { return charges.reserve(bytes); },
        [&](const auto& owner, auto bytes) { return charges.commit(owner, bytes); });
    std::atomic<int> reads{0};
    std::array<std::uint8_t, 64> output{};
    const auto first = tsg::next_host_moe_payload_source_id();
    const auto second = tsg::next_host_moe_payload_source_id();
    require(first != second && first != 0, "source identity reused");
    require(!cache.read_range(first, 128, 64, 9, output.data(), 17, true, loader(3, &reads)), "cold slice was a hit");
    verify(output.data(), 17, 3, 137);
    require(cache.read_range(first, 128, 64, 0, output.data(), 64, false, loader(99, &reads)), "different ring boundary missed canonical extent");
    verify(output.data(), 64, 3, 128);
    require(reads == 1 && cache.stats().source_bytes == 64, "canonical extent loaded twice");
    require(!cache.read(second, 128, output.data(), 64, true, loader(19, &reads)), "new source reused stale bytes");
    verify(output.data(), 64, 19, 128);
    for (int i = 0; i < 100; ++i)
        require(!cache.read(first, 1024 + i * 64, output.data(), 64, false, loader(4, &reads)), "prefill unexpectedly admitted new bytes");
    require(cache.stats().resident_bytes == 128 && charges.active == 128, "prefill scan grew/evicted decode cache");
    // A prefill hit must not make the oldest decode entry newer than `second`.
    require(cache.read(first, 128, output.data(), 64, false, loader(3)), "prefill could not probe decode cache");
    cache.read(first, 256, output.data(), 64, true, loader(5));
    require(cache.read(second, 128, output.data(), 64, false, loader(19)), "prefill aged its entry and evicted newer decode data");
    require(!cache.read(first, 128, output.data(), 64, false, loader(3)), "oldest decode entry was not evicted");
    cache.erase_source(second);
    require(cache.stats().entries == 1 && charges.active == 64, "source teardown retained bytes");
    cache.clear();
    require(cache.stats().reserved_bytes == 0 && charges.active == 0, "clear leaked physical charge");
    require(charges.reserves == charges.releases && charges.peak <= 128, "charge lifecycle or maximum violated");
}

static void duplicate_read_and_failures(bool fail) {
    Charges charges;
    tsg::HostMoePayloadCache cache(64, [&](auto bytes) { return charges.reserve(bytes); });
    std::promise<void> started, release;
    const auto allow = release.get_future().share();
    std::atomic<int> reads{0};
    auto slow = [&](std::uint64_t offset, void* destination, std::size_t count) {
        require(++reads == 1, "duplicate loader was not deduplicated");
        started.set_value(); allow.wait();
        loader(7)(offset, destination, count);
        if (fail) throw std::runtime_error("injected truncated read");
    };
    const auto run = [&] {
        std::array<std::uint8_t, 32> output{};
        bool threw = false;
        try {
            cache.read_range(11, 128, 64, 5, output.data(), output.size(), true, slow);
        } catch (const std::runtime_error&) { threw = true; if (!fail) throw; }
        require(threw == fail, "read failure did not propagate");
        if (!threw) verify(output.data(), output.size(), 7, 133);
    };
    std::vector<std::future<void>> pending;
    pending.push_back(std::async(std::launch::async, run));
    started.get_future().wait();
    for (int i = 0; i < 7; ++i) pending.push_back(std::async(std::launch::async, run));
    try {
        until([&] { return cache.stats().waits == 7; });
        require(cache.stats().reserved_bytes == 64 && charges.active == 64, "pending duplicates over-reserved");
    } catch (...) { release.set_value(); for (auto& future : pending) future.wait(); throw; }
    release.set_value();
    for (auto& future : pending) future.get();
    require(reads == 1, "concurrent duplicate read count changed");
    if (fail) {
        require(cache.stats().entries == 0 && charges.active == 0, "partial source payload published or leaked");
        std::array<std::uint8_t, 64> retry{};
        cache.read(11, 128, retry.data(), retry.size(), true, loader(9));
        verify(retry.data(), retry.size(), 9, 128);
    } else require(cache.stats().hits == 7, "waiting readers were not cache hits");
    cache.clear(); require(charges.active == 0, "duplicate test leaked host ownership");
}

static void clear_and_shrink_during_read(int mode) {
    Charges charges;
    tsg::HostMoePayloadCache cache(128, [&](auto bytes) { return charges.reserve(bytes); });
    std::promise<void> started, release;
    auto allow = release.get_future().share();
    auto pending = std::async(std::launch::async, [&] {
        std::array<std::uint8_t, 64> output{};
        cache.read(1, 0, output.data(), 64, true, [&](auto offset, void* destination, auto count) {
            started.set_value(); allow.wait(); loader(5)(offset, destination, count);
        });
        verify(output.data(), output.size(), 5, 0);
    });
    started.get_future().wait();
    if (mode == 0) cache.clear();
    else if (mode == 1) cache.set_budget(32);
    else cache.erase_source(1);
    require(cache.stats().reserved_bytes == 64 && charges.active == 64, "retirement refunded pending physical memory");
    require(cache.stats().resident_bytes == 0, "pending read became resident prematurely");
    std::array<std::uint8_t, 64> other{};
    if (mode == 1) {
        cache.read(2, 0, other.data(), 32, true, loader(13));
        require(charges.active == 64, "shrink allowed overlapping replacement over capacity");
    }
    release.set_value(); pending.get();
    require(cache.stats().reserved_bytes == 0 && cache.stats().entries == 0 && charges.active == 0,
        "retired in-flight read repopulated the cache");
}

static void optional_admission_failures() {
    std::array<std::uint8_t, 32> output{};
    for (int mode = 0; mode < 5; ++mode) {
        Charges charges;
        tsg::HostMoePayloadCache cache(64,
            [&](std::size_t bytes) -> std::shared_ptr<void> {
                if (mode == 0) return {};
                if (mode == 2) throw std::bad_alloc();
                return charges.reserve(bytes);
            }, [&](const auto& owner, auto bytes) {
                charges.commit(owner, bytes);
                if (mode == 3) throw std::runtime_error("injected commit error");
                if (mode == 4) throw std::bad_alloc();
                return false;
            });
        bool threw = false;
        try { cache.read_range(1, 16, 64, 9, output.data(), 32, true, loader(8)); }
        catch (const std::runtime_error&) { threw = true; }
        require(threw == (mode == 3), "wrong optional admission failure behavior");
        if (!threw) {
            verify(output.data(), output.size(), 8, 25);
            require(cache.stats().source_bytes == 32 && cache.stats().bypass_bytes == 32,
                "declined cache fetched speculative bytes");
        }
        require(cache.stats().reserved_bytes == 0 && cache.stats().entries == 0 && charges.active == 0,
            "failed admission leaked reservation or published bytes");
    }
    tsg::HostMoePayloadCache cache(31);
    cache.read(1, 0, output.data(), output.size(), true, loader(4));
    require(cache.stats().reserved_bytes == 0, "oversized payload exceeded budget");
    const auto rejects = [&](auto source, auto base, auto extent, auto relative, auto count) {
        bool failed = false;
        try { cache.read_range(source, base, extent, relative, output.data(), count, true, loader(4)); }
        catch (const std::invalid_argument&) { failed = true; }
        require(failed, "invalid range was accepted");
    };
    rejects(0, 0, 32, 0, 32); rejects(1, 0, 0, 0, 0);
    rejects(1, 0, 32, 33, 0); rejects(1, 0, 32, 16, 17);
    rejects(1, UINT64_MAX - 15, 32, 0, 1);
}

static void parallel_budget_stress() {
    Charges charges;
    tsg::HostMoePayloadCache cache(512, [&](auto bytes) { return charges.reserve(bytes); },
        [&](const auto& owner, auto bytes) { return charges.commit(owner, bytes); });
    std::vector<std::future<void>> pending;
    for (unsigned thread = 0; thread < 8; ++thread)
        pending.push_back(std::async(std::launch::async, [&, thread] {
            std::array<std::uint8_t, 31> output{};
            for (unsigned i = 0; i < 400; ++i) {
                const auto source = 1 + (i + thread * 3) % 5;
                const auto base = ((i * 7 + thread) % 13) * 64;
                cache.read_range(source, base, 64, i % 32, output.data(), output.size(), i % 7 != 0, loader(source));
                verify(output.data(), output.size(), source, base + i % 32);
                if (i % 47 == 0) cache.erase_source(source);
            }
        }));
    for (auto& future : pending) future.get();
    require(charges.peak <= 512 && cache.stats().reserved_bytes <= 512, "parallel cache exceeded physical ceiling");
    cache.set_budget(64);
    require(charges.active <= 64 && cache.stats().resident_bytes <= 64, "quiescent shrink exceeded budget");
    cache.clear();
    require(charges.active == 0 && charges.reserves == charges.releases, "parallel cache leaked charge");
}

static void reuse_without_refunding_or_overwriting_leases() {
    Charges charges;
    tsg::HostMoePayloadCache cache(64, [&](auto bytes) { return charges.reserve(bytes); },
        [&](const auto& owner, auto bytes) { return charges.commit(owner, bytes); });
    std::array<std::uint8_t, 64> output{};
    for (unsigned source = 1; source <= 100; ++source) {
        require(!cache.read(source, 64, output.data(), output.size(), true, loader(source)), "replacement was stale cache hit");
        verify(output.data(), output.size(), source, 64);
        require(charges.reserves == 1 && charges.commits == 1 && charges.releases == 0,
            "same-size replacement reallocated or refunded live charge");
        require(cache.stats().reserved_bytes == 64 && charges.active == 64, "reuse lost its physical reservation");
    }
    require(cache.stats().reuses == 99, "same-size victim was not reused");
    bool failed = false;
    try {
        cache.read(101, 64, output.data(), output.size(), true, [](auto, void* destination, auto) {
            static_cast<std::uint8_t*>(destination)[0] = 42;
            throw std::runtime_error("partial replacement failure");
        });
    } catch (const std::runtime_error&) { failed = true; }
    require(failed && charges.active == 0 && cache.stats().entries == 0, "failed replacement retained partial payload");
    require(!cache.read(100, 64, output.data(), output.size(), false, loader(100)), "old key survived replacement failure");
    verify(output.data(), output.size(), 100, 64);
    require(!cache.read(101, 64, output.data(), output.size(), false, loader(101)), "new key survived replacement failure");
    verify(output.data(), output.size(), 101, 64);

    cache.set_budget(128);
    cache.read(1, 0, output.data(), output.size(), true, loader(1));
    cache.read(2, 0, output.data(), output.size(), true, loader(2));
    const auto reserved_before = charges.reserves.load();
    auto lease = cache.test_lease(1, 0, 64);
    require(lease != nullptr, "failed to hold reader lease");
    cache.read(3, 0, output.data(), output.size(), true, loader(3));
    require(charges.reserves == reserved_before, "unleased equal-sized victim did not reuse allocation");
    require(cache.read(1, 0, output.data(), output.size(), false, loader(99)), "leased victim was overwritten");
    verify(output.data(), output.size(), 1, 0);
    require(!cache.read(2, 0, output.data(), output.size(), false, loader(2)), "newer unleased victim was not reused");
    cache.clear();
    require(charges.active == 64 && cache.stats().reserved_bytes == 64, "clear refunded an active immutable lease");
    lease.reset();
    require(charges.active == 0 && cache.stats().reserved_bytes == 0 && charges.reserves == charges.releases,
        "reader lease did not release final physical ownership");
}

static void optional_metadata_failures() {
    for (bool reuse : {false, true}) for (int stage : {1, 2, 3}) {
        if (reuse && stage == 1) continue; // A reused Entry needs no new shared owner.
        Charges charges;
        tsg::HostMoePayloadCache cache(64, [&](auto bytes) { return charges.reserve(bytes); },
            [&](const auto& owner, auto bytes) { return charges.commit(owner, bytes); });
        std::array<std::uint8_t, 32> output{};
        if (reuse) cache.read_range(1, 64, 64, 0, output.data(), output.size(), true, loader(3));
        const auto before = cache.stats();
        cache.test_fail_metadata_allocation(stage);
        require(!cache.read_range(2, 64, 64, 5, output.data(), output.size(), true, loader(9)),
            "metadata failure returned a cache hit");
        verify(output.data(), output.size(), 9, 69);
        const auto after = cache.stats();
        require(after.entries == 0 && after.reserved_bytes == 0 && charges.active == 0,
            "metadata failure left partial publication or physical charge");
        require(after.source_bytes - before.source_bytes == output.size()
            && after.bypass_bytes - before.bypass_bytes == output.size(),
            "metadata failure read beyond the requested slice");
        require(charges.reserves == (reuse ? 1 : 0), "metadata failure reserved new payload storage");
        if (reuse) require(!cache.read_range(1, 64, 64, 0, output.data(), output.size(), false, loader(3)),
            "reused victim kept stale identity after metadata failure");
        require(!cache.read_range(2, 64, 64, 5, output.data(), output.size(), true, loader(9)),
            "metadata retry found unpublished data");
        require(cache.read_range(2, 64, 64, 0, output.data(), output.size(), false, loader(99)),
            "metadata retry failed to publish original bytes");
        verify(output.data(), output.size(), 9, 64);
        cache.clear();
        require(charges.active == 0 && charges.reserves == charges.releases, "metadata retry leaked ownership");
    }
}

struct StorageTracker {
    std::atomic<std::size_t> live{0};
    std::atomic<int> allocations{0}, frees{0};
    tsg::HostMoePayloadCache::Storage allocate(std::size_t bytes, bool pinned) {
        auto* data = new std::uint8_t[bytes];
        live += bytes; ++allocations;
        std::shared_ptr<void> owner(data, [this, bytes](void* pointer) {
            delete[] static_cast<std::uint8_t*>(pointer);
            live -= bytes; ++frees;
        });
        return {std::move(owner), data, bytes, pinned};
    }
};

static void queued_dma_lease_lifetime(bool pinned) {
    StorageTracker storage;
    std::atomic<std::size_t> charged{0};
    std::atomic<bool> early_refund{false};
    tsg::HostMoePayloadCache cache(64, [&](std::size_t bytes) {
        charged += bytes;
        return std::shared_ptr<void>(new int(0), [&, bytes](void* pointer) {
            if (storage.live != 0) early_refund = true;
            delete static_cast<int*>(pointer); charged -= bytes;
        });
    }, [](const auto&, auto) { return true; }, false,
        [&](std::size_t bytes) { return storage.allocate(bytes, pinned); });
    std::array<std::uint8_t, 32> staging;
    staging.fill(0xcc);
    auto value = cache.acquire_range(1, 64, 64, 5, staging.data(), staging.size(), true, loader(7));
    require(value && !value.cache_hit && value.bytes == staging.size() && value.pinned == pinned,
        "cold acquire returned wrong slice lease");
    verify(value.data, value.bytes, 7, 69);
    require(std::all_of(staging.begin(), staging.end(), [](auto byte) { return byte == 0xcc; }),
        "acquire copied cached bytes to staging");
    auto hit = cache.acquire_range(1, 64, 64, 11, staging.data(), 17, false, loader(99));
    require(hit && hit.cache_hit && hit.owner == value.owner && hit.data == value.data + 6,
        "range lease did not reuse canonical storage");
    hit = {};
    require(cache.stats().pinned_bytes == (pinned ? 64u : 0u), "actual storage pinning not reported");
    std::promise<void> started, release;
    auto allow = release.get_future().share();
    auto transfer = std::async(std::launch::async, [value = std::move(value), &started, allow] {
        started.set_value(); allow.wait();
        // The queued transfer reads the original pointer only after eviction.
        verify(value.data, value.bytes, 7, 69);
    });
    started.get_future().wait();
    try {
        cache.clear();
        require(cache.stats().entries == 0 && cache.stats().reserved_bytes == 64
            && charged == 64 && storage.live == 64, "clear freed a queued transfer source");
        const auto declined = cache.acquire_range(2, 128, 64, 9, staging.data(), staging.size(), true, loader(5));
        require(!declined && storage.allocations == 1, "active transfer storage was reused or over-reserved");
        verify(staging.data(), staging.size(), 5, 137);
        require(cache.stats().pinned_bytes == (pinned ? 64u : 0u), "retired transfer lost pinned accounting");
    } catch (...) { release.set_value(); transfer.wait(); throw; }
    release.set_value(); transfer.get();
    require(charged == 0 && storage.live == 0 && storage.allocations == storage.frees && !early_refund,
        "transfer release order or storage lifetime violated");
    require(cache.stats().reserved_bytes == 0 && cache.stats().pinned_bytes == 0,
        "finished transfer retained physical accounting");
}

static void storage_lease_reuse_and_cache_lifetime() {
    for (bool frequency : {false, true}) {
        Charges charges;
        StorageTracker storage;
        tsg::HostMoePayloadCache::Lease held;
        std::array<std::uint8_t, 32> staging{};
        {
            tsg::HostMoePayloadCache cache(128, [&](auto bytes) { return charges.reserve(bytes); },
                [&](const auto& owner, auto bytes) { return charges.commit(owner, bytes); }, frequency,
                [&](auto bytes) { return storage.allocate(bytes, true); });
            held = cache.acquire_range(1, 0, 64, 0, staging.data(), staging.size(), true, loader(3));
            cache.read_range(2, 0, 64, 0, staging.data(), staging.size(), true, loader(5));
            require(storage.allocations == 2, "wrong initial custom allocation count");
            auto replacement = cache.acquire_range(3, 0, 64, 0, staging.data(), staging.size(), true, loader(7));
            require(replacement && storage.allocations == 2 && storage.frees == 0,
                "custom storage did not reuse unleased same-size victim");
            verify(held.data, held.bytes, 3, 0);
            verify(replacement.data, replacement.bytes, 7, 0);
            require(cache.stats().pinned_bytes == 128, "reuse changed pinned physical ownership");
            replacement = {};
            cache.set_budget(0);
            require(storage.live == 64 && charges.active == 64 && cache.stats().pinned_bytes == 64,
                "shrink freed outstanding lease or retained unused allocation");
        }
        verify(held.data, held.bytes, 3, 0);
        require(storage.live == 64 && charges.active == 64, "cache destruction invalidated external lease");
        held = {};
        require(storage.live == 0 && charges.active == 0 && storage.allocations == storage.frees,
            "external lease did not complete final storage release");
    }
}

static void optional_storage_failures() {
    for (int mode = 0; mode < 8; ++mode) {
        Charges charges;
        StorageTracker storage;
        tsg::HostMoePayloadCache cache(64, [&](auto bytes) { return charges.reserve(bytes); },
            [&](const auto& owner, auto bytes) {
                charges.commit(owner, bytes);
                if (mode == 6) return false;
                if (mode == 7) throw std::bad_alloc();
                return true;
            }, false, [&](std::size_t bytes) {
                if (mode == 4) throw std::bad_alloc();
                auto result = storage.allocate(bytes, true);
                if (mode == 0) result.owner.reset();
                if (mode == 1) result.data = nullptr;
                if (mode == 2) --result.bytes;
                if (mode == 3) ++result.bytes;
                return result;
            });
        std::array<std::uint8_t, 32> staging{};
        tsg::HostMoePayloadCache::Lease value;
        bool failed = false;
        try {
            value = cache.acquire_range(1, 64, 64, 7, staging.data(), staging.size(), true,
                [&](auto offset, void* destination, auto bytes) {
                    if (mode == 5) throw std::runtime_error("injected full source read failure");
                    loader(3)(offset, destination, bytes);
                });
        } catch (const std::runtime_error&) { failed = true; }
        require(failed == (mode == 5), "storage path failed to propagate source read failure");
        require(!value, "invalid/declined storage produced a payload lease");
        if (!failed) {
            verify(staging.data(), staging.size(), 3, 71);
            require(cache.stats().source_bytes == staging.size() && cache.stats().bypass_bytes == staging.size(),
                "invalid/declined storage read unrequested source bytes");
        }
        require(cache.stats().reserved_bytes == 0 && cache.stats().pinned_bytes == 0
            && charges.active == 0 && storage.live == 0, "invalid/declined storage leaked physical ownership");
    }
}

static void frequency_admission_resists_one_offs() {
    Charges charges;
    tsg::HostMoePayloadCache cache(128, [&](auto bytes) { return charges.reserve(bytes); },
        [&](const auto& owner, auto bytes) { return charges.commit(owner, bytes); }, true);
    std::array<std::uint8_t, 64> output{};
    for (int i = 0; i < 20; ++i) cache.read(1, 0, output.data(), 64, true, loader(1));
    for (int i = 0; i < 10; ++i) cache.read(2, 0, output.data(), 64, true, loader(2));
    const auto before = cache.stats();
    require(before.frequency_updates == 30 && before.frequency_counter_bytes == 65536,
        "frequency sketch is not fixed-size or misses warm read updates");
    for (unsigned source = 3; source < 163; ++source) {
        require(!cache.read_range(source, 0, 64, 9, output.data(), 7, true, loader(source)),
            "rejected one-off returned a cache hit");
        verify(output.data(), 7, source, 9);
    }
    const auto after = cache.stats();
    require(after.frequency_updates == before.frequency_updates + 160
        && after.frequency_rejections == before.frequency_rejections + 160,
        "one-offs were not counted/rejected by frequency admission");
    require(after.source_bytes - before.source_bytes == 160 * 7 && after.resident_bytes == 128
        && charges.reserves == 2 && charges.releases == 0,
        "rejected reads amplified source work, allocated, or displaced hot payloads");
    require(cache.read(1, 0, output.data(), 64, false, loader(99)), "stream displaced hottest source");
    verify(output.data(), 64, 1, 0);
    require(cache.read(2, 0, output.data(), 64, false, loader(99)), "stream displaced second hot source");
    verify(output.data(), 64, 2, 0);
    const auto updates = cache.stats().frequency_updates;
    for (unsigned source = 3; source < 163; ++source)
        cache.read_range(source, 0, 64, 9, output.data(), 7, false, loader(source));
    require(cache.stats().frequency_updates == updates,
        "prefill scan changed frequency admission history");
    // Earlier rejected accesses are evidence too. Repeated demand can enter
    // without a GPU victim download or a fresh host payload allocation.
    for (int i = 0; i < 9; ++i) cache.read(3, 0, output.data(), 64, true, loader(3));
    require(cache.read(3, 0, output.data(), 64, false, loader(99)),
        "repeated rejected demand never became eligible for admission");
    verify(output.data(), 64, 3, 0);
    require(cache.read(1, 0, output.data(), 64, false, loader(99)) && charges.reserves == 2,
        "frequency admission selected the hotter source or allocated a replacement");
    cache.clear(); require(charges.active == 0, "frequency admission leaked payload ownership");
}

static void frequency_aging_and_source_identity() {
    tsg::HostMoePayloadCache cache(64, {}, {}, true);
    std::array<std::uint8_t, 64> output{};
    for (int i = 0; i < 8; ++i) cache.read(1, 0, output.data(), 64, true, loader(1));
    require(cache.test_frequency(1, 0, 64) == 8, "frequency updates did not include host hits");
    // Oversized demand updates history without replacing the old resident.
    // This isolates real aging from a challenger simply accumulating its own
    // frequency until it beats an unchanging old count.
    while (cache.stats().frequency_updates < 49152)
        cache.read_range(999, 0, 65, 0, output.data(), 1, true, loader(9));
    require(cache.stats().frequency_ages == 1 && cache.test_frequency(1, 0, 64) == 4,
        "frequency epoch did not halve inactive history");
    require(cache.test_frequency(999, 0, 65) == 128,
        "frequency counter did not saturate at 255 before the aging boundary");
    for (int i = 0; i < 4; ++i) cache.read(2, 0, output.data(), 64, true, loader(2));
    require(cache.read(2, 0, output.data(), 64, false, loader(99)),
        "aged history prevented the changed workload from becoming resident");
    verify(output.data(), 64, 2, 0);
    require(!cache.read(1, 0, output.data(), 64, false, loader(1)),
        "replaced source identity survived in the payload map");
    verify(output.data(), 64, 1, 0);
    cache.erase_source(2);
    const auto replacement = tsg::next_host_moe_payload_source_id() + 1000;
    require(!cache.read(replacement, 0, output.data(), 64, true, loader(19)),
        "frequency metadata aliased a new source to old payload bytes");
    verify(output.data(), 64, 19, 0);
    require(cache.stats().frequency_counter_bytes == 65536 && cache.stats().resident_bytes == 64,
        "source changes grew frequency metadata or payload bytes");
}

static void frequency_sampling_respects_leases_and_sizes() {
    Charges charges;
    tsg::HostMoePayloadCache cache(128, [&](auto bytes) { return charges.reserve(bytes); },
        [&](const auto& owner, auto bytes) { return charges.commit(owner, bytes); }, true);
    std::array<std::uint8_t, 64> output{};
    cache.read(1, 0, output.data(), 64, true, loader(1));
    cache.read(2, 0, output.data(), 64, true, loader(2));
    auto first_lease = cache.test_lease(1, 0, 64), second_lease = cache.test_lease(2, 0, 64);
    const auto before = cache.stats();
    cache.read_range(3, 0, 64, 5, output.data(), 11, true, loader(3));
    verify(output.data(), 11, 3, 5);
    require(cache.stats().entries == 2 && charges.reserves == 2 && charges.active == 128,
        "frequency fallback trimmed or overwrote leased storage");
    first_lease.reset(); second_lease.reset();
    cache.read(4, 0, output.data(), 32, true, loader(4));
    verify(output.data(), 32, 4, 0);
    require(cache.stats().entries == 2 && charges.reserves == 2
        && cache.stats().frequency_rejections == before.frequency_rejections + 2,
        "non-matching payload size bypassed frequency policy through LRU trim");
    cache.clear(); require(charges.active == 0, "frequency lease/size fallback leaked ownership");
}

static void cache_only_exact_and_demand_protection(bool frequency) {
    Charges charges;
    tsg::HostMoePayloadCache cache(128, [&](auto bytes) { return charges.reserve(bytes); },
        [&](const auto& owner, auto bytes) { return charges.commit(owner, bytes); }, frequency);
    std::array<std::uint8_t, 64> output{};
    std::atomic<int> reads{0};
    for (int i = 0; i < 10; ++i) cache.read(1, 0, output.data(), 64, true, loader(1));
    const auto history = cache.stats().frequency_updates;
    auto next = cache.acquire_cached(2, 128, 64, loader(2, &reads));
    require(next && !next.cache_hit && reads == 1, "cache-only cold read did not publish a lease");
    verify(next.data, next.bytes, 2, 128);
    require(cache.stats().frequency_updates == history, "speculation changed demand frequency");
    auto refused = cache.acquire_cached(3, 128, 64, loader(3, &reads));
    require(!refused && reads == 1 && charges.reserves == 2 && cache.stats().entries == 2,
        "cache-only refusal read data, allocated, or unlinked leased/demand data");
    require(cache.read_range(2, 128, 64, 7, output.data(), 11, false, loader(99)),
        "wide demand did not reuse the speculative payload");
    verify(output.data(), 11, 2, 135);
    require(cache.stats().prefetch_hit_bytes == 11, "useful prefetch bytes did not count the exact consumed slice");
    next = {};
    auto replacement = cache.acquire_cached(3, 128, 64, loader(3, &reads));
    require(replacement && reads == 2 && charges.reserves == 2,
        "next layer did not reuse unleased speculative storage");
    verify(replacement.data, replacement.bytes, 3, 128);
    require(cache.read(1, 0, output.data(), 64, false, loader(99)), "prefetch displaced demand data");
    verify(output.data(), 64, 1, 0);
    replacement = {};
    require(cache.read(3, 128, output.data(), 64, true, loader(99)), "decode did not promote speculative data");
    require(cache.stats().prefetch_hit_bytes == 75, "promotion lost useful prefetch traffic");
    require(!cache.acquire_cached(4, 128, 64, loader(4, &reads)) && reads == 2,
        "prefetch evicted demand-promoted payloads");
    const auto useful = cache.stats().prefetch_hit_bytes;
    auto resident = cache.acquire_cached(3, 128, 64, loader(99, &reads));
    require(resident && resident.cache_hit && cache.stats().prefetch_hit_bytes == useful,
        "speculative probe was counted as useful demand traffic");
    resident = {};
    cache.set_budget(192);
    auto transient = cache.acquire_cached(5, 128, 64, loader(5, &reads));
    cache.discard_prefetched();
    require(cache.stats().resident_bytes == 192, "prefetch teardown unlinked a live lease");
    transient = {}; cache.discard_prefetched();
    require(cache.stats().resident_bytes == 128 && charges.active == 128,
        "prefetch teardown retained speculative storage or discarded demand-promoted data");
    cache.clear(); require(charges.active == 0, "cache-only path leaked storage charges");
}

static void cache_only_refusal_has_no_fallback_read() {
    for (int mode = 0; mode < 9; ++mode) {
        Charges charges;
        std::atomic<int> reads{0}, allocations{0};
        tsg::HostMoePayloadCache cache(mode == 0 ? 0 : mode == 1 ? 32 : 64,
            [&](auto bytes) -> std::shared_ptr<void> {
                return mode == 2 ? std::shared_ptr<void>{} : charges.reserve(bytes);
            }, [&](const auto& owner, auto bytes) {
                charges.commit(owner, bytes);
                if (mode == 4) return false;
                if (mode == 5) throw std::bad_alloc();
                return true;
            }, false, [&](std::size_t bytes) {
                ++allocations;
                if (mode == 3) throw std::bad_alloc();
                auto owner = std::shared_ptr<std::uint8_t[]>(new std::uint8_t[bytes]);
                return tsg::HostMoePayloadCache::Storage{owner, owner.get(), mode == 6 ? bytes + 1 : bytes, false};
            });
        if (mode == 7 || mode == 8) cache.test_fail_metadata_allocation(mode - 5);
        require(!cache.acquire_cached(1, 128, 64, loader(9, &reads)),
            "cache-only optional admission failure returned storage");
        const auto stats = cache.stats();
        require(reads == 0 && stats.source_bytes == 0 && stats.bypass_bytes == 0,
            "declined prefetch performed discarded file reads");
        require(stats.reserved_bytes == 0 && stats.entries == 0 && charges.active == 0,
            "declined prefetch leaked metadata, allocation or reservation");
        if (mode <= 2 || mode >= 7) require(allocations == 0, "pre-allocation refusal allocated payload storage");
    }
}

static void cache_only_mixed_sizes_and_source_identity() {
    tsg::HostMoePayloadCache cache(128, {}, {}, true);
    std::array<std::uint8_t, 64> output{};
    cache.read(1, 0, output.data(), 64, true, loader(1));
    auto old = cache.acquire_cached(2, 0, 64, loader(2));
    require(!cache.acquire_cached(3, 0, 32, loader(3)), "mixed-size prefetch unlinked a live payload lease");
    old = {};
    auto first = cache.acquire_cached(3, 0, 32, loader(3));
    auto second = cache.acquire_cached(4, 0, 32, loader(4));
    require(first && second && cache.stats().reserved_bytes == 128,
        "mixed-size prefetch failed to reclaim only old speculative storage");
    cache.erase_source(3);
    verify(first.data, first.bytes, 3, 0);
    const auto new_source = tsg::next_host_moe_payload_source_id() + 1000;
    require(!cache.acquire_cached(new_source, 0, 32, loader(8)), "retired source lease was refunded before release");
    first = {};
    auto replacement = cache.acquire_cached(new_source, 0, 32, loader(8));
    require(replacement && !replacement.cache_hit, "new source identity reused a retired payload");
    verify(replacement.data, replacement.bytes, 8, 0);
    require(cache.read(1, 0, output.data(), 64, false, loader(99)), "mixed-size prefetch trimmed demand data");
    verify(output.data(), 64, 1, 0);
}

static void cache_only_concurrent_demand_and_retirement() {
    for (bool retire : {false, true}) {
        tsg::HostMoePayloadCache cache(64);
        std::promise<void> started, release;
        auto allow = release.get_future().share();
        std::atomic<int> reads{0};
        auto prefetch = std::async(std::launch::async, [&] {
            return cache.acquire_cached(1, 0, 64, [&](auto offset, void* destination, auto count) {
                ++reads; started.set_value(); allow.wait(); loader(5)(offset, destination, count);
            });
        });
        started.get_future().wait();
        auto demand = std::async(std::launch::async, [&] {
            std::array<std::uint8_t, 19> output{};
            require(cache.read_range(1, 0, 64, 7, output.data(), output.size(), true, loader(99, &reads)),
                "demand did not deduplicate against a pending prefetch");
            verify(output.data(), output.size(), 5, 7);
        });
        try {
            until([&] { return cache.stats().waits == 1; });
            if (retire) { cache.erase_source(1); cache.set_budget(0); }
        } catch (...) { release.set_value(); throw; }
        release.set_value();
        auto lease = prefetch.get(); demand.get();
        verify(lease.data, lease.bytes, 5, 0);
        require(reads == 1 && cache.stats().prefetch_hit_bytes == 19, "prefetch and demand duplicated source traffic");
        require(cache.stats().reserved_bytes == 64, "in-flight prefetch retirement refunded leased bytes");
        lease = {};
        cache.discard_prefetched();
        require(cache.stats().resident_bytes == (retire ? 0 : 64),
            "retired prefetch was resurrected or demand promotion was discarded");
    }
}

int main() {
    try {
        exact_ranges_and_scan_resistance();
        duplicate_read_and_failures(false); duplicate_read_and_failures(true);
        for (int mode = 0; mode < 3; ++mode) clear_and_shrink_during_read(mode);
        optional_admission_failures(); parallel_budget_stress(); reuse_without_refunding_or_overwriting_leases();
        optional_metadata_failures();
        queued_dma_lease_lifetime(false); queued_dma_lease_lifetime(true);
        storage_lease_reuse_and_cache_lifetime(); optional_storage_failures();
        frequency_admission_resists_one_offs(); frequency_aging_and_source_identity();
        frequency_sampling_respects_leases_and_sizes();
        cache_only_exact_and_demand_protection(false); cache_only_exact_and_demand_protection(true);
        cache_only_refusal_has_no_fallback_read(); cache_only_mixed_sizes_and_source_identity();
        cache_only_concurrent_demand_and_retirement();
        std::printf("PASS host expert payload cache: %d checks; original bytes, canonical slices, scan resistance, concurrent dedup, failure/trim ownership and bounded parallel admission; CPU only\n", checks.load());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL host expert payload cache: %s\n", error.what()); return 1;
    }
}
