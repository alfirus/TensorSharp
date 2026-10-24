// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#define TSG_HOST_MOE_PAYLOAD_CACHE_TEST_HOOKS 1
#include "../ggml_ops_host_moe_victim_policy.h"
#include "../ggml_ops_host_moe_payload_cache.h"
#include <algorithm>
#include <cstdio>
#include <deque>
#include <map>
#include <stdexcept>
#include <thread>

namespace {
int checks = 0;
void require(bool value, const char* text) { ++checks; if (!value) throw std::runtime_error(text); }
tsg::HostMoeVictimKey key(int expert, std::uint64_t version = 1) { return {{version, version + 100, version + 200}, expert}; }
void policy_checks() {
    tsg::HostMoeVictimPolicy<32, 4> policy;
    std::deque<tsg::HostMoeVictimKey> reference;
    std::uint32_t random = 1234567;
    for (int step = 0; step < 20000; ++step) {
        random = random * 1664525u + 1013904223u;
        auto current = key(static_cast<int>(random % 71), 1 + (random >> 8) % 3);
        const bool hit = (random & 7) != 0;
        policy.observe(current, hit);
        reference.push_back(hit ? current : tsg::HostMoeVictimKey{});
        if (reference.size() > 32) reference.pop_front();
        for (int probe = 0; probe < 3; ++probe) {
            const auto selected = probe ? key((step + probe) % 71, 1 + probe) : current;
            const auto expected = std::count(reference.begin(), reference.end(), selected);
            require(policy.count(selected) == expected, "Bounded hit history differs from exact FIFO reference");
            require(policy.admit(selected) == (expected >= 4), "Victim threshold mismatch");
        }
        require(policy.keys() <= 32 && policy.events() <= 32, "Policy metadata exceeded its fixed bound");
    }
    policy.clear();
    for (int i = 0; i < 4; ++i) policy.observe(key(7), true);
    require(policy.admit(key(7)) && !policy.admit(key(7, 2)), "Source registration generations alias");
    for (int i = 0; i < 32; ++i) policy.observe(key(7), false);
    require(!policy.admit(key(7)) && policy.keys() == 0, "Misses did not expire past GPU hits");
    policy.clear(); require(policy.events() == 0 && policy.keys() == 0, "Policy clear retained history");
}
using Cache = tsg::HostMoePayloadCache;
using Ranges = std::array<Cache::Range, 3>;
using Writes = std::array<Cache::ExternalWrite, 3>;
Ranges ranges(std::uint64_t generation = 1) {
    return {{{generation, 17, 10}, {generation + 100, 23, 10}, {generation + 200, 39, 10}}};
}
Writes begin(Cache& cache, const Ranges& source, unsigned seed) {
    Writes result;
    for (int part = 0; part < 3; ++part) {
        result[part] = cache.begin_external(source[part]);
        require(bool(result[part]), "External writer allocation refused unexpectedly");
        for (std::size_t i = 0; i < result[part].bytes; ++i) result[part].data[i] = static_cast<std::uint8_t>(seed + part * 13 + i);
    }
    return result;
}
void lease_checks() {
    Cache cache(60);
    auto source = ranges(); auto writes = begin(cache, source, 5);
    require(!cache.take_triple(source)[0] && cache.stats().resident_bytes == 0,
        "Unfenced external bytes were visible");
    require(cache.publish_external_triple(writes), "Completed external triple was not published");
    writes = {};
    require(cache.stats().source_bytes == 0 && cache.stats().external_bytes == 30,
        "Victim provenance was counted as file source reads");
    auto leases = cache.take_triple(source);
    require(bool(leases[0]) && cache.stats().resident_bytes == 0 && cache.stats().reserved_bytes == 30,
        "Promotion retained duplicate entries or prematurely refunded live leases");
    for (int part = 0; part < 3; ++part) for (int i = 0; i < 10; ++i)
        require(leases[part].data[i] == 5 + part * 13 + i, "Promoted bytes changed");
    cache.clear(); cache.set_budget(0);
    require(cache.stats().reserved_bytes == 30, "Trim refunded a live promotion");
    leases = {}; require(cache.stats().reserved_bytes == 0, "Promotion release retained physical ownership");

    cache.set_budget(60); writes = begin(cache, source, 8);
    std::thread drop([&] { cache.erase_source(source[0].source); }); drop.join();
    require(!cache.publish_external_triple(writes) && cache.stats().resident_bytes == 0,
        "Late completion resurrected a dropped source or partial triple");
    writes = {};
    writes = begin(cache, ranges(2), 71);
    require(cache.publish_external_triple(writes), "New source generation could not publish");
    require(!cache.take_triple(source)[0], "Old identity consumed replacement bytes");
    auto newer = cache.take_triple(ranges(2));
    require(newer[0].data[0] == 71, "Source pointer generation reused stale data");
    newer = {}; writes = {};

    writes = begin(cache, source, 11); cache.set_budget(20);
    require(!cache.publish_external_triple(writes), "Budget-shrunk pending writer repopulated cache");
    writes = {}; require(cache.stats().reserved_bytes == 0, "Retired writer failed to free storage");
    cache.set_budget(60); writes = begin(cache, source, 33);
    Cache::test_fail_metadata_allocation(4); // First projection inserted, second fails.
    require(!cache.publish_external_triple(writes) && cache.stats().resident_bytes == 0,
        "Publication allocation failure retained a partial triple");
    writes = {};
    require(cache.stats().reserved_bytes == 0, "Failed publication leaked physical storage");
    require(!cache.begin_external(source[0], true) && cache.stats().reserved_bytes == 0,
        "Pinned-only admission accepted pageable storage");

    cache.set_budget(30); writes = begin(cache, source, 41);
    require(cache.publish_external_triple(writes), "Initial leased eviction setup failed"); writes = {};
    auto held = cache.test_lease(source[0].source, source[0].offset, source[0].bytes);
    auto* pointer = static_cast<const std::uint8_t*>(nullptr);
    // Ordinary read lease shares the same entry that external admission may evict.
    std::array<std::uint8_t, 10> scratch{};
    auto read = cache.acquire_range(source[0].source, source[0].offset, 10, 0, scratch.data(), 10, false,
        [](auto, auto, auto) { throw std::runtime_error("Unexpected file fallback"); });
    pointer = read.data;
    auto replacement = cache.begin_external(ranges(3)[0]);
    require(replacement && pointer[0] == 41 && cache.stats().reserved_bytes <= 30,
        "External admission overwrote a leased victim or exceeded its byte ceiling");
    held.reset(); read = {}; replacement = {}; cache.clear();
    require(cache.stats().reserved_bytes == 0, "Leased eviction retained ownership");
    cache.set_budget(30); writes = begin(cache, source, 43);
    require(cache.publish_external_triple(writes), "Partial victim fixture did not publish"); writes = {};
    cache.erase_source(source[1].source);
    require(!cache.take_triple(source, true)[0] && cache.stats().resident_bytes == 0
        && cache.stats().reserved_bytes == 0, "File fallback retained partial GPU/RAM duplication");
}
void refusal_checks() {
    unsigned held = 0, physical = 0; bool fail_commit = false;
    Cache cache(30, [&](std::size_t bytes) -> std::shared_ptr<void> {
        ++held; return std::shared_ptr<void>(new int(1), [&](void* p) { delete static_cast<int*>(p); --held; });
    }, [&](const auto&, auto) { return !fail_commit; }, false,
    [&](std::size_t bytes) -> Cache::Storage {
        auto* p = new std::uint8_t[bytes]; ++physical;
        return {std::shared_ptr<void>(p, [&](void* value) { delete[] static_cast<std::uint8_t*>(value); --physical; }), p, bytes, false};
    });
    fail_commit = true;
    require(!cache.begin_external(ranges()[0]) && !held && !physical && !cache.stats().reserved_bytes,
        "External commit refusal retained allocation or credit");
    fail_commit = false;
    auto writes = begin(cache, ranges(), 2);
    require(held == 3 && physical == 3, "External writes lack exact owners");
    require(!cache.begin_external(ranges(2)[0]), "Live unpublished writers exceeded host ceiling");
    writes = {}; require(!held && !physical, "Abandoned writes leaked charge/storage");
    Cache refused(30, [](auto) -> std::shared_ptr<void> { return {}; });
    require(!refused.begin_external(ranges()[0]) && !refused.stats().reserved_bytes,
        "Host reservation refusal allocated an external payload");
}
void recycling_checks() {
    std::size_t physical = 0, charged = 0; unsigned allocations = 0;
    bool allow_reserve = true;
    Cache cache(60, [&](std::size_t bytes) -> std::shared_ptr<void> {
        if (!allow_reserve) return {};
        charged += bytes;
        return std::shared_ptr<void>(new int(1), [&, bytes](void* p) {
            delete static_cast<int*>(p); charged -= bytes;
            require(physical <= charged, "Recycled extent refunded before physical retirement");
        });
    }, {}, false, [&](std::size_t bytes) -> Cache::Storage {
        auto* data = new std::uint8_t[bytes]; physical += bytes; ++allocations;
        return {std::shared_ptr<void>(data, [&, bytes](void* p) {
            delete[] static_cast<std::uint8_t*>(p); physical -= bytes;
        }), data, bytes, true};
    });
    for (int cycle = 0; cycle < 100; ++cycle) {
        auto source = ranges(1000 + cycle * 1000);
        auto writes = begin(cache, source, cycle);
        require(cache.publish_external_triple(writes), "Recycled writer did not publish"); writes = {};
        auto leases = cache.take_triple(source, true);
        require(leases[0].data[0] == static_cast<std::uint8_t>(cycle), "Recycled source generation returned stale bytes");
        require(cache.recycle_external(leases) == 30 && !leases[0], "Completed exclusive leases were not recycled");
        const auto stats = cache.stats();
        require(stats.resident_bytes == 0 && stats.external_free_bytes == 30 && stats.external_free_entries == 3
            && stats.reserved_bytes == 30 && stats.pinned_bytes == 30 && physical == charged && physical == 30,
            "Recycled extents retained readable identity or escaped physical accounting");
        require(!cache.take_triple(source, true)[0], "Free extent remained readable by its old source key");
        allow_reserve = false; // Reuse must retain the original charge, not request another.
    }
    require(allocations == 3 && cache.stats().external_allocations == 3
        && cache.stats().external_pool_reuses == 297 && cache.stats().external_frees == 0,
        "Warm promotion/demotion churned physical buffers");
    auto writes = begin(cache, ranges(), 61);
    require(cache.publish_external_triple(writes), "Copied-lease setup failed"); writes = {};
    auto leases = cache.take_triple(ranges()); auto held = leases[0];
    require(cache.recycle_external(leases) == 20 && leases[0] && !leases[1],
        "Pool reused a source with another outstanding lease");
    cache.set_budget(0); leases = {};
    require(cache.stats().reserved_bytes == 10 && physical == 10 && charged == 10,
        "Pool shrink freed a live copied lease or retained unused extents");
    held = {}; require(!physical && !charged && !cache.stats().reserved_bytes,
        "Final copied promotion retained physical ownership");
    cache.set_budget(60); allow_reserve = true;
    writes = begin(cache, ranges(71), 71);
    require(cache.publish_external_triple(writes), "Pool metadata-failure setup failed"); writes = {};
    leases = cache.take_triple(ranges(71));
    Cache::test_fail_metadata_allocation(6);
    require(cache.recycle_external(leases) == 20, "Optional pool metadata failure did not decline one extent");
    cache.clear(); leases = {};
    require(!physical && !charged && !cache.stats().reserved_bytes && !cache.stats().external_free_bytes,
        "Pool clear retained an allocation/charge after optional metadata failure");
    // Different projection classes may retire unmatched free extents, but may
    // never hide their still-live charge or grow beyond the configured ceiling.
    cache.set_budget(75);
    auto mixed = ranges(91); mixed[0].bytes = 10; mixed[1].bytes = 20; mixed[2].bytes = 30;
    writes = begin(cache, mixed, 91); require(cache.publish_external_triple(writes), "Mixed pool setup failed"); writes = {};
    leases = cache.take_triple(mixed); require(cache.recycle_external(leases) == 60, "Mixed pool did not retain all physical extents");
    mixed = ranges(92); mixed[0].bytes = 15; mixed[1].bytes = 25; mixed[2].bytes = 35;
    writes = begin(cache, mixed, 92);
    require(physical == charged && physical == 75 && cache.stats().reserved_bytes == 75,
        "Mixed-size free pool admission exceeded its exact physical ceiling");
    require(cache.publish_external_triple(writes), "Mixed reused classes did not publish"); writes = {};
    leases = cache.take_triple(mixed); require(cache.recycle_external(leases) == 75, "Mixed promotion failed recycling");
    cache.set_budget(60);
    require(physical <= 60 && physical == charged && physical == cache.stats().external_free_bytes,
        "Pool shrink did not retire unmatched physically owned extents");
    cache.set_budget(0); require(!physical && !charged, "Mixed pool teardown leaked physical credit");
}
}
int main() {
    try { policy_checks(); lease_checks(); refusal_checks(); recycling_checks(); std::printf("PASS: %d selective victim policy/lease checks\n", checks); }
    catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
