// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#define TSG_HOST_MOE_IMMUTABLE_CACHE_TEST_HOOKS 1
#include "../ggml_ops_host_moe_immutable_cache.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <future>
#include <thread>
#include <type_traits>
#include <vector>

using Cache = tsg::HostMoeImmutableCache;
static std::atomic<int> checks{0};
static void require(bool value, const char* message) {
    ++checks; if (!value) throw std::runtime_error(message);
}
static_assert(std::is_const<typename std::remove_pointer<decltype(Cache::Storage::data)>::type>::value,
    "Immutable factory data must be const");
static_assert(std::is_const<typename std::remove_pointer<decltype(Cache::Lease::data)>::type>::value,
    "Immutable lease data must be const");
template<class Predicate> static void until(Predicate predicate) {
    const auto end = std::chrono::steady_clock::now()+std::chrono::seconds(5);
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= end) throw std::runtime_error("Immutable concurrency handshake timed out");
        std::this_thread::yield();
    }
}
struct Ledger {
    std::atomic<std::size_t> active{0}, peak{0};
    std::atomic<int> reserves{0}, commits{0}, releases{0}, physical{0}, factories{0};
    std::atomic<bool> valid{true};
    bool strict_final_order = true; // Serial cases prove order; stress avoids cross-owner snapshot races.
    std::function<void()> on_release;
    struct Charge {
        Ledger* ledger; std::size_t bytes;
        ~Charge() {
            const auto before = ledger->active.fetch_sub(bytes);
            if (before < bytes || (ledger->strict_final_order && before == bytes && ledger->physical.load() != 0)) ledger->valid = false;
            ++ledger->releases;
            if (ledger->on_release) ledger->on_release();
        }
    };
    std::shared_ptr<void> reserve(std::size_t bytes) {
        auto owner = std::shared_ptr<Charge>(new Charge{this, bytes});
        const auto now = active.fetch_add(bytes)+bytes;
        auto old = peak.load(); while (old < now && !peak.compare_exchange_weak(old, now)) {}
        ++reserves; return owner;
    }
    bool commit(const std::shared_ptr<void>& owner, std::size_t bytes) {
        require(std::static_pointer_cast<Charge>(owner)->bytes == bytes && active >= bytes && physical > 0,
            "Immutable commit did not follow charged physical creation");
        ++commits; return true;
    }
    void empty() const {
        require(valid && active == 0 && physical == 0 && reserves == releases,
            "Immutable storage/ticket final-free order or balance failed");
    }
};
static std::uint8_t expected(std::uint64_t source, std::uint64_t offset) {
    return static_cast<std::uint8_t>(source*23 + offset*17);
}
struct BytesOwner {
    std::shared_ptr<void> lifetime; // Destroyed after the physical byte vector.
    std::vector<std::uint8_t> bytes;
    Ledger* ledger;
    BytesOwner(const Cache::Range& range, std::shared_ptr<void> token, Ledger& l)
        : lifetime(std::move(token)), bytes(range.bytes), ledger(&l) {
        for (std::size_t i=0; i<bytes.size(); ++i) bytes[i] = expected(range.source, range.offset+i);
        ++ledger->physical;
    }
    ~BytesOwner() { --ledger->physical; }
};
static Cache::Storage storage(Ledger& ledger, const Cache::Range& range, const std::shared_ptr<void>& token) {
    ++ledger.factories;
    auto owner = std::make_shared<BytesOwner>(range, token, ledger);
    return {owner, owner->bytes.data(), owner->bytes.size()};
}
static Cache::Factory factory(Ledger& ledger) {
    return [&](const auto& range, const auto& token) { return storage(ledger, range, token); };
}
static void verify(const Cache::Lease& lease, std::uint64_t source, std::uint64_t offset, std::size_t count) {
    require(bool(lease) && lease.bytes == count, "Immutable slice lease is empty/wrong size");
    for (std::size_t i=0; i<count; ++i) require(lease.data[i] == expected(source,offset+i), "Immutable source/slice bytes changed");
}
static constexpr std::size_t extent = 64, charge = extent+Cache::entry_metadata_bytes+4096;

static void exact_physical_charge_and_scan() {
    Ledger ledger;
    Cache cache(2*charge, [&](auto n) { return ledger.reserve(n); },
        [&](const auto& owner, auto n) { return ledger.commit(owner,n); });
    auto create = factory(ledger);
    auto a = cache.acquire_range(1,128,extent,charge,9,17,true,create);
    verify(a,1,137,17); require(!a.cache_hit, "New immutable admission was a cache hit");
    auto copied = a; a = {};
    require(cache.stats().active_leases == 1, "C++ Lease copy incorrectly created a second lease record");
    copied = {};
    auto b = cache.acquire_range(2,128,extent,charge,0,extent,true,create); b = {};
    const auto before = cache.stats();
    require(before.reserved_bytes == 2*charge && before.resident_logical_bytes == 2*extent
        && before.owned_entries == 2 && before.entries == 2 && before.pending_admissions == 0,
        "Logical extent incorrectly funded physical/metadata ownership");
    for (int i=0; i<100; ++i)
        require(!cache.acquire_range(1,1024+i*extent,extent,charge,0,7,false,create), "Prefill scan admitted immutable bytes");
    auto hit = cache.acquire_range(1,128,extent,charge,3,11,false,create);
    verify(hit,1,131,11); require(hit.cache_hit && cache.stats().hit_bytes == 0, "Lookup counted unguarded hit-copy bytes");
    cache.record_copy(hit.bytes); hit = {};
    auto c = cache.acquire_range(3,128,extent,charge,0,extent,true,create); c = {};
    require(!cache.acquire_range(1,128,extent,charge,0,1,false,create), "Hit-only prefill aged oldest decode entry");
    auto newer = cache.acquire_range(2,128,extent,charge,0,extent,false,create); newer = {};
    require(cache.stats().hit_bytes == 11 && ledger.factories == 3 && ledger.peak <= 2*charge,
        "Immutable scan accounting or physical ceiling changed");
    cache.erase_source(2);
    require(cache.stats().entries == 1 && cache.stats().reserved_bytes == charge, "Source erasure retained an unleased owner");
    cache.clear(); ledger.empty();
}

static void leases_pressure_and_cache_destruction() {
    Ledger ledger;
    auto cache = std::make_unique<Cache>(charge, [&](auto n) { return ledger.reserve(n); },
        [&](const auto& owner, auto n) { return ledger.commit(owner,n); });
    auto first = cache->acquire_range(11,0,extent,charge,0,extent,true,factory(ledger));
    auto second_record = cache->acquire_range(11,0,extent,charge,1,9,true,factory(ledger));
    auto record_copy = second_record;
    require(cache->stats().active_leases == 2, "Acquired records/copies have wrong active count");
    require(!cache->acquire_range(12,0,extent,charge,0,extent,true,factory(ledger)) && ledger.factories == 1,
        "Leased owner funded a replacement or was overwritten");
    cache->invalidate({11,0,extent});
    require(cache->stats().resident_logical_bytes == 0 && cache->stats().owned_entries == 1
        && cache->stats().reserved_bytes == charge, "Invalidation refunded leased physical storage");
    cache->set_budget(0);
    require(!cache->acquire_range(11,0,extent,charge,0,1,false,{}), "Invalidated entry remained addressable");
    verify(first,11,0,extent); cache.reset();
    require(ledger.active == charge && ledger.physical == 1, "Cache destruction released live lease storage");
    first = {}; second_record = {};
    require(ledger.active == charge, "Copied lease lost final ownership");
    record_copy = {}; ledger.empty();
}

static void duplicate_single_flight(bool fail) {
    Ledger ledger;
    Cache cache(charge, [&](auto n) { return ledger.reserve(n); },
        [&](const auto& owner, auto n) { return ledger.commit(owner,n); });
    std::promise<void> started, release;
    const auto allow = release.get_future().share();
    std::atomic<int> calls{0};
    Cache::Factory slow = [&](const auto& range, const auto& token) -> Cache::Storage {
        require(++calls == 1, "Immutable same-key factory was not single flight");
        started.set_value(); allow.wait();
        if (fail) throw std::runtime_error("injected immutable factory error");
        return storage(ledger,range,token);
    };
    auto run = [&] { return cache.acquire_range(7,256,extent,charge,5,17,true,slow); };
    std::vector<std::future<Cache::Lease>> pending;
    pending.push_back(std::async(std::launch::async, run)); started.get_future().wait();
    const auto misses_before_probe = cache.stats().misses;
    require(!cache.acquire_range(7,256,extent,charge,0,1,false,slow)
        && cache.stats().waits == 0 && cache.stats().pending_admissions == 1,
        "Hit-only prefill waited/admitted/aged an in-flight range");
    for (int i=0; i<7; ++i) pending.push_back(std::async(std::launch::async,run));
    try {
        until([&] { return cache.stats().waits == 7; });
        require(cache.stats().reserved_bytes == charge && cache.stats().owned_entries == 1,
            "Single-flight duplicates over-reserved metadata/physical owners");
    } catch (...) { release.set_value(); for (auto& future:pending) future.wait(); throw; }
    release.set_value();
    std::vector<Cache::Lease> leases;
    for (auto& future:pending) leases.push_back(future.get());
    require(calls == 1 && cache.stats().pending_admissions == 0, "Immutable factory completion leaked pending count");
    if (fail) {
        require(std::none_of(leases.begin(),leases.end(),[](const auto& l) { return bool(l); })
            && cache.stats().entries == 0 && cache.stats().reserved_bytes == 0
            && cache.stats().failures == 1, "Factory error published bytes or lost single-flight failure");
    } else {
        for (const auto& lease:leases) verify(lease,7,261,17);
        require(cache.stats().active_leases == 8 && cache.stats().hits == 7
            && cache.stats().misses == misses_before_probe+1 && cache.stats().hit_bytes == 0,
            "Concurrent leases/hits/copy bytes accounting changed");
        cache.clear(); require(cache.stats().reserved_bytes == charge, "Clear refunded concurrent live leases");
    }
    leases.clear(); cache.clear(); ledger.empty();
}

static void retire_during_factory(int mode) {
    Ledger ledger;
    Cache cache(charge, [&](auto n) { return ledger.reserve(n); },
        [&](const auto& owner, auto n) { return ledger.commit(owner,n); });
    std::promise<void> started, release;
    const auto allow = release.get_future().share();
    auto pending = std::async(std::launch::async,[&] {
        return cache.acquire_range(9,0,extent,charge,0,extent,true,[&](const auto& range,const auto& token) {
            started.set_value(); allow.wait(); return storage(ledger,range,token);
        });
    });
    started.get_future().wait();
    if (mode == 0) cache.clear();
    else if (mode == 1) cache.erase_source(9);
    else if (mode == 2) cache.invalidate({9,0,extent});
    else if (mode == 3) cache.set_budget(charge-1);
    else if (mode == 4) cache.set_budget(charge);
    else cache.set_budget(2*charge);
    require(cache.stats().entries == 0 && cache.stats().reserved_bytes == charge
        && cache.stats().owned_entries == 1 && cache.stats().pending_admissions == 1,
        "Retired factory refunded pending physical reservation");
    release.set_value(); auto lease = pending.get();
    require(!lease && cache.stats().entries == 0 && cache.stats().reserved_bytes == 0
        && cache.stats().pending_admissions == 0 && cache.stats().owned_entries == 0,
        "Retired loading generation resurrected or leaked its owner");
    ledger.empty();
}

static void factory_commit_and_quarantine_failures() {
    for (int mode=0; mode<9; ++mode) {
        Ledger ledger;
        std::vector<std::shared_ptr<BytesOwner>> quarantine;
        Cache cache(charge,[&](auto n) -> std::shared_ptr<void> {
            if (mode == 0) return {};
            if (mode == 1) throw std::bad_alloc();
            return ledger.reserve(n);
        },[&](const auto& owner,auto n) {
            ledger.commit(owner,n);
            if (mode == 4 || mode == 8) throw std::runtime_error("injected immutable commit exception");
            return mode != 3 && mode != 7;
        });
        auto result = cache.acquire_range(1,0,extent,charge,0,extent,true,[&](const auto& range,const auto& token) -> Cache::Storage {
            ++ledger.factories;
            auto owner = std::make_shared<BytesOwner>(range,token,ledger);
            if (mode >= 5) {
                quarantine.push_back(owner);
                if (mode == 6) throw std::runtime_error("factory failed after acquiring quarantined resources");
                if (mode == 5) return {owner,nullptr,range.bytes};
            }
            return {owner,owner->bytes.data(),mode == 2 ? range.bytes-1 : range.bytes};
        });
        require(!result && cache.stats().resident_logical_bytes == 0 && cache.stats().pending_admissions == 0,
            "Optional immutable failure published storage or pending state");
        if (mode >= 5) {
            require(cache.stats().reserved_bytes == charge && cache.stats().owned_entries == 1
                && ledger.active == charge && ledger.physical == 1,
                "Quarantined failed factory silently refunded its lifetime/Usage");
            cache.clear(); cache.set_budget(0);
            require(cache.stats().reserved_bytes == charge, "Clear/pressure refunded failed backend cleanup");
            quarantine.clear();
        }
        require(cache.stats().reserved_bytes == 0 && cache.stats().owned_entries == 0,
            "Failed storage lifetime did not release after actual cleanup");
        ledger.empty();
    }
}

static void parallel_pressure_stress() {
    Ledger ledger;
    ledger.strict_final_order = false;
    Cache cache(4*charge,[&](auto n) { return ledger.reserve(n); },
        [&](const auto& owner,auto n) { return ledger.commit(owner,n); });
    std::vector<std::future<void>> workers;
    for (unsigned thread=0; thread<8; ++thread) workers.push_back(std::async(std::launch::async,[&,thread] {
        for (unsigned i=0; i<120; ++i) {
            const auto source = 1+(i+thread)%5;
            const auto base = std::uint64_t((i*7+thread)%11)*extent;
            auto lease = cache.acquire_range(source,base,extent,charge,i%32,17,i%5 != 0,factory(ledger));
            if (lease) {
                verify(lease,source,base+i%32,17);
                if (lease.cache_hit) cache.record_copy(lease.bytes);
            }
            if (i%19 == 0) cache.erase_source(1+(i/19+thread)%5);
            if (i%37 == 0) cache.set_budget((1+(thread+i)%4)*charge);
        }
    }));
    for (auto& worker:workers) worker.get();
    require(ledger.peak <= 4*charge && cache.stats().reserved_bytes <= 4*charge
        && cache.stats().pending_admissions == 0 && cache.stats().active_leases == 0,
        "Parallel admission/pressure exceeded physical budget or retained work");
    cache.set_budget(0); cache.clear();
    require(cache.stats().owned_entries == 0 && cache.stats().reserved_bytes == 0,
        "Parallel retirement resurrected a key/owner");
    ledger.empty();
}

static void metadata_and_lease_record_bounds() {
    for (int stage=1; stage<=5; ++stage) {
        Ledger ledger;
        Cache cache(charge,[&](auto n) { return ledger.reserve(n); },
            [&](const auto& owner,auto n) { return ledger.commit(owner,n); });
        cache.test_fail_metadata(stage);
        require(!cache.acquire_range(1,0,extent,charge,0,extent,true,factory(ledger)),
            "Injected metadata failure did not bypass optional admission");
        require(cache.stats().entries == 0 && cache.stats().reserved_bytes == 0
            && cache.stats().owned_entries == 0 && cache.stats().failures == 1,
            "Metadata failure left a loading key/physical ticket");
        auto retry = cache.acquire_range(1,0,extent,charge,0,extent,true,factory(ledger));
        verify(retry,1,0,extent); retry = {}; cache.clear(); ledger.empty();
    }
    Ledger ledger;
    Cache cache(charge,[&](auto n) { return ledger.reserve(n); });
    std::vector<Cache::Lease> leases;
    for (std::size_t i=0; i<Cache::maximum_leases_per_entry; ++i) {
        leases.push_back(cache.acquire_range(1,0,extent,charge,0,1,true,factory(ledger)));
        require(bool(leases.back()), "Bounded immutable lease record unexpectedly refused");
    }
    auto copy = leases.front();
    require(cache.stats().active_leases == Cache::maximum_leases_per_entry
        && !cache.acquire_range(1,0,extent,charge,0,1,true,factory(ledger)),
        "Additional lease metadata exceeded conservative per-entry bound");
    leases.pop_back();
    auto next = cache.acquire_range(1,0,extent,charge,0,1,true,factory(ledger));
    require(bool(next), "Released lease record did not restore bounded capacity");
    next = {}; copy = {}; leases.clear(); cache.clear(); ledger.empty();
}

static void callbacks_outside_mutex_and_generations() {
    Ledger ledger;
    Cache* active = nullptr;
    Cache cache(charge,[&](auto n) { require(active->stats().pending_admissions == 1, "Reserve sees no pending Usage owner"); return ledger.reserve(n); },
        [&](const auto& owner,auto n) { require(active->stats().reserved_bytes == n, "Commit sees no physical charge"); return ledger.commit(owner,n); });
    active = &cache;
    ledger.on_release = [&] { (void) cache.stats(); };
    auto lease = cache.acquire_range(101,0,extent,charge,0,extent,true,[&](const auto& range,const auto& token) {
        require(cache.stats().pending_admissions == 1, "Factory callback cannot inspect unlocked cache state");
        return storage(ledger,range,token);
    });
    lease = {}; cache.erase_source(101);
    auto newer = cache.acquire_range(102,0,extent,charge,0,extent,true,factory(ledger));
    verify(newer,102,0,extent);
    require(!newer.cache_hit && ledger.factories == 2, "Source generation reused a stale immutable key");
    newer = {}; cache.clear(); ledger.empty(); ledger.on_release = {};
    const auto rejects = [&](std::uint64_t source,std::uint64_t base,std::size_t size,std::size_t charged,std::size_t relative,std::size_t count) {
        bool failed=false;
        try { cache.acquire_range(source,base,size,charged,relative,count,true,factory(ledger)); }
        catch (const std::invalid_argument&) { failed=true; }
        require(failed, "Invalid immutable range/charge accepted");
    };
    rejects(0,0,extent,charge,0,1); rejects(1,0,0,charge,0,1);
    rejects(1,0,extent,charge,extent,1); rejects(1,UINT64_MAX-1,extent,charge,0,1);
    rejects(1,0,extent,extent+Cache::entry_metadata_bytes-1,0,1);
    rejects(1,0,extent,charge,0,0);
}

static void ready_lookup_lazy_geometry_and_accounting() {
    Ledger ledger;
    Cache cache(2*charge,[&](auto n) { return ledger.reserve(n); },
        [&](const auto& owner,auto n) { return ledger.commit(owner,n); });
    int plans = 0;
    const auto lazy = [&](std::uint64_t source,bool admit,bool valid_plan=true) -> Cache::Lease {
        auto ready = cache.find_ready_range(source,0,extent,7,11,admit,!admit);
        if (ready.status == Cache::ReadyStatus::ready) return std::move(ready.lease);
        if (ready.status == Cache::ReadyStatus::lease_refused || !admit) return {};
        ++plans; // Represents the caller's physical geometry/API work.
        if (!valid_plan) { cache.record_miss_bypass(11); return {}; }
        return cache.acquire_range(source,0,extent,charge,7,11,true,factory(ledger));
    };
    require(!lazy(1,false) && plans == 0 && ledger.factories == 0,
        "Hit-only miss performed physical geometry/admission work");
    auto admitted = lazy(1,true);
    verify(admitted,1,7,11); require(!admitted.cache_hit && plans == 1, "Lazy admission was not planned exactly once");
    admitted = {};
    auto hit = lazy(1,true);
    verify(hit,1,7,11);
    require(hit.cache_hit && plans == 1 && ledger.factories == 1 && ledger.reserves == 1
        && ledger.commits == 1 && cache.stats().hit_bytes == 0,
        "Ready lookup invoked geometry/factory/charge or counted an unguarded copy");
    cache.record_copy(hit.bytes); hit = {};
    auto s = cache.stats();
    require(s.hits == 1 && s.misses == 2 && s.bypasses == 1 && s.bypass_logical_bytes == 11
        && s.admitted == 1 && s.hit_bytes == 11, "Lazy lookup/admission counted attempts twice");
    require(!lazy(2,true,false) && plans == 2 && ledger.factories == 1,
        "Declined geometry invoked factory");
    s = cache.stats();
    require(s.hits == 1 && s.misses == 3 && s.bypasses == 2 && s.bypass_logical_bytes == 22,
        "Declined geometry did not account one suppressed miss/bypass");
    cache.test_fail_metadata(5);
    require(!lazy(1,true) && plans == 2 && ledger.factories == 1,
        "Caller retried physical planning/admission after ready lease metadata refusal");
    s = cache.stats();
    require(s.hits == 2 && s.misses == 3 && s.bypasses == 3 && s.failures == 1,
        "Terminal ready lease refusal double-counted a miss/admission");
    require(cache.find_ready_range(2,0,extent,0,1,false,true).status == Cache::ReadyStatus::not_ready
        && cache.find_ready_range(1,1,extent,0,1,false,true).status == Cache::ReadyStatus::not_ready
        && cache.find_ready_range(1,0,extent-1,0,1,false,true).status == Cache::ReadyStatus::not_ready,
        "Ready lookup confused source generation/base/canonical extent");
    // The first probe may race a genuine admission. The normal admission API
    // rechecks the key and returns a hit without another physical factory.
    require(cache.find_ready_range(3,0,extent,0,1,true,false).status == Cache::ReadyStatus::not_ready,
        "Cold admission probe was ready");
    auto other = cache.acquire_range(3,0,extent,charge,0,1,true,factory(ledger)); other = {};
    const auto before_race = cache.stats();
    auto raced = cache.acquire_range(3,0,extent,charge,0,1,true,factory(ledger));
    require(raced && raced.cache_hit && ledger.factories == 2 && cache.stats().misses == before_race.misses
        && cache.stats().hits == before_race.hits+1, "Lookup/admission race duplicated a factory or miss");
    raced = {}; cache.clear(); ledger.empty();
}

static void ready_lookup_recency_and_validation() {
    Ledger ledger;
    Cache cache(2*charge,[&](auto n) { return ledger.reserve(n); });
    for (std::uint64_t source : {1u,2u}) {
        auto admitted = cache.acquire_range(source,0,extent,charge,0,1,true,factory(ledger));
    }
    auto old = cache.find_ready_range(1,0,extent,0,1,false,true);
    require(old.status == Cache::ReadyStatus::ready && old.lease.cache_hit, "Hit-only ready lookup missed");
    old = {};
    auto third = cache.acquire_range(3,0,extent,charge,0,1,true,factory(ledger)); third = {};
    require(cache.find_ready_range(1,0,extent,0,1,false,true).status == Cache::ReadyStatus::not_ready,
        "Hit-only ready lookup aged LRU recency");
    auto touched = cache.find_ready_range(2,0,extent,0,1,true,false); touched = {};
    auto fourth = cache.acquire_range(4,0,extent,charge,0,1,true,factory(ledger)); fourth = {};
    require(cache.find_ready_range(3,0,extent,0,1,false,true).status == Cache::ReadyStatus::not_ready,
        "Demand ready lookup did not age LRU recency");
    auto survived = cache.find_ready_range(2,0,extent,0,1,false,true);
    require(survived.status == Cache::ReadyStatus::ready, "Touched ready entry was evicted");
    survived = {};
    const auto rejects = [&](std::uint64_t source,std::uint64_t base,std::size_t size,std::size_t relative,std::size_t count) {
        const auto before = cache.stats();
        bool failed=false;
        try { cache.find_ready_range(source,base,size,relative,count,false,true); }
        catch (const std::invalid_argument&) { failed=true; }
        const auto after = cache.stats();
        require(failed && before.hits == after.hits && before.misses == after.misses
            && before.bypasses == after.bypasses, "Invalid ready range altered counters or was accepted");
    };
    rejects(0,0,extent,0,1); rejects(1,0,0,0,1); rejects(1,0,extent,0,0);
    rejects(1,0,extent,extent,1); rejects(1,UINT64_MAX-1,extent,0,1);
    // Logical lookup does not impose the admission-only +metadata charge bound.
    require(cache.find_ready_range(55,0,(std::numeric_limits<std::size_t>::max)(),0,1,false,false).status
        == Cache::ReadyStatus::not_ready, "Ready lookup required a physical charge/metadata envelope");
    cache.erase_source(2);
    require(cache.find_ready_range(2,0,extent,0,1,false,true).status == Cache::ReadyStatus::not_ready,
        "Source erasure left an old ready generation visible");
    cache.clear(); ledger.empty();
}

static void ready_lookup_refusal_is_terminal() {
    Ledger ledger;
    Cache cache(charge,[&](auto n) { return ledger.reserve(n); });
    auto admitted = cache.acquire_range(1,0,extent,charge,0,1,true,factory(ledger)); admitted = {};
    const auto before = cache.stats();
    cache.test_fail_metadata(5);
    auto refused = cache.find_ready_range(1,0,extent,0,1,true,false);
    auto s = cache.stats();
    require(refused.status == Cache::ReadyStatus::lease_refused && !refused.lease
        && s.hits == before.hits+1 && s.misses == before.misses && s.bypasses == before.bypasses+1
        && s.failures == before.failures+1 && s.pending_admissions == 0 && s.entries == 1
        && ledger.factories == 1 && ledger.reserves == 1,
        "Ready metadata refusal looked like an admission miss or repeated physical work");
    std::vector<Cache::Lease> leases;
    for (std::size_t i=0; i<Cache::maximum_leases_per_entry; ++i) {
        auto value = cache.find_ready_range(1,0,extent,0,1,false,false);
        require(value.status == Cache::ReadyStatus::ready, "Ready lease below record bound refused");
        leases.push_back(std::move(value.lease));
    }
    const auto before_cap = cache.stats();
    refused = cache.find_ready_range(1,0,extent,0,1,true,false);
    s = cache.stats();
    require(refused.status == Cache::ReadyStatus::lease_refused && !refused.lease
        && s.hits == before_cap.hits+1 && s.misses == before_cap.misses && s.bypasses == before_cap.bypasses+1
        && s.failures == before_cap.failures && s.active_leases == Cache::maximum_leases_per_entry
        && ledger.factories == 1, "Ready lease cap was counted as miss/failure or retried admission");
    leases.pop_back();
    auto retry = cache.find_ready_range(1,0,extent,0,1,true,false);
    require(retry.status == Cache::ReadyStatus::ready, "Released record did not restore ready lookup");
    retry = {}; leases.clear(); cache.clear(); ledger.empty();
}

static void ready_lookup_during_loading() {
    Ledger ledger;
    Cache cache(charge,[&](auto n) { return ledger.reserve(n); });
    std::promise<void> started, release;
    const auto allow = release.get_future().share();
    auto pending = std::async(std::launch::async,[&] {
        return cache.acquire_range(9,0,extent,charge,0,extent,true,[&](const auto& range,const auto& token) {
            started.set_value(); allow.wait(); return storage(ledger,range,token);
        });
    });
    started.get_future().wait();
    const auto before = cache.stats();
    auto probe = std::async(std::launch::async,[&] {
        auto suppressed = cache.find_ready_range(9,0,extent,0,7,true,false);
        auto counted = cache.find_ready_range(9,0,extent,0,7,false,true);
        return suppressed.status == Cache::ReadyStatus::not_ready && !suppressed.lease
            && counted.status == Cache::ReadyStatus::not_ready && !counted.lease;
    });
    const bool immediate = probe.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
    const auto during = cache.stats();
    release.set_value();
    const bool not_ready = probe.get();
    auto result = pending.get();
    require(immediate && not_ready && during.misses == before.misses+1
        && during.bypasses == before.bypasses+1 && during.bypass_logical_bytes == before.bypass_logical_bytes+7
        && during.waits == before.waits && during.pending_admissions == 1,
        "Ready-only lookup waited on/loading-claimed a key or double-counted its suppressed miss");
    verify(result,9,0,extent); result = {};
    auto ready = cache.find_ready_range(9,0,extent,3,9,false,true);
    require(ready.status == Cache::ReadyStatus::ready && ledger.factories == 1,
        "Completed single-flight key was not visible to ready lookup");
    ready = {}; cache.clear(); ledger.empty();
}

int main() {
    try {
        exact_physical_charge_and_scan(); leases_pressure_and_cache_destruction();
        duplicate_single_flight(false); duplicate_single_flight(true);
        for (int mode=0; mode<6; ++mode) retire_during_factory(mode);
        factory_commit_and_quarantine_failures(); metadata_and_lease_record_bounds();
        callbacks_outside_mutex_and_generations(); parallel_pressure_stress();
        ready_lookup_lazy_geometry_and_accounting(); ready_lookup_recency_and_validation();
        ready_lookup_refusal_is_terminal(); ready_lookup_during_loading();
        std::printf("PASS immutable cache: %d checks; const exact slices, physical/final-owner charge, single flight, hit-only scans, retire/no resurrection, quarantine, callbacks, bounded lease metadata, lazy geometry ready-only lookup/accounting/refusal/loading; no CUDA or model\n", checks.load());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr,"FAIL immutable cache: %s\n",error.what()); return 1;
    }
}
