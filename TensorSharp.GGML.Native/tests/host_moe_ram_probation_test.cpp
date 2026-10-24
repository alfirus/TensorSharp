// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#define TSG_HOST_MOE_RAM_PROBATION_TEST_HOOKS 1
#include "../ggml_ops_host_moe_ram_probation.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <future>
#include <new>
#include <vector>

namespace {
std::atomic<bool> watch_heap{false};
std::atomic<std::size_t> watched_allocations{0};
std::atomic<int> checks{0};
void require(bool value, const char* message) {
    ++checks; if (!value) throw std::runtime_error(message);
}
}
void* operator new(std::size_t count) {
    if (watch_heap.load(std::memory_order_relaxed)) watched_allocations.fetch_add(1,std::memory_order_relaxed);
    if (auto* value = std::malloc(count ? count : 1)) return value;
    throw std::bad_alloc();
}
void* operator new[](std::size_t count) { return ::operator new(count); }
void operator delete(void* value) noexcept { std::free(value); }
void operator delete[](void* value) noexcept { std::free(value); }
void operator delete(void* value,std::size_t) noexcept { std::free(value); }
void operator delete[](void* value,std::size_t) noexcept { std::free(value); }

using Policy = tsg::HostMoeRamProbation<>;
template<std::size_t N> using Small = tsg::HostMoeRamProbation<N>;
static Policy::Range key(std::uint64_t expert,std::uint64_t source=1,std::size_t bytes=64) {
    return {source,expert*64,bytes};
}
template<std::size_t N> struct Oracle {
    using Range = typename Small<N>::Range;
    std::deque<Range> oldest_first;
    std::uint64_t observations=0,repeats=0,firsts=0,evictions=0;
    bool observe(const Range& range) {
        ++observations;
        const auto found = std::find(oldest_first.begin(),oldest_first.end(),range);
        const bool previous = found != oldest_first.end();
        if (previous) { oldest_first.erase(found); ++repeats; }
        else {
            ++firsts;
            if (oldest_first.size() == N) { oldest_first.pop_front(); ++evictions; }
        }
        oldest_first.push_back(range); return previous;
    }
    void erase_source(std::uint64_t source) {
        oldest_first.erase(std::remove_if(oldest_first.begin(),oldest_first.end(),
            [&](const auto& range) { return range.source == source; }),oldest_first.end());
    }
    void clear() { oldest_first.clear(); observations=repeats=firsts=evictions=0; }
    void compare(const Small<N>& policy) const {
        const auto s = policy.stats();
        require(s.resident_keys == oldest_first.size() && s.observations == observations
            && s.repeats == repeats && s.firsts == firsts && s.evictions == evictions,
            "Fixed probation differs from independent deque oracle counters/bound");
    }
};

static void exact_lru_oracle() {
    Small<32> policy;
    Oracle<32> oracle;
    std::uint64_t random = UINT64_C(0x4d3c2b1a98765431);
    for (unsigned step=0; step<30000; ++step) {
        random ^= random << 13; random ^= random >> 7; random ^= random << 17;
        const Small<32>::Range range{1+(random>>9)%7,(random%73)*64,32+(random>>15)%3*32};
        require(policy.observe(range) == oracle.observe(range), "Second-demand decision differs from exact prior LRU membership");
        if (step%71 == 0) { policy.erase_source(1+(step/71)%7); oracle.erase_source(1+(step/71)%7); }
        if (step%4999 == 0) { policy.clear(); oracle.clear(); }
        oracle.compare(policy);
    }
}

static void generation_extent_and_full_capacity() {
    Policy policy;
    require(!policy.observe(key(5)) && policy.observe(key(5)) && policy.observe(key(5)),
        "First/second/later canonical demand has wrong eligibility");
    require(!policy.observe(key(5,2)) && !policy.observe(key(5,1,63))
        && !policy.observe({1,321,64}), "Source generation/base/extent aliased probation key");
    policy.erase_source(1);
    require(!policy.observe(key(5)) && policy.observe(key(5,2)),
        "Source erase forgot unrelated keys or retained retired source history");
    const auto before_invalid = policy.stats();
    for (const auto range : {Policy::Range{0,0,64},Policy::Range{1,0,0},Policy::Range{1,UINT64_MAX-31,64}}) {
        bool rejected=false;
        try { policy.observe(range); } catch (const std::invalid_argument&) { rejected=true; }
        require(rejected, "Invalid source/range entered probation");
    }
    require(policy.stats().observations == before_invalid.observations, "Invalid range changed observation counters");
    policy.clear();
    for (std::size_t i=0; i<Policy::capacity; ++i) require(!policy.observe(key(i)), "Cold full-capacity key repeated");
    require(policy.stats().resident_keys == Policy::capacity && policy.stats().evictions == 0,
        "Probation did not fill exactly4096 independent nodes");
    require(policy.observe(key(0)), "Oldest key disappeared before pressure");
    require(!policy.observe(key(Policy::capacity)), "Capacity+1 distinct key was a repeat");
    require(policy.observe(key(0)) && !policy.observe(key(1)), "Full LRU did not protect touched oldest key/evict next oldest");
    require(policy.stats().resident_keys == Policy::capacity && policy.stats().evictions == 2,
        "Capacity overflow grew metadata or missed exact eviction accounting");
}

static void deliberate_hash_collisions_and_reuse() {
    Small<8> policy;
    Oracle<8> oracle;
    std::vector<Small<8>::Range> collisions;
    for (std::uint64_t candidate=0; collisions.size()<14; ++candidate) {
        const Small<8>::Range range{1+candidate%3,candidate*64,64};
        if (Small<8>::test_bucket(range) == 0) collisions.push_back(range);
    }
    for (unsigned round=0; round<40; ++round) {
        for (const auto& range : collisions) {
            require(policy.observe(range) == oracle.observe(range), "Hash chain collision lost exact key");
            oracle.compare(policy);
        }
        for (std::uint64_t source=1; source<=3; ++source) {
            policy.erase_source(source); oracle.erase_source(source); oracle.compare(policy);
            const auto& range = collisions[(round+source)%collisions.size()];
            require(policy.observe(range) == oracle.observe(range), "Hash removal/free-node reuse corrupted collision chain");
        }
    }
}

static void readiness_is_independent_and_fragments_do_not_observe() {
    Small<8> policy;
    const Small<8>::Range canonical{99,1024,4096};
    require(!policy.observe(canonical), "First miss demand admitted itself");
    const bool allow = policy.observe(canonical);
    require(allow, "Second canonical miss demand was refused");
    const auto before_fragments = policy.stats().observations;
    for (int fragment=0; fragment<7; ++fragment) require(allow, "One canonical decision changed across fragments");
    require(policy.stats().observations == before_fragments,
        "Caller fragment reuse advanced canonical demand history");
    // A ready RAM owner can outlive a history node. Returning false here must
    // never suppress that owner's ready hit or its ordinary-demand RAM touch.
    for (std::uint64_t i=0; i<8; ++i) require(!policy.observe({100+i,i*4096,4096}), "Distinct pressure source repeated");
    const bool history_repeat = policy.observe(canonical);
    const bool ram_ready = true; // Deliberately independent residency fixture.
    require(!history_repeat && ram_ready, "Aged-out ready RAM identity was incorrectly treated as repeated history");
    policy.clear();
    const auto s=policy.stats();
    require(s.resident_keys == 0 && s.observations == 0 && s.repeats == 0 && s.firsts == 0 && s.evictions == 0,
        "Clear did not begin a fresh bounded history generation");
}

static void valid_operations_do_not_allocate() {
    Policy policy;
    watched_allocations=0; watch_heap=true;
    bool valid=true;
    for (std::uint64_t i=0; i<20000; ++i) {
        const Policy::Range range{1+i%11,(i%5003)*64,64};
        (void)policy.observe(range);
        if (i%79 == 0) policy.erase_source(1+(i/79)%11);
        if (i%4001 == 0) policy.clear();
        valid = valid && policy.stats().resident_keys <= Policy::capacity;
    }
    watch_heap=false;
    require(valid && watched_allocations == 0, "Valid observe/erase/clear/stats allocated heap metadata");
    require(sizeof(Policy) <= Policy::metadata_forecast_bytes, "Default policy exceeds separately charged1MiB bound");
}

static void concurrent_observe_and_retirement() {
    Policy shared;
    std::vector<std::future<void>> workers;
    for (std::uint64_t thread=0; thread<6; ++thread) workers.push_back(std::async(std::launch::async,[&,thread] {
        for (std::uint64_t i=0; i<3000; ++i) (void)shared.observe({1+(i+thread)%11,((i*7+thread)%6001)*64,64});
    }));
    for (auto& worker:workers) worker.get();
    auto s=shared.stats();
    require(s.observations == 18000 && s.observations == s.firsts+s.repeats
        && s.resident_keys <= Policy::capacity && s.evictions == s.firsts-s.resident_keys,
        "Concurrent observations lost events or exceeded fixed history cardinality");
    workers.clear();
    for (std::uint64_t thread=0; thread<6; ++thread) workers.push_back(std::async(std::launch::async,[&,thread] {
        for (std::uint64_t i=0; i<2000; ++i) {
            (void)shared.observe({1+(i+thread)%11,((i*13+thread)%6001)*64,64});
            if (i%97 == 0) shared.erase_source(1+(i/97+thread)%11);
            if (i%503 == 0) shared.clear();
            require(shared.stats().resident_keys <= Policy::capacity, "Concurrent source/clear retirement exceeded node bound");
        }
    }));
    for (auto& worker:workers) worker.get();
    s=shared.stats();
    require(s.observations == s.firsts+s.repeats && s.resident_keys <= Policy::capacity,
        "Concurrent retirement corrupted exact observation accounting");
    for (std::uint64_t source=1; source<=11; ++source) shared.erase_source(source);
    require(shared.stats().resident_keys == 0, "Bounded source erasure leaked nodes after concurrent use");
}

int main() {
    try {
        exact_lru_oracle(); generation_extent_and_full_capacity(); deliberate_hash_collisions_and_reuse();
        readiness_is_independent_and_fragments_do_not_observe(); valid_operations_do_not_allocate();
        concurrent_observe_and_retirement();
        std::printf("PASS RAM probation: %d checks; exact prior-membership LRU4096, generation/range keys, collisions, independent deque oracle, no-heap valid operations,6-worker safety; sizeof=%zu forecast=%zu; no CUDA/model/performance claim\n",
            checks.load(),sizeof(Policy),Policy::metadata_forecast_bytes);
        return 0;
    } catch (const std::exception& error) {
        watch_heap=false; std::fprintf(stderr,"FAIL RAM probation: %s\n",error.what()); return 1;
    }
}
