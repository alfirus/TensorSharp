// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_shared_cache_budget.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

static void require(bool value, const char* message)
{
    if (!value) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
struct Ledger {
    struct Allocation { int rank; std::int64_t bytes; bool committed = false; bool physical = false; };
    std::int64_t capacity[3] = {128, 64, 64}; // Shared RAM plus independent GPU constraints.
    std::int64_t reserved[3] = {}, committed[3] = {};
    std::uint64_t next = 0;
    std::map<std::uint64_t, Allocation> allocations;
    bool fail_commit = false;
    bool fail_checked_release = false, fail_release_ack = false;
    int checked_release_calls = 0;
    std::set<std::uint64_t> released_waiting_ack;
    int last_kind = -1;
    static std::vector<int> pools(int rank) { return rank < 0 ? std::vector<int>{0} : std::vector<int>{0, rank + 1}; }
    static std::uint64_t reserve(void* context, int rank, int kind, std::int64_t bytes)
    {
        auto& l = *static_cast<Ledger*>(context);
        require(kind == 0 || kind == 1 || kind == 3 || kind == 4 || kind == 5 || kind == 6 || kind == 7, "unknown cache allocation kind");
        l.last_kind = kind;
        if (rank < 0 || rank > 1) return 0;
        if (kind == 3 || kind == 6) rank = -1; // Host payload consumes RAM, no GPU credit.
        for (int pool : pools(rank))
            if (bytes > l.capacity[pool] - l.reserved[pool] - l.committed[pool]) return 0;
        const auto token = ++l.next;
        l.allocations.emplace(token, Allocation{rank, bytes});
        for (int pool : pools(rank)) l.reserved[pool] += bytes;
        return token;
    }
    static int commit(void* context, std::uint64_t token)
    {
        auto& l = *static_cast<Ledger*>(context);
        if (l.fail_commit) return 0;
        auto& a = l.allocations.at(token);
        require(!a.committed, "double commit");
        for (int pool : pools(a.rank)) { l.reserved[pool] -= a.bytes; l.committed[pool] += a.bytes; }
        a.committed = true;
        return 1;
    }
    static void release(void* context, std::uint64_t token)
    {
        auto& l = *static_cast<Ledger*>(context);
        auto a = l.allocations.at(token);
        require(!a.physical, "shared credits released before physical memory");
        for (int pool : pools(a.rank)) (a.committed ? l.committed[pool] : l.reserved[pool]) -= a.bytes;
        l.allocations.erase(token);
    }
    static int release_checked(void* context, std::uint64_t token) {
        auto& l = *static_cast<Ledger*>(context);
        ++l.checked_release_calls;
        if (l.fail_checked_release) return 0;
        if (l.released_waiting_ack.erase(token)) return 1;
        release(context, token);
        if (l.fail_release_ack) {
            l.released_waiting_ack.insert(token);
            return 0;
        }
        return 1;
    }
};
static bool attach(Ledger& ledger)
{
    return tsg::SharedCacheCharge::attach(&ledger, Ledger::reserve, Ledger::commit, Ledger::release);
}

int main()
{
    using tsg::SharedCacheCharge;
    Ledger ledger;
    // An allocation admitted before configuration cannot silently escape a new budget.
    auto unconfigured = SharedCacheCharge::reserve(0, 0, 64);
    require(bool(unconfigured) && !attach(ledger), "attach adopted unbudgeted in-flight allocation");
    require(unconfigured->commit(64) && !attach(ledger), "attach adopted an unbudgeted resident allocation");
    unconfigured.reset();
    auto old_expert = SharedCacheCharge::reserve(0, 4, 32);
    require(old_expert && !attach(ledger), "attach adopted pending unbudgeted selected experts");
    require(old_expert->commit(32) && !attach(ledger), "attach adopted resident unbudgeted selected experts");
    old_expert.reset();
    require(attach(ledger), "cannot attach an empty cache budget");
    require(!attach(ledger), "double registration accepted");
    require(!SharedCacheCharge::detach(nullptr), "wrong owner detached callbacks");

    auto first = SharedCacheCharge::reserve(0, 0, 64);
    auto second = SharedCacheCharge::reserve(1, 1, 64);
    require(first && second && ledger.reserved[0] == 128, "rank/UMA reservation is not atomic");
    require(!SharedCacheCharge::reserve(1, 0, 1), "shared quota exceeded");
    require(!SharedCacheCharge::detach(&ledger), "detached pending callbacks");
    require(first->commit(64) && second->commit(64), "commit failed");
    require(ledger.committed[0] == 128 && ledger.committed[1] == 64 && ledger.committed[2] == 64,
        "committed multi-pool accounting is incorrect");
    require(!SharedCacheCharge::detach(&ledger), "detached resident callbacks");
    first.reset(); second.reset();

    auto preload = SharedCacheCharge::reserve(0, 1, 32);
    require(preload && ledger.last_kind == 1, "preload kind changed");
    auto expert = SharedCacheCharge::reserve(0, 4, 32);
    require(expert && expert->is_accounted() && ledger.last_kind == 4,
        "selected-expert callback did not receive its dedicated kind");
    require(!SharedCacheCharge::reserve(0, 4, 1), "selected experts exceeded shared GPU credit");
    require(!SharedCacheCharge::detach(&ledger), "detached pending selected-expert ownership");
    require(preload->commit(32) && expert->commit(32) && ledger.committed[1] == 64,
        "preload/selected-expert shared accounting changed");
    preload.reset();
    require(!SharedCacheCharge::detach(&ledger), "detached resident selected-expert ownership");
    expert.reset();
    const auto calls_before_invalid = ledger.next;
    require(!SharedCacheCharge::reserve(0, -1, 1) && !SharedCacheCharge::reserve(0, 8, 1)
        && ledger.next == calls_before_invalid, "invalid kind reached the callback");

    // A native allocation/upload failure unwinds physical storage before its
    // ticket; both pending and committed failures obey the same ownership order.
    for (bool committed : {false, true}) {
        try {
            auto ticket = SharedCacheCharge::reserve(0, 4, 64);
            require(bool(ticket), "cannot reserve rollback case");
            auto& allocation = ledger.allocations.at(ledger.next);
            allocation.physical = true;
            struct Physical { bool& live; ~Physical() { live = false; } } physical{allocation.physical};
            if (committed) require(ticket->commit(64), "cannot commit rollback case");
            throw std::runtime_error("injected native allocation/upload failure");
        } catch (const std::runtime_error&) { }
        require(ledger.allocations.empty() && ledger.reserved[0] == 0 && ledger.committed[0] == 0,
            "rollback leaked shared credit");
    }
    auto too_large = SharedCacheCharge::reserve(0, 0, 64);
    require(!too_large->commit(65), "under-reserved allocation published");
    too_large.reset();
    ledger.fail_commit = true;
    auto refused_commit = SharedCacheCharge::reserve(0, 0, 64);
    require(!refused_commit->commit(64), "failed callback commit accepted");
    refused_commit.reset();
    ledger.fail_commit = false;

    std::atomic<int> attempted{0}, admitted{0};
    std::atomic<bool> release{false};
    std::vector<std::thread> workers;
    for (int i = 0; i < 32; ++i) workers.emplace_back([&, i] {
        const int kinds[] = {0, 1, 4};
        auto ticket = SharedCacheCharge::reserve(i % 2, kinds[i % 3], 16);
        if (ticket) admitted.fetch_add(1);
        attempted.fetch_add(1);
        while (!release.load()) std::this_thread::yield();
        if (ticket) require(ticket->commit(16), "concurrent reservation could not commit");
    });
    while (attempted.load() != 32) std::this_thread::yield();
    require(admitted == 8 && !SharedCacheCharge::detach(&ledger), "concurrent admission or detach escaped live ownership");
    release = true;
    for (auto& worker : workers) worker.join();
    require(ledger.allocations.empty() && ledger.committed[0] == 0 && ledger.reserved[0] == 0,
        "concurrent release leaked shared charges");
    require(SharedCacheCharge::detach(&ledger), "cannot detach fully released callbacks");
    require(attach(ledger) && SharedCacheCharge::detach(&ledger), "detach cannot be followed by clean registration");
    auto host_attach = [&] { return SharedCacheCharge::attach(&ledger, Ledger::reserve, Ledger::commit, Ledger::release, false, true); };
    auto old_host = SharedCacheCharge::reserve(0, 3, 32);
    require(old_host && old_host->commit(32) && !host_attach(), "host scope adopted a live uncharged arena");
    require(attach(ledger) && SharedCacheCharge::detach(&ledger), "legacy scope unexpectedly covers host storage");
    old_host.reset();
    require(host_attach(), "host scope could not attach after arena release");
    auto gpu = SharedCacheCharge::reserve(1, 1, 64);
    auto host = SharedCacheCharge::reserve(0, 3, 64);
    require(gpu && host && !SharedCacheCharge::reserve(0, 3, 1), "host staging escaped the shared RAM limit");
    require(gpu->commit(64) && host->commit(64), "host/device shared commits failed");
    require(ledger.committed[0] == 128 && ledger.committed[1] == 0 && ledger.committed[2] == 64,
        "host staging was charged to a GPU pool");
    gpu.reset();
    require(!SharedCacheCharge::detach(&ledger), "host arena did not retain callback ownership");
    host.reset();
    require(ledger.allocations.empty() && SharedCacheCharge::detach(&ledger), "host release leaked shared credit");
    auto old_ram_cache = SharedCacheCharge::reserve(0, 6, 32);
    require(old_ram_cache && !host_attach(), "host scope adopted pending RAM cache");
    require(old_ram_cache->commit(32) && !host_attach(), "host scope adopted resident RAM cache");
    require(attach(ledger) && SharedCacheCharge::detach(&ledger), "legacy scope unexpectedly covers RAM cache");
    old_ram_cache.reset();
    require(host_attach(), "cannot attach after RAM cache release");
    auto ram_cache = SharedCacheCharge::reserve(0, 6, 96);
    require(ram_cache && ledger.last_kind == 6 && ram_cache->is_accounted(), "RAM cache dedicated charge missing");
    require(!SharedCacheCharge::reserve(0, 3, 33), "RAM cache and staging escaped shared host quota");
    require(!SharedCacheCharge::detach(&ledger), "detached pending RAM cache");
    require(ram_cache->commit(96) && ledger.committed[0] == 96 && ledger.committed[1] == 0,
        "RAM cache charged GPU instead of host");
    require(!SharedCacheCharge::detach(&ledger), "detached resident RAM cache");
    ram_cache.reset();
    require(SharedCacheCharge::detach(&ledger) && ledger.allocations.empty(), "RAM cache leaked charge");
    auto graph_attach = [&] { return SharedCacheCharge::attach(&ledger, Ledger::reserve,
        Ledger::commit, Ledger::release, true); };
    auto old_joint = SharedCacheCharge::reserve(0, 5, 32);
    require(old_joint && !graph_attach(), "joint graph scope adopted an uncharged live arena");
    require(attach(ledger) && SharedCacheCharge::detach(&ledger), "legacy graph exclusion changed for joint arenas");
    old_joint.reset();
    require(graph_attach(), "joint graph scope attach failed");
    auto joint = SharedCacheCharge::reserve(0, 5, 32);
    require(joint && joint->is_accounted() && ledger.last_kind == 5 && joint->commit(32),
        "joint graph callback kind/commit missing");
    require(!SharedCacheCharge::detach(&ledger), "live joint graph did not protect callback lifetime");
    joint.reset();
    require(SharedCacheCharge::detach(&ledger) && ledger.allocations.empty(), "joint graph credit leaked");
    require(!SharedCacheCharge::reserve(0, 7, 32), "unconfigured speculation received an unaccounted owner");
    require(attach(ledger), "legacy scope attach for speculation test failed");
    require(!SharedCacheCharge::reserve(0, 7, 32), "legacy graph-excluded scope admitted speculation");
    require(SharedCacheCharge::detach(&ledger), "legacy speculation refusal leaked owner");
    require(graph_attach(), "speculation scope attach failed");
    auto speculation = SharedCacheCharge::reserve(0, 7, 64);
    require(speculation && speculation->is_accounted() && ledger.last_kind == 7,
        "speculation callback missed dedicated accounted owner");
    require(!SharedCacheCharge::reserve(0, 7, 1), "speculation exceeded device envelope");
    require(!SharedCacheCharge::detach(&ledger), "pending speculation lost callback ownership");
    require(speculation->commit(64) && !SharedCacheCharge::detach(&ledger), "committed speculation lost callback ownership");
    speculation.reset();
    require(SharedCacheCharge::detach(&ledger) && ledger.allocations.empty(), "speculation owner leaked credit");
    require(!SharedCacheCharge::checked_release_attached(), "detached registry advertises checked release");
    require(host_attach(), "legacy host attach for checked refusal failed");
    auto legacy = SharedCacheCharge::reserve(0, 3, 16);
    require(legacy && !legacy->release_checked() && !SharedCacheCharge::checked_release_attached()
        && !SharedCacheCharge::detach(&ledger), "void callback certified checked release or lost ownership");
    legacy.reset();
    require(SharedCacheCharge::detach(&ledger), "legacy checked refusal leaked registry");
    require(SharedCacheCharge::attach(&ledger, Ledger::reserve, Ledger::commit, Ledger::release,
        true, true, Ledger::release_checked) && SharedCacheCharge::checked_release_attached(),
        "cannot install explicit checked-release acknowledgment");
    for (bool committed : {false, true}) {
        auto checked = SharedCacheCharge::reserve(1, 4, 32);
        require(checked && checked->is_accounted() && checked->rank() == 1 && checked->kind() == 4
            && checked->reserved_bytes() == 32 && !checked->is_committed(), "immutable pending certificate changed");
        if (committed) require(checked->commit(32), "checked owner commit failed");
        require(checked->is_committed() == committed, "certificate did not reflect actual commit");
        ledger.fail_checked_release = true;
        require(!checked->release_checked() && ledger.allocations.size() == 1
            && !SharedCacheCharge::detach(&ledger), "failed checked release lost token or detach protection");
        ledger.fail_checked_release = false;
        require(checked->release_checked() && ledger.allocations.empty() && !checked->is_committed()
            && !checked->commit(32), "checked release did not end exactly this ownership");
        const auto calls = ledger.checked_release_calls;
        require(checked->release_checked() && ledger.checked_release_calls == calls,
            "idempotent checked release called accounting again");
        checked.reset();
        require(ledger.checked_release_calls == calls, "destructor duplicated an acknowledged refund");
    }
    auto delayed_ack = SharedCacheCharge::reserve(0, 3, 48);
    require(delayed_ack && delayed_ack->commit(48), "cannot create checked Host owner");
    ledger.fail_release_ack = true;
    require(!delayed_ack->release_checked() && ledger.allocations.empty()
        && ledger.released_waiting_ack.size() == 1 && !SharedCacheCharge::detach(&ledger),
        "post-ledger failure forgot exact ticket or prematurely detached");
    ledger.fail_release_ack = false;
    require(delayed_ack->release_checked() && ledger.released_waiting_ack.empty()
        && ledger.reserved[0] == 0 && ledger.committed[0] == 0, "ack retry duplicated physical credit");
    delayed_ack.reset();
    require(SharedCacheCharge::detach(&ledger) && !SharedCacheCharge::checked_release_attached(),
        "acknowledged cleanup retained callback registration");
    std::puts("PASS shared cache bridge: expert kind4, joint graph kind5, RAM kind6, admitted speculation kind7, atomic attach/detach, multi-pool/rank quota, physical-free order, rollback and concurrency");
}
