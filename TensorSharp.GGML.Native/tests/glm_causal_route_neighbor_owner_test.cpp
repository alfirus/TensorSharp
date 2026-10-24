// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Real checked SharedCacheCharge callbacks around the portable Host owner.
// Pure CPU: no cache enrollment, CUDA resource, OS allocator or model proof.
#include "../glm_causal_route_neighbor_owner.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#if !defined(TSG_GGML_TEST_HOOKS)
#error This fixture requires the bounded owner allocation/fault hooks.
#endif

namespace {
using namespace tsg_glm_neighbor;
std::size_t checks = 0, cases = 0;
void require(bool value, const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
using Info = std::array<std::int64_t, 16>;
Info info(const Owner& owner) {
    Info value{};
    require(owner.info(value.data(), static_cast<int>(value.size())), "Owner Info16 failed");
    return value;
}

struct Ledger {
    struct Ticket {
        std::uint64_t id = 0;
        std::size_t bytes = 0;
        bool committed = false, refunded = false;
    };
    std::array<Ticket, 8> tickets{};
    Owner* owner = nullptr;
    std::size_t pending = 0, committed = 0;
    std::uint64_t serial = 0, reserves = 0, commits = 0, checked = 0, legacy = 0, refunds = 0;
    std::array<std::uint64_t, 16> checked_ids{};
    bool valid = true, reject_reserve = false, reject_commit = false;
    bool fail_before_refund = false, fail_after_refund = false;
    Ticket* find(std::uint64_t id) noexcept {
        for (auto& value : tickets) if (value.id == id) return &value;
        return nullptr;
    }
    std::size_t held() const noexcept {
        std::size_t count = 0; for (const auto& value : tickets) count += value.id != 0; return count;
    }
    std::uint64_t only_id() const noexcept {
        std::uint64_t result = 0; for (const auto& value : tickets) if (value.id) result = value.id; return result;
    }
    static std::uint64_t reserve(void* context, int rank, int kind, std::int64_t bytes) noexcept {
        auto& self = *static_cast<Ledger*>(context); ++self.reserves;
        // Observes the actual owner at the accounting boundary. There is no
        // registry/cache call from inside a serialized accounting callback.
        if (rank != 0 || kind != 3 || bytes != static_cast<std::int64_t>(Owner::host_bytes())
            || !self.owner || self.owner->storage()) { self.valid = false; return 0; }
        if (self.reject_reserve) return 0;
        for (auto& value : self.tickets) if (!value.id) {
            value = {++self.serial, static_cast<std::size_t>(bytes), false, false};
            self.pending += value.bytes; return value.id;
        }
        self.valid = false; return 0;
    }
    static int commit(void* context, std::uint64_t id) noexcept {
        auto& self = *static_cast<Ledger*>(context); ++self.commits;
        auto* ticket = self.find(id);
        if (!ticket || ticket->committed || ticket->refunded || !self.owner || !self.owner->storage()) {
            self.valid = false; return 0;
        }
        if (self.reject_commit) return 0;
        if (self.pending < ticket->bytes) { self.valid = false; return 0; }
        self.pending -= ticket->bytes; self.committed += ticket->bytes; ticket->committed = true; return 1;
    }
    static int release_checked(void* context, std::uint64_t id) noexcept {
        auto& self = *static_cast<Ledger*>(context);
        if (self.checked < self.checked_ids.size()) self.checked_ids[self.checked] = id;
        ++self.checked;
        auto* ticket = self.find(id);
        if (!ticket || !self.owner || self.owner->storage()) { self.valid = false; return 0; }
        if (self.fail_before_refund) { self.fail_before_refund = false; return 0; }
        if (!ticket->refunded) {
            auto& held = ticket->committed ? self.committed : self.pending;
            if (held < ticket->bytes) { self.valid = false; return 0; }
            held -= ticket->bytes; ticket->refunded = true; ++self.refunds;
            if (self.fail_after_refund) { self.fail_after_refund = false; return 0; }
        }
        *ticket = {}; return 1;
    }
    static void release(void* context, std::uint64_t) noexcept {
        auto& self = *static_cast<Ledger*>(context); ++self.legacy; self.valid = false;
    }
};

struct Fixture {
    Ledger ledger; // Must outlive both the owner and its callback registry.
    Owner owner;
    Config config{};
    std::array<Source, 3> sources{};
    std::array<std::array<unsigned char, 3>, 3> source_tags{};
    unsigned char model = 0;
    bool attached = false;
    explicit Fixture(bool host = true, bool checked = true) {
        config.layers = 3; config.experts = 12; config.routes = 3;
        for (std::size_t layer = 0; layer < config.layers; ++layer) {
            config.layer_ids[layer] = static_cast<std::uint16_t>(3 + layer);
            sources[layer].layer = config.layer_ids[layer];
            for (std::size_t p = 0; p < 3; ++p) {
                sources[layer].pointer[p] = &source_tags[layer][p];
                sources[layer].stride[p] = 7 + p + layer;
                sources[layer].bytes[p] = sources[layer].stride[p] * config.experts;
                sources[layer].type[p] = static_cast<int>(16 + p);
            }
        }
        ledger.owner = &owner; owner.request(true);
        attached = tsg::SharedCacheCharge::attach(&ledger, Ledger::reserve, Ledger::commit,
            Ledger::release, false, host, checked ? Ledger::release_checked : nullptr);
        require(attached, "Cannot attach the isolated real SharedCacheCharge registry");
    }
    ~Fixture() noexcept {
        ledger.reject_reserve = ledger.reject_commit = ledger.fail_before_refund = ledger.fail_after_refund = false;
        if (owner.active_call() || owner.query_pending()) owner.fail_call();
        if (!owner.close() || !owner.release_ready() || !tsg::SharedCacheCharge::detach(&ledger)) {
            std::fputs("FATAL owner fixture cannot release exact retained callback ownership\n", stderr);
            std::abort();
        }
    }
    bool configure() { return owner.configure(&model, 73, 1, Owner::host_bytes(), config, sources.data()); }
    void clean() {
        require(owner.close() && owner.release_ready(), "Checked owner close failed");
        require(ledger.valid && !ledger.held() && !ledger.pending && !ledger.committed && !ledger.legacy,
            "Ledger retained an owner or used legacy void release");
    }
    void no_charge() const {
        require(!ledger.reserves && !ledger.commits && !ledger.checked && !ledger.held()
            && !ledger.pending && !ledger.committed && !owner.storage(),
            "Preflight refusal allocated or reserved a Host owner");
    }
    void call(std::int64_t head, std::int64_t sequence = 17) {
        require(owner.begin_call(true, head, 1, sequence), "Cannot begin a valid scalar call");
        for (std::size_t rank = 0; rank < config.layers; ++rank) {
            const std::int32_t ids[3] = {static_cast<std::int32_t>(rank), static_cast<std::int32_t>(rank + 1),
                static_cast<std::int32_t>(rank)};
            require(owner.prepare_query(rank, ids, 3), "Cannot prepare the complete source-qualified route");
            require(owner.score_slot(0, sources[rank].pointer, ids[0])
                && owner.score_slot(1, sources[rank].pointer, -1), "Cannot score resident and empty slots");
            require(owner.prefer_slot(1, -1, 99, 0, ids[0], 1), "Empty slot did not precede occupied slot");
            require(owner.commit_query(rank), "Cannot commit after simulated successful helper completion");
        }
        require(owner.finish_call(), "Cannot finish all completed own-layer queries");
    }
};

void capability_preflight() {
    for (int missing = 0; missing < 2; ++missing) {
        Fixture f(missing != 0, missing == 0);
        require(!f.configure() && !f.owner.configured() && f.owner.release_ready(),
            "Missing Host or checked-release capability was accepted");
        f.no_charge();
    }
}
void invalid_configuration() {
    Fixture f;
    const auto before = info(f.owner);
    require(!f.owner.configure(nullptr, 73, 1, Owner::host_bytes(), f.config, f.sources.data()), "NULL model accepted");
    require(!f.owner.configure(&f.model, 0, 1, Owner::host_bytes(), f.config, f.sources.data()), "Zero scope accepted");
    require(!f.owner.configure(&f.model, 73, 2, Owner::host_bytes(), f.config, f.sources.data()), "Wrong profile accepted");
    require(!f.owner.configure(&f.model, 73, 1, Owner::host_bytes() - 1, f.config, f.sources.data()), "Wrong byte plan accepted");
    require(!f.owner.configure(&f.model, 73, 1, Owner::host_bytes(), f.config, nullptr), "NULL source vector accepted");
    auto bad = f.sources;
    bad[1].pointer = bad[0].pointer;
    require(!f.owner.configure(&f.model, 73, 1, Owner::host_bytes(), f.config, bad.data()), "Duplicate source generation accepted");
    bad = f.sources; bad[1].bytes[2]--;
    require(!f.owner.configure(&f.model, 73, 1, Owner::host_bytes(), f.config, bad.data()), "Truncated projection accepted");
    bad = f.sources; bad[0].stride[0] = UINT64_MAX;
    require(!f.owner.configure(&f.model, 73, 1, Owner::host_bytes(), f.config, bad.data()), "Stride multiplication overflow accepted");
    bad = f.sources; bad[2].layer++;
    require(!f.owner.configure(&f.model, 73, 1, Owner::host_bytes(), f.config, bad.data()), "Wrong layer identity accepted");
    require(info(f.owner) == before && f.owner.scope() == 0 && !f.owner.model(), "Early refusal adopted a foreign scope");
    f.no_charge(); require(f.configure(), "Valid setup could not follow unchanged preflight refusals"); f.clean();
}
void immutable_single_charge() {
    Fixture f; require(f.configure(), "Initial configure failed");
    auto* slab = f.owner.storage(); const auto token = f.ledger.only_id(); const auto before = info(f.owner);
    require(slab && f.ledger.reserves == 1 && f.ledger.commits == 1 && f.ledger.pending == 0
        && f.ledger.committed == Owner::host_bytes() && f.ledger.held() == 1, "Actual initial ticket is not exact committed kind3");
    require(f.configure() && info(f.owner) == before && f.owner.storage() == slab && f.ledger.only_id() == token,
        "Idempotent configure recreated the slab/ticket");
    auto changed = f.sources; unsigned char replacement = 0; changed[0].pointer[1] = &replacement;
    require(!f.owner.configure(&f.model, 73, 1, Owner::host_bytes(), f.config, changed.data()), "Changed source generation accepted");
    require(!f.owner.configure(&f.model, 74, 1, Owner::host_bytes(), f.config, f.sources.data()), "Changed scope accepted");
    require(!f.owner.configure(&replacement, 73, 1, Owner::host_bytes(), f.config, f.sources.data()), "Changed model accepted");
    for (std::size_t rank = 0; rank < f.config.layers; ++rank)
        require(f.owner.find_rank(f.sources[rank].pointer) == static_cast<int>(rank), "Exact source triple has wrong rank");
    require(f.owner.find_rank(changed[0].pointer) == -1 && info(f.owner) == before && f.ledger.reserves == 1,
        "Refused replacement changed live identity or accounting");
    require(!tsg::SharedCacheCharge::detach(&f.ledger), "Registry detached while the exact owner ticket was live"); f.clean();
}
void reserve_failure() {
    Fixture f; f.ledger.reject_reserve = true;
    require(!f.configure(), "Injected real reserve refusal succeeded"); const auto value = info(f.owner);
    require(value[5] == 1 && value[7] == 0 && value[8] == 0 && f.ledger.reserves == 1
        && !f.ledger.commits && !f.ledger.held(), "Reserve refusal fabricated a slab/charge");
    require(!f.configure() && f.ledger.reserves == 1, "Poisoned setup silently retried reserve"); f.clean();
}
void allocation_failure() {
    Fixture f; f.owner.test_fail_allocation(); require(!f.configure(), "Injected post-reserve slab refusal succeeded");
    const auto value = info(f.owner);
    require(value[5] == 1 && value[7] == 0 && value[8] == static_cast<std::int64_t>(Owner::host_bytes())
        && f.ledger.pending == Owner::host_bytes() && !f.ledger.committed && f.ledger.held() == 1
        && f.ledger.reserves == 1 && !f.ledger.commits, "Allocation refusal did not retain its original pending ticket");
    require(!tsg::SharedCacheCharge::detach(&f.ledger), "Pending allocation ticket lost registry ownership"); f.clean();
}
void commit_failure() {
    Fixture f; f.ledger.reject_commit = true; require(!f.configure(), "Injected real commit refusal succeeded");
    const auto value = info(f.owner);
    require(value[5] == 1 && value[7] == static_cast<std::int64_t>(Owner::host_bytes()) && value[8] == value[7]
        && f.ledger.pending == Owner::host_bytes() && !f.ledger.committed && f.ledger.reserves == 1
        && f.ledger.commits == 1 && f.owner.storage(), "Commit refusal prematurely freed the slab/pending ticket");
    f.clean(); require(f.ledger.refunds == 1, "Pending commit refusal was credited more than once");
}
void acknowledgment_retry(bool after_refund) {
    Fixture f; require(f.configure(), "ACK case configure failed"); const auto id = f.ledger.only_id();
    f.ledger.fail_before_refund = !after_refund; f.ledger.fail_after_refund = after_refund;
    require(!f.owner.close(), "Injected checked ACK refusal was hidden"); const auto value = info(f.owner);
    require(value[3] == 1 && value[4] == 0 && value[7] == 0 && value[8] == static_cast<std::int64_t>(Owner::host_bytes())
        && f.ledger.held() == 1 && f.ledger.only_id() == id && f.ledger.checked == 1 && !f.owner.ready()
        && !f.owner.storage(), "ACK refusal did not retain only the exact original ticket");
    require(f.ledger.committed == (after_refund ? 0 : Owner::host_bytes())
        && f.ledger.refunds == static_cast<std::uint64_t>(after_refund), "Native ticket ownership was confused with refunded ledger bytes");
    require(!tsg::SharedCacheCharge::detach(&f.ledger), "Unacknowledged exact token allowed callback detach");
    require(!f.owner.begin_call(true) && !f.configure(), "Closing ACK owner resumed inference or setup");
    f.clean(); require(f.ledger.checked == 2 && f.ledger.checked_ids[0] == id && f.ledger.checked_ids[1] == id
        && f.ledger.refunds == 1 && f.ledger.reserves == 1, "ACK retry substituted a ticket or refunded twice");
    require(f.owner.close() && f.ledger.checked == 2, "Idempotent final close repeated release");
}
void completed_calls_and_resets() {
    Fixture f; require(f.configure(), "History case configure failed"); auto* slab = f.owner.storage();
    const auto token = f.ledger.only_id();
    require(f.owner.maintenance_reset() && info(f.owner)[12] == 1, "Empty maintenance reset invented a history epoch");
    for (int head = 0; head < 80; ++head) f.call(head);
    auto value = info(f.owner);
    require(value[10] == 80 && value[11] == 240 && value[14] > 0 && value[14] + value[15] == 240
        && f.owner.history_max() == 64 && f.owner.call_stats().committed == 3,
        "Completed calls did not preserve bounded own-layer history and cumulative attribution");
    const auto epoch = value[12]; f.call(400, 19);
    require(info(f.owner)[12] > epoch && f.owner.history_max() == 1 && info(f.owner)[10] == 81,
        "Discontinuous head/sequence reused old-topic history or erased cumulative counts");
    require(f.owner.begin_call(false, 401, 9, 19) && !f.owner.prepare_query(0, nullptr, 0)
        && f.owner.finish_call(), "Wide reset call learned a scalar label");
    value = info(f.owner);
    require(value[10] == 81 && value[11] == 243 && f.owner.history_max() == 0 && !value[9], "Wide call changed completed-token counters");
    require(f.owner.storage() == slab && f.ledger.only_id() == token && f.ledger.reserves == 1
        && f.ledger.commits == 1 && !f.ledger.checked && !f.ledger.pending,
        "History/reset hot path changed physical Host ownership or created another ticket"); f.clean();
}
void pending_failure_and_recovery() {
    Fixture f; require(f.configure(), "Recovery case configure failed"); f.call(0);
    require(f.owner.begin_call(true, 1, 1, 17), "Cannot begin pending call");
    const std::int32_t ids[3] = {0, 1, 0};
    require(f.owner.prepare_query(0, ids, 3) && info(f.owner)[9] == 1, "Pending query was not observable");
    require(!f.owner.finish_call(), "Partial layer prefix was certified completed");
    f.owner.fail_call(); const auto failed = info(f.owner);
    require(failed[5] == 1 && !failed[9] && failed[10] == 1 && failed[11] == 3
        && !f.owner.begin_call(true, 1) && f.owner.history_max() == 0, "Failed prepared query became a completed label");
    require(f.owner.recover_reset() && f.owner.ready(), "Checked public reset could not recover quiescent history owner");
    f.call(0, 88); const auto recovered = info(f.owner);
    require(!recovered[5] && recovered[10] == 2 && recovered[11] == 6 && recovered[12] > failed[12]
        && f.ledger.reserves == 1, "Reset recreated storage or erased cumulative observations"); f.clean();
}
void partial_token_maintenance() {
    Fixture f; require(f.configure(), "Partial maintenance case configure failed");
    for (int head = 0; head < 10; ++head) f.call(head);
    const auto before = info(f.owner); auto* slab = f.owner.storage();
    require(f.owner.begin_call(true, 10, 1, 17), "Cannot begin the matched partial token");
    const std::int32_t first[3] = {0, 1, 0};
    require(f.owner.prepare_query(0, first, 3) && f.owner.call_stats().matches == 1
        && f.owner.commit_query(0), "No actual first-rank matched/committed query before maintenance");
    const auto compared = f.owner.call_stats().compared;
    require(compared > 0 && f.owner.maintenance_reset(), "Active partial history could not be reset");
    for (std::size_t rank = 1; rank < f.config.layers; ++rank) {
        const std::int32_t ids[3] = {static_cast<std::int32_t>(rank), static_cast<std::int32_t>(rank + 1),
            static_cast<std::int32_t>(rank)};
        require(f.owner.prepare_query(rank, ids, 3) && f.owner.commit_query(rank), "Nonlearning remaining layer failed");
    }
    require(f.owner.finish_call(), "Nonlearning partial token could not finish actual commits");
    const auto after = info(f.owner); const auto& call = f.owner.call_stats();
    require(call.queries == f.config.layers && call.committed == f.config.layers && !call.matches
        && call.no_matches == f.config.layers && !call.min_lag && !call.history_max && call.compared == compared
        && !f.owner.history_max() && after[10] == before[10] + 1 && after[11] == before[11] + 3
        && after[14] == before[14] && after[15] == before[15] + 3 && after[12] > before[12],
        "Maintenance granted historical match credit or erased actual committed work");
    f.call(11);
    require(f.owner.history_max() == 1 && f.owner.call_stats().no_matches == 3
        && !f.owner.call_stats().compared && f.owner.storage() == slab && f.ledger.reserves == 1,
        "Next token did not start cold in the same retained owner"); f.clean();
}
void exhausted_epoch_abort() {
    Fixture f; require(f.configure() && f.owner.begin_call(true), "Cannot begin exhausted-epoch case");
    f.owner.storage()->policy.epoch = UINT64_MAX;
    const std::int32_t ids[3] = {0, 1, 0};
    require(f.owner.prepare_query(0, ids, 3) && f.owner.query_pending(), "Exhausted-epoch fixture did not prepare a real pending row");
    f.owner.fail_call(); const auto value = info(f.owner);
    require(value[5] == 1 && !value[9] && !value[10] && !value[11] && !f.owner.history_max()
        && !f.owner.storage()->policy.pending.prepared && !f.owner.storage()->policy.active
        && !f.owner.recover_reset() && !f.owner.ready(),
        "Terminal epoch overflow retained a publishable pending label or silently recovered");
    require(f.ledger.held() == 1 && f.ledger.committed == Owner::host_bytes(), "Terminal policy failure lost its physical/ticket owner"); f.clean();
}
void busy_close_and_fault_seams() {
    {
        Fixture f; require(f.configure(), "Busy-close case configure failed");
        require(f.owner.begin_call(true), "Cannot begin actual active call");
        require(!f.owner.close() && info(f.owner)[3] == 1 && f.owner.storage() && f.ledger.checked == 0,
            "Busy close retired an active slab or acknowledged it early");
        require(!f.owner.ready() && !f.owner.begin_call(true) && !f.owner.recover_reset(), "Closing owner resumed work");
        f.owner.fail_call(); f.clean();
    }
    for (int stage = 1; stage <= 3; ++stage) {
        Fixture hooks; require(hooks.configure(), "Hook case configure failed");
        const auto before = hooks.owner.test_failure_count();
        require(!hooks.owner.test_fault(-1) && !hooks.owner.test_fault(4) && hooks.owner.test_fault(stage), "Hook bounds wrong");
        require(!hooks.owner.test_fire(0) && !hooks.owner.test_fire(stage % 3 + 1), "Unarmed hook unexpectedly fired");
        require(hooks.owner.test_fire(stage) && !hooks.owner.test_fire(stage)
            && hooks.owner.test_failure_count() == before + 1, "Bounded one-shot hook was not consumed exactly once");
        // This tests the hook mechanism only, not the real cache/GLM fault cut.
        hooks.owner.fail_call(); require(hooks.owner.recover_reset(), "Hook owner could not explicitly recover"); hooks.clean();
    }
}
template<class F> void run(const char* name, F body) {
    body(); ++cases; std::printf("PASS owner_case=%s\n", name);
}
} // namespace

int main() {
    try {
        run("capability_preflight", capability_preflight);
        run("invalid_configuration", invalid_configuration);
        run("immutable_single_charge", immutable_single_charge);
        run("reserve_failure", reserve_failure);
        run("post_reserve_allocation_failure", allocation_failure);
        run("actual_commit_failure", commit_failure);
        run("pre_ledger_ack_retry", [] { acknowledgment_retry(false); });
        run("post_ledger_ack_retry", [] { acknowledgment_retry(true); });
        run("completed_calls_and_resets", completed_calls_and_resets);
        run("pending_failure_and_recovery", pending_failure_and_recovery);
        run("partial_token_maintenance", partial_token_maintenance);
        run("exhausted_epoch_abort", exhausted_epoch_abort);
        run("busy_close_and_hook_mechanism", busy_close_and_fault_seams);
        std::printf("PASS causal_route_neighbor_owner checks=%zu cases=%zu host_bytes=%zu device_bytes=0 legacy_release_calls=0 scope=pure_cpu_owner_checked_accounting_not_cache_or_model\n",
            checks, cases, Owner::host_bytes());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL causal_route_neighbor_owner: %s checks=%zu cases=%zu\n", error.what(), checks, cases);
        return 1;
    }
}
