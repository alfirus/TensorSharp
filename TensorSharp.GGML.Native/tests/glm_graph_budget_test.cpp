// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Strict device-type identity and actual CPU scheduler allocation. CUDA model
// qualification is separate: this fixture cannot substitute for a device run.
#define TSG_GGML_TEST_HOOKS 1
#include "../glm_graph_budget.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <map>
#include <limits>

namespace {
int checks = 0;
void require(bool value, const char* message) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL %s\n", message); std::exit(1); }
}
struct Ledger {
    std::int64_t used = 0, ceiling = 1 << 20;
    std::uint64_t next = 0;
    bool freed = false;
    int expected_kind = 5;
    std::map<std::uint64_t, std::int64_t> tickets;
    static std::uint64_t reserve(void* context, int rank, int kind, std::int64_t bytes) {
        auto& l = *static_cast<Ledger*>(context);
        require(rank == 0 && kind == l.expected_kind, "exact graph rank and shared owner");
        if (bytes > l.ceiling - l.used) return 0;
        l.used += bytes; l.tickets[++l.next] = bytes; return l.next;
    }
    static int commit(void* context, std::uint64_t token) {
        return static_cast<Ledger*>(context)->tickets.count(token) ? 1 : 0;
    }
    static void release(void* context, std::uint64_t token) {
        auto& l = *static_cast<Ledger*>(context);
        require(l.freed, "physical scheduler free before graph credit release");
        l.used -= l.tickets.at(token); l.tickets.erase(token);
    }
};
struct StrictBackend {
    ggml_backend_t cpu;
    ggml_backend_buffer_type_t type;
    ggml_backend_device device{};
    ggml_backend backend{};
    int dispatched = 0, syncs = 0, frees = 0, copies = 0;
    static StrictBackend& dev(ggml_backend_dev_t d) { return *static_cast<StrictBackend*>(d->context); }
    static StrictBackend& stream(ggml_backend_t b) {
        auto& fixture = *static_cast<StrictBackend*>(b->context);
        require(b == &fixture.backend, "vendor dispatch receives its original backend pointer");
        return fixture;
    }
    static const char* name(ggml_backend_dev_t) { return "strict-CPU"; }
    static void memory(ggml_backend_dev_t, size_t* free, size_t* total) { *free = *total = 1 << 20; }
    static enum ggml_backend_dev_type device_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_CPU; }
    static void props(ggml_backend_dev_t d, struct ggml_backend_dev_props* p) {
        ggml_backend_dev_get_props(ggml_backend_get_device(dev(d).cpu), p);
    }
    static ggml_backend_t init(ggml_backend_dev_t, const char*) { return ggml_backend_cpu_init(); }
    static ggml_backend_buffer_type_t buffer_type(ggml_backend_dev_t d) { return dev(d).type; }
    static bool supports_op(ggml_backend_dev_t d, const ggml_tensor* t) { return ggml_backend_supports_op(dev(d).cpu, t); }
    static bool supports_type(ggml_backend_dev_t d, ggml_backend_buffer_type_t t) { return t == dev(d).type; }
    static const char* stream_name(ggml_backend_t b) { stream(b); return "strict-CPU"; }
    static void free_stream(ggml_backend_t b) { ++stream(b).frees; }
    static void sync(ggml_backend_t b) { auto& f = stream(b); ++f.syncs; ggml_backend_synchronize(f.cpu); }
    static enum ggml_status compute(ggml_backend_t b, ggml_cgraph* g) {
        auto& f = stream(b); ++f.dispatched; return ggml_backend_graph_compute_async(f.cpu, g);
    }
    static bool copy(ggml_backend_t src, ggml_backend_t dst, const ggml_tensor*, ggml_tensor*) {
        stream(src); ++stream(dst).copies; return true;
    }
    StrictBackend() : cpu(ggml_backend_cpu_init()), type(ggml_backend_cpu_buffer_type()) {
        require(cpu != nullptr, "actual CPU backend available");
        ggml_backend_cpu_set_n_threads(cpu, 1);
        device.iface = { name, name, memory, device_type, props, init, buffer_type,
            nullptr, nullptr, supports_op, supports_type, nullptr, nullptr, nullptr, nullptr };
        device.context = this;
        backend.guid = cpu->guid; backend.device = &device; backend.context = this;
        backend.iface.get_name = stream_name; backend.iface.free = free_stream;
        backend.iface.synchronize = sync; backend.iface.graph_compute = compute;
        backend.iface.cpy_tensor_async = copy;
    }
    ~StrictBackend() { ggml_backend_free(cpu); }
};

// This allocator only simulates a nonhost GPU type. It never initializes CUDA
// or performs device work; real CUDA acceptance is a separate qualification.
struct CapLedger {
    std::int64_t charged = 0, physical = 0;
    std::uint64_t next = 0;
    int reserves = 0, commits = 0, releases = 0;
    bool refuse_reserve = false, refuse_commit = false, throw_commit = false;
    std::map<std::uint64_t, std::int64_t> tickets;
    tsg_glm::graph_budget* budget = nullptr;
    static std::uint64_t reserve(void* p, int rank, int kind, std::int64_t bytes) {
        auto& l = *static_cast<CapLedger*>(p); ++l.reserves;
        require(rank == 0 && kind == 5, "strict cap requires the exact device/kind ticket");
        if (l.refuse_reserve) return 0;
        l.charged += bytes; l.tickets[++l.next] = bytes; return l.next;
    }
    static int commit(void* p, std::uint64_t token) {
        auto& l = *static_cast<CapLedger*>(p); ++l.commits;
        require(l.tickets.count(token) == 1, "commit retains exact strict ticket");
        if (l.throw_commit) throw std::runtime_error("injected callback exception");
        return !l.refuse_commit;
    }
    static void release(void* p, std::uint64_t token) {
        auto& l = *static_cast<CapLedger*>(p); ++l.releases;
        const auto bytes = l.tickets.at(token);
        require(l.physical <= l.charged - bytes, "strict physical free precedes ticket refund");
        require(!l.budget || l.budget->device_payload_bytes() >= static_cast<std::size_t>(l.charged),
            "strict pending cap refund follows physical free and ticket refund");
        l.charged -= bytes; l.tickets.erase(token);
    }
    void attach(bool graphs = true) {
        require(tsg::SharedCacheCharge::attach(this, reserve, commit, release, graphs), "attach strict mock ledger");
    }
    void detach() {
        require(!charged && !physical && tickets.empty(), "strict failures and teardown retain no owners");
        require(tsg::SharedCacheCharge::detach(this), "detach strict mock ledger");
    }
};
struct CapBackend {
    ggml_backend_buffer_type type{}, host_type{}, foreign_type{};
    ggml_backend_device device{};
    ggml_backend backend{};
    CapLedger& ledger;
    int calls = 0, frees = 0;
    int actual_extra = 0;
    bool refuse_allocation = false, throw_allocation = false, wrong_type = false;
    static CapBackend& self(ggml_backend_buffer_type_t t) { return *static_cast<CapBackend*>(t->context); }
    static const char* name(ggml_backend_buffer_type_t) { return "mock strict payload"; }
    static const char* device_name(ggml_backend_dev_t) { return "mock GPU; no device work"; }
    static enum ggml_backend_dev_type device_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; }
    static ggml_backend_buffer_type_t default_type(ggml_backend_dev_t d) { return &static_cast<CapBackend*>(d->context)->type; }
    static size_t alignment(ggml_backend_buffer_type_t) { return 64; }
    static bool is_host(ggml_backend_buffer_type_t t) { return t == &self(t).host_type; }
    static void free_buffer(ggml_backend_buffer_t b) {
        auto& f = *static_cast<CapBackend*>(b->context); ++f.frees;
        require(!f.ledger.budget || f.ledger.budget->device_payload_bytes() >= b->size,
            "physical free still owns the strict cap reservation");
        f.ledger.physical -= static_cast<std::int64_t>(b->size);
    }
    static ggml_backend_buffer_t alloc(ggml_backend_buffer_type_t t, size_t bytes) {
        auto& f = self(t); ++f.calls;
        if (f.throw_allocation) throw std::runtime_error("injected physical allocator exception");
        if (f.refuse_allocation) return nullptr;
        const auto actual = static_cast<std::size_t>(static_cast<std::int64_t>(bytes) + f.actual_extra);
        ggml_backend_buffer_i iface{}; iface.free_buffer = free_buffer;
        f.ledger.physical += static_cast<std::int64_t>(actual);
        return ggml_backend_buffer_init(f.wrong_type ? &f.foreign_type : t, iface, &f, actual);
    }
    static ggml_backend_buffer_t alloc_n(ggml_backend_buffer_type_t, ggml_tensor**, int) { return nullptr; }
    static size_t size_n(ggml_backend_buffer_type_t, ggml_tensor**, int) { return 64; }
    explicit CapBackend(CapLedger& l) : ledger(l) {
        device.context = this; device.iface.get_name = device_name;
        device.iface.get_type = device_type; device.iface.get_buffer_type = default_type;
        type.iface = {name, alloc, nullptr, alignment, nullptr, nullptr, nullptr, is_host};
        type.context = this; type.device = &device;
        host_type = foreign_type = type;
        backend.device = &device; backend.context = this;
    }
};
void strict_cap_checks() {
    // The owned projection facade has its own device record but allocates the
    // exact original CUDA type. Device-address equality would reject every real
    // strict call before allocation; accepting a different physical type is
    // still forbidden. This reproduces that distinction without CUDA.
    {
        CapLedger l; l.attach(); CapBackend f(l);
        auto facade_device = f.device;
        auto facade_backend = f.backend; facade_backend.device = &facade_device;
        ggml_backend_buffer_type_t types[]{&f.type};
        ggml_backend_t backends[]{&facade_backend};
        {
            tsg_glm::graph_budget b; l.budget = &b;
            require(types[0]->device != ggml_backend_get_device(backends[0])
                && ggml_backend_get_default_buffer_type(backends[0]) == types[0],
                "known facade has a distinct device and exact physical default type");
            auto p = b.prepare(types, backends, 1, 1, 5, 64, &f.type);
            require(p && !b.policy_refused(), "strict scope accepts the exact type through a known device facade");
            auto buffer = ggml_backend_buft_alloc_buffer(p[0], 64);
            require(buffer && buffer->buft == &f.type && b.device_payload_bytes() == 64,
                "facade cap retains original physical buffer identity and ticket authority");
            ggml_backend_buffer_free(buffer);
        }
        l.budget = nullptr;
        facade_device.iface.get_buffer_type = [](ggml_backend_dev_t d) {
            return &static_cast<CapBackend*>(d->context)->foreign_type;
        };
        {
            tsg_glm::graph_budget b;
            require(!b.prepare(types, backends, 1, 1, 5, 64, &f.type) && b.policy_refused()
                && f.calls == 1, "a facade returning another physical default type is refused prephysical");
        }
        l.detach();
    }
    // Multiple physical requests through DIFFERENT descriptors share one cap.
    {
        CapLedger l; l.attach(); CapBackend f(l);
        {
            tsg_glm::graph_budget b; l.budget = &b;
            ggml_backend_buffer_type_t types[]{&f.type, &f.type, &f.host_type};
            ggml_backend_t backends[]{&f.backend, &f.backend, &f.backend};
            auto p = b.prepare(types, backends, 3, 1, 5, 128, &f.type);
            require(p && !b.policy_refused(), "strict known type prepares without policy failure");
            auto first = ggml_backend_buft_alloc_buffer(p[0], 64);
            auto second = ggml_backend_buft_alloc_buffer(p[1], 64);
            require(first && second && first->buft == &f.type && second->buft == &f.type
                && b.device_payload_bytes() == 128, "strict cap aggregates requests across owners");
            require(!ggml_backend_buft_alloc_buffer(p[1], 1) && b.policy_refused()
                && f.calls == 2 && l.reserves == 2, "cap refuses before ticket and physical allocation");
            require(!b.prepare(types, backends, 3, 1, 5, 1024, &f.type)
                && b.policy_refused() && b.device_payload_bytes() == 128, "reprepare cannot reset a live generation refusal/cap");
            ggml_backend_buffer_free(first); ggml_backend_buffer_free(second);
            require(b.device_payload_bytes() == 128 && l.charged == 128,
                "scheduler frees do not lend retained generation credit");
        }
        l.budget = nullptr; l.detach();
    }
    // A nonzero host allocation, custom tensor allocation or wrong type refuses
    // before any device/host physical call or accounting request.
    for (int mode = 0; mode < 3; ++mode) {
        CapLedger l; l.attach(); CapBackend f(l);
        {
            tsg_glm::graph_budget b; l.budget = &b;
            ggml_backend_buffer_type_t types[]{&f.type, &f.host_type};
            ggml_backend_t backends[]{&f.backend, &f.backend};
            auto p = b.prepare(types, backends, 2, 1, 5, 64, &f.type);
            require(p != nullptr, "strict host zero-allocation descriptor prepares");
            if (mode == 0) require(!ggml_backend_buft_alloc_buffer(p[1], 1), "strict host compute payload refused");
            if (mode == 1) require(!ggml_backend_buft_alloc_buffer_n(p[0], nullptr, 0), "strict alloc_n refused explicitly");
            if (mode == 2) require(!ggml_backend_buft_alloc_buffer(p[0], (std::numeric_limits<std::size_t>::max)()), "overflow request refused");
            require(b.policy_refused() && !b.device_payload_bytes() && !f.calls && !l.reserves,
                "unsupported strict allocation is prephysical and uncharged");
        }
        l.budget = nullptr; l.detach();
    }
    // Exact ticket authority, not an advisory registry snapshot, gates payload.
    for (bool attached : {false, true}) {
        CapLedger l; if (attached) l.attach(false); CapBackend f(l);
        {
            tsg_glm::graph_budget b;
            ggml_backend_buffer_type_t types[]{&f.type}; ggml_backend_t backends[]{&f.backend};
            auto p = b.prepare(types, backends, 1, 1, 5, 64, &f.type);
            require(p && !ggml_backend_buft_alloc_buffer(p[0], 64) && b.policy_refused()
                && !f.calls && !l.reserves && !b.device_payload_bytes(), "unaccounted strict ticket refuses before physical call");
        }
        if (attached) l.detach();
    }
    // Physical, commit, recorded-owner and unexpected vendor return failures
    // all roll back only after physical free. Ordinary failure remains distinct
    // from an unrecoverable cap/type/rounding policy refusal.
    for (int mode = 0; mode < 9; ++mode) {
        CapLedger l; l.attach(); CapBackend f(l);
        {
            tsg_glm::graph_budget b; l.budget = &b;
            ggml_backend_buffer_type_t types[]{&f.type}; ggml_backend_t backends[]{&f.backend};
            auto p = b.prepare(types, backends, 1, 1, 5, 256, &f.type);
            auto anchor = ggml_backend_buft_alloc_buffer(p[0], 64);
            require(anchor != nullptr, "failure fixture has an existing retained owner");
            if (mode == 0) f.refuse_allocation = true;
            if (mode == 1) f.throw_allocation = true;
            if (mode == 2) l.refuse_commit = true;
            if (mode == 3) l.throw_commit = true;
            if (mode == 4) b.test_fail_next_charge_record();
            if (mode == 5) f.actual_extra = 1;
            if (mode == 6) f.actual_extra = -1;
            if (mode == 7) f.wrong_type = true;
            if (mode == 8) l.refuse_reserve = true;
            require(!ggml_backend_buft_alloc_buffer(p[0], 64), "injected strict failure publishes no buffer");
            if (b.device_payload_bytes() != 64 || l.physical != 64 || l.charged != 64 || l.tickets.size() != 1)
                std::fprintf(stderr, "strict failure mode=%d cap=%zu physical=%lld charged=%lld tickets=%zu\n",
                    mode, b.device_payload_bytes(), static_cast<long long>(l.physical),
                    static_cast<long long>(l.charged), l.tickets.size());
            require(b.device_payload_bytes() == 64 && l.physical == 64 && l.charged == 64 && l.tickets.size() == 1,
                "failed strict allocation rolls back all new ownership and preserves old owner");
            require(b.policy_refused() == (mode >= 5 && mode <= 7), "policy latch distinguishes allocator/accounting failure");
            f.actual_extra = 0; f.wrong_type = f.refuse_allocation = f.throw_allocation = false;
            l.refuse_commit = l.throw_commit = l.refuse_reserve = false;
            if (!b.policy_refused()) {
                auto retry = ggml_backend_buft_alloc_buffer(p[0], 192);
                require(retry && b.device_payload_bytes() == 256, "failed reservation leaves full remaining cap available");
                ggml_backend_buffer_free(retry);
            }
            ggml_backend_buffer_free(anchor);
        }
        l.budget = nullptr; l.detach();
    }
    for (int mode = 0; mode < 8; ++mode) {
        CapLedger l; CapBackend f(l); tsg_glm::graph_budget b;
        ggml_backend_buffer_type_t types[]{mode == 3 ? &f.foreign_type : &f.type};
        ggml_backend_t backends[]{&f.backend};
        if (mode == 4) f.type.iface.alloc_buffer_n = CapBackend::alloc_n;
        if (mode == 5) f.type.iface.get_alloc_size_n = CapBackend::size_n;
        if (mode == 6) f.device.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_CPU; };
        if (mode == 7) f.type.iface.alloc_buffer = nullptr;
        auto p = b.prepare(types, backends, 1, mode == 0 ? 0 : 1, mode == 1 ? 7 : 5, 64,
            mode == 2 ? nullptr : &f.type);
        require(!p && b.policy_refused() && !f.calls, "strict scope eligibility refuses unsupported configuration");
    }
    // Unlimited/default behavior still accepts original backend padding and an
    // unaccounted graph allocation; no strict observation or latch is enabled.
    {
        CapLedger l; CapBackend f(l); tsg_glm::graph_budget b;
        f.actual_extra = 1;
        ggml_backend_buffer_type_t types[]{&f.type}; ggml_backend_t backends[]{&f.backend};
        auto p = b.prepare(types, backends, 1, 1, 5);
        auto buffer = ggml_backend_buft_alloc_buffer(p[0], 64);
        require(buffer && buffer->buft == &f.type && buffer->size == 65
            && !b.policy_refused() && !b.device_payload_bytes(), "default unlimited allocation behavior remains unchanged");
        ggml_backend_buffer_free(buffer);
    }
}
}
int main() {
    strict_cap_checks();
    for (int kind : {5, 7}) {
    StrictBackend strict;
    Ledger ledger;
    ledger.expected_kind = kind;
    require(tsg::SharedCacheCharge::attach(&ledger, Ledger::reserve, Ledger::commit, Ledger::release, true), "attach graph ledger");
    ggml_context* context = ggml_init({1 << 20, nullptr, true});
    require(context != nullptr, "graph metadata allocation");
    auto* input = ggml_new_tensor_1d(context, GGML_TYPE_F32, 32);
    ggml_set_input(input);
    auto* output = ggml_scale(context, input, 3.f);
    ggml_set_output(output);
    auto* graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    {
        tsg_glm::graph_budget budget;
        ggml_backend_buffer_type_t types[]{strict.type};
        ggml_backend_t originals[]{&strict.backend};
        auto* proxies = budget.prepare(types, originals, 1, 0, kind);
        require(proxies != nullptr && proxies[0] != types[0], "allocation descriptor has distinct identity");
        require(!ggml_backend_supports_buft(originals[0], proxies[0]), "strict vendor rejects bare allocation descriptor");
        auto alias = budget.scheduler_backend(originals[0]);
        require(alias && alias != originals[0] && alias == budget.scheduler_backends()[0], "exact scheduler facade identity");
        require(alias->guid == originals[0]->guid && alias->context == originals[0]->context, "vendor GUID and context preserved");
        require(ggml_backend_supports_buft(alias, proxies[0]), "facade unwraps its allocation descriptor");
        require(ggml_backend_supports_buft(alias, types[0]), "facade accepts original physical type");
        ggml_backend_buffer_type foreign = *types[0];
        require(!ggml_backend_supports_buft(alias, &foreign), "unknown lookalike type is rejected");
        require(budget.scheduler_backend(strict.cpu) == nullptr, "unknown backend cannot acquire a scheduler ID");
        require(ggml_backend_get_default_buffer_type(alias) == types[0], "default device type remains original");
        require(std::string(ggml_backend_name(alias)) == "strict-CPU", "name callback forwards original stream");
        StrictBackend second;
        tsg_glm::graph_backend second_facade(&second.backend, nullptr,
            [](void*, ggml_backend_buffer_type_t t) { return t; });
        require(alias->iface.cpy_tensor_async(second_facade.get(), alias, nullptr, nullptr), "copy unwraps both backend aliases");
        require(second.copies == 0 && strict.copies == 1, "destination backend owns the copy callback");
        require(second_facade.get()->iface.cpy_tensor_async(alias, &second.backend, nullptr, nullptr), "copy accepts mixed alias and original stream");
        require(second.copies == 1, "mixed copy reaches original destination stream");
        auto* scheduler = ggml_backend_sched_new(budget.scheduler_backends(), proxies, 1, 2048, false, false);
        require(scheduler != nullptr, "actual scheduler accepts strict facade/type pair");
        ggml_backend_sched_set_tensor_backend(scheduler, input, alias);
        ggml_backend_sched_set_tensor_backend(scheduler, output, alias);
        require(ggml_backend_sched_get_tensor_backend(scheduler, input) == alias, "input pin uses scheduler identity");
        require(ggml_backend_sched_get_tensor_backend(scheduler, output) == alias, "operation pin uses scheduler identity");
        require(ggml_backend_sched_alloc_graph(scheduler, graph), "budgeted actual CPU graph allocation");
        require(ledger.used > 0 && !tsg::SharedCacheCharge::detach(&ledger), "physical graph blocks ledger detach");
        require(ggml_backend_buffer_get_type(input->buffer) == types[0], "physical buffer retains original vendor type");
        float values[32], result[32];
        for (int i = 0; i < 32; ++i) values[i] = float(i) - 7.f;
        ggml_backend_tensor_set(input, values, 0, sizeof(values));
        require(ggml_backend_sched_graph_compute_async(scheduler, graph) == GGML_STATUS_SUCCESS, "actual scheduler compute through vendor dispatch");
        ggml_backend_sched_synchronize(scheduler);
        ggml_backend_tensor_get(output, result, 0, sizeof(result));
        for (int i = 0; i < 32; ++i) require(result[i] == 3.f * values[i], "actual CPU graph numerical result");
        require(strict.dispatched > 0 && strict.syncs > 0, "real dispatch and synchronization callbacks forwarded");
        ggml_backend_sched_free(scheduler);
        ledger.freed = true;
        require(ledger.used > 0, "scheduler physical free precedes owning descriptor refund");
        alias->iface.free(alias);
        require(strict.frees == 0, "graph alias never frees model-owned stream");
    }
    require(ledger.used == 0 && ledger.tickets.empty(), "every graph ticket returned");
    require(tsg::SharedCacheCharge::detach(&ledger), "empty graph ledger detaches");
    ggml_free(context);
    }
    std::printf("PASS %d cumulative cap mock, strict vendor-type and actual CPU scheduler checks; CUDA model acceptance remains separate\n", checks);
}
