// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Component qualification only. Original GGML CUDA graphs are the oracle;
// no model, cache policy, actual CUDA capture, or speed claim is implied.
#include "ggml_ops_host_moe_vmm_weights.h"
#include "ggml_ops_shared_cache_budget.h"
#include "ggml_ops_internal.h"
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-cuda.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// As in the existing standalone expert-cache fixture, the production component
// is compiled directly. These are its only process-error bridge dependencies.
namespace tsg {
std::atomic<std::uint64_t> g_ggml_error_count{0};
std::atomic<bool> g_backend_compute_failed{false};
}
namespace {
using namespace tsg::host_moe_vmm;
using Charge = tsg::SharedCacheCharge;
constexpr std::size_t MiB = std::size_t(1) << 20;
constexpr int maximum = 24, selected = 2;
std::uint64_t uid = UINT64_C(0x56574d5700000000), checked_f32 = 0;

void require(bool yes, const char* message) { if (!yes) throw std::runtime_error(message); }
void driver(CUresult result, const char* message) { require(result == CUDA_SUCCESS, message); }
void stage(const char* name) {
    std::fprintf(stderr, "[VMM-WEIGHTS-TEST] stage=%s\n", name); std::fflush(stderr);
}

// Fixed callback storage, no callback allocation or throwing. Kind4 is ONLY the
// production VMM physical owner; oracle/scratch have independent kind0/kind2.
struct Ledger {
    struct Entry { std::uint64_t token = 0; std::size_t bytes = 0; int kind = 0; bool committed = false, refunded = false; };
    std::array<Entry, 512> entries{};
    std::array<std::size_t, 8> pending{}, committed{};
    std::uint64_t next = 0, reserve_calls = 0, release_calls = 0;
    int refuse_reserve_kind = -1, refuse_commit_kind = -1, fail_ack_kind = -1;
    bool failed = false;
    Entry* find(std::uint64_t token) noexcept {
        for (auto& e : entries) if (e.token == token) return &e;
        return nullptr;
    }
    static bool host(int kind) noexcept { return kind == 3 || kind == 6; }
    std::size_t used(bool on_host) const noexcept {
        std::size_t result = 0;
        for (int k = 0; k != 8; ++k) if (host(k) == on_host) result += pending[k] + committed[k];
        return result;
    }
    std::size_t unacknowledged(int kind) const noexcept {
        std::size_t count = 0;
        for (const auto& e : entries) if (e.token && e.refunded && e.kind == kind) ++count;
        return count;
    }
    static std::uint64_t reserve(void* context, int rank, int kind, std::int64_t bytes) noexcept {
        auto& l = *static_cast<Ledger*>(context); ++l.reserve_calls;
        if (rank != 0 || bytes <= 0 || kind < 0 || kind >= 8) { l.failed = true; return 0; }
        if (kind == l.refuse_reserve_kind) { l.refuse_reserve_kind = -1; return 0; }
        const auto amount = static_cast<std::size_t>(bytes), ceiling = host(kind) ? 192 * MiB : 256 * MiB;
        if (l.used(host(kind)) > ceiling || amount > ceiling - l.used(host(kind))) return 0;
        for (auto& e : l.entries) if (!e.token) {
            e = { ++l.next, amount, kind, false, false }; l.pending[kind] += amount; return e.token;
        }
        l.failed = true; return 0;
    }
    static int commit(void* context, std::uint64_t token) noexcept {
        auto& l = *static_cast<Ledger*>(context); auto* e = l.find(token);
        if (!e || e->committed || e->refunded) { l.failed = true; return 0; }
        if (e->kind == l.refuse_commit_kind) { l.refuse_commit_kind = -1; return 0; }
        l.pending[e->kind] -= e->bytes; l.committed[e->kind] += e->bytes; e->committed = true; return 1;
    }
    static int release_checked(void* context, std::uint64_t token) noexcept {
        auto& l = *static_cast<Ledger*>(context); ++l.release_calls; auto* e = l.find(token);
        if (!e) { l.failed = true; return 0; }
        if (!e->refunded) {
            auto& amount = e->committed ? l.committed[e->kind] : l.pending[e->kind];
            if (amount < e->bytes) { l.failed = true; return 0; }
            amount -= e->bytes; e->refunded = true;
        }
        if (e->kind == l.fail_ack_kind) { l.fail_ack_kind = -1; return 0; }
        *e = {}; return 1;
    }
    static void release(void* context, std::uint64_t token) noexcept {
        if (!release_checked(context, token)) static_cast<Ledger*>(context)->failed = true;
    }
};

template<class T> struct HostVector {
    std::shared_ptr<Charge> charge;
    std::vector<T> values;
    HostVector() = default;
    HostVector(const HostVector&) = delete;
    void resize(std::size_t count, std::size_t metadata_allowance = 0) {
        require(!charge && count, "Host fixture allocation reused or empty");
        const auto bytes = count * sizeof(T) + metadata_allowance;
        charge = Charge::reserve(0, 3, bytes); require(charge && charge->is_accounted(), "Host fixture ticket refused");
        values.resize(count); require(values.capacity() == count, "Host vector allocation exceeds forecast");
        require(charge->commit(bytes), "Host fixture commit refused");
    }
    void close() noexcept { std::vector<T>().swap(values); charge.reset(); }
    ~HostVector() { close(); }
    T* data() { return values.data(); }
    const T* data() const { return values.data(); }
    std::size_t size() const { return values.size(); }
};

struct Raw {
    std::array<ggml_type, 3> types;
    int hidden, ff, expert_count;
    std::array<std::size_t, 3> stride{};
    std::array<HostVector<std::uint8_t>, 3> bytes;
    Raw(const std::array<ggml_type, 3>& t, int h, int f, int experts = maximum)
        : types(t), hidden(h), ff(f), expert_count(experts) {
        require(experts >= maximum, "Raw fixture source is smaller than the exercised table");
        HostVector<float> row, importance; row.resize((std::max)(h, f)); importance.resize((std::max)(h, f));
        std::fill(importance.values.begin(), importance.values.end(), 1.f);
        for (unsigned p = 0; p != 3; ++p) {
            if (!p || types[p] != types[p-1]) ggml_quantize_init(types[p]);
            const int columns = p == 2 ? f : h, rows = p == 2 ? h : f;
            const auto row_bytes = ggml_row_size(types[p], columns); stride[p] = row_bytes * rows;
            bytes[p].resize(stride[p] * expert_count);
            for (int expert = 0; expert != expert_count; ++expert) for (int r = 0; r != rows; ++r) {
                for (int c = 0; c != columns; ++c)
                    row.values[c] = .012f * std::sin(.071f*c + .037f*r + .13f*expert + .23f*p);
                require(ggml_quantize_chunk(types[p], row.data(), bytes[p].data() + expert*stride[p] + r*row_bytes,
                    0, 1, columns, importance.data()) == row_bytes, "Unique raw row quantization failed");
            }
        }
    }
    WeightSpec spec(const void* owner, int initial = 8) const {
        WeightSpec result; result.source_owner = owner; result.initial_capacity = initial; result.maximum_capacity = expert_count;
        for (unsigned p = 0; p != 3; ++p)
            result.projections[p] = { types[p], p == 2 ? ff : hidden, p == 2 ? hidden : ff, stride[p] };
        return result;
    }
    void close() noexcept { for (auto& payload : bytes) payload.close(); }
};

struct Oracle {
    HostVector<std::uint8_t> metadata;
    ggml_context* context = nullptr;
    std::array<ggml_tensor*, 3> tensors{};
    std::array<ggml_backend_buffer_t, 3> buffers{};
    std::array<std::shared_ptr<Charge>, 3> charges{};
    Oracle(ggml_backend_t backend, const Raw& raw, const std::vector<int>& slots) {
        try {
            metadata.resize(32768, 65536); context = ggml_init({metadata.size(), metadata.data(), true});
            require(context != nullptr, "Oracle metadata failed");
            const auto type = ggml_backend_get_default_buffer_type(backend);
            for (unsigned p = 0; p != 3; ++p) {
                tensors[p] = ggml_new_tensor_3d(context, raw.types[p], p == 2 ? raw.ff : raw.hidden,
                    p == 2 ? raw.hidden : raw.ff, slots.size());
                ggml_set_input(tensors[p]); ggml_set_output(tensors[p]);
                const auto amount = ggml_backend_buft_get_alloc_size(type, tensors[p]);
                charges[p] = Charge::reserve(0, 0, amount); require(charges[p] && charges[p]->is_accounted(), "Oracle ticket refused");
                buffers[p] = ggml_backend_buft_alloc_buffer(type, amount); require(buffers[p] != nullptr, "Oracle allocation failed");
                require(ggml_backend_buffer_get_size(buffers[p]) == amount, "Oracle allocator rounded unexpectedly");
                ggml_backend_buffer_set_usage(buffers[p], GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                ggml_backend_buffer_clear(buffers[p], 0);
                require(ggml_backend_tensor_alloc(buffers[p], tensors[p], ggml_backend_buffer_get_base(buffers[p])) == GGML_STATUS_SUCCESS,
                    "Oracle tensor bind failed");
                require(charges[p]->commit(amount), "Oracle commit failed");
                for (std::size_t s = 0; s != slots.size(); ++s) if (slots[s] >= 0)
                    ggml_backend_tensor_set(tensors[p], raw.bytes[p].data() + slots[s]*raw.stride[p], s*raw.stride[p], raw.stride[p]);
            }
        } catch (...) { close(); throw; }
    }
    void close() noexcept {
        for (unsigned p = 0; p != 3; ++p) { ggml_backend_buffer_free(buffers[p]); buffers[p] = nullptr; charges[p].reset(); }
        if (context) { ggml_free(context); context = nullptr; }
    }
    ~Oracle() { close(); }
};

struct Graph {
    ggml_backend_t backend;
    int hidden, rows;
    HostVector<std::uint8_t> metadata;
    HostVector<float> input_host, routes_host, result;
    HostVector<std::int32_t> ids_host;
    ggml_context* context = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_tensor *input = nullptr, *routes = nullptr, *ids = nullptr, *output = nullptr;
    std::shared_ptr<Charge> scratch;
    Graph(ggml_backend_t b, const std::array<ggml_tensor*, 3>& weights, int h, int n) : backend(b), hidden(h), rows(n) {
        try {
            metadata.resize(2*MiB, 65536); input_host.resize(h*n); routes_host.resize(selected*n);
            result.resize(h*n); ids_host.resize(selected*n);
            context = ggml_init({metadata.size(), metadata.data(), true}); require(context != nullptr, "Graph metadata failed");
            graph = ggml_new_graph_custom(context, 128, false); graph->uid = uid++;
            input = ggml_new_tensor_3d(context, GGML_TYPE_F32, hidden, 1, rows);
            ids = ggml_new_tensor_2d(context, GGML_TYPE_I32, selected, rows);
            routes = ggml_new_tensor_3d(context, GGML_TYPE_F32, 1, selected, rows);
            ggml_set_input(input); ggml_set_input(ids); ggml_set_input(routes);
            auto* gate = ggml_mul_mat_id(context, weights[0], input, ids);
            auto* up = ggml_mul_mat_id(context, weights[1], input, ids);
            auto* activation = ggml_mul(context, ggml_silu(context, gate), up);
            auto* down = ggml_mul_mat_id(context, weights[2], activation, ids);
            auto* weighted = ggml_mul(context, down, routes);
            auto* sum = ggml_add(context, ggml_view_2d(context, weighted, hidden, rows, weighted->nb[2], 0),
                ggml_view_2d(context, weighted, hidden, rows, weighted->nb[2], weighted->nb[1]));
            output = ggml_cont(context, sum); ggml_set_output(output); ggml_build_forward_expand(graph, output);
            allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)); require(allocator != nullptr, "Graph allocator failed");
            std::size_t size = 0; ggml_gallocr_reserve_n_size(allocator, graph, nullptr, nullptr, &size);
            scratch = Charge::reserve(0, 2, size); require(scratch && scratch->is_accounted(), "Graph scratch ticket refused");
            require(ggml_gallocr_reserve(allocator, graph) && ggml_gallocr_get_buffer_size(allocator, 0) == size
                && ggml_gallocr_alloc_graph(allocator, graph), "Graph scratch differs from forecast");
            require(scratch->commit(size), "Graph scratch commit failed");
        } catch (...) { close(); throw; }
    }
    void close() noexcept {
        ggml_backend_synchronize(backend);
        if (allocator) { ggml_gallocr_free(allocator); allocator = nullptr; }
        scratch.reset(); if (context) { ggml_free(context); context = nullptr; }
    }
    ~Graph() { close(); }
    void run(const std::vector<int>& slots, const char* path, int repeat) {
        std::array<int, maximum> occupied{}; int count = 0;
        for (int s = 0; s != static_cast<int>(slots.size()); ++s) if (slots[s] >= 0) occupied[count++] = s;
        require(count >= selected, "Insufficient occupied rows for graph");
        for (int r = 0; r != rows; ++r) {
            for (int c = 0; c != hidden; ++c) input_host.values[r*hidden+c] = .2f * std::cos(.019f*c + .031f*r);
            const int first = (r*3)%count;
            int second = count-1-(r*5)%count;
            if (second == first) second = (second+1)%count;
            ids_host.values[r*selected] = occupied[first];
            ids_host.values[r*selected+1] = occupied[second];
            // Real top-k routing selects distinct experts. Duplicate IDs can
            // overflow unchanged CUDA MMID sorting's per-expert row inventory.
            require(ids_host.values[r*selected] != ids_host.values[r*selected+1]
                && slots[ids_host.values[r*selected]] >= 0 && slots[ids_host.values[r*selected+1]] >= 0,
                "Fixture top-k IDs are not distinct occupied expert slots");
            routes_host.values[r*selected] = .625f; routes_host.values[r*selected+1] = .375f;
        }
        ggml_backend_tensor_set(input, input_host.data(), 0, input_host.size()*sizeof(float));
        ggml_backend_tensor_set(ids, ids_host.data(), 0, ids_host.size()*sizeof(std::int32_t));
        ggml_backend_tensor_set(routes, routes_host.data(), 0, routes_host.size()*sizeof(float));
        std::fprintf(stderr, "[VMM-WEIGHTS-TEST] stage=compute-begin path=%s rows=%d repeat=%d capacity=%zu uid=%llu\n",
            path, rows, repeat, slots.size(), static_cast<unsigned long long>(graph->uid)); std::fflush(stderr);
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Original CUDA graph failed");
        std::fprintf(stderr, "[VMM-WEIGHTS-TEST] stage=compute-end-before-output-get path=%s rows=%d repeat=%d capacity=%zu uid=%llu\n",
            path, rows, repeat, slots.size(), static_cast<unsigned long long>(graph->uid)); std::fflush(stderr);
        ggml_backend_tensor_get(output, result.data(), 0, result.size()*sizeof(float));
        require(!tsg::g_backend_compute_failed.load(), "Owned callback reported an execution error");
    }
};

struct Graphs {
    std::unique_ptr<Graph> narrow, wide;
    std::uint64_t barriers = 0;
    static bool freeze(void* context) noexcept {
        auto& self = *static_cast<Graphs*>(context);
        // No reader/upload producer exists in this component fixture. Both real
        // graph owners synchronize before all scratch/UID owners are destroyed.
        self.narrow.reset(); self.wide.reset(); ++self.barriers; return true;
    }
    WeightBarrier barrier() noexcept { return {freeze, this}; }
};
struct Arena {
    WeightArena* value = nullptr;
    Graphs graphs;
    Ledger& ledger;
    explicit Arena(Ledger& l, WeightOptions options = {}) : ledger(l) {
        options.maximum_total_chunks = 128;
        const auto status = create_weight_arena(options, value);
        if (status != WeightStatus::ready || !value) {
            for (int attempt = 0; value && attempt != 3; ++attempt) destroy_weight_arena_checked(value, graphs.barrier());
            if (value) ledger.failed = true;
            throw std::runtime_error("Accounted VMM arena failed");
        }
    }
    ~Arena() {
        ledger.refuse_reserve_kind = ledger.refuse_commit_kind = ledger.fail_ack_kind = -1;
        for (int attempt = 0; value && attempt != 3; ++attempt) destroy_weight_arena_checked(value, graphs.barrier());
        if (value) ledger.failed = true;
    }
};

std::array<ggml_tensor*, 3> tensors(Arena& arena, WeightHandle handle) {
    std::array<ggml_tensor*, 3> result{};
    for (unsigned p = 0; p != 3; ++p) {
        result[p] = weight_tensor(arena.value, handle, p); require(result[p] != nullptr, "Published VMM tensor unavailable");
    }
    return result;
}
void fill(ggml_backend_t backend, const Raw& raw, const std::array<ggml_tensor*, 3>& table, const std::vector<int>& slots) {
    for (unsigned p = 0; p != 3; ++p) for (std::size_t s = 0; s != slots.size(); ++s) if (slots[s] >= 0)
        ggml_backend_tensor_set_async(backend, table[p], raw.bytes[p].data() + slots[s]*raw.stride[p], s*raw.stride[p], raw.stride[p]);
    ggml_backend_synchronize(backend); // Join original backend's real async upload seam.
}
void async_2d(ggml_backend_t backend, const Raw& raw, const std::array<ggml_tensor*, 3>& table) {
    for (unsigned p = 0; p != 3; ++p) {
        const auto row = ggml_row_size(raw.types[p], p == 2 ? raw.ff : raw.hidden), pitch = row+17;
        HostVector<std::uint8_t> source, destination; source.resize(2*pitch); destination.resize(2*pitch);
        std::fill(source.values.begin(), source.values.end(), 0xa5);
        std::fill(destination.values.begin(), destination.values.end(), 0x5a);
        std::memcpy(source.data(), raw.bytes[p].data(), row);
        std::memcpy(source.data()+pitch, raw.bytes[p].data()+2*row, row);
        const std::uint8_t poison0 = static_cast<std::uint8_t>(source.values[0]^0xff),
            poison1 = static_cast<std::uint8_t>(source.values[pitch]^0xff);
        ggml_backend_tensor_set(table[p], &poison0, 0, 1);
        ggml_backend_tensor_set(table[p], &poison1, 2*row, 1);
        std::uint8_t observed0 = 0, observed1 = 0;
        ggml_backend_tensor_get(table[p], &observed0, 0, 1);
        ggml_backend_tensor_get(table[p], &observed1, 2*row, 1);
        require(observed0 == poison0 && observed1 == poison1, "2D upload poison did not change actual destination");
        ggml_backend_tensor_set_2d_async(backend, table[p], source.data(), 0, row, 2, 2*row, pitch);
        ggml_backend_tensor_get_2d_async(backend, table[p], destination.data(), 0, row, 2, 2*row, pitch);
        ggml_backend_synchronize(backend);
        for (std::size_t copy = 0; copy != 2; ++copy) {
            require(std::memcmp(destination.data()+copy*pitch, source.data()+copy*pitch, row) == 0,
                "Original backend 2D async upload/readback differs");
            for (std::size_t gap = row; gap != pitch; ++gap)
                require(destination.values[copy*pitch+gap] == 0x5a, "2D async readback overwrote host stride gap");
        }
    }
}
void verify(ggml_backend_t backend, const Raw& raw, const std::array<ggml_tensor*, 3>& table,
    const std::vector<int>& slots, bool use_callbacks = true, bool check_padding = true) {
    HostVector<std::uint8_t> one; one.resize(*std::max_element(raw.stride.begin(), raw.stride.end()));
    for (unsigned p = 0; p != 3; ++p) {
        for (std::size_t s = 0; s != slots.size(); ++s) if (slots[s] >= 0) {
            if (use_callbacks) {
                // The actual production async path checks exact CUDA buft and
                // submits to its original backend; no custom synchronous shim.
                ggml_backend_tensor_get_async(backend, table[p], one.data(), s*raw.stride[p], raw.stride[p]);
                ggml_backend_synchronize(backend);
            } else {
                // Quarantined owners forbid callbacks/compute. Read only the
                // known retained old physical prefix to prove failed writes
                // did not change valid rows; this is not permission to execute.
                driver(cuMemcpyDtoH(one.data(), reinterpret_cast<CUdeviceptr>(table[p]->data) + s*raw.stride[p], raw.stride[p]),
                    "Retained-prefix readback failed");
                driver(cuCtxSynchronize(), "Retained-prefix readback fence failed");
            }
            require(std::memcmp(one.data(), raw.bytes[p].data() + slots[s]*raw.stride[p], raw.stride[p]) == 0,
                "Raw source-qualified slot bytes differ");
        }
        const auto logical = ggml_nbytes(table[p]);
        const auto allocation = ggml_backend_buft_get_alloc_size(table[p]->buffer->buft, table[p]);
        require(allocation >= logical && allocation-logical <= 65536, "Unexpected padding envelope");
        if (check_padding && allocation > logical) {
            std::array<std::uint8_t, 65536> padding{};
            driver(cuMemcpyDtoH(padding.data(), reinterpret_cast<CUdeviceptr>(table[p]->data) + logical, allocation-logical), "Padding readback failed");
            driver(cuCtxSynchronize(), "Padding readback fence failed");
            require(std::all_of(padding.begin(), padding.begin() + (allocation-logical), [](std::uint8_t v) { return v == 0; }),
                "Actual CUDA allocation padding not initialized");
        }
    }
}
std::size_t check_usage(Arena& arena, WeightHandle handle, const std::array<ggml_tensor*, 3>& table,
    const std::array<void*, 3>& bases, int capacity, const WeightLayoutPlan& plan) {
    const auto u = weight_usage(arena.value, handle);
    require(u.usable && !u.failed && !u.context_lost && u.charge_certificate_api && u.actual_charge_certificate
        && u.host_ticket_held && !u.host_ticket_only && u.primary_context_owned, "Actual accounted VMM owner not usable");
    require(u.capacity == static_cast<std::size_t>(capacity) && !u.ticket_pending && !u.ticket_only && !u.quarantine,
        "Published VMM ownership contains pending/quarantined records");
    require(plan.capacity == static_cast<std::size_t>(capacity), "Actual owner check used a different planned capacity");
    std::size_t sum = 0, records = 0;
    for (unsigned p = 0; p != 3; ++p) {
        const auto& expected = plan.projections[p];
        const auto allocation = ggml_backend_buft_get_alloc_size(table[p]->buffer->buft, table[p]);
        require(expected.chunk && expected.granule && expected.chunk % expected.granule == 0
            && expected.allocation == allocation && expected.physical % expected.chunk == 0,
            "VMM owner does not match actual readonly allocation/chunk geometry");
        const auto actual = expected.physical;
        require(table[p]->data == bases[p] && table[p]->ne[2] == capacity && u.physical_prefix[p] == actual
            && ggml_backend_buffer_get_size(table[p]->buffer) == actual && u.reserved_va[p] >= actual,
            "VMM address/geometry changed outside committed prefix");
        require(ggml_backend_buffer_get_usage(table[p]->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE,
            "VMM usage changed original CUDA execution policy");
        sum += actual;
        records += actual / expected.chunk;
    }
    require(sum == plan.total && u.mapped == sum && u.ticket_committed == sum && !u.created_unmapped
        && u.ticket_records == records, "Physical prefixes and immutable whole-chunk ticket sum differ");
    require(arena.ledger.pending[4] == 0 && arena.ledger.committed[4] == sum, "VMM ledger differs from actual declared chunks");
    return sum;
}
void compare(ggml_backend_t backend, Arena& arena, const Raw& raw, WeightHandle handle, const std::vector<int>& slots) {
    auto weights = tensors(arena, handle);
    Oracle oracle(backend, raw, slots);
    Graph reference_narrow(backend, oracle.tensors, raw.hidden, 1), reference_wide(backend, oracle.tensors, raw.hidden, 9);
    if (!arena.graphs.narrow) arena.graphs.narrow = std::make_unique<Graph>(backend, weights, raw.hidden, 1);
    if (!arena.graphs.wide) arena.graphs.wide = std::make_unique<Graph>(backend, weights, raw.hidden, 9);
    require(arena.graphs.narrow->graph->uid != arena.graphs.wide->graph->uid, "Narrow/wide graph UIDs collide");
    for (int repeat = 0; repeat != 3; ++repeat) {
        for (auto pair : {std::pair<Graph*, Graph*>{&reference_narrow, arena.graphs.narrow.get()},
                          std::pair<Graph*, Graph*>{&reference_wide, arena.graphs.wide.get()}}) {
            pair.first->run(slots, "oracle", repeat); pair.second->run(slots, "owned", repeat);
            const auto& expected = pair.first->result.values; const auto& actual = pair.second->result.values;
            require(expected.size() == actual.size() && std::memcmp(expected.data(), actual.data(), actual.size()*sizeof(float)) == 0,
                "Original CUDA full-resident oracle differs in F32 bits");
            require(std::all_of(actual.begin(), actual.end(), [](float v) { return std::isfinite(v); }), "Nonfinite oracle comparison");
            checked_f32 += actual.size();
        }
    }
}

void close_with_host_ack(Arena& arena, Ledger& ledger) {
    // No ordinary fixture Host owner may confound this final acknowledgment-only
    // case. Only the arena's actual durable metadata ticket remains.
    const auto arena_host_bytes = weight_usage(arena.value, {}).declared_host_charge;
    require(ledger.used(false) == 0 && ledger.used(true) == arena_host_bytes, "Final Host test contains unrelated live payload owners");
    ledger.fail_ack_kind = 3;
    require(!destroy_weight_arena_checked(arena.value, arena.graphs.barrier()) && arena.value, "Host acknowledgment failure dropped arena shell");
    const auto host_only = weight_usage(arena.value, {});
    require(host_only.host_ticket_held && host_only.host_ticket_only && !host_only.portable_and_adapter_slab
        && !host_only.primary_context_owned && ledger.used(true) == 0 && ledger.unacknowledged(3) == 1
        && !Charge::detach(&ledger), "Isolated Host ticket did not survive real slab/context cleanup");
    require(destroy_weight_arena_checked(arena.value, arena.graphs.barrier()) && !arena.value, "Final Host acknowledgment retry failed");
    stage("cleanup-completed");
}

void exercise(ggml_backend_t backend, Ledger& ledger, const std::array<ggml_type, 3>& types, bool coarse = false) {
    stage("quantization-begin"); const auto began = std::chrono::steady_clock::now();
    const int source_experts = coarse ? 32 : maximum;
    Raw raw(types, coarse ? 1024 : 512, 1024, source_experts);
    std::fprintf(stderr, "[VMM-WEIGHTS-TEST] preparation_ms=%.3f\n",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-began).count()); std::fflush(stderr);
    const int source_owner = 17; WeightHandle handle;
    WeightOptions options; options.maximum_total_chunks = 128;
    if (coarse) options.preferred_chunk_bytes = 16 * MiB;
    const auto declared = raw.spec(&source_owner);
    const auto reserve_calls = ledger.reserve_calls;
    WeightHostPlan host_plan;
    require(plan_weight_host(options, host_plan) == WeightStatus::ready && host_plan.slab_bytes
        && host_plan.declared_charge_bytes > host_plan.slab_bytes && host_plan.maximum_live_ticket_records == 128
        && !host_plan.allocator_overhead_exact, "Pure durable Host metadata plan invalid");
    WeightLayoutPlan initial_plan, grown_plan, shrunk_plan;
    require(plan_weight_layout(options, declared.projections, source_experts, 8, initial_plan) == WeightStatus::ready
        && plan_weight_layout(options, declared.projections, source_experts, 24, grown_plan) == WeightStatus::ready
        && plan_weight_layout(options, declared.projections, source_experts, 12, shrunk_plan) == WeightStatus::ready
        && initial_plan.total < grown_plan.total && shrunk_plan.total < grown_plan.total,
        "Actual layout planner does not cross expected physical chunk boundaries");
    auto malformed = declared.projections; malformed[0].stride++;
    WeightLayoutPlan rejected = grown_plan;
    require(plan_weight_layout(options, malformed, source_experts, 8, rejected) == WeightStatus::refused && !rejected.total
        && !rejected.capacity && ledger.reserve_calls == reserve_calls, "Readonly plan allocated a ticket or accepted malformed stride");
    if (coarse) {
        require(types == std::array<ggml_type, 3>{GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0}
            && initial_plan.total == 48 * MiB && grown_plan.total == 96 * MiB && shrunk_plan.total == 48 * MiB,
            "Coarse Q8 fixture does not exercise the declared 48/96/48 MiB physical geometry");
        for (const auto* plan : {&initial_plan, &grown_plan, &shrunk_plan})
            for (const auto& p : plan->projections)
                require(p.chunk == 16 * MiB && p.allocation == p.stride * plan->capacity,
                    "Coarse Q8 actual plan has different chunks or unexpected CUDA row padding");
        auto bad_options = options; bad_options.preferred_chunk_bytes = 3 * MiB;
        WeightHostPlan bad_host;
        require(plan_weight_host(bad_options, bad_host) == WeightStatus::refused
            && plan_weight_layout(bad_options, declared.projections, source_experts, 8, rejected) == WeightStatus::refused
            && !rejected.total && ledger.reserve_calls == reserve_calls,
            "Malformed explicit chunk was accepted or caused a ticket reservation");
        std::fprintf(stderr, "[VMM-WEIGHTS-TEST] coarse_profile=1 hidden=1024 ff=1024 source_experts=32 "
            "planned_chunk_bytes=%zu planned_initial=%zu planned_grown=%zu planned_shrunk=%zu "
            "main_padding_bytes=0 padding_fault_qualified=0\n",
            options.preferred_chunk_bytes, initial_plan.total, grown_plan.total, shrunk_plan.total); std::fflush(stderr);
    }
    Arena arena(ledger, options);
    require(create_weight_table(arena.value, 0, raw.spec(&source_owner), handle) == WeightStatus::ready, "Initial VMM table failed");
    auto table = tensors(arena, handle); const std::array<void*, 3> bases{table[0]->data, table[1]->data, table[2]->data};
    require(bases[0] != bases[1] && bases[0] != bases[2] && bases[1] != bases[2], "Independent projection VAs alias");
    std::vector<int> slots(8); for (int s = 0; s != 8; ++s) slots[s] = s; slots[2] = -1;
    if (coarse) slots[7] = source_experts - 1; // Exercise the last immutable source row as well as slot-offset bounds.
    fill(backend, raw, table, slots); async_2d(backend, raw, table); verify(backend, raw, table, slots);
    const auto initial = check_usage(arena, handle, table, bases, 8, initial_plan);
    const auto initial_usage = weight_usage(arena.value, handle);
    require(initial == initial_plan.total && initial_usage.declared_host_charge == host_plan.declared_charge_bytes
        && initial_usage.portable_and_adapter_slab == host_plan.slab_bytes, "Factory physical/Host owner differs from readonly plan");
    stage("initial-oracle"); compare(backend, arena, raw, handle, slots);

    // All growth tickets are obtained before any new physical allocation.
    auto stats = weight_test_stats(arena.value); ledger.refuse_reserve_kind = 4;
    require(grow_weight_table(arena.value, handle, 24, arena.graphs.barrier()) == WeightStatus::unchanged,
        "Reserve refusal did not preserve original table");
    require(weight_test_stats(arena.value).successful_creates == stats.successful_creates, "Reserve refusal allocated physical memory");
    require(check_usage(arena, handle, table, bases, 8, initial_plan) == initial, "Reserve refusal changed old owner");

    stats = weight_test_stats(arena.value);
    require(weight_test_fault(arena.value, WeightTestFault::create_after_successes, 1), "Cannot arm partial-create failure");
    require(grow_weight_table(arena.value, handle, 24, arena.graphs.barrier()) == WeightStatus::unchanged,
        "Partial real-create failure did not roll back");
    auto after = weight_test_stats(arena.value);
    require(after.successful_creates == stats.successful_creates+1 && after.create_failures == stats.create_failures+1,
        "Partial-create fault did not follow exactly one real successful allocation");
    require(check_usage(arena, handle, table, bases, 8, initial_plan) == initial, "Partial-create rollback lost old owner or additive credit");
    verify(backend, raw, table, slots);

    stats = weight_test_stats(arena.value); ledger.refuse_commit_kind = 4;
    require(grow_weight_table(arena.value, handle, 24, arena.graphs.barrier()) == WeightStatus::unchanged,
        "Postphysical commit refusal did not roll back");
    require(weight_test_stats(arena.value).successful_creates > stats.successful_creates, "Commit refusal did not follow real allocation");
    require(check_usage(arena, handle, table, bases, 8, initial_plan) == initial, "Commit rollback lost old owner");
    verify(backend, raw, table, slots);

    stage("grow-delta");
    require(grow_weight_table(arena.value, handle, 24, arena.graphs.barrier()) == WeightStatus::ready, "Actual additive growth failed");
    slots.resize(24); for (int s = 8; s != 24; ++s) slots[s] = s;
    fill(backend, raw, table, slots); verify(backend, raw, table, slots);
    const auto grown = check_usage(arena, handle, table, bases, 24, grown_plan);
    require(grown > initial && grown == grown_plan.total, "Quantized geometry failed to match planned physical delta");
    stage("grown-oracle"); compare(backend, arena, raw, handle, slots);

    // Real synchronous owned-destination callback, poisoned first so a no-op
    // callback cannot pass. Never invoke generic tensor_copy[_async].
    {
        Oracle oracle(backend, raw, slots);
        for (unsigned p = 0; p != 3; ++p) {
            const auto poison = static_cast<std::uint8_t>(raw.bytes[p].values[0]^0xff);
            ggml_backend_tensor_set(table[p], &poison, 0, 1);
            std::uint8_t observed = 0;
            ggml_backend_tensor_get(table[p], &observed, 0, 1);
            require(observed == poison, "Stock-copy destination poison not observed");
            require(ggml_backend_buffer_copy_tensor(oracle.tensors[p], table[p]), "Owned D2D callback unavailable");
        }
        require(!tsg::g_backend_compute_failed.load(), "Owned D2D callback rejected stock CUDA source");
        verify(backend, raw, table, slots);
    }
    const auto barrier_count = arena.graphs.barriers;
    require(shrink_weight_table_empty_tail(arena.value, handle, 12, &source_owner, 1, slots.data(), slots.size(), arena.graphs.barrier())
        == WeightStatus::refused && arena.graphs.barriers == barrier_count, "Occupied tail was accepted or mutated before refusal");
    require(copy_weight_slot(arena.value, handle, 23, 2, &source_owner, slots.data(), slots.size(), arena.graphs.barrier()) == WeightStatus::ready,
        "Exact high-slot D2D compaction failed");
    slots[2] = slots[23]; for (int s = 12; s != 24; ++s) slots[s] = -1;
    verify(backend, raw, table, slots); stage("compacted-oracle"); compare(backend, arena, raw, handle, slots);
    require(shrink_weight_table_empty_tail(arena.value, handle, 12, &source_owner, 2, slots.data(), slots.size(), arena.graphs.barrier())
        == WeightStatus::ready, "Proved empty-tail shrink failed");
    slots.resize(12); verify(backend, raw, table, slots);
    const auto shrunk = check_usage(arena, handle, table, bases, 12, shrunk_plan);
    require(shrunk < grown && shrunk == shrunk_plan.total, "Empty-tail shrink failed to match actual planned physical refund");
    stage("shrunk-oracle"); compare(backend, arena, raw, handle, slots);
    if (coarse) {
        ledger.fail_ack_kind = 4;
        require(!close_weight_table(arena.value, handle, arena.graphs.barrier()),
            "Coarse Device refund acknowledgment failure was discarded");
        const auto held = weight_usage(arena.value, handle);
        require(held.ticket_only > 0 && held.quarantine > 0 && !held.usable
            && ledger.unacknowledged(4) == 1 && !Charge::detach(&ledger),
            "Coarse whole-chunk acknowledgment failure lost its exact token");
    }
    require(close_weight_table(arena.value, handle, arena.graphs.barrier()), "Main table close failed");
    require(!weight_tensor(arena.value, handle, 0) && !ledger.pending[4] && !ledger.committed[4], "Closed table handle or physical ticket remains");
    std::fprintf(stderr, "[VMM-WEIGHTS-TEST] physical_initial=%zu physical_grown=%zu physical_shrunk=%zu declared_payload_not_total_vram=1\n",
        initial, grown, shrunk); std::fflush(stderr);

    if (coarse) {
        // Aligned Q8 H/FF1024 has no CUDA row padding. The existing default/min
        // 256-wide fixture below remains the separate real padding-fault proof.
        raw.close(); close_with_host_ack(arena, ledger); return;
    }

    stage("tiny-padding-and-owned-source"); Raw tiny(types, 256, 256);
    WeightHandle tiny_a, tiny_b;
    require(create_weight_table(arena.value, 1, tiny.spec(&source_owner), tiny_a) == WeightStatus::ready
        && create_weight_table(arena.value, 2, tiny.spec(&source_owner), tiny_b) == WeightStatus::ready, "Tiny padding tables failed");
    auto source = tensors(arena, tiny_a), destination = tensors(arena, tiny_b);
    std::vector<int> tiny_slots(8); for (int s = 0; s != 8; ++s) tiny_slots[s] = s;
    fill(backend, tiny, source, tiny_slots);
    for (unsigned p = 0; p != 3; ++p) {
        ggml_backend_buffer_clear(destination[p]->buffer, 0xa5);
        std::array<std::uint8_t, 16> tail{};
        driver(cuMemcpyDtoH(tail.data(), reinterpret_cast<CUdeviceptr>(destination[p]->data)
            + ggml_backend_buffer_get_size(destination[p]->buffer)-tail.size(), tail.size()), "Mapped-prefix clear readback failed");
        driver(cuCtxSynchronize(), "Mapped-prefix clear fence failed");
        require(std::all_of(tail.begin(), tail.end(), [](std::uint8_t v) { return v == 0xa5; }), "Clear omitted committed chunk tail");
        const auto poison = static_cast<std::uint8_t>(tiny.bytes[p].values[0]^0xff);
        ggml_backend_tensor_set(destination[p], &poison, 0, 1);
        std::uint8_t observed = 0;
        ggml_backend_tensor_get(destination[p], &observed, 0, 1);
        require(observed == poison, "Owned-source copy destination poison not observed");
        require(ggml_backend_buffer_copy_tensor(source[p], destination[p]), "Owned-to-owned callback unavailable");
    }
    require(!tsg::g_backend_compute_failed.load(), "Owned-to-owned VMM source proof rejected");
    // Copy only logical source payload; clear has deliberately poisoned the
    // destination padding. Check copied bytes NOW, so later fill cannot hide a
    // no-op owned-to-owned callback. Then separately reinitialize the padding.
    verify(backend, tiny, destination, tiny_slots, true, false);
    for (unsigned p = 0; p != 3; ++p) ggml_backend_buffer_clear(destination[p]->buffer, 0);
    fill(backend, tiny, destination, tiny_slots); verify(backend, tiny, destination, tiny_slots);
    verify(backend, tiny, source, tiny_slots);
    require(close_weight_table(arena.value, tiny_b, arena.graphs.barrier()), "Tiny destination close failed");
    stats = weight_test_stats(arena.value);
    const auto error_epoch = tsg::g_ggml_error_count.load();
    require(weight_test_fault(arena.value, WeightTestFault::padding_after_write, 0), "Cannot arm real padding failure");
    require(grow_weight_table(arena.value, tiny_a, 24, arena.graphs.barrier()) == WeightStatus::quarantined, "Padding failure was not fail-closed");
    after = weight_test_stats(arena.value);
    require(after.padding_writes == stats.padding_writes+1 && after.padding_failures == stats.padding_failures+1,
        "Initialization fault did not follow a real nonzero padding write");
    require(tsg::g_ggml_error_count.load() == error_epoch+1 && tsg::g_backend_compute_failed.load(),
        "Initialization failure did not record exactly one real owner error epoch");
    require(weight_usage(arena.value, tiny_a).failed && !weight_usage(arena.value, tiny_a).usable
        && !weight_tensor(arena.value, tiny_a, 0), "Failed initialization published usable geometry");
    verify(backend, tiny, source, tiny_slots, false);
    ledger.fail_ack_kind = 4;
    require(!close_weight_table(arena.value, tiny_a, arena.graphs.barrier()), "Failed Device refund acknowledgment was discarded");
    const auto quarantined = weight_usage(arena.value, tiny_a);
    require(quarantined.ticket_only > 0 && quarantined.quarantine > 0 && !quarantined.usable
        && ledger.unacknowledged(4) == 1 && !Charge::detach(&ledger), "Device ticket-only quarantine lost exact token/lifetime");
    require(close_weight_table(arena.value, tiny_a, arena.graphs.barrier()), "Exact Device-ticket acknowledgment retry failed");
    require(!ledger.pending[4] && !ledger.committed[4] && !ledger.unacknowledged(4), "Device ticket retry refunded twice or leaked credit");
    raw.close(); tiny.close();
    close_with_host_ack(arena, ledger);
}
} // namespace

int main(int argc, char** argv) {
    static_assert(GGML_TYPE_IQ2_XXS == 16 && GGML_TYPE_IQ3_XXS == 18
        && GGML_TYPE_IQ1_S == 19 && GGML_TYPE_IQ4_XS == 23, "Saved GLM quantization IDs differ from pinned upstream");
    std::array<ggml_type, 3> types{GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0};
    const char* label = "q8_0";
    bool coarse = false;
    if (argc == 3 && std::strcmp(argv[1], "--quantization") == 0) {
        if (std::strcmp(argv[2], "iq1_m") == 0) types.fill(GGML_TYPE_IQ1_M);
        else if (std::strcmp(argv[2], "iq2_xxs") == 0) types.fill(GGML_TYPE_IQ2_XXS);
        else if (std::strcmp(argv[2], "q8_0") != 0) return 2;
        label = argv[2];
    } else if (argc == 3 && std::strcmp(argv[1], "--profile") == 0) {
        if (std::strcmp(argv[2], "glm_g3") == 0) types = {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS};
        else if (std::strcmp(argv[2], "glm_g4") == 0) types = {GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_S, GGML_TYPE_IQ3_XXS};
        else if (std::strcmp(argv[2], "glm_g11") == 0) types = {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ4_XS};
        else if (std::strcmp(argv[2], "coarse_q8") == 0) coarse = true;
        else return 2;
        label = argv[2];
    } else if (argc != 1) return 2;
    std::fprintf(stderr, "[VMM-WEIGHTS-TEST] profile=%s projection_types=%d,%d,%d\n", label,
        static_cast<int>(types[0]), static_cast<int>(types[1]), static_cast<int>(types[2])); std::fflush(stderr);
    if (const auto* value = std::getenv("GGML_CUDA_DEVICES"); value && *value) {
        std::puts("UNAVAILABLE: virtual CUDA mapping is outside this component fixture"); return 77;
    }
    int count = 0, supported = 0; std::size_t granule = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count != 1 || cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess
        || cuInit(0) != CUDA_SUCCESS || cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, 0) != CUDA_SUCCESS
        || !supported) { std::puts("UNAVAILABLE: requires one real CUDA device with VMM"); return 77; }
    CUmemAllocationProp prop{}; prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id = 0;
    if (cuMemGetAllocationGranularity(&granule, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS
        || !granule || granule > 2*MiB || (granule & (granule-1))) {
        std::puts("UNAVAILABLE: unsupported driver granularity"); return 77;
    }
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fputs("FAIL: CUDA backend unavailable after positive capability\n", stderr); return 1; }
    Ledger ledger; int result = 1; bool attached = false;
    try {
        require(ggml_backend_get_default_buffer_type(backend) == ggml_backend_cuda_buffer_type(0), "Original CUDA buft mismatch");
        require(Charge::attach(&ledger, Ledger::reserve, Ledger::commit, Ledger::release, true, true), "Legacy ledger attach failed"); attached = true;
        WeightArena* absent = nullptr; WeightOptions options;
        const auto calls = ledger.reserve_calls;
        require(create_weight_arena(options, absent) == WeightStatus::unsupported && !absent && ledger.reserve_calls == calls,
            "Legacy attachment escaped checked capability preflight");
        require(Charge::detach(&ledger), "Legacy scope detach failed"); attached = false;
        require(Charge::attach(&ledger, Ledger::reserve, Ledger::commit, Ledger::release, true, true, Ledger::release_checked), "Checked ledger attach failed"); attached = true;
        exercise(backend, ledger, types, coarse);
        require(!ledger.failed && ledger.used(true) == 0 && ledger.used(false) == 0, "Payload ledger or retained cleanup is not zero");
        require(std::all_of(ledger.entries.begin(), ledger.entries.end(), [](const Ledger::Entry& e) { return !e.token; }), "Acknowledgment-only token remains");
        require(Charge::detach(&ledger), "Clean ticket owners did not release attachment"); attached = false;
        std::printf("PASS VMM production owner type=%s exact_f32=%llu checked_payload_pending=0 committed=0 quarantine=0; projection_types=%d,%d,%d; "
            "CUDA capture unobserved; driver/context/workspace overhead excluded; no model/performance claim\n",
            label, static_cast<unsigned long long>(checked_f32),
            static_cast<int>(types[0]), static_cast<int>(types[1]), static_cast<int>(types[2])); result = 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); }
    if (attached && !Charge::detach(&ledger)) {
        // Keep backend/context alive with failed owners until process exit.
        std::fputs("FAIL: retained ticket/physical cleanup remains; backend not freed\n", stderr); return 1;
    }
    ggml_backend_synchronize(backend); ggml_backend_free(backend); ggml_quantize_free();
    return result;
}
