// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// The optional expert tier against an independent full-resident CUDA graph.
#include "ggml_ops_internal.h"
#include "ggml_ops_moe_prefetch.h"
#include "ggml_ops_file_source.h"
#include "ggml_ops_moe_prefetch_policy.h"
#include "ggml_ops_precision_policy.h"
#include "ggml_ops_host_moe_cache_policy.h"
#include "ggml_ops_shared_cache_budget.h"
#include "ggml-impl.h"
#ifdef TSG_GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include <array>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <fstream>
#include <stdexcept>

extern "C" int TSGgml_HostMoeExpertCacheBudgetKind();
extern "C" int TSGgml_HostMoeExpertCacheStats(std::int64_t*, std::int64_t*,
    std::int64_t*, std::int64_t*, std::int64_t*);
extern "C" int TSGgml_HostMoeExpertCacheExecutionStats(std::uint64_t*, std::uint64_t*, std::uint64_t*);
extern "C" std::int64_t TSGgml_TrimHostMoeExpertCache(std::int64_t);
extern "C" int TSGgml_BeginHostMoeExpertCacheScope(std::int64_t, int, int, std::uint64_t*);
extern "C" int TSGgml_UpdateHostMoeExpertCacheScope(std::uint64_t, std::int64_t, std::int64_t*);
extern "C" int TSGgml_TrimHostMoeExpertCacheScope(std::uint64_t, std::int64_t, std::int64_t*);
extern "C" int TSGgml_EndHostMoeExpertCacheScope(std::uint64_t, std::int64_t*);
extern "C" int TSGgml_RegisterHostFileSource(const void*, std::int64_t, const char*, std::int64_t);

// This standalone target compiles the production cache implementation with
// small bridge stubs; it never loads a model or depends on modified ggml code.
namespace tsg
{
    void host_moe_cpu_staging_release() {} // separate CPU staging fixture owns this path
    DeviceState g_device_states[TSG_MAX_DEVICES];
    std::atomic<int> g_device_count{1};
    std::atomic<int> g_tp_global_degree{0};
    thread_local int g_active_rank = 0;
    std::atomic<std::uint64_t> g_ggml_error_count{0};
    std::atomic<bool> g_backend_compute_failed{false};
    std::uint64_t g_test_prefetch_calls = 0;
    int g_test_copy_failure_countdown = 0;
    int g_test_file_failure_countdown = 0;
    std::uint64_t g_test_file_reads = 0;
    void set_last_error(const std::string& message) { std::fprintf(stderr, "%s\n", message.c_str()); }
    void clear_last_error() { }
    struct CacheOperatorShape {
        int projections = 0, rows = 0, selected = 0, adds = 0, weighted = 0;
        bool valid = true;
    } g_test_operator_shape;
    // Inspect the graph actually submitted by the production cache, not its
    // supergraph allocation plan. A single submission alone does not prove
    // that its expert operators batch multiple requests.
    ggml_backend_t q8_f32_execution_backend(ggml_backend_t backend, ggml_cgraph* graph) {
        g_test_operator_shape = {};
        const ggml_tensor* ids = nullptr;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            const auto* node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_MUL_MAT_ID) {
                auto& shape = g_test_operator_shape;
                if (shape.projections == 0) {
                    shape.rows = static_cast<int>(node->src[1]->ne[1]);
                    shape.selected = static_cast<int>(node->ne[1] / shape.rows);
                    ids = node->src[2];
                }
                const auto expected_inputs = shape.projections < 2 ? shape.rows : shape.rows * shape.selected;
                shape.valid &= node->ne[1] == shape.rows * shape.selected && node->ne[2] == 1
                    && node->src[1]->ne[1] == expected_inputs && node->src[1]->ne[2] == 1
                    && node->src[2] == ids && ids->ne[0] == shape.rows * shape.selected && ids->ne[1] == 1;
                ++shape.projections;
            }
            if (node->op == GGML_OP_MUL && node->src[1]->ne[0] == 1) {
                auto& shape = g_test_operator_shape;
                shape.valid &= node->src[0]->ne[1] == shape.selected && node->src[0]->ne[2] == shape.rows
                    && (node->src[0]->flags & GGML_TENSOR_FLAG_OUTPUT) != 0
                    && (shape.rows == 1 || node->src[0]->op == GGML_OP_CONT);
                ++shape.weighted;
            }
            if (node->op == GGML_OP_ADD) {
                g_test_operator_shape.valid &= node->ne[1] == g_test_operator_shape.rows;
                ++g_test_operator_shape.adds;
            }
        }
        // Expert tensors have no registered dense-Qwen precision policy in
        // this fixture. Dispatch unchanged upstream ggml CUDA operators.
        return backend;
    }
    void host_moe_expert_cache_test_fail_next(int stage);
    std::size_t host_moe_expert_cache_test_physical_bytes();
    void host_moe_expert_cache_test_file_slice_limit(std::size_t bytes);
    std::size_t host_moe_expert_cache_test_file_workspace_bytes();
    void host_moe_expert_cache_test_available_bytes(std::size_t bytes);
    std::uint64_t host_moe_expert_cache_test_output_drains();
#ifdef TSG_GGML_USE_CUDA
    void read_expert_files(std::vector<ExpertReadTask>& tasks,
        const std::function<void(std::size_t)>& consume)
    {
        for (std::size_t i = tasks.size(); i-- > 0;) {
            if (g_test_file_failure_countdown > 0 && --g_test_file_failure_countdown == 0)
                throw std::runtime_error("Injected interrupted expert file read");
            tasks[i].prepare();
            ++g_test_file_reads;
            consume(i);
        }
    }
    void host_pin_split(const void*, std::size_t bytes,
        std::vector<std::pair<std::size_t, std::size_t>>& pieces)
    {
        if (g_test_copy_failure_countdown > 0 && --g_test_copy_failure_countdown == 0)
            throw std::runtime_error("Injected partial expert upload failure");
        pieces.assign(1, {0, bytes});
    }
    std::chrono::steady_clock::duration prefetch_mapped_experts(
        const std::vector<MoePrefetchExpert>& experts,
        const std::function<void(std::size_t, std::size_t)>& consume)
    {
        // The production bridge reuses the existing MoE page-fault pool. This
        // standalone cache test observes its raw ranges and touches them without
        // linking the unrelated host-FFN engine or needing mapped model files.
        if (experts.empty()) throw std::runtime_error("Cache prefetch buffers are missing");
        volatile std::uint8_t sink = 0;
        for (std::size_t i = experts.size(); i-- > 0;)
        {
            for (std::size_t part = 3; part-- > 0;)
            {
                const auto& range = experts[i][part];
                if (range.first == nullptr || range.second == 0)
                    throw std::runtime_error("Cache prefetch buffer is empty");
                sink ^= range.first[0];
                sink ^= range.first[range.second - 1];
                consume(i, part);
            }
            // Reverse expert/projection completion; slots/IDs must still match.
        }
        (void)sink;
        ++g_test_prefetch_calls;
        return std::chrono::steady_clock::duration::zero();
    }
#endif
}

namespace
{
    void require(bool condition, const char* message)
    {
        if (!condition) throw std::runtime_error(message);
    }

    void require_operator_batch_shape(int rows, int selected)
    {
        const auto& shape = tsg::g_test_operator_shape;
        require(shape.valid && shape.projections == 3 && shape.rows == rows && shape.selected == selected
            && shape.weighted == 1 && shape.adds == selected - 1,
            "Actual cache graph does not contain three exact-width batched expert operators and one batched reduction");
    }

    void set_environment(const char* key, const char* value)
    {
#if defined(_WIN32)
        _putenv_s(key, value);
#else
        setenv(key, value, 1);
#endif
    }

    void prefetch_policy_checks()
    {
        tsg::ExpertPrefetchPolicy policy;
        constexpr std::size_t bytes = 16 << 20;
        using namespace std::chrono;
        require(!policy.should_prefetch(0, 0) && !policy.should_prefetch(1, bytes)
            && !policy.should_prefetch(2, 1024), "Small demand dispatched the read pool");
        require(policy.should_prefetch(8, bytes), "Cold demand did not enable parallel reads");
        policy.observe_read(bytes, milliseconds(20));
        require(policy.should_prefetch(8, bytes), "Slow source reads disabled prefetch");
        policy.observe_read(bytes, microseconds(100));
        require(!policy.should_prefetch(8, bytes), "Resident pages kept dispatching workers");
        for (int i = 0; i < 100; ++i) policy.observe_upload(bytes, microseconds(500));
        require(!policy.should_prefetch(8, bytes), "Fast uploads repeatedly re-armed prefetch");
        policy.observe_upload(bytes, milliseconds(30));
        require(policy.should_prefetch(8, bytes), "New expensive reads did not re-arm prefetch");
        require(!policy.should_prefetch(1, bytes), "Re-armed policy lost its minimum work guard");
        tsg::ExpertPrefetchPolicy reloaded;
        require(reloaded.should_prefetch(8, bytes), "Reload inherited another mapping's hot state");
    }

    void prefill_limit_checks()
    {
        const char* invalid_values[] = {nullptr, "", "0", "1", "8", "129", "+64", " 64", "64 ",
                                       "-1", "18446744073709551616", "64x"};
        for (const char* invalid : invalid_values)
            require(tsg::host_moe_cache_prefill_rows(invalid) == TSG_PRECISION_DECODE_COLUMNS,
                "Invalid/unset wide-prefill limit changed the default decode width");
        for (const int value : {9, 31, 63, 64, 88, 128})
            require(tsg::host_moe_cache_prefill_rows(std::to_string(value).c_str()) == value,
                "Bounded opt-in prefill row limit was not parsed exactly");
    }

    struct Stats
    {
        std::int64_t reserved = 0, budget = 0, hits = 0, misses = 0, calls = 0;
        Stats()
        {
            require(TSGgml_HostMoeExpertCacheStats(&reserved, &budget, &hits, &misses, &calls) == 1,
                "Could not read expert-cache engagement counters");
            require(reserved >= 0 && budget >= 0 && reserved <= budget,
                "Expert-cache allocations exceeded the explicit byte ceiling");
        }
    };

    struct ExecutionStats {
        std::uint64_t submissions = 0, joint = 0, rows = 0;
        ExecutionStats() { require(TSGgml_HostMoeExpertCacheExecutionStats(&submissions, &joint, &rows) == 1,
            "Missing exact expert graph submission counts"); }
    };

    constexpr int hidden = 256, ff = 256, experts = 16, used = 2;

    void scope_api_checks()
    {
        const auto original = Stats();
        auto env_value = [](const char* key) {
            const char* value = std::getenv(key); return std::string(value ? value : "");
        };
        const auto env_budget = env_value("TS_HOST_MOE_EXPERT_CACHE_MB");
        const auto env_layers = env_value("TS_HOST_MOE_EXPERT_CACHE_LAYERS");
        const auto env_rows = env_value("TS_HOST_MOE_EXPERT_CACHE_PREFILL_ROWS");
        std::uint64_t token = 99, second = 99;
        std::int64_t released = 99;
        require(TSGgml_BeginHostMoeExpertCacheScope(-1, 64, 4, &token) == 0 && token == 0,
            "Negative scope capacity did not fail before ownership");
        require(TSGgml_BeginHostMoeExpertCacheScope(1, 129, 4, &token) == 0 && token == 0
            && TSGgml_BeginHostMoeExpertCacheScope(1, 64, 0, &token) == 0 && token == 0
            && TSGgml_BeginHostMoeExpertCacheScope(1, 64, 4, nullptr) == 0,
            "Invalid scope settings or null output acquired ownership");
        require(TSGgml_BeginHostMoeExpertCacheScope(8 << 20, 64, 4, &token) == 1 && token != 0,
            "Valid native cache scope did not acquire settings");
        require(Stats().budget == (8 << 20) && Stats().reserved == 0,
            "Scoped budget was not visible through existing counters");
        require(TSGgml_BeginHostMoeExpertCacheScope(1, 1, 1, &second) == 0 && second == 0,
            "Second native cache scope replaced the live owner");
        require(TSGgml_UpdateHostMoeExpertCacheScope(token, (8 << 20) + 1, &released) == 0 && released == 0
            && TSGgml_UpdateHostMoeExpertCacheScope(token + 1, 0, &released) == 0 && released == 0
            && Stats().budget == (8 << 20), "Invalid update changed the scope ceiling");
        require(TSGgml_UpdateHostMoeExpertCacheScope(token, 0, &released) == 1 && released == 0
            && Stats().budget == 0, "Zero update did not disable an empty scope");
        require(TSGgml_UpdateHostMoeExpertCacheScope(token, 8 << 20, &released) == 1
            && TSGgml_TrimHostMoeExpertCacheScope(token, 0, &released) == 1 && released == 0
            && Stats().budget == (8 << 20), "Trim changed capacity or valid regrowth failed");
        require(TSGgml_EndHostMoeExpertCacheScope(token + 1, &released) == 0 && released == 0
            && Stats().budget == (8 << 20), "A stale token ended another owner's scope");
        require(TSGgml_EndHostMoeExpertCacheScope(token, &released) == 1 && released == 0
            && Stats().budget == original.budget, "Native scope did not restore its original budget");
        require(TSGgml_EndHostMoeExpertCacheScope(token, &released) == 0 && released == 0,
            "Ended native token remained reusable");
        require(TSGgml_BeginHostMoeExpertCacheScope(0, 1, 1, &second) == 1 && second != token,
            "A clean subsequent scope could not acquire independent ownership");
        require(TSGgml_TrimHostMoeExpertCacheScope(token, 0, &released) == 0
            && TSGgml_EndHostMoeExpertCacheScope(second, &released) == 1,
            "A stale token touched a subsequent scope");
        require(env_budget == env_value("TS_HOST_MOE_EXPERT_CACHE_MB")
            && env_layers == env_value("TS_HOST_MOE_EXPERT_CACHE_LAYERS")
            && env_rows == env_value("TS_HOST_MOE_EXPERT_CACHE_PREFILL_ROWS"),
            "Native scoped configuration mutated an environment variable");
        std::puts("PASS: native scope C ABI, validation, exclusive/stale ownership, regrowth and environment restoration");
    }

    struct SharedLedger
    {
        struct Allocation { std::int64_t bytes; bool committed = false; bool host = false; };
        std::int64_t capacity = 2 << 20, pending = 0, committed = 0;
        std::int64_t host_capacity = 32 << 20, host_pending = 0, host_committed = 0;
        std::uint64_t next = 0, reserve_calls = 0, commit_calls = 0, release_calls = 0;
        bool fail_commit = false, valid = true, attached = false, host_enabled = false;
        std::map<std::uint64_t, Allocation> allocations;

        ~SharedLedger()
        {
            // Keep callback context alive during exception unwinding too.
            if (attached)
            {
                tsg::host_moe_expert_cache_release();
                tsg::SharedCacheCharge::detach(this);
            }
        }

        static std::uint64_t reserve(void* context, int rank, int kind, std::int64_t bytes)
        {
            auto& l = *static_cast<SharedLedger*>(context);
            ++l.reserve_calls;
            if (rank != 0 || (kind != 4 && !(kind == 3 && l.host_enabled)) || bytes <= 0) { l.valid = false; return 0; }
            const bool host = kind == 3;
            if (bytes > (host ? l.host_capacity - l.host_pending - l.host_committed : l.capacity - l.pending - l.committed)) return 0;
            const auto token = ++l.next;
            l.allocations.emplace(token, Allocation{bytes, false, host});
            (host ? l.host_pending : l.pending) += bytes;
            return token;
        }
        static int commit(void* context, std::uint64_t token)
        {
            auto& l = *static_cast<SharedLedger*>(context);
            ++l.commit_calls;
            auto found = l.allocations.find(token);
            if (found == l.allocations.end() || found->second.committed) { l.valid = false; return 0; }
            const auto physical = found->second.host ? tsg::host_moe_expert_cache_test_file_workspace_bytes()
                : tsg::host_moe_expert_cache_test_physical_bytes();
            if (physical == 0) { l.valid = false; return 0; }
            if (l.fail_commit) return 0;
            auto& allocation = found->second;
            (allocation.host ? l.host_pending : l.pending) -= allocation.bytes;
            (allocation.host ? l.host_committed : l.committed) += allocation.bytes;
            allocation.committed = true;
            return 1;
        }
        static void release(void* context, std::uint64_t token)
        {
            auto& l = *static_cast<SharedLedger*>(context);
            ++l.release_calls;
            auto found = l.allocations.find(token);
            // These ownership fixtures keep one cache entry alive at a time;
            // the separate full-resident oracle is not charged to this cache.
            if (found == l.allocations.end()) { l.valid = false; return; }
            if ((found->second.host ? tsg::host_moe_expert_cache_test_file_workspace_bytes()
                    : tsg::host_moe_expert_cache_test_physical_bytes()) != 0)
            { l.valid = false; return; }
            auto& allocation = found->second;
            (allocation.host ? (allocation.committed ? l.host_committed : l.host_pending)
                : (allocation.committed ? l.committed : l.pending)) -= allocation.bytes;
            l.allocations.erase(found);
        }
        bool attach(bool include_host = false)
        {
            host_enabled = include_host;
            const bool result = tsg::SharedCacheCharge::attach(this, reserve, commit, release, false, include_host);
            if (result) attached = true;
            return result;
        }
        bool detach()
        {
            if (!tsg::SharedCacheCharge::detach(this)) return false;
            attached = false;
            return true;
        }
        void check_empty()
        {
            require(valid && allocations.empty() && pending == 0 && committed == 0 && host_pending == 0 && host_committed == 0,
                "Expert-cache shared credit escaped physical ownership or rollback");
            require(tsg::host_moe_expert_cache_test_physical_bytes() == 0 && Stats().reserved == 0,
                "Expert-cache rollback retained physical or private-accounted storage");
        }
    };

    struct Weights
    {
        tsg::HostMoeSegment segment;
        std::array<std::vector<std::uint8_t>, 3> quantized;
        std::array<ggml_type, 3> types{GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0};

        void fill(float seed, int expert_count = experts, int selected_count = used)
        {
            std::vector<float> values(static_cast<std::size_t>(hidden) * ff * expert_count);
            std::vector<float> importance(hidden, 1.0f);
            for (int part = 0; part < 3; ++part)
            {
                for (std::size_t i = 0; i < values.size(); ++i)
                    values[i] = 0.04f * std::sin(seed + float(i % 7919) * 0.013f + part * 0.7f);
                const auto bytes = ggml_row_size(types[part], hidden) * ff * expert_count;
                quantized[part].resize(bytes);
                const auto written = ggml_quantize_chunk(types[part], values.data(),
                    quantized[part].data(), 0, ff * expert_count, hidden,
                    ggml_quantize_requires_imatrix(types[part]) ? importance.data() : nullptr);
                require(written == bytes, "Synthetic quantized expert layout is incomplete");
            }
            segment.layer = 0;
            segment.hidden = hidden; segment.n_ff = ff; segment.num_experts = expert_count;
            segment.n_used = selected_count; segment.seq_len = 1; segment.activation = 0;
            segment.gate_data = quantized[0].data(); segment.gate_type = types[0];
            segment.gate_ne0 = hidden; segment.gate_ne1 = ff; segment.gate_bytes = quantized[0].size();
            segment.up_data = quantized[1].data(); segment.up_type = types[1];
            segment.up_ne0 = hidden; segment.up_ne1 = ff; segment.up_bytes = quantized[1].size();
            segment.down_data = quantized[2].data(); segment.down_type = types[2];
            segment.down_ne0 = ff; segment.down_ne1 = hidden; segment.down_bytes = quantized[2].size();
        }
    };

    void fallback_checks(Weights& weights)
    {
        std::array<float, hidden> x{}, result{};
        std::array<std::int32_t, used> ids{0, 1};
        std::array<float, used> routes{0.6f, 0.4f};
        auto hm = weights.segment;
        const auto run = [&] { return tsg::host_moe_cached_experts(hm, x.data(), ids.data(),
            routes.data(), result.data(), "qwen4exp fused token span"); };
        hm.seq_len = static_cast<int>(TSG_PRECISION_DECODE_COLUMNS) + 1;
        require(run() == 0, "Long prefill must retain the existing offload path");
        hm.seq_len = 1; hm.activation = 1;
        require(run() == 0, "Non-SiLU activation must retain the existing offload path");
        hm.activation = 0; hm.up_bias = routes.data();
        require(run() == 0, "Biased experts must retain the existing offload path");
        hm.up_bias = nullptr;
        require(tsg::host_moe_cached_experts(hm, x.data(), ids.data(), routes.data(), result.data(),
            "Other model") == 0, "The opt-in cache must not change other model families");
    }

#ifdef TSG_GGML_USE_CUDA
    struct DeviceInputs
    {
        ggml_context* context = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_tensor* x = nullptr;
        ggml_tensor* routes = nullptr;
        ggml_tensor* output = nullptr;

        explicit DeviceInputs(int rows = static_cast<int>(TSG_PRECISION_DECODE_COLUMNS), int selected_count = used)
        {
            context = ggml_init({ggml_tensor_overhead() * 4, nullptr, true});
            require(context != nullptr, "Could not allocate device-bridge input metadata");
            x = ggml_new_tensor_2d(context, GGML_TYPE_F32, hidden, rows);
            routes = ggml_new_tensor_2d(context, GGML_TYPE_F32, selected_count, rows);
            output = ggml_new_tensor_2d(context, GGML_TYPE_F32, hidden, rows);
            buffer = ggml_backend_alloc_ctx_tensors(context, g_backend);
            require(buffer != nullptr, "Could not allocate device-bridge inputs");
        }

        ~DeviceInputs()
        {
            if (buffer) ggml_backend_buffer_free(buffer);
            if (context) ggml_free(context);
        }
    };

    struct FullResidentGraph
    {
        ggml_context* context = nullptr;
        ggml_gallocr_t allocator = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* x = nullptr;
        ggml_tensor* ids = nullptr;
        ggml_tensor* routes = nullptr;
        ggml_tensor* output = nullptr;

        explicit FullResidentGraph(const Weights& weights)
        {
            context = ggml_init({ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(256, false), nullptr, true});
            require(context != nullptr, "Could not allocate reference tensor metadata");
            auto* gate = ggml_new_tensor_3d(context, weights.types[0], hidden, ff, weights.segment.num_experts);
            auto* up = ggml_new_tensor_3d(context, weights.types[1], hidden, ff, weights.segment.num_experts);
            auto* down = ggml_new_tensor_3d(context, weights.types[2], ff, hidden, weights.segment.num_experts);
            for (auto* t : {gate, up, down}) { ggml_set_input(t); ggml_set_output(t); }
            x = ggml_new_tensor_3d(context, GGML_TYPE_F32, hidden, 1, 1);
            ids = ggml_new_tensor_2d(context, GGML_TYPE_I32, weights.segment.n_used, 1);
            routes = ggml_new_tensor_3d(context, GGML_TYPE_F32, 1, weights.segment.n_used, 1);
            for (auto* t : {x, ids, routes}) ggml_set_input(t);
            // The outer Qwen host-MoE boundary keeps these source tensors live
            // as graph outputs. Mirror that lifetime when testing D2D reads
            // after the full-resident reference graph has finished computing.
            ggml_set_output(x);
            ggml_set_output(routes);
            auto* g = ggml_mul_mat_id(context, gate, x, ids);
            auto* u = ggml_mul_mat_id(context, up, x, ids);
            auto* h = ggml_mul(context, ggml_silu(context, g), u);
            auto* parts = ggml_mul_mat_id(context, down, h, ids);
            ggml_set_output(parts);
            auto* weighted = ggml_mul(context, parts, routes);
            auto* sum = ggml_view_2d(context, weighted, hidden, 1, weighted->nb[2], 0);
            for (int k = 1; k < weights.segment.n_used; ++k)
                sum = ggml_add(context, sum, ggml_view_2d(context, weighted, hidden, 1,
                    weighted->nb[2], static_cast<std::size_t>(k) * weighted->nb[1]));
            output = ggml_cont(context, sum);
            ggml_set_output(output);
            graph = ggml_new_graph_custom(context, 256, false);
            ggml_build_forward_expand(graph, output);
            allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
            require(allocator != nullptr && ggml_gallocr_alloc_graph(allocator, graph),
                "Could not allocate full-resident reference graph");
            ggml_backend_tensor_set(gate, weights.quantized[0].data(), 0, weights.quantized[0].size());
            ggml_backend_tensor_set(up, weights.quantized[1].data(), 0, weights.quantized[1].size());
            ggml_backend_tensor_set(down, weights.quantized[2].data(), 0, weights.quantized[2].size());
        }

        ~FullResidentGraph()
        {
            if (allocator) ggml_gallocr_free(allocator);
            if (context) ggml_free(context);
        }

        template<std::size_t Selected>
        std::array<float, hidden> compute(const std::array<float, hidden>& activation,
            const std::array<std::int32_t, Selected>& selected, const std::array<float, Selected>& probabilities)
        {
            require(Selected == static_cast<std::size_t>(ids->ne[0]), "Reference selected-expert geometry differs");
            ggml_backend_tensor_set(x, activation.data(), 0, sizeof(activation));
            ggml_backend_tensor_set(ids, selected.data(), 0, sizeof(selected));
            ggml_backend_tensor_set(routes, probabilities.data(), 0, sizeof(probabilities));
            require(ggml_backend_graph_compute(g_backend, graph) == GGML_STATUS_SUCCESS,
                "Full-resident reference expert graph failed");
            std::array<float, hidden> result{};
            ggml_backend_tensor_get(output, result.data(), 0, sizeof(result));
            return result;
        }
    };

    void wide_prefill_checks(Weights& weights)
    {
        tsg::host_moe_expert_cache_release();
        SharedLedger ledger;
        require(ledger.attach(), "Could not attach wide-prefill budget fixture");
        FullResidentGraph reference(weights);
        DeviceInputs device(128);
        std::int64_t retained_bytes = 0;
        std::uint64_t retained_reservations = 0;
        for (int rows : {9, 31, 63, 64, 88, 128})
        {
            auto hm = weights.segment;
            hm.seq_len = rows;
            hm.moe_in = device.x; hm.weights = device.routes; hm.moe_out = device.output;
            std::vector<float> inputs(rows * hidden), routes(rows * used), expected(rows * hidden);
            std::vector<float> actual(rows * hidden, -999.0f);
            std::vector<std::int32_t> selected(rows * used);
            for (int row = 0; row < rows; ++row)
            {
                std::array<float, hidden> x{};
                for (int i = 0; i < hidden; ++i)
                    x[i] = inputs[row * hidden + i] = std::sin(i * .07f + row + rows * .2f);
                std::array<std::int32_t, used> ids{(row * 5 + rows) % experts, (row * 5 + rows + 1) % experts};
                if (row % 3 == 0) ids[1] = ids[0];
                std::array<float, used> probabilities{.25f + .001f * row, .75f - .001f * row};
                std::copy(ids.begin(), ids.end(), selected.begin() + row * used);
                std::copy(probabilities.begin(), probabilities.end(), routes.begin() + row * used);
                const auto result = reference.compute(x, ids, probabilities);
                require(std::all_of(result.begin(), result.end(), [](float v) { return std::isfinite(v); }),
                    "Full-resident wide-prefill oracle produced a non-finite row");
                std::copy(result.begin(), result.end(), expected.begin() + row * hidden);
            }
            auto call = [&](int bridge) {
                return tsg::host_moe_cached_experts(hm, bridge ? nullptr : inputs.data(), selected.data(),
                    bridge ? nullptr : routes.data(), bridge == 2 ? nullptr : actual.data(),
                    "qwen4exp fused token span");
            };
            ggml_backend_tensor_set(device.x, inputs.data(), 0, inputs.size() * sizeof(float));
            ggml_backend_tensor_set(device.routes, routes.data(), 0, routes.size() * sizeof(float));
            auto rejected = [&](int bridge) {
                std::fill(actual.begin(), actual.end(), -999.0f);
                ggml_backend_tensor_set(device.output, actual.data(), 0, actual.size() * sizeof(float));
                const auto before = Stats();
                require(call(bridge) == 0, "Unsupported wide-prefill input did not decline before execution");
                const auto after = Stats();
                require(before.calls == after.calls && before.hits == after.hits && before.misses == after.misses
                    && before.reserved == after.reserved, "Rejected wide-prefill input changed cache state");
                if (bridge == 2) ggml_backend_tensor_get(device.output, actual.data(), 0, actual.size() * sizeof(float));
                require(std::all_of(actual.begin(), actual.end(), [](float v) { return v == -999.0f; }),
                    "Rejected wide-prefill input overwrote an output row");
            };
            // Validate the whole block, including its last row, before budget
            // admission or mutation. The three bridge modes obey one contract.
            const auto last_id = selected.back();
            selected.back() = experts;
            for (int bridge : {0, 1, 2}) rejected(bridge);
            selected.back() = last_id;
            const float last_route = routes.back();
            for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                                  std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()})
            {
                routes.back() = invalid;
                ggml_backend_tensor_set(device.routes, routes.data(), 0, routes.size() * sizeof(float));
                for (int bridge : {0, 1, 2}) rejected(bridge);
            }
            routes.back() = last_route;
            ggml_backend_tensor_set(device.routes, routes.data(), 0, routes.size() * sizeof(float));
            if (rows == 9)
            {
                ledger.capacity = 0;
                rejected(0); ledger.check_empty();
                ledger.capacity = 2 << 20;
                ledger.fail_commit = true;
                rejected(0); ledger.check_empty();
                ledger.fail_commit = false;
            }
            for (int bridge : {0, 1, 2})
            {
                const auto before = Stats();
                const auto drains = tsg::host_moe_expert_cache_test_output_drains();
                require(call(bridge) == 1, "Explicit wide-prefill cache did not engage");
                require(tsg::host_moe_expert_cache_test_output_drains() == drains,
                    "Successful wide-prefill handoff introduced an extra output drain");
                if (bridge == 2)
                {
                    tsg::sync_backend(g_backend);
                    ggml_backend_tensor_get(device.output, actual.data(), 0, actual.size() * sizeof(float));
                }
                require(std::memcmp(actual.data(), expected.data(), expected.size() * sizeof(float)) == 0,
                    "Wide-prefill rows differ from the independent scalar resident oracle");
                require(Stats().calls == before.calls + rows, "Wide-prefill cache skipped or repeated a row");
                if (!retained_bytes)
                {
                    retained_bytes = Stats().reserved;
                    retained_reservations = ledger.reserve_calls;
                }
                require(Stats().reserved == retained_bytes && ledger.reserve_calls == retained_reservations,
                    "Wider prefill acquired another cache allocation or shared reservation");
                require(ledger.pending == 0 && ledger.valid && ledger.committed == retained_bytes,
                    "Wide-prefill graph escaped its shared owner");
            }
            if (rows == 128)
            {
                // These failures are checked before even reading the caller's
                // row arrays, so the same 128-row buffers safely test the bound.
                hm.seq_len = 129; rejected(0); rejected(2); hm.seq_len = 128;
                tsg::g_device_count.store(2); rejected(0); tsg::g_device_count.store(1);
                tsg::g_tp_global_degree.store(2); rejected(0); tsg::g_tp_global_degree.store(0);
                tsg::g_active_rank = 1; rejected(0); tsg::g_active_rank = 0;
                // A backend read callback error is an execution failure, not
                // an unsupported layout that may fall back to CPU arithmetic.
                const auto before_read_failure = Stats();
                tsg::host_moe_expert_cache_test_fail_next(5);
                bool read_failed = false;
                try { call(2); }
                catch (const std::runtime_error& error)
                {
                    read_failed = std::string(error.what()).find("routing download failed") != std::string::npos;
                }
                const auto after_read_failure = Stats();
                require(read_failed && before_read_failure.calls == after_read_failure.calls
                    && before_read_failure.hits == after_read_failure.hits
                    && before_read_failure.misses == after_read_failure.misses
                    && before_read_failure.reserved == after_read_failure.reserved,
                    "Failed device routing read fell back or mutated cache state");
                tsg::host_moe_expert_cache_test_fail_next(4);
                const auto before = Stats();
                const auto drains_before = tsg::host_moe_expert_cache_test_output_drains();
                bool interrupted = false;
                try { call(2); }
                catch (const std::runtime_error& error)
                {
                    interrupted = std::string(error.what()).find("after a completed row") != std::string::npos;
                }
                require(interrupted && Stats().calls == before.calls + 1,
                    "Injected wide-prefill failure was hidden, retried, or executed later rows");
                require(tsg::host_moe_expert_cache_test_output_drains() == drains_before + 1,
                    "Wide-prefill exception returned before its pending output drain completed");
                // Error cleanup has already drained the pending output. Read
                // without an explicit sync, then retry all rows from fresh IDs.
                ggml_backend_tensor_get(device.output, actual.data(), 0, hidden * sizeof(float));
                require(std::memcmp(actual.data(), expected.data(), hidden * sizeof(float)) == 0,
                    "Failure cleanup did not preserve its completed device output row");
                require(call(2) == 1, "Wide-prefill cache did not recover after an interrupted block");
                // Success queues the final D2D copy. Immediate trim must drain
                // it before releasing graph storage and returning shared credit.
                require(TSGgml_TrimHostMoeExpertCache(0) == retained_bytes, "Wide-prefill trim retained cache storage");
                ledger.check_empty();
                ggml_backend_tensor_get(device.output, actual.data(), 0, actual.size() * sizeof(float));
                require(std::memcmp(actual.data(), expected.data(), expected.size() * sizeof(float)) == 0,
                    "Queued wide-prefill output changed during physical teardown");
            }
        }
        require(ledger.detach(), "Wide-prefill owners prevented budget detach after trim");
        std::puts("PASS: explicit 9/31/63/64/88/128-row scalar expert oracle, bridges, invalid routes, admission, interruption and trim");
    }

    template<std::size_t Selected>
    void wide_route_capacity_checks(const std::array<ggml_type, 3>& types)
    {
        tsg::host_moe_expert_cache_release();
        Weights weights;
        weights.types = types;
        weights.fill(0.71f, (std::max)(experts, static_cast<int>(Selected)), static_cast<int>(Selected));
        SharedLedger ledger;
        ledger.capacity = 32 << 20;
        require(ledger.attach(), "Could not attach route-capacity fixture");
        FullResidentGraph reference(weights);
        // Extra capacity makes every rejected request safe even if its guard
        // regresses. Rejection must still precede cache/output mutation.
        DeviceInputs device(129, static_cast<int>(Selected));
        std::vector<float> inputs(129 * hidden), routes(129 * Selected, 1.0f / Selected);
        std::vector<std::int32_t> selected(129 * Selected);
        std::vector<float> expected(129 * hidden), actual(129 * hidden, -999.0f);
        for (int row = 0; row < 129; ++row)
        {
            std::array<float, hidden> x{};
            std::array<std::int32_t, Selected> ids{};
            std::array<float, Selected> probabilities{};
            for (int i = 0; i < hidden; ++i)
                x[i] = inputs[row * hidden + i] = std::sin(i * .031f + row * .17f);
            for (std::size_t k = 0; k < Selected; ++k)
            {
                ids[k] = selected[row * Selected + k] = (row + static_cast<int>(k)) % weights.segment.num_experts;
                probabilities[k] = routes[row * Selected + k];
            }
            const auto result = reference.compute(x, ids, probabilities);
            require(std::all_of(result.begin(), result.end(), [](float v) { return std::isfinite(v); }),
                "Selected-expert capacity reference is nonfinite");
            std::copy(result.begin(), result.end(), expected.begin() + row * hidden);
        }
        ggml_backend_tensor_set(device.x, inputs.data(), 0, inputs.size() * sizeof(float));
        ggml_backend_tensor_set(device.routes, routes.data(), 0, routes.size() * sizeof(float));
        auto hm = weights.segment;
        hm.moe_in = device.x; hm.weights = device.routes; hm.moe_out = device.output;
        auto call = [&](int bridge) {
            return tsg::host_moe_cached_experts(hm, bridge ? nullptr : inputs.data(), selected.data(),
                bridge ? nullptr : routes.data(), bridge == 2 ? nullptr : actual.data(), "qwen4exp fused token span");
        };
        const std::vector<int> counts = Selected == 10 ? std::vector<int>{88, 128} : std::vector<int>{64};
        std::int64_t reserved = 0;
        for (const int rows : counts) for (const int bridge : {0, 1, 2})
        {
            hm.seq_len = rows;
            const auto before = Stats();
            require(call(bridge) == 1, "Bounded selected-expert geometry did not engage");
            if (bridge == 2) ggml_backend_tensor_get(device.output, actual.data(), 0, rows * hidden * sizeof(float));
            require(std::memcmp(actual.data(), expected.data(), rows * hidden * sizeof(float)) == 0,
                "Bounded selected-expert geometry differs from scalar resident oracle");
            require(Stats().calls == before.calls + rows, "Selected-expert boundary skipped/repeated rows");
            if (!reserved) reserved = Stats().reserved;
            require(Stats().reserved == reserved && ledger.pending == 0 && ledger.committed == reserved && ledger.valid,
                "Larger routed block changed retained scalar cache ownership");
        }
        // 64*65 exceeds the independent4096 route bound despite an allowed row
        // count.10*129 instead exceeds the128 row bound with only1290 routes.
        hm.seq_len = Selected == 64 ? 65 : 129;
        for (const int bridge : {0, 1, 2})
        {
            std::fill(actual.begin(), actual.end(), -999.0f);
            ggml_backend_tensor_set(device.output, actual.data(), 0, actual.size() * sizeof(float));
            const auto before = Stats();
            require(call(bridge) == 0, "Independent row/route bound did not decline");
            const auto after = Stats();
            require(before.calls == after.calls && before.hits == after.hits && before.misses == after.misses
                    && before.reserved == after.reserved, "Overbound routed block mutated the cache");
            if (bridge == 2) ggml_backend_tensor_get(device.output, actual.data(), 0, actual.size() * sizeof(float));
            require(std::all_of(actual.begin(), actual.end(), [](float v) { return v == -999.0f; }),
                "Overbound routed block overwrote output");
        }
        require(TSGgml_TrimHostMoeExpertCache(0) == reserved, "Route boundary trim retained cache storage");
        ledger.check_empty();
        require(ledger.detach(), "Route boundary left live budget owners");
        std::printf("PASS: selected=%zu, exact scalar CUDA oracle and independent128-row/4096-route limits\n", Selected);
    }

    void shared_budget_checks(Weights& weights)
    {
        using tsg::SharedCacheCharge;
        SharedLedger ledger;
        std::array<float, hidden> x{}, output{};
        for (int i = 0; i < hidden; ++i) x[i] = std::sin(i * 0.031f);
        std::array<std::int32_t, used> ids{0, 1};
        std::array<float, used> routes{0.6f, 0.4f};
        FullResidentGraph reference(weights);
        const auto expected = reference.compute(x, ids, routes);
        auto call = [&] {
            return tsg::host_moe_cached_experts(weights.segment, x.data(), ids.data(), routes.data(),
                output.data(), "qwen4exp fused token span");
        };
        require(call() == 1 && Stats().reserved > 0, "Unconfigured expert cache did not engage");
        require(!ledger.attach(), "Budget attach adopted an already-live uncharged expert graph");
        tsg::host_moe_expert_cache_release();
        require(ledger.attach(), "Could not attach budget after expert-cache physical teardown");

        // Quota refusal must precede even the first native-allocation hook.
        ledger.capacity = 0;
        output.fill(-999.0f);
        tsg::host_moe_expert_cache_test_fail_next(1);
        require(call() == 0, "Shared quota refusal did not retain the normal fallback");
        ledger.check_empty();
        require(std::all_of(output.begin(), output.end(), [](float value) { return value == -999.0f; }),
            "Refused expert-cache creation overwrote caller output");
        ledger.capacity = 2 << 20;
        require(call() == 0, "Denied quota consumed the later allocation-failure hook");
        ledger.check_empty();

        for (int stage : {1, 2, 3})
        {
            tsg::host_moe_expert_cache_test_fail_next(stage);
            require(call() == 0, "Injected expert-cache allocation/publication failure was ignored");
            ledger.check_empty();
        }
        ledger.fail_commit = true;
        require(call() == 0, "Refused shared commit published an expert-cache entry");
        ledger.check_empty();
        ledger.fail_commit = false;

        require(call() == 1, "Budgeted expert cache did not recover after allocation rollback");
        const auto initial = Stats();
        require(ledger.valid && ledger.pending == 0 && ledger.committed >= initial.reserved,
            "Committed expert graph is not covered by shared device credit");
        require(!ledger.detach(), "Detached callbacks with a live expert graph");
        const auto reservations = ledger.reserve_calls;
        const auto commits = ledger.commit_calls;
        require(call() == 1 && ledger.reserve_calls == reservations && ledger.commit_calls == commits,
            "Warm expert reuse acquired duplicate shared reservations");
        const auto exact_bytes = ledger.committed;
        tsg::host_moe_expert_cache_on_drop(weights.segment.gate_data);
        ledger.check_empty();
        ledger.capacity = exact_bytes - 1;
        require(call() == 1 && ledger.committed < exact_bytes,
            "Available shared credit could fit fewer slots but the cache fell back");
        require(std::memcmp(output.data(), expected.data(), sizeof(output)) == 0,
            "Shrinking the slot table changed expert arithmetic");
        require(ledger.committed <= ledger.capacity && ledger.valid,
            "Adaptive slot admission exceeded shared credit");
        const auto shrunk_reservations = ledger.reserve_calls;
        const auto shrunk_bytes = Stats().reserved;
        require(call() == 1 && ledger.reserve_calls == shrunk_reservations,
            "Warm shrunken table repeated admission or lost its weights");
        const auto before_trim = Stats();
        require(TSGgml_TrimHostMoeExpertCache(-1) == -1
            && TSGgml_TrimHostMoeExpertCache(before_trim.reserved) == 0
            && Stats().reserved == before_trim.reserved,
            "Invalid/no-op trim mutated a live cache");
        require(TSGgml_TrimHostMoeExpertCache(0) == shrunk_bytes,
            "Trim did not report all released accounted bytes");
        ledger.check_empty();
        require(Stats().calls == before_trim.calls && Stats().hits == before_trim.hits
            && Stats().misses == before_trim.misses,
            "Trim erased cumulative cache counters");

        // A two-expert checkpoint measures the same graph's minimum allocation
        // without assuming ggml padding, allocator rounding or graph reuse. The
        // full checkpoint must fit at that exact minimum, and never one byte less.
        ledger.capacity = 2 << 20;
        auto minimal = weights.segment;
        minimal.num_experts = used;
        minimal.gate_bytes = minimal.gate_bytes / experts * used;
        minimal.up_bytes = minimal.up_bytes / experts * used;
        minimal.down_bytes = minimal.down_bytes / experts * used;
        require(tsg::host_moe_cached_experts(minimal, x.data(), ids.data(), routes.data(),
            output.data(), "qwen4exp fused token span") == 1, "Could not measure minimum expert graph");
        const auto minimum_bytes = ledger.committed;
        require(minimum_bytes < exact_bytes, "Fixture did not have reducible weight slots");
        TSGgml_TrimHostMoeExpertCache(0);
        ledger.check_empty();
        ledger.capacity = minimum_bytes - 1;
        output.fill(-999.0f);
        require(call() == 0, "Minimum expert graph exceeded shared capacity by one byte");
        ledger.check_empty();
        require(std::all_of(output.begin(), output.end(), [](float v) { return v == -999.0f; }),
            "Rejected minimum graph changed output");
        ledger.capacity = minimum_bytes;
        require(call() == 1 && ledger.committed == minimum_bytes,
            "Full checkpoint did not shrink to the exact two-expert minimum");
        require(std::memcmp(output.data(), expected.data(), sizeof(output)) == 0,
            "Minimum slot table changed full-resident results");
        TSGgml_TrimHostMoeExpertCache(0);
        ledger.check_empty();
        // The same admission must respect observed VRAM even with abundant
        // shared credit. Headroom and one-byte boundary are not CUDA OOM tests.
        constexpr std::size_t physical_headroom = std::size_t(512) << 20;
        ledger.capacity = exact_bytes;
        const auto before_physical_refusal = ledger.reserve_calls;
        tsg::host_moe_expert_cache_test_available_bytes(physical_headroom + minimum_bytes - 1);
        require(call() == 0 && ledger.reserve_calls == before_physical_refusal,
            "Insufficient physical availability reached shared admission or allocated a graph");
        ledger.check_empty();
        tsg::host_moe_expert_cache_test_available_bytes(physical_headroom + minimum_bytes);
        require(call() == 1 && ledger.committed == minimum_bytes,
            "Physical pressure did not reduce slots to the minimum fitting graph");
        require(std::memcmp(output.data(), expected.data(), sizeof(output)) == 0,
            "Physical-pressure admission changed expert arithmetic");
        TSGgml_TrimHostMoeExpertCache(0);
        ledger.check_empty();
        tsg::host_moe_expert_cache_test_available_bytes(std::numeric_limits<std::size_t>::max());
        ledger.capacity = exact_bytes;
        require(call() == 1, "Budgeted expert graph failed to reload after invalidation");
        tsg::host_moe_expert_cache_release();
        ledger.check_empty();
        require(ledger.detach(), "Cannot detach after all expert-cache buffers were freed");
        std::printf("PASS: adaptive expert slots, minimum=%lld initial=%lld; shared quota, rollback, warm reuse and trim free-before-credit\n",
            static_cast<long long>(minimum_bytes), static_cast<long long>(exact_bytes));
    }

    void scoped_cuda_checks(Weights& weights)
    {
        FullResidentGraph reference(weights);
        std::array<float, hidden> x{}, actual{};
        for (int i = 0; i < hidden; ++i) x[i] = std::cos(i * 0.019f);
        std::array<std::int32_t, used> ids{0, 1};
        std::array<float, used> routes{0.4f, 0.6f};
        const auto expected = reference.compute(x, ids, routes);
        auto hm = weights.segment;
        auto call = [&] { return tsg::host_moe_cached_experts(hm, x.data(), ids.data(), routes.data(),
            actual.data(), "qwen4exp fused token span"); };
        auto exact_call = [&] {
            require(call() == 1, "Configured CUDA cache did not engage");
            require(std::memcmp(actual.data(), expected.data(), sizeof(actual)) == 0,
                "Scoped CUDA cache changed full-resident expert results");
        };
        exact_call();
        std::uint64_t token = 99;
        require(TSGgml_BeginHostMoeExpertCacheScope(8 << 20, 9, 4, &token) == 0 && token == 0,
            "Scope adopted a live legacy cache instead of refusing it");
        tsg::host_moe_expert_cache_release();
        SharedLedger ledger;
        ledger.capacity = 8 << 20;
        require(ledger.attach(), "Could not attach scoped CUDA budget fixture");
        require(TSGgml_BeginHostMoeExpertCacheScope(8 << 20, 9, 4, &token) == 1,
            "Could not begin scoped CUDA configuration");
        exact_call();
        // SharedLedger checks physical-before-credit for one entry at a time.
        // This entry already fits the new TOTAL limit, but exceeds its new
        // PER-LAYER share; the update must retire it instead of only doing LRU.
        const auto live_entry = Stats();
        require(live_entry.reserved > (1 << 20) && live_entry.reserved < (4 << 20)
            && ledger.committed == live_entry.reserved,
            "Scoped entry did not exercise per-layer pressure with shared ownership");
        std::int64_t released = -1;
        require(TSGgml_UpdateHostMoeExpertCacheScope(token, 4 << 20, &released) == 1
            && released == live_entry.reserved && Stats().reserved == 0,
            "Scope shrink did not enforce the smaller per-layer allowance");
        ledger.check_empty();
        require(call() == 0, "Too-small per-layer allowance allocated a cache entry");
        ledger.check_empty();
        require(TSGgml_UpdateHostMoeExpertCacheScope(token, 8 << 20, &released) == 1 && released == 0,
            "Scoped CUDA budget could not regrow to its ceiling");
        exact_call();
        // Nine rows use the scoped row limit; ten must fail before reading the
        // short arrays. The numerical oracle uses the exact same row inputs.
        std::vector<float> rows(hidden * 9), probabilities(used * 9), output(hidden * 9, -999.0f);
        std::vector<std::int32_t> selected(used * 9);
        for (int i = 0; i < 9; ++i) {
            std::copy(x.begin(), x.end(), rows.begin() + i * hidden);
            std::copy(routes.begin(), routes.end(), probabilities.begin() + i * used);
            std::copy(ids.begin(), ids.end(), selected.begin() + i * used);
        }
        hm.seq_len = 9;
        require(tsg::host_moe_cached_experts(hm, rows.data(), selected.data(), probabilities.data(),
            output.data(), "qwen4exp fused token span") == 1, "Scoped prefill width did not override the default eight");
        for (int i = 0; i < 9; ++i)
            require(std::memcmp(output.data() + i * hidden, expected.data(), sizeof(expected)) == 0,
                "Scoped wide-prefill arithmetic differs from the resident oracle");
        hm.seq_len = 10;
        const auto before_reject = Stats();
        require(call() == 0 && Stats().calls == before_reject.calls && Stats().reserved == before_reject.reserved,
            "Rows above the scoped limit executed or changed cache state");
        hm.seq_len = 1;
        const auto before_trim = Stats();
        require(TSGgml_TrimHostMoeExpertCacheScope(token, 0, &released) == 1
            && released == before_trim.reserved && Stats().budget == (8 << 20),
            "Scoped trim did not release cache storage independently of capacity");
        ledger.check_empty();
        exact_call();
        require(TSGgml_EndHostMoeExpertCacheScope(token, &released) == 1 && released > 0
            && Stats().budget == (2 << 20) && Stats().reserved == 0,
            "Scope disposal did not free before restoring legacy settings");
        ledger.check_empty();
        hm.seq_len = 9;
        require(call() == 0, "Disposal did not restore the legacy default row limit");
        hm.seq_len = 1;
        exact_call(); // Original layer quota has also been restored (2 MiB / one layer).
        tsg::host_moe_expert_cache_release();
        ledger.check_empty();
        require(ledger.detach(), "Scoped cache disposal retained shared native ownership");
        std::puts("PASS: scoped CUDA cache, all-row exact oracle, per-layer shrink, regrowth, trim and restored defaults");
    }

    void trim_lru_checks(Weights& weights)
    {
        // Separate process: 8 MiB global ceiling / four partitions leaves enough
        // space for three independent 2 MiB entries, unlike the eviction fixture.
        FullResidentGraph reference(weights);
        std::array<float, hidden> x{}, actual{};
        for (int i = 0; i < hidden; ++i) x[i] = std::cos(i * 0.017f);
        std::array<std::int32_t, used> ids{0, 1};
        std::array<float, used> routes{0.4f, 0.6f};
        const auto expected = reference.compute(x, ids, routes);
        auto hm = weights.segment;
        auto call = [&](int layer) {
            hm.layer = layer;
            require(tsg::host_moe_cached_experts(hm, x.data(), ids.data(), routes.data(),
                actual.data(), "qwen4exp fused token span") == 1, "LRU fixture cache did not engage");
            require(std::memcmp(actual.data(), expected.data(), sizeof(actual)) == 0,
                "LRU fixture changed numerical results");
        };
        call(0);
        const auto bytes = Stats().reserved;
        call(1); call(2); call(0); // Oldest=1, then 2; most recently used=0.
        const auto before = Stats();
        require(before.reserved == bytes * 3, "LRU fixture did not retain three entries");
        require(TSGgml_TrimHostMoeExpertCache(bytes * 2) == bytes
            && Stats().reserved == bytes * 2, "Trim did not retire one whole LRU entry");
        call(2); call(0);
        require(Stats().misses == before.misses, "Trim evicted a recent entry instead of the oldest");
        call(1);
        require(Stats().misses == before.misses + used, "Trimmed layer did not reload every selected weight");

        // No synchronization after a queued device-output copy. Trim must drain
        // the source owner while leaving the external output and KV/model live.
        hm.moe_in = reference.x; hm.weights = reference.routes; hm.moe_out = reference.output;
        require(tsg::host_moe_cached_experts(hm, nullptr, ids.data(), nullptr,
            nullptr, "qwen4exp fused token span") == 1, "Queued trim output did not engage");
        const auto live = Stats();
        require(TSGgml_TrimHostMoeExpertCache(0) == live.reserved && Stats().reserved == 0,
            "Complete trim retained expert buffers");
        ggml_backend_tensor_get(reference.output, actual.data(), 0, sizeof(actual));
        require(std::memcmp(actual.data(), expected.data(), sizeof(actual)) == 0,
            "Trim freed a still-pending device copy source");
        require(Stats().calls == live.calls && Stats().hits == live.hits && Stats().misses == live.misses,
            "LRU trim lost cumulative statistics");
        call(0);
        tsg::host_moe_expert_cache_release();
        std::puts("PASS: three-entry LRU trim, recent hits retained, reload, queued output and unchanged arithmetic");
    }

#if defined(_WIN32)
    void file_read_checks(Weights& weights, bool joint = false)
    {
        if (joint) tsg::host_moe_expert_cache_test_file_slice_limit(65536);
        const std::string path = "host-moe-source-" + std::to_string(GetCurrentProcessId()) + ".bin";
        std::array<std::int64_t, 3> offsets{};
        {
            std::ofstream file(path, std::ios::binary | std::ios::trunc);
            require(bool(file), "Cannot create expert file fixture");
            for (int part = 0; part < 3; ++part) {
                file.write("nonzero-offset!", 15);
                offsets[part] = static_cast<std::int64_t>(file.tellp());
                file.write(reinterpret_cast<const char*>(weights.quantized[part].data()), weights.quantized[part].size());
            }
            require(bool(file), "Cannot write expert file fixture");
        }
        for (int part = 0; part < 3; ++part)
            require(TSGgml_RegisterHostFileSource(weights.quantized[part].data(), weights.quantized[part].size(),
                path.c_str(), offsets[part]) == 1, "Cannot register exact file extent");
        require(TSGgml_RegisterHostFileSource(weights.quantized[0].data(), weights.quantized[0].size(),
            path.c_str(), offsets[0]) == 0, "Duplicate source silently replaced an existing owner");
        bool rejected = false;
        try { tsg::ExpertFileSource invalid(path.c_str(), UINT64_MAX-1, 3); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "Overflowing file extent accepted");
        {
            tsg::ExpertFileSource source(path.c_str(), offsets[0], weights.quantized[0].size());
            std::array<std::uint8_t, 113> tail{};
            source.read(source.bytes()-tail.size(), tail.data(), tail.size());
            require(std::memcmp(tail.data(), weights.quantized[0].data()+source.bytes()-tail.size(), tail.size()) == 0,
                "Partial source tail or file offset changed bytes");
            rejected = false;
            try { source.read(source.bytes()-tail.size()+1, tail.data(), tail.size()); }
            catch (const std::runtime_error&) { rejected = true; }
            require(rejected, "Out-of-range file read accepted");
        }
        FullResidentGraph reference(weights);
        std::array<float, hidden> x{}, actual{};
        for (int i = 0; i < hidden; ++i) x[i] = std::sin(0.03f * i);
        std::array<float, used> routes{0.65f,0.35f};
        auto pair = [&](int first) {
            std::array<std::int32_t, used> ids{first, first+1};
            const auto expected = reference.compute(x, ids, routes);
            if (joint) {
                auto hm = weights.segment; hm.seq_len = 2; hm.independent_decode_rows = true;
                std::array<float, hidden * 2> input{}, result{};
                std::copy(x.begin(), x.end(), input.begin()); std::copy(x.begin(), x.end(), input.begin() + hidden);
                std::array<std::int32_t, used * 2> selected{first, first+1, first, first+1};
                std::array<float, used * 2> probabilities{.65f,.35f,.65f,.35f};
                const auto before = ExecutionStats();
                require(tsg::host_moe_cached_experts(hm, input.data(), selected.data(), probabilities.data(),
                    result.data(), "qwen4exp arena batched decode") == 1, "Joint file cache did not execute");
                require(ExecutionStats().submissions == before.submissions + 1,
                    "Joint file rows were computed by separate submissions");
                require(std::memcmp(result.data(), expected.data(), sizeof(actual)) == 0
                    && std::memcmp(result.data() + hidden, expected.data(), sizeof(actual)) == 0,
                    "Joint file chunking changed scalar arithmetic");
            } else {
                require(tsg::host_moe_cached_experts(weights.segment, x.data(), ids.data(), routes.data(),
                    actual.data(), "qwen4exp fused token span") == 1, "File-backed cache did not execute");
                require(std::memcmp(actual.data(), expected.data(), sizeof(actual)) == 0, "File staging changed expert logits");
            }
        };
        SharedLedger ledger;
        require(ledger.attach(true), "Cannot attach shared host/device quota before file staging");
        std::size_t arena_bytes = 0;
        for (const auto& part : weights.quantized) arena_bytes += part.size() / experts * used;
        if (joint) arena_bytes = std::min<std::size_t>(arena_bytes, 65536);
        ledger.host_capacity = arena_bytes - 1;
        const auto before_denied = tsg::g_test_file_reads;
        rejected = false;
        try { pair(0); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected && ledger.host_pending == 0 && ledger.host_committed == 0
            && tsg::host_moe_expert_cache_test_file_workspace_bytes() == 0
            && before_denied == tsg::g_test_file_reads,
            "Host quota refusal allocated, read, or silently fell back to unbounded mmap");
        ledger.host_capacity = arena_bytes;
        ledger.fail_commit = true;
        rejected = false;
        try { pair(0); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected && ledger.host_pending == 0 && ledger.host_committed == 0
            && tsg::host_moe_expert_cache_test_file_workspace_bytes() == 0,
            "Refused host commit retained physical storage or credit");
        ledger.fail_commit = false;
        pair(0);
        require(ledger.host_committed == static_cast<std::int64_t>(arena_bytes)
            && ledger.host_pending == 0 && !ledger.detach(), "Exact host quota or lifetime ownership failed");
        for (int i = 0; i < experts; i += used) pair(i);
        const auto reads = tsg::g_test_file_reads;
        require(tsg::host_moe_expert_cache_test_file_workspace_bytes() > 0
            && tsg::host_moe_expert_cache_test_file_workspace_bytes() <= (32u << 20),
            "File workspace was absent or exceeded its process-wide ceiling");
        pair(experts-used);
        require(reads > 0 && reads == tsg::g_test_file_reads, "Warm slots reread files or file path was not exercised");
        require(TSGgml_TrimHostMoeExpertCache(0) > 0
            && tsg::host_moe_expert_cache_test_file_workspace_bytes() == 0 && ledger.host_committed == 0,
            "An idle cache trim retained its file transfer workspace");
        pair(0);
        require(tsg::g_test_file_reads > reads, "Cache trim lost the live model's source registrations");
        for (int i = 0; i < experts; i += used) pair(i);
        tsg::g_test_file_failure_countdown = 2;
        rejected = false;
        // A frequency policy may correctly retain the recently repeated pair 0.
        // Visit the full working set to force a miss against a filled cache,
        // without assuming which expert a particular eviction policy removed.
        for (int i = 0; i < experts && !rejected; i += used)
            try { pair(i); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected && tsg::g_test_file_failure_countdown == 0, "Interrupted file read was not exercised");
        for (int i = experts-used; i >= 0; i -= used) pair(i);
        // Invalidate one registered source: no incomplete triple may use file
        // staging; the existing mapped path still computes the exact answer.
        tsg::host_moe_expert_cache_on_drop(weights.segment.gate_data);
        const auto before_fallback = tsg::g_test_file_reads;
        pair(0);
        require(before_fallback == tsg::g_test_file_reads, "Dropped file source was retained");
        tsg::host_moe_expert_cache_release();
        ledger.check_empty();
        require(ledger.detach(), "File teardown did not release shared host/device quota");
        require(std::remove(path.c_str()) == 0, "File handles survived cache teardown");
        require(tsg::host_moe_expert_cache_test_file_workspace_bytes() == 0, "File workspace survived cache teardown");
        if (joint) tsg::host_moe_expert_cache_test_file_slice_limit(32u << 20);
        std::puts("PASS: exact file extents, cached reuse, out-of-order staging, partial-read recovery and source release");
    }
#endif

    void joint_budget_checks(Weights& weights)
    {
        tsg::host_moe_expert_cache_release();
        std::uint64_t token = 0;
        require(TSGgml_BeginHostMoeExpertCacheScope(8 << 20, 8, 1, &token) == 1, "Joint budget scope failed");
        SharedLedger ledger;
        ledger.capacity = 0;
        require(ledger.attach(), "Joint budget attach failed");
        FullResidentGraph reference(weights);
        auto hm = weights.segment;
        hm.seq_len = 2; hm.independent_decode_rows = true;
        std::array<float, hidden * 2> input{}, actual{}, expected{};
        std::array<std::int32_t, used * 2> ids{0, 1, 2, 3};
        std::array<float, used * 2> routes{.6f, .4f, .3f, .7f};
        for (int row = 0; row < 2; ++row) {
            std::array<float, hidden> x{};
            for (int h = 0; h < hidden; ++h) input[row * hidden + h] = x[h] = std::sin(h * .03f + row);
            std::array<std::int32_t, used> ri{ids[row * used], ids[row * used + 1]};
            std::array<float, used> rr{routes[row * used], routes[row * used + 1]};
            const auto answer = reference.compute(x, ri, rr);
            std::copy(answer.begin(), answer.end(), expected.begin() + row * hidden);
        }
        auto call = [&] { return tsg::host_moe_cached_experts(hm, input.data(), ids.data(), routes.data(),
            actual.data(), "qwen4exp arena batched decode"); };
        auto same = [&] { require(std::memcmp(actual.data(), expected.data(), sizeof(expected)) == 0,
            "Joint graph changed scalar-row arithmetic"); };
        actual.fill(-999);
        const auto before = ExecutionStats();
        require(call() == 0 && ExecutionStats().submissions == before.submissions, "Refused joint quota submitted work");
        require(std::all_of(actual.begin(), actual.end(), [](float f) { return f == -999; }), "Refused joint output changed");
        ledger.check_empty(); ledger.capacity = 8 << 20;
        for (int stage : {1, 2, 3}) {
            tsg::host_moe_expert_cache_test_fail_next(stage);
            require(call() == 0, "Joint allocation failure was not rejected"); ledger.check_empty();
        }
        ledger.fail_commit = true;
        require(call() == 0, "Joint commit refusal published a graph"); ledger.check_empty();
        ledger.fail_commit = false;
        // Measure the exact four-expert union floor, including two-row scratch.
        auto full = hm;
        hm.num_experts = 4;
        hm.gate_bytes = hm.gate_bytes / experts * 4;
        hm.up_bytes = hm.up_bytes / experts * 4;
        hm.down_bytes = hm.down_bytes / experts * 4;
        require(call() == 1, "Cannot measure joint union floor"); same();
        const auto minimum = ledger.committed;
        TSGgml_TrimHostMoeExpertCache(0); ledger.check_empty(); hm = full;
        ledger.capacity = minimum - 1;
        actual.fill(-999); const auto refused = ExecutionStats();
        require(call() == 0 && ExecutionStats().submissions == refused.submissions,
            "Joint union exceeded exact minimum quota"); ledger.check_empty();
        ledger.capacity = minimum;
        require(call() == 1 && ledger.committed == minimum, "Joint union floor did not retry exactly"); same();
        const auto reserved = ledger.reserve_calls;
        const auto warm = ExecutionStats();
        require(call() == 1 && ledger.reserve_calls == reserved && ExecutionStats().submissions == warm.submissions + 1,
            "Joint warm replay allocated or submitted more than one graph"); same();
        // The grown graph's one-row prefix must preserve solo arithmetic and
        // must not execute dummy lanes or create another retained allocation.
        hm.seq_len = 1;
        const auto scalar = ExecutionStats();
        require(call() == 1 && ledger.reserve_calls == reserved && ExecutionStats().submissions == scalar.submissions + 1,
            "Solo prefix of joint cache allocated or executed extra submissions");
        require(std::memcmp(actual.data(), expected.data(), hidden * sizeof(float)) == 0, "Solo prefix arithmetic changed");
        hm.seq_len = 2;
        for (int stage : {6, 7}) {
            actual.fill(-999); tsg::host_moe_expert_cache_test_fail_next(stage);
            bool failed = false;
            try { (void)call(); } catch (const std::runtime_error&) { failed = true; }
            require(failed && std::all_of(actual.begin(), actual.end(), [](float f) { return f == -999; }),
                "Joint post-upload/compute failure published partial output");
            require(call() == 1, "Joint execution did not recover after drained failure"); same();
        }
        // Invalid later route must reject before updating slots or submitting.
        ids.back() = experts;
        const auto invalid = ExecutionStats(); actual.fill(-999);
        require(call() == 0 && ExecutionStats().submissions == invalid.submissions, "Late invalid route submitted joint work");
        require(std::all_of(actual.begin(), actual.end(), [](float f) { return f == -999; }), "Invalid route published output");
        TSGgml_TrimHostMoeExpertCache(0); ledger.check_empty();
        require(ledger.detach(), "Joint cache leaked a callback owner");
        std::int64_t released = 0;
        require(TSGgml_EndHostMoeExpertCacheScope(token, &released) == 1, "Joint scope release failed");
        std::printf("PASS: joint union quota floor=%lld, one submission, prefix reuse, allocation/commit and queued/compute failure recovery\n", (long long)minimum);
    }

    template<int selected_count>
    void joint_exact_width_checks(const std::array<ggml_type, 3>& types)
    {
        constexpr int expert_count = 32;
        tsg::host_moe_expert_cache_release();
        std::uint64_t token = 0;
        require(TSGgml_BeginHostMoeExpertCacheScope(32 << 20, 8, 1, &token) == 1,
            "Cannot begin exact-width expert graph scope");
        SharedLedger ledger; ledger.capacity = 32 << 20;
        require(ledger.attach(), "Cannot attach exact-width expert graph budget");
        Weights weights; weights.types = types; weights.fill(0.71f, expert_count, selected_count);
        FullResidentGraph reference(weights);
        DeviceInputs device(8, selected_count);
        std::uint64_t first_reserves = 0;
        std::int64_t first_bytes = 0;
        for (const int rows : {8, 1, 7, 2, 6, 3, 5, 4, 8}) {
            auto hm = weights.segment; hm.seq_len = rows; hm.independent_decode_rows = true;
            std::vector<float> input(rows * hidden), routes(rows * selected_count), expected(rows * hidden);
            std::vector<std::int32_t> ids(rows * selected_count);
            std::vector<float> actual(expected.size(), -999.0f);
            for (int row = 0; row < rows; ++row) {
                std::array<float, hidden> x{};
                std::array<float, selected_count> probabilities{};
                std::array<std::int32_t, selected_count> selected{};
                for (int h = 0; h < hidden; ++h)
                    input[row * hidden + h] = x[h] = std::sin(h * .021f + row * .31f + rows * .07f);
                for (int k = 0; k < selected_count; ++k) {
                    selected[k] = (row * 7 + k * 3 + rows) % expert_count;
                    probabilities[k] = (k + 1.0f + row * .125f)
                        / (selected_count * (selected_count + 1.0f) * .5f + selected_count * .125f * row);
                }
                // Duplicate routes must preserve their positions and weights;
                // unlike the upstream multi-token ID kernel, scalar MMVQ has
                // no unique-per-token compaction requirement.
                if constexpr (selected_count > 2)
                    if (row % 3 == 1) selected[selected_count - 1] = selected[1];
                std::copy(selected.begin(), selected.end(), ids.begin() + row * selected_count);
                std::copy(probabilities.begin(), probabilities.end(), routes.begin() + row * selected_count);
                const auto answer = reference.compute(x, selected, probabilities);
                std::copy(answer.begin(), answer.end(), expected.begin() + row * hidden);
            }
            ggml_backend_tensor_set(device.x, input.data(), 0, input.size() * sizeof(float));
            ggml_backend_tensor_set(device.routes, routes.data(), 0, routes.size() * sizeof(float));
            hm.moe_in = device.x; hm.weights = device.routes; hm.moe_out = device.output;
            for (int bridge = 0; bridge < 3; ++bridge) {
                const auto before = ExecutionStats();
                require(tsg::host_moe_cached_experts(hm, bridge ? nullptr : input.data(), ids.data(),
                    bridge ? nullptr : routes.data(), bridge == 2 ? nullptr : actual.data(),
                    "qwen4exp arena batched decode") == 1, "Exact-width cache failed to execute");
                require_operator_batch_shape(rows, selected_count);
                if (bridge == 2) {
                    tsg::sync_backend(g_backend);
                    ggml_backend_tensor_get(device.output, actual.data(), 0, actual.size() * sizeof(float));
                }
                require(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0,
                    "Exact-width route-major expert operators differ from independent scalar CUDA oracle");
                require(ExecutionStats().submissions == before.submissions + 1,
                    "Exact-width cache submitted dummy or separate-row graphs");
                if (first_reserves == 0) { first_reserves = ledger.reserve_calls; first_bytes = ledger.committed; }
                require(ledger.reserve_calls == first_reserves && ledger.committed == first_bytes,
                    "Live-width changes duplicated or rebuilt expert owners");
            }
        }
        TSGgml_TrimHostMoeExpertCache(0); ledger.check_empty();
        require(ledger.detach(), "Exact-width cache retained a shared owner");
        std::int64_t released = 0;
        require(TSGgml_EndHostMoeExpertCacheScope(token, &released) == 1, "Exact-width scope did not release");
        std::printf("PASS: exact N=8/1/7/2/6/3/5/4/8, K%d, three actual batched projections, scalar CUDA bit-exact, 3 bridges and one owner %lld bytes\n",
            selected_count, static_cast<long long>(first_bytes));
    }

    void cuda_checks(Weights& weights, bool arena = false)
    {
        const char* kernel = arena ? "qwen4exp arena batched decode" : "qwen4exp fused token span";
        weights.segment.independent_decode_rows = arena;
        fallback_checks(weights);
        FullResidentGraph reference(weights);
        std::array<float, hidden> x{}, actual{};
        std::array<float, used> routes{0.65f, 0.35f};
        // A 2 MiB quota holds fewer than 16 experts (1 MiB is the workspace
        // reserve). Cycling sixteen IDs therefore requires real slot eviction.
        for (int iteration = 0; iteration < 80; ++iteration)
        {
            for (int i = 0; i < hidden; ++i)
                x[i] = std::sin(0.03f * i + 0.1f * iteration);
            std::array<std::int32_t, used> ids{(iteration * 3) % experts, (iteration * 3 + 1) % experts};
            if (iteration % 5 == 0) ids = {0, 2}; // miss can precede a later protected hit
            const auto expected = reference.compute(x, ids, routes);
            require(tsg::host_moe_cached_experts(weights.segment, x.data(), ids.data(), routes.data(),
                actual.data(), kernel) == 1,
                "The opt-in CUDA expert cache did not engage");
            require(std::memcmp(actual.data(), expected.data(), sizeof(actual)) == 0,
                "Compact/remapped or evicted expert results differ from the full-resident CUDA graph");
            const auto again = reference.compute(x, ids, routes);
            require(tsg::host_moe_cached_experts(weights.segment, x.data(), ids.data(), routes.data(),
                actual.data(), kernel) == 1,
                "Warm expert slots did not engage");
            require(std::memcmp(actual.data(), again.data(), sizeof(actual)) == 0,
                "Warm expert slots reused stale activations or different arithmetic");
            auto device_hm = weights.segment;
            device_hm.moe_in = reference.x;
            device_hm.weights = reference.routes;
            require(tsg::host_moe_cached_experts(device_hm, nullptr, ids.data(), nullptr,
                actual.data(), kernel) == 1,
                "The CUDA device-to-device scalar input bridge did not engage");
            require(std::memcmp(actual.data(), again.data(), sizeof(actual)) == 0,
                "CUDA device-to-device scalar inputs changed expert arithmetic");
            device_hm.moe_out = reference.output;
            require(tsg::host_moe_cached_experts(device_hm, nullptr, ids.data(), nullptr,
                nullptr, kernel) == 1,
                "The CUDA device-to-device scalar output bridge did not engage");
            tsg::sync_backend(g_backend);
            ggml_backend_tensor_get(reference.output, actual.data(), 0, sizeof(actual));
            require(std::memcmp(actual.data(), again.data(), sizeof(actual)) == 0,
                "CUDA device-to-device scalar output changed expert arithmetic");
            Stats stats;
            require(stats.reserved > 0, "Engaged cache did not account for its device buffers");
        }
        const Stats warm;
        require(warm.calls == 320 && warm.hits >= 480 && warm.misses > experts,
            "Test did not exercise warm hits and actual expert eviction");
        std::uint64_t arena_token = 0;
        if (arena) {
            tsg::host_moe_expert_cache_release();
            require(TSGgml_BeginHostMoeExpertCacheScope(8 << 20, 8, 1, &arena_token) == 1,
                "Cannot admit a bounded multi-row expert union fixture");
        }

        // Short prefill and speculative target verification must execute the
        // same one-row CUDA FFN arithmetic as successive autoregressive calls.
        // Changing activations, IDs, routes and repeated IDs exercises reuse
        // and slot eviction between rows inside a single cache-helper call.
        DeviceInputs device_inputs;
        for (int rows = 2; rows <= TSG_PRECISION_DECODE_COLUMNS; ++rows)
        {
            auto hm = weights.segment;
            hm.seq_len = rows;
            std::vector<float> inputs(static_cast<std::size_t>(rows) * hidden);
            std::vector<std::int32_t> selected(static_cast<std::size_t>(rows) * used);
            std::vector<float> probabilities(static_cast<std::size_t>(rows) * used);
            std::vector<float> expected(static_cast<std::size_t>(rows) * hidden);
            std::vector<float> block_result(expected.size(), -999.0f);
            for (int row = 0; row < rows; ++row)
            {
                for (int i = 0; i < hidden; ++i)
                    x[i] = inputs[static_cast<std::size_t>(row) * hidden + i]
                        = std::sin(0.07f * i + rows * 0.2f + row);
                std::array<std::int32_t, used> row_ids{(row * 5 + rows) % experts,
                    (row * 5 + rows + 1) % experts};
                if (row % 3 == 0) row_ids[1] = row_ids[0];
                std::array<float, used> row_routes{0.25f + 0.05f * row, 0.75f - 0.05f * row};
                std::copy(row_ids.begin(), row_ids.end(), selected.begin() + row * used);
                std::copy(row_routes.begin(), row_routes.end(), probabilities.begin() + row * used);
                const auto row_result = reference.compute(x, row_ids, row_routes);
                std::copy(row_result.begin(), row_result.end(), expected.begin() + row * hidden);
            }
            const auto before = Stats();
            const auto graphs_before = ExecutionStats();
            require(tsg::host_moe_cached_experts(hm, inputs.data(), selected.data(), probabilities.data(),
                block_result.data(), kernel) == 1,
                "Short-block expert cache did not replay the scalar CUDA graph");
            require(std::memcmp(block_result.data(), expected.data(), expected.size() * sizeof(float)) == 0,
                "Short-block expert rows differ from full-resident decode rounding");
            require(Stats().calls == before.calls + rows,
                "Short-block cache did not compute every row through the scalar graph");
            require(ExecutionStats().submissions == graphs_before.submissions + (arena ? 1 : rows),
                "Arena expert rows were serialized as separate graph submissions");
            if (arena) require(ExecutionStats().joint == graphs_before.joint + 1
                && ExecutionStats().rows == graphs_before.rows + rows, "Joint engagement was not recorded");
            if (arena) require_operator_batch_shape(rows, used);

            ggml_backend_tensor_set(device_inputs.x, inputs.data(), 0, inputs.size() * sizeof(float));
            ggml_backend_tensor_set(device_inputs.routes, probabilities.data(), 0, probabilities.size() * sizeof(float));
            hm.moe_in = device_inputs.x;
            hm.weights = device_inputs.routes;
            std::fill(block_result.begin(), block_result.end(), -999.0f);
            const auto device_before = Stats();
            const auto device_graphs_before = ExecutionStats();
            require(tsg::host_moe_cached_experts(hm, nullptr, selected.data(), nullptr,
                block_result.data(), kernel) == 1,
                "Short-block CUDA device-to-device input bridge did not engage");
            require(std::memcmp(block_result.data(), expected.data(), expected.size() * sizeof(float)) == 0,
                "Short-block CUDA device-to-device rows changed expert arithmetic");
            require(Stats().calls == device_before.calls + rows,
                "Short-block device bridge did not compute every row");
            require(ExecutionStats().submissions == device_graphs_before.submissions + (arena ? 1 : rows),
                "Device-input joint graph did not execute once");
            if (arena) require_operator_batch_shape(rows, used);
            hm.moe_out = device_inputs.output;
            std::fill(block_result.begin(), block_result.end(), -999.0f);
            const auto output_before = Stats();
            const auto output_graphs_before = ExecutionStats();
            require(tsg::host_moe_cached_experts(hm, nullptr, selected.data(), nullptr,
                nullptr, kernel) == 1,
                "Short-block CUDA device-to-device output bridge did not engage");
            // Production consumes this queued copy in the next graph on the
            // same backend stream. A standalone host read uses its own CUDA
            // stream and must wait for that pending copy explicitly.
            tsg::sync_backend(g_backend);
            ggml_backend_tensor_get(device_inputs.output, block_result.data(), 0,
                block_result.size() * sizeof(float));
            require(std::memcmp(block_result.data(), expected.data(), expected.size() * sizeof(float)) == 0,
                "Short-block CUDA device-to-device output changed expert arithmetic");
            require(Stats().calls == output_before.calls + rows,
                "Short-block device output did not compute every row");
            require(ExecutionStats().submissions == output_graphs_before.submissions + (arena ? 1 : rows),
                "Device-output joint graph did not execute once");
            if (arena) require_operator_batch_shape(rows, used);

            // A bad later row must reject the complete block before computing
            // the valid early rows or overwriting any caller output.
            selected.back() = experts;
            std::fill(block_result.begin(), block_result.end(), -999.0f);
            const auto invalid_before = Stats();
            require(tsg::host_moe_cached_experts(hm, inputs.data(), selected.data(), probabilities.data(),
                block_result.data(), kernel) == 0,
                "Invalid later-row expert ID did not reject the entire block");
            require(std::all_of(block_result.begin(), block_result.end(), [](float v) { return v == -999.0f; }),
                "Rejected short block exposed a partially computed CUDA result");
            const auto invalid_after = Stats();
            require(invalid_after.calls == invalid_before.calls && invalid_after.hits == invalid_before.hits
                && invalid_after.misses == invalid_before.misses && invalid_after.reserved == invalid_before.reserved,
                "Rejected short block mutated cache state before validating every row");
            require(tsg::host_moe_cached_experts(hm, nullptr, selected.data(), nullptr,
                block_result.data(), kernel) == 0,
                "Device bridge did not reject an invalid later-row expert ID");
            require(std::all_of(block_result.begin(), block_result.end(), [](float v) { return v == -999.0f; }),
                "Rejected device-bridge block exposed a partially computed CUDA result");

            selected.back() = 0;
            hm.weights = nullptr;
            require(tsg::host_moe_cached_experts(hm, nullptr, selected.data(), nullptr,
                block_result.data(), kernel) == 0,
                "Missing device routing source did not retain the staged fallback");
        }

        if (arena) {
            std::int64_t released = 0;
            require(TSGgml_EndHostMoeExpertCacheScope(arena_token, &released) == 1 && Stats().reserved == 0,
                "Joint graph scope retained physical payload");
        }
        tsg::host_moe_expert_cache_on_drop(weights.segment.gate_data);
        require(Stats().reserved == 0, "Invalidated model weights retained compact CUDA slots");
        // Rewrite the same vector allocations, as a reload can recycle an mmap
        // address. Invalidation must force fresh copies of every routed expert.
        weights.fill(1.4f);
        FullResidentGraph reloaded(weights);
        std::array<std::int32_t, used> ids{0, 2};
        const auto expected = reloaded.compute(x, ids, routes);
        require(tsg::host_moe_cached_experts(weights.segment, x.data(), ids.data(), routes.data(), actual.data(),
            kernel) == 1, "Reloaded cache did not engage");
        require(std::memcmp(actual.data(), expected.data(), sizeof(actual)) == 0,
            "Reloaded expert cache read bytes from the previous model identity");
        auto output_hm = weights.segment;
        output_hm.moe_in = reloaded.x;
        output_hm.weights = reloaded.routes;
        output_hm.moe_out = reloaded.output;
        require(tsg::host_moe_cached_experts(output_hm, nullptr, ids.data(), nullptr,
            nullptr, kernel) == 1, "Reloaded direct output did not engage");
        // No tensor read/synchronize between the final queued row copy and
        // invalidation. Entry retirement must drain that copy before freeing
        // its source graph buffer, leaving the external destination valid.
        tsg::host_moe_expert_cache_on_drop(weights.segment.gate_data);
        require(Stats().reserved == 0, "Direct output prevented cache source invalidation");
        ggml_backend_tensor_get(reloaded.output, actual.data(), 0, sizeof(actual));
        require(std::memcmp(actual.data(), expected.data(), sizeof(actual)) == 0,
            "Cache invalidation freed a pending direct-output copy's source");
        tsg::host_moe_expert_cache_release();
        const Stats released;
        require(released.reserved == 0 && released.calls == 0 && released.hits == 0 && released.misses == 0,
            "Explicit expert-cache teardown retained device allocations or old counters");

        // Fill a complete routing cycle, then fail after the first projection
        // of an evicted slot was uploaded. No old expert ID may remain valid
        // over those partially replaced bytes, even if the caller retries.
        auto run_pair = [&](int first) {
            std::array<std::int32_t, used> selected{first, first + 1};
            const auto expected_pair = reloaded.compute(x, selected, routes);
            require(tsg::host_moe_cached_experts(weights.segment, x.data(), selected.data(), routes.data(),
                actual.data(), kernel) == 1, "Failed upload prevented cache recovery");
            require(std::memcmp(actual.data(), expected_pair.data(), sizeof(actual)) == 0,
                "A partially overwritten slot retained a valid old expert ID");
        };
        for (int i = 0; i < experts; i += used) run_pair(i);
        tsg::g_test_copy_failure_countdown = 2;
        bool interrupted = false;
        try { run_pair(0); }
        catch (const std::runtime_error&) { interrupted = true; }
        require(interrupted && tsg::g_test_copy_failure_countdown == 0,
            "Partial expert upload failure injection did not execute");
        for (int i = experts - used; i >= 0; i -= used) run_pair(i);
        tsg::host_moe_expert_cache_release();
    }
#endif
}

int main(int argc, char** argv)
{
    require(TSGgml_HostMoeExpertCacheBudgetKind() == 4, "Selected-expert charge capability is missing");
    try
    {
        prefetch_policy_checks();
        prefill_limit_checks();
        const bool invalid_budget = argc == 3 && std::strcmp(argv[1], "--invalid-budget") == 0;
        const bool trim_only = argc == 2 && std::strcmp(argv[1], "--trim") == 0;
        set_environment("TS_HOST_MOE_EXPERT_CACHE_MB", invalid_budget ? argv[2] : trim_only ? "8" : "2");
        set_environment("TS_HOST_MOE_EXPERT_CACHE_LAYERS", trim_only ? "4" : "1");
        set_environment("TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS", "1");
        if (invalid_budget)
        {
            const auto stats = Stats();
            require(stats.budget == 0 && stats.reserved == 0 && stats.calls == 0,
                "Malformed expert-cache budget was accepted by the native parser");
            std::puts("PASS: malformed native cache budget remains disabled");
            return 0;
        }
        Weights weights;
        bool cpu_only = false, prefetch = false, file_read = false, lfu = false, wide_prefill = false, scoped = false, arena = false;
        for (int arg = 1; arg < argc; ++arg)
        {
            if (std::strcmp(argv[arg], "--cpu") == 0) cpu_only = true;
            else if (std::strcmp(argv[arg], "--scope") == 0) scoped = true;
            else if (std::strcmp(argv[arg], "--arena") == 0) arena = true;
            else if (std::strcmp(argv[arg], "--trim") == 0 && trim_only) { }
            else if (std::strcmp(argv[arg], "--prefetch") == 0) prefetch = true;
            else if (std::strcmp(argv[arg], "--file-read") == 0) file_read = true;
            else if (std::strcmp(argv[arg], "--lfu") == 0) lfu = true;
            else if (std::strcmp(argv[arg], "--wide-prefill") == 0) wide_prefill = true;
            else if (std::strcmp(argv[arg], "--quantization") == 0 && arg + 1 < argc)
            {
                ggml_type type = GGML_TYPE_COUNT;
                ++arg;
                if (std::strcmp(argv[arg], "iq1_m") == 0) type = GGML_TYPE_IQ1_M;
                else if (std::strcmp(argv[arg], "iq2_xxs") == 0) type = GGML_TYPE_IQ2_XXS;
                require(type != GGML_TYPE_COUNT, "Unsupported expert-cache parity quantization");
                weights.types = {type, type, GGML_TYPE_IQ4_NL};
            }
            else throw std::runtime_error("Unsupported expert-cache test argument");
        }
        set_environment("TS_HOST_MOE_EXPERT_CACHE_PREFETCH", prefetch ? "1" : "0");
        set_environment("TS_HOST_MOE_EXPERT_CACHE_LFU", lfu ? "1" : "0");
        set_environment("TS_HOST_MOE_EXPERT_CACHE_PREFILL_ROWS", wide_prefill ? "128" : "");
        if (wide_prefill) set_environment("TS_HOST_MOE_EXPERT_CACHE_MB", "32");
        if (scoped) scope_api_checks();
        weights.fill(0.5f);
#ifdef TSG_GGML_USE_CUDA
        if (!cpu_only)
        {
            if (ggml_backend_cuda_get_device_count() < 1)
            {
                std::fprintf(stderr, "SKIP: CUDA expert-cache parity needs an NVIDIA device\n");
                return 77;
            }
            g_backend = ggml_backend_cuda_init(0);
            require(g_backend != nullptr, "Could not initialize CUDA backend");
            if (scoped) {
                scoped_cuda_checks(weights);
                ggml_backend_free(g_backend); g_backend = nullptr;
                return 0;
            }
            if (wide_prefill)
            {
                wide_prefill_checks(weights);
                wide_route_capacity_checks<10>(weights.types);
                wide_route_capacity_checks<64>(weights.types);
                ggml_backend_free(g_backend); g_backend = nullptr;
                return 0;
            }
            fallback_checks(weights); // Default 9-row refusal must also hold on CUDA.
#if defined(_WIN32)
            if (file_read) {
                file_read_checks(weights, arena);
                ggml_backend_free(g_backend); g_backend = nullptr;
                return 0;
            }
#endif
            if (trim_only)
            {
                trim_lru_checks(weights);
                ggml_backend_free(g_backend);
                g_backend = nullptr;
                return 0;
            }
            shared_budget_checks(weights);
            cuda_checks(weights, arena);
            if (arena) {
                joint_budget_checks(weights);
                joint_exact_width_checks<1>(weights.types);
                joint_exact_width_checks<3>(weights.types);
                joint_exact_width_checks<10>(weights.types);
                joint_exact_width_checks<16>(weights.types);
            }
            require(prefetch == (tsg::g_test_prefetch_calls > 0),
                "Opt-in raw expert prefetch did not match its requested mode");
            ggml_backend_free(g_backend);
            g_backend = nullptr;
            std::puts("PASS: CUDA compact expert cache, warm hits, eviction, invalidation, reload and budget");
            return 0;
        }
#else
        (void)cpu_only;
#endif
        g_backend = ggml_backend_cpu_init();
        require(g_backend != nullptr, "Could not initialize CPU backend");
        SharedLedger ledger;
        require(ledger.attach(), "Could not attach the CPU fallback budget fixture");
        fallback_checks(weights);
        std::array<float, hidden> x{}, out{};
        std::array<std::int32_t, used> ids{0, 1};
        std::array<float, used> routes{0.6f, 0.4f};
        require(tsg::host_moe_cached_experts(weights.segment, x.data(), ids.data(), routes.data(), out.data(),
            "qwen4exp fused token span") == 0, "CPU backend incorrectly engaged the CUDA expert cache");
        require(Stats().reserved == 0 && Stats().calls == 0, "Fallback path allocated CUDA cache state");
        tsg::host_moe_expert_cache_release();
        ledger.check_empty();
        require(ledger.reserve_calls == 0 && ledger.detach(),
            "CPU fallback touched shared CUDA expert credit or prevented detach");
        ggml_backend_free(g_backend);
        g_backend = nullptr;
        std::puts("PASS: disabled/non-CUDA and unsupported expert-cache shapes fall back");
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        tsg::host_moe_expert_cache_release();
        if (g_backend) { ggml_backend_free(g_backend); g_backend = nullptr; }
        return 1;
    }
}
