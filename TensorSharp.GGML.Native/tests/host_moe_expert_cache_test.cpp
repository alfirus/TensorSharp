// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// The optional expert tier against an independent full-resident CUDA graph.
#include "ggml_ops_internal.h"
#include "ggml_ops_precision_policy.h"
#include "ggml-impl.h"
#ifdef TSG_GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

extern "C" int TSGgml_HostMoeExpertCacheStats(std::int64_t*, std::int64_t*,
    std::int64_t*, std::int64_t*, std::int64_t*);
extern "C" int TSGgml_HostMoeExpertHostCacheStats(std::int64_t*,std::int64_t*,std::int64_t*,std::int64_t*,std::int64_t*);

// This standalone target compiles the production cache implementation with
// small bridge stubs; it never loads a model or depends on modified ggml code.
namespace tsg
{
    DeviceState g_device_states[TSG_MAX_DEVICES];
    thread_local int g_active_rank = 0;
    std::atomic<std::uint64_t> g_ggml_error_count{0};
    std::atomic<bool> g_backend_compute_failed{false};
    std::uint64_t g_test_prefetch_calls = 0;
    void set_last_error(const std::string& message) { std::fprintf(stderr, "%s\n", message.c_str()); }
#ifdef TSG_GGML_USE_CUDA
    void host_pin_split(const void*, std::size_t bytes,
        std::vector<std::pair<std::size_t, std::size_t>>& pieces)
    {
        pieces.assign(1, {0, bytes});
    }
    void prefetch_mapped_ranges(const std::uint8_t* base,
        const std::vector<std::pair<std::size_t, std::size_t>>& ranges)
    {
        // The production bridge reuses the existing MoE page-fault pool. This
        // standalone cache test observes its raw ranges and touches them without
        // linking the unrelated host-FFN engine or needing mapped model files.
        if (base == nullptr || ranges.empty() || !std::is_sorted(ranges.begin(), ranges.end()))
            throw std::runtime_error("Cache prefetch ranges are missing or unordered");
        volatile std::uint8_t sink = 0;
        for (const auto& range : ranges)
        {
            if (range.second == 0) throw std::runtime_error("Cache prefetch range is empty");
            sink ^= base[range.first];
            sink ^= base[range.first + range.second - 1];
        }
        (void)sink;
        ++g_test_prefetch_calls;
    }
#endif
}

namespace
{
    void require(bool condition, const char* message)
    {
        if (!condition) throw std::runtime_error(message);
    }

    void set_environment(const char* key, const char* value)
    {
#if defined(_WIN32)
        _putenv_s(key, value);
#else
        setenv(key, value, 1);
#endif
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
    struct HostStats
    {
        std::int64_t reserved=0,budget=0,hits=0,sources=0,evictions=0;
        HostStats()
        {
            require(TSGgml_HostMoeExpertHostCacheStats(&reserved,&budget,&hits,&sources,&evictions)==1,
                "Could not read exclusive host cache statistics");
            require(reserved>=0 && reserved<=budget,"Anonymous host cache exceeded its explicit byte cap");
        }
    };

    constexpr int hidden = 256, ff = 256, experts = 16, used = 2;

    struct Weights
    {
        tsg::HostMoeSegment segment;
        std::array<std::vector<std::uint8_t>, 3> quantized;
        std::array<ggml_type, 3> types{GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0};

        void fill(float seed)
        {
            std::vector<float> values(static_cast<std::size_t>(hidden) * ff * experts);
            std::vector<float> importance(hidden, 1.0f);
            for (int part = 0; part < 3; ++part)
            {
                for (std::size_t i = 0; i < values.size(); ++i)
                    values[i] = 0.04f * std::sin(seed + float(i % 7919) * 0.013f + part * 0.7f);
                const auto bytes = ggml_row_size(types[part], hidden) * ff * experts;
                quantized[part].resize(bytes);
                const auto written = ggml_quantize_chunk(types[part], values.data(),
                    quantized[part].data(), 0, ff * experts, hidden,
                    ggml_quantize_requires_imatrix(types[part]) ? importance.data() : nullptr);
                require(written == bytes, "Synthetic quantized expert layout is incomplete");
            }
            segment.layer = 0;
            segment.hidden = hidden; segment.n_ff = ff; segment.num_experts = experts;
            segment.n_used = used; segment.seq_len = 1; segment.activation = 0;
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

        DeviceInputs()
        {
            context = ggml_init({ggml_tensor_overhead() * 4, nullptr, true});
            require(context != nullptr, "Could not allocate device-bridge input metadata");
            x = ggml_new_tensor_2d(context, GGML_TYPE_F32, hidden, TSG_PRECISION_DECODE_COLUMNS);
            routes = ggml_new_tensor_2d(context, GGML_TYPE_F32, used, TSG_PRECISION_DECODE_COLUMNS);
            output = ggml_new_tensor_2d(context, GGML_TYPE_F32, hidden, TSG_PRECISION_DECODE_COLUMNS);
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
        ggml_backend_t backend = nullptr;
        ggml_context* context = nullptr;
        ggml_gallocr_t allocator = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* x = nullptr;
        ggml_tensor* ids = nullptr;
        ggml_tensor* routes = nullptr;
        ggml_tensor* output = nullptr;

        explicit FullResidentGraph(const Weights& weights, bool deepseek = false, ggml_backend_t device = nullptr)
        {
            backend = device != nullptr ? device : g_backend;
            context = ggml_init({ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false), nullptr, true});
            require(context != nullptr, "Could not allocate reference tensor metadata");
            auto* gate = ggml_new_tensor_3d(context, weights.types[0], hidden, ff, experts);
            auto* up = ggml_new_tensor_3d(context, weights.types[1], hidden, ff, experts);
            auto* down = ggml_new_tensor_3d(context, weights.types[2], ff, hidden, experts);
            for (auto* t : {gate, up, down}) { ggml_set_input(t); ggml_set_output(t); }
            x = ggml_new_tensor_3d(context, GGML_TYPE_F32, hidden, 1, 1);
            ids = ggml_new_tensor_2d(context, GGML_TYPE_I32, used, 1);
            routes = ggml_new_tensor_3d(context, GGML_TYPE_F32, 1, used, 1);
            for (auto* t : {x, ids, routes}) ggml_set_input(t);
            // The outer Qwen host-MoE boundary keeps these source tensors live
            // as graph outputs. Mirror that lifetime when testing D2D reads
            // after the full-resident reference graph has finished computing.
            ggml_set_output(x);
            ggml_set_output(routes);
            auto* g = ggml_mul_mat_id(context, gate, x, ids);
            auto* u = ggml_mul_mat_id(context, up, x, ids);
            if (deepseek && weights.segment.clamp_limit > 1e-6f)
            {
                g = ggml_clamp(context, g, -INFINITY, weights.segment.clamp_limit);
                u = ggml_clamp(context, u, -weights.segment.clamp_limit, weights.segment.clamp_limit);
            }
            auto* h = deepseek ? ggml_swiglu_split(context, g, u) : ggml_mul(context, ggml_silu(context, g), u);
            auto* parts = ggml_mul_mat_id(context, down, h, ids);
            ggml_set_output(parts);
            auto* weighted = ggml_mul(context, parts, routes);
            auto* sum = ggml_view_2d(context, weighted, hidden, 1, weighted->nb[2], 0);
            for (int k = 1; k < used; ++k)
                sum = ggml_add(context, sum, ggml_view_2d(context, weighted, hidden, 1,
                    weighted->nb[2], static_cast<std::size_t>(k) * weighted->nb[1]));
            output = ggml_cont(context, sum);
            ggml_set_output(output);
            graph = ggml_new_graph_custom(context, 64, false);
            ggml_build_forward_expand(graph, output);
            allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
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

        std::array<float, hidden> compute(const std::array<float, hidden>& activation,
            const std::array<std::int32_t, used>& selected, const std::array<float, used>& probabilities)
        {
            ggml_backend_tensor_set(x, activation.data(), 0, sizeof(activation));
            ggml_backend_tensor_set(ids, selected.data(), 0, sizeof(selected));
            ggml_backend_tensor_set(routes, probabilities.data(), 0, sizeof(probabilities));
            require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
                "Full-resident reference expert graph failed");
            std::array<float, hidden> result{};
            ggml_backend_tensor_get(output, result.data(), 0, sizeof(result));
            return result;
        }
    };

    void cuda_checks(Weights& weights)
    {
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
                actual.data(), "qwen4exp fused token span") == 1,
                "The opt-in CUDA expert cache did not engage");
            require(std::memcmp(actual.data(), expected.data(), sizeof(actual)) == 0,
                "Compact/remapped or evicted expert results differ from the full-resident CUDA graph");
            const auto again = reference.compute(x, ids, routes);
            require(tsg::host_moe_cached_experts(weights.segment, x.data(), ids.data(), routes.data(),
                actual.data(), "qwen4exp fused token span") == 1,
                "Warm expert slots did not engage");
            require(std::memcmp(actual.data(), again.data(), sizeof(actual)) == 0,
                "Warm expert slots reused stale activations or different arithmetic");
            auto device_hm = weights.segment;
            device_hm.moe_in = reference.x;
            device_hm.weights = reference.routes;
            require(tsg::host_moe_cached_experts(device_hm, nullptr, ids.data(), nullptr,
                actual.data(), "qwen4exp fused token span") == 1,
                "The CUDA device-to-device scalar input bridge did not engage");
            require(std::memcmp(actual.data(), again.data(), sizeof(actual)) == 0,
                "CUDA device-to-device scalar inputs changed expert arithmetic");
            device_hm.moe_out = reference.output;
            require(tsg::host_moe_cached_experts(device_hm, nullptr, ids.data(), nullptr,
                nullptr, "qwen4exp fused token span") == 1,
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
            require(tsg::host_moe_cached_experts(hm, inputs.data(), selected.data(), probabilities.data(),
                block_result.data(), "qwen4exp fused token span") == 1,
                "Short-block expert cache did not replay the scalar CUDA graph");
            require(std::memcmp(block_result.data(), expected.data(), expected.size() * sizeof(float)) == 0,
                "Short-block expert rows differ from full-resident decode rounding");
            require(Stats().calls == before.calls + rows,
                "Short-block cache did not compute every row through the scalar graph");

            ggml_backend_tensor_set(device_inputs.x, inputs.data(), 0, inputs.size() * sizeof(float));
            ggml_backend_tensor_set(device_inputs.routes, probabilities.data(), 0, probabilities.size() * sizeof(float));
            hm.moe_in = device_inputs.x;
            hm.weights = device_inputs.routes;
            std::fill(block_result.begin(), block_result.end(), -999.0f);
            const auto device_before = Stats();
            require(tsg::host_moe_cached_experts(hm, nullptr, selected.data(), nullptr,
                block_result.data(), "qwen4exp fused token span") == 1,
                "Short-block CUDA device-to-device input bridge did not engage");
            require(std::memcmp(block_result.data(), expected.data(), expected.size() * sizeof(float)) == 0,
                "Short-block CUDA device-to-device rows changed expert arithmetic");
            require(Stats().calls == device_before.calls + rows,
                "Short-block device bridge did not compute every row");
            hm.moe_out = device_inputs.output;
            std::fill(block_result.begin(), block_result.end(), -999.0f);
            const auto output_before = Stats();
            require(tsg::host_moe_cached_experts(hm, nullptr, selected.data(), nullptr,
                nullptr, "qwen4exp fused token span") == 1,
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

            // A bad later row must reject the complete block before computing
            // the valid early rows or overwriting any caller output.
            selected.back() = experts;
            std::fill(block_result.begin(), block_result.end(), -999.0f);
            const auto invalid_before = Stats();
            require(tsg::host_moe_cached_experts(hm, inputs.data(), selected.data(), probabilities.data(),
                block_result.data(), "qwen4exp fused token span") == 0,
                "Invalid later-row expert ID did not reject the entire block");
            require(std::all_of(block_result.begin(), block_result.end(), [](float v) { return v == -999.0f; }),
                "Rejected short block exposed a partially computed CUDA result");
            const auto invalid_after = Stats();
            require(invalid_after.calls == invalid_before.calls && invalid_after.hits == invalid_before.hits
                && invalid_after.misses == invalid_before.misses && invalid_after.reserved == invalid_before.reserved,
                "Rejected short block mutated cache state before validating every row");
            require(tsg::host_moe_cached_experts(hm, nullptr, selected.data(), nullptr,
                block_result.data(), "qwen4exp fused token span") == 0,
                "Device bridge did not reject an invalid later-row expert ID");
            require(std::all_of(block_result.begin(), block_result.end(), [](float v) { return v == -999.0f; }),
                "Rejected device-bridge block exposed a partially computed CUDA result");

            selected.back() = 0;
            hm.weights = nullptr;
            require(tsg::host_moe_cached_experts(hm, nullptr, selected.data(), nullptr,
                block_result.data(), "qwen4exp fused token span") == 0,
                "Missing device routing source did not retain the staged fallback");
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
            "qwen4exp fused token span") == 1, "Reloaded cache did not engage");
        require(std::memcmp(actual.data(), expected.data(), sizeof(actual)) == 0,
            "Reloaded expert cache read bytes from the previous model identity");
        auto output_hm = weights.segment;
        output_hm.moe_in = reloaded.x;
        output_hm.weights = reloaded.routes;
        output_hm.moe_out = reloaded.output;
        require(tsg::host_moe_cached_experts(output_hm, nullptr, ids.data(), nullptr,
            nullptr, "qwen4exp fused token span") == 1, "Reloaded direct output did not engage");
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
    }

    void deepseek_cuda_checks(Weights& weights)
    {
        constexpr std::size_t budget = std::size_t(2) << 20;
        ggml_backend_t primary = g_backend;
        const int second_device = ggml_backend_cuda_get_device_count() > 1 ? 1 : 0;
        ggml_backend_t secondary = ggml_backend_cuda_init(second_device);
        ggml_backend_t bridge_cpu = ggml_backend_cpu_init();
        if (secondary == nullptr || bridge_cpu == nullptr)
        {
            if (secondary != nullptr) ggml_backend_free(secondary);
            if (bridge_cpu != nullptr) ggml_backend_free(bridge_cpu);
            throw std::runtime_error("Could not initialize independent cache-test backends");
        }
        try
        {
            // An independent DeepSeek backend must work even when the global
            // operation bridge points at a CPU backend.
            g_backend = bridge_cpu;
            auto hm = weights.segment;
            hm.clamp_limit = 0.1f;
            weights.segment.clamp_limit = hm.clamp_limit;
            {
                FullResidentGraph primary_reference(weights, true, primary);
                FullResidentGraph secondary_reference(weights, true, secondary);
                std::array<float, hidden> x{}, actual{};
                std::array<std::int32_t, used> ids{0, 1};
                std::array<float, used> routes{0.6f, 0.4f};
                for (int i = 0; i < hidden; ++i) x[i] = 4.0f * std::sin(0.13f * i);
                actual.fill(-123.0f);
                const auto sentinel = actual;
                const auto run = [&](const tsg::HostMoeSegment& segment, ggml_backend_t backend,
                                     std::size_t bytes = std::size_t(2) << 20, std::size_t layers = 1) {
                    return tsg::host_moe_cached_experts(segment, x.data(), ids.data(), routes.data(), actual.data(),
                        "deepseek41 native", backend, bytes, layers);
                };
                require(run(hm, nullptr) == 0 && actual == sentinel,
                    "DeepSeek cache inferred a backend from the global bridge");
                require(run(hm, primary, 0) == 0 && run(hm, primary, budget, 0) == 0 && actual == sentinel,
                    "DeepSeek cache engaged without an explicit reserved budget/layer count");
                require(run(hm, bridge_cpu) == 0 && actual == sentinel,
                    "DeepSeek cache incorrectly engaged a CPU backend");
                auto invalid = hm;
                invalid.clamp_limit = std::numeric_limits<float>::quiet_NaN();
                require(run(invalid, primary) == 0 && actual == sentinel,
                    "Invalid clamp metadata changed output instead of declining the cache");
                require(run(hm, primary) == 1, "Explicit DeepSeek cache backend did not engage");
                const auto expected = primary_reference.compute(x, ids, routes);
                require(std::memcmp(actual.data(), expected.data(), sizeof(actual)) == 0,
                    "DeepSeek cached clamp/SwiGLU differs from an independent full-resident graph");
                const Stats first;
                require(first.reserved > 0 && first.reserved <= static_cast<std::int64_t>(budget),
                    "First DeepSeek backend exceeded its reserved ceiling");
                require(run(hm, secondary) == 1, "Second independent DeepSeek backend did not engage");
                const auto expected_second = secondary_reference.compute(x, ids, routes);
                require(std::memcmp(actual.data(), expected_second.data(), sizeof(actual)) == 0,
                    "Second DeepSeek backend used another device's weights or activations");
                const Stats both;
                require(both.reserved > first.reserved && both.reserved <= static_cast<std::int64_t>(budget * 2),
                    "Independent backend allocations were combined into one device's budget");
                for (int iteration = 0; iteration < 48; ++iteration)
                {
                    ids = {(iteration * 3) % experts, (iteration * 3 + 1) % experts};
                    if (iteration % 7 == 0) ids[1] = ids[0];
                    require(run(hm, primary) == 1, "DeepSeek compact expert eviction declined unexpectedly");
                    const auto reference = primary_reference.compute(x, ids, routes);
                    require(std::memcmp(actual.data(), reference.data(), sizeof(actual)) == 0,
                        "DeepSeek expert eviction/remapping changed router-order arithmetic");
                    require(Stats().reserved <= static_cast<std::int64_t>(budget * 2),
                        "DeepSeek eviction exceeded the sum of independent backend ceilings");
                }
                ids = {0, 1};
                const Stats before_other;
                require(run(hm, secondary) == 1, "Secondary backend entry disappeared during primary eviction");
                const Stats after_other;
                require(after_other.misses == before_other.misses && after_other.hits >= before_other.hits + used,
                    "One backend evicted another backend's reserved expert slots");
                // The same source pointers with a different clamp must never
                // replay the earlier graph. Re-entering the old clamp tests
                // invalidation as well as the new-key miss.
                weights.segment.clamp_limit = 0.35f;
                FullResidentGraph changed_reference(weights, true, primary);
                auto changed = hm;
                changed.clamp_limit = weights.segment.clamp_limit;
                require(run(changed, primary) == 1, "Changed DeepSeek clamp did not rebuild the cache entry");
                const auto changed_expected = changed_reference.compute(x, ids, routes);
                require(std::memcmp(actual.data(), changed_expected.data(), sizeof(actual)) == 0,
                    "DeepSeek cache key omitted the clamp limit");
                require(run(hm, primary) == 1, "Original DeepSeek clamp could not be restored after eviction");
                const auto restored_expected = primary_reference.compute(x, ids, routes);
                require(std::memcmp(actual.data(), restored_expected.data(), sizeof(actual)) == 0,
                    "DeepSeek clamp restoration reused a graph with different clipping");
                // A Qwen call over the exact same pointers uses a different
                // arithmetic graph and an independent opt-in budget domain.
                auto qwen = hm;
                qwen.clamp_limit = 0;
                weights.segment.clamp_limit = 0;
                FullResidentGraph qwen_reference(weights, false, primary);
                const Stats before_qwen;
                require(tsg::host_moe_cached_experts(qwen, x.data(), ids.data(), routes.data(), actual.data(),
                    "qwen4exp fused token span", primary) == 1, "Existing Qwen explicit backend call declined");
                require(Stats().misses >= before_qwen.misses + used,
                    "Qwen reused another model family's cached arithmetic graph");
                const auto qwen_expected = qwen_reference.compute(x, ids, routes);
                require(std::memcmp(actual.data(), qwen_expected.data(), sizeof(actual)) == 0,
                    "Existing Qwen unclamped arithmetic changed under DeepSeek cache support");
                const Stats before_shrink;
                require(run(hm, primary, std::size_t(1) << 20) == 0,
                    "A workspace-only DeepSeek quota unexpectedly fit routed experts");
                require(Stats().reserved < before_shrink.reserved,
                    "A reduced DeepSeek quota retained an allocation above its explicit ceiling");
                require(run(hm, primary) == 1,
                    "DeepSeek cache could not recover after a bounded quota fallback");
                tsg::host_moe_expert_cache_release_backend(primary);
                const Stats primary_released;
                require(primary_released.reserved > 0 && primary_released.reserved <= static_cast<std::int64_t>(budget),
                    "Backend-specific release freed another backend's cache or retained its own slots");
                const Stats secondary_before;
                require(run(hm, secondary) == 1 && Stats().misses == secondary_before.misses,
                    "Backend-specific release invalidated an independent live backend");
                tsg::host_moe_expert_cache_release_backend(secondary);
                require(Stats().reserved == 0, "Backend-specific release retained cache allocations");
            }
            tsg::host_moe_expert_cache_release();
            ggml_backend_free(secondary);
            ggml_backend_free(bridge_cpu);
            g_backend = primary;
            std::printf("PASS: DeepSeek clamp, family/backend isolation, eviction and lifetime (secondary CUDA device %d)\n", second_device);
        }
        catch (...)
        {
            tsg::host_moe_expert_cache_release_backend(primary);
            tsg::host_moe_expert_cache_release_backend(secondary);
            ggml_backend_free(secondary);
            ggml_backend_free(bridge_cpu);
            g_backend = primary;
            throw;
        }
    }

#if !defined(_WIN32)
    // Keep the writer and the production read-only descriptor open throughout
    // truncation/restoration, matching a model shard changed after load.
    struct PreadWeightsFile
    {
        std::string path;
        int writer = -1;
        std::array<std::uint64_t, 3> offsets{};
        std::shared_ptr<tsg_dsv41_pread::file> source;

        explicit PreadWeightsFile(const Weights& weights)
        {
            char name[] = "/tmp/tensorsharp-expert-pread-XXXXXX";
            writer = ::mkstemp(name);
            require(writer >= 0, "Could not create selected-expert source fixture");
            path = name;
            try
            {
                std::uint64_t next = 4096;
                for (int part = 0; part < 3; ++part)
                {
                    offsets[part] = next;
                    write(weights.quantized[part].data(), weights.quantized[part].size(), next);
                    next += weights.quantized[part].size() + 4096;
                }
                source = tsg_dsv41_pread::file::open(path);
            }
            catch (...)
            {
                ::close(writer); writer = -1;
                ::unlink(path.c_str());
                throw;
            }
        }

        ~PreadWeightsFile()
        {
            if (writer >= 0) ::close(writer);
            if (!path.empty()) ::unlink(path.c_str());
        }

        void write(const std::uint8_t* bytes, std::size_t count, std::uint64_t offset)
        {
            while (count != 0)
            {
                const auto n = ::pwrite(writer, bytes, count, static_cast<off_t>(offset));
                if (n < 0 && errno == EINTR) continue;
                require(n > 0, "Could not write selected-expert source fixture");
                bytes += n; count -= static_cast<std::size_t>(n); offset += static_cast<std::uint64_t>(n);
            }
        }
    };

    void pread_cuda_checks(Weights& weights)
    {
        require(std::all_of(weights.types.begin(), weights.types.end(),
            [](ggml_type type) { return type == GGML_TYPE_MXFP4; }),
            "The later-wave selected-expert fixture requires MXFP4 weights");
        PreadWeightsFile file(weights);
        auto hm = weights.segment;
        hm.clamp_limit = 0.1f;
        weights.segment.clamp_limit = hm.clamp_limit;
        for (int part = 0; part < 3; ++part)
            hm.expert_files[part] = {file.source, file.offsets[part], weights.quantized[part].size()};
        const auto stride = weights.quantized[0].size() / experts;
        require(stride == 34816 && weights.quantized[1].size() / experts == stride
            && weights.quantized[2].size() / experts == stride,
            "Selected-expert fixture no longer forces separate 64KiB staging waves");
        // Exactly two slots before graph overhead: 2.5 experts of payload room
        // leaves ample small-graph scratch, but cannot admit a third slot.
        const std::size_t budget = (std::size_t(1) << 20) + 3 * stride * used + (3 * stride) / 2;
        std::array<float, hidden> x{}, actual{};
        std::array<std::int32_t, used> ids{0, 1};
        std::array<float, used> routes{0.6f, 0.4f};
        for (int i = 0; i < hidden; ++i) x[i] = 4.0f * std::sin(0.13f * i);
        FullResidentGraph reference(weights, true, g_backend);
        const auto run = [&] {
            return tsg::host_moe_cached_experts(hm, x.data(), ids.data(), routes.data(), actual.data(),
                "deepseek41 native", g_backend, budget, 1);
        };
        require(run() == 1, "File-backed selected-expert cache did not engage");
        auto expected = reference.compute(x, ids, routes);
        require(std::memcmp(actual.data(), expected.data(), sizeof(actual)) == 0,
            "File-backed quantized bytes/clamp differ from the full-resident graph");
        const Stats resident;
        require(resident.misses == used, "File-backed fixture did not populate exactly two initial slots");

        // Expert0 is the least recently used slot. The duplicate fresh ID takes
        // only that slot; with 34KiB per part, gate and up upload in earlier
        // waves before the down read encounters the deliberately shortened EOF.
        ids = {2, 2};
        const auto truncated_size = file.offsets[2] + 3 * stride - 1;
        require(::ftruncate(file.writer, static_cast<off_t>(truncated_size)) == 0,
            "Could not truncate a fresh expert's down source range");
        actual.fill(-123.0f);
        const auto sentinel = actual;
        require(run() == -1, "Later-wave source EOF did not fail the selected-expert cache");
        require(actual == sentinel, "Failed selected-expert upload exposed a partial result");

        // The former tag must have been cleared before gate/up were replaced.
        // Its original down bytes are before EOF and remain available to reload.
        ids = {0, 1};
        const Stats failed;
        require(run() == 1, "Cache could not reload the old victim after a later-wave source EOF");
        const Stats reloaded;
        require(reloaded.misses == failed.misses + 1 && reloaded.hits >= failed.hits + 1,
            "Failed replacement retained a stale victim tag or evicted the protected expert");
        expected = reference.compute(x, ids, routes);
        require(std::memcmp(actual.data(), expected.data(), sizeof(actual)) == 0,
            "Reloading the old victim used partially overwritten expert bytes");

        file.write(weights.quantized[2].data(), weights.quantized[2].size(), file.offsets[2]);
        ids = {2, 2};
        require(run() == 1, "Restored persistent source could not retry the failed fresh expert");
        expected = reference.compute(x, ids, routes);
        require(std::memcmp(actual.data(), expected.data(), sizeof(actual)) == 0,
            "Retry after source restoration retained failed-upload bytes or tags");
        tsg::host_moe_expert_cache_release();
        require(Stats().reserved == 0, "File-backed expert test retained cache allocations");
        std::puts("PASS: MXFP4 file-backed clamp parity, later-wave EOF, victim-tag reload and restored-source retry");
    }

    void host_cache_cuda_checks(Weights & weights)
    {
        PreadWeightsFile file(weights);
        auto hm=weights.segment;
        hm.clamp_limit=weights.segment.clamp_limit=0.1f;
        for (int p=0;p<3;++p) hm.expert_files[p]={file.source,file.offsets[p],weights.quantized[p].size()};
        const auto stride=weights.quantized[0].size()/experts;
        const std::size_t gpu_budget=(std::size_t(1)<<20)+3*stride*used+(3*stride)/2;
        std::array<float,hidden> x{},actual{};
        std::array<std::int32_t,used> ids{0,1};
        std::array<float,used> routes{0.6f,0.4f};
        for (int i=0;i<hidden;++i) x[i]=4.0f*std::sin(0.13f*i);
        FullResidentGraph reference(weights,true,g_backend);
        auto run=[&] {return tsg::host_moe_cached_experts(hm,x.data(),ids.data(),routes.data(),actual.data(),
            "deepseek41 native",g_backend,gpu_budget,1);};
        auto parity=[&] {
            const auto expected=reference.compute(x,ids,routes);
            require(std::memcmp(actual.data(),expected.data(),sizeof(actual))==0,
                "Exclusive quantized host cache changed routed clamp output bits");
            (void)HostStats();
        };
        auto restore=[&] {for (int p=0;p<3;++p) file.write(weights.quantized[p].data(),weights.quantized[p].size(),file.offsets[p]);};
        require(run()==1,"Initial exclusive host cache GPU fill failed");parity();
        require(HostStats().reserved==0,"Host tier duplicated initial GPU-resident experts");
        ids={2,2};require(run()==1,"GPU victim could not move to host cache");parity();
        require(HostStats().reserved==std::int64_t(3*stride),"Exactly one GPU victim was not retained as raw host bytes");
        require(::ftruncate(file.writer,0)==0,"Could not truncate host-hit source");
        const HostStats before_host;
        ids={0,1};require(run()==1,"Host cache hit incorrectly attempted to read truncated source");parity();
        const HostStats after_host;
        require(after_host.hits==before_host.hits+1 && after_host.sources==before_host.sources,
            "Exclusive host hit did not bypass source reads");
        require(after_host.reserved==std::int64_t(3*stride),"Host hit duplicated the exchanged GPU victim");
        restore();
        for (int expert=2;expert<experts;++expert)
        {
            ids={expert,expert};require(run()==1,"Bounded host cache eviction failed");parity();
        }
        require(HostStats().evictions>0,"One MiB host cap did not force LRU payload eviction");
        require(::ftruncate(file.writer,0)==0,"Could not truncate cold source after host eviction");
        ids={0,0};actual.fill(-123.0f);const auto sentinel=actual;
        const HostStats before_source;
        require(run()==-1 && actual==sentinel,"Missing both tiers did not preserve output on source EOF");
        const HostStats after_source;
        require(after_source.sources==before_source.sources+1,"Failed source attempt was not counted per expert");
        ids={14,15};
        require(run()==1,"Failed source replacement lost the original GPU victim's host bytes");parity();
        require(HostStats().sources==after_source.sources,"Old victim recovery unexpectedly read source");
        restore();ids={0,0};require(run()==1,"Restored source could not retry exclusive cache miss");parity();
        tsg::host_moe_expert_cache_on_drop(hm.gate_data);
        require(HostStats().reserved==0 && Stats().reserved==0,"Source release retained GPU or host expert payloads");
        ids={0,1};require(run()==1,"Entry reload after source release failed");
        ids={2,2};require(run()==1 && HostStats().reserved>0,"Reload did not create a host victim");
        tsg::host_moe_expert_cache_release_backend(g_backend);
        require(HostStats().reserved==0 && Stats().reserved==0,"Backend release retained host expert payloads");
        require(HostStats().hits>0,"Backend release unexpectedly reset monotonic host statistics");
        tsg::host_moe_expert_cache_release();
        require(HostStats().sources==0 && HostStats().hits==0,"Global release did not reset host statistics");
        std::puts("PASS: exclusive raw host victims, no-source host hits, strict LRU cap, EOF recovery and lifetime");
    }
#endif
#endif
}

int main(int argc, char** argv)
{
    try
    {
        const bool invalid_budget = argc == 3 && std::strcmp(argv[1], "--invalid-budget") == 0;
        set_environment("TS_HOST_MOE_EXPERT_CACHE_MB", invalid_budget ? argv[2] : "2");
        set_environment("TS_HOST_MOE_EXPERT_CACHE_LAYERS", "1");
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
        bool cpu_only = false, prefetch = false, deepseek = false, pread = false,host_cache=false;
        for (int arg = 1; arg < argc; ++arg)
        {
            if (std::strcmp(argv[arg], "--cpu") == 0) cpu_only = true;
            else if (std::strcmp(argv[arg], "--prefetch") == 0) prefetch = true;
            else if (std::strcmp(argv[arg], "--deepseek") == 0) deepseek = true;
            else if (std::strcmp(argv[arg], "--pread") == 0) pread = true;
            else if (std::strcmp(argv[arg], "--host-cache") == 0) {host_cache=true;pread=true;}
            else if (std::strcmp(argv[arg], "--quantization") == 0 && arg + 1 < argc)
            {
                ggml_type type = GGML_TYPE_COUNT;
                ++arg;
                if (std::strcmp(argv[arg], "iq1_m") == 0) type = GGML_TYPE_IQ1_M;
                else if (std::strcmp(argv[arg], "iq2_xxs") == 0) type = GGML_TYPE_IQ2_XXS;
                else if (std::strcmp(argv[arg], "mxfp4") == 0) type = GGML_TYPE_MXFP4;
                require(type != GGML_TYPE_COUNT, "Unsupported expert-cache parity quantization");
                weights.types = {type, type, type == GGML_TYPE_MXFP4 ? type : GGML_TYPE_IQ4_NL};
            }
            else throw std::runtime_error("Unsupported expert-cache test argument");
        }
        if (pread)
        {
            require(!cpu_only && !prefetch, "--pread requires CUDA and a separate run from --prefetch");
            deepseek = true;
            weights.types = {GGML_TYPE_MXFP4, GGML_TYPE_MXFP4, GGML_TYPE_MXFP4};
        }
        set_environment("TS_HOST_MOE_EXPERT_CACHE_PREFETCH", prefetch ? "1" : "0");
        set_environment("TS_DSV41_EXPERT_CACHE_MB", deepseek ? "2" : "0");
        set_environment("TS_DSV41_EXPERT_CACHE_PREFETCH", prefetch ? "1" : "0");
        set_environment("TS_DSV41_EXPERT_CACHE_DIAGNOSTICS", "1");
        set_environment("TS_DSV41_EXPERT_CACHE_PREAD", pread ? "1" : "0");
        set_environment("TS_DSV41_EXPERT_HOST_CACHE_MB",host_cache ? "1" : "0");
        if (pread) set_environment("TS_DSV41_EXPERT_CACHE_PREAD_STAGING_KB",host_cache ? "65536" : "64");
#if defined(_WIN32) || !defined(TSG_GGML_USE_CUDA)
        if (pread)
        {
            std::fprintf(stderr, "SKIP: selected-expert pread CUDA fixture requires Linux/POSIX and a CUDA build\n");
            return 77;
        }
#endif
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
#if !defined(_WIN32)
            if (pread)
            {
                if (host_cache) host_cache_cuda_checks(weights);
                else pread_cuda_checks(weights);
                ggml_backend_free(g_backend); g_backend = nullptr;
                return 0;
            }
#endif
            if (deepseek)
            {
                deepseek_cuda_checks(weights);
                require(prefetch == (tsg::g_test_prefetch_calls > 0),
                    "DeepSeek opt-in raw expert prefetch did not match its requested mode");
                ggml_backend_free(g_backend); g_backend = nullptr;
                return 0;
            }
            cuda_checks(weights);
            require(prefetch == (tsg::g_test_prefetch_calls > 0),
                "Opt-in raw expert prefetch did not match its requested mode");
            ggml_backend_free(g_backend);
            g_backend = nullptr;
            std::puts("PASS: CUDA compact expert cache, warm hits, eviction, invalidation, reload and budget");
            return 0;
        }
#else
        (void)cpu_only;
        (void)deepseek;
#endif
        g_backend = ggml_backend_cpu_init();
        require(g_backend != nullptr, "Could not initialize CPU backend");
        fallback_checks(weights);
        std::array<float, hidden> x{}, out{};
        std::array<std::int32_t, used> ids{0, 1};
        std::array<float, used> routes{0.6f, 0.4f};
        require(tsg::host_moe_cached_experts(weights.segment, x.data(), ids.data(), routes.data(), out.data(),
            "qwen4exp fused token span") == 0, "CPU backend incorrectly engaged the CUDA expert cache");
        require(Stats().reserved == 0 && Stats().calls == 0, "Fallback path allocated CUDA cache state");
        tsg::host_moe_expert_cache_release();
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
