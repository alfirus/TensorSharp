// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Qwen4Exp's optional CUDA expert tier. GGUF bytes remain quantized and are
// copied into compact, persistent slots; ordinary upstream ggml CUDA kernels
// compute the complete routed FFN. No CPU/GPU partial-sum split is involved.
#include "ggml_ops_internal.h"
#include "ggml_ops_moe_prefetch.h"
#include "ggml_ops_file_source.h"
#include "ggml_ops_host_moe_file_batch.h"
#include "ggml_ops_moe_prefetch_policy.h"
#include "ggml_ops_expert_eviction.h"
#include "ggml_ops_precision_policy.h"
#include "ggml_ops_host_moe_cache_policy.h"
#include "ggml_ops_host_moe_cache_config.h"
#include "ggml_ops_host_moe_cache_identity.h"
#include "ggml_ops_shared_cache_budget.h"
#include "ggml-impl.h"
#ifdef TSG_GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cerrno>
#include <cstdlib>
#include <list>

namespace tsg
{
    namespace
    {
        constexpr std::size_t kMiB = std::size_t(1) << 20;
        // This allowance covers small upstream CUDA work buffers in addition
        // to every byte the graph allocator reports. CUDA's shared backend
        // pool and driver capture storage also need physical-device headroom;
        // their total cannot be inferred from the graph allocator.
        constexpr std::size_t kWorkspaceAllowance = kMiB;
        constexpr std::size_t kDeviceHeadroom = std::size_t(512) << 20;
        constexpr int kMaxUsed = 64;

        std::mutex g_cache_mutex;
        std::size_t g_reserved = 0;
        std::uint64_t g_hits = 0, g_misses = 0, g_calls = 0;
        std::uint64_t g_graph_submissions = 0, g_joint_submissions = 0, g_joint_rows = 0;
        std::uint64_t g_growth_attempts = 0, g_growth_successes = 0, g_growth_refusals = 0;
        std::uint64_t g_growth_failures = 0, g_growth_noops = 0, g_growth_hot_slots = 0, g_growth_copy_bytes = 0;
        std::size_t g_growth_peak_accounted = 0;
#if defined(TSG_GGML_TEST_HOOKS)
        std::atomic<int> g_test_failure_stage{0};
        std::atomic<std::size_t> g_test_physical_bytes{0};
        std::atomic<std::uint64_t> g_test_completed_output_drains{0};
        std::atomic<std::size_t> g_test_available_bytes{std::numeric_limits<std::size_t>::max()};
        std::atomic<std::size_t> g_test_file_slice_bytes{std::size_t(32) << 20};
        bool test_failure(int stage)
        {
            int expected = stage;
            return g_test_failure_stage.compare_exchange_strong(expected, 0);
        }
#endif

        std::size_t read_positive_size(const char* name, std::size_t fallback)
        {
            const char* value = std::getenv(name);
            if (value == nullptr || value[0] == '\0') return fallback;
            // Match the managed placement parser: unsigned decimal digits,
            // with no sign or surrounding whitespace.
            const char* digits = value;
            if (digits[0] < '0' || digits[0] > '9') return 0;
            for (const char* p = digits; *p != '\0'; ++p)
                if (*p < '0' || *p > '9') return 0;
            errno = 0;
            char* end = nullptr;
            const unsigned long long n = std::strtoull(value, &end, 10);
            if (errno != 0 || end == value || *end != '\0'
                || n > std::numeric_limits<std::size_t>::max()) return 0;
            return static_cast<std::size_t>(n);
        }

        HostMoeCacheConfiguration& cache_configuration()
        {
            static HostMoeCacheConfiguration configuration([] {
                const std::size_t mb = read_positive_size("TS_HOST_MOE_EXPERT_CACHE_MB", 0);
                const auto maximum = static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
                return HostMoeCacheSettings{mb <= maximum / kMiB ? mb * kMiB : 0,
                    read_positive_size("TS_HOST_MOE_EXPERT_CACHE_LAYERS", 48),
                    host_moe_cache_prefill_rows(std::getenv("TS_HOST_MOE_EXPERT_CACHE_PREFILL_ROWS"))};
            }());
            return configuration;
        }

        std::size_t cache_budget()
        {
            return cache_configuration().fast_budget();
        }

        bool diagnostics()
        {
            static const bool enabled = [] {
                const char* e = std::getenv("TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS");
                return e != nullptr && e[0] == '1';
            }();
            return enabled;
        }

        bool route_trace()
        {
            static const bool enabled = [] {
                const char* value = std::getenv("TS_HOST_MOE_ROUTE_TRACE");
                return value && std::strcmp(value, "1") == 0;
            }();
            return enabled;
        }

        bool frequency_eviction()
        {
            static const bool enabled = [] {
                const char* value = std::getenv("TS_HOST_MOE_EXPERT_CACHE_LFU");
                return value == nullptr || std::strcmp(value, "1") == 0;
            }();
            return enabled;
        }

        struct CacheEntry
        {
            ggml_backend_t backend = nullptr;
            int layer = -1, hidden = 0, n_ff = 0, n_used = 0, num_experts = 0;
            const void* source[3] = {};
            int type[3] = {};
            std::size_t stride[3] = {};
            std::size_t bytes = 0;
            int capacity = 0, graph_rows = 1, bulk_rows = 0;
            std::uint64_t growth_generation = 0;
            bool pooled = false;
            int pool_expert_extent = 0;
            float swiglu_clamp = 0.0f;
            // One exact-width variant for each live batch width. All variants
            // share the expert leaves, but never execute padded/dummy lanes.
            std::vector<ggml_tensor*> width_inputs, width_ids, width_routes, width_outputs;
            std::vector<ggml_cgraph> width_graphs;
            std::vector<int> expert_for_slot;
            std::vector<ExpertSourceIdentity> source_for_slot;
            std::vector<std::uint64_t> last_used;
            std::uint64_t clock = 0;
            ExpertEvictionPolicy eviction_policy;
            ExpertPrefetchPolicy prefetch_policy;
            std::uint64_t prefetch_observations = 0, prefetch_reads = 0;
            std::uint64_t file_read_calls = 0, file_read_bytes = 0;
            ggml_context* ctx = nullptr;
            ggml_gallocr_t allocator = nullptr;
            ggml_cgraph* graph = nullptr;
            ggml_tensor* weight[3] = {};
            ggml_tensor* input = nullptr;
            ggml_tensor* ids = nullptr;
            ggml_tensor* routes = nullptr;
            ggml_tensor* output = nullptr;
            // The complete compact graph (weights, routing/activation payload,
            // padding and the existing workspace allowance) is one kind-4 owner.
            // This allocator deliberately does not use the kind-2 graph wrapper.
            std::shared_ptr<SharedCacheCharge> shared_charge;
#if defined(TSG_GGML_TEST_HOOKS)
            std::size_t test_physical_bytes = 0;
#endif

            ~CacheEntry()
            {
                if (diagnostics() && file_read_calls)
                    std::fprintf(stderr, "[HOSTMOE-CACHE-FILE] layer=%d calls=%llu file_bytes=%llu\n", layer,
                        static_cast<unsigned long long>(file_read_calls), static_cast<unsigned long long>(file_read_bytes));
                if (diagnostics() && prefetch_observations != 0)
                    std::fprintf(stderr, "[HOSTMOE-CACHE-PREFETCH] layer=%d observed=%llu parallel_reads=%llu\n",
                        layer, static_cast<unsigned long long>(prefetch_observations),
                        static_cast<unsigned long long>(prefetch_reads));
                // The last direct output copy can still be queued after the
                // helper returns. Drain it before eviction/invalidation frees
                // any source storage; CUDA remains alive at both release hooks.
                if (allocator != nullptr)
                {
                    sync_backend(backend);
                    ggml_gallocr_free(allocator);
                    allocator = nullptr;
#if defined(TSG_GGML_TEST_HOOKS)
                    g_test_physical_bytes.fetch_sub(test_physical_bytes);
                    test_physical_bytes = 0;
#endif
                }
                if (ctx != nullptr) ggml_free(ctx);
                // ggml's free is void/fatal on unrecoverable CUDA errors. Never
                // refund before it returns, including allocation/commit rollback.
                shared_charge.reset();
            }

            bool matches(const HostMoeSegment& hm, ggml_backend_t b) const
            {
                return backend == b && pooled == (hm.pooled_model_experts > 0)
                    && (pooled || layer == hm.layer) && hidden == hm.hidden
                    && n_ff == hm.n_ff && n_used == hm.n_used && num_experts == hm.num_experts
                    && (pooled || (source[0] == hm.gate_data && source[1] == hm.up_data && source[2] == hm.down_data))
                    && type[0] == hm.gate_type && type[1] == hm.up_type && type[2] == hm.down_type
                    && swiglu_clamp == hm.swiglu_clamp;
            }
            std::array<const void*, 3> current_source() const { return {source[0], source[1], source[2]}; }
            bool slot_matches(int slot, int expert) const {
                return pooled ? source_for_slot[slot].matches(current_source(), expert)
                    : expert_for_slot[slot] == expert;
            }
            void invalidate_slot(int slot) {
                expert_for_slot[slot] = -1; source_for_slot[slot].invalidate();
            }
            void publish_slot(int slot, int expert) {
                expert_for_slot[slot] = expert; source_for_slot[slot].publish(current_source(), expert);
            }
            bool references(const void* pointer) const {
                if (source[0] == pointer || source[1] == pointer || source[2] == pointer) return true;
                for (const auto& key : source_for_slot) if (key.references(pointer)) return true;
                return false;
            }
        };

        // LRU entries bound multiple loaded model identities as well as the
        // per-layer slots. Each layer has at most budget / layer_count bytes,
        // preventing early layers from consuming every slot on the first token.
        std::list<std::unique_ptr<CacheEntry>> g_entries;
        std::unordered_map<const void*, std::shared_ptr<ExpertFileSource>> g_file_sources;
        // One process-wide workspace under g_cache_mutex, never one per layer.
        // Kept through asynchronous uploads, drained on failure and released at
        // cache teardown. Larger routed sets retain the ordinary mmap path.
        constexpr std::size_t kFileReadWorkspace = std::size_t(32) << 20;
        struct FileReadWorkspace {
            ggml_backend_buffer_t buffer = nullptr;
            std::shared_ptr<SharedCacheCharge> shared_charge;
            std::size_t size() const { return buffer ? ggml_backend_buffer_get_size(buffer) : 0; }
            std::uint8_t* data() const { return static_cast<std::uint8_t*>(ggml_backend_buffer_get_base(buffer)); }
            void clear() {
                if (buffer) ggml_backend_buffer_free(buffer);
                buffer = nullptr;
                shared_charge.reset(); // Return host credit after physical free.
            }
            ~FileReadWorkspace() { clear(); }
            void reserve(std::size_t bytes) {
                if (bytes <= size()) return;
                if (bytes > kFileReadWorkspace) throw std::runtime_error("Expert read workspace exceeds its ceiling");
                // The previous row has completed; free before growing so the
                // transfer arena never temporarily holds two allocations.
                clear();
                auto charge = SharedCacheCharge::reserve(0, 3, bytes);
                if (!charge) throw std::runtime_error("Expert read workspace exceeds shared host budget");
#ifdef TSG_GGML_USE_CUDA
                auto type = ggml_backend_cuda_host_buffer_type();
#else
                auto type = ggml_backend_cpu_buffer_type();
#endif
                // Upstream falls back to ordinary RAM when pinning fails.
                // Only this small transfer arena is pinned, never model views.
                buffer = ggml_backend_buft_alloc_buffer(type, bytes);
                if (!buffer) throw std::runtime_error("Cannot allocate expert read workspace");
                if (!charge->commit(size())) {
                    clear();
                    throw std::runtime_error("Cannot commit expert read workspace to shared host budget");
                }
                shared_charge = std::move(charge);
            }
        } g_file_read_buffer;

#ifdef TSG_GGML_USE_CUDA
        bool valid_layout(const HostMoeSegment& hm, std::size_t (&stride)[3], int max_rows, bool explicit_model = false)
        {
            if (hm.seq_len <= 0 || hm.seq_len > max_rows
                || !host_moe_cache_routes_fit(hm.seq_len, hm.n_used)
                || (!explicit_model && hm.seq_len > TSG_PRECISION_DECODE_COLUMNS
                    && (g_device_count.load(std::memory_order_relaxed) != 1 || g_active_rank != 0
                        || tp_global_degree(1) != 1))
                || hm.activation != 0 || hm.tp_reduced != 0
                || !std::isfinite(hm.swiglu_clamp) || hm.swiglu_clamp < 0.0f
                || hm.pooled_model_experts < 0 || hm.pooled_model_experts > 65536
                || (hm.pooled_model_experts > 0 && (!explicit_model || hm.pooled_model_experts < hm.num_experts))
                || hm.layer < 0 || hm.hidden <= 0 || hm.n_ff <= 0
                || hm.n_used <= 0 || hm.n_used > kMaxUsed || hm.n_used > hm.num_experts
                || hm.gate_data == nullptr || hm.up_data == nullptr || hm.down_data == nullptr
                || hm.gate_bias != nullptr || hm.up_bias != nullptr || hm.down_bias != nullptr
                || hm.gate_ne0 != hm.hidden || hm.up_ne0 != hm.hidden
                || hm.gate_ne1 != hm.n_ff || hm.up_ne1 != hm.n_ff
                || hm.down_ne0 != hm.n_ff || hm.down_ne1 != hm.hidden)
                return false;
            const int types[3] = {hm.gate_type, hm.up_type, hm.down_type};
            const int widths[3] = {hm.hidden, hm.hidden, hm.n_ff};
            const int rows[3] = {hm.n_ff, hm.n_ff, hm.hidden};
            const std::int64_t totals[3] = {hm.gate_bytes, hm.up_bytes, hm.down_bytes};
            for (int i = 0; i < 3; ++i)
            {
                if (types[i] < 0 || types[i] >= GGML_TYPE_COUNT
                    || !ggml_is_quantized(static_cast<ggml_type>(types[i]))) return false;
                const int64_t block = ggml_blck_size(static_cast<ggml_type>(types[i]));
                if (block <= 0 || widths[i] % block != 0) return false;
                const std::size_t row = ggml_row_size(static_cast<ggml_type>(types[i]), widths[i]);
                if (row > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(rows[i]))
                    return false;
                stride[i] = row * static_cast<std::size_t>(rows[i]);
                if (stride[i] > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(hm.num_experts)
                    || totals[i] <= 0 || static_cast<std::uint64_t>(totals[i])
                        != stride[i] * static_cast<std::size_t>(hm.num_experts)) return false;
            }
            return true;
        }

        bool build_graph(CacheEntry& entry, int capacity, std::size_t quota, int graph_rows, int bulk_rows = 0)
        {
            ggml_init_params params{};
            const int variants = graph_rows + (bulk_rows > 0 ? 1 : 0);
            const std::size_t graph_size = 256 * static_cast<std::size_t>(variants);
            params.mem_size = ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false);
            params.no_alloc = true;
            entry.ctx = ggml_init(params);
            if (entry.ctx == nullptr) return false;
            auto* ctx = entry.ctx;
            entry.capacity = capacity;
            // Pool identities include source triples. The existing local-ID
            // frequency counter is meaningful only for a single layer.
            entry.eviction_policy.configure(entry.pooled ? entry.pool_expert_extent : entry.num_experts,
                capacity, !entry.pooled && frequency_eviction());
            entry.weight[0] = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(entry.type[0]),
                entry.hidden, entry.n_ff, capacity);
            entry.weight[1] = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(entry.type[1]),
                entry.hidden, entry.n_ff, capacity);
            entry.weight[2] = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(entry.type[2]),
                entry.n_ff, entry.hidden, capacity);
            for (auto* w : entry.weight)
            {
                // Slot bytes survive every execution of the dedicated allocator.
                ggml_set_input(w);
                ggml_set_output(w);
            }
            entry.graph_rows = graph_rows; entry.bulk_rows = bulk_rows;
            entry.width_inputs.resize(variants); entry.width_ids.resize(variants);
            entry.width_routes.resize(variants); entry.width_outputs.resize(variants);
            entry.width_graphs.reserve(variants);
            entry.graph = ggml_new_graph_custom(ctx, graph_size, false);
            // Allocate the supergraph once, including every retained OUTPUT.
            // Only a disjoint exact-width slice is submitted; shared weight
            // leaves are already resident and need no graph nodes of their own.
            static std::atomic<std::uint64_t> next_uid{UINT64_C(0x484d4f4500000000)};
            entry.graph->uid = next_uid.fetch_add(1, std::memory_order_relaxed);
            // Narrow variants remain independent scalar rows. Exactly one
            // wide variant uses ordinary bulk MMID geometry and the same
            // quantized weight leaves, measured in this one allocator.
            for (int index = 0; index < variants; ++index) {
                const bool bulk = index == graph_rows;
                const int width = bulk ? bulk_rows : index + 1;
                const int start = ggml_graph_n_nodes(entry.graph);
                auto* input = entry.width_inputs[index] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, entry.hidden, bulk ? 1 : width, bulk ? width : 1);
                auto* ids = entry.width_ids[index] = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, bulk ? entry.n_used : entry.n_used * width, bulk ? width : 1);
                // The caller's row-major [K,N] routes stay a single upload.
                auto* routes = entry.width_routes[index] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, entry.n_used, width);
                ggml_set_input(input); ggml_set_input(ids); ggml_set_input(routes);
                auto* gate = ggml_mul_mat_id(ctx, entry.weight[0], input, ids);
                auto* up = ggml_mul_mat_id(ctx, entry.weight[1], input, ids);
                auto* gate_projection = gate;
                auto* up_projection = up;
                if (entry.swiglu_clamp > 0.0f) {
                    up = ggml_clamp(ctx, up, -entry.swiglu_clamp, entry.swiglu_clamp);
                    gate = ggml_clamp(ctx, gate, -INFINITY, entry.swiglu_clamp);
                }
                auto* activation = ggml_mul(ctx, ggml_silu(ctx, gate), up);
                auto* down = ggml_mul_mat_id(ctx, entry.weight[2], activation, ids);
                // j=k*N+r: ordinary upstream scalar MMVQ sees ncols_dst=1,
                // expert=ids[j], input channel=j%N. Each projection is one
                // N-request operator/launch, with the same per-channel scalar
                // quantization and dot reduction as N independent calls.
                auto* down_by_route = bulk || width == 1 ? down
                    : ggml_cont(ctx, ggml_permute(ctx,
                        ggml_reshape_3d(ctx, down, entry.hidden, width, entry.n_used), 0, 2, 1, 3));
                // Restore [H,K,N] before the weighted reduction. Upstream's
                // scalar and batched weighted-reduction fusion recognizes this
                // layout; [H,N,K] would change both fusion and FMA rounding.
                // Protect its actual CONT producer against in-place MUL reuse.
                if (!bulk) ggml_set_output(down_by_route);
                auto* weighted = ggml_mul(ctx, down_by_route, routes);
                auto* sum = ggml_view_2d(ctx, weighted, entry.hidden, width, weighted->nb[2], 0);
                for (int k = 1; k < entry.n_used; ++k)
                    sum = ggml_add(ctx, sum, ggml_view_2d(ctx, weighted, entry.hidden, width,
                        weighted->nb[2], static_cast<std::size_t>(k) * weighted->nb[1]));
                // Each ADD handles all requests, preserving each row's exact
                // original k=0..K-1 weighted left-fold order.
                auto* output = entry.width_outputs[index] = ggml_cont(ctx, sum);
                ggml_set_output(output);
                ggml_build_forward_expand(entry.graph, output);
                entry.width_graphs.push_back(ggml_graph_view(entry.graph, start, ggml_graph_n_nodes(entry.graph)));
                entry.width_graphs.back().uid = next_uid.fetch_add(1, std::memory_order_relaxed);
                if (!ggml_backend_supports_op(entry.backend, gate_projection)
                    || !ggml_backend_supports_op(entry.backend, up_projection)
                    || !ggml_backend_supports_op(entry.backend, gate)
                    || !ggml_backend_supports_op(entry.backend, up)
                    || !ggml_backend_supports_op(entry.backend, down)) return false;
            }
            entry.input = entry.width_inputs[0]; entry.ids = entry.width_ids[0];
            entry.routes = entry.width_routes[0]; entry.output = entry.width_outputs[0];

            entry.allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(entry.backend));
            if (entry.allocator == nullptr) return false;
            std::size_t device_bytes = 0;
            ggml_gallocr_reserve_n_size(entry.allocator, entry.graph, nullptr, nullptr, &device_bytes);
            if (device_bytes > quota || quota - device_bytes < kWorkspaceAllowance) return false;
            entry.bytes = device_bytes + kWorkspaceAllowance;
            return true;
        }

        std::unique_ptr<CacheEntry> create_entry(const HostMoeSegment& hm,
            const std::size_t (&stride)[3], ggml_backend_t backend, std::size_t quota,
            std::size_t maximum_capacity, int graph_rows = 1, int minimum_capacity = 0, int bulk_rows = 0)
        {
            minimum_capacity = std::max(minimum_capacity, hm.n_used);
            std::size_t per_expert = 0;
            for (const auto s : stride)
            {
                if (s > std::numeric_limits<std::size_t>::max() - per_expert) return nullptr;
                per_expert += s;
            }
            if (quota <= kWorkspaceAllowance || per_expert == 0) return nullptr;
            std::size_t capacity = std::min({static_cast<std::size_t>(hm.pooled_model_experts > 0
                    ? hm.pooled_model_experts : hm.num_experts),
                maximum_capacity, (quota - kWorkspaceAllowance) / per_expert});
            if (capacity < static_cast<std::size_t>(minimum_capacity)) return nullptr;

            // Size-only candidates own CPU graph metadata, never backing device
            // buffers or a shared ticket. Destroy each rejected plan before the
            // next candidate so fitting cannot hide physical admission failures.
            const auto plan = [&](std::size_t slots) {
                auto entry = std::make_unique<CacheEntry>();
                entry->backend = backend;
                entry->layer = hm.layer; entry->hidden = hm.hidden; entry->n_ff = hm.n_ff;
                entry->n_used = hm.n_used; entry->num_experts = hm.num_experts;
                entry->pooled = hm.pooled_model_experts > 0;
                entry->pool_expert_extent = hm.pooled_model_experts;
                entry->swiglu_clamp = hm.swiglu_clamp;
                entry->source[0] = hm.gate_data; entry->source[1] = hm.up_data; entry->source[2] = hm.down_data;
                entry->type[0] = hm.gate_type; entry->type[1] = hm.up_type; entry->type[2] = hm.down_type;
                for (int i = 0; i < 3; ++i) entry->stride[i] = stride[i];
                if (!build_graph(*entry, static_cast<int>(slots), quota, graph_rows, bulk_rows)) entry.reset();
                return entry;
            };
            auto largest = plan(capacity);
            if (largest) return largest;
            if (hm.pooled_model_experts > 0) {
                // An explicit GLM layout pool can contain thousands of experts.
                // Graph scratch is almost independent of slot count; reducing
                // all payload by 25% wastes the quota after a one-byte miss.
                // For fixed graph geometry the allocator's weight/tail extent
                // is nondecreasing with capacity. At most 16 size-only probes
                // search the existing <=65536 pool bound, then rebuild one plan.
                std::size_t low = minimum_capacity, high = capacity - 1, best = 0;
                while (low <= high) {
                    const auto middle = low + (high - low) / 2;
                    auto candidate = plan(middle);
                    if (candidate) { best = middle; low = middle + 1; }
                    else { high = middle - 1; }
                }
                return best ? plan(best) : nullptr;
            }
            // Preserve the original Qwen capacity policy and admission behavior.
            for (;;) {
                if (capacity <= static_cast<std::size_t>(minimum_capacity)) return nullptr;
                capacity = std::max<std::size_t>(minimum_capacity, capacity * 3 / 4);
                auto entry = plan(capacity);
                if (entry) return entry;
            }
        }

        // Only an admission refusal may try a smaller slot table. Once physical
        // allocation starts, any failure must unwind the one owner and stop;
        // retrying then could hide a CUDA/commit error behind a successful graph.
        enum class Admission { refused, failed, admitted };

        Admission allocate_entry(CacheEntry& entry)
        {
            std::size_t free_bytes = 0, total_bytes = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(entry.backend), &free_bytes, &total_bytes);
#if defined(TSG_GGML_TEST_HOOKS)
            // Inject a tighter observation, never invent additional physical VRAM.
            free_bytes = std::min(free_bytes, g_test_available_bytes.load());
#endif
            if (total_bytes == 0 || free_bytes <= kDeviceHeadroom
                || entry.bytes > free_bytes - kDeviceHeadroom) return Admission::refused;
            entry.shared_charge = SharedCacheCharge::reserve(g_active_rank, 4, entry.bytes);
            if (!entry.shared_charge) return Admission::refused;
#if defined(TSG_GGML_TEST_HOOKS)
            if (test_failure(1)) return Admission::failed;
#endif
            // Size-only reservation above creates a valid allocation plan with
            // no backing buffers. alloc_graph alone sees the matching plan and
            // does not allocate them; explicitly materialize that reservation.
            if (!ggml_gallocr_reserve(entry.allocator, entry.graph)) return Admission::failed;
#if defined(TSG_GGML_TEST_HOOKS)
            entry.test_physical_bytes = ggml_gallocr_get_buffer_size(entry.allocator, 0);
            g_test_physical_bytes.fetch_add(entry.test_physical_bytes);
            if (test_failure(2)) return Admission::failed;
#endif
            if (!ggml_gallocr_alloc_graph(entry.allocator, entry.graph)) return Admission::failed;
            const std::size_t actual = ggml_gallocr_get_buffer_size(entry.allocator, 0);
            if (actual > entry.bytes - kWorkspaceAllowance) return Admission::failed;
            // Keep the exact reserved charge, even if the materialized buffer is
            // smaller than the size-only plan. The callback ticket cannot shrink;
            // cache statistics/trim must report the same committed ownership.
            // CUDA initializes each quantized tensor's over-read tail at alloc.
            // Clear its complete payload as well: unused slots must never hold
            // stale NaNs from a prior graph allocation.
            const auto errors_before = g_ggml_error_count.load(std::memory_order_acquire);
            for (auto* w : entry.weight) ggml_backend_tensor_memset(w, 0, 0, ggml_nbytes(w));
            entry.expert_for_slot.assign(entry.capacity, -1);
            entry.source_for_slot.resize(entry.capacity);
            entry.last_used.assign(entry.capacity, 0);
            sync_backend(entry.backend);
            if (g_ggml_error_count.load(std::memory_order_acquire) != errors_before
                || g_backend_compute_failed.load(std::memory_order_acquire)
                || !entry.shared_charge->commit(entry.bytes)) return Admission::failed;
#if defined(TSG_GGML_TEST_HOOKS)
            if (test_failure(3)) return Admission::failed;
#endif
            return Admission::admitted;
        }

        enum class Growth { unchanged, refused, replaced };

        // Caller holds the cache gate. The source table remains published and
        // owns its payload throughout planning, admission and copying. Neither
        // an optional refusal nor a copy failure may turn it into a cold table.
        Growth grow_entry(std::unique_ptr<CacheEntry>& published,
            std::unique_ptr<CacheEntry> candidate, std::size_t budget)
        {
            if (!candidate) { ++g_growth_refusals; return Growth::refused; }
            auto& old = *published;
            auto& next = *candidate;
            if (next.capacity == old.capacity && next.graph_rows == old.graph_rows && next.bulk_rows == old.bulk_rows) {
                ++g_growth_noops; return Growth::unchanged;
            }
            if (next.capacity < old.capacity || next.graph_rows < old.graph_rows
                || next.backend != old.backend || next.hidden != old.hidden || next.n_ff != old.n_ff
                || next.n_used != old.n_used || next.num_experts != old.num_experts
                || next.pooled != old.pooled || next.pool_expert_extent != old.pool_expert_extent
                || next.swiglu_clamp != old.swiglu_clamp)
                throw std::logic_error("Expert-cache growth changed its source or execution identity");
            for (int part = 0; part < 3; ++part)
                if (next.type[part] != old.type[part] || next.stride[part] != old.stride[part]
                    || (!old.pooled && next.source[part] != old.source[part]))
                    throw std::logic_error("Expert-cache growth changed its quantized layout");
            // The scope ceiling covers temporary coexistence, not just the
            // eventual net increase. Shared admission independently charges the
            // same old+new peak and observes physical device headroom.
            if (next.bytes > budget || g_reserved > budget - next.bytes) {
                ++g_growth_refusals; return Growth::refused;
            }
            const auto admission = allocate_entry(next);
            if (admission == Admission::refused) { ++g_growth_refusals; return Growth::refused; }
            if (admission == Admission::failed) {
                ++g_growth_failures;
                throw std::runtime_error("Expert-cache growth payload allocation or commit failed");
            }
            g_reserved += next.bytes;
            g_growth_peak_accounted = (std::max)(g_growth_peak_accounted, g_reserved);
            // Explicitly destroy the candidate before private-accounting
            // rollback. Its destructor drains all copies, frees device storage,
            // then refunds the real shared ticket.
            struct Rollback {
                std::unique_ptr<CacheEntry>& candidate;
                std::size_t bytes;
                bool published = false;
                ~Rollback() { if (!published) { candidate.reset(); g_reserved -= bytes; } }
            } rollback{candidate, next.bytes};
            try {
                // Any metadata allocation can fail while the old owner is
                // still intact. Fresh tail slots retain their invalid markers.
                std::copy(old.expert_for_slot.begin(), old.expert_for_slot.end(), next.expert_for_slot.begin());
                std::copy(old.source_for_slot.begin(), old.source_for_slot.end(), next.source_for_slot.begin());
                std::copy(old.last_used.begin(), old.last_used.end(), next.last_used.begin());
                next.clock = old.clock;
                next.eviction_policy = old.eviction_policy;
                next.eviction_policy.grow_capacity(next.capacity);
                next.prefetch_policy = old.prefetch_policy;
                next.prefetch_observations = old.prefetch_observations;
                next.prefetch_reads = old.prefetch_reads;
                next.growth_generation = cache_configuration().growth_generation();
                const auto epoch = g_ggml_error_count.load(std::memory_order_acquire);
                sync_backend(old.backend);
#if defined(TSG_GGML_TEST_HOOKS)
                if (test_failure(8)) throw std::runtime_error("Injected expert-cache growth before copying");
#endif
                // Matching contiguous views copy the entire old slot extent.
                // No weight is read from GGUF or requantized, and slot indices
                // remain identical. Descriptors live until the final fence.
                std::array<ggml_tensor, 3> sources{}, destinations{};
                struct CopyDrain {
                    ggml_backend_t backend;
                    bool pending = false;
                    ~CopyDrain() { if (pending) sync_backend(backend); }
                } copies{next.backend};
                std::uint64_t bytes = 0, hot = 0;
                for (const int expert : old.expert_for_slot) if (expert >= 0) ++hot;
                for (int part = 0; part < 3; ++part) {
                    sources[part] = *old.weight[part];
                    destinations[part] = *next.weight[part];
                    destinations[part].ne[2] = old.capacity;
                    destinations[part].nb[3] = sources[part].nb[3];
                    bytes += ggml_nbytes(&sources[part]);
                    copies.pending = true;
                    ggml_backend_tensor_copy_async(old.backend, next.backend, &sources[part], &destinations[part]);
#if defined(TSG_GGML_TEST_HOOKS)
                    if (part == 0 && test_failure(9))
                        throw std::runtime_error("Injected expert-cache growth after queued copy");
#endif
                }
                sync_backend(next.backend);
                copies.pending = false;
                if (g_ggml_error_count.load(std::memory_order_acquire) != epoch
                    || g_backend_compute_failed.load(std::memory_order_acquire))
                    throw std::runtime_error("Expert-cache growth device copy failed");
#if defined(TSG_GGML_TEST_HOOKS)
                if (test_failure(10)) throw std::runtime_error("Injected expert-cache growth before publication");
#endif
                const auto retired_bytes = old.bytes;
                auto retired = std::move(published);
                published = std::move(candidate); // Existing list node: no allocation can throw.
                rollback.published = true;
                retired.reset(); // Physical free and real credit return precede private refund.
                g_reserved -= retired_bytes;
                ++g_growth_successes;
                g_growth_hot_slots += hot; g_growth_copy_bytes += bytes;
                if (diagnostics()) std::fprintf(stderr,
                    "[HOSTMOE-CACHE-GROW] layer=%d slots=%d hot=%llu copied=%llu reserved=%zu budget=%zu\n",
                    published->layer, published->capacity, static_cast<unsigned long long>(hot),
                    static_cast<unsigned long long>(bytes), g_reserved, budget);
                return Growth::replaced;
            } catch (...) { ++g_growth_failures; throw; }
        }

        void copy_slot_part(CacheEntry& entry, int slot, int expert, int part,
            std::vector<std::pair<std::size_t, std::size_t>>& pieces)
        {
            const auto* src = static_cast<const std::uint8_t*>(entry.source[part])
                + static_cast<std::size_t>(expert) * entry.stride[part];
            // Respect prior registrations without pinning the expert stack or
            // creating a second RAM copy. Only the caller issues these uploads.
            host_pin_split(src, entry.stride[part], pieces);
            for (const auto& p : pieces)
                ggml_backend_tensor_set_async(entry.backend, entry.weight[part], src + p.first,
                    static_cast<std::size_t>(slot) * entry.stride[part] + p.first, p.second);
        }

        void copy_slot(CacheEntry& entry, int slot, int expert)
        {
            std::vector<std::pair<std::size_t, std::size_t>> pieces;
            for (int i = 0; i < 3; ++i)
                copy_slot_part(entry, slot, expert, i, pieces);
        }

        void copy_joint_file_misses(CacheEntry& entry,
            const std::vector<std::pair<int, int>>& misses,
            const std::array<std::shared_ptr<ExpertFileSource>, 3>& files, std::size_t bytes)
        {
            // A joint union can exceed the old one-row transfer ceiling. Read
            // bounded slices, never fall through to touching mmap weights.
            std::size_t window = kFileReadWorkspace;
#if defined(TSG_GGML_TEST_HOOKS)
            window = std::min(window, g_test_file_slice_bytes.load());
#endif
            if (window == 0) throw std::runtime_error("Joint expert file slice limit must be positive");
            window = std::min(bytes, window);
            g_file_read_buffer.reserve(window);
            struct Destination { int slot, part; std::size_t offset; };
            std::vector<ExpertReadTask> tasks;
            std::vector<Destination> destinations;
            std::size_t used = 0;
            auto flush = [&] {
                if (tasks.empty()) return;
                read_expert_files(tasks, [&](std::size_t i) {
                    const auto& d = destinations[i];
                    entry.file_read_bytes += tasks[i].bytes;
                    ggml_backend_tensor_set_async(entry.backend, entry.weight[d.part], tasks[i].destination,
                        static_cast<std::size_t>(d.slot) * entry.stride[d.part] + d.offset, tasks[i].bytes);
                });
                sync_backend(entry.backend); // source ring cannot be reused before H2D drains
                if (g_backend_compute_failed.load(std::memory_order_acquire))
                    throw std::runtime_error("Joint expert file upload failed");
                tasks.clear(); destinations.clear(); used = 0;
            };
            for (const auto& miss : misses) for (int part = 0; part < 3; ++part) {
                std::size_t offset = 0;
                while (offset < entry.stride[part]) {
                    if (used == window) flush();
                    const auto count = std::min(entry.stride[part] - offset, window - used);
                    tasks.push_back({files[part].get(), static_cast<std::size_t>(miss.second) * entry.stride[part] + offset,
                        g_file_read_buffer.data() + used, count});
                    destinations.push_back({miss.first, part, offset});
                    used += count; offset += count;
                }
            }
            flush();
            for (const auto& miss : misses) entry.publish_slot(miss.first, miss.second);
        }

        ggml_backend_buffer_t tensor_buffer(const ggml_tensor* tensor)
        {
            while (tensor != nullptr && tensor->buffer == nullptr) tensor = tensor->view_src;
            return tensor != nullptr ? tensor->buffer : nullptr;
        }

        bool device_source(const ggml_tensor* tensor, std::size_t bytes, ggml_backend_t backend)
        {
            const auto buffer = tensor_buffer(tensor);
            return tensor != nullptr && tensor->type == GGML_TYPE_F32 && tensor->data != nullptr
                && ggml_is_contiguous(tensor) && ggml_nbytes(tensor) >= bytes
                && buffer != nullptr && ggml_backend_buffer_get_type(buffer)
                    == ggml_backend_get_default_buffer_type(backend);
        }

        bool finite_device_routes(const ggml_tensor* routes, std::size_t count)
        {
            // Wide PP is an explicit bounded path. Read the complete small
            // routing block before any slot mutation, while activations and
            // results keep their device-to-device bridge. No retained owner
            // or additional graph/weight allocation is introduced here.
            std::array<float, kHostMoeCacheMaxRouteElements> values{};
            if (count > values.size()) return false;
            const auto errors_before = g_ggml_error_count.load(std::memory_order_acquire);
            ggml_backend_tensor_get(routes, values.data(), 0, count * sizeof(float));
#if defined(TSG_GGML_TEST_HOOKS)
            // Model a backend read reporting failure through ggml's callback.
            // This is not an actual CUDA allocation or driver failure test.
            if (test_failure(5)) g_ggml_error_count.fetch_add(1, std::memory_order_acq_rel);
#endif
            if (g_ggml_error_count.load(std::memory_order_acquire) != errors_before
                || g_backend_compute_failed.load(std::memory_order_acquire))
                throw std::runtime_error("qwen4exp expert cache: routing download failed");
            return std::all_of(values.begin(), values.begin() + count,
                [](float value) { return std::isfinite(value); });
        }

        void copy_device_row(ggml_backend_t backend, const ggml_tensor* source,
            ggml_tensor* destination, std::size_t offset, bool destination_row = false)
        {
            // A local copy descriptor has the destination's scalar layout and
            // points at one contiguous source row. It never changes tensors in
            // the outer graph, and the public backend copy queues on the same
            // CUDA stream as expert uploads and graph execution.
            ggml_tensor row = destination_row ? *source : *destination;
            const auto* storage = destination_row ? destination : source;
            row.buffer = tensor_buffer(storage);
            row.data = static_cast<std::uint8_t*>(storage->data) + offset;
            row.view_src = nullptr;
            row.view_offs = 0;
            row.extra = nullptr;
            if (destination_row) ggml_backend_tensor_copy_async(backend, backend, source, &row);
            else ggml_backend_tensor_copy_async(backend, backend, &row, destination);
        }
#endif
    }

    int host_moe_cached_experts(const HostMoeSegment& hm, const float* x,
        const std::int32_t* ids, const float* routes, float* out, const char* kernel_name,
        ggml_backend_t explicit_backend)
    {
#ifdef TSG_GGML_USE_CUDA
        // The default disabled path does not acquire the cache mutex. Recheck
        // under the gate before any allocation so concurrent scope maintenance
        // cannot publish a smaller ceiling underneath an old snapshot.
        const bool device_inputs = x == nullptr && routes == nullptr;
        const bool device_output = out == nullptr;
        const bool explicit_model = explicit_backend != nullptr;
        auto decline = [&](const char* reason) {
            if (explicit_model) set_last_error(std::string("GLM selected-expert cache declined: ") + reason);
            return 0;
        };
        if (cache_budget() == 0) return decline("configured cache capacity is zero");
        ggml_backend_t backend = explicit_model ? explicit_backend : active_backend();
        if (kernel_name == nullptr
            || (std::strcmp(kernel_name, "qwen4exp fused token span") != 0
                && std::strcmp(kernel_name, "qwen4exp arena batched decode") != 0
                && (!explicit_model || (std::strcmp(kernel_name, "glm5-next selected expert seam") != 0
                    && std::strcmp(kernel_name, "glm5-next bulk expert prefill") != 0)))
            || backend == nullptr || !ggml_backend_is_cuda(backend)
            || (!device_inputs && (x == nullptr || routes == nullptr))
            || ids == nullptr || (device_output && !device_inputs)) return decline("backend, caller or bridge mode is unsupported");
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        const auto settings = cache_configuration().settings();
        const std::size_t budget = settings.budget, layers = settings.layers;
        if (budget == 0 || layers == 0) return decline("scope has no capacity or layout groups");
        const bool joint = hm.independent_decode_rows && hm.seq_len > 1
            && (std::strcmp(kernel_name, "qwen4exp arena batched decode") == 0 || explicit_model);
        if (joint && hm.seq_len > kHostMoeCacheMaxDecodeRows) return decline("independent decode width exceeds compact graph variants");
        const std::size_t pool_capacity = static_cast<std::size_t>(hm.pooled_model_experts > 0 ? hm.pooled_model_experts : hm.num_experts);
        std::size_t stride[3] = {};
        if (!valid_layout(hm, stride, joint ? std::max(settings.max_rows, settings.max_decode_rows)
                : settings.max_rows, explicit_model)) return decline("expert type, strides, dimensions, biases or row bound is unsupported");
        // Validate the complete short block before allocating or computing any
        // row. Unsupported data always falls back as one block; execution
        // failure returns an error and never mixes CPU/GPU expert arithmetic.
        const std::size_t route_count = static_cast<std::size_t>(hm.n_used) * hm.seq_len;
        if (device_inputs && !device_source(hm.moe_in,
                static_cast<std::size_t>(hm.hidden) * hm.seq_len * sizeof(float), backend)) return decline("activation bridge is not contiguous F32 on the model CUDA stream");
        if (device_inputs && !device_source(hm.weights, route_count * sizeof(float), backend))
            return decline("routing bridge is not contiguous F32 on the model CUDA stream");
        if (device_output && !device_source(hm.moe_out,
                static_cast<std::size_t>(hm.hidden) * hm.seq_len * sizeof(float), backend)) return decline("output bridge is not contiguous F32 on the model CUDA stream");
        for (std::size_t k = 0; k < route_count; ++k)
            if (ids[k] < 0 || ids[k] >= hm.num_experts
                || (!device_inputs && !std::isfinite(routes[k]))) return decline("selected IDs or route weights are invalid");
        const bool bulk_prefill = explicit_model && std::strcmp(kernel_name, "glm5-next bulk expert prefill") == 0;
        if (bulk_prefill && (joint || hm.seq_len <= TSG_PRECISION_DECODE_COLUMNS))
            return decline("bulk prefill requires a genuine wide nonjoint token batch");
        const bool wide_prefill = hm.seq_len > TSG_PRECISION_DECODE_COLUMNS;
        const bool grouped_prefill = !joint && !bulk_prefill && hm.seq_len > 1 && settings.max_decode_rows > 1;
        if ((wide_prefill || grouped_prefill) && device_inputs && !finite_device_routes(hm.weights, route_count)) return 0;

        if (joint && device_inputs && !wide_prefill && !finite_device_routes(hm.weights, route_count)) return 0;
        int minimum_capacity = explicit_model && hm.pooled_model_experts > 0
            ? std::min(hm.num_experts, hm.n_used * settings.max_decode_rows) : hm.n_used;
        if (joint || bulk_prefill) {
            // Protect the complete routed union for the whole submission. This
            // sorted copy only plans capacity; original route-slot order stays.
            std::vector<std::int32_t> unique(ids, ids + route_count);
            std::sort(unique.begin(), unique.end());
            minimum_capacity = std::max(minimum_capacity, static_cast<int>(
                std::unique(unique.begin(), unique.end()) - unique.begin()));
        }
        // The caller supplies actual supported concurrency, not the much wider
        // prefill limit. Every variant shares one persistent weight generation;
        // only the exact live-width graph executes. Its measured reservation is
        // included in the existing per-layer and shared physical owner budgets.
        int graph_rows = std::max(joint ? hm.seq_len : 1, settings.max_decode_rows);
        auto found = g_entries.begin();
        while (found != g_entries.end() && !(*found)->matches(hm, backend)) ++found;
        std::unique_ptr<CacheEntry> planned;
        int bulk_rows = bulk_prefill ? hm.seq_len : 0;
        if (found != g_entries.end()) {
            if (!bulk_prefill) bulk_rows = (*found)->bulk_rows; // N1/N3 reuse wide table and weights.
            graph_rows = std::max(graph_rows, (*found)->graph_rows);
            const bool required = (*found)->graph_rows < graph_rows || (*found)->capacity < minimum_capacity
                || (bulk_prefill && (*found)->bulk_rows != bulk_rows);
            const bool optional_growth = settings.capacity_growth
                && (*found)->growth_generation != cache_configuration().growth_generation()
                && static_cast<std::size_t>((*found)->capacity) < pool_capacity;
            if (settings.capacity_growth && (required || optional_growth)) {
                (*found)->growth_generation = cache_configuration().growth_generation();
                ++g_growth_attempts;
                // New payload may never shrink away an old slot. In contrast
                // to the legacy cold replacement below, refusal leaves this
                // published generation available for all still-fitting rows.
                const auto peak_quota = g_reserved < budget ? (std::min)(budget / layers, budget - g_reserved) : 0;
                auto expanded = create_entry(hm, stride, backend, peak_quota,
                    pool_capacity, graph_rows, std::max((*found)->capacity, minimum_capacity), bulk_rows);
                const auto growth = grow_entry(*found, std::move(expanded), budget);
                if (required && growth != Growth::replaced)
                    return decline("required hot-table expansion cannot admit simultaneous old and new payloads");
            }
            else if (required) {
                // Plan without physical allocation first. Retire the old table
                // before growing so no duplicate weight generation is resident.
                planned = create_entry(hm, stride, backend, budget / layers,
                    pool_capacity, graph_rows, minimum_capacity, bulk_rows);
                if (!planned) return decline("larger route union cannot fit its compact graph quota");
                const auto retired = (*found)->bytes;
                g_entries.erase(found); g_reserved -= retired;
                found = g_entries.end();
            }
        }
        if (found == g_entries.end())
        {
            std::unique_ptr<CacheEntry> entry;
            std::size_t maximum_capacity = pool_capacity;
            for (;;)
            {
                entry = planned ? std::move(planned) : create_entry(hm, stride, backend, budget / layers, maximum_capacity, graph_rows, minimum_capacity, bulk_rows);
                if (!entry) return decline("compact graph type or workspace cannot fit its pool quota");
                while (g_reserved > budget - entry->bytes && !g_entries.empty())
                {
                    const auto retired_bytes = g_entries.back()->bytes;
                    g_entries.pop_back();
                    g_reserved -= retired_bytes;
                }
                if (g_reserved > budget - entry->bytes) return decline("other retained entries exhaust the shared cache ceiling");
                const auto admission = allocate_entry(*entry);
                if (admission == Admission::admitted) break;
                if (admission == Admission::failed || entry->capacity <= minimum_capacity) return decline("compact payload physical or shared admission was refused");
                // Keep the operator's per-layer ceiling, but do not equate that
                // ceiling with available shared credit or physical free VRAM.
                // Smaller tables preserve the same scalar expert arithmetic and
                // retain hot weights across calls. No payload is allocated for
                // refused candidates, and the final candidate holds every routed
                // expert even when all IDs are distinct.
                maximum_capacity = std::max<std::size_t>(minimum_capacity,
                    static_cast<std::size_t>(entry->capacity) * 3 / 4);
                entry.reset();
            }
            const auto bytes = entry->bytes;
            entry->growth_generation = cache_configuration().growth_generation();
            // A list-node allocation can throw. Publish before changing private
            // accounting so unique_ptr rollback still owns every physical byte.
            g_entries.push_front(std::move(entry));
            g_reserved += bytes;
            const auto& published = *g_entries.front();
            if (diagnostics()) std::fprintf(stderr,
                "[HOSTMOE-EVICTION] layer=%d policy=%s epoch=%u\n", hm.layer,
                published.eviction_policy.epoch() ? "lfu" : "lru", published.eviction_policy.epoch());
            if (route_trace()) std::fprintf(stderr,
                "[HOSTMOE-ROUTE-CREATE] layer=%d slots=%d experts=%d\n",
                hm.layer, published.capacity, hm.num_experts);
            if (diagnostics()) std::fprintf(stderr,
                "[HOSTMOE-CACHE] layer=%d slots=%d reserved=%zu budget=%zu graph=%zu workspace_allowance=%zu input_bridge=%s\n",
                hm.layer, published.capacity, g_reserved, budget,
                published.bytes - kWorkspaceAllowance, kWorkspaceAllowance, device_inputs ? "device" : "host");
        }
        else
        {
            g_entries.splice(g_entries.begin(), g_entries, found);
        }
        auto& entry = *g_entries.front();
        // Only this gated call uses the current source pointers for reads.
        // Resident slots retain their own immutable source keys across calls.
        entry.source[0] = hm.gate_data; entry.source[1] = hm.up_data; entry.source[2] = hm.down_data;
        const int maximum_execution_rows = joint || bulk_prefill ? hm.seq_len
            : grouped_prefill ? std::min(hm.seq_len, settings.max_decode_rows) : 1;
        const int maximum_selected_count = maximum_execution_rows * hm.n_used;
        std::vector<int> remapped(maximum_selected_count, -1);
        std::vector<std::int32_t> route_major_ids(maximum_selected_count);
        std::vector<std::uint8_t> protected_slots(entry.capacity, 0);
        std::vector<std::pair<int, int>> misses;
        misses.reserve(maximum_selected_count);
        static const int prefetch = [] {
            const char* e = std::getenv("TS_HOST_MOE_EXPERT_CACHE_PREFETCH");
            // Unset: follow observed source-read costs for this cache entry.
            // Explicit 0/1 retain the diagnostic disable/force contracts.
            return e == nullptr ? -1 : e[0] == '1' ? 1 : 0;
        }();
        // The preplanned scalar-channel variants preserve each token's decode
        // arithmetic while combining short prefill FFNs in bounded microgroups.
        // Only the expert helper changes; the outer prefill graph is untouched.
        // Legacy scopes retain one row. Every microgroup protects its complete
        // routed union and uses the same single retained allocation.
        struct DrainPendingOutput {
            ggml_backend_t backend;
            bool active = false;
            ~DrainPendingOutput()
            {
                if (!active) return;
                sync_backend(backend);
#if defined(TSG_GGML_TEST_HOOKS)
                g_test_completed_output_drains.fetch_add(1, std::memory_order_relaxed);
#endif
            }
        } pending_output{entry.backend};
        for (int row = 0; row < hm.seq_len;)
        {
            const auto* row_ids = ids + static_cast<std::size_t>(row) * hm.n_used;
            const int execution_rows = joint || bulk_prefill ? hm.seq_len : grouped_prefill
                ? host_moe_cache_prefill_group_rows(row_ids, hm.n_used, hm.seq_len - row,
                    maximum_execution_rows, entry.capacity) : 1;
            if (execution_rows <= 0) throw std::runtime_error("qwen4exp expert cache: routed prefill row exceeds slot capacity");
            const int selected_count = execution_rows * hm.n_used;
            remapped.resize(selected_count); route_major_ids.resize(selected_count);
            for (int lane = 0; lane < execution_rows; ++lane)
                entry.eviction_policy.observe(row_ids + lane * hm.n_used, hm.n_used);
            if (route_trace()) {
                std::fprintf(stderr, "[HOSTMOE-ROUTE] layer=%d ids=", hm.layer);
                for (int k = 0; k < selected_count; ++k)
                    std::fprintf(stderr, "%s%d", k ? "," : "", row_ids[k]);
                std::fputc('\n', stderr);
            }
            std::fill(remapped.begin(), remapped.end(), -1);
            std::fill(protected_slots.begin(), protected_slots.end(), 0);
            misses.clear();
            // Protect ALL hits before filling any miss, including hits later in
            // the router's order. Otherwise an early miss can evict a later hit.
            for (int k = 0; k < selected_count; ++k)
                for (int s = 0; s < entry.capacity; ++s)
                    if (entry.slot_matches(s, row_ids[k]))
                    {
                        remapped[k] = s;
                        protected_slots[s] = 1;
                        ++g_hits;
                        break;
                    }
            for (int k = 0; k < selected_count; ++k)
            {
                if (remapped[k] >= 0) continue;
                // Duplicated selected IDs are legal for the helper, even though
                // Qwen's top-k normally produces distinct experts.
                for (int previous = 0; previous < k; ++previous)
                    if (row_ids[previous] == row_ids[k]) { remapped[k] = remapped[previous]; break; }
                if (remapped[k] >= 0) { ++g_hits; continue; }
                int slot = -1;
                for (int s = 0; s < entry.capacity; ++s)
                    if (!protected_slots[s] && (slot < 0 || entry.eviction_policy.prefer(
                        entry.expert_for_slot[s], entry.last_used[s],
                        entry.expert_for_slot[slot], entry.last_used[slot]))) slot = s;
                if (slot < 0)
                {
                    set_last_error("qwen4exp expert cache: all slots protected before routed experts were filled.");
                    return -1;
                }
                misses.emplace_back(slot, row_ids[k]);
                remapped[k] = slot;
                protected_slots[slot] = 1;
                ++g_misses;
            }
            // Fault all selected misses concurrently before pageable CUDA
            // uploads serialize source page faults on the launching thread.
            // Only already-selected raw bytes are touched; no speculative experts,
            // pinned weight copies or expanded device quota. Join all three
            // projections together. Upload each ready projection on this caller
            // thread while the bounded I/O workers read the remaining misses.
            std::size_t miss_bytes = 0;
            for (const auto stride_bytes : entry.stride) miss_bytes += stride_bytes * misses.size();
            const bool observe = prefetch < 0 && ExpertPrefetchPolicy::eligible(misses.size(), miss_bytes);
            const bool do_prefetch = !misses.empty() && (prefetch == 1
                || (prefetch < 0 && entry.prefetch_policy.should_prefetch(misses.size(), miss_bytes)));
            if (observe) ++entry.prefetch_observations;
            // No slot may retain a valid old expert ID after its first source
            // projection is overwritten, including interrupted uploads.
            for (const auto& miss : misses) entry.invalidate_slot(miss.first);
            std::array<std::shared_ptr<ExpertFileSource>, 3> files;
            const bool group = execution_rows > 1;
            // GLM top-8 scalar misses exceed 32 MiB in real quantized layouts.
            // Keep their registered exact file sources on the same bounded ring
            // instead of falling through to pageable mmap faults on the caller.
            const bool bounded_file_group = joint || grouped_prefill || bulk_prefill
                || (explicit_model && hm.pooled_model_experts > 0);
            bool file_read = !misses.empty() && (bounded_file_group || miss_bytes <= kFileReadWorkspace);
            for (int part = 0; file_read && part < 3; ++part) {
                const auto source = g_file_sources.find(entry.source[part]);
                if (source == g_file_sources.end()
                    || source->second->bytes() != entry.stride[part] * static_cast<std::size_t>(hm.num_experts))
                    file_read = false;
                else files[part] = source->second;
            }
            struct DrainFileUploads {
                ggml_backend_t backend;
                bool active;
                ~DrainFileUploads() { if (active) sync_backend(backend); }
            } drain{entry.backend, true};
            if (file_read)
            {
                ++entry.file_read_calls;
                ++entry.prefetch_reads;
                if (bounded_file_group) {
                    copy_joint_file_misses(entry, misses, files, miss_bytes);
                } else {
                g_file_read_buffer.reserve(miss_bytes);
                std::vector<ExpertReadTask> tasks;
                tasks.reserve(misses.size() * 3);
                std::size_t offset = 0;
                for (const auto& miss : misses) for (int part = 0; part < 3; ++part) {
                    tasks.push_back({files[part].get(), static_cast<std::size_t>(miss.second) * entry.stride[part],
                        g_file_read_buffer.data() + offset, entry.stride[part]});
                    offset += entry.stride[part];
                }
                std::vector<int> copied_parts(misses.size(), 0);
                read_expert_files(tasks, [&](std::size_t index) {
                    const auto i = index / 3, part = index % 3;
                    const auto& task = tasks[index];
                    entry.file_read_bytes += task.bytes;
                    ggml_backend_tensor_set_async(entry.backend, entry.weight[part], task.destination,
                        static_cast<std::size_t>(misses[i].first) * entry.stride[part], task.bytes);
                    if (++copied_parts[i] == 3) entry.publish_slot(misses[i].first, misses[i].second);
                });
                }
            }
            else if (do_prefetch)
            {
                ++entry.prefetch_reads;
                std::vector<MoePrefetchExpert> experts(misses.size());
                for (std::size_t i = 0; i < misses.size(); ++i)
                {
                    for (int part = 0; part < 3; ++part)
                        experts[i][part] = {static_cast<const std::uint8_t*>(entry.source[part])
                                + static_cast<std::size_t>(misses[i].second) * entry.stride[part], entry.stride[part]};
                }
                std::vector<std::pair<std::size_t, std::size_t>> pieces;
                std::vector<int> copied_parts(misses.size(), 0);
                const auto read_time = prefetch_mapped_experts(experts, [&](std::size_t i, std::size_t part) {
                    const auto& miss = misses[i];
                    copy_slot_part(entry, miss.first, miss.second, static_cast<int>(part), pieces);
                    if (++copied_parts[i] == 3) entry.publish_slot(miss.first, miss.second);
                });
                if (observe) entry.prefetch_policy.observe_read(miss_bytes, read_time);
            }
            const auto upload_started = observe && !do_prefetch
                ? ExpertPrefetchPolicy::Clock::now() : ExpertPrefetchPolicy::Clock::time_point{};
            if (!file_read && !do_prefetch) for (const auto& miss : misses)
            {
                copy_slot(entry, miss.first, miss.second);
                entry.publish_slot(miss.first, miss.second);
            }
            if (observe && !do_prefetch && !file_read)
                entry.prefetch_policy.observe_upload(miss_bytes, ExpertPrefetchPolicy::Clock::now() - upload_started);
            for (const int slot : remapped) entry.last_used[slot] = ++entry.clock;
            // Exact active width, including when a prior larger batch grew this
            // cache. Input/routes/output each use one contiguous bridge; only
            // the small already-host-resident slot IDs need route transposition.
            const int variant = bulk_prefill ? entry.graph_rows : execution_rows - 1;
            for (int k = 0; k < entry.n_used; ++k)
                for (int lane = 0; lane < execution_rows; ++lane)
                    route_major_ids[bulk_prefill ? lane * entry.n_used + k : k * execution_rows + lane]
                        = remapped[lane * entry.n_used + k];
            if (device_inputs) {
                copy_device_row(entry.backend, hm.moe_in, entry.width_inputs[variant],
                    static_cast<std::size_t>(row) * entry.hidden * sizeof(float));
                copy_device_row(entry.backend, hm.weights, entry.width_routes[variant],
                    static_cast<std::size_t>(row) * entry.n_used * sizeof(float));
            } else {
                ggml_backend_tensor_set_async(entry.backend, entry.width_inputs[variant],
                    x + static_cast<std::size_t>(row) * hm.hidden, 0,
                    static_cast<std::size_t>(entry.hidden) * execution_rows * sizeof(float));
                ggml_backend_tensor_set_async(entry.backend, entry.width_routes[variant],
                    routes + static_cast<std::size_t>(row) * hm.n_used, 0,
                    static_cast<std::size_t>(entry.n_used) * execution_rows * sizeof(float));
            }
            ggml_backend_tensor_set_async(entry.backend, entry.width_ids[variant],
                route_major_ids.data(), 0, static_cast<std::size_t>(selected_count) * sizeof(std::int32_t));
            auto* execute = &entry.width_graphs[variant];
#if defined(TSG_GGML_TEST_HOOKS)
            if (joint && test_failure(6)) throw std::runtime_error("Injected joint cache failure after queued inputs");
#endif
            ++g_graph_submissions;
            if (joint) { ++g_joint_submissions; g_joint_rows += execution_rows; }
            if (compute_graph(entry.backend, execute) != GGML_STATUS_SUCCESS
                || g_backend_compute_failed.load(std::memory_order_acquire)) {
                set_last_error("qwen4exp expert cache: CUDA expert graph failed.");
                return -1;
            }
            drain.active = false; // graph_compute drained all preceding uploads.
#if defined(TSG_GGML_TEST_HOOKS)
            if (joint && test_failure(7)) throw std::runtime_error("Injected joint cache failure before publishing output");
#endif
            if (device_output) {
                copy_device_row(entry.backend, entry.width_outputs[variant], hm.moe_out,
                    static_cast<std::size_t>(row) * entry.hidden * sizeof(float), true);
                pending_output.active = wide_prefill || joint || group;
            } else
                ggml_backend_tensor_get(entry.width_outputs[variant], out + static_cast<std::size_t>(row) * hm.hidden,
                    0, static_cast<std::size_t>(entry.hidden) * execution_rows * sizeof(float));
            g_calls += execution_rows;
            row += execution_rows;

#if defined(TSG_GGML_TEST_HOOKS)
            if (wide_prefill && test_failure(4))
                throw std::runtime_error("Injected wide expert-cache failure after a completed row");
#endif
        }
        pending_output.active = false;
        return 1;
#else
        (void)hm; (void)x; (void)ids; (void)routes; (void)out; (void)kernel_name; (void)explicit_backend;
        return 0;
#endif
    }

    void host_moe_expert_cache_on_drop(const void* ptr)
    {
        if (ptr == nullptr) return;
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        g_file_sources.erase(ptr);
        for (auto it = g_entries.begin(); it != g_entries.end();)
        {
            const auto& entry = **it;
            if (entry.references(ptr))
            {
                const auto retired_bytes = entry.bytes;
                it = g_entries.erase(it);
                g_reserved -= retired_bytes;
            }
            else ++it;
        }
    }

    // A GLM stream belongs to the native model, unlike Qwen's borrowed global
    // stream. Drain and physically free every matching entry while it is alive.
    void host_moe_expert_cache_release_backend(ggml_backend_t backend)
    {
        if (!backend) return;
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        for (auto it = g_entries.begin(); it != g_entries.end();) {
            if ((*it)->backend != backend) { ++it; continue; }
            const auto bytes = (*it)->bytes;
            it = g_entries.erase(it); g_reserved -= bytes;
        }
    }

    void host_moe_expert_cache_release()
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        host_moe_cpu_staging_release();
        if (diagnostics() && (g_calls != 0 || g_reserved != 0)) std::fprintf(stderr,
            "[HOSTMOE-CACHE] calls=%llu hits=%llu misses=%llu reserved=%zu budget=%zu release\n",
            static_cast<unsigned long long>(g_calls), static_cast<unsigned long long>(g_hits),
            static_cast<unsigned long long>(g_misses), g_reserved, cache_budget());
        if (diagnostics() && g_file_read_buffer.size())
            std::fprintf(stderr, "[HOSTMOE-FILE-WORKSPACE] bytes=%zu ceiling=%zu type=%s\n",
                g_file_read_buffer.size(), kFileReadWorkspace, ggml_backend_buffer_name(g_file_read_buffer.buffer));
        g_entries.clear();
        g_file_sources.clear();
        g_file_read_buffer.clear();
        g_reserved = 0;
        g_hits = g_misses = g_calls = 0;
        g_graph_submissions = g_joint_submissions = g_joint_rows = 0;
        g_growth_attempts = g_growth_successes = g_growth_refusals = g_growth_failures = 0;
        g_growth_noops = g_growth_hot_slots = g_growth_copy_bytes = 0;
        g_growth_peak_accounted = 0;
    }

    // Caller holds g_cache_mutex. Scope updates also enforce the per-layer
    // share, so a formerly large entry cannot occupy another layer's new quota.
    std::size_t host_moe_expert_cache_trim_locked(std::size_t target,
        std::size_t per_layer = (std::numeric_limits<std::size_t>::max)())
    {
        // Request-boundary maintenance also releases the one serial CPU layer's
        // reusable host arena, without dropping registered file sources.
        host_moe_cpu_staging_release();
        const auto before = g_reserved;
        for (auto it = g_entries.begin(); per_layer != (std::numeric_limits<std::size_t>::max)()
            && it != g_entries.end();) {
            if ((*it)->bytes <= per_layer) { ++it; continue; }
            const auto bytes = (*it)->bytes;
            it = g_entries.erase(it);
            g_reserved -= bytes;
        }
        while (g_reserved > target && !g_entries.empty())
        {
            const auto bytes = g_entries.back()->bytes;
            // Destruction drains queued output copies and frees physical storage
            // before refunding shared credit. Never revoke an in-flight owner.
            g_entries.pop_back();
            g_reserved -= bytes;
        }
        if (g_entries.empty()) g_file_read_buffer.clear();
        return before - g_reserved;
    }

    std::size_t host_moe_expert_cache_trim(std::size_t target)
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        return host_moe_expert_cache_trim_locked(target);
    }

    std::uint64_t host_moe_expert_cache_scope_begin(HostMoeCacheSettings settings) {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        return cache_configuration().begin(settings,
            g_reserved == 0 && g_entries.empty() && g_file_read_buffer.size() == 0);
    }
    bool host_moe_expert_cache_scope_update(std::uint64_t token, std::size_t target, std::size_t& released) {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        return cache_configuration().update(token, target, [&](std::size_t budget, std::size_t layers) {
            released = host_moe_expert_cache_trim_locked(budget, budget / layers);
        });
    }
    bool host_moe_expert_cache_scope_trim(std::uint64_t token, std::size_t target, std::size_t& released) {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        if (!cache_configuration().owns(token)) return false;
        released = host_moe_expert_cache_trim_locked(target);
        return true;
    }
    bool host_moe_expert_cache_scope_end(std::uint64_t token, std::size_t& released) {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        return cache_configuration().end(token, [&] { released = host_moe_expert_cache_trim_locked(0); });
    }

    void host_moe_expert_cache_stats(std::int64_t* reserved, std::int64_t* budget,
        std::int64_t* hits, std::int64_t* misses, std::int64_t* calls)
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        *reserved = static_cast<std::int64_t>(g_reserved);
        *budget = static_cast<std::int64_t>(cache_budget());
        *hits = static_cast<std::int64_t>(g_hits);
        *misses = static_cast<std::int64_t>(g_misses);
        *calls = static_cast<std::int64_t>(g_calls);
    }

    void host_moe_expert_cache_growth_stats(std::int64_t* values)
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        values[0] = static_cast<std::int64_t>(g_growth_attempts);
        values[1] = static_cast<std::int64_t>(g_growth_successes);
        values[2] = static_cast<std::int64_t>(g_growth_refusals);
        values[3] = static_cast<std::int64_t>(g_growth_failures);
        values[4] = static_cast<std::int64_t>(g_growth_noops);
        values[5] = static_cast<std::int64_t>(g_growth_hot_slots);
        values[6] = static_cast<std::int64_t>(g_growth_copy_bytes);
        values[7] = static_cast<std::int64_t>(g_growth_peak_accounted);
    }

    int host_moe_expert_cache_entry_stats(std::int64_t* values, int capacity, int& count)
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        if (g_entries.size() > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
            throw std::overflow_error("Expert-cache entry metadata is too large");
        count = static_cast<int>(g_entries.size());
        if (!values && capacity == 0) return 1;
        if (capacity < count) return 2;
        if (!values) throw std::invalid_argument("Expert-cache entries require an output buffer");
        for (const auto& entry : g_entries) {
            int resident = 0;
            for (const int expert : entry->expert_for_slot) if (expert >= 0) ++resident;
            values[0] = entry->layer; values[1] = entry->capacity;
            values[2] = resident; values[3] = entry->graph_rows;
            values[4] = static_cast<std::int64_t>(entry->bytes);
            values[5] = entry->num_experts; values[6] = entry->pool_expert_extent;
            values[7] = static_cast<std::int64_t>(entry->stride[0] + entry->stride[1] + entry->stride[2]);
            values += 8;
        }
        return 1;
    }

    // Size-only planning of the smallest table that protects every possible
    // expert in the admitted joint union. No table is published, no live entry
    // is changed and reserve_n_size creates no CUDA backing allocation.
    bool host_moe_expert_cache_minimum_plan(const HostMoeSegment& hm, int rows,
        ggml_backend_t backend, std::size_t& bytes)
    {
        bytes = 0;
#ifdef TSG_GGML_USE_CUDA
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        if (!backend || !ggml_backend_is_cuda(backend) || !cache_configuration().active()
            || rows < 1 || rows > 3 || hm.pooled_model_experts <= 0) return false;
        std::size_t stride[3];
        if (!valid_layout(hm, stride, rows, true)) return false;
        const int capacity = std::min(hm.num_experts, hm.n_used * rows);
        auto planned = create_entry(hm, stride, backend,
            (std::numeric_limits<std::size_t>::max)(), capacity, rows, capacity);
        if (!planned || planned->capacity != capacity || planned->graph_rows != rows
            || ggml_gallocr_get_buffer_size(planned->allocator, 0) != 0 || planned->shared_charge) return false;
        bytes = planned->bytes;
        return bytes > 0;
#else
        (void)hm; (void)rows; (void)backend;
        return false;
#endif
    }
    bool host_moe_expert_cache_minimum_prefill_plan(const HostMoeSegment& hm, int rows,
        int prefill_rows, ggml_backend_t backend, std::size_t& bytes)
    {
        bytes = 0;
#ifdef TSG_GGML_USE_CUDA
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        if (!backend || !ggml_backend_is_cuda(backend) || !cache_configuration().active()
            || rows < 1 || rows > 3 || prefill_rows < 1 || hm.pooled_model_experts <= 0
            || !host_moe_cache_routes_fit(prefill_rows, hm.n_used)) return false;
        std::size_t stride[3];
        if (!valid_layout(hm, stride, rows, true)) return false;
        const int bulk_rows = prefill_rows > TSG_PRECISION_DECODE_COLUMNS ? prefill_rows : 0;
        const int capacity = std::min(hm.num_experts, hm.n_used * (bulk_rows > 0 ? std::max(rows, prefill_rows) : rows));
        auto planned = create_entry(hm, stride, backend,
            (std::numeric_limits<std::size_t>::max)(), capacity, rows, capacity, bulk_rows);
        if (!planned || planned->capacity != capacity || planned->graph_rows != rows
            || planned->bulk_rows != bulk_rows || ggml_gallocr_get_buffer_size(planned->allocator, 0) != 0
            || planned->shared_charge) return false;
        bytes = planned->bytes;
        return bytes > 0;
#else
        (void)hm; (void)rows; (void)prefill_rows; (void)backend;
        return false;
#endif
    }
    bool host_moe_expert_cache_scope_matches_plan(std::size_t minimum_bytes,
        int groups, int prefill_rows, int decode_rows)
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        const auto& settings = cache_configuration().settings();
        return cache_configuration().active() && minimum_bytes > 0 && groups > 0
            && settings.layers == static_cast<std::size_t>(groups)
            && settings.max_rows >= prefill_rows && settings.max_decode_rows == decode_rows
            && settings.budget >= minimum_bytes;
    }

#if defined(TSG_GGML_TEST_HOOKS)
    // Linked only into the standalone cache fixture, never a production export.
    void host_moe_expert_cache_test_fail_next(int stage) { g_test_failure_stage.store(stage); }
    std::size_t host_moe_expert_cache_test_physical_bytes() { return g_test_physical_bytes.load(); }
    void host_moe_expert_cache_test_file_slice_limit(std::size_t bytes) { g_test_file_slice_bytes.store(bytes); }
    std::size_t host_moe_expert_cache_test_file_workspace_bytes() { return g_file_read_buffer.size(); }
    void host_moe_expert_cache_test_available_bytes(std::size_t bytes) { g_test_available_bytes.store(bytes); }
    std::uint64_t host_moe_expert_cache_test_output_drains() { return g_test_completed_output_drains.load(); }
#ifdef TSG_GGML_USE_CUDA
    // A size-only capacity oracle for the standalone fixture; no production ABI.
    std::size_t host_moe_expert_cache_test_plan_capacity(const HostMoeSegment& hm,
        int capacity, int rows, ggml_backend_t backend) {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        std::size_t stride[3];
        if (capacity < hm.n_used || capacity > hm.pooled_model_experts || rows < 1
            || rows > kHostMoeCacheMaxDecodeRows || !backend || !ggml_backend_is_cuda(backend)
            || !valid_layout(hm, stride, rows, true)) return 0;
        auto candidate = create_entry(hm, stride, backend, (std::numeric_limits<std::size_t>::max)(),
            capacity, rows, capacity);
        if (!candidate || candidate->capacity != capacity || candidate->shared_charge
            || ggml_gallocr_get_buffer_size(candidate->allocator, 0) != 0) return 0;
        return candidate->bytes;
    }
#endif
#if defined(_WIN32)
#define TSG_MOE_TEST_EXPORT extern "C" __declspec(dllexport)
#else
#define TSG_MOE_TEST_EXPORT extern "C" __attribute__((visibility("default")))
#endif
    TSG_MOE_TEST_EXPORT int TSGgml_TestExpertHostWorkspaceReserve(std::int64_t bytes) {
        if (bytes <= 0) return 0;
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        try { g_file_read_buffer.reserve(static_cast<std::size_t>(bytes)); return 1; }
        catch (...) { return 0; }
    }
    TSG_MOE_TEST_EXPORT void TSGgml_TestExpertHostWorkspaceClear() {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        g_file_read_buffer.clear();
    }
#undef TSG_MOE_TEST_EXPORT
#endif

    int host_moe_file_batch_sources(const void* const* pointers, const std::int64_t* bytes,
        std::array<std::shared_ptr<ExpertFileSource>, 3>& result)
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        if (!cache_configuration().active() || !SharedCacheCharge::host_accounting_attached()) return 0;
        int found = 0;
        for (int part = 0; part < 3; ++part) {
            auto item = g_file_sources.find(pointers[part]);
            if (item == g_file_sources.end()) continue;
            if (bytes[part] <= 0 || item->second->bytes() != static_cast<std::uint64_t>(bytes[part])) return -1;
            result[part] = item->second; ++found;
        }
        return found == 3 ? 1 : found == 0 ? 0 : -1;
    }

    bool register_host_file_source(const void* pointer, std::int64_t bytes,
        const char* path, std::int64_t offset)
    {
        if (!pointer || bytes <= 0 || offset < 0) return false;
        auto source = std::make_shared<ExpertFileSource>(path, offset, bytes);
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        if (g_file_sources.count(pointer)) return false;
        g_file_sources.emplace(pointer, std::move(source));
        return true;
    }
}

TSG_EXPORT int TSGgml_RegisterHostFileSource(const void* pointer, std::int64_t bytes,
    const char* path, std::int64_t offset)
{
    try { return tsg::register_host_file_source(pointer, bytes, path, offset) ? 1 : 0; }
    catch (const std::exception& error) {
        tsg::set_last_error(std::string("Host file source registration failed: ") + error.what());
        return 0;
    }
}

TSG_EXPORT int TSGgml_HostMoeExpertCacheBudgetKind()
{
    // Lets managed session envelopes fail closed with an older native library.
    return 4;
}

TSG_EXPORT int TSGgml_HostMoeExpertCacheStats(std::int64_t* reserved, std::int64_t* budget,
    std::int64_t* hits, std::int64_t* misses, std::int64_t* calls)
{
    if (!reserved || !budget || !hits || !misses || !calls) return 0;
    tsg::host_moe_expert_cache_stats(reserved, budget, hits, misses, calls);
    return 1;
}

// Counts are submission attempts, not successful publications. A joint graph
// uses three route-major N-request MUL_MAT_ID operators with scalar dot order;
// weighted reduction is also N-request. Ordinary PP retains its scalar loop.
TSG_EXPORT int TSGgml_HostMoeExpertCacheExecutionStats(std::uint64_t* submissions,
    std::uint64_t* joint_submissions, std::uint64_t* joint_rows)
{
    if (!submissions || !joint_submissions || !joint_rows) return 0;
    std::lock_guard<std::mutex> lock(tsg::g_cache_mutex);
    *submissions = tsg::g_graph_submissions;
    *joint_submissions = tsg::g_joint_submissions;
    *joint_rows = tsg::g_joint_rows;
    return 1;
}

// Process-wide maintenance at a quiescent request boundary. This is an eviction
// target, not a new ceiling: subsequent requests can fill the configured cache.
// Counters remain cumulative, allowing the caller to observe the reload cost.
TSG_EXPORT std::int64_t TSGgml_TrimHostMoeExpertCache(std::int64_t target_bytes)
{
    if (target_bytes < 0 || static_cast<std::uint64_t>(target_bytes)
        > std::numeric_limits<std::size_t>::max()) return -1;
    try
    {
        return static_cast<std::int64_t>(tsg::host_moe_expert_cache_trim(
            static_cast<std::size_t>(target_bytes)));
    }
    catch (const std::exception& error)
    {
        tsg::set_last_error(std::string("Expert-cache trim failed: ") + error.what());
        return -1;
    }
}

namespace {
int cache_scope_failure(const char* message) noexcept {
    // Even reporting an allocation failure must not throw across the C ABI.
    try { tsg::set_last_error(message); } catch (...) {}
    return 0;
}
bool cache_scope_bytes(std::int64_t bytes) noexcept {
    return bytes >= 0 && static_cast<std::uint64_t>(bytes) <= (std::numeric_limits<std::size_t>::max)();
}
}

// Exclusive process-wide settings, independent of environment mutations.
// Quiesce model work before all maintenance; existing ticket ownership and
// physical-before-credit release are preserved by the original cache objects.
TSG_EXPORT int TSGgml_BeginHostMoeExpertCacheScopeWithCapacityGrowth(std::int64_t bytes,
    int max_rows, int layers, int max_decode_rows, int enable_capacity_growth, std::uint64_t* token)
{
    if (!token) return cache_scope_failure("Expert-cache scope requires an output token");
    *token = 0;
    try {
        tsg::clear_last_error();
        if (!cache_scope_bytes(bytes) || layers <= 0 || max_rows <= 0 || max_rows > tsg::kHostMoeCacheMaxPrefillRows
            || max_decode_rows <= 0 || max_decode_rows > tsg::kHostMoeCacheMaxDecodeRows
            || (enable_capacity_growth != 0 && enable_capacity_growth != 1))
            return cache_scope_failure("Invalid expert-cache scope capacity, prefill/decode row limit or layer count");
        *token = tsg::host_moe_expert_cache_scope_begin({static_cast<std::size_t>(bytes),
            static_cast<std::size_t>(layers), max_rows, max_decode_rows, enable_capacity_growth != 0});
        return *token ? 1 : cache_scope_failure("Expert-cache scope requires an empty cache and no active scope");
    }
    catch (const std::exception& error) { return cache_scope_failure(error.what()); }
    catch (...) { return cache_scope_failure("Expert-cache scope creation failed with an unknown exception"); }
}

TSG_EXPORT int TSGgml_BeginHostMoeExpertCacheScopeWithDecodeRows(std::int64_t bytes,
    int max_rows, int layers, int max_decode_rows, std::uint64_t* token)
{
    return TSGgml_BeginHostMoeExpertCacheScopeWithCapacityGrowth(bytes, max_rows, layers, max_decode_rows, 0, token);
}

// Preserve the original ABI and its scalar planning behavior for ordinary
// callers. Unified sessions opt into their known decode concurrency explicitly.
TSG_EXPORT int TSGgml_BeginHostMoeExpertCacheScope(std::int64_t bytes, int max_rows, int layers,
    std::uint64_t* token)
{
    return TSGgml_BeginHostMoeExpertCacheScopeWithDecodeRows(bytes, max_rows, layers, 1, token);
}

TSG_EXPORT int TSGgml_UpdateHostMoeExpertCacheScope(std::uint64_t token, std::int64_t bytes,
    std::int64_t* released)
{
    if (!released) return cache_scope_failure("Expert-cache update requires an output count");
    *released = 0;
    try {
        tsg::clear_last_error();
        if (!cache_scope_bytes(bytes)) return cache_scope_failure("Invalid expert-cache update capacity");
        std::size_t retired = 0;
        if (!tsg::host_moe_expert_cache_scope_update(token, static_cast<std::size_t>(bytes), retired))
            return cache_scope_failure("Expert-cache update requires its active token and cannot exceed the initial ceiling");
        *released = static_cast<std::int64_t>(retired);
        return 1;
    }
    catch (const std::exception& error) { return cache_scope_failure(error.what()); }
    catch (...) { return cache_scope_failure("Expert-cache update failed with an unknown exception"); }
}

TSG_EXPORT int TSGgml_TrimHostMoeExpertCacheScope(std::uint64_t token, std::int64_t bytes,
    std::int64_t* released)
{
    if (!released) return cache_scope_failure("Expert-cache trim requires an output count");
    *released = 0;
    try {
        tsg::clear_last_error();
        if (!cache_scope_bytes(bytes)) return cache_scope_failure("Invalid expert-cache trim target");
        std::size_t retired = 0;
        if (!tsg::host_moe_expert_cache_scope_trim(token, static_cast<std::size_t>(bytes), retired))
            return cache_scope_failure("Expert-cache trim requires its active scope token");
        *released = static_cast<std::int64_t>(retired);
        return 1;
    }
    catch (const std::exception& error) { return cache_scope_failure(error.what()); }
    catch (...) { return cache_scope_failure("Expert-cache trim failed with an unknown exception"); }
}

TSG_EXPORT int TSGgml_EndHostMoeExpertCacheScope(std::uint64_t token, std::int64_t* released)
{
    if (!released) return cache_scope_failure("Expert-cache disposal requires an output count");
    *released = 0;
    try {
        tsg::clear_last_error();
        std::size_t retired = 0;
        if (!tsg::host_moe_expert_cache_scope_end(token, retired))
            return cache_scope_failure("Expert-cache disposal requires its active scope token");
        *released = static_cast<std::int64_t>(retired);
        return 1;
    }
    catch (const std::exception& error) { return cache_scope_failure(error.what()); }
    catch (...) { return cache_scope_failure("Expert-cache disposal failed with an unknown exception"); }
}

// Read-only diagnostics. The peak is accounted cache payload plus its fixed
// allowance, never a claim about vendor/driver bytes or whole-board VRAM.
TSG_EXPORT int TSGgml_HostMoeExpertCacheGrowthStats(std::int64_t* values, int length)
{
    if (!values || length != 8) return cache_scope_failure("Expert-cache growth statistics require eight values");
    try { tsg::host_moe_expert_cache_growth_stats(values); return 1; }
    catch (const std::exception& error) { return cache_scope_failure(error.what()); }
    catch (...) { return cache_scope_failure("Expert-cache growth statistics failed"); }
}

TSG_EXPORT int TSGgml_HostMoeExpertCacheEntryStats(std::int64_t* values, int row_capacity, int* count)
{
    if (!count) return cache_scope_failure("Expert-cache entry statistics require a row count");
    *count = 0;
    if (row_capacity < 0 || (!values && row_capacity != 0))
        return cache_scope_failure("Invalid expert-cache entry statistics buffer");
    try { return tsg::host_moe_expert_cache_entry_stats(values, row_capacity, *count); }
    catch (const std::exception& error) { return cache_scope_failure(error.what()); }
    catch (...) { return cache_scope_failure("Expert-cache entry statistics failed"); }
}
