// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Qwen4Exp's optional CUDA expert tier. GGUF bytes remain quantized and are
// copied into compact, persistent slots; ordinary upstream ggml CUDA kernels
// compute the complete routed FFN. No CPU/GPU partial-sum split is involved.
#include "ggml_ops_internal.h"
#include "ggml_ops_moe_prefetch.h"
#include "ggml_ops_file_source.h"
#include "ggml_ops_file_source_view_prefetch.h"
#include "ggml_ops_host_moe_file_batch.h"
#include "ggml_ops_host_moe_layer_prefetch.h"
#include "ggml_ops_moe_prefetch_policy.h"
#include "ggml_ops_expert_eviction.h"
#include "ggml_ops_precision_policy.h"
#include "ggml_ops_host_moe_cache_policy.h"
#include "ggml_ops_host_moe_cache_config.h"
#include "ggml_ops_host_moe_cache_identity.h"
#include "ggml_ops_host_moe_victim_policy.h"
#include "ggml_ops_shared_cache_budget.h"
#include "ggml-impl.h"
#include "ggml_ops_host_moe_split_plan_draft.h"
#include "ggml_ops_host_moe_cache_timing.h"
#include "ggml_ops_file_read_workers.h"
#include "ggml_ops_host_moe_file_read_plan.h"
#include "ggml_ops_host_moe_file_ring.h"
#include "ggml_ops_host_moe_file_ring_coalesce.h"
#include "ggml_ops_host_moe_miss_read_order.h"
#include "ggml_ops_host_moe_down_overlap.h"
#include "ggml_ops_host_moe_immutable_cache.h"
#include "ggml_ops_host_moe_ram_probation.h"
#include "ggml_ops_host_moe_mapped_ram.h"
#include "ggml_ops_host_moe_decode_staging.h"
#include "ggml_ops_host_moe_vmm_weights.h"
#include "ggml_ops_host_moe_vmm_cache_plan.h"
#include "glm_causal_route_neighbor_owner.h"
#include "glm_causal_route_neighbor_cache.h"
#if defined(TSG_GGML_TEST_HOOKS)
#include "host_moe_cache_lifecycle_test_hooks.h"
#endif
#ifdef TSG_GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include "dsv41_engram_io.h"
#include <cerrno>
#include <condition_variable>
#include <cstdlib>
#include <list>
#include <mutex>
#include <thread>
#include <optional>
#include <exception>

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
        int file_source_view_option() noexcept {
            static const int option = [] {
                const char* value = std::getenv("TS_HOST_MOE_FILE_SOURCE_VIEW");
                if (!value || std::strcmp(value, "0") == 0) return 0;
                return std::strcmp(value, "1") == 0 ? 1 : -1;
            }();
            const int prefetch = ExpertFileSourceViewPrefetch::option();
            if (prefetch < 0 || (prefetch == 1 && (option != 1 || !ExpertFileSourceViewPrefetch::available())))
                return -1; // Existing loader Enabled gate refuses before model allocations.
            return option;
        }

        // Compact GLM prefill uses invariant <=8-column scalar-channel graphs.
        // This is helper geometry, never the admitted independent decode width.
        int compact_prefill_group_rows(int prefill_rows)
        {
            return std::min(prefill_rows, static_cast<int>(TSG_PRECISION_DECODE_COLUMNS));
        }

        int glm_compact_graph_rows(int decode_rows, int prefill_rows)
        {
            return std::max(decode_rows, compact_prefill_group_rows(prefill_rows));
        }

        std::mutex g_cache_mutex;
        std::size_t g_reserved = 0;
        std::uint64_t g_hits = 0, g_misses = 0, g_calls = 0;
        std::uint64_t g_graph_submissions = 0, g_joint_submissions = 0, g_joint_rows = 0;
        std::uint64_t g_growth_attempts = 0, g_growth_successes = 0, g_growth_refusals = 0;
        std::uint64_t g_growth_failures = 0, g_growth_noops = 0, g_growth_hot_slots = 0, g_growth_copy_bytes = 0;
        std::size_t g_growth_peak_accounted = 0;
        bool g_last_phase_wide = false;
        std::optional<HostMoeGroupKey> g_phase_cold_growth_group;
#if defined(TSG_GGML_TEST_HOOKS)
        std::atomic<int> g_test_failure_stage{0};
        std::atomic<std::size_t> g_test_physical_bytes{0};
        std::atomic<std::uint64_t> g_test_completed_output_drains{0};
        std::atomic<std::uint64_t> g_test_payload_upload_drains{0};
        std::atomic<std::size_t> g_test_available_bytes{std::numeric_limits<std::size_t>::max()};
        std::atomic<std::size_t> g_test_file_slice_bytes{std::size_t(32) << 20};
        std::atomic<int> g_test_file_workspace_failure{0};
        std::atomic<std::uint64_t> g_test_phase_cold_growth_attempts{0};
        std::atomic<std::uint64_t> g_test_file_windows{0};
        std::array<std::size_t, 3> g_test_last_completed_ring_plan{}; // under g_cache_mutex
        std::array<std::size_t, 5> g_test_last_coalesced_ring_plan{}; // before/after/payload/merged/cap
        std::atomic<std::uint64_t> g_test_coalesced_ring_calls{0};
        std::atomic<std::uint64_t> g_test_down_completed{0}, g_test_down_prefix{0}, g_test_down_suffix{0};
        std::atomic<std::uint64_t> g_test_down_drains{0}, g_test_down_partial_reads{0};
        bool g_test_vmm_profile = false; // Explicit quiescent fixture only, never an environment override.
        std::atomic<std::uint64_t> g_test_vmm_mutation_failures{0};
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

        bool file_read_sort_enabled() {
            static const bool enabled = [] {
                const bool requested = host_moe_file_sort_flag(std::getenv("TS_HOST_MOE_FILE_READ_SORT"));
                if (requested) for (const char* name : {
                    "TS_HOST_MOE_EXPERT_CACHE_DIRECT_READS", "TS_HOST_MOE_EXPERT_CACHE_ORDERED_READS",
                    "TS_HOST_MOE_PREFILL_RING_COALESCE16", "TS_HOST_MOE_PREFILL_DOWN_OVERLAP",
                    "TS_HOST_MOE_EXPERT_CACHE_PREFETCH", "TS_HOST_MOE_RAM_CACHE_VICTIM"}) {
                    const char* value = std::getenv(name);
                    if (value && std::strcmp(value, "1") == 0)
                        throw std::invalid_argument("File sort must be isolated from other source-order/cache experiments");
                }
                return requested;
            }();
            return enabled;
        }
#if defined(TSG_GGML_TEST_HOOKS) && defined(TSG_GGML_USE_CUDA)
        HostMoeFileSortTestSnapshot g_test_file_sort_snapshot;
#endif
        thread_local HostMoeCacheTiming* g_cache_timing = nullptr;
        using CacheStage = HostMoeCacheStage;
        using CacheStageTimer = HostMoeCacheStageTimer<>;

        struct ScopedCacheTiming {
            std::optional<HostMoeCacheTiming> timing;
            HostMoeCacheTiming* previous = nullptr;
            HostMoeFileReadActivity file_reads;
            HostMoeFileReadActivity* previous_file_reads = nullptr;
            std::chrono::steady_clock::time_point start{};
            int layer, rows, exceptions = 0;
            bool joint, bulk, completed = false;
            ScopedCacheTiming(int layer_, int rows_, bool joint_, bool bulk_)
                : layer(layer_), rows(rows_), joint(joint_), bulk(bulk_) {
                const bool sort_requested = file_read_sort_enabled(); // Validate before any entry admission.
                const int view_requested = file_source_view_option();
                if (view_requested < 0) throw std::runtime_error("TS_HOST_MOE_FILE_SOURCE_VIEW must be unset,0 or1");
                static const bool enabled = HostMoeCacheTiming::enabled(
                    std::getenv("TS_HOST_MOE_EXPERT_CACHE_TIMING"));
                if (!enabled) return;
                timing.emplace();
                timing->file_sort_requested = sort_requested ? 1 : 0;
                timing->file_view_requested = view_requested ? 1 : 0;
                timing->file_prefetch_requested = ExpertFileSourceViewPrefetch::option() == 1 ? 1 : 0;
                previous = g_cache_timing; g_cache_timing = &*timing;
                previous_file_reads = g_host_moe_file_read_activity;
                g_host_moe_file_read_activity = &file_reads;
                start = std::chrono::steady_clock::now();
                exceptions = std::uncaught_exceptions();
            }
            ~ScopedCacheTiming() noexcept {
                if (!timing) return;
                g_cache_timing = previous;
                g_host_moe_file_read_activity = previous_file_reads;
                const auto total = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
                const auto& t = *timing;
                const bool success = completed && std::uncaught_exceptions() <= exceptions;
                const auto ms = [&](CacheStage stage) { return t.milliseconds(stage); };
                const auto calls = [&](CacheStage stage) { return static_cast<unsigned long long>(
                    t.calls[static_cast<std::size_t>(stage)]); };
                std::fprintf(stderr,
                    "[HOSTMOE-CACHE-TIMING] layer=%d rows=%d joint=%d bulk=%d outcome=%s "
                    "total_ms=%.6f plan_ms=%.6f plan_regions=%llu allocation_ms=%.6f "
                    "wide_replace_ms=%.6f wide_replaces=%llu file_pipeline_ms=%.6f file_windows=%llu "
                    "file_bytes=%llu upload_enqueue_ms=%.6f upload_wait_ms=%.6f "
                    "input_bridge_ms=%.6f compute_ms=%.6f graph_submissions=%llu output_bridge_ms=%.6f "
                    "ring_calls=%llu ring_expanded_calls=%llu ring_arena_peak_bytes=%llu ring_slot_peak_bytes=%llu "
                    "ring_coalesced_calls=%llu ring_read_tasks_before=%llu ring_read_tasks_after=%llu "
                    "ring_coalesced_payload_bytes=%llu ring_coalesced_bytes=%llu ring_coalesced_read_cap_bytes=%llu "
                    "read_pool_requested=%u read_pool_workers=%u read_pool_maximum=%u read_pool_slots=%u "
                    "file_read_calls=%llu file_read_tasks=%llu file_read_completed_tasks=%llu file_read_failed_calls=%llu file_read_source_bytes=%llu file_read_random_access=%u "
                    "file_sort_requested=%llu file_sort_calls=%llu file_sort_changed_calls=%llu file_sort_misses=%llu file_sort_bytes=%llu "
                    "file_sort_inversions_before=%llu file_sort_inversions_after=%llu "
                    "file_view_requested=%llu file_view_calls=%llu file_view_bytes=%llu file_view_fallbacks=%llu file_view_buffered_bytes=%llu "
                    "file_prefetch_requested=%llu file_prefetch_calls=%llu file_prefetch_ranges=%llu file_prefetch_bytes=%llu file_prefetch_failures=%llu "
                    "vmm_profile=%llu vmm_capacity=%llu vmm_physical_bytes=%llu vmm_pending_bytes=%llu vmm_quarantine_bytes=%llu vmm_certificate=%llu "
                    "scope=nested_host_wall_not_cuda_kernel_or_physical_io\n",
                    layer, rows, joint ? 1 : 0, bulk ? 1 : 0,
                    std::uncaught_exceptions() > exceptions ? "exception" : completed ? "completed" : "not_completed",
                    total, ms(CacheStage::plan), calls(CacheStage::plan), ms(CacheStage::allocation),
                    ms(CacheStage::wide_replace), calls(CacheStage::wide_replace),
                    ms(CacheStage::file_pipeline), static_cast<unsigned long long>(t.file_windows),
                    static_cast<unsigned long long>(t.file_bytes), ms(CacheStage::upload_enqueue),
                    ms(CacheStage::upload_wait), ms(CacheStage::input_bridge), ms(CacheStage::compute),
                    static_cast<unsigned long long>(t.graph_submissions), ms(CacheStage::output_bridge),
                    static_cast<unsigned long long>(t.ring_calls), static_cast<unsigned long long>(t.ring_expanded_calls),
                    static_cast<unsigned long long>(t.ring_arena_peak_bytes), static_cast<unsigned long long>(t.ring_slot_peak_bytes),
                    static_cast<unsigned long long>(success ? t.ring_coalesced_calls : 0),
                    static_cast<unsigned long long>(success ? t.ring_read_tasks_before : 0),
                    static_cast<unsigned long long>(success ? t.ring_read_tasks_after : 0),
                    static_cast<unsigned long long>(success ? t.ring_coalesced_payload_bytes : 0),
                    static_cast<unsigned long long>(success ? t.ring_coalesced_bytes : 0),
                    static_cast<unsigned long long>(success ? t.ring_coalesced_read_cap_bytes : 0),
                    file_reads.requested, file_reads.workers, file_reads.maximum, file_reads.pool_slots,
                    static_cast<unsigned long long>(file_reads.calls), static_cast<unsigned long long>(file_reads.tasks),
                    static_cast<unsigned long long>(file_reads.completed_tasks),
                    static_cast<unsigned long long>(file_reads.valid ? file_reads.failed_calls : UINT64_MAX),
                    static_cast<unsigned long long>(file_reads.source_bytes), file_reads.random_access,
                    static_cast<unsigned long long>(t.file_sort_requested),
                    static_cast<unsigned long long>(success ? t.file_sort_calls : 0),
                    static_cast<unsigned long long>(success ? t.file_sort_changed_calls : 0),
                    static_cast<unsigned long long>(success ? t.file_sort_misses : 0),
                    static_cast<unsigned long long>(success ? t.file_sort_bytes : 0),
                    static_cast<unsigned long long>(success ? t.file_sort_inversions_before : 0),
                    static_cast<unsigned long long>(success ? t.file_sort_inversions_after : 0),
                    static_cast<unsigned long long>(t.file_view_requested),
                    static_cast<unsigned long long>(success ? file_reads.view_calls : 0),
                    static_cast<unsigned long long>(success ? file_reads.view_bytes : 0),
                    static_cast<unsigned long long>(success ? file_reads.view_fallbacks : 0),
                    static_cast<unsigned long long>(success ? file_reads.view_buffered_bytes : 0),
                    static_cast<unsigned long long>(t.file_prefetch_requested),
                    static_cast<unsigned long long>(success ? file_reads.prefetch_calls : 0),
                    static_cast<unsigned long long>(success ? file_reads.prefetch_ranges : 0),
                    static_cast<unsigned long long>(success ? file_reads.prefetch_bytes : 0),
                    static_cast<unsigned long long>(success ? file_reads.prefetch_failures : 0),
                    static_cast<unsigned long long>(success ? t.vmm_profile : 0),
                    static_cast<unsigned long long>(success ? t.vmm_capacity : 0),
                    static_cast<unsigned long long>(success ? t.vmm_physical_bytes : 0),
                    static_cast<unsigned long long>(success ? t.vmm_pending_bytes : 0),
                    static_cast<unsigned long long>(success ? t.vmm_quarantine_bytes : 0),
                    static_cast<unsigned long long>(success ? t.vmm_certificate : 0));
            }
        };

        // Persistent weights and each scratch owner have independent tickets.
#if defined(TSG_GGML_TEST_HOOKS)
        std::atomic<std::size_t> g_test_split_physical[3]{};
        thread_local int g_test_split_current_role = -1;
        thread_local std::uintptr_t g_test_split_current_identity = 0;
        thread_local std::size_t g_test_split_current_physical = 0;
        struct SplitTicketScope {
            int role; std::uintptr_t identity; std::size_t physical;
            SplitTicketScope(int next_role, const void* next_identity, std::size_t next_physical)
                : role(g_test_split_current_role), identity(g_test_split_current_identity), physical(g_test_split_current_physical) {
                g_test_split_current_role = next_role;
                g_test_split_current_identity = reinterpret_cast<std::uintptr_t>(next_identity);
                g_test_split_current_physical = next_physical;
            }
            ~SplitTicketScope() {
                g_test_split_current_role = role; g_test_split_current_identity = identity;
                g_test_split_current_physical = physical;
            }
        };
#else
        struct SplitTicketScope { SplitTicketScope(int, const void*, std::size_t) {} };
#endif
        std::shared_ptr<SharedCacheCharge> split_reserve(int role, const void* identity, std::size_t bytes) {
            SplitTicketScope scope(role, identity, 0);
            return SharedCacheCharge::reserve(g_active_rank, 4, bytes);
        }
        bool split_commit(const std::shared_ptr<SharedCacheCharge>& charge, int role,
            const void* identity, std::size_t physical, std::size_t bytes) {
            SplitTicketScope scope(role, identity, physical);
            return charge && charge->commit(bytes);
        }
        void split_refund(std::shared_ptr<SharedCacheCharge>& charge, int role, const void* identity) {
            SplitTicketScope scope(role, identity, 0);
            charge.reset();
        }
        struct SplitOwner {
            ggml_backend_t backend = nullptr;
            ggml_context* ctx = nullptr;
            ggml_gallocr_t allocator = nullptr;
            ggml_backend_buffer_t buffer = nullptr;
            ggml_cgraph* graph = nullptr;
            std::size_t bytes = 0;
            bool committed = false;
            bool cleanup_uncertain = false;
            int role = 0; // 0 weights; 1 narrow; 2 wide.
            std::shared_ptr<SharedCacheCharge> charge;
#if defined(TSG_GGML_TEST_HOOKS)
            std::size_t physical_bytes = 0;
#endif
            ~SplitOwner() {
                if (allocator || buffer) sync_backend(backend);
                if (allocator) { ggml_gallocr_free(allocator); allocator = nullptr; }
                if (buffer) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
#if defined(TSG_GGML_TEST_HOOKS)
                g_test_physical_bytes.fetch_sub(physical_bytes);
                g_test_split_physical[role].fetch_sub(physical_bytes);
                physical_bytes = 0;
#endif
                if (ctx) { ggml_free(ctx); ctx = nullptr; }
                split_refund(charge, role, this); // Refund only after the real physical frees return.
            }
        };

        struct SplitVariant {
            ggml_tensor* input = nullptr;
            ggml_tensor* ids = nullptr;
            ggml_tensor* routes = nullptr;
            ggml_tensor* output = nullptr;
            ggml_tensor* activation = nullptr;
            ggml_tensor* down = nullptr;
            ggml_cgraph view{};
        };

        struct SplitOverlapGraphs {
            std::vector<ggml_tensor*> prefix_nodes, suffix_nodes;
            ggml_cgraph prefix{}, suffix{};
            std::uint64_t full_uid = 0;
        };

        struct PlannedWide {
            std::unique_ptr<SplitOwner> owner;
            SplitVariant variant;
        };

        std::uint64_t split_graph_uid() {
            static std::atomic<std::uint64_t> next{UINT64_C(0x484d4f4600000000)};
            return next.fetch_add(1, std::memory_order_relaxed);
        }

        std::size_t split_add(std::size_t left, std::size_t right) {
            const auto limit = static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
            if (left > limit || right > limit - left)
                throw std::overflow_error("GLM split physical owner exceeds budget range");
            return left + right;
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
            bool split_weights = false, narrow_committed = false;
            std::size_t narrow_bytes = 0;
            std::unique_ptr<SplitOwner> weight_owner, wide_owner, vmm_pending_wide;
            // Descriptor-only ordinary packed leaf bytes and real independently
            // rounded VMM backing are deliberately different quantities.
            host_moe_vmm::WeightHandle vmm_weight;
            std::size_t vmm_weight_bytes = 0, vmm_maximum_capacity = 0;
            std::uint64_t vmm_mutation_generation = 1;
            bool vmm_planned = false, vmm_failed = false, vmm_unpublished = false, vmm_scratch_uncertain = false;
            std::uint64_t wide_generation = 0;
            std::uint64_t growth_generation = 0;
            bool pooled = false;
            int pool_expert_extent = 0;
            float swiglu_clamp = 0.0f;
            // One exact-width variant for each live batch width. All variants
            // share the expert leaves, but never execute padded/dummy lanes.
            std::vector<ggml_tensor*> width_inputs, width_ids, width_routes, width_outputs;
            std::vector<ggml_cgraph> width_graphs;
            ggml_tensor* wide_activation = nullptr;
            ggml_tensor* wide_down = nullptr;
            std::unique_ptr<SplitOverlapGraphs> down_overlap;
            std::vector<int> expert_for_slot;
            std::vector<ExpertSourceIdentity> source_for_slot;
            std::vector<std::uint64_t> last_used;
            // Slots a decode prefetch filled and no route has used yet (stats).
            std::vector<std::uint8_t> prefetched;
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
                down_overlap.reset();
                wide_owner.reset(); // Graphs reference the persistent external leaves.
                if (allocator != nullptr)
                {
                    sync_backend(backend);
                    ggml_gallocr_free(allocator);
                    allocator = nullptr;
#if defined(TSG_GGML_TEST_HOOKS)
                    g_test_physical_bytes.fetch_sub(test_physical_bytes);
                    if (split_weights) g_test_split_physical[1].fetch_sub(test_physical_bytes);
                    test_physical_bytes = 0;
#endif
                }
                if (ctx != nullptr) ggml_free(ctx);
                // ggml's free is void/fatal on unrecoverable CUDA errors. Never
                // refund before it returns, including allocation/commit rollback.
                if (split_weights) split_refund(shared_charge, 1, this); else shared_charge.reset();
                weight_owner.reset(); // Last: every narrow/wide leaf is no longer live.
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
                if (static_cast<std::size_t>(slot) < prefetched.size()) prefetched[static_cast<std::size_t>(slot)] = 0;
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
        // Caller holds g_cache_mutex; see DecodePrefetch. Must run before any
        // path that can erase an entry.
        void drain_decode_prefetch(bool publish);
        constexpr std::size_t kVmmCacheControlBytes = std::size_t(8) << 20;
        struct VmmCacheControl {
            std::array<HostMoeVmmSlotPlan, 3> slots;
            std::array<std::array<std::uint8_t, HostMoeVmmSlotPlan::maximum_slots>, 3> protect{};
        };
        // The allowance also retains bounded descriptor/shadow-graph planning
        // and existing entry slot vectors. It is not an exact malloc/driver cap.
        static_assert(sizeof(VmmCacheControl) < kVmmCacheControlBytes / 2,
            "VMM phase work must fit its precharged metadata allowance");
        struct VmmCacheState {
            std::uint64_t token = 0, attempts = 0, completed = 0, failures = 0, phase = 0;
            std::uint64_t pressure_mutation_serial = 0; // Scope-monotonic; never reset at transaction completion.
            bool requested = false, configured = false, closed = false, closing = false, failed = false, source_certified = false;
            host_moe_vmm::WeightOptions options{};
            host_moe_vmm::WeightArena* arena = nullptr;
            ggml_backend_t certified_backend = nullptr;
            std::shared_ptr<SharedCacheCharge> control_charge;
            std::unique_ptr<VmmCacheControl> control;
            std::size_t declared_host = 0;
        } g_vmm;
        bool vmm_active() noexcept { return g_vmm.configured && !g_vmm.closed && !g_vmm.closing && !g_vmm.failed && g_vmm.arena; }
        bool vmm_flag() {
            const char* value = std::getenv("TS_HOST_MOE_VMM_WEIGHTS");
            if (!value || std::strcmp(value, "0") == 0) return false;
            if (std::strcmp(value, "1") != 0) throw std::invalid_argument("Invalid TS_HOST_MOE_VMM_WEIGHTS value");
            return true;
        }
        bool vmm_retire_entry_checked(CacheEntry& entry);
        bool compact_vmm_pressure_locked(std::size_t target, const CacheEntry* keep);
        std::size_t vmm_owned_device_bytes(const CacheEntry& entry) {
            const auto usage = host_moe_vmm::weight_usage(g_vmm.arena, entry.vmm_weight);
            auto bytes = split_add(usage.ticket_pending, usage.ticket_committed);
            if (entry.shared_charge) bytes = split_add(bytes, entry.shared_charge->reserved_bytes());
            if (entry.wide_owner && entry.wide_owner->charge)
                bytes = split_add(bytes, entry.wide_owner->charge->reserved_bytes());
            if (entry.vmm_pending_wide && entry.vmm_pending_wide->charge)
                bytes = split_add(bytes, entry.vmm_pending_wide->charge->reserved_bytes());
            return bytes;
        }
        void refresh_vmm_accounting() {
            std::size_t total = 0;
            for (const auto& entry : g_entries) {
                if (entry->vmm_planned) entry->bytes = vmm_owned_device_bytes(*entry);
                total = split_add(total, entry->bytes);
            }
            g_reserved = total;
        }
        std::list<std::unique_ptr<CacheEntry>>::iterator erase_entry_checked(
            std::list<std::unique_ptr<CacheEntry>>::iterator it) {
            if (!cache_configuration().reset_child())
                throw std::runtime_error("Causal-route child could not invalidate retired cache history");
            const auto bytes = (*it)->bytes;
            if ((*it)->vmm_planned && !vmm_retire_entry_checked(**it)) {
                refresh_vmm_accounting();
                throw std::runtime_error("VMM expert owner still requires checked cleanup");
            }
            auto next = g_entries.erase(it);
            if (g_reserved < bytes) throw std::logic_error("Expert retirement accounting underflow");
            g_reserved -= bytes;
            return next;
        }
        std::optional<HostMoeGroupQuotaPlan> g_group_plan;
        std::optional<HostMoePhaseQuotaPlan> g_phase_plan;
        ggml_backend_t g_group_backend = nullptr;
        HostMoeGroupKey group_key(int gate, int up, int down, int hidden, int ff,
            int used, int experts, int pool, float clamp) {
            HostMoeGroupKey key{{gate, up, down, hidden, ff, used, experts, pool}, 0};
            if (clamp != 0.0f) std::memcpy(&key.clamp_bits, &clamp, sizeof(clamp));
            if (!key.valid()) throw std::invalid_argument("Invalid pooled expert group geometry");
            return key;
        }
        HostMoeGroupKey group_key(const CacheEntry& entry) {
            return group_key(entry.type[0], entry.type[1], entry.type[2], entry.hidden, entry.n_ff,
                entry.n_used, entry.num_experts, entry.pool_expert_extent, entry.swiglu_clamp);
        }
        int vmm_group_index(const HostMoeGroupKey& key) noexcept {
            const int* value = key.values.data();
            int experts = 288;
            bool geometry = value[3] == 4096 && value[4] == 2048 && value[5] == 8 && value[6] == experts;
#if defined(TSG_GGML_TEST_HOOKS)
            if (g_test_vmm_profile) { experts = 48; geometry = value[3] == 512 && value[4] == 512
                && value[5] == 2 && value[6] == experts; }
#endif
            // selected_expert_segment uses loaded_layers * expert_count for
            // every group, including the three dense leading trunk layers.
            // Group layer counts (11/28/3) are traffic weights, not pool IDs.
            if (!geometry || value[7] != 45 * experts) return -1;
            if (value[0] == GGML_TYPE_IQ2_XXS && value[1] == GGML_TYPE_IQ2_XXS
                && value[2] == GGML_TYPE_IQ3_XXS) return 0;
            if (value[0] == GGML_TYPE_IQ1_S && value[1] == GGML_TYPE_IQ1_S
                && value[2] == GGML_TYPE_IQ3_XXS) return 1;
            if (value[0] == GGML_TYPE_IQ2_XXS && value[1] == GGML_TYPE_IQ2_XXS
                && value[2] == GGML_TYPE_IQ4_XS) return 2;
            return -1;
        }
        std::array<host_moe_vmm::ProjectionShape, 3> vmm_shapes(const CacheEntry& entry) {
            return {{{static_cast<ggml_type>(entry.type[0]), entry.hidden, entry.n_ff, entry.stride[0]},
                {static_cast<ggml_type>(entry.type[1]), entry.hidden, entry.n_ff, entry.stride[1]},
                {static_cast<ggml_type>(entry.type[2]), entry.n_ff, entry.hidden, entry.stride[2]}}};
        }
        host_moe_vmm::WeightLayoutPlan vmm_layout(const CacheEntry& entry, std::size_t capacity) {
            host_moe_vmm::WeightLayoutPlan result;
            if (!g_vmm.configured || g_vmm.closed || !entry.vmm_maximum_capacity
                || host_moe_vmm::plan_weight_layout(g_vmm.options, vmm_shapes(entry),
                    entry.vmm_maximum_capacity, capacity, result) != host_moe_vmm::WeightStatus::ready)
                throw std::runtime_error("Exact VMM weight layout is unavailable");
            return result;
        }
        void vmm_fail(CacheEntry* entry = nullptr) noexcept {
            g_vmm.failed = true;
            if (entry) entry->vmm_failed = true;
            g_backend_compute_failed.store(true, std::memory_order_release);
        }
        std::size_t group_quota(const HostMoeGroupKey& key, ggml_backend_t backend,
            std::size_t budget, std::size_t layers) {
            if (g_group_plan) return backend == g_group_backend ? g_group_plan->quota(key, budget) : 0;
            return layers ? budget / layers : 0;
        }
        std::size_t entry_quota(const CacheEntry& entry, std::size_t budget, std::size_t layers) {
            // An already published phase entry may retain a large whole union.
            // Do not enforce cold-start fairness on every narrow decode call.
            if (g_phase_plan) return entry.pooled && entry.backend == g_group_backend
                && g_phase_plan->quota(group_key(entry), budget, false) ? budget : 0;
            if (!g_group_plan) return layers ? budget / layers : 0;
            return entry.pooled ? group_quota(group_key(entry), entry.backend, budget, layers) : 0;
        }
        // g_cache_mutex serializes helpers; successful return can leave the last
        // device-output copy queued until the GLM caller's fence. Inactive entry
        // graphs have no references in the caller's outer graph. Erase itself
        // synchronizes, physically frees, then refunds each owner.
        bool reclaim_phase_room(std::size_t incoming, std::size_t replacing,
            const CacheEntry* keep, std::size_t budget) {
            if (!g_phase_plan || incoming > budget || replacing > g_reserved) return false;
            drain_decode_prefetch(false);
            // Preserve source-qualified inactive VMM pools before falling back
            // to whole-entry retirement. The active owner and its current union
            // are fixed; incoming/replacing retain their original admission.
            if (g_vmm.configured && g_reserved - replacing > budget - incoming) {
                const auto target = split_add(budget - incoming, replacing);
                if (compact_vmm_pressure_locked(target, keep)) return true;
            }
            while (g_reserved - replacing > budget - incoming) {
                auto victim = g_entries.end();
                for (auto it = g_entries.begin(); it != g_entries.end(); ++it)
                    if (it->get() != keep) victim = it; // Last eligible entry is LRU.
                if (victim == g_entries.end()) return false;
                erase_entry_checked(victim);
            }
            return true;
        }
        std::unordered_map<const void*, std::shared_ptr<ExpertFileSource>> g_file_sources;
        bool victim_mode() {
            static const bool enabled = [] {
                const char* value = std::getenv("TS_HOST_MOE_RAM_CACHE_VICTIM");
                return value && value[0] == '1' && value[1] == '\0';
            }();
            return enabled;
        }
        HostMoePayloadCache::AllocateStorage payload_storage_allocator() {
            const char* value = std::getenv("TS_HOST_MOE_RAM_CACHE_PINNED");
            if (!victim_mode() && (!value || value[0] != '1' || value[1] != '\0')) return {};
            return [](std::size_t bytes) -> HostMoePayloadCache::Storage {
#ifdef TSG_GGML_USE_CUDA
                auto type = ggml_backend_cuda_host_buffer_type();
#else
                auto type = ggml_backend_cpu_buffer_type();
#endif
                auto buffer = ggml_backend_buft_alloc_buffer(type, bytes);
                if (!buffer) return {};
                std::shared_ptr<void> owner(buffer, [](void* pointer) {
                    ggml_backend_buffer_free(static_cast<ggml_backend_buffer_t>(pointer));
                });
                if (ggml_backend_buffer_get_size(buffer) != bytes) return {};
                bool pinned = false;
#ifdef TSG_GGML_USE_CUDA
                pinned = ggml_backend_buffer_get_type(buffer) == ggml_backend_cuda_host_buffer_type();
#endif
                return {std::move(owner), static_cast<std::uint8_t*>(ggml_backend_buffer_get_base(buffer)), bytes, pinned};
            };
        }
        HostMoePayloadCache g_payload_cache{0,
            [](std::size_t bytes) -> std::shared_ptr<void> {
                return SharedCacheCharge::reserve(0, 6, bytes);
            },
            [](const std::shared_ptr<void>& charge, std::size_t bytes) {
                return std::static_pointer_cast<SharedCacheCharge>(charge)->commit(bytes);
            }, [] {
                const char* value = std::getenv("TS_HOST_MOE_RAM_CACHE_FREQUENCY");
                return value && value[0] == '1' && value[1] == '\0';
            }(), payload_storage_allocator()};
        std::size_t g_payload_cache_budget = 0; // Protected by g_cache_mutex.
        bool mapped_ram_mode() {
            static const bool enabled = [] {
#if defined(_WIN32)
                const auto* value = std::getenv("TS_HOST_MOE_RAM_CACHE_MAPPED");
                return value && std::strcmp(value, "1") == 0;
#else
                return false;
#endif
            }();
            return enabled;
        }
        HostMoeImmutableCache g_mapped_cache{0,
            [](std::size_t bytes) -> std::shared_ptr<void> { return SharedCacheCharge::reserve(0, 6, bytes); },
            [](const std::shared_ptr<void>& charge, std::size_t bytes) {
                return std::static_pointer_cast<SharedCacheCharge>(charge)->commit(bytes);
            }};
        std::shared_ptr<SharedCacheCharge> g_mapped_control;
        std::shared_ptr<SharedCacheCharge> g_mapped_probation_charge;
        std::unique_ptr<HostMoeRamProbation<>> g_mapped_probation;
        std::atomic<std::uint64_t> g_probation_observations{0}, g_probation_firsts{0}, g_probation_repeats{0};
        std::size_t mapped_metadata_bytes() {
            return (g_mapped_control ? HostMoeMappedRam::control_metadata_bytes : 0)
                + (g_mapped_probation_charge ? HostMoeRamProbation<>::metadata_forecast_bytes : 0);
        }
        std::size_t g_mapped_capacity = 0;
        std::atomic<std::uint64_t> g_mapped_source_bytes{0};
        HostMoeMappedRam::Options mapped_ram_options() {
            const auto enabled = [](const char* name) {
                const auto* value = std::getenv(name);
                return value && std::strcmp(value, "1") == 0;
            };
            if (!enabled("TS_HOST_MOE_RAM_CACHE_MAPPED") || !mapped_ram_mode())
                throw std::runtime_error("Mapped RAM mode changed or is unavailable");
            for (const char* name : {"TS_HOST_MOE_RAM_CACHE_PINNED", "TS_HOST_MOE_RAM_CACHE_DIRECT_UPLOAD",
                    "TS_HOST_MOE_RAM_CACHE_VICTIM", "TS_HOST_MOE_RAM_CACHE_PREFETCH_ONLY",
                    "TS_HOST_MOE_RAM_CACHE_FREQUENCY", "TS_HOST_MOE_LAYER_PREFETCH",
                    "TS_HOST_MOE_EXPERT_CACHE_DIRECT_READS", "TS_HOST_MOE_EXPERT_CACHE_PREFETCH",
                    "TS_HOST_MOE_DECODE_PREFETCH"})
                if (enabled(name)) throw std::runtime_error(std::string("Mapped RAM mode conflicts with ") + name);
            HostMoeMappedRam::Options result;
            result.exclusive_quota = enabled("TS_HOST_MOE_RAM_CACHE_MAPPED_EXCLUSIVE_QUOTA");
            const char* quota = std::getenv("TS_HOST_MOE_RAM_CACHE_MAPPED_QUOTA_MB");
            if (quota && *quota && std::strcmp(quota, "0") != 0) {
                std::uint64_t value = 0;
                for (const char* p = quota; *p; ++p) {
                    if (*p < '0' || *p > '9' || value > (UINT64_MAX - (*p - '0')) / 10)
                        throw std::runtime_error("Mapped RAM quota must be a positive MiB count");
                    value = value * 10 + (*p - '0');
                }
                if (!value || value > (std::numeric_limits<std::size_t>::max)() / kMiB)
                    throw std::runtime_error("Mapped RAM quota is outside platform bounds");
                result.quota_ceiling_bytes = static_cast<std::size_t>(value) * kMiB;
            }
            if (result.exclusive_quota != (result.quota_ceiling_bytes != 0))
                throw std::runtime_error("Mapped RAM quota expansion requires its exclusive owner and explicit ceiling");
            return result;
        }
        bool configure_mapped_ram(std::size_t bytes) {
            auto& physical = HostMoeMappedRam::instance();
            if (!bytes) {
                if (g_mapped_probation) g_mapped_probation->clear();
                g_mapped_cache.set_budget(0);
                g_payload_cache_budget = g_mapped_capacity = 0;
                const bool retired = physical.finish_zero();
                if (!retired || g_mapped_cache.stats().reserved_bytes) return false;
                g_mapped_probation.reset();
                g_mapped_probation_charge.reset();
                g_mapped_control.reset(); // Quota/view retirement precedes the control refund.
                return true;
            }
            const auto options = mapped_ram_options();
            constexpr auto control = HostMoeMappedRam::control_metadata_bytes;
            constexpr auto probation = HostMoeRamProbation<>::metadata_forecast_bytes;
            // A tiny lazy ceiling can hold no useful tier. Retire completely,
            // including process quota, before reporting its zero physical use.
            if (bytes <= control + probation) {
                if (!configure_mapped_ram(0)) return false;
                g_mapped_capacity = bytes;
                return true;
            }
            if (!g_mapped_control) {
                auto charge = SharedCacheCharge::reserve(0, 6, control);
                if (!charge || !charge->is_accounted() || !charge->commit(control)) return false;
                g_mapped_control = std::move(charge);
            }
            if (!physical.configure(options, g_mapped_control)) {
                if (!g_mapped_cache.stats().reserved_bytes && physical.finish_zero()) g_mapped_control.reset();
                return false;
            }
            if (!g_mapped_probation) {
                auto charge = SharedCacheCharge::reserve(0, 6, probation);
                if (!charge || !charge->is_accounted()) {
                    if (!g_mapped_cache.stats().reserved_bytes && physical.finish_zero()) g_mapped_control.reset();
                    return false;
                }
                auto policy = std::make_unique<HostMoeRamProbation<>>();
                if (!charge->commit(probation)) {
                    policy.reset();
                    if (!g_mapped_cache.stats().reserved_bytes && physical.finish_zero()) g_mapped_control.reset();
                    return false;
                }
                g_mapped_probation_charge = std::move(charge);
                g_mapped_probation = std::move(policy);
            }
            g_mapped_cache.set_budget(bytes - control - probation);
            g_payload_cache_budget = bytes - control - probation;
            g_mapped_capacity = bytes;
            physical.retry_cleanup();
            return g_mapped_cache.stats().reserved_bytes <= bytes - control - probation;
        }
        void read_mapped_task(ExpertReadTask& task) {
            auto& physical = HostMoeMappedRam::instance();
            auto found = g_mapped_cache.find_ready_range(task.source->identity(), task.cache_offset, task.cache_bytes,
                static_cast<std::size_t>(task.offset - task.cache_offset), task.bytes, task.cache_admit, !task.immutable_admit);
            const auto read_exact = [&] {
                task.source->read(task.offset, task.destination, task.bytes);
                task.source_read_bytes = task.bytes;
                g_mapped_source_bytes.fetch_add(task.bytes, std::memory_order_relaxed);
            };
            if (found.lease && HostMoeMappedRam::guarded_copy(task.destination, found.lease.data, task.bytes)) {
                g_mapped_cache.record_copy(task.bytes);
                return;
            }
            // A failed guarded copy may have written a prefix. The exact
            // buffered read must overwrite the whole destination before H2D.
            if (found.status != HostMoeImmutableCache::ReadyStatus::not_ready) {
                if (found.lease) g_mapped_cache.invalidate({task.source->identity(), task.cache_offset, task.cache_bytes});
                found.lease = {};
                read_exact();
                return;
            }
            // Cold demand uses the ordinary buffered read first, preserving
            // its I/O geometry. Retain a view only after its whole logical
            // projection was read into the already charged arena. Fragments
            // may hit ready views but never fault unread payload for admission.
            try { read_exact(); }
            catch (...) {
                // A second-demand miss has deferred its accounting until
                // admission. Preserve that attempted miss if the read fails.
                if (task.immutable_admit) g_mapped_cache.record_miss_bypass(task.bytes);
                throw;
            }
            if (!task.immutable_admit) return; // Lookup already counted this bypass.
            const auto geometry = physical.plan(*task.source, task.cache_offset, task.cache_bytes);
            if (!geometry.valid) { g_mapped_cache.record_miss_bypass(task.bytes); return; }
            auto retained = g_mapped_cache.acquire_range(task.source->identity(), task.cache_offset, task.cache_bytes,
                geometry.charge_bytes, 0, task.bytes, true,
                [&](const HostMoeImmutableCache::Range& range, const std::shared_ptr<void>& lifetime) {
                    return physical.create(*task.source, range, lifetime);
                });
            // The current demand already has exact staged bytes. A late
            // concurrent cache hit here is not a successful RAM hit-copy.
        }
        struct VictimState {
            std::shared_ptr<SharedCacheCharge> charge;
            std::unique_ptr<HostMoeVictimPolicy<>> policy;
            std::size_t configured = 0;
            std::uint64_t candidates = 0, eligible = 0, admitted = 0, refusals = 0,
                d2h_bytes = 0, completed_bytes = 0, promoted_bytes = 0, failures = 0;
            std::size_t metadata_bytes() const { return policy ? sizeof(*policy) : 0; }
            void clear_metadata() { policy.reset(); charge.reset(); }
            void configure(std::size_t bytes) {
                configured = bytes;
                if (bytes <= sizeof(HostMoeVictimPolicy<>)) {
                    g_payload_cache.set_budget(0); g_payload_cache_budget = 0; clear_metadata(); return;
                }
                if (!policy) {
                    auto next_charge = SharedCacheCharge::reserve(0, 6, sizeof(HostMoeVictimPolicy<>));
                    if (!next_charge || !next_charge->is_accounted()) { ++refusals; return; }
                    try {
                        auto next = std::make_unique<HostMoeVictimPolicy<>>();
                        if (!next_charge->commit(sizeof(*next))) { ++refusals; return; }
                        charge = std::move(next_charge); policy = std::move(next);
                    } catch (const std::bad_alloc&) { ++refusals; return; }
                }
                g_payload_cache_budget = bytes - metadata_bytes();
                g_payload_cache.set_budget(g_payload_cache_budget);
            }
        } g_victim;
        void victim_diagnostics() {
            if (!victim_mode() || !diagnostics()) return;
            const auto stats = g_payload_cache.stats();
            std::fprintf(stderr, "[HOSTMOE-RAM-VICTIM] candidates=%llu eligible=%llu admitted=%llu refusals=%llu "
                "d2h_bytes=%llu completed_bytes=%llu promoted_bytes=%llu failures=%llu metadata_bytes=%zu "
                "payload_bytes=%zu configured_bytes=%zu history_events=%zu history_keys=%zu "
                "allocation_attempts=%llu allocation_bytes=%llu allocation_ns=%llu pool_reuses=%llu "
                "resident_reuses=%llu release_calls=%llu release_ns=%llu recycled_bytes=%llu "
                "free_bytes=%zu free_entries=%zu resident_bytes=%zu resident_entries=%zu\n",
                static_cast<unsigned long long>(g_victim.candidates), static_cast<unsigned long long>(g_victim.eligible),
                static_cast<unsigned long long>(g_victim.admitted), static_cast<unsigned long long>(g_victim.refusals),
                static_cast<unsigned long long>(g_victim.d2h_bytes), static_cast<unsigned long long>(g_victim.completed_bytes),
                static_cast<unsigned long long>(g_victim.promoted_bytes), static_cast<unsigned long long>(g_victim.failures),
                g_victim.metadata_bytes(), stats.reserved_bytes, g_victim.configured,
                g_victim.policy ? g_victim.policy->events() : 0, g_victim.policy ? g_victim.policy->keys() : 0,
                static_cast<unsigned long long>(stats.external_allocations),
                static_cast<unsigned long long>(stats.external_allocation_bytes),
                static_cast<unsigned long long>(stats.external_allocation_ns),
                static_cast<unsigned long long>(stats.external_pool_reuses),
                static_cast<unsigned long long>(stats.external_resident_reuses),
                static_cast<unsigned long long>(stats.external_frees),
                static_cast<unsigned long long>(stats.external_free_ns),
                static_cast<unsigned long long>(stats.external_recycled_bytes),
                stats.external_free_bytes, stats.external_free_entries, stats.resident_bytes, stats.entries);
        }
        using PayloadRanges = std::array<HostMoePayloadCache::Range, 3>;
        struct VictimSource {
            HostMoeVictimKey key;
            PayloadRanges ranges{};
            std::array<std::shared_ptr<ExpertFileSource>, 3> files;
        };
        VictimSource victim_source(const std::array<const void*, 3>& pointers, int expert,
            const std::size_t (&stride)[3], int experts) {
            VictimSource result;
            if (expert < 0 || expert >= experts) return result;
            for (int part = 0; part < 3; ++part) {
                const auto found = g_file_sources.find(pointers[part]);
                if (found == g_file_sources.end() || found->second->bytes() != stride[part] * static_cast<std::size_t>(experts)) return {};
                result.files[part] = found->second;
                result.key.source[part] = found->second->identity();
                result.ranges[part] = {found->second->identity(), static_cast<std::size_t>(expert) * stride[part], stride[part]};
            }
            result.key.expert = expert;
            return result;
        }
        struct VictimTransfers {
            ggml_backend_t backend;
            std::uint64_t error_epoch = g_ggml_error_count.load(std::memory_order_acquire);
            struct Download { VictimSource source; std::array<HostMoePayloadCache::ExternalWrite, 3> writes; };
            struct Promotion { int slot, expert; std::array<HostMoePayloadCache::Lease, 3> leases; };
            std::vector<Download> downloads;
            std::vector<Promotion> promotions;
            bool pending = false;
            explicit VictimTransfers(ggml_backend_t value) : backend(value) {}
            ~VictimTransfers() {
                if (!pending) return;
                sync_backend(backend);
                ++g_victim.failures;
#if defined(TSG_GGML_TEST_HOOKS)
                g_test_payload_upload_drains.fetch_add(1);
#endif
            }
            void demote(CacheEntry& entry, int slot) {
                if (!g_victim.policy || entry.expert_for_slot[slot] < 0) return;
                ++g_victim.candidates;
                const auto& old = entry.source_for_slot[slot];
                auto source = victim_source(old.source, old.expert, entry.stride, entry.num_experts);
                if (!source.key.valid() || !g_victim.policy->admit(source.key)) return;
                ++g_victim.eligible;
                Download download; download.source = std::move(source);
                for (int part = 0; part < 3; ++part) {
                    if (entry.weight[part]->nb[2] != entry.stride[part]
                        || entry.weight[part]->type != entry.type[part]) return;
                    download.writes[part] = g_payload_cache.begin_external(download.source.ranges[part], true);
                    if (!download.writes[part]) { ++g_victim.refusals; return; }
                }
                downloads.push_back(std::move(download)); // Own all destinations before the first DMA.
                pending = true;
                for (int part = 0; part < 3; ++part) {
                    const auto& write = downloads.back().writes[part];
                    ggml_backend_tensor_get_async(backend, entry.weight[part], write.data,
                        static_cast<std::size_t>(slot) * entry.stride[part], write.bytes);
                    g_victim.d2h_bytes += write.bytes;
                }
            }
            void upload_promotions(CacheEntry& entry) {
                for (const auto& promotion : promotions) {
                    pending = true;
                    for (int part = 0; part < 3; ++part) {
                        const auto& lease = promotion.leases[part];
                        ggml_backend_tensor_set_async(backend, entry.weight[part], lease.data,
                            static_cast<std::size_t>(promotion.slot) * entry.stride[part], lease.bytes);
                        g_victim.promoted_bytes += lease.bytes;
                    }
                }
            }
            void complete(CacheEntry& entry, bool already_drained) {
                if (pending && !already_drained) sync_backend(backend);
#if defined(TSG_GGML_TEST_HOOKS)
                if (pending) g_test_payload_upload_drains.fetch_add(1);
#endif
                pending = false;
                if (g_ggml_error_count.load(std::memory_order_acquire) != error_epoch
                    || g_backend_compute_failed.load(std::memory_order_acquire)) {
                    ++g_victim.failures;
                    throw std::runtime_error("Selective expert victim transfer failed before publication");
                }
                for (const auto& promotion : promotions)
                    entry.publish_slot(promotion.slot, promotion.expert);
                for (const auto& download : downloads) {
                    for (const auto& write : download.writes) g_victim.completed_bytes += write.bytes;
                    if (g_payload_cache.publish_external_triple(download.writes)) ++g_victim.admitted;
                }
                // The transfer fence has released every H2D source. Preserve
                // only its allocation/charge, with no readable source identity,
                // so later victims do not pay cudaMallocHost/cudaFreeHost again.
                for (auto& promotion : promotions) g_payload_cache.recycle_external(promotion.leases);
                downloads.clear(); promotions.clear(); // Physical frees follow the successful fence.
            }
        };
        // Read windows can split an expert projection at different offsets in
        // prefill/decode. Cache its canonical full projection, never the window
        // fragment, so phase changes retain useful exact bytes.
        void cache_file_task(ExpertReadTask& task, std::size_t stride, bool admit) {
            if (!stride || g_payload_cache_budget == 0 || victim_mode()) return;
            if (mapped_ram_mode()) {
                task.cache_offset = task.offset / stride * stride;
                task.cache_bytes = stride;
                task.cache_admit = admit && task.offset == task.cache_offset && task.bytes == stride;
                // Called on the constructing thread, before read workers.
                // History order is selected miss then gate/up/down, independent
                // of read completion. First history demand may still hit RAM.
                if (task.cache_admit && g_mapped_probation) {
                    task.immutable_admit = g_mapped_probation->observe({task.source->identity(), task.cache_offset, stride});
                    g_probation_observations.fetch_add(1, std::memory_order_relaxed);
                    (task.immutable_admit ? g_probation_repeats : g_probation_firsts).fetch_add(1, std::memory_order_relaxed);
                }
                task.immutable_read = read_mapped_task;
                return;
            }
            static const bool prefetch_only = [] {
                const char* value = std::getenv("TS_HOST_MOE_RAM_CACHE_PREFETCH_ONLY");
                return value && value[0] == '1' && value[1] == '\0';
            }();
            // The request has joined/cleaned up its speculative payloads before
            // decode. Avoid a useless cache mutex/lookup on every miss in this
            // mode; the ordinary source path still counts actual read bytes.
            // Wide prefill passes admit=false and retains its exact-byte probes.
            if (prefetch_only && admit) return;
            task.payload_cache = &g_payload_cache;
            task.cache_offset = task.offset / stride * stride;
            task.cache_bytes = stride;
            task.cache_admit = admit && !prefetch_only;
            static const bool direct = [] {
                for (const char* name : {"TS_HOST_MOE_RAM_CACHE_PINNED", "TS_HOST_MOE_RAM_CACHE_DIRECT_UPLOAD"}) {
                    const char* value = std::getenv(name);
                    if (value && value[0] == '1' && value[1] == '\0') return true;
                }
                return false;
            }();
            task.cache_direct_upload = direct;
        }
        // Declare after the task vector: queued reads may have published DMA
        // leases before a later read/upload throws. Drain before those leases
        // unwind, including pageable fallback and partially consumed windows.
        struct FilePayloadUploadDrain {
            ggml_backend_t backend;
            const std::vector<ExpertReadTask>& tasks;
            ~FilePayloadUploadDrain() {
                for (const auto& task : tasks) if (task.payload_lease) {
                    sync_backend(backend);
#if defined(TSG_GGML_TEST_HOOKS)
                    g_test_payload_upload_drains.fetch_add(1);
#endif
                    break;
                }
            }
        };
        // One process-wide workspace under g_cache_mutex, never one per layer.
        // Kept through asynchronous uploads, drained on failure and released at
        // cache teardown. Larger routed sets retain the ordinary mmap path.
        constexpr std::size_t kFileReadWorkspace = kHostMoeFileStagingBytes;
        bool scalar_file_staging_enabled() {
            static const bool enabled = host_moe_scalar_staging_flag(std::getenv("TS_HOST_MOE_DECODE_STAGING_72MB"));
            return enabled;
        }
        bool prefill_ring_staging_enabled() {
            static const bool enabled = host_moe_scalar_staging_flag(std::getenv("TS_HOST_MOE_PREFILL_RING_72MB"));
            return enabled;
        }

        bool prefill_down_overlap_enabled() {
            static const bool enabled = [] {
                const auto* value = std::getenv("TS_HOST_MOE_PREFILL_DOWN_OVERLAP");
                return value && std::strcmp(value, "1") == 0;
            }();
            return enabled;
        }
        bool prefill_ring_coalesce_enabled() {
            static const bool enabled = host_moe_ring_coalesce_flag(std::getenv("TS_HOST_MOE_PREFILL_RING_COALESCE16"));
            return enabled;
        }
        struct FileReadWorkspace {
            ggml_backend_buffer_t buffer = nullptr;
            std::shared_ptr<SharedCacheCharge> shared_charge;
            // The extra ticket covers only the increase over the original
            // arena. Retain the original credit while replacing its physical
            // buffer, so admission never requires a temporary 32+72 MiB peak.
            std::shared_ptr<SharedCacheCharge> expansion_charge;
            bool cleanup_uncertain = false;
            std::size_t size() const { return buffer ? ggml_backend_buffer_get_size(buffer) : 0; }
            std::uint8_t* data() const { return static_cast<std::uint8_t*>(ggml_backend_buffer_get_base(buffer)); }
            void clear() {
                if (g_vmm.configured) {
                    if (cleanup_uncertain) throw std::runtime_error("VMM staging physical cleanup is quarantined");
                    const auto epoch = g_ggml_error_count.load(std::memory_order_acquire);
                    if (buffer) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
                    if (g_ggml_error_count.load(std::memory_order_acquire) != epoch) {
                        cleanup_uncertain = true;
                        throw std::runtime_error("VMM staging free changed the backend error epoch");
                    }
                    if (expansion_charge && !expansion_charge->release_checked())
                        throw std::runtime_error("VMM staging expansion ticket needs checked acknowledgment");
                    expansion_charge.reset();
                    if (shared_charge && !shared_charge->release_checked())
                        throw std::runtime_error("VMM staging ticket needs checked acknowledgment");
                    shared_charge.reset(); return;
                }
                if (buffer) ggml_backend_buffer_free(buffer);
                buffer = nullptr;
                expansion_charge.reset();
                shared_charge.reset(); // Return host credit after physical free.
            }
            ~FileReadWorkspace() { clear(); }
            void reserve(std::size_t bytes) {
                if (bytes <= size()) return;
                if (bytes > kFileReadWorkspace) throw std::runtime_error("Expert read workspace exceeds its ceiling");
                // The previous row has completed; free before growing so the
                // transfer arena never temporarily holds two allocations.
                clear();
                if (g_vmm.configured) {
                    shared_charge = SharedCacheCharge::reserve(0, 3, bytes);
                    if (!shared_charge || !shared_charge->is_accounted())
                        throw std::runtime_error("VMM staging requires actual shared Host admission");
#ifdef TSG_GGML_USE_CUDA
                    auto type = ggml_backend_cuda_host_buffer_type();
#else
                    auto type = ggml_backend_cpu_buffer_type();
#endif
                    // The process-owned workspace retains its ticket through
                    // allocation/commit failure and failed cleanup acknowledgment.
                    buffer = ggml_backend_buft_alloc_buffer(type, bytes);
                    if (!buffer || size() != bytes || !shared_charge->commit(size()))
                        throw std::runtime_error("VMM staging allocation/commit failed");
                    return;
                }
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
            bool try_reserve_expanded() {
                constexpr auto target = kHostMoeScalarFileStagingBytes;
                const auto previous = size();
                if (previous >= target) return true;
                // This experiment requires actual host admission, including
                // standalone native callers. Refusal leaves the old arena and
                // its bytes intact; the caller keeps the ordinary window.
                if (!SharedCacheCharge::host_accounting_attached()) return false;
                auto charge = SharedCacheCharge::reserve(0, 3, target - previous);
                if (!charge || !charge->is_accounted()) return false;
                if (g_vmm.configured) expansion_charge = std::move(charge);
                const auto epoch = g_ggml_error_count.load(std::memory_order_acquire);
                if (buffer) ggml_backend_buffer_free(buffer);
                buffer = nullptr; // Physical old storage is gone; credit stays.
                if (g_vmm.configured && g_ggml_error_count.load(std::memory_order_acquire) != epoch) {
                    cleanup_uncertain = true;
                    throw std::runtime_error("VMM staging resize free is quarantined");
                }
                try {
#if defined(TSG_GGML_TEST_HOOKS)
                    if (g_test_file_workspace_failure.exchange(0) == 1)
                        throw std::runtime_error("Injected expanded expert staging allocation failure");
#endif
#ifdef TSG_GGML_USE_CUDA
                    auto type = ggml_backend_cuda_host_buffer_type();
#else
                    auto type = ggml_backend_cpu_buffer_type();
#endif
                    buffer = ggml_backend_buft_alloc_buffer(type, target);
                    if (!buffer) throw std::runtime_error("Cannot allocate expanded expert read workspace");
                    const auto& admitted = g_vmm.configured ? expansion_charge : charge;
                    if (size() != target || !admitted->commit(target - previous))
                        throw std::runtime_error("Cannot commit expanded expert read workspace to shared host budget");
                    if (!g_vmm.configured) expansion_charge = std::move(charge);
                    return true;
                } catch (...) {
                    // No retry/fallback after physical allocation begins.
                    // Both tickets outlive the allocation they jointly cover.
                    clear();
                    throw;
                }
            }
        } g_file_read_buffer;

#ifdef TSG_GGML_USE_CUDA
        // ---- Next-layer decode prefetch --------------------------------------
        // One prefetch in flight. g_cache_mutex guards every field the main
        // thread touches. The worker only reads files into `staging` and uploads
        // them into the target entry's chosen slots on its own CUDA stream; those
        // slots stay invalid (unmatched, free) until the main thread drains the
        // job and publishes them. Every consumer of the target entry and every
        // path that can erase entries drains first.
        struct DecodePrefetch {
            std::thread thread;
            std::mutex mutex;
            std::condition_variable cv;
            bool stop = false, job = false, done = true;
            std::exception_ptr error;
            CacheEntry* entry = nullptr;                 // target, valid until drained
            std::vector<std::pair<int, int>> fills;      // (slot, expert)
            std::vector<ExpertReadTask> tasks;
            std::vector<std::pair<int, int>> task_slot_part;
            ggml_backend_t upload = nullptr;             // second backend: its own stream
            ggml_backend_buffer_t staging = nullptr;
            std::shared_ptr<SharedCacheCharge> staging_charge;
            std::uint64_t issued = 0, published = 0, used = 0;

            static tsg_dsv41::engram_io_pool& readers() {
                // Separate from the seam's own reader pool, so the next layer's
                // reads do not queue behind this layer's misses.
                static tsg_dsv41::engram_io_pool pool(8);
                return pool;
            }
            void run() {
                for (;;) {
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        cv.wait(lock, [&] { return stop || job; });
                        if (stop) return;
                        job = false;
                    }
                    std::exception_ptr failure;
                    try {
                        readers().run(tasks.size(), [&](std::size_t i) { tasks[i].prepare(); });
                        for (std::size_t i = 0; i < tasks.size(); ++i) {
                            const auto [slot, part] = task_slot_part[i];
                            ggml_backend_tensor_set_async(upload, entry->weight[part], tasks[i].destination,
                                static_cast<std::size_t>(slot) * entry->stride[part], tasks[i].bytes);
                        }
                        ggml_backend_synchronize(upload);
                    } catch (...) { failure = std::current_exception(); }
                    std::lock_guard<std::mutex> lock(mutex);
                    error = failure;
                    done = true;
                    cv.notify_all();
                }
            }
            void start() {
                if (!thread.joinable()) thread = std::thread([this] { run(); });
                std::lock_guard<std::mutex> lock(mutex);
                done = false; job = true; error = nullptr;
                cv.notify_all();
            }
            void wait() {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&] { return done; });
            }
            // Caller holds g_cache_mutex. Publishes the filled slots unless the
            // read/upload failed or the caller is discarding the target.
            void drain(bool publish) {
                if (entry == nullptr) return;
                wait();
                if (publish && !error)
                    for (const auto& [slot, expert] : fills) {
                        entry->publish_slot(slot, expert);
                        entry->last_used[slot] = ++entry->clock;
                        if (entry->prefetched.size() != static_cast<std::size_t>(entry->capacity))
                            entry->prefetched.assign(static_cast<std::size_t>(entry->capacity), 0);
                        entry->prefetched[static_cast<std::size_t>(slot)] = 1;
                        ++published;
                    }
                entry = nullptr; fills.clear(); tasks.clear(); task_slot_part.clear(); error = nullptr;
            }
            void release_resources() {
                drain(false);
                if (staging) ggml_backend_buffer_free(staging);
                staging = nullptr; staging_charge.reset();
                if (upload) ggml_backend_free(upload);
                upload = nullptr;
            }
            ~DecodePrefetch() {
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    stop = true;
                    cv.notify_all();
                }
                if (thread.joinable()) thread.join();
            }
        } g_prefetch;
        void drain_decode_prefetch(bool publish) { g_prefetch.drain(publish); }

        // Serial cache gate permits one borrowed compute/owned upload pair.
        // No payload belongs to this backend: both phases use the existing
        // charged file arena and independently owned persistent weight leaves.
        struct PrefillDownUpload {
            ggml_backend_t owner = nullptr, upload = nullptr;
            void clear() noexcept {
                if (upload) {
                    try { sync_backend(upload); }
                    catch (...) { g_backend_compute_failed.store(true, std::memory_order_release); }
                    ggml_backend_free(upload);
                }
                upload = owner = nullptr;
            }
            bool prepare(ggml_backend_t backend, ggml_backend_buffer_type_t actual_type) {
                if (upload && owner == backend) return true;
                clear();
                // The precision facade's device deliberately cannot create a
                // backend. The owned physical buffer type identifies CUDA.
                auto device = ggml_backend_buft_get_device(actual_type);
                if (!device || ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) return false;
                auto candidate = ggml_backend_dev_init(device, nullptr);
                if (!candidate) throw std::runtime_error("Prefill Down upload backend creation failed");
                if (candidate == backend || !ggml_backend_is_cuda(candidate)
                    || ggml_backend_get_default_buffer_type(candidate) != actual_type) {
                    ggml_backend_free(candidate); return false;
                }
                owner = backend; upload = candidate;
                return true;
            }
        } g_prefill_down_upload;

        bool phase_cold_growth_enabled() {
            static const bool enabled = [] {
                const auto* value = std::getenv("TS_HOST_MOE_PHASE_COLD_GROWTH");
                return value && std::strcmp(value, "1") == 0;
            }();
            return enabled;
        }
        bool decode_rebalance_material_deficit() {
            static const bool enabled = [] {
                const auto* value = std::getenv("TS_HOST_MOE_DECODE_REBALANCE");
                return value && std::strcmp(value, "1") == 0;
            }();
            return enabled;
        }
        bool decode_rebalance_retain_peer() {
            static const bool requested = [] {
                const auto* value = std::getenv("TS_HOST_MOE_DECODE_REBALANCE_RETAIN_PEER");
                if (!value || std::strcmp(value, "0") == 0) return false;
                if (std::strcmp(value, "1") == 0) return true;
                throw std::invalid_argument("Invalid TS_HOST_MOE_DECODE_REBALANCE_RETAIN_PEER; expected 0 or 1");
            }();
            if (requested && !decode_rebalance_material_deficit())
                throw std::invalid_argument("Decode retained-peer rebalance requires TS_HOST_MOE_DECODE_REBALANCE=1");
            return requested;
        }
        std::size_t phase_resident_bytes(const HostMoeGroupKey& key, ggml_backend_t backend) {
            std::size_t bytes = 0;
            for (const auto& entry : g_entries)
                if (entry->pooled && entry->split_weights && entry->backend == backend
                    && group_key(*entry) == key) bytes += entry->bytes;
            return bytes;
        }
        void nominate_phase_cold_growth(std::size_t budget, ggml_backend_t backend) {
            g_phase_cold_growth_group.reset();
            HostMoeGroupKey selected;
            if (g_phase_plan && phase_cold_growth_enabled()
                && cache_configuration().settings().capacity_growth
                && g_phase_plan->nominate_cold_growth(budget, g_reserved,
                    [&](const auto& key) { return phase_resident_bytes(key, backend); }, selected))
                g_phase_cold_growth_group = selected;
        }
        // Caller holds g_cache_mutex. Erased tables are recreated lazily by the
        // next narrow call of their group, each at min(weighted share, room left).
        // Only a gross imbalance is rebuilt: one table holding over twice its
        // decode share while the groups below half of theirs (or experimentally
        // three quarters), or absent without room for full shares, are starved.
        // Affordable unions near their shares keep their hot slots.
        void rebalance_pooled_tables_locked(std::size_t budget, ggml_backend_t backend)
        {
            const bool material_deficit = decode_rebalance_material_deficit();
            const bool retain_peer = decode_rebalance_retain_peer();
            const auto errors_before = g_ggml_error_count.load(std::memory_order_acquire);
            const auto share = [&](const CacheEntry& entry) {
                return g_phase_plan->quota(group_key(entry), budget, false);
            };
            const auto skewed = [&](const CacheEntry& entry) {
                const auto quota = share(entry);
                return quota != 0 && (HostMoePhaseQuotaPlan::exceeds_twice_share(entry.bytes, quota)
                    || HostMoePhaseQuotaPlan::below_decode_share(entry.bytes, quota, material_deficit));
            };
            const auto owned = [&](const CacheEntry& entry) { return entry.pooled && entry.backend == backend; };
            // Full decode shares of the groups that currently own no table.
            const auto absent_shares = [&] {
                std::size_t total = 0;
                for (const auto& group : g_phase_plan->groups()) {
                    bool present = false;
                    for (const auto& entry : g_entries)
                        present = present || (owned(*entry) && group_key(*entry) == group.key);
                    if (!present) total += g_phase_plan->quota(group.key, budget, false);
                }
                return total;
            };
            const auto room = [&] { return budget > g_reserved ? budget - g_reserved : 0; };
            const auto absent_floors = [&] {
                return g_phase_plan->absent_narrow_floors([&](const HostMoeGroupKey& key) {
                    return phase_resident_bytes(key, backend);
                });
            };
            if (!g_phase_plan->should_rebalance_decode(budget, g_reserved, [&](const HostMoeGroupKey& key) {
                std::size_t bytes = 0;
                for (const auto& entry : g_entries)
                    if (owned(*entry) && group_key(*entry) == key) bytes += entry->bytes;
                return bytes;
            }, material_deficit)) return;
            drain_decode_prefetch(false);
            const auto erase = [&](std::list<std::unique_ptr<CacheEntry>>::iterator it) {
                const auto bytes = (*it)->bytes;
                if (diagnostics()) std::fprintf(stderr,
                    "[HOSTMOE-REBALANCE] layer=%d slots=%d bytes=%zu decode_quota=%zu material_deficit=%d\n",
                    (*it)->layer, (*it)->capacity, bytes, share(**it), material_deficit ? 1 : 0);
                return erase_entry_checked(it);
            };
            for (auto it = g_entries.begin(); it != g_entries.end();)
                it = owned(**it) && skewed(**it) ? erase(it) : std::next(it);
            // Recreated groups get only the room beside retained tables, and
            // hot growth still requires old+new room. Ordinary policy funds
            // full absent shares; retained-peer policy funds exact mandatory
            // floors and bounds later optional cold capacity independently.
            while (room() < (retain_peer ? absent_floors() : absent_shares())) {
                auto largest = g_entries.end();
                std::size_t excess = 0;
                for (auto it = g_entries.begin(); it != g_entries.end(); ++it) {
                    if (!owned(**it) || (*it)->bytes <= share(**it)) continue;
                    if ((*it)->bytes - share(**it) > excess) { excess = (*it)->bytes - share(**it); largest = it; }
                }
                if (largest == g_entries.end()) break;
                erase(largest);
            }
            if (retain_peer && room() < absent_floors())
                throw std::runtime_error("Retained-peer rebalance cannot preserve every absent mandatory floor");
            if (retain_peer && (g_ggml_error_count.load(std::memory_order_acquire) != errors_before
                || g_backend_compute_failed.load(std::memory_order_acquire)))
                throw std::runtime_error("Retained-peer rebalance retirement changed the backend error epoch");
            // The old full-share recovery would still erase an existing peer.
            // This event reports only the already-published payload we kept;
            // cold allocation must separately protect the absent floors below.
            if (retain_peer && room() < absent_shares()) {
                for (const auto& entry : g_entries) if (owned(*entry) && entry->bytes > share(*entry)) {
                    const auto hot = static_cast<std::size_t>(std::count_if(
                        entry->expert_for_slot.begin(), entry->expert_for_slot.end(),
                        [](int expert) { return expert >= 0; }));
                    if (!hot) continue;
                    std::fprintf(stderr,
                        "[HOSTMOE-REBALANCE-RETAIN] profile=1 layer=%d slots=%d hot=%zu owned_bytes=%zu "
                        "decode_quota=%zu room_bytes=%zu absent_floor_bytes=%zu absent_share_bytes=%zu "
                        "outcome=completed scope=published_gpu_payload_retained_not_new_capacity_or_physical_free\n",
                        entry->layer, entry->capacity, hot, entry->bytes, share(*entry), room(),
                        absent_floors(), absent_shares());
                }
            }
        }

        bool valid_layout(const HostMoeSegment& hm, std::size_t (&stride)[3], int max_rows, bool explicit_model = false)
        {
            if (hm.seq_len <= 0 || hm.seq_len > max_rows
                || !host_moe_cache_prefill_fits(hm.seq_len, hm.n_used, explicit_model)
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

        ggml_context* split_context(int variants) {
            if (variants < 1 || variants > kHostMoeCacheMaxDecodeRows + 1) return nullptr;
            ggml_init_params params{};
            const std::size_t nodes = 256 * static_cast<std::size_t>(variants);
            params.mem_size = ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false);
            params.no_alloc = true;
            return ggml_init(params);
        }

        bool emit_split_variant(CacheEntry& entry, ggml_context* ctx, ggml_cgraph* graph,
            int width, bool bulk, SplitVariant& result) {
            const int start = ggml_graph_n_nodes(graph);
            result.input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, entry.hidden, bulk ? 1 : width, bulk ? width : 1);
            result.ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, bulk ? entry.n_used : entry.n_used * width, bulk ? width : 1);
            result.routes = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, entry.n_used, width);
            ggml_set_input(result.input); ggml_set_input(result.ids); ggml_set_input(result.routes);
            auto* gate = ggml_mul_mat_id(ctx, entry.weight[0], result.input, result.ids);
            auto* up = ggml_mul_mat_id(ctx, entry.weight[1], result.input, result.ids);
            auto* gate_projection = gate;
            auto* up_projection = up;
            if (entry.swiglu_clamp > 0.0f) {
                up = ggml_clamp(ctx, up, -entry.swiglu_clamp, entry.swiglu_clamp);
                gate = ggml_clamp(ctx, gate, -INFINITY, entry.swiglu_clamp);
            }
            auto* activation = ggml_mul(ctx, ggml_silu(ctx, gate), up);
            auto* down = ggml_mul_mat_id(ctx, entry.weight[2], activation, result.ids);
            result.activation = activation; result.down = down;
            auto* down_by_route = bulk || width == 1 ? down
                : ggml_cont(ctx, ggml_permute(ctx,
                    ggml_reshape_3d(ctx, down, entry.hidden, width, entry.n_used), 0, 2, 1, 3));
            if (!bulk) ggml_set_output(down_by_route);
            auto* weighted = ggml_mul(ctx, down_by_route, result.routes);
            auto* sum = ggml_view_2d(ctx, weighted, entry.hidden, width, weighted->nb[2], 0);
            for (int k = 1; k < entry.n_used; ++k)
                sum = ggml_add(ctx, sum, ggml_view_2d(ctx, weighted, entry.hidden, width,
                    weighted->nb[2], static_cast<std::size_t>(k) * weighted->nb[1]));
            result.output = ggml_cont(ctx, sum);
            ggml_set_output(result.output);
            ggml_build_forward_expand(graph, result.output);
            result.view = ggml_graph_view(graph, start, ggml_graph_n_nodes(graph));
            result.view.uid = split_graph_uid();
            return ggml_backend_supports_op(entry.backend, gate_projection)
                && ggml_backend_supports_op(entry.backend, up_projection)
                && ggml_backend_supports_op(entry.backend, gate)
                && ggml_backend_supports_op(entry.backend, up)
                && ggml_backend_supports_op(entry.backend, down);
        }

        void publish_split_variant(CacheEntry& entry, int index, const SplitVariant& variant) {
            entry.width_inputs[index] = variant.input; entry.width_ids[index] = variant.ids;
            entry.width_routes[index] = variant.routes; entry.width_outputs[index] = variant.output;
            entry.width_graphs[index] = variant.view;
            if (index == entry.graph_rows) {
                entry.down_overlap.reset();
                entry.wide_activation = variant.activation; entry.wide_down = variant.down;
            }
        }

        std::size_t measure_split_scratch(CacheEntry& entry, ggml_cgraph* graph) {
            // Fresh planner only. Real type identity is used for weight leaves;
            // a distinct TS-only proxy preserves exact scratch padding/geometry.
            auto real = ggml_backend_get_default_buffer_type(entry.backend);
            native18_draft::SizeOnlyBufferType scratch(real);
            ggml_backend_buffer_type_t bufts[2] = {scratch.get(), real};
            ggml_gallocr_t planner = ggml_gallocr_new_n(bufts, 2);
            if (!planner) throw std::runtime_error("GLM scratch planner allocation failed");
            struct FreePlan { ggml_gallocr_t value; ~FreePlan() { ggml_gallocr_free(value); } } cleanup{planner};
            std::vector<int> nodes(graph->n_nodes, 0), leaves(graph->n_leafs, 0);
            for (int i = 0; i < graph->n_leafs; ++i)
                for (auto* weight : entry.weight)
                    if (graph->leafs[i] == weight) leaves[i] = 1;
            std::size_t sizes[2]{};
            ggml_gallocr_reserve_n_size(planner, graph, nodes.data(), leaves.data(), sizes);
            if (scratch.allocation_attempts() || ggml_gallocr_get_buffer_size(planner, 0)
                || ggml_gallocr_get_buffer_size(planner, 1))
                throw std::logic_error("GLM fresh scratch planner allocated physical storage");
            const bool bound = entry.weight[0]->data && entry.weight[1]->data && entry.weight[2]->data;
            if ((bound && sizes[1] != 0) || (!bound && sizes[1] != entry.weight_owner->bytes))
                throw std::logic_error("GLM independent weight sizing mismatch");
            return sizes[0];
        }

        PlannedWide plan_split_wide(CacheEntry& entry, int width) {
            CacheStageTimer timing(g_cache_timing, CacheStage::plan);
            PlannedWide result;
            if (width <= TSG_PRECISION_DECODE_COLUMNS) return result;
            result.owner = std::make_unique<SplitOwner>();
            auto& owner = *result.owner;
            owner.role = 2;
            owner.backend = entry.backend;
            owner.ctx = split_context(1);
            if (!owner.ctx) { result.owner.reset(); return result; }
            owner.graph = ggml_new_graph_custom(owner.ctx, 256, false);
            owner.graph->uid = split_graph_uid();
            if (!emit_split_variant(entry, owner.ctx, owner.graph, width, true, result.variant)) {
                result.owner.reset(); return result;
            }
            owner.bytes = measure_split_scratch(entry, owner.graph);
            owner.allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(entry.backend));
            if (!owner.allocator || !owner.bytes) result.owner.reset();
            return result;
        }

        bool build_split_graph(CacheEntry& entry, int capacity, std::size_t quota, int rows, int bulk_rows,
            int resource_prefill_rows, bool exact_phase_width) {
            // Static Native20 plans include the configured maximum wide owner
            // even on N1. Phase plans instead own only the actual wide phase;
            // their narrow floor must not secretly materialize a wide scratch.
            if (resource_prefill_rows < 1 || bulk_rows > resource_prefill_rows) return false;
            if (!exact_phase_width)
                bulk_rows = resource_prefill_rows > TSG_PRECISION_DECODE_COLUMNS ? resource_prefill_rows : 0;
            entry.split_weights = true; entry.capacity = capacity;
            entry.graph_rows = rows; entry.bulk_rows = bulk_rows;
            entry.eviction_policy.configure(entry.pool_expert_extent, capacity, false);
            entry.weight_owner = std::make_unique<SplitOwner>();
            auto& weights = *entry.weight_owner;
            weights.backend = entry.backend;
            weights.ctx = split_context(1);
            if (!weights.ctx) return false;
            entry.weight[0] = ggml_new_tensor_3d(weights.ctx, static_cast<ggml_type>(entry.type[0]), entry.hidden, entry.n_ff, capacity);
            entry.weight[1] = ggml_new_tensor_3d(weights.ctx, static_cast<ggml_type>(entry.type[1]), entry.hidden, entry.n_ff, capacity);
            entry.weight[2] = ggml_new_tensor_3d(weights.ctx, static_cast<ggml_type>(entry.type[2]), entry.n_ff, entry.hidden, capacity);
            for (auto* weight : entry.weight) { ggml_set_input(weight); ggml_set_output(weight); }
            weights.bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(weights.ctx,
                ggml_backend_get_default_buffer_type(entry.backend));
            if (!weights.bytes) return false;
            entry.ctx = split_context(rows);
            if (!entry.ctx) return false;
            entry.graph = ggml_new_graph_custom(entry.ctx, 256 * static_cast<std::size_t>(rows), false);
            entry.graph->uid = split_graph_uid();
            // One reserved vector slot for the current wide graph; no shape-indexed history.
            entry.width_inputs.resize(rows + 1); entry.width_ids.resize(rows + 1);
            entry.width_routes.resize(rows + 1); entry.width_outputs.resize(rows + 1);
            entry.width_graphs.resize(rows + 1);
            for (int index = 0; index < rows; ++index) {
                SplitVariant variant;
                if (!emit_split_variant(entry, entry.ctx, entry.graph, index + 1, false, variant)) return false;
                publish_split_variant(entry, index, variant);
            }
            entry.narrow_bytes = split_add(measure_split_scratch(entry, entry.graph), kWorkspaceAllowance);
            entry.allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(entry.backend));
            if (!entry.allocator) return false;
            if (bulk_rows > 0) {
                auto wide = plan_split_wide(entry, bulk_rows);
                if (!wide.owner) return false;
                publish_split_variant(entry, rows, wide.variant);
                entry.wide_owner = std::move(wide.owner);
            }
            entry.input = entry.width_inputs[0]; entry.ids = entry.width_ids[0];
            entry.routes = entry.width_routes[0]; entry.output = entry.width_outputs[0];
            entry.bytes = split_add(weights.bytes, entry.narrow_bytes);
            if (entry.wide_owner) entry.bytes = split_add(entry.bytes, entry.wide_owner->bytes);
            if (g_vmm.configured) {
                if (!vmm_active() || vmm_group_index(group_key(entry)) < 0)
                    throw std::runtime_error("VMM scope cannot use an ordinary or incompatible split layout");
                const auto stride_total = split_add(split_add(entry.stride[0], entry.stride[1]), entry.stride[2]);
                entry.vmm_maximum_capacity = (std::min)(static_cast<std::size_t>(entry.pool_expert_extent),
                    cache_configuration().maximum_budget() / stride_total);
                if (entry.vmm_maximum_capacity < static_cast<std::size_t>(capacity)
                    || entry.vmm_maximum_capacity > HostMoeVmmSlotPlan::maximum_slots)
                    throw std::runtime_error("VMM immutable capacity bound cannot cover this split plan");
                entry.vmm_planned = true;
                entry.vmm_weight_bytes = vmm_layout(entry, capacity).total;
                entry.bytes = split_add(entry.vmm_weight_bytes, entry.narrow_bytes);
                if (entry.wide_owner) entry.bytes = split_add(entry.bytes, entry.wide_owner->bytes);
            }
            return entry.bytes <= quota;
        }

        bool build_graph(CacheEntry& entry, int capacity, std::size_t quota, int graph_rows, int bulk_rows = 0,
            int resource_prefill_rows = 0, bool exact_phase_width = false)
        {
            if (entry.pooled) return build_split_graph(entry, capacity, quota, graph_rows, bulk_rows, resource_prefill_rows, exact_phase_width);
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
            // Measured even when refused, so a quota miss can refit capacity.
            entry.bytes = device_bytes + kWorkspaceAllowance;
            return device_bytes <= quota && quota - device_bytes >= kWorkspaceAllowance;
        }

        std::unique_ptr<CacheEntry> create_entry(const HostMoeSegment& hm,
            const std::size_t (&stride)[3], ggml_backend_t backend, std::size_t quota,
            std::size_t maximum_capacity, int graph_rows = 1, int minimum_capacity = 0, int bulk_rows = 0,
            int resource_prefill_rows = 0, bool exact_phase_width = false)
        {
            CacheStageTimer timing(g_cache_timing, CacheStage::plan);
            exact_phase_width = exact_phase_width || g_phase_plan.has_value();
            if (hm.pooled_model_experts > 0 && resource_prefill_rows == 0) {
                if (!cache_configuration().active()) return nullptr;
                resource_prefill_rows = cache_configuration().settings().max_rows;
                if (g_phase_plan) {
                    int decode_rows = 0, prefill_rows = 0;
                    const auto& settings = cache_configuration().settings();
                    if (!g_phase_plan->row_limits(host_moe_expert_group_key(hm), settings.max_decode_rows,
                            settings.max_rows, decode_rows, prefill_rows)
                        || graph_rows > glm_compact_graph_rows(decode_rows, prefill_rows)
                        || bulk_rows > prefill_rows) return nullptr;
                    resource_prefill_rows = prefill_rows;
                }
            }
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
            std::size_t refused_bytes = 0;
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
                if (!build_graph(*entry, static_cast<int>(slots), quota, graph_rows, bulk_rows, resource_prefill_rows, exact_phase_width)) {
                    refused_bytes = entry->bytes;
                    entry.reset();
                }
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
            // The slot estimate above ignores graph scratch, which barely depends
            // on slot count, so a quota-bound plan usually misses by a few MiB.
            // Refit once from the measured overshoot rather than giving up a
            // quarter of the table; then keep the original 3/4 steps.
            if (refused_bytes > quota) {
                const auto fewer = (refused_bytes - quota + per_expert - 1) / per_expert;
                for (std::size_t extra = 0; extra < 2 && capacity > fewer + extra
                        && capacity - fewer - extra >= static_cast<std::size_t>(minimum_capacity); ++extra) {
                    auto entry = plan(capacity - fewer - extra);
                    if (entry) return entry;
                }
            }
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

        bool split_device_headroom(ggml_backend_t backend, std::size_t bytes) {
            std::size_t free_bytes = 0, total_bytes = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(backend), &free_bytes, &total_bytes);
#if defined(TSG_GGML_TEST_HOOKS)
            free_bytes = std::min(free_bytes, g_test_available_bytes.load());
#endif
            return total_bytes && free_bytes > kDeviceHeadroom && bytes <= free_bytes - kDeviceHeadroom;
        }

        bool materialize_split_scratch(SplitOwner& owner) {
            CacheStageTimer timing(g_cache_timing, CacheStage::allocation);
            if (!owner.allocator || !owner.graph || !owner.charge) return false;
            std::size_t measured = 0;
            // This allocator is fresh, with no old physical buffer to free.
            ggml_gallocr_reserve_n_size(owner.allocator, owner.graph, nullptr, nullptr, &measured);
            if (measured != owner.bytes || ggml_gallocr_get_buffer_size(owner.allocator, 0) != 0) return false;
            if (!ggml_gallocr_reserve(owner.allocator, owner.graph)) return false;
#if defined(TSG_GGML_TEST_HOOKS)
            owner.physical_bytes = ggml_gallocr_get_buffer_size(owner.allocator, 0);
            g_test_physical_bytes.fetch_add(owner.physical_bytes);
            g_test_split_physical[owner.role].fetch_add(owner.physical_bytes);
#endif
            return ggml_gallocr_get_buffer_size(owner.allocator, 0) == owner.bytes
                && ggml_gallocr_alloc_graph(owner.allocator, owner.graph);
        }

        struct VmmBarrierContext { CacheEntry* entry = nullptr; bool all = false; };
        bool vmm_freeze_graphs(void* opaque) noexcept {
            auto* context = static_cast<VmmBarrierContext*>(opaque);
            if (!context) return false;
            const auto epoch = g_ggml_error_count.load(std::memory_order_acquire);
            try {
                drain_decode_prefetch(false);
                const auto freeze = [](CacheEntry& entry) {
                    sync_backend(entry.backend);
                    entry.down_overlap.reset();
                    if (entry.graph) entry.graph->uid = split_graph_uid();
                    for (auto& graph : entry.width_graphs) if (graph.n_nodes) graph.uid = split_graph_uid();
                    if (entry.wide_owner && entry.wide_owner->graph)
                        entry.wide_owner->graph->uid = split_graph_uid();
                };
                if (context->all) {
                    for (const auto& entry : g_entries) if (entry->vmm_planned) freeze(*entry);
                } else if (context->entry) freeze(*context->entry);
                return g_ggml_error_count.load(std::memory_order_acquire) == epoch;
            } catch (...) { return false; }
        }
        host_moe_vmm::WeightBarrier vmm_barrier(VmmBarrierContext& context) noexcept {
            return {vmm_freeze_graphs, &context};
        }
        bool retire_vmm_scratch_owner(SplitOwner& owner) {
            if (owner.cleanup_uncertain) return false;
            const auto epoch = g_ggml_error_count.load(std::memory_order_acquire);
            if (owner.allocator) {
                ggml_gallocr_free(owner.allocator); owner.allocator = nullptr;
#if defined(TSG_GGML_TEST_HOOKS)
                g_test_physical_bytes.fetch_sub(owner.physical_bytes);
                g_test_split_physical[owner.role].fetch_sub(owner.physical_bytes);
                owner.physical_bytes = 0;
#endif
            }
            if (owner.buffer) { ggml_backend_buffer_free(owner.buffer); owner.buffer = nullptr; }
            if (owner.ctx) { ggml_free(owner.ctx); owner.ctx = nullptr; owner.graph = nullptr; }
            if (g_ggml_error_count.load(std::memory_order_acquire) != epoch) { owner.cleanup_uncertain = true; return false; }
            if (owner.charge && !owner.charge->release_checked()) return false;
            owner.charge.reset(); owner.committed = false;
            return true;
        }
        bool vmm_retire_entry_checked(CacheEntry& entry) {
            if (!entry.vmm_planned) return true;
            if (entry.vmm_scratch_uncertain) return false;
            VmmBarrierContext context{&entry, false};
            if (!vmm_freeze_graphs(&context)) { vmm_fail(&entry); return false; }
            try {
                const auto epoch = g_ggml_error_count.load(std::memory_order_acquire);
                if (entry.vmm_pending_wide) {
                    if (!retire_vmm_scratch_owner(*entry.vmm_pending_wide)) { vmm_fail(&entry); return false; }
                    entry.vmm_pending_wide.reset();
                }
                if (entry.wide_owner) {
                    if (!retire_vmm_scratch_owner(*entry.wide_owner)) { vmm_fail(&entry); return false; }
                    entry.wide_owner.reset();
                }
                if (entry.allocator) {
                    ggml_gallocr_free(entry.allocator); entry.allocator = nullptr;
#if defined(TSG_GGML_TEST_HOOKS)
                    g_test_physical_bytes.fetch_sub(entry.test_physical_bytes);
                    g_test_split_physical[1].fetch_sub(entry.test_physical_bytes);
                    entry.test_physical_bytes = 0;
#endif
                }
                if (entry.ctx) { ggml_free(entry.ctx); entry.ctx = nullptr; entry.graph = nullptr; }
                entry.width_graphs.clear(); entry.width_inputs.clear(); entry.width_ids.clear();
                entry.width_routes.clear(); entry.width_outputs.clear();
                entry.input = entry.ids = entry.routes = entry.output = nullptr;
                entry.wide_activation = entry.wide_down = nullptr;
                if (g_ggml_error_count.load(std::memory_order_acquire) != epoch) {
                    entry.vmm_scratch_uncertain = true; vmm_fail(&entry); return false;
                }
                if (entry.shared_charge && !entry.shared_charge->release_checked()) { vmm_fail(&entry); return false; }
                entry.shared_charge.reset(); entry.narrow_committed = false;
                if (entry.vmm_weight.table && !host_moe_vmm::close_weight_table(g_vmm.arena,
                    entry.vmm_weight, vmm_barrier(context))) { vmm_fail(&entry); return false; }
                entry.vmm_weight = {}; entry.vmm_weight_bytes = 0;
                for (auto*& weight : entry.weight) weight = nullptr;
                return true;
            } catch (...) { vmm_fail(&entry); return false; }
        }
        bool vmm_bound_weights(const CacheEntry& entry, const host_moe_vmm::WeightLayoutPlan& plan) {
            std::array<std::uintptr_t, 3> starts{}, ends{};
            const auto type = ggml_backend_get_default_buffer_type(entry.backend);
            for (unsigned p = 0; p != 3; ++p) {
                const auto* weight = entry.weight[p];
                if (!weight || weight != host_moe_vmm::weight_tensor(g_vmm.arena, entry.vmm_weight, p)
                    || !weight->buffer || weight->view_src || weight->view_offs
                    || !weight->data || weight->type != entry.type[p]
                    || weight->ne[0] != (p == 2 ? entry.n_ff : entry.hidden)
                    || weight->ne[1] != (p == 2 ? entry.hidden : entry.n_ff)
                    || !ggml_is_contiguous(weight) || weight->ne[2] != static_cast<int64_t>(plan.capacity)
                    || weight->ne[3] != 1 || weight->nb[2] != entry.stride[p]
                    || ggml_backend_buffer_get_type(weight->buffer) != type
                    || ggml_backend_buffer_get_size(weight->buffer) != plan.projections[p].physical
                    || ggml_backend_buffer_get_base(weight->buffer) != weight->data
                    || ggml_nbytes(weight) > plan.projections[p].allocation) return false;
                starts[p] = reinterpret_cast<std::uintptr_t>(weight->data);
                if (plan.projections[p].physical > (std::numeric_limits<std::uintptr_t>::max)() - starts[p]) return false;
                ends[p] = starts[p] + plan.projections[p].physical;
                for (unsigned q = 0; q != p; ++q)
                    if (starts[p] < ends[q] && starts[q] < ends[p]) return false;
            }
            return true;
        }
        // Compare canonical graph roles rather than raw temporary addresses.
        // Every source edge (including views) must name the same role. Weight
        // capacity is the only permitted shape change; all compute geometry and
        // precision/op parameters remain byte-for-byte identical.
        bool vmm_same_graph(const CacheEntry& actual, const ggml_cgraph* a,
            const CacheEntry& target, const ggml_cgraph* b) {
            if (!a || !b || a->n_nodes != b->n_nodes || a->n_leafs != b->n_leafs) return false;
            const auto role = [](const CacheEntry& entry, const ggml_cgraph* graph, const ggml_tensor* tensor) {
                if (!tensor) return -1;
                for (int p = 0; p != 3; ++p) if (tensor == entry.weight[p]) return -2 - p;
                for (int i = 0; i != graph->n_nodes; ++i) if (graph->nodes[i] == tensor) return i;
                for (int i = 0; i != graph->n_leafs; ++i) if (graph->leafs[i] == tensor) return graph->n_nodes + i;
                return (std::numeric_limits<int>::min)();
            };
            const auto equal = [&](const ggml_tensor* x, const ggml_tensor* y) {
                const int xr = role(actual, a, x), yr = role(target, b, y);
                if (xr != yr || xr == (std::numeric_limits<int>::min)()) return false;
                if (xr <= -2 && xr >= -4) return true;
                if (x->op != y->op || x->type != y->type || x->flags != y->flags
                    || std::memcmp(x->ne, y->ne, sizeof(x->ne)) || std::memcmp(x->nb, y->nb, sizeof(x->nb))
                    || std::memcmp(x->op_params, y->op_params, sizeof(x->op_params))
                    || x->view_offs != y->view_offs
                    || role(actual, a, x->view_src) != role(target, b, y->view_src)) return false;
                for (unsigned p = 0; p != GGML_MAX_SRC; ++p) {
                    const int left = role(actual, a, x->src[p]), right = role(target, b, y->src[p]);
                    if (left != right || left == (std::numeric_limits<int>::min)()) return false;
                }
                return true;
            };
            for (int i = 0; i != a->n_nodes; ++i) if (!equal(a->nodes[i], b->nodes[i])) return false;
            for (int i = 0; i != a->n_leafs; ++i) if (!equal(a->leafs[i], b->leafs[i])) return false;
            return true;
        }
        std::unique_ptr<CacheEntry> vmm_resize_proof(const CacheEntry& entry, std::size_t capacity,
            bool retire_wide = false) {
            HostMoeSegment hm{};
            hm.layer = entry.layer; hm.hidden = entry.hidden; hm.n_ff = entry.n_ff;
            hm.n_used = entry.n_used; hm.num_experts = entry.num_experts;
            hm.pooled_model_experts = entry.pool_expert_extent; hm.swiglu_clamp = entry.swiglu_clamp;
            hm.gate_type = entry.type[0]; hm.up_type = entry.type[1]; hm.down_type = entry.type[2];
            hm.gate_data = const_cast<void*>(entry.source[0]); hm.up_data = const_cast<void*>(entry.source[1]);
            hm.down_data = const_cast<void*>(entry.source[2]);
            auto shadow = create_entry(hm, entry.stride, entry.backend, (std::numeric_limits<std::size_t>::max)(),
                capacity, entry.graph_rows, static_cast<int>(capacity), retire_wide ? 0 : entry.bulk_rows, 0, true);
            if (!shadow || shadow->capacity != static_cast<int>(capacity) || !shadow->vmm_planned
                || shadow->narrow_bytes != entry.narrow_bytes
                || (!retire_wide && static_cast<bool>(shadow->wide_owner) != static_cast<bool>(entry.wide_owner))
                || (retire_wide && shadow->wide_owner)
                || !vmm_same_graph(entry, entry.graph, *shadow, shadow->graph)
                || ggml_gallocr_get_buffer_size(entry.allocator, 0) != entry.narrow_bytes - kWorkspaceAllowance)
                throw std::runtime_error("VMM resize cannot prove unchanged narrow graph/scratch geometry");
            if (!retire_wide && entry.wide_owner && (entry.wide_owner->bytes != shadow->wide_owner->bytes
                || !vmm_same_graph(entry, entry.wide_owner->graph, *shadow, shadow->wide_owner->graph)
                || ggml_gallocr_get_buffer_size(entry.wide_owner->allocator, 0) != entry.wide_owner->bytes))
                throw std::runtime_error("VMM resize cannot prove unchanged wide graph/scratch geometry");
            return shadow;
        }
        bool bind_vmm_split_graphs(CacheEntry& entry) {
            const auto expected_narrow = entry.narrow_bytes;
            const auto expected_wide = entry.wide_owner ? entry.wide_owner->bytes : 0;
            if (ggml_gallocr_get_buffer_size(entry.allocator, 0) != 0) return false;
            ggml_gallocr_free(entry.allocator); entry.allocator = nullptr;
            ggml_free(entry.ctx); entry.ctx = nullptr; entry.graph = nullptr;
            for (unsigned p = 0; p != 3; ++p) entry.weight[p] = host_moe_vmm::weight_tensor(g_vmm.arena, entry.vmm_weight, p);
            if (!vmm_bound_weights(entry, vmm_layout(entry, entry.capacity))) return false;
            entry.ctx = split_context(entry.graph_rows);
            if (!entry.ctx) return false;
            entry.graph = ggml_new_graph_custom(entry.ctx, 256 * static_cast<std::size_t>(entry.graph_rows), false);
            entry.graph->uid = split_graph_uid();
            for (int i = 0; i != entry.graph_rows; ++i) {
                SplitVariant variant;
                if (!emit_split_variant(entry, entry.ctx, entry.graph, i + 1, false, variant)) return false;
                publish_split_variant(entry, i, variant);
            }
            if (split_add(measure_split_scratch(entry, entry.graph), kWorkspaceAllowance) != expected_narrow) return false;
            entry.allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(entry.backend));
            if (!entry.allocator) return false;
            if (entry.bulk_rows > 0) {
                auto wide = plan_split_wide(entry, entry.bulk_rows);
                if (!wide.owner || wide.owner->bytes != expected_wide) return false;
                wide.owner->charge = std::move(entry.wide_owner->charge);
                publish_split_variant(entry, entry.graph_rows, wide.variant);
                entry.wide_owner = std::move(wide.owner);
            }
            entry.input = entry.width_inputs[0]; entry.ids = entry.width_ids[0];
            entry.routes = entry.width_routes[0]; entry.output = entry.width_outputs[0];
            return true;
        }
        Admission allocate_vmm_entry(CacheEntry& entry) {
            if (!vmm_active() || !g_vmm.source_certified || !entry.vmm_planned
                || !entry.vmm_unpublished || !split_device_headroom(entry.backend, entry.bytes)) return Admission::refused;
            const auto epoch = g_ggml_error_count.load(std::memory_order_acquire);
            entry.shared_charge = split_reserve(1, &entry, entry.narrow_bytes);
            if (!entry.shared_charge || !entry.shared_charge->is_accounted()) return Admission::refused;
            if (entry.wide_owner) {
                entry.wide_owner->charge = split_reserve(2, entry.wide_owner.get(), entry.wide_owner->bytes);
                if (!entry.wide_owner->charge || !entry.wide_owner->charge->is_accounted()) return Admission::refused;
            }
            host_moe_vmm::WeightSpec spec{vmm_shapes(entry), entry.vmm_maximum_capacity,
                static_cast<std::size_t>(entry.capacity), &entry};
            const auto status = host_moe_vmm::create_weight_table(g_vmm.arena,
                static_cast<unsigned>(vmm_group_index(group_key(entry))), spec, entry.vmm_weight);
            if (status == host_moe_vmm::WeightStatus::refused || status == host_moe_vmm::WeightStatus::unsupported)
                return Admission::refused;
            if (status != host_moe_vmm::WeightStatus::ready || !bind_vmm_split_graphs(entry)) return Admission::failed;
            std::size_t measured = 0;
            ggml_gallocr_reserve_n_size(entry.allocator, entry.graph, nullptr, nullptr, &measured);
            if (measured != entry.narrow_bytes - kWorkspaceAllowance
                || !ggml_gallocr_reserve(entry.allocator, entry.graph)) return Admission::failed;
#if defined(TSG_GGML_TEST_HOOKS)
            entry.test_physical_bytes = ggml_gallocr_get_buffer_size(entry.allocator, 0);
            g_test_physical_bytes.fetch_add(entry.test_physical_bytes); g_test_split_physical[1].fetch_add(entry.test_physical_bytes);
#endif
            if (ggml_gallocr_get_buffer_size(entry.allocator, 0) != measured
                || !ggml_gallocr_alloc_graph(entry.allocator, entry.graph)) return Admission::failed;
            if (entry.wide_owner && !materialize_split_scratch(*entry.wide_owner)) return Admission::failed;
            sync_backend(entry.backend);
            if (g_ggml_error_count.load(std::memory_order_acquire) != epoch
                || g_backend_compute_failed.load(std::memory_order_acquire)
                || !split_commit(entry.shared_charge, 1, &entry, measured, entry.narrow_bytes)) return Admission::failed;
            entry.narrow_committed = true;
            if (entry.wide_owner && !split_commit(entry.wide_owner->charge, 2, entry.wide_owner.get(),
                entry.wide_owner->bytes, entry.wide_owner->bytes)) return Admission::failed;
            if (entry.wide_owner) entry.wide_owner->committed = true;
            entry.expert_for_slot.assign(entry.capacity, -1); entry.source_for_slot.resize(entry.capacity);
            entry.last_used.assign(entry.capacity, 0); entry.prefetched.assign(entry.capacity, 0);
            const auto usage = host_moe_vmm::weight_usage(g_vmm.arena, entry.vmm_weight);
            return usage.usable && usage.actual_charge_certificate && !usage.ticket_pending && !usage.quarantine
                && usage.mapped == entry.vmm_weight_bytes ? Admission::admitted : Admission::failed;
        }

        void vmm_publish_capacity(CacheEntry& entry, const CacheEntry& shadow) {
            const auto count = static_cast<std::size_t>(shadow.capacity);
            entry.expert_for_slot.resize(count, -1); entry.source_for_slot.resize(count);
            entry.last_used.resize(count, 0); entry.prefetched.resize(count, 0);
            entry.capacity = shadow.capacity; entry.vmm_weight_bytes = shadow.vmm_weight_bytes;
            entry.weight_owner->bytes = shadow.weight_owner->bytes; // Logical packed metadata, never physical charge.
            entry.eviction_policy.configure(entry.pool_expert_extent, entry.capacity, false);
            ++entry.vmm_mutation_generation;
            if (!vmm_bound_weights(entry, vmm_layout(entry, count)))
                throw std::runtime_error("VMM resized stable tensor binding is inconsistent");
        }
        // Request-boundary Trim or a selected call's mandatory room request.
        // False is a PRE-mutation refusal: callers may use the old checked whole
        // eviction policy. After any graph withdrawal, tag write or physical
        // retirement, failure poisons the scope and throws; no cold fallback.
        // The cache gate excludes new producers/source rebinding throughout.
        bool compact_vmm_pressure_locked(std::size_t target, const CacheEntry* keep) {
            if (g_reserved <= target) return true;
            if (!target || !vmm_active() || !g_phase_plan || !g_vmm.source_certified
                || !g_vmm.control || !g_vmm.certified_backend) return false;
            const auto pressure_epoch = g_ggml_error_count.load(std::memory_order_acquire);
            if (g_backend_compute_failed.load(std::memory_order_acquire)) {
                vmm_fail(); throw std::runtime_error("VMM pressure cannot bypass a failed backend");
            }
            std::array<CacheEntry*, 3> entries{};
            std::array<std::unique_ptr<CacheEntry>, 3> shadows;
            std::array<std::size_t, 3> floors{}, weights{}, capacities{}, before_cap{}, before_weight{};
            std::size_t fixed = 0, floor_total = 0, weight_total = 0, wide_total = 0;
            unsigned count = 0, seen = 0;
            std::size_t accounted = 0;
            // Every existing held byte is either fixed (the active owner) or
            // represented exactly once by a published eligible pressure plan.
            for (const auto& owner : g_entries) {
                auto& entry = *owner;
                const auto group = vmm_group_index(group_key(entry));
                if (!entry.vmm_planned || entry.vmm_unpublished || entry.vmm_failed
                    || entry.vmm_pending_wide || entry.backend != g_vmm.certified_backend
                    || group < 0 || (seen & (1u << group))) return false;
                seen |= 1u << group;
                const auto usage = host_moe_vmm::weight_usage(g_vmm.arena, entry.vmm_weight);
                if (!usage.usable || !usage.actual_charge_certificate || usage.ticket_pending || usage.quarantine
                    || usage.mapped != entry.vmm_weight_bytes || !entry.narrow_committed
                    || !entry.shared_charge || !entry.shared_charge->is_committed()
                    || (entry.wide_owner && (!entry.wide_owner->committed || !entry.wide_owner->charge
                        || !entry.wide_owner->charge->is_committed()))) {
                    vmm_fail(&entry); throw std::runtime_error("VMM pressure cannot use an uncertified owner");
                }
                auto actual = split_add(split_add(usage.ticket_pending, usage.ticket_committed),
                    entry.shared_charge->reserved_bytes());
                if (entry.wide_owner) actual = split_add(actual, entry.wide_owner->charge->reserved_bytes());
                if (actual != entry.bytes) {
                    vmm_fail(&entry); throw std::runtime_error("VMM pressure private owner accounting is inconsistent");
                }
                accounted = split_add(accounted, actual);
                if (&entry == keep) { fixed = split_add(fixed, actual); continue; }
                entries[group] = &entry; ++count;
                const auto minimum = static_cast<std::size_t>(entry.n_used);
                if (entry.capacity < entry.n_used || entry.graph_rows != TSG_PRECISION_DECODE_COLUMNS)
                    return false;
                floors[group] = split_add(entry.narrow_bytes, vmm_layout(entry, minimum).total);
                if (floors[group] < g_phase_plan->floor(group_key(entry), false)) return false;
                for (const auto& phase : g_phase_plan->groups()) if (phase.key == group_key(entry))
                    weights[group] = phase.weight;
                if (!weights[group]) return false;
                floor_total = split_add(floor_total, floors[group]);
                weight_total = split_add(weight_total, weights[group]);
                if (entry.wide_owner) wide_total = split_add(wide_total, entry.wide_owner->bytes);
                before_cap[group] = entry.capacity; before_weight[group] = entry.vmm_weight_bytes;
            }
            if (accounted != g_reserved) {
                vmm_fail(); throw std::runtime_error("VMM pressure aggregate accounting is inconsistent");
            }
            if (!count || fixed > target || floor_total > target - fixed) return false;
            const auto before = g_reserved;
            const bool scratch_only = wide_total >= before - target;
            std::size_t predicted = fixed;
            auto spare = target - fixed - floor_total;
            auto remaining_weight = weight_total;
            // Only metadata may allocate here (within the existing charged
            // three-live/three-shadow bound). No payload or ticket is created.
            // A pure plan refusal may still use checked whole-entry eviction.
            try {
                for (unsigned group = 0; group != 3; ++group) if (entries[group]) {
                    auto& entry = *entries[group];
                    std::size_t share = 0;
                    if (remaining_weight == weights[group]) share = spare;
                    else share = (std::min)(spare, static_cast<std::size_t>(
                        static_cast<long double>(spare) * weights[group] / remaining_weight));
                    spare -= share; remaining_weight -= weights[group];
                    const auto quota = split_add(floors[group], share);
                    capacities[group] = scratch_only ? static_cast<std::size_t>(entry.capacity)
                        : host_moe_vmm_capacity(quota, entry.narrow_bytes, static_cast<std::size_t>(entry.n_used),
                            static_cast<std::size_t>(entry.capacity),
                            [&](std::size_t n) { return vmm_layout(entry, n).total; });
                    if (!capacities[group] || capacities[group] > static_cast<std::size_t>(entry.capacity)) return false;
                    g_vmm.control->slots[group].prepare(entry.capacity, capacities[group],
                        entry.expert_for_slot.data(), entry.last_used.data());
                    shadows[group] = vmm_resize_proof(entry, capacities[group], true);
                    predicted = split_add(predicted, split_add(entry.narrow_bytes, shadows[group]->vmm_weight_bytes));
                }
            } catch (...) {
                if (g_ggml_error_count.load(std::memory_order_acquire) != pressure_epoch
                    || g_backend_compute_failed.load(std::memory_order_acquire)) { vmm_fail(); throw; }
                return false;
            }
            if (g_ggml_error_count.load(std::memory_order_acquire) != pressure_epoch
                || g_backend_compute_failed.load(std::memory_order_acquire)) {
                vmm_fail(); throw std::runtime_error("VMM pressure planning changed its error epoch");
            }
            if (predicted > target || predicted >= before) return false;
            const auto epoch = pressure_epoch;
            if (g_vmm.pressure_mutation_serial == (std::numeric_limits<std::uint64_t>::max)()) {
                vmm_fail(); throw std::runtime_error("VMM pressure mutation generation cannot wrap");
            }
            // Even a completed pressure transaction remains a published
            // mutation for its enclosing selected call. Later admission may
            // not retry a smaller geometry after any such mutation.
            ++g_vmm.pressure_mutation_serial; // Before the first graph UID withdrawal.
            std::size_t wide_refunded = 0, weight_refunded = 0, retained_hot = 0, moved = 0, discarded = 0;
            try {
                VmmBarrierContext all{nullptr, true};
                if (!vmm_freeze_graphs(&all)) throw std::runtime_error("VMM pressure global drain failed");
                for (unsigned group = 0; group != 3; ++group) if (entries[group]) {
                    auto& entry = *entries[group];
                    if (entry.wide_owner) {
                        const auto bytes = entry.wide_owner->bytes;
                        // Withdraw all views BEFORE their metadata/storage is
                        // freed. A failed ACK keeps this real owner shell/ticket
                        // in the entry, with no executable dangling wide view.
                        entry.down_overlap.reset(); entry.bulk_rows = 0;
                        entry.wide_activation = entry.wide_down = nullptr;
                        entry.width_inputs[entry.graph_rows] = nullptr; entry.width_ids[entry.graph_rows] = nullptr;
                        entry.width_routes[entry.graph_rows] = nullptr; entry.width_outputs[entry.graph_rows] = nullptr;
                        entry.width_graphs[entry.graph_rows] = {}; ++entry.wide_generation;
                        if (!retire_vmm_scratch_owner(*entry.wide_owner))
                            throw std::runtime_error("VMM pressure wide scratch cleanup remains owned");
                        entry.wide_owner.reset(); refresh_vmm_accounting();
                        wide_refunded = split_add(wide_refunded, bytes);
#if defined(TSG_GGML_TEST_HOOKS)
                        if (bytes && test_failure(34)) {
                            g_test_vmm_mutation_failures.fetch_add(1);
                            throw std::runtime_error("Injected VMM pressure failure after checked wide retirement");
                        }
#endif
                    }
                    auto& slots = g_vmm.control->slots[group];
                    retained_hot = split_add(retained_hot, slots.kept_hot());
                    discarded = split_add(discarded, slots.discarded_count());
                    if (capacities[group] == static_cast<std::size_t>(entry.capacity)) continue;
                    for (std::size_t i = 0; i != slots.discarded_count(); ++i)
                        entry.invalidate_slot(static_cast<int>(slots.discarded(i)));
                    ++entry.vmm_mutation_generation;
                    for (std::size_t i = 0; i != slots.move_count(); ++i) {
                        const auto move = slots.move(i);
                        if (host_moe_vmm::copy_weight_slot(g_vmm.arena, entry.vmm_weight, move.from, move.to,
                            &entry, entry.expert_for_slot.data(), entry.expert_for_slot.size(), vmm_barrier(all))
                            != host_moe_vmm::WeightStatus::ready)
                            throw std::runtime_error("VMM pressure survivor compaction failed");
                        entry.expert_for_slot[move.to] = entry.expert_for_slot[move.from];
                        entry.source_for_slot[move.to] = entry.source_for_slot[move.from];
                        entry.last_used[move.to] = entry.last_used[move.from];
                        entry.prefetched[move.to] = entry.prefetched[move.from];
                        entry.invalidate_slot(static_cast<int>(move.from)); ++entry.vmm_mutation_generation; ++moved;
                    }
                    const auto status = host_moe_vmm::shrink_weight_table_empty_tail(g_vmm.arena, entry.vmm_weight,
                        capacities[group], &entry, entry.vmm_mutation_generation, entry.expert_for_slot.data(),
                        entry.expert_for_slot.size(), vmm_barrier(all));
                    refresh_vmm_accounting();
                    if (status != host_moe_vmm::WeightStatus::ready)
                        throw std::runtime_error("VMM pressure tail shrink refused after mutation");
                    vmm_publish_capacity(entry, *shadows[group]);
                    weight_refunded = split_add(weight_refunded, before_weight[group] - entry.vmm_weight_bytes);
#if defined(TSG_GGML_TEST_HOOKS)
                    if (before_weight[group] > entry.vmm_weight_bytes && test_failure(34)) {
                        g_test_vmm_mutation_failures.fetch_add(1);
                        throw std::runtime_error("Injected VMM pressure failure after checked weight retirement");
                    }
#endif
                }
                if (!vmm_freeze_graphs(&all) || g_ggml_error_count.load(std::memory_order_acquire) != epoch
                    || g_backend_compute_failed.load(std::memory_order_acquire))
                    throw std::runtime_error("VMM pressure final fence failed");
                refresh_vmm_accounting();
                for (const auto& owner : g_entries) {
                    const auto& entry = *owner;
                    const auto usage = host_moe_vmm::weight_usage(g_vmm.arena, entry.vmm_weight);
                    if (!usage.usable || !usage.actual_charge_certificate || usage.ticket_pending || usage.quarantine
                        || usage.mapped != entry.vmm_weight_bytes || !entry.shared_charge
                        || !entry.shared_charge->is_accounted() || !entry.shared_charge->is_committed()
                        || (entry.wide_owner && (!entry.wide_owner->charge || !entry.wide_owner->committed
                            || !entry.wide_owner->charge->is_accounted() || !entry.wide_owner->charge->is_committed())))
                        throw std::runtime_error("VMM pressure final owner certificate failed");
                }
                if (g_reserved != predicted || g_reserved > target
                    || before - g_reserved != split_add(wide_refunded, weight_refunded))
                    throw std::runtime_error("VMM pressure actual ownership does not meet its target");
                std::fprintf(stderr, "[HOSTMOE-VMM-PRESSURE] profile=1 mutation=%llu outcome=completed target=%zu "
                    "owned_before=%zu owned_after=%zu wide_refunded_bytes=%zu weight_refunded_bytes=%zu "
                    "retained_hot=%zu moved_slots=%zu discarded_hot=%zu active_owner_excluded=%d "
                    "g3_cap_before=%zu g3_cap_after=%zu g4_cap_before=%zu g4_cap_after=%zu "
                    "g11_cap_before=%zu g11_cap_after=%zu pending_bytes=0 quarantine_bytes=0 certificate=1 "
                    "scope=published_gpu_payload_shrink_not_new_capacity_or_physical_free\n",
                    static_cast<unsigned long long>(g_vmm.pressure_mutation_serial),
                    target, before, g_reserved, wide_refunded, weight_refunded, retained_hot, moved, discarded, keep ? 1 : 0,
                    before_cap[0], entries[0] ? capacities[0] : 0, before_cap[1], entries[1] ? capacities[1] : 0,
                    before_cap[2], entries[2] ? capacities[2] : 0);
                return true;
            } catch (...) {
                vmm_fail(); refresh_vmm_accounting(); throw;
            }
        }
        // Caller has validated the complete current call and holds the cache
        // gate. No forecast/maximum ceiling is converted into a live grant here.
        void reshape_vmm_decode(const HostMoeSegment& hm, const std::int32_t* ids, std::size_t budget,
            ggml_backend_t backend) {
            if (!vmm_active() || !g_vmm.control || !g_phase_plan || !g_vmm.source_certified)
                throw std::runtime_error("VMM decode reshape lacks its configured owners");
            std::array<CacheEntry*, 3> entries{};
            for (const auto& owner : g_entries) {
                auto& entry = *owner;
                const int group = vmm_group_index(group_key(entry));
                if (!entry.vmm_planned || entry.vmm_unpublished || entry.vmm_failed || entry.backend != backend
                    || group < 0 || entries[group]) throw std::runtime_error("VMM phase contains an unsupported table");
                entries[group] = &entry;
            }
            // An incomplete wide working set has no three-way resident pool to
            // reshape. Later cold admission still protects every absent floor.
            if (!entries[0] || !entries[1] || !entries[2]) return;
            std::array<std::unique_ptr<CacheEntry>, 3> shadows;
            std::array<std::size_t, 3> before_cap{}, before_bytes{}, after_bytes{};
            std::size_t planned_total = 0, retained_hot = 0, moved = 0, discarded = 0;
            bool changed = false;
            for (unsigned group = 0; group != 3; ++group) {
                auto& entry = *entries[group];
                const auto usage = host_moe_vmm::weight_usage(g_vmm.arena, entry.vmm_weight);
                if (!usage.usable || !usage.actual_charge_certificate || usage.ticket_pending || usage.quarantine
                    || usage.mapped != entry.vmm_weight_bytes || !entry.narrow_committed
                    || !entry.shared_charge || !entry.shared_charge->is_committed()
                    || (entry.wide_owner && (!entry.wide_owner->committed || !entry.wide_owner->charge
                        || !entry.wide_owner->charge->is_committed())))
                    throw std::runtime_error("VMM reshape requires a complete committed source generation");
                before_cap[group] = entry.capacity; before_bytes[group] = usage.mapped;
                const auto scratch = split_add(entry.narrow_bytes, entry.wide_owner ? entry.wide_owner->bytes : 0);
                const auto quota = g_phase_plan->quota(group_key(entry), budget, false);
                auto& protect = g_vmm.control->protect[group];
                std::fill(protect.begin(), protect.end(), 0);
                const std::array<const void*, 3> current{hm.gate_data, hm.up_data, hm.down_data};
                if (entry.matches(hm, backend)) for (int slot = 0; slot != entry.capacity; ++slot)
                    for (int k = 0; k != hm.n_used; ++k)
                        if (entry.source_for_slot[slot].matches(current, ids[k])) protect[slot] = 1;
                const auto capacity = host_moe_vmm_capacity(quota, scratch, static_cast<std::size_t>(entry.n_used),
                    entry.vmm_maximum_capacity, [&](std::size_t count) { return vmm_layout(entry, count).total; });
                if (!capacity) throw std::runtime_error("Rounded VMM weighted share cannot cover its mandatory floor");
                g_vmm.control->slots[group].prepare(entry.capacity, capacity,
                    entry.expert_for_slot.data(), entry.last_used.data(), protect.data());
                retained_hot += g_vmm.control->slots[group].kept_hot();
                moved += g_vmm.control->slots[group].move_count();
                discarded += g_vmm.control->slots[group].discarded_count();
                shadows[group] = vmm_resize_proof(entry, capacity);
                planned_total = split_add(planned_total, split_add(scratch, shadows[group]->vmm_weight_bytes));
                changed = changed || capacity != static_cast<std::size_t>(entry.capacity);
                // All metadata expansion happens before the first tag or GPU
                // mutation. Resize after commit is nonallocating for these PODs.
                entry.expert_for_slot.reserve(capacity); entry.source_for_slot.reserve(capacity);
                entry.last_used.reserve(capacity); entry.prefetched.reserve(capacity);
            }
            if (!changed) return;
            if (planned_total > budget || g_reserved > budget)
                throw std::runtime_error("VMM decode reshape cannot spend unadmitted capacity");
            const auto owned_before = g_reserved;
            const auto epoch = g_ggml_error_count.load(std::memory_order_acquire);
            ++g_vmm.attempts; const auto phase = ++g_vmm.phase;
            std::size_t refunded = 0, grown = 0;
            try {
                VmmBarrierContext all{nullptr, true};
                if (!vmm_freeze_graphs(&all)) throw std::runtime_error("VMM phase drain failed");
                for (unsigned group = 0; group != 3; ++group) {
                    auto& entry = *entries[group]; auto& shadow = *shadows[group];
                    if (shadow.capacity >= entry.capacity) continue;
                    auto& slots = g_vmm.control->slots[group];
                    for (std::size_t i = 0; i != slots.discarded_count(); ++i)
                        entry.invalidate_slot(static_cast<int>(slots.discarded(i)));
                    ++entry.vmm_mutation_generation;
                    for (std::size_t i = 0; i != slots.move_count(); ++i) {
                        const auto move = slots.move(i);
                        if (host_moe_vmm::copy_weight_slot(g_vmm.arena, entry.vmm_weight, move.from, move.to,
                            &entry, entry.expert_for_slot.data(), entry.expert_for_slot.size(), vmm_barrier(all))
                            != host_moe_vmm::WeightStatus::ready)
                            throw std::runtime_error("VMM survivor compaction failed");
                        entry.expert_for_slot[move.to] = entry.expert_for_slot[move.from];
                        entry.source_for_slot[move.to] = entry.source_for_slot[move.from];
                        entry.last_used[move.to] = entry.last_used[move.from];
                        entry.prefetched[move.to] = entry.prefetched[move.from];
                        entry.invalidate_slot(static_cast<int>(move.from)); ++entry.vmm_mutation_generation;
                    }
                    const auto status = host_moe_vmm::shrink_weight_table_empty_tail(g_vmm.arena, entry.vmm_weight,
                        shadow.capacity, &entry, entry.vmm_mutation_generation, entry.expert_for_slot.data(),
                        entry.expert_for_slot.size(), vmm_barrier(all));
                    refresh_vmm_accounting();
                    if (status != host_moe_vmm::WeightStatus::ready)
                        throw std::runtime_error("VMM donor shrink refused after phase mutation");
                    vmm_publish_capacity(entry, shadow);
                    refunded = split_add(refunded, before_bytes[group] - entry.vmm_weight_bytes);
#if defined(TSG_GGML_TEST_HOOKS)
                    if (before_bytes[group] > entry.vmm_weight_bytes && test_failure(33)) {
                        g_test_vmm_mutation_failures.fetch_add(1);
                        throw std::runtime_error("Injected VMM failure after a physically refunded donor");
                    }
#endif
                }
                for (unsigned group = 0; group != 3; ++group) {
                    auto& entry = *entries[group]; auto& shadow = *shadows[group];
                    if (shadow.capacity <= entry.capacity) continue;
                    const auto delta = shadow.vmm_weight_bytes - entry.vmm_weight_bytes;
                    if (g_reserved > budget || delta > budget - g_reserved
                        || (delta && !split_device_headroom(backend, delta)))
                        throw std::runtime_error("VMM recipient delta lacks admitted physical headroom");
                    const auto status = host_moe_vmm::grow_weight_table(g_vmm.arena, entry.vmm_weight,
                        shadow.capacity, vmm_barrier(all));
                    refresh_vmm_accounting();
                    if (status != host_moe_vmm::WeightStatus::ready)
                        throw std::runtime_error("VMM recipient growth refused after phase mutation");
                    vmm_publish_capacity(entry, shadow); grown = split_add(grown, delta);
                }
                if (!vmm_freeze_graphs(&all) || g_ggml_error_count.load(std::memory_order_acquire) != epoch
                    || g_backend_compute_failed.load(std::memory_order_acquire)) throw std::runtime_error("VMM phase final fence failed");
                refresh_vmm_accounting();
                for (unsigned group = 0; group != 3; ++group) {
                    const auto usage = host_moe_vmm::weight_usage(g_vmm.arena, entries[group]->vmm_weight);
                    if (!usage.usable || !usage.actual_charge_certificate || usage.ticket_pending || usage.quarantine
                        || usage.mapped != shadows[group]->vmm_weight_bytes)
                        throw std::runtime_error("VMM phase publication certificate failed");
                    after_bytes[group] = usage.mapped;
                }
                if (g_reserved != owned_before - refunded + grown || g_reserved > budget)
                    throw std::logic_error("VMM phase changed unrelated owned credit");
                ++g_vmm.completed;
                std::fprintf(stderr, "[HOSTMOE-VMM-RESHAPE] profile=1 phase=%llu outcome=completed "
                    "g3_cap_before=%zu g3_cap_after=%d g4_cap_before=%zu g4_cap_after=%d g11_cap_before=%zu g11_cap_after=%d "
                    "g3_bytes_before=%zu g3_bytes_after=%zu g4_bytes_before=%zu g4_bytes_after=%zu g11_bytes_before=%zu g11_bytes_after=%zu "
                    "refunded_bytes=%zu grown_bytes=%zu retained_hot=%zu moved_slots=%zu discarded_hot=%zu "
                    "budget=%zu owned_before=%zu owned_after=%zu pending_bytes=0 quarantine_bytes=0 certificate=1 "
                    "scope=published_gpu_payload_not_new_capacity_or_physical_free\n",
                    static_cast<unsigned long long>(phase), before_cap[0], entries[0]->capacity, before_cap[1], entries[1]->capacity,
                    before_cap[2], entries[2]->capacity, before_bytes[0], after_bytes[0], before_bytes[1], after_bytes[1], before_bytes[2], after_bytes[2],
                    refunded, grown, retained_hot, moved, discarded, budget, owned_before, g_reserved);
            } catch (...) {
                ++g_vmm.failures; vmm_fail();
                for (auto* entry : entries) entry->vmm_failed = true;
                refresh_vmm_accounting();
                std::fprintf(stderr, "[HOSTMOE-VMM-RESHAPE] profile=1 phase=%llu outcome=failed\n", static_cast<unsigned long long>(phase));
                throw;
            }
        }

        Admission allocate_split_entry(CacheEntry& entry) {
            if (entry.vmm_planned) throw std::logic_error("VMM candidate must be retained before physical allocation");
            if (!entry.weight_owner || !entry.split_weights || !split_device_headroom(entry.backend, entry.bytes))
                return Admission::refused;
            auto& weights = *entry.weight_owner;
            // Reserve every nonzero device owner BEFORE the first physical allocation.
            weights.charge = split_reserve(0, &weights, weights.bytes);
            if (!weights.charge) return Admission::refused;
            entry.shared_charge = split_reserve(1, &entry, entry.narrow_bytes);
            if (!entry.shared_charge) return Admission::refused;
            if (entry.wide_owner) {
                entry.wide_owner->charge = split_reserve(2, entry.wide_owner.get(), entry.wide_owner->bytes);
                if (!entry.wide_owner->charge) return Admission::refused;
            }
#if defined(TSG_GGML_TEST_HOOKS)
            if (test_failure(1)) return Admission::failed;
#endif
            const auto errors_before = g_ggml_error_count.load(std::memory_order_acquire);
            weights.buffer = ggml_backend_alloc_ctx_tensors_from_buft(weights.ctx,
                ggml_backend_get_default_buffer_type(entry.backend));
            if (!weights.buffer) return Admission::failed;
            // Preserve the original galloc COMPUTE buffer policy. Upstream CUDA
            // selects padding/quantized execution paths using this real usage.
            ggml_backend_buffer_set_usage(weights.buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
#if defined(TSG_GGML_TEST_HOOKS)
            weights.physical_bytes = ggml_backend_buffer_get_size(weights.buffer);
            g_test_physical_bytes.fetch_add(weights.physical_bytes);
            g_test_split_physical[0].fetch_add(weights.physical_bytes);
            if (test_failure(11)) return Admission::failed;
#endif
            if (ggml_backend_buffer_get_size(weights.buffer) != weights.bytes) return Admission::failed;
            std::size_t measured = 0;
            ggml_gallocr_reserve_n_size(entry.allocator, entry.graph, nullptr, nullptr, &measured);
            if (measured != entry.narrow_bytes - kWorkspaceAllowance
                || ggml_gallocr_get_buffer_size(entry.allocator, 0) != 0) return Admission::failed;
            if (!ggml_gallocr_reserve(entry.allocator, entry.graph)) return Admission::failed;
#if defined(TSG_GGML_TEST_HOOKS)
            entry.test_physical_bytes = ggml_gallocr_get_buffer_size(entry.allocator, 0);
            g_test_physical_bytes.fetch_add(entry.test_physical_bytes);
            g_test_split_physical[1].fetch_add(entry.test_physical_bytes);
            if (test_failure(12)) return Admission::failed;
#endif
            if (ggml_gallocr_get_buffer_size(entry.allocator, 0) != measured
                || !ggml_gallocr_alloc_graph(entry.allocator, entry.graph)) return Admission::failed;
            if (entry.wide_owner && !materialize_split_scratch(*entry.wide_owner)) return Admission::failed;
#if defined(TSG_GGML_TEST_HOOKS)
            if (test_failure(13) || test_failure(2)) return Admission::failed;
#endif
            for (auto* weight : entry.weight) ggml_backend_tensor_memset(weight, 0, 0, ggml_nbytes(weight));
            entry.expert_for_slot.assign(entry.capacity, -1);
            entry.source_for_slot.resize(entry.capacity);
            entry.last_used.assign(entry.capacity, 0);
            sync_backend(entry.backend);
            if (g_ggml_error_count.load(std::memory_order_acquire) != errors_before
                || g_backend_compute_failed.load(std::memory_order_acquire)) return Admission::failed;
#if defined(TSG_GGML_TEST_HOOKS)
            if (test_failure(14)) return Admission::failed;
#endif
            if (!split_commit(weights.charge, 0, &weights, ggml_backend_buffer_get_size(weights.buffer), weights.bytes)) return Admission::failed;
            weights.committed = true;
#if defined(TSG_GGML_TEST_HOOKS)
            if (test_failure(15)) return Admission::failed;
#endif
            if (!split_commit(entry.shared_charge, 1, &entry,
                ggml_gallocr_get_buffer_size(entry.allocator, 0), entry.narrow_bytes)) return Admission::failed;
            entry.narrow_committed = true;
            if (entry.wide_owner && !split_commit(entry.wide_owner->charge, 2, entry.wide_owner.get(),
                ggml_gallocr_get_buffer_size(entry.wide_owner->allocator, 0), entry.wide_owner->bytes)) return Admission::failed;
            if (entry.wide_owner) entry.wide_owner->committed = true;
#if defined(TSG_GGML_TEST_HOOKS)
            if (test_failure(16) || test_failure(3)) return Admission::failed;
#endif
            return Admission::admitted;
        }

        void grow_vmm_required(CacheEntry& entry, std::size_t capacity, std::size_t budget) {
            if (capacity <= static_cast<std::size_t>(entry.capacity)) return;
            auto shadow = vmm_resize_proof(entry, capacity);
            entry.expert_for_slot.reserve(capacity); entry.source_for_slot.reserve(capacity);
            entry.last_used.reserve(capacity); entry.prefetched.reserve(capacity);
            const auto delta = shadow->vmm_weight_bytes - entry.vmm_weight_bytes;
            try {
                if (!reclaim_phase_room(delta, 0, &entry, budget) || g_reserved > budget || delta > budget - g_reserved
                    || (delta && !split_device_headroom(entry.backend, delta)))
                    throw std::runtime_error("VMM required union lacks actual rounded delta capacity");
                VmmBarrierContext context{&entry, false};
                const auto epoch = g_ggml_error_count.load(std::memory_order_acquire);
                const auto status = host_moe_vmm::grow_weight_table(g_vmm.arena, entry.vmm_weight, capacity, vmm_barrier(context));
                refresh_vmm_accounting();
                if (status != host_moe_vmm::WeightStatus::ready || g_ggml_error_count.load(std::memory_order_acquire) != epoch
                    || g_backend_compute_failed.load(std::memory_order_acquire))
                    throw std::runtime_error("VMM required union growth refused or failed");
                vmm_publish_capacity(entry, *shadow); refresh_vmm_accounting();
            } catch (...) { vmm_fail(&entry); refresh_vmm_accounting(); throw; }
        }

        Admission replace_wide_scratch(CacheEntry& entry, int rows, std::size_t budget, std::size_t layers) {
            CacheStageTimer timing(g_cache_timing, CacheStage::wide_replace);
            if (!entry.split_weights || !entry.weight_owner || (!entry.vmm_planned && !entry.weight_owner->buffer)
                || rows <= TSG_PRECISION_DECODE_COLUMNS || !layers) return Admission::failed;
            auto planned = plan_split_wide(entry, rows);
            if (!planned.owner) return Admission::failed;
            const auto old_bytes = entry.wide_owner ? entry.wide_owner->bytes : 0;
            const auto retained_bytes = entry.bytes - old_bytes;
            const auto quota = entry_quota(entry, budget, layers);
            if (retained_bytes > quota || planned.owner->bytes > quota - retained_bytes
                || g_reserved < old_bytes) return Admission::refused;
            // Finish the size-only plan before evicting any inactive group.
            // Only old scratch is replaced; keep the current weight/narrow owner.
            if (g_phase_plan && !reclaim_phase_room(planned.owner->bytes, old_bytes, &entry, budget))
                return Admission::refused;
            if (g_reserved - old_bytes > budget
                || planned.owner->bytes > budget - (g_reserved - old_bytes)) return Admission::refused;
            if (entry.vmm_planned) {
                // Publish a retained pending owner before scratch retirement or
                // allocation. A failed acknowledgment/physical call must never
                // reach a local unique_ptr's legacy refund destructor.
                entry.vmm_pending_wide = std::move(planned.owner);
                try {
                    VmmBarrierContext context{&entry, false};
                    if (!vmm_freeze_graphs(&context) || (entry.wide_owner && !retire_vmm_scratch_owner(*entry.wide_owner)))
                        throw std::runtime_error("VMM wide scratch retirement is still owned");
                    entry.wide_owner.reset(); entry.bulk_rows = 0;
                    entry.wide_activation = entry.wide_down = nullptr;
                    entry.width_inputs[entry.graph_rows] = nullptr; entry.width_ids[entry.graph_rows] = nullptr;
                    entry.width_routes[entry.graph_rows] = nullptr; entry.width_outputs[entry.graph_rows] = nullptr;
                    entry.width_graphs[entry.graph_rows] = {};
                    refresh_vmm_accounting();
                    auto& owner = *entry.vmm_pending_wide;
                    owner.charge = split_reserve(2, &owner, owner.bytes);
                    const auto epoch = g_ggml_error_count.load(std::memory_order_acquire);
                    if (!owner.charge || !owner.charge->is_accounted() || !split_device_headroom(entry.backend, owner.bytes)
                        || !materialize_split_scratch(owner)) throw std::runtime_error("VMM wide scratch allocation refused");
                    sync_backend(entry.backend);
                    if (g_ggml_error_count.load(std::memory_order_acquire) != epoch
                        || g_backend_compute_failed.load(std::memory_order_acquire)
                        || !split_commit(owner.charge, 2, &owner, owner.bytes, owner.bytes))
                        throw std::runtime_error("VMM wide scratch commit failed");
                    owner.committed = true;
                    publish_split_variant(entry, entry.graph_rows, planned.variant);
                    entry.wide_owner = std::move(entry.vmm_pending_wide);
                    entry.bulk_rows = rows; ++entry.wide_generation; refresh_vmm_accounting();
                    return Admission::admitted;
                } catch (...) { vmm_fail(&entry); refresh_vmm_accounting(); throw; }
            }
            // No old/new scratch overlap: fence/free, then credit, then admit.
            entry.down_overlap.reset(); entry.wide_activation = entry.wide_down = nullptr;
            entry.wide_owner.reset();
            entry.bulk_rows = 0;
            entry.width_inputs[entry.graph_rows] = nullptr; entry.width_ids[entry.graph_rows] = nullptr;
            entry.width_routes[entry.graph_rows] = nullptr; entry.width_outputs[entry.graph_rows] = nullptr;
            entry.width_graphs[entry.graph_rows] = {};
            entry.bytes -= old_bytes; g_reserved -= old_bytes;
            auto& owner = *planned.owner;
            if (!split_device_headroom(entry.backend, owner.bytes)) return Admission::refused;
            owner.charge = split_reserve(2, &owner, owner.bytes);
            if (!owner.charge) return Admission::refused;
#if defined(TSG_GGML_TEST_HOOKS)
            if (test_failure(17)) return Admission::failed;
#endif
            const auto errors_before = g_ggml_error_count.load(std::memory_order_acquire);
            if (!materialize_split_scratch(owner)) return Admission::failed;
#if defined(TSG_GGML_TEST_HOOKS)
            if (test_failure(18)) return Admission::failed;
#endif
            sync_backend(entry.backend);
            if (g_ggml_error_count.load(std::memory_order_acquire) != errors_before
                || g_backend_compute_failed.load(std::memory_order_acquire)
                || !split_commit(owner.charge, 2, &owner,
                    ggml_gallocr_get_buffer_size(owner.allocator, 0), owner.bytes)) return Admission::failed;
            owner.committed = true;
#if defined(TSG_GGML_TEST_HOOKS)
            if (test_failure(19)) return Admission::failed;
#endif
            // Vector slots are fixed at cold creation; publication cannot allocate.
            const auto bytes = owner.bytes;
            publish_split_variant(entry, entry.graph_rows, planned.variant);
            entry.wide_owner = std::move(planned.owner);
            entry.bulk_rows = rows; ++entry.wide_generation;
            entry.bytes += bytes; g_reserved += bytes;
            return Admission::admitted;
        }

        Admission allocate_entry(CacheEntry& entry)
        {
            CacheStageTimer timing(g_cache_timing, CacheStage::allocation);
            if (entry.split_weights) return allocate_split_entry(entry);
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
            if (published->vmm_planned || (candidate && candidate->vmm_planned))
                throw std::logic_error("Ordinary whole-buffer growth cannot own VMM tensors");
            if (!candidate) { ++g_growth_refusals; return Growth::refused; }
            auto& old = *published;
            auto& next = *candidate;
            if (next.capacity == old.capacity && next.graph_rows == old.graph_rows
                && ((next.split_weights && old.split_weights) || next.bulk_rows == old.bulk_rows)) {
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

        // Optional once-per-wide->narrow phase recovery. Full old+new growth
        // cannot use a few hundred MiB of slack beside a multi-GiB table. The
        // selected table is reloadable from immutable GGUF, so preplan both its
        // replacement and same-capacity fallback, fence, and free it first.
        Growth cold_grow_phase_entry(std::list<std::unique_ptr<CacheEntry>>::iterator& found,
            const HostMoeSegment& hm, const std::size_t (&stride)[3], ggml_backend_t backend,
            std::size_t budget, std::size_t pool_capacity, int graph_rows, int minimum_capacity,
            int bulk_rows) {
            if ((*found)->vmm_planned) throw std::logic_error("Ordinary cold replacement cannot retire a VMM table");
            if (!g_phase_cold_growth_group || !(group_key(**found) == *g_phase_cold_growth_group))
                return Growth::unchanged;
            g_phase_cold_growth_group.reset(); // Consume even an optional refusal.
            const auto& old = **found;
            const auto key = group_key(old);
            for (const auto& group : g_phase_plan->groups())
                if (phase_resident_bytes(group.key, backend) < group.narrow_bytes) return Growth::refused;
            const auto quota = g_phase_plan->cold_growth_quota(key, budget, g_reserved, old.bytes);
            if (!HostMoePhaseQuotaPlan::below_decode_share(old.bytes,
                g_phase_plan->quota(key, budget, false), true)) return Growth::unchanged;
            auto candidate = create_entry(hm, stride, backend, quota, pool_capacity,
                graph_rows, std::max(old.capacity + 1, minimum_capacity), bulk_rows);
            if (!candidate || candidate->bytes <= old.bytes
                || candidate->bytes - old.bytes < (std::size_t(64) << 20)) return Growth::refused;
            // Complete all size-only planning while the old physical owner is
            // intact. A plan failure must leave its hot slots untouched.
            auto fallback = create_entry(hm, stride, backend, old.bytes,
                static_cast<std::size_t>(old.capacity), old.graph_rows, old.capacity, old.bulk_rows);
            if (!fallback || fallback->capacity != old.capacity || fallback->bytes != old.bytes)
                return Growth::refused;
            const auto old_bytes = old.bytes;
            const auto old_capacity = old.capacity;
            std::size_t lost_hot_slots = 0;
            for (const auto expert : old.expert_for_slot) if (expert >= 0) ++lost_hot_slots;
            const auto epoch = g_ggml_error_count.load(std::memory_order_acquire);
            const auto require_clean_epoch = [&] {
                if (g_ggml_error_count.load(std::memory_order_acquire) != epoch
                    || g_backend_compute_failed.load(std::memory_order_acquire))
                    throw std::runtime_error("GLM cold phase growth changed its original backend error epoch");
            };
            drain_decode_prefetch(false);
            sync_backend(backend);
            require_clean_epoch();
            ++g_growth_attempts;
#if defined(TSG_GGML_TEST_HOOKS)
            g_test_phase_cold_growth_attempts.fetch_add(1);
#endif
            try {
                if (!cache_configuration().reset_child())
                    throw std::runtime_error("Causal-route child could not invalidate replaced cache history");
                g_entries.erase(found); // Destruction drains, frees, then refunds shared credit.
                found = g_entries.end();
                g_reserved -= old_bytes; // Private refund follows the physical free.
#if defined(TSG_GGML_TEST_HOOKS)
                if (test_failure(21)) g_ggml_error_count.fetch_add(1, std::memory_order_acq_rel);
#endif
                require_clean_epoch(); // Retirement callbacks/destructors must not reset the baseline.
                bool used_fallback = false;
                auto admission = allocate_entry(*candidate);
                require_clean_epoch();
                if (admission == Admission::refused) {
                    candidate.reset(); // Release unused partial reservation tickets first.
#if defined(TSG_GGML_TEST_HOOKS)
                    if (test_failure(22)) g_ggml_error_count.fetch_add(1, std::memory_order_acq_rel);
#endif
                    require_clean_epoch(); // Refusal cleanup errors cannot become a fallback success.
                    candidate = std::move(fallback); used_fallback = true;
                    admission = allocate_entry(*candidate);
                    require_clean_epoch();
                }
                if (admission != Admission::admitted)
                    throw std::runtime_error("GLM cold phase growth allocation/commit failed after retirement");
                const auto bytes = candidate->bytes;
                candidate->growth_generation = cache_configuration().growth_generation();
#if defined(TSG_GGML_TEST_HOOKS)
                if (test_failure(20)) throw std::bad_alloc(); // list-node publication allocation failure.
#endif
                require_clean_epoch();
                g_entries.push_front(std::move(candidate)); // Node allocation can fail without private debit.
                found = g_entries.begin(); g_reserved += bytes;
                g_growth_peak_accounted = (std::max)(g_growth_peak_accounted, g_reserved);
                if (used_fallback) ++g_growth_refusals; else ++g_growth_successes;
                if (diagnostics()) std::fprintf(stderr,
                    "[HOSTMOE-PHASE-COLD-GROW] layer=%d slots=%d->%d bytes=%zu->%zu lost_hot_slots=%zu reload=immutable_source fallback=%d reserved=%zu budget=%zu\n",
                    hm.layer, old_capacity, (*found)->capacity, old_bytes, bytes, lost_hot_slots,
                    used_fallback ? 1 : 0, g_reserved, budget);
                return used_fallback ? Growth::refused : Growth::replaced;
            } catch (...) { ++g_growth_failures; throw; }
        }

        bool compact_shrink()
        {
            static const bool enabled = [] {
                const char* value = std::getenv("TS_HOST_MOE_EXPERT_CACHE_SHRINK_COPY");
                return value == nullptr || std::strcmp(value, "1") == 0;
            }();
            return enabled;
        }

        // Caller holds the cache gate. A budget cut used to erase every
        // per-layer table above the new share, so decode after each wide
        // prefill restarted cold. Rebuild one ordinary table under `quota`
        // and keep the slots that misses would evict last. The old table
        // stays published until the copy completes; refusal leaves it intact
        // for the caller to erase. Old and new payloads coexist briefly within
        // the credit the caller has not yet returned. `unchanged` means no smaller
        // table fits; `refused` means its admission was refused.
        Growth shrink_entry(std::unique_ptr<CacheEntry>& published, std::size_t quota)
        {
            auto& old = *published;
            if (old.pooled || old.split_weights || old.wide_owner || old.capacity <= old.n_used) return Growth::unchanged;
            HostMoeSegment hm;
            hm.layer = old.layer; hm.hidden = old.hidden; hm.n_ff = old.n_ff;
            hm.n_used = old.n_used; hm.num_experts = old.num_experts; hm.swiglu_clamp = old.swiglu_clamp;
            hm.gate_data = const_cast<void*>(old.source[0]); hm.gate_type = old.type[0];
            hm.up_data = const_cast<void*>(old.source[1]); hm.up_type = old.type[1];
            hm.down_data = const_cast<void*>(old.source[2]); hm.down_type = old.type[2];
            const std::size_t stride[3] = {old.stride[0], old.stride[1], old.stride[2]};
            using Clock = std::chrono::steady_clock;
            const auto ms = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            const auto t0 = Clock::now();
            auto candidate = create_entry(hm, stride, old.backend, quota,
                static_cast<std::size_t>(old.capacity - 1), old.graph_rows, old.n_used, old.bulk_rows);
            if (!candidate || candidate->capacity >= old.capacity || candidate->bytes > quota) return Growth::unchanged;
            auto& next = *candidate;
            const auto t1 = Clock::now();
            const auto admission = allocate_entry(next);
            if (admission == Admission::refused) return Growth::refused;
            if (admission == Admission::failed)
                throw std::runtime_error("Expert-cache compaction payload allocation or commit failed");
            const auto t2 = Clock::now();
            g_reserved += next.bytes;
            struct Rollback {
                std::unique_ptr<CacheEntry>& candidate;
                std::size_t bytes;
                bool published = false;
                ~Rollback() { if (!published) { candidate.reset(); g_reserved -= bytes; } }
            } rollback{candidate, next.bytes};
            // Rank by the miss path's own victim order; drop the most
            // preferred victims. Kept slots retain their relative order, so
            // most survivors form a few long runs of one copy each.
            std::vector<int> order(static_cast<std::size_t>(old.capacity));
            for (int s = 0; s < old.capacity; ++s) order[static_cast<std::size_t>(s)] = s;
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                // Empty slots go first even when LRU ages tie at zero.
                const bool empty_a = old.expert_for_slot[a] < 0, empty_b = old.expert_for_slot[b] < 0;
                if (empty_a != empty_b) return empty_a;
                return old.eviction_policy.prefer(old.expert_for_slot[a], old.last_used[a],
                    old.expert_for_slot[b], old.last_used[b]);
            });
            std::vector<std::uint8_t> keep(static_cast<std::size_t>(old.capacity), 0);
            for (std::size_t i = static_cast<std::size_t>(old.capacity - next.capacity); i < order.size(); ++i)
                keep[static_cast<std::size_t>(order[i])] = old.expert_for_slot[order[i]] >= 0 ? 1 : 0;
            struct Run { int from, to, count; };
            std::vector<Run> runs;
            int hot = 0;
            for (int s = 0; s < old.capacity; ++s) {
                if (!keep[static_cast<std::size_t>(s)]) continue;
                if (!runs.empty() && runs.back().from + runs.back().count == s) ++runs.back().count;
                else runs.push_back({s, hot, 1});
                ++hot;
            }
            for (const auto& run : runs)
                for (int i = 0; i < run.count; ++i) {
                    const auto from = static_cast<std::size_t>(run.from + i), to = static_cast<std::size_t>(run.to + i);
                    next.expert_for_slot[to] = old.expert_for_slot[from];
                    next.source_for_slot[to] = old.source_for_slot[from];
                    next.last_used[to] = old.last_used[from];
                }
            if (!old.prefetched.empty()) {
                next.prefetched.assign(static_cast<std::size_t>(next.capacity), 0);
                for (const auto& run : runs)
                    for (int i = 0; i < run.count; ++i)
                        next.prefetched[static_cast<std::size_t>(run.to + i)] = old.prefetched[static_cast<std::size_t>(run.from + i)];
            }
            next.clock = old.clock;
            next.eviction_policy = old.eviction_policy;
            next.eviction_policy.shrink_capacity(next.capacity);
            next.prefetch_policy = old.prefetch_policy;
            next.prefetch_observations = old.prefetch_observations;
            next.prefetch_reads = old.prefetch_reads;
            next.growth_generation = cache_configuration().growth_generation();
            const auto epoch = g_ggml_error_count.load(std::memory_order_acquire);
            sync_backend(old.backend);
            struct CopyDrain {
                ggml_backend_t backend;
                bool pending = false;
                ~CopyDrain() { if (pending) sync_backend(backend); }
            } copies{next.backend};
            std::uint64_t bytes = 0;
            for (int part = 0; part < 3; ++part) {
                for (const auto& run : runs) {
                    ggml_tensor source = *old.weight[part], destination = *next.weight[part];
                    const auto slot_bytes = source.nb[2];
                    source.data = static_cast<char*>(source.data) + static_cast<std::size_t>(run.from) * slot_bytes;
                    destination.data = static_cast<char*>(destination.data) + static_cast<std::size_t>(run.to) * slot_bytes;
                    source.ne[2] = destination.ne[2] = run.count;
                    source.nb[3] = destination.nb[3] = static_cast<std::size_t>(run.count) * slot_bytes;
                    bytes += ggml_nbytes(&source);
                    copies.pending = true;
                    ggml_backend_tensor_copy_async(old.backend, next.backend, &source, &destination);
                }
            }
            sync_backend(next.backend);
            copies.pending = false;
            if (g_ggml_error_count.load(std::memory_order_acquire) != epoch
                || g_backend_compute_failed.load(std::memory_order_acquire))
                throw std::runtime_error("Expert-cache compaction device copy failed");
            const auto t3 = Clock::now();
            const auto retired_bytes = old.bytes;
            const int retired_capacity = old.capacity;
            auto retired = std::move(published);
            published = std::move(candidate);
            rollback.published = true;
            retired.reset();
            g_reserved -= retired_bytes;
            const auto t4 = Clock::now();
            if (diagnostics()) std::fprintf(stderr,
                "[HOSTMOE-CACHE-SHRINK] layer=%d slots=%d->%d hot=%d runs=%zu copied=%llu reserved=%zu quota=%zu plan=%.2f alloc=%.2f copy=%.2f retire=%.2f ms\n",
                published->layer, retired_capacity, published->capacity, hot, runs.size(),
                static_cast<unsigned long long>(bytes), g_reserved, quota, ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4));
            return Growth::replaced;
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

        void copy_ordered_file_misses(CacheEntry& entry,
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
            // Keep parallel workers supplied with bounded, ascending reads.
            // Coalescing never crosses an unselected expert or projection.
            constexpr std::size_t maximum_read = std::size_t(2) << 20;
            host_moe_for_each_file_read_window(misses,
                std::array<std::size_t, 3>{entry.stride[0], entry.stride[1], entry.stride[2]}, window, maximum_read,
                [&](const auto& runs, const auto& pieces, std::size_t) {
                std::vector<ExpertReadTask> tasks;
                tasks.reserve(runs.size());
                for (const auto& run : runs)
                    tasks.push_back({files[run.part].get(), run.source_offset,
                        g_file_read_buffer.data() + run.staging_offset, run.bytes});
                if (g_cache_timing) HostMoeCacheTiming::add(g_cache_timing->file_windows, 1);
                {
                    CacheStageTimer read_timing(g_cache_timing, CacheStage::file_pipeline);
                    read_expert_files(tasks, [&](std::size_t i) {
                        entry.file_read_bytes += tasks[i].source_read_bytes;
                        if (g_cache_timing) HostMoeCacheTiming::add(g_cache_timing->file_bytes, tasks[i].source_read_bytes);
                        CacheStageTimer upload_timing(g_cache_timing, CacheStage::upload_enqueue);
                        const auto& run = runs[i];
                        for (std::size_t p = run.first_piece; p < run.first_piece + run.piece_count; ++p) {
                            const auto& piece = pieces[p];
                            ggml_backend_tensor_set_async(entry.backend, entry.weight[piece.part],
                                g_file_read_buffer.data() + piece.staging_offset,
                                static_cast<std::size_t>(piece.slot) * entry.stride[piece.part]
                                    + piece.expert_offset, piece.bytes);
                        }
                    });
                }
                {
                    CacheStageTimer wait_timing(g_cache_timing, CacheStage::upload_wait);
                    sync_backend(entry.backend); // source ring cannot be reused before H2D drains
                }
                if (g_backend_compute_failed.load(std::memory_order_acquire))
                    throw std::runtime_error("Joint expert file upload failed");
            });
            for (const auto& miss : misses) entry.publish_slot(miss.first, miss.second);
        }

        void copy_direct_file_misses(CacheEntry& entry,
            const std::vector<std::pair<int, int>>& misses,
            const std::array<std::shared_ptr<ExpertFileSource>, 3>& files, std::size_t bytes)
        {
            std::size_t window = kFileReadWorkspace;
#if defined(TSG_GGML_TEST_HOOKS)
            window = std::min(window, g_test_file_slice_bytes.load());
#endif
            if (window == 0) throw std::runtime_error("Direct expert file slice limit must be positive");
            window = std::min(bytes, window);
            g_file_read_buffer.reserve(window); // EXACT same allocation/charge as ordinary reads.
            std::array<HostMoeDirectFileGeometry, 3> geometry{};
            for (int part = 0; part < 3; ++part)
                geometry[part] = {files[part]->offset(), files[part]->bytes(), files[part]->direct_alignment()};
            std::uint64_t logical_bytes = 0, direct_bytes = 0, buffered_bytes = 0, direct_reads = 0, buffered_reads = 0;
            std::size_t padding_peak = 0;
            host_moe_for_each_direct_read_window(misses,
                std::array<std::size_t, 3>{entry.stride[0], entry.stride[1], entry.stride[2]}, geometry,
                window, reinterpret_cast<std::uintptr_t>(g_file_read_buffer.data()),
                [&](const auto& pieces, std::size_t used) {
                std::vector<ExpertReadTask> tasks;
                tasks.reserve(pieces.size());
                std::size_t payload = 0;
                for (const auto& piece : pieces) {
                    ExpertReadTask task{files[piece.part].get(), piece.source_offset,
                        g_file_read_buffer.data() + piece.staging_offset, piece.bytes};
                    if (piece.direct_bytes) {
                        task.direct_offset = piece.direct_offset;
                        task.direct_destination = g_file_read_buffer.data() + piece.direct_staging_offset;
                        task.direct_bytes = piece.direct_bytes;
                    }
                    tasks.push_back(task); payload += piece.bytes;
                }
                padding_peak = (std::max)(padding_peak, used - payload);
                if (g_cache_timing) HostMoeCacheTiming::add(g_cache_timing->file_windows, 1);
                {
                    CacheStageTimer read_timing(g_cache_timing, CacheStage::file_pipeline);
                    read_expert_files(tasks, [&](std::size_t i) {
                        const auto& task = tasks[i]; const auto& piece = pieces[i];
                        entry.file_read_bytes += task.source_read_bytes; logical_bytes += task.bytes;
                        if (task.direct_completed) { direct_bytes += task.direct_bytes; ++direct_reads; }
                        else { buffered_bytes += task.bytes; ++buffered_reads; }
                        if (g_cache_timing) HostMoeCacheTiming::add(g_cache_timing->file_bytes, task.source_read_bytes);
                        CacheStageTimer upload_timing(g_cache_timing, CacheStage::upload_enqueue);
                        ggml_backend_tensor_set_async(entry.backend, entry.weight[piece.part], task.destination,
                            static_cast<std::size_t>(piece.slot) * entry.stride[piece.part] + piece.expert_offset, task.bytes);
                    });
                }
                {
                    CacheStageTimer wait_timing(g_cache_timing, CacheStage::upload_wait);
                    sync_backend(entry.backend); // all aligned/ordinary destinations settle before reuse
                }
                if (g_backend_compute_failed.load(std::memory_order_acquire))
                    throw std::runtime_error("Direct expert file upload failed");
            });
            for (const auto& miss : misses) entry.publish_slot(miss.first, miss.second);
            if (diagnostics() || g_cache_timing) std::fprintf(stderr,
                "[HOSTMOE-DIRECT-READ] logical_h2d_bytes=%llu completed_direct_source_bytes=%llu buffered_source_bytes=%llu "
                "completed_direct_reads=%llu buffered_reads=%llu window_bytes=%zu padding_peak=%zu "
                "scope=completed_requested_spans_not_physical_ssd_bytes\n",
                static_cast<unsigned long long>(logical_bytes), static_cast<unsigned long long>(direct_bytes),
                static_cast<unsigned long long>(buffered_bytes), static_cast<unsigned long long>(direct_reads),
                static_cast<unsigned long long>(buffered_reads), window, padding_peak);
        }

        void validate_coalesced_ring_targets(const CacheEntry& entry) {
            if (!entry.weight_owner || !entry.weight_owner->buffer || entry.capacity < 1)
                throw std::runtime_error("Coalesced ring has no physical weight owner");
            std::array<HostMoeRingTargetSpan, 3> targets{};
            for (int p = 0; p < 3; ++p) {
                const auto* weight = entry.weight[p];
                const auto extent = HostMoeCoalescedFileRingPlan::mul(entry.capacity, entry.stride[p]);
                if (!weight || weight->view_src || weight->view_offs || !weight->data || !weight->buffer
                    || weight->buffer != entry.weight_owner->buffer || !ggml_is_contiguous(weight)
                    || weight->type != entry.type[p] || weight->ne[2] != entry.capacity
                    || weight->nb[2] != entry.stride[p] || ggml_nbytes(weight) != extent)
                    throw std::runtime_error("Coalesced ring target is not a distinct contiguous owned weight tensor");
                targets[p] = {reinterpret_cast<std::uintptr_t>(weight->data),
                    reinterpret_cast<std::uintptr_t>(ggml_backend_buffer_get_base(weight->buffer)),
                    extent, ggml_backend_buffer_get_size(weight->buffer), false};
            }
#if defined(TSG_GGML_TEST_HOOKS)
            if (test_failure(25)) targets[1].data = targets[0].data;
#endif
            if (!host_moe_ring_targets_disjoint(targets))
                throw std::runtime_error("Coalesced ring GPU target spans overlap or exceed their backing buffer");
        }

        SplitOverlapGraphs& prepare_down_overlap_graphs(CacheEntry& entry) {
            if (!entry.wide_owner || !entry.wide_owner->committed || !entry.weight_owner
                || !entry.weight_owner->committed || !entry.wide_activation || !entry.wide_down)
                throw std::runtime_error("Prefill Down overlap has no committed full graph");
            auto& full = entry.width_graphs[entry.graph_rows];
            if (!entry.down_overlap || entry.down_overlap->full_uid != full.uid) {
                auto result = std::make_unique<SplitOverlapGraphs>();
                int cut = 0;
                while (cut < full.n_nodes && full.nodes[cut] != entry.wide_down) ++cut;
                if (cut < 1 || cut >= full.n_nodes)
                    throw std::runtime_error("Prefill Down overlap cut is missing");
                result->prefix_nodes.assign(full.nodes, full.nodes + cut);
                result->suffix_nodes.assign(full.nodes + cut, full.nodes + full.n_nodes);
                result->prefix = ggml_graph_view(&full, 0, cut);
                result->suffix = ggml_graph_view(&full, cut, full.n_nodes);
                result->prefix.nodes = result->prefix_nodes.data();
                result->suffix.nodes = result->suffix_nodes.data();
                result->prefix.uid = split_graph_uid(); result->suffix.uid = split_graph_uid();
                result->full_uid = full.uid;
                entry.down_overlap = std::move(result);
            }
            auto& result = *entry.down_overlap;
            std::vector<HostMoeOverlapNode> nodes;
            nodes.reserve(result.prefix_nodes.size() + result.suffix_nodes.size());
            auto describe = [&](ggml_tensor* tensor) {
                HostMoeOverlapNode node; node.id = reinterpret_cast<std::uintptr_t>(tensor);
                static_assert(std::tuple_size<decltype(node.sources)>::value >= GGML_MAX_SRC + 1,
                    "Overlap graph source proof must include view_src");
                for (int s = 0; s < GGML_MAX_SRC; ++s)
                    node.sources[s] = reinterpret_cast<std::uintptr_t>(tensor->src[s]);
                node.sources[GGML_MAX_SRC] = reinterpret_cast<std::uintptr_t>(tensor->view_src);
                nodes.push_back(node);
            };
            for (auto* node : result.prefix_nodes) describe(node);
            for (auto* node : result.suffix_nodes) describe(node);
            if (!host_moe_overlap_cut_valid(nodes, result.prefix_nodes.size(),
                    reinterpret_cast<std::uintptr_t>(entry.wide_activation),
                    reinterpret_cast<std::uintptr_t>(entry.wide_down))
                || entry.wide_down->op != GGML_OP_MUL_MAT_ID
                || entry.wide_down->src[0] != entry.weight[2]
                || entry.wide_down->src[1] != entry.wide_activation
                || entry.wide_down->src[2] != entry.width_ids[entry.graph_rows]
                || entry.wide_activation->op != GGML_OP_MUL)
                throw std::runtime_error("Prefill Down overlap changed the original graph topology");
            validate_coalesced_ring_targets(entry);
            const auto actual_type = ggml_backend_buffer_get_type(entry.weight_owner->buffer);
            auto down_begin = reinterpret_cast<std::uintptr_t>(entry.weight[2]->data);
#if defined(TSG_GGML_TEST_HOOKS)
            if (test_failure(26)) down_begin = reinterpret_cast<std::uintptr_t>(entry.weight[0]->data);
#endif
            const auto down_bytes = ggml_nbytes(entry.weight[2]);
            std::vector<ggml_tensor*> reachable(result.prefix_nodes);
            std::vector<ggml_tensor*> visited;
            int projections = 0; unsigned projection_parts = 0;
            for (std::size_t index = 0; index < reachable.size(); ++index) {
                auto* tensor = reachable[index];
                if (std::find(visited.begin(), visited.end(), tensor) != visited.end()) continue;
                if (visited.size() >= 256 || tensor == entry.weight[2] || !tensor->buffer || !tensor->data
                    || ggml_backend_buffer_get_type(tensor->buffer) != actual_type
                    || ggml_backend_buffer_is_host(tensor->buffer))
                    throw std::runtime_error("Prefill prefix source is not bounded owned CUDA storage");
                visited.push_back(tensor);
                const auto data = reinterpret_cast<std::uintptr_t>(tensor->data);
                const auto base = reinterpret_cast<std::uintptr_t>(ggml_backend_buffer_get_base(tensor->buffer));
                const auto size = ggml_backend_buffer_get_size(tensor->buffer), bytes = ggml_nbytes(tensor);
                if (data < base || data - base > size || bytes > size - (data - base)
                    || !host_moe_overlap_spans_disjoint(data, bytes, down_begin, down_bytes))
                    throw std::runtime_error("Prefill prefix aliases the Down upload destination");
                if (tensor->op == GGML_OP_MUL_MAT_ID) {
                    if ((tensor->src[0] != entry.weight[0] && tensor->src[0] != entry.weight[1])
                        || tensor->src[1] != entry.width_inputs[entry.graph_rows]
                        || tensor->src[2] != entry.width_ids[entry.graph_rows])
                        throw std::runtime_error("Prefill prefix changed Gate/Up sources");
                    ++projections;
                    projection_parts |= tensor->src[0] == entry.weight[0] ? 1u : 2u;
                }
                for (auto* source : tensor->src) if (source) reachable.push_back(source);
                if (tensor->view_src) reachable.push_back(tensor->view_src);
            }
            if (projections != 2 || projection_parts != 3) throw std::runtime_error("Prefill prefix is not exactly Gate and Up");
            return result;
        }

        struct DownOverlapActivity {
            int layer, rows;
            std::size_t misses, arena = 0, slot = 0;
            std::uint64_t prefix = 0, suffix = 0, gate_bytes = 0, down_bytes = 0;
            std::uint64_t gate_windows = 0, down_windows = 0;
            double gate_ms = 0, prefix_ms = 0, down_ms = 0, join_ms = 0, suffix_ms = 0;
            bool completed = false;
            ~DownOverlapActivity() {
                std::fprintf(stderr, "[HOSTMOE-PREFILL-DOWN-OVERLAP] profile=1 layer=%d rows=%d outcome=%s "
                    "selected_misses=%zu prefix_submissions=%llu suffix_submissions=%llu gate_up_bytes=%llu "
                    "down_bytes=%llu gate_up_windows=%llu down_windows=%llu arena_bytes=%zu slot_bytes=%zu "
                    "distinct_upload_backend=1 gate_up_file_ms=%.6f prefix_submit_ms=%.6f down_file_ms=%.6f "
                    "join_ms=%.6f suffix_compute_ms=%.6f "
                    "scope=exact_selected_payload_and_host_wall_not_kernel_overlap_or_physical_io\n",
                    layer, rows, completed ? "completed" : "failed", misses,
                    static_cast<unsigned long long>(prefix), static_cast<unsigned long long>(suffix),
                    static_cast<unsigned long long>(gate_bytes), static_cast<unsigned long long>(down_bytes),
                    static_cast<unsigned long long>(gate_windows), static_cast<unsigned long long>(down_windows),
                    arena, slot, gate_ms, prefix_ms, down_ms, join_ms, suffix_ms);
            }
        };

        void copy_projection_file_misses(CacheEntry& entry, const HostMoeProjectionRingPlan& plan,
            const std::array<std::shared_ptr<ExpertFileSource>, 3>& files,
            ggml_backend_t upload, bool cache_admit, std::uint64_t epoch, bool down,
            DownOverlapActivity& activity) {
            std::vector<ExpertReadTask> tasks;
            std::vector<std::size_t> counts;
            tasks.reserve(plan.pieces.size()); counts.reserve(plan.windows.size());
            for (std::size_t w = 0; w < plan.windows.size(); ++w) {
                const auto& window = plan.windows[w]; counts.push_back(window.count);
                auto* base = g_file_read_buffer.data() + (w % 2) * plan.slot_bytes;
                for (auto i = window.first; i < window.first + window.count; ++i) {
                    const auto& p = plan.pieces[i];
                    tasks.push_back({files[p.part].get(), p.source_offset, base + p.staging_offset, p.bytes});
                    cache_file_task(tasks.back(), entry.stride[p.part], cache_admit);
                }
            }
#if defined(TSG_GGML_TEST_HOOKS)
            if (down && test_failure(31)) {
                auto& task = tasks.front();
                task.cache_offset = task.offset; task.cache_bytes = task.bytes;
                task.immutable_read = [](ExpertReadTask& selected) {
                    selected.source->read(selected.offset, selected.destination, selected.bytes / 2);
                    g_test_down_partial_reads.fetch_add(1);
                    throw std::runtime_error("Injected joined Down reader failure after exact source prefix");
                };
            }
#endif
            struct Drain {
                ggml_backend_t backend;
                std::array<ggml_backend_event_t, 2> fences{};
                ~Drain() {
                    try { sync_backend(backend); }
                    catch (...) { g_backend_compute_failed.store(true, std::memory_order_release); }
#if defined(TSG_GGML_TEST_HOOKS)
                    g_test_payload_upload_drains.fetch_add(1);
#endif
                    for (auto event : fences) if (event) {
                        try { ggml_backend_event_free(event); }
                        catch (...) { g_backend_compute_failed.store(true, std::memory_order_release); }
                    }
                }
            } drain{upload}; // Producer join and DMA drain precede task-lease release.
            auto check = [&] {
                if (g_ggml_error_count.load(std::memory_order_acquire) != epoch
                    || g_backend_compute_failed.load(std::memory_order_acquire))
                    throw std::runtime_error("Prefill Down overlap backend epoch changed");
            };
            const auto device = ggml_backend_get_device(upload);
            for (auto& event : drain.fences) {
                event = ggml_backend_event_new(device);
                if (!event) throw std::runtime_error("Prefill projection ring fence allocation failed");
            }
            std::array<bool, 2> submitted{};
            CacheStageTimer read_timing(g_cache_timing, CacheStage::file_pipeline);
            read_expert_file_windows(tasks, counts, [&](std::size_t slot) {
                if (submitted[slot]) ggml_backend_event_synchronize(drain.fences[slot]);
                check();
            }, [&](std::size_t w) {
                const auto& window = plan.windows[w];
                CacheStageTimer upload_timing(g_cache_timing, CacheStage::upload_enqueue);
                for (auto i = window.first; i < window.first + window.count; ++i) {
                    const auto& p = plan.pieces[i];
                    entry.file_read_bytes += tasks[i].source_read_bytes;
                    if (g_cache_timing) HostMoeCacheTiming::add(g_cache_timing->file_bytes, tasks[i].source_read_bytes);
                    auto& bytes = down ? activity.down_bytes : activity.gate_bytes;
                    bytes += tasks[i].source_read_bytes;
                    ggml_backend_tensor_set_async(upload, entry.weight[p.part], tasks[i].upload_data(),
                        static_cast<std::size_t>(p.slot) * entry.stride[p.part] + p.expert_offset, p.bytes);
                }
#if defined(TSG_GGML_TEST_HOOKS)
                if (down && test_failure(28)) throw std::runtime_error("Injected Down upload failure before fence");
#endif
                ggml_backend_event_record(drain.fences[w % 2], upload); submitted[w % 2] = true;
                if (g_cache_timing) HostMoeCacheTiming::add(g_cache_timing->file_windows, 1);
                ++(down ? activity.down_windows : activity.gate_windows);
                check();
            });
            sync_backend(upload); check();
            if ((down ? activity.down_bytes : activity.gate_bytes) != plan.payload_bytes)
                throw std::runtime_error("Prefill projection ring selected bytes differ");
        }

        void copy_ring_file_misses(CacheEntry& entry,
            const std::vector<std::pair<int, int>>& misses,
            const std::array<std::shared_ptr<ExpertFileSource>, 3>& files, std::size_t bytes,
            bool cache_admit, bool expanded_staging, bool genuine_wide_prefill,
            const std::vector<std::pair<int, int>>* read_order = nullptr)
        {
            std::size_t arena_bytes = kFileReadWorkspace;
            if (expanded_staging && g_file_read_buffer.try_reserve_expanded())
                arena_bytes = kHostMoeScalarFileStagingBytes;
#if defined(TSG_GGML_TEST_HOOKS)
            if (g_test_file_slice_bytes.load() != kFileReadWorkspace)
                arena_bytes = std::min(arena_bytes, g_test_file_slice_bytes.load());
#endif
            arena_bytes = std::min(bytes, arena_bytes);
            HostMoeFileRingPlan plan(read_order ? *read_order : misses,
                {entry.stride[0], entry.stride[1], entry.stride[2]}, arena_bytes);
            std::optional<HostMoeCoalescedFileRingPlan> coalesced;
            if (host_moe_ring_coalesce_eligible(prefill_ring_coalesce_enabled(),
                    entry.pooled && entry.split_weights && genuine_wide_prefill,
                    entry.n_used, g_payload_cache_budget, bytes, arena_bytes, misses.size())
                && HostMoeCoalescedFileRingPlan::metadata_fits(misses.size(), bytes, plan.slot_bytes)) {
                validate_coalesced_ring_targets(entry); // Actual absolute spans, before any source I/O.
                std::array<std::size_t, 3> extents{};
                std::array<std::uint64_t, 3> identities{};
                for (int p = 0; p < 3; ++p) {
                    if (!files[p] || files[p]->bytes() > (std::numeric_limits<std::size_t>::max)())
                        throw std::runtime_error("Coalesced ring source owner/extent is unavailable");
                    extents[p] = static_cast<std::size_t>(files[p]->bytes());
                    identities[p] = files[p]->identity();
                }
                coalesced.emplace(misses, std::array<std::size_t, 3>{entry.stride[0], entry.stride[1], entry.stride[2]},
                    arena_bytes, extents, identities, entry.capacity, entry.num_experts);
                if (coalesced->payload_bytes != bytes)
                    throw std::runtime_error("Coalesced ring selected byte proof differs from caller");
                if (coalesced->windows.size() != plan.windows.size())
                    throw std::runtime_error("Coalesced ring changed the bounded slot/window geometry");
                if (!coalesced->coalesced_bytes || coalesced->reads.size() >= plan.pieces.size()) coalesced.reset();
            }
            g_file_read_buffer.reserve(arena_bytes);
            std::vector<ExpertReadTask> tasks;
            std::vector<std::size_t> counts;
            tasks.reserve(coalesced ? coalesced->reads.size() : plan.pieces.size()); counts.reserve(plan.windows.size());
            for (std::size_t index = 0; index < plan.windows.size(); ++index) {
                const auto& window = plan.windows[index];
                auto* base = g_file_read_buffer.data() + (index % 2) * plan.slot_bytes;
                if (coalesced) {
                    const auto& selected = coalesced->windows[index];
                    counts.push_back(selected.read_count);
                    for (std::size_t i = selected.first_read; i < selected.first_read + selected.read_count; ++i) {
                        const auto& read = coalesced->reads[i];
                        tasks.push_back({files[read.part].get(), read.source_offset, base + read.staging_offset, read.bytes});
                    }
                    continue;
                }
                counts.push_back(window.count);
                for (std::size_t i = window.first; i < window.first + window.count; ++i) {
                    const auto& piece = plan.pieces[i];
                    tasks.push_back({files[piece.part].get(), piece.source_offset,
                        base + piece.staging_offset, piece.bytes});
                    cache_file_task(tasks.back(), entry.stride[piece.part], cache_admit);
                }
            }
            // This guard outlives the joined producers and drains even if an
            // upload fails before its fence can be recorded. No slot is valid
            // until all three projections and every queued DMA have completed.
            struct Drain {
                ggml_backend_t backend;
                std::array<ggml_backend_event_t, 2> fences{};
                ~Drain() {
                    sync_backend(backend);
#if defined(TSG_GGML_TEST_HOOKS)
                    g_test_payload_upload_drains.fetch_add(1);
#endif
                    for (auto event : fences) if (event) ggml_backend_event_free(event);
                }
            } drain{entry.backend};
            for (auto& event : drain.fences) {
                event = ggml_backend_event_new(ggml_backend_get_device(entry.backend));
                if (!event) throw std::runtime_error("Expert read ring fence allocation failed");
            }
            std::array<bool, 2> submitted{};
            const auto error_epoch = g_ggml_error_count.load(std::memory_order_acquire);
            auto check = [&] {
                if (g_ggml_error_count.load(std::memory_order_acquire) != error_epoch
                    || g_backend_compute_failed.load(std::memory_order_acquire))
                    throw std::runtime_error("Expert read ring upload failed");
            };
            {
                CacheStageTimer read_timing(g_cache_timing, CacheStage::file_pipeline);
                read_expert_file_windows(tasks, counts, [&](std::size_t slot) {
                    if (submitted[slot]) ggml_backend_event_synchronize(drain.fences[slot]);
                    check();
                }, [&](std::size_t index) {
                    const auto& window = plan.windows[index];
                    if (g_cache_timing) HostMoeCacheTiming::add(g_cache_timing->file_windows, 1);
                    CacheStageTimer upload_timing(g_cache_timing, CacheStage::upload_enqueue);
                    if (coalesced) {
                        const auto& selected = coalesced->windows[index];
                        for (std::size_t i = selected.first_read; i < selected.first_read + selected.read_count; ++i) {
                            entry.file_read_bytes += tasks[i].source_read_bytes;
                            if (g_cache_timing) HostMoeCacheTiming::add(g_cache_timing->file_bytes, tasks[i].source_read_bytes);
                        }
                        auto* base = g_file_read_buffer.data() + (index % 2) * plan.slot_bytes;
                        for (std::size_t i = selected.first_piece; i < selected.first_piece + selected.piece_count; ++i) {
                            const auto& piece = coalesced->pieces[i];
                            ggml_backend_tensor_set_async(entry.backend, entry.weight[piece.part], base + piece.staging_offset,
                                static_cast<std::size_t>(piece.slot) * entry.stride[piece.part] + piece.expert_offset, piece.bytes);
                        }
                    } else for (std::size_t i = window.first; i < window.first + window.count; ++i) {
                        const auto& piece = plan.pieces[i];
                        entry.file_read_bytes += tasks[i].source_read_bytes;
                        if (g_cache_timing) HostMoeCacheTiming::add(g_cache_timing->file_bytes, tasks[i].source_read_bytes);
                        ggml_backend_tensor_set_async(entry.backend, entry.weight[piece.part], tasks[i].upload_data(),
                            static_cast<std::size_t>(piece.slot) * entry.stride[piece.part]
                                + piece.expert_offset, piece.bytes);
                    }
#if defined(TSG_GGML_TEST_HOOKS)
                    if (test_failure(23)) throw std::runtime_error("Injected ring failure after queued uploads before fence");
#endif
                    ggml_backend_event_record(drain.fences[index % 2], entry.backend);
                    submitted[index % 2] = true;
#if defined(TSG_GGML_TEST_HOOKS)
                    if (test_failure(24)) g_ggml_error_count.fetch_add(1, std::memory_order_acq_rel);
#endif
                    check();
                });
            }
            {
                CacheStageTimer wait_timing(g_cache_timing, CacheStage::upload_wait);
                sync_backend(entry.backend);
            }
            check();
            if (coalesced) {
                std::size_t completed_bytes = 0;
                for (const auto& task : tasks) completed_bytes = HostMoeCoalescedFileRingPlan::add(completed_bytes, task.source_read_bytes);
                if (completed_bytes != bytes) throw std::runtime_error("Coalesced ring completed source bytes differ from selected payload");
            }
            for (const auto& miss : misses) entry.publish_slot(miss.first, miss.second);
            if (g_cache_timing) {
                HostMoeCacheTiming::add(g_cache_timing->ring_calls, 1);
                if (arena_bytes == kHostMoeScalarFileStagingBytes)
                    HostMoeCacheTiming::add(g_cache_timing->ring_expanded_calls, 1);
                g_cache_timing->ring_arena_peak_bytes = (std::max)(g_cache_timing->ring_arena_peak_bytes,
                    static_cast<std::uint64_t>(arena_bytes));
                g_cache_timing->ring_slot_peak_bytes = (std::max)(g_cache_timing->ring_slot_peak_bytes,
                    static_cast<std::uint64_t>(plan.slot_bytes));
                if (coalesced) {
                    HostMoeCacheTiming::add(g_cache_timing->ring_coalesced_calls, 1);
                    HostMoeCacheTiming::add(g_cache_timing->ring_read_tasks_before, plan.pieces.size());
                    HostMoeCacheTiming::add(g_cache_timing->ring_read_tasks_after, tasks.size());
                    HostMoeCacheTiming::add(g_cache_timing->ring_coalesced_payload_bytes, bytes);
                    HostMoeCacheTiming::add(g_cache_timing->ring_coalesced_bytes, coalesced->coalesced_bytes);
                    g_cache_timing->ring_coalesced_read_cap_bytes = kHostMoeRingCoalesceReadBytes;
                }
            }
#if defined(TSG_GGML_TEST_HOOKS)
            g_test_last_completed_ring_plan = {arena_bytes, plan.slot_bytes, plan.windows.size()};
            if (coalesced) {
                g_test_last_coalesced_ring_plan = {plan.pieces.size(), tasks.size(), bytes, coalesced->coalesced_bytes, kHostMoeRingCoalesceReadBytes};
                g_test_coalesced_ring_calls.fetch_add(1);
            }
#endif
        }

        void copy_joint_file_misses(CacheEntry& entry,
            const std::vector<std::pair<int, int>>& misses,
            const std::array<std::shared_ptr<ExpertFileSource>, 3>& files, std::size_t bytes,
            bool genuine_wide_prefill, bool cache_admit, bool scalar_staging, bool pooled_glm)
        {
            std::optional<HostMoeMissReadOrder> sorted;
            const auto* read_order = &misses;
            if (file_read_sort_enabled()) {
                if (g_payload_cache_budget != 0 || victim_mode())
                    throw std::invalid_argument("File sort requires RAM payload caches disabled");
                if (pooled_glm && entry.pooled && entry.split_weights && entry.n_used == 8
                    && entry.num_experts <= 288 && SharedCacheCharge::host_accounting_attached()) {
                    std::array<HostMoeRingTargetSpan, 3> targets{};
                    for (int p = 0; p < 3; ++p) {
                        const auto extent = HostMoeCoalescedFileRingPlan::mul(entry.capacity, entry.stride[p]);
                        const auto* weight = entry.weight[p];
                        if (!files[p] || files[p]->bytes() != HostMoeCoalescedFileRingPlan::mul(entry.num_experts, entry.stride[p])
                            || !weight || weight->view_src || weight->view_offs || !weight->data || !weight->buffer
                            || !ggml_is_contiguous(weight) || weight->type != entry.type[p]
                            || weight->ne[2] != entry.capacity || weight->nb[2] != entry.stride[p]
                            || ggml_nbytes(weight) != extent)
                            throw std::runtime_error("File sort exact source/physical target proof failed");
                        targets[p] = {reinterpret_cast<std::uintptr_t>(weight->data),
                            reinterpret_cast<std::uintptr_t>(ggml_backend_buffer_get_base(weight->buffer)), extent,
                            ggml_backend_buffer_get_size(weight->buffer), false};
                    }
                    if (!host_moe_ring_targets_disjoint(targets))
                        throw std::runtime_error("File sort GPU targets overlap or exceed owned buffers");
                    sorted.emplace(misses, entry.capacity, entry.num_experts);
                    read_order = &sorted->ordered;
                }
            }
#if defined(TSG_GGML_TEST_HOOKS) && defined(TSG_GGML_USE_CUDA)
            std::size_t completed_tasks = 0, completed_windows = 0, completed_arena = 0;
#endif
            const auto record_sort = [&](bool ring) {
                (void)ring;
                if (!sorted) return;
                if (g_cache_timing) {
                    HostMoeCacheTiming::add(g_cache_timing->file_sort_calls, 1);
                    HostMoeCacheTiming::add(g_cache_timing->file_sort_changed_calls, sorted->changed ? 1 : 0);
                    HostMoeCacheTiming::add(g_cache_timing->file_sort_misses, misses.size());
                    HostMoeCacheTiming::add(g_cache_timing->file_sort_bytes, bytes);
                    HostMoeCacheTiming::add(g_cache_timing->file_sort_inversions_before, sorted->inversions_before);
                    // Unique ascending source experts have exactly zero inversions after sorting.
                }
#if defined(TSG_GGML_TEST_HOOKS) && defined(TSG_GGML_USE_CUDA)
                auto& proof = g_test_file_sort_snapshot;
                ++proof.completed_calls; proof.changed_calls += sorted->changed ? 1 : 0;
                proof.misses = misses.size(); proof.bytes = bytes; proof.ring = ring;
                proof.inversions_before = sorted->inversions_before;
                proof.arena = completed_arena; proof.slot_bytes = ring ? completed_arena / 2 : completed_arena;
                proof.windows = completed_windows; proof.tasks = completed_tasks;
                for (int p = 0; p < 3; ++p) proof.source_identity[p] = files[p]->identity();
                for (std::size_t i = 0; i < misses.size(); ++i) {
                    proof.original_slot[i] = misses[i].first; proof.original_expert[i] = misses[i].second;
                    proof.ordered_slot[i] = (*read_order)[i].first; proof.ordered_expert[i] = (*read_order)[i].second;
                }
#endif
            };
            // The ordered 2 MiB read experiment regressed matched Windows GLM
            // prefill. Keep the qualified source scheduling as the default;
            // this explicit diagnostic opt-in is not a hardware heuristic.
            static const bool ordered_reads = [] {
                const char* value = std::getenv("TS_HOST_MOE_EXPERT_CACHE_ORDERED_READS");
                return value && value[0] == '1' && value[1] == '\0';
            }();
            static const bool direct_reads = [] {
                const char* value = std::getenv("TS_HOST_MOE_EXPERT_CACHE_DIRECT_READS");
                return value && value[0] == '1' && value[1] == '\0';
            }();
            static const int ring_reads = [] {
                const char* value = std::getenv("TS_HOST_MOE_EXPERT_CACHE_RING_READS");
                if (value == nullptr) return -1;
                // 1: all exact selected reads; 2: genuine wide GLM prefill only.
                // Scalar decode has smaller transfers and can lose to fence
                // overhead even when prefill benefits from the same pipeline.
                return value && (value[0] == '1' || value[0] == '2') && value[1] == '\0'
                    ? value[0] - '0' : 0;
            }();
            if (static_cast<int>(direct_reads) + static_cast<int>(ordered_reads) + static_cast<int>(ring_reads > 0) > 1)
                throw std::invalid_argument("Direct, ordered and ring expert read experiments must be tested separately");
            bool automatic_ring = false;
#if defined(_WIN32)
            // Full-model Windows measurements improve prefill but regress
            // scalar decode. Keep other platforms opt-in until measured, and
            // preserve explicit direct/ordered experiments as isolated controls.
            automatic_ring = ring_reads < 0 && genuine_wide_prefill && !direct_reads && !ordered_reads;
#endif
            if (automatic_ring || ring_reads == 1 || (ring_reads == 2 && genuine_wide_prefill)) {
                copy_ring_file_misses(entry, misses, files, bytes, cache_admit,
                    host_moe_prefill_ring_staging_eligible(prefill_ring_staging_enabled(),
                        entry.pooled && entry.split_weights, genuine_wide_prefill, entry.n_used, bytes), genuine_wide_prefill,
                    sorted ? read_order : nullptr);
#if defined(TSG_GGML_TEST_HOOKS) && defined(TSG_GGML_USE_CUDA)
                if (sorted) {
                    const auto geometry = g_test_last_completed_ring_plan;
                    completed_arena = geometry[0]; completed_windows = geometry[2];
                    HostMoeFileRingPlan proof(*read_order, {entry.stride[0], entry.stride[1], entry.stride[2]}, completed_arena);
                    completed_tasks = proof.pieces.size();
                }
#endif
                record_sort(true);
                return;
            }
            if (direct_reads && genuine_wide_prefill) {
                copy_direct_file_misses(entry, misses, files, bytes);
                return;
            }
            if (ordered_reads) {
                copy_ordered_file_misses(entry, misses, files, bytes);
                return;
            }
            // A joint union can exceed the old one-row transfer ceiling. Read
            // bounded slices, never fall through to touching mmap weights.
            std::size_t window = kFileReadWorkspace;
            if (scalar_staging && g_file_read_buffer.try_reserve_expanded())
                window = kHostMoeScalarFileStagingBytes;
#if defined(TSG_GGML_TEST_HOOKS)
            // A smaller explicit test slice still exercises split/fault paths;
            // the normal 32 MiB default is not an experimental-size override.
            if (g_test_file_slice_bytes.load() != kFileReadWorkspace)
                window = std::min(window, g_test_file_slice_bytes.load());
#endif
            if (window == 0) throw std::runtime_error("Joint expert file slice limit must be positive");
            window = std::min(bytes, window);
            g_file_read_buffer.reserve(window);
#if defined(TSG_GGML_TEST_HOOKS) && defined(TSG_GGML_USE_CUDA)
            completed_arena = window;
#endif
            struct Destination { int slot, part; std::size_t offset; };
            std::vector<ExpertReadTask> tasks;
            std::vector<Destination> destinations;
            FilePayloadUploadDrain lease_drain{entry.backend, tasks};
            std::size_t used = 0;
            auto flush = [&] {
                if (tasks.empty()) return;
#if defined(TSG_GGML_TEST_HOOKS)
                ++g_test_file_windows;
#endif
                if (g_cache_timing) HostMoeCacheTiming::add(g_cache_timing->file_windows, 1);
                {
                    CacheStageTimer read_timing(g_cache_timing, CacheStage::file_pipeline);
                    read_expert_files(tasks, [&](std::size_t i) {
                        const auto& d = destinations[i];
                        entry.file_read_bytes += tasks[i].source_read_bytes;
                        if (g_cache_timing) HostMoeCacheTiming::add(g_cache_timing->file_bytes, tasks[i].source_read_bytes);
                        CacheStageTimer upload_timing(g_cache_timing, CacheStage::upload_enqueue);
                        ggml_backend_tensor_set_async(entry.backend, entry.weight[d.part], tasks[i].upload_data(),
                            static_cast<std::size_t>(d.slot) * entry.stride[d.part] + d.offset, tasks[i].bytes);
                    });
                }
                {
                    CacheStageTimer wait_timing(g_cache_timing, CacheStage::upload_wait);
                    sync_backend(entry.backend); // source ring cannot be reused before H2D drains
                }
                if (g_backend_compute_failed.load(std::memory_order_acquire))
                    throw std::runtime_error("Joint expert file upload failed");
#if defined(TSG_GGML_TEST_HOOKS) && defined(TSG_GGML_USE_CUDA)
                completed_tasks += tasks.size(); ++completed_windows;
#endif
                tasks.clear(); destinations.clear(); used = 0;
            };
            for (const auto& miss : *read_order) for (int part = 0; part < 3; ++part) {
                std::size_t offset = 0;
                while (offset < entry.stride[part]) {
                    if (used == window) flush();
                    const auto count = std::min(entry.stride[part] - offset, window - used);
                    tasks.push_back({files[part].get(), static_cast<std::size_t>(miss.second) * entry.stride[part] + offset,
                        g_file_read_buffer.data() + used, count});
                    cache_file_task(tasks.back(), entry.stride[part], cache_admit);
                    destinations.push_back({miss.first, part, offset});
                    used += count; offset += count;
                }
            }
            flush();
            for (const auto& miss : misses) entry.publish_slot(miss.first, miss.second);
            record_sort(false);
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

        template<std::size_t MaximumRoutes>
#if defined(_MSC_VER)
        __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
        __attribute__((noinline))
#endif
        bool finite_device_routes_bounded(const ggml_tensor* routes, std::size_t count)
        {
            // Wide PP is an explicit bounded path. Read the complete small
            // routing block before any slot mutation, while activations and
            // results keep their device-to-device bridge. No retained owner
            // or additional graph/weight allocation is introduced here.
            // Separate non-inlined frames retain Qwen's16KiB physical scratch
            // while the explicit GLM adapter uses its accounted32KiB frame.
            std::array<float, MaximumRoutes> values{};
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

        bool finite_device_routes(const ggml_tensor* routes, std::size_t count, bool explicit_glm)
        {
            return explicit_glm ? finite_device_routes_bounded<kHostMoeCacheMaxGlmRouteElements>(routes, count)
                : finite_device_routes_bounded<kHostMoeCacheMaxRouteElements>(routes, count);
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

    static int host_moe_cached_experts_impl(const HostMoeSegment& hm, const float* x,
        const std::int32_t* ids, const float* routes, float* out, const char* kernel_name,
        ggml_backend_t explicit_backend, const HostMoePrefetchHooks* prefetch_hooks,
        tsg_glm_neighbor::Owner* neighbor)
    {
#ifdef TSG_GGML_USE_CUDA
        // Pause and drain each complete background fill before taking the
        // global cache gate. A demand lookup may wait for that same entry.
        if (prefetch_hooks && prefetch_hooks->demand_begin)
            prefetch_hooks->demand_begin(prefetch_hooks->context);
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
        const auto vmm_call_epoch = g_ggml_error_count.load(std::memory_order_acquire);
        if (tsg_glm_neighbor::option() < 0 || (tsg_glm_neighbor::option() == 1 && !neighbor))
            return decline("causal-route request lacks its immutable model context");
        if (neighbor && (!neighbor->ready() || !neighbor->active_call()
            || cache_configuration().child_cookie() != neighbor
            || !cache_configuration().owns(neighbor->scope()) || !explicit_model
            || hm.pooled_model_experts <= 0 || hm.n_used != 8))
            return decline("causal-route owner/source/profile is not certified");
        if (file_source_view_option() == 1) {
            if (!ExpertFileSourceViews::execution_ready())
                return decline("source-view profile is not configured for execution");
        }
        const auto vmm_pressure_serial_at_entry = g_vmm.pressure_mutation_serial;
        const auto settings = cache_configuration().settings();
        const std::size_t budget = settings.budget, layers = settings.layers;
        if (budget == 0 || layers == 0) return decline("scope has no capacity or layout groups");
        int planned_decode_rows = settings.max_decode_rows, planned_prefill_rows = settings.max_rows;
        if (g_phase_plan && (!explicit_model || hm.pooled_model_experts <= 0
            || !g_phase_plan->row_limits(host_moe_expert_group_key(hm), settings.max_decode_rows,
                settings.max_rows, planned_decode_rows, planned_prefill_rows)))
            return decline("missing or incompatible per-group phase geometry");
        const bool joint = hm.independent_decode_rows && hm.seq_len > 1
            && (std::strcmp(kernel_name, "qwen4exp arena batched decode") == 0 || explicit_model);
        ScopedCacheTiming cache_timing(hm.layer, hm.seq_len, joint,
            std::strcmp(kernel_name, "glm5-next bulk expert prefill") == 0);
        if (joint && hm.seq_len > kHostMoeCacheMaxDecodeRows) return decline("independent decode width exceeds compact graph variants");
        if (g_phase_plan && hm.seq_len > (joint ? planned_decode_rows : planned_prefill_rows))
            return decline("execution width exceeds the measured group phase plan");
        const std::size_t pool_capacity = static_cast<std::size_t>(hm.pooled_model_experts > 0 ? hm.pooled_model_experts : hm.num_experts);
        std::size_t stride[3] = {};
        if (!valid_layout(hm, stride, joint ? std::max(planned_prefill_rows, planned_decode_rows)
                : planned_prefill_rows, explicit_model)) return decline("expert type, strides, dimensions, biases or row bound is unsupported");
        const bool phase_wide = explicit_model && std::strcmp(kernel_name, "glm5-next bulk expert prefill") == 0;
        // Bulk prefill leaves pooled group tables sized for its phase: the
        // active group's whole union and the others' leftovers, in visiting
        // order. When narrow calls resume, rebuild every table that is far from
        // its weighted decode share so decode does not inherit that split.
        const bool vmm_transition = g_vmm.configured && !phase_wide && g_last_phase_wide;
        if (!g_vmm.configured && explicit_model && hm.pooled_model_experts > 0 && g_phase_plan) {
            if (phase_wide) g_phase_cold_growth_group.reset();
            if (!phase_wide && g_last_phase_wide) {
                rebalance_pooled_tables_locked(budget, backend);
                nominate_phase_cold_growth(budget, backend);
            }
            g_last_phase_wide = phase_wide;
        }
        const auto phase_cold_quota = [&]() -> std::size_t {
            if (hm.pooled_model_experts <= 0 || backend != g_group_backend) return 0;
            const auto key = host_moe_expert_group_key(hm);
            const bool retain_peer = g_vmm.configured || decode_rebalance_retain_peer();
            const auto peer_floors = !phase_wide && retain_peer && !phase_resident_bytes(key, backend)
                ? g_phase_plan->absent_narrow_floors([&](const HostMoeGroupKey& peer) {
                    return phase_resident_bytes(peer, backend);
                }, &key) : 0;
            return g_phase_plan->cold_quota(key, budget, g_reserved, phase_wide, peer_floors);
        };
        std::size_t quota = g_phase_plan
            ? phase_cold_quota()
            : g_group_plan
                ? (hm.pooled_model_experts > 0 ? group_quota(host_moe_expert_group_key(hm), backend, budget, layers) : 0)
                : budget / layers;
        if (!quota) return decline("complete group floor or exact pooled identity is not covered by the current scope");
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
        const bool ram_cache_admit = host_moe_cache_ram_demand_admission(
            hm.seq_len, hm.independent_decode_rows)
            && (!mapped_ram_mode() || (explicit_model && !joint && hm.seq_len == 1));
        const bool pooled_glm = explicit_model && hm.pooled_model_experts > 0;
        const int compact_group_rows = pooled_glm
            ? compact_prefill_group_rows(planned_prefill_rows) : planned_decode_rows;
        // Only narrow nonjoint GLM prefill changes execution grouping. Existing
        // joint, ordinary-wide and Qwen execution limits keep their contracts.
        const int prefill_group_limit = pooled_glm && !wide_prefill
            ? compact_group_rows : planned_decode_rows;
        const bool grouped_prefill = !joint && !bulk_prefill && hm.seq_len > 1 && prefill_group_limit > 1;
        if ((wide_prefill || grouped_prefill) && device_inputs && !finite_device_routes(hm.weights, route_count, explicit_model)) return 0;

        if (joint && device_inputs && !wide_prefill && !finite_device_routes(hm.weights, route_count, explicit_model)) return 0;
        int minimum_capacity = explicit_model && hm.pooled_model_experts > 0
            ? std::min(hm.num_experts, hm.n_used * planned_decode_rows) : hm.n_used;
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
        int graph_rows = std::max(joint ? hm.seq_len : 1, planned_decode_rows);
        if (pooled_glm) graph_rows = std::max(graph_rows, compact_group_rows);
        struct VmmCallGuard {
            bool active = false, completed = false;
            ~VmmCallGuard() { if (active && !completed) vmm_fail(); }
        } vmm_call_guard;
        if (g_vmm.requested) {
            if (!vmm_active() || g_backend_compute_failed.load(std::memory_order_acquire)
                || !g_vmm.source_certified || !explicit_model || joint || !pooled_glm
                || backend != g_group_backend || backend != g_vmm.certified_backend
                || !g_phase_plan || graph_rows != TSG_PRECISION_DECODE_COLUMNS
                || g_active_rank != g_vmm.options.budget_rank
                || vmm_group_index(host_moe_expert_group_key(hm)) < 0)
                throw std::runtime_error("VMM selected profile is unavailable; ordinary fallback is forbidden");
            std::array<std::shared_ptr<ExpertFileSource>, 3> sources;
            const void* pointers[3] = {hm.gate_data, hm.up_data, hm.down_data};
            for (unsigned p = 0; p != 3; ++p) {
                const auto found_source = g_file_sources.find(pointers[p]);
                if (found_source == g_file_sources.end() || !found_source->second
                    || stride[p] > (std::numeric_limits<std::size_t>::max)() / static_cast<std::size_t>(hm.num_experts)
                    || found_source->second->bytes() != stride[p] * static_cast<std::size_t>(hm.num_experts))
                    throw std::runtime_error("VMM projection is not its complete exact registered immutable source");
                sources[p] = found_source->second;
                for (unsigned q = 0; q != p; ++q)
                    if (sources[q]->identity() == sources[p]->identity()) throw std::runtime_error("VMM projection generations alias");
            }
            vmm_call_guard.active = true;
            if (vmm_transition) reshape_vmm_decode(hm, ids, budget, backend);
            g_last_phase_wide = phase_wide;
            quota = phase_cold_quota();
            if (!quota) throw std::runtime_error("VMM phase has no current admitted quota");
        }
        auto found = g_entries.begin();
        while (found != g_entries.end() && !(*found)->matches(hm, backend)) ++found;
        // A prefetch into this entry is published before its lookup. Creating
        // an entry can evict others, so any prefetch is drained first then.
        if (g_prefetch.entry != nullptr && (found == g_entries.end() || found->get() == g_prefetch.entry))
            drain_decode_prefetch(true);
        std::unique_ptr<CacheEntry> planned;
        int bulk_rows = bulk_prefill ? hm.seq_len : 0;
        if (found != g_entries.end()) {
            if (!bulk_prefill) bulk_rows = (*found)->bulk_rows; // N1/N3 reuse wide table and weights.
            graph_rows = std::max(graph_rows, (*found)->graph_rows);
            if (!(*found)->vmm_planned && !bulk_prefill && g_phase_plan && g_phase_cold_growth_group
                && (*found)->split_weights && (*found)->capacity >= minimum_capacity
                && (*found)->graph_rows >= graph_rows) {
                cold_grow_phase_entry(found, hm, stride, backend, budget, pool_capacity,
                    graph_rows, minimum_capacity, bulk_rows);
            }
            if ((*found)->split_weights && bulk_prefill && (*found)->bulk_rows != bulk_rows
                && (*found)->graph_rows >= graph_rows && (*found)->capacity >= minimum_capacity) {
                const auto resized = replace_wide_scratch(**found, bulk_rows, budget, layers);
                if (resized != Admission::admitted)
                    throw std::runtime_error("GLM wide scratch replacement refused/failed; persistent weights remain owned");
            }
            const bool required = (*found)->graph_rows < graph_rows || (*found)->capacity < minimum_capacity
                || (bulk_prefill && !(*found)->split_weights && (*found)->bulk_rows != bulk_rows);
            const bool optional_growth = settings.capacity_growth
                && (*found)->growth_generation != cache_configuration().growth_generation()
                && static_cast<std::size_t>((*found)->capacity) < pool_capacity;
            bool grown = false;
            if ((*found)->vmm_planned && required) {
                if ((*found)->graph_rows < graph_rows) throw std::runtime_error("VMM graph width cannot change inside its immutable profile");
                grow_vmm_required(**found, minimum_capacity, budget); grown = true;
            }
            // Pressure may retain only narrow scratch at a smaller capacity.
            // Ensure the future wide graph AFTER required union growth; the
            // earlier pre-growth replacement is ineligible while slots are few.
            if ((*found)->vmm_planned && bulk_prefill && (*found)->bulk_rows != bulk_rows) {
                const auto resized = replace_wide_scratch(**found, bulk_rows, budget, layers);
                if (resized != Admission::admitted)
                    throw std::runtime_error("VMM future-wide scratch could not be admitted after union growth");
            }
            // Phase-plan recovery is optional and narrow-only. Reuse the
            // transactional old+new path after an admitted budget increase;
            // wide union replacement keeps its existing cold fallback rules.
            const bool phase_growth = g_phase_plan && optional_growth && !bulk_prefill && !required;
            if (!(*found)->vmm_planned && settings.capacity_growth && ((!g_phase_plan && (required || optional_growth)) || phase_growth)) {
                (*found)->growth_generation = cache_configuration().growth_generation();
                ++g_growth_attempts;
                // New payload may never shrink away an old slot. In contrast
                // to the legacy cold replacement below, refusal leaves this
                // published generation available for all still-fitting rows.
                const auto peak_quota = g_phase_plan
                    ? g_phase_plan->growth_quota(group_key(**found), budget, g_reserved)
                    : g_reserved < budget ? (std::min)(quota, budget - g_reserved) : 0;
                auto expanded = create_entry(hm, stride, backend, peak_quota,
                    pool_capacity, graph_rows, std::max((*found)->capacity, minimum_capacity), bulk_rows);
                grown = grow_entry(*found, std::move(expanded), budget) == Growth::replaced;
            }
            if (required && !grown) {
                // Plan without physical allocation first. Retire the old table
                // before growing so no duplicate weight generation is resident.
                // Also the fallback when growth cannot hold old and new payloads
                // at once: losing hot slots beats declining every call until
                // the next budget change.
                planned = create_entry(hm, stride, backend, quota,
                    pool_capacity, graph_rows, minimum_capacity, bulk_rows);
                if (!planned) return decline("larger route union cannot fit its compact graph quota");
                // The cold loop below may evict a prefetch target.
                drain_decode_prefetch(true);
                erase_entry_checked(found);
                found = g_entries.end();
            }
        }
        if (found == g_entries.end())
        {
            std::unique_ptr<CacheEntry> entry;
            std::size_t maximum_capacity = pool_capacity;
            for (;;)
            {
                entry = planned ? std::move(planned) : create_entry(hm, stride, backend, quota, maximum_capacity, graph_rows, minimum_capacity, bulk_rows);
                if (!entry) return decline("compact graph type or workspace cannot fit its pool quota");
                if (entry->vmm_planned && !reclaim_phase_room(entry->bytes, 0, nullptr, budget))
                    throw std::runtime_error("VMM cold mandatory plan lacks actual shared room");
                while (g_reserved > budget - entry->bytes && !g_entries.empty())
                {
                    erase_entry_checked(std::prev(g_entries.end()));
                }
                if (g_reserved > budget - entry->bytes) return decline("other retained entries exhaust the shared cache ceiling");
                if (entry->vmm_planned) {
                    // Allocate the list node BEFORE any ticket or physical
                    // object. This stable globally retained owner also covers
                    // failed constructors, commits and cleanup acknowledgments.
                    const auto candidate_capacity = entry->capacity;
                    entry->vmm_unpublished = true;
                    g_entries.push_front(std::move(entry));
                    auto candidate = g_entries.begin();
                    Admission admission = Admission::failed;
                    try {
                        CacheStageTimer allocation(g_cache_timing, CacheStage::allocation);
#if defined(TSG_GGML_TEST_HOOKS)
                        if (g_vmm.pressure_mutation_serial != vmm_pressure_serial_at_entry && test_failure(35)) {
                            // The caller has already completed real pressure
                            // retirement in this selected call. Refuse before
                            // this candidate reserves or allocates anything.
                            g_test_vmm_mutation_failures.fetch_add(1);
                            admission = Admission::refused;
                        } else
#endif
                            admission = allocate_vmm_entry(**candidate);
                        refresh_vmm_accounting();
                    } catch (...) { vmm_fail(candidate->get()); refresh_vmm_accounting(); throw; }
                    if (admission == Admission::admitted) {
                        (*candidate)->vmm_unpublished = false;
                        (*candidate)->growth_generation = cache_configuration().growth_generation();
                        break;
                    }
                    if (admission == Admission::refused
                        && g_vmm.pressure_mutation_serial != vmm_pressure_serial_at_entry) {
                        vmm_fail(candidate->get());
                        throw std::runtime_error("VMM admission refused after this selected call published pressure mutation");
                    }
                    if (admission != Admission::refused) {
                        vmm_fail(candidate->get());
                        throw std::runtime_error("VMM physical allocation/commit failed; selected seam stops and retains cleanup owner");
                    }
                    erase_entry_checked(candidate);
                    if (g_ggml_error_count.load(std::memory_order_acquire) != vmm_call_epoch
                        || g_backend_compute_failed.load(std::memory_order_acquire)) {
                        vmm_fail(); throw std::runtime_error("VMM refused admission cleanup changed the original call error epoch");
                    }
                    if (candidate_capacity <= minimum_capacity)
                        throw std::runtime_error("VMM mandatory physical/shared admission was refused");
                    maximum_capacity = (std::max)(static_cast<std::size_t>(minimum_capacity),
                        static_cast<std::size_t>(candidate_capacity) * 3 / 4);
                    continue;
                }
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
            if (entry) {
                const auto bytes = entry->bytes;
                entry->growth_generation = cache_configuration().growth_generation();
                // Ordinary path preserves its original allocation/publication.
                g_entries.push_front(std::move(entry)); g_reserved += bytes;
            }
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
        if (entry.split_weights && bulk_prefill && entry.bulk_rows != hm.seq_len) {
            const auto resized = replace_wide_scratch(entry, hm.seq_len, budget, layers);
            if (resized != Admission::admitted)
                throw std::runtime_error("GLM cold wide scratch replacement refused/failed; persistent weights remain owned");
        }
        // Only this gated call uses the current source pointers for reads.
        // Resident slots retain their own immutable source keys across calls.
        entry.source[0] = hm.gate_data; entry.source[1] = hm.up_data; entry.source[2] = hm.down_data;
        const int maximum_execution_rows = joint || bulk_prefill ? hm.seq_len
            : grouped_prefill ? std::min(hm.seq_len, prefill_group_limit) : 1;
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
            // Multi-group calls resume speculative work during each graph.
            // Workers use only the payload cache and never need our gate.
            if (row && prefetch_hooks && prefetch_hooks->demand_begin)
                prefetch_hooks->demand_begin(prefetch_hooks->context);
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
                        if (static_cast<std::size_t>(s) < entry.prefetched.size() && entry.prefetched[static_cast<std::size_t>(s)])
                        {
                            entry.prefetched[static_cast<std::size_t>(s)] = 0;
                            ++g_prefetch.used;
                        }
                        break;
                    }
            tsg_glm_neighbor::WallTimer neighbor_prepare(neighbor);
            if (neighbor && neighbor->scalar_call()) {
                if (entry.capacity < 1 || static_cast<std::size_t>(entry.capacity) > tsg_glm_neighbor::max_slots)
                    throw std::runtime_error("Causal-route table exceeds its exact bounded slot workspace");
                const int rank = neighbor->find_rank(entry.current_source());
                if (rank < 0) throw std::runtime_error("Causal-route current source is not enrolled");
                const auto& enrolled = neighbor->storage()->source[static_cast<std::size_t>(rank)];
                if (enrolled.layer != hm.layer || enrolled.type != std::array<int,3>{hm.gate_type,hm.up_type,hm.down_type}
                    || enrolled.stride != std::array<std::uint64_t,3>{entry.stride[0],entry.stride[1],entry.stride[2]}
                    || enrolled.bytes != std::array<std::uint64_t,3>{static_cast<std::uint64_t>(hm.gate_bytes),
                        static_cast<std::uint64_t>(hm.up_bytes),static_cast<std::uint64_t>(hm.down_bytes)}
                    || !neighbor->prepare_query(static_cast<std::size_t>(rank), row_ids, static_cast<std::size_t>(selected_count)))
                    throw std::runtime_error("Causal-route query order or complete IDs refused");
                for (int s = 0; s < entry.capacity; ++s)
                    if (!neighbor->score_slot(static_cast<std::size_t>(s), entry.source_for_slot[s].source,
                        entry.expert_for_slot[s]))
                        throw std::runtime_error("Causal-route resident source identity is not enrolled");
#if defined(TSG_GGML_TEST_HOOKS)
                if (neighbor->test_fire(1)) throw std::runtime_error("Injected causal-route failure after query prepare");
#endif
            }
            // The experimental history sees only actual GPU hits and advances
            // on every narrow-demand access. Wide prefill scans never train it.
            if (victim_mode() && g_victim.policy && pooled_glm && ram_cache_admit) {
                auto source = victim_source(entry.current_source(), 0, entry.stride, entry.num_experts);
                for (int k = 0; k < selected_count; ++k) {
                    auto key = source.key; key.expert = row_ids[k];
                    g_victim.policy->observe(key, remapped[k] >= 0);
                }
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
                    if (!protected_slots[s] && (slot < 0 || (neighbor && neighbor->scalar_call()
                        ? neighbor->prefer_slot(static_cast<std::size_t>(s), entry.expert_for_slot[s], entry.last_used[s],
                            static_cast<std::size_t>(slot), entry.expert_for_slot[slot], entry.last_used[slot])
                        : entry.eviction_policy.prefer(
                        entry.expert_for_slot[s], entry.last_used[s],
                        entry.expert_for_slot[slot], entry.last_used[slot])))) slot = s;
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
            neighbor_prepare.finish();
            std::optional<VictimTransfers> victims;
            if (victim_mode() && g_victim.policy && pooled_glm && !misses.empty()) {
                victims.emplace(entry.backend);
                // Protect every incoming RAM hit before admitting any victims.
                // Full projections outlive every window and its queued upload.
                for (const auto& miss : misses) {
                    auto source = victim_source(entry.current_source(), miss.second, entry.stride, entry.num_experts);
                    if (!source.key.valid()) continue;
                    auto leases = g_payload_cache.take_triple(source.ranges, true);
                    if (leases[0]) victims->promotions.push_back({miss.first, miss.second, std::move(leases)});
                }
                if (ram_cache_admit) for (const auto& miss : misses) victims->demote(entry, miss.first);
                // D2H was queued on this same stream before any old slot bytes
                // can be overwritten. Publication waits for the upload drain.
                for (const auto& miss : misses) entry.invalidate_slot(miss.first);
                victims->upload_promotions(entry);
                misses.erase(std::remove_if(misses.begin(), misses.end(), [&](const auto& miss) {
                    return std::any_of(victims->promotions.begin(), victims->promotions.end(),
                        [&](const auto& promotion) { return promotion.slot == miss.first; });
                }), misses.end());
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
            const int variant = bulk_prefill ? entry.graph_rows : execution_rows - 1;
            auto bridge_inputs = [&] {
                for (int k = 0; k < entry.n_used; ++k)
                    for (int lane = 0; lane < execution_rows; ++lane)
                        route_major_ids[bulk_prefill ? lane * entry.n_used + k : k * execution_rows + lane]
                            = remapped[lane * entry.n_used + k];
                CacheStageTimer input_timing(g_cache_timing, CacheStage::input_bridge);
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
            };
            bool overlap_computed = false;
            struct DrainFileUploads {
                ggml_backend_t backend;
                bool active;
                ~DrainFileUploads() { if (active) sync_backend(backend); }
            } drain{entry.backend, true};
            const auto flag_on = [](const char* name) {
                const auto* value = std::getenv(name); return value && std::strcmp(value, "1") == 0;
            };
            const bool overlap_candidate = prefill_down_overlap_enabled() && file_read && bulk_prefill
                && entry.pooled && entry.split_weights && entry.n_used == 8
                && SharedCacheCharge::host_accounting_attached()
                && entry.weight_owner && entry.weight_owner->charge && entry.weight_owner->charge->is_accounted()
                && entry.wide_owner && entry.wide_owner->charge && entry.wide_owner->charge->is_accounted()
                && entry.shared_charge && entry.shared_charge->is_accounted()
                && execution_rows > TSG_PRECISION_DECODE_COLUMNS && execution_rows <= 512
                && misses.size() <= HostMoeProjectionRingPlan::max_misses && g_payload_cache_budget == 0
                && !flag_on("TS_HOST_MOE_EXPERT_CACHE_DIRECT_READS")
                && !flag_on("TS_HOST_MOE_EXPERT_CACHE_ORDERED_READS") && !prefill_ring_coalesce_enabled();
            if (overlap_candidate) {
                const auto overlap_epoch = g_ggml_error_count.load(std::memory_order_acquire);
                auto& graphs = prepare_down_overlap_graphs(entry);
                auto* full = &entry.width_graphs[variant];
                auto compute = q8_f32_execution_backend(entry.backend, full);
                const auto prefix_backend = q8_f32_execution_backend(entry.backend, &graphs.prefix);
                const auto suffix_backend = q8_f32_execution_backend(entry.backend, &graphs.suffix);
                const auto actual_type = ggml_backend_buffer_get_type(entry.weight_owner->buffer);
                // All eligibility refusal remains before the first file read,
                // upload or graph submission. A runtime failure never replays.
                if (compute == prefix_backend && compute == suffix_backend
                    && g_prefill_down_upload.prepare(entry.backend, actual_type)) {
                    if (g_ggml_error_count.load(std::memory_order_acquire) != overlap_epoch
                        || g_backend_compute_failed.load(std::memory_order_acquire))
                        throw std::runtime_error("Prefill Down overlap preflight backend failed");
                    DownOverlapActivity activity{hm.layer, execution_rows, misses.size()};
                    std::size_t arena = kFileReadWorkspace;
                    if (prefill_ring_staging_enabled() && g_file_read_buffer.try_reserve_expanded())
                        arena = kHostMoeScalarFileStagingBytes;
#if defined(TSG_GGML_TEST_HOOKS)
                    if (g_test_file_slice_bytes.load() != kFileReadWorkspace)
                        arena = (std::min)(arena, g_test_file_slice_bytes.load());
#endif
                    arena = (std::min)(arena, miss_bytes);
                    std::array<std::size_t, 3> strides{}, extents{};
                    for (int p = 0; p < 3; ++p) {
                        strides[p] = entry.stride[p]; extents[p] = static_cast<std::size_t>(files[p]->bytes());
                    }
                    HostMoeProjectionRingPlan gate(misses, strides, 3, arena, extents, entry.capacity, entry.num_experts);
                    HostMoeProjectionRingPlan down(misses, strides, 4, arena, extents, entry.capacity, entry.num_experts);
                    g_file_read_buffer.reserve(arena);
                    if ((!g_file_read_buffer.shared_charge && !g_file_read_buffer.expansion_charge)
                        || (g_file_read_buffer.shared_charge && !g_file_read_buffer.shared_charge->is_accounted())
                        || (g_file_read_buffer.expansion_charge && !g_file_read_buffer.expansion_charge->is_accounted()))
                        throw std::runtime_error("Prefill Down overlap staging is not accounted");
                    activity.arena = arena; activity.slot = gate.slot_bytes;
                    const auto epoch = overlap_epoch;
                    auto check = [&] {
                        if (g_ggml_error_count.load(std::memory_order_acquire) != epoch
                            || g_backend_compute_failed.load(std::memory_order_acquire))
                            throw std::runtime_error("Prefill Down overlap execution failed");
                    };
                    auto elapsed = [](auto start) {
                        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                    };
                    struct BothDrains {
                        ggml_backend_t upload, compute;
                        ~BothDrains() {
                            if (!host_moe_overlap_drain_both([&] { sync_backend(upload); }, [&] { sync_backend(compute); }))
                                g_backend_compute_failed.store(true, std::memory_order_release);
#if defined(TSG_GGML_TEST_HOOKS)
                            g_test_down_drains.fetch_add(1);
#endif
                        }
                    } both{g_prefill_down_upload.upload, compute};
                    ++entry.file_read_calls; ++entry.prefetch_reads;
                    auto start = std::chrono::steady_clock::now();
                    copy_projection_file_misses(entry, gate, files, entry.backend, ram_cache_admit, epoch, false, activity);
                    activity.gate_ms = elapsed(start);
                    bridge_inputs(); check();
                    if (prefetch_hooks && prefetch_hooks->before_compute)
                        prefetch_hooks->before_compute(prefetch_hooks->context);
                    check();
                    start = std::chrono::steady_clock::now();
                    ++activity.prefix;
#if defined(TSG_GGML_TEST_HOOKS)
                    g_test_down_prefix.fetch_add(1);
#endif
                    if (ggml_backend_graph_compute_async(compute, &graphs.prefix) != GGML_STATUS_SUCCESS)
                        throw std::runtime_error("Prefill Gate/Up prefix submission failed");
                    activity.prefix_ms = elapsed(start); check();
#if defined(TSG_GGML_TEST_HOOKS)
                    if (test_failure(27)) throw std::runtime_error("Injected failure after prefill prefix submission");
                    if (test_failure(29)) g_ggml_error_count.fetch_add(1, std::memory_order_acq_rel);
                    if (test_failure(32)) throw std::bad_alloc();
#endif
                    start = std::chrono::steady_clock::now();
                    copy_projection_file_misses(entry, down, files, g_prefill_down_upload.upload,
                        ram_cache_admit, epoch, true, activity);
                    activity.down_ms = elapsed(start);
                    start = std::chrono::steady_clock::now();
                    sync_backend(g_prefill_down_upload.upload); check();
                    activity.join_ms = elapsed(start);
                    start = std::chrono::steady_clock::now();
                    ++activity.suffix;
#if defined(TSG_GGML_TEST_HOOKS)
                    g_test_down_suffix.fetch_add(1);
#endif
                    {
                        CacheStageTimer compute_timing(g_cache_timing, CacheStage::compute);
                        if (compute_graph(compute, &graphs.suffix) != GGML_STATUS_SUCCESS)
                            throw std::runtime_error("Prefill Down suffix execution failed");
                    }
                    activity.suffix_ms = elapsed(start); check();
#if defined(TSG_GGML_TEST_HOOKS)
                    if (test_failure(30)) g_ggml_error_count.fetch_add(1, std::memory_order_acq_rel);
#endif
                    // Whole-call epoch survives all submissions and both drains.
                    sync_backend(g_prefill_down_upload.upload); sync_backend(compute); check();
                    for (const auto& miss : misses) entry.publish_slot(miss.first, miss.second);
                    activity.completed = true; overlap_computed = true;
#if defined(TSG_GGML_TEST_HOOKS)
                    g_test_down_completed.fetch_add(1);
#endif
                    ++g_graph_submissions;
                    if (g_cache_timing) {
                        HostMoeCacheTiming::add(g_cache_timing->graph_submissions, 1);
                        HostMoeCacheTiming::add(g_cache_timing->ring_calls, 2);
                        if (arena == kHostMoeScalarFileStagingBytes)
                            HostMoeCacheTiming::add(g_cache_timing->ring_expanded_calls, 2);
                        g_cache_timing->ring_arena_peak_bytes = (std::max)(g_cache_timing->ring_arena_peak_bytes,
                            static_cast<std::uint64_t>(arena));
                        g_cache_timing->ring_slot_peak_bytes = (std::max)(g_cache_timing->ring_slot_peak_bytes,
                            static_cast<std::uint64_t>(gate.slot_bytes));
                    }
                }
            }
            if (file_read && !overlap_computed)
            {
                ++entry.file_read_calls;
                ++entry.prefetch_reads;
                if (bounded_file_group) {
                    copy_joint_file_misses(entry, misses, files, miss_bytes,
                        host_moe_cache_direct_read_phase(bulk_prefill, execution_rows, hm.independent_decode_rows),
                        ram_cache_admit, host_moe_scalar_staging_eligible(scalar_file_staging_enabled(),
                            pooled_glm, hm.seq_len, hm.n_used, miss_bytes), pooled_glm);
                } else {
                g_file_read_buffer.reserve(miss_bytes);
                // One task per projection: splitting a scalar token's few misses
                // into smaller reads measured no faster from the page cache or
                // NVMe, and each extra piece costs its own upload launch.
                std::vector<ExpertReadTask> tasks;
                FilePayloadUploadDrain lease_drain{entry.backend, tasks};
                tasks.reserve(misses.size() * 3);
                std::size_t offset = 0;
                for (const auto& miss : misses) for (int part = 0; part < 3; ++part) {
                    tasks.push_back({files[part].get(), static_cast<std::size_t>(miss.second) * entry.stride[part],
                        g_file_read_buffer.data() + offset, entry.stride[part]});
                    cache_file_task(tasks.back(), entry.stride[part], ram_cache_admit);
                    offset += entry.stride[part];
                }
                std::vector<int> copied_parts(misses.size(), 0);
                if (g_cache_timing) HostMoeCacheTiming::add(g_cache_timing->file_windows, 1);
                CacheStageTimer read_timing(g_cache_timing, CacheStage::file_pipeline);
                read_expert_files(tasks, [&](std::size_t index) {
                    const auto i = index / 3, part = index % 3;
                    const auto& task = tasks[index];
                    entry.file_read_bytes += task.source_read_bytes;
                    if (g_cache_timing) HostMoeCacheTiming::add(g_cache_timing->file_bytes, task.source_read_bytes);
                    CacheStageTimer upload_timing(g_cache_timing, CacheStage::upload_enqueue);
                    ggml_backend_tensor_set_async(entry.backend, entry.weight[part], task.upload_data(),
                        static_cast<std::size_t>(misses[i].first) * entry.stride[part], task.bytes);
                    if (++copied_parts[i] == 3) entry.publish_slot(misses[i].first, misses[i].second);
                });
                }
            }
            else if (!overlap_computed && do_prefetch)
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
            if (!overlap_computed && !file_read && !do_prefetch) for (const auto& miss : misses)
            {
                copy_slot(entry, miss.first, miss.second);
                entry.publish_slot(miss.first, miss.second);
            }
            if (victims) victims->complete(entry, file_read && bounded_file_group);
            if (observe && !do_prefetch && !file_read)
                entry.prefetch_policy.observe_upload(miss_bytes, ExpertPrefetchPolicy::Clock::now() - upload_started);
            for (const int slot : remapped) entry.last_used[slot] = ++entry.clock;
#if defined(TSG_GGML_TEST_HOOKS)
            if (neighbor && neighbor->scalar_call() && !misses.empty() && neighbor->test_fire(2))
                throw std::runtime_error("Injected causal-route failure after actual selected miss uploads");
#endif
            // Exact active width, including when a prior larger batch grew this
            // cache. Input/routes/output each use one contiguous bridge; only
            // the small already-host-resident slot IDs need route transposition.
            if (!overlap_computed) bridge_inputs();
            auto* execute = &entry.width_graphs[variant];
#if defined(TSG_GGML_TEST_HOOKS)
            if (joint && test_failure(6)) throw std::runtime_error("Injected joint cache failure after queued inputs");
#endif
            if (!overlap_computed) ++g_graph_submissions;
            if (joint) { ++g_joint_submissions; g_joint_rows += execution_rows; }
            ggml_status compute_status = GGML_STATUS_SUCCESS;
            if (!overlap_computed && prefetch_hooks && prefetch_hooks->before_compute)
                prefetch_hooks->before_compute(prefetch_hooks->context);
            if (!overlap_computed) {
                CacheStageTimer compute_timing(g_cache_timing, CacheStage::compute);
                if (g_cache_timing) HostMoeCacheTiming::add(g_cache_timing->graph_submissions, 1);
                compute_status = compute_graph(entry.backend, execute);
            }
            if (compute_status != GGML_STATUS_SUCCESS || g_backend_compute_failed.load(std::memory_order_acquire)) {
                set_last_error("qwen4exp expert cache: CUDA expert graph failed.");
                return -1;
            }
            drain.active = false; // graph_compute drained all preceding uploads.
#if defined(TSG_GGML_TEST_HOOKS)
            if (joint && test_failure(7)) throw std::runtime_error("Injected joint cache failure before publishing output");
#endif
            {
                CacheStageTimer output_timing(g_cache_timing, CacheStage::output_bridge);
                if (device_output) {
                    copy_device_row(entry.backend, entry.width_outputs[variant], hm.moe_out,
                        static_cast<std::size_t>(row) * entry.hidden * sizeof(float), true);
                    pending_output.active = wide_prefill || joint || group;
                } else
                    ggml_backend_tensor_get(entry.width_outputs[variant], out + static_cast<std::size_t>(row) * hm.hidden,
                        0, static_cast<std::size_t>(entry.hidden) * execution_rows * sizeof(float));
            }
            g_calls += execution_rows;
            row += execution_rows;

#if defined(TSG_GGML_TEST_HOOKS)
            if (wide_prefill && test_failure(4))
                throw std::runtime_error("Injected wide expert-cache failure after a completed row");
#endif
        }
        if (entry.vmm_planned) {
            // Unlike the ordinary asynchronous return, the VMM certificate is
            // emitted only after the actual selected output and compute drain.
            sync_backend(entry.backend);
            const auto usage = host_moe_vmm::weight_usage(g_vmm.arena, entry.vmm_weight);
            if (!vmm_active() || entry.vmm_failed || !usage.usable || !usage.actual_charge_certificate
                || usage.ticket_pending || usage.quarantine || usage.mapped != entry.vmm_weight_bytes
                || g_ggml_error_count.load(std::memory_order_acquire) != vmm_call_epoch
                || g_backend_compute_failed.load(std::memory_order_acquire)) {
                vmm_fail(&entry); throw std::runtime_error("VMM selected output publication certificate failed");
            }
            if (g_cache_timing) {
                g_cache_timing->vmm_profile = 1; g_cache_timing->vmm_capacity = usage.capacity;
                g_cache_timing->vmm_physical_bytes = usage.mapped;
                g_cache_timing->vmm_pending_bytes = usage.ticket_pending;
                g_cache_timing->vmm_quarantine_bytes = usage.quarantine;
                g_cache_timing->vmm_certificate = 1;
            }
        }
        pending_output.active = false;
        vmm_call_guard.completed = true;
        cache_timing.completed = true;
        return 1;
#else
        (void)hm; (void)x; (void)ids; (void)routes; (void)out; (void)kernel_name; (void)explicit_backend; (void)prefetch_hooks; (void)neighbor;
        return 0;
#endif
    }

    int host_moe_cached_experts(const HostMoeSegment& hm, const float* x, const std::int32_t* ids,
        const float* routes, float* out, const char* kernel, ggml_backend_t backend,
        const HostMoePrefetchHooks* hooks) {
        return host_moe_cached_experts_impl(hm, x, ids, routes, out, kernel, backend, hooks, nullptr);
    }
    int host_moe_cached_experts_with_neighbor(const HostMoeSegment& hm, const float* x, const std::int32_t* ids,
        const float* routes, float* out, const char* kernel, ggml_backend_t backend,
        const HostMoePrefetchHooks* hooks, tsg_glm_neighbor::Owner* neighbor) {
        return host_moe_cached_experts_impl(hm, x, ids, routes, out, kernel, backend, hooks, neighbor);
    }

    bool host_moe_decode_prefetch_enabled()
    {
        static const bool enabled = [] {
            // Opt-in. Measured on a 16 GB laptop GPU with NVMe/page-cache misses
            // read at ~4-5 GB/s: top-10 prediction raised the decode expert hit
            // rate 79% -> ~87% but only ~47% of prefetched experts were used, and
            // the extra bytes on a bandwidth-bound read path made decode slower
            // (13-15 -> 10 tok/s). Useful where reads are latency-bound instead.
            const char* e = std::getenv("TS_HOST_MOE_DECODE_PREFETCH");
            return e != nullptr && e[0] == '1';
        }();
        return enabled;
    }

    void host_moe_cache_prefetch(const HostMoeSegment& next, const std::int32_t* predicted, int count,
                                 ggml_backend_t explicit_backend)
    {
#ifdef TSG_GGML_USE_CUDA
        if (!host_moe_decode_prefetch_enabled() || next.seq_len != 1 || predicted == nullptr || count <= 0) return;
        ggml_backend_t backend = explicit_backend != nullptr ? explicit_backend : active_backend();
        if (backend == nullptr || !ggml_backend_is_cuda(backend)) return;
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        if (cache_budget() == 0) return;
        // The previous prediction's layer normally consumed it already.
        drain_decode_prefetch(true);
        CacheEntry* entry = nullptr;
        for (const auto& candidate : g_entries)
            if (candidate->matches(next, backend)) { entry = candidate.get(); break; }
        // Pooled/split GLM tables keep their own staging contracts.
        if (entry == nullptr || entry->pooled || entry->split_weights || entry->capacity < 2) return;
        std::array<std::shared_ptr<ExpertFileSource>, 3> files;
        for (int part = 0; part < 3; ++part) {
            const auto source = g_file_sources.find(entry->source[part]);
            if (source == g_file_sources.end()
                || source->second->bytes() != entry->stride[part] * static_cast<std::size_t>(entry->num_experts)) return;
            files[part] = source->second;
        }
        auto& p = g_prefetch;
        // Keep every predicted expert that is already resident; fill the others
        // into the slots the eviction policy would give up first.
        std::vector<std::uint8_t> keep(static_cast<std::size_t>(entry->capacity), 0);
        std::vector<int> wanted;
        for (int k = 0; k < count; ++k) {
            const int expert = predicted[k];
            if (expert < 0 || expert >= entry->num_experts) return;
            bool resident = false;
            for (int s = 0; s < entry->capacity; ++s)
                if (entry->slot_matches(s, expert)) { keep[static_cast<std::size_t>(s)] = 1; resident = true; break; }
            if (!resident && std::find(wanted.begin(), wanted.end(), expert) == wanted.end()) wanted.push_back(expert);
        }
        // Never displace more than half of a layer's slots on a prediction.
        const int limit = std::min(static_cast<int>(wanted.size()), entry->capacity / 2);
        p.fills.clear();
        for (int i = 0; i < limit; ++i) {
            int victim = -1;
            for (int s = 0; s < entry->capacity; ++s)
                if (!keep[static_cast<std::size_t>(s)] && (victim < 0 || entry->eviction_policy.prefer(
                        entry->expert_for_slot[s], entry->last_used[s],
                        entry->expert_for_slot[victim], entry->last_used[victim]))) victim = s;
            if (victim < 0) break;
            keep[static_cast<std::size_t>(victim)] = 1;
            p.fills.emplace_back(victim, wanted[static_cast<std::size_t>(i)]);
        }
        if (p.fills.empty()) return;
        const std::size_t expert_bytes = entry->stride[0] + entry->stride[1] + entry->stride[2];
        const std::size_t bytes = expert_bytes * p.fills.size();
        if (p.staging == nullptr || ggml_backend_buffer_get_size(p.staging) < bytes) {
            if (p.staging) ggml_backend_buffer_free(p.staging);
            p.staging = nullptr; p.staging_charge.reset();
            const std::size_t capacity = expert_bytes * static_cast<std::size_t>(kHostMoeDecodePrefetchCandidates);
            auto charge = SharedCacheCharge::reserve(0, 3, capacity);
            if (!charge || !charge->is_accounted()) { p.fills.clear(); return; }
            ggml_backend_buffer_t staging = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), capacity);
            if (staging == nullptr || !charge->commit(capacity)) {
                if (staging) ggml_backend_buffer_free(staging);
                p.fills.clear();
                return;
            }
            p.staging = staging; p.staging_charge = std::move(charge);
        }
        if (p.upload == nullptr) {
            p.upload = ggml_backend_dev_init(ggml_backend_get_device(backend), nullptr);
            if (p.upload == nullptr) { p.fills.clear(); return; }
        }
        auto* base = static_cast<std::uint8_t*>(ggml_backend_buffer_get_base(p.staging));
        p.tasks.clear(); p.task_slot_part.clear();
        std::size_t offset = 0;
        for (const auto& [slot, expert] : p.fills) {
            // The slot no longer holds its old expert from here on, even if the
            // read fails: a failed prefetch leaves it free.
            entry->invalidate_slot(slot);
            for (int part = 0; part < 3; ++part) {
                p.tasks.push_back({files[part].get(), static_cast<std::size_t>(expert) * entry->stride[part],
                    base + offset, entry->stride[part]});
                p.task_slot_part.emplace_back(slot, part);
                offset += entry->stride[part];
            }
        }
        p.entry = entry;
        p.issued += p.fills.size();
        static std::uint64_t prefetch_calls = 0;
        if (diagnostics() && ++prefetch_calls % 2400 == 0)
            std::fprintf(stderr, "[HOSTMOE-PREFETCH] issued=%llu published=%llu used=%llu hits=%llu misses=%llu\n",
                static_cast<unsigned long long>(p.issued), static_cast<unsigned long long>(p.published),
                static_cast<unsigned long long>(p.used), static_cast<unsigned long long>(g_hits),
                static_cast<unsigned long long>(g_misses));
        p.start();
#else
        (void)next; (void)predicted; (void)count; (void)explicit_backend;
#endif
    }

    void host_moe_expert_cache_on_drop(const void* ptr)
    {
        if (ptr == nullptr) return;
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        if (cache_configuration().child_cookie() && g_file_sources.count(ptr))
            throw std::runtime_error("Causal-route source enrollment requires checked owner close before unregister");
        drain_decode_prefetch(false);
        // A VMM slot retains exact source keys. Do not release source leases
        // until every graph/table referring to this registration is retired.
        for (auto it = g_entries.begin(); it != g_entries.end();)
            if ((*it)->references(ptr)) it = erase_entry_checked(it); else ++it;
        const auto source = g_file_sources.find(ptr);
        if (source != g_file_sources.end()) {
            g_payload_cache.erase_source(source->second->identity());
            g_mapped_cache.erase_source(source->second->identity());
            if (g_mapped_probation) g_mapped_probation->erase_source(source->second->identity());
            if (g_victim.policy && g_victim.policy->events()) g_victim.policy->clear();
            g_file_sources.erase(source);
        }
    }

    // A GLM stream belongs to the native model, unlike Qwen's borrowed global
    // stream. Drain and physically free every matching entry while it is alive.
    void host_moe_expert_cache_release_backend(ggml_backend_t backend)
    {
        if (!backend) return;
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        if (cache_configuration().child_cookie())
            throw std::runtime_error("Causal-route owner requires checked final close before backend release");
        if (g_vmm.configured && !g_vmm.closed)
            throw std::runtime_error("VMM arena requires final checked close before backend release");
        if (!ExpertFileSourceViews::release_ready())
            throw std::runtime_error("Source views require final checked close before backend release");
        g_prefetch.release_resources();
        if (g_prefill_down_upload.owner == backend) g_prefill_down_upload.clear();
        for (auto it = g_entries.begin(); it != g_entries.end();) {
            if ((*it)->backend != backend) { ++it; continue; }
            it = erase_entry_checked(it);
        }
        if (backend == g_group_backend) {
            g_group_plan.reset(); g_phase_plan.reset(); g_group_backend = nullptr;
            g_last_phase_wide = false; g_phase_cold_growth_group.reset();
        }
    }

    void host_moe_expert_cache_release()
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        if (cache_configuration().child_cookie())
            throw std::runtime_error("Causal-route owner requires checked final close before global release");
        if (g_vmm.configured && !g_vmm.closed)
            throw std::runtime_error("VMM arena requires final checked close before global release");
        if (!ExpertFileSourceViews::release_ready())
            throw std::runtime_error("Source views require final checked close before global release");
        g_prefetch.release_resources();
        g_prefill_down_upload.clear();
        host_moe_cpu_staging_release();
        if (diagnostics() && g_prefetch.issued) std::fprintf(stderr,
            "[HOSTMOE-PREFETCH] issued=%llu published=%llu used=%llu\n",
            static_cast<unsigned long long>(g_prefetch.issued), static_cast<unsigned long long>(g_prefetch.published),
            static_cast<unsigned long long>(g_prefetch.used));
        if (diagnostics() && (g_calls != 0 || g_reserved != 0)) std::fprintf(stderr,
            "[HOSTMOE-CACHE] calls=%llu hits=%llu misses=%llu reserved=%zu budget=%zu release\n",
            static_cast<unsigned long long>(g_calls), static_cast<unsigned long long>(g_hits),
            static_cast<unsigned long long>(g_misses), g_reserved, cache_budget());
        if (diagnostics() && g_file_read_buffer.size())
            std::fprintf(stderr, "[HOSTMOE-FILE-WORKSPACE] bytes=%zu ceiling=%zu type=%s\n",
                g_file_read_buffer.size(), scalar_file_staging_enabled() || prefill_ring_staging_enabled()
                    ? kHostMoeScalarFileStagingBytes : kFileReadWorkspace,
                ggml_backend_buffer_name(g_file_read_buffer.buffer));
        while (!g_entries.empty()) erase_entry_checked(g_entries.begin());
        // Every owner is physically retired and its checked ticket acknowledged.
        g_group_plan.reset(); g_phase_plan.reset(); g_group_backend = nullptr;
        g_last_phase_wide = false; g_phase_cold_growth_group.reset();
        g_file_sources.clear();
        victim_diagnostics();
        g_payload_cache.set_budget(0);
        g_payload_cache_budget = 0;
        if (!configure_mapped_ram(0))
            throw std::runtime_error("Mapped expert RAM resources still require cleanup");
        g_victim.clear_metadata(); g_victim = {};
        g_file_read_buffer.clear();
        g_reserved = 0;
        g_hits = g_misses = g_calls = 0;
        g_graph_submissions = g_joint_submissions = g_joint_rows = 0;
        g_growth_attempts = g_growth_successes = g_growth_refusals = g_growth_failures = 0;
        g_growth_noops = g_growth_hot_slots = g_growth_copy_bytes = 0;
        g_growth_peak_accounted = 0;
#if defined(TSG_GGML_TEST_HOOKS)
        g_test_phase_cold_growth_attempts.store(0);
#endif
    }

    // Caller holds g_cache_mutex. Scope updates also enforce the per-layer
    // share, so a formerly large entry cannot occupy another layer's new quota.
    std::size_t host_moe_expert_cache_trim_locked(std::size_t target,
        std::size_t per_layer = (std::numeric_limits<std::size_t>::max)())
    {
        // Request-boundary maintenance also releases the one serial CPU layer's
        // reusable host arena, without dropping registered file sources.
        drain_decode_prefetch(false);
        host_moe_cpu_staging_release();
        const auto before = g_reserved;
        const auto started = std::chrono::steady_clock::now();
        int compacted = 0, erased = 0;
        // This is the existing target, never a new grant. Zero retains the
        // complete checked teardown path and leaves no reusable table owner.
        if (target && g_vmm.configured && g_reserved > target
            && compact_vmm_pressure_locked(target, nullptr)) return before - g_reserved;
        for (auto it = g_entries.begin(); per_layer != (std::numeric_limits<std::size_t>::max)()
            && it != g_entries.end();) {
            // Phase sharing imposes one aggregate ceiling. Large hot entries
            // survive a valid resize if physically affordable, irrespective of
            // the fair cold-narrow share. Ordinary/static weighted stays equal.
            const auto quota = g_phase_plan ? target
                : g_group_plan ? entry_quota(**it, target, cache_configuration().settings().layers) : per_layer;
            if ((*it)->bytes <= quota) { ++it; continue; }
            if (!g_phase_plan && !g_group_plan && compact_shrink()) {
                // A failed compaction left the old table published and owned;
                // trimming then erases it exactly as before.
                const auto compact = [&] {
                    try { return shrink_entry(*it, quota); }
                    catch (const std::exception&) { return Growth::unchanged; }
                };
                auto outcome = compact();
                if (outcome == Growth::refused && std::next(it) != g_entries.end()) {
                    // The first compaction may lack transient credit. One
                    // least-recent table pays for it; later ones are funded
                    // by the slots each compaction returns.
                    erase_entry_checked(std::prev(g_entries.end()));
                    outcome = compact();
                }
                if (outcome == Growth::replaced) { ++compacted; ++it; continue; }
            }
            ++erased;
            it = erase_entry_checked(it);
        }
        // A refused VMM admission can retain a zero-payload registry node.
        // Complete teardown must also retire such nodes before clearing Host
        // staging; an empty byte total alone does not prove an empty registry.
        while ((target == 0 || g_reserved > target) && !g_entries.empty())
        {
            // Keep failed VMM cleanup in the registry, with all native tickets.
            erase_entry_checked(std::prev(g_entries.end()));
        }
        if (g_entries.empty()) { g_prefill_down_upload.clear(); g_file_read_buffer.clear(); }
        if (diagnostics() && (compacted || erased)) std::fprintf(stderr,
            "[HOSTMOE-CACHE-TRIM] target=%zu compacted=%d erased=%d reserved=%zu ms=%.1f\n", target, compacted, erased,
            g_reserved, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
        return before - g_reserved;
    }

    std::size_t host_moe_expert_cache_trim(std::size_t target)
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        return host_moe_expert_cache_trim_locked(target);
    }

    static bool causal_configure_impl(tsg_glm_neighbor::Owner& owner, const void* model, std::uint64_t token,
        int profile, std::size_t bytes, const tsg_glm_neighbor::Config& config,
        const tsg_glm_neighbor::Source* sources, ggml_backend_t backend, bool tiny) noexcept {
        try {
            std::lock_guard<std::mutex> lock(g_cache_mutex);
            if (tsg_glm_neighbor::option() != 1 || !cache_configuration().owns(token) || !sources
                || !backend || backend != g_group_backend || !g_phase_plan || g_phase_plan->size() != 3
                || (tiny ? (config.layers != 4 || config.experts != 48 || config.routes != 8)
                    : (config.layers != 42 || config.experts != 288 || config.routes != 8))
                || g_file_sources.size() != static_cast<std::size_t>(config.layers) * 3
                || (!owner.configured() && !g_entries.empty())
                || g_vmm.configured || !ExpertFileSourceViews::release_ready()) return false;
            for (std::size_t i = 0; i < config.layers; ++i) for (std::size_t p = 0; p < 3; ++p) {
                const auto found = g_file_sources.find(sources[i].pointer[p]);
                if (found == g_file_sources.end() || found->second->bytes() != sources[i].bytes[p]) return false;
            }
            if (!cache_configuration().bind_child(token, &owner,
                [](void* value) { const auto& child = *static_cast<tsg_glm_neighbor::Owner*>(value);
                    return child.closed() && child.release_ready(); },
                [](void* value) { return static_cast<tsg_glm_neighbor::Owner*>(value)->maintenance_reset(); })) return false;
            return owner.configure(model, token, profile, bytes, config, sources);
        } catch (...) { return false; }
    }
    bool host_moe_causal_configure(tsg_glm_neighbor::Owner& owner, const void* model, std::uint64_t token,
        int profile, std::size_t bytes, const tsg_glm_neighbor::Config& config,
        const tsg_glm_neighbor::Source* sources, ggml_backend_t backend) noexcept {
        return causal_configure_impl(owner, model, token, profile, bytes, config, sources, backend, false);
    }
#if defined(TSG_GGML_TEST_HOOKS)
    bool host_moe_causal_test_configure(tsg_glm_neighbor::Owner& owner, const void* model, std::uint64_t token,
        std::size_t bytes, const tsg_glm_neighbor::Config& config,
        const tsg_glm_neighbor::Source* sources, ggml_backend_t backend) noexcept {
        return causal_configure_impl(owner, model, token, 1, bytes, config, sources, backend, true);
    }
#endif
    bool host_moe_causal_close(tsg_glm_neighbor::Owner& owner) noexcept {
        try {
            std::lock_guard<std::mutex> lock(g_cache_mutex);
            if (!owner.close()) return false;
            const auto bound_token = cache_configuration().child_token(&owner);
            if (bound_token) return cache_configuration().detach_child(bound_token, &owner);
            return true;
        } catch (...) { return false; }
    }
    bool host_moe_causal_begin(tsg_glm_neighbor::Owner& owner, bool scalar, std::int64_t head,
        int rows, std::int64_t sequence) noexcept {
        try { std::lock_guard<std::mutex> lock(g_cache_mutex);
            return cache_configuration().child_cookie() == &owner && cache_configuration().owns(owner.scope())
                && owner.begin_call(scalar, head, rows, sequence); } catch (...) { return false; }
    }
    bool host_moe_causal_commit(tsg_glm_neighbor::Owner& owner, int layer) noexcept {
        try { std::lock_guard<std::mutex> lock(g_cache_mutex);
            if (cache_configuration().child_cookie() != &owner || !cache_configuration().owns(owner.scope())
                || !owner.storage()) return false;
            const auto& config = owner.storage()->policy.config;
            for (std::size_t rank = 0; rank < config.layers; ++rank)
                if (config.layer_ids[rank] == layer) return owner.commit_query(rank);
            return false;
        } catch (...) { return false; }
    }
    bool host_moe_causal_finish(tsg_glm_neighbor::Owner& owner) noexcept {
        try { std::lock_guard<std::mutex> lock(g_cache_mutex); return owner.finish_call(); } catch (...) { return false; }
    }
    void host_moe_causal_abort(tsg_glm_neighbor::Owner& owner) noexcept {
        try { std::lock_guard<std::mutex> lock(g_cache_mutex); owner.fail_call(); } catch (...) {}
    }
    bool host_moe_causal_reset(tsg_glm_neighbor::Owner& owner, bool recover) noexcept {
        try { std::lock_guard<std::mutex> lock(g_cache_mutex);
            return recover ? owner.recover_reset() : owner.reset(); } catch (...) { return false; }
    }
    bool host_moe_causal_info(tsg_glm_neighbor::Owner& owner, std::int64_t* values, int count) noexcept {
        try { std::lock_guard<std::mutex> lock(g_cache_mutex); return owner.info(values, count); } catch (...) { return false; }
    }
    bool host_moe_causal_model_release_ready() noexcept {
        try { std::lock_guard<std::mutex> lock(g_cache_mutex);
            return cache_configuration().child_cookie() == nullptr; } catch (...) { return false; }
    }

    bool host_moe_vmm_model_release_ready() noexcept {
        try {
            std::lock_guard<std::mutex> lock(g_cache_mutex);
            return !g_vmm.configured || g_vmm.closed;
        } catch (...) { return false; }
    }
    bool host_moe_file_source_view_requested() noexcept { return file_source_view_option() == 1; }
    bool host_moe_file_source_view_model_release_ready() noexcept {
        try { std::lock_guard<std::mutex> lock(g_cache_mutex); return ExpertFileSourceViews::release_ready(); }
        catch (...) { return false; }
    }
    bool host_moe_file_source_view_execution_ready() noexcept {
        try {
            if (file_source_view_option() != 1) return file_source_view_option() == 0;
            return ExpertFileSourceViews::execution_ready();
        } catch (...) { return false; }
    }
    // Only the serialized GLM owner calls this, after its existing backend
    // drain. No VA/alias/resource/accounting transition is made by the hint.
    int host_moe_file_source_view_trim_quiescent() noexcept {
        const auto failed = [](const char* reason) noexcept {
            try { set_last_error(reason); } catch (...) {}
            return -1;
        };
        try {
            std::lock_guard<std::mutex> lock(g_cache_mutex);
            if (file_source_view_option() != 1 || ExpertFileSourceViews::release_ready()) return 0;
            const auto token = ExpertFileSourceViews::scope_token();
            if (!cache_configuration().owns(token)) return failed("Source view pressure hint lost its active scope");
            const auto before = ExpertFileSourceViews::pressure_stats();
            const int result = ExpertFileSourceViews::pressure_hint(token);
            if (result < 0) return failed(ExpertFileSourceViews::last_error());
            const auto after = ExpertFileSourceViews::pressure_stats();
            const auto usage = ExpertFileSourceViews::stats();
            const auto calls = after.range_calls - before.range_calls;
            std::fprintf(stderr, "[HOSTMOE-SOURCE-VIEW-PRESSURE] version=1 sequence=%llu outcome=%s "
                "ranges=%llu attempted=%llu accepted=%llu unexpected=%llu active_readers=%llu "
                "control_bytes=%llu held_control_ticket_bytes=%llu total_host_promise=%llu "
                "scope=working_set_hint_not_reclaimed_or_physical_free_credit\n",
                static_cast<unsigned long long>(after.attempts), result == 1 ? "completed" : calls ? "advisory" : "busy",
                static_cast<unsigned long long>(after.owned_records), static_cast<unsigned long long>(calls),
                static_cast<unsigned long long>(after.accepted_ranges - before.accepted_ranges),
                static_cast<unsigned long long>(after.unexpected_results - before.unexpected_results),
                static_cast<unsigned long long>(usage.active_readers), static_cast<unsigned long long>(usage.owned_control_bytes),
                static_cast<unsigned long long>(usage.held_control_ticket_bytes),
                static_cast<unsigned long long>(usage.total_host_promise));
            return result;
        } catch (...) { return failed("Source view pressure hint owner query failed"); }
    }
    bool host_moe_vmm_requested() noexcept {
        try { std::lock_guard<std::mutex> lock(g_cache_mutex); return g_vmm.requested; }
        catch (...) { return true; } // A failed query cannot bypass native containment.
    }
    bool host_moe_vmm_certify_model(ggml_backend_t backend) noexcept {
        try {
            std::lock_guard<std::mutex> lock(g_cache_mutex);
            if (!g_vmm.requested) return true;
            if (!vmm_active() || !backend || g_phase_plan || g_group_plan || g_reserved || !g_entries.empty()) return false;
            if (g_vmm.certified_backend && g_vmm.certified_backend != backend) return false;
            g_vmm.certified_backend = backend; return true;
        } catch (...) { return false; }
    }
    host_moe_vmm::WeightOptions vmm_weight_options(int device, int rank, std::size_t chunks) {
        host_moe_vmm::WeightOptions options{device, rank, chunks};
        // Production retains whole 16 MiB physical owners. The explicit tiny
        // fixture keeps the driver minimum so its rich boundary/fault coverage
        // remains independent of this production chunk policy.
        options.preferred_chunk_bytes = std::size_t(16) << 20;
#if defined(TSG_GGML_TEST_HOOKS)
        if (g_test_vmm_profile) options.preferred_chunk_bytes = 0;
#endif
        return options;
    }
    std::size_t vmm_host_plan(int profile, int device, int rank, std::size_t chunks) {
        host_moe_vmm::WeightHostPlan plan;
        if (profile != 1 || device != 0 || rank != 0 || chunks != 8192
            || host_moe_vmm::plan_weight_host(vmm_weight_options(device, rank, chunks), plan) != host_moe_vmm::WeightStatus::ready
            || !host_moe_vmm::weight_charge_certificate_available()) return 0;
        return split_add(plan.declared_charge_bytes, kVmmCacheControlBytes);
    }
    bool vmm_close_locked(std::uint64_t token) {
        if (!cache_configuration().owns(token)) return false;
        if (!g_vmm.configured) return !g_vmm.arena && !g_vmm.control_charge;
        if (g_vmm.token != token) return false;
        if (g_vmm.closed) return true;
        g_vmm.closing = true; // A failed final close cannot reopen request admission.
        host_moe_expert_cache_trim_locked(0);
        VmmBarrierContext barrier{nullptr, true};
        if (g_vmm.arena && !host_moe_vmm::destroy_weight_arena_checked(g_vmm.arena, vmm_barrier(barrier)))
            return false;
        g_vmm.control.reset(); // Physical metadata is gone before the final acknowledgment.
        if (g_vmm.control_charge && !g_vmm.control_charge->release_checked()) return false;
        g_vmm.control_charge.reset(); g_vmm.closed = true;
        return true;
    }
    bool vmm_configure(std::uint64_t token, int profile, int device, int rank, std::size_t chunks) {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        const auto options = vmm_weight_options(device, rank, chunks);
        const auto bytes = vmm_host_plan(profile, device, rank, chunks);
        const auto& settings = cache_configuration().settings();
        int expected_rows = 512;
#if defined(TSG_GGML_TEST_HOOKS)
        if (g_test_vmm_profile) expected_rows = 16;
#endif
        if (!bytes || !cache_configuration().owns(token) || !g_vmm.requested || g_vmm.closed
            || g_reserved || !g_entries.empty() || g_group_plan || g_phase_plan || g_group_backend
            || settings.layers != 3 || settings.max_rows != expected_rows || settings.max_decode_rows != 1
            || !SharedCacheCharge::host_accounting_attached() || !SharedCacheCharge::checked_release_attached()) return false;
        if (g_vmm.configured) return g_vmm.token == token && vmm_active() && g_vmm.declared_host == bytes
            && g_vmm.options == options;
        // At most three live entries and three simultaneous size-only resize
        // shadows. Each has one 8-variant narrow context, one wide context and
        // one descriptor-only weight context; rejected search plans die before
        // the next one is created. Slot arrays exist only on the three live
        // entries and never exceed the immutable maximum derived at startup.
        const auto context_bytes = [](std::size_t variants) {
            const auto nodes = 256 * variants;
            return split_add(ggml_tensor_overhead() * nodes, ggml_graph_overhead_custom(nodes, false));
        };
        std::size_t metadata = split_add(sizeof(VmmCacheControl), 6 * sizeof(CacheEntry));
        metadata = split_add(metadata, 6 * split_add(context_bytes(8), 2 * context_bytes(1)));
        metadata = split_add(metadata, 6 * 9 * (4 * sizeof(ggml_tensor*) + sizeof(ggml_cgraph)));
        int hidden = 4096, ff = 2048, experts = 288;
#if defined(TSG_GGML_TEST_HOOKS)
        if (g_test_vmm_profile) { hidden = ff = 512; experts = 48; }
#endif
        const ggml_type triples[3][3] = {{GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS},
            {GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_S, GGML_TYPE_IQ3_XXS}, {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ4_XS}};
        for (unsigned group = 0; group != 3; ++group) {
            const auto stride = split_add(split_add(ggml_row_size(triples[group][0], hidden) * ff,
                ggml_row_size(triples[group][1], hidden) * ff), ggml_row_size(triples[group][2], ff) * hidden);
            const auto maximum = (std::min)(static_cast<std::size_t>(45 * experts),
                cache_configuration().maximum_budget() / stride);
            if (!maximum || maximum > HostMoeVmmSlotPlan::maximum_slots) return false;
            metadata = split_add(metadata, maximum * (sizeof(int) + sizeof(ExpertSourceIdentity)
                + sizeof(std::uint64_t) + sizeof(std::uint8_t)));
        }
        // Additional bounded vector/gallocr bookkeeping policy allowance. This
        // does not certify malloc fragmentation or vendor/driver allocations.
        metadata = split_add(metadata, std::size_t(512) << 10);
        if (metadata > kVmmCacheControlBytes) return false;
        // Latch the attempt before reserving or creating anything. Any failure
        // keeps this scope alive for explicit checked cleanup, never a destructor.
        g_vmm.configured = true; g_vmm.token = token; g_vmm.declared_host = bytes;
        g_vmm.options = options;
        try {
            g_vmm.control_charge = SharedCacheCharge::reserve(rank, 3, kVmmCacheControlBytes);
            if (!g_vmm.control_charge || !g_vmm.control_charge->is_accounted()) return false;
            g_vmm.control = std::make_unique<VmmCacheControl>();
            if (!g_vmm.control_charge->commit(kVmmCacheControlBytes)) return false;
            const auto status = host_moe_vmm::create_weight_arena(g_vmm.options, g_vmm.arena);
            return status == host_moe_vmm::WeightStatus::ready && g_vmm.arena;
        } catch (...) { g_vmm.failed = true; throw; }
    }
    void vmm_stats(std::int64_t* values) {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        const auto arena = host_moe_vmm::weight_usage(g_vmm.arena, {});
        std::size_t tables = 0, owned = 0, pending = 0, quarantine = 0;
        for (const auto& entry : g_entries) if (entry->vmm_planned) {
            tables += entry->vmm_weight.table != 0;
            const auto usage = host_moe_vmm::weight_usage(g_vmm.arena, entry->vmm_weight);
            owned = split_add(owned, vmm_owned_device_bytes(*entry));
            pending = split_add(pending, usage.ticket_pending); quarantine = split_add(quarantine, usage.quarantine);
            const auto scratch = [&](const std::shared_ptr<SharedCacheCharge>& charge, bool uncertain) {
                if (!charge) return;
                if (!charge->is_committed()) pending = split_add(pending, charge->reserved_bytes());
                if (uncertain) quarantine = split_add(quarantine, charge->reserved_bytes());
            };
            scratch(entry->shared_charge, entry->vmm_scratch_uncertain);
            if (entry->wide_owner) scratch(entry->wide_owner->charge, entry->wide_owner->cleanup_uncertain);
            if (entry->vmm_pending_wide) scratch(entry->vmm_pending_wide->charge, entry->vmm_pending_wide->cleanup_uncertain);
        }
        const std::size_t host_tickets = static_cast<std::size_t>(arena.host_ticket_held)
            + static_cast<std::size_t>(static_cast<bool>(g_vmm.control_charge));
        const std::uint64_t result[12] = {1, g_vmm.configured, g_vmm.closed, g_vmm.declared_host,
            host_tickets, tables, owned, pending, quarantine, g_vmm.attempts, g_vmm.completed, g_vmm.failures};
        for (unsigned i = 0; i != 12; ++i) {
            if (result[i] > static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)()))
                throw std::overflow_error("VMM statistics exceed ABI range");
            values[i] = static_cast<std::int64_t>(result[i]);
        }
    }

    std::uint64_t host_moe_expert_cache_scope_begin(HostMoeCacheSettings settings) {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        if (file_source_view_option() < 0 || !ExpertFileSourceViews::release_ready()) return 0;
        const bool requested = vmm_flag(); // Immutable for this exclusive scope.
        const auto token = cache_configuration().begin(settings,
            g_reserved == 0 && g_entries.empty() && !g_group_plan && !g_phase_plan && g_file_read_buffer.size() == 0);
        if (token) { g_vmm = {}; g_vmm.requested = requested; }
        return token;
    }
    bool host_moe_expert_cache_scope_update(std::uint64_t token, std::size_t target, std::size_t& released) {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        if (target && g_vmm.configured && (g_vmm.closed || g_vmm.closing)) { released = 0; return false; }
        if (target && file_source_view_option() == 1 && ExpertFileSourceViews::stats().closing) { released = 0; return false; }
        // A sub-floor reconfiguration is refused before any mutation. Explicit
        // Trim(0) remains the cancellation/reclamation operation; End also works.
        if (g_phase_plan && !g_phase_plan->admits(target)) { released = 0; return false; }
        return cache_configuration().update(token, target, [&](std::size_t budget, std::size_t layers) {
            released = host_moe_expert_cache_trim_locked(budget, budget / layers);
        });
    }
    bool host_moe_expert_cache_scope_trim(std::uint64_t token, std::size_t target, std::size_t& released) {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        if (!cache_configuration().owns(token)) return false;
        if (g_vmm.configured && g_vmm.closed) { released = 0; return target == 0; }
        released = host_moe_expert_cache_trim_locked(target);
        return true;
    }
    bool host_moe_expert_cache_scope_end(std::uint64_t token, std::size_t& released) {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        return cache_configuration().end(token, [&] {
            if (g_vmm.configured) g_vmm.closing = true;
            released = host_moe_expert_cache_trim_locked(0);
            if (!vmm_close_locked(token)) throw std::runtime_error("VMM scope final cleanup still requires its owner");
            if (!ExpertFileSourceViews::release_ready() && !ExpertFileSourceViews::close(token))
                throw std::runtime_error("Source view scope final cleanup still requires its owner");
            g_group_plan.reset(); g_phase_plan.reset(); g_group_backend = nullptr;
            g_last_phase_wide = false; g_phase_cold_growth_group.reset();
            g_vmm = {};
        });
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

    HostMoeGroupKey host_moe_expert_group_key(const HostMoeSegment& hm) {
        return group_key(hm.gate_type, hm.up_type, hm.down_type, hm.hidden, hm.n_ff,
            hm.n_used, hm.num_experts, hm.pooled_model_experts, hm.swiglu_clamp);
    }

    bool host_moe_expert_cache_install_group_plan(const std::vector<HostMoeGroupMinimum>& groups,
        int prefill_rows, int decode_rows, ggml_backend_t backend) {
#ifdef TSG_GGML_USE_CUDA
        // Allocate/validate the complete candidate before touching publication.
        HostMoeGroupQuotaPlan candidate(groups);
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        const auto& settings = cache_configuration().settings();
        if (!backend || !ggml_backend_is_cuda(backend) || !cache_configuration().active()
            || g_group_plan || g_phase_plan || g_group_backend || g_reserved || !g_entries.empty()
            || candidate.size() != settings.layers || candidate.minimum() > settings.budget
            || settings.max_rows != prefill_rows || settings.max_decode_rows != decode_rows) return false;
        // Model-specific grouping is permitted only for uniform trunk geometry.
        // Different quantization triples may have different payload floors.
        const auto& first = candidate.groups().front().key;
        if (g_vmm.requested) return false; // VMM requires the independently measured phase plan.
        for (const auto& group : candidate.groups()) {
            for (std::size_t axis = 3; axis < first.values.size(); ++axis)
                if (group.key.values[axis] != first.values[axis]) return false;
            if (group.key.clamp_bits != first.clamp_bits) return false;
        }
        g_group_plan.emplace(std::move(candidate));
        g_group_backend = backend;
        if (diagnostics()) for (const auto& group : g_group_plan->groups())
            std::fprintf(stderr, "[HOSTMOE-GROUP-QUOTA] types=%d,%d,%d minimum=%zu quota=%zu budget=%zu\n",
                group.key.values[0], group.key.values[1], group.key.values[2], group.bytes,
                g_group_plan->quota(group.key, settings.budget), settings.budget);
        return true;
#else
        (void)groups; (void)prefill_rows; (void)decode_rows; (void)backend; return false;
#endif
    }

    bool host_moe_expert_cache_install_phase_plan(const std::vector<HostMoePhaseGroupMinimum>& groups,
        int prefill_rows, int decode_rows, ggml_backend_t backend) {
#ifdef TSG_GGML_USE_CUDA
        HostMoePhaseQuotaPlan candidate(groups);
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        const auto& settings = cache_configuration().settings();
        if (!backend || !ggml_backend_is_cuda(backend) || !cache_configuration().active()
            || g_group_plan || g_phase_plan || g_group_backend || g_reserved || !g_entries.empty()
            || candidate.size() != settings.layers || !candidate.admits(settings.budget)
            || prefill_rows < 1 || prefill_rows > kHostMoeCacheMaxGlmPrefillRows || decode_rows < 1 || decode_rows > 3
            || settings.max_rows != prefill_rows || settings.max_decode_rows != decode_rows) return false;
        const auto& first = candidate.groups().front().key;
        for (const auto& group : candidate.groups()) {
            int group_decode = 0, group_prefill = 0;
            if (!candidate.row_limits(group.key, decode_rows, prefill_rows, group_decode, group_prefill)) return false;
            for (std::size_t axis = 3; axis < first.values.size(); ++axis)
                if (group.key.values[axis] != first.values[axis]) return false;
            if (group.key.clamp_bits != first.clamp_bits) return false;
        }
        if (g_vmm.requested) {
            int expected_rows = 512; std::size_t required_sources = 126; bool fixture = false;
#if defined(TSG_GGML_TEST_HOOKS)
            if (g_test_vmm_profile) { expected_rows = 16; required_sources = 9; fixture = true; }
#endif
            if (!vmm_active() || (!fixture && g_vmm.certified_backend != backend)
                || prefill_rows != expected_rows || decode_rows != 1
                || candidate.size() != 3 || g_file_sources.size() < required_sources
                || (!fixture && g_file_sources.size() != required_sources)) return false;
            unsigned mask = 0;
            for (const auto& group : candidate.groups()) {
                const int index = vmm_group_index(group.key);
                if (index < 0 || (mask & (1u << index))) return false;
                int d = 0, p = 0;
                if (!candidate.row_limits(group.key, decode_rows, prefill_rows, d, p) || d != 1 || p <= 8) return false;
                mask |= 1u << index;
            }
            if (mask != 7) return false;
        }
        g_phase_plan.emplace(std::move(candidate)); // No throwing work follows publication.
        g_group_backend = backend;
        if (g_vmm.requested) {
            g_vmm.source_certified = true;
#if defined(TSG_GGML_TEST_HOOKS)
            if (g_test_vmm_profile) g_vmm.certified_backend = backend;
#endif
        }
        if (diagnostics()) for (const auto& group : g_phase_plan->groups())
            std::fprintf(stderr, "[HOSTMOE-PHASE-QUOTA] types=%d,%d,%d narrow=%zu wide=%zu aggregate_min=%zu budget=%zu decode_rows=%d prefill_rows=%d\n",
                group.key.values[0], group.key.values[1], group.key.values[2], group.narrow_bytes,
                group.wide_bytes, g_phase_plan->minimum(), settings.budget,
                group.decode_rows ? group.decode_rows : decode_rows, group.prefill_rows ? group.prefill_rows : prefill_rows);
        return true;
#else
        (void)groups; (void)prefill_rows; (void)decode_rows; (void)backend; return false;
#endif
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
        const int graph_rows = glm_compact_graph_rows(rows, cache_configuration().settings().max_rows);
        auto planned = create_entry(hm, stride, backend,
            (std::numeric_limits<std::size_t>::max)(), capacity, graph_rows, capacity);
        if (!planned || planned->capacity != capacity || planned->graph_rows != graph_rows
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
            || cache_configuration().settings().max_rows != prefill_rows
            || !host_moe_cache_prefill_fits(prefill_rows, hm.n_used, true)) return false;
        std::size_t stride[3];
        if (!valid_layout(hm, stride, rows, true)) return false;
        const int bulk_rows = prefill_rows > TSG_PRECISION_DECODE_COLUMNS ? prefill_rows : 0;
        const int graph_rows = glm_compact_graph_rows(rows, prefill_rows);
        const int capacity = std::min(hm.num_experts, hm.n_used * (bulk_rows > 0 ? std::max(rows, prefill_rows) : rows));
        auto planned = create_entry(hm, stride, backend,
            (std::numeric_limits<std::size_t>::max)(), capacity, graph_rows, capacity, bulk_rows);
        if (!planned || planned->capacity != capacity || planned->graph_rows != graph_rows
            || planned->bulk_rows != bulk_rows || ggml_gallocr_get_buffer_size(planned->allocator, 0) != 0
            || planned->shared_charge) return false;
        bytes = planned->bytes;
        return bytes > 0;
#else
        (void)hm; (void)rows; (void)prefill_rows; (void)backend;
        return false;
#endif
    }
    bool host_moe_expert_cache_minimum_phase_plan(const HostMoeSegment& hm, int rows,
        int prefill_rows, ggml_backend_t backend, std::size_t& narrow_bytes, std::size_t& wide_bytes) {
        narrow_bytes = wide_bytes = 0;
#ifdef TSG_GGML_USE_CUDA
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        if (!backend || !ggml_backend_is_cuda(backend) || !cache_configuration().active()
            || rows < 1 || rows > 3 || prefill_rows < 1 || prefill_rows > kHostMoeCacheMaxGlmPrefillRows || hm.pooled_model_experts <= 0
            || prefill_rows > cache_configuration().settings().max_rows
            || rows > cache_configuration().settings().max_decode_rows
            || !host_moe_cache_prefill_fits(prefill_rows, hm.n_used, true)) return false;
        std::size_t stride[3];
        if (!valid_layout(hm, stride, rows, true)) return false;
        const int graph_rows = glm_compact_graph_rows(rows, prefill_rows);
        const auto plan = [&](bool wide) -> std::size_t {
            const int bulk = wide && prefill_rows > TSG_PRECISION_DECODE_COLUMNS ? prefill_rows : 0;
            const int capacity = std::min(hm.num_experts, hm.n_used * (bulk ? std::max(rows, prefill_rows) : rows));
            // Keep full resource-prefill policy in the narrow plan: it uses
            // the same split owners and all retained narrow variants as runtime.
            auto entry = create_entry(hm, stride, backend, (std::numeric_limits<std::size_t>::max)(),
                capacity, graph_rows, capacity, bulk, prefill_rows, true);
            if (!entry || entry->capacity != capacity || entry->graph_rows != graph_rows
                || entry->bulk_rows != bulk || ggml_gallocr_get_buffer_size(entry->allocator, 0) != 0
                || entry->shared_charge) return 0;
            return entry->bytes;
        };
        const auto narrow = plan(false), wide = plan(true);
        if (!narrow || wide < narrow) return false;
        narrow_bytes = narrow; wide_bytes = wide; return true;
#else
        (void)hm; (void)rows; (void)prefill_rows; (void)backend; return false;
#endif
    }

    bool host_moe_expert_cache_scope_matches_plan(std::size_t minimum_bytes,
        int groups, int prefill_rows, int decode_rows)
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        const auto& settings = cache_configuration().settings();
        return cache_configuration().active() && minimum_bytes > 0 && groups > 0
            && settings.layers == static_cast<std::size_t>(groups)
            && settings.max_rows == prefill_rows && settings.max_decode_rows == decode_rows
            && compact_prefill_group_rows(settings.max_rows) == compact_prefill_group_rows(prefill_rows)
            && settings.budget >= minimum_bytes;
    }

#if defined(TSG_GGML_TEST_HOOKS)
    // Linked only into the standalone cache fixture, never a production export.
    void host_moe_expert_cache_test_fail_next(int stage) { g_test_failure_stage.store(stage); }
    std::size_t host_moe_expert_cache_test_physical_bytes() { return g_test_physical_bytes.load(); }
    void host_moe_expert_cache_test_file_slice_limit(std::size_t bytes) { g_test_file_slice_bytes.store(bytes); }
    std::size_t host_moe_expert_cache_test_file_workspace_bytes() { return g_file_read_buffer.size(); }
    void host_moe_expert_cache_test_file_workspace_fail(int stage) { g_test_file_workspace_failure.store(stage); }
    std::uint64_t host_moe_expert_cache_test_file_windows() { return g_test_file_windows.load(); }
    const void* host_moe_expert_cache_test_file_workspace_data() { return g_file_read_buffer.data(); }
    bool host_moe_expert_cache_test_try_scalar_workspace() {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        return g_file_read_buffer.try_reserve_expanded();
    }
    std::array<std::size_t, 3> host_moe_expert_cache_test_completed_ring_plan() {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        return g_test_last_completed_ring_plan;
    }
#ifdef TSG_GGML_USE_CUDA
    HostMoeFileSortTestSnapshot host_moe_expert_cache_test_file_sort_snapshot() {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        return g_test_file_sort_snapshot;
    }
#endif
    std::uint64_t host_moe_expert_cache_test_coalesced_ring_calls() { return g_test_coalesced_ring_calls.load(); }
    std::array<std::size_t, 5> host_moe_expert_cache_test_completed_coalesced_ring_plan() {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        return g_test_last_coalesced_ring_plan;
    }
    std::uint64_t host_moe_expert_cache_test_payload_upload_drains() { return g_test_payload_upload_drains.load(); }
    std::array<std::uint64_t, 5> host_moe_expert_cache_test_down_overlap_stats() {
        return {g_test_down_completed.load(), g_test_down_prefix.load(), g_test_down_suffix.load(),
            g_test_down_drains.load(), g_test_down_partial_reads.load()};
    }
    std::uint64_t host_moe_expert_cache_test_phase_cold_growth_attempts() { return g_test_phase_cold_growth_attempts.load(); }
    HostMoePayloadCache::Stats host_moe_expert_cache_test_payload_stats() {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        return g_payload_cache.stats();
    }
    void host_moe_expert_cache_test_available_bytes(std::size_t bytes) { g_test_available_bytes.store(bytes); }
    std::uint64_t host_moe_expert_cache_test_output_drains() { return g_test_completed_output_drains.load(); }
#ifdef TSG_GGML_USE_CUDA
    void host_moe_expert_cache_test_vmm_profile(bool enabled) {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        if (cache_configuration().active() || g_vmm.configured || !g_entries.empty() || g_reserved)
            throw std::logic_error("VMM fixture profile requires an empty quiescent scope");
        g_test_vmm_profile = enabled;
    }
    std::uint64_t host_moe_expert_cache_test_vmm_mutation_failures() { return g_test_vmm_mutation_failures.load(); }
    bool host_moe_expert_cache_test_copy_slot_metadata(const HostMoeSegment& hm, ggml_backend_t backend,
        HostMoeCacheTestSlot* output, std::size_t count) {
        if (!output) return false;
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        for (const auto& entry : g_entries) if (entry->matches(hm, backend)) {
            if (count != static_cast<std::size_t>(entry->capacity) || entry->expert_for_slot.size() != count) return false;
            for (std::size_t i = 0; i != count; ++i) {
                output[i].expert = entry->expert_for_slot[i]; output[i].last_used = entry->last_used[i];
                for (unsigned p = 0; p != 3; ++p)
                    output[i].source[p] = reinterpret_cast<std::uintptr_t>(entry->source_for_slot[i].source[p]);
            }
            return true;
        }
        return false;
    }
    void host_moe_expert_cache_test_split_physical_bytes(std::size_t values[3]) {
        if (!values) return;
        for (int role = 0; role < 3; ++role) values[role] = g_test_split_physical[role].load();
    }
    bool host_moe_expert_cache_test_current_split_owner(int& role, std::uintptr_t& identity, std::size_t& physical) {
        role = g_test_split_current_role; identity = g_test_split_current_identity;
        physical = g_test_split_current_physical;
        return role >= 0 && role < 3 && identity != 0;
    }

    bool host_moe_expert_cache_test_split_plan(const HostMoeSegment& hm, int capacity,
        int decode_rows, int prefill_rows, ggml_backend_t backend, HostMoeCacheTestPlan& result) {
        result = {};
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        if (capacity < hm.n_used || capacity > hm.pooled_model_experts
            || decode_rows < 1 || decode_rows > kHostMoeCacheMaxDecodeRows
            || prefill_rows < 1 || !host_moe_cache_prefill_fits(prefill_rows, hm.n_used, true)
            || !backend || !ggml_backend_is_cuda(backend) || hm.pooled_model_experts <= 0) return false;
        std::size_t stride[3];
        if (!valid_layout(hm, stride, std::max(decode_rows, prefill_rows), true)) return false;
        const auto narrow_rows = glm_compact_graph_rows(decode_rows, prefill_rows);
        const auto wide_rows = prefill_rows > TSG_PRECISION_DECODE_COLUMNS ? prefill_rows : 0;
        auto planned = create_entry(hm, stride, backend, (std::numeric_limits<std::size_t>::max)(),
            capacity, narrow_rows, capacity, wide_rows, prefill_rows);
        if (!planned || !planned->split_weights || planned->capacity != capacity
            || planned->shared_charge || planned->weight_owner->charge || planned->weight_owner->buffer
            || ggml_gallocr_get_buffer_size(planned->allocator, 0) != 0
            || (planned->wide_owner && (planned->wide_owner->charge
                || ggml_gallocr_get_buffer_size(planned->wide_owner->allocator, 0) != 0))) return false;
        result.capacity = capacity; result.narrow_rows = narrow_rows; result.wide_rows = wide_rows;
        result.weight_bytes = planned->weight_owner->bytes;
        result.narrow_bytes = planned->narrow_bytes;
        result.wide_bytes = planned->wide_owner ? planned->wide_owner->bytes : 0;
        result.weight_physical_bytes = planned->vmm_planned ? planned->vmm_weight_bytes : result.weight_bytes;
        result.narrow_physical_bytes = result.narrow_bytes - kWorkspaceAllowance;
        result.wide_physical_bytes = result.wide_bytes;
        return true;
    }

    bool host_moe_expert_cache_test_lifecycle_snapshot(const HostMoeSegment& hm,
        ggml_backend_t backend, HostMoeCacheTestSnapshot& result) {
        result = {};
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        for (const auto& value : g_entries) {
            auto& entry = *value;
            if (!entry.matches(hm, backend) || !entry.split_weights) continue;
            result.entry_identity = reinterpret_cast<std::uintptr_t>(&entry);
            result.capacity = entry.capacity; result.narrow_rows = entry.graph_rows; result.wide_rows = entry.bulk_rows;
            result.weight_bytes = entry.weight_owner->bytes; result.narrow_bytes = entry.narrow_bytes;
            result.wide_bytes = entry.wide_owner ? entry.wide_owner->bytes : 0;
            result.weight_physical_bytes = entry.weight_owner->physical_bytes;
            if (entry.vmm_planned) {
                const auto usage = host_moe_vmm::weight_usage(g_vmm.arena, entry.vmm_weight);
                result.vmm = true; result.vmm_certificate = usage.actual_charge_certificate;
                result.vmm_usable = usage.usable; result.vmm_failed = entry.vmm_failed || usage.failed;
                result.vmm_table_identity = static_cast<std::uintptr_t>(entry.vmm_weight.table);
                result.vmm_pending_bytes = usage.ticket_pending; result.vmm_quarantine_bytes = usage.quarantine;
                result.vmm_mutation_generation = entry.vmm_mutation_generation;
                result.weight_physical_bytes = usage.mapped;
                if (usage.usable) {
                    try { result.vmm_graph_binding_valid = vmm_bound_weights(entry, vmm_layout(entry, entry.capacity))
                        && static_cast<bool>(vmm_resize_proof(entry, entry.capacity)); }
                    catch (...) { result.vmm_graph_binding_valid = false; }
                }
            }
            result.narrow_physical_bytes = entry.test_physical_bytes;
            result.wide_physical_bytes = entry.wide_owner ? entry.wide_owner->physical_bytes : 0;
            result.weight_buffer = reinterpret_cast<std::uintptr_t>(entry.vmm_planned && entry.weight[0]
                ? entry.weight[0]->buffer : entry.weight_owner->buffer);
            result.narrow_buffer = reinterpret_cast<std::uintptr_t>(entry.width_inputs[0]->buffer);
            // Checked pressure retirement may retain a ticket-only wide owner
            // after its executable view was withdrawn and physical buffer freed.
            result.wide_buffer = entry.wide_owner && entry.width_inputs[entry.graph_rows]
                ? reinterpret_cast<std::uintptr_t>(entry.width_inputs[entry.graph_rows]->buffer) : 0;
            result.weight_charge = reinterpret_cast<std::uintptr_t>(entry.weight_owner->charge.get());
            result.narrow_charge = reinterpret_cast<std::uintptr_t>(entry.shared_charge.get());
            result.wide_charge = entry.wide_owner ? reinterpret_cast<std::uintptr_t>(entry.wide_owner->charge.get()) : 0;
            result.committed_owners = (entry.weight_owner->committed ? 1 : 0)
                + (entry.narrow_committed ? 1 : 0) + (entry.wide_owner && entry.wide_owner->committed ? 1 : 0);
            result.clock = entry.clock; result.growth_generation = entry.growth_generation;
            result.narrow_graph_uid = entry.width_graphs.empty() ? 0 : entry.width_graphs[0].uid;
            for (std::size_t i = 0; i < entry.width_graphs.size() && i < 8; ++i)
                result.width_graph_uid[i] = entry.width_graphs[i].uid;
            result.wide_graph_uid = entry.bulk_rows > 0
                ? entry.width_graphs[entry.graph_rows].uid : 0;
            result.wide_generation = entry.wide_generation;
            result.file_read_calls = entry.file_read_calls; result.file_read_bytes = entry.file_read_bytes;
            for (int part = 0; part < 3; ++part) {
                result.weight_data[part] = reinterpret_cast<std::uintptr_t>(entry.weight[part]->data);
                result.weight_part_bytes[part] = ggml_nbytes(entry.weight[part]);
                result.source[part] = reinterpret_cast<std::uintptr_t>(entry.source[part]);
            }
            for (const auto expert : entry.expert_for_slot) if (expert >= 0) ++result.resident;
            for (int index = 0; index < std::min(entry.capacity, 64); ++index) {
                result.slots[index].expert = entry.expert_for_slot[index];
                result.slots[index].last_used = entry.last_used[index];
                for (int part = 0; part < 3; ++part)
                    result.slots[index].source[part] = reinterpret_cast<std::uintptr_t>(entry.source_for_slot[index].source[part]);
            }
            return true;
        }
        return false;
    }

    bool host_moe_expert_cache_test_copy_weight_payload(const HostMoeSegment& hm,
        ggml_backend_t backend, int part, void* destination, std::size_t bytes) {
        if (!destination || part < 0 || part >= 3) return false;
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        for (const auto& value : g_entries) {
            auto& entry = *value;
            if (!entry.matches(hm, backend) || !entry.split_weights || bytes != ggml_nbytes(entry.weight[part])) continue;
            const auto errors_before = g_ggml_error_count.load(std::memory_order_acquire);
            sync_backend(backend);
            ggml_backend_tensor_get(entry.weight[part], destination, 0, bytes);
            return g_ggml_error_count.load(std::memory_order_acquire) == errors_before
                && !g_backend_compute_failed.load(std::memory_order_acquire);
        }
        return false;
    }

    // A size-only capacity oracle for the standalone fixture; no production ABI.
    std::size_t host_moe_expert_cache_test_plan_capacity(const HostMoeSegment& hm,
        int capacity, int rows, ggml_backend_t backend) {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        std::size_t stride[3];
        if (capacity < hm.n_used || capacity > hm.pooled_model_experts || rows < 1
            || rows > kHostMoeCacheMaxDecodeRows || !backend || !ggml_backend_is_cuda(backend)
            || !valid_layout(hm, stride, rows, true)) return 0;
        auto candidate = create_entry(hm, stride, backend, (std::numeric_limits<std::size_t>::max)(),
            capacity, glm_compact_graph_rows(rows, TSG_PRECISION_DECODE_COLUMNS), capacity, 0, TSG_PRECISION_DECODE_COLUMNS);
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
    TSG_MOE_TEST_EXPORT void TSGgml_TestMappedRamProbationClear() {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        if (g_mapped_probation) g_mapped_probation->clear();
    }
    TSG_MOE_TEST_EXPORT std::size_t TSGgml_TestMappedRamProbationKeys() {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        return g_mapped_probation ? g_mapped_probation->stats().resident_keys : 0;
    }
#undef TSG_MOE_TEST_EXPORT
#endif

    std::size_t host_moe_prefetch_capacity() {
        return g_payload_cache.stats().budget_bytes;
    }

    HostMoePayloadCache::Lease host_moe_prefetch_acquire(std::uint64_t source,
        std::uint64_t base, std::size_t extent, const HostMoePrefetchLoader& loader) {
        if (victim_mode()) return {}; // Never fill a victim tier from speculative file reads.
        return g_payload_cache.acquire_cached(source, base, extent, loader);
    }

    void host_moe_prefetch_discard() {
        g_payload_cache.discard_prefetched();
    }

    std::uint64_t host_moe_prefetch_hit_bytes() {
        return g_payload_cache.stats().prefetch_hit_bytes;
    }

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
        const char* direct = std::getenv("TS_HOST_MOE_EXPERT_CACHE_DIRECT_READS");
        auto source = std::make_shared<ExpertFileSource>(path, offset, bytes,
            direct && direct[0] == '1' && direct[1] == '\0');
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        if (g_file_sources.count(pointer)) return false;
        if (cache_configuration().child_cookie()) return false; // Immutable causal-source enrollment.
        if (!ExpertFileSourceViews::release_ready()) return false; // Enrollment is immutable after eager configuration.
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

TSG_EXPORT int TSGgml_HostMoeDecodeStagingVersion() { return 1; }
TSG_EXPORT int TSGgml_HostMoeDecodeStagingEnabled() { return tsg::scalar_file_staging_enabled() ? 1 : 0; }
TSG_EXPORT int TSGgml_HostMoePrefillRingStagingVersion() { return 1; }
TSG_EXPORT int TSGgml_HostMoePrefillRingStagingEnabled() { return tsg::prefill_ring_staging_enabled() ? 1 : 0; }
TSG_EXPORT int TSGgml_HostMoePrefillDownOverlapVersion() { return 1; }
TSG_EXPORT int TSGgml_HostMoePrefillDownOverlapEnabled() { return tsg::prefill_down_overlap_enabled() ? 1 : 0; }
#if defined(TSG_GGML_TEST_HOOKS) && defined(TSG_GGML_USE_CUDA)
TSG_EXPORT int TSGgml_TestHostMoePrefillDownOverlapFailNext(int stage) {
    if (stage != 0 && (stage < 26 || stage > 32)) return 0;
    tsg::g_test_failure_stage.store(stage, std::memory_order_release); return 1;
}
TSG_EXPORT int TSGgml_TestHostMoePrefillDownOverlapStats(std::uint64_t* values, int length) {
    if (!values || length != 5) return 0;
    const auto stats = tsg::host_moe_expert_cache_test_down_overlap_stats();
    std::copy(stats.begin(), stats.end(), values); return 1;
}
#endif
TSG_EXPORT int TSGgml_HostMoeMappedRamVersion() { return 1; }
TSG_EXPORT int TSGgml_HostMoeMappedRamEnabled() { return tsg::mapped_ram_mode() ? 1 : 0; }

// A quiescent caller owns the persistent host envelope. All exact-read workers
// are joined before releasing g_cache_mutex, including exceptional paths.
TSG_EXPORT int TSGgml_HostMoeRamCacheConfigure(std::int64_t bytes)
{
    if (bytes < 0 || static_cast<std::uint64_t>(bytes) > std::numeric_limits<std::size_t>::max()) return 0;
    try {
        std::lock_guard<std::mutex> lock(tsg::g_cache_mutex);
        if (bytes && !tsg::SharedCacheCharge::host_accounting_attached()) return 0;
        tsg::drain_decode_prefetch(false);
        const auto* mapped = std::getenv("TS_HOST_MOE_RAM_CACHE_MAPPED");
        const bool requested = mapped && std::strcmp(mapped, "1") == 0;
        if (bytes && requested != tsg::mapped_ram_mode())
            throw std::runtime_error("Mapped RAM request differs from the native process latch");
        if (tsg::mapped_ram_mode() || tsg::g_mapped_control)
            return tsg::configure_mapped_ram(static_cast<std::size_t>(bytes)) ? 1 : 0;
        if (tsg::victim_mode()) tsg::g_victim.configure(static_cast<std::size_t>(bytes));
        else {
            tsg::g_payload_cache.set_budget(static_cast<std::size_t>(bytes));
            tsg::g_payload_cache_budget = static_cast<std::size_t>(bytes);
        }
        return tsg::g_payload_cache.stats().reserved_bytes + tsg::g_victim.metadata_bytes()
            <= static_cast<std::uint64_t>(bytes) ? 1 : 0;
    } catch (const std::exception& error) {
        tsg::set_last_error(std::string("Host expert RAM cache configuration failed: ") + error.what());
        return 0;
    }
}

TSG_EXPORT int TSGgml_HostMoeRamCacheStats(std::int64_t* values, int length)
{
    if (!values || length != 9) return 0;
    std::lock_guard<std::mutex> lock(tsg::g_cache_mutex);
    if (tsg::mapped_ram_mode()) {
        const auto s = tsg::g_mapped_cache.stats();
        const std::uint64_t fields[] = {tsg::g_mapped_capacity, s.resident_logical_bytes,
            s.reserved_bytes + tsg::mapped_metadata_bytes(),
            s.hits, s.misses, s.hit_bytes, tsg::g_mapped_source_bytes.load(std::memory_order_relaxed),
            s.bypass_logical_bytes, s.evictions};
        for (int i = 0; i < length; ++i)
            values[i] = static_cast<std::int64_t>((std::min)(fields[i], std::uint64_t(INT64_MAX)));
        return 1;
    }
    const auto s = tsg::g_payload_cache.stats();
    tsg::victim_diagnostics();
    if (tsg::diagnostics()) std::fprintf(stderr,
        "[HOSTMOE-RAM-STORAGE] retained=%zu physically_pinned=%zu includes_live_leases=1\n",
        s.reserved_bytes, s.pinned_bytes);
    const std::uint64_t fields[] = {tsg::victim_mode() ? tsg::g_victim.configured : s.budget_bytes,
        s.resident_bytes, s.reserved_bytes + tsg::g_victim.metadata_bytes(),
        s.hits, s.misses, s.hit_bytes, s.source_bytes, s.bypass_bytes, s.evictions};
    for (int i = 0; i < length; ++i)
        values[i] = static_cast<std::int64_t>(std::min(fields[i],
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())));
    return 1;
}

// v1 typed mapped residency view. This is separate from the nine-field logical
// source-cache ABI; page locks are Windows residency, never CUDA host pinning.
TSG_EXPORT int TSGgml_HostMoeMappedRamStats(std::int64_t* values, int length)
{
    if (!values || length != 16 || !tsg::mapped_ram_mode()) return 0;
    std::lock_guard<std::mutex> lock(tsg::g_cache_mutex);
    const auto s = tsg::g_mapped_cache.stats();
    const auto p = tsg::HostMoeMappedRam::instance().stats();
    const auto control = tsg::mapped_metadata_bytes();
    const std::uint64_t fields[] = {tsg::g_mapped_capacity, s.resident_logical_bytes,
        s.reserved_bytes + control, p.locked_page_bytes,
        control + s.owned_entries * tsg::HostMoeMappedRam::entry_metadata_bytes,
        s.entries, s.active_leases, p.quarantine_charge_bytes, s.hits, s.misses,
        s.hit_bytes, s.admitted_logical_bytes, s.bypass_logical_bytes, s.evictions,
        s.pending_admissions, p.quota_restore_pending ? 1u : 0u};
    for (int i = 0; i < length; ++i)
        values[i] = static_cast<std::int64_t>((std::min)(fields[i], std::uint64_t(INT64_MAX)));
    if (tsg::diagnostics()) {
        std::fprintf(stderr, "[HOSTMOE-RAM-MAPPED] capacity_charge_bytes=%llu logical_resident_bytes=%llu "
            "owned_charge_bytes=%llu locked_page_bytes=%llu metadata_charge_bytes=%llu resident_entries=%llu "
            "active_leases=%llu quarantine_charge_bytes=%llu hits=%llu misses=%llu hit_copy_bytes=%llu "
            "admitted_logical_bytes=%llu bypass_logical_bytes=%llu evictions=%llu pending_admissions=%llu "
            "quota_restore_pending=%llu quota_conflict=%u cleanup_failures=%llu\n",
            static_cast<unsigned long long>(fields[0]), static_cast<unsigned long long>(fields[1]),
            static_cast<unsigned long long>(fields[2]), static_cast<unsigned long long>(fields[3]),
            static_cast<unsigned long long>(fields[4]), static_cast<unsigned long long>(fields[5]),
            static_cast<unsigned long long>(fields[6]), static_cast<unsigned long long>(fields[7]),
            static_cast<unsigned long long>(fields[8]), static_cast<unsigned long long>(fields[9]),
            static_cast<unsigned long long>(fields[10]), static_cast<unsigned long long>(fields[11]),
            static_cast<unsigned long long>(fields[12]), static_cast<unsigned long long>(fields[13]),
            static_cast<unsigned long long>(fields[14]), static_cast<unsigned long long>(fields[15]),
            p.quota_conflict ? 1u : 0u, static_cast<unsigned long long>(p.cleanup_failures));
#define TSG_MAPPED_COST_FIELDS(X) \
    X(plan_calls) X(plan_nanoseconds) X(admission_calls) X(admission_successes) X(admission_nanoseconds) \
    X(quota_claim_calls) X(quota_claim_wait_nanoseconds) \
    X(quota_read_calls) X(quota_read_nanoseconds) X(quota_write_calls) X(quota_write_nanoseconds) \
    X(quota_expand_writes) X(quota_restore_writes) X(quota_reuse_claims) \
    X(memory_query_calls) X(memory_query_nanoseconds) \
    X(duplicate_calls) X(duplicate_nanoseconds) X(section_calls) X(section_nanoseconds) \
    X(map_calls) X(map_nanoseconds) X(protection_query_calls) X(protection_query_nanoseconds) \
    X(lock_calls) X(lock_nanoseconds) X(page_query_calls) X(page_query_nanoseconds) \
    X(retire_calls) X(retire_nanoseconds) \
    X(guarded_copy_calls) X(guarded_copy_bytes) X(guarded_copy_failures) X(guarded_copy_nanoseconds) \
    X(peak_claim_bytes) X(quota_min_high_water_bytes)
#define TSG_MAPPED_COST_FORMAT(name) " " #name "=%llu"
#define TSG_MAPPED_COST_ARGUMENT(name) , static_cast<unsigned long long>(p.name)
        std::fprintf(stderr, "[HOSTMOE-RAM-MAPPED-COST]"
            TSG_MAPPED_COST_FIELDS(TSG_MAPPED_COST_FORMAT)
            " probation_observations=%llu probation_firsts=%llu probation_repeats=%llu probation_keys=%zu "
            "probation_metadata_bytes=%zu scope=nested_host_wall_not_additive_or_physical_io\n"
            TSG_MAPPED_COST_FIELDS(TSG_MAPPED_COST_ARGUMENT),
            static_cast<unsigned long long>(tsg::g_probation_observations.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(tsg::g_probation_firsts.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(tsg::g_probation_repeats.load(std::memory_order_relaxed)),
            tsg::g_mapped_probation ? tsg::g_mapped_probation->stats().resident_keys : 0,
            tsg::g_mapped_probation_charge ? tsg::HostMoeRamProbation<>::metadata_forecast_bytes : 0);
#undef TSG_MAPPED_COST_FORMAT
#undef TSG_MAPPED_COST_ARGUMENT
#undef TSG_MAPPED_COST_FIELDS
    }
    return 1;
}

TSG_EXPORT int TSGgml_HostMoeVictimCacheStats(std::int64_t* values, int length)
{
    if (!values || length != 16) return 0;
    std::lock_guard<std::mutex> lock(tsg::g_cache_mutex);
    const auto s = tsg::g_payload_cache.stats();
    const auto& v = tsg::g_victim;
    const std::uint64_t fields[] = {v.configured, v.metadata_bytes(), s.reserved_bytes,
        v.candidates, v.eligible, v.admitted, v.refusals, v.d2h_bytes, v.completed_bytes,
        v.promoted_bytes, v.failures, v.policy ? v.policy->events() : 0,
        v.policy ? v.policy->keys() : 0, s.external_bytes, s.entries, s.pinned_bytes};
    for (int i = 0; i < length; ++i)
        values[i] = static_cast<std::int64_t>(std::min(fields[i],
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())));
    return 1;
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
TSG_EXPORT int TSGgml_HostMoeFileSourceViewVersion() { return 1; }
TSG_EXPORT int TSGgml_HostMoeFileSourceViewEnabled() { return tsg::file_source_view_option(); }
TSG_EXPORT int TSGgml_HostMoeFileSourceViewHostPlan(int profile, int sources,
    std::int64_t source_bytes, std::int64_t* values, int length) {
    if (!values || length < 8) return cache_scope_failure("Source view Host plan requires eight values");
    std::fill(values, values + 8, 0);
    try {
        tsg::ExpertFileSourceViewPlan plan;
        // Pure pre-load planning: no cache token, budget attachment, source
        // handle, file payload, mapping or native buffer allocation required.
        if (tsg::file_source_view_option() < 0 || profile != 1 || sources != 126 || source_bytes <= 0
            || !tsg::ExpertFileSourceViews::plan(sources, static_cast<std::uint64_t>(source_bytes), plan))
            return cache_scope_failure("Source view Host plan profile or geometry refused");
        const std::uint64_t fields[] = {plan.version, plan.total_host_promise, plan.owned_control_bytes,
            plan.pte_forecast_bytes, plan.sources, plan.source_bytes, plan.page_bytes, plan.map_granularity};
        for (unsigned i = 0; i < 8; ++i) {
            if (fields[i] > INT64_MAX) return cache_scope_failure("Source view Host plan overflow");
            values[i] = static_cast<std::int64_t>(fields[i]);
        }
        return 1;
    } catch (...) { return cache_scope_failure("Source view Host planning failed"); }
}
TSG_EXPORT int TSGgml_ConfigureHostMoeFileSourceView(std::uint64_t token, int profile,
    int sources, std::int64_t source_bytes, std::int64_t total_host_promise) {
    try {
        std::lock_guard<std::mutex> lock(tsg::g_cache_mutex);
        const auto& settings = tsg::cache_configuration().settings();
        if (tsg::file_source_view_option() != 1 || profile != 1 || sources != 126 || source_bytes <= 0
            || total_host_promise <= 0 || !tsg::cache_configuration().owns(token) || tsg::g_vmm.requested
            || settings.layers != 3 || settings.max_rows != 512 || settings.max_decode_rows != 1
            || tsg::g_reserved || !tsg::g_entries.empty() || !tsg::g_phase_plan
            || tsg::g_file_sources.size() != static_cast<std::size_t>(sources))
            return cache_scope_failure("Source view configuration requires the exact registered empty selected profile");
        for (const char* flag : {
            "TS_HOST_MOE_EXPERT_CACHE_DIRECT_READS", "TS_HOST_MOE_EXPERT_CACHE_ORDERED_READS",
            "TS_HOST_MOE_PREFILL_RING_COALESCE16", "TS_HOST_MOE_PREFILL_DOWN_OVERLAP",
            "TS_HOST_MOE_EXPERT_CACHE_PREFETCH", "TS_HOST_MOE_LAYER_PREFETCH", "TS_HOST_MOE_DECODE_PREFETCH",
            "TS_HOST_MOE_RAM_CACHE_MB", "TS_HOST_MOE_RAM_CACHE_MAPPED", "TS_HOST_MOE_RAM_CACHE_PINNED",
            "TS_HOST_MOE_RAM_CACHE_DIRECT_UPLOAD", "TS_HOST_MOE_RAM_CACHE_VICTIM", "TS_HOST_MOE_RAM_CACHE_FREQUENCY",
            "TS_HOST_MOE_RAM_CACHE_PREFETCH_ONLY", "TS_HOST_MOE_FILE_READ_SORT", "TS_HOST_MOE_FILE_RANDOM_ACCESS",
            "TS_HOST_MOE_FILE_READ_WORKERS", "TS_HOST_MOE_DECODE_REBALANCE_RETAIN_PEER",
            "TS_HOST_MOE_PHASE_COLD_GROWTH"}) {
            const char* value = std::getenv(flag);
            if (value && std::strcmp(value, "0") != 0)
                return cache_scope_failure("Source view experiment conflicts with another file/RAM service mode");
        }
        // LFU is enabled by its legacy unset default; this isolated profile
        // therefore requires an explicit zero rather than interpreting unset
        // as off. The ordinary default-OFF cache keeps its original behavior.
        const char* lfu = std::getenv("TS_HOST_MOE_EXPERT_CACHE_LFU");
        if (!lfu || std::strcmp(lfu, "0") != 0)
            return cache_scope_failure("Source view profile requires TS_HOST_MOE_EXPERT_CACHE_LFU=0");
        tsg::ExpertFileSourceViewPlan plan;
        if (!tsg::ExpertFileSourceViews::plan(sources, static_cast<std::uint64_t>(source_bytes), plan)
            || plan.total_host_promise != static_cast<std::uint64_t>(total_host_promise))
            return cache_scope_failure("Source view durable Host promise differs from its actual native plan");
        std::vector<std::shared_ptr<tsg::ExpertFileSource>> exact;
        exact.reserve(static_cast<std::size_t>(sources));
        for (const auto& item : tsg::g_file_sources) exact.push_back(item.second);
        if (!tsg::ExpertFileSourceViews::configure(token, plan, exact))
            return cache_scope_failure(tsg::ExpertFileSourceViews::last_error());
        return 1;
    } catch (...) { return cache_scope_failure("Source view configuration failed; retain owner for checked cleanup"); }
}
TSG_EXPORT int TSGgml_CloseHostMoeFileSourceView(std::uint64_t token) {
    try {
        std::lock_guard<std::mutex> lock(tsg::g_cache_mutex);
        if (!tsg::cache_configuration().owns(token)) return cache_scope_failure("Source view close scope token differs");
        tsg::drain_decode_prefetch(false);
        return tsg::ExpertFileSourceViews::release_ready() || tsg::ExpertFileSourceViews::close(token)
            ? 1 : cache_scope_failure(tsg::ExpertFileSourceViews::last_error());
    } catch (...) { return cache_scope_failure("Source view close failed; complete owner remains retained"); }
}
TSG_EXPORT int TSGgml_HostMoeFileSourceViewStats(std::uint64_t token, std::int64_t* values, int length) {
    if (!values || length < 16) return cache_scope_failure("Source view statistics require sixteen values");
    std::fill(values, values + 16, 0);
    try {
        std::lock_guard<std::mutex> lock(tsg::g_cache_mutex);
        if (!tsg::cache_configuration().owns(token)) return cache_scope_failure("Source view statistics scope token differs");
        const auto s = tsg::ExpertFileSourceViews::stats();
        const std::uint64_t fields[] = {s.version, s.configured, s.closing, s.closed, s.owned_records,
            s.live_views, s.active_readers, s.mapped_va_bytes, s.owned_control_bytes, s.held_control_ticket_bytes,
            s.pending_records, s.quarantined_records, s.disabled_records, s.map_calls, s.view_close_calls, s.total_host_promise};
        for (unsigned i = 0; i < 16; ++i) {
            if (fields[i] > INT64_MAX) return cache_scope_failure("Source view statistics overflow");
            values[i] = static_cast<std::int64_t>(fields[i]);
        }
        return 1;
    } catch (...) { return cache_scope_failure("Source view statistics failed"); }
}
TSG_EXPORT int TSGgml_HostMoeFileSourceViewPressureStats(std::uint64_t token, std::int64_t* values, int length) {
    if (!values || length < 8) return cache_scope_failure("Source view pressure statistics require eight values");
    std::fill(values, values + 8, 0);
    try {
        std::lock_guard<std::mutex> lock(tsg::g_cache_mutex);
        if (!tsg::cache_configuration().owns(token)) return cache_scope_failure("Source view pressure scope token differs");
        const auto s = tsg::ExpertFileSourceViews::pressure_stats();
        const std::uint64_t fields[] = {s.version, s.attempts, s.completed, s.range_calls,
            s.accepted_ranges, s.unexpected_results, s.busy_refusals, s.owned_records};
        for (unsigned i = 0; i < 8; ++i) {
            if (fields[i] > INT64_MAX) return cache_scope_failure("Source view pressure statistics overflow");
            values[i] = static_cast<std::int64_t>(fields[i]);
        }
        return 1;
    } catch (...) { return cache_scope_failure("Source view pressure statistics failed"); }
}
TSG_EXPORT int TSGgml_HostMoeVmmWeightsVersion() { return 1; }
TSG_EXPORT int TSGgml_HostMoeVmmWeightsHostPlan(int profile, int device, int rank,
    std::int64_t chunks, std::int64_t* bytes) {
    if (!bytes) return cache_scope_failure("VMM Host plan requires an output");
    *bytes = 0;
    try {
        if (!cache_scope_bytes(chunks)) return cache_scope_failure("Invalid VMM chunk bound");
        *bytes = static_cast<std::int64_t>(tsg::vmm_host_plan(profile, device, rank, static_cast<std::size_t>(chunks)));
        return *bytes > 0 ? 1 : cache_scope_failure("VMM profile or checked ownership capability is unavailable");
    } catch (...) { return cache_scope_failure("VMM Host planning failed"); }
}
TSG_EXPORT int TSGgml_ConfigureHostMoeVmmWeights(std::uint64_t token, int profile, int device,
    int rank, std::int64_t chunks) {
    try {
        if (!cache_scope_bytes(chunks) || !tsg::vmm_configure(token, profile, device, rank, static_cast<std::size_t>(chunks)))
            return cache_scope_failure("VMM configuration refused; retain scope until checked close succeeds");
        return 1;
    } catch (...) { return cache_scope_failure("VMM configuration failed; checked cleanup is required"); }
}
TSG_EXPORT int TSGgml_CloseHostMoeVmmWeights(std::uint64_t token) {
    try {
        std::lock_guard<std::mutex> lock(tsg::g_cache_mutex);
        return tsg::vmm_close_locked(token) ? 1 : cache_scope_failure("VMM final cleanup remains owned");
    } catch (...) { return cache_scope_failure("VMM final cleanup failed; retain owner and retry"); }
}
TSG_EXPORT int TSGgml_HostMoeVmmWeightsStats(std::int64_t* values, int length) {
    if (!values || length < 12) return cache_scope_failure("VMM statistics require twelve values");
    try { tsg::vmm_stats(values); return 1; }
    catch (...) { return cache_scope_failure("VMM statistics failed"); }
}

TSG_EXPORT int TSGgml_BeginHostMoeExpertCacheScopeWithCapacityGrowth(std::int64_t bytes,
    int max_rows, int layers, int max_decode_rows, int enable_capacity_growth, std::uint64_t* token)
{
    if (!token) return cache_scope_failure("Expert-cache scope requires an output token");
    *token = 0;
    try {
        tsg::clear_last_error();
        if (!cache_scope_bytes(bytes) || layers <= 0 || max_rows <= 0 || max_rows > tsg::kHostMoeCacheMaxGlmPrefillRows
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
