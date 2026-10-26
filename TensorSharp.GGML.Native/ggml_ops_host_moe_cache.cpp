// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Optional CUDA expert tiers for Qwen4Exp and DeepSeek V4.1. GGUF bytes remain quantized and are
// copied into compact, persistent slots; ordinary upstream ggml CUDA kernels
// compute the complete routed FFN. No CPU/GPU partial-sum split is involved.
#include "ggml_ops_internal.h"
#include "ggml_ops_precision_policy.h"
#include "dsv41_engram_io.h"
#include "ggml-impl.h"
#ifdef TSG_GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cerrno>
#include <cstdlib>
#include <iterator>
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
        std::size_t g_host_reserved = 0;
        std::uint64_t g_host_hits = 0, g_source_reads = 0, g_host_evictions = 0;

        enum class CacheFamily { Qwen4Exp, DeepSeek41 };

        struct CachePolicy
        {
            CacheFamily family = CacheFamily::Qwen4Exp;
            ggml_backend_t backend = nullptr;
            std::size_t budget = 0, layers = 0;
        };

        const char* family_name(CacheFamily family)
        {
            return family == CacheFamily::DeepSeek41 ? "deepseek41" : "qwen4exp";
        }

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

        std::size_t cache_budget()
        {
            static const std::size_t bytes = [] {
                const std::size_t mb = read_positive_size("TS_HOST_MOE_EXPERT_CACHE_MB", 0);
                const auto maximum = static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
                return mb <= maximum / kMiB ? mb * kMiB : 0;
            }();
            return bytes;
        }

        std::size_t cache_layer_count()
        {
            // Existing Qwen4Exp callers keep their original quota. Its production
            // checkpoint has 48 layers; DeepSeek supplies its own local count.
            static const std::size_t layers = read_positive_size("TS_HOST_MOE_EXPERT_CACHE_LAYERS", 48);
            return layers;
        }

        std::size_t host_cache_budget()
        {
            const auto mb = read_positive_size("TS_DSV41_EXPERT_HOST_CACHE_MB",0);
            const auto maximum = static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
            return mb <= maximum/kMiB ? mb*kMiB : 0;
        }

        std::size_t deepseek_cache_budget()
        {
            static const std::size_t bytes = [] {
                const std::size_t mb = read_positive_size("TS_DSV41_EXPERT_CACHE_MB", 0);
                const auto maximum = static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
                return mb <= maximum / kMiB ? mb * kMiB : 0;
            }();
            return bytes;
        }

        bool policy_for(const char* kernel_name, ggml_backend_t backend,
            std::size_t budget_per_device, std::size_t layer_count, CachePolicy& policy)
        {
            if (kernel_name == nullptr) return false;
            if (std::strcmp(kernel_name, "qwen4exp fused token span") == 0)
            {
                policy.backend = backend != nullptr ? backend : g_backend;
                policy.budget = cache_budget(); policy.layers = cache_layer_count();
            }
            else if (std::strcmp(kernel_name, "deepseek41 native") == 0)
            {
                // The model reserves this explicit per-device ceiling before
                // weight placement. It cannot inherit Qwen's opt-in or infer a
                // device from the global operation bridge.
                if (backend == nullptr || budget_per_device == 0 || layer_count == 0) return false;
                policy.family = CacheFamily::DeepSeek41; policy.backend = backend;
                policy.budget = std::min(deepseek_cache_budget(), budget_per_device);
                policy.layers = layer_count;
            }
            else return false;
            return policy.backend != nullptr && policy.budget != 0 && policy.layers != 0;
        }

        bool diagnostics(CacheFamily family = CacheFamily::Qwen4Exp)
        {
            static const bool qwen_enabled = [] {
                const char* e = std::getenv("TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS");
                return e != nullptr && e[0] == '1';
            }();
            static const bool deepseek_enabled = [] {
                const char* e = std::getenv("TS_DSV41_EXPERT_CACHE_DIAGNOSTICS");
                return e != nullptr && e[0] == '1';
            }();
            return family == CacheFamily::DeepSeek41 ? deepseek_enabled : qwen_enabled;
        }

        bool prefetch_enabled(CacheFamily family)
        {
            static const bool qwen = [] {
                const char* e = std::getenv("TS_HOST_MOE_EXPERT_CACHE_PREFETCH");
                return e != nullptr && e[0] == '1';
            }();
            static const bool deepseek = [] {
                const char* e = std::getenv("TS_DSV41_EXPERT_CACHE_PREFETCH");
                return e != nullptr && e[0] == '1';
            }();
            return family == CacheFamily::DeepSeek41 ? deepseek : qwen;
        }

        struct CacheEntry;
        struct HostVictim
        {
            const CacheEntry * owner;
            int expert;
            std::size_t bytes;
            // Pageable anonymous storage, never a pinned mirror of the model.
            std::unique_ptr<std::uint8_t[]> data;
        };
        std::list<HostVictim> g_host_victims;
        void drop_host_entry(const CacheEntry * owner)
        {
            for (auto it=g_host_victims.begin();it!=g_host_victims.end();)
                if (it->owner==owner) { g_host_reserved-=it->bytes; it=g_host_victims.erase(it); }
                else ++it;
        }
        void trim_host(std::size_t ceiling)
        {
            while (g_host_reserved>ceiling && !g_host_victims.empty())
            {
                g_host_reserved-=g_host_victims.back().bytes;
                g_host_victims.pop_back(); ++g_host_evictions;
            }
        }

        struct CacheEntry
        {
            ggml_backend_t backend = nullptr;
            CacheFamily family = CacheFamily::Qwen4Exp;
            std::size_t policy_budget = 0, policy_layers = 0;
            int layer = -1, hidden = 0, n_ff = 0, n_used = 0, num_experts = 0;
            float clamp_limit = 0.0f;
            const void* source[3] = {};
            int type[3] = {};
            std::size_t stride[3] = {};
            tsg_dsv41_pread::file_range files[3];
            std::size_t bytes = 0;
            int capacity = 0;
            std::vector<int> expert_for_slot;
            std::vector<std::uint64_t> last_used;
            std::uint64_t clock = 0;
            ggml_context* ctx = nullptr;
            ggml_gallocr_t allocator = nullptr;
            ggml_cgraph* graph = nullptr;
            ggml_tensor* weight[3] = {};
            ggml_tensor* input = nullptr;
            ggml_tensor* ids = nullptr;
            ggml_tensor* routes = nullptr;
            ggml_tensor* output = nullptr;

            ~CacheEntry()
            {
                drop_host_entry(this);
                // The last direct output copy can still be queued after the
                // helper returns. Drain it before eviction/invalidation frees
                // any source storage; CUDA remains alive at both release hooks.
                if (allocator != nullptr)
                {
                    sync_backend(backend);
                    ggml_gallocr_free(allocator);
                }
                if (ctx != nullptr) ggml_free(ctx);
            }

            bool same_domain(const CachePolicy& policy) const
            {
                return family == policy.family && (family == CacheFamily::Qwen4Exp || backend == policy.backend);
            }

            bool matches(const HostMoeSegment& hm, const CachePolicy& policy) const
            {
                const float limit = hm.clamp_limit > 1e-6f ? hm.clamp_limit : 0.0f;
                return backend == policy.backend && family == policy.family
                    && policy_budget == policy.budget && policy_layers == policy.layers
                    && clamp_limit == limit && layer == hm.layer && hidden == hm.hidden
                    && n_ff == hm.n_ff && n_used == hm.n_used && num_experts == hm.num_experts
                    && source[0] == hm.gate_data && source[1] == hm.up_data && source[2] == hm.down_data
                    && type[0] == hm.gate_type && type[1] == hm.up_type && type[2] == hm.down_type
                    && files[0].source == hm.expert_files[0].source && files[0].offset == hm.expert_files[0].offset
                    && files[1].source == hm.expert_files[1].source && files[1].offset == hm.expert_files[1].offset
                    && files[2].source == hm.expert_files[2].source && files[2].offset == hm.expert_files[2].offset;
            }
        };

        // LRU entries bound multiple loaded model identities as well as the
        // per-layer slots. Each layer has at most budget / layer_count bytes,
        // preventing early layers from consuming every slot on the first token.
        std::list<std::unique_ptr<CacheEntry>> g_entries;

        std::size_t domain_reserved(const CachePolicy& policy)
        {
            std::size_t bytes = 0;
            for (const auto& entry : g_entries) if (entry->same_domain(policy)) bytes += entry->bytes;
            return bytes;
        }

        bool trim_domain(const CachePolicy& policy, std::size_t ceiling)
        {
            std::size_t reserved = domain_reserved(policy);
            while (reserved > ceiling)
            {
                auto victim = std::find_if(g_entries.rbegin(), g_entries.rend(),
                    [&](const std::unique_ptr<CacheEntry>& item) { return item->same_domain(policy); });
                if (victim == g_entries.rend()) return false;
                reserved -= (*victim)->bytes; g_reserved -= (*victim)->bytes;
                g_entries.erase(std::prev(victim.base()));
            }
            return true;
        }

        std::size_t aggregate_budget()
        {
            // Qwen keeps its existing process-global ceiling. DeepSeek ceilings
            // are per independent backend; aggregate statistics never imply
            // that one GPU may borrow another backend's reserved budget.
            std::size_t total = cache_budget();
            std::unordered_map<ggml_backend_t, std::size_t> deepseek;
            for (const auto& entry : g_entries) if (entry->family == CacheFamily::DeepSeek41)
                deepseek[entry->backend] = std::max(deepseek[entry->backend], entry->policy_budget);
            const auto maximum = static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
            for (const auto& item : deepseek)
            {
                if (item.second > maximum - total) return maximum;
                total += item.second;
            }
            return total;
        }

#ifdef TSG_GGML_USE_CUDA
        // One small buffer per independent backend, shared by all its layer
        // entries. Unlike registering the model mmap this cannot pin hundreds
        // of GiB or displace the host's entire file cache. The model already
        // serializes execution; g_cache_mutex also protects reuse/lifetime.
        struct PreadStaging
        {
            ggml_backend_t backend;
            const std::size_t capacity;
            tsg_dsv41::engram_io_pool io{16};
            ggml_backend_buffer_t buffer = nullptr;
            std::uint8_t * data = nullptr;
            static std::size_t configured_capacity()
            {
                const auto kib = read_positive_size("TS_DSV41_EXPERT_CACHE_PREAD_STAGING_KB",65536);
                if (kib < 64 || kib > 65536)
                    throw std::runtime_error("DeepSeek expert pread staging must be between 64 and 65536 KiB");
                return kib*1024;
            }
            explicit PreadStaging(ggml_backend_t b) : backend(b),capacity(configured_capacity())
            {
                const auto host_type = ggml_backend_cuda_host_buffer_type();
                buffer = ggml_backend_buft_alloc_buffer(host_type,capacity);
                if (!buffer) throw std::runtime_error("DeepSeek expert pread staging allocation failed");
                data = static_cast<std::uint8_t *>(ggml_backend_buffer_get_base(buffer));
                if (!data)
                {
                    ggml_backend_buffer_free(buffer); buffer = nullptr;
                    throw std::runtime_error("DeepSeek expert pread staging has no host storage");
                }
                std::fprintf(stderr,"[HOSTMOE-PREAD] backend=%p staging=%zu pinned=%d threads=%u persistent=yes\n",
                    static_cast<void *>(backend),capacity,ggml_backend_buffer_get_type(buffer)==host_type,io.threads());
            }
            ~PreadStaging()
            {
                sync_backend(backend);
                if (buffer) ggml_backend_buffer_free(buffer);
            }
        };
        std::unordered_map<ggml_backend_t,std::unique_ptr<PreadStaging>> g_pread_staging;

        bool valid_layout(const HostMoeSegment& hm, std::size_t (&stride)[3])
        {
            if (hm.seq_len <= 0 || hm.seq_len > TSG_PRECISION_DECODE_COLUMNS
                || hm.activation != 0 || hm.tp_reduced != 0
                || !std::isfinite(hm.clamp_limit) || hm.clamp_limit < 0.0f
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

        bool build_graph(CacheEntry& entry, int capacity, std::size_t quota)
        {
            ggml_init_params params{};
            params.mem_size = ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(256, false);
            params.no_alloc = true;
            entry.ctx = ggml_init(params);
            if (entry.ctx == nullptr) return false;
            auto* ctx = entry.ctx;
            entry.capacity = capacity;
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
            entry.input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, entry.hidden, 1, 1);
            entry.ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, entry.n_used, 1);
            entry.routes = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, entry.n_used, 1);
            ggml_set_input(entry.input);
            ggml_set_input(entry.ids);
            ggml_set_input(entry.routes);

            auto* gate_linear = ggml_mul_mat_id(ctx, entry.weight[0], entry.input, entry.ids);
            auto* up_linear = ggml_mul_mat_id(ctx, entry.weight[1], entry.input, entry.ids);
            auto* gate = gate_linear;
            auto* up = up_linear;
            if (entry.clamp_limit > 1e-6f)
            {
                gate = ggml_clamp(ctx, gate, -INFINITY, entry.clamp_limit);
                up = ggml_clamp(ctx, up, -entry.clamp_limit, entry.clamp_limit);
            }
            auto* activation = entry.family == CacheFamily::DeepSeek41
                ? ggml_swiglu_split(ctx, gate, up)
                : ggml_mul(ctx, ggml_silu(ctx, gate), up);
            auto* down = ggml_mul_mat_id(ctx, entry.weight[2], activation, entry.ids);
            // Match qwen4exp's decode graph: the down output cannot alias the
            // weighted-reduction destination, making CUDA's fusion independent
            // of which freed scratch block the allocator would otherwise reuse.
            ggml_set_output(down);
            auto* weighted = ggml_mul(ctx, down, entry.routes);
            auto* sum = ggml_view_2d(ctx, weighted, entry.hidden, 1, weighted->nb[2], 0);
            for (int k = 1; k < entry.n_used; ++k)
                sum = ggml_add(ctx, sum, ggml_view_2d(ctx, weighted, entry.hidden, 1,
                    weighted->nb[2], static_cast<std::size_t>(k) * weighted->nb[1]));
            entry.output = ggml_cont(ctx, sum);
            ggml_set_output(entry.output);
            entry.graph = ggml_new_graph_custom(ctx, 256, false);
            // A disjoint UID range prevents a destroyed graph's captured CUDA
            // executable from being mistaken for a subsequently allocated graph.
            static std::atomic<std::uint64_t> next_uid{UINT64_C(0x484d4f4500000000)};
            entry.graph->uid = next_uid.fetch_add(1, std::memory_order_relaxed);
            ggml_build_forward_expand(entry.graph, entry.output);
            if (!ggml_backend_supports_op(entry.backend, gate_linear)
                || !ggml_backend_supports_op(entry.backend, up_linear)
                || !ggml_backend_supports_op(entry.backend, gate)
                || !ggml_backend_supports_op(entry.backend, up)
                || !ggml_backend_supports_op(entry.backend, activation)
                || !ggml_backend_supports_op(entry.backend, down)) return false;

            entry.allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(entry.backend));
            if (entry.allocator == nullptr) return false;
            std::size_t device_bytes = 0;
            ggml_gallocr_reserve_n_size(entry.allocator, entry.graph, nullptr, nullptr, &device_bytes);
            if (device_bytes > quota || quota - device_bytes < kWorkspaceAllowance) return false;
            entry.bytes = device_bytes + kWorkspaceAllowance;
            return true;
        }

        std::unique_ptr<CacheEntry> create_entry(const HostMoeSegment& hm,
            const std::size_t (&stride)[3], const CachePolicy& policy, std::size_t quota)
        {
            std::size_t per_expert = 0;
            for (const auto s : stride)
            {
                if (s > std::numeric_limits<std::size_t>::max() - per_expert) return nullptr;
                per_expert += s;
            }
            if (quota <= kWorkspaceAllowance || per_expert == 0) return nullptr;
            std::size_t capacity = std::min<std::size_t>(hm.num_experts,
                (quota - kWorkspaceAllowance) / per_expert);
            if (capacity < static_cast<std::size_t>(hm.n_used)) return nullptr;

            // Graph scratch and quantized-tensor tail padding must fit too.
            // Shrink a measured graph before allocating any device memory.
            for (;;)
            {
                auto entry = std::make_unique<CacheEntry>();
                entry->backend = policy.backend; entry->family = policy.family;
                entry->policy_budget = policy.budget; entry->policy_layers = policy.layers;
                entry->clamp_limit = hm.clamp_limit > 1e-6f ? hm.clamp_limit : 0.0f;
                entry->layer = hm.layer; entry->hidden = hm.hidden; entry->n_ff = hm.n_ff;
                entry->n_used = hm.n_used; entry->num_experts = hm.num_experts;
                entry->source[0] = hm.gate_data; entry->source[1] = hm.up_data; entry->source[2] = hm.down_data;
                entry->type[0] = hm.gate_type; entry->type[1] = hm.up_type; entry->type[2] = hm.down_type;
                for (int i = 0; i < 3; ++i)
                {
                    entry->stride[i] = stride[i];
                    entry->files[i] = hm.expert_files[i];
                }
                if (build_graph(*entry, static_cast<int>(capacity), quota)) return entry;
                if (capacity <= static_cast<std::size_t>(hm.n_used)) return nullptr;
                capacity = std::max<std::size_t>(hm.n_used, capacity * 3 / 4);
            }
        }

        bool allocate_entry(CacheEntry& entry)
        {
            std::size_t free_bytes = 0, total_bytes = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(entry.backend), &free_bytes, &total_bytes);
            if (total_bytes == 0 || free_bytes <= kDeviceHeadroom
                || entry.bytes > free_bytes - kDeviceHeadroom) return false;
            // Size-only reservation above creates a valid allocation plan with
            // no backing buffers. alloc_graph alone sees the matching plan and
            // does not allocate them; explicitly materialize that reservation.
            if (!ggml_gallocr_reserve(entry.allocator, entry.graph)
                || !ggml_gallocr_alloc_graph(entry.allocator, entry.graph)) return false;
            const std::size_t actual = ggml_gallocr_get_buffer_size(entry.allocator, 0);
            if (actual > entry.bytes - kWorkspaceAllowance) return false;
            entry.bytes = actual + kWorkspaceAllowance;
            // CUDA initializes each quantized tensor's over-read tail at alloc.
            // Clear its complete payload as well: unused slots must never hold
            // stale NaNs from a prior graph allocation.
            for (auto* w : entry.weight) ggml_backend_tensor_memset(w, 0, 0, ggml_nbytes(w));
            entry.expert_for_slot.assign(entry.capacity, -1);
            entry.last_used.assign(entry.capacity, 0);
            return true;
        }

        void copy_slot(CacheEntry& entry, int slot, int expert)
        {
            std::vector<std::pair<std::size_t, std::size_t>> pieces;
            for (int i = 0; i < 3; ++i)
            {
                const auto* src = static_cast<const std::uint8_t*>(entry.source[i])
                    + static_cast<std::size_t>(expert) * entry.stride[i];
                // Respect prior registrations from streamed prefill without
                // pinning the whole expert stack or creating a second RAM copy.
                host_pin_split(src, entry.stride[i], pieces);
                for (const auto& p : pieces)
                    ggml_backend_tensor_set_async(entry.backend, entry.weight[i], src + p.first,
                        static_cast<std::size_t>(slot) * entry.stride[i] + p.first, p.second);
            }
        }

        void pread_slots(CacheEntry& entry,const std::vector<std::pair<int,int>> & misses)
        {
            for (int part = 0; part < 3; ++part)
                if (!entry.files[part].valid(entry.stride[part]*std::size_t(entry.num_experts)))
                    throw std::runtime_error("DeepSeek expert pread requires exact file ranges for all routed weights");
            auto & owned = g_pread_staging[entry.backend];
            if (!owned) owned = std::make_unique<PreadStaging>(entry.backend);
            auto & stage = *owned;
            struct Upload { int part; std::size_t source, destination, bytes, staging; };
            std::vector<Upload> batch;
            std::size_t used = 0;
            auto flush = [&]
            {
                if (batch.empty()) return;
                // Every selected byte is read contiguously into bounded host
                // storage. No mmap page walk or entire expert stack is read.
                // A short/error read joins every worker and fails before this
                // batch is uploaded; cache tags remain invalid on any failure.
                stage.io.run(batch.size(),[&](std::size_t i)
                {
                    const auto & u = batch[i];
                    entry.files[u.part].read_exact(stage.data+u.staging,u.bytes,u.source);
                });
                for (const auto & u : batch)
                    ggml_backend_tensor_set_async(entry.backend,entry.weight[u.part],stage.data+u.staging,u.destination,u.bytes);
                // CUDA may DMA after set_async returns. Never overwrite or
                // release the staging bytes until the owning stream drains.
                sync_backend(entry.backend);
                if (g_backend_compute_failed.load(std::memory_order_acquire))
                    throw std::runtime_error("DeepSeek expert pread CUDA upload failed");
                batch.clear(); used = 0;
            };
            for (const auto & miss : misses)
                for (int part = 0; part < 3; ++part)
                    for (std::size_t offset = 0; offset < entry.stride[part];)
                    {
                        const std::size_t n = std::min(entry.stride[part]-offset,stage.capacity);
                        if (n > stage.capacity-used) flush();
                        batch.push_back({part,std::size_t(miss.second)*entry.stride[part]+offset,
                            std::size_t(miss.first)*entry.stride[part]+offset,n,used});
                        used += n; offset += n;
                    }
            flush();
        }

        PreadStaging & staging_for(ggml_backend_t backend)
        {
            auto & owned=g_pread_staging[backend];
            if (!owned) owned=std::make_unique<PreadStaging>(backend);
            return *owned;
        }
        void checked_drain(ggml_backend_t backend)
        {
            sync_backend(backend);
            if (g_backend_compute_failed.load(std::memory_order_acquire))
                throw std::runtime_error("DeepSeek host expert cache CUDA transfer failed");
        }
        std::size_t expert_bytes(const CacheEntry & entry)
        {
            std::size_t n=0;
            for (auto stride:entry.stride)
            {
                if (stride>SIZE_MAX-n) throw std::runtime_error("DeepSeek host expert cache payload overflows");
                n+=stride;
            }
            return n;
        }
        void download_slot(CacheEntry & entry,int slot,std::uint8_t * destination)
        {
            auto & stage=staging_for(entry.backend);
            const auto total=expert_bytes(entry);
            if (total<=stage.capacity)
            {
                std::size_t out=0;
                for (int p=0;p<3;++p)
                {
                    ggml_backend_tensor_get_async(entry.backend,entry.weight[p],stage.data+out,
                        std::size_t(slot)*entry.stride[p],entry.stride[p]);
                    out+=entry.stride[p];
                }
                checked_drain(entry.backend);
                std::memcpy(destination,stage.data,total);
                return;
            }
            std::size_t out=0;
            for (int p=0;p<3;++p)
                for (std::size_t off=0;off<entry.stride[p];)
                {
                    const auto n=std::min(entry.stride[p]-off,stage.capacity);
                    ggml_backend_tensor_get_async(entry.backend,entry.weight[p],stage.data,
                        std::size_t(slot)*entry.stride[p]+off,n);
                    checked_drain(entry.backend);
                    std::memcpy(destination+out+off,stage.data,n);
                    off+=n;
                    if (off==entry.stride[p]) out+=entry.stride[p];
                }
        }
        // Exchange in place: the host allocation containing a selected expert
        // becomes the GPU victim's allocation. Holding the pending node in the
        // byte accounting makes the anonymous cap strict even during exchange.
        // Two bounded pinned half-buffers preserve both sides until DMA drains.
        void exchange_slot(CacheEntry & entry,int slot,std::uint8_t * host,bool keep_victim)
        {
            auto & stage=staging_for(entry.backend);
            const auto half=stage.capacity/2,total=expert_bytes(entry);
            if (total<=half)
            {
                std::size_t offset=0;
                if (keep_victim) for (int p=0;p<3;++p)
                {
                    ggml_backend_tensor_get_async(entry.backend,entry.weight[p],stage.data+offset,
                        std::size_t(slot)*entry.stride[p],entry.stride[p]);
                    offset+=entry.stride[p];
                }
                std::memcpy(stage.data+half,host,total);
                if (keep_victim) checked_drain(entry.backend);
                offset=0;
                for (int p=0;p<3;++p)
                {
                    ggml_backend_tensor_set_async(entry.backend,entry.weight[p],stage.data+half+offset,
                        std::size_t(slot)*entry.stride[p],entry.stride[p]);
                    offset+=entry.stride[p];
                }
                checked_drain(entry.backend);
                if (keep_victim) std::memcpy(host,stage.data,total);
                return;
            }
            std::size_t host_offset=0;
            for (int p=0;p<3;++p)
            {
                for (std::size_t off=0;off<entry.stride[p];)
                {
                    const auto n=std::min(entry.stride[p]-off,half);
                    if (keep_victim)
                        ggml_backend_tensor_get_async(entry.backend,entry.weight[p],stage.data,
                            std::size_t(slot)*entry.stride[p]+off,n);
                    std::memcpy(stage.data+half,host+host_offset+off,n);
                    if (keep_victim) checked_drain(entry.backend);
                    ggml_backend_tensor_set_async(entry.backend,entry.weight[p],stage.data+half,
                        std::size_t(slot)*entry.stride[p]+off,n);
                    checked_drain(entry.backend);
                    if (keep_victim) std::memcpy(host+host_offset+off,stage.data,n);
                    off+=n;
                }
                host_offset+=entry.stride[p];
            }
        }
        void prefetch_slots(CacheEntry & entry,const std::vector<std::pair<int,int>> & misses)
        {
            std::vector<std::pair<std::size_t,std::size_t>> ranges;
            ranges.reserve(misses.size());
            for (int part=0;part<3;++part)
            {
                ranges.clear();
                for (const auto & miss:misses)
                    ranges.emplace_back(std::size_t(miss.second)*entry.stride[part],entry.stride[part]);
                std::sort(ranges.begin(),ranges.end());
                prefetch_mapped_ranges(static_cast<const std::uint8_t *>(entry.source[part]),ranges);
            }
        }
        void fill_host_tier(CacheEntry & entry,const std::vector<std::pair<int,int>> & misses,
                            std::size_t budget,bool pread,bool prefetch)
        {
            trim_host(budget);
            std::vector<std::pair<int,int>> source_misses;
            source_misses.reserve(misses.size());
            // Consume all selected host hits before inserting other victims:
            // a new victim must not evict a selected host expert not yet used.
            for (const auto & miss:misses)
            {
                auto found=std::find_if(g_host_victims.begin(),g_host_victims.end(),
                    [&](const HostVictim & h) {return h.owner==&entry && h.expert==miss.second;});
                if (found==g_host_victims.end()) {source_misses.push_back(miss);continue;}
                std::list<HostVictim> pending;
                pending.splice(pending.begin(),g_host_victims,found);
                auto & host=pending.front();
                const int victim=entry.expert_for_slot[miss.first];
                entry.expert_for_slot[miss.first]=-1;
                try {exchange_slot(entry,miss.first,host.data.get(),victim>=0);}
                catch (...) {g_host_reserved-=host.bytes;throw;}
                if (victim>=0)
                {
                    host.expert=victim;
                    g_host_victims.splice(g_host_victims.begin(),pending,pending.begin());
                }
                else {g_host_reserved-=host.bytes;pending.clear();}
                entry.expert_for_slot[miss.first]=miss.second;
                ++g_host_hits;
            }
            const auto bytes=expert_bytes(entry);
            for (const auto & miss:source_misses)
            {
                const int victim=entry.expert_for_slot[miss.first];
                entry.expert_for_slot[miss.first]=-1;
                if (victim<0 || bytes>budget) continue;
                trim_host(budget-bytes);
                auto data=std::unique_ptr<std::uint8_t[]>(new std::uint8_t[bytes]);
                download_slot(entry,miss.first,data.get());
                g_host_victims.push_front({&entry,victim,bytes,std::move(data)});
                g_host_reserved+=bytes;
            }
            if (!source_misses.empty())
            {
                g_source_reads+=source_misses.size();
                if (pread) pread_slots(entry,source_misses);
                else
                {
                    if (prefetch) prefetch_slots(entry,source_misses);
                    for (const auto & miss:source_misses) copy_slot(entry,miss.first,miss.second);
                    checked_drain(entry.backend);
                }
                for (const auto & miss:source_misses) entry.expert_for_slot[miss.first]=miss.second;
            }
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
        ggml_backend_t backend, std::size_t budget_per_device, std::size_t layer_count)
    {
#ifdef TSG_GGML_USE_CUDA
        CachePolicy policy;
        if (!policy_for(kernel_name, backend, budget_per_device, layer_count, policy)) return 0;
        const std::size_t budget = policy.budget, layers = policy.layers;
        backend = policy.backend;
        const bool device_inputs = x == nullptr && routes == nullptr;
        const bool device_output = out == nullptr;
        if (!ggml_backend_is_cuda(backend)
            || (!device_inputs && (x == nullptr || routes == nullptr))
            || ids == nullptr || (device_output && !device_inputs)) return 0;
        std::size_t stride[3] = {};
        if (!valid_layout(hm, stride)) return 0;
        // Validate the complete short block before allocating or computing any
        // row. Unsupported data always falls back as one block; execution
        // failure returns an error and never mixes CPU/GPU expert arithmetic.
        const std::size_t route_count = static_cast<std::size_t>(hm.n_used) * hm.seq_len;
        if (device_inputs && (!device_source(hm.moe_in,
                static_cast<std::size_t>(hm.hidden) * hm.seq_len * sizeof(float), backend)
            || !device_source(hm.weights, route_count * sizeof(float), backend))) return 0;
        if (device_output && !device_source(hm.moe_out,
                static_cast<std::size_t>(hm.hidden) * hm.seq_len * sizeof(float), backend)) return 0;
        for (std::size_t k = 0; k < route_count; ++k)
            if (ids[k] < 0 || ids[k] >= hm.num_experts
                || (!device_inputs && !std::isfinite(routes[k]))) return 0;

        std::lock_guard<std::mutex> lock(g_cache_mutex);
        // A caller can lower its reserved ceiling between graph instances.
        // Retire old entries even when the new quota cannot fit an expert and
        // this request will fall back; disabled arithmetic must not retain an
        // allocation above the explicitly supplied device reservation.
        if (policy.family == CacheFamily::DeepSeek41 && !trim_domain(policy, budget)) return 0;
        auto found = g_entries.begin();
        while (found != g_entries.end() && !(*found)->matches(hm, policy)) ++found;
        if (found == g_entries.end())
        {
            auto entry = create_entry(hm, stride, policy, budget / layers);
            if (!entry) return 0;
            if (!trim_domain(policy, budget - entry->bytes)) return 0;
            const std::size_t reserved = domain_reserved(policy);
            if (entry->bytes > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) - g_reserved
                || !allocate_entry(*entry)) return 0;
            g_reserved += entry->bytes;
            if (diagnostics(policy.family)) std::fprintf(stderr,
                "[HOSTMOE-CACHE] family=%s backend=%p layer=%d slots=%d domain_reserved=%zu budget=%zu total_reserved=%zu graph=%zu workspace_allowance=%zu input_bridge=%s\n",
                family_name(policy.family), static_cast<void*>(backend), hm.layer, entry->capacity, reserved + entry->bytes, budget, g_reserved,
                entry->bytes - kWorkspaceAllowance, kWorkspaceAllowance, device_inputs ? "device" : "host");
            g_entries.push_front(std::move(entry));
        }
        else
        {
            g_entries.splice(g_entries.begin(), g_entries, found);
        }
        auto& entry = *g_entries.front();
        std::vector<int> remapped(hm.n_used, -1);
        std::vector<std::uint8_t> protected_slots(entry.capacity, 0);
        std::vector<std::pair<int, int>> misses;
        misses.reserve(hm.n_used);
        const bool prefetch = prefetch_enabled(policy.family);
        const bool pread = policy.family == CacheFamily::DeepSeek41 &&
            tsg_dsv41_pread::enabled(std::getenv("TS_DSV41_EXPERT_CACHE_PREAD"));
        const auto host_budget=policy.family==CacheFamily::DeepSeek41 ? host_cache_budget() : 0;
        if (policy.family==CacheFamily::DeepSeek41) trim_host(host_budget);
        // Qwen4Exp's precision policy uses decode kernels for every row in a
        // short prefill/speculative target block. Replay this same scalar graph
        // so compact-cache verification has exactly the decode rounding. The
        // one allocation above serves every row; eviction only replaces slots.
        for (int row = 0; row < hm.seq_len; ++row)
        {
            const auto* row_ids = ids + static_cast<std::size_t>(row) * hm.n_used;
            std::fill(remapped.begin(), remapped.end(), -1);
            std::fill(protected_slots.begin(), protected_slots.end(), 0);
            misses.clear();
            // Protect ALL hits before filling any miss, including hits later in
            // the router's order. Otherwise an early miss can evict a later hit.
            for (int k = 0; k < hm.n_used; ++k)
                for (int s = 0; s < entry.capacity; ++s)
                    if (entry.expert_for_slot[s] == row_ids[k])
                    {
                        remapped[k] = s;
                        protected_slots[s] = 1;
                        ++g_hits;
                        break;
                    }
            for (int k = 0; k < hm.n_used; ++k)
            {
                if (remapped[k] >= 0) continue;
                // Duplicated selected IDs are legal for the helper, even though
                // Qwen's top-k normally produces distinct experts.
                for (int previous = 0; previous < k; ++previous)
                    if (row_ids[previous] == row_ids[k]) { remapped[k] = remapped[previous]; break; }
                if (remapped[k] >= 0) { ++g_hits; continue; }
                int slot = -1;
                for (int s = 0; s < entry.capacity; ++s)
                    if (!protected_slots[s] && (slot < 0 || entry.last_used[s] < entry.last_used[slot])) slot = s;
                if (slot < 0)
                {
                    set_last_error(std::string(family_name(policy.family)) + " expert cache: all slots protected before routed experts were filled.");
                    return -1;
                }
                misses.emplace_back(slot, row_ids[k]);
                remapped[k] = slot;
                protected_slots[slot] = 1;
                ++g_misses;
            }
            // Fault all selected misses concurrently before pageable CUDA
            // uploads serialize source page faults on the launching thread.
            // Only raw bytes are touched; cache-hit payloads and row arithmetic
            // are unchanged. Keep this opt-in for cold/warm trained A/B runs.
            if (host_budget && !misses.empty())
            {
                try {fill_host_tier(entry,misses,host_budget,pread,prefetch);}
                catch (const std::exception & error)
                {
                    set_last_error(error.what());
                    std::fprintf(stderr,"[HOSTMOE-HOST] failure: %s\n",error.what());
                    return -1;
                }
            }
            else if (!misses.empty())
            {
                if (policy.family==CacheFamily::DeepSeek41) g_source_reads+=misses.size();
                if (prefetch && !pread) prefetch_slots(entry,misses);
                if (pread)
                {
                    // Replacements must never retain the former expert identity
                    // if a later read/upload fails after overwriting its payload.
                    for (const auto & miss : misses) entry.expert_for_slot[miss.first] = -1;
                    try { pread_slots(entry,misses); }
                    catch (const std::exception & error)
                    {
                        set_last_error(error.what());
                        std::fprintf(stderr,"[HOSTMOE-PREAD] failure: %s\n",error.what());
                        return -1;
                    }
                }
                for (const auto & miss:misses)
                {
                    if (!pread) copy_slot(entry,miss.first,miss.second);
                    entry.expert_for_slot[miss.first]=miss.second;
                }
            }
            for (const int slot : remapped) entry.last_used[slot] = ++entry.clock;
            // Copies and compute use the backend's own stream; graph topology and
            // addresses never change when slots are replaced or IDs are remapped.
            if (device_inputs)
            {
                copy_device_row(entry.backend, hm.moe_in, entry.input,
                    static_cast<std::size_t>(row) * entry.hidden * sizeof(float));
                copy_device_row(entry.backend, hm.weights, entry.routes,
                    static_cast<std::size_t>(row) * entry.n_used * sizeof(float));
            }
            else
            {
                ggml_backend_tensor_set_async(entry.backend, entry.input,
                    x + static_cast<std::size_t>(row) * hm.hidden, 0,
                    static_cast<std::size_t>(entry.hidden) * sizeof(float));
                ggml_backend_tensor_set_async(entry.backend, entry.routes,
                    routes + static_cast<std::size_t>(row) * hm.n_used, 0,
                    static_cast<std::size_t>(entry.n_used) * sizeof(float));
            }
            ggml_backend_tensor_set_async(entry.backend, entry.ids, remapped.data(), 0,
                static_cast<std::size_t>(entry.n_used) * sizeof(std::int32_t));
            if (compute_graph(entry.backend, entry.graph) != GGML_STATUS_SUCCESS
                || g_backend_compute_failed.load(std::memory_order_acquire))
            {
                set_last_error(std::string(family_name(policy.family)) + " expert cache: CUDA expert graph failed.");
                return -1;
            }
            if (device_output)
                copy_device_row(entry.backend, entry.output, hm.moe_out,
                    static_cast<std::size_t>(row) * entry.hidden * sizeof(float), true);
            else
                ggml_backend_tensor_get(entry.output, out + static_cast<std::size_t>(row) * hm.hidden,
                    0, static_cast<std::size_t>(entry.hidden) * sizeof(float));
            ++g_calls;
        }
        return 1;
#else
        (void)hm; (void)x; (void)ids; (void)routes; (void)out; (void)kernel_name;
        (void)backend; (void)budget_per_device; (void)layer_count;
        return 0;
#endif
    }

    void host_moe_expert_cache_on_drop(const void* ptr)
    {
        if (ptr == nullptr) return;
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        for (auto it = g_entries.begin(); it != g_entries.end();)
        {
            const auto& entry = **it;
            if (entry.source[0] == ptr || entry.source[1] == ptr || entry.source[2] == ptr)
            {
                g_reserved -= entry.bytes;
                it = g_entries.erase(it);
            }
            else ++it;
        }
    }

    void host_moe_expert_cache_release()
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        if ((diagnostics() || diagnostics(CacheFamily::DeepSeek41)) && (g_calls != 0 || g_reserved != 0)) std::fprintf(stderr,
            "[HOSTMOE-CACHE] calls=%llu hits=%llu misses=%llu reserved=%zu budget=%zu release\n",
            static_cast<unsigned long long>(g_calls), static_cast<unsigned long long>(g_hits),
            static_cast<unsigned long long>(g_misses), g_reserved, aggregate_budget());
        g_entries.clear();
        if (diagnostics(CacheFamily::DeepSeek41)) std::fprintf(stderr,
            "[HOSTMOE-HOST] hits=%llu source_reads=%llu evictions=%llu reserved=%zu budget=%zu release\n",
            static_cast<unsigned long long>(g_host_hits),static_cast<unsigned long long>(g_source_reads),
            static_cast<unsigned long long>(g_host_evictions),g_host_reserved,host_cache_budget());
        g_host_victims.clear();g_host_reserved=0;
        g_host_hits=g_source_reads=g_host_evictions=0;
#ifdef TSG_GGML_USE_CUDA
        g_pread_staging.clear();
#endif
        g_reserved = 0;
        g_hits = g_misses = g_calls = 0;
    }

    void host_moe_expert_cache_release_backend(ggml_backend_t backend)
    {
        if (backend == nullptr) return;
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        for (auto it = g_entries.begin(); it != g_entries.end();)
        {
            if ((*it)->backend == backend)
            {
                g_reserved -= (*it)->bytes;
                it = g_entries.erase(it); // drains pending output copies before freeing its source buffers
            }
            else ++it;
        }
#ifdef TSG_GGML_USE_CUDA
        g_pread_staging.erase(backend);
#endif
    }

    void host_moe_expert_cache_stats(std::int64_t* reserved, std::int64_t* budget,
        std::int64_t* hits, std::int64_t* misses, std::int64_t* calls)
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        *reserved = static_cast<std::int64_t>(g_reserved);
        *budget = static_cast<std::int64_t>(aggregate_budget());
        *hits = static_cast<std::int64_t>(g_hits);
        *misses = static_cast<std::int64_t>(g_misses);
        *calls = static_cast<std::int64_t>(g_calls);
    }
    void host_moe_expert_host_cache_stats(std::int64_t * reserved,std::int64_t * budget,
        std::int64_t * hits,std::int64_t * sources,std::int64_t * evictions)
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        *reserved=static_cast<std::int64_t>(g_host_reserved);
        *budget=static_cast<std::int64_t>(host_cache_budget());
        *hits=static_cast<std::int64_t>(g_host_hits);
        *sources=static_cast<std::int64_t>(g_source_reads);
        *evictions=static_cast<std::int64_t>(g_host_evictions);
    }
}

TSG_EXPORT int TSGgml_HostMoeExpertCacheStats(std::int64_t* reserved, std::int64_t* budget,
    std::int64_t* hits, std::int64_t* misses, std::int64_t* calls)
{
    if (!reserved || !budget || !hits || !misses || !calls) return 0;
    tsg::host_moe_expert_cache_stats(reserved, budget, hits, misses, calls);
    return 1;
}
TSG_EXPORT int TSGgml_HostMoeExpertHostCacheStats(std::int64_t * reserved,std::int64_t * budget,
    std::int64_t * hits,std::int64_t * source_reads,std::int64_t * evictions)
{
    if (!reserved || !budget || !hits || !source_reads || !evictions) return 0;
    tsg::host_moe_expert_host_cache_stats(reserved,budget,hits,source_reads,evictions);
    return 1;
}
