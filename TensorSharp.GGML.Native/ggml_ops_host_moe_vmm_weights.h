// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include "ggml.h"
#include <array>
#include <cstddef>
#include <cstdint>

namespace tsg::host_moe_vmm {
// Internal native API; no environment flag, cache policy or new budget grant.
// The entire arena is retained independently of a CacheEntry until checked
// destroy succeeds. One arena is supported, matching the exclusive cache scope.
// Its caller is the sole owner of the opaque pointer: do not copy aliases or
// retain/use one after checked destruction, including after a later reopen.
struct WeightArena;
struct WeightHandle { std::uint64_t arena = 0, table = 0; unsigned slot = 0; };
enum class WeightStatus { ready, unchanged, unsupported, refused, failed, quarantined };
struct WeightOptions {
    int cuda_device = 0, budget_rank = 0;
    std::size_t maximum_total_chunks = 8192;
    // Zero selects the actual driver minimum. An explicit power-of-two chunk
    // must also be an exact multiple of that minimum; it is never rounded up.
    // The arena freezes this choice for every table, layout and ticket.
    std::size_t preferred_chunk_bytes = 0;
    bool operator==(const WeightOptions& other) const noexcept {
        return cuda_device == other.cuda_device && budget_rank == other.budget_rank
            && maximum_total_chunks == other.maximum_total_chunks
            && preferred_chunk_bytes == other.preferred_chunk_bytes;
    }
};
struct ProjectionShape { ggml_type type = GGML_TYPE_COUNT; std::int64_t columns = 0, rows = 0; std::size_t stride = 0; };
struct WeightSpec {
    std::array<ProjectionShape, 3> projections{};
    std::size_t maximum_capacity = 0, initial_capacity = 0;
    const void* source_owner = nullptr; // Stable CacheEntry identity; never dereferenced after detach.
};
struct WeightHostPlan {
    std::size_t declared_charge_bytes = 0, slab_bytes = 0, maximum_live_ticket_records = 0;
    bool allocator_overhead_exact = false;
};
struct WeightProjectionPlan {
    std::size_t stride = 0, allocation = 0, granule = 0, chunk = 0;
    std::size_t physical = 0, reserved_va = 0, maximum_buffer = 0, alignment = 0;
};
struct WeightLayoutPlan {
    std::size_t capacity = 0, maximum_capacity = 0, total = 0, reserved_va_total = 0;
    std::array<WeightProjectionPlan, 3> projections{};
};
struct WeightBarrier {
    // Hold the caller's cache gate. Join/cancel producers BEFORE CUDA drains,
    // invalidate ALL narrow/wide/overlap graphs and capacity-dependent userdata.
    // Preserve the source owner, its complete slot vector and mutation generation
    // through this call. Those arguments must describe the same live CacheEntry;
    // the adapter cannot discover or authenticate arbitrary external slot tags.
    // This callback must not enter this API. No context is retained after call.
    bool (*freeze_join_invalidate)(void*) noexcept = nullptr;
    void* context = nullptr;
};
struct WeightUsage {
    // Native held-ticket state, not physical use or an observed managed ledger
    // snapshot. A failed checked acknowledgment may follow an actual refund.
    // Physical payload = mapped + created_unmapped; reserved_va is address space.
    std::size_t capacity = 0, ticket_pending = 0, ticket_committed = 0;
    std::size_t mapped = 0, created_unmapped = 0, ticket_only = 0, quarantine = 0;
    std::array<std::size_t, 3> physical_prefix{}, reserved_va{};
    std::size_t declared_host_charge = 0, portable_and_adapter_slab = 0, ticket_records = 0;
    bool failed = false, usable = false, charge_certificate_api = false;
    bool actual_charge_certificate = false, context_lost = false;
    bool host_ticket_held = false, host_ticket_only = false, primary_context_owned = false;
    // Slab/record count is exact. The extra per-ticket allocation allowance is
    // conservative policy; allocator fragmentation/driver context internals are
    // outside this payload owner, not proved bounded by sizeof the slab.
    bool allocator_overhead_exact = false;
};

// Compile-time native capability only; factory also requires an actual attached
// checked-release/Host-accounting scope and the caller's durable envelope.
bool weight_charge_certificate_available() noexcept;
// Pure bounded Host calculation; does not require a budget scope or driver.
WeightStatus plan_weight_host(const WeightOptions&, WeightHostPlan&) noexcept;
// Actual ordinary CUDA buft/device metadata query. May initialize vendor driver/
// runtime metadata; does not reserve tickets, tables, retained contexts, VAs or
// payload. No arena/sourceOwner is required. capacity is positive and <=maximum.
// This plan is neither a physical-free observation nor a reservation/certificate.
WeightStatus plan_weight_layout(const WeightOptions&, const std::array<ProjectionShape, 3>&,
    std::size_t maximum_capacity, std::size_t capacity, WeightLayoutPlan&) noexcept;
// Future managed caller MUST hold EnterAllocationOwner on a dedicated durable
// Host metadata envelope at scope startup (before any request). Private factory
// mints/verifies its own actual kind3 ticket; arbitrary caller leases rejected.
// Missing SharedCacheCharge certificate getters OR an attached acknowledged
// release callback is unsupported before reserve/CUDA calls. Managed wiring and
// accounting/device qualification remain separate prerequisites.
// The caller must already have options.cuda_device's primary context current;
// metadata-only ggml_backend_cuda_init does not establish this precondition.
WeightStatus create_weight_arena(const WeightOptions&, WeightArena*& result) noexcept;
WeightStatus create_weight_table(WeightArena*, unsigned slot, const WeightSpec&, WeightHandle&) noexcept;
WeightStatus grow_weight_table(WeightArena*, WeightHandle, std::size_t capacity, const WeightBarrier&) noexcept;
WeightStatus shrink_weight_table_empty_tail(WeightArena*, WeightHandle, std::size_t capacity,
    const void* source_owner, std::uint64_t mutation_generation, const int* complete_expert_for_slot,
    std::size_t count, const WeightBarrier&) noexcept;
// Explicit same-device D2D; destination must be empty in the same complete slot
// vector. No generic ggml tensor_copy[_async] call is permitted for these weights.
WeightStatus copy_weight_slot(WeightArena*, WeightHandle, std::size_t source, std::size_t destination,
    const void* source_owner, const int* complete_expert_for_slot, std::size_t count,
    const WeightBarrier&) noexcept;
ggml_tensor* weight_tensor(WeightArena*, WeightHandle, unsigned projection) noexcept;
WeightUsage weight_usage(WeightArena*, WeightHandle) noexcept;
bool close_weight_table(WeightArena*, WeightHandle, const WeightBarrier&) noexcept;
// Terminal whole-arena cleanup, not a capacity/policy operation: all table
// handles and dependent graph owners must be retired by the caller's barrier.
bool retry_weight_cleanup(WeightArena*, const WeightBarrier&) noexcept;
// Failure retains every still-owned object and ticket in the process shell;
// physical cleanup failure retains its entire slab/adapter/context/VA state.
// Final Host acknowledgment failure can retain a ticket after physical frees.
// Result pointer is unchanged. Do not detach budget or its managed owner.
bool destroy_weight_arena_checked(WeightArena*&, const WeightBarrier&) noexcept;
#if defined(TSG_GGML_TEST_HOOKS)
enum class WeightTestFault { none, create_after_successes, padding_after_write };
struct WeightTestStats {
    std::uint64_t successful_creates = 0, padding_writes = 0;
    std::uint64_t create_failures = 0, padding_failures = 0;
};
// Source/device fixtures only, under the same exclusive gate and with no active
// graph/read producer. One-shot faults; none/0 disarms. Stats remain observable
// after quarantine. These do not emulate driver cleanup/context error recovery.
bool weight_test_fault(WeightArena*, WeightTestFault, int successful_operations_before_failure) noexcept;
WeightTestStats weight_test_stats(WeightArena*) noexcept;
#endif
} // namespace tsg::host_moe_vmm
