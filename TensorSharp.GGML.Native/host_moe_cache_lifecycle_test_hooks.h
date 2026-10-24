// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Lifecycle-fixture contract. No production ABI or exports.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include "ggml_ops_internal.h"

#if defined(TSG_GGML_TEST_HOOKS) && defined(TSG_GGML_USE_CUDA)
namespace tsg {
    std::uint64_t host_moe_expert_cache_test_payload_upload_drains();
    std::uint64_t host_moe_expert_cache_test_phase_cold_growth_attempts();
    std::array<std::size_t, 3> host_moe_expert_cache_test_completed_ring_plan();
    std::uint64_t host_moe_expert_cache_test_coalesced_ring_calls();
    // Last successfully copied coalesced file call: ordinary tasks, actual
    // reads, exact payload, bytes in adjacent multi-piece runs, fixed read cap.
    std::array<std::size_t, 5> host_moe_expert_cache_test_completed_coalesced_ring_plan();
struct HostMoeFileSortTestSnapshot {
    std::size_t misses = 0, bytes = 0, arena = 0, slot_bytes = 0, windows = 0, tasks = 0;
    std::uint64_t completed_calls = 0, changed_calls = 0, inversions_before = 0;
    std::uint64_t source_identity[3] = {};
    int original_slot[288] = {}, original_expert[288] = {};
    int ordered_slot[288] = {}, ordered_expert[288] = {};
    bool ring = false;
};
HostMoeFileSortTestSnapshot host_moe_expert_cache_test_file_sort_snapshot();
struct HostMoeCacheTestPlan {
    int capacity = 0, narrow_rows = 0, wide_rows = 0;
    std::size_t weight_bytes = 0, narrow_bytes = 0, wide_bytes = 0;
    std::size_t weight_physical_bytes = 0, narrow_physical_bytes = 0, wide_physical_bytes = 0;
};
struct HostMoeCacheTestSlot {
    std::int32_t expert = -1;
    std::uintptr_t source[3] = {};
    std::uint64_t last_used = 0;
};
struct HostMoeCacheTestSnapshot {
    std::uintptr_t entry_identity = 0;
    std::uintptr_t weight_data[3] = {};
    std::size_t weight_part_bytes[3] = {};
    std::uintptr_t source[3] = {};
    std::uintptr_t weight_buffer = 0, narrow_buffer = 0, wide_buffer = 0;
    // These identify SharedCacheCharge objects, not numerical ledger tickets.
    std::uintptr_t weight_charge = 0, narrow_charge = 0, wide_charge = 0;
    std::size_t weight_bytes = 0, narrow_bytes = 0, wide_bytes = 0;
    std::size_t weight_physical_bytes = 0, narrow_physical_bytes = 0, wide_physical_bytes = 0;
    int capacity = 0, narrow_rows = 0, wide_rows = 0, resident = 0, committed_owners = 0;
    std::uint64_t clock = 0, growth_generation = 0, wide_generation = 0;
    std::uint64_t narrow_graph_uid = 0, wide_graph_uid = 0;
    std::uint64_t file_read_calls = 0, file_read_bytes = 0;
    // Actual VMM adapter/binding state, only in this standalone fixture build.
    bool vmm = false, vmm_certificate = false, vmm_usable = false;
    bool vmm_failed = false, vmm_graph_binding_valid = false;
    std::uintptr_t vmm_table_identity = 0;
    std::size_t vmm_pending_bytes = 0, vmm_quarantine_bytes = 0;
    std::uint64_t vmm_mutation_generation = 0, width_graph_uid[8] = {};
    HostMoeCacheTestSlot slots[64] = {};
};
// Explicitly selected tiny geometry exists only in this native test build.
// Change it only while no scope, arena, entry or producer is live.
void host_moe_expert_cache_test_vmm_profile(bool enabled);
std::uint64_t host_moe_expert_cache_test_vmm_mutation_failures();
// Caller owns exactly capacity records; the cache does not allocate here.
bool host_moe_expert_cache_test_copy_slot_metadata(const HostMoeSegment&,
    ggml_backend_t, HostMoeCacheTestSlot*, std::size_t exact_capacity);
// Size-only: plan all three owners at exactly capacity; no physical allocation,
// shared charge, source-key publication, live cache mutation, or file I/O.
bool host_moe_expert_cache_test_split_plan(const HostMoeSegment&, int capacity,
    int decode_rows, int prefill_rows, ggml_backend_t, HostMoeCacheTestPlan&);
// Mutex-protected metadata only. Return false for no match. Larger tables
// report full ownership/capacity metadata and only the first 64 slot records.
bool host_moe_expert_cache_test_lifecycle_snapshot(const HostMoeSegment&,
    ggml_backend_t, HostMoeCacheTestSnapshot&);
// Mutex-free, allocation-free atomic counters for callback use: [weights,
// narrow scratch, wide scratch]. Increment after actual allocation; decrement
// only after its actual physical free returns. No registry/cache/CUDA calls.
void host_moe_expert_cache_test_split_physical_bytes(std::size_t values[3]);
// Callback-only TLS view of the actual resource; never takes a lock or queries CUDA.
bool host_moe_expert_cache_test_current_split_owner(int& role,
    std::uintptr_t& identity, std::size_t& physical_bytes);
// Fixture-controlled synchronized D2H read of one complete weight tensor.
// The byte count must equal weight_part_bytes[part], with part in [0,3).
bool host_moe_expert_cache_test_copy_weight_payload(const HostMoeSegment&,
    ggml_backend_t, int part, void* destination, std::size_t exact_bytes);
}
#endif
