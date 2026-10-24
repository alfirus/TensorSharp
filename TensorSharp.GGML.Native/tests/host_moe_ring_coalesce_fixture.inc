// Reuse the frozen v20 ring shapes/oracle/ledger and extend only this opt-in mode.
#ifdef TSG_GGML_USE_CUDA
void host_moe_prefill_ring_coalesce_checks(const std::array<ggml_type, 3>& types) {
    const auto calls = tsg::host_moe_expert_cache_test_coalesced_ring_calls();
    const auto partial = tsg::g_test_partial_file_reads.load();
    tsg::g_test_partial_file_failure = true;
    struct Reset { ~Reset() { tsg::g_test_partial_file_failure = false; } } reset;
    host_moe_prefill_ring_staging_checks(types, false, true);
    require(tsg::host_moe_expert_cache_test_coalesced_ring_calls() > calls,
        "No completed coalesced72/36 ring call executed");
    require(tsg::g_test_partial_file_reads.load() > partial,
        "Coalesced fixture lacks a real partially filled merged read");
    std::puts("PASS: coalesced ring16 exact resident oracle, fewer actual read tasks, unchanged selected payload/arena/tickets, RAM/scalar/refusal gates, overlap preflight, partial real-file read and queued-upload/epoch drainage");
}
#endif
