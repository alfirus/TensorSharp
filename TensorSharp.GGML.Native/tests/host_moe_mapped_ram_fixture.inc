// Exact CUDA parity for the optional immutable locked file-view source tier.
#if defined(TSG_GGML_USE_CUDA) && defined(_WIN32)
extern "C" int TSGgml_HostMoeMappedRamVersion();
extern "C" int TSGgml_HostMoeMappedRamEnabled();
extern "C" int TSGgml_HostMoeMappedRamStats(std::int64_t*, int);
extern "C" void TSGgml_TestMappedRamProbationClear();
extern "C" std::size_t TSGgml_TestMappedRamProbationKeys();
struct MappedRamFixtureStats {
    std::array<std::int64_t, 16> values{};
    MappedRamFixtureStats() {
        require(TSGgml_HostMoeMappedRamStats(values.data(), values.size()) == 1,
            "Cannot observe mapped RAM ownership");
        require(values[1] <= values[2] && values[3] + values[4] <= values[2]
            && values[7] <= values[2] && values[2] <= values[0], "Mapped RAM charge invariant failed");
    }
};
void host_moe_mapped_ram_checks(const std::array<ggml_type, 3>& types) {
    struct ParallelReads {
        bool previous = tsg::g_test_parallel_file_reads;
        ParallelReads() {
            require(tsg::g_test_parallel_file_active.load() == 0, "Mapped read workers were already active");
            tsg::g_test_parallel_file_peak.store(0);
            tsg::g_test_parallel_file_reads = true;
        }
        ~ParallelReads() { tsg::g_test_parallel_file_reads = previous; }
    } parallel_reads;
    constexpr std::int64_t capacity = 16LL << 20, control = 65536 + (1LL << 20);
    tsg::host_moe_expert_cache_release();
    require(TSGgml_HostMoeMappedRamVersion() == 1 && TSGgml_HostMoeMappedRamEnabled() == 1,
        "Mapped RAM capability/latch unavailable");
    require(TSGgml_HostMoeRamCacheConfigure(capacity) == 0,
        "Mapped RAM admitted without Host accounting");
    Weights first, replacement; first.types = replacement.types = types;
    first.fill(.72f); replacement.fill(2.12f);
    // Exercise the same bounded scalar file path as the pooled GLM adapter.
    // The legacy nonpooled path intentionally ignores test slice limits.
    first.segment.pooled_model_experts = first.segment.num_experts;
    replacement.segment.pooled_model_experts = replacement.segment.num_experts;
    FullResidentGraph oracle(first), next_oracle(replacement), bulk_oracle(first, 16);
    std::array<std::string, 2> paths{fixture_file_path("host-moe-mapped-original-"),
        fixture_file_path("host-moe-mapped-replacement-")};
    std::array<std::array<std::int64_t, 3>, 2> offsets{};
    RamFixtureLedger ledger; ledger.host_capacity = 64LL << 20;
    std::uint64_t token = 0;
    struct Cleanup {
        std::array<std::string, 2>& paths; std::uint64_t& token;
        ~Cleanup() {
            tsg::g_test_file_failure_countdown = 0;
            tsg::g_test_partial_file_failure = false;
            tsg::host_moe_expert_cache_test_file_slice_limit(32u << 20);
            if (token) { std::int64_t released = 0; TSGgml_EndHostMoeExpertCacheScope(token, &released); }
            tsg::host_moe_expert_cache_release();
            for (const auto& path : paths) std::remove(path.c_str());
        }
    } cleanup{paths, token};
    for (int version = 0; version < 2; ++version) {
        const auto& weights = version ? replacement : first;
        std::ofstream file(paths[version], std::ios::binary | std::ios::trunc);
        for (int part = 0; part < 3; ++part) {
            file.write("source-offset!", 14); offsets[version][part] = static_cast<std::int64_t>(file.tellp());
            file.write(reinterpret_cast<const char*>(weights.quantized[part].data()), weights.quantized[part].size());
        }
        std::array<char, 4096> tail{}; file.write(tail.data(), tail.size());
        require(bool(file), "Cannot write mapped RAM source fixture");
    }
    auto register_version = [&](int version) {
        for (int part = 0; part < 3; ++part) {
            auto& bytes = first.quantized[part];
            require(TSGgml_RegisterHostFileSource(bytes.data(), bytes.size(), paths[version].c_str(), offsets[version][part]) == 1,
                "Cannot register mapped RAM source");
            std::fill(bytes.begin(), bytes.end(), 0);
        }
    };
    register_version(0);
    require(TSGgml_BeginHostMoeExpertCacheScopeWithDecodeRows(32LL << 20, 32, 1, 3, &token) == 1,
        "Cannot start mapped RAM GPU fixture");
    ledger.attached = tsg::SharedCacheCharge::attach(&ledger, RamFixtureLedger::reserve,
        RamFixtureLedger::commit, RamFixtureLedger::release, false, true);
    require(ledger.attached && TSGgml_HostMoeRamCacheConfigure(capacity) == 1, "Cannot enable mapped RAM fixture");
    require(TSGgml_HostMoeRamCacheConfigure(capacity) == 1, "Mapped RAM repeated configuration failed");
    require(TSGgml_HostMoeRamCacheConfigure(1) == 1 && MappedRamFixtureStats().values[2] == 0
        && TSGgml_HostMoeRamCacheConfigure(capacity) == 1, "Tiny ceiling did not retire fixed mapped metadata safely");
    std::array<float, hidden> input{}, actual{};
    for (int i = 0; i < hidden; ++i) input[i] = std::sin(.032f * i);
    std::array<std::int32_t, used> ids{3, 11}; std::array<float, used> routes{.65f, .35f};
    const auto expected = oracle.compute(input, ids, routes), next_expected = next_oracle.compute(input, ids, routes);
    auto run = [&](const std::array<float, hidden>& answer) {
        actual.fill(-999.f);
        require(tsg::host_moe_cached_experts(first.segment, input.data(), ids.data(), routes.data(), actual.data(),
            "glm5-next selected expert seam", g_backend) == 1, "Mapped RAM scalar call failed");
        require(std::memcmp(actual.data(), answer.data(), sizeof(actual)) == 0, "Mapped RAM differs from resident CUDA oracle");
    };
    const auto first_cost = tsg::HostMoeMappedRam::instance().stats();
    run(expected);
    require(MappedRamFixtureStats().values[1] == 0
        && tsg::HostMoeMappedRam::instance().stats().admission_calls == first_cost.admission_calls,
        "First GPU-miss demand allocated mapped payload before probation");
    require(TSGgml_HostMoeRamCacheConfigure(capacity) == 1, "Positive resize discarded probation history");
    tsg::g_test_parallel_file_peak.store(0);
    const auto cold_parallel_calls = tsg::g_test_parallel_file_calls;
    TSGgml_TrimHostMoeExpertCache(0); run(expected);
    const auto cold = MappedRamFixtureStats(); const auto load_cold = RamFixtureStats();
    const auto staging_bytes = ledger.held(3);
    require(cold.values[1] > 0 && cold.values[3] > 0 && cold.values[11] > 0
        && cold.values[2] == ledger.held(6) && cold.values[6] == 0 && cold.values[14] == 0,
        "Cold mapped RAM did not retain real charged locked pages");
    require(tsg::g_test_parallel_file_calls > cold_parallel_calls
        && tsg::g_test_parallel_file_peak.load() >= 2 && tsg::g_test_parallel_file_active.load() == 0,
        "Cold mapped admission did not use concurrent joined production-style workers");
    TSGgml_TestMappedRamProbationClear();
    const auto ready_cost = tsg::HostMoeMappedRam::instance().stats();
    TSGgml_TrimHostMoeExpertCache(0); run(expected);
    const auto warm = MappedRamFixtureStats();
    require(warm.values[10] > cold.values[10] && RamFixtureStats().values[6] == load_cold.values[6],
        "Warm mapped RAM reread source instead of staging exact hits");
    const auto ready_after = tsg::HostMoeMappedRam::instance().stats();
    require(ready_after.plan_calls == ready_cost.plan_calls && ready_after.admission_calls == ready_cost.admission_calls,
        "Ready RAM hit with first probation membership repeated geometry or admission");
    TSGgml_TestMappedRamProbationClear();
    TSGgml_TrimHostMoeExpertCache(0);
    tsg::host_moe_expert_cache_test_file_slice_limit(512);
    run(expected);
    tsg::host_moe_expert_cache_test_file_slice_limit(32u << 20);
    require(TSGgml_TestMappedRamProbationKeys() == 0
        && tsg::HostMoeMappedRam::instance().stats().plan_calls == ready_after.plan_calls
        && RamFixtureStats().values[6] == load_cold.values[6],
        "Fragmented ready hits observed probation, planned geometry or reread the source");
    std::vector<float> bx(16 * hidden), br(16 * used), by(16 * hidden);
    std::vector<std::int32_t> bi(16 * used);
    for (int row = 0; row < 16; ++row) {
        for (int i = 0; i < hidden; ++i) bx[row * hidden + i] = std::sin(.032f * i + row * .071f);
        std::copy(ids.begin(), ids.end(), bi.begin() + row * used); std::copy(routes.begin(), routes.end(), br.begin() + row * used);
    }
    auto wide_segment = first.segment; wide_segment.seq_len = 16;
    const auto wide_expected = bulk_oracle.compute_batch(bx, bi, br);
    TSGgml_TrimHostMoeExpertCache(0);
    require(tsg::host_moe_cached_experts(wide_segment, bx.data(), bi.data(), br.data(), by.data(),
        "glm5-next bulk expert prefill", g_backend) == 1
        && std::memcmp(by.data(), wide_expected.data(), by.size() * sizeof(float)) == 0,
        "Mapped hit-only wide prefill parity failed");
    const auto wide = MappedRamFixtureStats();
    require(wide.values[10] > warm.values[10] && wide.values[11] == warm.values[11],
        "Wide prefill failed hit-only immutable reuse");
    require(TSGgml_HostMoeRamCacheConfigure(0) == 1 && TSGgml_HostMoeRamCacheConfigure(capacity) == 1,
        "Cannot clear mapped tier before wide cold scan");
    TSGgml_TrimHostMoeExpertCache(0);
    require(tsg::host_moe_cached_experts(wide_segment, bx.data(), bi.data(), br.data(), by.data(),
        "glm5-next bulk expert prefill", g_backend) == 1
        && std::memcmp(by.data(), wide_expected.data(), by.size() * sizeof(float)) == 0,
        "Cold hit-only wide prefill fallback parity failed");
    const auto scan = MappedRamFixtureStats();
    require(scan.values[1] == 0 && scan.values[3] == 0 && scan.values[11] == wide.values[11]
        && TSGgml_TestMappedRamProbationKeys() == 0,
        "Wide scan admitted mapped payloads or aged decode admission");
    TSGgml_TrimHostMoeExpertCache(0); run(expected);
    TSGgml_TrimHostMoeExpertCache(0); run(expected);
    for (auto fault : {tsg::HostMoeMappedRam::TestFault::copy, tsg::HostMoeMappedRam::TestFault::in_page}) {
        TSGgml_TrimHostMoeExpertCache(0);
        const auto prior = RamFixtureStats(); tsg::HostMoeMappedRam::test_fail_next(fault); run(expected);
        require(RamFixtureStats().values[6] > prior.values[6], "Guarded mapped-copy failure did not use exact buffered fallback");
    }
    TSGgml_TrimHostMoeExpertCache(0);
    const auto prefix_before = tsg::g_test_partial_file_reads.load();
    const auto failed_parallel_calls = tsg::g_test_parallel_file_calls;
    tsg::g_test_partial_file_failure = true;
    tsg::g_test_file_failure_countdown = 2; actual.fill(-999.f); bool interrupted = false;
    try { tsg::host_moe_cached_experts(first.segment, input.data(), ids.data(), routes.data(), actual.data(),
        "glm5-next selected expert seam", g_backend); }
    catch (const std::runtime_error&) { interrupted = true; }
    tsg::g_test_partial_file_failure = false;
    require(interrupted && std::all_of(actual.begin(), actual.end(), [](float x) { return x == -999.f; }),
        "Interrupted mapped staging published a partial result");
    require(tsg::g_test_partial_file_reads.load() == prefix_before + 1
        && tsg::g_test_parallel_file_calls > failed_parallel_calls
        && tsg::g_test_parallel_file_active.load() == 0 && tsg::g_test_file_failure_countdown == 0,
        "Partially filled mapped read did not join every worker before failure returned");
    TSGgml_TrimHostMoeExpertCache(0);
    require(TSGgml_HostMoeRamCacheConfigure(0) == 1 && ledger.held(6) == 0, "Mapped zero capacity did not fully retire");
    ledger.host_capacity = staging_bytes + control;
    require(TSGgml_HostMoeRamCacheConfigure(capacity) == 1, "Cannot configure optional mapped quota-refusal case");
    const auto refusals = ledger.refused; run(expected);
    TSGgml_TrimHostMoeExpertCache(0); run(expected);
    require(ledger.refused > refusals && MappedRamFixtureStats().values[1] == 0 && ledger.held(6) == control,
        "Host credit refusal did not retain only control and exact fallback");
    TSGgml_TrimHostMoeExpertCache(0);
    require(TSGgml_HostMoeRamCacheConfigure(0) == 1, "Cannot retire refused mapped case");
    ledger.host_capacity = 64LL << 20;
    require(TSGgml_HostMoeRamCacheConfigure(capacity) == 1, "Cannot restore mapped capacity");
    run(expected); TSGgml_TrimHostMoeExpertCache(0);
    ledger.fail_ram_commit = true; run(expected); ledger.fail_ram_commit = false;
    const auto refused = MappedRamFixtureStats();
    require(refused.values[1] == 0 && refused.values[3] == 0 && refused.values[2] == control,
        "Mapped failed commit retained a payload or refunded control ownership");
    TSGgml_TrimHostMoeExpertCache(0); run(expected);
    for (auto& part : first.quantized) tsg::host_moe_expert_cache_on_drop(part.data());
    require(MappedRamFixtureStats().values[1] == 0 && ledger.held(6) == control,
        "Source drop retained old mapped payloads");
    register_version(1); run(next_expected);
    TSGgml_TrimHostMoeExpertCache(0); run(next_expected);
    std::int64_t released = 0;
    require(TSGgml_EndHostMoeExpertCacheScope(token, &released) == 1, "Mapped fixture scope end failed"); token = 0;
    tsg::host_moe_expert_cache_release();
    require(ledger.valid && ledger.allocations.empty() && MappedRamFixtureStats().values[2] == 0,
        "Mapped fixture retained physical/quota/ledger ownership");
    require(tsg::SharedCacheCharge::detach(&ledger), "Mapped RAM prevented Host ledger detach"); ledger.attached = false;
    for (const auto& path : paths) require(std::remove(path.c_str()) == 0, "Mapped source handles survived teardown");
    std::puts("PASS: mapped RAM cold/warm and hit-only wide CUDA exact oracle; concurrent cold mapped admission; actual locked/charged pages; guarded partial/in-page fallback; strict-prefix worker failure joined/no publication; commit refusal; source generation; full teardown");
}
#endif
