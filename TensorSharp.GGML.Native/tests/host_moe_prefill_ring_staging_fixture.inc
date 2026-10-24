// Default-off wide ring expansion: real joined source workers and CUDA H2D.
#ifdef TSG_GGML_USE_CUDA
extern "C" int TSGgml_HostMoePrefillRingStagingVersion();
extern "C" int TSGgml_HostMoePrefillRingStagingEnabled();

extern "C" int TSGgml_HostMoeRamCacheConfigure(std::int64_t);
void host_moe_prefill_ring_staging_checks(const std::array<ggml_type, 3>& types, bool combined, bool coalesce = false, bool sort_only = false) {
    constexpr std::size_t ordinary = std::size_t(32) << 20, expanded = std::size_t(72) << 20;
    constexpr int h = 4096, count = 48, selected = 8, rows = 16;
    struct PartialReset {
        bool previous;
        explicit PartialReset(bool enabled) : previous(tsg::g_test_partial_file_failure) {
            if (enabled) tsg::g_test_partial_file_failure = true;
        }
        ~PartialReset() { tsg::g_test_partial_file_failure = previous; }
    } partial_reset(sort_only);
    require(TSGgml_HostMoePrefillRingStagingVersion() == 1 && TSGgml_HostMoePrefillRingStagingEnabled() == 1,
        "Wide ring fixture lacks the enabled native capability");
    require(TSGgml_HostMoeDecodeStagingEnabled() == (combined ? 1 : 0), "Wide ring scalar-staging configuration differs");
    tsg::host_moe_expert_cache_release();
    Weights weights; weights.types = types;
    weights.fill(.87f, count, selected, h, types[0] == GGML_TYPE_Q8_0 ? 512 : 1280);
    weights.segment.pooled_model_experts = count;
    const auto stride_bytes = (weights.segment.gate_bytes + weights.segment.up_bytes + weights.segment.down_bytes) / count;
    const auto payload = stride_bytes * 24;
    require(payload > expanded, "Wide fixture must reuse an expanded ring slot");
    FullResidentGraph scalar(weights), wide(weights, rows);
    std::vector<float> input(h * rows), routes(selected * rows), actual(h * rows);
    std::vector<std::int32_t> low(selected * rows), high(selected * rows);
    for (int row = 0; row < rows; ++row) {
        for (int i = 0; i < h; ++i) input[row*h+i] = std::cos(.021f*i + .007f*row);
        for (int k = 0; k < selected; ++k) {
            routes[row*selected+k] = float(k+1)/36;
            constexpr int permutation[selected] = {7,1,6,0,5,2,4,3};
            const auto original_id = sort_only ? permutation[k] : k;
            low[row*selected+k] = original_id;
            high[row*selected+k] = 24 + (row%3)*selected + original_id;
        }
    }
    const auto expected_low = wide.compute_batch(input, low, routes);
    const auto expected_high = wide.compute_batch(input, high, routes);
    const auto scalar_expected = scalar.compute_batch(std::vector<float>(input.begin(), input.begin()+h),
        std::vector<std::int32_t>(low.begin(), low.begin()+selected), std::vector<float>(routes.begin(), routes.begin()+selected));
    const auto path = fixture_file_path("host-moe-wide-ring72-");
    struct FileCleanup { const std::string& path; ~FileCleanup() { std::remove(path.c_str()); } } file_cleanup{path};
    std::array<std::int64_t, 3> offsets{};
    {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        for (int part = 0; part < 3; ++part) {
            file.write("unaligned-source!", 17); offsets[part] = static_cast<std::int64_t>(file.tellp());
            file.write(reinterpret_cast<const char*>(weights.quantized[part].data()), weights.quantized[part].size());
        }
        require(bool(file), "Cannot write wide ring source");
    }
    for (auto& part : weights.quantized) std::fill(part.begin(), part.end(), 0);
    auto call = [&](bool is_wide, bool upper) {
        auto hm = weights.segment; hm.seq_len = is_wide ? rows : 1;
        std::fill(actual.begin(), actual.end(), -999);
        const auto& ids = upper ? high : low;
        const auto sort_before = sort_only ? tsg::host_moe_expert_cache_test_file_sort_snapshot() : tsg::HostMoeFileSortTestSnapshot{};
        tsg::HostMoeCacheTestSnapshot lru_before{};
        if (sort_only) {
            (void)tsg::host_moe_expert_cache_test_lifecycle_snapshot(hm,g_backend,lru_before);
            tsg::g_test_file_spans.clear();
        }
        const auto reads_before = tsg::g_test_file_reads;
        struct CaptureReset {
            bool previous;
            explicit CaptureReset(bool enabled) : previous(tsg::g_test_capture_file_spans) {
                if (enabled) tsg::g_test_capture_file_spans = true;
            }
            ~CaptureReset() { tsg::g_test_capture_file_spans = previous; }
        } capture(sort_only);
        require(tsg::host_moe_cached_experts(hm, input.data(), ids.data(), routes.data(), actual.data(),
            is_wide ? "glm5-next bulk expert prefill" : "glm5-next selected expert seam", g_backend) == 1,
            "Wide ring selected seam declined");
        const auto& expected = is_wide ? upper ? expected_high : expected_low : scalar_expected;
        require(std::all_of(expected.begin(), expected.end(), [](float v) { return std::isfinite(v); })
            && std::any_of(expected.begin(), expected.end(), [](float v) { return v != 0; })
            && std::memcmp(actual.data(), expected.data(), expected.size()*sizeof(float)) == 0,
            "Wide ring changed complete finite resident-oracle output");
        if (sort_only) file_sort_fixture_completed(hm,ids,is_wide ? selected*rows : selected,lru_before,sort_before,reads_before);
    };
    const std::vector<int> faults = sort_only ? std::vector<int>{0,-4,23,24} : combined ? std::vector<int>{0} : coalesce
        ? std::vector<int>{0, -1, -2, -3, -4, -5, -6, 23, 24, 25}
        : std::vector<int>{0, -1, -2, -3, -4, -5, 23, 24};
    for (int fault : faults) {
        tsg::host_moe_expert_cache_release();
        for (int part = 0; part < 3; ++part)
            require(TSGgml_RegisterHostFileSource(weights.quantized[part].data(), weights.quantized[part].size(),
                path.c_str(), offsets[part]) == 1, "Cannot register poisoned wide source");
        SharedLedger ledger; ledger.capacity = 512LL << 20; ledger.host_capacity = expanded;
        require(ledger.attach(true), "Cannot attach wide staging ledger");
        std::uint64_t token = 0;
        require(TSGgml_BeginHostMoeExpertCacheScopeWithDecodeRows(ledger.capacity, rows, 1, 1, &token) == 1,
            "Cannot begin wide staging scope");
        struct Cleanup {
            std::uint64_t& token;
            ~Cleanup() {
                tsg::g_test_file_failure_countdown = 0; tsg::host_moe_expert_cache_test_fail_next(0);
                tsg::host_moe_expert_cache_test_file_workspace_fail(0);
                tsg::host_moe_expert_cache_test_file_slice_limit(std::size_t(32) << 20);
                if (token) { std::int64_t bytes = 0; TSGgml_EndHostMoeExpertCacheScope(token, &bytes); }
            }
        } cleanup{token};
        call(false, false); // Ordinary scalar creates its original32MiB arena.
        call(true, false); // Materialize wide device scratch with hot experts, no file read.
        require(tsg::host_moe_expert_cache_test_file_workspace_bytes() == (combined ? expanded : ordinary), "Hot wide preparation changed staging");
        const void* original = tsg::host_moe_expert_cache_test_file_workspace_data();
        std::array<std::uint8_t, 64> prefix{}; std::memcpy(prefix.data(), original, prefix.size());
        if (fault == -1) {
            ledger.host_capacity = expanded-1;
            require(!tsg::host_moe_expert_cache_test_try_scalar_workspace()
                && original == tsg::host_moe_expert_cache_test_file_workspace_data()
                && std::memcmp(prefix.data(), original, prefix.size()) == 0 && ledger.host_pending == 0,
                "Wide prephysical refusal replaced its old arena/bytes");
        }
        if (fault == -2) ledger.fail_commit = true;
        if (fault == -3) tsg::host_moe_expert_cache_test_file_workspace_fail(1);
        if (fault == -4) tsg::g_test_file_failure_countdown = 2;
        if (fault == -5) tsg::host_moe_expert_cache_test_file_slice_limit(std::size_t(1) << 20);
        if (fault == -6) require(TSGgml_HostMoeRamCacheConfigure(1) == 1, "Cannot enable bounded RAM eligibility refusal");
        if (fault == 23 || fault == 24 || fault == 25) tsg::host_moe_expert_cache_test_fail_next(fault);
        const auto drains = tsg::host_moe_expert_cache_test_payload_upload_drains();
        const auto coalesced_calls = tsg::host_moe_expert_cache_test_coalesced_ring_calls();
        const auto reads_before = tsg::g_test_file_reads;
        const auto partial_before = tsg::g_test_partial_file_reads.load();
        const auto sort_before_failure = sort_only ? tsg::host_moe_expert_cache_test_file_sort_snapshot() : tsg::HostMoeFileSortTestSnapshot{};
        const auto before_wide_reserves = ledger.reserve_calls;
        bool failed = false;
        try { call(true, true); } catch (const std::exception&) { failed = true; }
        require(failed == (fault != 0 && fault != -1 && fault != -5 && fault != -6), "Wide staging failure fell back or refusal failed");
        if (combined) require(ledger.reserve_calls == before_wide_reserves,
            "Combined scalar/wide expansion allocated or reserved a second arena");
        ledger.fail_commit = false;
        if (failed) {
            if (sort_only) {
                require(tsg::host_moe_expert_cache_test_file_sort_snapshot().completed_calls == sort_before_failure.completed_calls,
                    "Failed sorted upload published completed sort activity");
                if (fault == -4) require(tsg::g_test_partial_file_reads.load() == partial_before+1,
                    "Sorted joined-read failure did not copy a strict real-file prefix");
            }
            if (coalesce) require(tsg::host_moe_expert_cache_test_coalesced_ring_calls() == coalesced_calls,
                "Failed coalesced file call published successful activity");
            require(std::all_of(actual.begin(), actual.end(), [](float v) { return v == -999; }),
                "Failed wide staging published caller output");
            if (fault == -2 || fault == -3)
                require(tsg::host_moe_expert_cache_test_file_workspace_bytes() == 0 && ledger.host_committed == 0,
                    "Failed expanded arena retained physical storage or tickets");
            else if (fault == 25) require(tsg::g_test_file_reads == reads_before
                && tsg::host_moe_expert_cache_test_payload_upload_drains() == drains,
                "Aliased target preflight performed source I/O or queued an upload");
            else require(tsg::host_moe_expert_cache_test_payload_upload_drains() > drains,
                "Queued wide upload failure escaped its drain");
            if (coalesce && fault == -4) require(tsg::g_test_partial_file_reads.load() == partial_before+1,
                "Merged read failure did not fill a strict real-file prefix");
            call(true, true); // Invalid slots must reread complete immutable payloads.
        }
        if (coalesce) {
            const bool active = fault != -1 && fault != -5 && fault != -6;
            require(tsg::host_moe_expert_cache_test_coalesced_ring_calls() == coalesced_calls + (active ? 1 : 0),
                "Actual coalesced activity differs from strict eligibility/refusal");
            if (active) {
                const auto proof = tsg::host_moe_expert_cache_test_completed_coalesced_ring_plan();
                require(proof[0] > proof[1] && proof[1] > 0 && proof[2] == payload
                    && proof[3] > 0 && proof[3] <= payload && proof[4] == (std::size_t(16) << 20),
                    "Coalesced ring lacks exact payload/task reduction/adjacency/cap proof");
                std::printf("[TEST-COALESCE16] fault=%d read_tasks=%zu->%zu payload=%zu coalesced_bytes=%zu read_cap=%zu\n",
                    fault, proof[0], proof[1], proof[2], proof[3], proof[4]);
            }
        }
        const auto plan = tsg::host_moe_expert_cache_test_completed_ring_plan();
        const auto arena = fault == -1 ? ordinary : expanded;
        const auto plan_arena = fault == -5 ? std::size_t(1) << 20 : arena;
        require(plan[0] == plan_arena && plan[1] == plan_arena/2 && plan[2] == (payload+plan_arena/2-1)/(plan_arena/2)
            && plan[2] >= 3 && tsg::host_moe_expert_cache_test_file_workspace_bytes() == arena,
            "Wide ring did not use its actual admitted plan/retained arena");
        require(ledger.valid && ledger.host_pending == 0 && ledger.host_committed == arena,
            "Wide staging violated physical-before-refund accounting");
        tsg::host_moe_expert_cache_test_file_slice_limit(ordinary);
        const auto reads = tsg::g_test_file_reads, reserves = ledger.reserve_calls;
        const auto coalesced_before_warm = tsg::host_moe_expert_cache_test_coalesced_ring_calls();
        call(true, true);
        require(tsg::g_test_file_reads == reads && ledger.reserve_calls == reserves, "Warm wide ring reread or regrew storage");
        if (coalesce) require(tsg::host_moe_expert_cache_test_coalesced_ring_calls() == coalesced_before_warm,
            "Warm no-I/O call claimed new coalesced activity");
        call(false, false);
        if (coalesce) require(tsg::host_moe_expert_cache_test_coalesced_ring_calls() == coalesced_before_warm,
            "Scalar decode executed coalesced prefill planner");
        require(tsg::host_moe_expert_cache_test_file_workspace_bytes() == arena, "Narrow transition dropped retained expanded ownership");
        if (fault == -6) require(TSGgml_HostMoeRamCacheConfigure(0) == 1, "Cannot clear bounded RAM eligibility refusal");
        std::int64_t released = 0;
        require(TSGgml_EndHostMoeExpertCacheScope(token, &released) == 1, "Wide ring scope end refused");
        token = 0; ledger.check_empty(); require(ledger.detach(), "Wide ring retained native tickets");
    }
    if (sort_only) {
        tsg::host_moe_expert_cache_release(); // Close exact registry handles before checked fixture deletion.
        require(std::remove(path.c_str()) == 0, "Sorted fixture retained a source handle or generated file");
        std::puts("PASS: sorted PP+N1 original slots/route-LRU, exact source/file spans, unchanged72MiB task/window/H2D geometry, complete CUDA bit oracle, real prefix read/queued upload/epoch failure and joined cleanup");
    }
    if (!sort_only) {
        std::printf("PASS: wide ring72 exact payload=%lld, joined workers/two36MiB slots, prephysical refusal, physical/commit/queued-upload/epoch failures, narrow retention and teardown\n", static_cast<long long>(payload));
    }
}
#endif
