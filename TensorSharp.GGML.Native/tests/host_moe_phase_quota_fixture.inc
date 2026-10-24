// Three real quantized type groups share one physical ceiling. Included in the
// CUDA lifecycle fixture; this is operator/ownership evidence, not model quality.
#ifdef TSG_GGML_USE_CUDA
void glm_phase_quota_checks(bool coexist = false, bool direct_only_wide = false, bool ring_only_wide = false) {
    constexpr int count = 64, top = 8, maximum_rows = 1024, decode_rows = 3;
    tsg::host_moe_expert_cache_release();
    std::array<Weights, 3> weights;
    weights[0].types = {GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0};
    weights[1].types = {GGML_TYPE_IQ1_M, GGML_TYPE_IQ1_M, GGML_TYPE_IQ4_NL};
    weights[2].types = {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ4_NL};
    std::vector<tsg::HostMoePhaseGroupMinimum> minima;
    std::array<std::map<int, std::unique_ptr<FullResidentGraph>>, 3> oracles;
    std::uint64_t planning_token = 0;
    require(TSGgml_BeginHostMoeExpertCacheScopeWithDecodeRows(64LL << 20,
        maximum_rows, 3, decode_rows, &planning_token) == 1, "Cannot begin size-only group plan");
    struct PlanningCleanup {
        std::uint64_t& token;
        ~PlanningCleanup() { if (token) { std::int64_t released = 0; TSGgml_EndHostMoeExpertCacheScope(token, &released); } }
    } planning_cleanup{planning_token};
    for (int group = 0; group < 3; ++group) {
        auto& w = weights[group]; w.fill(.7f + group * .4f, count, top);
        w.segment.layer = group; w.segment.pooled_model_experts = coexist ? count * 16 : count;
        std::size_t narrow = 0, wide = 0;
        require(tsg::host_moe_expert_cache_minimum_phase_plan(w.segment, decode_rows,
            maximum_rows, g_backend, narrow, wide), "Cannot measure exact phase group minimum");
        minima.push_back({tsg::host_moe_expert_group_key(w.segment), narrow, wide});
        for (int rows : {1, 20, 32, 64, 128, 256, 384, 512, 1024})
            oracles[group].emplace(rows, std::make_unique<FullResidentGraph>(w, rows));
    }
    std::int64_t planning_released = 0;
    require(TSGgml_EndHostMoeExpertCacheScope(planning_token, &planning_released) == 1
        && planning_released == 0, "Metadata-only group planning retained cache payload");
    planning_token = 0;
    tsg::HostMoePhaseQuotaPlan plan(minima);
    const auto all_wide = minima[0].wide_bytes + minima[1].wide_bytes + minima[2].wide_bytes;
    const auto budget = (coexist ? all_wide : plan.minimum()) + 17;
    require(coexist ? budget >= all_wide : budget < all_wide,
        "Fixture does not distinguish simultaneous-wide and pressure phases");
    SharedLedger ledger; ledger.capacity = static_cast<std::int64_t>(budget);
    ledger.host_capacity = 65536; ledger.multiple_cache_entries = true;
    require(ledger.attach(true), "Cannot attach actual three-group ownership ledger");
    std::uint64_t token = 0;
    const auto path = fixture_file_path("glm-phase-groups-");
    struct Cleanup {
        std::string path; std::uint64_t& token;
        ~Cleanup() {
            std::int64_t released = 0;
            if (token) TSGgml_EndHostMoeExpertCacheScope(token, &released);
            tsg::g_test_file_failure_countdown = 0;
            tsg::host_moe_expert_cache_test_fail_next(0);
            tsg::host_moe_expert_cache_release();
            tsg::host_moe_expert_cache_test_file_slice_limit(32u << 20);
            std::remove(path.c_str());
        }
    } cleanup{path, token};
    std::int64_t offsets[3][3]{};
    {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        require(bool(file), "Cannot create three-group source");
        for (int group = 0; group < 3; ++group) for (int part = 0; part < 3; ++part) {
            file.write("group-offset", 12); offsets[group][part] = static_cast<std::int64_t>(file.tellp());
            const auto& bytes = weights[group].quantized[part];
            file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }
        require(bool(file), "Cannot write three-group source");
    }
    for (int group = 0; group < 3; ++group) for (int part = 0; part < 3; ++part) {
        auto& bytes = weights[group].quantized[part];
        require(TSGgml_RegisterHostFileSource(bytes.data(), bytes.size(), path.c_str(), offsets[group][part]) == 1,
            "Cannot register three-group immutable extent");
        std::fill(bytes.begin(), bytes.end(), 0); // GPU oracle owns original bytes.
    }
    tsg::host_moe_expert_cache_test_file_slice_limit(65536);
    require(TSGgml_BeginHostMoeExpertCacheScopeWithDecodeRows(budget + 1,
        maximum_rows, 3, decode_rows, &token) == 1, "Cannot begin three-group scope");
    std::int64_t released = 0;
    require(TSGgml_UpdateHostMoeExpertCacheScope(token, plan.minimum() - 1, &released) == 1,
        "Cannot lower the initial scope before phase installation");
    require(!tsg::host_moe_expert_cache_install_phase_plan(minima, maximum_rows, decode_rows, g_backend)
        && ledger.allocations.empty(), "Insufficient total floor installed a partial plan");
    require(TSGgml_UpdateHostMoeExpertCacheScope(token, budget, &released) == 1,
        "Cannot recover complete group floor");
    require(!tsg::host_moe_expert_cache_install_phase_plan(minima, maximum_rows - 1, decode_rows, g_backend),
        "Mismatching prefill shape installed a plan");
    auto duplicate = minima; duplicate[1] = duplicate[0];
    bool rejected_duplicate = false;
    try { rejected_duplicate = !tsg::host_moe_expert_cache_install_phase_plan(duplicate, maximum_rows, decode_rows, g_backend); }
    catch (const std::invalid_argument&) { rejected_duplicate = true; }
    require(rejected_duplicate && tsg::host_moe_expert_cache_install_phase_plan(minima, maximum_rows, decode_rows, g_backend)
        && !tsg::host_moe_expert_cache_install_phase_plan(minima, maximum_rows, decode_rows, g_backend),
        "Plan publication is not atomic and exclusive");
    std::size_t compared = 0;
    std::uint64_t narrow_file_reads = 0, wide_direct_reads = 0;
    DeviceInputs widest_device(maximum_rows, top);
    const auto reject_complete_shape = [&](int rows, int selected) {
        auto hm = weights[0].segment; hm.seq_len = rows; hm.n_used = selected;
        std::vector<float> input(rows * hidden), routes(rows * selected, 1.0f / selected), actual(rows * hidden, -999.0f);
        std::vector<std::int32_t> ids(rows * selected, 0);
        const auto before = Stats(); const auto reserved = ledger.reserve_calls, files_before = tsg::g_test_file_reads;
        require(tsg::host_moe_cached_experts(hm, input.data(), ids.data(), routes.data(), actual.data(),
            "glm5-next bulk expert prefill", g_backend) == 0,
            "GLM overbound row/high-topk route shape was accepted");
        const auto after = Stats();
        require(before.reserved == after.reserved && before.calls == after.calls && before.hits == after.hits
            && before.misses == after.misses && ledger.reserve_calls == reserved && tsg::g_test_file_reads == files_before
            && std::all_of(actual.begin(), actual.end(), [](float value) { return value == -999.0f; }),
            "GLM complete overbound shape read/allocated/mutated/published before refusal");
    };
    reject_complete_shape(1025, 8); reject_complete_shape(1024, 9); reject_complete_shape(129, 64);
    const auto reject_complete_device_routes = [&](bool backend_fault) {
        auto hm = weights[0].segment; hm.seq_len = maximum_rows;
        hm.moe_in = widest_device.x; hm.weights = widest_device.routes; hm.moe_out = widest_device.output;
        std::vector<float> input(maximum_rows * hidden), routes(maximum_rows * top, 1.0f / top), actual(maximum_rows * hidden, -999.0f);
        std::vector<std::int32_t> ids(maximum_rows * top, 0);
        if (!backend_fault) routes.back() = (std::numeric_limits<float>::quiet_NaN)();
        ggml_backend_tensor_set(widest_device.x, input.data(), 0, input.size() * sizeof(float));
        ggml_backend_tensor_set(widest_device.routes, routes.data(), 0, routes.size() * sizeof(float));
        ggml_backend_tensor_set(widest_device.output, actual.data(), 0, actual.size() * sizeof(float));
        const auto before = Stats(); const auto reserved = ledger.reserve_calls, files_before = tsg::g_test_file_reads;
        bool threw = false; int result = -99;
        if (backend_fault) tsg::host_moe_expert_cache_test_fail_next(5);
        try { result = tsg::host_moe_cached_experts(hm, nullptr, ids.data(), nullptr, nullptr,
            "glm5-next bulk expert prefill", g_backend); }
        catch (const std::runtime_error& error) {
            if (backend_fault && std::strcmp(error.what(), "qwen4exp expert cache: routing download failed") == 0) threw = true;
            else throw;
        }
        require(backend_fault ? threw : result == 0, "Last8192 route NaN/backend failure was accepted");
        const auto after = Stats();
        ggml_backend_tensor_get(widest_device.output, actual.data(), 0, actual.size() * sizeof(float));
        require(before.reserved == after.reserved && before.calls == after.calls && before.hits == after.hits
            && before.misses == after.misses && ledger.reserve_calls == reserved && tsg::g_test_file_reads == files_before
            && ledger.pending == 0 && ledger.valid
            && std::all_of(actual.begin(), actual.end(), [](float value) { return value == -999.0f; }),
            "Complete8192 device route refusal mutated/read/allocated/published before validation");
    };
    reject_complete_device_routes(false); reject_complete_device_routes(true);
    // fault=1 is an exact pre-dispatch wide-replacement error; fault=2 is
    // a deliberately injected file read error. Other exceptions fail the test.
    auto run = [&](int group, int rows, bool expected_success, int fault = 0) {
        std::vector<float> input(rows * hidden), routes(rows * top), actual(rows * hidden, -999.0f), expected;
        std::vector<std::int32_t> ids(rows * top);
        for (int row = 0; row < rows; ++row) {
            for (int h = 0; h < hidden; ++h) input[row * hidden + h] = std::cos(h * .041f + row * .31f);
            for (int k = 0; k < top; ++k) {
                ids[row * top + k] = (row * top + k) % count;
                routes[row * top + k] = (k + 1.0f) / 36.0f;
            }
        }
        if (rows <= 8) {
            expected.resize(actual.size());
            for (int row = 0; row < rows; ++row) {
                std::vector<float> x(hidden), probability(top); std::vector<std::int32_t> selected(top);
                std::copy_n(input.begin() + row * hidden, hidden, x.begin());
                std::copy_n(routes.begin() + row * top, top, probability.begin());
                std::copy_n(ids.begin() + row * top, top, selected.begin());
                const auto answer = oracles[group].at(1)->compute_batch(x, selected, probability);
                std::copy(answer.begin(), answer.end(), expected.begin() + row * hidden);
            }
        } else expected = oracles[group].at(rows)->compute_batch(input, ids, routes);
        auto hm = weights[group].segment; hm.seq_len = rows; hm.independent_decode_rows = rows == 3;
        // Exercise the COMPLETE8192 routing download and device/output bridge
        // at the real maximum; smaller rows retain the existing host oracle.
        const bool device_bridge = rows == maximum_rows;
        if (device_bridge) {
            ggml_backend_tensor_set(widest_device.x, input.data(), 0, input.size() * sizeof(float));
            ggml_backend_tensor_set(widest_device.routes, routes.data(), 0, routes.size() * sizeof(float));
            ggml_backend_tensor_set(widest_device.output, actual.data(), 0, actual.size() * sizeof(float));
            hm.moe_in = widest_device.x; hm.weights = widest_device.routes; hm.moe_out = widest_device.output;
        }
        const auto before_execution = ExecutionStats();
        const auto before_files = tsg::g_test_file_reads, before_direct = tsg::g_test_direct_file_reads;
        const auto before_ring = tsg::g_test_file_ring_calls;
        int ok = -99; bool threw = false;
        try {
            ok = tsg::host_moe_cached_experts(hm, device_bridge ? nullptr : input.data(), ids.data(),
                device_bridge ? nullptr : routes.data(), device_bridge ? nullptr : actual.data(),
                rows > 8 ? "glm5-next bulk expert prefill" : "glm5-next selected expert seam", g_backend);
            if (device_bridge) {
                tsg::sync_backend(g_backend);
                ggml_backend_tensor_get(widest_device.output, actual.data(), 0, actual.size() * sizeof(float));
            }
        } catch (const std::runtime_error& error) {
            if (fault == 1 && std::strcmp(error.what(),
                "GLM wide scratch replacement refused/failed; persistent weights remain owned") == 0) { ok = 0; threw = true; }
            else if (fault == 2 && tsg::g_test_file_failure_countdown == 0
                && std::strcmp(error.what(), "Injected interrupted expert file read") == 0) { ok = 0; threw = true; }
            else throw;
        }
        if (direct_only_wide) {
            if (rows <= 8) {
                require(tsg::g_test_direct_file_reads == before_direct,
                    "Direct read experiment leaked into N1/N3 decode phase");
                narrow_file_reads += tsg::g_test_file_reads - before_files;
            } else wide_direct_reads += tsg::g_test_direct_file_reads - before_direct;
        }
        if (ring_only_wide && rows <= 8)
            require(tsg::g_test_file_ring_calls == before_ring, "Prefill ring scheduling leaked into N1/N3/compact execution");
        if (fault) require(threw && ExecutionStats().submissions == before_execution.submissions,
            "Injected phase failure did not abort before compute");
        require((ok == 1) == expected_success, "Three-group execution/refusal mismatch");
        if (expected_success) {
            require(std::all_of(actual.begin(), actual.end(), [](float x) { return std::isfinite(x); })
                && std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0,
                "Group quota or changing wide scratch changed complete CUDA output");
            compared += actual.size();
        } else require(std::all_of(actual.begin(), actual.end(), [](float x) { return x == -999.0f; }),
            "Refused group partially published output");
        require(ledger.valid && ledger.pending == 0 && ledger.committed <= static_cast<std::int64_t>(budget),
            "Three-group transaction exceeded shared credit or lost physical ownership");
    };
    using Snapshot = tsg::HostMoeCacheTestSnapshot;
    auto snapshot = [&](int group) {
        Snapshot value{};
        require(tsg::host_moe_expert_cache_test_lifecycle_snapshot(weights[group].segment, g_backend, value),
            "Expected current phase group has no live owner");
        return value;
    };
    auto stable_weights = [&](const Snapshot& a, const Snapshot& b) {
        require(a.entry_identity == b.entry_identity && a.weight_buffer == b.weight_buffer
            && a.weight_charge == b.weight_charge && a.narrow_buffer == b.narrow_buffer
            && a.narrow_charge == b.narrow_charge && a.weight_bytes == b.weight_bytes
            && a.narrow_bytes == b.narrow_bytes, "Narrow phase or reshape rebuilt affordable persistent weights");
    };
    if (!coexist && !direct_only_wide) {
        // Reproduce the weighted decode starvation gate with real allocations:
        // the rare group's complete prefill union leaves enough room for HALF
        // the absent shares, but not their full shares. Cold decode tables do
        // not grow later, so the phase transition must retire that hoarder.
        require(TSGgml_EndHostMoeExpertCacheScope(token, &released) == 1,
            "Cannot isolate weighted decode rebalance scope");
        token = 0; ledger.check_empty();
        auto weighted = minima;
        weighted[0].weight = 16; weighted[1].weight = 1; weighted[2].weight = 24;
        const tsg::HostMoePhaseQuotaPlan weighted_plan(weighted);
        require(TSGgml_BeginHostMoeExpertCacheScopeWithDecodeRows(budget,
            maximum_rows, 3, decode_rows, &token) == 1
            && tsg::host_moe_expert_cache_install_phase_plan(weighted, maximum_rows, decode_rows, g_backend),
            "Cannot install weighted decode rebalance fixture");
        // PP512 retains the same complete 64-expert union while its smaller
        // graph leaves room between half and all of the absent decode shares.
        // PP1024's scratch alone makes the old half-share gate fire as well.
        run(1, 512, true);
        const auto rare_before = snapshot(1);
        const auto rare_owned = static_cast<std::size_t>(Stats().reserved);
        const auto rare_share = weighted_plan.quota(weighted[1].key, budget, false);
        const auto absent = weighted_plan.quota(weighted[0].key, budget, false)
            + weighted_plan.quota(weighted[2].key, budget, false);
        require(tsg::HostMoePhaseQuotaPlan::exceeds_twice_share(rare_owned, rare_share)
            && budget - rare_owned >= absent / 2 && budget - rare_owned < absent,
            "Weighted fixture does not reproduce the old half-share false negative");
        const auto before_releases = ledger.release_calls;
        run(0, 3, true);
        Snapshot retired{};
        require(!tsg::host_moe_expert_cache_test_lifecycle_snapshot(weights[1].segment, g_backend, retired)
            && ledger.release_calls > before_releases,
            "Decode transition retained the rare hoarder and starved absent full shares");
        run(2, 3, true); run(1, 3, true);
        const auto rare_after = snapshot(1);
        require(rare_after.capacity < rare_before.capacity && rare_after.wide_rows == 0
            && Stats().reserved <= static_cast<std::int64_t>(budget) && ledger.valid && ledger.pending == 0,
            "Decode rebalance failed to restore weighted capacity with real physical retirement");
        require(TSGgml_EndHostMoeExpertCacheScope(token, &released) == 1,
            "Weighted decode rebalance scope did not drain");
        token = 0; ledger.check_empty();
        require(TSGgml_BeginHostMoeExpertCacheScopeWithDecodeRows(budget + 1,
            maximum_rows, 3, decode_rows, &token) == 1
            && TSGgml_UpdateHostMoeExpertCacheScope(token, budget, &released) == 1
            && tsg::host_moe_expert_cache_install_phase_plan(minima, maximum_rows, decode_rows, g_backend),
            "Cannot restore original affordable-union fixture");
        std::printf("PASS: weighted decode rebalance rare=%zu share=%zu absent=%zu room=%zu; old half-share gate false, real retirement and complete CUDA output exact\n",
            rare_owned, rare_share, absent, budget - rare_owned);
    }
    if (coexist) {
        // The pool permits 1024 hot slots, while each complete current union
        // has only 64. The old near-global cold ceiling greedily fills hundreds
        // and evicts peers even though all three measured 64-slot plans fit.
        // Start at the measured maximum width: a smaller graph may legally
        // spend its scratch savings on optional slots, beyond this fixture's
        // 64-slot lifecycle snapshot. That is not a cache ownership failure.
        std::array<Snapshot, 3> original{};
        for (int group = 0; group < 3; ++group) {
            run(group, maximum_rows, true); original[group] = snapshot(group);
            require(original[group].capacity == count,
                "Optional cold slots consumed another affordable wide group's protected floor");
        }
        std::array<std::int64_t, 24> entries{}; int actual_entries = 0;
        require(TSGgml_HostMoeExpertCacheEntryStats(entries.data(), 3, &actual_entries) == 1 && actual_entries == 3,
            "Greedy cold wide planning evicted an otherwise affordable type group");
        const auto reserves = ledger.reserve_calls, releases = ledger.release_calls;
        const auto reads = tsg::g_test_file_reads;
        for (int rows : {3, 1, maximum_rows, 3, maximum_rows}) for (int group = 0; group < 3; ++group) {
            run(group, rows, true); stable_weights(original[group], snapshot(group));
        }
        require(ledger.reserve_calls == reserves && ledger.release_calls == releases
            && tsg::g_test_file_reads == reads,
            "Affordable whole unions were reallocated, evicted or reread during warm group switches");
        require(TSGgml_HostMoeExpertCacheEntryStats(entries.data(), 3, &actual_entries) == 1 && actual_entries == 3,
            "Warm phase transitions lost a coexisting type group");
        for (int rows : {128, 256, 384, 512, 1024}) for (int group = 0; group < 3; ++group) {
            run(group, rows, true); stable_weights(original[group], snapshot(group));
        }
        for (int group = 0; group < 3; ++group) original[group] = snapshot(group);
        // Tighten to the actual owned total after shrinking only the current
        // graph. Growing it again must reclaim an inactive group before the
        // injected allocation failure, rather than refunding virtual credit.
        run(0, 20, true); const auto small = snapshot(0);
        stable_weights(original[0], small);
        const auto tightened = Stats().reserved;
        require(tightened >= static_cast<std::int64_t>(plan.minimum())
            && tightened <= static_cast<std::int64_t>(budget)
            && original[0].wide_bytes > small.wide_bytes,
            "Fixture does not force real inactive-owner reclaim before wide growth");
        const auto before_tighten_releases = ledger.release_calls;
        require(TSGgml_UpdateHostMoeExpertCacheScope(token, tightened, &released) == 1
            && released == 0 && ledger.release_calls == before_tighten_releases,
            "Tightening to the actual owned total unnecessarily discarded a group");
        ledger.capacity = tightened;
        tsg::host_moe_expert_cache_test_fail_next(17);
        run(0, maximum_rows, false, 1);
        const auto failed = snapshot(0); stable_weights(small, failed);
        require(failed.wide_rows == 0 && failed.wide_bytes == 0
            && failed.wide_buffer == 0 && failed.wide_charge == 0,
            "Failed post-reclaim allocation must retain weights/narrow, not a stale wide graph");
        require(TSGgml_HostMoeExpertCacheEntryStats(entries.data(), 3, &actual_entries) == 1
            && actual_entries < 3 && ledger.release_calls > before_tighten_releases
            && Stats().reserved <= tightened && ledger.pending == 0 && ledger.valid,
            "Post-reclaim failure did not physically retire inactive owners or release its pending ticket");
        ledger.capacity = static_cast<std::int64_t>(budget);
        require(TSGgml_UpdateHostMoeExpertCacheScope(token, budget, &released) == 1,
            "Cannot restore an admitted phase budget after the injected failure");
        run(0, maximum_rows, true); stable_weights(small, snapshot(0));
        run(1, maximum_rows, true); run(2, maximum_rows, true);
    } else {
    // Actual cache retention: a large affordable entry is not reduced to its
    // fair cold-narrow quota merely because the current phase is N3/N1 decode.
    run(0, 128, true); const auto first = snapshot(0);
    const auto first_reserves = ledger.reserve_calls;
    run(0, 3, true); run(0, 1, true);
    stable_weights(first, snapshot(0));
    require(ledger.reserve_calls == first_reserves && snapshot(0).wide_rows == 128,
        "Decode invalidated the retained whole-union graph");
    // Cold sibling decode uses affordable spare capacity, not its optional fair
    // share if that would unnecessarily evict this still-affordable wide owner.
    run(1, 3, true); stable_weights(first, snapshot(0));
    // Complete min-1 gate precedes any eviction, ticket release or metadata
    // mutation. The same native scope resumes without reinstallation.
    const auto before_budget = Stats(); const auto before_resizes = ledger.release_calls;
    require(TSGgml_UpdateHostMoeExpertCacheScope(token, plan.minimum() - 1, &released) == 0
        && released == 0 && Stats().reserved == before_budget.reserved && Stats().budget == before_budget.budget
        && ledger.release_calls == before_resizes, "Sub-floor resize mutated the accepted phase state");
    stable_weights(first, snapshot(0)); run(0, 3, true);
    // A failed cold allocation is allowed to discard inactive optional cache,
    // but may not retain pending credit or publish a partial output.
    ledger.fail_commit = true; run(2, 128, false); ledger.fail_commit = false;
    require(ledger.pending == 0 && ledger.valid, "Failed phase commit stranded physical credit");
    run(1, 128, true); run(1, 3, true);
    run(2, 128, true); run(2, 3, true);
    run(0, 128, true); run(0, 3, true);
    // The admitted budget is below simultaneous-wide sum: at least one real
    // inactive owner must have been freed, not merely relabeled or discounted.
    require(ledger.release_calls > before_resizes && Stats().reserved <= static_cast<std::int64_t>(budget),
        "Wide phase did not perform real shared-owner retirement");
    for (int fault : {17, 18, 19}) {
        run(0, 128, true); const auto before = snapshot(0);
        tsg::host_moe_expert_cache_test_fail_next(fault);
        run(0, 64, false, 1);
        const auto after = snapshot(0); stable_weights(before, after);
        require(after.wide_bytes == 0 && after.wide_buffer == 0 && after.wide_charge == 0,
            "Failed reshape retained an uncommitted or duplicate wide buffer");
        run(0, 64, true); stable_weights(before, snapshot(0));
    }
    // Largest complete8192 device routing block and replacement ownership
    // stay correct at1024->512, including each physical/commit failure phase.
    for (int fault : {17, 18, 19}) {
        run(0, 1024, true); const auto before = snapshot(0);
        tsg::host_moe_expert_cache_test_fail_next(fault);
        run(0, 512, false, 1);
        const auto after = snapshot(0); stable_weights(before, after);
        require(after.wide_bytes == 0 && after.wide_buffer == 0 && after.wide_charge == 0
            && ledger.pending == 0 && ledger.valid, "Failed1024-to512 reshape retained duplicate or uncommitted ownership");
        run(0, 512, true); stable_weights(before, snapshot(0));
    }
    for (int rows : {20, 32, 128, 256, 384, 512, 1024, 8, 3, 1}) for (int group = 0; group < 3; ++group) run(group, rows, true);
    // Quiescent cancellation is represented by the real scope Trim operation,
    // not an asynchronous abort during a CUDA graph. Plan metadata survives.
    require(TSGgml_TrimHostMoeExpertCacheScope(token, 0, &released) == 1, "Phase cancellation trim failed");
    ledger.check_empty();
    require(TSGgml_UpdateHostMoeExpertCacheScope(token, budget + 1, &released) == 1
        && TSGgml_UpdateHostMoeExpertCacheScope(token, budget, &released) == 1,
        "Empty phase scope cannot resize/resume above its minimum");
    tsg::g_test_file_failure_countdown = 2;
    run(1, 1024, false, 2);
    require(ledger.valid && ledger.pending == 0, "Interrupted source read stranded a pending owner");
    run(1, 1024, true); run(1, 3, true);
    require(TSGgml_TrimHostMoeExpertCacheScope(token, 0, &released) == 1, "Cannot drain interrupted/resumed phase");
    ledger.check_empty();
    for (int group = 0; group < 3; ++group) run(group, 3, true);
    // Drop invalidates only referencing payload; the complete geometry plan
    // remains until backend/scope teardown. No source is silently re-registered.
    for (auto& bytes : weights[0].quantized) tsg::host_moe_expert_cache_on_drop(bytes.data());
    Snapshot dropped{};
    require(!tsg::host_moe_expert_cache_test_lifecycle_snapshot(weights[0].segment, g_backend, dropped),
        "Source drop retained its referencing payload");
    run(1, 3, true); run(2, 3, true);
    }
    require(TSGgml_EndHostMoeExpertCacheScope(token, &released) == 1, "Three-group scope failed cleanup");
    token = 0; ledger.check_empty(); require(ledger.detach(), "Three-group callbacks retained an owner");
    for (auto& w : weights) for (auto& bytes : w.quantized) tsg::host_moe_expert_cache_on_drop(bytes.data());
    require(std::remove(path.c_str()) == 0, "Three-group file handles survived unregister");
    require(TSGgml_BeginHostMoeExpertCacheScopeWithDecodeRows(budget, maximum_rows, 3, decode_rows, &token) == 1
        && tsg::host_moe_expert_cache_install_phase_plan(minima, maximum_rows, decode_rows, g_backend),
        "Ended scope leaked its phase metadata into a fresh session");
    require(TSGgml_EndHostMoeExpertCacheScope(token, &released) == 1, "Fresh empty group scope did not close");
    token = 0;
    std::printf("PASS: actual phase coexist=%d minimum=%zu budget=%zu all-wide=%zu; %zu complete finite CUDA operator floats byte-exact, PP128/256/384/512/1024/N3/group switch, min-1 nonmutation, reshape faults, read interruption, trim/resume/drop/end. Not whole-model logits or asynchronous cancellation qualification.\n",
        coexist ? 1 : 0, plan.minimum(), budget, all_wide, compared);
    if (direct_only_wide) {
        if (!wide_direct_reads) std::puts("SKIP: wide phase completed buffered fallback but filesystem did not complete direct I/O");
        else std::printf("PASS: wide-only direct phase coexist=%d completed_direct_reads=%llu narrow_buffered_reads=%llu narrow_direct_reads=0; complete phase exact output/ownership/fault proof\n",
            coexist ? 1 : 0, static_cast<unsigned long long>(wide_direct_reads), static_cast<unsigned long long>(narrow_file_reads));
        require(coexist || narrow_file_reads > 0, "Wide-only direct fixture lacks cold N1/N3 buffered miss evidence");
    }
}
void glm_phase_capacity_growth_checks() {
    constexpr int count = 64, decode_rows = 3, prefill_rows = 32;
    tsg::host_moe_expert_cache_release();
    std::array<Weights, 4> weights;
    weights[1].types = {GGML_TYPE_IQ1_M, GGML_TYPE_IQ1_M, GGML_TYPE_IQ4_NL};
    weights[2].types = {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ4_NL};
    std::array<std::unique_ptr<FullResidentGraph>, 4> references;
    std::vector<tsg::HostMoePhaseGroupMinimum> minima;
    std::uint64_t planning = 0;
    require(TSGgml_BeginHostMoeExpertCacheScopeWithDecodeRows(256LL << 20,
        prefill_rows, 3, decode_rows, &planning) == 1, "Cannot begin phase growth metadata scope");
    for (int i = 0; i < 4; ++i) {
        weights[i].fill(.7f + i * .4f, count);
        weights[i].segment.layer = i; weights[i].segment.pooled_model_experts = count;
        if (i < 3) {
            std::size_t narrow = 0, wide = 0;
            require(tsg::host_moe_expert_cache_minimum_phase_plan(weights[i].segment, decode_rows,
                prefill_rows, g_backend, narrow, wide), "Cannot measure phase growth floors");
            minima.push_back({tsg::host_moe_expert_group_key(weights[i].segment), narrow, wide});
        }
        references[i] = std::make_unique<FullResidentGraph>(weights[i]);
    }
    std::int64_t released = 0;
    require(TSGgml_EndHostMoeExpertCacheScope(planning, &released) == 1, "Growth metadata scope retained ownership");
    const tsg::HostMoePhaseQuotaPlan plan(minima);
    const auto pressure = plan.minimum(), ceiling = pressure * 3;
    auto snapshot = [&](int group) {
        tsg::HostMoeCacheTestSnapshot result{};
        require(tsg::host_moe_expert_cache_test_lifecycle_snapshot(weights[group].segment, g_backend, result),
            "Phase growth lost its published table");
        return result;
    };
    const auto growth = [] {
        std::array<std::int64_t, 8> values{};
        require(TSGgml_HostMoeExpertCacheGrowthStats(values.data(), 8) == 1, "Missing phase growth counters");
        return values;
    };
    auto call = [&](int group, int rows) {
        std::vector<float> x(rows * hidden), routes(rows * used), actual(rows * hidden, -999), expected;
        std::vector<std::int32_t> ids(rows * used);
        for (int row = 0; row < rows; ++row) {
            std::array<float, hidden> input{};
            const std::array<float, used> probabilities{.6f, .4f};
            const std::array<std::int32_t, used> selected{0, 1};
            for (int h = 0; h < hidden; ++h) input[h] = std::sin(h * .013f + row * .17f);
            std::copy(input.begin(), input.end(), x.begin() + row * hidden);
            std::copy(probabilities.begin(), probabilities.end(), routes.begin() + row * used);
            std::copy(selected.begin(), selected.end(), ids.begin() + row * used);
            const auto answer = references[group]->compute(input, selected, probabilities);
            expected.insert(expected.end(), answer.begin(), answer.end());
        }
        auto hm = weights[group].segment; hm.seq_len = rows; hm.independent_decode_rows = rows > 1;
        require(tsg::host_moe_cached_experts(hm, x.data(), ids.data(), routes.data(), actual.data(),
            "glm5-next selected expert seam", g_backend) == 1, "Phase growth execution declined");
        require(std::memcmp(actual.data(), expected.data(), expected.size() * sizeof(float)) == 0,
            "Phase growth changed exact scalar CUDA output");
    };
    for (int fault : {0, -1, -2, -3, 11, 12, 15, 8, 9, 10}) {
        tsg::host_moe_expert_cache_release();
        SharedLedger ledger; ledger.capacity = static_cast<std::int64_t>(ceiling); ledger.multiple_cache_entries = true;
        require(ledger.attach(), "Cannot attach phase growth physical ledger");
        std::uint64_t token = 0;
        require(TSGgml_BeginHostMoeExpertCacheScopeWithCapacityGrowth(ceiling, prefill_rows, 3,
            decode_rows, 1, &token) == 1
            && TSGgml_UpdateHostMoeExpertCacheScope(token, pressure, &released) == 1
            && tsg::host_moe_expert_cache_install_phase_plan(minima, prefill_rows, decode_rows, g_backend),
            "Cannot establish pressure-limited phase growth scope");
        struct Cleanup {
            std::uint64_t& token;
            ~Cleanup() {
                tsg::g_test_copy_failure_countdown = 0; tsg::host_moe_expert_cache_test_fail_next(0);
                tsg::host_moe_expert_cache_test_available_bytes((std::numeric_limits<std::size_t>::max)());
                if (token) { std::int64_t bytes = 0; TSGgml_EndHostMoeExpertCacheScope(token, &bytes); }
            }
        } cleanup{token};
        // Two layers share the first pooled group and local IDs0/1, but their
        // immutable source tensors and weights differ. Preserve both identities.
        call(0, 1); call(3, 3); call(0, 3);
        const auto before = snapshot(0); const auto before_stats = Stats();
        require(before.capacity < count, "Pressure phase did not produce a smaller table");
        require(TSGgml_UpdateHostMoeExpertCacheScope(token, ceiling, &released) == 1 && released == 0,
            "Idle budget recovery retired a live table before admission");
        if (fault == -1) ledger.capacity = ledger.committed;
        if (fault == -2) tsg::host_moe_expert_cache_test_available_bytes((512u << 20) + 1);
        if (fault == -3) ledger.fail_commit = true;
        if (fault > 0) tsg::host_moe_expert_cache_test_fail_next(fault);
        tsg::g_test_copy_failure_countdown = 1;
        bool caught = false;
        try { call(0, 1); } catch (const std::runtime_error&) { caught = true; }
        const bool failure = fault == -3 || fault > 0;
        require(caught == failure, "Phase growth failure/refusal contract changed");
        ledger.fail_commit = false; ledger.capacity = static_cast<std::int64_t>(ceiling);
        tsg::host_moe_expert_cache_test_available_bytes((std::numeric_limits<std::size_t>::max)());
        const auto after = snapshot(0);
        require(fault == 0 ? after.capacity > before.capacity : after.entry_identity == before.entry_identity
            && after.weight_buffer == before.weight_buffer && after.weight_charge == before.weight_charge
            && after.narrow_buffer == before.narrow_buffer && after.capacity == before.capacity,
            "Phase growth refusal/failure lost the original physical owner");
        for (int rows : {3, 1, 3}) { call(0, rows); call(3, rows); }
        require(growth()[0] == 1 && Stats().misses == before_stats.misses
            && tsg::g_test_copy_failure_countdown == 1 && ledger.valid && ledger.pending == 0,
            "Phase growth retried per token, reread hot experts, or leaked physical credit");
        if (fault != 0) {
            require(TSGgml_UpdateHostMoeExpertCacheScope(token, pressure, &released) == 1
                && TSGgml_UpdateHostMoeExpertCacheScope(token, ceiling, &released) == 1,
                "Cannot rearm a refused phase generation");
            call(0, 1);
            require(growth()[0] == 2 && growth()[1] == 1 && snapshot(0).capacity > before.capacity,
                "Phase recovery failed to retry exactly once on later budget growth");
        }
        tsg::g_test_copy_failure_countdown = 0;
        if (fault == 0) {
            // All groups warm at N1, then N3 admission reduces the real shared
            // ceiling. LRU retirement makes the next N3 table smaller; N1
            // recovery must grow it again without reading its hot routes.
            call(1, 1); call(2, 1);
            require(TSGgml_UpdateHostMoeExpertCacheScope(token, pressure, &released) == 1 && released > 0,
                "N3 pressure did not retire actual oversized cache ownership");
            ledger.capacity = static_cast<std::int64_t>(pressure);
            call(0, 3); const auto narrow = snapshot(0); const auto misses = Stats().misses;
            require(narrow.capacity < after.capacity && ledger.committed <= static_cast<std::int64_t>(pressure),
                "N3 pressure did not fit the reduced shared envelope");
            ledger.capacity = static_cast<std::int64_t>(ceiling);
            require(TSGgml_UpdateHostMoeExpertCacheScope(token, ceiling, &released) == 1, "N1 recovery budget refused");
            tsg::g_test_copy_failure_countdown = 1; call(0, 1);
            require(snapshot(0).capacity > narrow.capacity && Stats().misses == misses
                && tsg::g_test_copy_failure_countdown == 1, "N1 recovery lost warm routes or retained pressure capacity");
            tsg::g_test_copy_failure_countdown = 0;
        }
        require(ledger.peak_owned <= static_cast<std::int64_t>(ceiling) && ledger.valid && ledger.pending == 0,
            "Phase growth exceeded the old-plus-new physical/shared ceiling");
        require(TSGgml_EndHostMoeExpertCacheScope(token, &released) == 1, "Phase growth teardown failed");
        token = 0; ledger.check_empty(); require(ledger.detach(), "Phase growth leaked a ticket");
    }
    std::printf("PASS: phase-plan capacity recovery pressure=%zu ceiling=%zu; N1/N3 pressure/N1 recovery, exact CUDA output, warm copies, joined failures, one attempt per generation and real peak ledger\n",
        pressure, ceiling);
}
void glm_phase_retained_peer_checks(bool material_deficit) {
    constexpr int count = 64, rows = 32;
    tsg::host_moe_expert_cache_release();
    std::array<Weights, 3> weights;
    weights[0].types = {GGML_TYPE_IQ1_M, GGML_TYPE_IQ1_M, GGML_TYPE_IQ4_NL};
    weights[1].types = {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ4_NL};
    weights[2].types = {GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0};
    std::array<std::unique_ptr<FullResidentGraph>, 3> scalar, wide;
    std::vector<tsg::HostMoePhaseGroupMinimum> minima;
    std::uint64_t planning = 0;
    require(TSGgml_BeginHostMoeExpertCacheScopeWithDecodeRows(128LL << 20, rows, 3, 1, &planning) == 1,
        "Cannot begin retained-peer size-only scope");
    struct PlanningCleanup {
        std::uint64_t& token;
        ~PlanningCleanup() { if (token) { std::int64_t bytes = 0; TSGgml_EndHostMoeExpertCacheScope(token, &bytes); } }
    } planning_cleanup{planning};
    for (int group = 0; group < 3; ++group) {
        weights[group].fill(.7f + group * .3f, count);
        weights[group].segment.layer = group;
        weights[group].segment.pooled_model_experts = count * 16;
        std::size_t narrow_bytes = 0, wide_bytes = 0;
        require(tsg::host_moe_expert_cache_minimum_phase_plan(weights[group].segment, 1, rows,
            g_backend, narrow_bytes, wide_bytes), "Cannot measure retained-peer group floors");
        minima.push_back({tsg::host_moe_expert_group_key(weights[group].segment), narrow_bytes, wide_bytes});
        scalar[group] = std::make_unique<FullResidentGraph>(weights[group]);
        wide[group] = std::make_unique<FullResidentGraph>(weights[group], rows);
    }
    std::int64_t released = 0;
    require(TSGgml_EndHostMoeExpertCacheScope(planning, &released) == 1 && released == 0,
        "Retained-peer metadata planning retained physical ownership");
    planning = 0;
    const auto budget = minima[0].wide_bytes + minima[1].wide_bytes + minima[2].wide_bytes + 17;
    const auto narrow_total = minima[0].narrow_bytes + minima[1].narrow_bytes + minima[2].narrow_bytes;
    const auto spare = budget - narrow_total;
    // Real measured floors, artificial traffic weights: common is slightly
    // above its fair quota; frequent is deficient; Q8 is a >2x hoarder. The
    // existing full-share loop would erase common solely for this small excess.
    const auto excess = (minima[0].wide_bytes - minima[0].narrow_bytes) / 8;
    require(excess > 17 && minima[0].wide_bytes - minima[0].narrow_bytes > excess,
        "Retained-peer fixture lacks a material common excess");
    minima[0].weight = minima[0].wide_bytes - minima[0].narrow_bytes - excess;
    minima[2].weight = 1;
    minima[1].weight = spare - minima[0].weight - minima[2].weight;
    const tsg::HostMoePhaseQuotaPlan plan(minima);
    const auto common_quota = plan.quota(minima[0].key, budget, false);
    require(minima[0].wide_bytes > common_quota && minima[0].wide_bytes < common_quota * 2
        && !tsg::HostMoePhaseQuotaPlan::below_decode_share(minima[0].wide_bytes, common_quota, true)
        && tsg::HostMoePhaseQuotaPlan::below_decode_share(minima[1].wide_bytes,
            plan.quota(minima[1].key, budget, false), true)
        && tsg::HostMoePhaseQuotaPlan::exceeds_twice_share(minima[2].wide_bytes,
            plan.quota(minima[2].key, budget, false)), "Retained-peer measured geometry lacks required skew");
    const auto snapshot = [&](int group) {
        tsg::HostMoeCacheTestSnapshot value{};
        require(tsg::host_moe_expert_cache_test_lifecycle_snapshot(weights[group].segment, g_backend, value),
            "Retained-peer fixture lost a required table");
        return value;
    };
    const auto bytes = [](const tsg::HostMoeCacheTestSnapshot& value) {
        return value.weight_bytes + value.narrow_bytes + value.wide_bytes;
    };
    const auto same_common = [&](const auto& before, const auto& after, bool allow_touch) {
        require(after.entry_identity == before.entry_identity && after.capacity == before.capacity
            && after.weight_buffer == before.weight_buffer && after.weight_charge == before.weight_charge
            && after.narrow_buffer == before.narrow_buffer && after.narrow_charge == before.narrow_charge
            && after.wide_buffer == before.wide_buffer && after.wide_charge == before.wide_charge
            && after.narrow_graph_uid == before.narrow_graph_uid && after.wide_graph_uid == before.wide_graph_uid
            && after.wide_generation == before.wide_generation && after.resident == before.resident,
            "Partial rebalance rebuilt a retained peer owner or graph generation");
        require(bytes(after) == bytes(before) && after.committed_owners == before.committed_owners
            && (allow_touch ? after.clock > before.clock : after.clock == before.clock),
            "Partial rebalance changed retained accounting or LRU clock");
        for (int slot = 0; slot < count; ++slot) {
            require(after.slots[slot].expert == before.slots[slot].expert,
                "Partial rebalance changed a retained source-qualified slot");
            for (int part = 0; part < 3; ++part)
                require(after.slots[slot].source[part] == before.slots[slot].source[part],
                    "Partial rebalance changed a retained immutable source");
            if (!allow_touch || before.slots[slot].expert > 1)
                require(after.slots[slot].last_used == before.slots[slot].last_used,
                    "Partial rebalance reset retained LRU history");
        }
    };
    const auto call = [&](int group, int width) {
        std::vector<float> input(width * hidden), routes(width * used), actual(width * hidden, -999.0f);
        std::vector<std::int32_t> ids(width * used);
        for (int row = 0; row < width; ++row) {
            for (int h = 0; h < hidden; ++h) input[row * hidden + h] = std::sin(h * .013f + row * .07f);
            ids[row * used] = row * used; ids[row * used + 1] = row * used + 1;
            routes[row * used] = .6f; routes[row * used + 1] = .4f;
        }
        const auto expected = width == 1 ? scalar[group]->compute_batch(input, ids, routes)
            : wide[group]->compute_batch(input, ids, routes);
        auto hm = weights[group].segment; hm.seq_len = width;
        const auto submissions = ExecutionStats().submissions;
        try {
            const auto result = tsg::host_moe_cached_experts(hm, input.data(), ids.data(), routes.data(), actual.data(),
                width > 8 ? "glm5-next bulk expert prefill" : "glm5-next selected expert seam", g_backend);
            if (result != 1) throw std::runtime_error("Retained-peer selected seam declined");
        } catch (...) {
            require(ExecutionStats().submissions == submissions
                && std::all_of(actual.begin(), actual.end(), [](float value) { return value == -999.0f; }),
                "Retained-peer allocation/guard failure executed or published output");
            throw;
        }
        require(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0,
            "Retained-peer policy changed complete CUDA operator output");
    };
    for (int fault : {0, 11, 12, 15}) {
        tsg::host_moe_expert_cache_release();
        SharedLedger ledger; ledger.capacity = static_cast<std::int64_t>(budget);
        ledger.multiple_cache_entries = true;
        require(ledger.attach(), "Cannot attach retained-peer physical ledger");
        std::uint64_t token = 0;
        struct Cleanup {
            std::uint64_t& token;
            ~Cleanup() {
                tsg::g_test_copy_failure_countdown = 0; tsg::host_moe_expert_cache_test_fail_next(0);
                if (token) { std::int64_t bytes = 0; TSGgml_EndHostMoeExpertCacheScope(token, &bytes); }
            }
        } cleanup{token};
        require(TSGgml_BeginHostMoeExpertCacheScopeWithDecodeRows(budget, rows, 3, 1, &token) == 1
            && tsg::host_moe_expert_cache_install_phase_plan(minima, rows, 1, g_backend),
            "Cannot install retained-peer exact three-group profile");
        if (!material_deficit) {
            const auto reserves = ledger.reserve_calls;
            bool refused = false;
            try { call(0, rows); } catch (const std::invalid_argument&) { refused = true; }
            require(refused && ledger.reserve_calls == reserves && Stats().reserved == 0,
                "Retained-peer flag without75% gate mutated cache ownership");
            require(TSGgml_EndHostMoeExpertCacheScope(token, &released) == 1,
                "Retained-peer guard teardown refused");
            token = 0; ledger.check_empty(); require(ledger.detach(), "Retained-peer guard leaked tickets");
            std::puts("PASS: retained-peer strict flag guard refused before cache allocation");
            return;
        }
        for (int group = 0; group < 3; ++group) call(group, rows);
        const auto before = snapshot(0);
        require(before.capacity == count && before.resident == count && bytes(before) == minima[0].wide_bytes,
            "Retained-peer warmup did not fill the expected exact64-slot peer");
        const auto misses = Stats().misses;
        tsg::g_test_copy_failure_countdown = 1;
        call(0, 1); // Actual wide->narrow examination, then a hot peer hit.
        require(Stats().misses == misses && tsg::g_test_copy_failure_countdown == 1,
            "Retained peer was cold-reloaded after phase rebalance");
        tsg::g_test_copy_failure_countdown = 0;
        const auto retained = snapshot(0); same_common(before, retained, true);
        require(Stats().reserved == static_cast<std::int64_t>(bytes(retained))
            && ledger.committed == Stats().reserved && ledger.pending == 0,
            "Partial rebalance did not retire only the skewed owners");
        if (fault) tsg::host_moe_expert_cache_test_fail_next(fault);
        bool failed = false;
        try { call(1, 1); } catch (const std::runtime_error&) { failed = true; }
        require(failed == (fault != 0), "Retained-peer cold allocation failure contract changed");
        tsg::host_moe_expert_cache_test_fail_next(0);
        same_common(retained, snapshot(0), false);
        if (fault) {
            require(Stats().reserved == static_cast<std::int64_t>(bytes(retained))
                && ledger.committed == Stats().reserved && ledger.pending == 0,
                "Retained-peer failed candidate leaked credit or lost the peer");
            call(1, 1);
        }
        const auto frequent = snapshot(1);
        require(bytes(retained) + bytes(frequent) + minima[2].narrow_bytes <= budget,
            "First cold optional capacity consumed the final absent group's floor");
        call(2, 1);
        same_common(retained, snapshot(0), false);
        const auto rare = snapshot(2);
        require(bytes(rare) >= minima[2].narrow_bytes && rare.wide_rows == 0
            && ledger.valid && ledger.pending == 0 && ledger.peak_owned <= static_cast<std::int64_t>(budget)
            && Stats().reserved == ledger.committed,
            "Final cold group could not acquire its real shared mandatory floor");
        call(1, 1); call(2, 1); same_common(retained, snapshot(0), false);
        require(TSGgml_EndHostMoeExpertCacheScope(token, &released) == 1, "Retained-peer teardown refused");
        token = 0; ledger.check_empty(); require(ledger.detach(), "Retained-peer leaked a real ownership ticket");
    }
    std::puts("PASS: partial75% phase rebalance; three real groups, retained hot source/LRU/owners/UIDs, absent-floor protection, complete exact CUDA outputs, allocation/commit failures, physical ledger and zero teardown");
}

void glm_phase_cold_growth_checks() {
    constexpr int count = 64, prefill_rows = 32, decode_rows = 1;
    tsg::host_moe_expert_cache_release();
    std::array<Weights, 3> weights;
    weights[1].types = {GGML_TYPE_IQ1_M, GGML_TYPE_IQ1_M, GGML_TYPE_IQ4_NL};
    weights[2].types = {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ4_NL};
    std::array<std::unique_ptr<FullResidentGraph>, 3> scalar;
    std::unique_ptr<FullResidentGraph> wide;
    std::vector<tsg::HostMoePhaseGroupMinimum> minima;
    std::uint64_t planning = 0;
    require(TSGgml_BeginHostMoeExpertCacheScopeWithDecodeRows(512LL << 20,
        prefill_rows, 3, decode_rows, &planning) == 1, "Cannot begin cold growth metadata scope");
    for (int group = 0; group < 3; ++group) {
        weights[group].fill(.6f + group * .3f, count);
        weights[group].segment.layer = group;
        weights[group].segment.pooled_model_experts = count * 512;
        std::size_t narrow = 0, maximum_wide = 0;
        require(tsg::host_moe_expert_cache_minimum_phase_plan(weights[group].segment,
            decode_rows, prefill_rows, g_backend, narrow, maximum_wide), "Cannot measure cold growth floors");
        minima.push_back({tsg::host_moe_expert_group_key(weights[group].segment),
            narrow, maximum_wide, group == 1 ? 9u : 1u});
        scalar[group] = std::make_unique<FullResidentGraph>(weights[group]);
    }
    wide = std::make_unique<FullResidentGraph>(weights[0], prefill_rows);
    std::int64_t released = 0;
    require(TSGgml_EndHostMoeExpertCacheScope(planning, &released) == 1 && released == 0,
        "Cold growth metadata planning retained ownership");
    const tsg::HostMoePhaseQuotaPlan plan(minima);
    const auto pressure = plan.minimum() + (16u << 20), ceiling = std::size_t(256) << 20;
    require(pressure < ceiling / 2, "Cold growth fixture does not leave material net slack");
    const auto snapshot = [&](int group, tsg::HostMoeCacheTestSnapshot& result) {
        return tsg::host_moe_expert_cache_test_lifecycle_snapshot(weights[group].segment, g_backend, result);
    };
    const auto bytes = [](const tsg::HostMoeCacheTestSnapshot& value) {
        return value.weight_bytes + value.narrow_bytes + value.wide_bytes;
    };
    const auto call = [&](int group, int rows) {
        std::vector<float> input(rows * hidden), routes(rows * used), actual(rows * hidden, -999.0f);
        std::vector<std::int32_t> ids(rows * used);
        for (int row = 0; row < rows; ++row) {
            for (int h = 0; h < hidden; ++h) input[row * hidden + h] = std::sin(h * .013f + row * .07f);
            ids[row * used] = 0; ids[row * used + 1] = 1;
            routes[row * used] = .6f; routes[row * used + 1] = .4f;
        }
        const auto expected = rows == 1 ? scalar[group]->compute_batch(input, ids, routes)
            : wide->compute_batch(input, ids, routes);
        auto hm = weights[group].segment; hm.seq_len = rows;
        const auto submissions_before = ExecutionStats().submissions;
        try {
            require(tsg::host_moe_cached_experts(hm, input.data(), ids.data(), routes.data(), actual.data(),
                rows > 8 ? "glm5-next bulk expert prefill" : "glm5-next selected expert seam", g_backend) == 1,
                "Cold growth selected seam declined");
        } catch (...) {
            require(ExecutionStats().submissions == submissions_before
                && std::all_of(actual.begin(), actual.end(), [](float value) { return value == -999.0f; }),
                "Cold growth failure submitted experts or changed output sentinels");
            throw;
        }
        require(std::all_of(actual.begin(), actual.end(), [](float value) { return std::isfinite(value); })
            && std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0,
            "Cold growth changed complete finite CUDA operator output");
    };
    for (int fault : {0, -1, -2, -3, 11, 12, 15, 20, 21, 22}) {
        tsg::host_moe_expert_cache_release();
        SharedLedger ledger; ledger.capacity = static_cast<std::int64_t>(ceiling); ledger.multiple_cache_entries = true;
        require(ledger.attach(), "Cannot attach cold growth ledger");
        std::uint64_t token = 0;
        require(TSGgml_BeginHostMoeExpertCacheScopeWithCapacityGrowth(ceiling, prefill_rows,
            3, decode_rows, 1, &token) == 1
            && TSGgml_UpdateHostMoeExpertCacheScope(token, pressure, &released) == 1
            && tsg::host_moe_expert_cache_install_phase_plan(minima, prefill_rows, decode_rows, g_backend),
            "Cannot establish cold growth phase scope");
        struct Cleanup {
            std::uint64_t& token;
            ~Cleanup() {
                tsg::host_moe_expert_cache_test_fail_next(0);
                tsg::host_moe_expert_cache_test_available_bytes((std::numeric_limits<std::size_t>::max)());
                if (token) { std::int64_t bytes = 0; TSGgml_EndHostMoeExpertCacheScope(token, &bytes); }
            }
        } cleanup{token};
        for (int group = 0; group < 3; ++group) call(group, 1);
        require(TSGgml_UpdateHostMoeExpertCacheScope(token, ceiling, &released) == 1 && released == 0,
            "Budget growth prematurely retired cold fixture tables");
        call(0, prefill_rows); // Real wide->narrow marker with all narrow peer floors already live.
        std::array<tsg::HostMoeCacheTestSnapshot, 3> before{};
        for (int group = 0; group < 3; ++group)
            require(snapshot(group, before[group]), "Cold growth fixture lost a mandatory peer");
        const auto reserve_calls_before = ledger.reserve_calls;
        if (fault == -1 || fault == 22) ledger.capacity = ledger.committed;
        if (fault == -2) tsg::host_moe_expert_cache_test_available_bytes((512u << 20) + bytes(before[1]) + 1);
        if (fault == -3) ledger.fail_commit = true;
        if (fault > 0) tsg::host_moe_expert_cache_test_fail_next(fault);
        bool failed = false;
        try { call(1, 1); } catch (const std::exception&) { failed = true; }
        const bool expected_failure = fault == -3 || fault > 0;
        require(failed == expected_failure, "Cold growth physical failure was retried or optional refusal failed");
        if (fault == 21 || fault == 22)
            require(ledger.reserve_calls == reserve_calls_before + (fault == 22 ? 1 : 0),
                "Cold growth retired/refused error epoch still attempted fallback/allocation");
        ledger.fail_commit = false; ledger.capacity = static_cast<std::int64_t>(ceiling);
        tsg::host_moe_expert_cache_test_available_bytes((std::numeric_limits<std::size_t>::max)());
        tsg::host_moe_expert_cache_test_fail_next(0);
        for (int group : {0, 2}) {
            tsg::HostMoeCacheTestSnapshot after{};
            require(snapshot(group, after) && after.entry_identity == before[group].entry_identity
                && after.weight_buffer == before[group].weight_buffer && after.weight_charge == before[group].weight_charge,
                "Optional cold growth evicted or rebuilt a peer");
        }
        tsg::HostMoeCacheTestSnapshot after{};
        if (expected_failure) require(!snapshot(1, after), "Failed cold replacement retained a partial published table");
        else require(snapshot(1, after)
            && (fault == 0 ? after.capacity > before[1].capacity && bytes(after) - bytes(before[1]) >= (64u << 20)
                : after.capacity == before[1].capacity && bytes(after) == bytes(before[1])),
            "Cold growth did not realize its net capacity or preplanned fallback");
        require(ledger.valid && ledger.pending == 0
            && ledger.committed == static_cast<std::int64_t>(bytes(before[0]) + bytes(before[2])
                + (expected_failure ? 0 : bytes(after))),
            "Cold growth failure/refusal left credit outside its surviving peer owners");
        require(tsg::host_moe_expert_cache_test_phase_cold_growth_attempts() == 1,
            "Cold phase did not nominate exactly one material replacement");
        for (int group = 0; group < 3; ++group) call(group, 1);
        require(tsg::host_moe_expert_cache_test_phase_cold_growth_attempts() == 1
            && ledger.valid && ledger.pending == 0 && ledger.peak_owned <= static_cast<std::int64_t>(ceiling),
            "Cold growth retried per token or exceeded its real physical/shared ceiling");
        require(TSGgml_EndHostMoeExpertCacheScope(token, &released) == 1, "Cold growth teardown refused");
        token = 0; ledger.check_empty(); require(ledger.detach(), "Cold growth leaked an ownership ticket");
    }
    std::puts("PASS: one-group phase cold growth; exact finite CUDA output, material net capacity, peer preservation, one attempt, refusal fallback, physical/commit failure stop, trim/end ownership");
}
#endif
