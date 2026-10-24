// CUDA coverage for the opt-in scalar selected-file staging arena. File
// mappings are poisoned; independent resident graphs retain the original bytes.
#ifdef TSG_GGML_USE_CUDA
extern "C" int TSGgml_HostMoeDecodeStagingVersion();
extern "C" int TSGgml_HostMoeDecodeStagingEnabled();

void host_moe_scalar_staging_checks(const std::array<ggml_type, 3>& types) {
    constexpr std::size_t ordinary = std::size_t(32) << 20, expanded = std::size_t(72) << 20;
    constexpr int h = 4096, count = 16, selected = 8;
    require(TSGgml_HostMoeDecodeStagingVersion() == 1 && TSGgml_HostMoeDecodeStagingEnabled() == 1,
        "Scalar staging fixture did not enable the actual native capability");
    tsg::host_moe_expert_cache_release();
    Weights weights; weights.types = types;
    weights.fill(.87f, count, selected, h, types[0] == GGML_TYPE_Q8_0 ? 512 : 1280);
    weights.segment.pooled_model_experts = count;
    const auto payload = (weights.segment.gate_bytes + weights.segment.up_bytes + weights.segment.down_bytes) / count * selected;
    require(payload > ordinary && payload <= expanded, "Scalar staging fixture does not cross the real32MiB boundary");
    FullResidentGraph oracle(weights);
    const auto path = fixture_file_path("host-moe-scalar-staging-");
    SharedLedger ledger; ledger.capacity = 128LL << 20; ledger.host_capacity = ordinary;
    std::uint64_t token = 0;
    struct Cleanup {
        const std::string& path; std::uint64_t& token;
        ~Cleanup() {
            tsg::g_test_file_failure_countdown = 0; tsg::g_test_copy_failure_countdown = 0;
            tsg::host_moe_expert_cache_test_file_workspace_fail(0);
            if (token) { std::int64_t released = 0; TSGgml_EndHostMoeExpertCacheScope(token, &released); }
            tsg::host_moe_expert_cache_release(); std::remove(path.c_str());
        }
    } cleanup{path, token};
    std::array<std::int64_t, 3> offsets{};
    {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        require(bool(file), "Cannot create scalar staging source");
        for (int part = 0; part < 3; ++part) {
            file.write("unaligned-source!", 17); offsets[part] = static_cast<std::int64_t>(file.tellp());
            file.write(reinterpret_cast<const char*>(weights.quantized[part].data()), weights.quantized[part].size());
        }
        require(bool(file), "Cannot write scalar staging source");
    }
    for (int part = 0; part < 3; ++part) {
        require(TSGgml_RegisterHostFileSource(weights.quantized[part].data(), weights.quantized[part].size(),
            path.c_str(), offsets[part]) == 1, "Cannot register scalar staging source");
        std::fill(weights.quantized[part].begin(), weights.quantized[part].end(), 0);
    }
    require(ledger.attach(true), "Cannot attach scalar staging accounting");
    require(TSGgml_BeginHostMoeExpertCacheScopeWithDecodeRows(ledger.capacity, 8, 1, 3, &token) == 1,
        "Cannot begin scalar staging scope");
    std::vector<float> input(h), output(h), routes(selected);
    std::vector<std::int32_t> low{7, 3, 0, 6, 2, 5, 1, 4}, high = low;
    for (auto& id : high) id += selected;
    for (int i = 0; i < h; ++i) input[i] = std::cos(.021f * i);
    for (int i = 0; i < selected; ++i) routes[i] = float(i + 1) / 36;
    const auto expected_low = oracle.compute_batch(input, low, routes);
    const auto expected_high = oracle.compute_batch(input, high, routes);
    for (const auto* expected : {&expected_low, &expected_high})
        require(std::all_of(expected->begin(), expected->end(), [](float value) { return std::isfinite(value); })
            && std::any_of(expected->begin(), expected->end(), [](float value) { return value != 0; }),
            "Scalar staging resident oracle is nonfinite or degenerate");
    auto call = [&](const std::vector<std::int32_t>& ids, const std::vector<float>& expected) {
        std::fill(output.begin(), output.end(), -999);
        require(tsg::host_moe_cached_experts(weights.segment, input.data(), ids.data(), routes.data(), output.data(),
            "glm5-next selected expert seam", g_backend) == 1, "Scalar staging did not execute");
        require(std::memcmp(output.data(), expected.data(), expected.size() * sizeof(float)) == 0,
            "Scalar staging changed complete resident-oracle output");
    };
    auto host_tickets = [&] {
        std::vector<std::int64_t> result;
        for (const auto& item : ledger.allocations) if (item.second.host) {
            require(item.second.committed, "Staging retained an uncommitted host ticket");
            result.push_back(item.second.bytes);
        }
        std::sort(result.begin(), result.end()); return result;
    };
    auto trim = [&] {
        require(TSGgml_TrimHostMoeExpertCache(0) > 0, "Staging fixture did not retire its cache");
        ledger.check_empty();
        require(tsg::host_moe_expert_cache_test_file_workspace_bytes() == 0, "Trim retained staging physical storage");
    };

    auto windows = tsg::host_moe_expert_cache_test_file_windows();
    call(low, expected_low);
    require(tsg::host_moe_expert_cache_test_file_windows() - windows == (payload + ordinary - 1) / ordinary
        && host_tickets() == std::vector<std::int64_t>{ordinary},
        "Extra host refusal did not use exact ordinary32MiB windows");
    const auto* original = tsg::host_moe_expert_cache_test_file_workspace_data();
    std::array<std::uint8_t, 64> prefix{}; std::memcpy(prefix.data(), original, prefix.size());
    const auto reads = tsg::g_test_file_reads;
    require(!tsg::host_moe_expert_cache_test_try_scalar_workspace()
        && original == tsg::host_moe_expert_cache_test_file_workspace_data()
        && std::memcmp(prefix.data(), original, prefix.size()) == 0
        && tsg::g_test_file_reads == reads && ledger.host_committed == ordinary && ledger.host_pending == 0,
        "Prephysical expansion refusal replaced or modified the original arena");

    ledger.host_capacity = expanded;
    windows = tsg::host_moe_expert_cache_test_file_windows();
    const auto before_expanded_reads = tsg::g_test_file_reads;
    call(high, expected_high);
    require(tsg::host_moe_expert_cache_test_file_windows() == windows + 1
        && tsg::g_test_file_reads == before_expanded_reads + selected * 3
        && tsg::host_moe_expert_cache_test_file_workspace_bytes() == expanded
        && host_tickets() == std::vector<std::int64_t>{ordinary, expanded - ordinary}
        && ledger.host_committed == expanded && ledger.host_pending == 0 && ledger.valid,
        "Expanded scalar payload was split, overcharged or not covered by32+40MiB tickets");
    const auto warm_reads = tsg::g_test_file_reads, warm_reserves = ledger.reserve_calls;
    call(high, expected_high);
    require(tsg::g_test_file_reads == warm_reads && ledger.reserve_calls == warm_reserves,
        "Warm scalar cache regrew staging or reread its source");
    trim();

    // Multirow prefill remains on the original bounded windows even though the
    // experimental session has enough host credit for72MiB.
    auto prefill = weights.segment; prefill.seq_len = 2;
    std::vector<float> two_input = input, two_routes = routes, two_output(2 * h, -999);
    std::vector<std::int32_t> two_ids = low;
    two_input.insert(two_input.end(), input.begin(), input.end());
    two_routes.insert(two_routes.end(), routes.begin(), routes.end());
    two_ids.insert(two_ids.end(), low.begin(), low.end());
    windows = tsg::host_moe_expert_cache_test_file_windows();
    require(tsg::host_moe_cached_experts(prefill, two_input.data(), two_ids.data(), two_routes.data(), two_output.data(),
        "glm5-next selected expert seam", g_backend) == 1, "Multirow prefill staging did not execute");
    for (int row = 0; row < 2; ++row)
        require(std::memcmp(two_output.data() + row * h, expected_low.data(), h * sizeof(float)) == 0,
            "Multirow prefill changed scalar arithmetic under staging opt-in");
    require(tsg::host_moe_expert_cache_test_file_workspace_bytes() == ordinary
        && host_tickets() == std::vector<std::int64_t>{ordinary}
        && tsg::host_moe_expert_cache_test_file_windows() - windows == (payload + ordinary - 1) / ordinary,
        "Multirow prefill used expanded staging or changed its original windows");

    // An actual attempted allocation fails closed after the old physical
    // buffer is retired. Both its retained ticket and the expansion ticket must
    // be released only after physical storage is gone; SharedLedger checks this.
    tsg::host_moe_expert_cache_test_file_workspace_fail(1);
    bool failed = false;
    try { call(high, expected_high); } catch (const std::runtime_error&) { failed = true; }
    require(failed && tsg::host_moe_expert_cache_test_file_workspace_bytes() == 0
        && ledger.host_committed == 0 && ledger.host_pending == 0 && ledger.valid
        && std::all_of(output.begin(), output.end(), [](float value) { return value == -999; }),
        "Physical staging allocation failure fell back or escaped ownership/output fencing");
    call(high, expected_high);
    require(host_tickets() == std::vector<std::int64_t>{expanded}, "Cold72MiB recovery did not own its complete allocation");
    trim();

    ledger.host_capacity = ordinary; call(low, expected_low);
    ledger.host_capacity = expanded; ledger.fail_commit = true; failed = false;
    try { call(high, expected_high); } catch (const std::runtime_error&) { failed = true; }
    ledger.fail_commit = false;
    require(failed && tsg::host_moe_expert_cache_test_file_workspace_bytes() == 0
        && ledger.host_committed == 0 && ledger.host_pending == 0 && ledger.valid
        && std::all_of(output.begin(), output.end(), [](float value) { return value == -999; }),
        "Expanded staging commit failure hid a physical owner or produced output");
    call(high, expected_high); trim();

    tsg::g_test_file_failure_countdown = 2; failed = false;
    try { call(low, expected_low); } catch (const std::runtime_error&) { failed = true; }
    require(failed && tsg::g_test_file_failure_countdown == 0 && ledger.valid
        && std::all_of(output.begin(), output.end(), [](float value) { return value == -999; }),
        "Partial expanded source upload was not failed before output");
    call(low, expected_low);
    std::int64_t released = 0;
    require(TSGgml_EndHostMoeExpertCacheScope(token, &released) == 1, "Scalar staging scope did not end");
    token = 0; ledger.check_empty(); require(ledger.detach(), "Scalar staging retained native tickets");
    std::printf("PASS: scalar72MiB exact payload=%lld, one window,32+40MiB tickets, refusal retains original bytes, multirow unchanged, allocation/commit/read failures and recovery\n",
        static_cast<long long>(payload));
}
#endif
