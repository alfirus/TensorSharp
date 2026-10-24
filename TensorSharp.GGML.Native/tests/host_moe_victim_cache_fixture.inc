// Selective GPU victim RAM tier, using immutable file bytes and independent
// resident CUDA references. Run in a fresh process with VICTIM=1: the allocator
// selection is deliberately fixed before any cache ownership is acquired.
#ifdef TSG_GGML_USE_CUDA
extern "C" int TSGgml_HostMoeVictimCacheStats(std::int64_t*, int);
struct VictimFixtureStats {
    std::array<std::int64_t, 16> v{};
    VictimFixtureStats() {
        require(TSGgml_HostMoeVictimCacheStats(v.data(), static_cast<int>(v.size())) == 1,
            "Cannot read selective victim counters");
        require(v[1] + v[2] <= v[0] && v[11] <= 16384 && v[12] <= 16384
            && v[8] <= v[7] && v[15] <= v[2], "Victim cache physical/history accounting escaped its bound");
    }
};
void host_moe_victim_cache_checks(const std::array<ggml_type, 3>& types) {
    const auto enabled = [](const char* name) {
        const auto* value = std::getenv(name);
        return value && value[0] == '1' && value[1] == '\0';
    };
    require(enabled("TS_HOST_MOE_RAM_CACHE_VICTIM"), "Victim fixture needs an explicit process flag");
    const bool pageable = enabled("GGML_CUDA_NO_PINNED");
    constexpr std::int64_t ram_capacity = 16LL << 20;
    tsg::host_moe_expert_cache_release();
    require(TSGgml_HostMoeRamCacheConfigure(ram_capacity) == 0,
        "Victim policy allocated without a shared host ledger");
    Weights first, second, replacement;
    first.types = second.types = replacement.types = types;
    first.fill(.37f); second.fill(1.29f); replacement.fill(2.12f);
    const auto original_bytes = first.quantized;
    first.segment.pooled_model_experts = second.segment.pooled_model_experts = 2 * experts;
    second.segment.layer = 1;
    FullResidentGraph first_oracle(first), second_oracle(second), replacement_oracle(replacement), wide_oracle(first, 16);
    std::array<std::string, 3> paths{fixture_file_path("victim-first-"),
        fixture_file_path("victim-second-"), fixture_file_path("victim-replacement-")};
    std::array<std::array<std::int64_t, 3>, 3> offsets{};
    RamFixtureLedger ledger; ledger.host_capacity = 128LL << 20;
    std::uint64_t token = 0;
    struct Cleanup {
        std::array<std::string, 3>& paths; std::uint64_t& token; RamFixtureLedger& ledger;
        ~Cleanup() {
            tsg::g_test_file_failure_countdown = 0; ledger.watch_ram_release = false;
            if (token) { std::int64_t bytes = 0; TSGgml_EndHostMoeExpertCacheScope(token, &bytes); }
            tsg::host_moe_expert_cache_release();
            for (const auto& path : paths) std::remove(path.c_str());
        }
    } cleanup{paths, token, ledger};
    for (int version = 0; version < 3; ++version) {
        const auto& weights = version == 0 ? first : version == 1 ? second : replacement;
        std::ofstream file(paths[version], std::ios::binary | std::ios::trunc);
        require(bool(file), "Cannot create immutable victim fixture");
        for (int part = 0; part < 3; ++part) {
            file.write("source-offset!", 14);
            offsets[version][part] = static_cast<std::int64_t>(file.tellp());
            file.write(reinterpret_cast<const char*>(weights.quantized[part].data()), weights.quantized[part].size());
        }
        require(bool(file), "Cannot write immutable victim fixture");
    }
    auto register_source = [&](Weights& mapped, int version) {
        for (int part = 0; part < 3; ++part) {
            auto& bytes = mapped.quantized[part];
            require(TSGgml_RegisterHostFileSource(bytes.data(), bytes.size(), paths[version].c_str(), offsets[version][part]) == 1,
                "Cannot register victim source generation");
            std::fill(bytes.begin(), bytes.end(), 0);
        }
    };
    register_source(first, 0); register_source(second, 1);
    tsg::HostMoeCacheTestPlan plan{};
    require(tsg::host_moe_expert_cache_test_split_plan(first.segment, used, 1, 16, g_backend, plan),
        "Cannot measure two-slot victim table");
    const auto device_capacity = plan.weight_bytes + plan.narrow_bytes + plan.wide_bytes;
    ledger.device_capacity = static_cast<std::int64_t>(device_capacity);
    require(TSGgml_BeginHostMoeExpertCacheScopeWithDecodeRows(ledger.device_capacity, 16, 1, 1, &token) == 1,
        "Cannot begin victim CUDA scope");
    ledger.attached = tsg::SharedCacheCharge::attach(&ledger, RamFixtureLedger::reserve,
        RamFixtureLedger::commit, RamFixtureLedger::release, false, true);
    require(ledger.attached && TSGgml_HostMoeRamCacheConfigure(ram_capacity) == 1,
        "Cannot configure charged victim history and payloads");
    require(VictimFixtureStats().v[1] > 0 && ledger.held(6) == VictimFixtureStats().v[1],
        "Fixed victim metadata is not physically charged");
    std::array<float, hidden> input{}, actual{};
    for (int i = 0; i < hidden; ++i) input[i] = std::sin(.032f * i);
    std::array<std::int32_t, used> ids{0, 1};
    const std::array<float, used> routes{.65f, .35f};
    const auto first_expected = first_oracle.compute(input, ids, routes);
    const auto second_expected = second_oracle.compute(input, ids, routes);
    const auto replacement_expected = replacement_oracle.compute(input, ids, routes);
    require(std::memcmp(first_expected.data(), second_expected.data(), sizeof(actual)) != 0
        && std::memcmp(first_expected.data(), replacement_expected.data(), sizeof(actual)) != 0,
        "Victim source identity fixtures do not distinguish bytes");
    auto call = [&](const Weights& source) {
        return tsg::host_moe_cached_experts(source.segment, input.data(), ids.data(), routes.data(), actual.data(),
            "glm5-next selected expert seam", g_backend);
    };
    auto run = [&](const Weights& source, const std::array<float, hidden>& expected) {
        std::fill(actual.begin(), actual.end(), -999.f);
        require(call(source) == 1, "Victim scalar CUDA call declined");
        require(std::memcmp(actual.data(), expected.data(), sizeof(actual)) == 0,
            "Victim scalar output differs from independent resident CUDA graph");
    };
    auto reset = [&] {
        TSGgml_TrimHostMoeExpertCache(0);
        require(TSGgml_HostMoeRamCacheConfigure(0) == 1 && ledger.held(6) == 0
            && TSGgml_HostMoeRamCacheConfigure(ram_capacity) == 1,
            "Victim reset retained payload/history ownership");
    };
    auto train = [&] { for (int hit = 0; hit < 5; ++hit) run(first, first_expected); };
    train();
    std::array<std::int64_t, 8> entry{}; int count = 0;
    require(TSGgml_HostMoeExpertCacheEntryStats(entry.data(), 1, &count) == 1
        && count == 1 && entry[1] == used, "Victim fixture did not force exactly two pooled slots");
    require(VictimFixtureStats().v[7] == 0 && RamFixtureStats().values[1] == 0,
        "Read-through payloads duplicated live GPU slots");
    const auto before_demote = VictimFixtureStats();
    run(second, second_expected);
    const auto demoted = VictimFixtureStats();
    if (pageable) {
        require(demoted.v[7] == 0 && demoted.v[6] > before_demote.v[6] && demoted.v[2] == 0,
            "Victim admission silently accepted pageable D2H destinations");
        run(first, first_expected);
    } else {
        require(demoted.v[5] == 2 && demoted.v[7] > 0 && demoted.v[7] == demoted.v[8]
            && demoted.v[2] == demoted.v[15], "Eligible old-layer GPU slots were not fenced into pinned RAM");
        const auto reads = tsg::g_test_file_reads;
        const auto initial_storage = tsg::host_moe_expert_cache_test_payload_stats();
        run(first, first_expected);
        const auto promoted = VictimFixtureStats();
        const auto promoted_storage = tsg::host_moe_expert_cache_test_payload_stats();
        require(tsg::g_test_file_reads == reads && promoted.v[9] == demoted.v[7]
            && promoted_storage.resident_bytes == 0 && promoted.v[2] == promoted_storage.external_free_bytes
            && promoted_storage.external_free_entries == 6 && ledger.held(6) == promoted.v[1] + promoted.v[2]
            && promoted_storage.external_frees == initial_storage.external_frees,
            "Exclusive promotion reread source, retained keyed duplicates or lost charged reusable allocations");
        tsg::HostMoeCacheTestSnapshot snapshot{};
        require(tsg::host_moe_expert_cache_test_lifecycle_snapshot(first.segment, g_backend, snapshot),
            "Cannot inspect promoted raw slot provenance");
        for (int part = 0; part < 3; ++part) {
            const auto stride = original_bytes[part].size() / experts;
            std::vector<std::uint8_t> payload(snapshot.weight_part_bytes[part]);
            require(tsg::host_moe_expert_cache_test_copy_weight_payload(first.segment, g_backend, part,
                payload.data(), payload.size()), "Cannot read back promoted raw quantized payload");
            for (int slot = 0; slot < snapshot.capacity; ++slot)
                require(snapshot.slots[slot].expert >= 0
                    && snapshot.slots[slot].source[part] == reinterpret_cast<std::uintptr_t>(first.quantized[part].data())
                    && std::memcmp(payload.data() + slot * stride,
                        original_bytes[part].data() + snapshot.slots[slot].expert * stride, stride) == 0,
                    "D2H/H2D promotion changed original quantized bytes or old-layer identity");
        }
        // A wide request may consume prior victims but never trains/demotes.
        train(); run(second, second_expected);
        const auto reused_storage = tsg::host_moe_expert_cache_test_payload_stats();
        require(reused_storage.external_pool_reuses >= initial_storage.external_pool_reuses + 6
            && reused_storage.external_allocations == initial_storage.external_allocations
            && reused_storage.external_frees == initial_storage.external_frees,
            "Warm victim demotion allocated/freed pinned storage instead of reusing completed extents");
        const auto before_wide = VictimFixtureStats();
        TSGgml_TrimHostMoeExpertCache(0);
        require(VictimFixtureStats().v[2] == before_wide.v[2] && ledger.held(4) == 0,
            "GPU trim discarded retained victim payloads");
        std::vector<float> wide_input(16 * hidden), wide_routes(16 * used), wide_actual(16 * hidden);
        std::vector<std::int32_t> wide_ids(16 * used);
        for (int row = 0; row < 16; ++row) {
            for (int i = 0; i < hidden; ++i) wide_input[row * hidden + i] = std::sin(.032f * i + row * .071f);
            std::copy(ids.begin(), ids.end(), wide_ids.begin() + row * used);
            std::copy(routes.begin(), routes.end(), wide_routes.begin() + row * used);
        }
        const auto expected = wide_oracle.compute_batch(wide_input, wide_ids, wide_routes);
        auto segment = first.segment; segment.seq_len = 16;
        const auto wide_reads = tsg::g_test_file_reads;
        require(tsg::host_moe_cached_experts(segment, wide_input.data(), wide_ids.data(), wide_routes.data(),
            wide_actual.data(), "glm5-next bulk expert prefill", g_backend) == 1
            && std::memcmp(wide_actual.data(), expected.data(), expected.size() * sizeof(float)) == 0,
            "Victim wide promotion changed genuine bulk CUDA output");
        const auto after_wide = VictimFixtureStats();
        require(tsg::g_test_file_reads == wide_reads && after_wide.v[7] == before_wide.v[7]
            && after_wide.v[11] == before_wide.v[11] && after_wide.v[9] > before_wide.v[9],
            "Wide reuse reread files, demoted scan victims or trained narrow history");

        // D2H is queued before a later source read fails. Pending destinations
        // must drain before refund; none become visible on exception unwind.
        reset(); train();
        ledger.watch_ram_release = true; ledger.watched_ram_token = 0;
        ledger.watched_ram_released = false;
        ledger.drains_before_watched_upload = tsg::host_moe_expert_cache_test_payload_upload_drains();
        const auto before_failure = VictimFixtureStats();
        tsg::g_test_file_failure_countdown = 1;
        std::fill(actual.begin(), actual.end(), -999.f);
        bool interrupted = false;
        try { call(second); } catch (const std::runtime_error& error) {
            if (std::strcmp(error.what(), "Injected interrupted expert file read") != 0) throw;
            interrupted = true;
        }
        const auto after_failure = VictimFixtureStats();
        require(interrupted && tsg::g_test_file_failure_countdown == 0
            && std::all_of(actual.begin(), actual.end(), [](float value) { return value == -999.f; })
            && after_failure.v[7] > before_failure.v[7] && after_failure.v[8] == before_failure.v[8]
            && after_failure.v[2] == 0 && ledger.watched_ram_released && ledger.valid,
            "Interrupted D2H published partial bytes/output or refunded before its drain");
        ledger.watch_ram_release = false;
        run(second, second_expected); run(first, first_expected);

        // Partial RAM promotion plus a file miss must retain every DMA source
        // through failure and leave slots recoverable from immutable files.
        reset(); train(); run(second, second_expected);
        ids = {0, 7};
        const auto mixed_expected = first_oracle.compute(input, ids, routes);
        const auto drains = tsg::host_moe_expert_cache_test_payload_upload_drains();
        tsg::g_test_file_failure_countdown = 1;
        std::fill(actual.begin(), actual.end(), -999.f); interrupted = false;
        try { call(first); } catch (const std::runtime_error& error) {
            if (std::strcmp(error.what(), "Injected interrupted expert file read") != 0) throw;
            interrupted = true;
        }
        require(interrupted && tsg::host_moe_expert_cache_test_payload_upload_drains() > drains
            && std::all_of(actual.begin(), actual.end(), [](float value) { return value == -999.f; }) && ledger.valid,
            "Interrupted mixed promotion did not drain or published incomplete output");
        run(first, mixed_expected); ids = {0, 1};

        // Optional host credit/commit refusal preserves the exact file path.
        reset(); train(); ledger.fail_ram_commit = true;
        const auto before_commit = VictimFixtureStats();
        run(second, second_expected);
        require(VictimFixtureStats().v[7] == before_commit.v[7] && VictimFixtureStats().v[2] == 0,
            "Refused victim commit queued D2H or retained its allocation");
        ledger.fail_ram_commit = false;
        reset(); train();
        ledger.host_capacity = ledger.held(3) + ledger.held(6);
        const auto refusals = ledger.refused;
        run(second, second_expected);
        require(ledger.refused > refusals && VictimFixtureStats().v[2] == 0,
            "Host quota refusal failed exact file fallback");
        ledger.host_capacity = 128LL << 20;
    }
    // Replacement uses the same mapped addresses and local expert IDs.
    reset(); train(); run(second, second_expected);
    for (auto& part : first.quantized) tsg::host_moe_expert_cache_on_drop(part.data());
    require(tsg::host_moe_expert_cache_test_payload_stats().resident_bytes == 0 && VictimFixtureStats().v[11] == 0,
        "Source drop retained old-generation payload/history");
    register_source(first, 2);
    const auto replacement_reads = tsg::g_test_file_reads;
    run(first, replacement_expected);
    require(tsg::g_test_file_reads > replacement_reads, "Replacement pointers consumed stale victim bytes");
    std::int64_t released = 0;
    require(TSGgml_EndHostMoeExpertCacheScope(token, &released) == 1, "Victim scope did not end"); token = 0;
    tsg::host_moe_expert_cache_release();
    require(ledger.valid && ledger.allocations.empty() && VictimFixtureStats().v[0] == 0,
        "Victim teardown retained physical allocation or shared credit");
    require(tsg::SharedCacheCharge::detach(&ledger), "Victim ledger could not detach"); ledger.attached = false;
    for (const auto& path : paths) require(std::remove(path.c_str()) == 0, "Victim source handle survived teardown");
    std::puts(pageable
        ? "PASS: selective victim pageable refusal; exact CUDA file fallback, source generation and teardown (D2H deliberately unavailable)"
        : "PASS: selective victim pinned D2H/exclusive promotion; exact scalar/wide CUDA, old-layer provenance, physical/history bounds, drop/address reuse, D2H and mixed-promotion failure drains, host quota/commit refusal, complete teardown");
}
#endif
