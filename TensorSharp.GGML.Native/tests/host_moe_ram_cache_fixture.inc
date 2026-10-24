// CUDA end-to-end coverage for the optional charged RAM source tier. The
// independent resident graphs own the original bytes; mapped pointers below
// are poisoned so accidental file/cache bypass cannot satisfy exact parity.
#ifdef TSG_GGML_USE_CUDA
extern "C" int TSGgml_HostMoeRamCacheConfigure(std::int64_t);
extern "C" int TSGgml_HostMoeRamCacheStats(std::int64_t*, int);

struct RamFixtureStats {
    std::array<std::int64_t, 9> values{};
    RamFixtureStats() {
        require(TSGgml_HostMoeRamCacheStats(values.data(), static_cast<int>(values.size())) == 1,
            "Cannot observe RAM expert cache counters");
        require(values[0] >= 0 && values[1] >= 0 && values[2] >= 0 && values[1] <= values[2]
            && values[2] <= values[0], "RAM expert cache exceeded its physical byte ceiling");
    }
};

struct RamFixtureLedger {
    struct Allocation { int kind; std::int64_t bytes; bool committed = false; };
    std::map<std::uint64_t, Allocation> allocations;
    std::uint64_t next = 0, refused = 0;
    std::int64_t host_capacity = 32LL << 20, device_capacity = 32LL << 20;
    bool attached = false, valid = true, fail_ram_commit = false;
    bool watch_ram_release = false, watched_ram_released = false;
    std::uint64_t watched_ram_token = 0, drains_before_watched_upload = 0;
    std::int64_t held(int kind) const {
        std::int64_t result = 0;
        for (const auto& item : allocations) if (item.second.kind == kind) result += item.second.bytes;
        return result;
    }
    static std::uint64_t reserve(void* context, int rank, int kind, std::int64_t bytes) {
        auto& ledger = *static_cast<RamFixtureLedger*>(context);
        if (rank != 0 || (kind != 3 && kind != 4 && kind != 6) || bytes <= 0) {
            ledger.valid = false; return 0;
        }
        const auto free = kind == 4 ? ledger.device_capacity - ledger.held(4)
            : ledger.host_capacity - ledger.held(3) - ledger.held(6);
        if (bytes > free) { ++ledger.refused; return 0; }
        const auto token = ++ledger.next;
        ledger.allocations.emplace(token, Allocation{kind, bytes});
        if (kind == 6 && ledger.watch_ram_release && !ledger.watched_ram_token)
            ledger.watched_ram_token = token;
        return token;
    }
    static int commit(void* context, std::uint64_t token) {
        auto& ledger = *static_cast<RamFixtureLedger*>(context);
        const auto found = ledger.allocations.find(token);
        if (found == ledger.allocations.end() || found->second.committed) { ledger.valid = false; return 0; }
        if (found->second.kind == 6 && ledger.fail_ram_commit) return 0;
        found->second.committed = true; return 1;
    }
    static void release(void* context, std::uint64_t token) {
        auto& ledger = *static_cast<RamFixtureLedger*>(context);
        if (ledger.watch_ram_release && token == ledger.watched_ram_token) {
            // This getter only loads an atomic; accounting callbacks must not
            // reenter cache APIs or take their global ownership mutex.
            if (tsg::host_moe_expert_cache_test_payload_upload_drains() <= ledger.drains_before_watched_upload)
                ledger.valid = false;
            ledger.watched_ram_released = true;
        }
        if (ledger.allocations.erase(token) != 1) ledger.valid = false;
    }
    ~RamFixtureLedger() {
        if (attached) {
            tsg::host_moe_expert_cache_release();
            tsg::SharedCacheCharge::detach(this);
        }
    }
};

void host_moe_ram_cache_checks(const std::array<ggml_type, 3>& types) {
    constexpr std::int64_t capacity = 16LL << 20;
    tsg::host_moe_expert_cache_release();
    require(TSGgml_HostMoeRamCacheConfigure(capacity) == 0,
        "Persistent RAM payloads were enabled without attached host accounting");
    require(TSGgml_HostMoeRamCacheConfigure(-1) == 0,
        "Negative RAM cache capacity was accepted");
    std::array<std::int64_t, 9> short_stats{};
    require(TSGgml_HostMoeRamCacheStats(nullptr, 9) == 0
        && TSGgml_HostMoeRamCacheStats(short_stats.data(), 8) == 0,
        "RAM cache statistics accepted a missing/short destination");

    Weights first, replacement;
    first.types = replacement.types = types;
    first.fill(.72f); replacement.fill(2.12f);
    FullResidentGraph original_oracle(first), replacement_oracle(replacement), bulk_oracle(first, 16);
    std::array<std::string, 2> paths{fixture_file_path("host-moe-ram-original-"),
                                   fixture_file_path("host-moe-ram-replacement-")};
    std::array<std::array<std::int64_t, 3>, 2> offsets{};
    RamFixtureLedger ledger;
    std::uint64_t token = 0;
    struct Cleanup {
        std::array<std::string, 2>& paths; std::uint64_t& token;
        ~Cleanup() {
            tsg::g_test_file_failure_countdown = 0;
            if (token) { std::int64_t released = 0; TSGgml_EndHostMoeExpertCacheScope(token, &released); }
            tsg::host_moe_expert_cache_release();
            for (const auto& path : paths) std::remove(path.c_str());
        }
    } cleanup{paths, token};
    for (int version = 0; version < 2; ++version) {
        const auto& weights = version ? replacement : first;
        std::ofstream file(paths[version], std::ios::binary | std::ios::trunc);
        require(bool(file), "Cannot create RAM cache immutable source fixture");
        for (int part = 0; part < 3; ++part) {
            file.write("source-offset!", 14);
            offsets[version][part] = static_cast<std::int64_t>(file.tellp());
            file.write(reinterpret_cast<const char*>(weights.quantized[part].data()), weights.quantized[part].size());
        }
        require(bool(file), "Cannot write RAM cache immutable source fixture");
    }
    auto register_version = [&](int version) {
        for (int part = 0; part < 3; ++part) {
            auto& bytes = first.quantized[part];
            require(TSGgml_RegisterHostFileSource(bytes.data(), bytes.size(), paths[version].c_str(), offsets[version][part]) == 1,
                "Cannot register RAM cache exact source extent");
            std::fill(bytes.begin(), bytes.end(), 0);
        }
    };
    register_version(0);
    require(TSGgml_BeginHostMoeExpertCacheScopeWithDecodeRows(32LL << 20, 32, 1, 3, &token) == 1,
        "Cannot begin RAM cache CUDA scope");
    ledger.attached = tsg::SharedCacheCharge::attach(&ledger, RamFixtureLedger::reserve,
        RamFixtureLedger::commit, RamFixtureLedger::release, false, true);
    require(ledger.attached && TSGgml_HostMoeRamCacheConfigure(capacity) == 1,
        "Cannot enable charged RAM cache fixture");
    std::array<float, hidden> input{}, actual{};
    for (int i = 0; i < hidden; ++i) input[i] = std::sin(.032f * i);
    std::array<std::int32_t, used> ids{3, 11};
    std::array<float, used> routes{.65f, .35f};
    const auto original_expected = original_oracle.compute(input, ids, routes);
    const auto replacement_expected = replacement_oracle.compute(input, ids, routes);
    require(std::memcmp(original_expected.data(), replacement_expected.data(), sizeof(actual)) != 0,
        "Replacement source fixture did not change the reference output");
    auto run = [&](const std::array<float, hidden>& expected) {
        std::fill(actual.begin(), actual.end(), -999.0f);
        require(tsg::host_moe_cached_experts(first.segment, input.data(), ids.data(), routes.data(), actual.data(),
            "glm5-next selected expert seam", g_backend) == 1,
            "RAM backed scalar expert call did not execute");
        require(std::memcmp(actual.data(), expected.data(), sizeof(actual)) == 0,
            "RAM backed scalar output differs from the independent full-resident CUDA graph");
    };
    const auto before_cold = RamFixtureStats();
    run(original_expected);
    const auto cold = RamFixtureStats();
    const auto staging_bytes = ledger.held(3);
    require(cold.values[1] > 0 && cold.values[6] > before_cold.values[6]
        && ledger.held(6) > 0 && staging_bytes > 0,
        "Cold selected file reads did not populate charged RAM payloads");
    require(TSGgml_TrimHostMoeExpertCache(0) > 0 && Stats().reserved == 0
        && ledger.held(3) == 0 && ledger.held(4) == 0 && ledger.held(6) > 0,
        "Device cache trim dropped RAM reuse or retained device/staging credit");
    run(original_expected);
    const auto warm = RamFixtureStats();
    require(warm.values[5] > cold.values[5] && warm.values[6] == cold.values[6],
        "RAM hit after physical VRAM retirement reread the source file");

    // A genuine wide graph must also consume RAM hits without touching poisoned
    // mapped bytes. Each row selects the exact same previously loaded payload.
    TSGgml_TrimHostMoeExpertCache(0);
    std::vector<float> bulk_input(16 * hidden), bulk_routes(16 * used), bulk_actual(16 * hidden);
    std::vector<std::int32_t> bulk_ids(16 * used);
    for (int row = 0; row < 16; ++row) {
        for (int i = 0; i < hidden; ++i) bulk_input[row * hidden + i] = std::sin(.032f * i + row * .071f);
        std::copy(ids.begin(), ids.end(), bulk_ids.begin() + row * used);
        std::copy(routes.begin(), routes.end(), bulk_routes.begin() + row * used);
    }
    const auto bulk_expected = bulk_oracle.compute_batch(bulk_input, bulk_ids, bulk_routes);
    auto bulk_segment = first.segment; bulk_segment.seq_len = 16;
    require(tsg::host_moe_cached_experts(bulk_segment, bulk_input.data(), bulk_ids.data(), bulk_routes.data(),
        bulk_actual.data(), "glm5-next bulk expert prefill", g_backend) == 1,
        "RAM backed wide expert prefill did not execute");
    require(std::memcmp(bulk_actual.data(), bulk_expected.data(), bulk_actual.size() * sizeof(float)) == 0,
        "RAM backed wide prefill differs from the full-resident CUDA graph");
    const auto wide = RamFixtureStats();
    require(wide.values[5] > warm.values[5] && wide.values[6] == warm.values[6],
        "Wide prefill did not reuse the complete previously loaded RAM payload");

    TSGgml_TrimHostMoeExpertCache(0);
    require(TSGgml_HostMoeRamCacheConfigure(0) == 1 && RamFixtureStats().values[1] == 0
        && RamFixtureStats().values[2] == 0 && ledger.held(6) == 0,
        "Zero RAM capacity retained physical payload credit");
    ledger.host_capacity = staging_bytes;
    require(TSGgml_HostMoeRamCacheConfigure(capacity) == 1, "Cannot restore optional RAM capacity");
    const auto refusals = ledger.refused;
    run(original_expected);
    require(ledger.refused > refusals && ledger.held(6) == 0 && RamFixtureStats().values[1] == 0,
        "Host quota refusal allocated RAM payloads or skipped exact file fallback");
    TSGgml_TrimHostMoeExpertCache(0);
    ledger.host_capacity = 32LL << 20; ledger.fail_ram_commit = true;
    run(original_expected);
    require(ledger.held(6) == 0 && RamFixtureStats().values[1] == 0,
        "Refused RAM commit retained an allocation or prevented exact file fallback");
    ledger.fail_ram_commit = false;
    TSGgml_TrimHostMoeExpertCache(0);
    run(original_expected);
    require(RamFixtureStats().values[1] > 0, "RAM cache did not recover after quota/commit refusal");

    const auto enabled = [](const char* name) {
        const char* value = std::getenv(name);
        return value && value[0] == '1' && value[1] == '\0';
    };
    const bool direct_lease = enabled("TS_HOST_MOE_RAM_CACHE_DIRECT_UPLOAD")
        || enabled("TS_HOST_MOE_RAM_CACHE_PINNED");
    const bool frequency = enabled("TS_HOST_MOE_RAM_CACHE_FREQUENCY");
    const auto interrupted_upload = [&](int fail_after, std::int64_t ram_limit, bool require_unwind_release) {
        TSGgml_TrimHostMoeExpertCache(0);
        require(TSGgml_HostMoeRamCacheConfigure(0) == 1
            && TSGgml_HostMoeRamCacheConfigure(ram_limit) == 1,
            "Cannot reset RAM capacity before interrupted upload");
        ledger.watch_ram_release = direct_lease;
        ledger.watched_ram_token = 0; ledger.watched_ram_released = false;
        ledger.drains_before_watched_upload = tsg::host_moe_expert_cache_test_payload_upload_drains();
        const auto reads_before = tsg::g_test_file_reads;
        std::fill(actual.begin(), actual.end(), -999.0f);
        tsg::g_test_file_failure_countdown = fail_after;
        bool interrupted = false;
        try {
            tsg::host_moe_cached_experts(first.segment, input.data(), ids.data(), routes.data(), actual.data(),
                "glm5-next selected expert seam", g_backend);
        } catch (const std::runtime_error& error) {
            if (std::strcmp(error.what(), "Injected interrupted expert file read") == 0) interrupted = true;
            else throw;
        }
        require(interrupted && tsg::g_test_file_failure_countdown == 0
            && tsg::g_test_file_reads == reads_before + fail_after - 1,
            "Later file failure did not occur after the first upload was consumed");
        require(std::all_of(actual.begin(), actual.end(), [](float value) { return value == -999.0f; }),
            "Interrupted expert upload published an incomplete inference result");
        if (direct_lease) {
            require(ledger.watched_ram_token != 0
                && tsg::host_moe_expert_cache_test_payload_upload_drains() > ledger.drains_before_watched_upload,
                "Direct cached upload did not drain its acquired payload lease on failure");
            if (require_unwind_release)
                require(ledger.watched_ram_released,
                    "Unlinked payload lease did not physically release during exception unwind");
        }
        require(TSGgml_HostMoeRamCacheConfigure(0) == 1 && ledger.held(6) == 0
            && RamFixtureStats().values[2] == 0 && ledger.valid,
            "Failure cleanup refunded a live DMA source before drain or retained its RAM lease");
        if (direct_lease) require(ledger.watched_ram_released,
            "Zero capacity did not release the interrupted cached upload source");
        ledger.watch_ram_release = false;
        TSGgml_TrimHostMoeExpertCache(0);
        require(TSGgml_HostMoeRamCacheConfigure(capacity) == 1,
            "Cannot restore RAM cache after interrupted upload");
        run(original_expected);
    };
    // The standalone reader completes tasks in reverse order: task N-1 is
    // prepared and uploaded before task N-2 throws. No failure is inferred
    // from a skipped branch or a synthetic callback-only source.
    interrupted_upload(2, capacity, false);
    if (direct_lease && !frequency) {
        // With only one projection of RAM, reading the second projection
        // unlinks the first LRU entry while its upload lease is still live.
        // The third read then throws; releasing that first physical owner must
        // happen after the inner upload drain, during task-vector unwinding.
        interrupted_upload(3, static_cast<std::int64_t>(first.quantized[2].size() / experts), true);
    }

    for (auto& part : first.quantized) tsg::host_moe_expert_cache_on_drop(part.data());
    require(RamFixtureStats().values[1] == 0 && ledger.held(6) == 0 && Stats().reserved == 0,
        "Source drop retained payloads under their old source identity");
    register_version(1);
    const auto before_reload = RamFixtureStats();
    run(replacement_expected);
    require(RamFixtureStats().values[6] > before_reload.values[6],
        "Reused source pointers inherited old RAM payloads instead of the new file");
    TSGgml_TrimHostMoeExpertCache(0);
    const auto reload_cold = RamFixtureStats();
    run(replacement_expected);
    require(RamFixtureStats().values[5] > reload_cold.values[5]
        && RamFixtureStats().values[6] == reload_cold.values[6],
        "Replacement source failed RAM-hit reuse after GPU trim");
    std::int64_t released = 0;
    require(TSGgml_EndHostMoeExpertCacheScope(token, &released) == 1, "RAM fixture scope failed to end");
    token = 0;
    tsg::host_moe_expert_cache_release();
    require(ledger.valid && ledger.allocations.empty() && RamFixtureStats().values[0] == 0
        && RamFixtureStats().values[1] == 0 && RamFixtureStats().values[2] == 0,
        "RAM cache teardown retained shared ownership or configuration");
    require(tsg::SharedCacheCharge::detach(&ledger), "RAM cache teardown prevented host ledger detach");
    ledger.attached = false;
    for (const auto& path : paths) require(std::remove(path.c_str()) == 0, "RAM fixture retained a source handle");
    std::puts("PASS: charged RAM expert cache; exact cold/warm scalar and wide CUDA output; GPU retirement reuse; host refusal/commit fallback; partial-upload failure and drain-before-lease-release; trim0; source pointer reuse; complete teardown");
}
#endif
