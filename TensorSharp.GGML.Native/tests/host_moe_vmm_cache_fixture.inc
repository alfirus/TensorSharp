// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// The production cache and real VMM adapter, under an explicit test-only small
// geometry. This does not qualify the public GLM model profile or performance.
#ifdef TSG_GGML_USE_CUDA
struct VmmCacheLedger {
    struct Ticket { std::uint64_t token = 0; std::size_t bytes = 0; int kind = 0; bool committed = false, refunded = false; };
    std::array<Ticket, 8256> tickets{};
    std::size_t capacity = 0, host_capacity = 0, pending = 0, committed = 0, host_pending = 0, host_committed = 0;
    std::size_t peak_device = 0, peak_host = 0;
    std::uint64_t next = 0, reserves = 0, commits = 0, releases = 0, acknowledgments = 0;
    bool valid = true, attached = false, fail_device_ack = false, fail_host_ack = false;
    Ticket* find(std::uint64_t token) noexcept { for (auto& t : tickets) if (t.token == token) return &t; return nullptr; }
    static std::uint64_t reserve(void* context, int rank, int kind, std::int64_t bytes) noexcept {
        auto& l = *static_cast<VmmCacheLedger*>(context); ++l.reserves;
        if (rank != 0 || (kind != 3 && kind != 4) || bytes <= 0) { l.valid = false; return 0; }
        const bool host = kind == 3;
        const auto size = static_cast<std::size_t>(bytes), owned = host ? l.host_pending + l.host_committed : l.pending + l.committed;
        const auto limit = host ? l.host_capacity : l.capacity;
        if (owned > limit || size > limit - owned) return 0;
        Ticket* slot = nullptr; for (auto& t : l.tickets) if (!t.token) { slot = &t; break; }
        if (!slot) { l.valid = false; return 0; }
        *slot = {++l.next, size, kind};
        (host ? l.host_pending : l.pending) += size;
        (host ? l.peak_host : l.peak_device) = (std::max)(host ? l.peak_host : l.peak_device, owned + size);
        return slot->token;
    }
    static int commit(void* context, std::uint64_t token) noexcept {
        auto& l = *static_cast<VmmCacheLedger*>(context); ++l.commits;
        auto* t = l.find(token);
        if (!t || t->committed || t->refunded) { l.valid = false; return 0; }
        auto& pending = t->kind == 3 ? l.host_pending : l.pending;
        auto& committed = t->kind == 3 ? l.host_committed : l.committed;
        if (pending < t->bytes) { l.valid = false; return 0; }
        pending -= t->bytes; committed += t->bytes; t->committed = true; return 1;
    }
    static int release_checked(void* context, std::uint64_t token) noexcept {
        auto& l = *static_cast<VmmCacheLedger*>(context); ++l.releases;
        auto* t = l.find(token); if (!t) { l.valid = false; return 0; }
        if (!t->refunded) {
            auto& owned = t->kind == 3 ? (t->committed ? l.host_committed : l.host_pending)
                : (t->committed ? l.committed : l.pending);
            if (owned < t->bytes) { l.valid = false; return 0; }
            owned -= t->bytes; t->refunded = true;
            bool& fail = t->kind == 3 ? l.fail_host_ack : l.fail_device_ack;
            if (fail) { fail = false; return 0; }
        }
        *t = {}; ++l.acknowledgments; return 1;
    }
    static void release(void* context, std::uint64_t token) noexcept { (void)release_checked(context, token); }
    bool attach() {
        attached = tsg::SharedCacheCharge::attach(this, reserve, commit, release, false, true, release_checked);
        return attached;
    }
    std::size_t held(int kind = 0) const noexcept {
        std::size_t result = 0; for (const auto& t : tickets) result += t.token && (!kind || t.kind == kind); return result;
    }
    std::size_t unacknowledged(int kind) const noexcept {
        std::size_t result = 0; for (const auto& t : tickets) result += t.token && t.kind == kind && t.refunded; return result;
    }
    void check() const {
        require(valid && pending == 0 && host_pending == 0 && pending + committed <= capacity
            && host_pending + host_committed <= host_capacity && peak_device <= capacity && peak_host <= host_capacity,
            "VMM cache exceeded a shared payload ceiling or corrupted a ticket");
    }
};

void host_moe_vmm_cache_checks(bool pressure_mode = false) {
    constexpr int model_hidden = 512, model_ff = 512, count = 48, top = 2, rows = 16;
    // selected_expert_segment uses every loaded trunk layer for the common
    // pooled identity. Quantization-group frequency is a separate quota weight.
    constexpr int loaded_layers = 45, pool_extent = loaded_layers * count;
    constexpr int group_layers[3] = {11, 28, 3};
    using Snapshot = tsg::HostMoeCacheTestSnapshot;
    using Slot = tsg::HostMoeCacheTestSlot;
    tsg::host_moe_expert_cache_release();
    tsg::host_moe_expert_cache_test_vmm_profile(true);
    require(TSGgml_HostMoeVmmWeightsVersion() == 1, "VMM cache ABI profile is unavailable");
    // The checked arena requires the caller's actual CUDA primary context to
    // be current. Backend discovery alone does not establish that precondition.
    // Previous reference graph allocations did so implicitly; make it explicit
    // with one bounded ordinary allocation, retired before any cache ledger.
    const auto bootstrap_epoch = tsg::g_ggml_error_count.load(std::memory_order_acquire);
    {
        struct Bootstrap {
            ggml_backend_buffer_t buffer;
            ~Bootstrap() { ggml_backend_buffer_free(buffer); }
        } bootstrap{ggml_backend_alloc_buffer(g_backend, 256)};
        require(bootstrap.buffer != nullptr
            && ggml_backend_buffer_get_type(bootstrap.buffer) == ggml_backend_get_default_buffer_type(g_backend),
            "Cannot establish the ordinary CUDA caller context for the VMM arena");
        ggml_backend_synchronize(g_backend);
    }
    require(tsg::g_ggml_error_count.load(std::memory_order_acquire) == bootstrap_epoch
        && !tsg::g_backend_compute_failed.load(std::memory_order_acquire),
        "Ordinary CUDA context bootstrap reported a backend error");
    std::fprintf(stderr, "[vmm-cache-fixture] caller_context_bootstrap_bytes=256 ordinary_buffer_freed_before_cache_ledger=1\n");
    std::fflush(stderr);
    std::array<Weights, 4> weights;
    weights[0].types = {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS};
    weights[1].types = {GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_S, GGML_TYPE_IQ3_XXS};
    weights[2].types = {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ4_XS};
    weights[3].types = weights[0].types;
    const auto path = fixture_file_path("glm-vmm-cache-");
    struct FileCleanup {
        const std::string& path; bool active = true;
        ~FileCleanup() {
            if (!active) return;
            try { tsg::host_moe_expert_cache_release(); } catch (...) {}
            try { tsg::host_moe_expert_cache_test_vmm_profile(false); } catch (...) {}
            std::remove(path.c_str());
        }
    } file_cleanup{path};
    std::array<std::array<std::int64_t, 3>, 4> offsets{};
    const auto preparing = std::chrono::steady_clock::now();
    for (int source = 0; source != 4; ++source) {
        std::fprintf(stderr, "[vmm-cache-fixture] prepare source=%d\n", source); std::fflush(stderr);
        auto& w = weights[source];
        for (auto type : w.types) ggml_quantize_init(type);
        w.fill(.43f + source * .71f, count, top, model_hidden, model_ff);
        const int group = source == 3 ? 0 : source;
        w.segment.layer = source == 3 ? 1 : (group == 0 ? 0 : group == 1 ? 11 : 39);
        w.segment.pooled_model_experts = pool_extent;
        const auto key = tsg::host_moe_expert_group_key(w.segment);
        require(key.values[3] == model_hidden && key.values[4] == model_ff && key.values[5] == top
            && key.values[6] == count && key.values[7] == pool_extent && key.clamp_bits == 0,
            "Tiny VMM source does not preserve the actual common loaded-layer pool contract");
    }
    {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        require(bool(file), "Cannot create VMM cache source file");
        for (int source = 0; source != 4; ++source) for (int part = 0; part != 3; ++part) {
            file.write("source-generation-offset", 24); offsets[source][part] = static_cast<std::int64_t>(file.tellp());
            const auto& payload = weights[source].quantized[part];
            file.write(reinterpret_cast<const char*>(payload.data()), payload.size());
        }
        require(bool(file), "Cannot write VMM cache source file");
    }
    std::fprintf(stderr, "[vmm-cache-fixture] preparation_ms=%.3f\n",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - preparing).count()); std::fflush(stderr);

    std::size_t compared = 0;
    const auto stats = [] {
        std::array<std::int64_t, 12> value{};
        require(TSGgml_HostMoeVmmWeightsStats(value.data(), static_cast<int>(value.size())) == 1,
            "Cannot read VMM cache owner certificate");
        return value;
    };
    const auto snapshot = [&](int group) {
        Snapshot value{};
        require(tsg::host_moe_expert_cache_test_lifecycle_snapshot(weights[group].segment, g_backend, value),
            "VMM cache lost an expected group table");
        require(value.vmm && value.vmm_certificate && value.vmm_usable && !value.vmm_failed
            && value.vmm_graph_binding_valid && value.vmm_table_identity && value.vmm_pending_bytes == 0
            && value.vmm_quarantine_bytes == 0 && value.weight_physical_bytes > 0
            && value.weight_physical_bytes >= value.weight_bytes,
            "VMM cache graph is not bound to actual certified stable-VMM leaves");
        tsg::HostMoeCacheTestPlan exact{};
        require(tsg::host_moe_expert_cache_test_split_plan(weights[group].segment, value.capacity, 1,
            value.wide_rows > 8 ? value.wide_rows : 8, g_backend, exact)
            && exact.weight_bytes == value.weight_bytes && exact.weight_physical_bytes == value.weight_physical_bytes,
            "VMM snapshot conflates logical packed leaves with the actual rounded backing plan");
        return value;
    };
    const auto slots = [&](int group, const Snapshot& value) {
        std::vector<Slot> result(value.capacity);
        require(tsg::host_moe_expert_cache_test_copy_slot_metadata(weights[group].segment, g_backend,
            result.data(), result.size()), "Cannot read the complete same-owner VMM slot vector");
        return result;
    };
    // Assigned after the source-byte verifier below, before any selected call.
    // The reference independently reconstructs the actual table geometry from
    // source-qualified slot tags and the generated file, never GPU payload.
    std::function<void(int)> diagnose_payload;
    std::function<std::vector<float>(int, int, const std::vector<float>&,
        const std::vector<std::int32_t>&, const std::vector<float>&)> matched_reference;
    const auto call = [&](int source, int width, int first, bool repeated_pair = false) {
        std::vector<float> input(static_cast<std::size_t>(width) * model_hidden), routes(width * top), actual(input.size(), -999.0f), expected;
        std::vector<std::int32_t> ids(width * top);
        for (int r = 0; r != width; ++r) {
            for (int h = 0; h != model_hidden; ++h) input[r * model_hidden + h] = std::sin(h * .013f + r * .07f);
            ids[r * top] = (first + (repeated_pair ? 0 : r * top)) % count;
            ids[r * top + 1] = (ids[r * top] + 1) % count;
            routes[r * top] = .6f; routes[r * top + 1] = .4f;
        }
        auto hm = weights[source].segment; hm.seq_len = width;
        const auto before_submissions = ExecutionStats().submissions;
        try {
            require(tsg::host_moe_cached_experts(hm, input.data(), ids.data(), routes.data(), actual.data(),
                width > 8 ? "glm5-next bulk expert prefill" : "glm5-next selected expert seam", g_backend) == 1,
                "VMM cache selected-expert call refused");
        } catch (...) {
            require(ExecutionStats().submissions == before_submissions
                && std::all_of(actual.begin(), actual.end(), [](float x) { return x == -999.0f; }),
                "VMM postmutation refusal submitted a helper or published partial output");
            throw;
        }
        require(static_cast<bool>(matched_reference), "VMM matched-geometry reference was not prepared");
        expected = matched_reference(source, width, input, ids, routes);
        require(expected.size() == actual.size(), "VMM matched-geometry reference output extent changed");
        if (std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) != 0
            || !std::all_of(actual.begin(), actual.end(), [](float x) { return std::isfinite(x); })) {
            std::size_t mismatch_count = 0, first_mismatch = actual.size(), actual_nonfinite = 0, expected_nonfinite = 0;
            double maximum_absolute = 0, maximum_relative = 0, maximum_actual_magnitude = 0, maximum_expected_magnitude = 0;
            for (std::size_t i = 0; i != actual.size(); ++i) {
                std::uint32_t a = 0, e = 0;
                std::memcpy(&a, &actual[i], sizeof(a)); std::memcpy(&e, &expected[i], sizeof(e));
                if (a != e) { ++mismatch_count; first_mismatch = (std::min)(first_mismatch, i); }
                actual_nonfinite += !std::isfinite(actual[i]); expected_nonfinite += !std::isfinite(expected[i]);
                if (std::isfinite(actual[i])) maximum_actual_magnitude = (std::max)(maximum_actual_magnitude, std::abs(static_cast<double>(actual[i])));
                if (std::isfinite(expected[i])) maximum_expected_magnitude = (std::max)(maximum_expected_magnitude, std::abs(static_cast<double>(expected[i])));
                if (std::isfinite(actual[i]) && std::isfinite(expected[i])) {
                    const double difference = std::abs(static_cast<double>(actual[i]) - expected[i]);
                    maximum_absolute = (std::max)(maximum_absolute, difference);
                    maximum_relative = (std::max)(maximum_relative,
                        difference / (std::max)(std::abs(static_cast<double>(expected[i])), 1e-30));
                }
            }
            // A same-bit nonfinite result still fails, independently of the
            // numerical mismatch count, so keep the diagnostic index bounded.
            if (first_mismatch == actual.size()) first_mismatch = 0;
            std::uint32_t a = 0, e = 0;
            std::memcpy(&a, &actual[first_mismatch], sizeof(a)); std::memcpy(&e, &expected[first_mismatch], sizeof(e));
            std::fprintf(stderr, "[vmm-cache-fixture] exact-mismatch source=%d width=%d first_expert=%d repeated_pair=%d "
                "row=%zu column=%zu mismatches=%zu total=%zu actual=%a expected=%a actual_bits=%08x expected_bits=%08x "
                "max_abs=%.17g max_rel=%.17g max_actual_magnitude=%.17g max_expected_magnitude=%.17g actual_nonfinite=%zu expected_nonfinite=%zu "
                "submitted_projections=%d submitted_rows=%d submitted_topk=%d submitted_bulk=%d submitted_shape_valid=%d\n",
                source, width, first, repeated_pair ? 1 : 0, first_mismatch / model_hidden, first_mismatch % model_hidden,
                mismatch_count, actual.size(), static_cast<double>(actual[first_mismatch]), static_cast<double>(expected[first_mismatch]),
                static_cast<unsigned>(a), static_cast<unsigned>(e), maximum_absolute, maximum_relative,
                maximum_actual_magnitude, maximum_expected_magnitude,
                actual_nonfinite, expected_nonfinite, tsg::g_test_operator_shape.projections, tsg::g_test_operator_shape.rows,
                tsg::g_test_operator_shape.selected, tsg::g_test_operator_shape.bulk ? 1 : 0, tsg::g_test_operator_shape.valid ? 1 : 0);
            std::fflush(stderr);
            require(static_cast<bool>(diagnose_payload), "VMM mismatch payload diagnostics were not prepared");
            diagnose_payload(source);
            throw std::runtime_error("VMM cache changed matched-geometry standard-CUDA expert output bits");
        }
        compared += actual.size();
    };

    // Poison the registered pointer ranges after writing the immutable file.
    // Both the actual cache and independent reference must read file bytes.
    for (auto& w : weights) for (auto& payload : w.quantized) std::fill(payload.begin(), payload.end(), 0);
    const auto total = [](const tsg::HostMoeCacheTestPlan& p) { return p.weight_physical_bytes + p.narrow_bytes + p.wide_bytes; };
    const auto same_key = [](const Slot& a, const Slot& b) {
        return a.expert == b.expert && a.source[0] == b.source[0]
            && a.source[1] == b.source[1] && a.source[2] == b.source[2];
    };
    const auto source_of = [&](const Slot& slot) {
        for (int s = 0; s != 4; ++s) {
            bool same = true;
            for (int p = 0; p != 3; ++p) same &= slot.source[p] == reinterpret_cast<std::uintptr_t>(weights[s].quantized[p].data());
            if (same) return s;
        }
        throw std::runtime_error("A VMM slot lost its complete immutable source identity");
    };
    const auto verify_payload = [&](int group, const Snapshot& value, const std::vector<Slot>& live) {
        std::ifstream file(path, std::ios::binary);
        require(bool(file), "Cannot reopen generated VMM fixture reference bytes");
        for (int p = 0; p != 3; ++p) {
            const auto stride = weights[group].quantized[p].size() / count;
            require(value.weight_part_bytes[p] == stride * static_cast<std::size_t>(value.capacity),
                "VMM logical tensor extent differs from its complete slot vector");
            std::vector<std::uint8_t> actual(value.weight_part_bytes[p]), expected(stride);
            require(tsg::host_moe_expert_cache_test_copy_weight_payload(weights[group].segment, g_backend,
                p, actual.data(), actual.size()), "Cannot inspect the actual bound VMM projection payload");
            for (int i = 0; i != value.capacity; ++i) if (live[i].expert >= 0) {
                const int source = source_of(live[i]);
                require(live[i].expert < count && weights[source].types == weights[group].types,
                    "VMM slot source has a different projection type or expert bound");
                file.seekg(offsets[source][p] + static_cast<std::int64_t>(stride) * live[i].expert);
                file.read(reinterpret_cast<char*>(expected.data()), expected.size());
                require(bool(file) && std::memcmp(actual.data() + static_cast<std::size_t>(i) * stride,
                    expected.data(), stride) == 0, "VMM compaction changed source-qualified raw expert bytes");
            }
        }
    };
    diagnose_payload = [&](int source) {
        const int group = source == 3 ? 0 : source;
        Snapshot value{};
        require(tsg::host_moe_expert_cache_test_lifecycle_snapshot(weights[group].segment, g_backend, value),
            "VMM mismatch could not inspect its actual cache owner");
        const auto live = slots(group, value);
        std::fprintf(stderr, "[vmm-cache-fixture] mismatch-payload-check source=%d group=%d capacity=%d resident=%d "
            "logical_weight_bytes=%zu mapped_weight_bytes=%zu narrow_bytes=%zu wide_bytes=%zu "
            "certificate=%d usable=%d failed=%d graph_binding=%d pending=%zu quarantine=%zu state=checking\n",
            source, group, value.capacity, value.resident, value.weight_bytes, value.weight_physical_bytes,
            value.narrow_bytes, value.wide_bytes, value.vmm_certificate ? 1 : 0, value.vmm_usable ? 1 : 0,
            value.vmm_failed ? 1 : 0, value.vmm_graph_binding_valid ? 1 : 0, value.vmm_pending_bytes, value.vmm_quarantine_bytes);
        std::fflush(stderr);
        verify_payload(group, value, live);
        std::fprintf(stderr, "[vmm-cache-fixture] mismatch-payload-check source=%d group=%d "
            "all_occupied_projection_rows_exact=1 state=passed matched_cuda_output_still_failed=1\n", source, group);
        std::fflush(stderr);
    };
    matched_reference = [&](int source, int width, const std::vector<float>& input,
        const std::vector<std::int32_t>& ids, const std::vector<float>& routes) {
        const int group = source == 3 ? 0 : source;
        const auto value = snapshot(group);
        require(value.capacity >= top && value.capacity <= pool_extent,
            "VMM reference table capacity is outside the certified fixture pool");
        const auto live = slots(group, value);
        // This raw readback is a separate assertion. It is never an oracle
        // input: the ordinary CUDA weights below are reconstructed from FILE.
        verify_payload(group, value, live);
        std::size_t resident = 0;
        for (int slot = 0; slot != value.capacity; ++slot) {
            require(live[slot].expert >= -1 && live[slot].expert < count,
                "VMM reference encountered an invalid original expert identity");
            if (live[slot].expert < 0) continue;
            ++resident;
            const int owner = source_of(live[slot]);
            require(weights[owner].types == weights[group].types,
                "VMM reference encountered an incompatible immutable source triplet");
            for (int prior = 0; prior != slot; ++prior)
                require(live[prior].expert < 0 || !same_key(live[prior], live[slot]),
                    "VMM reference encountered duplicate source-qualified slots");
        }
        require(resident == static_cast<std::size_t>(value.resident),
            "VMM reference complete slot vector disagrees with resident accounting");
        std::vector<std::int32_t> selected(ids.size(), -1);
        for (std::size_t requested = 0; requested != ids.size(); ++requested) {
            require(ids[requested] >= 0 && ids[requested] < count,
                "VMM reference original requested ID is invalid");
            int matches = 0;
            for (int slot = 0; slot != value.capacity; ++slot) {
                bool same = live[slot].expert == ids[requested];
                for (int p = 0; p != 3; ++p)
                    same &= live[slot].source[p] == reinterpret_cast<std::uintptr_t>(weights[source].quantized[p].data());
                if (same) { selected[requested] = slot; ++matches; }
            }
            require(matches == 1, "VMM reference cannot uniquely resolve the original source-qualified request");
            for (std::size_t prior = 0; prior != requested; ++prior)
                require(ids[prior] == ids[requested] || selected[prior] != selected[requested],
                    "VMM reference falsely aliases two distinct original expert requests");
        }
        Weights reference; reference.types = weights[group].types; reference.segment = weights[source].segment;
        reference.segment.num_experts = value.capacity; reference.segment.pooled_model_experts = 0;
        std::ifstream file(path, std::ios::binary);
        require(bool(file), "Cannot open independent source bytes for matched-geometry CUDA reference");
        for (int p = 0; p != 3; ++p) {
            const auto stride = weights[group].quantized[p].size() / count;
            reference.quantized[p].resize(stride * static_cast<std::size_t>(value.capacity), 0);
            for (int slot = 0; slot != value.capacity; ++slot) if (live[slot].expert >= 0) {
                const int owner = source_of(live[slot]);
                file.seekg(offsets[owner][p] + static_cast<std::int64_t>(stride) * live[slot].expert);
                file.read(reinterpret_cast<char*>(reference.quantized[p].data() + static_cast<std::size_t>(slot) * stride), stride);
                require(bool(file), "Cannot reconstruct a source-qualified ordinary CUDA expert slot from FILE");
            }
        }
        reference.segment.gate_data = reference.quantized[0].data(); reference.segment.gate_bytes = reference.quantized[0].size();
        reference.segment.up_data = reference.quantized[1].data(); reference.segment.up_bytes = reference.quantized[1].size();
        reference.segment.down_data = reference.quantized[2].data(); reference.segment.down_bytes = reference.quantized[2].size();
        const bool scalar_channels = width <= TSG_PRECISION_DECODE_COLUMNS;
        std::fprintf(stderr, "[vmm-cache-fixture] matched-reference source=%d group=%d rows=%d capacity=%d resident=%zu "
            "scalar_channels=%d independently_mapped_ids=%zu file_source_rows=%zu empty_slots_zero=1 "
            "scope=matched_standard_cuda_operator_geometry_not_model_quality state=running\n",
            source, group, width, value.capacity, resident, scalar_channels ? 1 : 0, selected.size(), resident * 3);
        std::fflush(stderr);
        FullResidentGraph ordinary(reference, width, scalar_channels);
        return ordinary.compute_batch(input, selected, routes);
    };
    // -1 injects the real wide-owner ACK callback, not a native fault hook.
    const std::vector<int> fault_stages = pressure_mode ? std::vector<int>{0, -1, 34, 35} : std::vector<int>{0, 33};
    for (int fault : fault_stages) {
        auto ledger = std::make_unique<VmmCacheLedger>();
        std::uint64_t token = 0;
        struct ScopeCleanup {
            std::unique_ptr<VmmCacheLedger>& ledger; std::uint64_t& token;
            ~ScopeCleanup() {
                tsg::host_moe_expert_cache_test_fail_next(0);
                if (!ledger) return;
                ledger->fail_device_ack = ledger->fail_host_ack = false;
                std::int64_t released = 0;
                if (token) {
                    (void)TSGgml_TrimHostMoeExpertCacheScope(token, 0, &released);
                    (void)TSGgml_CloseHostMoeVmmWeights(token);
                    if (TSGgml_EndHostMoeExpertCacheScope(token, &released) == 1) token = 0;
                }
                if (ledger->attached && tsg::SharedCacheCharge::detach(ledger.get())) ledger->attached = false;
                // An unrecoverable native cleanup failure must not destroy the
                // callback context retained by its original native tickets.
                if (token || ledger->attached) (void)ledger.release();
            }
        } scope_cleanup{ledger, token};
        std::int64_t host_plan = 0, released = 0;
        require(TSGgml_HostMoeVmmWeightsHostPlan(1, 0, 0, 8192, &host_plan) == 1 && host_plan > 0,
            "Cannot plan durable VMM cache Host metadata");
        ledger->capacity = std::size_t(256) << 20;
        ledger->host_capacity = static_cast<std::size_t>(host_plan) + (std::size_t(32) << 20);
        require(ledger->attach(), "Cannot attach actual checked Host/device cache ledger");
        require(TSGgml_BeginHostMoeExpertCacheScopeWithDecodeRows(ledger->capacity, rows, 3, 1, &token) == 1
            && TSGgml_ConfigureHostMoeVmmWeights(token, 1, 0, 0, 8192) == 1,
            "Cannot configure the empty, charged VMM cache arena");
        auto empty = stats();
        require(empty[0] == 1 && empty[1] == 1 && empty[2] == 0 && empty[3] == host_plan
            && empty[4] == 2 && empty[5] == 0 && empty[6] == 0 && empty[7] == 0 && empty[8] == 0,
            "VMM empty arena certificate differs from its admitted metadata owner");
        std::vector<tsg::HostMoePhaseGroupMinimum> minima;
        std::array<tsg::HostMoeCacheTestPlan, 3> wide_minimum_plans{};
        std::size_t budget = 17;
        for (int g = 0; g != 3; ++g) {
            tsg::HostMoeCacheTestPlan narrow{}, wide{};
            require(tsg::host_moe_expert_cache_test_split_plan(weights[g].segment, top, 1, 8, g_backend, narrow)
                && tsg::host_moe_expert_cache_test_split_plan(weights[g].segment, count, 1, rows, g_backend, wide)
                && narrow.weight_physical_bytes >= narrow.weight_bytes && wide.weight_physical_bytes >= wide.weight_bytes
                && narrow.narrow_rows == 8 && wide.narrow_rows == 8 && wide.wide_rows == rows,
                "Exact VMM split planning confused logical leaves with rounded physical charge");
            const auto stride = (weights[g].segment.gate_bytes + weights[g].segment.up_bytes + weights[g].segment.down_bytes) / count;
            tsg::HostMoePhaseGroupMinimum minimum;
            minimum.key = tsg::host_moe_expert_group_key(weights[g].segment);
            minimum.narrow_bytes = total(narrow); minimum.wide_bytes = total(wide);
            minimum.weight = static_cast<std::size_t>(group_layers[g]) * stride;
            minimum.decode_rows = 1; minimum.prefill_rows = rows;
            require(minimum.key.values[7] == pool_extent && minimum.weight > 0,
                "VMM phase proof confused common pool extent with quantization traffic");
            minima.push_back(minimum);
            wide_minimum_plans[g] = wide;
            budget += total(wide);
        }
        const tsg::HostMoePhaseQuotaPlan plan(minima);
        require(budget >= plan.minimum() && budget < ledger->capacity,
            "VMM lifecycle geometry lacks an affordable fixed physical ceiling");
        require(plan.size() == 3, "VMM fixture does not describe exactly three quantization groups");
        for (int g = 0; g != 3; ++g) {
            int decode = 0, prefill = 0;
            require(plan.row_limits(minima[g].key, 1, rows, decode, prefill) && decode == 1 && prefill == rows,
                "VMM phase proof lost its explicit scalar/wide row certificate");
            for (unsigned axis = 3; axis != 8; ++axis)
                require(minima[g].key.values[axis] == minima[0].key.values[axis],
                    "VMM phase proof has nonuniform trunk geometry or pooled identity");
            std::fprintf(stderr, "[vmm-cache-fixture] phase-proof group=%d types=%d,%d,%d pool_extent=%d "
                "loaded_layers=%d traffic_layers=%d traffic_weight=%zu decode=%d prefill=%d narrow=%zu wide=%zu groups=%zu\n",
                g, minima[g].key.values[0], minima[g].key.values[1], minima[g].key.values[2], pool_extent,
                loaded_layers, group_layers[g], minima[g].weight, decode, prefill,
                minima[g].narrow_bytes, minima[g].wide_bytes, plan.size());
        }
        std::fflush(stderr);
        require(TSGgml_UpdateHostMoeExpertCacheScope(token, budget, &released) == 1 && released == 0,
            "Empty VMM scope could not take its exact fixed fixture ceiling");
        ledger->capacity = budget;
        for (int s = 0; s != 4; ++s) for (int p = 0; p != 3; ++p)
            require(TSGgml_RegisterHostFileSource(weights[s].quantized[p].data(), weights[s].quantized[p].size(),
                path.c_str(), offsets[s][p]) == 1, "Cannot register an exact VMM cache immutable source extent");
        const auto before_invalid_reserves = ledger->reserves;
        auto per_group_pool = minima, uniform_wrong_pool = minima;
        for (int g = 0; g != 3; ++g) {
            per_group_pool[g].key.values[7] = group_layers[g] * count;
            uniform_wrong_pool[g].key.values[7] = group_layers[0] * count;
        }
        require(!tsg::host_moe_expert_cache_install_phase_plan(per_group_pool, rows, 1, g_backend)
            && !tsg::host_moe_expert_cache_install_phase_plan(uniform_wrong_pool, rows, 1, g_backend)
            && ledger->reserves == before_invalid_reserves && stats()[5] == 0 && Stats().reserved == 0,
            "Invalid per-group or uniformly wrong VMM pool installed or allocated before rejection");
        require(tsg::host_moe_expert_cache_install_phase_plan(minima, rows, 1, g_backend),
            "Cannot install the complete three-group VMM phase proof");
        std::array<Snapshot, 3> before{};
        std::array<std::vector<Slot>, 3> old_slots;
        if (pressure_mode) {
            // The callback ceiling stays at the original fixed scope ceiling.
            // Public admission and one-off trim targets are tested separately.
            const auto original_ceiling = ledger->capacity;
            const auto minimum = plan.minimum();
            require(minimum > 0 && minimum < budget, "Pressure fixture has no distinct admitted nonzero targets");
            const auto capture = [&](int groups) {
                for (int g = 0; g != groups; ++g) {
                    before[g] = snapshot(g); old_slots[g] = slots(g, before[g]);
                    verify_payload(g, before[g], old_slots[g]);
                }
            };
            const auto preserved = [&](int groups, bool all_wide_retired, bool poisoned_scope = false) {
                for (int g = 0; g != groups; ++g) {
                    Snapshot after{};
                    if (poisoned_scope) require(tsg::host_moe_expert_cache_test_lifecycle_snapshot(weights[g].segment, g_backend, after)
                        && after.vmm && after.vmm_table_identity, "Poisoned pressure lost its retained donor metadata");
                    else after = snapshot(g);
                    const auto live = slots(g, after);
                    require(after.entry_identity == before[g].entry_identity
                        && after.vmm_table_identity == before[g].vmm_table_identity
                        && after.narrow_buffer == before[g].narrow_buffer && after.narrow_charge == before[g].narrow_charge
                        && after.narrow_bytes == before[g].narrow_bytes
                        && after.narrow_physical_bytes == before[g].narrow_physical_bytes
                        && after.capacity <= before[g].capacity && after.weight_physical_bytes <= before[g].weight_physical_bytes,
                        "Pressure replaced a retained table/narrow owner or grew a projection");
                    for (int p = 0; p != 3; ++p) require(after.weight_data[p] == before[g].weight_data[p],
                        "Pressure changed a retained projection VA");
                    for (int w = 0; w != 8; ++w) require(after.width_graph_uid[w]
                        && after.width_graph_uid[w] != before[g].width_graph_uid[w],
                        "Pressure kept a stale narrow CUDA graph UID");
                    if (all_wide_retired) require(after.wide_rows == 0 && after.wide_buffer == 0
                        && after.wide_bytes == 0 && after.wide_graph_uid == 0,
                        "Pressure left an executable retired wide view or physical owner");
                    require(after.resident == (std::min)(before[g].resident, after.capacity),
                        "Pressure discarded an affordable hot source or invented a resident tag");
                    std::uint64_t youngest_discard = 0, oldest_survivor = (std::numeric_limits<std::uint64_t>::max)();
                    for (const auto& old : old_slots[g]) if (old.expert >= 0) {
                        const auto found = std::find_if(live.begin(), live.end(), [&](const Slot& now) { return same_key(old, now); });
                        if (found != live.end()) {
                            require(found->last_used == old.last_used, "Pressure changed an untouched source-qualified LRU timestamp");
                            oldest_survivor = (std::min)(oldest_survivor, old.last_used);
                        } else youngest_discard = (std::max)(youngest_discard, old.last_used);
                    }
                    require(oldest_survivor >= youngest_discard, "Pressure discarded a newer source-qualified hot key");
                    // A poisoned scope explicitly forbids the successful-read
                    // hook and graph proof. Its ownership/tags remain observable;
                    // only the healthy success arm claims raw payload parity.
                    if (!poisoned_scope) verify_payload(g, after, live);
                }
            };
            const auto owned_at_target = [&](std::size_t target) {
                const auto owner = stats(); const auto cache = Stats();
                require(owner[5] == 3 && owner[6] == cache.reserved && owner[6] == static_cast<std::int64_t>(ledger->committed)
                    && static_cast<std::size_t>(cache.reserved) <= target && !owner[7] && !owner[8]
                    && ledger->capacity == original_ceiling,
                    "Pressure actual owned payload differs from target, native accounting or fixed callback ceiling");
                ledger->check();
            };
            if (fault == 35) {
                require(TSGgml_UpdateHostMoeExpertCacheScope(token, minimum, &released) == 1 && released == 0,
                    "Cold pressure fixture cannot install its derived phase minimum");
                call(0, rows, 0); call(1, 1, 0, true);
                capture(2);
                const auto owned_before = Stats().reserved;
                const auto incoming = total(wide_minimum_plans[2]);
                require(stats()[5] == 2 && before[0].wide_buffer != 0 && before[1].wide_buffer == 0
                    && static_cast<std::size_t>(owned_before) + incoming > minimum
                    && minima[0].narrow_bytes + minima[1].narrow_bytes + incoming <= minimum,
                    "Cold pressure setup does not require real inactive-peer retirement for the mandatory wide floor");
                const auto fired = tsg::host_moe_expert_cache_test_vmm_mutation_failures();
                const auto reserves = ledger->reserves; const auto submissions = ExecutionStats().submissions;
                tsg::host_moe_expert_cache_test_fail_next(35);
                bool failed = false; try { call(2, rows, 0); } catch (const std::runtime_error&) { failed = true; }
                require(failed && tsg::host_moe_expert_cache_test_vmm_mutation_failures() == fired + 1
                    && ledger->reserves == reserves && ExecutionStats().submissions == submissions
                    && Stats().reserved < owned_before && stats()[5] == 2,
                    "Cold admission refusal did not follow real pressure or retried/allocated/submitted after mutation");
                // The global scope is poisoned. Its retained donor adapters can
                // still be individually healthy; do not require fictitious
                // per-entry failure bits or a third physical table.
                preserved(2, true, true);
                const auto retry_reserves = ledger->reserves;
                bool refused = false; try { call(2, rows, 0); } catch (const std::runtime_error&) { refused = true; }
                require(refused && ledger->reserves == retry_reserves && ExecutionStats().submissions == submissions
                    && tsg::host_moe_expert_cache_test_vmm_mutation_failures() == fired + 1,
                    "Poisoned same-call pressure refusal resumed allocation or helper execution");
                std::fprintf(stderr, "[vmm-pressure-fixture] fault=35 actual_retirement=1 prephysical_candidate_refusal=1 "
                    "no_reserve=1 no_helper=1 sentinel=1 no_smaller_retry=1 retained_donors=2\n"); std::fflush(stderr);
            } else {
                call(0, rows, 0); call(3, rows, 0); call(0, rows, 16);
                for (int g : {1, 2}) { call(g, rows, 0); call(g, rows, 16); }
                capture(3);
                const auto owned_before = Stats().reserved;
                require(static_cast<std::size_t>(owned_before) > minimum,
                    "Warm pressure fixture has no reducible owned payload above the actual phase minimum");
                const auto update_target = minimum + (static_cast<std::size_t>(owned_before) - minimum) / 2;
                require(update_target >= minimum && update_target < static_cast<std::size_t>(owned_before),
                    "Pressure update target is not derived inside the affordable range");
                const auto reserves = ledger->reserves;
                if (fault == -1) {
                    const auto held = ledger->held(4); const auto submissions = ExecutionStats().submissions;
                    require(before[0].wide_buffer && before[0].wide_charge && before[0].wide_physical_bytes,
                        "Wide ACK fault requires an actual first-group wide owner");
                    ledger->fail_device_ack = true;
                    require(TSGgml_UpdateHostMoeExpertCacheScope(token, update_target, &released) == 0
                        && ledger->reserves == reserves && ledger->held(4) == held && ledger->unacknowledged(4) == 1
                        && Stats().budget == static_cast<std::int64_t>(budget) && stats()[5] == 3,
                        "Pressure wide ACK failure did not retain its original ticket and failed Update admission");
                    Snapshot dirty{};
                    require(tsg::host_moe_expert_cache_test_lifecycle_snapshot(weights[0].segment, g_backend, dirty)
                        && dirty.entry_identity == before[0].entry_identity
                        && dirty.vmm_table_identity == before[0].vmm_table_identity
                        && dirty.narrow_buffer == before[0].narrow_buffer && dirty.narrow_charge == before[0].narrow_charge
                        && dirty.wide_rows == 0 && dirty.wide_buffer == 0 && dirty.wide_physical_bytes == 0
                        && dirty.wide_graph_uid == 0 && dirty.wide_charge == before[0].wide_charge
                        && dirty.wide_bytes == before[0].wide_bytes,
                        "Failed wide ACK did not withdraw graph/physical storage before retaining its exact owner shell");
                    for (int p = 0; p != 3; ++p) require(dirty.weight_data[p] == before[0].weight_data[p],
                        "Failed wide ACK replaced a retained projection VA");
                    bool refused = false; try { call(0, 1, 0, true); } catch (const std::runtime_error&) { refused = true; }
                    require(refused && ledger->reserves == reserves && ExecutionStats().submissions == submissions
                        && !tsg::SharedCacheCharge::detach(ledger.get()),
                        "Failed wide ACK resumed execution or detached its retained callback owner");
                    std::fprintf(stderr, "[vmm-pressure-fixture] fault=wide_ack actual_wide_free=1 wide_view_withdrawn=1 "
                        "same_charge_retained=1 unacknowledged_kind4=1 failed_update_kept_budget=1 no_helper=1 sentinel=1\n");
                    std::fflush(stderr);
                } else if (fault == 34) {
                    const auto fired = tsg::host_moe_expert_cache_test_vmm_mutation_failures();
                    const auto submissions = ExecutionStats().submissions;
                    tsg::host_moe_expert_cache_test_fail_next(34);
                    require(TSGgml_UpdateHostMoeExpertCacheScope(token, update_target, &released) == 0
                        && tsg::host_moe_expert_cache_test_vmm_mutation_failures() == fired + 1
                        && ledger->reserves == reserves && Stats().reserved < owned_before
                        && Stats().budget == static_cast<std::int64_t>(budget) && stats()[5] == 3,
                        "Pressure retirement fault did not fire after actual free or changed failed Update admission");
                    bool refused = false; try { call(0, 1, 0, true); } catch (const std::runtime_error&) { refused = true; }
                    require(refused && ledger->reserves == reserves && ExecutionStats().submissions == submissions,
                        "Post-retirement pressure failure published output, submitted a helper or resumed allocation");
                    std::fprintf(stderr, "[vmm-pressure-fixture] fault=34 actual_retirement=1 failed_update_kept_budget=1 "
                        "no_reserve=1 no_helper=1 sentinel=1 retained_tables=3\n"); std::fflush(stderr);
                } else {
                    require(TSGgml_UpdateHostMoeExpertCacheScope(token, update_target, &released) == 1
                        && released > 0 && released == owned_before - Stats().reserved && ledger->reserves == reserves
                        && Stats().budget == static_cast<std::int64_t>(update_target),
                        "Nonzero Update did not shrink before publishing its admission target or reserved new payload");
                    preserved(3, true); owned_at_target(update_target);
                    for (int g = 0; g != 3; ++g) for (int width = 1; width <= 8; ++width) call(g, width, 0, true);
                    capture(3);
                    const auto trim_before = Stats().reserved; const auto trim_reserves = ledger->reserves;
                    require(static_cast<std::size_t>(trim_before) > minimum
                        && TSGgml_TrimHostMoeExpertCacheScope(token, minimum, &released) == 1 && released > 0
                        && released == trim_before - Stats().reserved && ledger->reserves == trim_reserves
                        && Stats().budget == static_cast<std::int64_t>(update_target),
                        "Nonzero Trim did not meet its independent actual target while preserving admission budget");
                    preserved(3, true); owned_at_target(minimum);
                    for (int g = 0; g != 3; ++g) for (int width = 1; width <= 8; ++width) call(g, width, 0, true);
                    const auto restore_reserves = ledger->reserves;
                    require(TSGgml_UpdateHostMoeExpertCacheScope(token, minimum, &released) == 1 && released == 0
                        && ledger->reserves == restore_reserves && Stats().budget == static_cast<std::int64_t>(minimum),
                        "Future-wide admission was not narrowed to the already met actual trim target");
                    std::array<Snapshot, 3> future_before{};
                    for (int g = 0; g != 3; ++g) future_before[g] = snapshot(g);
                    const auto future_owned = static_cast<std::size_t>(Stats().reserved);
                    int restore_group = -1;
                    for (int g = 0; g != 3; ++g) if (future_before[g].capacity < rows * top) {
                        tsg::HostMoeCacheTestPlan required{};
                        require(tsg::host_moe_expert_cache_test_split_plan(weights[g].segment, rows * top, 1, rows, g_backend, required)
                            && required.narrow_bytes == future_before[g].narrow_bytes
                            && required.weight_physical_bytes >= future_before[g].weight_physical_bytes,
                            "Future-wide required-union plan is incompatible with its retained narrow owner");
                        const auto incoming = required.weight_physical_bytes - future_before[g].weight_physical_bytes + required.wide_bytes;
                        if (future_owned <= minimum && incoming > minimum - future_owned) { restore_group = g; break; }
                    }
                    require(restore_group >= 0, "Pressure did not leave a real active-owner-excluded reclamation opportunity");
                    const auto retained = future_before[restore_group];
                    require(retained.wide_buffer == 0, "Future-wide fixture still owns its old optional graph");
                    call(restore_group, rows, 0);
                    const auto restored = snapshot(restore_group);
                    require(restored.capacity >= rows * top && restored.capacity > retained.capacity
                        && restored.wide_rows == rows && restored.wide_buffer && restored.wide_graph_uid
                        && restored.wide_generation > retained.wide_generation
                        && restored.entry_identity == retained.entry_identity && restored.vmm_table_identity == retained.vmm_table_identity
                        && restored.narrow_buffer == retained.narrow_buffer && restored.narrow_charge == retained.narrow_charge,
                        "Required union growth did not recreate an admitted future-wide graph on the retained owner");
                    for (int p = 0; p != 3; ++p) require(restored.weight_data[p] == retained.weight_data[p],
                        "Future wide restoration replaced a stable projection VA");
                    for (int g = 0; g != 3; ++g) {
                        const auto peer = snapshot(g);
                        require(peer.vmm_table_identity == future_before[g].vmm_table_identity
                            && peer.narrow_buffer == future_before[g].narrow_buffer,
                            "Active-owner reclamation evicted an inactive reusable table/narrow owner");
                        for (int p = 0; p != 3; ++p) require(peer.weight_data[p] == future_before[g].weight_data[p],
                            "Active-owner reclamation changed an inactive projection VA");
                        verify_payload(g, peer, slots(g, peer));
                    }
                    call(restore_group, 9, 0); require(snapshot(restore_group).wide_rows == 9,
                        "Future exact-width9 graph was not replanned after wide16 restoration");
                    call(restore_group, 1, 0, true); owned_at_target(minimum);
                    std::fprintf(stderr, "[vmm-pressure-fixture] fault=0 update_target=%zu trim_target=%zu "
                        "callback_ceiling=%zu all_three_retained=1 all_narrow_widths=1..8 future_required_growth_group=%d "
                        "future_admission_target=%zu active_owner_excluded=1 future_wide_widths=16,9 final_n1_exact=1\n",
                        update_target, minimum, original_ceiling, restore_group, minimum);
                    std::fflush(stderr);
                }
            }
            if (fault > 0) {
                ledger->check(); ledger->fail_device_ack = true;
                require(TSGgml_TrimHostMoeExpertCacheScope(token, 0, &released) == 0
                    && ledger->unacknowledged(4) == 1 && ledger->held(4) > 0
                    && !tsg::SharedCacheCharge::detach(ledger.get()),
                    "Pressure failure cleanup discarded the exact retained Device acknowledgment token");
            }
        } else {
        std::fprintf(stderr, "[vmm-cache-fixture] warm fault=%d fixed_ceiling=%zu\n", fault, budget); std::fflush(stderr);
        call(0, rows, 0); call(3, rows, 0); call(0, rows, 16);
        for (int g : {1, 2}) { call(g, rows, 0); call(g, rows, 16); }
        for (int g = 0; g != 3; ++g) {
            before[g] = snapshot(g); old_slots[g] = slots(g, before[g]); verify_payload(g, before[g], old_slots[g]);
        }
        bool saw_alternate = false; int protected_first = -1, protected_source = -1;
        std::uint64_t protected_age = (std::numeric_limits<std::uint64_t>::max)();
        for (const auto& slot : old_slots[0]) if (slot.expert >= 0) {
            const int source = source_of(slot); saw_alternate |= source == 3;
            if (slot.expert == count - 1 || slot.last_used >= protected_age) continue;
            const bool next_present = std::any_of(old_slots[0].begin(), old_slots[0].end(), [&](const Slot& next) {
                return next.expert == slot.expert + 1 && next.source[0] == slot.source[0]
                    && next.source[1] == slot.source[1] && next.source[2] == slot.source[2];
            });
            if (next_present) { protected_first = slot.expert; protected_source = source; protected_age = slot.last_used; }
        }
        require(saw_alternate && protected_first >= 0, "VMM warmup did not retain distinct source-qualified hot slots");
        const auto mutations_before = tsg::host_moe_expert_cache_test_vmm_mutation_failures();
        if (fault) tsg::host_moe_expert_cache_test_fail_next(fault);
        bool failed = false;
        try { call(protected_source, 1, protected_first, true); } catch (const std::runtime_error&) { failed = true; }
        require(failed == (fault != 0), "Post-donor-shrink VMM failure was retried or silently fell back");
        ledger->check();
        if (!fault) {
            require(tsg::host_moe_expert_cache_test_vmm_mutation_failures() == mutations_before,
                "Unarmed VMM mutation hook fired");
            bool shrunk = false, grew = false, physical_shrink = false, physical_grow = false;
            for (int g = 0; g != 3; ++g) {
                const auto after = snapshot(g); const auto live = slots(g, after);
                shrunk |= after.capacity < before[g].capacity; grew |= after.capacity > before[g].capacity;
                physical_shrink |= after.weight_physical_bytes < before[g].weight_physical_bytes;
                physical_grow |= after.weight_physical_bytes > before[g].weight_physical_bytes;
                require(after.entry_identity == before[g].entry_identity && after.vmm_table_identity == before[g].vmm_table_identity
                    && after.narrow_buffer == before[g].narrow_buffer && after.narrow_charge == before[g].narrow_charge,
                    "VMM reshape replaced its stable table or retained narrow scratch owner");
                for (int p = 0; p != 3; ++p) require(after.weight_data[p] == before[g].weight_data[p],
                    "VMM capacity change replaced a stable projection VA");
                if (after.capacity != before[g].capacity) {
                    require(after.vmm_mutation_generation > before[g].vmm_mutation_generation,
                        "VMM capacity mutation did not advance its owner generation");
                    for (int w = 0; w != 8; ++w) require(after.width_graph_uid[w]
                        && after.width_graph_uid[w] != before[g].width_graph_uid[w],
                        "VMM resize retained a stale narrow CUDA graph UID");
                    if (after.wide_buffer && before[g].wide_buffer) require(after.wide_graph_uid != before[g].wide_graph_uid,
                        "VMM resize retained a stale wide CUDA graph UID");
                }
                require(after.resident == (std::min)(before[g].resident, after.capacity),
                    "VMM reshape discarded an affordable hot expert or invented a valid slot");
                std::uint64_t youngest_discard = 0, oldest_survivor = (std::numeric_limits<std::uint64_t>::max)();
                for (const auto& old : old_slots[g]) if (old.expert >= 0) {
                    const auto found = std::find_if(live.begin(), live.end(), [&](const Slot& now) { return same_key(old, now); });
                    const bool protected_slot = g == 0 && source_of(old) == protected_source
                        && (old.expert == protected_first || old.expert == protected_first + 1);
                    if (protected_slot) require(found != live.end(), "Current source-qualified union was evicted during donor shrink");
                    if (found != live.end()) {
                        require(protected_slot ? found->last_used > old.last_used : found->last_used == old.last_used,
                            "VMM compaction reset an unrelated LRU timestamp");
                        if (!protected_slot) oldest_survivor = (std::min)(oldest_survivor, old.last_used);
                    } else if (!protected_slot) youngest_discard = (std::max)(youngest_discard, old.last_used);
                }
                require(oldest_survivor >= youngest_discard,
                    "VMM compaction discarded a newer unprotected source before an older survivor");
                verify_payload(g, after, live);
            }
            const auto frequent = snapshot(1);
            require(shrunk && grew && physical_shrink && physical_grow && frequent.capacity > before[1].capacity
                && frequent.weight_physical_bytes > before[1].weight_physical_bytes,
                "Actual rounded geometry did not shrink donors then grow the deficient G4 table");
            const auto owner = stats();
            require(owner[5] == 3 && owner[6] > 0 && owner[7] == 0 && owner[8] == 0
                && owner[9] >= 1 && owner[10] >= 1 && owner[11] == 0 && owner[6] == Stats().reserved
                && owner[6] == static_cast<std::int64_t>(ledger->committed),
                "Completed VMM reshape lacks actual activity or disagrees with held shared ownership");
            for (int g = 0; g != 3; ++g) {
                for (int width = 1; width <= 8; ++width) call(g, width, 0, true);
                call(g, 9, 0); call(g, rows, 16); call(g, 1, 0, true);
            }
            ledger->check();
        } else {
            require(tsg::host_moe_expert_cache_test_vmm_mutation_failures() == mutations_before + 1,
                "VMM postmutation failure was not actually reached after a donor shrink");
            const auto dirty = stats();
            require(dirty[5] > 0 && dirty[9] >= 1 && dirty[11] >= 1,
                "VMM mutation failure discarded its owned table/failed activity evidence");
            const auto reserves_before_retry = ledger->reserves;
            bool refused_again = false;
            try { call(protected_source, 1, protected_first, true); }
            catch (const std::runtime_error&) { refused_again = true; }
            require(refused_again && ledger->reserves == reserves_before_retry
                && tsg::host_moe_expert_cache_test_vmm_mutation_failures() == mutations_before + 1,
                "A poisoned postmutation VMM scope resumed allocation/execution");
            ledger->fail_device_ack = true;
            require(TSGgml_TrimHostMoeExpertCacheScope(token, 0, &released) == 0
                && ledger->unacknowledged(4) == 1 && ledger->held(4) > 0,
                "Failed checked Device acknowledgment did not retain its exact original ticket");
            require(!tsg::SharedCacheCharge::detach(ledger.get()),
                "VMM cleanup detached the callback owner with native-held tickets");
        }
        }
        if (!fault) {
            const auto staging_host_tickets = ledger->held(3);
            require(tsg::host_moe_expert_cache_test_file_workspace_bytes() > 0 && staging_host_tickets > 2,
                "VMM staging acknowledgment fixture has no actual Host payload");
            ledger->fail_host_ack = true;
            require(TSGgml_TrimHostMoeExpertCacheScope(token, 0, &released) == 0
                && ledger->unacknowledged(3) == 1 && ledger->held(3) == staging_host_tickets
                && tsg::host_moe_expert_cache_test_file_workspace_bytes() == 0 && stats()[2] == 0,
                "VMM staging did not free its actual buffer before retaining a failed Host acknowledgment");
        }
        require(TSGgml_TrimHostMoeExpertCacheScope(token, 0, &released) == 1,
            "Checked VMM table cleanup retry failed");
        const auto trimmed = stats();
        require(trimmed[1] == 1 && trimmed[2] == 0 && trimmed[4] == 2 && trimmed[5] == 0
            && trimmed[6] == 0 && trimmed[7] == 0 && trimmed[8] == 0 && ledger->held(4) == 0 && ledger->held(3) == 2,
            "Reusable VMM Trim(0) lost durable metadata or retained a payload owner");
        if (!fault) {
            call(0, 1, 0, true);
            const auto reopened = snapshot(0);
            require(reopened.vmm_table_identity != before[0].vmm_table_identity && stats()[4] == 2,
                "Trim(0) did not permit a new table in the same admitted durable arena");
            require(TSGgml_TrimHostMoeExpertCacheScope(token, 0, &released) == 1 && ledger->held(4) == 0,
                "Reused VMM arena did not retire its new table before terminal close");
        }
        ledger->fail_host_ack = true;
        require(TSGgml_CloseHostMoeVmmWeights(token) == 0 && ledger->unacknowledged(3) == 1,
            "Final VMM Host acknowledgment failure did not retain its exact ticket");
        require(TSGgml_CloseHostMoeVmmWeights(token) == 1, "Final VMM Host acknowledgment retry failed");
        const auto closed = stats();
        require(closed[2] == 1 && closed[4] == 0 && closed[5] == 0 && closed[6] == 0
            && closed[7] == 0 && closed[8] == 0 && ledger->held() == 0,
            "Final VMM close claimed success with a live owner or unacknowledged ticket");
        require(TSGgml_EndHostMoeExpertCacheScope(token, &released) == 1, "VMM cache scope end failed"); token = 0;
        ledger->check();
        require(ledger->committed == 0 && ledger->host_committed == 0 && Stats().reserved == 0,
            "VMM cache terminal cleanup retained declared payload credit");
        require(tsg::SharedCacheCharge::detach(ledger.get()), "VMM cache lost its final checked attachment"); ledger->attached = false;
        tsg::host_moe_expert_cache_release(); // Drop registered file handles before unlink.
        tsg::g_backend_compute_failed.store(false, std::memory_order_release);
    }
    tsg::host_moe_expert_cache_release();
    tsg::host_moe_expert_cache_test_vmm_profile(false);
    require(std::remove(path.c_str()) == 0, "VMM cache fixture source remained open after checked retirement");
    file_cleanup.active = false;
    if (pressure_mode) std::fprintf(stderr, "PASS VMM cache pressure exact_f32=%zu mixed_groups=3 distinct_sources=4 "
        "nonzero_update=1 nonzero_trim=1 no_pressure_reserve=1 retained_tables=3 stable_va_narrow=1 refreshed_uids=8 "
        "source_lru_preserved=1 narrow_widths=1..8 future_wide_widths=16,9 final_n1_exact=1 faults=wide_ack,34,35 "
        "checked_retry=1 pending=0 committed=0 tiny_TEST_profile_only=1 model_quality_unqualified=1 CUDA_capture_unobserved=1 "
        "scope=cache_payload_not_total_fixture_or_vendor_memory\n", compared);
    else std::fprintf(stderr, "PASS VMM cache integration exact_f32=%zu mixed_groups=3 distinct_sources=4 donor_shrink=1 recipient_grow=1 "
        "narrow_widths=1..8 wide_widths=9,16 mutation_fault=33 checked_retry=1 pending=0 committed=0; "
        "common_pool_layers=45 traffic_layers=11,28,3 matched_geometry_file_oracle=1 raw_payload_separately_exact=1 "
        "tiny_TEST_profile_only=1 CUDA_capture_unobserved=1 model_quality_unqualified=1 "
        "scope=cache_payload_not_total_fixture_or_vendor_memory\n", compared); std::fflush(stderr);
}
#endif
