// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Actual cache + original CUDA operators at a TEST-only 4/48/top8 geometry.
// FILE-derived same-capacity references; no model/capture/performance claim.
#ifdef TSG_GGML_USE_CUDA
struct CausalCacheLedger {
    struct Ticket { std::uint64_t id = 0; std::size_t bytes = 0; int kind = 0; bool committed = false, refunded = false; };
    std::array<Ticket, 128> tickets{};
    std::size_t device_limit = 0, host_limit = 128u << 20, pending[2] = {}, committed[2] = {};
    std::uint64_t serial = 0, reserves = 0, commits = 0, checked = 0, legacy = 0, refunds = 0;
    std::uint64_t owner_id = 0, failed_id = 0;
    tsg_glm_neighbor::Owner* owner = nullptr;
    bool valid = true, fail_owner_ack = false;
    Ticket* find(std::uint64_t id) { for (auto& t : tickets) if (t.id == id) return &t; return nullptr; }
    static std::uint64_t reserve(void* context, int rank, int kind, std::int64_t bytes) noexcept {
        auto& l = *static_cast<CausalCacheLedger*>(context); ++l.reserves;
        if (rank || (kind != 3 && kind != 4) || bytes <= 0) { l.valid = false; return 0; }
        const int h = kind == 3; const auto n = static_cast<std::size_t>(bytes);
        const auto limit = h ? l.host_limit : l.device_limit;
        if (l.pending[h] + l.committed[h] > limit || n > limit - l.pending[h] - l.committed[h]) return 0;
        for (auto& t : l.tickets) if (!t.id) {
            t = {++l.serial, n, kind}; l.pending[h] += n;
            if (h && l.owner && !l.owner->configured() && !l.owner->storage()
                && n == tsg_glm_neighbor::Owner::host_bytes() && !l.owner_id) l.owner_id = t.id;
            return t.id;
        }
        l.valid = false; return 0;
    }
    static int commit(void* context, std::uint64_t id) noexcept {
        auto& l = *static_cast<CausalCacheLedger*>(context); ++l.commits; auto* t = l.find(id);
        if (!t || t->committed || t->refunded) { l.valid = false; return 0; }
        const int h = t->kind == 3;
        if (l.pending[h] < t->bytes) { l.valid = false; return 0; }
        l.pending[h] -= t->bytes; l.committed[h] += t->bytes; t->committed = true; return 1;
    }
    static int release_checked(void* context, std::uint64_t id) noexcept {
        auto& l = *static_cast<CausalCacheLedger*>(context); ++l.checked; auto* t = l.find(id);
        if (!t) { l.valid = false; return 0; }
        if (id == l.owner_id && l.owner && l.owner->storage()) { l.valid = false; return 0; }
        if (!t->refunded) {
            const int h = t->kind == 3; auto& bytes = t->committed ? l.committed[h] : l.pending[h];
            if (bytes < t->bytes) { l.valid = false; return 0; }
            bytes -= t->bytes; t->refunded = true; ++l.refunds;
            if (id == l.owner_id && l.fail_owner_ack) { l.fail_owner_ack = false; l.failed_id = id; return 0; }
        }
        *t = {}; return 1;
    }
    static void release(void* context, std::uint64_t id) noexcept {
        auto& l = *static_cast<CausalCacheLedger*>(context); ++l.legacy;
        // Ordinary selected-cache/staging owners retain their historical void
        // release path. Only the new metadata owner must always checked-ACK.
        if (id == l.owner_id) l.valid = false;
        (void)release_checked(context, id);
    }
    std::size_t held() const { std::size_t n = 0; for (const auto& t : tickets) n += t.id != 0; return n; }
    void check() const {
        require(valid && !pending[0] && !pending[1] && committed[0] <= device_limit && committed[1] <= host_limit,
            "Causal cache exceeded its actual shared envelope or retained pending accounting");
    }
};

// Independent full-past set matcher. The production State/query/due arrays
// never feed this oracle. Each rank's saved absolute due is expired in place.
struct CausalPast {
    using Row = std::vector<std::int32_t>;
    std::array<std::vector<Row>, 4> rows;
    std::array<std::array<std::uint64_t, 48>, 4> due{};
    struct Prediction { std::array<std::uint64_t, 48> values{}; bool matched = false; };
    Prediction query(int rank, const Row& selected) const {
        Prediction out; const auto& past = rows[rank]; const auto t = past.size();
        std::size_t best = 0, overlap_best = 0, union_best = 1;
        const auto first = t > 64 ? t - 64 : 0;
        for (std::size_t i = first; i + 8 < t; ++i) {
            std::size_t overlap = 0;
            for (auto id : selected) overlap += std::find(past[i].begin(), past[i].end(), id) != past[i].end();
            const auto united = selected.size() + past[i].size() - overlap;
            if (overlap < 2) continue;
            if (!out.matched || overlap * union_best > overlap_best * united
                || (overlap * union_best == overlap_best * united && i > best)) {
                out.matched = true; best = i; overlap_best = overlap; union_best = united;
            }
        }
        if (out.matched) for (std::size_t d = 1; d <= 8; ++d)
            for (auto id : past[best + d]) if (!out.values[static_cast<std::size_t>(id)])
                out.values[static_cast<std::size_t>(id)] = 4 * (t + d) + static_cast<std::size_t>(rank);
        return out;
    }
    std::uint64_t value(int resident_rank, int expert, int current_rank, const Prediction& p) const {
        if (expert < 0) return UINT64_MAX;
        const auto d = resident_rank == current_rank ? p.values[static_cast<std::size_t>(expert)]
            : due[resident_rank][static_cast<std::size_t>(expert)];
        const auto now = 4 * rows[current_rank].size() + static_cast<std::size_t>(current_rank);
        return d > now ? d : UINT64_MAX;
    }
    void commit(int rank, const Row& selected, const Prediction& p) { due[rank] = p.values; rows[rank].push_back(selected); }
    void clear() { for (auto& v : rows) v.clear(); for (auto& v : due) v.fill(0); }
};

void host_moe_causal_route_neighbor_checks() {
    using Owner = tsg_glm_neighbor::Owner;
    using Snapshot = tsg::HostMoeCacheTestSnapshot;
    using Slot = tsg::HostMoeCacheTestSlot;
    constexpr int h = 512, ff = 512, count = 48, top = 8, width = 16, pool = 45 * count;
    tsg::host_moe_expert_cache_release();
    require(tsg_glm_neighbor::option() == 1, "Causal cache fixture flag was not fixed before first use");
    std::array<Weights, 4> weights;
    weights[0].types = {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS};
    weights[1].types = {GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_S, GGML_TYPE_IQ3_XXS};
    weights[2].types = {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ4_XS};
    weights[3].types = weights[0].types;
    std::array<std::array<std::int64_t, 3>, 4> offsets{};
    std::array<tsg_glm_neighbor::Source, 4> sources{};
    tsg_glm_neighbor::Config settings; settings.layers = 4; settings.experts = count; settings.routes = top;
    const auto path = fixture_file_path("glm-causal-route-neighbor-");
    struct Unlink { const std::string& path; ~Unlink() { std::remove(path.c_str()); } } unlink{path};
    for (int rank = 0; rank < 4; ++rank) {
        for (auto type : weights[rank].types) ggml_quantize_init(type);
        weights[rank].fill(.71f + rank * .83f, count, top, h, ff);
        weights[rank].segment.layer = 3 + rank; weights[rank].segment.pooled_model_experts = pool;
        settings.layer_ids[rank] = sources[rank].layer = static_cast<std::uint16_t>(3 + rank);
        for (int p = 0; p < 3; ++p) {
            sources[rank].pointer[p] = weights[rank].quantized[p].data();
            sources[rank].bytes[p] = weights[rank].quantized[p].size();
            sources[rank].stride[p] = sources[rank].bytes[p] / count;
            sources[rank].type[p] = weights[rank].types[p];
        }
    }
    {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        require(bool(file), "Cannot create bounded generated causal source file");
        for (int rank = 0; rank < 4; ++rank) for (int p = 0; p < 3; ++p) {
            file.write("causal-source-generation", 24); offsets[rank][p] = static_cast<std::int64_t>(file.tellp());
            file.write(reinterpret_cast<const char*>(weights[rank].quantized[p].data()), weights[rank].quantized[p].size());
        }
        require(bool(file), "Generated source FILE write failed");
    }
    // Exact source registration pointers are poison, never the reference data.
    for (auto& w : weights) for (auto& bytes : w.quantized) std::fill(bytes.begin(), bytes.end(), 0xa5);
    std::vector<tsg::HostMoePhaseGroupMinimum> minima;
    std::size_t budget = 0;
    for (int g = 0; g < 3; ++g) {
        tsg::HostMoeCacheTestPlan narrow{}, wide{}, fair{};
        require(tsg::host_moe_expert_cache_test_split_plan(weights[g].segment, top, 1, 8, g_backend, narrow)
            && tsg::host_moe_expert_cache_test_split_plan(weights[g].segment, count, 1, width, g_backend, wide)
            && tsg::host_moe_expert_cache_test_split_plan(weights[g].segment, 32, 1, 8, g_backend, fair),
            "Cannot obtain actual three-group cold plans");
        const auto total = [](const tsg::HostMoeCacheTestPlan& p) { return p.weight_bytes + p.narrow_bytes + p.wide_bytes; };
        tsg::HostMoePhaseGroupMinimum m; m.key = tsg::host_moe_expert_group_key(weights[g].segment);
        m.narrow_bytes = total(narrow); m.wide_bytes = total(wide); m.weight = 1; m.decode_rows = 1; m.prefill_rows = width;
        minima.push_back(m); budget += total(fair);
    }
    budget = (std::max)(budget, tsg::HostMoePhaseQuotaPlan(minima).minimum()) + 17;
    require(tsg::HostMoePhaseQuotaPlan(minima).admits(budget), "Actual source fixture lacks affordable phase floors");
    unsigned char model_cookie = 0;
    std::size_t compared_f32 = 0, victim_proofs = 0, replaced = 0, actual_faults = 0;

    struct Scope {
        CausalCacheLedger ledger;
        Owner owner;
        std::uint64_t token = 0;
        bool attached = false;
        ~Scope() noexcept {
            ledger.fail_owner_ack = false;
            if (owner.active_call() || owner.query_pending()) tsg::host_moe_causal_abort(owner);
            std::int64_t released = 0;
            if (!tsg::host_moe_causal_close(owner)
                || (token && TSGgml_EndHostMoeExpertCacheScope(token, &released) != 1)) {
                std::fputs("FATAL causal fixture retains callback/model ownership after cleanup refusal\n", stderr); std::abort();
            }
            try { tsg::host_moe_expert_cache_release(); } catch (...) { std::abort(); }
            if (attached && !tsg::SharedCacheCharge::detach(&ledger)) std::abort();
        }
    };
    const auto setup = [&](Scope& s, bool checked) {
        s.owner.request(true); s.ledger.owner = &s.owner; s.ledger.device_limit = budget;
        s.attached = tsg::SharedCacheCharge::attach(&s.ledger, CausalCacheLedger::reserve, CausalCacheLedger::commit,
            CausalCacheLedger::release, false, true, checked ? CausalCacheLedger::release_checked : nullptr);
        require(s.attached && TSGgml_BeginHostMoeExpertCacheScopeWithDecodeRows(budget, width, 3, 1, &s.token) == 1,
            "Cannot create checked cache setup scope");
        for (int rank = 0; rank < 4; ++rank) for (int p = 0; p < 3; ++p)
            require(TSGgml_RegisterHostFileSource(sources[rank].pointer[p], sources[rank].bytes[p], path.c_str(), offsets[rank][p]) == 1,
                "Cannot register all twelve exact immutable source extents");
        require(tsg::host_moe_expert_cache_install_phase_plan(minima, width, 1, g_backend), "Cannot install tiny three-group phase proof");
    };
    // Wrapper binds a child cookie before Owner's early validation. These cases
    // must detach that actual bound cookie even though Owner.scope() is zero.
    for (int fault = 0; fault < 3; ++fault) {
        Scope s; setup(s, fault != 1); auto wrong = sources;
        if (fault == 0) wrong[3] = wrong[0];
        const auto bytes = Owner::host_bytes() - (fault == 2 ? 1 : 0);
        require(!tsg::host_moe_causal_test_configure(s.owner, &model_cookie, s.token, bytes,
            settings, wrong.data(), g_backend), "Malformed bound-cookie setup succeeded");
        require(s.owner.scope() == 0 && s.owner.release_ready() && s.ledger.reserves == 0
            && !s.ledger.held() && Stats().reserved == 0, "Early setup refusal adopted payload/accounting ownership");
        std::int64_t released = 0;
        require(TSGgml_EndHostMoeExpertCacheScope(s.token, &released) == 0,
            "Unclosed failed-setup child cookie was silently discarded");
        require(tsg::host_moe_causal_close(s.owner)
            && TSGgml_EndHostMoeExpertCacheScope(s.token, &released) == 1 && released == 0,
            "Checked failed-setup close did not detach the bound child token"); s.token = 0;
    }

    // A source-qualified identity is independent of the borrowed source bytes.
    const auto source_of = [&](const Slot& slot) {
        int found = -1;
        for (int rank = 0; rank < 4; ++rank) {
            bool same = true; for (int p = 0; p < 3; ++p)
                same &= slot.source[p] == reinterpret_cast<std::uintptr_t>(sources[rank].pointer[p]);
            if (same) { require(found == -1, "Ambiguous source generation in complete slot vector"); found = rank; }
        }
        require(found >= 0, "Resident key refers to an unenrolled source generation"); return found;
    };
    const auto snapshot = [&](int rank, Snapshot& out, std::vector<Slot>& slots) {
        if (!tsg::host_moe_expert_cache_test_lifecycle_snapshot(weights[rank].segment, g_backend, out)) return false;
        require(!out.vmm && out.capacity >= top && out.capacity <= pool && out.weight_buffer && out.narrow_buffer,
            "Actual causal table lost ordinary split-buffer ownership");
        slots.resize(out.capacity);
        require(tsg::host_moe_expert_cache_test_copy_slot_metadata(weights[rank].segment, g_backend, slots.data(), slots.size()),
            "Cannot inspect complete source-qualified slots"); return true;
    };
    const auto reference = [&](int rank, int rows, const std::vector<float>& x,
        const std::vector<std::int32_t>& ids, const std::vector<float>& routes) {
        Snapshot snap{}; std::vector<Slot> slots;
        require(snapshot(rank, snap, slots), "Successful helper published no table");
        Weights packed; packed.types = weights[rank].types; packed.segment = weights[rank].segment;
        packed.segment.num_experts = snap.capacity; packed.segment.pooled_model_experts = 0;
        std::ifstream file(path, std::ios::binary); require(bool(file), "Cannot open FILE oracle");
        for (int p = 0; p < 3; ++p) {
            const auto stride = sources[rank].stride[p]; packed.quantized[p].resize(stride * snap.capacity, 0);
            std::vector<unsigned char> actual(snap.weight_part_bytes[p]), raw(stride);
            require(tsg::host_moe_expert_cache_test_copy_weight_payload(weights[rank].segment, g_backend, p, actual.data(), actual.size()),
                "Cannot independently verify actual CUDA weight bytes");
            for (int slot = 0; slot < snap.capacity; ++slot) if (slots[slot].expert >= 0) {
                const int r = source_of(slots[slot]);
                require(slots[slot].expert < count && weights[r].types == packed.types, "Bad original expert/type in resident slot");
                file.seekg(offsets[r][p] + static_cast<std::int64_t>(stride) * slots[slot].expert);
                file.read(reinterpret_cast<char*>(raw.data()), raw.size());
                require(bool(file) && std::memcmp(actual.data() + stride * slot, raw.data(), stride) == 0,
                    "Actual routed projection bytes differ from immutable source FILE");
                std::memcpy(packed.quantized[p].data() + stride * slot, raw.data(), stride);
            }
        }
        std::vector<std::int32_t> mapped(ids.size(), -1);
        for (std::size_t k = 0; k < ids.size(); ++k) {
            int matches = 0;
            for (int slot = 0; slot < snap.capacity; ++slot) if (slots[slot].expert == ids[k] && source_of(slots[slot]) == rank) {
                mapped[k] = slot; ++matches;
            }
            require(matches == 1, "Original route cannot uniquely resolve exact source + expert after forced admission");
        }
        packed.segment.gate_data = packed.quantized[0].data(); packed.segment.gate_bytes = packed.quantized[0].size();
        packed.segment.up_data = packed.quantized[1].data(); packed.segment.up_bytes = packed.quantized[1].size();
        packed.segment.down_data = packed.quantized[2].data(); packed.segment.down_bytes = packed.quantized[2].size();
        FullResidentGraph graph(packed, rows, rows <= 8);
        return graph.compute_batch(x, mapped, routes);
    };

    Scope s; setup(s, true);
    require(tsg::host_moe_causal_test_configure(s.owner, &model_cookie, s.token, Owner::host_bytes(),
        settings, sources.data(), g_backend), "Real tiny cache owner configuration refused");
    auto* slab = s.owner.storage(); const auto host_ticket = s.ledger.owner_id;
    require(slab && host_ticket && s.ledger.held() == 1 && s.ledger.committed[1] == Owner::host_bytes(),
        "New Host metadata owner is not the exact single checked kind3 ticket");
    CausalPast oracle;
    const auto run = [&](int rank, int rows, const std::vector<std::int32_t>& ids, bool scalar) {
        std::vector<float> x(static_cast<std::size_t>(rows) * h), routes(static_cast<std::size_t>(rows) * top), out(x.size(), -999.0f);
        for (std::size_t k = 0; k < x.size(); ++k) x[k] = std::sin(float(k % h) * .013f + float(k / h) * .071f);
        for (int row = 0; row < rows; ++row) for (int k = 0; k < top; ++k) routes[row * top + k] = (k + 1) / 36.0f;
        Snapshot before{}; std::vector<Slot> old;
        const bool existed = snapshot(rank, before, old);
        CausalPast::Prediction prediction;
        if (scalar) prediction = oracle.query(rank, ids);
        auto hm = weights[rank].segment; hm.seq_len = rows;
        require(tsg::host_moe_cached_experts_with_neighbor(hm, x.data(), ids.data(), routes.data(), out.data(),
            rows > 8 ? "glm5-next bulk expert prefill" : "glm5-next selected expert seam", g_backend, nullptr, &s.owner) == 1,
            "Actual causal selected-expert helper refused");
        ggml_backend_synchronize(g_backend); // Commit only after the original helper's actual drain.
        const auto expected = reference(rank, rows, x, ids, routes);
        require(out.size() == expected.size() && std::memcmp(out.data(), expected.data(), out.size() * sizeof(float)) == 0
            && std::all_of(out.begin(), out.end(), [](float v) { return std::isfinite(v); }),
            "Causal eviction changed FILE-derived same-geometry original CUDA output bits"); compared_f32 += out.size();
        Snapshot after{}; std::vector<Slot> actual; require(snapshot(rank, after, actual), "Missing post-call table");
        if (scalar && existed && before.entry_identity == after.entry_identity && before.capacity == after.capacity) {
            // Simulate forced admission with ALL later hits protected first.
            auto planned = old; std::vector<bool> protect(planned.size(), false); std::vector<int> remapped(top, -1);
            for (int k = 0; k < top; ++k) for (std::size_t slot = 0; slot < planned.size(); ++slot)
                if (planned[slot].expert == ids[k] && source_of(planned[slot]) == rank) { remapped[k] = static_cast<int>(slot); protect[slot] = true; break; }
            for (int k = 0; k < top; ++k) if (remapped[k] < 0) {
                int victim = -1;
                for (std::size_t candidate = 0; candidate < planned.size(); ++candidate) if (!protect[candidate]) {
                    if (victim < 0) { victim = static_cast<int>(candidate); continue; }
                    const auto& a = planned[candidate]; const auto& b = planned[victim];
                    const bool ea = a.expert < 0, eb = b.expert < 0;
                    const auto da = ea ? UINT64_MAX : oracle.value(source_of(a), a.expert, rank, prediction);
                    const auto db = eb ? UINT64_MAX : oracle.value(source_of(b), b.expert, rank, prediction);
                    const bool first = ea != eb ? ea : (!ea && da != db ? da > db
                        : a.last_used < b.last_used || (a.last_used == b.last_used && candidate < static_cast<std::size_t>(victim)));
                    if (first) victim = static_cast<int>(candidate);
                }
                require(victim >= 0, "Independent forced-admission oracle found all slots protected");
                protect[victim] = true; remapped[k] = victim; replaced += planned[victim].expert >= 0;
                planned[victim].expert = ids[k];
                for (int p = 0; p < 3; ++p) planned[victim].source[p] = reinterpret_cast<std::uintptr_t>(sources[rank].pointer[p]);
            }
            auto clock = before.clock;
            for (int slot : remapped) planned[slot].last_used = ++clock;
            require(after.clock == clock, "Original occurrence-order LRU clock changed");
            for (std::size_t slot = 0; slot < planned.size(); ++slot)
                require(actual[slot].expert == planned[slot].expert && actual[slot].last_used == planned[slot].last_used
                    && std::equal(std::begin(actual[slot].source), std::end(actual[slot].source), std::begin(planned[slot].source)),
                    "Actual protected victim/source/final-LRU differs from independent past-only oracle");
            ++victim_proofs;
        }
        if (scalar) {
            require(tsg::host_moe_causal_commit(s.owner, hm.layer), "Actual helper drain could not commit this own-layer row");
            oracle.commit(rank, ids, prediction);
        }
        s.ledger.check();
    };
    require(tsg::host_moe_causal_begin(s.owner, false, 0, width, 44), "Cannot begin PP reset phase");
    std::vector<std::int32_t> scan(width * top);
    for (int row = 0; row < width; ++row) for (int k = 0; k < top; ++k) scan[row * top + k] = (row * top + k) % count;
    for (int rank = 0; rank < 4; ++rank) run(rank, width, scan, false);
    require(tsg::host_moe_causal_finish(s.owner), "PP call could not finish reset-only scope");
    std::array<std::int64_t, 16> state{};
    require(tsg::host_moe_causal_info(s.owner, state.data(), 16) && state[10] == 0 && state[11] == 0 && state[14] == 0,
        "Wide prefill trained causal metadata");
    // Start the policy differential from fair cold narrow tables. Wide phase
    // reclaim is independently exercised above, not assumed to preserve a
    // narrow history through unrelated table replacement/maintenance resets.
    {
        std::int64_t released = 0;
        require(TSGgml_TrimHostMoeExpertCacheScope(s.token, 0, &released) == 1 && Stats().reserved == 0,
            "Cannot establish the exact cold scalar policy seed");
        oracle.clear();
    }
    for (int token = 0; token < 16; ++token) {
        require(tsg::host_moe_causal_begin(s.owner, true, width + token, 1, 44), "Cannot begin exact N1 chain");
        for (int rank = 0; rank < 4; ++rank) {
            std::vector<std::int32_t> ids(top);
            for (int k = 0; k < 4; ++k) { ids[k] = (token % 6) * 8 + k; ids[k + 4] = ((token + 5) % 6) * 8 + k + 4; }
            run(rank, 1, ids, true);
        }
        require(tsg::host_moe_causal_finish(s.owner), "Cannot finish all four actual N1 helper commits");
    }
    require(tsg::host_moe_causal_info(s.owner, state.data(), 16) && state[10] == 16 && state[11] == 64
        && state[14] > 0 && state[14] + state[15] == 64 && victim_proofs >= 32 && replaced > 0,
        "Actual cache did not exercise causal matches, existing-table victims and cumulative completed rows");
    require(s.owner.storage() == slab && s.ledger.owner_id == host_ticket && !s.ledger.pending[0] && !s.ledger.pending[1],
        "Causal hot path replaced the admitted slab/ticket or left pending allocations");

    // Actual production helper fault cuts. Fresh Trim0 guarantees stage2 has
    // real miss uploads. It is maintenance, not a new source generation.
    for (int stage = 1; stage <= 2; ++stage) {
        std::int64_t released = 0; require(TSGgml_TrimHostMoeExpertCacheScope(s.token, 0, &released) == 1, "Cannot cold-clear fault table");
        oracle.clear(); require(tsg::host_moe_causal_reset(s.owner, true), "Cannot recover clean fault scope");
        require(tsg::host_moe_causal_begin(s.owner, true, 100 + stage, 1, 50 + stage), "Cannot begin actual helper fault call");
        const auto before = s.owner.test_failure_count(); const auto calls = Stats().calls; const auto submitted = ExecutionStats().submissions;
        const auto old_info = [&] { std::array<std::int64_t, 16> v{}; require(s.owner.info(v.data(), 16), "Fault Info unavailable"); return v; }();
        require(s.owner.test_fault(stage), "Cannot arm bounded helper fault");
        std::vector<float> x(h, .05f), routes(top, .125f), sentinel(h, -777.0f);
        std::int32_t ids[top] = {0, 1, 2, 3, 4, 5, 6, 7}; auto hm = weights[0].segment;
        const auto reads = tsg::g_test_file_reads;
        bool failed = false;
        try { (void)tsg::host_moe_cached_experts_with_neighbor(hm, x.data(), ids, routes.data(), sentinel.data(),
            "glm5-next selected expert seam", g_backend, nullptr, &s.owner); }
        catch (const std::runtime_error&) { failed = true; }
        ggml_backend_synchronize(g_backend); tsg::host_moe_causal_abort(s.owner);
        require(failed && s.owner.test_failure_count() == before + 1 && Stats().calls == calls
            && ExecutionStats().submissions == submitted && std::all_of(sentinel.begin(), sentinel.end(), [](float v) { return v == -777.0f; }),
            "Actual helper fault did not stop before math/output publication");
        require(stage == 1 ? tsg::g_test_file_reads == reads : tsg::g_test_file_reads > reads,
            "Fault stage did not occur on the intended side of actual FILE uploads");
        std::array<std::int64_t, 16> now{}; require(s.owner.info(now.data(), 16) && now[5] == 1 && !now[9]
            && now[10] == old_info[10] && now[11] == old_info[11] && s.owner.storage() == slab,
            "Failed uncommitted query trained history or lost durable owner");
        require(tsg::host_moe_cached_experts_with_neighbor(hm, x.data(), ids, routes.data(), sentinel.data(),
            "glm5-next selected expert seam", g_backend, nullptr, &s.owner) == 0
            && ExecutionStats().submissions == submitted, "Poisoned model context resumed ordinary helper work");
        ++actual_faults;
    }
    require(tsg::host_moe_causal_reset(s.owner, true), "Explicit reset did not recover actual cache fault"); oracle.clear();
    require(tsg::host_moe_causal_begin(s.owner, true, 0, 1, 99), "Cannot begin recovered exact helper chain");
    for (int rank = 0; rank < 4; ++rank) run(rank, 1, {0, 1, 2, 3, 4, 5, 6, 7}, true);
    require(tsg::host_moe_causal_finish(s.owner), "Recovered complete N1 chain failed");
    // Stage3's actual GLM callback cut is covered by the separate checkpoint.
    // This component deliberately gives it no runtime credit.
    s.ledger.fail_owner_ack = true;
    require(!tsg::host_moe_causal_close(s.owner) && !s.owner.storage() && s.ledger.failed_id == host_ticket,
        "Post-ledger metadata ACK failure did not retain exact original ticket");
    require(!tsg::SharedCacheCharge::detach(&s.ledger), "Failed ACK detached callback lifetime");
    std::int64_t released = 0;
    require(TSGgml_EndHostMoeExpertCacheScope(s.token, &released) == 0,
        "Cache scope ended before its child ticket was acknowledged");
    require(tsg::host_moe_causal_close(s.owner) && s.owner.closed(), "Same-token checked metadata ACK retry failed");
    require(TSGgml_EndHostMoeExpertCacheScope(s.token, &released) == 1, "Final causal scope End failed"); s.token = 0;
    require(s.ledger.held() == 0 && !s.ledger.pending[0] && !s.ledger.pending[1]
        && !s.ledger.committed[0] && !s.ledger.committed[1] && Stats().reserved == 0 && s.ledger.valid,
        "Checked metadata/cache/staging ownership did not fully retire");
    tsg::host_moe_expert_cache_release(); // Retire all registered FILE handles before success/unlink.
    require(tsg::SharedCacheCharge::detach(&s.ledger), "Final empty callback registry could not detach"); s.attached = false;
    std::printf("PASS: causal-route cache matched_f32=%zu source_generations=4 quant_groups=3 enrolled_files=12 protected_topk=8 "
        "victim_oracles=%zu replaced_slots=%zu actual_helper_faults=%zu checked_host_ack_retry=1 "
        "scope=tiny_actual_cache_cuda_not_glm_hook3_capture_model_quality_or_speed\n", compared_f32, victim_proofs, replaced, actual_faults);
}
#endif
