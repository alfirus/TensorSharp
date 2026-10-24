// Included by the existing CPU ring executable; no CUDA/device dependency.
static void coalesced_ring_checks() {
    using Plan = tsg::HostMoeCoalescedFileRingPlan;
    constexpr std::size_t mib = std::size_t(1) << 20;
    require(!tsg::host_moe_ring_coalesce_flag(nullptr) && !tsg::host_moe_ring_coalesce_flag("")
        && !tsg::host_moe_ring_coalesce_flag("01") && !tsg::host_moe_ring_coalesce_flag("1 ")
        && tsg::host_moe_ring_coalesce_flag("1"), "coalescing flag must be exact/default off");
    const auto eligible = [&](bool enabled, bool wide, int selected, std::size_t ram,
        std::size_t bytes, std::size_t arena, std::size_t misses) {
        return tsg::host_moe_ring_coalesce_eligible(enabled, wide, selected, ram, bytes, arena, misses);
    };
    require(eligible(true, true, 8, 0, 73*mib, 72*mib, 24), "qualified coalesced policy refused");
    require(!eligible(false, true, 8, 0, 73*mib, 72*mib, 24)
        && !eligible(true, false, 8, 0, 73*mib, 72*mib, 24)
        && !eligible(true, true, 7, 0, 73*mib, 72*mib, 24)
        && !eligible(true, true, 8, 1, 73*mib, 72*mib, 24)
        && !eligible(true, true, 8, 0, 72*mib, 72*mib, 24)
        && !eligible(true, true, 8, 0, 73*mib, 32*mib, 24)
        && !eligible(true, true, 8, 0, 73*mib, 72*mib, 0)
        && !eligible(true, true, 8, 0, 73*mib, 72*mib, 289), "ineligible coalesced policy accepted");
    require(!Plan::metadata_fits(288, (std::numeric_limits<std::size_t>::max)(), 1)
        && !Plan::metadata_fits(0, 1, 1), "unbounded metadata admitted");
    std::array<tsg::HostMoeRingTargetSpan, 3> spans{{{0x1100, 0x1000, 256, 4096, false},
        {0x1300, 0x1000, 256, 4096, false}, {0x1500, 0x1000, 256, 4096, false}}};
    require(tsg::host_moe_ring_targets_disjoint(spans), "disjoint physical targets refused");
    auto invalid = spans; invalid[1].data = spans[0].data + 1;
    require(!tsg::host_moe_ring_targets_disjoint(invalid), "distinct tensors with overlapping absolute targets admitted");
    invalid = spans; invalid[0].view = true;
    require(!tsg::host_moe_ring_targets_disjoint(invalid), "view target admitted");
    invalid = spans; invalid[0].bytes = 4096;
    require(!tsg::host_moe_ring_targets_disjoint(invalid), "target exceeds physical backing");
    invalid = spans; invalid[0].data = (std::numeric_limits<std::uintptr_t>::max)() - 32;
    invalid[0].buffer = invalid[0].data; invalid[0].buffer_bytes = 256;
    require(!tsg::host_moe_ring_targets_disjoint(invalid), "target absolute end overflow admitted");

    const std::array<std::size_t, 3> stride{7, 13, 19}, extents{84, 156, 228};
    const std::array<std::uint64_t, 3> identities{101, 202, 303};
    const std::vector<std::pair<int, int>> misses{{5, 8}, {2, 1}, {9, 0}, {0, 2}, {7, 5}};
    tsg_dsv41::engram_io_pool pool(8);
    for (std::size_t arena : {2, 7, 23, 64, 1024}) {
        const Plan plan(misses, stride, arena, extents, identities, 10, 12);
        require(plan.payload_bytes == 195 && plan.source_identity == identities && plan.slot_bytes == arena/2,
            "coalesced plan changed payload/source identity/arena");
        for (const auto& window : plan.windows) {
            std::vector<bool> read_coverage(window.bytes), upload_coverage(window.bytes);
            for (std::size_t i = window.first_read; i < window.first_read + window.read_count; ++i) {
                const auto& read = plan.reads[i];
                require(read.bytes && read.bytes <= tsg::kHostMoeRingCoalesceReadBytes
                    && read.source_offset <= extents[read.part] && read.bytes <= extents[read.part]-read.source_offset
                    && read.staging_offset <= window.bytes && read.bytes <= window.bytes-read.staging_offset,
                    "coalesced read exceeds exact source or slot");
                for (std::size_t b = read.staging_offset; b < read.staging_offset + read.bytes; ++b) {
                    require(!read_coverage[b], "coalesced reads overlap arena bytes"); read_coverage[b] = true;
                    bool selected = false;
                    for (std::size_t p = window.first_piece; p < window.first_piece + window.piece_count; ++p) {
                        const auto& piece = plan.pieces[p];
                        if (b >= piece.staging_offset && b-piece.staging_offset < piece.bytes) {
                            selected = piece.part == read.part && piece.source_offset + b-piece.staging_offset
                                == read.source_offset + b-read.staging_offset;
                            break;
                        }
                    }
                    require(selected, "merged read crossed an unselected gap/projection/source");
                }
            }
            for (std::size_t i = window.first_piece; i < window.first_piece + window.piece_count; ++i) {
                const auto& piece = plan.pieces[i];
                require(piece.bytes && piece.expert_offset < stride[piece.part]
                    && piece.bytes <= stride[piece.part]-piece.expert_offset
                    && piece.staging_offset <= window.bytes && piece.bytes <= window.bytes-piece.staging_offset,
                    "coalesced upload exceeds expert/arena");
                for (std::size_t b = piece.staging_offset; b < piece.staging_offset + piece.bytes; ++b) {
                    require(!upload_coverage[b], "coalesced uploads repeat arena bytes"); upload_coverage[b] = true;
                }
            }
            require(read_coverage == upload_coverage
                && std::all_of(read_coverage.begin(), read_coverage.end(), [](bool v) { return v; }),
                "read/upload partitions do not cover identical exact bytes");
        }
        for (int failure : {-1, 0, 1, 2}) {
            std::vector<std::uint8_t> staging(arena, 0xff);
            std::vector<std::size_t> counts;
            for (const auto& window : plan.windows) counts.push_back(window.read_count);
            std::array<std::future<void>, 2> transfers;
            std::array<std::atomic<bool>, 2> in_flight{};
            std::atomic<int> active{0};
            std::map<std::pair<int, int>, std::vector<std::uint8_t>> output;
            for (const auto& miss : misses) for (int part = 0; part < 3; ++part)
                output[{miss.first, part}].resize(stride[part], 0xff);
            const auto fail_index = (std::min)(std::size_t(2), plan.windows.size()-1);
            bool threw = false, published = false;
            try {
                tsg::run_bounded_parallel_read_pipeline(pool, counts, 2,
                    [&](std::size_t index, std::size_t slot) {
                        if (transfers[slot].valid()) transfers[slot].get();
                        if (failure == 0 && index == fail_index) throw std::runtime_error("coalesced fence failure");
                    }, [&](std::size_t index, std::size_t job, std::size_t slot) {
                        ++active; struct Exit { std::atomic<int>& active; ~Exit() { --active; } } exit{active};
                        if (in_flight[slot]) throw std::runtime_error("coalesced DMA source overwritten");
                        const auto& read = plan.reads[plan.windows[index].first_read+job];
                        auto* target = staging.data()+slot*plan.slot_bytes+read.staging_offset;
                        for (std::size_t b = 0; b < read.bytes; ++b) {
                            target[b] = static_cast<std::uint8_t>(read.part*41+read.source_offset+b);
                            if (failure == 1 && index == fail_index && job == 0 && b == read.bytes/2)
                                throw std::runtime_error("coalesced partial read failure");
                        }
                    }, [&](std::size_t index, std::size_t slot) {
                        in_flight[slot] = true;
                        const auto& window = plan.windows[index];
                        std::vector<std::uint8_t*> targets;
                        for (std::size_t i = window.first_piece; i < window.first_piece+window.piece_count; ++i) {
                            const auto& piece = plan.pieces[i];
                            targets.push_back(output.at({piece.slot, piece.part}).data()+piece.expert_offset);
                        }
                        transfers[slot] = std::async(std::launch::async, [&, index, slot, targets] {
                            for (int i = 0; i < 40; ++i) std::this_thread::yield();
                            const auto& window = plan.windows[index];
                            for (std::size_t job = 0; job < window.piece_count; ++job) {
                                const auto& piece = plan.pieces[window.first_piece+job];
                                const auto* source = staging.data()+slot*plan.slot_bytes+piece.staging_offset;
                                for (std::size_t b = 0; b < piece.bytes; ++b) {
                                    if (source[b] != static_cast<std::uint8_t>(piece.part*41+piece.source_offset+b))
                                        throw std::runtime_error("coalesced source changed during DMA");
                                    targets[job][b] = source[b];
                                }
                            }
                            in_flight[slot] = false;
                        });
                        if (failure == 2 && index == fail_index) throw std::runtime_error("coalesced queued-upload failure");
                    });
                for (auto& transfer : transfers) if (transfer.valid()) transfer.get();
                published = true;
            } catch (const std::runtime_error&) { threw = true; }
            for (auto& transfer : transfers) if (transfer.valid()) transfer.get();
            require(active == 0 && !in_flight[0] && !in_flight[1] && threw == (failure >= 0)
                && published == (failure < 0), "coalesced failure escaped live work/publication");
            if (published) for (const auto& miss : misses) for (int part = 0; part < 3; ++part)
                for (std::size_t b = 0; b < stride[part]; ++b)
                    require(output.at({miss.first, part})[b] == static_cast<std::uint8_t>(part*41+miss.second*stride[part]+b),
                        "coalesced scatter changed exact target bytes");
        }
    }
    for (const auto& strides : {std::array<std::size_t, 3>{2162688,2162688,3211264},
            std::array<std::size_t, 3>{1638400,1638400,3211264},
            std::array<std::size_t, 3>{2162688,2162688,4456448},
            std::array<std::size_t, 3>{36*mib+1,36*mib-1,73*mib+3}}) {
        std::vector<std::pair<int,int>> all;
        const int count = strides[0] > 36*mib ? 4 : 288;
        for (int e = 0; e < count; ++e) all.emplace_back(count-1-e, e);
        const std::array<std::size_t,3> source{Plan::mul(strides[0],count), Plan::mul(strides[1],count), Plan::mul(strides[2],count)};
        const Plan plan(all, strides, 72*mib, source, identities, count, count);
        const tsg::HostMoeFileRingPlan ordinary(all, strides, 72*mib);
        require((count == 288 ? plan.reads.size() < ordinary.pieces.size() : plan.reads.size() >= ordinary.pieces.size())
            && plan.coalesced_bytes > 0 && plan.pieces.size() <= tsg::kHostMoeRingCoalesceMaxItems
            && plan.reads.size() <= tsg::kHostMoeRingCoalesceMaxItems, "large bounded plan has no real task reduction");
        std::size_t bytes = 0; bool cap_crosses_piece = false;
        for (const auto& window : plan.windows) {
            require(window.bytes <= plan.slot_bytes, "coalesced window exceeds slot");
            if (window.bytes == 36*mib) require(window.read_count >= 3, "full window cannot supply read concurrency");
            for (std::size_t i = window.first_read; i < window.first_read+window.read_count; ++i) {
                const auto& read = plan.reads[i];
                require(read.bytes && read.bytes <= 16*mib, "coalesced read cap changed"); bytes += read.bytes;
                const auto end = read.staging_offset+read.bytes;
                for (std::size_t p = window.first_piece; p < window.first_piece+window.piece_count; ++p)
                    if (end > plan.pieces[p].staging_offset && end < plan.pieces[p].staging_offset+plan.pieces[p].bytes)
                        cap_crosses_piece = true;
            }
        }
        require(bytes == plan.payload_bytes && cap_crosses_piece, "cap cuts must preserve independent upload partition");
        auto next_ids = identities; ++next_ids[0];
        const Plan next(all, strides, 72*mib, source, next_ids, count, count);
        require(plan.source_identity != next.source_identity, "source generation identity was aliased");
    }
    const auto rejects = [&](auto selected, auto strides, auto bytes, auto ids, int capacity, int experts) {
        bool refused = false;
        try { Plan candidate(selected, strides, 72*mib, bytes, ids, capacity, experts); }
        catch (const std::exception&) { refused = true; }
        require(refused, "invalid coalesced plan accepted");
    };
    rejects(std::vector<std::pair<int,int>>{{0,0},{0,1}}, stride, extents, identities, 10, 12);
    rejects(std::vector<std::pair<int,int>>{{0,0},{1,0}}, stride, extents, identities, 10, 12);
    rejects(std::vector<std::pair<int,int>>{{-1,0}}, stride, extents, identities, 10, 12);
    rejects(std::vector<std::pair<int,int>>{{10,0}}, stride, extents, identities, 10, 12);
    rejects(std::vector<std::pair<int,int>>{{0,12}}, stride, extents, identities, 10, 12);
    auto bad_extent = extents; --bad_extent[0]; rejects(misses, stride, bad_extent, identities, 10, 12);
    auto bad_identity = identities; bad_identity[0] = 0; rejects(misses, stride, extents, bad_identity, 10, 12);
    rejects(misses, std::array<std::size_t,3>{(std::numeric_limits<std::size_t>::max)(),13,19}, extents, identities,10,12);
}
