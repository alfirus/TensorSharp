// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_host_moe_down_overlap.h"
#include <cstdio>
#include <map>
#include <stdexcept>

namespace {
std::size_t checks = 0;
void require(bool value, const char* text) { ++checks; if (!value) throw std::runtime_error(text); }
template<class F> void rejects(F fn) {
    bool rejected = false; try { fn(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "Invalid geometry was accepted");
}
void plans() {
    using Plan = tsg::HostMoeProjectionRingPlan;
    const std::vector<std::pair<int, int>> misses{{3, 16}, {0, 0}, {8, 7}, {2, 11}};
    for (const auto strides : {std::array<std::size_t, 3>{13, 13, 29}, {1, 2, 3}, {2162688, 2162688, 3211264}})
        for (const auto arena : {std::size_t(2), std::size_t(37), std::size_t(72) << 20}) {
            // Tiny slots with production strides intentionally exceed the
            // bounded metadata allowance; refusal precedes any source read.
            if (arena < 100 && strides[0] > 100) {
                rejects([&] { Plan p(misses, strides, 3, arena, {strides[0]*17,strides[1]*17,strides[2]*17}, 9, 17); });
                continue;
            }
            std::size_t total = 0;
            for (unsigned mask : {3u, 4u}) {
                Plan p(misses, strides, mask, arena, {strides[0]*17,strides[1]*17,strides[2]*17}, 9, 17);
                std::map<std::pair<int, int>, std::size_t> covered;
                std::size_t bytes = 0, pieces = 0;
                for (const auto& w : p.windows) {
                    require(w.first == pieces && w.count && w.bytes && w.bytes <= p.slot_bytes, "Invalid window partition");
                    std::size_t cursor = 0;
                    for (auto i = w.first; i < w.first+w.count; ++i) {
                        const auto& q = p.pieces[i];
                        require((mask & (1u<<q.part)) != 0 && q.bytes && q.staging_offset == cursor
                            && q.bytes <= p.slot_bytes-cursor, "Piece changed projection/slot bounds");
                        const auto found = std::find_if(misses.begin(), misses.end(), [&](const auto& m) { return m.first == q.slot; });
                        require(found != misses.end() && q.source_offset == std::size_t(found->second)*strides[q.part]+q.expert_offset,
                            "Piece changed the exact selected file offset");
                        auto& extent = covered[{q.slot,q.part}];
                        require(q.expert_offset == extent && q.bytes <= strides[q.part]-extent, "Projection has a hole/duplicate");
                        extent += q.bytes; cursor += q.bytes; bytes += q.bytes; ++pieces;
                    }
                    require(cursor == w.bytes, "Window has padding payload");
                }
                for (const auto& m : misses) for (int part = 0; part < 3; ++part)
                    require(covered[{m.first,part}] == ((mask & (1u<<part)) ? strides[part] : 0), "Phase missed or read an unselected projection");
                require(bytes == p.payload_bytes && pieces == p.pieces.size(), "Phase payload differs");
                total += bytes;
            }
            require(total == misses.size()*(strides[0]+strides[1]+strides[2]), "Two phases changed total payload");
        }
    const std::array<std::size_t, 3> s{13,17,19}, e{221,289,323};
    rejects([&] { Plan p(misses,s,0,72,e,9,17); });
    rejects([&] { Plan p(misses,s,8,72,e,9,17); });
    rejects([&] { Plan p(misses,s,3,1,e,9,17); });
    rejects([&] { Plan p(misses,s,3,72,{220,289,323},9,17); });
    rejects([&] { Plan p({{0,0},{0,1}},s,3,72,e,9,17); });
    rejects([&] { Plan p({{0,0},{1,0}},s,3,72,e,9,17); });
    rejects([&] { Plan p({{9,0}},s,3,72,e,9,17); });
    rejects([&] { Plan p({{0,17}},s,3,72,e,9,17); });
    rejects([&] { Plan p({{0,-1}},s,3,72,e,9,17); });
    rejects([&] { Plan p({},s,3,72,e,9,17); });
    rejects([&] { Plan p(misses,{SIZE_MAX,17,19},3,72,e,9,17); });
    require(tsg::host_moe_overlap_spans_disjoint(100,20,120,20), "Adjacent buffers must be allowed");
    require(!tsg::host_moe_overlap_spans_disjoint(100,21,120,20), "Overlapping buffers were accepted");
    require(!tsg::host_moe_overlap_spans_disjoint(UINTPTR_MAX-4,8,100,20), "Overflowing buffer was accepted");
}
void cuts_and_drains() {
    std::vector<tsg::HostMoeOverlapNode> n(4);
    n[0].id=10; n[1].id=20; n[2].id=30; n[3].id=40;
    n[1].sources[0]=10; n[2].sources[9]=20; n[3].sources[10]=30;
    require(tsg::host_moe_overlap_cut_valid(n,2,20,30), "Original dependency cut was refused");
    n[0].sources[9]=30;
    require(!tsg::host_moe_overlap_cut_valid(n,2,20,30), "High source slot crossed backward over the cut");
    n[0].sources[9]=0; n[1].sources[10]=40;
    require(!tsg::host_moe_overlap_cut_valid(n,2,20,30), "View dependency crossed backward over the cut");
    n[1].sources[10]=0;
    require(!tsg::host_moe_overlap_cut_valid(n,1,20,30), "Wrong activation boundary was accepted");
    n[3].id=10;
    require(!tsg::host_moe_overlap_cut_valid(n,2,20,30), "Repeated node identity was accepted");
    for (unsigned failures=0; failures<4; ++failures) {
        unsigned attempts=0;
        const bool okay=tsg::host_moe_overlap_drain_both([&] { attempts|=1; if(failures&1) throw std::runtime_error("upload"); },
            [&] { attempts|=2; if(failures&2) throw std::runtime_error("compute"); });
        require(attempts==3 && okay==(failures==0), "One failing drain skipped the other stream");
    }
}
}
int main() {
    try { plans(); cuts_and_drains(); std::printf("PASS: %zu bounded projection/cut/source/span/drain checks; CPU only\n",checks); return 0; }
    catch(const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
