// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "../ggml_ops_host_moe_miss_read_order.h"
#include "../ggml_ops_host_moe_file_ring.h"
#include <array>
#include <cstdio>
#include <map>
#include <set>
#include <tuple>

static std::size_t checks = 0;
static void require(bool value, const char* reason) {
    ++checks; if (!value) throw std::runtime_error(reason);
}

int main() {
    try {
        require(!tsg::host_moe_file_sort_flag(nullptr) && !tsg::host_moe_file_sort_flag("0")
            && tsg::host_moe_file_sort_flag("1"), "exact default/off/on parser failed");
        for (const char* value : {"", "00", "01", " 1", "1 ", "+1", "2", "true", "1x"}) {
            bool failed = false;
            try { (void)tsg::host_moe_file_sort_flag(value); } catch (const std::invalid_argument&) { failed = true; }
            require(failed, "malformed file-sort policy accepted");
        }
        const auto rejects = [&](const std::vector<std::pair<int,int>>& selected, int c, int e) {
            bool failed = false;
            try { tsg::HostMoeMissReadOrder plan(selected, c, e); }
            catch (const std::invalid_argument&) { failed = true; }
            require(failed, "invalid selected-union assignment accepted");
        };
        rejects({}, 32, 48); rejects({{0,0}}, 0, 48); rejects({{0,0}}, 32, 289);
        rejects({{-1,0}}, 32, 48); rejects({{32,0}}, 32, 48);
        rejects({{0,-1}}, 32, 48); rejects({{0,48}}, 32, 48);
        rejects({{0,7},{0,4}}, 32, 48); rejects({{0,7},{2,7}}, 32, 48);
        const std::vector<std::pair<int,int>> assignment{{11,17},{0,2},{29,31},{6,0},{4,3},{22,16},{31,47},{8,9}};
        const auto immutable = assignment;
        const tsg::HostMoeMissReadOrder order(assignment, 32, 48);
        require(assignment == immutable && order.changed && order.inversions_before == 12,
            "sort changed slot selection or failed the independent inversion count");
        const std::array<int,8> experts{0,2,3,9,16,17,31,47};
        for (std::size_t i = 0; i < experts.size(); ++i) {
            require(order.ordered[i].second == experts[i], "source order differs from independently enumerated union");
            require(std::find(assignment.begin(),assignment.end(),order.ordered[i]) != assignment.end(),
                "slot/expert pair was rebound by sorting");
        }
        const tsg::HostMoeMissReadOrder repeated(order.ordered,32,48);
        require(!repeated.changed && repeated.inversions_before == 0 && repeated.ordered == order.ordered,
            "already sorted input claimed changed activity");
        const std::array<std::size_t,3> strides{7,13,19};
        using Key = std::pair<int,int>;
        for (const auto arena : {2u,7u,23u,64u,512u}) {
            const tsg::HostMoeFileRingPlan original(assignment,strides,arena), sorted(order.ordered,strides,arena);
            require(original.slot_bytes == sorted.slot_bytes && original.pieces.size() == sorted.pieces.size()
                && original.windows.size() == sorted.windows.size(), "sort changed geometry/task count");
            std::map<Key,std::vector<std::uint8_t>> copied;
            std::map<Key,std::vector<bool>> coverage;
            for (const auto& pair : assignment) for (int p=0;p<3;++p) {
                copied[{pair.first,p}].resize(strides[p],0xa5); coverage[{pair.first,p}].resize(strides[p]);
            }
            std::array<std::size_t,3> last_end{};
            std::size_t bytes=0;
            for (std::size_t w=0;w<original.windows.size();++w) {
                const auto& a=original.windows[w]; const auto& b=sorted.windows[w];
                require(a.bytes==b.bytes && a.count==b.count && a.first==b.first,
                    "sort changed a bounded window partition");
                for (std::size_t i=b.first;i<b.first+b.count;++i) {
                    const auto& op=original.pieces[i]; const auto& piece=sorted.pieces[i];
                    require(op.part==piece.part && op.bytes==piece.bytes && op.expert_offset==piece.expert_offset
                        && op.staging_offset==piece.staging_offset, "sort changed a task/H2D size or stage offset");
                    const auto found=std::find_if(assignment.begin(),assignment.end(),[&](const auto& x){return x.first==piece.slot;});
                    require(found!=assignment.end() && piece.source_offset==std::size_t(found->second)*strides[piece.part]+piece.expert_offset,
                        "read crosses unselected source or wrong assigned target");
                    require(piece.source_offset>=last_end[piece.part], "projection source offsets are not monotonic");
                    last_end[piece.part]=piece.source_offset+piece.bytes;
                    for (std::size_t j=0;j<piece.bytes;++j) {
                        auto& seen=coverage.at({piece.slot,piece.part});
                        require(!seen[piece.expert_offset+j], "selected destination byte copied twice");
                        seen[piece.expert_offset+j]=true;
                        copied.at({piece.slot,piece.part})[piece.expert_offset+j]=
                            std::uint8_t(piece.part*73+(piece.source_offset+j)*17+3);
                    }
                    bytes+=piece.bytes;
                }
            }
            require(bytes==assignment.size()*39, "sort changed exact selected payload bytes");
            for (const auto& pair:assignment) for (int p=0;p<3;++p) for (std::size_t b=0;b<strides[p];++b)
                require(coverage.at({pair.first,p})[b]
                    && copied.at({pair.first,p})[b]==std::uint8_t(p*73+(std::size_t(pair.second)*strides[p]+b)*17+3),
                    "exact source-tagged payload differs at original target");
        }
        // Real mixed GLM strides, dense and sparse required unions: metadata only.
        for (const auto count : {1,8,271,284,288}) {
            std::vector<std::pair<int,int>> selected;
            for (int i=0;i<count;++i) selected.emplace_back(i,287-i);
            const tsg::HostMoeMissReadOrder plan(selected,288,288);
            const std::array<std::size_t,3> real{2162688,2162688,4456448};
            const tsg::HostMoeFileRingPlan a(selected,real,72u<<20),b(plan.ordered,real,72u<<20);
            require(a.windows.size()==b.windows.size() && a.pieces.size()==b.pieces.size()
                && plan.inversions_before==std::uint64_t(count)*(count-1)/2,
                "bounded real-stride union changed counts or inversion arithmetic");
            for (std::size_t i=0;i<a.pieces.size();++i)
                require(a.pieces[i].bytes==b.pieces[i].bytes && a.pieces[i].part==b.pieces[i].part,
                    "real-stride sort changed per-piece operation size");
        }
        std::printf("PASS: file-sort strict policy, immutable pair/source assignment, exact byte coverage, unchanged tasks/windows/H2D geometry; checks=%zu\n",checks);
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr,"FAIL: %s\n",error.what()); return 1; }
}
