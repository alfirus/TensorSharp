// Source-only companion to the existing real-file/ring CUDA fixture.
#ifdef TSG_GGML_USE_CUDA
static bool file_sort_fixture_source(const tsg::HostMoeCacheTestSlot& slot,
                                     const tsg::HostMoeSegment& hm, int expert) {
    return slot.expert == expert && slot.source[0] == reinterpret_cast<std::uintptr_t>(hm.gate_data)
        && slot.source[1] == reinterpret_cast<std::uintptr_t>(hm.up_data)
        && slot.source[2] == reinterpret_cast<std::uintptr_t>(hm.down_data);
}
static void file_sort_fixture_completed(const tsg::HostMoeSegment& hm,
    const std::vector<std::int32_t>& requested, std::size_t count,
    const tsg::HostMoeCacheTestSnapshot& before,
    const tsg::HostMoeFileSortTestSnapshot& previous, std::uint64_t reads_before) {
    tsg::HostMoeCacheTestSnapshot after;
    require(tsg::host_moe_expert_cache_test_lifecycle_snapshot(hm,g_backend,after)
        && after.capacity <= 64 && after.clock == before.clock + count,
        "File sort changed complete original-route LRU clock");
    std::vector<int> missing;
    for (std::size_t i=0;i<count;++i) {
        const auto expert=requested[i];
        bool was_hit=false;
        for (int s=0;s<before.capacity;++s) if (file_sort_fixture_source(before.slots[s],hm,expert)) was_hit=true;
        if (!was_hit && std::find(missing.begin(),missing.end(),expert)==missing.end()) missing.push_back(expert);
        int found=-1;
        for (int s=0;s<after.capacity;++s) if (file_sort_fixture_source(after.slots[s],hm,expert)) {
            require(found<0,"Sorted read published a false duplicate source-qualified slot"); found=s;
        }
        require(found>=0,"Sorted read lost a required source-qualified expert");
        std::size_t last=i;
        for (std::size_t j=i+1;j<count;++j) if (requested[j]==expert) last=j;
        require(after.slots[found].last_used==before.clock+last+1,
            "Sorted read changed original route-order recency timestamp");
        for (int s=0;s<before.capacity;++s) if (file_sort_fixture_source(before.slots[s],hm,expert))
            require(s==found,"Sorted read evicted/reassigned a protected required hit");
    }
    const auto proof=tsg::host_moe_expert_cache_test_file_sort_snapshot();
    require(proof.completed_calls==previous.completed_calls+(missing.empty()?0:1),
        "File sort actual completed service count differs from original demand misses");
    if (missing.empty()) {
        require(proof.changed_calls==previous.changed_calls && tsg::g_test_file_reads==reads_before
            && tsg::g_test_file_spans.empty(),"All-hit call claimed sorted read activity");
        return;
    }
    const std::array<std::size_t,3> stride{
        static_cast<std::size_t>(hm.gate_bytes/hm.num_experts),
        static_cast<std::size_t>(hm.up_bytes/hm.num_experts),
        static_cast<std::size_t>(hm.down_bytes/hm.num_experts)};
    require(proof.misses==missing.size() && proof.bytes==missing.size()*(stride[0]+stride[1]+stride[2])
        && proof.inversions_before>0 && proof.changed_calls==previous.changed_calls+1,
        "Sorted fixture lacks a real nontrivial unique-demand permutation");
    std::vector<std::pair<int,int>> original,ordered;
    for (std::size_t i=0;i<proof.misses;++i) {
        require(proof.original_expert[i]==missing[i],"Sort ran before original route/victim assignment");
        original.emplace_back(proof.original_slot[i],proof.original_expert[i]);
        ordered.emplace_back(proof.ordered_slot[i],proof.ordered_expert[i]);
        require(proof.ordered_slot[i]>=0 && proof.ordered_slot[i]<after.capacity
            && file_sort_fixture_source(after.slots[proof.ordered_slot[i]],hm,proof.ordered_expert[i]),
            "Read order rebound original source to a different published slot");
    }
    auto by_pair=original; auto compare=ordered; std::sort(by_pair.begin(),by_pair.end()); std::sort(compare.begin(),compare.end());
    require(by_pair==compare,"Sorted read changed the selected slot/source pair multiset");
    for (std::size_t i=1;i<ordered.size();++i)
        require(ordered[i-1].second<ordered[i].second,"Actual planned expert offsets are not strictly ascending");
    require(proof.source_identity[0] && proof.source_identity[1] && proof.source_identity[2]
        && proof.source_identity[0]!=proof.source_identity[1] && proof.source_identity[1]!=proof.source_identity[2]
        && proof.source_identity[0]!=proof.source_identity[2],"Sort lost exact independent source identities");
    // Serial bounded windows use a full arena; ring uses two half-arena slots.
    const auto planner_arena=proof.ring?proof.arena:proof.arena*2;
    const tsg::HostMoeFileRingPlan old_plan(original,stride,planner_arena),new_plan(ordered,stride,planner_arena);
    require(old_plan.windows.size()==new_plan.windows.size() && old_plan.pieces.size()==new_plan.pieces.size()
        && proof.windows==new_plan.windows.size() && proof.tasks==new_plan.pieces.size()
        && proof.slot_bytes==new_plan.slot_bytes && tsg::g_test_file_reads-reads_before==proof.tasks,
        "Sorting changed actual task/window/H2D geometry or completed task count");
    using Span=std::tuple<std::uint64_t,std::uint64_t,std::size_t>;
    std::vector<Span> actual,expected;
    for (const auto& span:tsg::g_test_file_spans) actual.emplace_back(span.source,span.offset,span.bytes);
    for (std::size_t i=0;i<new_plan.pieces.size();++i) {
        const auto& a=old_plan.pieces[i]; const auto& b=new_plan.pieces[i];
        require(a.part==b.part && a.bytes==b.bytes && a.expert_offset==b.expert_offset
            && a.staging_offset==b.staging_offset,"Sort changed a read/upload operation's size or arena offset");
        expected.emplace_back(proof.source_identity[b.part],b.source_offset,b.bytes);
    }
    for (std::size_t i=0;i<new_plan.windows.size();++i)
        require(old_plan.windows[i].bytes==new_plan.windows[i].bytes && old_plan.windows[i].count==new_plan.windows[i].count,
            "Sort changed a per-window task or byte bound");
    std::sort(actual.begin(),actual.end()); std::sort(expected.begin(),expected.end());
    require(actual==expected,"Actual completed exact-source read spans differ from the selected permutation");
    std::printf("[TEST-FILE-SORT] rows=%d misses=%zu inversions=%llu tasks=%zu windows=%zu payload=%zu ring=%d original_slot_pairs=1 source_spans=1 original_lru=1\n",
        hm.seq_len,proof.misses,static_cast<unsigned long long>(proof.inversions_before),proof.tasks,proof.windows,proof.bytes,proof.ring?1:0);
}
#endif
