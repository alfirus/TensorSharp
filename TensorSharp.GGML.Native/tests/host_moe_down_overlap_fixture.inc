// Production selected-cache pipeline versus unchanged full-resident CUDA math.
#ifdef TSG_GGML_USE_CUDA
extern "C" int TSGgml_HostMoePrefillDownOverlapVersion();
extern "C" int TSGgml_HostMoePrefillDownOverlapEnabled();
extern "C" int TSGgml_HostMoePrefillRingStagingEnabled();
extern "C" int TSGgml_TestHostMoePrefillDownOverlapFailNext(int);
extern "C" int TSGgml_TestHostMoePrefillDownOverlapStats(std::uint64_t*, int);

void host_moe_down_overlap_checks(const std::array<ggml_type, 3>& types) {
    constexpr int h=4096, count=48, selected=8, rows=16, tail=9;
    const bool compact_iq2=types[0]==GGML_TYPE_IQ2_XXS;
    const int fixture_ff=compact_iq2?256:types[0]==GGML_TYPE_Q8_0?512:1280;
    const std::size_t arena=std::size_t(compact_iq2?8:72)<<20;
    // Native staging is a process latch. Bind the explicit smaller IQ2 test
    // policy before its first query; the Q8/IQ1 cases retain real72MiB coverage.
    set_environment("TS_HOST_MOE_PREFILL_RING_72MB",compact_iq2?"0":"1");
    require(TSGgml_HostMoePrefillRingStagingEnabled()==(compact_iq2?0:1),"Unexpected native staging latch");
    tsg::host_moe_expert_cache_test_file_slice_limit(compact_iq2?arena:std::size_t(32)<<20);
    struct RestoreSlice { ~RestoreSlice(){tsg::host_moe_expert_cache_test_file_slice_limit(std::size_t(32)<<20);} } restore_slice;
    const auto prep_start=std::chrono::steady_clock::now();
    auto stage=[&](const char* name) {
        const auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-prep_start).count();
        std::fprintf(stderr,"[TEST-DOWN-OVERLAP-PREP] stage=%s type=%d hidden=%d ff=%d experts=%d arena_bytes=%zu elapsed_ms=%.3f\n",
            name,static_cast<int>(types[0]),h,fixture_ff,count,arena,ms); std::fflush(stderr);
    };
    require(TSGgml_HostMoePrefillDownOverlapVersion()==1 && TSGgml_HostMoePrefillDownOverlapEnabled()==1,
        "Prefill Down overlap capability is disabled");
    tsg::host_moe_expert_cache_release();
    Weights weights; weights.types=types;
    stage("fill_begin");
    weights.fill(.73f,count,selected,h,fixture_ff);
    stage("fill_ready");
    weights.segment.pooled_model_experts=count; weights.segment.swiglu_clamp=7.0f;
    stage("oracle_allocation_begin");
    FullResidentGraph full(weights,rows), short_full(weights,tail);
    stage("oracle_allocation_ready");
    std::vector<float> input(h*rows), routes(selected*rows), actual(h*rows);
    std::vector<std::int32_t> low(selected*rows), high(selected*rows), mixed(selected*rows), final_ids(selected*rows);
    for(int row=0;row<rows;++row) {
        for(int i=0;i<h;++i) input[row*h+i]=std::cos(.021f*i+.007f*row);
        for(int k=0;k<selected;++k) {
            routes[row*selected+k]=float(k+1)/36;
            low[row*selected+k]=k;
            high[row*selected+k]=24+(row%3)*selected+k;
            mixed[row*selected+k]=k<4?24+k:12+(row%2)*4+k-4;
            final_ids[row*selected+k]=k<4?24+k:20+k-4;
        }
    }
    auto expected=[&](int width,const std::vector<std::int32_t>& ids) {
        auto& graph=width==tail?short_full:full;
        return graph.compute_batch(std::vector<float>(input.begin(),input.begin()+h*width),
            std::vector<std::int32_t>(ids.begin(),ids.begin()+selected*width),
            std::vector<float>(routes.begin(),routes.begin()+selected*width));
    };
    stage("oracle_math_begin");
    const auto low_expected=expected(rows,low), high_expected=expected(rows,high);
    const auto mixed_expected=expected(rows,mixed), tail_expected=expected(tail,final_ids), final_expected=expected(rows,final_ids);
    stage("oracle_math_ready");
    const auto path=fixture_file_path("host-moe-down-overlap-");
    struct FileCleanup {
        const std::string& path; bool removed=false;
        ~FileCleanup(){if(!removed){try{tsg::host_moe_expert_cache_release();}catch(...){}std::remove(path.c_str());}}
    } file_cleanup{path};
    std::array<std::int64_t,3> offsets{};
    {
        std::ofstream file(path,std::ios::binary|std::ios::trunc);
        for(int p=0;p<3;++p) {
            file.write("unaligned-source!",17); offsets[p]=static_cast<std::int64_t>(file.tellp());
            file.write(reinterpret_cast<const char*>(weights.quantized[p].data()),weights.quantized[p].size());
        }
        require(bool(file),"Cannot write authoritative Down overlap source");
    }
    for(auto& p:weights.quantized) std::fill(p.begin(),p.end(),0); // Only registered file bytes can produce the oracle.
    stage("validation_begin");
    auto stats=[] {
        std::array<std::uint64_t,5> result{};
        require(TSGgml_TestHostMoePrefillDownOverlapStats(result.data(),5)==1,"Cannot observe actual overlap submissions/drains");
        return result;
    };
    const auto call=[&](int width,const auto& ids,const auto& oracle) {
        auto hm=weights.segment; hm.seq_len=width;
        std::fill(actual.begin(),actual.end(),-999);
        require(tsg::host_moe_cached_experts(hm,input.data(),ids.data(),routes.data(),actual.data(),
            "glm5-next bulk expert prefill",g_backend)==1,"Prefill Down overlap declined");
        require(std::all_of(oracle.begin(),oracle.end(),[](float x){return std::isfinite(x);})
            &&std::any_of(oracle.begin(),oracle.end(),[](float x){return x!=0;})
            &&std::memcmp(actual.data(),oracle.data(),oracle.size()*sizeof(float))==0,
            "Split Gate/Up and Down changed full original-graph F32 bits");
    };
    for(int fault:{0,26,27,28,29,30,31,32}) {
        tsg::host_moe_expert_cache_release();
        for(int p=0;p<3;++p) require(TSGgml_RegisterHostFileSource(weights.quantized[p].data(),weights.quantized[p].size(),
            path.c_str(),offsets[p])==1,"Cannot register immutable overlap source");
        SharedLedger ledger; ledger.capacity=512LL<<20; ledger.host_capacity=arena;
        require(ledger.attach(true),"Cannot attach overlap physical/shared ledger");
        std::uint64_t token=0;
        require(TSGgml_BeginHostMoeExpertCacheScopeWithDecodeRows(ledger.capacity,rows,1,1,&token)==1,
            "Cannot begin overlap scope");
        struct Cleanup {
            std::uint64_t& token;
            ~Cleanup(){TSGgml_TestHostMoePrefillDownOverlapFailNext(0);tsg::g_backend_compute_failed.store(false);
                tsg::g_test_capture_file_spans=false;
                if(token){std::int64_t released=0;TSGgml_EndHostMoeExpertCacheScope(token,&released);}}
        } cleanup{token};
        const void* pointers[3]={weights.segment.gate_data,weights.segment.up_data,weights.segment.down_data};
        const std::int64_t extents[3]={weights.segment.gate_bytes,weights.segment.up_bytes,weights.segment.down_bytes};
        std::array<std::shared_ptr<tsg::ExpertFileSource>,3> files;
        require(tsg::host_moe_file_batch_sources(pointers,extents,files)==1,"Cannot bind exact registered source generations");
        auto verify_spans=[&](const std::vector<int>& ids) {
            auto spans=tsg::g_test_file_spans;
            std::sort(spans.begin(),spans.end(),[](const auto& a,const auto& b){return a.source<b.source||(a.source==b.source&&a.offset<b.offset);});
            std::vector<tsg::FileReadSpan> wanted;
            for(int p=0;p<3;++p) for(int id:ids) {
                const auto stride=static_cast<std::size_t>(extents[p])/count;
                wanted.push_back({files[p]->identity(),std::size_t(id)*stride,stride});
            }
            std::sort(wanted.begin(),wanted.end(),[](const auto& a,const auto& b){return a.source<b.source||(a.source==b.source&&a.offset<b.offset);});
            std::size_t index=0;
            for(const auto& range:wanted) {
                std::size_t done=0;
                while(done<range.bytes) {
                    require(index<spans.size()&&spans[index].source==range.source&&spans[index].offset==range.offset+done
                        &&spans[index].bytes&&spans[index].bytes<=range.bytes-done,
                        "Actual selected read has a wrong generation/offset/hole/duplicate");
                    done+=spans[index++].bytes;
                }
            }
            require(index==spans.size(),"Overlap read bytes outside exact selected misses");
        };
        auto clear_reads=[] {tsg::g_test_file_spans.clear();tsg::g_test_file_window_counts.clear();};
        tsg::g_test_capture_file_spans=true; clear_reads();
        const auto initial=stats(); call(rows,low,low_expected);
        verify_spans({0,1,2,3,4,5,6,7});
        require(stats()[0]==initial[0]+1,"Cold call did not complete an actual split prefix/suffix");
        clear_reads(); const auto warm=stats(); call(rows,low,low_expected);
        require(tsg::g_test_file_spans.empty()&&stats()==warm,"Warm fullgraph claimed new overlap/read activity");
        const auto before=stats();
        require(TSGgml_TestHostMoePrefillDownOverlapFailNext(fault)==1,"Cannot arm selected overlap failure");
        bool failed=false;
        try{call(rows,high,high_expected);}catch(const std::exception&){failed=true;}
        require(failed==(fault!=0),"Overlap fault did not fail closed");
        if(fault) {
            const auto after=stats();
            require(std::all_of(actual.begin(),actual.end(),[](float x){return x==-999;}),"Failed overlap published caller output");
            require(after[0]==before[0]&&after[1]==before[1]+(fault==26?0:1)
                &&after[2]==before[2]+(fault==30?1:0)&&after[3]==before[3]+(fault==26?0:1)
                &&after[4]==before[4]+(fault==31?1:0),"Fault missed its actual prefix/suffix/reader/drain boundary");
            if(fault==26) require(tsg::g_test_file_spans.empty(),"Aliased prefix performed source I/O");
            tsg::g_backend_compute_failed.store(false);
            require(TSGgml_TestHostMoePrefillDownOverlapFailNext(0)==1,"Cannot disarm overlap fault");
            clear_reads(); call(rows,high,high_expected);
        }
        verify_spans({24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47});
        if(compact_iq2) require(tsg::g_test_file_window_counts==std::vector<std::size_t>{4,4},
            "Compact IQ2 must complete four real windows in each phase and reuse both slots");
        clear_reads(); const auto hot=stats(); call(rows,high,high_expected);
        require(tsg::g_test_file_spans.empty()&&stats()==hot,"Repeated hot wide call reread or resubmitted split work");
        call(rows,mixed,mixed_expected); verify_spans({12,13,14,15,16,17,18,19});
        clear_reads(); const auto tail_before=stats(); call(tail,final_ids,tail_expected);
        verify_spans({20,21,22,23});
        require(stats()[0]==tail_before[0]+1,"Exact tail width did not exercise split graph");
        clear_reads(); call(rows,final_ids,final_expected);
        require(tsg::g_test_file_spans.empty(),"Wider hot replacement reread payload");
        require(ledger.valid&&ledger.host_pending==0&&ledger.host_committed==arena,
            "Overlap expanded payload escaped the same charged arena");
        tsg::g_test_capture_file_spans=false;
        std::int64_t released=0;
        require(TSGgml_EndHostMoeExpertCacheScope(token,&released)==1,"Overlap scope cleanup failed"); token=0;
        ledger.check_empty(); require(ledger.detach(),"Overlap retained shared ticket after physical cleanup");
        // Rebinding exactly the same source addresses creates fresh generations
        // next iteration; empty trim also retired the owned upload backend.
    }
    // EndScope retains source registration; drop its read-only handles before
    // deleting the fixture on Windows, after every callback ledger is detached.
    tsg::host_moe_expert_cache_release();
    require(std::remove(path.c_str())==0,"Cannot remove retired Down overlap fixture"); file_cleanup.removed=true;
    stage("validation_ready");
    std::printf("PASS: Down overlap exact original CUDA math at16/9 rows; cold/warm/partial source ranges; alias, prefix, queued-DMA, epoch, joined partial-read and allocation failures; exact%zuMiB ownership/cleanup%s\n",
        arena>>20,compact_iq2?"; IQ2 bounded FF256/four windows per phase":"");
}
#endif
