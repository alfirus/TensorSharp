// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "../dsv41_expert_cache.h"
#include <iostream>

namespace { int cache_status = 0, cache_calls = 0, prefetch_calls = 0, checks = 0; }
namespace tsg {
thread_local std::string g_last_error;
std::atomic<uint64_t> g_ggml_error_count{0};
std::atomic<bool> g_backend_compute_failed{false};
void prefetch_mapped_ranges(const uint8_t * base, const std::vector<std::pair<size_t,size_t>> & ranges) {
    if (!base || ranges.size()!=1 || ranges[0].first!=0 || ranges[0].second!=192)
        throw std::runtime_error("Prefetch did not coalesce the selected expert ranges");
    ++prefetch_calls;
}
int host_moe_cached_experts(const HostMoeSegment & segment, const float * x, const int32_t * ids,
    const float * weights, float * out, const char * name, ggml_backend_t backend, size_t bytes, size_t layers) {
    ++cache_calls;
    if (!x || !ids || !weights || !out || !backend || bytes != 1048576 || layers != 1 ||
        std::string(name) != "deepseek41 native" || segment.clamp_limit != 2.0f)
        throw std::runtime_error("Invalid explicit cache bridge parameters");
    return cache_status;
}
}

namespace {
void check(bool value, const char * message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
void configuration() {
    for (const char * value : {"", "0", "000"}) check(tsg_dsv41_cache::configured_bytes(value)==0,"zero cache config");
    check(tsg_dsv41_cache::configured_bytes("1024")==size_t(1024)*1048576,"cache MiB conversion");
    for (const char * value : {"-1","+1"," 1","1 ","1.0","18446744073709551616"}) {
        bool refused = false;
        try { (void)tsg_dsv41_cache::configured_bytes(value); } catch (const std::exception &) { refused=true; }
        check(refused,"invalid cache config refused");
    }
    for (const char * value : {"", "0"})
        check(!tsg_dsv41_cache::configured_device_bridge(value),"device bridge remains opt-in");
    check(!tsg_dsv41_cache::configured_device_bridge(nullptr),"unset device bridge disabled");
    check(tsg_dsv41_cache::configured_device_bridge("1"),"explicit device bridge enabled");
    for (const char * value : {"-1","2","true"," 1","1 "}) {
        bool refused=false;
        try {(void)tsg_dsv41_cache::configured_device_bridge(value);}catch(const std::exception &){refused=true;}
        check(refused,"invalid device bridge config refused");
    }
    bool named=false;
    try {(void)tsg_dsv41_cache::configured_bytes("-1","TS_DSV41_EXPERT_HOST_CACHE_MB");}
    catch(const std::exception & error){named=std::string(error.what()).find("TS_DSV41_EXPERT_HOST_CACHE_MB")!=std::string::npos;}
    check(named,"host victim cache refusal names its own setting");
}
}

int main() {
    try {
        configuration();
        constexpr int hidden=4, ff=4, experts=3, used=2, rows=3;
        auto * backend=ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(backend,2);
        auto * context=ggml_init({1024*1024,nullptr,true});
        check(backend && context,"outer CPU resources");
        auto * gate=ggml_new_tensor_3d(context,GGML_TYPE_F32,hidden,ff,experts);
        auto * up=ggml_new_tensor_3d(context,GGML_TYPE_F32,hidden,ff,experts);
        auto * down=ggml_new_tensor_3d(context,GGML_TYPE_F32,ff,hidden,experts);
        auto * input=ggml_new_tensor_3d(context,GGML_TYPE_F32,hidden,1,rows);
        auto * selected=ggml_new_tensor_2d(context,GGML_TYPE_I32,used,rows);
        auto * routes=ggml_new_tensor_3d(context,GGML_TYPE_F32,1,used,rows);
        auto * buffer=ggml_backend_alloc_ctx_tensors(context,backend);
        check(buffer!=nullptr,"outer tensor allocation");
        std::vector<float> g(hidden*ff*experts),u(g.size()),d(g.size());
        for (size_t i=0;i<g.size();++i) {g[i]=float(int(i%9)-4)*0.75f;u[i]=float(int(i%7)-3)*0.5f;d[i]=float(int(i%5)-2)*0.25f;}
        float x[rows*hidden]={1,2,-1,0.5f, -0.5f,1,2,1, 3,-2,1,-1};
        int32_t ids[rows*used]={2,0,1,2,0,0};
        float weights[rows*used]={0.25f,0.75f,0.5f,0.5f,0.125f,0.875f};
        ggml_backend_tensor_set(gate,g.data(),0,g.size()*4);ggml_backend_tensor_set(up,u.data(),0,u.size()*4);
        ggml_backend_tensor_set(down,d.data(),0,d.size()*4);ggml_backend_tensor_set(input,x,0,sizeof(x));
        ggml_backend_tensor_set(selected,ids,0,sizeof(ids));ggml_backend_tensor_set(routes,weights,0,sizeof(weights));
        tsg::HostMoeSegment descriptor;
        descriptor.layer=1;descriptor.hidden=hidden;descriptor.n_ff=ff;descriptor.num_experts=experts;
        descriptor.n_used=used;descriptor.seq_len=rows;descriptor.clamp_limit=2;
        std::string error;bool failed=false;
        {
            tsg_dsv41_cache::state bridge(descriptor,gate,up,down,backend,1048576,1,2,error,failed);
            auto * result=bridge.build(context,input,selected,routes);
            auto * graph=ggml_new_graph_custom(context,64,false);
            ggml_build_forward_expand(graph,result);
            auto * allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            check(allocator && ggml_gallocr_alloc_graph(allocator,graph),"outer graph allocation");
            float actual[hidden*rows];
            for (int repeat=0;repeat<3;++repeat) {
                check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS && error.empty(),"nested CPU fallback returns");
                ggml_backend_tensor_get(result,actual,0,sizeof(actual));
                for (int row=0;row<rows;++row) for (int h=0;h<hidden;++h) {
                    double expected=0;
                    for (int k=0;k<used;++k) {
                        const int expert=ids[row*used+k];double output=0;
                        for (int f=0;f<ff;++f) {
                            double gv=0,uv=0;
                            for (int j=0;j<hidden;++j) {gv+=g[(expert*ff+f)*hidden+j]*x[row*hidden+j];uv+=u[(expert*ff+f)*hidden+j]*x[row*hidden+j];}
                            gv=std::min(gv,2.0);uv=std::clamp(uv,-2.0,2.0);
                            output+=gv/(1+std::exp(-gv))*uv*d[(expert*hidden+h)*ff+f];
                        }
                        expected+=output*weights[row*used+k];
                    }
                    check(std::abs(actual[row*hidden+h]-expected)<2e-6,"CPU fallback clamp/routing matches scalar oracle");
                }
            }
            check(cache_calls==3,"cache attempted for every short span");
            cache_status=-1;
            check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS && failed && !error.empty(),"GPU cache failure latched");
            ggml_backend_tensor_get(result,actual,0,sizeof(actual));
            for (float value:actual) check(std::isnan(value),"failed GPU output poisoned");
            cache_status=0;error.clear();failed=false;
            check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS && error.empty(),"fallback reusable after reset");
            ids[0]=-1;ggml_backend_tensor_set(selected,ids,0,sizeof(ids));
            check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS && !error.empty(),"invalid route refused before GGML matmul");
            ggml_gallocr_free(allocator);
        }
        ids[0]=2;ggml_backend_tensor_set(selected,ids,0,sizeof(ids));error.clear();failed=false;
        {
            tsg_dsv41_prefetch::node prefetch(gate,up,down,experts,used,error,failed);
            auto * result=prefetch.build(context,selected);
            auto * graph=ggml_new_graph_custom(context,64,false);ggml_build_forward_expand(graph,result);
            auto * allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            check(allocator && ggml_gallocr_alloc_graph(allocator,graph),"prefetch identity allocation");
            check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS && !failed,"prefetch identity executes");
            int32_t actual[rows*used];ggml_backend_tensor_get(result,actual,0,sizeof(actual));
            check(std::equal(std::begin(ids),std::end(ids),std::begin(actual)),"prefetch preserves every route ID and duplicate");
            check(prefetch_calls==3,"all three expert stacks prefetched once");
            ids[0]=-1;ggml_backend_tensor_set(selected,ids,0,sizeof(ids));
            check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS && failed,"prefetch invalid ID is latched");
            ggml_backend_tensor_get(result,actual,0,sizeof(actual));
            check(std::all_of(std::begin(actual),std::end(actual),[](int32_t value){return value==0;}),"failed prefetch emits safe IDs");
            ggml_gallocr_free(allocator);
        }
        ggml_backend_buffer_free(buffer);ggml_free(context);ggml_backend_free(backend);
        std::cout<<checks<<" DeepSeek expert bridge checks passed\n";
        return 0;
    } catch (const std::exception & error) {std::cerr<<error.what()<<'\n';return 1;}
}
