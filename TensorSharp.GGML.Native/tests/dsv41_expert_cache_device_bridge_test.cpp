// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Linux CUDA integration fixture: use the production fused dispatcher, cache,
// bounded file reader and complete CPU fallback. No model or GGML patch needed.
#include "../dsv41_expert_cache.h"
#include "ggml-impl.h"
#include <fcntl.h>
#include <unistd.h>
#include <iostream>

namespace {
int checks = 0;
void require(bool condition, const char * message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
constexpr int hidden=256, ff=256, experts=16, used=2;

struct source {
    ggml_backend_t cpu = nullptr;
    ggml_context * context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * tensors[3]{};
    std::array<std::vector<uint8_t>,3> bytes;
    std::array<uint64_t,3> offsets{};
    int writer = -1;
    std::string path;
    std::shared_ptr<tsg_dsv41_pread::file> file;
    tsg::HostMoeSegment descriptor;

    void write(const uint8_t * data, size_t count, uint64_t offset) {
        while (count) {
            const auto n=::pwrite(writer,data,count,static_cast<off_t>(offset));
            if (n<0 && errno==EINTR) continue;
            require(n>0,"write expert fixture");
            data+=n;count-=size_t(n);offset+=uint64_t(n);
        }
    }
    void restore() { for (int part=0;part<3;++part) write(bytes[part].data(),bytes[part].size(),offsets[part]); }
    source() {
        cpu=ggml_backend_cpu_init();
        require(cpu!=nullptr,"CPU weight backend");
        ggml_backend_cpu_set_n_threads(cpu,2);
        context=ggml_init({ggml_tensor_overhead()*3,nullptr,true});
        require(context!=nullptr,"CPU weight metadata");
        for (auto & tensor:tensors) tensor=ggml_new_tensor_3d(context,GGML_TYPE_MXFP4,hidden,ff,experts);
        buffer=ggml_backend_alloc_ctx_tensors(context,cpu);
        require(buffer!=nullptr,"CPU weight buffer");
        std::vector<float> values(size_t(hidden)*ff*experts);
        for (int part=0;part<3;++part) {
            for (size_t i=0;i<values.size();++i) values[i]=0.04f*std::sin(0.5f+float(i%7919)*0.013f+part*0.7f);
            bytes[part].resize(ggml_nbytes(tensors[part]));
            require(ggml_quantize_chunk(GGML_TYPE_MXFP4,values.data(),bytes[part].data(),0,ff*experts,hidden,nullptr)
                ==bytes[part].size(),"MXFP4 fixture quantization");
            ggml_backend_tensor_set(tensors[part],bytes[part].data(),0,bytes[part].size());
        }
        char name[]="/tmp/tensorsharp-device-expert-XXXXXX";
        writer=::mkstemp(name);require(writer>=0,"create persistent expert fixture");path=name;
        uint64_t next=4096;
        for (int part=0;part<3;++part) { offsets[part]=next;next+=bytes[part].size()+4096; }
        restore();file=tsg_dsv41_pread::file::open(path);
        descriptor.hidden=hidden;descriptor.n_ff=ff;descriptor.num_experts=experts;descriptor.n_used=used;
        descriptor.clamp_limit=0.1f;descriptor.activation=0;
        descriptor.gate_data=tensors[0]->data;descriptor.gate_type=GGML_TYPE_MXFP4;
        descriptor.gate_ne0=hidden;descriptor.gate_ne1=ff;descriptor.gate_bytes=bytes[0].size();
        descriptor.up_data=tensors[1]->data;descriptor.up_type=GGML_TYPE_MXFP4;
        descriptor.up_ne0=hidden;descriptor.up_ne1=ff;descriptor.up_bytes=bytes[1].size();
        descriptor.down_data=tensors[2]->data;descriptor.down_type=GGML_TYPE_MXFP4;
        descriptor.down_ne0=ff;descriptor.down_ne1=hidden;descriptor.down_bytes=bytes[2].size();
        for (int part=0;part<3;++part) descriptor.expert_files[part]={file,offsets[part],bytes[part].size()};
    }
    ~source() {
        tsg::host_moe_expert_cache_on_drop(descriptor.gate_data);
        if (writer>=0) ::close(writer);
        if (!path.empty()) ::unlink(path.c_str());
        if (buffer) ggml_backend_buffer_free(buffer);
        if (context) ggml_free(context);
        if (cpu) ggml_backend_free(cpu);
    }
};

// Independent materialized resident graph, using the exact existing clamp,
// SwiGLU and ordered expert sum on the requested CPU or CUDA backend.
struct reference {
    ggml_backend_t backend;
    ggml_context * context = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * x=nullptr,* ids=nullptr,* routes=nullptr,* output=nullptr;
    reference(const source & weights,ggml_backend_t device,int rows):backend(device) {
        context=ggml_init({ggml_tensor_overhead()*64+ggml_graph_overhead_custom(64,false),nullptr,true});
        require(context!=nullptr,"resident reference metadata");
        ggml_tensor * tensors[3];
        for (auto & tensor:tensors) {tensor=ggml_new_tensor_3d(context,GGML_TYPE_MXFP4,hidden,ff,experts);ggml_set_input(tensor);ggml_set_output(tensor);}
        x=ggml_new_tensor_3d(context,GGML_TYPE_F32,hidden,1,rows);
        ids=ggml_new_tensor_2d(context,GGML_TYPE_I32,used,rows);
        routes=ggml_new_tensor_3d(context,GGML_TYPE_F32,1,used,rows);
        for (auto * tensor:{x,ids,routes}) ggml_set_input(tensor);
        auto * g=ggml_clamp(context,ggml_mul_mat_id(context,tensors[0],x,ids),-INFINITY,0.1f);
        auto * u=ggml_clamp(context,ggml_mul_mat_id(context,tensors[1],x,ids),-0.1f,0.1f);
        auto * down=ggml_mul_mat_id(context,tensors[2],ggml_swiglu_split(context,g,u),ids);ggml_set_output(down);
        auto * weighted=ggml_mul(context,down,routes);
        auto * sum=ggml_view_2d(context,weighted,hidden,rows,weighted->nb[2],0);
        for (int k=1;k<used;++k) sum=ggml_add(context,sum,ggml_view_2d(context,weighted,hidden,rows,weighted->nb[2],size_t(k)*weighted->nb[1]));
        output=ggml_cont(context,sum);ggml_set_output(output);
        graph=ggml_new_graph_custom(context,64,false);ggml_build_forward_expand(graph,output);
        allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(allocator && ggml_gallocr_alloc_graph(allocator,graph),"resident reference allocation");
        for (int part=0;part<3;++part) ggml_backend_tensor_set(tensors[part],weights.bytes[part].data(),0,weights.bytes[part].size());
    }
    ~reference() {if(allocator)ggml_gallocr_free(allocator);if(context)ggml_free(context);}
    std::vector<float> compute(const std::vector<float> & input,const std::vector<int32_t> & selected,const std::vector<float> & probabilities) {
        ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float));
        ggml_backend_tensor_set(ids,selected.data(),0,selected.size()*sizeof(int32_t));
        ggml_backend_tensor_set(routes,probabilities.data(),0,probabilities.size()*sizeof(float));
        require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"resident reference compute");
        std::vector<float> result(input.size());ggml_backend_tensor_get(output,result.data(),0,result.size()*sizeof(float));return result;
    }
};

struct bridge {
    ggml_backend_t backend,wrapped;
    ggml_context * inputs=nullptr,* context=nullptr;
    ggml_backend_buffer_t buffer=nullptr;
    ggml_gallocr_t allocator=nullptr;
    ggml_cgraph * graph=nullptr;
    ggml_tensor * x=nullptr,* ids=nullptr,* routes=nullptr,* output=nullptr;
    std::string error;bool failed=false;
    std::unique_ptr<tsg_dsv41_cache::state> state;
    bridge(source & weights,ggml_backend_t device,ggml_backend_t fused,int rows,size_t budget):backend(device),wrapped(fused) {
        inputs=ggml_init({ggml_tensor_overhead()*3,nullptr,true});
        context=ggml_init({ggml_tensor_overhead()*4+ggml_graph_overhead_custom(64,false),nullptr,true});
        require(inputs && context,"device bridge metadata");
        x=ggml_new_tensor_3d(inputs,GGML_TYPE_F32,hidden,1,rows);
        ids=ggml_new_tensor_2d(inputs,GGML_TYPE_I32,used,rows);
        routes=ggml_new_tensor_3d(inputs,GGML_TYPE_F32,1,used,rows);
        for (auto * tensor:{x,ids,routes}) ggml_set_input(tensor);
        buffer=ggml_backend_alloc_ctx_tensors(inputs,backend);require(buffer!=nullptr,"device bridge inputs");
        auto descriptor=weights.descriptor;descriptor.seq_len=rows;
        state=std::make_unique<tsg_dsv41_cache::state>(descriptor,weights.tensors[0],weights.tensors[1],weights.tensors[2],
            backend,budget,1,2,error,failed);
        output=state->build(context,x,ids,routes,true);ggml_set_output(output);
        ggml_custom_op_params params;std::memcpy(&params,output->op_params,sizeof(params));
        require(static_cast<tsg_dsv4_fused_desc *>(params.userdata)->kind==TSG_DSV41_CACHED_MOE,"production device descriptor kind");
        graph=ggml_new_graph_custom(context,64,false);ggml_build_forward_expand(graph,output);
        allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(allocator && ggml_gallocr_alloc_graph(allocator,graph),"device bridge output allocation");
        require(ggml_backend_buffer_get_type(output->buffer)==ggml_backend_get_default_buffer_type(backend),"result stays on owning device");
    }
    ~bridge() {state.reset();if(allocator)ggml_gallocr_free(allocator);if(buffer)ggml_backend_buffer_free(buffer);if(context)ggml_free(context);if(inputs)ggml_free(inputs);}
    void reset_failure() {error.clear();failed=false;tsg::g_backend_compute_failed.store(false);}
    std::vector<float> compute(const std::vector<float> & input,const std::vector<int32_t> & selected,const std::vector<float> & probabilities) {
        ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float));
        ggml_backend_tensor_set(ids,selected.data(),0,selected.size()*sizeof(int32_t));
        ggml_backend_tensor_set(routes,probabilities.data(),0,probabilities.size()*sizeof(float));
        require(ggml_backend_graph_compute(wrapped,graph)==GGML_STATUS_SUCCESS,"production fused graph compute");
        ggml_backend_synchronize(backend);
        std::vector<float> result(input.size());ggml_backend_tensor_get(output,result.data(),0,result.size()*sizeof(float));return result;
    }
};
void equal(const std::vector<float> & actual,const std::vector<float> & expected,const char * message) {
    require(actual.size()==expected.size() && std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float))==0,message);
}
void poisoned(const bridge & node,const std::vector<float> & values) {
    require(node.failed && !node.error.empty(),"device failure latched");
    require(std::all_of(values.begin(),values.end(),[](float value){return std::isnan(value);}),"device failure poisons entire span");
}

void run(ggml_backend_t backend,ggml_backend_t fused) {
    source weights;
    const size_t stride=weights.bytes[0].size()/experts;
    const size_t budget=(size_t(1)<<20)+3*stride*used+(3*stride)/2;
    for (int rows:{1,3,8}) {
        bridge node(weights,backend,fused,rows,budget);
        // The production helper replays each row through its scalar GPU graph.
        reference resident(weights,backend,1);
        std::vector<float> x(size_t(rows)*hidden),probabilities(size_t(rows)*used);
        std::vector<int32_t> ids(size_t(rows)*used);
        for (int row=0;row<rows;++row) {
            for(int i=0;i<hidden;++i)x[size_t(row)*hidden+i]=4.0f*std::sin(0.13f*i+row);
            ids[row*used]=(row*3)%experts;ids[row*used+1]=(row*3+1)%experts;
            probabilities[row*used]=0.6f;probabilities[row*used+1]=0.4f;
        }
        std::vector<float> expected;
        for (int row=0;row<rows;++row) {
            const auto values=resident.compute(std::vector<float>(x.begin()+row*hidden,x.begin()+(row+1)*hidden),
                std::vector<int32_t>(ids.begin()+row*used,ids.begin()+(row+1)*used),
                std::vector<float>(probabilities.begin()+row*used,probabilities.begin()+(row+1)*used));
            expected.insert(expected.end(),values.begin(),values.end());
        }
        equal(node.compute(x,ids,probabilities),expected,"device bridge rows differ from exact resident scalar graph");
        require(!node.failed,"valid device rows accepted");
        equal(node.compute(x,ids,probabilities),expected,"warm/evicted device bridge rows changed arithmetic");

        auto invalid=ids;invalid.back()=experts;
        poisoned(node,node.compute(x,invalid,probabilities));node.reset_failure();
        auto nonfinite=x;nonfinite.back()=std::numeric_limits<float>::quiet_NaN();
        poisoned(node,node.compute(nonfinite,ids,probabilities));node.reset_failure();
        auto bad_route=probabilities;bad_route.back()=INFINITY;
        poisoned(node,node.compute(x,ids,bad_route));node.reset_failure();
        equal(node.compute(x,ids,probabilities),expected,"device bridge did not recover after failure reset");

        if (rows==1) {
            ids={0,1};(void)node.compute(x,ids,probabilities);
            ids={2,2};
            require(stride==34816,"64KiB fixture must force multiple staging waves");
            require(::ftruncate(weights.writer,static_cast<off_t>(weights.offsets[2]+3*stride-1))==0,"truncate device expert down source");
            poisoned(node,node.compute(x,ids,probabilities));node.reset_failure();
            ids={0,1};
            equal(node.compute(x,ids,probabilities),resident.compute(x,ids,probabilities),"device source EOF retained a stale victim tag");
            weights.restore();ids={2,2};
            equal(node.compute(x,ids,probabilities),resident.compute(x,ids,probabilities),"device source restore/retry differs");
        }

        // A workspace-only quota cannot fit experts. The production device
        // hook must use the complete CPU graph, then upload the complete span.
        bridge declined(weights,backend,fused,rows,size_t(1)<<20);
        reference cpu_reference(weights,weights.cpu,rows);
        equal(declined.compute(x,ids,probabilities),cpu_reference.compute(x,ids,probabilities),
            "device cache decline does not match complete independent CPU fallback");
        require(!declined.failed,"complete device-to-CPU fallback accepted");
    }
    tsg::host_moe_expert_cache_release_backend(backend);
}
}

int main() {
    try {
        setenv("TS_DSV41_EXPERT_CACHE_MB","2",1);
        setenv("TS_DSV41_EXPERT_CACHE_PREAD","1",1);
        setenv("TS_DSV41_EXPERT_CACHE_PREAD_STAGING_KB","64",1);
        setenv("TS_DSV41_EXPERT_HOST_CACHE_MB","0",1);
        setenv("TS_DSV41_EXPERT_CACHE_DIAGNOSTICS","1",1);
        const int devices=ggml_backend_cuda_get_device_count();
        if(devices<1){std::cerr<<"SKIP: CUDA device bridge requires NVIDIA hardware\n";return 77;}
        for(int device=0;device<std::min(devices,2);++device){
            auto * backend=ggml_backend_cuda_init(device);require(backend!=nullptr,"owning CUDA backend");
            auto * fused=tsg_dsv4_fused_backend_init(backend);require(fused!=nullptr,"owning fused CUDA backend");
            try{run(backend,fused);}catch(...){tsg::host_moe_expert_cache_release_backend(backend);ggml_backend_free(fused);ggml_backend_free(backend);throw;}
            ggml_backend_free(fused);ggml_backend_free(backend);
        }
        tsg::host_moe_expert_cache_release();
        std::cout<<checks<<" device bridge checks passed: MXFP4 scalar parity, finite/ID failure, file EOF/retry, complete CPU fallback\n";
        return 0;
    } catch(const std::exception & error){tsg::host_moe_expert_cache_release();std::cerr<<error.what()<<'\n';return 1;}
}
