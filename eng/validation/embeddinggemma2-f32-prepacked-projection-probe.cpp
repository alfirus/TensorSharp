// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Validation-only component. Existing native and upstream sources are untouched.
#include "embeddinggemma2-f32-prepacked-probe.h"
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error This validation tool requires finite IEEE arithmetic without fast-math.
#endif
#ifndef TSG_PREPACKED_WITH_GGML
#define TSG_PREPACKED_WITH_GGML 0
#endif
#if TSG_PREPACKED_WITH_GGML
#include "embedding_linux_blas.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "gguf.h"
#endif
#include <array>
#include <atomic>
#include <cerrno>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace {
namespace packed = tsg_validation::prepacked_f32;
using timer = std::chrono::steady_clock;
void require(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
std::string quote(const std::string & text) {
    std::string out = "\"";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') { out += '\\'; out += char(c); }
        else if (c < 32) { char escaped[8]; std::snprintf(escaped, sizeof(escaped), "\\u%04x", c); out += escaped; }
        else out += char(c);
    }
    return out + '"';
}
std::string number(double v) {
    if (!std::isfinite(v)) return "null";
    std::ostringstream out; out << std::setprecision(17) << v; return out.str();
}
const char * boolean(bool value) { return value ? "true" : "false"; }
uint64_t fingerprint(const void * data, size_t bytes) {
    auto * p = static_cast<const uint8_t *>(data); uint64_t result = 14695981039346656037ULL;
    for (size_t i = 0; i < bytes; ++i) result = (result ^ p[i]) * 1099511628211ULL;
    return result;
}

// Portable SHA256 binds original checkpoint matrix bytes. The outer execution
// wrapper must separately bind model, executable, sources, provider and runtime.
std::string sha256(const void * source, size_t size) {
    static constexpr uint32_t constants[] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    require(size <= std::numeric_limits<uint64_t>::max()/8 && size <= std::numeric_limits<size_t>::max()-72,"SHA input too large");
    std::vector<uint8_t> data((size+9+63)/64*64,0);
    if(size) std::memcpy(data.data(),source,size);
    data[size]=128; const uint64_t bits=uint64_t(size)*8;
    for(int i=0;i<8;++i) data[data.size()-1-i]=uint8_t(bits>>(8*i));
    std::array<uint32_t,8> h{{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}};
    const auto rotate=[](uint32_t x,unsigned shift){return (x>>shift)|(x<<(32-shift));};
    for(size_t offset=0;offset<data.size();offset+=64){
        uint32_t words[64];
        for(int i=0;i<16;++i){const auto * p=data.data()+offset+i*4;words[i]=(uint32_t(p[0])<<24)|(uint32_t(p[1])<<16)|(uint32_t(p[2])<<8)|uint32_t(p[3]);}
        for(int i=16;i<64;++i){const uint32_t a=words[i-15],b=words[i-2];words[i]=words[i-16]+(rotate(a,7)^rotate(a,18)^(a>>3))+words[i-7]+(rotate(b,17)^rotate(b,19)^(b>>10));}
        auto a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],j=h[7];
        for(int i=0;i<64;++i){const uint32_t t1=j+(rotate(e,6)^rotate(e,11)^rotate(e,25))+((e&f)^((~e)&g))+constants[i]+words[i];const uint32_t t2=(rotate(a,2)^rotate(a,13)^rotate(a,22))+((a&b)^(a&c)^(b&c));j=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=j;
    }
    std::ostringstream out;out<<std::hex<<std::setfill('0');
    for(auto word:h){out<<std::setw(8)<<word;}
    return out.str();
}

struct guarded {
    static constexpr size_t guard=64;
    static constexpr uint32_t canary_bits=0x7fc12345;
    std::vector<float> allocation;
    float * base=nullptr;
    size_t count,storage;
    explicit guarded(size_t elements):allocation(elements+2*guard+15),count(elements),storage(elements+2*guard){
        const auto address=reinterpret_cast<uintptr_t>(allocation.data());base=reinterpret_cast<float *>((address+63)&~uintptr_t(63));
        require(base+storage<=allocation.data()+allocation.size(),"Aligned guard storage overflow");
        for(size_t i=0;i<storage;++i)std::memcpy(base+i,&canary_bits,4);
    }
    guarded(const guarded &)=delete;
    guarded & operator=(const guarded &)=delete;
    float * data(){return base+guard;}
    const float * data()const{return base+guard;}
    uint64_t hash()const{return fingerprint(base,storage*4);}
    void reset(){for(size_t i=0;i<count;++i)std::memcpy(data()+i,&canary_bits,4);}
    bool guards()const{
        for(size_t i=0;i<guard;++i)if(std::memcmp(base+i,&canary_bits,4))return false;
        for(size_t i=guard+count;i<storage;++i)if(std::memcmp(base+i,&canary_bits,4))return false;
        return true;
    }
};
struct options{
    std::string output,model,library,baseline="ggml-f32";
    int threads=8,warmup=3,repeats=20;
    double minimum_seconds=.1;
    bool host=false,benchmark=false;
};
int integer(const std::string & value){size_t used=0;int out=std::stoi(value,&used);require(used==value.size(),"Incomplete integer");return out;}
options parse(int argc,char ** argv){
    options out;std::set<std::string> seen;
    for(int i=1;i<argc;++i){
        const std::string name=argv[i];require(seen.insert(name).second,"Duplicate option");
        if(name=="--host-controls"){out.host=true;continue;}
        if(name=="--benchmark"){out.benchmark=true;continue;}
        require(i+1<argc,"Option needs a value");const std::string value=argv[++i];
        if(name=="--output")out.output=value;
        else if(name=="--model")out.model=value;
        else if(name=="--library")out.library=value;
        else if(name=="--baseline")out.baseline=value;
        else if(name=="--threads")out.threads=integer(value);
        else if(name=="--warmup")out.warmup=integer(value);
        else if(name=="--repeats")out.repeats=integer(value);
        else if(name=="--minimum-seconds"){size_t used=0;out.minimum_seconds=std::stod(value,&used);require(used==value.size(),"Incomplete seconds");}
        else throw std::runtime_error("Unknown option: "+name);
    }
    require(out.threads==7||out.threads==8,"Only7/8worker policies are supported");
    require(out.warmup>=1&&out.warmup<=100&&out.repeats>=1&&out.repeats<=10000&&std::isfinite(out.minimum_seconds)&&out.minimum_seconds>=0&&out.minimum_seconds<=10,"Invalid phase bounds");
    require(out.baseline=="ggml-f32"||out.baseline=="blas-policy","Unknown baseline");
    require(out.library.empty()||out.baseline=="blas-policy","Explicit provider requires the optional BLAS baseline");
    require(out.host||(!out.output.empty()&&!out.model.empty()),"Require --output and --model, or --host-controls");
    require(!out.host||(!out.benchmark&&out.output.empty()&&out.model.empty()),"Host mode accepts no execution model/output");
    return out;
}
struct report_file{
    int fd=-1;
    explicit report_file(const std::string & path){fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0644);require(fd>=0,"Require fresh output file");}
    ~report_file(){if(fd>=0)::close(fd);}
    void write(const std::string & value){
        size_t done=0;
        while(done<value.size()){const auto n=::write(fd,value.data()+done,value.size()-done);if(n<0&&errno==EINTR)continue;require(n>0,"Report write failed");done+=size_t(n);}
        const int result=::close(fd);fd=-1;require(result==0,"Report close failed");
    }
};
size_t host_controls(){
    size_t checks=0;auto check=[&](bool value){require(value,"Host control failed");++checks;};
    check(sha256("",0)=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    check(sha256("abc",3)=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    for(int k:{1,17,31,33})for(int n:{1,7,8,9,15,16,17}){
        const packed::dimensions s{k,32,n};guarded w(size_t(k)*32),p(size_t(k)*32),x(size_t(k)*n),y(size_t(n)*32);
        for(size_t i=0;i<w.count;++i){const uint32_t bits=uint32_t(i)*2654435761U;std::memcpy(w.data()+i,&bits,4);}
        const auto wh=w.hash();
        check(packed::pack(s,{w.data(),w.count},{p.data(),p.count}));
        check(packed::pack_bits_equal(s,{w.data(),w.count},{p.data(),p.count}));
        check(w.hash()==wh&&w.guards()&&p.guards());
        check(packed::admissible(s,{p.data(),p.count},{x.data(),x.count},{y.data(),y.count}));
        for(int workers:{7,8}){
            std::vector<unsigned> coverage(y.count,0);
            for(int worker=0;worker<workers;++worker)for(int block=packed::block_begin(s.m/16,worker,workers);block<packed::block_end(s.m/16,worker,workers);++block)
                for(int row=0;row<n;row+=8)for(int tail=0;tail<std::min(8,n-row);++tail)for(int channel=0;channel<16;++channel)++coverage[size_t(row+tail)*s.m+block*16+channel];
            check(std::all_of(coverage.begin(),coverage.end(),[](unsigned v){return v==1;}));
        }
        check(!packed::pack(s,{w.data(),w.count},{w.data(),w.count}));
        check(!packed::admissible(s,{p.data(),p.count},{x.data(),x.count},{x.data(),y.count}));
        check(!packed::admissible({k,31,n},{p.data(),p.count},{x.data(),x.count},{y.data(),y.count}));
        check(!packed::admissible(s,{p.data(),p.count-1},{x.data(),x.count},{y.data(),y.count}));
        const auto before_y=y.hash();
        check(!packed::project_partition(s,{p.data(),p.count},{x.data(),x.count},{y.data(),y.count},-1,8));
        check(!packed::project_partition(s,{p.data(),p.count},{x.data(),x.count},{y.data(),y.count},0,9));
        check(!packed::project_partition(s,{p.data(),p.count},{x.data(),x.count},{y.data(),y.count},0,0));
        check(y.hash()==before_y);
        uint32_t bits;std::memcpy(&bits,p.data(),4);bits^=1;std::memcpy(p.data(),&bits,4);
        check(!packed::pack_bits_equal(s,{w.data(),w.count},{p.data(),p.count}));
        if(!packed::supported()){
            const auto yh=y.hash();check(!packed::project_partition(s,{p.data(),p.count},{x.data(),x.count},{y.data(),y.count},0,8));check(y.hash()==yh);
        }
    }
    check(!packed::span(reinterpret_cast<void *>(std::numeric_limits<uintptr_t>::max()-3),2));
    check(!packed::geometry({1,16,8193}));check(!packed::geometry({0,16,1}));
    for(int workers:{7,8}){
        std::vector<unsigned> coverage(512,0);bool all_nonempty=true;
        for(int worker=0;worker<workers;++worker){
            const int begin=packed::block_begin(32,worker,workers),end=packed::block_end(32,worker,workers);
            all_nonempty&=begin<end;
            for(int block=begin;block<end;++block)for(int lane=0;lane<16;++lane)++coverage[size_t(block)*16+lane];
        }
        check(all_nonempty&&std::all_of(coverage.begin(),coverage.end(),[](unsigned v){return v==1;}));
    }
    return checks;
}
struct quality{
    bool passed=true,finite=true,canaries=true,immutable=true,pack_bits=true,fma_order_bits=true,fma_checked=false;
    uint64_t sampled=0,full_elements=0;
    double max_error=0,max_ratio=0,baseline_max_error=0;
    void retain(const quality & q){
        passed&=q.passed;finite&=q.finite;canaries&=q.canaries;immutable&=q.immutable;pack_bits&=q.pack_bits;fma_order_bits&=q.fma_order_bits;
        fma_checked|=q.fma_checked;
        sampled+=q.sampled;full_elements+=q.full_elements;
        max_error=std::max(max_error,q.max_error);max_ratio=std::max(max_ratio,q.max_ratio);baseline_max_error=std::max(baseline_max_error,q.baseline_max_error);
    }
    std::string json()const{
        std::ostringstream out;out<<"{\"passed\":"<<boolean(passed)<<",\"finite\":"<<boolean(finite)<<",\"canaries\":"<<boolean(canaries)<<",\"immutable\":"<<boolean(immutable)<<",\"pack_bits\":"<<boolean(pack_bits)<<",\"sequential_fma_oracle_bits\":"<<(fma_checked?boolean(fma_order_bits):"null")<<",\"double_samples\":"<<sampled<<",\"full_output_elements\":"<<full_elements<<",\"max_absolute_error\":"<<number(max_error)<<",\"max_error_over_original_bound\":"<<number(max_ratio)<<",\"full_baseline_max_absolute_error\":"<<number(baseline_max_error)<<'}';return out.str();
    }
};
size_t quality_controls(){
    size_t checks=0;
    const auto check=[&](bool value){require(value,"Quality retention control failed");++checks;};
    quality failed;failed.passed=false;failed.finite=false;failed.max_error=7;failed.max_ratio=2;failed.sampled=3;failed.full_elements=16;
    quality later;later.sampled=2;later.full_elements=8;later.max_error=.1;later.max_ratio=.1;
    quality all;all.retain(failed);all.retain(later);all.retain(later);
    check(!all.passed&&!all.finite);
    check(all.max_error==7&&all.max_ratio==2);
    check(all.sampled==7&&all.full_elements==32);
    check(number(std::numeric_limits<double>::infinity())=="null"&&number(std::numeric_limits<double>::quiet_NaN())=="null");
    return checks;
}

#if TSG_PREPACKED_WITH_GGML
struct session{
    ggml_backend_t backend=nullptr;ggml_threadpool_t pool=nullptr;
    explicit session(int workers){
        backend=ggml_backend_cpu_init();require(backend,"CPU backend unavailable");
        auto parameters=ggml_threadpool_params_default(workers);pool=ggml_threadpool_new(&parameters);
        if(!pool){ggml_backend_free(backend);backend=nullptr;throw std::runtime_error("Threadpool unavailable");}
        ggml_backend_cpu_set_n_threads(backend,workers);ggml_backend_cpu_set_threadpool(backend,pool);
    }
    ~session(){if(backend)ggml_backend_free(backend);if(pool)ggml_threadpool_free(pool);}
};
struct checkpoint{
    gguf_context * metadata=nullptr;std::ifstream file;
    explicit checkpoint(const std::string & path):file(path,std::ios::binary){
        metadata=gguf_init_from_file(path.c_str(),{true,nullptr});
        if(!metadata||!file){if(metadata)gguf_free(metadata);metadata=nullptr;throw std::runtime_error("Model unavailable");}
    }
    ~checkpoint(){if(metadata)gguf_free(metadata);}
    std::vector<uint8_t> load(const std::string & name,int k,int m){
        const int64_t id=gguf_find_tensor(metadata,name.c_str());require(id>=0,"Checkpoint weight missing");
        const auto * shape=gguf_get_tensor_ne(metadata,id);require(gguf_get_tensor_type(metadata,id)==GGML_TYPE_Q8_0&&shape[0]==k&&shape[1]==m&&shape[2]==1&&shape[3]==1,"OriginalQ8 geometry required");
        std::vector<uint8_t> bytes(gguf_get_tensor_size(metadata,id));file.clear();file.seekg(std::streamoff(gguf_get_data_offset(metadata)+gguf_get_tensor_offset(metadata,id)));
        require(bool(file.read(reinterpret_cast<char *>(bytes.data()),std::streamsize(bytes.size()))),"Truncated originalQ8 checkpoint");return bytes;
    }
};
struct matrix_storage{
    guarded weights,packed_weights;std::vector<uint8_t> original;
    bool prepared=false;double cold_ms=0;
    matrix_storage(int k,int m):weights(size_t(k)*m),packed_weights(size_t(k)*m){}
};
struct problem{
    packed::dimensions shape;std::shared_ptr<matrix_storage> matrix;
    guarded & weights;guarded & packed_weights;guarded inputs;std::vector<uint8_t> & original;
    std::string name,original_sha;uint64_t wh=0,ph=0,xh=0,oh=0;double pack_ms=0;bool synthetic=false;
    problem(packed::dimensions s,const std::string & label,std::shared_ptr<matrix_storage> shared=nullptr):
        shape(s),matrix(shared?std::move(shared):std::make_shared<matrix_storage>(s.k,s.m)),weights(matrix->weights),
        packed_weights(matrix->packed_weights),inputs(size_t(s.k)*s.n),original(matrix->original),name(label){}
    void finish(){
        if(!matrix->prepared){
            const auto begin=timer::now();require(packed::pack(shape,{weights.data(),weights.count},{packed_weights.data(),packed_weights.count}),"Pack rejected");
            matrix->cold_ms=std::chrono::duration<double,std::milli>(timer::now()-begin).count();matrix->prepared=true;
        }
        pack_ms=matrix->cold_ms;
        require(packed::pack_bits_equal(shape,{weights.data(),weights.count},{packed_weights.data(),packed_weights.count}),"Pack bits differ");
        for(size_t i=0;i<weights.count;++i)require(std::isfinite(weights.data()[i]),"Nonfinite dequantized weight");
        wh=weights.hash();ph=packed_weights.hash();xh=inputs.hash();oh=fingerprint(original.data(),original.size());original_sha=sha256(original.data(),original.size());
    }
    bool immutable()const{return weights.hash()==wh&&packed_weights.hash()==ph&&inputs.hash()==xh&&fingerprint(original.data(),original.size())==oh;}
};
void fill_inputs(problem & p,uint32_t seed){
    for(size_t i=0;i<p.inputs.count;++i){seed=seed*1664525U+1013904223U;p.inputs.data()[i]=float(int(seed>>8)-8388608)/8388608.f;}
    for(int k=0;k<p.shape.k;++k)p.inputs.data()[k]=((k&1)?-1.f:1.f)*(1.f+float(k%31)*.00017f);
}
struct state{
    packed::dimensions shape;bool instrument=false;std::atomic<bool> error{false};
    tsg_embeddings::linux_blas_plan * blas=nullptr;
    std::array<uint64_t,8> tiles{},elements{},calls{};
    std::array<int,8> rounding{};
    explicit state(packed::dimensions s):shape(s){}
};
void projection(ggml_tensor * y,int ith,int nth,void * userdata){
    auto & s=*static_cast<state *>(userdata);const auto * w=y->src[0];const auto * x=y->src[1];
    if(s.blas){
        const tsg_embeddings::blas_const_matrix weights{static_cast<const float *>(w->data),size_t(s.shape.k)*s.shape.m,s.shape.m,s.shape.k,s.shape.k};
        const tsg_embeddings::blas_const_matrix input{static_cast<const float *>(x->data),size_t(s.shape.k)*s.shape.n,s.shape.n,s.shape.k,s.shape.k};
        const tsg_embeddings::blas_matrix output{static_cast<float *>(y->data),size_t(s.shape.m)*s.shape.n,s.shape.n,s.shape.m,s.shape.m};
        const auto axis=s.shape.n>=128?tsg_embeddings::blas_partition::token_rows:tsg_embeddings::blas_partition::output_channels;
        if(!s.blas->try_partition(weights,input,output,ith,nth,axis))s.error.store(true);
        return;
    }
    uint64_t tiles=0,elements=0;
    const bool okay=packed::project_partition(s.shape,{static_cast<const float *>(w->data),size_t(s.shape.k)*s.shape.m},{static_cast<const float *>(x->data),size_t(s.shape.k)*s.shape.n},{static_cast<float *>(y->data),size_t(s.shape.m)*s.shape.n},ith,nth,s.instrument?&tiles:nullptr,s.instrument?&elements:nullptr);
    if(!okay){s.error.store(true);return;}
    if(s.instrument){s.tiles[size_t(ith)]=tiles;s.elements[size_t(ith)]=elements;s.calls[size_t(ith)]=1;s.rounding[size_t(ith)]=std::fegetround();}
}
struct route{
    guarded output;state parameters;ggml_context * context=nullptr;ggml_backend_graph_plan_t plan=nullptr;
    ggml_backend_buffer_t wb=nullptr,xb=nullptr,yb=nullptr;ggml_backend_t backend;bool owned;
    route(session & owner,problem & p,int workers,bool candidate,tsg_embeddings::linux_blas_plan * blas):output(size_t(p.shape.m)*p.shape.n),parameters(p.shape),backend(owner.backend),owned(candidate){
        try{
            context=ggml_init({1024*1024,nullptr,true});require(context,"Metadata allocation failed");
            auto * w=owned?ggml_new_tensor_3d(context,GGML_TYPE_F32,16,p.shape.k,p.shape.m/16):ggml_new_tensor_2d(context,GGML_TYPE_F32,p.shape.k,p.shape.m);
            auto * x=ggml_new_tensor_2d(context,GGML_TYPE_F32,p.shape.k,p.shape.n);ggml_tensor * y=nullptr;
            if(owned||blas){parameters.blas=blas;ggml_tensor * args[]={w,x};y=ggml_custom_4d(context,GGML_TYPE_F32,p.shape.m,p.shape.n,1,1,args,2,projection,workers,&parameters);}
            else y=ggml_mul_mat(context,w,x);
            require(y,"Selected baseline rejected; no hidden fallback");ggml_set_input(w);ggml_set_input(x);ggml_set_output(y);
            auto & weight=owned?p.packed_weights:p.weights;
            wb=ggml_backend_cpu_buffer_from_ptr(weight.base,weight.storage*4);xb=ggml_backend_cpu_buffer_from_ptr(p.inputs.base,p.inputs.storage*4);yb=ggml_backend_cpu_buffer_from_ptr(output.base,output.storage*4);
            require(wb&&xb&&yb,"Buffer wrapper failed");
            require(ggml_backend_tensor_alloc(wb,w,weight.data())==GGML_STATUS_SUCCESS&&ggml_backend_tensor_alloc(xb,x,p.inputs.data())==GGML_STATUS_SUCCESS&&ggml_backend_tensor_alloc(yb,y,output.data())==GGML_STATUS_SUCCESS,"Guarded binding failed");
            auto * graph=ggml_new_graph_custom(context,16,false);ggml_build_forward_expand(graph,y);require(ggml_backend_supports_op(backend,y),"Backend rejected graph");plan=ggml_backend_graph_plan_create(backend,graph);require(plan,"Plan allocation failed");
        }catch(...){release();throw;}
    }
    void release(){
        if(plan){ggml_backend_graph_plan_free(backend,plan);plan=nullptr;}
        ggml_backend_buffer_free(yb);yb=nullptr;ggml_backend_buffer_free(xb);xb=nullptr;ggml_backend_buffer_free(wb);wb=nullptr;
        if(context){ggml_free(context);context=nullptr;}
    }
    ~route(){release();}
    void compute(){require(ggml_backend_graph_plan_compute(backend,plan)==GGML_STATUS_SUCCESS,"Plan compute failed");require(!parameters.error.load(),"Owned callback refused execution");}
};
quality check(const problem & p,const route & r,int workers,const route * baseline=nullptr){
    quality out;out.fma_checked=r.owned;out.canaries=r.output.guards()&&p.weights.guards()&&p.packed_weights.guards()&&p.inputs.guards();out.immutable=p.immutable();out.pack_bits=packed::pack_bits_equal(p.shape,{p.weights.data(),p.weights.count},{p.packed_weights.data(),p.packed_weights.count});
    for(size_t i=0;i<r.output.count;++i){
        const double actual=r.output.data()[i];out.finite&=std::isfinite(actual);++out.full_elements;
        if(baseline){const double error=std::abs(actual-baseline->output.data()[i]);if(!std::isfinite(error))out.finite=false;else out.baseline_max_error=std::max(out.baseline_max_error,error);}
    }
    std::set<int> rows{0,1,p.shape.n/2,p.shape.n-1},columns{0,1,7,15,16,p.shape.m/2,p.shape.m-1};
    for(int worker=1;worker<workers;++worker)for(int offset:{-1,0,1}){rows.insert(p.shape.n*worker/workers+offset);columns.insert(p.shape.m*worker/workers+offset);columns.insert(packed::block_begin(p.shape.m/16,worker,workers)*16+offset);}
    for(int boundary:{7,8,9,15,16,17})rows.insert(boundary);
    for(int boundary:{31,32,33,47,48,49})columns.insert(boundary);
    uint32_t seed=1729;for(int i=0;i<32;++i){seed=seed*1664525U+1013904223U;columns.insert(int(seed%p.shape.m));}
    if(p.synthetic){
        for(int i=0;i<p.shape.n;++i){rows.insert(i);}
        for(int i=0;i<p.shape.m;++i){columns.insert(i);}
    }
    for(int row:rows)for(int column:columns){
        if(row<0||row>=p.shape.n||column<0||column>=p.shape.m)continue;
        double expected=0,absolute=0;
        for(int k=0;k<p.shape.k;++k){const double product=double(p.weights.data()[size_t(column)*p.shape.k+k])*p.inputs.data()[size_t(row)*p.shape.k+k];expected+=product;absolute+=std::abs(product);}
        const double error=std::abs(double(r.output.data()[size_t(row)*p.shape.m+column])-expected),bound=std::max(2e-5,1e-6*absolute);++out.sampled;
        if(r.owned){
            float fma=0;
            for(int k=0;k<p.shape.k;++k)fma=std::fma(p.weights.data()[size_t(column)*p.shape.k+k],p.inputs.data()[size_t(row)*p.shape.k+k],fma);
            out.fma_order_bits&=std::memcmp(&fma,r.output.data()+size_t(row)*p.shape.m+column,sizeof(float))==0;
        }
        if(!std::isfinite(error)){out.finite=false;out.passed=false;}else{out.max_error=std::max(out.max_error,error);out.max_ratio=std::max(out.max_ratio,error/bound);out.passed&=error<=bound;}
    }
    out.passed&=out.finite&&out.canaries&&out.immutable&&out.pack_bits&&out.fma_order_bits;return out;
}
std::string worker_proof(route & r,const problem & p,int workers){
    r.parameters.instrument=true;r.compute();r.parameters.instrument=false;
    uint64_t total_tiles=0,total_elements=0,mask=0;
    for(int worker=0;worker<workers;++worker){
        const uint64_t blocks=uint64_t(packed::block_end(p.shape.m/16,worker,workers)-packed::block_begin(p.shape.m/16,worker,workers));
        require(r.parameters.calls[size_t(worker)]==1&&r.parameters.tiles[size_t(worker)]==blocks*uint64_t((p.shape.n+7)/8)&&r.parameters.elements[size_t(worker)]==blocks*uint64_t(p.shape.n)*16&&r.parameters.rounding[size_t(worker)]==FE_TONEAREST,"Incomplete worker coverage or unexpected rounding");
        total_tiles+=r.parameters.tiles[size_t(worker)];total_elements+=r.parameters.elements[size_t(worker)];if(blocks)mask|=uint64_t(1)<<worker;
    }
    require(total_elements==r.output.count,"Output coverage mismatch");
    std::ostringstream out;out<<"{\"outside_timing\":true,\"all_workers_round_to_nearest\":true,\"worker_mask\":"<<mask<<",\"elements\":"<<total_elements<<",\"tiles\":"<<total_tiles<<",\"worker_elements\":[";
    for(int i=0;i<workers;++i){if(i)out<<',';out<<r.parameters.elements[size_t(i)];}out<<"]}";return out.str();
}
struct phase{quality checked;std::vector<double> times;std::string name,route_name;};
phase measure(route & r,const problem & p,const options & o,const std::string & name,const std::string & route_name,const route * baseline){
    phase out;out.name=name;out.route_name=route_name;
    for(int i=0;i<o.warmup;++i){r.output.reset();r.compute();out.checked.retain(check(p,r,o.threads,baseline));}
    const auto start=timer::now();
    do{r.output.reset();const auto begin=timer::now();r.compute();out.times.push_back(std::chrono::duration<double,std::milli>(timer::now()-begin).count());}
    while(out.times.size()<size_t(o.repeats)||std::chrono::duration<double>(timer::now()-start).count()<o.minimum_seconds);
    out.checked.retain(check(p,r,o.threads,baseline));return out;
}
std::string phase_json(const phase & p){
    auto sorted=p.times;std::sort(sorted.begin(),sorted.end());const size_t mid=sorted.size()/2;const double median=sorted.size()%2?sorted[mid]:(sorted[mid-1]+sorted[mid])*.5;
    std::ostringstream out;out<<"{\"phase\":"<<quote(p.name)<<",\"route\":"<<quote(p.route_name)<<",\"quality\":"<<p.checked.json()<<",\"median_ms\":"<<number(median)<<",\"samples_ms\":[";
    for(size_t i=0;i<p.times.size();++i){if(i)out<<',';out<<number(p.times[i]);}out<<"]}";return out.str();
}
#endif
} // namespace

int main(int argc,char ** argv){
    options o;try{o=parse(argc,argv);}catch(const std::exception & e){std::fprintf(stderr,"%s\n",e.what());return 2;}
    size_t controls=0;try{controls=host_controls()+quality_controls();}catch(const std::exception & e){std::fprintf(stderr,"%s\n",e.what());return 1;}
    if(o.host){std::printf("F32_PACKED_HOST_CONTROLS passed=%zu failed=0 target_calls=0 native_graphs=0 model_forwards=0\n",controls);return 0;}
    std::unique_ptr<report_file> file;try{file=std::make_unique<report_file>(o.output);}catch(const std::exception & e){std::fprintf(stderr,"%s\n",e.what());return 2;}
    bool completed=false,passed=false,benchmark_allowed=false;uint64_t target_graphs=0,baseline_graphs=0;
    std::string error,provider_path,provider_config;std::vector<std::string> cases,phases;
    try{
        require(packed::supported(),"Linuxx64 AVX512F/FMA unavailable; zero target/native calls");
#if TSG_PREPACKED_WITH_GGML
        require(ggml_cpu_has_llamafile()&&ggml_cpu_has_avx(),"Captured default baseline must be AVX LLAMAFILE-enabled");
        require(std::fegetround()==FE_TONEAREST,"Round-to-nearest mode required");
        const char * fusion=std::getenv("GGML_CPU_DISABLE_FUSION");require(!fusion||!*fusion,"Fusion override is outside the baseline contract");
        std::unique_ptr<tsg_embeddings::linux_blas_plan> blas;
        if(o.baseline=="blas-policy"){
            blas=std::make_unique<tsg_embeddings::linux_blas_plan>(o.library.empty()?nullptr:o.library.c_str(),o.threads);
            require(blas->available()&&blas->thread_policy_current(),"Qualified optional one-worker BLAS unavailable");
            provider_path=blas->library_path();provider_config=blas->library_config();
        }
        session owner(o.threads);checkpoint model(o.model);
        struct family{const char * name;const char * tensor;int k;};
        const family families[]={{"down","blk.0.ffn_down.weight",2048},{"attn-swa","blk.0.attn_output.weight",1024},{"attn-global","blk.5.attn_output.weight",2048}};
        std::vector<std::unique_ptr<problem>> problems;
        for(int k:{1,17,31,33})for(int n:{1,7,8,9,15,16,17}){
            auto p=std::make_unique<problem>(packed::dimensions{k,32,n},"synthetic-tail-k"+std::to_string(k)+"-n"+std::to_string(n));p->synthetic=true;
            for(size_t i=0;i<p->weights.count;++i)p->weights.data()[i]=float(int(i%19)-9)*.03125f;
            fill_inputs(*p,1729);
            p->weights.data()[0]=-0.f;if(k>1)p->weights.data()[1]=std::numeric_limits<float>::denorm_min();
            p->finish();problems.push_back(std::move(p));
        }
        for(const auto & family:families){
            auto shared=std::make_shared<matrix_storage>(family.k,512);shared->original=model.load(family.tensor,family.k,512);
            const auto * traits=ggml_get_type_traits(GGML_TYPE_Q8_0);require(traits&&traits->to_float,"Unchanged Q8 decoder unavailable");
            traits->to_float(shared->original.data(),shared->weights.data(),int64_t(family.k)*512);
            for(int n:{128,800,1024}){
                auto p=std::make_unique<problem>(packed::dimensions{family.k,512,n},family.name,shared);
                fill_inputs(*p,101+uint32_t(n));p->finish();problems.push_back(std::move(p));
            }
        }
        bool all=true;
        for(auto & ptr:problems){
            auto & p=*ptr;route base(owner,p,o.threads,false,blas.get()),candidate(owner,p,o.threads,true,nullptr);
            base.compute();++baseline_graphs;const auto bq=check(p,base,o.threads);
            candidate.compute();++target_graphs;quality cq=check(p,candidate,o.threads,&base);const auto proof=worker_proof(candidate,p,o.threads);++target_graphs;cq.retain(check(p,candidate,o.threads,&base));all&=bq.passed&&cq.passed;
            std::ostringstream record;record<<"{\"name\":"<<quote(p.name)<<",\"synthetic\":"<<boolean(p.synthetic)<<",\"K\":"<<p.shape.k<<",\"M\":"<<p.shape.m<<",\"N\":"<<p.shape.n<<",\"original_q8_bytes\":"<<p.original.size()<<",\"original_q8_sha256\":"<<(p.synthetic?"null":quote(p.original_sha))
                <<",\"dequantized_f32_sha256\":"<<quote(sha256(p.weights.data(),p.weights.count*4))<<",\"packed_f32_sha256\":"<<quote(sha256(p.packed_weights.data(),p.packed_weights.count*4))<<",\"input_f32_sha256\":"<<quote(sha256(p.inputs.data(),p.inputs.count*4))
                <<",\"baseline_output_sha256\":"<<quote(sha256(base.output.data(),base.output.count*4))<<",\"candidate_output_sha256\":"<<quote(sha256(candidate.output.data(),candidate.output.count*4))
                <<",\"cold_pack_ms\":"<<number(p.pack_ms)<<",\"packed_bytes\":"<<p.packed_weights.count*4<<",\"baseline\":"<<bq.json()<<",\"candidate\":"<<cq.json()<<",\"worker_proof\":"<<proof<<'}';cases.push_back(record.str());
        }
        benchmark_allowed=all;
        if(o.benchmark&&all)for(auto & ptr:problems){
            auto & p=*ptr;if(p.synthetic)continue;
            route base(owner,p,o.threads,false,blas.get()),candidate(owner,p,o.threads,true,nullptr);
            base.compute();++baseline_graphs;quality gate=check(p,base,o.threads);candidate.compute();++target_graphs;gate.retain(check(p,candidate,o.threads,&base));all&=gate.passed;require(gate.passed,"Recreated resident plan preflight failed before timing");
            for(int step=0;step<4;++step){
                const bool owned=step==1||step==2;
                auto value=measure(owned?candidate:base,p,o,(step==0?"baseline-a":step==1?"candidate-a":step==2?"candidate-b":"baseline-b"),owned?"owned-prepacked-f32":o.baseline,owned?&base:nullptr);
                all&=value.checked.passed;std::ostringstream item;item<<"{\"name\":"<<quote(p.name)<<",\"N\":"<<p.shape.n<<",\"result\":"<<phase_json(value)<<'}';phases.push_back(item.str());
                const uint64_t calls=uint64_t(o.warmup)+value.times.size();if(owned)target_graphs+=calls;else baseline_graphs+=calls;
            }
        }
        completed=true;passed=all;
#else
        throw std::runtime_error("Standalone built without frozen GGML integration; zero native/model calls");
#endif
    }catch(const std::exception & e){error=e.what();}
    std::ostringstream out;out<<"{\"completed\":"<<boolean(completed)<<",\"passed\":"<<boolean(passed)<<",\"scope\":\"Validation-only resident projection component; no production/model/service claim\",\"compiler\":"<<quote(__VERSION__)
        <<",\"upstream_revision\":\"3612c0af93f2cfc8f048295300a61187723b12ae\",\"source_baseline_native_sha256\":\"638b6206de314b99386d7eb66fed968cf1c0877af6483ef612689dfdf35d71d4\""
        <<",\"baseline_route\":"<<quote(o.baseline)<<",\"workers\":"<<o.threads<<",\"host_controls\":"<<controls<<",\"isa_available\":"<<boolean(packed::supported())<<",\"completed_target_graph_calls\":"<<target_graphs<<",\"completed_baseline_graph_calls\":"<<baseline_graphs
        <<",\"provider_path\":"<<(provider_path.empty()?"null":quote(provider_path))<<",\"provider_config\":"<<(provider_config.empty()?"null":quote(provider_config))<<",\"provider_workers\":"<<(provider_path.empty()?"null":"1")
        <<",\"model_forwards\":0,\"benchmark_requested\":"<<boolean(o.benchmark)<<",\"all_initial_cases_passed\":"<<boolean(benchmark_allowed)<<",\"error\":"<<(error.empty()?"null":quote(error))<<",\"cases\":[";
    for(size_t i=0;i<cases.size();++i){if(i)out<<',';out<<cases[i];}
    out<<"],\"phases\":[";for(size_t i=0;i<phases.size();++i){if(i)out<<',';out<<phases[i];}out<<"]}\n";
    try{file->write(out.str());}catch(const std::exception & e){std::fprintf(stderr,"%s\n",e.what());return 1;}
    return completed&&passed?0:1;
}
