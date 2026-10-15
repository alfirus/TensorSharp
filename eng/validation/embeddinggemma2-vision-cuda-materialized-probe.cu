// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Validation only: full-score F32 QK, unnormalized exp, F32 PV, divide.
// No production/upstream change and no approximate or reduced-precision math.
#include <algorithm>
#include <array>
#include <cerrno>
#include <cfloat>
#include <climits>
#include <cstdio>
#include <memory>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>
#ifndef TSG_ATTN_HOST_CONTROLS_ONLY
#include "ggml.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <math_constants.h>
#endif

namespace {
constexpr size_t guard = 16;
constexpr float sentinel = -1234567.25f;
constexpr size_t score_budget = size_t(512) * 1024 * 1024;
void require(bool value, const char * reason) { if (!value) throw std::runtime_error(reason); }
size_t product(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b) throw std::overflow_error("Size product overflow");
    return a * b;
}
size_t addition(size_t a, size_t b) {
    if (a > std::numeric_limits<size_t>::max() - b) throw std::overflow_error("Size addition overflow");
    return a + b;
}
std::string quote(const std::string & value) {
    std::string out = "\"";
    for (unsigned char c : value) {
        if (c == '\\' || c == '"') { out += '\\'; out += char(c); }
        else if (c < 32) { char text[8]; std::snprintf(text, sizeof(text), "\\u%04x", c); out += text; }
        else out += char(c);
    }
    return out + "\"";
}
struct fixture {
    int queries, keys, heads, batches;
    bool padded, interleaved, token_major;
    int data;
    const char * name;
    bool original, actual, eligible;
};
constexpr fixture fixtures[] = {
    {1,1,1,1,false,false,false,0,"single",true,false,true},
    {37,37,3,2,true,false,false,0,"batched-padded",true,false,true},
    {65,129,2,1,true,true,false,0,"strided-key-tail",true,false,true},
    {129,521,4,1,false,false,false,0,"query-key-tail",true,false,true},
    {3,8193,2,1,true,false,false,0,"long-key-tail",true,false,true},
    {65,257,2,1,false,false,false,1,"large-values",true,false,true},
    {129,17,2,2,true,true,false,2,"exact-80000",true,false,true},
    {129,2,2,2,true,true,false,3,"source-bits",true,false,true},
    {1089,1089,12,1,false,false,true,0,"actual-1089",false,true,true},
    {2304,2304,12,1,false,false,true,0,"actual-2304",false,true,true},
    {2340,2340,12,1,false,false,true,0,"actual-2340",false,true,true},
    {131,133,12,1,false,false,true,0,"canonical-key-query-tails",false,false,true},
    {129,257,12,2,true,false,true,0,"canonical-padded-batches",false,false,true},
    {65,129,12,2,true,true,true,0,"canonical-interleaved-batches",false,false,true},
    {2,17,1,1,false,false,true,4,"unnormalized-range-refusal",false,false,false},
};
struct geometry {
    int rows, heads, batches;
    std::array<size_t,4> nb{};
    size_t offset, bytes;
    geometry(int n, int h, int b, bool pad, bool interleave, bool token_major)
        : rows(n), heads(h), batches(b), offset(64 + (pad ? 8 : 0)), bytes(0) {
        require(n>0 && h>0 && b>0, "Positive geometry required");
        nb[0] = interleave ? 8 : 4;
        if (token_major) {
            nb[2] = addition(product(64,nb[0]),pad?12:0);
            nb[1] = product(nb[2],size_t(h)+(pad?1:0));
            nb[3] = product(nb[1],size_t(n)+(pad?2:0));
        } else {
            nb[1] = addition(product(64,nb[0]),pad?12:0);
            nb[2] = product(nb[1],size_t(n)+(pad?2:0));
            nb[3] = product(nb[2],size_t(h)+(pad?1:0));
        }
        bytes = addition(addition(offset,product(nb[3],size_t(b))),64);
    }
    size_t logical(int x,int y,int h,int b) const { return size_t(x)+64*(size_t(y)+size_t(rows)*(size_t(h)+size_t(heads)*b)); }
    size_t address(int x,int y,int h,int b) const { return offset+x*nb[0]+y*nb[1]+h*nb[2]+b*nb[3]; }
};
size_t host_controls() {
    size_t passed=0;
    require(guard==16&&std::isfinite(sentinel)&&sentinel<-80000.125f,"Guard/sentinel contract changed");++passed;
    require(sizeof(fixtures)/sizeof(fixtures[0])==15,"Fixture coverage changed"); ++passed;
    require(std::count_if(std::begin(fixtures),std::end(fixtures),[](auto f){return f.original;})==8,"Original8fixtures required");++passed;
    for (const auto & f:fixtures) {
        geometry q(f.queries,f.heads,f.batches,f.padded,f.interleaved,f.token_major);
        geometry k(f.keys,f.heads,f.batches,f.padded,f.interleaved,f.token_major);
        for (const auto & g:{q,k}) {
            std::set<size_t> addresses;
            for(int b=0;b<g.batches;++b) for(int h=0;h<g.heads;++h)
                for(int y:{0,g.rows/2,g.rows-1}) for(int x:{0,1,31,32,63}) {
                    size_t at=g.address(x,y,h,b);require(at>=g.offset && at+4<=g.bytes-64,"Layout escaped source guards");++passed;
                    // Repeated sample rows are intentional; every logical cell still has one address.
                    addresses.insert(at);
                }
            require(!addresses.empty(),"Missing layout samples");++passed;
        }
        size_t scores=product(product(product(size_t(f.queries),size_t(f.keys)),size_t(f.heads)*f.batches),4);
        require(scores<=score_budget,"Fixture score budget changed");++passed;
    }
    const std::pair<size_t,size_t> overflow_cases[] = {{std::numeric_limits<size_t>::max(),2},{size_t(1)<<63,3}};
    for(const auto & pair:overflow_cases) {
        bool rejected=false;try{(void)product(pair.first,pair.second);}catch(const std::overflow_error&){rejected=true;}
        require(rejected,"Checked product overflow not rejected");++passed;
    }
    bool rejected=false;try{(void)addition(std::numeric_limits<size_t>::max(),1);}catch(const std::overflow_error&){rejected=true;}
    require(rejected,"Checked addition overflow not rejected");++passed;
    return passed;
}
#ifndef TSG_ATTN_HOST_CONTROLS_ONLY
// BEGIN EXACT OWNED BASELINE
struct view {
    const char * data;
    int64_t ne[4];
    size_t nb[4];
    int type;
};
view tensor_view(const ggml_tensor * t) {
    view result;
    result.data = static_cast<const char *>(t->data);
    result.type = t->type;
    for (int i = 0; i < 4; ++i) {
        result.ne[i] = t->ne[i];
        result.nb[i] = t->nb[i];
    }
    return result;
}
__device__ __forceinline__ float read(view t, int64_t x, int64_t y = 0, int64_t h = 0, int64_t b = 0) {
    const char * p = t.data + x * t.nb[0] + y * t.nb[1] + h * t.nb[2] + b * t.nb[3];
    if (t.type == GGML_TYPE_F32)
        return *reinterpret_cast<const float *>(p);
    if (t.type == GGML_TYPE_F16)
        return __half2float(*reinterpret_cast<const half *>(p));
    return __uint_as_float(uint32_t(*reinterpret_cast<const uint16_t *>(p)) << 16);
}

// One thread owns one query vector, while the block shares each K/V tile.
// All operands, dot products, online-softmax state and weighted sums stay F32.
// Unlike upstream's generic 72-wide flash kernel, this never rounds Q/K/V or
// accumulators to half; unlike decomposed attention, scratch is O(N * D).
template <int WIDTH>
__global__ __launch_bounds__(128) void vision_kernel(view q, view k, view v, float scale, float * output) {
    constexpr int QUERIES = 128, KEYS = 16;
    __shared__ float keys[KEYS * WIDTH], values[KEYS * WIDTH];
    // Query-major register arrays for all scores add register pressure and make
    // the compiler emit an enormous unrolled instruction stream. Keep this
    // small score tile key-major in shared memory: adjacent query threads use
    // adjacent banks, and the two vector loops below keep their independent
    // per-component accumulators in registers.
    __shared__ float probabilities[KEYS * QUERIES];
    const int head = blockIdx.y, batch = blockIdx.z;
    const int64_t query = int64_t(blockIdx.x) * QUERIES + threadIdx.x;
    const bool active = query < q.ne[1];
    float query_values[WIDTH], accumulators[WIDTH];
#pragma unroll
    for (int x = 0; x < WIDTH; ++x) {
        query_values[x] = active ? read(q, x, query, head, batch) : 0;
        accumulators[x] = 0;
    }
    float maximum = -CUDART_INF_F, denominator = 0;
    for (int64_t first = 0; first < k.ne[1]; first += KEYS) {
        const int count = int(min(int64_t(KEYS), k.ne[1] - first));
        for (int i = threadIdx.x; i < count * WIDTH; i += QUERIES) {
            keys[i] = read(k, i % WIDTH, first + i / WIDTH, head, batch);
            values[i] = read(v, i % WIDTH, first + i / WIDTH, head, batch);
        }
        __syncthreads();
        if (active) {
            float next = maximum;
#pragma unroll 1
            for (int j = 0; j < KEYS; ++j) {
                float dot0 = 0, dot1 = 0, dot2 = 0, dot3 = 0;
                if (j < count) {
#pragma unroll
                    for (int x = 0; x < WIDTH; x += 4) {
                        dot0 = fmaf(query_values[x], keys[j * WIDTH + x], dot0);
                        dot1 = fmaf(query_values[x + 1], keys[j * WIDTH + x + 1], dot1);
                        dot2 = fmaf(query_values[x + 2], keys[j * WIDTH + x + 2], dot2);
                        dot3 = fmaf(query_values[x + 3], keys[j * WIDTH + x + 3], dot3);
                    }
                }
                const float dot = (dot0 + dot1) + (dot2 + dot3);
                const float score = j < count ? __fmul_rn(dot, scale) : -CUDART_INF_F;
                probabilities[j * QUERIES + threadIdx.x] = score;
                next = fmaxf(next, score);
            }
            const float correction = expf(maximum - next);
            denominator *= correction;
#pragma unroll 1
            for (int j = 0; j < KEYS; ++j) {
                const float probability = j < count ? expf(probabilities[j * QUERIES + threadIdx.x] - next) : 0;
                probabilities[j * QUERIES + threadIdx.x] = probability;
                denominator += probability;
            }
#pragma unroll
            for (int x = 0; x < WIDTH; ++x) accumulators[x] *= correction;
#pragma unroll 1
            for (int j = 0; j < count; ++j) {
                const float probability = probabilities[j * QUERIES + threadIdx.x];
#pragma unroll
                for (int x = 0; x < WIDTH; ++x)
                    accumulators[x] = fmaf(probability, values[j * WIDTH + x], accumulators[x]);
            }
            maximum = next;
        }
        // An inactive tail query participates in both barriers; no tile can be
        // overwritten until all active queries have consumed it.
        __syncthreads();
    }
    if (active) {
        const int64_t row = head + q.ne[2] * (query + q.ne[1] * batch);
#pragma unroll
        for (int x = 0; x < WIDTH; ++x)
            output[row * WIDTH + x] = accumulators[x] / denominator;
    }
}

// END EXACT OWNED BASELINE

void check(cudaError_t code) { if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code)); }
void blas_check(cublasStatus_t code) { if(code!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS API status "+std::to_string(int(code))); }
bool cleanup_failed=false;
struct session {
    cudaStream_t stream=nullptr;
    cublasHandle_t handle=nullptr;
    session() {
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        try {
            blas_check(cublasCreate(&handle)); blas_check(cublasSetStream(handle,stream));
            blas_check(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));
            blas_check(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST));
        } catch(...) {
            if(handle && cublasDestroy(handle)!=CUBLAS_STATUS_SUCCESS)cleanup_failed=true;
            if(cudaStreamDestroy(stream)!=cudaSuccess)cleanup_failed=true;
            throw;
        }
    }
    ~session() {
        if(cudaStreamSynchronize(stream)!=cudaSuccess)cleanup_failed=true;
        if(cublasDestroy(handle)!=CUBLAS_STATUS_SUCCESS)cleanup_failed=true;
        if(cudaStreamDestroy(stream)!=cudaSuccess)cleanup_failed=true;
    }
};
struct buffer {
    session & owner; void * data=nullptr; size_t bytes;
    buffer(session & s,size_t n):owner(s),bytes(n) { check(cudaMalloc(&data,n)); }
    buffer(const buffer&)=delete;
    ~buffer() {
        if(cudaStreamSynchronize(owner.stream)!=cudaSuccess)cleanup_failed=true;
        if(cudaFree(data)!=cudaSuccess)cleanup_failed=true;
    }
    void upload(const void * source) { check(cudaMemcpyAsync(data,source,bytes,cudaMemcpyHostToDevice,owner.stream)); }
    std::vector<unsigned char> read_bytes() const {
        std::vector<unsigned char> out(bytes);
        check(cudaMemcpyAsync(out.data(),data,bytes,cudaMemcpyDeviceToHost,owner.stream));
        check(cudaStreamSynchronize(owner.stream));return out;
    }
};
struct guarded {
    buffer storage; size_t count;
    guarded(session & s,size_t n):storage(s,product(addition(n,2*guard),4)),count(n) {
        std::vector<float> values(n+2*guard,sentinel);storage.upload(values.data());check(cudaStreamSynchronize(s.stream));
    }
    float * data() const { return static_cast<float*>(storage.data)+guard; }
    std::vector<float> read() const {
        auto bytes=storage.read_bytes();std::vector<float> values(count+2*guard);
        std::memcpy(values.data(),bytes.data(),bytes.size());return values;
    }
    bool intact() const {
        auto values=read();
        for(size_t i=0;i<guard;++i) {
            if(values[i]!=sentinel || values[values.size()-1-i]!=sentinel)return false;
        }
        return true;
    }
    void reset() {
        std::vector<float> values(count+2*guard,sentinel);storage.upload(values.data());check(cudaStreamSynchronize(storage.owner.stream));
    }
};
struct input {
    geometry g;std::vector<float> values;std::vector<unsigned char> bytes;buffer gpu;
    input(session & s,int rows,int heads,int batches,bool pad,bool interleave,bool token_major)
        :g(rows,heads,batches,pad,interleave,token_major),values(product(product(size_t(64)*rows,heads),batches)),bytes(g.bytes,0xff),gpu(s,g.bytes) {}
    float at(int x,int y,int h,int b)const{return values[g.logical(x,y,h,b)];}
    void upload() {
        for(int b=0;b<g.batches;++b)for(int h=0;h<g.heads;++h)for(int y=0;y<g.rows;++y)for(int x=0;x<64;++x) {
            float value=at(x,y,h,b);std::memcpy(bytes.data()+g.address(x,y,h,b),&value,4);
        }
        gpu.upload(bytes.data());
    }
    bool unchanged() const {return gpu.read_bytes()==bytes;}
    view as_view() const {
        view result{};result.data=static_cast<const char*>(gpu.data)+g.offset;result.type=GGML_TYPE_F32;
        result.ne[0]=64;result.ne[1]=g.rows;result.ne[2]=g.heads;result.ne[3]=g.batches;
        for(int i=0;i<4;++i) { result.nb[i]=g.nb[i]; }
        return result;
    }
};
size_t span(const view & v) {
    size_t n=4;
    for(int i=0;i<4;++i){require(v.ne[i]>0,"Positive view dimensions required");n=addition(n,product(size_t(v.ne[i]-1),v.nb[i]));}
    require(reinterpret_cast<uintptr_t>(v.data)<=std::numeric_limits<uintptr_t>::max()-n,"Pointer span overflow");return n;
}
bool valid_view(const view & v) {
    if(!v.data || v.type!=GGML_TYPE_F32 || v.ne[0]!=64 || reinterpret_cast<uintptr_t>(v.data)%4)return false;
    std::array<std::pair<size_t,int64_t>,4> axes;
    for(int i=0;i<4;++i) {
        if(v.ne[i]<=0 || v.ne[i]>INT_MAX || !v.nb[i] || v.nb[i]%4)return false;
        axes[i]={v.nb[i],v.ne[i]};
    }
    std::sort(axes.begin(),axes.end());size_t used=4;
    for(auto axis:axes) {
        if(axis.second>1 && axis.first<used)return false;
        used=addition(used,product(size_t(axis.second-1),axis.first));
    }
    (void)span(v);return true;
}
bool overlap(uintptr_t a,size_t an,uintptr_t b,size_t bn){return a<b+bn && b<a+an;}
bool admit(view q,view k,view v,float * output,float scale,size_t budget) {
    try {
        if(!valid_view(q)||!valid_view(k)||!valid_view(v)||!output||!std::isfinite(scale))return false;
        for(int i:{2,3})if(q.ne[i]!=k.ne[i]||q.ne[i]!=v.ne[i])return false;
        if(k.ne[1]!=v.ne[1]||q.ne[2]>65535||q.ne[3]>65535)return false;
        size_t scores=product(product(size_t(q.ne[1]),size_t(k.ne[1])),product(size_t(q.ne[2])*q.ne[3],4));
        size_t bytes=product(product(size_t(q.ne[1])*64,size_t(q.ne[2])*q.ne[3]),4);
        uintptr_t out=reinterpret_cast<uintptr_t>(output);if(out%4||out>std::numeric_limits<uintptr_t>::max()-bytes||scores>budget)return false;
        for(auto x:{q,k,v})if(overlap(out,bytes,reinterpret_cast<uintptr_t>(x.data),span(x)))return false;
        return true;
    }catch(const std::exception&){return false;}
}
__global__ void pack_kernel(view source,float * output,size_t count) {
    for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<count;i+=size_t(blockDim.x)*gridDim.x) {
        size_t j=i;const int x=int(j%64);j/=64;const int y=int(j%source.ne[1]);j/=source.ne[1];
        const int h=int(j%source.ne[2]);const int b=int(j/source.ne[2]);output[i]=read(source,x,y,h,b);
    }
}
__global__ void range_kernel(view source,size_t count,int keys,bool values,unsigned * status) {
    for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<count;i+=size_t(blockDim.x)*gridDim.x) {
        size_t j=i;const int x=int(j%64);j/=64;const int y=int(j%source.ne[1]);j/=source.ne[1];
        const int h=int(j%source.ne[2]);const int b=int(j/source.ne[2]);const float value=read(source,x,y,h,b);
        if(!isfinite(value) || (values && double(fabsf(value))*keys>double(FLT_MAX)))atomicOr(status,1u);
    }
}
// Same256-thread F32 max/expf/sum order as the owned softmax, with normalization
// deferred until AFTER the F32 PV product. No approximate exp or fixture branch.
__global__ void exponential_kernel(float * scores,float * sums,int keys,int rows,float scale,unsigned * status) {
    __shared__ float reduction[256];const int tid=threadIdx.x;const int row=blockIdx.x;
    if(row>=rows)return;
    float * result=scores+size_t(row)*keys;float maximum=-CUDART_INF_F;
    for(int key=tid;key<keys;key+=256) {
        const float value=__fmul_rn(result[key],scale)+0.0f;
        result[key]=value;if(!isfinite(value))atomicOr(status,2u);maximum=fmaxf(maximum,value);
    }
    reduction[tid]=maximum;__syncthreads();
    for(int stride=128;stride;stride/=2){if(tid<stride)reduction[tid]=fmaxf(reduction[tid],reduction[tid+stride]);__syncthreads();}
    maximum=reduction[0];float sum=0;
    for(int key=tid;key<keys;key+=256){const float value=isfinite(maximum)?expf(result[key]-maximum):0;result[key]=value;sum+=value;}
    __syncthreads();reduction[tid]=sum;__syncthreads();
    for(int stride=128;stride;stride/=2){if(tid<stride)reduction[tid]+=reduction[tid+stride];__syncthreads();}
    if(tid==0){sums[row]=reduction[0];if(!(reduction[0]>0)||!isfinite(reduction[0]))atomicOr(status,2u);}
}
__global__ void divide_kernel(float * output,const float * sums,int queries,int heads,size_t count,unsigned * status) {
    for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<count;i+=size_t(blockDim.x)*gridDim.x) {
        const size_t head=i/64%heads,query=i/(size_t(64)*heads)%queries,batch=i/(size_t(64)*heads*queries);
        const float sum=sums[query+size_t(queries)*(head+size_t(heads)*batch)];
        const float value=output[i]/sum;output[i]=value;if(!isfinite(value))atomicOr(status,4u);
    }
}
__global__ void downstream_kernel(const float * source,float * output,size_t count) {
    for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<count;i+=size_t(blockDim.x)*gridDim.x)output[i]=__fmul_rn(source[i],0.5f);
}
long long stride_elements(size_t bytes) {
    require(bytes%4==0,"cuBLAS stride must contain complete F32 elements");
    require(bytes/4<=size_t(LLONG_MAX),"cuBLAS longlong stride is not representable");
    return static_cast<long long>(bytes/4);
}
bool needs_pack(view source) {
    return source.nb[0]!=4 || source.nb[1]/4<64 || source.nb[1]/4>INT_MAX;
}
long long head_stride(view source) {
    // A singleton head never dereferences a second matrix. Canonicalize its
    // unused descriptor stride rather than forwarding arbitrarily huge values.
    return stride_elements(source.ne[2]==1
        ? addition(product(size_t(source.ne[1]-1),source.nb[1]),64*4) : source.nb[2]);
}
unsigned blocks(size_t n){return unsigned(std::min<size_t>((n+255)/256,65535));}
struct materialized {
    session & owner;view original_q,original_k,original_v;
    size_t budget_limit,elements,score_elements,rows;
    static size_t checked_scores(view q,view k,view v,size_t budget) {
        require(valid_view(q)&&valid_view(k)&&valid_view(v),"Unsupported input layout");
        require(q.ne[2]==k.ne[2]&&q.ne[2]==v.ne[2]&&q.ne[3]==k.ne[3]&&q.ne[3]==v.ne[3]&&k.ne[1]==v.ne[1],"Mismatched input shapes");
        require(q.ne[2]<=65535&&q.ne[3]<=65535,"Unsupported launch dimensions before allocation");
        size_t count=product(product(size_t(q.ne[1]),size_t(k.ne[1])),product(size_t(q.ne[2]),size_t(q.ne[3])));
        require(product(count,4)<=budget,"Score budget refused before allocation");
        return count;
    }
    guarded scores,sums,work,result,status;
    std::vector<std::unique_ptr<guarded>> packs;
    materialized(session & s,view q,view k,view v,size_t budget=score_budget):owner(s),original_q(q),original_k(k),original_v(v),budget_limit(budget),
        elements(product(product(size_t(64)*q.ne[1],q.ne[2]),q.ne[3])),
        score_elements(checked_scores(q,k,v,budget)),rows(product(size_t(q.ne[1])*q.ne[2],q.ne[3])),
        scores(s,score_elements),sums(s,rows),work(s,elements),result(s,elements),status(s,1) {
        require(admit(q,k,v,result.data(),1.0f,budget_limit),"Unsupported materialized geometry");
        for(auto source:{q,k,v}) {
            const bool need=needs_pack(source);
            packs.push_back(need?std::make_unique<guarded>(s,product(product(size_t(64)*source.ne[1],source.ne[2]),source.ne[3])):nullptr);
        }
    }
    view prepare(view source,size_t index) {
        if(!packs[index])return source;
        size_t count=packs[index]->count;pack_kernel<<<blocks(count),256,0,owner.stream>>>(source,packs[index]->data(),count);check(cudaGetLastError());
        source.data=reinterpret_cast<const char*>(packs[index]->data());source.nb[0]=4;source.nb[1]=64*4;
        source.nb[2]=size_t(source.ne[1])*source.nb[1];source.nb[3]=size_t(source.ne[2])*source.nb[2];return source;
    }
    unsigned read_status() {
        unsigned value=0;check(cudaMemcpyAsync(&value,status.data(),4,cudaMemcpyDeviceToHost,owner.stream));check(cudaStreamSynchronize(owner.stream));return value;
    }
    bool execute(float scale) {
        if(!admit(original_q,original_k,original_v,result.data(),scale,budget_limit))return false;
        check(cudaMemsetAsync(status.data(),0,4,owner.stream));
        const std::pair<view,bool> range_inputs[] = {{original_q,false},{original_k,false},{original_v,true}};
        for(const auto & pair:range_inputs) {
            size_t count=product(product(size_t(64)*pair.first.ne[1],pair.first.ne[2]),pair.first.ne[3]);
            range_kernel<<<blocks(count),256,0,owner.stream>>>(pair.first,count,int(original_k.ne[1]),pair.second,reinterpret_cast<unsigned*>(status.data()));check(cudaGetLastError());
        }
        if(read_status())return false;
        view q=prepare(original_q,0),k=prepare(original_k,1),v=prepare(original_v,2);
        const int queries=int(q.ne[1]),keys=int(k.ne[1]),heads=int(q.ne[2]);const float one=1.0f,zero=0.0f;
        for(int b=0;b<q.ne[3];++b)
            blas_check(cublasGemmStridedBatchedEx(owner.handle,CUBLAS_OP_T,CUBLAS_OP_N,keys,queries,64,&one,
                k.data+b*k.nb[3],CUDA_R_32F,int(k.nb[1]/4),head_stride(k),
                q.data+b*q.nb[3],CUDA_R_32F,int(q.nb[1]/4),head_stride(q),&zero,
                scores.data()+size_t(b)*heads*keys*queries,CUDA_R_32F,keys,int64_t(keys)*queries,heads,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
        check(cudaGetLastError());
        exponential_kernel<<<unsigned(rows),256,0,owner.stream>>>(scores.data(),sums.data(),keys,int(rows),scale,reinterpret_cast<unsigned*>(status.data()));check(cudaGetLastError());
        for(int b=0;b<q.ne[3];++b)
            blas_check(cublasGemmStridedBatchedEx(owner.handle,CUBLAS_OP_N,CUBLAS_OP_N,64,queries,keys,&one,
                v.data+b*v.nb[3],CUDA_R_32F,int(v.nb[1]/4),head_stride(v),
                scores.data()+size_t(b)*heads*keys*queries,CUDA_R_32F,keys,int64_t(keys)*queries,&zero,
                work.data()+size_t(b)*heads*64*queries,CUDA_R_32F,64*heads,64,heads,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
        check(cudaGetLastError());
        divide_kernel<<<blocks(elements),256,0,owner.stream>>>(work.data(),sums.data(),queries,heads,elements,reinterpret_cast<unsigned*>(status.data()));check(cudaGetLastError());
        downstream_kernel<<<blocks(elements),256,0,owner.stream>>>(work.data(),result.data(),elements);check(cudaGetLastError());
        if(read_status())return false;
        return true;
    }
    bool publish(std::vector<float> & published,bool readonly_inputs) {
        if(!readonly_inputs || read_status() || !intact())return false;
        auto values=result.read();
        if(!std::all_of(values.begin()+guard,values.end()-guard,[](float x){return std::isfinite(x)&&x!=sentinel;}))return false;
        published.assign(values.begin()+guard,values.end()-guard);return true;
    }
    bool intact()const {
        if(!scores.intact()||!sums.intact()||!work.intact()||!result.intact()||!status.intact())return false;
        for(const auto & pack:packs)if(pack && !pack->intact())return false;
        return true;
    }
};
struct oracle_row { int query,head,batch;std::array<double,64> values{}; };
std::vector<oracle_row> independent_double(const input & q,const input & k,const input & v,float scale,bool complete) {
    std::set<int> selected;
    if(complete)for(int i=0;i<q.g.rows;++i)selected.insert(i);
    else for(int i:{0,1,15,16,31,32,63,64,127,128,q.g.rows/2,q.g.rows-2,q.g.rows-1})if(i>=0&&i<q.g.rows)selected.insert(i);
    std::vector<oracle_row> out;std::vector<double> weights(k.g.rows);
    for(int b=0;b<q.g.batches;++b)for(int h=0;h<q.g.heads;++h)for(int t:selected) {
        double maximum=-std::numeric_limits<double>::infinity(),sum=0;
        for(int s=0;s<k.g.rows;++s){double dot=0;for(int x=0;x<64;++x)dot+=double(q.at(x,t,h,b))*k.at(x,s,h,b);weights[s]=dot*scale;maximum=std::max(maximum,weights[s]);}
        for(auto & p:weights){p=std::exp(p-maximum);sum+=p;}
        oracle_row row;row.query=t;row.head=h;row.batch=b;
        for(int x=0;x<64;++x){double value=0;for(int s=0;s<k.g.rows;++s)value+=weights[s]*v.at(x,s,h,b);row.values[x]=value*(0.5/sum);}
        out.push_back(row);
    }
    return out;
}
struct quality {
    bool passed=true,finite=true,written=true,canaries=true,inputs_immutable=true,exact=true;
    size_t double_outputs=0,complete_baseline_outputs=0;
    double max_double=0,max_double_ratio=0,max_baseline=0,max_baseline_ratio=0;
};
quality verify(const fixture & f,const std::vector<float> & actual,const std::vector<oracle_row> & oracle,
               const std::vector<float> * baseline,bool integrity) {
    quality q;q.canaries=integrity;q.passed=integrity;
    const size_t count=size_t(64)*f.queries*f.heads*f.batches;
    if(actual.size()!=count){q.passed=q.written=false;return q;}
    for(size_t i=0;i<count;++i) {
        const float x=actual[i];q.finite &= std::isfinite(x);q.written &= x!=sentinel;
        if(f.data==2)q.exact &= x==40000.0625f;
        if(baseline) {
            ++q.complete_baseline_outputs;
            if(!std::isfinite(x)||!std::isfinite((*baseline)[i])){q.passed=false;continue;}
            const double error=std::abs(double(x)-(*baseline)[i]);
            const double bound=f.data==3?2e-6:8e-5+2e-5*std::abs((*baseline)[i]);
            q.max_baseline=std::max(q.max_baseline,error);q.max_baseline_ratio=std::max(q.max_baseline_ratio,error/bound);q.passed &= error<=bound;
        }
    }
    for(const auto & row:oracle)for(int x=0;x<64;++x) {
        const float value=actual[x+size_t(64)*(row.head+f.heads*(row.query+size_t(f.queries)*row.batch))];++q.double_outputs;
        if(!std::isfinite(value)){q.passed=false;continue;}
        const double error=std::abs(double(value)-row.values[x]),bound=f.data==3?2e-6:8e-5+2e-5*std::abs(row.values[x]);
        q.max_double=std::max(q.max_double,error);q.max_double_ratio=std::max(q.max_double_ratio,error/bound);q.passed &= error<=bound;
    }
    q.passed &= q.finite&&q.written&&q.canaries&&q.exact;return q;
}
void merge(quality & a,const quality & b) {
    a.passed &= b.passed;a.finite &= b.finite;a.written &= b.written;a.canaries &= b.canaries;a.inputs_immutable &= b.inputs_immutable;a.exact &= b.exact;
    a.double_outputs+=b.double_outputs;a.complete_baseline_outputs+=b.complete_baseline_outputs;
    a.max_double=std::max(a.max_double,b.max_double);a.max_double_ratio=std::max(a.max_double_ratio,b.max_double_ratio);
    a.max_baseline=std::max(a.max_baseline,b.max_baseline);a.max_baseline_ratio=std::max(a.max_baseline_ratio,b.max_baseline_ratio);
}
void write_quality(std::ostream & out,const quality & q) {
    out << "{\"passed\":" << (q.passed?"true":"false") << ",\"finite\":" << (q.finite?"true":"false")
        << ",\"written\":" << (q.written?"true":"false") << ",\"canaries\":" << (q.canaries?"true":"false")
        << ",\"inputs_immutable\":" << (q.inputs_immutable?"true":"false") << ",\"exact_uniform_gate\":" << (q.exact?"true":"false")
        << ",\"independent_double_outputs\":" << q.double_outputs << ",\"complete_baseline_outputs\":" << q.complete_baseline_outputs
        << ",\"max_double_error\":" << q.max_double << ",\"max_double_error_over_bound\":" << q.max_double_ratio
        << ",\"max_full_baseline_error\":" << q.max_baseline << ",\"max_full_baseline_error_over_bound\":" << q.max_baseline_ratio << '}';
}
void populate(input & q,input & k,input & v,const fixture & f) {
    std::mt19937 random(46581+f.keys+f.queries);std::uniform_real_distribution<float> uniform(-1.5f,1.5f);
    for(auto & x:q.values)x=uniform(random);
    for(auto & x:k.values)x=uniform(random);
    for(auto & x:v.values)x=f.data==1?80000.125f+uniform(random):uniform(random);
    if(f.data>=2) {
        std::fill(q.values.begin(),q.values.end(),0);std::fill(k.values.begin(),k.values.end(),0);
        std::fill(v.values.begin(),v.values.end(),f.data==4?std::numeric_limits<float>::max()/2:80000.125f);
        if(f.data==3)for(int b=0;b<f.batches;++b)for(int h=0;h<f.heads;++h) {
            for(int t=0;t<f.queries;++t)q.values[q.g.logical(0,t,h,b)]=1.0f+0x1p-12f;
            k.values[k.g.logical(0,0,h,b)]=1;k.values[k.g.logical(0,1,h,b)]=-1;
            for(int x=0;x<64;++x){v.values[v.g.logical(x,0,h,b)]=1;v.values[v.g.logical(x,1,h,b)]=-1;}
        }
    }
}
#endif
void write_all(int fd,const void * data,size_t bytes) {
    const auto * p=static_cast<const char*>(data);
    while(bytes){ssize_t count=::write(fd,p,bytes);if(count<0&&errno==EINTR)continue;if(count<=0)throw std::runtime_error("Cannot write reserved output");p+=count;bytes-=size_t(count);}
}
#ifndef TSG_ATTN_HOST_CONTROLS_ONLY
void save_vector(const std::string & path,const std::vector<float> & values) {
    int fd=::open(path.c_str(),O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC,0644);require(fd>=0,"Fresh vector output required");
    try{write_all(fd,values.data(),product(values.size(),4));}catch(...){::close(fd);throw;}
    require(::close(fd)==0,"Cannot close vector output");
}
size_t rejection_controls(materialized & plan,view q,view k,view v,float scale) {
    const auto before=plan.result.storage.read_bytes();size_t count=0;
    auto reject=[&](view a,view b,view c,float * output,float s,size_t budget){require(!admit(a,b,c,output,s,budget),"Invalid admission accepted");++count;};
    reject(q,k,v,plan.result.data(),scale,0);
    reject(q,k,v,plan.result.data(),std::numeric_limits<float>::infinity(),score_budget);
    reject(q,k,v,reinterpret_cast<float*>(const_cast<char*>(q.data)),scale,score_budget);
    reject(q,k,v,reinterpret_cast<float*>(std::numeric_limits<uintptr_t>::max()-3),scale,score_budget);
    view invalid=q;invalid.type=GGML_TYPE_F16;reject(invalid,k,v,plan.result.data(),scale,score_budget);
    invalid=q;invalid.ne[0]=72;reject(invalid,k,v,plan.result.data(),scale,score_budget);
    invalid=q;invalid.ne[1]=0;reject(invalid,k,v,plan.result.data(),scale,score_budget);
    invalid=q;invalid.ne[2]+=1;reject(invalid,k,v,plan.result.data(),scale,score_budget);
    invalid=q;invalid.nb[0]=3;reject(invalid,k,v,plan.result.data(),scale,score_budget);
    invalid=q;invalid.data=reinterpret_cast<const char*>(std::numeric_limits<uintptr_t>::max()-3);reject(invalid,k,v,plan.result.data(),scale,score_budget);
    view singleton=q;singleton.ne[1]=singleton.ne[2]=singleton.ne[3]=1;
    singleton.nb[0]=4;singleton.nb[1]=4;
    singleton.nb[2]=singleton.nb[3]=std::numeric_limits<size_t>::max()&~size_t(3);
    require(valid_view(singleton)&&needs_pack(singleton),"Unused smalllda must use exactF32 packing");
    {
        materialized packed(plan.owner,singleton,singleton,singleton);
        require(packed.packs[0]&&packed.packs[1]&&packed.packs[2]&&packed.intact(),"Smalllda packing plan missing");
    }
    ++count;
    singleton.nb[1]=64*4;
    require(valid_view(singleton)&&!needs_pack(singleton)&&head_stride(singleton)==64,"Unused hugehead stride not canonicalized");
    require(stride_elements(std::numeric_limits<size_t>::max()&~size_t(3))<=LLONG_MAX,"Representable stride boundary failed");
    bool partial_rejected=false;try{(void)stride_elements(3);}catch(const std::exception&){partial_rejected=true;}
    require(partial_rejected,"PartialF32 stride accepted");++count;
    bool budget_refused=false;
    try{materialized unsupported(plan.owner,q,k,v,0);}catch(const std::exception&){budget_refused=true;}
    require(budget_refused,"Constructor must refuse budget before scratch allocation");++count;
    require(before==plan.result.storage.read_bytes()&&plan.intact(),"Host admission refusal mutated scratch/guards");return count;
}
std::string run_device(const std::string & mode,const std::string & path,size_t controls) {
    (void)&tensor_view;
    check(cudaSetDevice(0));cudaDeviceProp device{};check(cudaGetDeviceProperties(&device,0));
    require(device.major==8&&device.minor==6,"This standalone screen requires actualSM86");
    std::ostringstream records,timings;records<<std::setprecision(17);timings<<std::setprecision(17);
    bool passed=true,original_passed=true,record_comma=false,timing_comma=false;int eligible=0,unavailable=0,failed=0,baseline_failed=0;size_t refusals=0;
    int version=0;size_t peak_score=0;cudaFuncAttributes attrs{};check(cudaFuncGetAttributes(&attrs,vision_kernel<64>));
    {
        session s;blas_check(cublasGetVersion(s.handle,&version));
        cublasMath_t math_mode;blas_check(cublasGetMathMode(s.handle,&math_mode));require(math_mode==CUBLAS_PEDANTIC_MATH,"Private pedantic math mode differs");
        // Each fixture owns all scratch through final completion; no model pointer or persistent cache exists.
        for(const auto & f:fixtures) {
            input q(s,f.queries,f.heads,f.batches,f.padded,f.interleaved,f.token_major);
            input k(s,f.keys,f.heads,f.batches,f.padded,f.interleaved,f.token_major);
            input v(s,f.keys,f.heads,f.batches,f.padded,f.interleaved,f.token_major);
            populate(q,k,v,f);q.upload();k.upload();v.upload();check(cudaStreamSynchronize(s.stream));
            materialized candidate(s,q.as_view(),k.as_view(),v.as_view());peak_score=std::max(peak_score,candidate.score_elements*4);
            const float scale=f.original&&f.data!=3?1.0f/std::sqrt(64.0f):1.0f;
            refusals+=rejection_controls(candidate,q.as_view(),k.as_view(),v.as_view(),scale);
            if(record_comma) { records<<','; }
            record_comma=true;
            records<<"{\"case\":"<<quote(f.name)<<",\"queries\":"<<f.queries<<",\"keys\":"<<f.keys<<",\"heads\":"<<f.heads<<",\"batches\":"<<f.batches
                <<",\"padded\":"<<(f.padded?"true":"false")<<",\"interleaved\":"<<(f.interleaved?"true":"false")<<",\"token_major\":"<<(f.token_major?"true":"false")
                <<",\"data_kind\":"<<f.data<<",\"seed\":"<<(46581+f.keys+f.queries)
                <<",\"q_byte_strides\":["<<q.g.nb[0]<<','<<q.g.nb[1]<<','<<q.g.nb[2]<<','<<q.g.nb[3]<<']'
                <<",\"k_v_byte_strides\":["<<k.g.nb[0]<<','<<k.g.nb[1]<<','<<k.g.nb[2]<<','<<k.g.nb[3]<<']'
                <<",\"original_fixture\":"<<(f.original?"true":"false")<<",\"actual_geometry\":"<<(f.actual?"true":"false")<<",\"scale\":"<<scale<<",\"eligible\":"<<(f.eligible?"true":"false");
            if(!f.eligible) {
                ++unavailable;std::vector<float> published(candidate.elements,37.125f);const auto before=published;const auto private_before=candidate.result.storage.read_bytes();
                const bool executed=candidate.execute(scale);const bool integrity=candidate.intact()&&q.unchanged()&&k.unchanged()&&v.unchanged();
                const bool refusal=!executed&&before==published&&private_before==candidate.result.storage.read_bytes()&&integrity;
                passed &= refusal;records<<",\"candidate_executed\":false,\"range_control_executed\":true,\"refusal_no_publication_and_private_output_unchanged\":"<<(refusal?"true":"false")<<",\"quality\":null}";continue;
            }
            ++eligible;const auto oracle=independent_double(q,k,v,scale,!f.actual);
            guarded baseline_work(s,candidate.elements),baseline_result(s,candidate.elements);
            auto baseline_execute=[&] {
                const dim3 grid(unsigned((f.queries+127)/128),unsigned(f.heads),unsigned(f.batches));
                vision_kernel<64><<<grid,128,0,s.stream>>>(q.as_view(),k.as_view(),v.as_view(),scale,baseline_work.data());check(cudaGetLastError());
                downstream_kernel<<<blocks(candidate.elements),256,0,s.stream>>>(baseline_work.data(),baseline_result.data(),candidate.elements);check(cudaGetLastError());check(cudaStreamSynchronize(s.stream));
            };
            auto read_baseline=[&] {auto guarded_values=baseline_result.read();return std::vector<float>(guarded_values.begin()+guard,guarded_values.end()-guard);};
            baseline_execute();auto baseline=read_baseline();
            auto base_quality=verify(f,baseline,oracle,nullptr,baseline_work.intact()&&baseline_result.intact());
            base_quality.inputs_immutable=q.unchanged()&&k.unchanged()&&v.unchanged();base_quality.passed &=base_quality.inputs_immutable;
            passed &=base_quality.passed;if(!base_quality.passed)++baseline_failed;
            save_vector(path+"-"+f.name+"-baseline.f32",baseline);
            std::vector<float> published(candidate.elements,sentinel);quality combined;
            auto candidate_check=[&] {
                const auto previous=published;const bool done=candidate.execute(scale);
                const bool input_integrity=q.unchanged()&&k.unchanged()&&v.unchanged();
                bool publication=done&&candidate.publish(published,input_integrity);
                quality gate=verify(f,published,oracle,&baseline,candidate.intact());gate.passed &=publication;
                if(!publication)require(published==previous,"Failed candidate published a partial result");
                gate.inputs_immutable=q.unchanged()&&k.unchanged()&&v.unchanged();gate.passed &=gate.inputs_immutable;return gate;
            };
            auto initial=candidate_check();merge(combined,initial);
            for(int warm=0;warm<3;++warm)merge(combined,candidate_check());
            merge(combined,candidate_check());
            passed &=combined.passed;
            if(f.original) { original_passed &=combined.passed&&base_quality.passed; }
            if(!combined.passed) { ++failed; }
            save_vector(path+"-"+f.name+"-candidate.f32",published);
            records<<",\"candidate_executed\":true,\"baseline_quality\":";write_quality(records,base_quality);
            records<<",\"initial_candidate_quality\":";write_quality(records,initial);records<<",\"combined_initial_warm_final_quality\":";write_quality(records,combined);records<<'}';
            // Timing is globally deferred until a separate qualified build invocation.
            // Even there any preceding quality failure stops further timing, without weakening that failure.
            if(mode=="benchmark"&&passed&&f.actual) {
                for(int phase=0;phase<4;++phase) {
                    const bool baseline_phase=phase==0||phase==3;quality phase_quality;
                    for(int warm=0;warm<3;++warm) {
                        if(baseline_phase){baseline_execute();merge(phase_quality,verify(f,read_baseline(),oracle,&baseline,baseline_work.intact()&&baseline_result.intact()));}
                        else merge(phase_quality,candidate_check());
                    }
                    std::vector<double> samples;const auto started=std::chrono::steady_clock::now();bool api=true;
                    do {
                        const auto begin=std::chrono::steady_clock::now();
                        if(baseline_phase)baseline_execute();else api &=candidate.execute(scale);
                        samples.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count());
                    }while(samples.size()<20||std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<.1);
                    if(baseline_phase)merge(phase_quality,verify(f,read_baseline(),oracle,&baseline,baseline_work.intact()&&baseline_result.intact()));
                    else {require(candidate.publish(published,q.unchanged()&&k.unchanged()&&v.unchanged()),"Timed candidate final publication failed");merge(phase_quality,verify(f,published,oracle,&baseline,candidate.intact()));}
                    phase_quality.inputs_immutable=q.unchanged()&&k.unchanged()&&v.unchanged();phase_quality.passed &=api&&phase_quality.inputs_immutable;passed &=phase_quality.passed;
                    if(!phase_quality.passed)++failed;
                    auto sorted=samples;std::sort(sorted.begin(),sorted.end());
                    if(timing_comma) { timings<<','; }
                    timing_comma=true;
                    timings<<"{\"case\":"<<quote(f.name)<<",\"phase\":"<<phase<<",\"route\":"<<quote(baseline_phase?"copied-online-baseline":"materialized-unnormalized-pv-divide")
                        <<",\"median_ms\":"<<(sorted[(sorted.size()-1)/2]+sorted[sorted.size()/2])*.5<<",\"quality\":";write_quality(timings,phase_quality);timings<<",\"samples_ms\":[";
                    for(size_t i=0;i<samples.size();++i) {
                        if(i) { timings<<','; }
                        timings<<samples[i];
                    }
                    timings<<"]}";
                }
            }
            std::cerr<<"MATERIALIZED_ATTENTION "<<f.name<<" passed="<<combined.passed<<" double_ratio="<<combined.max_double_ratio<<" baseline_ratio="<<combined.max_baseline_ratio<<'\n';
        }
    }
    passed &=!cleanup_failed;
    std::ostringstream out;out<<std::setprecision(17)<<"{\"completed\":true,\"passed\":"<<(passed?"true":"false")<<",\"mode\":"<<quote(mode)
        <<",\"device\":"<<quote(device.name)<<",\"sm_major\":"<<device.major<<",\"sm_minor\":"<<device.minor<<",\"cublas_version\":"<<version
        <<",\"private_cublas_pedantic_math_mode_verified\":true,\"host_controls_passed\":"<<controls<<",\"host_refusal_controls_passed\":"<<refusals<<",\"eligible_candidate_quality_passed\":"<<(passed?"true":"false")
        <<",\"candidate_shapes_executed\":"<<eligible<<",\"unavailable_candidate_shapes\":"<<unavailable<<",\"failed_candidate_cases_or_timing_phases\":"<<failed
        <<",\"baseline_cases_failed\":"<<baseline_failed<<",\"all_original_fixtures_candidate_passed\":"<<(original_passed?"true":"false")
        <<",\"device_allocator_failure_injected\":false,\"device_allocator_failure_executed\":false,\"cleanup_succeeded\":"<<(!cleanup_failed?"true":"false")<<",\"peak_score_workspace_bytes\":"<<peak_score
        <<",\"baseline_kernel_registers\":"<<attrs.numRegs<<",\"baseline_kernel_shared_bytes\":"<<attrs.sharedSizeBytes
        <<",\"quality_cases\":["<<records.str()<<"],\"timings\":["<<timings.str()<<"],\"warmup\":3,\"minimum_repeats\":20,\"minimum_seconds\":0.1"
        <<",\"model_qualified\":false,\"production_promoted\":false,\"performance_qualified\":false"
        <<",\"timing_scope\":\"Resident device inputs/outputs/private handle/workspaces. Includes candidate range/status scans and host status completion, any F32 packing, QK, exp/sum, PV, divide, downstream0.5 and stream completion. Excludes cold allocation/initial input upload, independent oracle, full output readback and canary/immutability quality checks. Baseline includes original onlinekernel plus same downstream0.5 and stream completion.\""
        <<",\"limitations\":[\"Validation only; no production/native/upstream edit or learned model result.\",\"All8original fixtures use completeDouble; actualN1089/2304/2340 use sampledcompleteDouble queryrows/all12heads and completebaseline comparisons.\",\"UnnormalizedPV range has an explicit conservativeVmax*keys finite bound; refusal is not universalF32 arithmetic qualification.\",\"Priorv4/v5 vision already used tiledmaterializedF32 math; this distinct fullscore/directstride screen has no forecastgain.\",\"Private cuBLAS pedantic call policy is explicit; no TensorCore or bitidentity claim fromsource alone.\"]}";
    return out.str();
}
#endif
} // namespace

int main(int argc,char ** argv) {
    if(argc==2&&std::strcmp(argv[1],"--host-controls")==0) {
        try{std::cout<<"HOST_CONTROLS passed="<<host_controls()<<" failed=0 cuda_executed=0\n";return 0;}catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}
    }
    if(argc!=5||std::strcmp(argv[1],"--mode")||std::strcmp(argv[3],"--output")||
       (std::strcmp(argv[2],"quality")&&std::strcmp(argv[2],"benchmark"))){std::cerr<<"Require --mode quality|benchmark --output FRESH_PATH\n";return 2;}
    const int fd=::open(argv[4],O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC,0644);if(fd<0){std::cerr<<"Fresh output reservation failed\n";return 2;}
    std::string report;int code=1;
    try {
#ifdef TSG_ATTN_HOST_CONTROLS_ONLY
        throw std::runtime_error("CUDA compilation/device unavailable in host-controls-only executable");
#else
        if(std::strcmp(argv[2],"benchmark")==0) {
            const std::string preflight=run_device("quality",std::string(argv[4])+"-preflight",host_controls());
            const std::string preflight_path=std::string(argv[4])+"-preflight.json";
            const int preflight_fd=::open(preflight_path.c_str(),O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC,0644);
            require(preflight_fd>=0,"Fresh benchmark qualitypreflight required");
            try{write_all(preflight_fd,preflight.data(),preflight.size());}catch(...){::close(preflight_fd);throw;}
            require(::close(preflight_fd)==0,"Cannot close benchmark preflight");
            require(preflight.find("\"passed\":true,\"mode\"")!=std::string::npos,"Allfixtures must qualify before anytiming");
        }
        report=run_device(argv[2],argv[4],host_controls());
        code=report.find("\"passed\":true,\"mode\"")==std::string::npos?1:0;
#endif
    }catch(const std::exception & e){report="{\"completed\":false,\"passed\":false,\"model_qualified\":false,\"performance_qualified\":false,\"error\":"+quote(e.what())+"}";std::cerr<<e.what()<<'\n';}
    try{write_all(fd,report.data(),report.size());require(::close(fd)==0,"Cannot close output report");}catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}
    return code;
}
