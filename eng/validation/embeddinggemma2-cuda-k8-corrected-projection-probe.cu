// Copyright (c) Zhongkai Fu. All rights reserved.
// Standalone validation only. No production or upstream dependency changes.
// K8 correction/order adapted from Ootomo/Yokota's CUTLASS fork, revision
// 268162f4bf652fdfebd3d132ebcf55a3986fe570, mma_sm75.h:280-325 and
// mma_multistage.h:362-371,440-448,532-535. Paper: arxiv.org/abs/2203.03341.
// The small CTA and manual mapping below are owned validation code, not the
// author's tuned CUTLASS implementation. Adapted source notice follows:
// Copyright (c) 2017 - 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of the NVIDIA CORPORATION nor the names of its
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <algorithm>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#if !defined(TSG_K8_HOST_CONTROLS)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#endif

namespace {
[[maybe_unused]] constexpr float canary = -1234567.25f;
constexpr size_t guard_count = 32;
uint32_t float_bits(float x) { uint32_t b; std::memcpy(&b, &x, 4); return b; }
[[maybe_unused]] float from_bits(uint32_t b) { float x; std::memcpy(&x, &b, 4); return x; }
void require(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
size_t product(int a, int b) {
    require(a > 0 && b > 0, "Nonpositive matrix dimension");
    require(size_t(a) <= (std::numeric_limits<size_t>::max() / sizeof(float) - 2 * guard_count) / size_t(b), "Matrix byte count overflow");
    return size_t(a) * size_t(b);
}
[[maybe_unused]] std::string quote(const std::string & s) {
    std::string y = "\"";
    for (unsigned char c : s) {
        if (c == '\\' || c == '"') { y += '\\'; y += char(c); }
        else if (c < 32) { char v[8]; std::snprintf(v, sizeof(v), "\\u%04x", c); y += v; }
        else y += char(c);
    }
    return y + '"';
}
[[maybe_unused]] std::string number(double x) {
    if (!std::isfinite(x)) return "null";
    std::ostringstream s; s << std::setprecision(17) << x; return s.str();
}
// Independent host oracle: search binary16 values and compare exact Double
// distances, rather than duplicating a device bit-rounding implementation.
double half_double(uint16_t h) {
    const unsigned e = (h >> 10) & 31, f = h & 1023;
    double x = e == 0 ? std::ldexp(double(f), -24) : e == 31
        ? (f ? std::numeric_limits<double>::quiet_NaN() : std::numeric_limits<double>::infinity())
        : std::ldexp(double(1024 + f), int(e) - 25);
    return h & 0x8000 ? -x : x;
}
uint16_t host_half(float x) {
    const uint16_t sign = uint16_t(float_bits(x) >> 16) & 0x8000;
    if (std::isnan(x)) return uint16_t(sign | 0x7e00);
    if (std::isinf(x)) return uint16_t(sign | 0x7c00);
    const double value = std::abs(double(x));
    unsigned lo = 0, hi = 0x7bff;
    while (lo < hi) {
        const unsigned middle = (lo + hi + 1) / 2;
        if (half_double(uint16_t(middle)) <= value) lo = middle;
        else hi = middle - 1;
    }
    const unsigned upper = lo + 1;
    const double a = value - half_double(uint16_t(lo));
    const double b = (upper == 0x7c00 ? 65536. : half_double(uint16_t(upper))) - value;
    const bool up = b < a || (b == a && (lo & 1));
    return uint16_t(sign | (up ? upper : lo));
}
struct split_value { uint16_t high, low; bool eligible; };
split_value host_split(float x) {
    const auto high = host_half(x);
    if (!std::isfinite(x) || (high & 0x7c00) == 0x7c00) return {high, 0, false};
    // Difference and power-of-two scale are evaluated in Double, then one
    // explicit F32 cast before independent half rounding. RN host required.
    const float scaled = float((double(x) - half_double(high)) * 2048.);
    const auto low = host_half(scaled);
    return {high, low, (low & 0x7c00) != 0x7c00};
}
struct shape { int n, m, k; const char * kind; bool original; };
[[maybe_unused]] std::vector<shape> shapes() {
    std::vector<shape> out = {{36,768,768,"normalized",true}, {2304,768,768,"normalized",true},
        {2340,3072,768,"normalized",true}, {1089,768,3072,"normalized",true},
        {129,64,64,"unrounded",true}, {33,64,64,"large_cancellation_80000",true}};
    for (int n : {36,108,1089,2304,2340}) {
        out.push_back({n,768,768,"actual_geometry",false});
        out.push_back({n,3072,768,"actual_geometry",false});
        out.push_back({n,768,3072,"actual_geometry",false});
    }
    for (const shape s : {shape{17,19,23,"tail",false}, shape{7,15,7,"tail",false},
        shape{8,16,8,"tail",false}, shape{9,17,9,"tail",false},
        shape{31,33,17,"tail",false}, shape{32,31,16,"tail",false}, shape{33,32,15,"tail",false}}) out.push_back(s);
    return out;
}
[[maybe_unused]] void initialize(shape s, std::vector<float> & w, std::vector<float> & x) {
    w.resize(product(s.m,s.k)); x.resize(product(s.n,s.k));
    // Exactly the six preserved previous TF32/API probe fixture generators.
    for (size_t i=0;i<w.size();++i) w[i]=float(std::sin(double(i*17+3)*.013)*.21+std::cos(double(i+11)*.003)*.007);
    for (size_t i=0;i<x.size();++i) x[i]=float(std::sin(double(i*13+7)*.017)*.83+std::cos(double(i+5)*.009)*.011);
    if (std::string(s.kind)=="unrounded") {
        for(size_t i=0;i<w.size();++i) w[i]=1.f+float(i%31)*.0000013f;
        for(size_t i=0;i<x.size();++i) x[i]=((i%s.k)%2?-1.f:1.f)+float(i%29)*.0000021f;
    }
    if (std::string(s.kind)=="large_cancellation_80000") {
        for(size_t i=0;i<w.size();++i) w[i]=80000.f+float(i%31)*.015625f;
        for(size_t i=0;i<x.size();++i) x[i]=((i%s.k)%2?-80000.f:80000.f)+float(i%23)*.015625f;
    }
}
// Admission is deliberately narrow. Unsupported descriptors do not access
// matrices, launch kernels or mutate the separate published output.
bool span_valid(uintptr_t begin,size_t elements) {
    return begin!=0 && elements<=(std::numeric_limits<uintptr_t>::max()-begin)/sizeof(float);
}
bool spans_overlap(uintptr_t a,size_t ac,uintptr_t b,size_t bc) {
    if(!span_valid(a,ac)||!span_valid(b,bc)) return true;
    return a<b+bc*sizeof(float)&&b<a+ac*sizeof(float);
}
struct descriptor { int n,m,k; size_t w_count,x_count,y_count; int w_stride,x_stride,y_stride,batch; bool f32,overlap; };
bool layout_eligible(const descriptor & d) {
    if (d.n<=0 || d.m<=0 || d.k<=0 || !d.f32 || d.overlap || d.batch!=1 || d.w_stride!=d.k || d.x_stride!=d.k || d.y_stride!=d.m) return false;
    try { return d.w_count==product(d.m,d.k) && d.x_count==product(d.n,d.k) && d.y_count==product(d.n,d.m); }
    catch (...) { return false; }
}
size_t host_controls() {
    require(std::fegetround()==FE_TONEAREST,"Require RN-even host floating-point mode");
    size_t count=0;
    auto test=[&](bool ok) { require(ok,"Host conversion/layout control failed"); ++count; };
    // Exhaust all finite half values: exact round trip and adjacent midpoints,
    // both signs; this tests signed zero and subnormal transitions as well.
    for (unsigned h=0; h<=0x7bff; ++h) {
        test(host_half(float(half_double(uint16_t(h))))==h);
        test(host_half(float(-half_double(uint16_t(h))))==(h|0x8000));
        const double next=h==0x7bff?65536.:half_double(uint16_t(h+1));
        const float mid=float((half_double(uint16_t(h))+next)*.5);
        const uint16_t tied=uint16_t((h&1)?h+1:h);
        test(host_half(mid)==tied); test(host_half(-mid)==uint16_t(tied|0x8000));
    }
    const descriptor good{17,19,23,19*23,17*23,17*19,23,23,19,1,true,false};
    test(layout_eligible(good));
    for(int fault=0;fault<10;++fault) {
        auto bad=good;
        switch(fault) {case 0:bad.n=0;break;case 1:bad.m=-1;break;case 2:bad.k=0;break;
            case 3:bad.w_count--;break;case 4:bad.x_count--;break;case 5:bad.y_count--;break;
            case 6:bad.w_stride++;break;case 7:bad.batch=2;break;case 8:bad.f32=false;break;case 9:bad.overlap=true;break;}
        test(!layout_eligible(bad));
    }
    test(!span_valid(std::numeric_limits<uintptr_t>::max()-3,1));
    test(spans_overlap(64,16,96,16)); test(!spans_overlap(64,16,128,16));
    test(!host_split(80000.f).eligible); test(!host_split(std::numeric_limits<float>::infinity()).eligible);
    test(!host_split(std::numeric_limits<float>::quiet_NaN()).eligible);
    // Independently enumerate fragment destinations: four warps, two N8 tiles,
    // every one of1024 CTA output entries exactly once, including row/col order.
    std::vector<int> touched(32*32);
    for(int warp=0;warp<4;++warp) for(int lane=0;lane<32;++lane) for(int tile=0;tile<2;++tile) for(int v=0;v<4;++v) {
        const int row=(warp/2)*16+lane/4+(v>=2?8:0), col=(warp%2)*16+tile*8+(lane%4)*2+(v&1);
        test(row>=0&&row<32&&col>=0&&col<32); ++touched[size_t(col)*32+row];
    }
    test(std::all_of(touched.begin(),touched.end(),[](int n){return n==1;}));
    return count;
}
} // namespace

#if defined(TSG_K8_HOST_CONTROLS)
int main() try {
    const auto n=host_controls();
    std::cout<<"{\"completed\":true,\"host_controls_passed\":"<<n<<",\"host_controls_failed\":0,\"cuda_executed\":false,\"model_qualified\":false,\"performance_qualified\":false}\n";
    return 0;
} catch(const std::exception & e) { std::cerr<<e.what()<<'\n'; return 1; }
#else
namespace {
bool teardown_failed=false;
void cleanup(cudaError_t x,const char * operation) noexcept { if(x!=cudaSuccess) { teardown_failed=true; std::cerr<<operation<<": "<<cudaGetErrorString(x)<<'\n'; } }
void cleanup(cublasStatus_t x,const char * operation) noexcept { if(x!=CUBLAS_STATUS_SUCCESS) { teardown_failed=true; std::cerr<<operation<<": cuBLAS status "<<int(x)<<'\n'; } }
void check(cudaError_t x) { if(x!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
void check(cublasStatus_t x) { if(x!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS status "+std::to_string(x)); }
struct stream {
    cudaStream_t value=nullptr;
    stream() { check(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking)); }
    ~stream() { if(value) { cleanup(cudaStreamSynchronize(value),"stream teardown synchronize"); cleanup(cudaStreamDestroy(value),"stream destroy"); } }
    stream(const stream &)=delete;
};
struct handle {
    cublasHandle_t value=nullptr;
    explicit handle(cudaStream_t s) {
        try { check(cublasCreate(&value)); check(cublasSetStream(value,s));
            check(cublasSetMathMode(value,static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH|CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION))); }
        catch(...) { if(value) cleanup(cublasDestroy(value),"partial cuBLAS destroy"); throw; }
    }
    ~handle() { if(value) cleanup(cublasDestroy(value),"cuBLAS destroy"); }
    handle(const handle &)=delete;
};
struct buffer {
    float * base=nullptr; size_t count; cudaStream_t owner;
    explicit buffer(size_t n,cudaStream_t s):count(n),owner(s) { check(cudaMalloc(reinterpret_cast<void **>(&base),(n+2*guard_count)*sizeof(float))); }
    ~buffer() { if(base) { cleanup(cudaStreamSynchronize(owner),"buffer teardown synchronize"); cleanup(cudaFree(base),"buffer free"); } }
    buffer(const buffer &)=delete;
    float * data() const { return base+guard_count; }
    void reset(cudaStream_t s,const std::vector<float> * values=nullptr) {
        std::vector<float> all(count+2*guard_count,canary);
        if(values) { require(values->size()==count,"Upload count mismatch"); std::copy(values->begin(),values->end(),all.begin()+guard_count); }
        check(cudaMemcpyAsync(base,all.data(),all.size()*sizeof(float),cudaMemcpyHostToDevice,s));
        check(cudaStreamSynchronize(s)); // pageable all remains alive
    }
    std::vector<float> read(cudaStream_t s) const {
        std::vector<float> all(count+2*guard_count);
        check(cudaMemcpyAsync(all.data(),base,all.size()*sizeof(float),cudaMemcpyDeviceToHost,s)); check(cudaStreamSynchronize(s)); return all;
    }
};
bool guards(const std::vector<float> & x) {
    for(size_t i=0;i<guard_count;++i) if(float_bits(x[i])!=float_bits(canary)||float_bits(x[x.size()-1-i])!=float_bits(canary)) return false;
    return true;
}
bool immutable(const std::vector<float> & guarded,const std::vector<float> & original) {
    return guarded.size()==original.size()+2*guard_count&&guards(guarded)&&std::memcmp(guarded.data()+guard_count,original.data(),original.size()*sizeof(float))==0;
}
struct device_status {
    unsigned * base=nullptr; unsigned * value=nullptr; cudaStream_t owner;
    explicit device_status(cudaStream_t s):owner(s) {
        check(cudaMalloc(reinterpret_cast<void **>(&base),3*sizeof(unsigned))); value=base+1;
        const unsigned initial[3]={0x51a7c0deu,0u,0x51a7c0deu};
        try { check(cudaMemcpyAsync(base,initial,sizeof(initial),cudaMemcpyHostToDevice,owner)); check(cudaStreamSynchronize(owner)); }
        catch(...) { cleanup(cudaStreamSynchronize(owner),"partial status synchronize"); cleanup(cudaFree(base),"partial status free"); base=nullptr; value=nullptr; throw; }
    }
    ~device_status() { if(base) { cleanup(cudaStreamSynchronize(owner),"status teardown synchronize"); cleanup(cudaFree(base),"status free"); } }
    bool guards_intact() const {
        unsigned all[3]={}; check(cudaMemcpyAsync(all,base,sizeof(all),cudaMemcpyDeviceToHost,owner)); check(cudaStreamSynchronize(owner));
        return all[0]==0x51a7c0deu&&all[2]==0x51a7c0deu;
    }
    device_status(const device_status &)=delete;
};
__device__ __forceinline__ void split_device(float x,__half & high,__half & low,unsigned * status) {
    high=__float2half_rn(x);
    if(!isfinite(x)||!isfinite(__half2float(high))) {
        atomicOr(status,1u); high=__float2half_rn(0.f); low=__float2half_rn(0.f); return;
    }
    low=__float2half_rn(__fmul_rn(__fsub_rn(x,__half2float(high)),2048.f));
    if(!isfinite(__half2float(low))) { atomicOr(status,2u); low=__float2half_rn(0.f); }
}
__device__ __forceinline__ uint32_t packed(__half a,__half b) {
    return uint32_t(__half_as_ushort(a))|(uint32_t(__half_as_ushort(b))<<16);
}
__device__ __forceinline__ void mma8(float * d,uint32_t a0,uint32_t a1,uint32_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) : "r"(a0),"r"(a1),"r"(b));
#else
    asm volatile("trap;");
#endif
}
// CTA output logical[M32,N32]. W[M,K] row-major, X[N,K] row-major;
// published Y[N,M] row-major. Exactly author K8 reset/cross order, untuned CTA.
__global__ void corrected_k8(const float * w,const float * x,float * y,int m,int n,int k,unsigned * status) {
    __shared__ __half ah[32*8],al[32*8],bh[32*8],bl[32*8];
    const int tid=int(threadIdx.x),warp=tid/32,lane=tid%32,group=lane/4,t=lane%4;
    const int wm=(warp/2)*16,wn=(warp%2)*16,base_m=int(blockIdx.x)*32,base_n=int(blockIdx.y)*32;
    float main[2][4]={{0.f}},correction[2][4]={{0.f}};
    for(int kk=0;kk<k;kk+=8) {
        for(int i=tid;i<32*8;i+=128) {
            const int row=i/8,component=i%8,ki=kk+component;
            const float av=base_m+row<m&&ki<k?w[size_t(base_m+row)*k+ki]:0.f;
            const float bv=base_n+row<n&&ki<k?x[size_t(base_n+row)*k+ki]:0.f;
            split_device(av,ah[i],al[i],status); split_device(bv,bh[i],bl[i],status);
        }
        __syncthreads();
        const int ai0=(wm+group)*8+2*t,ai1=(wm+group+8)*8+2*t;
        const uint32_t a0=packed(ah[ai0],ah[ai0+1]),a1=packed(ah[ai1],ah[ai1+1]);
        const uint32_t da0=packed(al[ai0],al[ai0+1]),da1=packed(al[ai1],al[ai1+1]);
        for(int tile=0;tile<2;++tile) {
            const int bi=(wn+tile*8+group)*8+2*t;
            const uint32_t b=packed(bh[bi],bh[bi+1]),db=packed(bl[bi],bl[bi+1]);
            float hh[4]={0.f,0.f,0.f,0.f};
            mma8(hh,a0,a1,b); // Zero dominant accumulator at EVERY K8.
            for(int i=0;i<4;++i) main[tile][i]=__fadd_rn(main[tile][i],hh[i]);
            mma8(correction[tile],da0,da1,b); // Author lowA*highB first.
            mma8(correction[tile],a0,a1,db); // Then highA*lowB; persists across K.
        }
        __syncthreads(); // Every thread joins before shared K8 buffers reused.
    }
    for(int tile=0;tile<2;++tile) for(int i=0;i<4;++i) {
        const int mi=base_m+wm+group+(i>=2?8:0),ni=base_n+wn+tile*8+2*t+(i&1);
        if(mi<m&&ni<n) y[size_t(ni)*m+mi]=__fadd_rn(main[tile][i],__fmul_rn(correction[tile][i],0x1p-11f));
    }
}
__global__ void inspect_split(const float * x,float * high_bits,float * low_bits,float * classification,size_t count) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x) {
        const __half h=__float2half_rn(x[i]);
        const bool finite=isfinite(x[i])&&isfinite(__half2float(h));
        const __half l=finite?__float2half_rn(__fmul_rn(__fsub_rn(x[i],__half2float(h)),2048.f)):__float2half_rn(0.f);
        high_bits[i]=float(__half_as_ushort(h)); low_bits[i]=float(__half_as_ushort(l));
        classification[i]=finite&&isfinite(__half2float(l))?1.f:0.f;
    }
}
struct pack_quality { size_t count=0; bool eligible=true; double max_reconstruction_error=0; };
pack_quality verify_pack(const buffer & src,const std::vector<float> & input,cudaStream_t s) {
    buffer high(input.size(),s),low(input.size(),s),eligible(input.size(),s);
    high.reset(s); low.reset(s); eligible.reset(s);
    inspect_split<<<unsigned((input.size()+255)/256),256,0,s>>>(src.data(),high.data(),low.data(),eligible.data(),input.size());
    check(cudaGetLastError()); check(cudaStreamSynchronize(s));
    const auto hs=high.read(s),ls=low.read(s),es=eligible.read(s);
    require(immutable(src.read(s),input)&&guards(hs)&&guards(ls)&&guards(es),"Packing guards/input changed");
    pack_quality q; q.count=input.size();
    for(size_t i=0;i<input.size();++i) {
        const auto expected=host_split(input[i]);
        if(std::isnan(input[i])) require((unsigned(hs[i+guard_count])&0x7c00)==0x7c00&&(unsigned(hs[i+guard_count])&1023)!=0,"Half NaN classification");
        else require(unsigned(hs[i+guard_count])==expected.high,"Host/device high half bits differ");
        require(unsigned(ls[i+guard_count])==expected.low&&es[i+guard_count]==(expected.eligible?1.f:0.f),"Host/device residual/admission differ");
        q.eligible&=expected.eligible;
        if(expected.eligible) q.max_reconstruction_error=std::max(q.max_reconstruction_error,
            std::abs(double(input[i])-(half_double(expected.high)+half_double(expected.low)/2048.)));
    }
    return q;
}
size_t device_pack_controls(cudaStream_t s) {
    std::vector<float> input;
    for(unsigned h=0;h<=0x7bff;++h) {
        const double a=half_double(uint16_t(h)),b=h==0x7bff?65536.:half_double(uint16_t(h+1));
        const float mid=float((a+b)*.5);
        for(float v : {float(a),mid,std::nextafter(mid,-std::numeric_limits<float>::infinity()),std::nextafter(mid,std::numeric_limits<float>::infinity())}) {
            input.push_back(v); input.push_back(-v);
        }
    }
    for(uint32_t b:{0u,0x80000000u,1u,0x80000001u,0x007fffffu,0x00800000u,0x7f7fffffu,0xff7fffffu,0x7f800000u,0xff800000u,0x7fc12345u,0xffc12345u}) input.push_back(from_bits(b));
    buffer src(input.size(),s); src.reset(s,&input);
    require(verify_pack(src,input,s).count==input.size(),"Device pack control count");
    return input.size();
}
struct sample { size_t index; double expected; };
std::vector<sample> samples(shape s,const std::vector<float> & w,const std::vector<float> & x) {
    const size_t count=product(s.n,s.m),original_samples=std::min(count,size_t(4096));
    std::set<size_t> indices;
    for(size_t j=0;j<original_samples;++j) indices.insert(original_samples==1?0:j*(count-1)/(original_samples-1));
    // Complete Double for bounded direct fixtures; original4096 evenly spaced
    // samples are retained, plus tile/partial-row boundary samples.
    if(std::string(s.kind)=="tail") for(size_t i=0;i<count;++i) indices.insert(i);
    for(int ni:{0,7,8,15,16,31,32,s.n-1}) for(int mi:{0,7,8,15,16,31,32,s.m-1})
        if(ni>=0&&ni<s.n&&mi>=0&&mi<s.m) indices.insert(size_t(ni)*s.m+mi);
    std::vector<sample> out;
    for(size_t i:indices) { const size_t mi=i%s.m,ni=i/s.m; double d=0;
        for(int ki=0;ki<s.k;++ki) d+=double(w[mi*s.k+ki])*double(x[ni*s.k+ki]);
        out.push_back({i,d}); }
    return out;
}
struct quality {
    bool finite=true,written=true,canaries=true,inputs_immutable=true;
    double max_double=0,double_ratio=0,max_baseline=0,baseline_ratio=0;
    size_t sampled=0,complete=0;
    bool passed() const { return finite&&written&&canaries&&inputs_immutable&&double_ratio<=1&&baseline_ratio<=1; }
};
void accumulate(quality & a,const quality & b) {
    a.finite&=b.finite; a.written&=b.written; a.canaries&=b.canaries; a.inputs_immutable&=b.inputs_immutable;
    a.max_double=std::max(a.max_double,b.max_double); a.double_ratio=std::max(a.double_ratio,b.double_ratio);
    a.max_baseline=std::max(a.max_baseline,b.max_baseline); a.baseline_ratio=std::max(a.baseline_ratio,b.baseline_ratio);
    a.sampled+=b.sampled; a.complete+=b.complete;
}
void write_quality(std::ostream & out,const quality & q) {
    out<<"{\"passed\":"<<(q.passed()?"true":"false")<<",\"finite\":"<<(q.finite?"true":"false")<<",\"written\":"<<(q.written?"true":"false")
       <<",\"canaries\":"<<(q.canaries?"true":"false")<<",\"inputs_immutable\":"<<(q.inputs_immutable?"true":"false")
       <<",\"sampled_original_double_outputs\":"<<q.sampled<<",\"complete_baseline_outputs\":"<<q.complete
       <<",\"max_double_error\":"<<number(q.max_double)<<",\"max_double_error_over_bound\":"<<number(q.double_ratio)
       <<",\"max_full_baseline_error\":"<<number(q.max_baseline)<<",\"max_full_baseline_error_over_bound\":"<<number(q.baseline_ratio)<<'}';
}
struct problem {
    shape s; stream execution; handle pedantic; std::vector<float> w,x,reference,published;
    buffer dw,dx,dy; device_status status; std::vector<sample> independent;
    bool eligible=true; pack_quality w_pack,x_pack;
    explicit problem(shape sh):s(sh),pedantic(execution.value),dw(product(s.m,s.k),execution.value),dx(product(s.n,s.k),execution.value),dy(product(s.n,s.m),execution.value),status(execution.value) {
        initialize(s,w,x); published.assign(product(s.n,s.m),canary);
        dw.reset(execution.value,&w); dx.reset(execution.value,&x); dy.reset(execution.value);
        w_pack=verify_pack(dw,w,execution.value); x_pack=verify_pack(dx,x,execution.value);
        eligible=w_pack.eligible&&x_pack.eligible;
        baseline(); const auto all=dy.read(execution.value); require(guards(all),"Initial baseline guards");
        reference.assign(all.begin()+guard_count,all.end()-guard_count); independent=samples(s,w,x);
    }
    ~problem() { cleanup(cudaStreamSynchronize(execution.value),"problem teardown synchronize"); } // Before member buffers free.
    void baseline() {
        const float one=1,zero=0;
        check(cublasGemmStridedBatchedEx(pedantic.value,CUBLAS_OP_T,CUBLAS_OP_N,s.m,s.n,s.k,&one,
            dw.data(),CUDA_R_32F,s.k,0,dx.data(),CUDA_R_32F,s.k,static_cast<long long>(dx.count),&zero,
            dy.data(),CUDA_R_32F,s.m,static_cast<long long>(dy.count),1,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
        check(cudaGetLastError()); check(cudaStreamSynchronize(execution.value));
    }
    bool candidate(bool force_range_control=false) {
        descriptor d{s.n,s.m,s.k,dw.count,dx.count,dy.count,s.k,s.k,s.m,1,true,false};
        if(!layout_eligible(d)||spans_overlap(reinterpret_cast<uintptr_t>(dw.data()),dw.count,reinterpret_cast<uintptr_t>(dy.data()),dy.count)||
            spans_overlap(reinterpret_cast<uintptr_t>(dx.data()),dx.count,reinterpret_cast<uintptr_t>(dy.data()),dy.count)||(!eligible&&!force_range_control)) return false;
        // Both packing and range/status work is inside every candidate call.
        check(cudaMemsetAsync(status.value,0,sizeof(unsigned),execution.value));
        corrected_k8<<<dim3(unsigned((s.m+31)/32),unsigned((s.n+31)/32)),128,0,execution.value>>>(dw.data(),dx.data(),dy.data(),s.m,s.n,s.k,status.value);
        check(cudaGetLastError()); unsigned result=0;
        check(cudaMemcpyAsync(&result,status.value,sizeof(unsigned),cudaMemcpyDeviceToHost,execution.value));
        check(cudaStreamSynchronize(execution.value));
        return result==0;
    }
    void publish() {
        const auto all=dy.read(execution.value);
        require(guards(all),"Candidate publication guard failure");
        published.assign(all.begin()+guard_count,all.end()-guard_count);
    }
    quality evaluate() const {
        quality q; const auto all=dy.read(execution.value); q.canaries=guards(all)&&status.guards_intact();
        q.inputs_immutable=immutable(dw.read(execution.value),w)&&immutable(dx.read(execution.value),x);
        q.complete=dy.count; q.sampled=independent.size();
        for(size_t i=0;i<dy.count;++i) {
            const double v=all[i+guard_count]; q.finite&=std::isfinite(v); q.written&=float_bits(float(v))!=float_bits(canary);
            const double e=std::abs(v-double(reference[i])); q.max_baseline=std::max(q.max_baseline,e);
            q.baseline_ratio=std::max(q.baseline_ratio,e/(2e-6+2e-6*std::abs(double(reference[i]))));
        }
        for(const auto & sample:independent) {
            const double e=std::abs(double(all[sample.index+guard_count])-sample.expected);
            q.max_double=std::max(q.max_double,e); q.double_ratio=std::max(q.double_ratio,e/(2e-6+2e-6*std::abs(sample.expected)));
        }
        return q;
    }
};
void shape_fields(std::ostream & out,shape s) {
    out<<"\"n\":"<<s.n<<",\"m\":"<<s.m<<",\"k\":"<<s.k<<",\"kind\":"<<quote(s.kind)<<",\"original_fixture\":"<<(s.original?"true":"false");
}
} // namespace

int main(int argc,char ** argv) {
    std::string output,mode="quality"; FILE * report_file=nullptr; std::ostringstream report; report<<std::setprecision(17);
    bool completed=false,all_quality=true; size_t executed=0,unavailable=0,failures=0,baseline_failures=0;
    std::string error;
    try {
        std::set<std::string> seen;
        for(int i=1;i<argc;i+=2) {
            require(i+1<argc,"Option requires value"); const std::string key=argv[i],value=argv[i+1]; require(seen.insert(key).second,"Duplicate option");
            if(key=="--output") output=value; else if(key=="--mode") mode=value; else throw std::runtime_error("Unknown option");
        }
        require(!output.empty()&&(mode=="quality"||mode=="benchmark"),"Usage: probe --output fresh.json --mode quality|benchmark");
        // Reserve report before any CUDA/model work; no existing evidence lost.
        report_file=std::fopen(output.c_str(),"wx"); require(report_file!=nullptr,"Output exists or cannot reserve it");
        const auto host_count=host_controls(); check(cudaSetDevice(0)); cudaDeviceProp device{}; check(cudaGetDeviceProperties(&device,0));
        require(device.major==8&&device.minor==6,"First owned screen requires actual SM86");
        stream controls; handle version_handle(controls.value); int cublas_version=0; check(cublasGetVersion(version_handle.value,&cublas_version));
        const auto conversion_count=device_pack_controls(controls.value);
        int runtime=0,driver=0; check(cudaRuntimeGetVersion(&runtime)); check(cudaDriverGetVersion(&driver));
        report<<"{\"mode\":"<<quote(mode)<<",\"device\":"<<quote(device.name)<<",\"sm_major\":"<<device.major<<",\"sm_minor\":"<<device.minor
            <<",\"cublas_version\":"<<cublas_version<<",\"cuda_runtime\":"<<runtime<<",\"cuda_driver\":"<<driver<<",\"host_controls_passed\":"<<host_count<<",\"device_pack_controls_passed\":"<<conversion_count
            <<",\"algorithm\":\"owned CTA32x32, K8 zeroHH plus external RNmain, author persistent scaled cross order\",\"source_only_instruction_attribution\":true,\"tensor_core_instructions_reviewed\":false,\"quality_cases\":[";
        bool comma=false;
        for(shape s:shapes()) {
            problem p(s); const auto baseline=p.evaluate(); all_quality&=baseline.passed()||!p.eligible;
            if(!baseline.passed()) { ++baseline_failures; }
            if(comma) { report<<','; }
            comma=true; report<<'{'; shape_fields(report,s); report<<",\"baseline_quality\":"; write_quality(report,baseline);
            report<<",\"eligible\":"<<(p.eligible?"true":"false")<<",\"matrix_host_device_pack_bits_agree\":true,\"pack_checked_original_elements\":"<<(p.w_pack.count+p.x_pack.count)<<",\"weight_max_reconstruction_error\":"<<number(p.w_pack.max_reconstruction_error)<<",\"input_max_reconstruction_error\":"<<number(p.x_pack.max_reconstruction_error);
            if(!p.eligible) {
                const auto before=p.published,private_before=p.dy.read(p.execution.value); const bool refused=!p.candidate();
                const auto private_after=p.dy.read(p.execution.value);
                const bool transactional=refused&&p.published==before&&private_before==private_after;
                require(transactional,"Host refusal mutated published/private output");
                // Force only the internal range rejection control: kernel may
                // write private scratch, but never publishes a failed result.
                const bool device_refused=!p.candidate(true); require(device_refused&&p.published==before,"Device range refusal publication failure");
                require(p.status.guards_intact()&&guards(p.dy.read(p.execution.value))&&immutable(p.dw.read(p.execution.value),p.w)&&immutable(p.dx.read(p.execution.value),p.x),"Device range refusal scratch/status/input integrity");
                ++unavailable; report<<",\"candidate_executed\":false,\"candidate_quality\":null,\"reason\":\"FP16 finite-range overflow; original F32 fallback separate\",\"host_refusal_no_mutation\":true,\"device_range_refusal_control_executed\":true,\"device_refusal_no_publication\":true,\"device_refusal_scratch_status_input_integrity\":true,\"host_refusal_private_output_unchanged\":true}";
                continue;
            }
            ++executed; p.dy.reset(p.execution.value); require(p.candidate(),"Eligible kernel range failure");
            const auto initial=p.evaluate(); quality all_checks=initial;
            for(int warm=0;warm<3;++warm) { require(p.candidate(),"Warm range failure"); accumulate(all_checks,p.evaluate()); }
            const auto final=p.evaluate(); accumulate(all_checks,final); const bool pass=all_checks.passed(); all_quality&=pass; if(!pass) ++failures;
            if(pass) p.publish();
            report<<",\"candidate_executed\":true,\"initial_quality\":"; write_quality(report,initial);
            report<<",\"final_quality\":"; write_quality(report,final); report<<",\"combined_initial_warm_final_quality\":"; write_quality(report,all_checks); report<<",\"all_initial_warm_final_passed\":"<<(pass?"true":"false")<<'}';
        }
        report<<"],\"timings\":["; comma=false;
        // Never time a candidate that failed any eligible shape's quality gate.
        if(mode=="benchmark"&&all_quality) for(shape s:shapes()) {
            problem p(s); if(!p.eligible) continue;
            for(int phase=0;phase<4;++phase) {
                const bool candidate=phase==1||phase==2; p.dy.reset(p.execution.value); quality all_checks;
                for(int warm=0;warm<3;++warm) { if(candidate) require(p.candidate(),"Timed warm range failure"); else p.baseline(); accumulate(all_checks,p.evaluate()); }
                require(all_checks.passed(),"Timing warm quality failed; no timed calls permitted");
                std::vector<double> times; const auto began=std::chrono::steady_clock::now();
                do { const auto t=std::chrono::steady_clock::now(); if(candidate) require(p.candidate(),"Timed range failure"); else p.baseline();
                    times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());
                } while(times.size()<20||std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count()<.1);
                const auto final=p.evaluate(); accumulate(all_checks,final); const bool pass=all_checks.passed(); all_quality&=pass; if(!pass) ++failures;
                auto sorted=times; std::sort(sorted.begin(),sorted.end()); const size_t n=sorted.size();
                const double median=(sorted[(n-1)/2]+sorted[n/2])*.5;
                if(comma) { report<<','; }
            comma=true; report<<'{'; shape_fields(report,s);
                report<<",\"phase\":"<<phase<<",\"candidate\":"<<(candidate?"true":"false")<<",\"median_ms\":"<<median<<",\"repeats\":"<<times.size()<<",\"warm_final_quality_passed\":"<<(pass?"true":"false")<<",\"final_quality\":"; write_quality(report,final);
                report<<",\"combined_warm_final_quality\":"; write_quality(report,all_checks); report<<",\"samples_ms\":["; for(size_t i=0;i<times.size();++i) { if(i) { report<<','; } report<<times[i]; } report<<"]}";
                require(pass,"Final timed quality failed; remaining timing aborted");
            }
        }
        report<<"]"; completed=true;
    } catch(const std::exception & e) { error=e.what(); all_quality=false; }
    if(teardown_failed) { error="Checked CUDA/cuBLAS teardown failed"; completed=false; all_quality=false; }
    // A failed setup/partial record is a valid small failure report, not a
    // malformed successful JSON or a passing unavailable CUDA scenario.
    std::string data;
    if(!completed) data="{\"completed\":false,\"passed\":false,\"error\":"+quote(error)+",\"candidate_shapes_executed_before_failure\":"+std::to_string(executed)+",\"cuda_qualified\":false,\"partial_evidence_json_prefix\":"+quote(report.str())+"}\n";
    else {
        report<<",\"completed\":true,\"eligible_candidate_quality_passed\":"<<(all_quality?"true":"false")<<",\"candidate_shapes_executed\":"<<executed<<",\"unavailable_candidate_shapes\":"<<unavailable<<",\"failed_candidate_cases_or_timing_phases\":"<<failures<<",\"baseline_cases_failed\":"<<baseline_failures
            <<",\"all_original_fixtures_candidate_passed\":false,\"warmup\":3,\"minimum_repeats\":20,\"minimum_seconds\":0.1,\"model_qualified\":false,\"production_promoted\":false,\"performance_qualified\":false,\"limitations\":[\"Original80000 fixture unavailable for FP16; its baseline Double failure is retained, not passed.\",\"Two half components do not exactly reconstruct arbitrary F32. Host/device pack bits and original Double output gates are mandatory.\",\"Untuned owned32x32 tile is not author tunedCUTLASS implementation; no performance prediction.\",\"Candidate timing includes on-the-fly W/X packing, range/status, RNmain/reset/correction and stream completion. Setup/readback quality outside timings.\",\"PTX/SASS instruction and externalRN ordering review remains a separate required gate; API success is not hardware attribution.\",\"Synthetic component inputs,15actual geometries; no learned full tower, pooled embedding or service qualification.\"]}\n";
        data=report.str();
    }
    try {
        if(report_file) {
            // Keep the exclusively reserved descriptor; never reopen a path or
            // overwrite an existing report after failed reservation.
            const bool ok=std::fwrite(data.data(),1,data.size(),report_file)==data.size();
            const int closed=std::fclose(report_file); report_file=nullptr; require(ok&&closed==0,"Report write failed");
        } else { std::cerr<<error<<'\n'; return 1; }
    } catch(const std::exception & e) { std::cerr<<e.what()<<'\n'; return 1; }
    std::cout<<"K8_PROBE completed="<<completed<<" eligible_quality="<<all_quality<<" executed="<<executed<<" unavailable="<<unavailable<<'\n';
    return completed&&all_quality?0:1;
}
#endif
