// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Standalone API-success residual TF32 screen. No production/upstream source change.
// Derived from the preserved attribution screen SHA256 4f7dc84f567777b794500c08f4107a1de5c3882d9ac43396e074dbb222cb01b2.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t s) { if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
void check(cublasStatus_t s) { if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS status " + std::to_string(s)); }
std::string quote(const std::string & value) {
    std::string out = "\"";
    for (unsigned char c : value) {
        if (c == '\\' || c == '"') { out += '\\'; out += char(c); }
        else if (c < 32) { char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", c); out += b; }
        else out += char(c);
    }
    return out + '"';
}
std::string number(double value) {
    if (!std::isfinite(value)) return "null";
    std::ostringstream out; out << std::setprecision(17) << value; return out.str();
}
uint32_t bits(float x) { uint32_t b; std::memcpy(&b, &x, 4); return b; }
float from_bits(uint32_t b) { float x; std::memcpy(&x, &b, 4); return x; }
constexpr float canary = -1234567.25f;
struct stream {
    cudaStream_t value = nullptr;
    stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~stream() { if (value) { cudaStreamSynchronize(value); cudaStreamDestroy(value); } }
    stream(const stream &) = delete;
};
struct handle {
    cublasHandle_t value = nullptr;
    explicit handle(cudaStream_t execution) {
        try {
            check(cublasCreate(&value)); check(cublasSetStream(value, execution));
            check(cublasSetMathMode(value, static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
        } catch (...) { if (value) cublasDestroy(value); throw; }
    }
    ~handle() { if (value) cublasDestroy(value); }
    handle(const handle &) = delete;
};
struct guarded {
    float * base = nullptr;
    size_t count;
    explicit guarded(size_t n) : count(n) { check(cudaMalloc(reinterpret_cast<void **>(&base), (n + 32) * sizeof(float))); }
    ~guarded() { if (base) cudaFree(base); }
    guarded(const guarded &) = delete;
    float * data() const { return base + 16; }
    void reset(cudaStream_t s, const std::vector<float> * interior = nullptr) {
        std::vector<float> all(count + 32, canary);
        if (interior) { if (interior->size() != count) throw std::runtime_error("Guarded upload shape"); std::copy(interior->begin(), interior->end(), all.begin() + 16); }
        check(cudaMemcpyAsync(base, all.data(), all.size() * 4, cudaMemcpyHostToDevice, s));
        // Pageable host storage remains alive until this transfer completes.
        check(cudaStreamSynchronize(s));
    }
    std::vector<float> read() const {
        std::vector<float> all(count + 32); check(cudaMemcpy(all.data(), base, all.size() * 4, cudaMemcpyDeviceToHost)); return all;
    }
};
bool guards(const std::vector<float> & x) {
    for (size_t i = 0; i < 16; ++i) if (x[i] != canary || x[x.size() - 1 - i] != canary) return false;
    return true;
}
bool same(const std::vector<float> & guarded_values, const std::vector<float> & expected) {
    return guards(guarded_values) && guarded_values.size() == expected.size() + 32 &&
        std::memcmp(guarded_values.data() + 16, expected.data(), expected.size() * 4) == 0;
}

__device__ __forceinline__ float tf32_rn(float x) {
    const uint32_t b = __float_as_uint(x);
    if ((b & 0x7fffffffu) >= 0x7f800000u) return x;
    // 23->10 explicit mantissa bits, nearest with ties to an even retained bit.
    return __uint_as_float((b + 0xfffu + ((b >> 13) & 1u)) & 0xffffe000u);
}
__global__ void split_tf32(const float * source, float * high, float * low, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
        const float h = tf32_rn(source[i]); high[i] = h;
        // Nonfinite values/finite RN overflow preserve classification and never
        // evaluate inf-inf. Such inputs are rejected before candidate GEMM.
        low[i] = isfinite(source[i]) && isfinite(h) ? tf32_rn(__fsub_rn(source[i], h)) : 0.f;
    }
}
__global__ void combine(const float * hl, const float * lh, const float * hh, const float * ll, float * y, size_t count, bool four) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
        const float cross = __fadd_rn(hl[i], lh[i]);
        const float residual = four ? __fadd_rn(cross, ll[i]) : cross;
        y[i] = __fadd_rn(residual, hh[i]);
    }
}
int blocks(size_t n) { return int(std::min(size_t(65535), (n + 255) / 256)); }
void split(const guarded & x, guarded & h, guarded & l, cudaStream_t s) {
    split_tf32<<<blocks(x.count),256,0,s>>>(x.data(),h.data(),l.data(),x.count); check(cudaGetLastError());
}
// Independent host rounding: choose between adjacent TF32 grid points using
// Double distances, including a Double finite representation of 2^128.
float host_tf32(float x) {
    const uint32_t b = bits(x), magnitude = b & 0x7fffffffu, sign = b & 0x80000000u;
    if (magnitude >= 0x7f800000u) return x;
    const uint32_t lower_bits = magnitude / 8192u * 8192u, upper_bits = lower_bits + 8192u;
    const double lower = from_bits(lower_bits), upper = upper_bits == 0x7f800000u ? std::ldexp(1.,128) : double(from_bits(upper_bits));
    const double value = std::abs(double(x)), a = value - lower, c = upper - value;
    const bool up = c < a || (c == a && ((lower_bits / 8192u) & 1u));
    return from_bits(sign | (up ? upper_bits : lower_bits));
}
void host_split(const std::vector<float> & x, std::vector<float> & high, std::vector<float> & low, bool finite_required) {
    high.resize(x.size()); low.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        high[i] = host_tf32(x[i]);
        low[i] = std::isfinite(x[i]) && std::isfinite(high[i]) ? host_tf32(float(x[i] - high[i])) : 0.f;
        if (finite_required && (!std::isfinite(x[i]) || !std::isfinite(high[i]) || !std::isfinite(low[i])))
            throw std::runtime_error("Nonfinite/finite-overflow TF32 decomposition is unavailable");
    }
}
void conversion_fixtures(cudaStream_t execution, std::ostream & out) {
    const uint32_t patterns[] = {0,0x80000000u,1,0x80000001u,0x1000u,0x1001u,0x3000u,0x7fffffu,0x800000u,
        0x3f800000u,0x3f801000u,0x3f803000u,0xbf801000u,0xbf803000u,0x3fffffffu,
        0x7f7fdfffu,0x7f7fe000u,0x7f7ff000u,0x7f7fffffu,0xff7fffffu,0x7f800000u,0xff800000u,0x7fc12345u,0xffc12345u};
    std::vector<float> x; for (uint32_t b : patterns) x.push_back(from_bits(b));
    std::vector<float> h,l;host_split(x,h,l,false);guarded dx(x.size()),dh(x.size()),dl(x.size());
    dx.reset(execution,&x);dh.reset(execution);dl.reset(execution);split(dx,dh,dl,execution);check(cudaStreamSynchronize(execution));
    const auto actual_h=dh.read(),actual_l=dl.read();
    if (!same(dx.read(),x)||!same(actual_h,h)||!same(actual_l,l))throw std::runtime_error("TF32 RN-even conversion fixture mismatch");
    out << "\"conversion_fixtures\":[";
    for(size_t i=0;i<x.size();++i){if(i)out<<',';
        const char*classification=!std::isfinite(x[i])?"nonfinite":!std::isfinite(h[i])?"finite-rounding-overflow":std::fpclassify(x[i])==FP_SUBNORMAL?"finite-subnormal":"finite";
        out<<"{\"source_bits\":"<<patterns[i]<<",\"high_bits\":"<<bits(h[i])<<",\"low_bits\":"<<bits(l[i])<<",\"classification\":"<<quote(classification)<<",\"matrix_eligible\":"<<(std::isfinite(x[i])&&std::isfinite(h[i])&&std::isfinite(l[i])?"true":"false")<<",\"passed\":true}";}
    out << "],";
}
struct shape { int n,m,k; const char * kind; };
struct sample { size_t index; double expected; };
struct quality {
    bool finite=true, written=true, canaries=true, immutable=true, split_exact=true;
    double max_double=0,max_double_ratio=0,max_baseline=0,max_baseline_ratio=0;
    bool passed()const{return finite&&written&&canaries&&immutable&&split_exact&&max_double_ratio<=1&&max_baseline_ratio<=1;}
};
void write_quality(std::ostream & out,const quality&q){out<<"{\"passed\":"<<(q.passed()?"true":"false")<<",\"finite\":"<<(q.finite?"true":"false")
    <<",\"written\":"<<(q.written?"true":"false")<<",\"canaries\":"<<(q.canaries?"true":"false")<<",\"immutable\":"<<(q.immutable?"true":"false")
    <<",\"split_matches_independent_RN_even\":"<<(q.split_exact?"true":"false")<<",\"max_double_error\":"<<number(q.max_double)<<",\"max_double_error_over_bound\":"<<number(q.max_double_ratio)
    <<",\"max_full_baseline_error\":"<<number(q.max_baseline)<<",\"max_full_baseline_error_over_bound\":"<<number(q.max_baseline_ratio)<<'}';}
} // namespace

int main(int argc,char**argv)try{
    if(argc!=3||std::string(argv[1])!="--output")throw std::runtime_error("Usage: probe --output report.json");
    if(std::filesystem::exists(argv[2]))throw std::runtime_error("Preserve prior evidence; choose a new report");
    std::ofstream out(argv[2]);if(!out)throw std::runtime_error("Cannot create report");out<<std::setprecision(17);
    const char*override_value=std::getenv("NVIDIA_TF32_OVERRIDE");
    if(override_value&&std::string(override_value)=="0") {out<<"{\"completed\":false,\"available\":false,\"NVIDIA_TF32_OVERRIDE\":\"0\",\"reason\":\"Explicit TF32 disable; no candidate executed.\"}\n";return 2;}
    check(cudaSetDevice(0));cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));
    if(prop.major<8){out<<"{\"completed\":false,\"available\":false,\"reason\":\"Require SM80 or higher.\"}\n";return 2;}
    stream execution;handle pedantic(execution.value),fast(execution.value);int version=0;check(cublasGetVersion(pedantic.value,&version));
    out<<"{\"screen\":\"api-success-rounded-products\",\"api_success_is_tf32_attribution\":false,\"prior_attribution_screen_sha256\":\"4f7dc84f567777b794500c08f4107a1de5c3882d9ac43396e074dbb222cb01b2\",\"device\":"<<quote(prop.name)<<",\"SM_major\":"<<prop.major<<",\"SM_minor\":"<<prop.minor<<",\"cublas_version\":"<<version
       <<",\"NVIDIA_TF32_OVERRIDE\":"<<(override_value?quote(override_value):"null")<<",\"candidate_compute\":\"CUBLAS_COMPUTE_32F_FAST_TF32\",\"baseline_compute\":\"CUBLAS_COMPUTE_32F_PEDANTIC\",";
    conversion_fixtures(execution.value,out);out<<"\"records\":[";bool comma=false;size_t pass=0,fail=0,candidate_phases=0,unavailable=0;
    size_t candidate_pass=0,candidate_fail=0,baseline_pass=0,baseline_fail=0;
    for(const shape s:{shape{36,768,768,"normalized"},shape{2304,768,768,"normalized"},shape{2340,3072,768,"normalized"},shape{1089,768,3072,"normalized"},shape{129,64,64,"unrounded"},shape{33,64,64,"large_cancellation"}}){
        const size_t wc=size_t(s.k)*s.m,xc=size_t(s.k)*s.n,yc=size_t(s.m)*s.n;
        std::vector<float>w(wc),x(xc),reference(yc),wh,wl,xh,xl;
        for(size_t i=0;i<wc;++i)w[i]=float(std::sin(double(i*17+3)*.013)*.21+std::cos(double(i+11)*.003)*.007);
        for(size_t i=0;i<xc;++i)x[i]=float(std::sin(double(i*13+7)*.017)*.83+std::cos(double(i+5)*.009)*.011);
        if(std::string(s.kind)=="unrounded"){for(size_t i=0;i<wc;++i)w[i]=1.f+float(i%31)*.0000013f;for(size_t i=0;i<xc;++i)x[i]=((i%s.k)%2?-1.f:1.f)+float(i%29)*.0000021f;}
        if(std::string(s.kind)=="large_cancellation"){for(size_t i=0;i<wc;++i)w[i]=80000.f+float(i%31)*.015625f;for(size_t i=0;i<xc;++i)x[i]=((i%s.k)%2?-80000.f:80000.f)+float(i%23)*.015625f;}
        host_split(w,wh,wl,true);host_split(x,xh,xl,true);
        guarded dw(wc),dx(xc),dwh(wc),dwl(wc),dxh(xc),dxl(xc),y(yc),hl(yc),lh(yc),hh(yc),ll(yc);
        dw.reset(execution.value,&w);dx.reset(execution.value,&x);for(auto*p:{&dwh,&dwl,&dxh,&dxl,&y,&hl,&lh,&hh,&ll})p->reset(execution.value);
        const auto cold_started=std::chrono::steady_clock::now();split(dw,dwh,dwl,execution.value);check(cudaStreamSynchronize(execution.value));
        const double cold_weight_pack_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-cold_started).count();
        split(dx,dxh,dxl,execution.value);check(cudaStreamSynchronize(execution.value));
        const float one=1,zero=0;
        auto submit=[&](bool tf32,const guarded&a,const guarded&b,guarded&dst){return cublasGemmStridedBatchedEx(tf32?fast.value:pedantic.value,CUBLAS_OP_T,CUBLAS_OP_N,s.m,s.n,s.k,&one,a.data(),CUDA_R_32F,s.k,0,b.data(),CUDA_R_32F,s.k,xc,&zero,dst.data(),CUDA_R_32F,s.m,yc,1,tf32?CUBLAS_COMPUTE_32F_FAST_TF32:CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT);};
        auto gemm=[&](bool tf32,const guarded&a,const guarded&b,guarded&dst){check(submit(tf32,a,b,dst));};
        auto run=[&](int products){if(products==0)gemm(false,dw,dx,y);else{split(dx,dxh,dxl,execution.value);gemm(true,dwh,dxl,hl);gemm(true,dwl,dxh,lh);gemm(true,dwh,dxh,hh);if(products==4)gemm(true,dwl,dxl,ll);combine<<<blocks(yc),256,0,execution.value>>>(hl.data(),lh.data(),hh.data(),ll.data(),y.data(),yc,products==4);check(cudaGetLastError());}check(cudaStreamSynchronize(execution.value));};
        run(0);const auto initial=y.read();if(!guards(initial))throw std::runtime_error("Initial F32 output guard");std::copy(initial.begin()+16,initial.end()-16,reference.begin());
        for(float value:reference)if(!std::isfinite(value)||value==canary)throw std::runtime_error("Initial F32 output invalid");
        // Arithmetic-effect canary only; it does not prove Tensor Core instructions.
        const auto tf32_status=submit(true,dw,dx,y);
        if(tf32_status!=CUBLAS_STATUS_SUCCESS&&tf32_status!=CUBLAS_STATUS_NOT_SUPPORTED)check(tf32_status);
        gemm(false,dwh,dxh,hh);check(cudaStreamSynchronize(execution.value));const auto effective=y.read(),high_reference=hh.read();
        double original_l2=0,high_l2=0;bool canary_finite=tf32_status==CUBLAS_STATUS_SUCCESS&&guards(effective)&&guards(high_reference);
        for(size_t i=0;i<yc;++i){const double a=double(effective[i+16])-reference[i],b=double(effective[i+16])-high_reference[i+16];original_l2+=a*a;high_l2+=b*b;canary_finite&=std::isfinite(effective[i+16])&&std::isfinite(high_reference[i+16]);}
        const bool arithmetic_qualified=canary_finite&&original_l2>1e-24&&high_l2<=.0025*original_l2;
        // API execution eligibility is separate from the original attribution
        // heuristic. Each actual rounded product is preflighted on this same
        // private handle/stream; a successful API does not prove Tensor Cores.
        const cublasStatus_t rounded_status[] = {
            submit(true,dwh,dxl,hl), submit(true,dwl,dxh,lh),
            submit(true,dwh,dxh,hh), submit(true,dwl,dxl,ll)};
        for(const auto status:rounded_status)
            if(status!=CUBLAS_STATUS_SUCCESS&&status!=CUBLAS_STATUS_NOT_SUPPORTED)check(status);
        check(cudaStreamSynchronize(execution.value));
        std::vector<sample>independent;const size_t samples=std::min(yc,size_t(4096));
        for(size_t j=0;j<samples;++j){const size_t i=samples==1?0:j*(yc-1)/(samples-1),row=i%s.m,token=i/s.m;double value=0;for(int k=0;k<s.k;++k)value+=double(w[row*s.k+k])*x[token*s.k+k];independent.push_back({i,value});}
        auto evaluate=[&](int products){quality q;const auto values=y.read();q.canaries=guards(values);q.immutable=same(dw.read(),w)&&same(dx.read(),x);q.split_exact=same(dwh.read(),wh)&&same(dwl.read(),wl)&&same(dxh.read(),xh)&&same(dxl.read(),xl);
            for(auto*p:{&hl,&lh,&hh,&ll}){const auto a=p->read();q.canaries&=guards(a);if(products==0||(p==&ll&&products==3))continue;for(size_t i=16;i<a.size()-16;++i){q.finite&=std::isfinite(a[i]);q.written&=a[i]!=canary;}}
            for(size_t i=0;i<yc;++i){q.finite&=std::isfinite(values[i+16]);q.written&=values[i+16]!=canary;const double error=std::abs(double(values[i+16])-reference[i]);q.max_baseline=std::max(q.max_baseline,error);q.max_baseline_ratio=std::max(q.max_baseline_ratio,error/(2e-6+2e-6*std::abs(double(reference[i]))));}
            for(const auto&item:independent){const double error=std::abs(double(values[item.index+16])-item.expected);q.max_double=std::max(q.max_double,error);q.max_double_ratio=std::max(q.max_double_ratio,error/(2e-6+2e-6*std::abs(item.expected)));}return q;};
        for(int products:{3,4}){
            const bool api_available=std::all_of(rounded_status,rounded_status+products,
                [](cublasStatus_t status){return status==CUBLAS_STATUS_SUCCESS;});
            if(!api_available)++unavailable;
            for(int phase=0;phase<(api_available?4:1);++phase){const int executed=api_available&&(phase==1||phase==2)?products:0;
            for(auto*p:{&y,&hl,&lh,&hh,&ll})p->reset(execution.value);for(int i=0;i<3;++i)run(executed);const auto warm=evaluate(executed);std::vector<double>times;const auto start=std::chrono::steady_clock::now();
            do{const auto before=std::chrono::steady_clock::now();run(executed);times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count());}while(times.size()<20||std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<.1);
            const auto final=evaluate(executed);const bool phase_pass=warm.passed()&&final.passed();
            if(phase_pass)++pass;else++fail;
            if(executed){++candidate_phases;if(phase_pass)++candidate_pass;else++candidate_fail;}
            else{if(phase_pass)++baseline_pass;else++baseline_fail;}
            auto sorted=times;std::sort(sorted.begin(),sorted.end());const size_t n=times.size();const double median=n%2?sorted[n/2]:(sorted[n/2-1]+sorted[n/2])*.5;
            if(comma)out<<',';comma=true;out<<"{\"n\":"<<s.n<<",\"m\":"<<s.m<<",\"k\":"<<s.k<<",\"kind\":"<<quote(s.kind)<<",\"products\":"<<products<<",\"phase\":"<<phase<<",\"executed_products\":"<<executed
               <<",\"available\":"<<(api_available?"true":"false")
               <<",\"execution_basis\":\"required-rounded-product-API-success\",\"required_rounded_product_api_success\":"<<(api_available?"true":"false")
               <<",\"rounded_product_API_statuses\":{\"high_low\":"<<int(rounded_status[0])<<",\"low_high\":"<<int(rounded_status[1])<<",\"high_high\":"<<int(rounded_status[2])<<",\"low_low\":"<<int(rounded_status[3])<<'}'
               <<",\"TF32_preflight_role\":\"original-input-attribution-canary\",\"TF32_preflight_status\":"<<int(tf32_status)
               <<",\"TF32_attribution_status\":"<<quote(arithmetic_qualified?"arithmetic-effect-canary-passed-instructions-unproven":"unknown-canary-not-qualified")
               <<",\"effective_TF32_arithmetic_canary\":"<<(arithmetic_qualified?"true":"false")<<",\"tensor_core_instructions_proven\":false,\"canary_finite\":"<<(canary_finite?"true":"false")<<",\"canary_original_l2_error\":"<<number(std::sqrt(original_l2))<<",\"canary_TF32_high_l2_error\":"<<number(std::sqrt(high_l2))<<",\"cold_weight_pack_ms\":"<<cold_weight_pack_ms<<",\"repeats\":"<<n<<",\"median_ms\":"<<median<<",\"warm_quality\":";write_quality(out,warm);out<<",\"final_quality\":";write_quality(out,final);out<<",\"samples_ms\":[";for(size_t i=0;i<times.size();++i){if(i)out<<',';out<<times[i];}out<<"]}";out.flush();}}
    }
    out<<"],\"completed\":true,\"exit_code_means_execution_only\":true,\"phases_passed\":"<<pass<<",\"phases_failed\":"<<fail<<",\"candidate_phases_executed\":"<<candidate_phases<<",\"unavailable_candidate_shapes\":"<<unavailable
       <<",\"candidate_phases_passed\":"<<candidate_pass<<",\"candidate_phases_failed\":"<<candidate_fail
       <<",\"baseline_phases_passed\":"<<baseline_pass<<",\"baseline_phases_failed\":"<<baseline_fail
       <<",\"unavailable_candidate_shapes_unit\":\"shape-product-combinations\""
       <<",\"quality_all_executed_phases_passed\":"<<(fail==0&&candidate_phases>0?"true":"false")<<",\"warmup\":3,\"minimum_repeats\":20,\"minimum_seconds\":0.1,\"model_qualified\":false,\"production_promoted\":false,\"limitations\":[\"Synthetic six-shape operator screen; no model/HTTP qualification.\",\"Original arithmetic-effect canary is retained as attribution metadata, not an execution gate. API success does not prove TF32 or Tensor Core execution; failed canary attribution remains unknown and separate trace is required.\",\"Timing includes activation splitting, explicit F32 addition, submission and completion; weight splitting is separately recorded cold and excluded from warmed phases.\",\"Original Double and complete pedantic-baseline bounds2e-6+2e-6*abs(reference) are unchanged; failed phases remain timed.\",\"Nonfinite or finite-overflow matrix decompositions are unavailable; conversion fixtures preserve classification without inf-inf. Subnormal packing is tested but does not qualify hardware GEMM subnormal behavior.\"]}\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
