// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Standalone component screen. No production/upstream source or build changes.
// Compile with -ffp-contract=off; never use fast-math.
#if defined(__FAST_MATH__)
#error This exact-order screen forbids fast-math.
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>
#ifndef TSG_RMS_RESIDUAL_HOST_ONLY
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#endif

namespace {
constexpr int width = 512;
constexpr float epsilon = 1e-6f;
void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}
std::string quote(const std::string & value) {
    std::string result = "\"";
    for (unsigned char c : value) {
        if (c == '\\' || c == '"') { result += '\\'; result += char(c); }
        else if (c < 32) { char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", c); result += b; }
        else result += char(c);
    }
    return result + '"';
}
std::string environment(const char * name) {
    const char * value = std::getenv(name);
    return value ? quote(value) : "null";
}
uint32_t bits(float x) { uint32_t result; std::memcpy(&result, &x, sizeof(result)); return result; }
struct layout {
    int64_t columns = width, rows = 1;
    size_t element_stride = sizeof(float), row_stride = width * sizeof(float);
    bool f32 = true, two_dimensional = true;
};
bool valid_layout(const layout & a) {
    return a.f32 && a.two_dimensional && a.columns == width && a.rows > 0 && a.rows <= 8192 &&
        a.element_stride == sizeof(float) && a.row_stride >= width * sizeof(float) &&
        a.row_stride % sizeof(float) == 0 &&
        uint64_t(a.rows - 1) <= (std::numeric_limits<size_t>::max() - width * sizeof(float)) / a.row_stride;
}
bool admissible(const layout & x, const layout & gain, const layout & residual, float eps) {
    return valid_layout(x) && valid_layout(gain) && valid_layout(residual) &&
        residual.rows == x.rows && (gain.rows == 1 || gain.rows == x.rows) &&
        std::isfinite(eps) && eps > 0;
}
// This is the exact upstream fused RMS+MUL arithmetic, followed by a separate
// F32 ADD. Explicit temporaries and -ffp-contract=off retain both multiply
// roundings and the ADD rounding. No reciprocal/rsqrt approximation or FMA.
void row(float * y, const float * x, const float * gain, const float * residual, float eps) {
    double sum = 0;
    for (int i = 0; i < width; ++i) sum += double(x[i] * x[i]);
    const float mean = sum / width;
    const float scale = 1.0f / sqrtf(mean + eps);
    for (int i = 0; i < width; ++i) {
        const float scaled = x[i] * scale;
        const float weighted = scaled * gain[i];
        y[i] = weighted + residual[i];
    }
}
// Independent real-valued normalization, not an implementation of the F32
// square/mean/sqrt sequence. Full baseline bit identity is an additional gate.
void reference(double * y, const float * x, const float * gain, const float * residual, float eps) {
    double sum = 0;
    for (int i = 0; i < width; ++i) sum += double(x[i]) * double(x[i]);
    const double scale = 1.0 / std::sqrt(sum / width + double(eps));
    for (int i = 0; i < width; ++i) y[i] = double(x[i]) * scale * double(gain[i]) + double(residual[i]);
}
void fill_row(const std::string & family, int token, float * x, float * gain, float * residual) {
    uint32_t state = 1729U + uint32_t(token) * 3571U;
    auto random = [&] { state = state * 1664525U + 1013904223U; return float(int(state >> 8) - 8388608) / 8388608.f; };
    for (int i = 0; i < width; ++i) {
        x[i] = random() * 1.000021f;
        gain[i] = .83f + random() * .71f;
        residual[i] = random() * .71f;
        if (family == "signed-zero") {
            x[i] = i % 2 ? -0.0f : 0.0f;
            gain[i] = i % 3 ? -1.13f : .71f;
            residual[i] = i % 2 ? -0.0f : 0.0f;
        } else if (family == "subnormal") {
            x[i] = std::numeric_limits<float>::denorm_min() * float(1 + i % 17) * (i % 2 ? -1 : 1);
            residual[i] = std::numeric_limits<float>::denorm_min() * float(1 + i % 7) * (i % 3 ? -1 : 1);
        } else if (family == "extreme-finite") {
            x[i] *= i % 3 ? 1e15f : 1e-15f;
            residual[i] *= i % 2 ? 1e14f : 1e-14f;
        }
    }
    if (family == "cancellation") {
        float weighted[width], zero[width]{};
        row(weighted, x, gain, zero, epsilon);
        for (int i = 0; i < width; ++i) residual[i] = -weighted[i];
    }
}
struct quality { bool passed=true,bits=true;size_t elements=0;double max_error=0,max_ratio=0; };
void retain(quality & aggregate,const quality & current) {
    aggregate.passed&=current.passed;aggregate.bits&=current.bits;
    aggregate.max_error=std::max(aggregate.max_error,current.max_error);
    aggregate.max_ratio=std::max(aggregate.max_ratio,current.max_ratio);
    aggregate.elements+=current.elements;
}
size_t host_controls() {
    size_t count = 0;
    auto check = [&](bool value) { require(value, "Host control failed"); ++count; };
    for (int n : {1,8,16,32,128,320,1024,8192}) {
        for (bool padded : {false,true}) {
            layout x{width,n,sizeof(float),size_t(width + (padded ? 64 : 0))*sizeof(float)}, gain, residual = x;
            check(admissible(x,gain,residual,epsilon));
            gain.rows = n; check(admissible(x,gain,residual,epsilon));
            for (int which = 0; which < 7; ++which) {
                auto bad = x;
                if (which == 0) bad.columns = width - 1;
                if (which == 1) bad.rows = 0;
                if (which == 2) bad.rows = 8193;
                if (which == 3) bad.element_stride = 8;
                if (which == 4) bad.row_stride = width * sizeof(float) - 4;
                if (which == 5) bad.f32 = false;
                if (which == 6) bad.two_dimensional = false;
                check(!admissible(bad,gain,residual,epsilon));
                check(!admissible(x,gain,bad,epsilon));
            }
            for (float eps : {0.f,-1.f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()})
                check(!admissible(x,gain,residual,eps));
            gain.rows = n + 1; check(!admissible(x,gain,residual,epsilon));
        }
    }
    for (const std::string family : {"ordinary","signed-zero","subnormal","extreme-finite","cancellation"}) {
        for (int token : {0,7,31}) {
            float x[width], gain[width], residual[width], actual[width]; double expected[width];
            fill_row(family,token,x,gain,residual); row(actual,x,gain,residual,epsilon); reference(expected,x,gain,residual,epsilon);
            for (int i = 0; i < width; ++i) {
                const double error = std::abs(double(actual[i])-expected[i]), bound = 2e-6+2e-6*std::abs(expected[i]);
                check(std::isfinite(actual[i]) && error <= bound);
            }
        }
    }
    // Directed signed-zero contract: ((-0 * positive) * negative) + (-0) = +0.
    float x[width]{}, gain[width], residual[width], y[width];
    std::fill(x,x+width,-0.f); std::fill(gain,gain+width,-1.f); std::fill(residual,residual+width,-0.f);
    row(y,x,gain,residual,epsilon); check(bits(y[0]) == bits(+0.f));
    quality warm;
    retain(warm,quality{false,false,width,.01,2});
    retain(warm,quality{true,true,width,.00001,.1});
    retain(warm,quality{true,true,width,0,0});
    check(!warm.passed);check(!warm.bits);check(warm.max_error==.01&&warm.max_ratio==2);check(warm.elements==3*width);
    return count;
}
struct options {
    std::string output, mode = "quality";
    int threads = 8, warmup = 3, repeats = 20;
    double seconds = .1;
};
options parse(int argc, char ** argv) {
    options o; std::set<std::string> keys;
    for (int i = 1; i < argc; i += 2) {
        require(i+1 < argc && keys.insert(argv[i]).second,"Require unique option/value pairs");
        const std::string key = argv[i], value = argv[i+1];
        size_t used = 0;
        if (key == "--output") o.output = value;
        else if (key == "--mode") o.mode = value;
        else if (key == "--threads") { o.threads = std::stoi(value,&used); require(used == value.size(),"Invalid threads"); }
        else if (key == "--warmup") { o.warmup = std::stoi(value,&used); require(used == value.size(),"Invalid warmup"); }
        else if (key == "--repeats") { o.repeats = std::stoi(value,&used); require(used == value.size(),"Invalid repeats"); }
        else if (key == "--minimum-seconds") { o.seconds = std::stod(value,&used); require(used == value.size(),"Invalid minimum seconds"); }
        else throw std::runtime_error("Unknown option: " + key);
    }
    require(!o.output.empty() && (o.mode == "quality" || o.mode == "benchmark"),"Require fresh output and quality|benchmark mode");
    require((o.threads == 7 || o.threads == 8) && o.warmup == 3 && o.repeats == 20 && o.seconds == .1,"Only reviewed7/8workers W3/R20/min.1 policy");
    require(std::getenv("GGML_CPU_DISABLE_FUSION") == nullptr,"Original baseline requires upstream fusion enabled");
    require(std::fegetround() == FE_TONEAREST,"Required default RN floating-point environment unavailable");
    return o;
}
void write_all(int fd, const std::string & text) {
    size_t done = 0;
    while (done < text.size()) {
        const auto n = ::write(fd,text.data()+done,text.size()-done);
        if (n < 0 && errno == EINTR) continue;
        require(n > 0,"Cannot write report"); done += size_t(n);
    }
}
#ifndef TSG_RMS_RESIDUAL_HOST_ONLY
struct guarded {
    static constexpr size_t border = 64;
    static constexpr float canary = 9876.5f;
    std::vector<float> owner;
    float * base;
    size_t count, span;
    explicit guarded(size_t n) : owner(n+2*border+15,canary),
        base(reinterpret_cast<float *>((reinterpret_cast<uintptr_t>(owner.data())+63)&~uintptr_t(63))),count(n),span(n+2*border) {
        require(base >= owner.data() && base+span <= owner.data()+owner.size(),"Aligned owner bounds differ");
    }
    float * data() const { return base+border; }
    bool guards() const {
        for (const float * p = owner.data(); p < owner.data()+owner.size(); ++p)
            if ((p < data() || p >= data()+count) && bits(*p) != bits(canary)) return false;
        return true;
    }
    uint64_t fingerprint() const {
        uint64_t h = 14695981039346656037ULL;
        const auto * p = reinterpret_cast<const unsigned char *>(owner.data());
        for (size_t i = 0; i < owner.size()*sizeof(float); ++i) h = (h ^ p[i])*1099511628211ULL;
        return h;
    }
};
struct spec { const char * name; bool padded, row_gain; };
struct problem {
    spec s; int n; size_t xstride,rstride,wstride; int wrows;
    guarded x,w,r; uint64_t xhash,whash,rhash; std::vector<double> oracle;
    problem(spec input, int tokens) : s(input),n(tokens),xstride(width+(s.padded?64:0)),
        rstride(width+(s.padded?96:0)),wstride(width+(s.padded?32:0)),wrows(s.row_gain?n:1),
        x(xstride*n),w(wstride*wrows),r(rstride*n),oracle(size_t(width)*n) {
        float temp_gain[width];
        for (int token = 0; token < n; ++token) fill_row(s.name,token,x.data()+token*xstride,temp_gain,r.data()+token*rstride);
        for (int token = 0; token < wrows; ++token) {
            float tx[width],tr[width]; fill_row(s.name,token,tx,w.data()+token*wstride,tr);
        }
        for (int token = 0; token < n; ++token) {
            const float * gain = w.data()+(s.row_gain?token:0)*wstride;
            if (std::string(s.name) == "cancellation") {
                float temp[width],zero[width]{}; row(temp,x.data()+token*xstride,gain,zero,epsilon);
                for (int i = 0; i < width; ++i) r.data()[token*rstride+i] = -temp[i];
            }
            reference(oracle.data()+size_t(token)*width,x.data()+token*xstride,gain,r.data()+token*rstride,epsilon);
        }
        xhash=x.fingerprint();whash=w.fingerprint();rhash=r.fingerprint();
    }
    bool immutable() const { return x.guards() && w.guards() && r.guards() && x.fingerprint()==xhash && w.fingerprint()==whash && r.fingerprint()==rhash; }
};
struct session {
    ggml_backend_t backend=nullptr; ggml_threadpool_t pool=nullptr;
    explicit session(int threads) {
        backend=ggml_backend_cpu_init(); require(backend,"CPU backend unavailable");
        auto params=ggml_threadpool_params_default(threads); pool=ggml_threadpool_new(&params);
        if (!pool) { ggml_backend_free(backend);backend=nullptr;throw std::runtime_error("CPU pool unavailable"); }
        ggml_backend_cpu_set_n_threads(backend,threads);ggml_backend_cpu_set_threadpool(backend,pool);ggml_backend_cpu_set_use_ref(backend,false);
    }
    ~session() { if(backend)ggml_backend_free(backend);if(pool)ggml_threadpool_free(pool); }
};
struct state {
    float eps=epsilon; std::atomic<bool> instrument{false};std::array<std::atomic<uint64_t>,8> counts{};
};
layout describe(const ggml_tensor * t) { return {t->ne[0],t->ne[1],t->nb[0],t->nb[1],t->type==GGML_TYPE_F32,t->ne[2]==1&&t->ne[3]==1}; }
void fused(ggml_tensor * y,const ggml_tensor * x,const ggml_tensor * w,const ggml_tensor * r,int ith,int nth,void * userdata) {
    auto & s=*static_cast<state *>(userdata);uint64_t completed=0;
    for(int64_t token=ith;token<x->ne[1];token+=nth) {
        row(reinterpret_cast<float *>(static_cast<char *>(y->data)+token*y->nb[1]),
            reinterpret_cast<const float *>(static_cast<const char *>(x->data)+token*x->nb[1]),
            reinterpret_cast<const float *>(static_cast<const char *>(w->data)+(token%w->ne[1])*w->nb[1]),
            reinterpret_cast<const float *>(static_cast<const char *>(r->data)+token*r->nb[1]),s.eps);
        ++completed;
    }
    if(s.instrument.load(std::memory_order_relaxed))s.counts[size_t(ith)].fetch_add(completed,std::memory_order_relaxed);
}
ggml_tensor * make(ggml_context * ctx,ggml_tensor * x,ggml_tensor * w,ggml_tensor * r,state * s) {
    if(!s || !admissible(describe(x),describe(w),describe(r),s->eps))return nullptr;
    return ggml_map_custom3(ctx,x,w,r,fused,GGML_N_TASKS_MAX,s);
}
struct route {
    guarded output;state userdata;ggml_backend_t backend;ggml_context * ctx=nullptr;
    ggml_backend_graph_plan_t plan=nullptr;std::vector<ggml_backend_buffer_t> bindings;ggml_backend_buffer_t scratch=nullptr;
    bool candidate,fusion=false;
    route(session & owner,problem & p,bool owned) : output(size_t(width)*p.n),backend(owner.backend),candidate(owned) {
        try {
            ctx=ggml_init({1024*1024,nullptr,true});require(ctx,"Graph context unavailable");
            auto bind=[&](guarded & values,int rows,size_t stride,bool gain_leaf=false) {
                auto * packed=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,gain_leaf?width:int64_t(stride),rows);
                if(gain_leaf) {
                    // A weight VIEW would appear between RMS and MUL in graph
                    // postorder and defeat the original fusion being compared.
                    // Bind the same logical strided gain as a read-only leaf.
                    packed->nb[1]=stride*sizeof(float);packed->nb[2]=packed->nb[1]*size_t(rows);packed->nb[3]=packed->nb[2];
                }
                auto buffer=ggml_backend_cpu_buffer_from_ptr(values.base,values.span*sizeof(float));require(buffer,"Guarded CPU wrapper unavailable");bindings.push_back(buffer);
                require(ggml_backend_tensor_alloc(buffer,packed,values.data())==GGML_STATUS_SUCCESS,"Guarded operand allocation failed");ggml_set_input(packed);
                if(gain_leaf)return packed;
                auto * view=ggml_view_2d(ctx,packed,width,rows,stride*sizeof(float),0);
                require(ggml_backend_view_init(view)==GGML_STATUS_SUCCESS,"Padded input view bind failed");return view;
            };
            auto * x=bind(p.x,p.n,p.xstride),*w=bind(p.w,p.wrows,p.wstride,true),*residual=bind(p.r,p.n,p.rstride);
            ggml_tensor * result=nullptr,*rms=nullptr;
            if(owned) { result=make(ctx,x,w,residual,&userdata);require(result,"Eligible owned factory refused"); }
            else { rms=ggml_rms_norm(ctx,x,epsilon);result=ggml_add(ctx,ggml_mul(ctx,rms,w),residual); }
            require(ggml_is_contiguous(result)&&ggml_are_same_shape(result,x),"Complete output layout differs");ggml_set_output(result);
            auto buffer=ggml_backend_cpu_buffer_from_ptr(output.base,output.span*sizeof(float));require(buffer,"Output wrapper unavailable");bindings.push_back(buffer);
            require(ggml_backend_tensor_alloc(buffer,result,output.data())==GGML_STATUS_SUCCESS,"Guarded output binding failed");
            auto * graph=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(graph,result);
            if(!owned) {
                const ggml_op sequence[]={GGML_OP_RMS_NORM,GGML_OP_MUL};
                for(int i=0;i<graph->n_nodes;++i)if(graph->nodes[i]==rms)fusion=ggml_can_fuse(graph,i,sequence,2);
                require(fusion,"Unchanged baseline RMS+MUL not fusion-eligible");
            }
            scratch=ggml_backend_alloc_ctx_tensors(ctx,backend);require(owned||scratch,"Original intermediates unavailable");
            plan=ggml_backend_graph_plan_create(backend,graph);require(plan,"Persistent graph plan unavailable");
        }catch(...){release();throw;}
    }
    void release() {
        if(plan){ggml_backend_graph_plan_free(backend,plan);plan=nullptr;}
        ggml_backend_buffer_free(scratch);scratch=nullptr;
        for(auto buffer:bindings)ggml_backend_buffer_free(buffer);
        bindings.clear();if(ctx){ggml_free(ctx);ctx=nullptr;}
    }
    ~route(){release();}
    void compute(){require(ggml_backend_graph_plan_compute(backend,plan)==GGML_STATUS_SUCCESS,"CPU graph compute failed");}
};
quality check(problem & p,route & r,const std::vector<float> & baseline) {
    require(p.immutable()&&r.output.guards(),"Immutable input or guard changed");quality q;
    q.bits=std::memcmp(r.output.data(),baseline.data(),baseline.size()*sizeof(float))==0;q.passed=q.bits;
    for(size_t i=0;i<p.oracle.size();++i) {
        const float actual=r.output.data()[i];require(std::isfinite(actual)&&actual!=guarded::canary,"Nonfinite or unwritten output");
        const double error=std::abs(double(actual)-p.oracle[i]),bound=2e-6+2e-6*std::abs(p.oracle[i]);
        q.max_error=std::max(q.max_error,error);q.max_ratio=std::max(q.max_ratio,error/bound);q.passed&=error<=bound;++q.elements;
    }
    return q;
}
struct phase { quality instrument,warm,final;std::array<uint64_t,8> workers{};std::vector<double> samples;double median=0; };
phase run_phase(problem & p,route & r,const std::vector<float> & baseline,const options & o,bool benchmark) {
    phase t;
    std::fill(r.output.data(),r.output.data()+r.output.count,guarded::canary);
    for(auto & c:r.userdata.counts) { c.store(0); }
    r.userdata.instrument.store(true);r.compute();r.userdata.instrument.store(false);
    t.instrument=check(p,r,baseline);
    for(size_t i=0;i<t.workers.size();++i)t.workers[i]=r.userdata.counts[i].load();
    if(r.candidate)for(int i=0;i<8;++i) {
        const uint64_t expected=i<o.threads?uint64_t(p.n/o.threads+(i<p.n%o.threads)):0;
        require(t.workers[size_t(i)]==expected,"Untimed worker row coverage differs");
    }
    for(int i=0;i<o.warmup;++i) {
        r.compute();
        retain(t.warm,check(p,r,baseline));
    }
    if(benchmark) {
        const auto start=std::chrono::steady_clock::now();
        do {const auto before=std::chrono::steady_clock::now();r.compute();t.samples.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count());}
        while(t.samples.size()<size_t(o.repeats)||std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<o.seconds);
        auto sorted=t.samples;std::sort(sorted.begin(),sorted.end());t.median=(sorted[(sorted.size()-1)/2]+sorted[sorted.size()/2])/2;
    }else r.compute();
    t.final=check(p,r,baseline);return t;
}
bool phase_passed(const phase & p){return p.instrument.passed&&p.warm.passed&&p.final.passed;}
void write_phase(std::ostream & out,const char * name,bool candidate,const phase & p) {
    out<<"{\"name\":"<<quote(name)<<",\"candidate\":"<<(candidate?"true":"false")<<",\"passed\":"<<(phase_passed(p)?"true":"false")
        <<",\"full_baseline_bit_identity\":"<<(p.instrument.bits&&p.warm.bits&&p.final.bits?"true":"false")
        <<",\"full_double_max_error\":"<<std::max({p.instrument.max_error,p.warm.max_error,p.final.max_error})
        <<",\"full_double_max_error_over_bound\":"<<std::max({p.instrument.max_ratio,p.warm.max_ratio,p.final.max_ratio})
        <<",\"complete_elements_per_check\":"<<p.final.elements<<",\"warm_elements_checked\":"<<p.warm.elements<<",\"untimed_worker_rows\":[";
    for(size_t i=0;i<p.workers.size();++i)out<<(i?",":"")<<p.workers[i];
    out<<"],\"median_ms\":"<<(p.samples.empty()?"null":std::to_string(p.median))<<",\"samples_ms\":[";
    for(size_t i=0;i<p.samples.size();++i)out<<(i?",":"")<<p.samples[i];
    out<<"]}";
}
bool execute(std::ostream & out,const options & o,bool benchmark) {
    session owner(o.threads);bool passed=true,comma=false;size_t count=0;
    for(const auto s:{spec{"ordinary",false,false},spec{"padded-row-gain",true,true},spec{"signed-zero",true,false},
        spec{"subnormal",false,true},spec{"extreme-finite",true,false},spec{"cancellation",true,true}}) {
        for(int n:{1,8,16,32,128,320,1024,8192}) {
            problem p(s,n);route baseline(owner,p,false),candidate(owner,p,true);baseline.compute();
            std::vector<float> original(baseline.output.data(),baseline.output.data()+baseline.output.count);
            const auto initial=check(p,baseline,original);passed&=initial.passed;
            const auto a=run_phase(p,baseline,original,o,benchmark),b=run_phase(p,candidate,original,o,benchmark),
                c=run_phase(p,candidate,original,o,benchmark),d=run_phase(p,baseline,original,o,benchmark);
            const bool success=initial.passed&&phase_passed(a)&&phase_passed(b)&&phase_passed(c)&&phase_passed(d);passed&=success;
            if(comma)out<<',';
            comma=true;++count;
            out<<"{\"family\":"<<quote(s.name)<<",\"tokens\":"<<n<<",\"width\":512,\"padded_input_rows\":"<<(s.padded?"true":"false")
                <<",\"gain_rows\":"<<p.wrows<<",\"residual_rows\":"<<n<<",\"upstream_rms_mul_fusion_eligible\":true,\"initial_baseline_double_passed\":"<<(initial.passed?"true":"false")
                <<",\"passed\":"<<(success?"true":"false")<<",\"candidate_over_baseline_ratios\":";
            if(benchmark)out<<'['<<b.median/a.median<<','<<c.median/d.median<<']';
            else out<<"null";
            out<<",\"phases\":[";
            write_phase(out,"baseline-a",false,a);out<<',';write_phase(out,"candidate-a",true,b);out<<',';
            write_phase(out,"candidate-b",true,c);out<<',';write_phase(out,"baseline-b",false,d);out<<"]}";
        }
    }
    require(count==48,"Fixture coverage differs");return passed;
}
#endif
} // namespace

int main(int argc,char ** argv) {
    if(argc==2&&std::strcmp(argv[1],"--host-controls")==0) {
        try{const auto count=host_controls();std::cout<<"RMS_RESIDUAL_HOST_CONTROLS passed="<<count<<" failed=0 native_graphs=0 model_executed=0\n";return 0;}
        catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}
    }
    options o;
    try{o=parse(argc,argv);}catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 2;}
    const int fd=::open(o.output.c_str(),O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC,0644);
    if(fd<0){std::cerr<<"Require fresh output\n";return 2;}
    bool completed=false,passed=false;std::string error;std::ostringstream cases,preflight;
    cases<<std::setprecision(17);preflight<<std::setprecision(17);
    try {
        (void)host_controls();
#ifdef TSG_RMS_RESIDUAL_HOST_ONLY
        throw std::runtime_error("Native graph compilation unavailable in host-only executable");
#else
        if(o.mode=="benchmark")require(execute(preflight,o,false),"Complete all-case untimed quality preflight failed; no timing allowed");
        passed=execute(cases,o,o.mode=="benchmark");completed=true;
#endif
    }catch(const std::exception & e){error=e.what();}
    std::ostringstream report;report<<std::setprecision(17)<<"{\"completed\":"<<(completed?"true":"false")<<",\"passed\":"<<(completed&&passed?"true":"false")
        <<",\"mode\":"<<quote(o.mode)<<",\"threads\":"<<o.threads<<",\"cases_expected\":48,\"cases\":["<<cases.str()
        <<"],\"untimed_all_case_preflight\":["<<preflight.str()<<"],\"error\":"<<(error.empty()?"null":quote(error))
        <<",\"warmup\":3,\"minimum_repeats\":20,\"minimum_seconds\":0.1,\"model_qualified\":false,\"production_promoted\":false,\"performance_qualified\":false,\"environment\":{";
    bool comma=false;
    for(const char * key:{"GGML_CPU_DISABLE_FUSION","TS_GGML_CPU_THREADS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","OMP_WAIT_POLICY","GOMP_SPINCOUNT","KMP_BLOCKTIME","GGML_CPU_TILED_MM","GGML_TOTAL_THREADS"}) {
        if(comma)report<<',';
        comma=true;report<<quote(key)<<':'<<environment(key);
    }
    report<<"},\"limitations\":[\"Standalone unchanged-upstream component, not model/HTTP/llama parity.\",\"Original fusion eligibility is source/graph checked; no dispatch counter or barrier timing inferred.\",\"Shared512-vector andper-row512-gain,2Dinput/residual paddedrows; contiguousoutput. No higher-axis orquantizedgeometry qualification.\",\"Untimed instrumentation provesassignedrows; timedcallbackretainsdisabledatomiccheck/localrowcount.\",\"Full baseline bit identity and complete independentDouble bound2e-6+2e-6abs(reference) required.\",\"No optional learnedlayer-scale/GELU/newapproximation.\"]}";
    try{write_all(fd,report.str());require(::close(fd)==0,"Cannot close report");}
    catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}
    if(!error.empty())std::cerr<<error<<'\n';
    return completed&&passed?0:1;
}
