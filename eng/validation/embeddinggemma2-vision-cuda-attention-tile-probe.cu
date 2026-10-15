// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Standalone head64 tile experiment; no production or upstream dispatch change.
// Kernel body copied from ggml_ops_attention_precision.cu, with only QUERIES,
// KEYS and launch bounds made template parameters. Plain-expf arithmetic is unchanged.
// Separately labeled BOUNDED_EXP variants use __expf only for [-80,0].
// Production source SHA256: 1a610f3f4b264c8fc7e53b1c01f0eba04899f8b7e5457a19f0d4b89d16454b71
#include "ggml.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace {
void check(cudaError_t status) { if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); }
struct device_buffer {
    void * data = nullptr;
    explicit device_buffer(size_t bytes) { check(cudaMalloc(&data, bytes)); }
    ~device_buffer() { cudaFree(data); }
    device_buffer(const device_buffer &) = delete;
};
struct stream_owner {
    cudaStream_t value = nullptr;
    stream_owner() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~stream_owner() { cudaStreamDestroy(value); }
};
std::string quote(const std::string & value) {
    std::string result = "\"";
    for (unsigned char c : value) {
        if (c == '\\' || c == '"') { result += '\\'; result += char(c); }
        else if (c < 32) { char text[8]; std::snprintf(text, sizeof(text), "\\u%04x", c); result += text; }
        else result += char(c);
    }
    return result + "\"";
}
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

template <bool BOUNDED_EXP>
__device__ __forceinline__ float probability_exp(float value) {
    if (BOUNDED_EXP && value >= -80.f && value <= 0.f) return __expf(value);
    return expf(value);
}

template <int WIDTH, int QUERIES, int KEYS, bool BOUNDED_EXP = false>
__global__ __launch_bounds__(QUERIES) void vision_kernel(view q, view k, view v, float scale, float * output) {
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
            const float correction = probability_exp<BOUNDED_EXP>(maximum - next);
            denominator *= correction;
#pragma unroll 1
            for (int j = 0; j < KEYS; ++j) {
                const float probability = j < count ? probability_exp<BOUNDED_EXP>(probabilities[j * QUERIES + threadIdx.x] - next) : 0;
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


struct input {
    int rows, heads, batches;
    size_t offset = 64, nb[4];
    std::vector<float> values;
    std::vector<unsigned char> bytes;
    std::unique_ptr<device_buffer> gpu;
    input(int n, int h, int b, bool padded, bool interleaved) : rows(n), heads(h), batches(b), values(size_t(64) * n * h * b) {
        nb[0] = sizeof(float) * (interleaved ? 2 : 1);
        nb[1] = nb[0] * 64 + (padded ? 12 : 0);
        nb[2] = nb[1] * (rows + (padded ? 2 : 0));
        nb[3] = nb[2] * (heads + (padded ? 1 : 0));
        offset += padded ? 8 : 0;
        bytes.resize(offset + nb[3] * batches + 64, 0xff);
        gpu = std::make_unique<device_buffer>(bytes.size());
    }
    size_t index(int x, int y, int h, int b) const { return x + size_t(64) * (y + size_t(rows) * (h + size_t(heads) * b)); }
    float at(int x, int y, int h, int b) const { return values[index(x, y, h, b)]; }
    void upload(cudaStream_t stream) {
        for (int b = 0; b < batches; ++b) for (int h = 0; h < heads; ++h)
        for (int y = 0; y < rows; ++y) for (int x = 0; x < 64; ++x) {
            const float value = at(x, y, h, b);
            std::memcpy(bytes.data() + offset + x * nb[0] + y * nb[1] + h * nb[2] + b * nb[3], &value, sizeof(float));
        }
        check(cudaMemcpyAsync(gpu->data, bytes.data(), bytes.size(), cudaMemcpyHostToDevice, stream));
    }
    bool unchanged() const {
        std::vector<unsigned char> actual(bytes.size());
        check(cudaMemcpy(actual.data(), gpu->data, actual.size(), cudaMemcpyDeviceToHost));
        return actual == bytes;
    }
    view as_view() const {
        view value{}; value.data = static_cast<const char *>(gpu->data) + offset; value.type = GGML_TYPE_F32;
        value.ne[0] = 64; value.ne[1] = rows; value.ne[2] = heads; value.ne[3] = batches;
        for (int i = 0; i < 4; ++i) value.nb[i] = nb[i];
        return value;
    }
};
struct variant { int queries, keys; bool bounded_exp; const char * name; };
constexpr variant variants[] = {{128,16,false,"Q128-K16"},{128,32,false,"Q128-K32"},
    {128,48,false,"Q128-K48"},{64,16,false,"Q64-K16"},{64,32,false,"Q64-K32"},
    {128,16,true,"Q128-K16-bounded-exp"},{128,32,true,"Q128-K32-bounded-exp"},{64,32,true,"Q64-K32-bounded-exp"}};
constexpr int variant_count = sizeof(variants) / sizeof(variants[0]);
void resource_attributes(std::ostream & out) {
    cudaFuncAttributes attributes[variant_count]{}; int blocks[variant_count]{};
#define ATTRIBUTES(INDEX,Q,K,EXP) \
    check(cudaFuncGetAttributes(&attributes[INDEX],vision_kernel<64,Q,K,EXP>)); \
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks[INDEX],vision_kernel<64,Q,K,EXP>,Q,0))
    ATTRIBUTES(0,128,16,false); ATTRIBUTES(1,128,32,false); ATTRIBUTES(2,128,48,false);
    ATTRIBUTES(3,64,16,false); ATTRIBUTES(4,64,32,false);
    ATTRIBUTES(5,128,16,true); ATTRIBUTES(6,128,32,true); ATTRIBUTES(7,64,32,true);
#undef ATTRIBUTES
    out << "\"kernel_resources\":[";
    for(int i=0;i<variant_count;++i) {
        if(i)out<<',';
        out << "{\"variant\":" << quote(variants[i].name) << ",\"bounded_exp\":" << (variants[i].bounded_exp?"true":"false") << ",\"registers_per_thread\":" << attributes[i].numRegs
            << ",\"static_shared_bytes\":" << attributes[i].sharedSizeBytes << ",\"local_bytes_per_thread\":" << attributes[i].localSizeBytes
            << ",\"maximum_active_blocks_per_SM\":" << blocks[i] << '}';
    }
    out << "],";
}
void launch(int variant_id, const input & q, const input & k, const input & v, float scale, float * output, cudaStream_t stream) {
    const auto & variant = variants[variant_id];
    const dim3 grid(unsigned((q.rows + variant.queries - 1) / variant.queries), unsigned(q.heads), unsigned(q.batches));
    const auto qv = q.as_view(), kv = k.as_view(), vv = v.as_view();
    switch (variant_id) {
        case 0: vision_kernel<64,128,16><<<grid,128,0,stream>>>(qv,kv,vv,scale,output); break;
        case 1: vision_kernel<64,128,32><<<grid,128,0,stream>>>(qv,kv,vv,scale,output); break;
        case 2: vision_kernel<64,128,48><<<grid,128,0,stream>>>(qv,kv,vv,scale,output); break;
        case 3: vision_kernel<64,64,16><<<grid,64,0,stream>>>(qv,kv,vv,scale,output); break;
        case 4: vision_kernel<64,64,32><<<grid,64,0,stream>>>(qv,kv,vv,scale,output); break;
        case 5: vision_kernel<64,128,16,true><<<grid,128,0,stream>>>(qv,kv,vv,scale,output); break;
        case 6: vision_kernel<64,128,32,true><<<grid,128,0,stream>>>(qv,kv,vv,scale,output); break;
        case 7: vision_kernel<64,64,32,true><<<grid,64,0,stream>>>(qv,kv,vv,scale,output); break;
        default: throw std::runtime_error("Unknown tile variant");
    }
    check(cudaGetLastError()); check(cudaStreamSynchronize(stream));
}
struct test_case { int queries, keys, heads, batches; bool padded, interleaved; int data; const char * name; };
struct oracle_row { int query, head, batch; std::vector<double> values; };
std::vector<oracle_row> oracle(const input & q, const input & k, const input & v, float scale, bool all_queries) {
    std::set<int> queries, heads;
    if (all_queries) { for (int i = 0; i < q.rows; ++i) queries.insert(i); for (int i = 0; i < q.heads; ++i) heads.insert(i); }
    else {
        for (int i : {0,1,63,64,127,128,q.rows/2,q.rows-2,q.rows-1}) if (i >= 0 && i < q.rows) queries.insert(i);
        for (int i : {0,1,q.heads/2,q.heads-1}) if (i < q.heads) heads.insert(i);
    }
    std::vector<oracle_row> result;
    std::vector<double> scores(k.rows);
    for (int b = 0; b < q.batches; ++b) for (int h : heads) for (int t : queries) {
        double maximum = -INFINITY, denominator = 0;
        for (int s = 0; s < k.rows; ++s) {
            double dot = 0;
            for (int x = 0; x < 64; ++x) dot += double(q.at(x,t,h,b)) * k.at(x,s,h,b);
            scores[s] = dot * scale; maximum = std::max(maximum,scores[s]);
        }
        for (double & p : scores) { p = std::exp(p - maximum); denominator += p; }
        oracle_row row{t,h,b,std::vector<double>(64)};
        for (int x = 0; x < 64; ++x) {
            for (int s = 0; s < k.rows; ++s) row.values[x] += scores[s] * v.at(x,s,h,b);
            row.values[x] /= denominator;
        }
        result.push_back(std::move(row));
    }
    return result;
}
constexpr float canary = -1234567.25f;
struct quality { bool passed = true; double max_double = 0, max_double_ratio = 0, max_baseline = 0; size_t double_positions = 0; };
quality verify(const input & q, const std::vector<float> & output, const std::vector<oracle_row> & expected,
    const std::vector<float> * baseline, int data) {
    quality result;
    for (int i = 0; i < 16; ++i) result.passed &= output[i] == canary && output[output.size()-1-i] == canary;
    for (size_t i = 16; i < output.size()-16; ++i) {
        result.passed &= std::isfinite(output[i]) && output[i] != canary;
        if (baseline) {
            const double error = std::abs(double(output[i])-(*baseline)[i]);
            result.max_baseline = std::max(result.max_baseline,error);
            result.passed &= error <= 8e-5 + 2e-5 * std::abs((*baseline)[i]);
        }
        if (data == 2) result.passed &= output[i] == 80000.125f;
    }
    for (const auto & row : expected) for (int x = 0; x < 64; ++x) {
        const size_t index = 16 + x + size_t(64) * (row.head + q.heads * (row.query + q.rows * row.batch));
        const double error = std::abs(double(output[index])-row.values[x]);
        const double bound = data == 3 ? 2e-6 : 8e-5 + 2e-5 * std::abs(row.values[x]);
        result.max_double = std::max(result.max_double,error);
        result.max_double_ratio = std::max(result.max_double_ratio,error/bound);
        result.passed &= std::isfinite(error) && error <= bound;
        ++result.double_positions;
    }
    return result;
}
void write_quality(std::ostream & out, const quality & q) {
    out << ",\"passed\":" << (q.passed ? "true" : "false") << ",\"double_positions\":" << q.double_positions
        << ",\"max_double_error\":" << q.max_double << ",\"max_error_over_bound\":" << q.max_double_ratio
        << ",\"max_full_baseline_error\":" << q.max_baseline;
}
} // namespace

int main(int argc, char ** argv) {
    if (argc != 3 || std::strcmp(argv[1],"--output")) { std::fprintf(stderr,"Require --output PATH\n"); return 2; }
    const std::string output_path = argv[2]; std::ostringstream out; out << std::setprecision(17);
    try {
        check(cudaSetDevice(0)); cudaDeviceProp device{}; check(cudaGetDeviceProperties(&device,0));
        if (device.sharedMemPerBlock < 49152) throw std::runtime_error("Require at least48KiB per-block shared memory for all variants");
        stream_owner stream;
        out << "{\"device\":" << quote(device.name) << ",\"visible_device\":0,\"SM_major\":" << device.major
            << ",\"SM_minor\":" << device.minor << ",\"shared_bytes_per_SM\":" << device.sharedMemPerMultiprocessor << ',';
        resource_attributes(out);
        out << "\"records\":[";
        bool comma = false, passed = true;
        const test_case cases[] = {{1,1,1,1,false,false,0,"single"},{37,37,3,2,true,false,0,"batched-padded"},
            {65,129,2,1,true,true,0,"strided-key-tail"},{129,521,4,1,false,false,0,"query-key-tail"},
            {3,8193,2,1,true,false,0,"long-key-tail"},{65,257,2,1,false,false,1,"large-values"},
            {129,17,2,2,true,true,2,"exact-80000"},{129,2,2,2,true,true,3,"source-bits"},
            {2304,2304,12,1,false,false,0,"actual-2304"},{2340,2340,12,1,false,false,0,"actual-2340"},
            {1089,1089,12,1,false,false,0,"actual-1089"}};
        for (const auto & c : cases) {
            const bool timed = c.queries >= 1000;
            input q(c.queries,c.heads,c.batches,c.padded,c.interleaved), k(c.keys,c.heads,c.batches,c.padded,c.interleaved), v(c.keys,c.heads,c.batches,c.padded,c.interleaved);
            std::mt19937 random(46581+c.keys+c.queries); std::uniform_real_distribution<float> uniform(-1.5f,1.5f);
            for (auto & value : q.values) value = uniform(random);
            for (auto & value : k.values) value = uniform(random);
            for (auto & value : v.values) value = c.data == 1 ? 80000.125f + uniform(random) : uniform(random);
            if (c.data == 2 || c.data == 3) {
                std::fill(q.values.begin(),q.values.end(),0); std::fill(k.values.begin(),k.values.end(),0);
                std::fill(v.values.begin(),v.values.end(),80000.125f);
                if (c.data == 3) for (int b=0;b<c.batches;++b) for(int h=0;h<c.heads;++h) {
                    for(int t=0;t<c.queries;++t) q.values[q.index(0,t,h,b)]=1.f+0x1p-12f;
                    k.values[k.index(0,0,h,b)]=1; k.values[k.index(0,1,h,b)]=-1;
                    for(int x=0;x<64;++x) {v.values[v.index(x,0,h,b)]=1;v.values[v.index(x,1,h,b)]=-1;}
                }
            }
            const float scale = timed || c.data == 3 ? 1.f : 1.f/std::sqrt(64.f);
            q.upload(stream.value); k.upload(stream.value); v.upload(stream.value);
            check(cudaStreamSynchronize(stream.value));
            const auto expected = oracle(q,k,v,scale,!timed);
            const size_t elements=size_t(64)*c.queries*c.heads*c.batches;
            device_buffer output((elements+32)*sizeof(float)); std::vector<float> values(elements+32,canary),baseline;
            auto reset=[&]{std::fill(values.begin(),values.end(),canary);
                check(cudaMemcpyAsync(output.data,values.data(),values.size()*sizeof(float),cudaMemcpyHostToDevice,stream.value));
                check(cudaStreamSynchronize(stream.value));};
            auto compute=[&](int variant){launch(variant,q,k,v,scale,static_cast<float*>(output.data)+16,stream.value);};
            auto quality_check=[&]{check(cudaMemcpy(values.data(),output.data,values.size()*sizeof(float),cudaMemcpyDeviceToHost));
                auto gate=verify(q,values,expected,baseline.empty()?nullptr:&baseline,c.data);gate.passed &= q.unchanged()&&k.unchanged()&&v.unchanged();return gate;};
            reset();compute(0);auto base_quality=quality_check();baseline=values;passed &= base_quality.passed;
            std::fprintf(stderr,"vision tile case%s queries%d keys%d timed%d\n",c.name,c.queries,c.keys,timed);
            for(int candidate=timed?1:0;candidate<variant_count;++candidate) for(int phase=0;phase<(timed?4:1);++phase) {
                const int executed=timed&&(phase==0||phase==3)?0:candidate;
                reset(); for(int i=0;i<(timed?3:1);++i) compute(executed);
                auto gate=quality_check(); std::vector<double> samples;
                if(timed) {
                    const auto started=std::chrono::steady_clock::now();
                    do {const auto before=std::chrono::steady_clock::now();compute(executed);
                        samples.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count());
                    }while(samples.size()<20||std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<.25);
                    const auto after=quality_check();gate.passed &= after.passed;
                    gate.max_double=std::max(gate.max_double,after.max_double);gate.max_double_ratio=std::max(gate.max_double_ratio,after.max_double_ratio);gate.max_baseline=std::max(gate.max_baseline,after.max_baseline);
                }
                passed &= gate.passed;if(comma)out<<',';comma=true;
                out << "{\"case\":" << quote(c.name) << ",\"queries\":" << c.queries << ",\"keys\":" << c.keys
                    << ",\"heads\":" << c.heads << ",\"batches\":" << c.batches << ",\"candidate\":" << quote(variants[candidate].name)
                    << ",\"executed\":" << quote(variants[executed].name) << ",\"bounded_exp\":" << (variants[executed].bounded_exp?"true":"false") << ",\"phase\":" << phase << ",\"timed\":" << (timed?"true":"false");
                write_quality(out,gate);
                if(timed) {auto sorted=samples;std::sort(sorted.begin(),sorted.end());
                    out<<",\"median_ms\":"<<(sorted[(sorted.size()-1)/2]+sorted[sorted.size()/2])*.5<<",\"samples_ms\":[";
                    for(size_t i=0;i<samples.size();++i){if(i)out<<',';out<<samples[i];}out<<']';}
                out << '}';
            }
        }
        out << "],\"completed\":true,\"passed\":" << (passed?"true":"false")
            << ",\"warmup_per_timed_phase\":3,\"minimum_repeats\":20,\"minimum_seconds\":0.25,\"model_qualified\":false,\"production_promoted\":false"
            << ",\"scope\":\"F32 copied owned head64 kernel tile and separately labeled bounded-exp screen; complete eight operator fixtures plus three actual projection-grid shapes with synthetic inputs.\""
            << ",\"limitations\":[\"Actual-size cases use sampled complete Double query/head rows and complete baseline comparisons; not a full vision model oracle.\",\"Timings include kernel submission and private-stream completion; exclude allocation, input transfer, quality checks and full tower/HTTP.\",\"Native WIDTH72 specialization is unchanged; no production dispatch changed.\",\"Bounded-exp candidates use __expf only in [-80,0], with ordinary expf elsewhere; no global fast-math mode.\"]}\n";
        std::ofstream file(output_path);if(!file||!(file<<out.str()))throw std::runtime_error("Cannot write output report");return passed?0:1;
    } catch(const std::exception & e) {std::ofstream(output_path)<<"{\"completed\":false,\"passed\":false,\"error\":"<<quote(e.what())<<"}\n";std::fprintf(stderr,"%s\n",e.what());return 1;}
}
