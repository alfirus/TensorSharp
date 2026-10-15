// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Standalone diagnostic: unchanged upstream ggml plus TensorSharp's existing
// owned F32 Metal projection helper. Not part of model/runtime dispatch.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include "ggml-backend-impl.h"
#include "ggml-metal/ggml-metal-device.h"
#include "ggml_gemma4_audio_metal_precision.h"
#include <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Experimental only. A float matrix declaration is not itself evidence that
// the device preserves all F32 operand bits: measure it against independent
// CPU SGEMM and sampled double dots before considering runtime integration.
constexpr const char* simd_source = R"METAL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
struct P { uint rows, inner, columns, weightType; };
kernel void projection_simd(device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]], device float* output [[buffer(2)]],
    constant P& p [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float xs[32*32];
    threadgroup float ws[32*64];
    simdgroup_float8x8 c[8];
    for (uint i=0; i<8; ++i) c[i] = make_filled_simdgroup_matrix<float,8>(0);
    const uint token = (sg / 2) * 16, channel = (sg % 2) * 32;
    for (uint start = 0; start < p.inner; start += 32) {
        for (uint i = tid; i < 1024; i += 128) {
            const uint r = group.y * 32 + i / 32, k = start + i % 32;
            xs[i] = r < p.rows && k < p.inner ? input[ulong(r)*p.inner+k] : 0;
        }
        for (uint i = tid; i < 2048; i += 128) {
            const uint oc = group.x * 64 + i % 64, wk = start + i / 64;
            float value = 0;
            if (oc < p.columns && wk < p.inner) {
                if (p.weightType == 8) {
                    device const uchar* block = weights + (ulong(oc)*(p.inner/32)+wk/32)*34;
                    value = float(char(block[2+wk%32])) * float(*reinterpret_cast<device const half*>(block));
                } else value = reinterpret_cast<device const float*>(weights)[ulong(oc)*p.inner+wk];
            }
            ws[i] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 32; k += 8) {
            simdgroup_float8x8 a0,a1,b[4];
            simdgroup_load(a0,xs+token*32+k,32,0,false);
            simdgroup_load(a1,xs+(token+8)*32+k,32,0,false);
            for (uint j=0; j<4; ++j) {
                simdgroup_load(b[j],ws+k*64+channel+j*8,64,0,false);
                simdgroup_multiply_accumulate(c[j],a0,b[j],c[j]);
                simdgroup_multiply_accumulate(c[j+4],a1,b[j],c[j+4]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint j=0; j<4; ++j) {
        simdgroup_store(c[j],ws+token*64+channel+j*8,64,0,false);
        simdgroup_store(c[j+4],ws+(token+8)*64+channel+j*8,64,0,false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 2048; i += 128) {
        const uint r = group.y*32+i/64, oc = group.x*64+i%64;
        if (r < p.rows && oc < p.columns) output[ulong(r)*p.columns+oc] = ws[i];
    }
}
)METAL";

void simd_compute(ggml_backend_t backend, ggml_tensor* w, ggml_tensor* x, ggml_tensor* y) {
    @autoreleasepool {
    auto* device = static_cast<ggml_metal_device_t>(ggml_backend_get_device(backend)->context);
    id<MTLDevice> metal = (__bridge id<MTLDevice>)ggml_metal_device_get_obj(device);
    static id<MTLComputePipelineState> pipeline = nil;
    if (!pipeline) {
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init]; options.fastMathEnabled = NO;
        NSError* error = nil;
        id<MTLLibrary> library = [metal newLibraryWithSource:[NSString stringWithUTF8String:simd_source] options:options error:&error];
        if (!library) throw std::runtime_error([[error localizedDescription] UTF8String]);
        pipeline = [metal newComputePipelineStateWithFunction:[library newFunctionWithName:@"projection_simd"] error:&error];
        if (!pipeline) throw std::runtime_error([[error localizedDescription] UTF8String]);
    }
    auto binding = [](ggml_tensor* tensor) {
        return ggml_metal_buffer_get_id(static_cast<ggml_metal_buffer_t>(tensor->buffer->context),tensor);
    };
    auto wb = binding(w), xb = binding(x), yb = binding(y);
    struct parameters { uint32_t rows, inner, columns, weightType; } p{uint32_t(x->ne[1]),uint32_t(x->ne[0]),uint32_t(w->ne[1]),uint32_t(w->type)};
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ggml_metal_device_get_queue(device);
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)wb.metal offset:wb.offs atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)xb.metal offset:xb.offs atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)yb.metal offset:yb.offs atIndex:2];
    [encoder setBytes:&p length:sizeof(p) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake((p.columns+63)/64,(p.rows+31)/32,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) throw std::runtime_error("SIMD projection did not complete.");
    }
}

std::string sha256(const void* bytes, size_t length) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(bytes, CC_LONG(length), digest);
    char result[2*CC_SHA256_DIGEST_LENGTH+1];
    for (size_t i=0; i<CC_SHA256_DIGEST_LENGTH; ++i) std::snprintf(result+2*i,3,"%02x",digest[i]);
    return result;
}

template<class T> std::vector<T> read(const char* path, size_t elements) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() != std::streamoff(elements * sizeof(T)))
        throw std::runtime_error("Weight file length differs from the required layout.");
    stream.seekg(0); std::vector<T> result(elements);
    if (!stream.read(reinterpret_cast<char*>(result.data()), result.size() * sizeof(T)))
        throw std::runtime_error("Cannot read weight file.");
    return result;
}

struct result {
    std::string mode;
    int rows, inner, columns;
    std::vector<double> samples;
    double error, double_sample_error;
    std::string input_hash, output_hash, reference_hash;
};

result measure(ggml_backend_t backend, const void* weights, ggml_type type,
    const std::vector<float>& input, const std::vector<float>& expanded,
    const std::vector<float>& reference, int rows, int inner, int columns, int mode) {
    const bool owned = mode < 2;
    ggml_init_params parameters{4 * 1024 * 1024, nullptr, true};
    auto* ctx = ggml_init(parameters);
    if (!ctx) throw std::runtime_error("Context allocation failed.");
    auto* w = ggml_new_tensor_2d(ctx, type, inner, columns);
    auto* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, inner, rows);
    auto* y = ggml_mul_mat(ctx, w, x);
    ggml_prec_set_acc(y, GGML_PREC_F32);
    ggml_set_name(y, owned ? "tsg_gemma4_vision_f32_projection" : "upstream_q8_projection");
    ggml_set_output(y);
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    auto* buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { ggml_free(ctx); throw std::runtime_error("Metal allocation failed."); }
    ggml_backend_tensor_set(w, weights, 0, ggml_nbytes(w));
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    result value{mode == 3 ? "f32-simd-float8x8" : mode == 4 ? "q8-simd-float8x8" :
        owned ? type == GGML_TYPE_F32 ? "f32-mps" : "q8-owned-f32" : "q8-upstream",
        rows, inner, columns, {}, 0, 0};
    auto execute = [&] {
        ggml_status status = GGML_STATUS_SUCCESS;
        if (mode >= 3) simd_compute(backend,w,x,y);
        else if (owned) {
            if (!tsg_gemma4_audio_metal_graph_compute(backend, graph, &status, true))
                throw std::runtime_error("Required owned projection did not execute.");
        } else status = ggml_backend_graph_compute(backend, graph);
        if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("Projection compute failed.");
        ggml_backend_synchronize(backend);
    };
    for (int i = 0; i < 3; ++i) execute();
    using clock = std::chrono::steady_clock;
    auto started = clock::now();
    do {
        auto before = clock::now(); execute();
        value.samples.push_back(std::chrono::duration<double, std::milli>(clock::now() - before).count());
    } while (value.samples.size() < 20 || std::chrono::duration<double>(clock::now() - started).count() < .2);
    std::vector<float> actual(size_t(rows) * columns);
    ggml_backend_tensor_get(y, actual.data(), 0, actual.size() * sizeof(float));
    value.input_hash = sha256(input.data(),input.size()*sizeof(float));
    value.output_hash = sha256(actual.data(),actual.size()*sizeof(float));
    value.reference_hash = sha256(reference.data(),reference.size()*sizeof(float));
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) throw std::runtime_error("Nonfinite projection result.");
        value.error = std::max(value.error, std::abs(double(actual[i]) - reference[i]));
    }
    // Independent double dot products, covering boundaries and interior cells.
    // Complete output comparison above uses separate CPU F32 SGEMM, not MPS.
    for (int sample = 0; sample < 64; ++sample) {
        const int row = sample == 0 ? 0 : sample == 63 ? rows - 1 : (sample * 163) % rows;
        const int column = sample == 0 ? 0 : sample == 63 ? columns - 1 : (sample * 211) % columns;
        double expected = 0;
        for (int k = 0; k < inner; ++k)
            expected += double(input[size_t(row) * inner + k]) * expanded[size_t(column) * inner + k];
        value.double_sample_error = std::max(value.double_sample_error,
            std::abs(double(actual[size_t(row) * columns + column]) - expected));
    }
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
    return value;
}

int main(int argc, char** argv) {
    try {
        if (argc != 6 && !(argc == 7 && std::string(argv[6]) == "--precise-screen"))
            throw std::runtime_error("Pass Q8 raw file, expanded F32 file, input channels, output channels, report path and optional --precise-screen.");
        const int inner = std::stoi(argv[3]), columns = std::stoi(argv[4]);
        if (inner <= 0 || inner > 8192 || inner % 32 || columns <= 0 || columns > 8192)
            throw std::runtime_error("Invalid bounded matrix geometry.");
        const auto compressed = read<unsigned char>(argv[1], size_t(inner / 32) * columns * 34);
        const auto expanded = read<float>(argv[2], size_t(inner) * columns);
        // Ensure both supplied weight formats describe exactly the same values.
        for (size_t i = 0; i < expanded.size(); ++i) {
            const auto* block = compressed.data() + i / 32 * 34;
            ggml_fp16_t scale; std::memcpy(&scale, block, sizeof(scale));
            float exact = float(static_cast<signed char>(block[2 + i % 32])) * ggml_fp16_to_fp32(scale);
            if (exact != expanded[i]) throw std::runtime_error("Q8/F32 weight files differ.");
        }
        auto* backend = ggml_backend_metal_init();
        if (!backend) throw std::runtime_error("Metal unavailable.");
        std::vector<result> results;
        for (int rows : {36, 108, 1024, 1664, 2304, 2340}) {
            std::vector<float> input(size_t(rows) * inner), reference(size_t(rows) * columns);
            for (size_t i = 0; i < input.size(); ++i)
                input[i] = std::sin(float(i % 65521) * .0013f) + .2f * std::cos(float(i % 8191) * .017f);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, columns, inner,
                1, input.data(), inner, expanded.data(), inner, 0, reference.data(), columns);
            results.push_back(measure(backend, expanded.data(), GGML_TYPE_F32, input, expanded, reference, rows, inner, columns, 0));
            if (argc == 6) {
                results.push_back(measure(backend, compressed.data(), GGML_TYPE_Q8_0, input, expanded, reference, rows, inner, columns, 1));
                results.push_back(measure(backend, compressed.data(), GGML_TYPE_Q8_0, input, expanded, reference, rows, inner, columns, 2));
            }
            results.push_back(measure(backend, expanded.data(), GGML_TYPE_F32, input, expanded, reference, rows, inner, columns, 3));
            results.push_back(measure(backend, compressed.data(), GGML_TYPE_Q8_0, input, expanded, reference, rows, inner, columns, 4));
            std::fprintf(stderr, "Completed N=%d K=%d M=%d\n", rows, inner, columns);
        }
        ggml_backend_free(backend);
        std::ofstream output(argv[5]); output.precision(17);
        output << "{\"completed\":true,\"warmup\":3,\"minimum_repeats\":20,\"minimum_seconds\":0.2,\"results\":[";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i]; auto sorted = r.samples; std::sort(sorted.begin(), sorted.end());
            if (i) output << ',';
            output << "{\"mode\":\"" << r.mode << "\",\"rows\":" << r.rows << ",\"inner\":" << r.inner
                << ",\"columns\":" << r.columns << ",\"median_ms\":" << (sorted[(sorted.size()-1)/2]+sorted[sorted.size()/2])/2
                << ",\"max_absolute_error_cpu_sgemm\":" << r.error << ",\"max_absolute_error_64_double_samples\":" << r.double_sample_error
                << ",\"input_sha256\":\"" << r.input_hash << "\",\"output_sha256\":\"" << r.output_hash
                << "\",\"cpu_reference_sha256\":\"" << r.reference_hash << "\""
                << ",\"samples_ms\":[";
            for (size_t j = 0; j < r.samples.size(); ++j) { if (j) output << ','; output << r.samples[j]; }
            output << "]}";
        }
        output << "],\"limitations\":\"Projection microbenchmark only; original quantized weights and synthetic F32 activations. Upload/model load/output readback excluded; completed GPU compute included. No complete vision, pooled embedding or HTTP parity claim.\"}\n";
        if (!output) throw std::runtime_error("Cannot write report.");
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
