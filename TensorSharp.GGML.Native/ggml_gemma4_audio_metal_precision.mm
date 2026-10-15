// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Uses the unchanged ggml Metal device/encoder helpers at the build revision.
// Kernel math and graph submission here are entirely TensorSharp-owned.
#include "ggml_gemma4_audio_metal_precision.h"
#include "ggml-backend-impl.h"
#include "ggml-metal.h"
#include "ggml-metal/ggml-metal-device.h"
#include "ggml-metal/ggml-metal-ops.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>

namespace {
constexpr const char * projection_name = "tsg_gemma4_audio_f32_projection";
constexpr const char * vision_projection_name = "tsg_gemma4_vision_f32_projection";
constexpr const char * shared_projection_name = "tsg_gemma4_f32_projection";
constexpr const char * attention_name = "tsg_gemma4_audio_f32_attention";
constexpr const char * depthwise_name = "tsg_gemma4_audio_f32_depthwise";
constexpr const char * vision_attention_name = "tsg_gemma4_vision_f32_attention";

// 8 output columns x 16 input rows share each 32-channel weight/input tile.
// Every product and accumulator is float, including Q8_0 dequantization.
constexpr const char * source = R"METAL(
#include <metal_stdlib>
using namespace metal;
struct Parameters {
    uint columns, rows, inner, weightType;
    ulong weightStride, inputStride, outputStride;
};
kernel void audio_projection_f32(
    device const uchar * weights [[buffer(0)]],
    device const uchar * inputs [[buffer(1)]],
    device uchar * outputs [[buffer(2)]],
    constant Parameters& p [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float wtile[8 * 32];
    threadgroup float xtile[16 * 32];
    const uint columnInTile = tid % 8, rowInTile = tid / 8;
    const uint column = group.x * 8 + columnInTile;
    const uint row = group.y * 16 + rowInTile;
    float sum = 0.0f;
    for (uint start = 0; start < p.inner; start += 32) {
        for (uint index = tid; index < 8 * 32; index += 128) {
            const uint oc = group.x * 8 + index / 32;
            const uint channel = start + index % 32;
            float value = 0.0f;
            if (oc < p.columns && channel < p.inner) {
                device const uchar * weight = weights + ulong(oc) * p.weightStride;
                if (p.weightType == 8) {
                    device const uchar * block = weight + ulong(channel / 32) * 34;
                    const float scale = float(*reinterpret_cast<device const half *>(block));
                    value = float(char(block[2 + channel % 32])) * scale;
                } else {
                    value = reinterpret_cast<device const float *>(weight)[channel];
                }
            }
            wtile[index] = value;
        }
        for (uint index = tid; index < 16 * 32; index += 128) {
            const uint inputRow = group.y * 16 + index / 32;
            const uint channel = start + index % 32;
            xtile[index] = inputRow < p.rows && channel < p.inner
                ? reinterpret_cast<device const float *>(inputs + ulong(inputRow) * p.inputStride)[channel] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint channel = 0; channel < 32; ++channel)
            sum = fma(wtile[columnInTile * 32 + channel], xtile[rowInTile * 32 + channel], sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (column < p.columns && row < p.rows)
        reinterpret_cast<device float *>(outputs + ulong(row) * p.outputStride)[column] = sum;
}
struct AttentionParameters {
    uint sequence, heads, head, past, future;
    float cap;
};
struct VisionAttentionParameters { uint sequence, heads, head; };
kernel void vision_attention_f32(
    device const float * q [[buffer(0)]], device const float * k [[buffer(1)]],
    device const float * v [[buffer(2)]], device float * output [[buffer(3)]],
    constant VisionAttentionParameters& p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scores[128];
    const uint time = group.x, h = group.y;
    for (uint key = 0; key < p.sequence; ++key) {
        float dot = 0.0f;
        for (uint d = lane; d < p.head; d += 32)
            dot = fma(q[(ulong(h) * p.sequence + time) * p.head + d],
                k[(ulong(h) * p.sequence + key) * p.head + d], dot);
        dot = simd_sum(dot);
        if (lane == 0) scores[key] = dot;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float maximum = -INFINITY, denominator = 0.0f;
        for (uint key = 0; key < p.sequence; ++key) maximum = max(maximum, scores[key]);
        for (uint key = 0; key < p.sequence; ++key) {
            scores[key] = exp(scores[key] - maximum); denominator += scores[key];
        }
        for (uint key = 0; key < p.sequence; ++key) scores[key] /= denominator;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = lane; d < p.head; d += 32) {
        float sum = 0.0f;
        for (uint key = 0; key < p.sequence; ++key)
            sum = fma(scores[key], v[(ulong(h) * p.sequence + key) * p.head + d], sum);
        output[(ulong(time) * p.heads + h) * p.head + d] = sum;
    }
}
kernel void audio_attention_f32(
    device const float * q [[buffer(0)]], device const float * k [[buffer(1)]],
    device const float * v [[buffer(2)]], device const float * relative [[buffer(3)]],
    device float * output [[buffer(4)]], constant AttentionParameters& p [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    // One SIMD group owns one query/head. No score or window materialization.
    threadgroup float probabilities[1024];
    const uint time = group.x, h = group.y, hidden = p.heads * p.head;
    const uint first = time > p.past ? time - p.past : 0;
    const uint last = min(p.sequence - 1, time + p.future);
    const uint count = last - first + 1;
    for (uint index = 0; index < count; ++index) {
        const uint keyTime = first + index, position = keyTime + p.past - time;
        float content = 0.0f, positional = 0.0f;
        for (uint d = lane; d < p.head; d += 32) {
            const float query = q[ulong(time) * hidden + h * p.head + d];
            content = fma(query, k[ulong(keyTime) * hidden + h * p.head + d], content);
            positional = fma(query, relative[ulong(position) * hidden + h * p.head + d], positional);
        }
        content = simd_sum(content); positional = simd_sum(positional);
        if (lane == 0) probabilities[index] = p.cap * tanh((content + positional) / p.cap);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float maximum = -INFINITY;
        for (uint index = 0; index < count; ++index) maximum = max(maximum, probabilities[index]);
        float denominator = 0.0f;
        for (uint index = 0; index < count; ++index) {
            probabilities[index] = exp(probabilities[index] - maximum);
            denominator += probabilities[index];
        }
        for (uint index = 0; index < count; ++index) probabilities[index] /= denominator;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = lane; d < p.head; d += 32) {
        float sum = 0.0f;
        for (uint index = 0; index < count; ++index)
            sum = fma(probabilities[index], v[ulong(first + index) * hidden + h * p.head + d], sum);
        output[ulong(time) * hidden + h * p.head + d] = sum;
    }
}
struct DepthwiseParameters { uint sequence, channels, kernelWidth; };
kernel void audio_depthwise_f32(
    device const float * input [[buffer(0)]], device const float * weight [[buffer(1)]],
    device float * output [[buffer(2)]], constant DepthwiseParameters& p [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    const ulong total = ulong(p.sequence) * p.channels;
    if (index >= total) return;
    const uint time = index / p.channels, channel = index % p.channels;
    float sum = 0.0f;
    for (uint tap = 0; tap < p.kernelWidth; ++tap) {
        const long inputTime = long(time) + tap - (p.kernelWidth - 1);
        if (inputTime >= 0)
            sum = fma(input[ulong(inputTime) * p.channels + channel],
                weight[ulong(channel) * p.kernelWidth + tap], sum);
    }
    output[index] = sum;
}
)METAL";

struct parameters {
    uint32_t columns, rows, inner, weight_type;
    uint64_t weight_stride, input_stride, output_stride;
};
static_assert(sizeof(parameters) == 40);
struct attention_parameters { uint32_t sequence, heads, head, past, future; float cap; };
struct vision_attention_parameters { uint32_t sequence, heads, head; };
struct depthwise_parameters { uint32_t sequence, channels, kernel; };
static_assert(sizeof(attention_parameters) == 24);
static_assert(sizeof(depthwise_parameters) == 12);
static_assert(sizeof(vision_attention_parameters) == 12);

struct state {
    std::mutex mutex;
    id<MTLDevice> device = nil;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLComputePipelineState> vision_attention_pipeline = nil;
    id<MTLComputePipelineState> attention_pipeline = nil;
    id<MTLComputePipelineState> depthwise_pipeline = nil;
    // Shape-only objects never retain model weights or graph buffers.
    using matrix_key = std::tuple<uint32_t, uint32_t, uint32_t, uint64_t, uint64_t, uint64_t>;
    std::map<matrix_key, MPSMatrixMultiplication *> matrix_kernels;

};
state& kernel_state() { static state value; return value; }

bool tagged(const ggml_tensor * node) {
    return (node->op == GGML_OP_MUL_MAT && (std::strcmp(node->name, projection_name) == 0 ||
            std::strcmp(node->name, vision_projection_name) == 0 || std::strcmp(node->name, shared_projection_name) == 0)) ||
        (node->op == GGML_OP_DUP && (std::strcmp(node->name, attention_name) == 0 ||
            std::strcmp(node->name, depthwise_name) == 0 || std::strcmp(node->name, vision_attention_name) == 0));
}

ggml_backend_buffer_t buffer(const ggml_tensor * tensor) {
    return tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
}

ggml_metal_buffer_id buffer_id(ggml_backend_dev_t device, const ggml_tensor * tensor) {
    auto * bound = buffer(tensor);
    if (!bound || !ggml_backend_dev_supports_buft(device, ggml_backend_buffer_get_type(bound)))
        throw std::runtime_error("Audio F32 Metal projection requires Metal-resident tensors.");
    return ggml_metal_buffer_get_id(static_cast<ggml_metal_buffer_t>(bound->context), tensor);
}

bool valid_projection(const ggml_tensor * output) {
    const auto * w = output->src[0]; const auto * x = output->src[1];
    if (!w || !x || x->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
        (w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_Q8_0) ||
        !ggml_is_contiguous(w) || !ggml_is_contiguous(x) || !ggml_is_contiguous(output) ||
        w->ne[2] != 1 || w->ne[3] != 1 || x->ne[2] != 1 || x->ne[3] != 1 ||
        output->ne[2] != 1 || output->ne[3] != 1 || w->ne[0] != x->ne[0] ||
        output->ne[0] != w->ne[1] || output->ne[1] != x->ne[1] ||
        w->ne[0] <= 0 || w->ne[1] <= 0 || x->ne[1] <= 0 ||
        w->ne[0] > INT32_MAX || w->ne[1] > INT32_MAX || x->ne[1] > INT32_MAX ||
        (w->type == GGML_TYPE_Q8_0 && w->ne[0] % 32 != 0)) return false;
    return true;
}

bool valid_owned_elementwise(const ggml_tensor * node) {
    if (node->type != GGML_TYPE_F32 || !ggml_is_contiguous(node)) return false;
    const bool attention = std::strcmp(node->name, attention_name) == 0;
    const bool vision = std::strcmp(node->name, vision_attention_name) == 0;
    const int sources = vision ? 3 : attention ? 4 : 2;
    for (int index = 0; index < sources; ++index)
        if (!node->src[index] || node->src[index]->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(node->src[index])) return false;
    return true;
}

void ensure_pipeline(state& cache, id<MTLDevice> device) {
    if (cache.pipeline != nil && cache.device == device) return;
    MTLCompileOptions * options = [MTLCompileOptions new];
    options.fastMathEnabled = NO;
    NSError * error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source]
        options:options error:&error];
    if (!library) throw std::runtime_error("Cannot compile audio F32 Metal projection: " +
        std::string(error.localizedDescription.UTF8String ?: "unknown Metal error"));
    id<MTLFunction> function = [library newFunctionWithName:@"audio_projection_f32"];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 128)
        throw std::runtime_error("Cannot create audio F32 Metal projection pipeline.");
    id<MTLComputePipelineState> attention = [device newComputePipelineStateWithFunction:
        [library newFunctionWithName:@"audio_attention_f32"] error:&error];
    id<MTLComputePipelineState> depthwise = [device newComputePipelineStateWithFunction:
        [library newFunctionWithName:@"audio_depthwise_f32"] error:&error];
    if (!attention || attention.threadExecutionWidth != 32 || attention.maxTotalThreadsPerThreadgroup < 32 ||
        !depthwise || depthwise.maxTotalThreadsPerThreadgroup < 256)
        throw std::runtime_error("Cannot create audio F32 Metal attention/depthwise pipelines.");
    id<MTLComputePipelineState> vision_attention = [device newComputePipelineStateWithFunction:
        [library newFunctionWithName:@"vision_attention_f32"] error:&error];
    if (!vision_attention || vision_attention.threadExecutionWidth != 32 || vision_attention.maxTotalThreadsPerThreadgroup < 32)
        throw std::runtime_error("Cannot create the F32 small vision attention pipeline.");
    cache.matrix_kernels.clear(); cache.device = device; cache.pipeline = pipeline;
    cache.attention_pipeline = attention; cache.depthwise_pipeline = depthwise;
    cache.vision_attention_pipeline = vision_attention;

}

void encode_owned_elementwise(ggml_backend_dev_t device, id<MTLCommandBuffer> command,
    state& cache, ggml_tensor * output) {
    const bool attention = std::strcmp(output->name, attention_name) == 0;
    const bool vision = std::strcmp(output->name, vision_attention_name) == 0;
    const int sources = vision ? 3 : attention ? 4 : 2;
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:vision ? cache.vision_attention_pipeline : attention ? cache.attention_pipeline : cache.depthwise_pipeline];
    for (int index = 0; index < sources; ++index) {
        const auto input = buffer_id(device, output->src[index]);
        [encoder setBuffer:(__bridge id<MTLBuffer>)input.metal offset:input.offs atIndex:index];
    }
    const auto result = buffer_id(device, output);
    [encoder setBuffer:(__bridge id<MTLBuffer>)result.metal offset:result.offs atIndex:sources];
    if (vision) {
        vision_attention_parameters args{}; std::memcpy(&args, output->op_params, sizeof(args));
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(args.sequence, args.heads, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    } else if (attention) {
        attention_parameters args{}; std::memcpy(&args, output->op_params, sizeof(args));
        [encoder setBytes:&args length:sizeof(args) atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake(args.sequence, args.heads, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    } else {
        depthwise_parameters args{}; std::memcpy(&args, output->op_params, sizeof(args));
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
        const size_t total = static_cast<size_t>(args.sequence) * args.channels;
        [encoder dispatchThreadgroups:MTLSizeMake((total + 255) / 256, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers]; [encoder endEncoding];
}

void encode_mps_projection(state& cache, ggml_backend_dev_t device, id<MTLCommandBuffer> command,
    ggml_tensor * output) {
    const auto * w = output->src[0]; const auto * x = output->src[1];
    if (!MPSSupportsMTLDevice(cache.device))
        throw std::runtime_error("This Metal device cannot execute the requested MPS F32 audio projection.");
    const auto weight = buffer_id(device, w), input = buffer_id(device, x), result = buffer_id(device, output);
    const auto key = state::matrix_key{ static_cast<uint32_t>(output->ne[1]), static_cast<uint32_t>(output->ne[0]),
        static_cast<uint32_t>(w->ne[0]), x->nb[1], w->nb[1], output->nb[1] };
    auto found = cache.matrix_kernels.find(key);
    if (found == cache.matrix_kernels.end()) {
        MPSMatrixMultiplication * operation = [[MPSMatrixMultiplication alloc] initWithDevice:cache.device
            transposeLeft:NO transposeRight:YES resultRows:output->ne[1] resultColumns:output->ne[0]
            interiorColumns:w->ne[0] alpha:1.0 beta:0.0];
        if (!operation) throw std::runtime_error("Cannot create the MPS F32 audio projection.");
        // Apple's default forbids reduced internal precision. Independent
        // numerical gates still check the implementation's reduction order.
        operation.options = MPSKernelOptionsNone;
        if (cache.matrix_kernels.size() >= 64) cache.matrix_kernels.clear();
        found = cache.matrix_kernels.emplace(key, operation).first;
    }
    MPSMatrixDescriptor * xd = [MPSMatrixDescriptor matrixDescriptorWithRows:x->ne[1] columns:x->ne[0]
        rowBytes:x->nb[1] dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor * wd = [MPSMatrixDescriptor matrixDescriptorWithRows:w->ne[1] columns:w->ne[0]
        rowBytes:w->nb[1] dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor * yd = [MPSMatrixDescriptor matrixDescriptorWithRows:output->ne[1] columns:output->ne[0]
        rowBytes:output->nb[1] dataType:MPSDataTypeFloat32];
    MPSMatrix * xm = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)input.metal offset:input.offs descriptor:xd];
    MPSMatrix * wm = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)weight.metal offset:weight.offs descriptor:wd];
    MPSMatrix * ym = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)result.metal offset:result.offs descriptor:yd];
    if (!xm || !wm || !ym) throw std::runtime_error("Cannot bind the MPS F32 audio projection matrices.");
    [found->second encodeToCommandBuffer:command leftMatrix:xm rightMatrix:wm resultMatrix:ym];
}

void encode_projection(ggml_backend_dev_t device, id<MTLCommandBuffer> command,
    state& cache, ggml_tensor * output, bool prefer_mps) {
    const auto * w = output->src[0]; const auto * x = output->src[1];
    const char * mps = std::getenv(std::strcmp(output->name, vision_projection_name) == 0
        ? "TS_GEMMA4_VISION_METAL_MPS_MM" : "TS_GEMMA4_AUDIO_METAL_MPS_MM");
    const bool opt_out = mps && mps[0] == '0' && mps[1] == '\0';
    if (!opt_out && (prefer_mps || (mps && mps[0] == '1' && mps[1] == '\0')) && w->type == GGML_TYPE_F32) {
        encode_mps_projection(cache, device, command, output); return;
    }
    const auto weight = buffer_id(device, w), input = buffer_id(device, x), result = buffer_id(device, output);
    parameters args{ static_cast<uint32_t>(output->ne[0]), static_cast<uint32_t>(output->ne[1]),
        static_cast<uint32_t>(w->ne[0]), static_cast<uint32_t>(w->type),
        w->nb[1], x->nb[1], output->nb[1] };
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:cache.pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)weight.metal offset:weight.offs atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)input.metal offset:input.offs atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)result.metal offset:result.offs atIndex:2];
    [encoder setBytes:&args length:sizeof(args) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake((args.columns + 7) / 8, (args.rows + 15) / 16, 1)
        threadsPerThreadgroup:MTLSizeMake(8, 16, 1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder endEncoding];
}

void encode_ordinary(ggml_metal_device_t device, id<MTLCommandBuffer> command,
    ggml_cgraph * graph, int start, int end) {
    if (start == end) return;
    ggml_metal_op_t operations = ggml_metal_op_init(device, (__bridge void *)command, graph,
        ggml_metal_device_get_fusion_info(device), start, end, false, false, 0);
    try {
        const int count = ggml_metal_op_n_nodes(operations);
        for (int index = 0; index < count;) index += ggml_metal_op_encode(operations, index);
    } catch (...) { ggml_metal_op_free(operations); throw; }
    ggml_metal_op_free(operations);
}
}

ggml_tensor * tsg_gemma4_audio_metal_attention(ggml_context * ctx,
    ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * relative,
    int sequence, int heads, int head, int past, int future, float cap) {
    if (!ctx || !q || !k || !v || !relative || sequence <= 0 || heads <= 0 || head <= 0 ||
        past < 0 || future < 0 || static_cast<int64_t>(past) + future + 1 > 1024 ||
        static_cast<int64_t>(heads) * head > INT32_MAX ||
        static_cast<int64_t>(sequence) + future > UINT32_MAX || !std::isfinite(cap) || cap <= 0)
        return nullptr;
    auto * output = ggml_dup(ctx, ggml_cont(ctx, q));
    output->src[1] = ggml_cont(ctx, k); output->src[2] = ggml_cont(ctx, v);
    output->src[3] = ggml_cont(ctx, relative);
    const attention_parameters args{static_cast<uint32_t>(sequence), static_cast<uint32_t>(heads),
        static_cast<uint32_t>(head), static_cast<uint32_t>(past), static_cast<uint32_t>(future), cap};
    std::memcpy(output->op_params, &args, sizeof(args)); ggml_set_name(output, attention_name);
    return output;
}

ggml_tensor * tsg_gemma4_audio_metal_depthwise(ggml_context * ctx,
    ggml_tensor * input, ggml_tensor * weight, int sequence, int channels, int kernel) {
    if (!ctx || !input || !weight || sequence <= 0 || channels <= 0 || kernel <= 0 ||
        static_cast<uint64_t>(sequence) * channels > UINT32_MAX) return nullptr;
    auto * output = ggml_dup(ctx, ggml_cont(ctx, input)); output->src[1] = ggml_cont(ctx, weight);
    const depthwise_parameters args{static_cast<uint32_t>(sequence), static_cast<uint32_t>(channels),
        static_cast<uint32_t>(kernel)};
    std::memcpy(output->op_params, &args, sizeof(args)); ggml_set_name(output, depthwise_name);
    return output;
}

ggml_tensor * tsg_gemma4_vision_metal_attention(ggml_context * ctx,
    ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, int sequence, int heads, int head) {
    if (!ctx || !q || !k || !v || sequence <= 0 || sequence > 128 || heads <= 0 || head != 64) return nullptr;
    for (const auto * input : {q, k, v})
        if (input->type != GGML_TYPE_F32 || input->ne[0] != head || input->ne[1] != sequence ||
            input->ne[2] != heads || input->ne[3] != 1) return nullptr;
    auto * output = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head, heads, sequence);
    output->op = GGML_OP_DUP;
    output->src[0] = ggml_cont(ctx, q); output->src[1] = ggml_cont(ctx, k); output->src[2] = ggml_cont(ctx, v);
    const vision_attention_parameters args{static_cast<uint32_t>(sequence), static_cast<uint32_t>(heads), static_cast<uint32_t>(head)};
    std::memcpy(output->op_params, &args, sizeof(args)); ggml_set_name(output, vision_attention_name);
    return output;
}

bool tsg_gemma4_audio_metal_graph_compute(ggml_backend_t backend, ggml_cgraph * graph, ggml_status * status, bool prefer_mps) {
    if (!status || !backend || !graph || !ggml_backend_is_metal(backend)) return false;
    const auto backend_device = ggml_backend_get_device(backend);
    const int count = ggml_graph_n_nodes(graph);
    int projections = 0, owned_operations = 0;
    for (int index = 0; index < count; ++index) {
        const auto * node = ggml_graph_node(graph, index);
        if (tagged(node)) {
            const bool projection = node->op == GGML_OP_MUL_MAT;
            if (!(projection ? valid_projection(node) : valid_owned_elementwise(node)))
                throw std::runtime_error("Unsupported audio F32 Metal owned operation geometry.");
            if (projection) ++projections;
            ++owned_operations;
            // Resolve/check every tensor before encoding any GPU work.
            const int sources = projection || std::strcmp(node->name, depthwise_name) == 0 ? 2 :
                std::strcmp(node->name, vision_attention_name) == 0 ? 3 : 4;
            for (int source = 0; source < sources; ++source) buffer_id(backend_device, node->src[source]);
            buffer_id(backend_device, node);
        } else if (!ggml_backend_supports_op(backend, node)) {
            throw std::runtime_error("Audio F32 Metal graph contains an unsupported ordinary operation.");
        }
    }
    if (!owned_operations) return false;
    auto& cache = kernel_state();
    std::lock_guard<std::mutex> lock(cache.mutex);
    const auto synchronization_start = std::chrono::steady_clock::now();
    ggml_backend_synchronize(backend);
    const auto synchronization_end = std::chrono::steady_clock::now();
    auto device = static_cast<ggml_metal_device_t>(ggml_backend_get_device(backend)->context);
    @autoreleasepool {
        id<MTLDevice> metal_device = (__bridge id<MTLDevice>)ggml_metal_device_get_obj(device);
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ggml_metal_device_get_queue(device);
        ensure_pipeline(cache, metal_device);
        const auto encoding_start = std::chrono::steady_clock::now();
        id<MTLCommandBuffer> command = [queue commandBuffer];
        if (!command) throw std::runtime_error("Cannot allocate audio F32 Metal command buffer.");
        int start = 0;
        for (int index = 0; index < count; ++index) {
            auto * node = ggml_graph_node(graph, index);
            if (!tagged(node)) continue;
            encode_ordinary(device, command, graph, start, index);
            if (node->op == GGML_OP_MUL_MAT) encode_projection(backend_device, command, cache, node, prefer_mps);
            else encode_owned_elementwise(backend_device, command, cache, node);
            start = index + 1;
        }
        encode_ordinary(device, command, graph, start, count);
        const auto encoding_end = std::chrono::steady_clock::now();
        [command commit]; [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted)
            throw std::runtime_error("Audio F32 Metal command buffer failed: " +
                std::string(command.error.localizedDescription.UTF8String ?: "unknown Metal error"));
        *status = GGML_STATUS_SUCCESS;
        const char * profile = std::getenv("TS_GEMMA4_AUDIO_METAL_PROFILE");
        if (profile && profile[0] == '1' && profile[1] == '\0') {
            const auto complete = std::chrono::steady_clock::now();
            auto milliseconds = [](auto a, auto b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            const double gpu_ms = command.GPUEndTime > command.GPUStartTime
                ? (command.GPUEndTime - command.GPUStartTime) * 1000.0 : -1.0;
            // Diagnostic only: CPU encoding excludes uploads/preflight/cache
            // compilation; completion includes commit/wait, GPU spans the whole
            // command. It does not time individual kernels or whole requests.
            std::fprintf(stderr, "[gemma4-audio-metal-profile] {\"nodes\":%d,\"projections\":%d,\"owned_operations\":%d,"
                "\"synchronize_ms\":%.6f,\"encode_ms\":%.6f,\"commit_wait_ms\":%.6f,\"gpu_ms\":%.6f}\n",
                count, projections, owned_operations, milliseconds(synchronization_start, synchronization_end),
                milliseconds(encoding_start, encoding_end), milliseconds(encoding_end, complete), gpu_ms);
        }
    }
    return true;
}
