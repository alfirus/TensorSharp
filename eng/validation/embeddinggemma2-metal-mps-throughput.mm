// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Standalone MPS throughput/capability screen. No model routing or quality claim.
// Build: clang++ -std=c++17 -O2 -fobjc-arc <source> -framework Foundation
//        -framework Metal -framework MetalPerformanceShaders -o <ignored output>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>
struct Sample { double host_ms, gpu_ms; };
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return (values[(values.size()-1)/2] + values[values.size()/2]) / 2;
}
int main() {
    @autoreleasepool {
      try {
        @try {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device || !MPSSupportsMTLDevice(device)) return 2;
            id<MTLCommandQueue> queue = [device newCommandQueue]; if (!queue) return 2;
            constexpr NSUInteger rows = 2304;
            std::printf("{\"device\":\"%s\",\"rows\":%lu,\"warmup\":3,\"minimum_repeats\":30,\"minimum_seconds\":0.25,\"options\":\"MPSKernelOptionsNone\",\"output\":\"F32\"}\n", device.name.UTF8String, rows);
            for (auto [inner, columns] : {std::pair<NSUInteger,NSUInteger>{768,768}, {768,3072}, {3072,768}}) {
                id<MTLBuffer> xf = [device newBufferWithLength:rows * inner * 4 options:MTLResourceStorageModeShared];
                id<MTLBuffer> wf = [device newBufferWithLength:columns * inner * 4 options:MTLResourceStorageModeShared];
                id<MTLBuffer> xh = [device newBufferWithLength:rows * inner * 2 options:MTLResourceStorageModeShared];
                id<MTLBuffer> wh = [device newBufferWithLength:columns * inner * 2 options:MTLResourceStorageModeShared];
                id<MTLBuffer> output = [device newBufferWithLength:rows * columns * 4 options:MTLResourceStorageModeShared];
                if (!xf || !wf || !xh || !wh || !output) return 2;
                auto * x = static_cast<float *>(xf.contents); auto * w = static_cast<float *>(wf.contents);
                auto * hx = static_cast<_Float16 *>(xh.contents); auto * hw = static_cast<_Float16 *>(wh.contents);
                // Exactly representable Half values keep F32/F16 operands
                // identical; throughput comparisons do not hide quantization.
                for (NSUInteger i=0;i<rows*inner;++i) hx[i] = static_cast<_Float16>(x[i] = (static_cast<int>(i%17)-8)/16.f);
                for (NSUInteger i=0;i<columns*inner;++i) hw[i] = static_cast<_Float16>(w[i] = (static_cast<int>(i%13)-6)/16.f);
                auto matrix = [&](id<MTLBuffer> buffer, NSUInteger nrows, NSUInteger ncols, bool half) {
                    MPSMatrixDescriptor * descriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:nrows columns:ncols
                        rowBytes:ncols*(half?2:4) dataType:half?MPSDataTypeFloat16:MPSDataTypeFloat32];
                    return [[MPSMatrix alloc] initWithBuffer:buffer descriptor:descriptor];
                };
                MPSMatrix * fx = matrix(xf,rows,inner,false), * fw = matrix(wf,columns,inner,false);
                MPSMatrix * hxm = matrix(xh,rows,inner,true), * hwm = matrix(wh,columns,inner,true);
                MPSMatrix * y = matrix(output,rows,columns,false);
                if (!fx || !fw || !hxm || !hwm || !y) return 2;
                std::vector<double> mode_medians;
                for (int mode=0;mode<4;++mode) {
                    const bool half = mode!=0; const int products = mode<2?1:mode==2?3:6;
                    MPSMatrixMultiplication * first = [[MPSMatrixMultiplication alloc] initWithDevice:device transposeLeft:NO transposeRight:YES
                        resultRows:rows resultColumns:columns interiorColumns:inner alpha:1 beta:0];
                    MPSMatrixMultiplication * accumulate = [[MPSMatrixMultiplication alloc] initWithDevice:device transposeLeft:NO transposeRight:YES
                        resultRows:rows resultColumns:columns interiorColumns:inner alpha:1 beta:1];
                    if (!first || !accumulate) return 2;
                    first.options = MPSKernelOptionsNone; accumulate.options = MPSKernelOptionsNone;
                    auto dispatch = [&]() {
                        const auto start = std::chrono::steady_clock::now();
                        id<MTLCommandBuffer> command = [queue commandBuffer]; if (!command) throw std::runtime_error("No command buffer.");
                        for (int product=0;product<products;++product)
                            [(product?accumulate:first) encodeToCommandBuffer:command leftMatrix:(half?hxm:fx) rightMatrix:(half?hwm:fw) resultMatrix:y];
                        [command commit]; [command waitUntilCompleted];
                        if (command.status != MTLCommandBufferStatusCompleted)
                            throw std::runtime_error(command.error.localizedDescription.UTF8String ?: "MPS command failed.");
                        const double gpu = (command.GPUEndTime-command.GPUStartTime)*1000;
                        if (!(gpu>0) || !std::isfinite(gpu)) throw std::runtime_error("GPU timestamps unavailable.");
                        return Sample{std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count(),gpu};
                    };
                    for (int warm=0;warm<3;++warm) dispatch();
                    std::vector<double> host, gpu; const auto run_start = std::chrono::steady_clock::now();
                    while (host.size()<30 || std::chrono::duration<double>(std::chrono::steady_clock::now()-run_start).count()<.25) {
                        auto sample=dispatch();host.push_back(sample.host_ms);gpu.push_back(sample.gpu_ms);
                    }
                    double maximum = 0; const auto * actual = static_cast<const float *>(output.contents);
                    for (NSUInteger row : {0ul,37ul,rows-1})for (NSUInteger column : {0ul,19ul,columns-1}) {
                        double expected=0;
                        for (NSUInteger k=0;k<inner;++k) expected+=static_cast<double>(x[row*inner+k])*w[column*inner+k];
                        expected*=products;
                        const double value=actual[row*columns+column];
                        if (!std::isfinite(value)) return 1;
                        maximum=std::fmax(maximum,std::fabs(value-expected));
                    }
                    if (maximum>3e-5) { std::fprintf(stderr,"Unchanged double check failed: %.12g\n",maximum);return 1; }
                    const double gpu_median=median(gpu);mode_medians.push_back(gpu_median);
                    std::printf("{\"inner\":%lu,\"columns\":%lu,\"input\":\"%s\",\"products\":%d,\"samples\":%zu,\"host_median_ms\":%.9g,\"gpu_median_ms\":%.9g,\"max_double_error\":%.12g,\"passed\":true}\n",
                        inner,columns,half?"F16":"F32",products,host.size(),median(host),gpu_median,maximum);
                }
                std::printf("{\"inner\":%lu,\"columns\":%lu,\"half_1_vs_float_1\":%.9g,\"half_3_vs_float_1\":%.9g,\"half_6_vs_float_1\":%.9g}\n",
                    inner,columns,mode_medians[1]/mode_medians[0],mode_medians[2]/mode_medians[0],mode_medians[3]/mode_medians[0]);
            }
            std::puts("{\"completed\":true,\"executed_shapes\":3,\"executed_modes\":12,\"skipped\":0,\"passed\":true}");return 0;
        } @catch(NSException *error) { std::fprintf(stderr,"MPS unsupported/error: %s\n",error.reason.UTF8String);return 1; }
      } catch (const std::exception& error) { std::fprintf(stderr,"MPS validation failed: %s\n",error.what()); return 1; }
    }
}
