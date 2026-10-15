// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Standalone capability/accumulator probe; no TensorSharp model or timing claim.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cmath>
#include <cstdio>

int main() {
    @autoreleasepool {
        @try {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device || !MPSSupportsMTLDevice(device)) return 2;
            constexpr NSUInteger rows = 3, inner = 64, columns = 8;
            id<MTLBuffer> xb = [device newBufferWithLength:rows * inner * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> wb = [device newBufferWithLength:columns * inner * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> yb = [device newBufferWithLength:rows * columns * 4 options:MTLResourceStorageModeShared];
            auto * x = static_cast<_Float16 *>(xb.contents); auto * w = static_cast<_Float16 *>(wb.contents);
            for (NSUInteger i = 0; i < rows * inner; ++i) x[i] = static_cast<_Float16>((static_cast<int>(i % 17) - 8) / 11.0f);
            for (NSUInteger i = 0; i < columns * inner; ++i) w[i] = static_cast<_Float16>((static_cast<int>(i % 13) - 6) / 7.0f);
            x[0] = 32000; w[0] = 4; // Exact Half operands, product exceeds Half range.
            MPSMatrixDescriptor * xd = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:inner rowBytes:inner * 2 dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor * wd = [MPSMatrixDescriptor matrixDescriptorWithRows:columns columns:inner rowBytes:inner * 2 dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor * yd = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:columns rowBytes:columns * 4 dataType:MPSDataTypeFloat32];
            MPSMatrix * xm = [[MPSMatrix alloc] initWithBuffer:xb descriptor:xd];
            MPSMatrix * wm = [[MPSMatrix alloc] initWithBuffer:wb descriptor:wd];
            MPSMatrix * ym = [[MPSMatrix alloc] initWithBuffer:yb descriptor:yd];
            MPSMatrixMultiplication * operation = [[MPSMatrixMultiplication alloc] initWithDevice:device
                transposeLeft:NO transposeRight:YES resultRows:rows resultColumns:columns interiorColumns:inner alpha:1 beta:0];
            operation.options = MPSKernelOptionsNone;
            id<MTLCommandQueue> queue = [device newCommandQueue]; id<MTLCommandBuffer> command = [queue commandBuffer];
            [operation encodeToCommandBuffer:command leftMatrix:xm rightMatrix:wm resultMatrix:ym];
            [command commit]; [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted) {
                std::fprintf(stderr, "GPU failure: %s\n", command.error.localizedDescription.UTF8String); return 1;
            }
            double maximum = 0; const auto * y = static_cast<const float *>(yb.contents);
            for (NSUInteger row = 0; row < rows; ++row) for (NSUInteger column = 0; column < columns; ++column) {
                double expected = 0;
                for (NSUInteger k = 0; k < inner; ++k) expected += static_cast<double>(x[row * inner + k]) * static_cast<double>(w[column * inner + k]);
                if (!std::isfinite(y[row * columns + column])) return 1;
                maximum = std::fmax(maximum, std::fabs(y[row * columns + column] - expected));
            }
            std::printf("{\"device\":\"%s\",\"input\":\"F16\",\"weight\":\"F16\",\"output\":\"F32\",\"maximum_error\":%.12g,\"range_output\":%.12g,\"passed\":%s}\n",
                device.name.UTF8String, maximum, static_cast<double>(y[0]), maximum <= .02 && y[0] > 65504 ? "true" : "false");
            return maximum <= .02 && y[0] > 65504 ? 0 : 1;
        } @catch (NSException * error) {
            std::fprintf(stderr, "Mixed MPS matrix types unsupported: %s\n", error.reason.UTF8String); return 1;
        }
    }
}
