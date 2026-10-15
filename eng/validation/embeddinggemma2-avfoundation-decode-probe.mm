// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Independent application of the platform codec. No OpenCV/TensorSharp imports.
#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {
void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}
size_t checked_product(size_t a, size_t b) {
    require(b == 0 || a <= std::numeric_limits<size_t>::max() / b, "Pixel span overflow");
    return a * b;
}
std::vector<uint8_t> rgba(const uint8_t * pixels, size_t width, size_t height, size_t stride) {
    require(pixels && width > 0 && height > 0 && width <= 1024 && height <= 1024, "Unsupported pixel geometry");
    const size_t row = checked_product(width, 4);
    require(stride >= row && stride <= row + 4096, "Invalid source row stride");
    (void)checked_product(height, stride);
    std::vector<uint8_t> out(checked_product(row, height));
    for (size_t y = 0; y < height; ++y) for (size_t x = 0; x < width; ++x) {
        const auto * src = pixels + y * stride + x * 4;
        auto * dst = out.data() + (y * width + x) * 4;
        require(src[3] == 255, "This bounded proof requires opaque decoded alpha");
        dst[0] = src[2]; dst[1] = src[1]; dst[2] = src[0]; dst[3] = src[3];
    }
    return out;
}
void write_file(NSString * path, const void * data, size_t bytes) {
    const int fd = ::open(path.fileSystemRepresentation, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    require(fd >= 0, "Require a fresh output file");
    const auto * p = static_cast<const uint8_t *>(data);
    size_t done = 0;
    while (done < bytes) {
        const ssize_t n = ::write(fd, p + done, bytes - done);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { ::close(fd); throw std::runtime_error("Cannot write output"); }
        done += size_t(n);
    }
    require(::close(fd) == 0, "Cannot close output");
}
void controls() {
    const uint8_t source[] = {1,2,3,255,4,5,6,255,91,92,93,94,7,8,9,255,10,11,12,255,95,96,97,98};
    const uint8_t expected[] = {3,2,1,255,6,5,4,255,9,8,7,255,12,11,10,255};
    const auto result = rgba(source, 2, 2, 12);
    require(result.size() == sizeof(expected) && !std::memcmp(result.data(), expected, sizeof(expected)), "Padded channel copy control failed");
    size_t rejected = 0;
    for (int which = 0; which < 4; ++which) {
        try {
            if (which == 0) (void)rgba(source, 0, 2, 12);
            if (which == 1) (void)rgba(source, 2, 2, 7);
            if (which == 2) (void)rgba(source, 1025, 2, 4100);
            if (which == 3) (void)checked_product(std::numeric_limits<size_t>::max(), 2);
        } catch (const std::exception &) { ++rejected; }
    }
    require(rejected == 4, "Invalid span controls failed");
    std::vector<uint8_t> translucent(std::begin(source), std::end(source)); translucent[3] = 254;
    bool refused = false;
    try { (void)rgba(translucent.data(), 2, 2, 12); } catch (const std::exception &) { refused = true; }
    require(refused, "Alpha refusal control failed");
}
}

int main(int argc, char ** argv) {
    @autoreleasepool {
        try { controls(); } catch (const std::exception & e) { fprintf(stderr, "%s\n", e.what()); return 1; }
        if (argc == 2 && !std::strcmp(argv[1], "--host-controls")) {
            puts("AVFOUNDATION_COPY_CONTROLS passed=6 failed=0 decoded_frames=0 model_forwards=0"); return 0;
        }
        if (argc != 5 || std::strcmp(argv[1], "--video") || std::strcmp(argv[3], "--output")) {
            fputs("Require --video FILE --output FRESH_DIRECTORY\n", stderr); return 2;
        }
        NSString * video = [[NSString stringWithUTF8String:argv[2]] stringByStandardizingPath];
        NSString * output = [[NSString stringWithUTF8String:argv[4]] stringByStandardizingPath];
        if (!video.isAbsolutePath || !output.isAbsolutePath) { fputs("Require absolute paths\n", stderr); return 2; }
        NSFileManager * files = NSFileManager.defaultManager;
        if ([files fileExistsAtPath:output] || ![files createDirectoryAtPath:output withIntermediateDirectories:NO attributes:nil error:nil]) {
            fputs("Require a fresh output directory with an existing parent\n", stderr); return 2;
        }
        NSMutableDictionary * report = [@{ @"completed": @NO, @"passed": @NO,
            @"decoder": @"AVAssetReaderTrackOutput", @"requested_format": @"32BGRA",
            @"video": video, @"os": NSProcessInfo.processInfo.operatingSystemVersionString,
            @"independent_codec_implementation": @NO, @"ffmpeg_equivalence_qualified": @NO,
            @"model_forwards": @0, @"frames": [NSMutableArray array] } mutableCopy];
        AVAssetReader * reader = nil;
        try {
            require([files fileExistsAtPath:video], "Missing video input");
            AVURLAsset * asset = [AVURLAsset URLAssetWithURL:[NSURL fileURLWithPath:video] options:nil];
            NSArray<AVAssetTrack *> * tracks = [asset tracksWithMediaType:AVMediaTypeVideo];
            require(tracks.count == 1, "This bounded proof requires exactly one video track");
            AVAssetTrack * track = tracks.firstObject;
            const CGAffineTransform transform = track.preferredTransform;
            require(std::abs(transform.a - 1) <= 1e-9 && std::abs(transform.d - 1) <= 1e-9 &&
                std::abs(transform.b) <= 1e-9 && std::abs(transform.c) <= 1e-9 &&
                std::abs(transform.tx) <= 1e-9 && std::abs(transform.ty) <= 1e-9, "Nonidentity track transform is outside this proof");
            const double duration = CMTimeGetSeconds(asset.duration), fps = track.nominalFrameRate;
            require(std::isfinite(duration) && duration > 0 && std::isfinite(fps) && fps > 0 && fps <= 240, "Invalid track timing");
            report[@"duration_seconds"] = @(duration); report[@"nominal_fps"] = @(fps);
            NSError * error = nil;
            reader = [[AVAssetReader alloc] initWithAsset:asset error:&error];
            require(reader && !error, "Cannot initialize AVAssetReader");
            AVAssetReaderTrackOutput * stream = [[AVAssetReaderTrackOutput alloc] initWithTrack:track outputSettings:@{
                (NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA) }];
            stream.alwaysCopiesSampleData = NO;
            require([reader canAddOutput:stream], "Cannot add video track output"); [reader addOutput:stream];
            require([reader startReading], "Cannot start sequential decode");
            size_t index = 0; double previous = -std::numeric_limits<double>::infinity();
            while (CMSampleBufferRef sample = [stream copyNextSampleBuffer]) {
                CVPixelBufferRef pixels = CMSampleBufferGetImageBuffer(sample); bool locked = false;
                try {
                    require(index < 64 && pixels, "Fixture frame bound exceeded or missing decoded pixels");
                    const CMTime pts = CMSampleBufferGetPresentationTimeStamp(sample);
                    const double seconds = CMTimeGetSeconds(pts);
                    require(CMTIME_IS_NUMERIC(pts) && std::isfinite(seconds) && seconds >= previous, "Invalid presentation timestamp order");
                    previous = seconds;
                    require(!CVPixelBufferIsPlanar(pixels) && CVPixelBufferGetPixelFormatType(pixels) == kCVPixelFormatType_32BGRA,
                        "Decoder did not produce packed32BGRA");
                    require(CVPixelBufferLockBaseAddress(pixels, kCVPixelBufferLock_ReadOnly) == kCVReturnSuccess, "Cannot lock decoded pixels"); locked = true;
                    const size_t width = CVPixelBufferGetWidth(pixels), height = CVPixelBufferGetHeight(pixels), stride = CVPixelBufferGetBytesPerRow(pixels);
                    require(CVPixelBufferGetDataSize(pixels) >= checked_product(height, stride), "Decoded backing span is too short");
                    const auto packed = rgba(static_cast<const uint8_t *>(CVPixelBufferGetBaseAddress(pixels)), width, height, stride);
                    NSString * name = [NSString stringWithFormat:@"frame-%zu.rgba", index];
                    write_file([output stringByAppendingPathComponent:name], packed.data(), packed.size());
                    [report[@"frames"] addObject:@{ @"index": @(index), @"width": @(width), @"height": @(height),
                        @"decoded_bytes_per_row": @(stride), @"pts_value": @(pts.value), @"pts_timescale": @(pts.timescale),
                        @"pts_seconds": @(seconds), @"opaque_alpha": @YES, @"rgba_file": name, @"rgba_bytes": @(packed.size()) }];
                    require(CVPixelBufferUnlockBaseAddress(pixels, kCVPixelBufferLock_ReadOnly) == kCVReturnSuccess, "Cannot unlock decoded pixels"); locked = false;
                    CFRelease(sample); ++index;
                } catch (...) { if (locked) CVPixelBufferUnlockBaseAddress(pixels, kCVPixelBufferLock_ReadOnly); CFRelease(sample); throw; }
            }
            require(reader.status == AVAssetReaderStatusCompleted && index > 0, "Sequential decode did not complete");
            report[@"frame_count"] = @(index); report[@"completed"] = @YES; report[@"passed"] = @YES;
        } catch (const std::exception & e) {
            [reader cancelReading]; report[@"error"] = [NSString stringWithUTF8String:e.what()];
            if (reader.error) report[@"platform_error"] = reader.error.localizedDescription;
        }
        NSError * error = nil;
        NSData * json = [NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:&error];
        if (!json || error) { fputs("Cannot serialize decode report\n", stderr); return 1; }
        try { write_file([output stringByAppendingPathComponent:@"decode.json"], json.bytes, json.length); }
        catch (const std::exception & e) { fprintf(stderr, "%s\n", e.what()); return 1; }
        return [report[@"passed"] boolValue] ? 0 : 1;
    }
}
