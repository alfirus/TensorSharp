// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Full-bit original-tile oracle plus the current production-auto timing control.
// Synthetic stream timings do not assert or establish end-to-end model speed.
#include "ggml_ops_q8_precision.h"
#include "ggml_ops_q8_ordered_policy.h"
#include "q8_precision_test_hooks.h"
#include "precision_test_utils.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
void checked(cudaError_t status) { require(status == cudaSuccess, cudaGetErrorString(status)); }
struct allocation {
    void* data = nullptr;
    explicit allocation(size_t bytes) { checked(cudaMalloc(&data, bytes)); }
    ~allocation() { if (data) cudaFree(data); }
    allocation(const allocation&) = delete;
    allocation& operator=(const allocation&) = delete;
};

void run(int k, int m, int n, int layout, bool benchmark, bool wide, bool tiles = false,
        bool gate_tile16 = false) {
    const size_t ws = size_t(k / 32) * 34 + (layout ? 6 : 0);
    const size_t xs = layout == 2 ? size_t(n + 3) * 4 : layout == 1 ? 8 : 4;
    const size_t cs = layout == 2 ? 4 : size_t(k) * xs + (layout ? 20 : 0);
    const size_t input_bytes = size_t(k - 1) * xs + size_t(n - 1) * cs + 4;
    const size_t output_bytes = (size_t(m) * n + 2) * 4;
    std::vector<unsigned char> weights(size_t(m) * ws, 0xff), input(input_bytes, 0xff);
    std::vector<float> logical(size_t(k) * n), reference(size_t(m) * n + 2), actual(reference.size());
    allocation dw(weights.size()), dx(input.size()), dy(output_bytes);
    enum class route { original, control, candidate, tile16, tile32, tile64 };
    auto launch = [&](route selected) {
        if (selected == route::tile16 || selected == route::tile32 || selected == route::tile64) {
            const int tile = selected == route::tile16 ? 16 : selected == route::tile32 ? 32 : 64;
            checked(static_cast<cudaError_t>(tsg_matmul_q8_cuda_launch_test_tile(dw.data, dx.data,
                static_cast<float*>(dy.data) + 1, k, m, n, ws, xs, cs, tile, nullptr)));
            return;
        }
        auto fn = selected == route::candidate ? tsg_matmul_q8_cuda_launch
            : selected == route::control ? tsg_matmul_q8_cuda_launch_control
            : tsg_matmul_q8_cuda_launch_reference;
        checked(static_cast<cudaError_t>(fn(dw.data, dx.data, static_cast<float*>(dy.data) + 1,
            k, m, n, ws, xs, cs, nullptr)));
    };
    auto route_name = [](route selected) {
        return selected == route::tile16 ? "tile16" : selected == route::tile32 ? "tile32"
            : selected == route::tile64 ? "tile64" : selected == route::control ? "production_auto" : "candidate";
    };
    std::vector<route> checked_routes{route::original, route::control, route::candidate};
    if (tiles) checked_routes.insert(checked_routes.end(), {route::tile16, route::tile32, route::tile64});
    constexpr float canary = -12345.625f;
    for (int revision = 0; revision < 2; ++revision) {
        for (int r = 0; r < m; ++r) for (int b = 0; b < k / 32; ++b) {
            const uint16_t scale = uint16_t(((8 + (r + b) % 4) << 10)
                | ((r * 71 + b * 19 + revision * 13) % 1024));
            auto* block = weights.data() + size_t(r) * ws + size_t(b) * 34;
            std::memcpy(block, &scale, 2);
            for (int j = 0; j < 32; ++j)
                block[2 + j] = uint8_t((uint32_t(b * 32 + j) * 37 + uint32_t(r) * 19 + revision * 11) % 256 - 128);
        }
        for (int c = 0; c < n; ++c) for (int j = 0; j < k; ++j) {
            const uint32_t hash = uint32_t(j + 1) * 2654435761u ^ uint32_t(c + 7) * 2246822519u;
            const float value = (float(int(hash % 65537) - 32768) / 32768.0f
                + float((hash >> 17) % 13) * 0x1p-22f) * (revision ? 0.5f : 1.0f);
            logical[size_t(c) * k + j] = value;
            std::memcpy(input.data() + size_t(c) * cs + size_t(j) * xs, &value, 4);
        }
        checked(cudaMemcpy(dw.data, weights.data(), weights.size(), cudaMemcpyHostToDevice));
        checked(cudaMemcpy(dx.data, input.data(), input.size(), cudaMemcpyHostToDevice));
        for (route selected : checked_routes) {
            std::fill(actual.begin(), actual.end(), canary);
            checked(cudaMemcpy(dy.data, actual.data(), output_bytes, cudaMemcpyHostToDevice));
            launch(selected); checked(cudaDeviceSynchronize());
            checked(cudaMemcpy(actual.data(), dy.data, output_bytes, cudaMemcpyDeviceToHost));
            require(actual.front() == canary && actual.back() == canary, "Ordered small-batch output guard overwritten");
            require(std::all_of(actual.begin(), actual.end(), [](float x) { return std::isfinite(x); }), "Nonfinite small-batch output");
            if (selected == route::original) reference = actual;
            else require(std::memcmp(reference.data(), actual.data(), output_bytes) == 0,
                "Production control or ordered candidate differs from original tile output bits");
        }
        // Independent FP64 dots for every column: all rows in bounded narrow
        // correctness mode, sampled rows for wide/tile/timing runs. The full
        // byte comparison above always covers every output, including guards.
        const int samples = benchmark || wide || tiles || gate_tile16 ? std::min(m, 17) : m;
        for (int c = 0; c < n; ++c) for (int sample = 0; sample < samples; ++sample) {
            const int r = samples == 1 ? 0 : int(int64_t(sample) * (m - 1) / (samples - 1));
            double dot = 0, absolute = 0;
            for (int j = 0; j < k; ++j) {
                const double scale = std::ldexp(1.0 + double((r * 71 + j / 32 * 19 + revision * 13) % 1024) / 1024.0,
                    -7 + (r + j / 32) % 4);
                const int quant = int((uint32_t(j) * 37 + uint32_t(r) * 19 + revision * 11) % 256) - 128;
                const double product = scale * quant * logical[size_t(c) * k + j];
                dot += product; absolute += std::abs(product);
            }
            const double ku = k * 0x1p-24;
            require(std::abs(reference[1 + size_t(c) * m + r] - dot) <= ku / (1 - ku) * absolute,
                "Ordered small-batch independent sequential-FMA error bound failed");
        }
        std::vector<unsigned char> readback(weights.size());
        checked(cudaMemcpy(readback.data(), dw.data, weights.size(), cudaMemcpyDeviceToHost));
        require(readback == weights, "Ordered small-batch modified weight bytes");
        readback.resize(input.size()); checked(cudaMemcpy(readback.data(), dx.data, input.size(), cudaMemcpyDeviceToHost));
        require(readback == input, "Ordered small-batch modified input bytes");
    }
    const bool uses_ordered = wide ? tsg_q8_ordered_gate_prefill_shape(k, m, n)
        : tsg_q8_ordered_small_batch_shape(k, m, n);
    const bool uses_gate_tile16 = gate_tile16 && !uses_ordered && tsg_q8_gate_prefill_tile16_shape(k, m, n);
    std::printf("{\"check\":true,\"k\":%d,\"m\":%d,\"n\":%d,\"layout\":%d,\"revisions\":2,\"candidate_ordered\":%s,\"candidate_gate_tile16\":%s,\"forced_tiles_checked\":%s,\"production_control_equal\":true,\"all_output_bits_equal\":true}\n",
        k, m, n, layout, uses_ordered ? "true" : "false", uses_gate_tile16 ? "true" : "false", tiles ? "true" : "false");
    if (!benchmark) return;
    cudaEvent_t start, end; checked(cudaEventCreate(&start)); checked(cudaEventCreate(&end));
    const std::vector<route> alternatives = tiles
        ? std::vector<route>{route::tile16, route::tile32, route::tile64}
        : std::vector<route>{route::candidate};
    for (route alternative : alternatives) {
        // Sustained warmup avoids timing the initial device-clock ramp. Both
        // paths use identical data and allocations for each bracketed ABBA run.
        for (route selected : {route::control, alternative}) {
            checked(cudaEventRecord(start));
            float elapsed = 0;
            do {
                for (int i = 0; i < 32; ++i) launch(selected);
                checked(cudaEventRecord(end)); checked(cudaEventSynchronize(end));
                checked(cudaEventElapsedTime(&elapsed, start, end));
            } while (elapsed < 250.0f);
        }
        for (int pass = 0; pass < 4; ++pass) {
            const bool candidate = pass == 1 || pass == 2; // ABBA on the same buffers.
            const route selected = candidate ? alternative : route::control;
            for (int i = 0; i < 3; ++i) launch(selected);
            checked(cudaEventRecord(start));
            for (int i = 0; i < 3; ++i) launch(selected);
            checked(cudaEventRecord(end)); checked(cudaEventSynchronize(end));
            float elapsed = 0; checked(cudaEventElapsedTime(&elapsed, start, end));
            const int repeats = std::clamp(int(std::ceil(50.0 / std::max(double(elapsed) / 3, 0.001))), 3, 10000);
            checked(cudaEventRecord(start));
            for (int i = 0; i < repeats; ++i) launch(selected);
            checked(cudaEventRecord(end)); checked(cudaEventSynchronize(end));
            checked(cudaEventElapsedTime(&elapsed, start, end));
            std::printf("{\"k\":%d,\"m\":%d,\"n\":%d,\"comparison\":\"%s\",\"route\":\"%s\",\"pass\":%d,\"repeats\":%d,\"stream_us\":%.6f}\n",
                k, m, n, route_name(alternative), route_name(selected), pass, repeats, elapsed * 1000.0 / repeats);
        }
    }
    checked(cudaEventDestroy(start)); checked(cudaEventDestroy(end));
}
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    require(argc == 2, "Use --check/--benchmark with optional -wide, -tiles or -gate-tile16; --check-gate-combined also supported");
    const std::string mode = argv[1];
    const bool combined = mode == "--check-gate-combined";
    const bool wide = mode == "--check-wide" || mode == "--benchmark-wide" || combined;
    const bool tiles = mode == "--check-tiles" || mode == "--benchmark-tiles";
    const bool gate_tile16 = mode == "--check-gate-tile16" || mode == "--benchmark-gate-tile16" || combined;
    const bool benchmark = mode == "--benchmark" || mode == "--benchmark-wide"
        || mode == "--benchmark-tiles" || mode == "--benchmark-gate-tile16";
    require(wide || tiles || gate_tile16 || mode == "--check" || mode == "--benchmark", "Unexpected benchmark mode");
    // Dispatch bounds matter: extending this exact-arithmetic experiment to
    // poorly matched shapes can still regress performance substantially.
    for (int n = 0; n <= 513; ++n) {
        require(tsg_q8_ordered_small_batch_shape(128, 8192, n) == (n >= 2 && n <= 4),
            "Short-K output-wide gate must retain tile reuse after N=4");
        for (int m : {63, 64, 65, 127, 128, 129}) {
            require(tsg_q8_ordered_gate_prefill_shape(4096, m, n)
                    == (n >= 9 && ((m == 64 && n <= 336) || (m == 128 && n <= 128))),
                "Wide ordered gate dispatch escaped its measured shape bounds");
            require(!tsg_q8_ordered_gate_prefill_shape(4064, m, n)
                    && !tsg_q8_ordered_gate_prefill_shape(4128, m, n),
                "Unqualified K must keep production dispatch");
            require(tsg_q8_gate_prefill_tile16_shape(4096, m, n)
                    == ((m == 64 || m == 128) && n >= 64 && n <= 512),
                "Gate tile16 dispatch escaped its measured shape bounds");
            require(!tsg_q8_gate_prefill_tile16_shape(4064, m, n)
                    && !tsg_q8_gate_prefill_tile16_shape(4128, m, n),
                "Unqualified K must retain original tile policy");
        }
    }
#ifdef _WIN32
    require(_putenv_s("TS_GGML_Q8_ORDERED_SMALL_BATCH", combined || (!wide && !tiles && !gate_tile16) ? "1" : "0") == 0, "Cannot configure ordered batch");
    require(_putenv_s("TS_GGML_Q8_ORDERED_GATE_PREFILL", wide ? "1" : "0") == 0, "Cannot configure ordered gate prefill");
    require(_putenv_s("TS_GGML_Q8_GATE_PREFILL_TILE16", gate_tile16 ? "1" : "0") == 0, "Cannot configure gate tile16");
    require(_putenv_s("TS_GGML_Q8_PREFILL_TILE", "auto") == 0, "Cannot configure production auto control");
    require(_putenv_s("TS_GGML_Q8_PARALLEL_VECTOR", "0") == 0, "Cannot disable parallel vector");
    require(_putenv_s("TS_GGML_Q8_PARALLEL_SMALL_BATCH", "0") == 0, "Cannot disable parallel batch");
#else
    require(setenv("TS_GGML_Q8_ORDERED_SMALL_BATCH", combined || (!wide && !tiles && !gate_tile16) ? "1" : "0", 1) == 0, "Cannot configure ordered batch");
    require(setenv("TS_GGML_Q8_ORDERED_GATE_PREFILL", wide ? "1" : "0", 1) == 0, "Cannot configure ordered gate prefill");
    require(setenv("TS_GGML_Q8_GATE_PREFILL_TILE16", gate_tile16 ? "1" : "0", 1) == 0, "Cannot configure gate tile16");
    require(setenv("TS_GGML_Q8_PREFILL_TILE", "auto", 1) == 0, "Cannot configure production auto control");
    require(setenv("TS_GGML_Q8_PARALLEL_VECTOR", "0", 1) == 0, "Cannot disable parallel vector");
    require(setenv("TS_GGML_Q8_PARALLEL_SMALL_BATCH", "0", 1) == 0, "Cannot disable parallel batch");
#endif
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    checked(cudaSetDevice(0));
    if (gate_tile16) {
        for (int m : {64, 128}) for (int n : {63, 64, 129, 337, 512, 513}) {
            if (benchmark && (n == 63 || n == 513)) continue;
            for (int layout = 0; layout < (benchmark ? 1 : 3); ++layout)
                run(4096, m, n, layout, benchmark, wide, false, true);
        }
        if (!benchmark) {
            for (int m : {63, 65, 127, 129}) run(4096, m, 129, 1, false, wide, false, true);
            for (int k : {4064, 4128}) run(k, 64, 129, 2, false, wide, false, true);
        }
        return 0;
    }
    if (tiles) {
        for (int m : {64, 128}) for (int n : {64, 128, 336, 512})
            for (int layout = 0; layout < (benchmark ? 1 : 3); ++layout)
                run(4096, m, n, layout, benchmark, false, true);
        return 0;
    }
    if (wide) {
        for (int m : {64, 128}) for (int n : {9, 16, 32, 33, 64, 128, 336, 512})
            for (int layout = 0; layout < (benchmark ? 1 : 3); ++layout)
                run(4096, m, n, layout, benchmark, true);
        if (!benchmark) {
            for (int m : {63, 65, 127, 129}) run(4096, m, 33, 1, false, true);
            for (int k : {4064, 4128}) run(k, 64, 33, 2, false, true);
            for (int n : {8, 513}) run(4096, 64, n, 1, false, true);
            run(4096, 64, 337, 1, false, true);
            run(4096, 128, 129, 2, false, true);
            run(128, 8192, 33, 0, false, true);
        }
        return 0;
    }
    if (!benchmark) for (int n = 1; n <= 8; ++n)
        for (const auto shape : {std::array<int, 2>{32, 1}, {96, 33}, {1024, 65}})
            for (int layout : {0, 1, 2}) run(shape[0], shape[1], n, layout, false, false);
    // Shapes from the real Flash checkpoint: beta, f_a/g_a and f_b/g_b.
    for (const auto shape : {std::array<int, 2>{4096, 64}, {4096, 128}, {128, 8192}})
        for (int n : {1, 2, 3, 4, 8}) run(shape[0], shape[1], n, 0, benchmark, false);
    return 0;
}
