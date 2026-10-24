// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Same-device original K-ordered arithmetic is the full-byte oracle. FP64 dots
// below are an independent sanity bound, never a replacement exactness bar.
#include "q8_hc_compact_test_hooks.h"
#include "../ggml_ops_q8_hc_compact_policy.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
static_assert(sizeof(float) == 4 && sizeof(std::uint16_t) == 2 && sizeof(std::uint32_t) == 4,
    "Fixture requires IEEE-width Q8/F32 storage");
static_assert(std::numeric_limits<float>::is_iec559, "Fixture requires IEEE binary32");
std::size_t checks = 0, compared_values = 0, cases = 0, live_allocations = 0;
std::size_t graph_cases = 0, graph_replays = 0;
void require(bool condition, const char * message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void checked(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void environment(const char * name, const char * value) {
#ifdef _WIN32
    require(_putenv_s(name, value) == 0, "Cannot set fixture environment");
#else
    require(setenv(name, value, 1) == 0, "Cannot set fixture environment");
#endif
}
struct device_memory {
    void * data = nullptr;
    explicit device_memory(std::size_t bytes) { checked(cudaMalloc(&data, bytes)); ++live_allocations; }
    device_memory(const device_memory &) = delete;
    device_memory & operator=(const device_memory &) = delete;
    void close() { if (data) { checked(cudaFree(data)); data = nullptr; --live_allocations; } }
    ~device_memory() { if (data) { cudaFree(data); --live_allocations; } }
};
struct stream_owner {
    cudaStream_t value = nullptr;
    stream_owner() { checked(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    void close() { if (value) { checked(cudaStreamDestroy(value)); value = nullptr; } }
    ~stream_owner() { if (value) cudaStreamDestroy(value); }
};
struct events {
    cudaEvent_t first = nullptr, last = nullptr;
    events() {
        checked(cudaEventCreate(&first));
        const auto status = cudaEventCreate(&last);
        if (status != cudaSuccess) { cudaEventDestroy(first); first = nullptr; checked(status); }
    }
    void close() { checked(cudaEventDestroy(last)); last = nullptr; checked(cudaEventDestroy(first)); first = nullptr; }
    ~events() { if (last) cudaEventDestroy(last); if (first) cudaEventDestroy(first); }
};
struct captured_graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    void close() {
        if (executable) { checked(cudaGraphExecDestroy(executable)); executable = nullptr; }
        if (graph) { checked(cudaGraphDestroy(graph)); graph = nullptr; }
    }
    ~captured_graph() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }
};
struct capture_scope {
    cudaStream_t stream;
    bool active = false;
    explicit capture_scope(cudaStream_t value) : stream(value) {
        checked(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)); active = true;
    }
    cudaError_t end(cudaGraph_t * graph) {
        const auto status = cudaStreamEndCapture(stream, graph); active = false; return status;
    }
    ~capture_scope() {
        if (active) {
            cudaGraph_t abandoned = nullptr;
            cudaStreamEndCapture(stream, &abandoned);
            if (abandoned) cudaGraphDestroy(abandoned);
        }
    }
};

enum class pattern { ordinary, cancellation, subnormal, signed_zero, half_extremes };
struct shape { int k, m, n, layout; pattern data = pattern::ordinary; };
constexpr std::size_t guard = 64;
constexpr float output_guard = -12345.625f;

// Independently decode the exact public IEEE binary16 scale bits used below.
// No production/device conversion routine or quantized payload is read here.
double half_value(std::uint16_t bits) {
    const int sign = (bits & 0x8000u) ? -1 : 1;
    const int exponent = (bits >> 10) & 31, fraction = bits & 1023;
    return sign * (exponent == 0 ? std::ldexp(double(fraction), -24)
        : std::ldexp(1.0 + double(fraction) / 1024.0, exponent - 15));
}
std::uint16_t scale_bits(const shape & s, int row, int block, int revision) {
    if (s.data == pattern::subnormal) return 1; // Smallest positive half scale.
    if (s.data == pattern::signed_zero) return (row + block + revision) & 1 ? 0x8000u : 0;
    if (s.data == pattern::cancellation) return 0x3c00u;
    if (s.data == pattern::half_extremes) {
        constexpr std::uint16_t samples[] = {0x7bffu, 0xfbffu, 0x0001u, 0x8001u, 0x0400u, 0x8400u};
        return samples[(row + block + revision) % 6];
    }
    // Non-power-of-two, exactly representable half scales in every ordinary case.
    return std::uint16_t(((8 + (row + block) % 4) << 10)
        | (1 + (row * 71 + block * 19 + revision * 13) % 1023));
}
int quant(const shape & s, int row, int k, int revision) {
    if (s.data == pattern::cancellation) return 1;
    if (s.data == pattern::subnormal) return k == 0 ? 1 : 0;
    return int((std::uint32_t(k) * 37 + std::uint32_t(row) * 19 + revision * 11) % 256) - 128;
}
float activation(const shape & s, int col, int k, int revision) {
    if (s.data == pattern::cancellation)
        return k % 4 == 0 ? 100000000.0f : k % 4 == 2 ? -100000000.0f : revision ? 0.5f : 1.0f;
    if (s.data == pattern::subnormal) return k == 0 ? 0x1p-110f : 0.0f;
    if (s.data == pattern::signed_zero) return (k + col + revision) & 1 ? -0.0f : 0.0f;
    const std::uint32_t hash = std::uint32_t(k + 1) * 2654435761u ^ std::uint32_t(col + 7) * 2246822519u;
    return (float(int(hash % 65537) - 32768) / 32768.0f
        + float((hash >> 17) % 13) * 0x1p-22f) * (revision ? 0.5f : 1.0f);
}

void run(const shape & s, bool enabled, bool benchmark, cudaStream_t stream,
         bool kda_probe = false, bool kda_mode = false, bool kda_b_probe = false) {
    require(s.k > 0 && s.k % 32 == 0 && s.m > 0 && s.n > 0, "Invalid bounded fixture shape");
    const std::size_t ws = std::size_t(s.k / 32) * 34 + (s.layout ? 6 : 0);
    const std::size_t xs = s.layout == 2 ? std::size_t(s.n + 3) * 4 : s.layout == 1 ? 8 : 4;
    const std::size_t cs = s.layout == 2 ? 4 : std::size_t(s.k) * xs + (s.layout ? 20 : 0);
    const std::size_t xb = std::size_t(s.k - 1) * xs + std::size_t(s.n - 1) * cs + 4;
    const std::size_t count = std::size_t(s.m) * s.n;
    std::vector<unsigned char> weights(guard + std::size_t(s.m) * ws + guard, 0xff);
    std::vector<unsigned char> input(guard + xb + guard, 0xff);
    std::vector<float> reference(count + 32, output_guard), actual(reference.size()), poison(reference.size());
    device_memory dw(weights.size()), dx(input.size()), dy(reference.size() * sizeof(float));
    captured_graph captured;
    std::vector<float> graph_first_output;
    cudaGraphExec_t first_executable = nullptr;
    const bool capture_shape = kda_mode ? s.k == 4096 && (s.m == 64 || s.m == 128)
                                        : s.k == 16384 && s.m == 24;
    const bool capture_case = capture_shape && s.data == pattern::ordinary
        && ((s.n == 1 && s.layout == 0) || (s.n == 13 && s.layout == 1));
    const bool kda_target = s.k == 4096 && (s.m == 64 || s.m == 128)
        && (s.n == 1 || (s.n >= 9 && s.n <= 512));
    const int kda_variant = enabled && kda_target ? (s.n == 1 ? 1 : 2) : 0;
    if (kda_mode) require(tsg_matmul_q8_kda_cuda_test_selected(s.k, s.m, s.n, enabled) == kda_variant,
        "KDA policy changed an OFF, boundary or non-target shape");
    void * wp = static_cast<unsigned char *>(dw.data) + guard;
    void * xp = static_cast<unsigned char *>(dx.data) + guard;
    float * yp = static_cast<float *>(dy.data) + 16;
    auto launch = [&](TsgQ8HcTestRoute route, int tile) {
        checked(static_cast<cudaError_t>(tsg_matmul_q8_hc_cuda_test_launch(route, wp, xp, yp,
            s.k, s.m, s.n, ws, xs, cs, tile, stream)));
    };
    auto download = [&] {
        checked(cudaMemcpyAsync(actual.data(), dy.data, actual.size() * 4, cudaMemcpyDeviceToHost, stream));
        checked(cudaStreamSynchronize(stream));
    };
    auto sources_unchanged = [&] {
        std::vector<unsigned char> readback(weights.size());
        checked(cudaMemcpy(readback.data(), dw.data, readback.size(), cudaMemcpyDeviceToHost));
        require(readback == weights, "Q8 kernel changed weights, stride padding or guards");
        readback.resize(input.size());
        checked(cudaMemcpy(readback.data(), dx.data, readback.size(), cudaMemcpyDeviceToHost));
        require(readback == input, "Q8 kernel changed input, stride padding or guards");
    };
    const int base_tile = s.n <= 8 ? 8 : s.n <= 16 ? 16 : 32;
    for (int revision = 0; revision < 2; ++revision) {
        for (int row = 0; row < s.m; ++row) for (int block = 0; block < s.k / 32; ++block) {
            unsigned char * dst = weights.data() + guard + std::size_t(row) * ws + std::size_t(block) * 34;
            const std::uint16_t bits = scale_bits(s, row, block, revision);
            std::memcpy(dst, &bits, 2);
            for (int j = 0; j < 32; ++j) {
                const std::int8_t q = static_cast<std::int8_t>(quant(s, row, block * 32 + j, revision));
                std::memcpy(dst + 2 + j, &q, 1);
            }
        }
        for (int col = 0; col < s.n; ++col) for (int k = 0; k < s.k; ++k) {
            const float x = activation(s, col, k, revision);
            std::memcpy(input.data() + guard + std::size_t(col) * cs + std::size_t(k) * xs, &x, 4);
        }
        checked(cudaMemcpyAsync(dw.data, weights.data(), weights.size(), cudaMemcpyHostToDevice, stream));
        checked(cudaMemcpyAsync(dx.data, input.data(), input.size(), cudaMemcpyHostToDevice, stream));
        std::fill(actual.begin(), actual.end(), output_guard);
        checked(cudaMemcpyAsync(dy.data, actual.data(), actual.size() * 4, cudaMemcpyHostToDevice, stream));
        // The b-shape rejection screen compares the one fixed forced candidate
        // against ACTUAL production dispatch; no forced-tile baseline sweep.
        launch(kda_b_probe ? TsgQ8HcTestRoute::production : TsgQ8HcTestRoute::control,
            kda_b_probe ? 0 : base_tile);
        download(); reference = actual;
        for (std::size_t i = 0; i < reference.size(); ++i) {
            require(std::isfinite(reference[i]), "Original control produced nonfinite output");
            if (i < 16 || i >= count + 16) require(reference[i] == output_guard, "Original control overwrote output guard");
        }
        if (revision == 0) {
            const unsigned before = tsg_matmul_q8_hc_cuda_test_reported();
            const unsigned kda_before = kda_mode ? tsg_matmul_q8_kda_cuda_test_reported() : 0;
            require(tsg_matmul_q8_hc_cuda_test_launch(TsgQ8HcTestRoute::compact_tile, wp, xp, yp,
                s.k, s.m, s.n, ws, xs, cs, 8, stream) == int(cudaErrorInvalidValue),
                "Unsupported compact tile must refuse before launch");
            require(tsg_matmul_q8_hc_cuda_test_launch(TsgQ8HcTestRoute::production, nullptr, xp, yp,
                s.k, s.m, s.n, ws, xs, cs, 0, stream) == int(cudaErrorInvalidValue),
                "Null source must refuse before production launch");
            download();
            require(std::memcmp(actual.data(), reference.data(), actual.size() * 4) == 0
                && tsg_matmul_q8_hc_cuda_test_reported() == before,
                "Refused launch changed output or production activity");
            if (kda_mode) require(tsg_matmul_q8_kda_cuda_test_reported() == kda_before,
                "Refused launch changed KDA production activity");
        }
        auto compare_submitted = [&](auto && submit) {
            poison = reference;
            for (std::size_t i = 16; i < count + 16; ++i) {
                std::uint32_t bits; std::memcpy(&bits, &reference[i], 4); bits = ~bits;
                std::memcpy(&poison[i], &bits, 4);
            }
            checked(cudaMemcpyAsync(dy.data, poison.data(), poison.size() * 4, cudaMemcpyHostToDevice, stream));
            download();
            require(std::memcmp(actual.data(), poison.data(), actual.size() * 4) == 0,
                "Output poison was not observed on the actual device allocation");
            require(std::memcmp(actual.data() + 16, reference.data() + 16, count * 4) != 0,
                "Output poison must differ from every expected output vector");
            submit(); download();
            require(std::memcmp(actual.data(), reference.data(), actual.size() * 4) == 0,
                "Candidate/production differs from the original K-ordered full output bits or guards");
            compared_values += count;
            sources_unchanged();
        };
        auto compare = [&](TsgQ8HcTestRoute route, int tile) {
            compare_submitted([&] { launch(route, tile); });
        };
        const unsigned forced_before = tsg_matmul_q8_hc_cuda_test_reported();
        const unsigned kda_forced_before = kda_mode ? tsg_matmul_q8_kda_cuda_test_reported() : 0;
        if (s.n == 1) compare(TsgQ8HcTestRoute::row_warp, 0);
        else if (kda_b_probe) compare(TsgQ8HcTestRoute::compact_tile, 16);
        else for (int tile : {16, 32, 64, 128}) {
            // Existing tile arithmetic itself must also agree across columns.
            compare(TsgQ8HcTestRoute::control, tile);
            compare(TsgQ8HcTestRoute::compact_tile, tile);
        }
        require(tsg_matmul_q8_hc_cuda_test_reported() == forced_before,
            "Forced test routes must not report production dispatch activity");
        if (kda_mode) require(tsg_matmul_q8_kda_cuda_test_reported() == kda_forced_before,
            "Forced test routes must not report KDA production activity");
        const unsigned production_before = tsg_matmul_q8_hc_cuda_test_reported();
        compare(TsgQ8HcTestRoute::production, 0);
        if (kda_mode || !enabled || !tsg_q8_hc_target_shape(s.k, s.m, s.n))
            require(tsg_matmul_q8_hc_cuda_test_reported() == production_before,
                "OFF/non-target production changed the compact enqueue mask");
        if (kda_mode) {
            const unsigned expected_bit = kda_variant == 0 ? 0u
                : 1u << ((s.m == 128 ? 2 : 0) + (kda_variant == 2 ? 1 : 0));
            require(tsg_matmul_q8_kda_cuda_test_reported() == (kda_forced_before | expected_bit),
                "Actual KDA production enqueue did not match the exact shape/mode");
        }
        if (capture_case) {
            // The synchronized production comparison above warms this exact
            // route before capture. Source uploads, output poisoning, the old
            // control and all comparisons stay OUTSIDE the captured graph.
            if (revision == 0) {
                checked(cudaStreamSynchronize(stream));
                capture_scope capture(stream);
                const int submitted = tsg_matmul_q8_hc_cuda_test_launch(TsgQ8HcTestRoute::production,
                    wp, xp, yp, s.k, s.m, s.n, ws, xs, cs, 0, stream);
                // End a begun capture even when the raw producer refuses.
                const auto ended = capture.end(&captured.graph);
                checked(static_cast<cudaError_t>(submitted)); checked(ended);
                require(captured.graph != nullptr, "Capture did not produce a graph");
                std::size_t nodes = 0;
                checked(cudaGraphGetNodes(captured.graph, nullptr, &nodes));
                require(nodes == 1, "Producer-only capture must contain exactly one kernel node");
                cudaGraphNode_t node = nullptr;
                checked(cudaGraphGetNodes(captured.graph, &node, &nodes));
                cudaGraphNodeType type;
                checked(cudaGraphNodeGetType(node, &type));
                require(nodes == 1 && type == cudaGraphNodeTypeKernel,
                    "Empty/non-kernel graph cannot qualify capture correctness");
                checked(cudaGraphInstantiate(&captured.executable, captured.graph, 0));
                first_executable = captured.executable;
                graph_first_output = reference;
            }
            else require(graph_first_output.size() == reference.size()
                && std::memcmp(graph_first_output.data(), reference.data(), reference.size() * 4) != 0,
                "Changed source revision must change the fresh original-control output");
            require(captured.executable != nullptr && captured.executable == first_executable,
                "Both source revisions must replay the SAME graph executable");
            compare_submitted([&] { checked(cudaGraphLaunch(captured.executable, stream)); });
            ++graph_replays;
        }

        // Sample independent logical row/column coordinates, including all
        // boundaries. Sequential-FMA rounding and underflow bound, not loose
        // model tolerance; the strict same-device byte test above is primary.
        const int nr = std::min(s.m, 5), nc = std::min(s.n, 5);
        for (int ri = 0; ri < nr; ++ri) for (int ci = 0; ci < nc; ++ci) {
            const int row = nr == 1 ? 0 : int(std::int64_t(ri) * (s.m - 1) / (nr - 1));
            const int col = nc == 1 ? 0 : int(std::int64_t(ci) * (s.n - 1) / (nc - 1));
            double dot = 0, absolute = 0;
            for (int k = 0; k < s.k; ++k) {
                const double w = half_value(scale_bits(s, row, k / 32, revision)) * quant(s, row, k, revision);
                const double product = w * double(activation(s, col, k, revision));
                dot += product; absolute += std::abs(product);
            }
            const double ku = s.k * 0x1p-24, du = s.k * 0x1p-53;
            const double bound = (ku / (1 - ku) + du / (1 - du)) * absolute
                + s.k * std::ldexp(1.0, -150) / (1 - ku);
            require(std::abs(double(reference[16 + std::size_t(col) * s.m + row]) - dot) <= bound,
                "Independent FP64 logical dot exceeded sequential-FMA/underflow bound");
        }
    }
    ++cases;
    if (kda_b_probe) {
        std::printf("{\"check\":true,\"probe\":\"kda_b_fixed_forced_kernel\",\"k\":%d,\"m\":%d,\"n\":%d,\"layout\":%d,\"revisions\":2,\"all_output_bits_equal\":true,\"reference\":\"production\",\"candidate\":\"%s\",\"hc_production_mask\":%u,\"kda_production_mask\":%u,\"scope\":\"synthetic_exact_arithmetic_not_model_speed\"}\n",
            s.k, s.m, s.n, s.layout, s.n == 1 ? "forced_row_warp" : "forced_rows32_columns16",
            tsg_matmul_q8_hc_cuda_test_reported(), tsg_matmul_q8_kda_cuda_test_reported());
    } else if (kda_mode) {
        std::printf("{\"check\":true,\"profile\":\"kda\",\"k\":%d,\"m\":%d,\"n\":%d,\"layout\":%d,\"pattern\":%d,\"revisions\":2,\"all_output_bits_equal\":true,\"production_mask\":%u,\"hc_production_mask\":%u}\n",
            s.k, s.m, s.n, s.layout, int(s.data), tsg_matmul_q8_kda_cuda_test_reported(),
            tsg_matmul_q8_hc_cuda_test_reported());
    } else {
        std::printf("{\"check\":true,\"k\":%d,\"m\":%d,\"n\":%d,\"layout\":%d,\"pattern\":%d,\"revisions\":2,\"all_output_bits_equal\":true,\"production_mask\":%u}\n",
            s.k, s.m, s.n, s.layout, int(s.data), tsg_matmul_q8_hc_cuda_test_reported());
    }
    const bool micro_target = (s.k == 16384 && s.m == 24)
        || (kda_probe && s.k == 4096 && (s.m == 64 || s.m == 128))
        || (kda_b_probe && s.k == 128 && s.m == 8192);
    if (benchmark && micro_target && s.layout == 0 && s.data == pattern::ordinary) {
        events timing;
        std::vector<std::pair<TsgQ8HcTestRoute, int>> routes;
        // --kda-probe keeps production KDA disabled. Compare production for
        // exactness above, but do not time that unchanged route as a candidate.
        if (kda_b_probe) routes.push_back({s.n == 1 ? TsgQ8HcTestRoute::row_warp
                                                   : TsgQ8HcTestRoute::compact_tile,
                                         s.n == 1 ? 0 : 16});
        else {
            if (!kda_probe) routes.push_back({TsgQ8HcTestRoute::production, 0});
            if (s.n == 1) routes.push_back({TsgQ8HcTestRoute::row_warp, 0});
            else for (int tile : {16, 32, 64, 128}) routes.push_back({TsgQ8HcTestRoute::compact_tile, tile});
        }
        for (const auto & candidate : routes) {
            for (int pass = 0; pass < 4; ++pass) {
                const bool alternate = pass == 1 || pass == 2;
                const auto selected = alternate ? candidate.first
                    : kda_b_probe ? TsgQ8HcTestRoute::production : TsgQ8HcTestRoute::control;
                const int tile = alternate ? candidate.second : kda_b_probe ? 0 : base_tile;
                for (int i = 0; i < 3; ++i) launch(selected, tile);
                checked(cudaEventRecord(timing.first, stream));
                for (int i = 0; i < 8; ++i) launch(selected, tile);
                checked(cudaEventRecord(timing.last, stream)); checked(cudaEventSynchronize(timing.last));
                float ms = 0; checked(cudaEventElapsedTime(&ms, timing.first, timing.last));
                if (kda_b_probe) require(std::isfinite(ms) && ms > 0,
                    "KDA-b event interval must be finite and positive");
                const int control_tile = s.n == 1 ? 1 : base_tile;
                const int actual_tile = selected == TsgQ8HcTestRoute::production ? -1
                    : selected == TsgQ8HcTestRoute::control ? control_tile
                    : selected == TsgQ8HcTestRoute::row_warp ? 1 : tile;
                if (kda_b_probe) {
                    std::printf("{\"microtiming\":true,\"probe\":\"kda_b_fixed_forced_kernel\",\"scope\":\"synthetic_same_stream_not_model_speed\",\"k\":%d,\"m\":%d,\"n\":%d,\"comparison_route\":%d,\"forced_tile\":%d,\"baseline_route\":%d,\"baseline_kernel\":\"%s\",\"baseline_tile\":%d,\"baseline_tile_policy\":\"%s\",\"pass\":%d,\"actual_route\":%d,\"actual_tile\":%d,\"repeats\":8,\"stream_us\":%.6f,\"production_candidate_dispatch\":false,\"graph_qualified\":false}\n",
                        s.k, s.m, s.n, int(candidate.first), candidate.second,
                        int(TsgQ8HcTestRoute::production),
                        s.n == 1 ? "production_legacy_vector" : "production_legacy_auto_rows64",
                        s.n == 1 ? 1 : -1, s.n == 1 ? "known_vector" : "production_auto_unknown", pass,
                        int(selected), selected == TsgQ8HcTestRoute::production ? (s.n == 1 ? 1 : -1) : actual_tile,
                        double(ms) * 1000 / 8);
                } else if (kda_probe) {
                    std::printf("{\"microtiming\":true,\"probe\":\"kda_existing_forced_kernels\",\"scope\":\"synthetic_same_stream_not_model_speed\",\"k\":%d,\"m\":%d,\"n\":%d,\"comparison_route\":%d,\"forced_tile\":%d,\"control_kernel\":\"%s\",\"candidate_kernel\":\"%s\",\"control_tile\":%d,\"pass\":%d,\"actual_route\":%d,\"actual_tile\":%d,\"repeats\":8,\"stream_us\":%.6f,\"production_candidate_dispatch\":false,\"graph_qualified\":false}\n",
                        s.k, s.m, s.n, int(candidate.first), candidate.second,
                        s.n == 1 ? "legacy_vector" : "legacy_rows64",
                        s.n == 1 ? "forced_row_warp" : "forced_rows32", control_tile, pass,
                        int(selected), actual_tile, double(ms) * 1000 / 8);
                } else {
                    std::printf("{\"microtiming\":true,\"scope\":\"synthetic_same_stream_not_model_speed\",\"k\":%d,\"m\":%d,\"n\":%d,\"comparison_route\":%d,\"forced_tile\":%d,\"control_tile\":%d,\"pass\":%d,\"actual_route\":%d,\"actual_tile\":%d,\"actual_tile_policy\":\"%s\",\"repeats\":8,\"stream_us\":%.6f}\n",
                        s.k, s.m, s.n, int(candidate.first), candidate.second, control_tile, pass,
                        int(selected), actual_tile, actual_tile < 0 ? "production_auto_unknown" : "explicit",
                        double(ms) * 1000 / 8);
                }
            }
        }
        timing.close();
    }
    checked(cudaStreamSynchronize(stream));
    sources_unchanged();
    captured.close();
    if (capture_case) {
        ++graph_cases;
        if (kda_mode) {
            std::printf("{\"graph_check\":true,\"profile\":\"kda\",\"mode\":\"%s\",\"k\":%d,\"m\":%d,\"n\":%d,\"kernel_nodes\":1,\"same_exec_source_revisions\":2,\"all_output_bits_equal\":true,\"graph_exec_closed\":true,\"scope\":\"synthetic_producer_capture_not_model_capture\"}\n",
                enabled ? "on" : "off", s.k, s.m, s.n);
        } else {
            std::printf("{\"graph_check\":true,\"mode\":\"%s\",\"k\":%d,\"m\":%d,\"n\":%d,\"kernel_nodes\":1,\"same_exec_source_revisions\":2,\"all_output_bits_equal\":true,\"graph_exec_closed\":true,\"scope\":\"synthetic_producer_capture_not_model_capture\"}\n",
                enabled ? "on" : "off", s.k, s.m, s.n);
        }
    }
    dy.close(); dx.close(); dw.close();
}
} // namespace

int main(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        require(argc == 2 || argc == 3, "Use --off/--on with optional --benchmark, --kda-probe, --kda-b-probe, or --kda-off/--kda-on");
        const bool kda_probe = std::string(argv[1]) == "--kda-probe";
        const bool kda_b_probe = std::string(argv[1]) == "--kda-b-probe";
        const bool kda_mode = std::string(argv[1]) == "--kda-off" || std::string(argv[1]) == "--kda-on";
        const bool hc_enabled = std::string(argv[1]) == "--on";
        const bool kda_enabled = std::string(argv[1]) == "--kda-on";
        const bool enabled = hc_enabled || kda_enabled;
        require(hc_enabled || std::string(argv[1]) == "--off" || kda_probe || kda_b_probe || kda_mode, "Unknown fixture mode");
        if (kda_probe) require(argc == 2, "--kda-probe is a complete bounded micro mode");
        if (kda_b_probe) require(argc == 2, "--kda-b-probe is a fixed four-shape rejection screen");
        if (kda_mode) require(argc == 2, "KDA OFF/ON are correctness/capture modes without microtiming");
        const bool benchmark = kda_probe || argc == 3;
        require(argc != 3 || std::string(argv[2]) == "--benchmark", "Unknown optional argument");
        environment("TS_GGML_Q8_HC_COMPACT", hc_enabled ? "1" : "0");
        environment("TS_GGML_Q8_KDA_COMPACT", kda_enabled ? "1" : "0");
        for (const char * name : {"TS_GGML_Q8_PARALLEL_VECTOR", "TS_GGML_Q8_PARALLEL_SMALL_BATCH",
             "TS_GGML_Q8_ORDERED_SMALL_BATCH", "TS_GGML_Q8_ORDERED_GATE_PREFILL", "TS_GGML_Q8_GATE_PREFILL_TILE16"})
            environment(name, "0");
        environment("TS_GGML_Q8_PREFILL_TILE", "auto");
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::fprintf(stderr, "SKIP: CUDA unavailable; no Q8 HC correctness credit\n"); return 77;
        }
        checked(cudaSetDevice(0));
        require(tsg_matmul_q8_hc_cuda_test_reported() == 0, "Fixture must begin in a fresh process");
        if (kda_mode) require(tsg_matmul_q8_kda_cuda_test_reported() == 0,
            "KDA fixture must begin with no successful production enqueues");
        stream_owner stream;
        if (kda_b_probe) {
            require(tsg_matmul_q8_kda_cuda_test_reported() == 0,
                "KDA-b probe must begin in a fresh process with both compact modes OFF");
            for (int n : {1, 33, 336, 512})
                run({128, 8192, n, 0}, false, true, stream.value, false, false, true);
            checked(cudaStreamSynchronize(stream.value)); stream.close();
            require(cases == 4 && graph_cases == 0 && graph_replays == 0,
                "KDA-b probe must complete exactly four non-capture shapes");
            require(compared_values == std::size_t(4) * 8192 * (1 + 33 + 336 + 512),
                "KDA-b probe must compare candidate and production at both source revisions");
            require(tsg_matmul_q8_hc_cuda_test_reported() == 0
                && tsg_matmul_q8_kda_cuda_test_reported() == 0 && live_allocations == 0,
                "KDA-b probe changed production activity or retained CUDA storage");
            std::printf("PASS q8_kda_b_probe cases=%zu checks=%zu compared_f32=%zu source_revisions=2 micro_groups=4 micro_records=16 timed_enqueues=128 production_candidate_dispatch=0 graph_cases=0 graph_qualified=0 cuda_closed=1 scope=synthetic_existing_fixed_forced_kernel_not_model_speed\n",
                cases, checks, compared_values);
            return 0;
        }
        if (kda_probe || kda_mode) {
            // Metadata-bound K4096/M64 beta and M128 f_a/g_a only. All inputs
            // are generated; this mode never opens model/header/payload files.
            std::vector<shape> probe;
            for (int m : {64, 128}) {
                for (int n : {1, 33, 336, 512}) for (int layout : {0, 1, 2})
                    probe.push_back({4096, m, n, layout});
                for (pattern p : {pattern::cancellation, pattern::subnormal, pattern::signed_zero, pattern::half_extremes})
                    for (int n : {1, 33}) probe.push_back({4096, m, n, n == 1 ? 2 : 1, p});
                for (int n : {9, 13, 16, 17, 65, 129}) probe.push_back({4096, m, n, 1});
                for (int n = 2; n <= 8; ++n) probe.push_back({4096, m, n, 1});
            }
            if (kda_mode) {
                // Test non-targets before any positive mask bit exists: an
                // accidental KDA enqueue cannot hide behind an earlier hit.
                for (int m : {64, 128}) {
                    probe.push_back({4096, m, 513, 1});
                    probe.push_back({4064, m, 1, 1});
                    probe.push_back({4064, m, 33, 2});
                    probe.push_back({4128, m, 1, 0});
                    probe.push_back({4128, m, 33, 1});
                }
                for (int m : {63, 65, 127, 129}) for (int n : {1, 33})
                    probe.push_back({4096, m, n, n == 1 ? 0 : 1});
                probe.push_back({16384, 24, 1, 0});
                probe.push_back({16384, 24, 33, 1});
                std::stable_partition(probe.begin(), probe.end(), [](const shape & s) {
                    return !(s.k == 4096 && (s.m == 64 || s.m == 128)
                        && (s.n == 1 || (s.n >= 9 && s.n <= 512)));
                });
            }
            for (const auto & s : probe) run(s, enabled, kda_probe, stream.value, kda_probe, kda_mode);
            checked(cudaStreamSynchronize(stream.value)); stream.close();
            if (kda_mode) {
                const unsigned mask = tsg_matmul_q8_kda_cuda_test_reported();
                require(cases == 86 && graph_cases == 4 && graph_replays == 8,
                    "KDA mode must complete all86 cases and four producer captures/eight replays");
                require(mask == (enabled ? 15u : 0u), "KDA OFF/ON must prove both M64/M128 row and wide production paths");
                require(tsg_matmul_q8_hc_cuda_test_reported() == 0 && live_allocations == 0,
                    "KDA mode changed HC activity or retained CUDA storage");
                std::printf("PASS q8_kda_compact mode=%s cases=%zu checks=%zu compared_f32=%zu production_mask=%u graph_cases=%zu graph_replays=%zu cuda_closed=1 scope=synthetic_exact_arithmetic_not_model_capture_or_performance\n",
                    enabled ? "on" : "off", cases, checks, compared_values, mask, graph_cases, graph_replays);
                return 0;
            }
            require(cases == 66 && graph_cases == 0 && graph_replays == 0,
                "KDA probe must execute the full66-case non-capture matrix");
            require(tsg_matmul_q8_hc_cuda_test_reported() == 0 && live_allocations == 0,
                "KDA forced probe must not create HC production activity or retain CUDA storage");
            std::printf("PASS q8_kda_probe cases=%zu checks=%zu compared_f32=%zu production_candidate_dispatch=0 graph_cases=0 graph_qualified=0 cuda_closed=1 scope=synthetic_existing_forced_kernels_not_model_speed\n",
                cases, checks, compared_values);
            return 0;
        }
        std::vector<shape> matrix;
        for (int n : {1, 33}) for (int layout : {0, 1, 2}) matrix.push_back({16384, 24, n, layout});
        for (int n : {336, 512}) for (int layout : {0, 2}) matrix.push_back({16384, 24, n, layout});
        for (int n : {9, 13, 16, 65, 129}) matrix.push_back({16384, 24, n, 1});
        for (pattern p : {pattern::cancellation, pattern::subnormal, pattern::signed_zero, pattern::half_extremes})
            for (int n : {1, 33}) matrix.push_back({16384, 24, n, n == 1 ? 2 : 1, p});
        // Unchanged N2..8 and neighboring target shape boundaries.
        for (int n = 2; n <= 8; ++n) matrix.push_back({16384, 24, n, 1});
        matrix.insert(matrix.end(), {{16352, 24, 1, 1}, {16352, 24, 33, 2},
            {16384, 23, 1, 0}, {16384, 23, 33, 1}, {16384, 25, 33, 2}});
        // Forced candidate kernels support bounded diagnostic row/column tails.
        for (int m : {1, 31, 33, 65}) for (int n : {1, 17}) matrix.push_back({96, m, n, 2});
        matrix.push_back({160, 35, 129, 1, pattern::half_extremes});
        for (const auto & s : matrix) run(s, enabled, benchmark, stream.value);
        const unsigned mask = tsg_matmul_q8_hc_cuda_test_reported();
        require(enabled ? ((mask & 1u) && (mask & 0x1eu)) : mask == 0,
            "Actual synchronized production row/wide activity must match ON/OFF mode");
        require((mask & ~31u) == 0, "Unknown production activity bit");
        require(graph_cases == 2 && graph_replays == 4,
            "N1 and N13 must each replay one captured producer across both source revisions");
        checked(cudaStreamSynchronize(stream.value)); stream.close();
        require(live_allocations == 0, "Fixture retained device allocation");
        std::printf("PASS q8_hc_compact mode=%s cases=%zu checks=%zu compared_f32=%zu production_mask=%u graph_cases=%zu graph_replays=%zu cuda_closed=1 scope=synthetic_exact_arithmetic_not_model_capture_or_performance\n",
            enabled ? "on" : "off", cases, checks, compared_values, mask, graph_cases, graph_replays);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL q8_hc_compact: %s\n", error.what()); return 1;
    }
}
