// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Isolated pedantic F32 algorithm screen; no production dispatch is changed.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t s) { if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
void check(cublasStatus_t s) { if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS status " + std::to_string(s)); }
template<class T> struct buffer {
    T * p = nullptr;
    explicit buffer(size_t n) { check(cudaMalloc(reinterpret_cast<void **>(&p), n * sizeof(T))); }
    ~buffer() { if (p) cudaFree(p); }
    buffer(const buffer &) = delete;
};
struct stream {
    cudaStream_t value = nullptr;
    stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~stream() { if (value) { cudaStreamSynchronize(value); cudaStreamDestroy(value); } }
    stream(const stream &) = delete;
};
struct handles {
    cublasHandle_t classic = nullptr;
    cublasLtHandle_t lt = nullptr;
    explicit handles(cudaStream_t s) {
        try {
            check(cublasCreate(&classic)); check(cublasSetStream(classic, s));
            check(cublasSetMathMode(classic, static_cast<cublasMath_t>(
                CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
            check(cublasLtCreate(&lt));
        } catch (...) { if (classic) cublasDestroy(classic); throw; }
    }
    ~handles() { if (lt) cublasLtDestroy(lt); if (classic) cublasDestroy(classic); }
    handles(const handles &) = delete;
};
struct descriptors {
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    void clear() {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (operation) cublasLtMatmulDescDestroy(operation);
    }
    descriptors(int m, int n, int k, uint64_t workspace) {
        try {
            check(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F_PEDANTIC, CUDA_R_32F));
            const cublasOperation_t transpose = CUBLAS_OP_T;
            check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));
            check(cublasLtMatrixLayoutCreate(&a, CUDA_R_32F, k, m, k));
            check(cublasLtMatrixLayoutCreate(&b, CUDA_R_32F, k, n, k));
            check(cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, m, n, m));
            check(cublasLtMatmulPreferenceCreate(&preference));
            check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &workspace, sizeof(workspace)));
            // cudaMalloc bases are aligned, but the output follows 16 F32 canaries.
            const uint32_t input_alignment = 256, output_alignment = 64;
            check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                &input_alignment, sizeof(input_alignment)));
            check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                &input_alignment, sizeof(input_alignment)));
            check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                &output_alignment, sizeof(output_alignment)));
            check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
                &output_alignment, sizeof(output_alignment)));
        } catch (...) { clear(); throw; }
    }
    ~descriptors() { clear(); }
    descriptors(const descriptors &) = delete;
};
struct shape { int n, m, k; const char * kind; };
struct sample { size_t index; double expected; };
struct quality {
    bool finite = true, canaries = true, inputs_unchanged = true;
    double max_error = 0, max_error_over_bound = 0, max_baseline_error = 0, max_baseline_error_over_bound = 0;
    bool passed() const { return finite && canaries && inputs_unchanged && max_error_over_bound <= 1 && max_baseline_error_over_bound <= 1; }
};
int attribute(const cublasLtMatmulAlgo_t & algo, cublasLtMatmulAlgoConfigAttributes_t key) {
    int value = -1; size_t written = 0;
    check(cublasLtMatmulAlgoConfigGetAttribute(&algo, key, &value, sizeof(value), &written));
    if (written != sizeof(value)) throw std::runtime_error("Unexpected algorithm attribute size");
    return value;
}
void write_quality(std::ostream & out, const quality & q) {
    out << "{\"finite\":" << (q.finite ? "true" : "false") << ",\"canaries\":" << (q.canaries ? "true" : "false")
        << ",\"inputs_unchanged\":" << (q.inputs_unchanged ? "true" : "false") << ",\"max_double_error\":" << q.max_error
        << ",\"max_error_over_bound\":" << q.max_error_over_bound << ",\"max_baseline_error\":" << q.max_baseline_error
        << ",\"max_baseline_error_over_bound\":" << q.max_baseline_error_over_bound
        << ",\"passed\":" << (q.passed() ? "true" : "false") << '}';
}
} // namespace

int main(int argc, char ** argv) try {
    if (argc != 3 || std::string(argv[1]) != "--output") throw std::runtime_error("Usage: probe --output report.json");
    std::ofstream out(argv[2]); if (!out) throw std::runtime_error("Cannot create report");
    check(cudaSetDevice(0)); cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop, 0));
    stream execution; handles api(execution.value);
    constexpr uint64_t workspace_bytes = 32ull * 1024 * 1024;
    buffer<unsigned char> workspace(workspace_bytes);
    int version = 0; check(cublasGetVersion(api.classic, &version));
    out << std::setprecision(12) << "{\"device\":\"" << prop.name << "\",\"cublas_version\":" << version
        << ",\"precision\":\"CUBLAS_COMPUTE_32F_PEDANTIC\",\"workspace_limit_bytes\":" << workspace_bytes
        << ",\"warmup\":3,\"minimum_repeats\":20,\"minimum_seconds\":0.1,\"records\":[";
    bool first_record = true;
    size_t phases_passed = 0, phases_failed = 0, candidate_phases = 0, unavailable_algorithms = 0;
    for (const shape s : {shape{36,768,768,"normalized"}, shape{2304,768,768,"normalized"},
            shape{2340,3072,768,"normalized"}, shape{1089,768,3072,"normalized"},
            shape{129,64,64,"unrounded"}, shape{33,64,64,"large_cancellation"}}) {
        const size_t wc = size_t(s.k) * s.m, xc = size_t(s.k) * s.n, yc = size_t(s.m) * s.n;
        std::vector<float> w(wc), x(xc), reference(yc), result(yc + 32, -1234567.25f), read_w(wc), read_x(xc);
        for (size_t i = 0; i < wc; ++i) w[i] = float(std::sin(double(i * 17 + 3) * .013) * .21 + std::cos(double(i + 11) * .003) * .007);
        for (size_t i = 0; i < xc; ++i) x[i] = float(std::sin(double(i * 13 + 7) * .017) * .83 + std::cos(double(i + 5) * .009) * .011);
        if (std::string(s.kind) == "unrounded") {
            for (size_t i = 0; i < wc; ++i) w[i] = 1.f + float(i % 31) * .0000013f;
            for (size_t i = 0; i < xc; ++i) x[i] = ((i % s.k) % 2 ? -1.f : 1.f) + float(i % 29) * .0000021f;
        }
        if (std::string(s.kind) == "large_cancellation") {
            for (size_t i = 0; i < wc; ++i) w[i] = 80000.f + float(i % 31) * .015625f;
            for (size_t i = 0; i < xc; ++i) x[i] = ((i % s.k) % 2 ? -80000.f : 80000.f) + float(i % 23) * .015625f;
        }
        buffer<float> dw(wc), dx(xc), dy(yc + 32);
        check(cudaMemcpyAsync(dw.p, w.data(), wc * sizeof(float), cudaMemcpyHostToDevice, execution.value));
        check(cudaMemcpyAsync(dx.p, x.data(), xc * sizeof(float), cudaMemcpyHostToDevice, execution.value));
        check(cudaStreamSynchronize(execution.value));
        descriptors desc(s.m, s.n, s.k, workspace_bytes);
        cublasLtMatmulHeuristicResult_t algorithms[16]{}; int returned = 0;
        const auto heuristic_status = cublasLtMatmulAlgoGetHeuristic(api.lt, desc.operation, desc.a, desc.b,
            desc.c, desc.c, desc.preference, 16, algorithms, &returned);
        if (heuristic_status != CUBLAS_STATUS_SUCCESS && heuristic_status != CUBLAS_STATUS_NOT_SUPPORTED) check(heuristic_status);
        const float one = 1, zero = 0;
        auto submit = [&](int algorithm) {
            if (algorithm < 0) return cublasGemmStridedBatchedEx(api.classic, CUBLAS_OP_T, CUBLAS_OP_N,
                s.m, s.n, s.k, &one, dw.p, CUDA_R_32F, s.k, 0, dx.p, CUDA_R_32F, s.k, xc,
                &zero, dy.p + 16, CUDA_R_32F, s.m, yc, 1, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT);
            return cublasLtMatmul(api.lt, desc.operation, &one, dw.p, desc.a, dx.p, desc.b, &zero,
                dy.p + 16, desc.c, dy.p + 16, desc.c, &algorithms[algorithm].algo,
                workspace.p, workspace_bytes, execution.value);
        };
        auto run = [&](int algorithm) {
            check(submit(algorithm));
            check(cudaStreamSynchronize(execution.value));
        };
        check(cudaMemcpyAsync(dy.p, result.data(), result.size() * sizeof(float), cudaMemcpyHostToDevice, execution.value));
        run(-1); check(cudaMemcpy(reference.data(), dy.p + 16, yc * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<sample> independent;
        const size_t samples = std::min(yc, size_t(4096));
        for (size_t j = 0; j < samples; ++j) {
            const size_t i = samples == 1 ? 0 : j * (yc - 1) / (samples - 1), row = i % s.m, token = i / s.m;
            double value = 0;
            for (int k = 0; k < s.k; ++k) value += double(w[row * s.k + k]) * x[token * s.k + k];
            independent.push_back({i, value});
        }
        auto evaluate = [&] {
            quality q;
            check(cudaMemcpy(result.data(), dy.p, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
            check(cudaMemcpy(read_w.data(), dw.p, wc * sizeof(float), cudaMemcpyDeviceToHost));
            check(cudaMemcpy(read_x.data(), dx.p, xc * sizeof(float), cudaMemcpyDeviceToHost));
            q.inputs_unchanged = std::memcmp(w.data(), read_w.data(), wc * sizeof(float)) == 0 &&
                std::memcmp(x.data(), read_x.data(), xc * sizeof(float)) == 0;
            for (size_t i = 0; i < yc; ++i) {
                q.finite &= std::isfinite(result[i + 16]);
                const double error = std::abs(double(result[i + 16]) - reference[i]);
                q.max_baseline_error = std::max(q.max_baseline_error, error);
                q.max_baseline_error_over_bound = std::max(q.max_baseline_error_over_bound,
                    error / (2e-6 + 2e-6 * std::abs(double(reference[i]))));
            }
            for (int i = 0; i < 16; ++i) q.canaries &= result[i] == -1234567.25f && result[yc + 16 + i] == -1234567.25f;
            for (const auto & item : independent) {
                const double error = std::abs(double(result[item.index + 16]) - item.expected);
                q.max_error = std::max(q.max_error, error);
                q.max_error_over_bound = std::max(q.max_error_over_bound, error / (2e-6 + 2e-6 * std::abs(item.expected)));
            }
            return q;
        };
        // Include one baseline-only ABBA when no Lt algorithm is available.
        for (int candidate = 0; candidate < std::max(1, returned); ++candidate) {
            bool available = candidate < returned && algorithms[candidate].state == CUBLAS_STATUS_SUCCESS &&
                algorithms[candidate].workspaceSize <= workspace_bytes;
            auto execution_status = CUBLAS_STATUS_NOT_SUPPORTED;
            if (available) {
                execution_status = submit(candidate);
                if (execution_status == CUBLAS_STATUS_NOT_SUPPORTED) available = false;
                else check(execution_status);
                check(cudaStreamSynchronize(execution.value));
            }
            if (!available) ++unavailable_algorithms;
            for (int phase = 0; phase < (available ? 4 : 1); ++phase) {
                const int executed = available && (phase == 1 || phase == 2) ? candidate : -1;
                std::fill(result.begin(), result.end(), -1234567.25f);
                check(cudaMemcpyAsync(dy.p, result.data(), result.size() * sizeof(float), cudaMemcpyHostToDevice, execution.value));
                for (int warmup = 0; warmup < 3; ++warmup) run(executed);
                const auto warm_quality = evaluate();
                std::vector<double> times;
                const auto start = std::chrono::steady_clock::now();
                do {
                    const auto before = std::chrono::steady_clock::now(); run(executed);
                    times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - before).count());
                } while (times.size() < 20 || std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < .1);
                const auto final_quality = evaluate(); std::sort(times.begin(), times.end());
                if (warm_quality.passed() && final_quality.passed()) ++phases_passed; else ++phases_failed;
                if (executed >= 0) ++candidate_phases;
                const auto count = times.size();
                const double median = count % 2 ? times[count / 2] : (times[count / 2 - 1] + times[count / 2]) * .5;
                if (!first_record) out << ','; first_record = false;
                out << "{\"n\":" << s.n << ",\"m\":" << s.m << ",\"k\":" << s.k << ",\"kind\":\"" << s.kind
                    << "\",\"candidate_index\":" << candidate << ",\"available\":" << (available ? "true" : "false")
                    << ",\"heuristic_status\":" << int(heuristic_status) << ",\"returned_algorithms\":" << returned
                    << ",\"execution_preflight_status\":" << int(execution_status)
                    << ",\"algorithm_id\":" << (available ? attribute(algorithms[candidate].algo, CUBLASLT_ALGO_CONFIG_ID) : -1)
                    << ",\"split_k\":" << (available ? attribute(algorithms[candidate].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM) : -1)
                    << ",\"workspace_bytes\":" << (available ? algorithms[candidate].workspaceSize : 0)
                    << ",\"phase\":" << phase << ",\"executed\":\"" << (executed < 0 ? "classic_pedantic" : "lt_pedantic")
                    << "\",\"repeats\":" << count << ",\"median_ms\":" << median << ",\"warm_quality\":";
                write_quality(out, warm_quality); out << ",\"final_quality\":"; write_quality(out, final_quality);
                out << '}'; out.flush();
            }
        }
    }
    out << "],\"completed\":true,\"exit_code_means_execution_only\":true,\"phases_passed\":" << phases_passed
        << ",\"phases_failed\":" << phases_failed << ",\"candidate_phases_executed\":" << candidate_phases
        << ",\"unavailable_algorithms\":" << unavailable_algorithms
        << ",\"quality_all_executed_phases_passed\":" << (phases_failed == 0 && candidate_phases > 0 ? "true" : "false")
        << ",\"model_qualified\":false,\"production_promoted\":false,\"limitations\":["
        << "\"Synthetic inputs at actual projection dimensions; not a model oracle or HTTP benchmark.\","
        << "\"Double checks sample at most 4096 positions; complete outputs compare with the classic baseline.\","
        << "\"Unavailable algorithms and failed quality phases are retained and are not passing coverage.\","
        << "\"Timing includes private-stream submission/completion and excludes input transfers, allocation and heuristic setup.\"]}\n";
    return 0;
} catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
