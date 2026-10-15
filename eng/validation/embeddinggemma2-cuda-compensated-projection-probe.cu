// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Standalone diagnostic. No TensorSharp/ggml production dispatch is changed.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void check(cudaError_t s) { if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
void check(cublasStatus_t s) { if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS status " + std::to_string(s)); }
template<class T> struct buffer {
    T * p = nullptr;
    explicit buffer(size_t n) { check(cudaMalloc(reinterpret_cast<void **>(&p), n * sizeof(T))); }
    ~buffer() { cudaFree(p); }
    buffer(const buffer &) = delete;
};
struct private_handle {
    cublasHandle_t value = nullptr;
    private_handle() { check(cublasCreate(&value)); }
    ~private_handle() { if (value) cublasDestroy(value); }
    private_handle(const private_handle &) = delete;
};

__global__ void split(const float * x, __nv_bfloat16 * hi, __nv_bfloat16 * mid,
                      __nv_bfloat16 * lo, size_t n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(blockDim.x) * gridDim.x) {
        hi[i] = __float2bfloat16_rn(x[i]);
        const float r = x[i] - __bfloat162float(hi[i]);
        mid[i] = __float2bfloat16_rn(r);
        lo[i] = __float2bfloat16_rn(r - __bfloat162float(mid[i]));
    }
}
void pack(const float * x, __nv_bfloat16 * hi, __nv_bfloat16 * mid, __nv_bfloat16 * lo, size_t n) {
    split<<<std::min(size_t(4096), (n + 255) / 256), 256>>>(x, hi, mid, lo, n);
    check(cudaGetLastError());
}
uint64_t fingerprint(const std::vector<float> & v) {
    uint64_t h = 1469598103934665603ull;
    for (float f : v) { uint32_t bits; std::memcpy(&bits, &f, 4); h = (h ^ bits) * 1099511628211ull; }
    return h;
}
struct shape { int n, m, k; const char * kind; };

int main() try {
    int device = 0; check(cudaSetDevice(device));
    cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop, device));
    if (prop.major < 8) throw std::runtime_error("Required BF16 CUDA device is unavailable");
    private_handle owner;
    const auto handle = owner.value;
    check(cublasSetMathMode(handle, static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    std::cout << std::setprecision(12) << "{\"device\":\"" << prop.name
              << "\",\"warmup\":3,\"minimum_repeats\":20,\"minimum_seconds\":0.25,\"records\":[";
    bool first = true;
    for (const shape s : {shape{36,768,768,"normalized"}, shape{2304,768,768,"normalized"},
            shape{2340,3072,768,"normalized"}, shape{1089,768,3072,"normalized"},
            shape{129,64,64,"unrounded"}, shape{33,64,64,"large_cancellation"}}) {
        const size_t wc = size_t(s.k) * s.m, xc = size_t(s.k) * s.n, yc = size_t(s.m) * s.n;
        std::vector<float> w(wc), x(xc), reference(yc), result(yc + 32, -1234567.25f);
        for (size_t i = 0; i < wc; ++i) w[i] = float(std::sin(double(i * 17 + 3) * .013) * .21 + std::cos(double(i + 11) * .003) * .007);
        for (size_t i = 0; i < xc; ++i) x[i] = float(std::sin(double(i * 13 + 7) * .017) * .83 + std::cos(double(i + 5) * .009) * .011);
        if (std::string(s.kind) == "unrounded") {
            for (size_t i = 0; i < wc; ++i) w[i] = 1.f + float(i % 31) * 0.0000013f;
            for (size_t i = 0; i < xc; ++i) x[i] = ((i % s.k) % 2 ? -1.f : 1.f) + float(i % 29) * .0000021f;
        }
        if (std::string(s.kind) == "large_cancellation") {
            for (size_t i = 0; i < wc; ++i) w[i] = 80000.f + float(i % 31) * .015625f;
            for (size_t i = 0; i < xc; ++i) x[i] = ((i % s.k) % 2 ? -80000.f : 80000.f) + float(i % 23) * .015625f;
        }
        const uint64_t wh = fingerprint(w), xh = fingerprint(x);
        buffer<float> dw(wc), dx(xc), dy(yc + 32);
        buffer<__nv_bfloat16> whi(wc), wmid(wc), wlo(wc), xhi(xc), xmid(xc), xlo(xc);
        check(cudaMemcpy(dw.p, w.data(), wc * 4, cudaMemcpyHostToDevice));
        check(cudaMemcpy(dx.p, x.data(), xc * 4, cudaMemcpyHostToDevice));
        pack(dw.p, whi.p, wmid.p, wlo.p, wc);
        pack(dx.p, xhi.p, xmid.p, xlo.p, xc);
        check(cudaDeviceSynchronize());
        const float one = 1.f, zero = 0.f;
        auto run = [&](int terms) {
            if (terms <= 0) {
                if (terms == 0)
                    check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, s.m, s.n, s.k, &one,
                        dw.p, CUDA_R_32F, s.k, dx.p, CUDA_R_32F, s.k, &zero, dy.p + 16, CUDA_R_32F,
                        s.m, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
                else
                    check(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, s.m, s.n, s.k, &one,
                        dw.p, CUDA_R_32F, s.k, 0, dx.p, CUDA_R_32F, s.k, xc, &zero, dy.p + 16, CUDA_R_32F,
                        s.m, yc, 1, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
            } else {
                // Weight packing is cold/resident. Include activation splitting
                // on every timed call; this is still a component diagnostic.
                pack(dx.p, xhi.p, xmid.p, xlo.p, xc);
                const __nv_bfloat16 * wa[] = {whi.p, wmid.p, wlo.p};
                const __nv_bfloat16 * xb[] = {xhi.p, xmid.p, xlo.p};
                const int pairs[][2] = {{1,2},{2,1},{2,2},{0,2},{1,1},{2,0},{0,1},{1,0},{0,0}};
                int count = 0;
                for (const auto & pair : pairs) {
                    const int order = pair[0] + pair[1];
                    if ((terms == 3 && order > 1) || (terms == 6 && order > 2)) continue;
                    check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, s.m, s.n, s.k, &one,
                        wa[pair[0]], CUDA_R_16BF, s.k, xb[pair[1]], CUDA_R_16BF, s.k,
                        count++ ? &one : &zero, dy.p + 16, CUDA_R_32F, s.m,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                }
                if (count != terms) throw std::runtime_error("Incorrect compensated term count");
            }
            check(cudaDeviceSynchronize());
        };
        check(cudaMemcpy(dy.p, result.data(), result.size() * 4, cudaMemcpyHostToDevice));
        run(0);
        check(cudaMemcpy(reference.data(), dy.p + 16, yc * 4, cudaMemcpyDeviceToHost));
        // All outputs compare with the full pedantic result; independent Double
        // checks use at most 4096 deterministic positions covering matrix edges.
        std::vector<std::pair<size_t,double>> independent;
        const size_t samples = std::min(yc, size_t(4096));
        for (size_t j = 0; j < samples; ++j) {
            const size_t i = samples == 1 ? 0 : j * (yc - 1) / (samples - 1), row = i % s.m, token = i / s.m;
            double value = 0;
            for (int k = 0; k < s.k; ++k) value += double(w[row * s.k + k]) * x[token * s.k + k];
            independent.emplace_back(i, value);
        }
        // ABBA for each candidate against the same precision baseline.
        for (int candidate : {-1,3,6,9}) for (int phase = 0; phase < 4; ++phase) {
            const int terms = phase == 0 || phase == 3 ? 0 : candidate;
            for (int warmup = 0; warmup < 3; ++warmup) run(terms);
            std::vector<double> times;
            const auto start = std::chrono::steady_clock::now();
            do {
                const auto before = std::chrono::steady_clock::now(); run(terms);
                times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - before).count());
            } while (times.size() < 20 || std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < .25);
            check(cudaMemcpy(result.data(), dy.p, result.size() * 4, cudaMemcpyDeviceToHost));
            bool finite = true, canaries = true;
            double max_ref = 0, max_double = 0, max_scaled_double = 0;
            for (size_t i = 0; i < yc; ++i) {
                finite &= std::isfinite(result[i + 16]);
                max_ref = std::max(max_ref, std::abs(double(result[i + 16]) - reference[i]));
            }
            for (const auto & item : independent) {
                const double error = std::abs(double(result[item.first + 16]) - item.second);
                max_double = std::max(max_double, error);
                max_scaled_double = std::max(max_scaled_double, error / (2e-6 + 2e-6 * std::abs(item.second)));
            }
            for (size_t i = 0; i < 16; ++i) canaries &= result[i] == -1234567.25f && result[yc + 16 + i] == -1234567.25f;
            std::vector<float> afterw(wc), afterx(xc);
            check(cudaMemcpy(afterw.data(), dw.p, wc * 4, cudaMemcpyDeviceToHost));
            check(cudaMemcpy(afterx.data(), dx.p, xc * 4, cudaMemcpyDeviceToHost));
            const bool immutable = fingerprint(afterw) == wh && fingerprint(afterx) == xh;
            std::sort(times.begin(), times.end());
            const double median = (times[(times.size() - 1) / 2] + times[times.size() / 2]) * .5;
            if (!first) std::cout << ',';
            first = false;
            std::cout << "{\"N\":" << s.n << ",\"M\":" << s.m << ",\"K\":" << s.k
                << ",\"kind\":\"" << s.kind << "\",\"candidate_terms\":" << candidate << ",\"phase\":" << phase
                << ",\"executed_terms\":" << terms << ",\"samples\":" << times.size() << ",\"median_ms\":" << median
                << ",\"max_pedantic_difference\":" << max_ref << ",\"independent_double_positions\":" << independent.size()
                << ",\"max_double_error\":" << max_double << ",\"max_scaled_double_error\":" << max_scaled_double
                << ",\"finite\":" << (finite ? "true" : "false") << ",\"canaries\":" << (canaries ? "true" : "false")
                << ",\"inputs_unchanged\":" << (immutable ? "true" : "false") << ",\"diagnostic_numerical_pass\":"
                << (finite && canaries && immutable && max_scaled_double <= 1 ? "true" : "false") << '}';
        }
    }
    const auto destroy_status = cublasDestroy(handle);
    owner.value = nullptr;
    check(destroy_status);
    std::cout << "],\"completed\":true,\"model_qualified\":false,\"production_promoted\":false}\n";
    return 0;
} catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
