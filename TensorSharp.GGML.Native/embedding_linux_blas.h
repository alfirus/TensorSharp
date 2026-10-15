// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

// TensorSharp-owned optional LP64 OpenBLAS dispatch. No ggml headers or
// upstream implementation changes are needed. The embedding graph owns the
// plan; its callbacks may partition a projection over the existing ggml
// workers only when the loaded BLAS actually reports one worker.
// No thread setters, environment writes, math modes or global caches.
// Primary API/thread ABI: https://github.com/OpenMathLib/OpenBLAS/blob/develop/cblas.h
// https://github.com/OpenMathLib/OpenBLAS/wiki/Faq
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <climits>
#include <limits>
#include <memory>
#include <string>
#if defined(__linux__)
#include <dlfcn.h>
#include <cstdlib>
#endif

namespace tsg_embeddings {

enum class blas_partition { output_channels, token_rows };

struct blas_const_matrix {
    const float * data = nullptr;
    size_t elements = 0; // Remaining accessible elements from data, including padding.
    int64_t rows = 0, columns = 0, stride = 0; // Row-major, F32 unit inner stride.
};
struct blas_matrix {
    float * data = nullptr;
    size_t elements = 0;
    int64_t rows = 0, columns = 0, stride = 0;
};

struct linux_blas_counts {
    uint64_t gemm_calls = 0, output_channel_calls = 0, token_row_calls = 0;
    uint64_t output_elements = 0, empty_partitions = 0, worker_mask = 0;
    uint64_t shape_rejections = 0, policy_rejections = 0;
    uint64_t active_calls = 0;
};

class linux_blas_plan final {
    struct counters {
        std::atomic<uint64_t> gemm{0}, channels{0}, rows{0}, elements{0}, empty{0}, workers{0}, shape{0}, policy{0}, active{0};
    };
    struct library {
        // CBLAS LP64 ABI: all enum/dimension/leading-dimension arguments are int.
        using sgemm_fn = void (*)(int, int, int, int, int, int, float, const float *, int, const float *, int, float, float *, int);
        using get_int_fn = int (*)();
        using get_config_fn = char *(*)();
        void * handle = nullptr;
        sgemm_fn sgemm = nullptr;
        get_int_fn threads = nullptr, parallel = nullptr;
        std::string path, config;
        std::shared_ptr<counters> counts;
        ~library() {
#if defined(__linux__)
            if (handle) dlclose(handle); // Only this owned loader reference.
#endif
        }
    };
    std::shared_ptr<library> loaded_;
    std::shared_ptr<counters> counts_ = std::make_shared<counters>();
    std::string reason_, path_, config_;
    int worker_budget_ = 8;

    static bool matrix_range(const float * data, size_t elements, int64_t rows, int64_t columns,
        int64_t stride, uintptr_t & begin, uintptr_t & end) {
        if (!data || rows <= 0 || columns <= 0 || rows > INT_MAX || columns > INT_MAX ||
            stride < columns || stride > INT_MAX || reinterpret_cast<uintptr_t>(data) % alignof(float)) return false;
        const uint64_t extent = uint64_t(rows - 1) * uint64_t(stride) + uint64_t(columns);
        if (extent > elements || extent > std::numeric_limits<size_t>::max() / sizeof(float)) return false;
        begin = reinterpret_cast<uintptr_t>(data);
        const size_t bytes = size_t(extent) * sizeof(float);
        if (bytes > std::numeric_limits<uintptr_t>::max() - begin) return false;
        end = begin + bytes; return true;
    }
    static bool overlap(uintptr_t a, uintptr_t ae, uintptr_t b, uintptr_t be) { return a < be && b < ae; }
    static bool layout(const blas_const_matrix & weights, const blas_const_matrix & input, const blas_matrix & output) {
        uintptr_t w, we, x, xe, y, ye;
        return weights.columns == input.columns && output.rows == input.rows && output.columns == weights.rows &&
            matrix_range(weights.data, weights.elements, weights.rows, weights.columns, weights.stride, w, we) &&
            matrix_range(input.data, input.elements, input.rows, input.columns, input.stride, x, xe) &&
            matrix_range(output.data, output.elements, output.rows, output.columns, output.stride, y, ye) &&
            !overlap(y, ye, w, we) && !overlap(y, ye, x, xe);
    }
#if defined(__linux__)
    static std::string provider_path(void * symbol) {
        Dl_info info{};
        if (!symbol || !dladdr(symbol, &info) || !info.dli_fname) return {};
        char * actual = realpath(info.dli_fname, nullptr);
        std::string result = actual ? actual : info.dli_fname;
        std::free(actual); return result;
    }
    bool try_load(const char * name) {
        auto value = std::make_shared<library>();
        value->handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (!value->handle) { reason_ = "library-unavailable"; return false; }
        void * gemm = dlsym(value->handle, "cblas_sgemm");
        void * threads = dlsym(value->handle, "openblas_get_num_threads");
        void * parallel = dlsym(value->handle, "openblas_get_parallel");
        void * config = dlsym(value->handle, "openblas_get_config");
        if (!gemm || !threads || !parallel || !config) { reason_ = "missing-openblas-abi-or-thread-metadata"; return false; }
        value->path = provider_path(gemm);
        if (value->path.empty() || value->path != provider_path(threads) || value->path != provider_path(parallel) ||
            value->path != provider_path(config)) { reason_ = "mixed-symbol-providers"; return false; }
        value->sgemm = reinterpret_cast<library::sgemm_fn>(gemm);
        value->threads = reinterpret_cast<library::get_int_fn>(threads);
        value->parallel = reinterpret_cast<library::get_int_fn>(parallel);
        const char * text = reinterpret_cast<library::get_config_fn>(config)();
        if (!text) { reason_ = "missing-openblas-config"; return false; }
        value->config = text;
        path_ = value->path; config_ = value->config;
        if (value->config.find("OpenBLAS ") == std::string::npos || value->config.find("USE64BITINT") != std::string::npos ||
            value->config.find("INTERFACE64") != std::string::npos) { reason_ = "unproven-or-non-lp64-abi"; return false; }
        // Pthread OpenBLAS with one worker supports concurrent BLAS clients.
        // A USE_THREAD=0 library needs separate USE_LOCKING evidence; an OpenMP
        // library has another nesting policy. Neither is assumed safe here.
        if (value->parallel() != 1 || value->threads() != 1) { reason_ = "requires-pthread-openblas-with-one-worker"; return false; }
        value->counts = counts_; std::atomic_store(&loaded_, value); reason_.clear(); return true;
    }
#endif

public:
    explicit linux_blas_plan(const char * explicit_library = nullptr, int worker_budget = 8) : worker_budget_(worker_budget) {
        if (worker_budget < 1 || worker_budget > 8) { reason_ = "worker-budget-must-be-1-to-8"; return; }
#if defined(__linux__)
        if (explicit_library && *explicit_library) {
            if (*explicit_library != '/') { reason_ = "explicit-library-path-must-be-absolute"; return; }
            try_load(explicit_library);
        } else {
            if (!try_load("libopenblas.so.0")) try_load("libblas.so.3");
        }
#else
        (void)explicit_library; reason_ = "linux-only-candidate";
#endif
    }
    linux_blas_plan(const linux_blas_plan &) = delete;
    linux_blas_plan & operator=(const linux_blas_plan &) = delete;
    ~linux_blas_plan() { close(); }

    bool available() const { return bool(std::atomic_load(&loaded_)); }
    bool thread_policy_current() const {
        auto value = std::atomic_load(&loaded_);
        return value && value->parallel() == 1 && value->threads() == 1;
    }
    const std::string & reason() const { return reason_; }
    const std::string & library_path() const { return path_; }
    const std::string & library_config() const { return config_; }
    int worker_budget() const { return worker_budget_; }

    // Close only this plan's reference. A synchronous in-flight call retains
    // its own shared loader reference and counters until SGEMM returns. The
    // graph owner must still prevent new callbacks after destroying the plan.
    void close() { std::atomic_store(&loaded_, std::shared_ptr<library>{}); }

    /// Execute one existing-GGML-worker partition. Empty tails are successful
    /// without a BLAS call. False means no write for this invocation; graph
    /// integration must qualify this plan/layout before constructing a custom
    /// callback and keep the observed library thread policy unchanged.
    bool try_partition(const blas_const_matrix & weights, const blas_const_matrix & input, const blas_matrix & output,
        int ith, int nth, blas_partition axis) const {
        auto value = std::atomic_load(&loaded_);
        auto counts = counts_; // The caller keeps the plan object alive during entry.
        if (!value || nth < 1 || nth > worker_budget_ || ith < 0 || ith >= nth ||
            (axis != blas_partition::output_channels && axis != blas_partition::token_rows)) {
            counts->policy.fetch_add(1, std::memory_order_relaxed); return false;
        }
        if (!layout(weights, input, output)) { counts->shape.fetch_add(1, std::memory_order_relaxed); return false; }
        if (value->parallel() != 1 || value->threads() != 1) {
            counts->policy.fetch_add(1, std::memory_order_relaxed); return false;
        }
        const int64_t length = axis == blas_partition::output_channels ? weights.rows : input.rows;
        const int64_t begin = length * ith / nth, end = length * (ith + 1) / nth;
        if (begin == end) { counts->empty.fetch_add(1, std::memory_order_relaxed); return true; }
        const bool channels = axis == blas_partition::output_channels;
        const int m = int(channels ? end - begin : weights.rows), n = int(channels ? input.rows : end - begin);
        // W[M,K], X[N,K], Y[N,M] are column-major K×M, K×N, M×N
        // views. Channel partitions retain the full Y leading dimension.
        counts->active.fetch_add(1, std::memory_order_relaxed);
        value->sgemm(102 /*ColMajor*/, 112 /*Trans*/, 111 /*NoTrans*/, m, n, int(weights.columns), 1.0f,
            weights.data + (channels ? begin * weights.stride : 0), int(weights.stride),
            input.data + (channels ? 0 : begin * input.stride), int(input.stride), 0.0f,
            output.data + (channels ? begin : begin * output.stride), int(output.stride));
        counts->active.fetch_sub(1, std::memory_order_relaxed);
        counts->gemm.fetch_add(1, std::memory_order_relaxed);
        (channels ? counts->channels : counts->rows).fetch_add(1, std::memory_order_relaxed);
        counts->elements.fetch_add(uint64_t(m) * uint64_t(n), std::memory_order_relaxed);
        counts->workers.fetch_or(uint64_t(1) << ith, std::memory_order_relaxed); return true;
    }

    // A row-major A[M,K] * B[K,N] product for an already partitioned attention
    // tile. Keep the same provider, ABI, bounds, alias and worker-policy checks
    // as projections; no process-wide BLAS configuration is changed.
    bool try_product(const blas_const_matrix & a, const blas_const_matrix & b, const blas_matrix & output, int worker) const {
        auto value = std::atomic_load(&loaded_);
        auto counts = counts_;
        if (!value || worker < 0 || worker >= worker_budget_ || value->parallel() != 1 || value->threads() != 1) {
            counts->policy.fetch_add(1, std::memory_order_relaxed); return false;
        }
        uintptr_t ap, ae, bp, be, yp, ye;
        if (a.columns != b.rows || output.rows != a.rows || output.columns != b.columns ||
            !matrix_range(a.data, a.elements, a.rows, a.columns, a.stride, ap, ae) ||
            !matrix_range(b.data, b.elements, b.rows, b.columns, b.stride, bp, be) ||
            !matrix_range(output.data, output.elements, output.rows, output.columns, output.stride, yp, ye) ||
            overlap(yp, ye, ap, ae) || overlap(yp, ye, bp, be)) {
            counts->shape.fetch_add(1, std::memory_order_relaxed); return false;
        }
        counts->active.fetch_add(1, std::memory_order_relaxed);
        value->sgemm(101 /*RowMajor*/, 111 /*NoTrans*/, 111 /*NoTrans*/,
            int(a.rows), int(b.columns), int(a.columns), 1.0f, a.data, int(a.stride),
            b.data, int(b.stride), 0.0f, output.data, int(output.stride));
        counts->active.fetch_sub(1, std::memory_order_relaxed);
        counts->gemm.fetch_add(1, std::memory_order_relaxed);
        counts->rows.fetch_add(1, std::memory_order_relaxed);
        counts->elements.fetch_add(uint64_t(output.rows) * uint64_t(output.columns), std::memory_order_relaxed);
        counts->workers.fetch_or(uint64_t(1) << worker, std::memory_order_relaxed); return true;
    }

    linux_blas_counts counts() const {
        const auto & c = *counts_;
        return {c.gemm.load(), c.channels.load(), c.rows.load(), c.elements.load(), c.empty.load(), c.workers.load(), c.shape.load(), c.policy.load(), c.active.load()};
    }
};

} // namespace tsg_embeddings
