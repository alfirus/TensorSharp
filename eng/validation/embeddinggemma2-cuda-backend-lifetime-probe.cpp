// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Diagnostic only: measure the actual owned CUDA wrapper's create/free cycle.
// Link against the existing GgmlOps library; no upstream or runtime edits.
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml_ops_dsv4_fused.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

int main() {
    ggml_backend_t base = nullptr;
    try {
        base = ggml_backend_cuda_init(0);
        if (!base) throw std::runtime_error("CUDA backend unavailable");
        using clock = std::chrono::steady_clock;
        auto cycle = [&] {
            ggml_backend_synchronize(base);
            const auto start = clock::now();
            auto * wrapped = tsg_dsv4_fused_backend_init(base);
            if (!wrapped) throw std::runtime_error("owned precise wrapper unavailable");
            const bool owned = std::strncmp(ggml_backend_name(wrapped), "TSDSV4-", 7) == 0;
            ggml_backend_synchronize(wrapped);
            ggml_backend_free(wrapped);
            if (!owned) throw std::runtime_error("unexpected wrapper implementation");
            return std::chrono::duration<double, std::milli>(clock::now() - start).count();
        };
        for (int i = 0; i < 3; ++i) cycle();
        std::vector<double> samples;
        for (int i = 0; i < 30; ++i) samples.push_back(cycle());
        auto sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        std::printf("{\"completed\":true,\"diagnostic\":true,\"device\":0,\"warmup\":3,\"repeats\":30,"
            "\"median_ms\":%.17g,\"p95_ms\":%.17g,\"mean_ms\":%.17g,\"samples_ms\":[",
            (sorted[14] + sorted[15]) / 2, sorted[28],
            std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size());
        for (size_t i = 0; i < samples.size(); ++i) std::printf("%s%.17g", i ? "," : "", samples[i]);
        std::printf("],\"scope\":\"Actual owned private pedantic CUDA backend create/synchronize/free; excludes base CUDA startup and all model/graph math.\","
            "\"limitations\":\"No model quality, scratch-allocation cost, service latency or reuse speedup is established by this diagnostic.\"}\n");
        ggml_backend_free(base);
        return 0;
    } catch (const std::exception & error) {
        if (base) ggml_backend_free(base);
        std::fprintf(stderr, "backend lifetime diagnostic: %s\n", error.what());
        return 1;
    }
}
