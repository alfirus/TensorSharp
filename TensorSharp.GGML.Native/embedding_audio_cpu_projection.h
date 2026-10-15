// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include "embedding_cpu_matmul.h"
#include <cstring>

namespace tsg_embeddings {

// Optional embedding audio learned projections only. The graph owns the
// provider for its complete lifetime; no process-wide BLAS state is changed.
class audio_cpu_projection final {
    std::unique_ptr<linux_blas_plan> plan_;
    size_t nodes_ = 0;
public:
    audio_cpu_projection(bool cpu_backend, bool require_float32_precision,
        const char * optional_route, const char * library, bool prefer_upstream_tiled = false) {
        // The unchanged upstream tiled F32 path wins for the qualified audio
        // tower when compiled with AVX. Keep the optional provider for builds
        // without that path; do not acquire an unused provider lease.
        if (prefer_upstream_tiled || !cpu_backend || !require_float32_precision || !optional_route ||
            std::strcmp(optional_route, "1") != 0) return;
        auto plan = std::make_unique<linux_blas_plan>(library, 8);
        if (plan->available()) plan_ = std::move(plan);
    }
    bool available() const { return bool(plan_); }
    size_t nodes() const { return nodes_; }
    linux_blas_counts counts() const { return plan_ ? plan_->counts() : linux_blas_counts{}; }

    // These are the actual classic tower's FFN up/down, Q/K/V/output/PW2,
    // and SSCP learned projection shapes qualified by the standalone probe.
    // The 1024->2048 convolution GLU and convolution/attention products keep
    // their existing implementation. The shared helper also checks F32,
    // contiguous two-dimensional operands and the current provider policy.
    static bool shape(int64_t input, int64_t output) {
        return (input == 1024 && (output == 1024 || output == 4096)) ||
            (input == 4096 && output == 1024);
    }
    ggml_tensor * multiply(ggml_context * context, ggml_tensor * weight, ggml_tensor * input) {
        if (!plan_ || !weight || !input || !shape(weight->ne[0], weight->ne[1])) return nullptr;
        auto * output = cpu_matmul(context, weight, input, 1, plan_.get());
        if (output) { ++nodes_; ggml_set_name(output, "tsg_gemma4_audio_cpu_blas_projection"); }
        return output;
    }
};
} // namespace tsg_embeddings
