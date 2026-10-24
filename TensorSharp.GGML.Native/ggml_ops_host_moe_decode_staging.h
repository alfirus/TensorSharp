// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <cstddef>

namespace tsg {
constexpr std::size_t kHostMoeFileStagingBytes = std::size_t(32) << 20;
constexpr std::size_t kHostMoeScalarFileStagingBytes = std::size_t(72) << 20;

inline bool host_moe_scalar_staging_flag(const char* value) noexcept {
    return value && value[0] == '1' && value[1] == '\0';
}

// A width-based experiment: one-token prefill has the same contract as scalar
// decode. Multirow prefill, joint decode and non-GLM callers stay unchanged.
// The complete selected payload must fit; never grow for a partial large union.
inline bool host_moe_scalar_staging_eligible(bool enabled, bool pooled_glm,
    int rows, int selected, std::size_t missing_bytes) noexcept {
    return enabled && pooled_glm && rows == 1 && selected == 8
        && missing_bytes > kHostMoeFileStagingBytes
        && missing_bytes <= kHostMoeScalarFileStagingBytes;
}

// Genuine wide, nonjoint GLM uses the existing two-slot pipeline. Unlike the
// scalar experiment, its complete selected union may span many 72 MiB arenas.
inline bool host_moe_prefill_ring_staging_eligible(bool enabled, bool pooled_glm,
    bool genuine_wide_prefill, int selected, std::size_t missing_bytes) noexcept {
    return enabled && pooled_glm && genuine_wide_prefill && selected == 8
        && missing_bytes > kHostMoeFileStagingBytes;
}
} // namespace tsg
