// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <cstring>

struct TsgQ8KdaOption { bool valid; bool enabled; };
inline TsgQ8KdaOption tsg_q8_kda_parse_option(const char * value) noexcept {
    if (!value || std::strcmp(value, "0") == 0) return {true, false};
    if (std::strcmp(value, "1") == 0) return {true, true};
    return {false, false};
}

// Exact original Q8 KDA a/beta shapes confirmed by the saved header inventory.
// K128/M8192 b projections, N2..8 and N>512 keep their existing dispatch.
constexpr bool tsg_q8_kda_target_shape(int inner, int rows, int columns) noexcept {
    return inner == 4096 && (rows == 64 || rows == 128)
        && (columns == 1 || (columns >= 9 && columns <= 512));
}
constexpr int tsg_q8_kda_selected_variant(int inner, int rows, int columns,
                                       bool enabled) noexcept {
    return !enabled || !tsg_q8_kda_target_shape(inner, rows, columns) ? 0
        : columns == 1 ? 1 : 2;
}
