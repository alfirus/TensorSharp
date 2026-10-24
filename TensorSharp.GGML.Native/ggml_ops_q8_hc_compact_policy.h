// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <cstdint>
#include <cstring>
#include <initializer_list>

struct TsgQ8HcOption { bool valid; bool enabled; };
inline TsgQ8HcOption tsg_q8_hc_parse_option(const char * value) noexcept {
    if (!value || std::strcmp(value, "0") == 0) return {true, false};
    if (std::strcmp(value, "1") == 0) return {true, true};
    return {false, false};
}

constexpr bool tsg_q8_hc_target_shape(int inner, int rows, int columns) noexcept {
    return inner == 16384 && rows == 24 && (columns == 1 || columns > 8);
}
// Return0 for unchanged,1 for row-warp,2 for compact rows. Explicit independent
// parallel-K modes retain their existing precedence outside this predicate.
constexpr int tsg_q8_hc_selected_variant(int inner, int rows, int columns,
                                      bool enabled) noexcept {
    return !enabled || !tsg_q8_hc_target_shape(inner, rows, columns) ? 0
        : columns == 1 ? 1 : 2;
}

// Same reuse/padding criterion as the existing policy, using the ACTUAL row
// tile and the ACTUAL compact kernel's64/128-column occupancy observations.
// No device query or allocation belongs in this pure decision helper.
inline int tsg_q8_hc_auto_columns(int rows, int columns, int multiprocessors,
                                 int active64, int active128, int tile_rows) noexcept {
    if (rows <= 0 || columns < 64 || multiprocessors <= 0 || tile_rows <= 0) return 16;
    const int64_t row_tiles = (int64_t(rows) + tile_rows - 1) / tile_rows;
    for (int tile : {128, 64}) {
        const int active = tile == 128 ? active128 : active64;
        if (columns < tile || active <= 0) continue;
        const int64_t column_tiles = (int64_t(columns) + tile - 1) / tile;
        if (int64_t(columns) * 4 < column_tiles * tile * 3) continue;
        const int64_t target = int64_t(multiprocessors) * (active < 2 ? active : 2);
        if (row_tiles * column_tiles >= target) return tile;
    }
    return 16;
}

// `tile` is the resolved automatic/explicit choice. Below64 columns explicit
// global policies keep their old32-column behavior; the caller resolves only
// automatic policy to16. All valid outputs retain the same K-FMA recurrence.
constexpr int tsg_q8_hc_compact_columns(int columns, int tile) noexcept {
    return columns <= 16 || tile == 16 ? 16
        : columns >= 128 && tile == 128 ? 128
        : columns >= 64 && tile >= 64 ? 64 : 32;
}
