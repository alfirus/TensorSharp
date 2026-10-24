// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_q8_hc_compact_policy.h"
#include <cstdio>
#include <stdexcept>
#include <limits>

namespace {
int checks = 0;
void require(bool value, const char * message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
// Enumerate candidate grids directly. The choice is the widest valid tile;
// no production selection helper is used to form this oracle.
int independent_tile(int rows, int columns, int sms, int a64, int a128, int row_tile) {
    if (rows <= 0 || columns < 64 || sms <= 0 || row_tile <= 0) return 16;
    int best = 16;
    for (int tile = 64; tile <= 128; tile *= 2) {
        const int active = tile == 64 ? a64 : a128;
        if (columns < tile || active < 1) continue;
        const int64_t padded_columns = ((int64_t(columns) - 1) / tile + 1) * tile;
        const int64_t blocks = ((int64_t(rows) - 1) / row_tile + 1) * (padded_columns / tile);
        const int64_t required = int64_t(sms) * (active == 1 ? 1 : 2);
        if (4 * int64_t(columns) >= 3 * padded_columns && blocks >= required) best = tile;
    }
    return best;
}
}

int main() {
    try {
        for (const char * value : {static_cast<const char *>(nullptr), "0", "1"}) {
            const auto parsed = tsg_q8_hc_parse_option(value);
            require(parsed.valid && parsed.enabled == (value && value[0] == '1'), "Valid flag rejected");
        }
        for (const char * value : {"", "01", "true", "false", " 1", "1 ", "2", "-1", "0x1", "1\n"}) {
            const auto parsed = tsg_q8_hc_parse_option(value);
            require(!parsed.valid && !parsed.enabled, "Malformed flag accepted");
        }
        for (int k : {0, 32, 4096, 16352, 16384, 16416})
        for (int m : {0, 1, 23, 24, 25, 32, 64, 128})
        for (int n : {0, 1, 2, 8, 9, 13, 16, 33, 336, 512}) {
            const bool exact_target = k == 16384 && m == 24 && (n == 1 || n >= 9);
            require(tsg_q8_hc_target_shape(k,m,n) == exact_target, "Shape gate widened");
            require(tsg_q8_hc_selected_variant(k,m,n,false) == 0, "OFF selected a candidate");
            require(tsg_q8_hc_selected_variant(k,m,n,true) == (!exact_target ? 0 : n == 1 ? 1 : 2), "Wrong variant");
        }
        require(tsg_q8_hc_auto_columns(33,128,2,2,2,32) == 64, "Compact grid count not used");
        require(tsg_q8_hc_auto_columns(33,128,2,2,2,64) == 16, "Old row-grid control wrong");
        require(tsg_q8_hc_auto_columns(24,512,1,1,1,32) == 128, "Reuse-eligible128 rejected");
        require(tsg_q8_hc_auto_columns(24,512,60,2,2,32) == 16, "Underpopulated wide tile selected");
        require(tsg_q8_hc_auto_columns(24,95,1,2,2,32) == 16, "Excessive column padding accepted");
        struct column_case { int columns, resolved, expected; };
        const column_case column_cases[] = {{13,32,16}, {17,16,16}, {33,16,16},
             {33,32,32}, {33,64,32}, {33,128,32}, {64,16,16}, {64,32,32},
             {64,64,64}, {64,128,64}, {128,128,128}, {336,16,16}, {512,16,16}};
        for (const auto & item : column_cases)
            require(tsg_q8_hc_compact_columns(item.columns,item.resolved) == item.expected,
                "Resolved HC tile changed explicit-policy geometry");
        for (int m : {1,24,31,32,33,63,64,65,129})
        for (int n : {1,8,33,63,64,65,95,96,127,128,129,192,336,512})
        for (int sms : {0,1,2,4,60})
        for (int a64 : {0,1,2,4})
        for (int a128 : {0,1,2,4})
        for (int row_tile : {32,64})
            require(tsg_q8_hc_auto_columns(m,n,sms,a64,a128,row_tile)
                == independent_tile(m,n,sms,a64,a128,row_tile), "Actual row/occupancy policy mismatch");
        const int high = std::numeric_limits<int>::max();
        require(tsg_q8_hc_auto_columns(high,high,high,2,2,32)
            == independent_tile(high,high,high,2,2,32), "Policy overflowed32-bit bounds");
        std::printf("PASS q8-hc-compact-policy checks=%d\n",checks);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr,"FAIL q8-hc-compact-policy: %s\n",error.what());
        return 1;
    }
}
