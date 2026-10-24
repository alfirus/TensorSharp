// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

// These predicates only select independent output columns. The kernel retains
// the original K-increasing F32 FMA sequence, including decoded-weight rounding.
constexpr bool tsg_q8_ordered_small_batch_shape(int inner, int rows, int columns) {
    return columns >= 2 && columns <= 8
        // The wide output of the short-K Flash gates benefits from tile reuse
        // by N=8. Keep its unqualified N=5..8 range on the existing tile.
        && !(inner == 128 && rows == 8192 && columns > 4);
}

constexpr bool tsg_q8_ordered_gate_prefill_shape(int inner, int rows, int columns) {
    // Wider M128 gates lose to the existing tile; M64/N512 was only marginal.
    // Keep the opt-in inside the measured beneficial region for each shape.
    return inner == 4096 && columns >= 9
        && ((rows == 64 && columns <= 336) || (rows == 128 && columns <= 128));
}

constexpr bool tsg_q8_gate_prefill_tile16_shape(int inner, int rows, int columns) {
    return inner == 4096 && (rows == 64 || rows == 128)
        && columns >= 64 && columns <= 512;
}
