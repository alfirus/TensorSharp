// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <cstddef>

enum class TsgQ8HcTestRoute : int { control=0, row_warp=1, compact_tile=2, production=3 };
// Test builds only; raw launchers borrow caller-owned device storage/stream.
// Control forces the old ordered arithmetic, independent of experimental ENV.
int tsg_matmul_q8_hc_cuda_test_launch(TsgQ8HcTestRoute route,
    const void * weights, const void * input, float * output,
    int inner, int rows, int columns, std::size_t weight_stride,
    std::size_t input_inner_stride, std::size_t input_column_stride,
    int tile_columns, void * stream);
int tsg_matmul_q8_hc_cuda_test_selected(int inner, int rows, int columns,
                                      int tile_columns, bool enabled);
// Actual successful production enqueue markers: bit0 row-warp, bits1..4 tile
//16/32/64/128. Forced test routes never modify this process-lifetime mask.
unsigned tsg_matmul_q8_hc_cuda_test_reported();
// Actual successful KDA production enqueues: bit0 M64 row, bit1 M64 wide16,
// bit2 M128 row, bit3 M128 wide16. Forced routes never modify this mask.
unsigned tsg_matmul_q8_kda_cuda_test_reported();
// Pure policy: 0 old path, 1 row-warp, 2 Rows32/Columns16.
int tsg_matmul_q8_kda_cuda_test_selected(int inner, int rows, int columns, bool enabled);
