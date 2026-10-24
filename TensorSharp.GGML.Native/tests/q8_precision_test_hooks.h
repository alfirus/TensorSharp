// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include <cstddef>

// Test builds only: submit an existing K-ordered tile without changing the
// production policy or environment. Supported column tiles are 16, 32 and 64.
int tsg_matmul_q8_cuda_launch_test_tile(const void * weights, const void * input, float * output,
    int inner, int rows, int columns, std::size_t weight_stride, std::size_t input_inner_stride,
    std::size_t input_column_stride, int tile_columns, void * stream);
