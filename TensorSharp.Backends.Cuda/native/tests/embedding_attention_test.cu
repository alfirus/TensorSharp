// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <vector>
#include "../kernels/tensorsharp_embedding_attention.cuh"

static bool check(cudaError_t status) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status)); return false;
}

static bool verify(int dim, int batch, int seq, int radius, bool packed) {
    const int heads = 4, kv_heads = dim == 256 ? 2 : 1, group = heads / kv_heads;
    std::vector<float> q(size_t(batch) * seq * heads * dim), k(size_t(batch) * seq * kv_heads * dim), v(k.size()), output(q.size());
    for (size_t i = 0; i < q.size(); ++i) q[i] = std::sin(float(i) * .011f) * .12f;
    for (size_t i = 0; i < k.size(); ++i) { k[i] = std::cos(float(i) * .019f) * .2f; v[i] = std::sin(float(i) * .007f); }
    std::vector<int> bounds(size_t(batch) * seq * 2);
    for (int b = 0; b < batch; ++b) for (int row = 0; row < seq; ++row) {
        bounds[(b * seq + row) * 2] = packed && row >= 17 ? 17 : 0;
        bounds[(b * seq + row) * 2 + 1] = packed && row < 17 ? 17 : seq;
    }
    float *dq = nullptr, *dk = nullptr, *dv = nullptr, *dy = nullptr; int *db = nullptr;
    if (!check(cudaMalloc(&dq, q.size() * sizeof(float))) || !check(cudaMalloc(&dk, k.size() * sizeof(float))) ||
        !check(cudaMalloc(&dv, v.size() * sizeof(float))) || !check(cudaMalloc(&dy, output.size() * sizeof(float))) ||
        !check(cudaMalloc(&db, bounds.size() * sizeof(int)))) return false;
    bool success = check(cudaMemcpy(dq, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice)) &&
        check(cudaMemcpy(dk, k.data(), k.size() * sizeof(float), cudaMemcpyHostToDevice)) &&
        check(cudaMemcpy(dv, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice)) &&
        check(cudaMemcpy(db, bounds.data(), bounds.size() * sizeof(int), cudaMemcpyHostToDevice));
    if (success) {
        const int rows = 16, qrows = dim == 256 ? 8 : 4;
        const size_t shared = (rows * (dim + 128) + 3 * rows) * sizeof(float);
        const dim3 grid(kv_heads, (seq + qrows - 1) / qrows, batch);
        if (dim == 256) ts_embedding_attention_group2_d256_f32<<<grid,256,shared>>>(dq, dk, dv, packed ? db : nullptr, dy, batch, heads, kv_heads, seq, radius, 1.f);
        else ts_embedding_attention_group4_d512_f32<<<grid,256,shared>>>(dq, dk, dv, packed ? db : nullptr, dy, batch, heads, kv_heads, seq, radius, 1.f);
        success = check(cudaGetLastError()) && check(cudaMemcpy(output.data(), dy, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }
    double maximum_error = 0;
    std::vector<int> queries;
    if (seq < 128) for (int row = 0; row < seq; ++row) queries.push_back(row);
    else for (int row : {0, 1, seq / 2 - 1, seq / 2, seq / 2 + 1, seq - 2, seq - 1}) queries.push_back(row);
    if (success) for (int b = 0; b < batch; ++b) for (int row : queries) for (int h = 0; h < heads; ++h) {
        int start = packed ? bounds[(b * seq + row) * 2] : 0, end = packed ? bounds[(b * seq + row) * 2 + 1] : seq;
        if (radius > 0) { start = std::max(start, row - radius); end = std::min(end, row + radius + 1); }
        std::vector<double> scores(end - start), expected(dim, 0.0);
        double maximum = -DBL_MAX;
        const size_t qbase = ((size_t(b) * seq + row) * heads + h) * dim;
        for (int key = start; key < end; ++key) {
            const size_t base = ((size_t(b) * seq + key) * kv_heads + h / group) * dim;
            double score = 0;
            for (int d = 0; d < dim; ++d) score += double(q[qbase + d]) * k[base + d];
            scores[key - start] = score; maximum = std::max(maximum, score);
        }
        double denominator = 0;
        for (int key = start; key < end; ++key) {
            const double p = std::exp(scores[key - start] - maximum); denominator += p;
            const size_t base = ((size_t(b) * seq + key) * kv_heads + h / group) * dim;
            for (int d = 0; d < dim; ++d) expected[d] += p * v[base + d];
        }
        for (int d = 0; d < dim; ++d) {
            const double error = std::abs(expected[d] / denominator - output[qbase + d]);
            maximum_error = std::max(maximum_error, error);
            success = success && std::isfinite(output[qbase + d]) && error < 2e-5;
        }
    }
    std::printf("dim=%d batch=%d seq=%d radius=%d packed=%d max_error=%.9g %s\n", dim, batch, seq, radius, packed, maximum_error, success ? "PASS" : "FAIL");
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dy); cudaFree(db);
    return success;
}

int main() {
    bool success = true;
    for (int dim : {256, 512}) {
        success = verify(dim, 2, 7, 0, false) && success;
        success = verify(dim, 2, 33, 16, false) && success;
        success = verify(dim, 1, 41, 0, true) && success;
        success = verify(dim, 1, 41, 16, true) && success;
        success = verify(dim, 1, 513, 512, false) && success;
        success = verify(dim, 1, 1025, 512, false) && success;
    }
    return success ? 0 : 1;
}
