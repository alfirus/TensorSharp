// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "embedding_cpu_attention.h"
#include "embedding_cpu_profile.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

using namespace tsg_embeddings;

static bool verify(int dim, int length, int heads, int kv, int batch, int radius, int threads,
    bool f32_mask, bool no_mask, bool close_before_compute, bool require_blas,
    bool broadcast_mask = false, bool fully_masked_row = false, bool grow_workspace = false) {
    linux_blas_plan blas(std::getenv("TSG_EMBEDDING_BLAS_LIBRARY"), std::min(threads, 8));
    if (require_blas && !blas.available()) { std::fprintf(stderr, "BLAS unavailable: %s\n", blas.reason().c_str()); return false; }
    cpu_attention_workspace workspace(&blas);
    workspace.reserve(length);
    std::vector<std::unique_ptr<cpu_attention_node>> nodes;
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    auto * backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, threads);
    // Head/token permutation exercises the fused-QKV row strides used by the model.
    auto * qb = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, heads, length, batch);
    auto * kb = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, kv, length, batch);
    auto * vb = ggml_dup_tensor(ctx, kb);
    auto * q = ggml_permute(ctx, qb, 0, 2, 1, 3);
    auto * k = ggml_permute(ctx, kb, 0, 2, 1, 3);
    auto * v = ggml_permute(ctx, vb, 0, 2, 1, 3);
    const int mask_batch = broadcast_mask ? 1 : batch;
    auto * mask = no_mask ? nullptr : ggml_new_tensor_4d(ctx, f32_mask ? GGML_TYPE_F32 : GGML_TYPE_F16, length + 3, length + 5, 1, mask_batch);
    ggml_set_input(qb); ggml_set_input(kb); ggml_set_input(vb);
    if (mask) ggml_set_input(mask);
    constexpr float scale = .31f;
    auto * output = cpu_attention(ctx, q, k, v, mask, radius, scale, workspace, nodes);
    if (!output) {
        if (require_blas) return false;
        nodes.push_back(std::make_unique<cpu_attention_node>(cpu_attention_node{&workspace, radius, scale}));
        ggml_tensor * args[] = {q, k, v, mask};
        output = ggml_custom_4d(ctx, GGML_TYPE_F32, dim * heads, length, batch, 1,
            args, mask ? 4 : 3, cpu_attention_compute, std::min(threads, 8), nodes.back().get());
    }
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, output);
    auto * allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    bool success = ggml_gallocr_alloc_graph(allocator, graph);
    std::vector<float> qdata(ggml_nelements(qb)), kdata(ggml_nelements(kb)), vdata(ggml_nelements(vb));
    for (size_t i = 0; i < qdata.size(); ++i) qdata[i] = .7f * std::sin(float(i) * .17317f);
    for (size_t i = 0; i < kdata.size(); ++i) kdata[i] = .5f * std::cos(float(i) * .13719f);
    for (size_t i = 0; i < vdata.size(); ++i) vdata[i] = .3f * std::sin(float(i) * .11391f);
    std::vector<float> mdata(size_t(length + 3) * (length + 5) * mask_batch, -INFINITY);
    if (mask) for (int b = 0; b < mask_batch; ++b) for (int r = 0; r < length; ++r) for (int c = 0; c < length; ++c)
        if (!(fully_masked_row && r == std::min(1, length - 1)) &&
            (radius < 0 || std::abs(c - r) <= radius) && (c == r || c % 11 != 3))
            mdata[(size_t(b) * (length + 5) + r) * (length + 3) + c] = (c % 3 - 1) * .125f;
    std::vector<ggml_fp16_t> half_mask(mdata.size());
    for (size_t i = 0; i < mdata.size(); ++i) half_mask[i] = ggml_fp32_to_fp16(mdata[i]);
    std::vector<float> result(ggml_nelements(output), std::numeric_limits<float>::quiet_NaN());
    if (success) {
        ggml_backend_tensor_set(qb, qdata.data(), 0, qdata.size() * sizeof(float));
        ggml_backend_tensor_set(kb, kdata.data(), 0, kdata.size() * sizeof(float));
        ggml_backend_tensor_set(vb, vdata.data(), 0, vdata.size() * sizeof(float));
        if (mask) ggml_backend_tensor_set(mask, f32_mask ? static_cast<void *>(mdata.data()) : static_cast<void *>(half_mask.data()), 0, ggml_nbytes(mask));
        if (close_before_compute) blas.close();
        auto * plan = grow_workspace ? ggml_backend_graph_plan_create(backend, graph) : nullptr;
        success = !grow_workspace || plan != nullptr;
        if (success) success = (plan ? ggml_backend_graph_plan_compute(backend, plan) :
            ggml_backend_graph_compute(backend, graph)) == GGML_STATUS_SUCCESS;
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
        if (success && grow_workspace) {
            const auto initial = result;
            // A cached graph/plan retains userdata, not a pointer into the
            // scratch vector. Growth must update every worker's row stride.
            workspace.reserve(length + 67);
            success = ggml_backend_graph_plan_compute(backend, plan) == GGML_STATUS_SUCCESS;
            ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
            success = success && initial == result;
        }
        if (plan) ggml_backend_graph_plan_free(backend, plan);
        for (float value : result) success = success && std::isfinite(value);
        // An independent double-product softmax oracle reads original operands.
        // All rows are checked on small cases; large cases sample both tile and
        // window edges while executing and checking finiteness of the full graph.
        for (int b = 0; b < batch; ++b) for (int h = 0; h < heads; ++h) for (int r = 0; r < length; ++r) {
            if (length > 137 && r != 0 && r != 1 && r != 63 && r != 64 && r != 65 && r != length / 2 && r != length - 1) continue;
            std::vector<double> scores(length, -INFINITY);
            double maximum = -INFINITY;
            for (int c = 0; c < length; ++c) {
                if (radius >= 0 && std::abs(c - r) > radius) continue;
                double score = 0;
                for (int d = 0; d < dim; ++d) score += double(qdata[((size_t(b) * length + r) * heads + h) * dim + d]) *
                    kdata[((size_t(b) * length + c) * kv + h / (heads / kv)) * dim + d];
                scores[c] = score * scale + (mask ? mdata[(size_t(broadcast_mask ? 0 : b) * (length + 5) + r) * (length + 3) + c] : 0);
                maximum = std::max(maximum, scores[c]);
            }
            double sum = 0;
            for (auto & score : scores) { score = std::isfinite(maximum) ? std::exp(score - maximum) : 0; sum += score; }
            for (int d = 0; d < dim; ++d) {
                double expected = 0;
                if (sum > 0) for (int c = 0; c < length; ++c) expected += scores[c] / sum * vdata[((size_t(b) * length + c) * kv + h / (heads / kv)) * dim + d];
                const float actual = result[((size_t(b) * length + r) * heads + h) * dim + d];
                if (std::abs(expected - actual) > 2e-5) {
                    std::fprintf(stderr, "Attention oracle failed dim=%d length=%d radius=%d row=%d head=%d d=%d expected=%.9g actual=%.9g\n",
                        dim, length, radius, r, h, d, expected, actual); success = false;
                }
            }
        }
        if (require_blas && !close_before_compute) success = success && workspace.scalar_products == 0 && blas.counts().gemm_calls > 0;
        if (close_before_compute) success = success && workspace.scalar_products > 0;
    }
    std::printf("attention dim=%d length=%d heads=%d kv=%d batch=%d radius=%d threads=%d tiles=%llu scalar=%llu pass=%d\n",
        dim, length, heads, kv, batch, radius, threads, (unsigned long long)workspace.tiles.load(),
        (unsigned long long)workspace.scalar_products.load(), success);
    ggml_gallocr_free(allocator); ggml_free(ctx); ggml_backend_free(backend);
    return success;
}

static bool verify_profile_plan_lifetime() {
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    auto * backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 2);
    auto * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 5);
    ggml_set_input(weights); ggml_set_input(input);
    auto * projection = ggml_mul_mat(ctx, weights, input);
    auto * attention = ggml_scale(ctx, projection, .5f);
    auto * output = ggml_sqr(ctx, attention);
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, output);
    auto * allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    bool success = ggml_gallocr_alloc_graph(allocator, graph);
    ggml_backend_graph_plan_t full_plan = success ? ggml_backend_graph_plan_create(backend, graph) : nullptr;
    success = success && full_plan != nullptr;
    if (success) {
        // The split plans borrow graph nodes and allocated tensor storage.
        // Exercise repeated submissions with changing inputs and destroy them
        // before the full plan, allocator, graph context and backend.
        cpu_profile_plan split(backend, graph, {weights}, {attention});
        success = split.valid();
        const float w[] = {.25f, -.5f, .75f, -.125f, .375f, .625f};
        std::vector<float> x(15), expected(10), actual(10);
        ggml_backend_tensor_set(weights, w, 0, sizeof(w));
        for (int step = 0; success && step < 3; ++step) {
            for (size_t i = 0; i < x.size(); ++i) x[i] = float(int(i) - 7 + step) / 8;
            ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
            success = ggml_backend_graph_plan_compute(backend, full_plan) == GGML_STATUS_SUCCESS;
            ggml_backend_tensor_get(output, expected.data(), 0, expected.size() * sizeof(float));
            success = success && split.compute(5, 1, 1) == GGML_STATUS_SUCCESS;
            ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
            success = success && expected == actual;
            for (int r = 0; r < 5; ++r) for (int c = 0; c < 2; ++c) {
                double value = 0;
                for (int d = 0; d < 3; ++d) value += double(w[c * 3 + d]) * x[r * 3 + d];
                success = success && std::abs(actual[r * 2 + c] - .25 * value * value) < 2e-5;
            }
        }
    }
    if (full_plan) ggml_backend_graph_plan_free(backend, full_plan);
    ggml_gallocr_free(allocator); ggml_free(ctx); ggml_backend_free(backend);
    std::printf("profile-plan-lifetime pass=%d\n", success);
    return success;
}

static bool verify_softmax_ranges(bool require_simd) {
    const bool simd = cpu_softmax_simd_supported();
    if (require_simd && !simd) { std::fprintf(stderr, "Required AVX512 softmax unavailable\n"); return false; }
    double maximum_error = 0;
    size_t checked = 0;
    for (int count : {1, 3, 15, 16, 17, 31, 32, 137, 1025, 8192, 131073}) {
        // Dense exponent sweep crosses every range-reduction boundary, the
        // normal/subnormal transition and the explicit libm fallback at -80.
        std::vector<float> values(count + 17, 9876.5f), original(count);
        for (int c = 0; c < count; ++c) original[c] = values[c] = -104.0f * c / std::max(1, count - 1);
        cpu_softmax(values.data(), count, nullptr, GGML_TYPE_F32, 0, 0, -1, 1, simd);
        double sum = 0;
        for (float value : original) sum += std::exp(double(value));
        for (int c = 0; c < count; ++c) {
            const double expected = std::exp(double(original[c])) / sum;
            const double error = std::abs(values[c] - expected);
            maximum_error = std::max(maximum_error, error);
            if (!std::isfinite(values[c]) || error > std::max(3e-7 * expected, 1e-43)) return false;
            ++checked;
        }
        for (size_t c = count; c < values.size(); ++c) if (values[c] != 9876.5f) return false;
    }
    constexpr int begin = 7;
    for (int count : {15, 16, 17, 31, 32, 137}) for (int radius : {-1, 0, 9})
    for (bool half : {false, true}) for (float scale : {.03125f, 1.0f, 2.0f}) {
        const int query = begin + count / 2;
        std::vector<float> values(count + 17, 9876.5f), original(count), mask(begin + count + 3, -INFINITY);
        std::vector<ggml_fp16_t> half_mask(mask.size());
        for (int c = 0; c < count; ++c) {
            values[c] = original[c] = (c % 29 - 14) * .125f;
            if (c % 11 != 3 || begin + c == query) mask[begin + c] = (c % 3 - 1) * .125f;
        }
        for (size_t c = 0; c < mask.size(); ++c) half_mask[c] = ggml_fp32_to_fp16(mask[c]);
        cpu_softmax(values.data(), count, half ? static_cast<const void *>(half_mask.data()) : mask.data(),
            half ? GGML_TYPE_F16 : GGML_TYPE_F32, begin, query, radius, scale, simd);
        std::vector<double> scores(count);
        double maximum = -INFINITY;
        for (int c = 0; c < count; ++c) {
            scores[c] = radius >= 0 && std::abs(begin + c - query) > radius ? -INFINITY :
                double(original[c]) * scale + mask[begin + c];
            maximum = std::max(maximum, scores[c]);
        }
        double sum = 0;
        for (auto & value : scores) { value = std::exp(value - maximum); sum += value; }
        for (int c = 0; c < count; ++c) {
            const double expected = scores[c] / sum, error = std::abs(values[c] - expected);
            maximum_error = std::max(maximum_error, error);
            if (!std::isfinite(values[c]) || error > std::max(3e-7 * expected, 1e-43)) return false;
            ++checked;
        }
        for (size_t c = count; c < values.size(); ++c) if (values[c] != 9876.5f) return false;
    }
    std::printf("softmax-double-range simd_executed=%d elements=%zu max_absolute_error=%.9g pass=1\n", simd, checked, maximum_error);
    return true;
}

int main(int argc, char ** argv) {
    bool blas = false, require_simd = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--linux-blas") == 0) blas = true;
        else if (std::strcmp(argv[i], "--require-simd") == 0) require_simd = true;
        else return 2;
    }
    if (!verify_softmax_ranges(require_simd)) return 1;
    for (int threads : {1, 2, 8}) for (int radius : {-1, 0, 9})
        if (!verify(37, 65, 4, 2, 2, radius, threads, threads == 2, false, false, blas)) return 1;
    if (!verify(16, 1, 4, 1, 2, 0, 8, false, false, false, blas) ||
        !verify(37, 3, 4, 2, 1, 1, 2, false, true, false, blas) ||
        !verify(256, 137, 4, 2, 1, -1, 8, false, true, false, blas) ||
        !verify(512, 513, 4, 1, 1, 512, 8, false, false, false, blas) ||
        !verify(16, 1025, 4, 1, 1, -1, 8, false, false, false, blas) ||
        !verify(16, 8192, 4, 2, 1, 16, 8, false, false, false, blas) ||
        !verify(37, 65, 4, 1, 1, 9, 8, true, false, true, blas) ||
        !verify(37, 65, 4, 2, 2, -1, 8, false, false, false, blas, true) ||
        !verify(37, 65, 4, 2, 2, 9, 8, true, false, false, blas, true, true) ||
        !verify(37, 65, 4, 2, 2, 9, 8, false, false, false, blas, false, false, true) ||
        !verify_profile_plan_lifetime()) return 1;
    return 0;
}
