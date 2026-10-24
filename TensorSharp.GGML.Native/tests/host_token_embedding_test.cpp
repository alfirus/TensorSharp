// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Isolated CUDA correctness oracle: original GGML_GET_ROWS from a resident
// Q4_K table versus the same original op from raw, Host-gathered Q4_K rows.
// Synthetic geometry here never bypasses the real model's production profile.
#include "../glm_host_token_embedding.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
constexpr std::size_t H = 4096, row_bytes = 2304, maximum_owned_bytes = 32u << 20;
int checks = 0;
std::size_t floats_compared = 0, peak_host = 0, peak_device = 0;
void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
struct CaseBuffers {
    ggml_backend_t backend;
    ggml_context* context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    explicit CaseBuffers(ggml_backend_t b) : backend(b) {
        context = ggml_init({1u << 20, nullptr, true});
        require(context != nullptr, "fixture metadata context");
    }
    ~CaseBuffers() {
        // Also fence failed compute/upload attempts before physical payload free.
        if (buffer) { ggml_backend_synchronize(backend); ggml_backend_buffer_free(buffer); }
        if (context) ggml_free(context);
    }
};

void run_case(ggml_backend_t backend, std::size_t vocabulary, std::size_t count) {
    const auto layout = tsg_glm::checked_host_token_embedding_layout(
        vocabulary * row_bytes, row_bytes, H, vocabulary, int(GGML_TYPE_Q4_K));
    require(ggml_row_size(GGML_TYPE_Q4_K, H) == row_bytes && int(GGML_TYPE_Q4_K) == 12,
        "original Q4_K row stride and type");
    std::vector<std::uint8_t> raw_table(layout.table_bytes), packed(count * row_bytes);
    std::vector<float> float_row(H);
    ggml_quantize_init(GGML_TYPE_Q4_K);
    for (std::size_t r = 0; r < vocabulary; ++r) {
        for (std::size_t c = 0; c < H; ++c)
            float_row[c] = float(int((r * 31 + c * 17 + c / 7) % 103) - 51) / 37.0f;
        require(ggml_quantize_chunk(GGML_TYPE_Q4_K, float_row.data(),
            raw_table.data() + r * row_bytes, 0, 1, H, nullptr) == row_bytes,
            "original quantizer produces a complete valid raw Q4_K row");
    }
    std::vector<std::int32_t> ids(count), packed_ids(count);
    for (std::size_t i = 0; i < count; ++i) {
        ids[i] = static_cast<std::int32_t>((i * 47 + 13) % vocabulary);
        packed_ids[i] = static_cast<std::int32_t>(i);
    }
    ids.front() = static_cast<std::int32_t>(vocabulary - 1);
    if (count > 1) ids[1] = 0;
    if (count > 2) { ids[2] = ids[0]; ids.back() = 0; }
    std::size_t copied = 0;
    require(tsg_glm::pack_host_token_embedding_rows(layout, raw_table.data(), raw_table.size(),
        ids.data(), count, packed.data(), packed.size(),
        [](void* d, const void* s, std::size_t n) { std::memcpy(d, s, n); return true; }, &copied)
        && copied == packed.size(), "raw gather completes before upload");

    CaseBuffers owned(backend);
    auto* ctx = owned.context;
    auto* resident = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, H, vocabulary);
    auto* compact = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, H, count);
    auto* original_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, count);
    auto* compact_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, count);
    ggml_set_input(resident); ggml_set_input(compact);
    ggml_set_input(original_ids); ggml_set_input(compact_ids);
    auto* oracle = ggml_get_rows(ctx, resident, original_ids);
    auto* gathered = ggml_get_rows(ctx, compact, compact_ids);
    ggml_set_output(oracle); ggml_set_output(gathered);
    require(oracle->op == GGML_OP_GET_ROWS && gathered->op == GGML_OP_GET_ROWS &&
        oracle->src[0] == resident && gathered->src[0] == compact &&
        resident->nb[1] == row_bytes && compact->nb[1] == row_bytes,
        "both paths use unchanged original GET_ROWS with Q4_K source rows");
    require(oracle->type == GGML_TYPE_F32 && gathered->type == GGML_TYPE_F32 &&
        oracle->ne[0] == std::int64_t(H) && oracle->ne[1] == std::int64_t(count) &&
        gathered->ne[0] == std::int64_t(H) && gathered->ne[1] == std::int64_t(count),
        "identical count-by4096 F32 output geometry");
    require(ggml_backend_supports_op(backend, oracle) && ggml_backend_supports_op(backend, gathered),
        "actual original CUDA backend supports both GET_ROWS nodes");
    auto* graph_oracle = ggml_new_graph_custom(ctx, 16, false);
    auto* graph_gathered = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph_oracle, oracle);
    ggml_build_forward_expand(graph_gathered, gathered);
    owned.buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(owned.buffer != nullptr && !ggml_backend_buffer_is_host(owned.buffer),
        "all table, token and output payloads physically allocated on CUDA");
    require(ggml_backend_buffer_get_type(owned.buffer) == ggml_backend_get_default_buffer_type(backend),
        "original CUDA buffer type, no CPU facade or fallback");
    const auto device_bytes = ggml_backend_buffer_get_size(owned.buffer);
    require(device_bytes <= maximum_owned_bytes, "bounded actual owned device allocation");
    require(oracle->buffer == owned.buffer && gathered->buffer == owned.buffer &&
        oracle->data != resident->data && gathered->data != compact->data && oracle->data != gathered->data,
        "distinct GPU output storage, never a raw-source alias");
    std::vector<float> expected(H * count), actual(H * count);
    const auto host_bytes = raw_table.capacity() + packed.capacity() + float_row.capacity() * sizeof(float) +
        (ids.capacity() + packed_ids.capacity()) * sizeof(std::int32_t) +
        (expected.capacity() + actual.capacity()) * sizeof(float) + ggml_get_mem_size(ctx);
    require(host_bytes <= maximum_owned_bytes, "bounded exact owned Host capacities plus ggml metadata");
    peak_host = std::max(peak_host, host_bytes); peak_device = std::max(peak_device, device_bytes);
    ggml_backend_tensor_set(resident, raw_table.data(), 0, raw_table.size());
    ggml_backend_tensor_set(compact, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(original_ids, ids.data(), 0, ids.size() * sizeof(ids[0]));
    ggml_backend_tensor_set(compact_ids, packed_ids.data(), 0, packed_ids.size() * sizeof(packed_ids[0]));
    require(ggml_backend_graph_compute(backend, graph_oracle) == GGML_STATUS_SUCCESS,
        "resident oracle dispatched on original CUDA backend");
    require(ggml_backend_graph_compute(backend, graph_gathered) == GGML_STATUS_SUCCESS,
        "packed raw rows dispatched on original CUDA backend");
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(oracle, expected.data(), 0, expected.size() * sizeof(float));
    ggml_backend_tensor_get(gathered, actual.data(), 0, actual.size() * sizeof(float));
    for (std::size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(expected[i]) && std::isfinite(actual[i]), "both original GPU outputs finite");
        require(std::memcmp(&expected[i], &actual[i], sizeof(float)) == 0,
            "every F32 output bit-exact to original resident CUDA GET_ROWS");
    }
    floats_compared += actual.size();
    std::printf("host-token-embedding-cuda case vocab=%zu rows=%zu row_bytes=%zu copied_bytes=%zu host_owned_bytes=%zu device_owned_bytes=%zu floats=%zu exact=1\n",
        vocabulary, count, row_bytes, copied, host_bytes, device_bytes, actual.size());
}
}

int main(int argc, char**) {
    if (argc != 1) { std::fprintf(stderr, "FAIL this isolated fixture accepts no production-profile override\n"); return 1; }
    if (ggml_backend_cuda_get_device_count() < 1) {
        std::fprintf(stderr, "SKIP host token embedding requires an actual CUDA device\n"); return 77;
    }
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cuda_init(0), ggml_backend_free);
    if (!backend) { std::fprintf(stderr, "FAIL cannot initialize original CUDA backend\n"); return 1; }
    try {
        for (std::size_t vocabulary : {std::size_t(317), std::size_t(617)})
            for (std::size_t count : {std::size_t(1), std::size_t(9), std::size_t(512)})
                run_case(backend.get(), vocabulary, count);
        std::printf("host-token-embedding-cuda passed cases=6 checks=%d skips=0 floats=%zu host_owned_peak=%zu device_owned_peak=%zu scope=fixture_owned_payload_and_metadata_not_backend_pools_or_model_profile\n",
            checks, floats_compared, peak_host, peak_device);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL host token embedding: %s\n", e.what()); return 1;
    }
}
