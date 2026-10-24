// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <cstddef>
#include <cstdint>

namespace tsg_glm {
// Version1 is an explicit declared-payload policy, not a graph-fit estimate or
// a cap on CUDA work pools/vendor memory. The native owner independently checks
// actual tensors, backend, graph topology, source owners and allocator tickets.
struct layer_major_graph_cap {
    static constexpr int profile = 1, context = 2048, chunk = 512, minimum_rows = 9;
    static constexpr std::size_t mib = std::size_t(1) << 20;
    static bool valid_limit(int version, std::uint64_t bytes) noexcept {
        return version == profile && (bytes == 512 * mib || bytes == 768 * mib);
    }
    struct shape {
        int hidden = 0, hc = 0, vocab = 0, layers = 0, recurrent_layers = 0, mla_layers = 0;
        int heads = 0, kda_heads = 0, kda_dim = 0, conv = 0;
        int dense_ff = 0, shared_ff = 0, expert_ff = 0, experts = 0, used = 0;
        int context_rows = 0, ubatch = 0, dense_lead = 0, indexer_top_k = 0, indexer_pool = 0;
    };
    static bool valid_shape(const shape& s) noexcept {
        return s.hidden == 4096 && s.hc == 4 && s.vocab == 154880 && s.layers == 45
            && s.recurrent_layers == 34 && s.mla_layers == 11 && s.heads == 64 && s.kda_heads == 64
            && s.kda_dim == 128 && s.conv == 4 && s.dense_ff > 0 && s.dense_ff <= 12288
            && s.shared_ff > 0 && s.shared_ff <= 2048 && s.expert_ff == 2048
            && s.experts == 288 && s.used == 8 && s.context_rows == context && s.ubatch == chunk
            && s.dense_lead == 3 && s.indexer_top_k == context && s.indexer_pool == 4;
    }
    static bool valid_call(std::int64_t tokens, std::int64_t past, int rows,
        bool hidden_export, bool overrides, bool final_logits_only) noexcept {
        if (rows != chunk || tokens <= chunk || past < 0 || past > context
            || tokens > context - past || hidden_export || overrides || !final_logits_only) return false;
        const auto tail = tokens % chunk;
        return tail == 0 || tail >= minimum_rows;
    }
    static bool valid_range(int begin, int end, int layers, std::int64_t rows,
        std::int64_t past, std::int64_t padded_kv, std::int64_t outputs,
        int kind, bool hidden_export, std::int64_t overrides, bool capture_prefix) noexcept {
        const auto effective_end = end > 0 ? end : layers;
        return layers > 0 && begin >= 0 && begin < layers && effective_end == begin + 1
            && rows >= minimum_rows && rows <= chunk && past >= 0 && past <= context
            && rows <= context - past && padded_kv > 0 && padded_kv <= context
            && outputs == 1 && kind == 0 && !hidden_export && overrides == 0 && !capture_prefix;
    }
};
} // namespace tsg_glm
