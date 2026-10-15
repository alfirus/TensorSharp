// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <unordered_set>
#include <vector>

namespace tsg_embeddings {

// Diagnostic only: graph slicing prevents fusion across category boundaries and
// adds CPU scheduling calls. Never use these durations as paired latency data.
// Plans and tensor storage must remain alive until this object is destroyed.
class cpu_profile_plan {
public:
    enum class category : unsigned { projection, attention, other };
    using tensor_set = std::unordered_set<const ggml_tensor *>;

    static bool enabled() {
        const char * value = std::getenv("TSG_EMBEDDING_CPU_PROFILE");
        return value && std::strcmp(value, "1") == 0;
    }

    static category classify(const ggml_tensor * node, const tensor_set & learned_weights,
                             const tensor_set & attention_nodes) {
        if (node->op == GGML_OP_FLASH_ATTN_EXT || attention_nodes.count(node)) return category::attention;
        if ((node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_CUSTOM) &&
            node->src[0] && learned_weights.count(node->src[0])) return category::projection;
        return category::other;
    }

    cpu_profile_plan(ggml_backend_t backend, ggml_cgraph * graph, const tensor_set & learned_weights,
                     const tensor_set & attention_nodes = {}) : backend_(backend) {
        if (!backend || !ggml_backend_is_cpu(backend) || !graph || graph->n_nodes <= 0) return;
        nodes_ = graph->n_nodes;
        for (int begin = 0; begin < nodes_;) {
            const auto kind = classify(graph->nodes[begin], learned_weights, attention_nodes);
            int end = begin + 1;
            while (end < nodes_ && classify(graph->nodes[end], learned_weights, attention_nodes) == kind) ++end;
            auto part = std::make_unique<segment>();
            part->kind = kind;
            part->backend = backend_;
            part->view = ggml_graph_view(graph, begin, end);
            part->plan = ggml_backend_graph_plan_create(backend_, &part->view);
            segments_.push_back(std::move(part));
            if (!segments_.back()->plan) return;
            node_counts_[unsigned(kind)] += end - begin;
            begin = end;
        }
        valid_ = true;
    }

    ~cpu_profile_plan() = default;
    cpu_profile_plan(const cpu_profile_plan &) = delete;
    cpu_profile_plan & operator=(const cpu_profile_plan &) = delete;
    bool valid() const { return valid_; }

    ggml_status compute(int length, int batch, int sequences) {
        if (!valid_) return GGML_STATUS_FAILED;
        using clock = std::chrono::steady_clock;
        std::array<double, 3> elapsed{};
        std::array<unsigned, 3> calls{};
        const auto start = clock::now();
        for (const auto & part : segments_) {
            const auto before = clock::now();
            const auto status = ggml_backend_graph_plan_compute(backend_, part->plan);
            elapsed[unsigned(part->kind)] += std::chrono::duration<double, std::milli>(clock::now() - before).count();
            ++calls[unsigned(part->kind)];
            if (status != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "EMBEDDING_GGML_CPU_PROFILE_ABORT status=%d\n", int(status));
                return status;
            }
        }
        const double wall = std::chrono::duration<double, std::milli>(clock::now() - start).count();
        std::fprintf(stderr,
            "EMBEDDING_GGML_CPU_PROFILE {\"diagnostic\":true,\"split_graph\":true,\"step\":%llu,"
            "\"length\":%d,\"batch\":%d,\"sequences\":%d,\"nodes\":%d,\"segments\":%zu,\"wall_ms\":%.9f,"
            "\"category_ms\":{\"projection\":%.9f,\"attention\":%.9f,\"other\":%.9f},"
            "\"category_nodes\":{\"projection\":%u,\"attention\":%u,\"other\":%u},"
            "\"category_segments\":{\"projection\":%u,\"attention\":%u,\"other\":%u}}\n",
            ++step_, length, batch, sequences, nodes_, segments_.size(), wall,
            elapsed[0], elapsed[1], elapsed[2], node_counts_[0], node_counts_[1], node_counts_[2],
            calls[0], calls[1], calls[2]);
        return GGML_STATUS_SUCCESS;
    }

private:
    struct segment {
        category kind = category::other;
        ggml_cgraph view{};
        ggml_backend_t backend = nullptr;
        ggml_backend_graph_plan_t plan = nullptr;
        // Also releases already-created plans if a later allocation in the
        // owning constructor throws before cpu_profile_plan exists in full.
        ~segment() { if (plan) ggml_backend_graph_plan_free(backend, plan); }
    };
    ggml_backend_t backend_;
    bool valid_ = false;
    int nodes_ = 0;
    unsigned long long step_ = 0;
    std::array<unsigned, 3> node_counts_{};
    std::vector<std::unique_ptr<segment>> segments_;
};

} // namespace tsg_embeddings
