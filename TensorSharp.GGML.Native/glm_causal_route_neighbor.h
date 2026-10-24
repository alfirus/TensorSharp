// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace tsg_glm_neighbor {
constexpr std::size_t max_layers = 42, max_experts = 288, max_routes = 8;
constexpr std::size_t history_rows = 64, suffix_rows = 8, minimum_intersection = 2;
constexpr std::size_t max_slots = 45 * max_experts;
constexpr std::uint64_t unknown_due = std::numeric_limits<std::uint64_t>::max();

struct Config {
    std::uint16_t layers = 0, experts = 0, routes = 0;
    std::array<std::uint16_t, max_layers> layer_ids{};
};
inline bool valid_config(const Config& config) noexcept {
    if (!config.layers || config.layers > max_layers || !config.experts || config.experts > max_experts
        || !config.routes || config.routes > max_routes || config.routes > config.experts) return false;
    for (std::size_t i = 0; i < config.layers; ++i)
        for (std::size_t j = 0; j < i; ++j) if (config.layer_ids[i] == config.layer_ids[j]) return false;
    return true;
}
struct HistoricalRoute {
    std::uint64_t ordinal = 0;
    std::array<std::uint16_t, max_routes> experts{};
    std::uint8_t count = 0;
};
struct Layer {
    std::array<HistoricalRoute, history_rows> history{};
    std::array<std::uint64_t, max_experts> due{};
    std::uint64_t completed = 0;
};
struct Query {
    std::array<std::uint64_t, max_experts> due{};
    std::array<std::uint16_t, max_routes> experts{};
    std::uint64_t epoch = 0, cycle = 0, anchor = 0, suffix_last = 0;
    std::uint64_t compared = 0, min_completed_lag = 0, suffix_references = 0;
    std::uint16_t rank = 0;
    std::uint8_t count = 0;
    bool prepared = false, matched = false;
};
struct State {
    Config config{};
    std::array<Layer, max_layers> layer{};
    Query pending{};
    std::array<std::uint64_t, max_slots> slot_due{};
    std::uint64_t epoch = 0, cycle = 0;
    std::uint16_t next_rank = 0;
    bool active = false, skip_token = false;
};
static_assert(std::is_trivially_copyable<State>::value, "Fixed neighbor payload must not own heap members");
static_assert(sizeof(State) > 0 && sizeof(State) <= 1024 * 1024, "Exact Host slab must fit profile1 bound");

inline bool initialize(State& state, const Config& config) noexcept {
    if (!valid_config(config)) return false;
    state.config = config;
    for (auto& layer : state.layer) layer = {};
    state.pending = {};
    state.slot_due.fill(unknown_due);
    state.epoch = 1; state.cycle = 0; state.next_rank = 0;
    state.active = state.skip_token = false;
    return true;
}
// Existing in-flight caller keeps its old pending epoch. A reset deliberately
// invalidates that pending proof; it never converts it into a completed label.
inline bool reset_history(State& state) noexcept {
    if (!valid_config(state.config) || state.epoch == unknown_due) return false;
    const bool partial = state.active && state.next_rank != 0;
    for (auto& layer : state.layer) layer = {};
    state.pending = {};
    state.slot_due.fill(unknown_due);
    ++state.epoch; state.cycle = 0;
    state.skip_token = partial;
    if (!state.active) state.next_rank = 0;
    return true;
}
inline bool begin_token(State& state) noexcept {
    if (!valid_config(state.config) || state.active || state.pending.prepared) return false;
    state.active = true; state.skip_token = false; state.next_rank = 0;
    return true;
}
inline bool validate_row(const State& state, std::size_t rank, const std::int32_t* ids,
                         std::size_t count) noexcept {
    if (!state.active || state.pending.prepared || rank != state.next_rank || rank >= state.config.layers
        || !ids || count != state.config.routes || count > max_routes) return false;
    for (std::size_t i = 0; i < count; ++i)
        if (ids[i] < 0 || ids[i] >= state.config.experts) return false;
    return true;
}
inline bool prepare(State& state, std::size_t rank, const std::int32_t* ids,
                    std::size_t count) noexcept {
    // Validate the ENTIRE route/order and timestamp bound before any mutation.
    if (!validate_row(state, rank, ids, count)
        || state.cycle > (unknown_due - max_layers) / state.config.layers - suffix_rows) return false;
    auto& layer = state.layer[rank];
    if (!state.skip_token && layer.completed != state.cycle) return false;
    std::array<std::uint8_t, max_experts> current{};
    Query query{}; query.rank = static_cast<std::uint16_t>(rank);
    query.epoch = state.epoch; query.cycle = state.cycle;
    for (std::size_t i = 0; i < count; ++i) if (!current[static_cast<std::size_t>(ids[i])]) {
        current[static_cast<std::size_t>(ids[i])] = 1;
        query.experts[query.count++] = static_cast<std::uint16_t>(ids[i]);
    }
    std::uint64_t best_overlap = 0, best_union = 1;
    if (!state.skip_token) {
        const auto t = layer.completed, first = t > history_rows ? t - history_rows : 0;
        for (auto i = first; i + suffix_rows < t; ++i) {
            const auto& prior = layer.history[i % history_rows];
            if (!prior.count || prior.ordinal != i) return false;
            ++query.compared;
            std::uint64_t overlap = 0;
            for (std::size_t j = 0; j < prior.count; ++j) overlap += current[prior.experts[j]];
            const auto united = static_cast<std::uint64_t>(query.count + prior.count) - overlap;
            if (overlap < minimum_intersection) continue;
            if (!query.matched || overlap * best_union > best_overlap * united
                || (overlap * best_union == best_overlap * united && i > query.anchor)) {
                query.matched = true; query.anchor = i;
                best_overlap = overlap; best_union = united;
            }
        }
        if (query.matched) {
            query.suffix_last = query.anchor + suffix_rows;
            if (!(query.suffix_last < t)) return false;
            query.min_completed_lag = t - query.suffix_last;
            for (std::size_t d = 1; d <= suffix_rows; ++d) {
                const auto& old = layer.history[(query.anchor + d) % history_rows];
                if (!old.count || old.ordinal != query.anchor + d) return false;
                const auto due = state.config.layers * (state.cycle + d) + rank;
                query.suffix_references += old.count;
                for (std::size_t j = 0; j < old.count; ++j)
                    if (!query.due[old.experts[j]]) query.due[old.experts[j]] = due;
            }
        }
    }
    query.prepared = true;
    state.pending = query;
    return true;
}
inline std::uint64_t predicted_due(const State& state, std::size_t rank, std::size_t expert) noexcept {
    if (!state.pending.prepared || rank >= state.config.layers || expert >= state.config.experts) return unknown_due;
    const auto due = rank == state.pending.rank ? state.pending.due[expert] : state.layer[rank].due[expert];
    const auto now = state.config.layers * state.pending.cycle + state.pending.rank;
    return due && due > now ? due : unknown_due;
}
inline bool prefer(bool empty_a, std::uint64_t due_a, std::uint64_t age_a, std::size_t slot_a,
                   bool empty_b, std::uint64_t due_b, std::uint64_t age_b, std::size_t slot_b) noexcept {
    if (empty_a != empty_b) return empty_a;
    if (!empty_a && due_a != due_b) return due_a > due_b;
    return age_a < age_b || (age_a == age_b && slot_a < slot_b);
}
inline bool commit(State& state, std::uint64_t expected_epoch, std::size_t rank) noexcept {
    const auto& query = state.pending;
    if (!state.active || !query.prepared || expected_epoch != state.epoch || query.epoch != expected_epoch
        || rank != state.next_rank || query.rank != rank || query.cycle != state.cycle) return false;
    if (!state.skip_token) {
        auto& layer = state.layer[rank];
        if (layer.completed != state.cycle || layer.completed == unknown_due) return false;
        auto& row = layer.history[layer.completed % history_rows];
        row.ordinal = layer.completed; row.experts = query.experts; row.count = query.count;
        layer.due = query.due; // A no-match replaces old confidence with all unknown.
        ++layer.completed;
    }
    state.pending = {};
    ++state.next_rank;
    return true;
}
inline bool finish_token(State& state) noexcept {
    if (!state.active || state.pending.prepared || state.next_rank != state.config.layers
        || state.cycle == unknown_due) return false;
    if (!state.skip_token) ++state.cycle;
    state.active = state.skip_token = false; state.next_rank = 0;
    return true;
}
inline bool abort_token(State& state) noexcept {
    state.active = false; state.next_rank = 0;
    return reset_history(state);
}
} // namespace tsg_glm_neighbor
