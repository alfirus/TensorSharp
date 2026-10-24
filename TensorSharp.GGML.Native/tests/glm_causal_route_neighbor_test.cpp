// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Independent past-list and pooled-cache reference. Pure CPU metadata only:
// no GGML/CUDA, native source enrollment, budget callback or model proof.
#include "../glm_causal_route_neighbor.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <list>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
namespace production = tsg_glm_neighbor;
std::size_t checks = 0, cases = 0;
void require(bool value, const char* reason) {
    ++checks;
    if (!value) throw std::runtime_error(reason);
}
production::Config config(std::size_t layers = 1, std::size_t experts = 12, std::size_t routes = 3) {
    production::Config value;
    value.layers = static_cast<std::uint16_t>(layers);
    value.experts = static_cast<std::uint16_t>(experts);
    value.routes = static_cast<std::uint16_t>(routes);
    for (std::size_t i = 0; i < layers; ++i) value.layer_ids[i] = static_cast<std::uint16_t>(3 + i);
    return value;
}
std::unique_ptr<production::State> state(const production::Config& settings) {
    auto value = std::make_unique<production::State>();
    require(production::initialize(*value, settings), "Valid fixed state could not initialize");
    return value;
}
std::vector<unsigned char> snapshot(const production::State& value) {
    const auto* first = reinterpret_cast<const unsigned char*>(&value);
    return {first, first + sizeof(value)};
}
template<class F> void unchanged_refusal(production::State& value, F operation) {
    const auto before = snapshot(value);
    require(!operation(), "Malformed metadata operation succeeded");
    require(std::memcmp(before.data(), &value, sizeof(value)) == 0,
        "Refused metadata operation mutated the fixed state");
}
using Route = std::vector<std::int32_t>;
using Due = std::vector<std::uint64_t>;
struct Expected {
    Due due;
    std::vector<std::uint16_t> distinct;
    std::uint64_t compared = 0, anchor = 0, last = 0, lag = 0, references = 0;
    bool matched = false;
};

// Full unbounded past lists, set intersection and rational comparison.
// No production ring, query, predicted_due or prefer participates in this oracle.
struct PastReference {
    std::size_t layers, experts;
    std::vector<std::vector<Route>> past;
    std::vector<Due> maps;
    explicit PastReference(const production::Config& settings)
        : layers(settings.layers), experts(settings.experts), past(layers), maps(layers, Due(experts, 0)) {}
    Expected query(std::size_t rank, const Route& ids) const {
        Expected result; result.due.assign(experts, 0);
        std::set<std::int32_t> current(ids.begin(), ids.end());
        for (const auto expert : ids)
            if (std::find(result.distinct.begin(), result.distinct.end(), expert) == result.distinct.end())
                result.distinct.push_back(static_cast<std::uint16_t>(expert));
        const auto& history = past.at(rank);
        const auto t = history.size();
        const auto first = t > 64 ? t - 64 : 0;
        std::uint64_t best_numerator = 0, best_denominator = 1;
        for (std::size_t candidate = first; candidate + 8 < t; ++candidate) {
            ++result.compared;
            std::set<std::int32_t> old(history[candidate].begin(), history[candidate].end());
            std::vector<std::int32_t> common, united;
            std::set_intersection(current.begin(), current.end(), old.begin(), old.end(), std::back_inserter(common));
            std::set_union(current.begin(), current.end(), old.begin(), old.end(), std::back_inserter(united));
            if (common.size() < 2) continue;
            const auto numerator = static_cast<std::uint64_t>(common.size());
            const auto denominator = static_cast<std::uint64_t>(united.size());
            if (!result.matched || numerator * best_denominator > best_numerator * denominator
                || (numerator * best_denominator == best_numerator * denominator && candidate > result.anchor)) {
                result.matched = true; result.anchor = candidate;
                best_numerator = numerator; best_denominator = denominator;
            }
        }
        if (result.matched) {
            result.last = result.anchor + 8; result.lag = t - result.last;
            for (std::size_t distance = 1; distance <= 8; ++distance) {
                const auto& old = history[result.anchor + distance];
                std::set<std::int32_t> unique(old.begin(), old.end());
                result.references += unique.size();
                for (const auto expert : unique)
                    if (!result.due[expert]) result.due[expert] = layers * (t + distance) + rank;
            }
        }
        return result;
    }
    std::uint64_t predicted(const Expected& current, std::size_t current_rank,
        std::size_t candidate_rank, std::size_t expert) const {
        const auto now = layers * past[current_rank].size() + current_rank;
        const auto due = candidate_rank == current_rank ? current.due[expert] : maps[candidate_rank][expert];
        return due != 0 && due > now ? due : production::unknown_due;
    }
    void complete(std::size_t rank, const Route& ids, const Expected& current) {
        past[rank].push_back(ids); maps[rank] = current.due;
    }
};

void compare_query(production::State& actual, PastReference& oracle, std::size_t rank, const Route& ids) {
    const auto expected = oracle.query(rank, ids);
    require(production::prepare(actual, rank, ids.data(), ids.size()), "Valid route preparation failed");
    const auto& query = actual.pending;
    require(query.matched == expected.matched && query.compared == expected.compared,
        "Match/candidate count differs from full-past reference");
    require(query.count == expected.distinct.size(), "Duplicate routes changed similarity cardinality");
    for (std::size_t i = 0; i < expected.distinct.size(); ++i)
        require(query.experts[i] == expected.distinct[i], "Distinct route changed original occurrence order");
    if (expected.matched)
        require(query.anchor == expected.anchor && query.suffix_last == expected.last
            && query.min_completed_lag == expected.lag && query.suffix_references == expected.references,
            "Anchor/suffix completion lag differs from independent rational history");
    for (std::size_t r = 0; r < oracle.layers; ++r) for (std::size_t e = 0; e < oracle.experts; ++e)
        require(production::predicted_due(actual, r, e) == oracle.predicted(expected, rank, r, e),
            "Source-layer prediction was rebased, borrowed or expired incorrectly");
    const auto old_completed = actual.layer[rank].completed;
    require(old_completed == oracle.past[rank].size(), "Prepared-only route became a completed label");
    require(production::commit(actual, actual.epoch, rank), "Valid completed route could not commit");
    oracle.complete(rank, ids, expected);
    require(actual.layer[rank].completed == old_completed + 1, "Commit did not append exactly one visit");
}

void malformed_whole_row_nonmutation() {
    ++cases;
    auto actual = state(config());
    auto invalid = config(); invalid.layers = 0;
    unchanged_refusal(*actual, [&] { return production::initialize(*actual, invalid); });
    invalid = config(2); invalid.layer_ids[1] = invalid.layer_ids[0];
    unchanged_refusal(*actual, [&] { return production::initialize(*actual, invalid); });
    invalid = config(); invalid.routes = 9;
    unchanged_refusal(*actual, [&] { return production::initialize(*actual, invalid); });
    invalid = config(); invalid.experts = 289;
    unchanged_refusal(*actual, [&] { return production::initialize(*actual, invalid); });
    Route valid{0, 1, 2}, bad_last{0, 1, 12}, negative{0, 1, -1};
    unchanged_refusal(*actual, [&] { return production::prepare(*actual, 0, valid.data(), valid.size()); });
    require(production::begin_token(*actual), "Begin failed");
    unchanged_refusal(*actual, [&] { return production::begin_token(*actual); });
    unchanged_refusal(*actual, [&] { return production::prepare(*actual, 1, valid.data(), valid.size()); });
    unchanged_refusal(*actual, [&] { return production::prepare(*actual, 0, nullptr, valid.size()); });
    unchanged_refusal(*actual, [&] { return production::prepare(*actual, 0, valid.data(), 2); });
    unchanged_refusal(*actual, [&] { return production::prepare(*actual, 0, bad_last.data(), bad_last.size()); });
    unchanged_refusal(*actual, [&] { return production::prepare(*actual, 0, negative.data(), negative.size()); });
    require(production::prepare(*actual, 0, valid.data(), valid.size()), "Valid pending row failed");
    unchanged_refusal(*actual, [&] { return production::prepare(*actual, 0, valid.data(), valid.size()); });
    unchanged_refusal(*actual, [&] { return production::commit(*actual, actual->epoch + 1, 0); });
    unchanged_refusal(*actual, [&] { return production::commit(*actual, actual->epoch, 1); });
    unchanged_refusal(*actual, [&] { return production::finish_token(*actual); });
    require(actual->layer[0].completed == 0, "Failed/pending helper taught a route");
    require(production::abort_token(*actual), "Abort failed");
    require(!actual->active && !actual->pending.prepared && actual->layer[0].completed == 0,
        "Aborted row retained training state");
}

void strict_suffix_and_many_wraps() {
    ++cases;
    const auto settings = config(); auto actual = state(settings); PastReference oracle(settings);
    const Route ids{0, 1, 2};
    for (std::size_t t = 0; t < 257; ++t) {
        require(production::begin_token(*actual), "Begin failed");
        const auto expected = oracle.query(0, ids);
        if (t <= 8) require(!expected.matched && expected.compared == 0, "Uncompleted suffix was eligible");
        if (t == 9) require(expected.matched && expected.anchor == 0 && expected.lag == 1,
            "First strictly completed suffix boundary differs");
        if (t >= 64) require(expected.compared == 56 && expected.anchor == t - 9,
            "Ring window or newest exact-ratio tie differs");
        compare_query(*actual, oracle, 0, ids);
        require(production::finish_token(*actual), "Finish failed");
    }
    require(actual->layer[0].completed == 257 && actual->cycle == 257, "Completed repeats collapsed into one visit");
}

void oldest_retained_anchor_and_no_match_clear() {
    ++cases;
    const auto settings = config(1, 8, 2); auto actual = state(settings); PastReference oracle(settings);
    for (std::size_t t = 0; t < 66; ++t) {
        const Route ids = t == 0 || t >= 64 ? Route{0, 1} : Route{2, 3};
        require(production::begin_token(*actual), "Begin failed");
        const auto expected = oracle.query(0, ids);
        if (t == 64) require(expected.matched && expected.anchor == 0 && expected.lag == 56,
            "Oldest retained anchor or completed lag56 was lost");
        if (t == 65) require(!expected.matched, "Expired oldest anchor remained eligible");
        compare_query(*actual, oracle, 0, ids);
        if (t == 65) for (const auto due : actual->layer[0].due)
            require(due == 0, "No-match query preserved a stale own-layer prediction");
        require(production::finish_token(*actual), "Finish failed");
    }
}

void rational_matching_and_duplicate_cardinality() {
    ++cases;
    const auto settings = config(1, 8, 3); auto actual = state(settings); PastReference oracle(settings);
    for (std::size_t t = 0; t < 17; ++t) {
        Route ids = t == 0 ? Route{0, 1, 0} : (t <= 2 || t == 16 ? Route{0, 1, 2} : Route{3, 4, 5});
        require(production::begin_token(*actual), "Begin failed");
        if (t == 16) {
            const auto query = oracle.query(0, ids);
            require(query.matched && query.anchor == 2,
                "Exact rational score/newest tie did not outrank partial intersection");
        }
        compare_query(*actual, oracle, 0, ids);
        require(production::finish_token(*actual), "Finish failed");
    }
    require(production::reset_history(*actual), "Quiescent reset failed");
    PastReference one(settings);
    for (std::size_t t = 0; t < 12; ++t) {
        const Route ids{0, 0, 0};
        require(production::begin_token(*actual), "Begin failed");
        require(!one.query(0, ids).matched, "One intersection was admitted despite minimum2");
        compare_query(*actual, one, 0, ids);
        require(production::finish_token(*actual), "Finish failed");
    }
}

using Key = std::pair<std::size_t, std::int32_t>;
struct ActualSlot { Key key{}; std::uint64_t age = 0; bool occupied = false; };
struct PooledReference {
    std::size_t capacity;
    std::list<Key> recency;
    explicit PooledReference(std::size_t bytes_as_slots) : capacity(bytes_as_slots) {}
    std::vector<Key> visit(const PastReference& history, const Expected& pending, std::size_t rank, const Route& ids) {
        std::set<Key> protected_keys;
        for (const auto expert : ids) protected_keys.emplace(rank, expert);
        std::vector<Key> missing;
        for (const auto expert : ids) {
            const Key key{rank, expert};
            if (std::find(recency.begin(), recency.end(), key) != recency.end()) continue;
            if (recency.size() == capacity) {
                auto victim = recency.end(); std::uint64_t farthest = 0;
                for (auto old = recency.begin(); old != recency.end(); ++old) {
                    if (protected_keys.count(*old)) continue;
                    const auto due = history.predicted(pending, rank, old->first, old->second);
                    // Oldest-first list iteration supplies the independent LRU tie.
                    if (victim == recency.end() || due > farthest) { victim = old; farthest = due; }
                }
                require(victim != recency.end(), "Reference had no unprotected victim");
                recency.erase(victim);
            }
            recency.push_back(key); missing.push_back(key);
        }
        for (const auto expert : ids) {
            const Key key{rank, expert}; recency.remove(key); recency.push_back(key);
        }
        return missing;
    }
};
void pooled_differential(std::size_t capacity) {
    ++cases;
    const auto settings = config(3, 12, 3); auto actual = state(settings); PastReference history(settings);
    PooledReference oracle(capacity); std::vector<ActualSlot> slots(capacity);
    std::uint64_t clock = 0; std::size_t matching_rows = 0, missed_rows = 0;
    for (std::size_t t = 0; t < 173; ++t) {
        require(production::begin_token(*actual), "Begin failed");
        for (std::size_t rank = 0; rank < settings.layers; ++rank) {
            // One immutable source per rank. Identical local expert IDs across
            // ranks stay distinct; repeated routes and duplicates are intentional.
            const auto base = static_cast<std::int32_t>((t % 17 + rank * 3) % 10);
            Route ids{base, (base + 1) % 12, (t % 7 == 0 ? base : (base + 3) % 12)};
            const auto expected = history.query(rank, ids);
            if (expected.matched) ++matching_rows; else ++missed_rows;
            require(production::prepare(*actual, rank, ids.data(), ids.size()), "Pooled preparation failed");
            require(actual->pending.matched == expected.matched && actual->pending.anchor == expected.anchor,
                "Pooled matcher differs from full-past query");
            for (std::size_t r = 0; r < settings.layers; ++r) for (std::size_t e = 0; e < settings.experts; ++e)
                require(production::predicted_due(*actual, r, e) == history.predicted(expected, rank, r, e),
                    "Pooled due timestamp differs from absolute own-layer reference");
            const auto expected_misses = oracle.visit(history, expected, rank, ids);
            std::set<Key> protected_keys;
            for (const auto expert : ids) protected_keys.emplace(rank, expert);
            std::vector<Key> actual_misses;
            for (const auto expert : ids) {
                const Key key{rank, expert};
                auto found = std::find_if(slots.begin(), slots.end(), [&](const auto& slot) {
                    return slot.occupied && slot.key == key;
                });
                if (found != slots.end()) continue;
                std::size_t chosen = capacity;
                for (std::size_t i = 0; i < capacity; ++i) {
                    if (slots[i].occupied && protected_keys.count(slots[i].key)) continue;
                    const auto due = slots[i].occupied
                        ? production::predicted_due(*actual, slots[i].key.first, slots[i].key.second) : production::unknown_due;
                    const auto prior_due = chosen == capacity || !slots[chosen].occupied ? production::unknown_due
                        : production::predicted_due(*actual, slots[chosen].key.first, slots[chosen].key.second);
                    if (chosen == capacity || production::prefer(!slots[i].occupied, due, slots[i].age, i,
                            !slots[chosen].occupied, prior_due, slots[chosen].age, chosen)) chosen = i;
                }
                require(chosen < capacity, "Production had no complete-union-safe victim");
                slots[chosen].key = key; slots[chosen].occupied = true; actual_misses.push_back(key);
            }
            for (const auto expert : ids) {
                const Key key{rank, expert};
                const auto found = std::find_if(slots.begin(), slots.end(), [&](const auto& slot) {
                    return slot.occupied && slot.key == key;
                });
                require(found != slots.end(), "A current hit/newly admitted key was evicted");
                found->age = ++clock;
            }
            require(actual_misses == expected_misses, "Pooled forced misses differ from independent past-list cache");
            std::vector<std::pair<std::uint64_t, Key>> ordered;
            for (const auto& slot : slots) if (slot.occupied) ordered.emplace_back(slot.age, slot.key);
            std::sort(ordered.begin(), ordered.end());
            require(ordered.size() == oracle.recency.size(), "Pooled occupancy differs");
            auto old = oracle.recency.begin();
            for (const auto& item : ordered) { require(item.second == *old, "Original final LRU occurrence order changed"); ++old; }
            require(production::commit(*actual, actual->epoch, rank), "Pooled completed commit failed");
            history.complete(rank, ids, expected);
        }
        require(production::finish_token(*actual), "Pooled finish failed");
    }
    require(matching_rows > 0 && missed_rows > 0, "Pooled fixture did not exercise both matched and unknown policy");
}

void independent_victim_ties_and_absolute_expiry() {
    ++cases;
    require(production::prefer(true, 1, 100, 9, false, production::unknown_due, 0, 0), "Empty slot did not win");
    require(production::prefer(false, production::unknown_due, 100, 9, false, 1000, 0, 0), "Unknown victim did not win");
    require(production::prefer(false, 99, 7, 9, false, 98, 0, 0), "Farthest due did not win");
    require(production::prefer(false, 99, 7, 9, false, 99, 8, 0), "Original LRU tie changed");
    require(production::prefer(false, 99, 7, 2, false, 99, 7, 3), "Increasing slot tie changed");
    require(!production::prefer(false, 99, 7, 3, false, 99, 7, 2), "Reversed equal-age slot tie won");
    auto actual = state(config(3));
    actual->cycle = 100;
    for (auto& layer : actual->layer) layer.completed = 100;
    actual->active = true; actual->next_rank = 1;
    actual->pending.prepared = true; actual->pending.rank = 1; actual->pending.cycle = 100;
    actual->layer[0].due[4] = 300;
    actual->layer[0].due[5] = 301;
    actual->layer[0].due[6] = 302;
    require(production::predicted_due(*actual, 0, 4) == production::unknown_due,
        "Stale other-layer due was rolled forward");
    require(production::predicted_due(*actual, 0, 5) == production::unknown_due,
        "Due at current global ordinal was not expired");
    require(production::predicted_due(*actual, 0, 6) == 302,
        "Future other-layer due was normalized to the current layer");
    actual->pending.due[6] = 999;
    require(production::predicted_due(*actual, 0, 6) == 302,
        "Current-layer prediction overwrote another layer");
    require(production::predicted_due(*actual, 1, 6) == 999, "Current pending map was not used");
}

void reset_partial_token_and_overflow() {
    ++cases;
    const auto settings = config(3); auto actual = state(settings); const Route ids{0, 1, 2};
    require(production::begin_token(*actual), "Begin failed");
    require(production::prepare(*actual, 0, ids.data(), ids.size()), "Prepare failed");
    require(production::commit(*actual, actual->epoch, 0), "Commit failed");
    require(production::prepare(*actual, 1, ids.data(), ids.size()), "Prepare failed");
    const auto retired_epoch = actual->epoch;
    require(production::reset_history(*actual) && actual->epoch == retired_epoch + 1 && actual->skip_token,
        "Partial retirement did not create a fresh nonlearning epoch");
    unchanged_refusal(*actual, [&] { return production::commit(*actual, retired_epoch, 1); });
    for (std::size_t rank = 1; rank < settings.layers; ++rank) {
        require(production::prepare(*actual, rank, ids.data(), ids.size()), "Partial suffix prepare failed");
        require(!actual->pending.matched, "Partial token borrowed history");
        require(production::commit(*actual, actual->epoch, rank), "Partial suffix commit failed");
    }
    require(production::finish_token(*actual) && actual->cycle == 0, "Partial token became a learned cycle");
    for (const auto& layer : actual->layer) require(layer.completed == 0, "Partial layer history survived");
    PastReference oracle(settings);
    require(production::begin_token(*actual), "Fresh complete begin failed");
    for (std::size_t rank = 0; rank < settings.layers; ++rank) compare_query(*actual, oracle, rank, ids);
    require(production::finish_token(*actual) && actual->cycle == 1, "Fresh complete token did not resume learning");
    require(production::begin_token(*actual), "Begin failed");
    actual->cycle = production::unknown_due;
    unchanged_refusal(*actual, [&] { return production::prepare(*actual, 0, ids.data(), ids.size()); });
    actual->epoch = production::unknown_due;
    unchanged_refusal(*actual, [&] { return production::reset_history(*actual); });
}

void maximum_geometry_and_prepared_not_completed() {
    ++cases;
    const auto settings = config(42, 288, 8); auto actual = state(settings);
    require(sizeof(*actual) <= 1024 * 1024 && actual->slot_due.size() == 12960,
        "Fixed source/pool slab bound differs from declared profile");
    require(production::begin_token(*actual), "Maximum begin failed");
    const Route ids{280, 281, 282, 283, 284, 285, 286, 287};
    for (std::size_t rank = 0; rank < 42; ++rank) {
        require(production::prepare(*actual, rank, ids.data(), ids.size()), "Maximum source ID/route failed");
        require(actual->layer[rank].completed == 0 && !actual->pending.matched,
            "Prepared maximum route trained history early");
        require(production::commit(*actual, actual->epoch, rank), "Maximum commit failed");
    }
    require(production::finish_token(*actual), "Maximum finish failed");
    require(production::reset_history(*actual) && actual->cycle == 0, "Wide-prefill reset simulation failed");
    for (const auto& layer : actual->layer) require(layer.completed == 0, "Reset retained a completed history row");
}

void exhaustive_tiny_continuations_and_prefix() {
    ++cases;
    const auto settings = config(1, 4, 2); auto prefix = state(settings); PastReference history(settings);
    const std::array<Route, 6> choices{{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};
    for (std::size_t row = 0; row < 24; ++row) {
        require(production::begin_token(*prefix), "Prefix begin failed");
        compare_query(*prefix, history, 0, choices[(row * 5 + row / 7) % choices.size()]);
        require(production::finish_token(*prefix), "Prefix finish failed");
    }
    const auto immutable_prefix = snapshot(*prefix);
    const auto expected_first = history.query(0, choices[0]);
    for (const auto& second : choices) for (const auto& third : choices) {
        auto branch = std::make_unique<production::State>(*prefix);
        auto oracle = history;
        require(production::begin_token(*branch), "Continuation begin failed");
        require(production::prepare(*branch, 0, choices[0].data(), choices[0].size()), "First continuation failed");
        require(branch->pending.matched == expected_first.matched && branch->pending.anchor == expected_first.anchor,
            "A future continuation altered its identical causal prefix");
        require(production::commit(*branch, branch->epoch, 0) && production::finish_token(*branch),
            "First continuation commit failed");
        oracle.complete(0, choices[0], expected_first);
        for (const auto* next : {&second, &third}) {
            require(production::begin_token(*branch), "Continuation begin failed");
            compare_query(*branch, oracle, 0, *next);
            require(production::finish_token(*branch), "Continuation finish failed");
        }
        require(std::memcmp(immutable_prefix.data(), prefix.get(), sizeof(*prefix)) == 0,
            "Independent hypothetical future mutated the retained prefix");
    }
}
} // namespace

int main() {
    try {
        malformed_whole_row_nonmutation(); strict_suffix_and_many_wraps();
        oldest_retained_anchor_and_no_match_clear(); rational_matching_and_duplicate_cardinality();
        pooled_differential(3); pooled_differential(7); pooled_differential(17);
        independent_victim_ties_and_absolute_expiry(); reset_partial_token_and_overflow();
        maximum_geometry_and_prepared_not_completed(); exhaustive_tiny_continuations_and_prefix();
        std::printf("[GLM-CAUSAL-ROUTE-NEIGHBOR-CPU] PASS checks=%zu cases=%zu scope=pure_fixed_metadata_independent_past_list_and_pooled_lru_not_native_lifecycle_or_cuda\n", checks, cases);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[GLM-CAUSAL-ROUTE-NEIGHBOR-CPU] FAIL checks=%zu cases=%zu error=%s\n", checks, cases, error.what());
        return 1;
    }
}
