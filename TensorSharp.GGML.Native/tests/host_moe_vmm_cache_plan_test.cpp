// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Pure bounded slot/capacity policy. No CUDA/GGML/model/physical resource proof.
#include "ggml_ops_host_moe_vmm_cache_plan.h"
#include <cstdio>
#include <set>
#include <vector>

namespace {
std::size_t checks = 0;
void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
template<class F> void refused(F operation) {
    bool caught = false;
    try { operation(); } catch (const std::invalid_argument&) { caught = true; }
    require(caught, "Malformed compaction/search input accepted");
}
struct Tag {
    int expert = -1;
    unsigned source = 0;
    std::uint64_t last = 0;
    bool prefetched = false;
};

void compaction(std::size_t current, std::size_t target, bool protect_one) {
    std::vector<int> experts(current);
    std::vector<std::uint64_t> last(current);
    std::vector<std::uint8_t> protected_slots(current);
    std::vector<Tag> before(current), actual(current);
    std::size_t hot = 0, protected_index = current;
    for (std::size_t i = 0; i != current; ++i) {
        // Equal expert IDs from different immutable source identities remain
        // different tags. Timestamps deliberately include ties and old highs.
        experts[i] = i % 5 == 2 ? -1 : static_cast<int>(i % 3);
        last[i] = (i * 7) % 11;
        actual[i] = before[i] = {experts[i], static_cast<unsigned>(i + 100), last[i], (i % 2) != 0};
        if (experts[i] >= 0) { ++hot; protected_index = i; }
    }
    if (protect_one && protected_index != current) protected_slots[protected_index] = 1;
    tsg::HostMoeVmmSlotPlan plan;
    plan.prepare(current, target, experts.data(), last.data(), protected_slots.data());
    require(plan.kept_hot() == (std::min)(hot, target), "Survivor count differs from admitted capacity");
    require(plan.discarded_count() == hot - plan.kept_hot(), "Discard count differs from original occupancy");
    std::set<unsigned> dropped, moved_sources, moved_destinations;
    for (std::size_t i = 0; i != plan.discarded_count(); ++i) {
        const auto slot = plan.discarded(i);
        require(slot < current && actual[slot].expert >= 0 && !protected_slots[slot]
            && dropped.insert(slot).second, "Discard is invalid, repeated or current-union protected");
        actual[slot].expert = -1;
    }
    for (std::size_t i = 0; i != plan.move_count(); ++i) {
        const auto move = plan.move(i);
        require(move.from >= target && move.from < current && move.to < target
            && actual[move.from].expert >= 0 && actual[move.to].expert == -1
            && moved_sources.insert(move.from).second && moved_destinations.insert(move.to).second,
            "Move overwrites a valid slot or does not compact a retained high tag into an empty low hole");
        actual[move.to] = actual[move.from]; actual[move.from].expert = -1;
    }
    std::set<unsigned> survivors;
    for (std::size_t i = 0; i != current; ++i) if (actual[i].expert >= 0) {
        const auto original = static_cast<std::size_t>(actual[i].source - 100);
        require(i < target && original < current && survivors.insert(static_cast<unsigned>(original)).second,
            "A survivor remains beyond the target prefix or duplicates another source-qualified tag");
        require(actual[i].expert == before[original].expert && actual[i].last == before[original].last
            && actual[i].prefetched == before[original].prefetched,
            "Compaction changed the complete source/expert/LRU/prefetch metadata");
    }
    require(survivors.size() == plan.kept_hot(), "A required survivor was lost during applying the move plan");
    if (protect_one && protected_index != current)
        require(survivors.count(static_cast<unsigned>(protected_index)) == 1, "The current known union was evicted");
    // Rank property, independent of the planner's internal sorting/move layout:
    // every nonprotected survivor must outrank every discarded past-only tag.
    for (const auto kept : survivors) if (!protected_slots[kept]) for (const auto lost : dropped)
        require(before[kept].last > before[lost].last
            || (before[kept].last == before[lost].last && kept > lost),
            "A newer source-qualified hot key was discarded before an older key");
    if (target >= current) require(!plan.move_count() && dropped.empty(), "Growth unnecessarily moved or evicted hot payload");
}

void invalid_inputs() {
    tsg::HostMoeVmmSlotPlan plan;
    int experts[3] = {0, 1, -1}; std::uint64_t used[3] = {1, 2, 0};
    std::uint8_t protected_slots[3] = {1, 1, 0};
    refused([&] { plan.prepare(0, 1, experts, used); });
    refused([&] { plan.prepare(3, 0, experts, used); });
    refused([&] { plan.prepare(8193, 1, experts, used); });
    refused([&] { plan.prepare(3, 8193, experts, used); });
    refused([&] { plan.prepare(3, 1, nullptr, used); });
    refused([&] { plan.prepare(3, 1, experts, nullptr); });
    refused([&] { plan.prepare(3, 1, experts, used, protected_slots); });
    protected_slots[0] = protected_slots[1] = 0; protected_slots[2] = 1;
    refused([&] { plan.prepare(3, 2, experts, used, protected_slots); });
    protected_slots[2] = 0; protected_slots[0] = 2;
    refused([&] { plan.prepare(3, 2, experts, used, protected_slots); });
    experts[1] = -2;
    refused([&] { plan.prepare(3, 2, experts, used); });
}

void rounded_capacity() {
    const auto bytes = [](std::size_t capacity) {
        std::size_t result = 0;
        for (const auto stride : {7u, 11u, 13u}) result += ((capacity * stride + 15) / 16) * 16;
        return result;
    };
    for (std::size_t scratch : {0u, 13u, 127u}) for (std::size_t minimum : {1u, 3u, 8u}) {
        for (std::size_t quota = 0; quota != 1500; ++quota) {
            std::size_t expected = 0;
            for (std::size_t candidate = minimum; candidate <= 40; ++candidate)
                if (scratch <= quota && bytes(candidate) <= quota - scratch) expected = candidate;
            const auto actual = tsg::host_moe_vmm_capacity(quota, scratch, minimum, 40, bytes);
            require(actual == expected, "Rounded capacity search does not maximize the existing admitted quota");
        }
    }
    refused([&] { tsg::host_moe_vmm_capacity(100, 0, 0, 40, bytes); });
    refused([&] { tsg::host_moe_vmm_capacity(100, 0, 4, 3, bytes); });
    refused([&] { tsg::host_moe_vmm_capacity(100, 0, 1, 8193, bytes); });
}
} // namespace

int main() {
    try {
        for (std::size_t current = 1; current <= 32; ++current)
            for (std::size_t target = 1; target <= 32; ++target) {
                compaction(current, target, false); compaction(current, target, true);
            }
        invalid_inputs(); rounded_capacity();
        std::printf("PASS VMM cache policy: %zu CPU checks; no CUDA, GGML, payload or model work\n", checks);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL VMM cache policy after %zu checks: %s\n", checks, error.what()); return 1;
    }
}
