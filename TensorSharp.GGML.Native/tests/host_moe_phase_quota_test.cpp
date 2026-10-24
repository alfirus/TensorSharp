// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "../ggml_ops_host_moe_phase_quota.h"
#include <cstdio>
#include <cstdlib>

int main() {
    int checks = 0;
    const auto require = [&](bool value) {
        if (++checks && !value) { std::fprintf(stderr, "FAIL phase quota check %d\n", checks); std::exit(1); }
    };
    const auto key = [](int type) { return tsg::HostMoeGroupKey{{type, type, 18, 4096, 2048, 8, 288, 12960}, 0}; };
    const std::vector<tsg::HostMoePhaseGroupMinimum> groups{{key(16), 30, 300}, {key(19), 20, 200}, {key(23), 40, 400}};
    tsg::HostMoePhaseQuotaPlan plan(groups);
    require(plan.narrow_total() == 90 && plan.minimum() == 450 && plan.size() == 3);
    for (std::size_t budget = 450; budget < 551; ++budget) {
        std::size_t sum = 0;
        for (const auto& group : groups) {
            const auto narrow = plan.quota(group.key, budget, false), wide = plan.quota(group.key, budget, true);
            require(narrow >= group.narrow_bytes && wide >= group.wide_bytes);
            require(wide + plan.narrow_total() - group.narrow_bytes == budget);
            sum += narrow;
        }
        require(sum == budget);
    }
    auto reversed = groups; std::reverse(reversed.begin(), reversed.end());
    tsg::HostMoePhaseQuotaPlan reverse_plan(reversed);
    for (const auto& group : groups) for (bool wide : {false, true})
        require(plan.quota(group.key, 451, wide) == reverse_plan.quota(group.key, 451, wide));
    for (auto budget : {std::size_t(0), std::size_t(449)}) for (const auto& group : groups)
        require(!plan.admits(budget) && plan.quota(group.key, budget, false) == 0 && plan.quota(group.key, budget, true) == 0);
    require(plan.quota(key(24), 500, true) == 0);
    require(plan.cold_quota(key(19), 450, 400, false) == 50);
    require(plan.cold_quota(key(19), 450, 440, false) == 20);
    require(plan.cold_quota(key(19), 450, 451, false) == 0);
    require(plan.cold_quota(key(19), 450, 400, true) == 200);
    require(plan.cold_quota(key(16), 450, 0, true) == 300);
    for (std::size_t budget = 900; budget < 1001; ++budget) {
        std::size_t total_wide = 0, total_limits = 0;
        require(plan.all_wide_fit(budget, total_wide) && total_wide == 900);
        for (const auto& group : groups) {
            const auto cold = plan.cold_quota(group.key, budget, 0, true);
            require(cold >= group.wide_bytes && cold <= plan.quota(group.key, budget, true));
            total_limits += cold;
        }
        require(total_limits == budget);
    }
    std::size_t no_total = 123;
    require(!plan.all_wide_fit(899, no_total) && no_total == 0);
    // All-wide sum overflows signed size but one phase remains valid.
    const auto half_max = static_cast<std::size_t>((std::numeric_limits<std::int64_t>::max)()) / 2;
    tsg::HostMoePhaseQuotaPlan huge({{key(16), 1, half_max + 1}, {key(19), 1, half_max + 1}});
    require(huge.admits(half_max + 2) && !huge.all_wide_fit(half_max + 2, no_total));
    require(huge.cold_quota(key(16), half_max + 2, 0, true) == half_max + 1);
    auto changed = key(16); changed.values[3]++;
    require(plan.quota(changed, 500, false) == 0);
    const auto invalid = [&](std::vector<tsg::HostMoePhaseGroupMinimum> candidate) {
        bool refused = false;
        try { tsg::HostMoePhaseQuotaPlan rejected(std::move(candidate)); }
        catch (const std::exception&) { refused = true; }
        require(refused && plan.minimum() == 450);
    };
    int d = 0, p = 0;
    require(plan.row_limits(key(16), 3, 512, d, p) && d == 3 && p == 512);
    tsg::HostMoePhaseQuotaPlan shapes({{key(16), 30, 300, 1, 3, 512}, {key(19), 20, 20, 1, 1, 1}});
    require(shapes.row_limits(key(19), 3, 512, d, p) && d == 1 && p == 1);
    require(!shapes.row_limits(key(16), 2, 512, d, p) && d == 0 && p == 0);
    require(!shapes.row_limits(key(16), 3, 511, d, p));
    require(!shapes.row_limits(key(24), 3, 512, d, p));
    require(shapes.minimum() == 320 && !shapes.admits(319));
    // Merge widths before measuring bytes, including incomparable profiles.
    tsg::HostMoePhaseGroupMinimum a{key(16), 0, 0, 7, 3, 1}, b{key(16), 0, 0, 5, 1, 8};
    auto ab = a, ba = b; ab.merge(b); ba.merge(a);
    require(ab.decode_rows == 3 && ab.prefill_rows == 8 && ab.weight == 12);
    require(ab.decode_rows == ba.decode_rows && ab.prefill_rows == ba.prefill_rows && ab.weight == ba.weight);
    invalid({{key(16), 1, 1, 1, 0, 1}});
    invalid({{key(16), 1, 1, 1, 4, 8}});
    invalid({{key(16), 1, 1, 1, 1, 1025}});
    bool merge_overflow = false;
    try { auto huge_weight = a; huge_weight.weight = static_cast<std::size_t>((std::numeric_limits<std::int64_t>::max)()); huge_weight.merge(b); }
    catch (const std::overflow_error&) { merge_overflow = true; }
    require(merge_overflow);
    invalid({}); invalid({groups[0], groups[0]});
    invalid({{key(16), 0, 1}}); invalid({{key(16), 2, 1}});
    changed = key(16); changed.values[5] = 289;
    invalid({{changed, 1, 1}});
    const auto maximum = static_cast<std::size_t>((std::numeric_limits<std::int64_t>::max)());
    invalid({{key(16), maximum, maximum}, {key(19), 1, 1}});
    invalid({{key(16), 1, maximum}, {key(19), 1, 1}}); // Phase sum overflow, narrow sum does fit.
    tsg::HostMoePhaseQuotaPlan narrow_only({{key(16), 30, 30}, {key(19), 20, 20}});
    require(narrow_only.minimum() == 50 && narrow_only.quota(key(16), 50, true) == 30);
    tsg::HostMoePhaseQuotaPlan single({{key(16), 30, 300}});
    require(single.minimum() == 300 && single.quota(key(16), 301, false) == 301);
    // CPU ownership state machine: no credit is borrowed until a physical owner
    // has actually been removed. This supplements, not replaces, the CUDA ledger.
    std::array<std::size_t, 3> live{300, 20, 40};
    auto total = [&] { return live[0] + live[1] + live[2]; };
    const auto saved = live;
    require(!plan.admits(449) && live == saved); // Sub-floor resize is nonmutating.
    live[0] = 0; // Synchronous physical retirement before B is admitted.
    require(total() + 200 <= plan.minimum()); live[1] = 200;
    live[2] = 0; live[1] = 0; // Retire B before C's wide allocation.
    require(total() + 400 <= plan.minimum()); live[2] = 400;
    live = {}; require(total() == 0); // Cancellation/trim.
    live = {30, 20, 40}; require(total() == plan.narrow_total()); // N3 resume.
    // Recorded GLM5.3-Flash IQ1_S geometry and 42 routed-layer traffic. Bulk
    // prefill leaves only the rare type group resident (312 slots). The old
    // half-share gate misses this despite decode's other groups being absent.
    const auto glm_key = [&](int gate, int down) {
        auto value = key(gate); value.values[2] = down; return value;
    };
    const auto common = glm_key(16, 18), rare = glm_key(16, 23), frequent = glm_key(19, 18);
    const tsg::HostMoePhaseQuotaPlan glm({
        {common, 189271040, 2397113344, 11u * 7536640},
        {rare, 219155456, 2755726336, 3u * 8781824},
        {frequent, 164105216, 2095123456, 28u * 6488064}});
    constexpr std::size_t glm_budget = 5121432328ull, rare_owned = 2751090688ull;
    const auto common_share = glm.quota(common, glm_budget, false);
    const auto rare_share = glm.quota(rare, glm_budget, false);
    const auto frequent_share = glm.quota(frequent, glm_budget, false);
    require(common_share == 1485589868 && rare_share == 631108158 && frequent_share == 3004734300ull);
    const auto absent = common_share + frequent_share, room = glm_budget - rare_owned;
    require(rare_owned > 2 * rare_share && room >= absent / 2 && room < absent);
    require(glm.should_rebalance_decode(glm_budget, rare_owned,
        [&](const auto& group) { return group == rare ? rare_owned : 0; }));
    require(!glm.should_rebalance_decode(glm_budget, 0, [](const auto&) { return std::size_t(0); }));
    require(!glm.should_rebalance_decode(glm_budget, common_share + rare_share + frequent_share,
        [&](const auto& group) { return glm.quota(group, glm_budget, false); }));
    // No absent/undersized peer: even an oversized affordable union survives.
    require(!glm.should_rebalance_decode(glm_budget, rare_owned + common_share / 2 + frequent_share / 2,
        [&](const auto& group) { return group == rare ? rare_owned : glm.quota(group, glm_budget, false) / 2; }));
    // A real peer below half its share also requires the >2x hoarder gate.
    require(glm.should_rebalance_decode(glm_budget, rare_owned + common_share / 3 + frequent_share / 2,
        [&](const auto& group) { return group == rare ? rare_owned : group == common ? common_share / 3 : frequent_share / 2; }));
    require(!glm.should_rebalance_decode(glm_budget, rare_share + common_share / 3 + frequent_share / 2,
        [&](const auto& group) { return group == rare ? rare_share : group == common ? common_share / 3 : frequent_share / 2; }));
    require(!glm.should_rebalance_decode(glm.minimum() - 1, rare_owned,
        [&](const auto& group) { return group == rare ? rare_owned : 0; }));
    require(!tsg::HostMoePhaseQuotaPlan::exceeds_twice_share(200, 100)
        && tsg::HostMoePhaseQuotaPlan::exceeds_twice_share(201, 100)
        && !tsg::HostMoePhaseQuotaPlan::exceeds_twice_share(maximum, maximum / 2 + 1));
    require(glm.growth_quota(frequent, glm_budget, rare_owned) == glm_budget - rare_owned);
    require(glm.growth_quota(rare, glm_budget, rare_owned) == rare_share);
    require(glm.growth_quota(common, glm_budget, glm_budget) == 0
        && glm.growth_quota(common, glm_budget, glm_budget + 1) == 0);
    require(glm.growth_quota(common, glm.minimum() - 1, 0) == 0);
    require(glm.growth_quota(key(24), glm_budget, 0) == 0);
    // Recorded plain2048-stage-v10 after a 33-token prefill: all three groups
    // remain present, but 28-layer traffic has only 353 slots and the 3-layer
    // group has 304. The frequent group is just ABOVE half its weighted share,
    // so the legacy predicate cannot recover this persistent allocation skew.
    const tsg::HostMoePhaseQuotaPlan short_prefill({
        {common, 68684800, 2288028672, 11u * 7536640},
        {rare, 78646272, 2646641664, 3u * 8781824},
        {frequent, 60296192, 1986038784, 28u * 6488064}});
    constexpr std::size_t short_budget = 7436500992ull;
    const std::array<std::size_t, 3> measured{2042778880ull, 2305709312ull, 2685097216ull};
    const auto recorded_owned = [&](const auto& group) {
        return group == common ? measured[0] : group == frequent ? measured[1] : measured[2];
    };
    const auto recorded_total = measured[0] + measured[1] + measured[2];
    require(recorded_total == 7033585408ull && recorded_total < short_budget);
    const auto frequent_target = short_prefill.quota(frequent, short_budget, false);
    const auto rare_target = short_prefill.quota(rare, short_budget, false);
    require(measured[1] > frequent_target / 2 && measured[1] < frequent_target - frequent_target / 4
        && tsg::HostMoePhaseQuotaPlan::exceeds_twice_share(measured[2], rare_target));
    require(!short_prefill.should_rebalance_decode(short_budget, recorded_total, recorded_owned));
    require(short_prefill.should_rebalance_decode(short_budget, recorded_total, recorded_owned, true));
    require(!tsg::HostMoePhaseQuotaPlan::below_decode_share(measured[0],
        short_prefill.quota(common, short_budget, false), true));
    // Retain the near-share common table. Only the hoarder and deficient peer
    // need physical retirement; both new decode shares fit beside that owner.
    const auto after_retire = measured[0];
    require(after_retire + frequent_target + rare_target <= short_budget);
    require(!short_prefill.should_rebalance_decode(short_budget,
        after_retire + frequent_target + rare_target, [&](const auto& group) {
            return group == common ? measured[0] : short_prefill.quota(group, short_budget, false);
        }, true));
    // A deficient peer alone is insufficient: no automatic cold rebuild of
    // balanced or merely underfilled tables. Missing/floor-invalid rules stay.
    require(!short_prefill.should_rebalance_decode(short_budget, measured[0] + measured[1] + rare_target,
        [&](const auto& group) { return group == rare ? rare_target : recorded_owned(group); }, true));
    require(!short_prefill.should_rebalance_decode(short_prefill.minimum() - 1, recorded_total, recorded_owned, true));
    require(!short_prefill.should_rebalance_decode(short_budget, 0, [](const auto&) { return std::size_t(0); }, true));
    // Strict 75% boundary, odd quotas and overflow-safe maximum-size behavior.
    for (std::size_t q = 1; q != 101; ++q) {
        const auto edge = q - q / 4;
        require(tsg::HostMoePhaseQuotaPlan::below_decode_share(edge - 1, q, true));
        require(!tsg::HostMoePhaseQuotaPlan::below_decode_share(edge, q, true));
        require(!tsg::HostMoePhaseQuotaPlan::below_decode_share(q, q, true));
    }
    const auto max_size = (std::numeric_limits<std::size_t>::max)();
    require(!tsg::HostMoePhaseQuotaPlan::below_decode_share(0, 0, true));
    require(tsg::HostMoePhaseQuotaPlan::below_decode_share(max_size - max_size / 4 - 1, max_size, true));
    require(!tsg::HostMoePhaseQuotaPlan::below_decode_share(max_size - max_size / 4, max_size, true));
    // Recorded ordinary-executor v17 + RAM-victim PP33 tables leave 397.6 MiB
    // unused. Old+new replacement cannot grow the frequent 2.3 GB table; its
    // own cold retirement can turn that exact slack into ~64 additional slots.
    constexpr std::size_t v17_budget = 7435452416ull;
    // Common table includes its subsequently installed 7,031,040-byte PP33 scratch.
    const std::array<std::size_t, 3> v17{2027705600ull, 2305709312ull, 2685097216ull};
    const auto v17_owned = [&](const auto& group) {
        return group == common ? v17[0] : group == frequent ? v17[1] : v17[2];
    };
    const auto v17_total = v17[0] + v17[1] + v17[2];
    tsg::HostMoeGroupKey nominated;
    require(v17_total == 7018512128ull && v17_budget - v17_total == 416940288ull);
    require(short_prefill.nominate_cold_growth(v17_budget, v17_total, v17_owned, nominated)
        && nominated == frequent);
    require(short_prefill.cold_growth_quota(frequent, v17_budget, v17_total, v17[1]) == 2722649600ull);
    require(short_prefill.growth_quota(frequent, v17_budget, v17_total) < v17[1]);
    require(!short_prefill.nominate_cold_growth(v17_total, v17_total, v17_owned, nominated));
    require(!short_prefill.nominate_cold_growth(v17_budget, v17_total, v17_owned, nominated,
        v17_budget - v17_total + 1));
    require(!short_prefill.nominate_cold_growth(v17_budget, v17_total, [&](const auto& group) {
        return group == rare ? std::size_t(0) : v17_owned(group);
    }, nominated));
    require(!short_prefill.nominate_cold_growth(v17_budget, v17_total, [&](const auto& group) {
        return group == rare ? short_prefill.floor(rare, false) - 1 : v17_owned(group);
    }, nominated));
    require(short_prefill.cold_growth_quota(frequent, v17_budget, v17_total, 0) == 0);
    require(short_prefill.cold_growth_quota(frequent, v17_budget, v17_total, v17_total + 1) == 0);
    require(short_prefill.cold_growth_quota(frequent, v17_total - 1, v17_total, v17[1]) == 0);
    require(short_prefill.cold_growth_quota(key(24), v17_budget, v17_total, v17[1]) == 0);
    // Actual v28 PP33 ownership: the near-share common group exceeds its quota
    // by about51MiB. Full absent shares force its otherwise unnecessary erase.
    // Partial rebalance preserves it and protects both exact cold-narrow floors.
    constexpr std::size_t v28_budget = 7538484736ull, v28_common = 2208584960ull;
    const auto absent_owned = [&](const auto& group) { return group == common ? v28_common : std::size_t(0); };
    const auto all_missing_floors = short_prefill.absent_narrow_floors(absent_owned);
    require(all_missing_floors == 138942464ull);
    require(short_prefill.absent_narrow_floors(absent_owned, &frequent) == 78646272ull);
    require(short_prefill.absent_narrow_floors(absent_owned, &rare) == 60296192ull);
    require(short_prefill.absent_narrow_floors([](const auto&) { return std::size_t(1); }) == 0);
    require(short_prefill.absent_narrow_floors([](const auto&) { return std::size_t(0); })
        == short_prefill.narrow_total());
    const auto full_missing_shares = short_prefill.quota(frequent, v28_budget, false)
        + short_prefill.quota(rare, v28_budget, false);
    require(v28_budget - v28_common >= all_missing_floors
        && v28_budget - v28_common < full_missing_shares);
    const auto first_quota = short_prefill.cold_quota(frequent, v28_budget, v28_common,
        false, short_prefill.absent_narrow_floors(absent_owned, &frequent));
    require(first_quota >= short_prefill.floor(frequent, false)
        && first_quota <= short_prefill.quota(frequent, v28_budget, false));
    require(v28_common + first_quota + short_prefill.floor(rare, false) <= v28_budget);
    const auto second_quota = short_prefill.cold_quota(rare, v28_budget, v28_common + first_quota, false);
    require(second_quota >= short_prefill.floor(rare, false)
        && v28_common + first_quota + second_quota <= v28_budget);
    // At the exact floor-only boundary, first allocation receives exactly its
    // own floor; one byte less refuses without borrowing the other peer's.
    const auto floor_room_owned = v28_budget - all_missing_floors;
    require(short_prefill.cold_quota(frequent, v28_budget, floor_room_owned, false,
        short_prefill.floor(rare, false)) == short_prefill.floor(frequent, false));
    require(short_prefill.cold_quota(frequent, v28_budget, floor_room_owned + 1, false,
        short_prefill.floor(rare, false)) == 0);
    require(short_prefill.cold_quota(frequent, v28_budget, v28_budget, false, 1) == 0);
    require(short_prefill.cold_quota(frequent, v28_budget, v28_budget + 1, false, 1) == 0);
    require(short_prefill.cold_quota(key(24), v28_budget, 0, false, 1) == 0);
    // Exercise all cold orders with rounded/underfilled actual candidates. A
    // new owner cannot consume the final absent peer's mandatory reservation.
    for (bool frequent_first : {false, true}) for (std::size_t shortfall : {0u, 1u, 1048576u}) {
        const auto& first = frequent_first ? frequent : rare;
        const auto& last = frequent_first ? rare : frequent;
        const auto reserve_last = short_prefill.floor(last, false);
        const auto granted = short_prefill.cold_quota(first, v28_budget, v28_common, false, reserve_last);
        require(granted >= short_prefill.floor(first, false) + shortfall);
        const auto actual = granted - shortfall;
        const auto last_granted = short_prefill.cold_quota(last, v28_budget, v28_common + actual, false);
        require(last_granted >= reserve_last && v28_common + actual + last_granted <= v28_budget);
    }
    // The optional extra argument leaves every existing wide path unchanged.
    require(short_prefill.cold_quota(common, v28_budget, v28_common, true, 123)
        == short_prefill.cold_quota(common, v28_budget, v28_common, true));
    std::printf("PASS %d pure phase-plan checks; real shared tickets and CUDA remain separate qualification\n", checks);
}
