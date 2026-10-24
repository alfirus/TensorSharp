// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include "ggml_ops_host_moe_group_quota.h"

namespace tsg {
struct HostMoePhaseGroupMinimum {
    HostMoeGroupKey key;
    std::size_t narrow_bytes = 0, wide_bytes = 0;
    // Relative decode traffic: layers in the group x bytes of one expert. Spare
    // budget beyond the floors is shared in this proportion, so a group serving
    // most layers is not starved by one serving a few. Zero: share equally.
    std::size_t weight = 0;
    // Zero/zero inherits the enclosing scope (legacy callers). Explicit limits
    // describe every member of this pooled identity, not its first caller.
    int decode_rows = 0, prefill_rows = 0;

    // A changed combined shape must be measured again before installation;
    // maxima of byte estimates alone cannot certify incomparable row profiles.
    void merge(const HostMoePhaseGroupMinimum& other) {
        if (!(key == other.key)) throw std::invalid_argument("cannot merge different expert groups");
        const auto limit = static_cast<std::size_t>((std::numeric_limits<std::int64_t>::max)());
        if (weight > limit || other.weight > limit - weight)
            throw std::overflow_error("expert group traffic weight overflow");
        narrow_bytes = (std::max)(narrow_bytes, other.narrow_bytes);
        wide_bytes = (std::max)(wide_bytes, other.wide_bytes);
        weight += other.weight;
        if (!decode_rows || !other.decode_rows) decode_rows = prefill_rows = 0;
        else {
            decode_rows = (std::max)(decode_rows, other.decode_rows);
            prefill_rows = (std::max)(prefill_rows, other.prefill_rows);
        }
    }
};

// Serialized helper phases may reclaim inactive groups. The minimum protects
// one whole wide union plus every other group's narrow decode floor. It is a
// feasibility bound, not a reservation or permission to refund a live owner.
class HostMoePhaseQuotaPlan {
    std::vector<HostMoePhaseGroupMinimum> groups_;
    std::size_t narrow_total_ = 0, minimum_ = 0;
    std::size_t weight_total_ = 0; // zero unless every group has a weight
    // Share of `amount` for group i: by weight, else equally.
    std::size_t share(std::size_t i, std::size_t amount) const noexcept {
        if (weight_total_ == 0)
            return amount / groups_.size() + (i < amount % groups_.size() ? 1 : 0);
        // 128-bit-free proportional split: amount and weights stay far below
        // 2^63, and the quotient is computed in long double for safety.
        const long double part = static_cast<long double>(amount) * groups_[i].weight / weight_total_;
        return static_cast<std::size_t>(part);
    }
public:
    explicit HostMoePhaseQuotaPlan(std::vector<HostMoePhaseGroupMinimum> groups)
        : groups_(std::move(groups)) {
        if (groups_.empty()) throw std::invalid_argument("GLM phase plan is empty");
        std::sort(groups_.begin(), groups_.end(), [](const auto& a, const auto& b) { return a.key < b.key; });
        const auto limit = (std::min)(static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)()),
            static_cast<std::uint64_t>((std::numeric_limits<std::size_t>::max)()));
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const auto& group = groups_[i];
            if (!group.key.valid() || !group.narrow_bytes || group.wide_bytes < group.narrow_bytes
                || group.decode_rows < 0 || group.decode_rows > 3 || group.prefill_rows < 0 || group.prefill_rows > 1024
                || ((group.decode_rows == 0) != (group.prefill_rows == 0))
                || (i && groups_[i - 1].key == group.key))
                throw std::invalid_argument("GLM phase plan has invalid geometry or floors");
            if (group.wide_bytes > limit || narrow_total_ > limit - group.narrow_bytes)
                throw std::overflow_error("GLM phase floors exceed shared-budget range");
            narrow_total_ += group.narrow_bytes;
        }
        bool weighted = true;
        for (const auto& group : groups_) {
            if (group.weight == 0 || group.weight > limit - weight_total_) { weighted = false; break; }
            weight_total_ += group.weight;
        }
        if (!weighted) weight_total_ = 0;
        for (const auto& group : groups_) {
            const auto others = narrow_total_ - group.narrow_bytes;
            if (others > limit - group.wide_bytes)
                throw std::overflow_error("GLM phase minimum exceeds shared-budget range");
            minimum_ = (std::max)(minimum_, others + group.wide_bytes);
        }
    }
    const std::vector<HostMoePhaseGroupMinimum>& groups() const noexcept { return groups_; }
    std::size_t size() const noexcept { return groups_.size(); }
    std::size_t minimum() const noexcept { return minimum_; }
    std::size_t narrow_total() const noexcept { return narrow_total_; }
    bool row_limits(const HostMoeGroupKey& key, int scope_decode, int scope_prefill,
        int& decode, int& prefill) const noexcept {
        decode = prefill = 0;
        for (const auto& group : groups_) if (group.key == key) {
            const int d = group.decode_rows ? group.decode_rows : scope_decode;
            const int p = group.prefill_rows ? group.prefill_rows : scope_prefill;
            if (d < 1 || d > scope_decode || p < 1 || p > scope_prefill) return false;
            decode = d; prefill = p; return true;
        }
        return false;
    }
    std::size_t floor(const HostMoeGroupKey& key, bool wide) const noexcept {
        for (const auto& group : groups_) if (group.key == key)
            return wide ? group.wide_bytes : group.narrow_bytes;
        return 0;
    }
    bool admits(std::size_t budget) const noexcept {
        return budget >= minimum_ && budget <= static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)());
    }
    bool all_wide_fit(std::size_t budget, std::size_t& total) const noexcept {
        total = 0;
        if (!admits(budget)) return false;
        for (const auto& group : groups_) {
            // A sum of all-wide floors may exceed signed range even though one
            // wide phase fits. Never overflow or reject the valid phase plan.
            if (group.wide_bytes > budget - total) { total = 0; return false; }
            total += group.wide_bytes;
        }
        return true;
    }
    std::size_t quota(const HostMoeGroupKey& key, std::size_t budget, bool wide) const noexcept {
        if (!admits(budget)) return 0;
        for (std::size_t i = 0; i < groups_.size(); ++i) if (groups_[i].key == key) {
            if (wide) return budget - (narrow_total_ - groups_[i].narrow_bytes);
            return groups_[i].narrow_bytes + share(i, budget - narrow_total_);
        }
        return 0;
    }
    static bool exceeds_twice_share(std::size_t bytes, std::size_t quota) noexcept {
        return quota != 0 && bytes > quota && bytes - quota > quota;
    }
    static bool below_decode_share(std::size_t bytes, std::size_t quota,
        bool material_deficit = false) noexcept {
        // Keep the default half-share gate. The experimental phase policy also
        // repairs present groups below 75%: a table just above half can remain
        // permanently undersized because hot growth must hold old+new at once.
        // ceil(3*quota/4) avoids multiplication overflow, including SIZE_MAX.
        const auto threshold = material_deficit ? quota - quota / 4 : quota / 2;
        return bytes < threshold;
    }
    // Inspect real resident ownership without allocating or changing credit.
    // A missing group's full decode share must fit: accepting merely half can
    // permanently strand it at a small cold allocation after the phase switch.
    // Preserve affordable whole unions unless one group exceeds twice its
    // own share AND another is undersized or cannot obtain its complete share.
    template<class ResidentBytes>
    bool should_rebalance_decode(std::size_t budget, std::size_t total_reserved,
        ResidentBytes resident_bytes, bool material_deficit = false) const {
        if (!admits(budget)) return false;
        bool hoarding = false, starved = false;
        std::size_t absent_shares = 0;
        for (const auto& group : groups_) {
            const auto desired = quota(group.key, budget, false);
            const auto owned = resident_bytes(group.key);
            if (!owned) absent_shares += desired;
            else {
                hoarding = hoarding || exceeds_twice_share(owned, desired);
                starved = starved || below_decode_share(owned, desired, material_deficit);
            }
        }
        const auto room = budget > total_reserved ? budget - total_reserved : 0;
        return hoarding && (starved || room < absent_shares);
    }
    // Exact mandatory ownership of groups that do not currently have a table.
    // The caller may exclude the group it is about to allocate. A published
    // peer, including an affordable wide owner, is never counted as absent.
    template<class ResidentBytes>
    std::size_t absent_narrow_floors(ResidentBytes resident_bytes,
        const HostMoeGroupKey* exclude = nullptr) const {
        std::size_t total = 0;
        for (const auto& group : groups_)
            if ((!exclude || !(group.key == *exclude)) && !resident_bytes(group.key))
                total += group.narrow_bytes; // Constructor checked this complete sum.
        return total;
    }
    // A replacement is charged beside every live owner until its copies drain.
    // The full decode share is the target; cold_quota cannot be reused here
    // because it treats the old owner as an unrelated consumer of that share.
    std::size_t growth_quota(const HostMoeGroupKey& key, std::size_t budget,
        std::size_t total_reserved) const noexcept {
        if (total_reserved >= budget) return 0;
        return (std::min)(quota(key, budget, false), budget - total_reserved);
    }
    // A cold replacement first retires exactly its own reloadable payload.
    // Its final quota cannot borrow another group's physical ownership.
    std::size_t cold_growth_quota(const HostMoeGroupKey& key, std::size_t budget,
        std::size_t total_reserved, std::size_t old_bytes) const noexcept {
        if (!old_bytes || old_bytes > total_reserved || total_reserved > budget) return 0;
        return (std::min)(quota(key, budget, false), budget - (total_reserved - old_bytes));
    }
    template<class ResidentBytes>
    bool nominate_cold_growth(std::size_t budget, std::size_t total_reserved,
        ResidentBytes resident_bytes, HostMoeGroupKey& selected,
        std::size_t minimum_gain = std::size_t(64) << 20) const {
        if (!admits(budget) || total_reserved > budget || !minimum_gain) return false;
        // Missing or under-floor peers must recover their mandatory ownership
        // first. An optional replacement never evicts a peer to fund itself.
        for (const auto& group : groups_)
            if (resident_bytes(group.key) < group.narrow_bytes) return false;
        std::size_t largest_gain = 0;
        bool found = false;
        for (const auto& group : groups_) {
            const auto owned = resident_bytes(group.key);
            const auto desired = quota(group.key, budget, false);
            if (!below_decode_share(owned, desired, true)) continue;
            const auto target = cold_growth_quota(group.key, budget, total_reserved, owned);
            const auto gain = target > owned ? target - owned : 0;
            if (gain < minimum_gain || (found && gain <= largest_gain)) continue;
            selected = group.key; largest_gain = gain; found = true;
        }
        return found;
    }
    std::size_t cold_quota(const HostMoeGroupKey& key, std::size_t budget,
        std::size_t already_owned, bool wide,
        std::size_t absent_peer_narrow_floors = 0) const noexcept {
        const auto desired = quota(key, budget, wide);
        if (!desired || already_owned > budget) return 0;
        if (wide) {
            std::size_t all_wide = 0;
            if (all_wide_fit(budget, all_wide)) {
                // When all real groups can coexist, protect their measured wide
                // floors before assigning optional cold capacity. The sum of
                // these cold limits is exactly budget, so early groups cannot
                // fill most of the pool and evict otherwise affordable peers.
                for (std::size_t i = 0; i < groups_.size(); ++i) if (groups_[i].key == key) {
                    return groups_[i].wide_bytes + share(i, budget - all_wide);
                }
                return 0;
            }
            // Under real phase pressure, admit the complete wide union, but do
            // not spend every borrowable byte on optional initial hot slots.
            // Existing affordable large tables are retained by entry_quota.
            return (std::max)(floor(key, true), quota(key, budget, false));
        }
        if (absent_peer_narrow_floors) {
            // Experimental partial rebalance: optional capacity cannot consume
            // an absent peer's exact mandatory floor. Refuse before subtraction
            // or allocation if this complete set of floors cannot coexist.
            const auto available = budget - already_owned;
            if (absent_peer_narrow_floors > available) return 0;
            const auto own_room = available - absent_peer_narrow_floors;
            if (own_room < floor(key, false)) return 0;
            return (std::min)(desired, own_room);
        }
        // Prefer an affordable narrow table to evicting somebody else's hot
        // wide table for optional capacity. A genuine floor deficit may still
        // reclaim inactive owners later, after exact metadata planning.
        return (std::max)(floor(key, false), (std::min)(desired, budget - already_owned));
    }
};
} // namespace tsg
