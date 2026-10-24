#pragma once

#include <cstdint>
#include <limits>

namespace tsg {

// One retained N2 verification prefix. Storage is owned by the model; this
// token only grants permission to publish/commit after all GPU work succeeds.
class GlmVerifiedPrefixWindow {
public:
    static constexpr bool capture_shape(bool enabled, bool fresh_snapshot,
            std::int64_t rows, int ubatch, bool all_logits, bool hidden, bool logits) noexcept {
        return enabled && fresh_snapshot && rows == 2 && ubatch >= 2
            && all_logits && hidden && logits;
    }

    void invalidate() noexcept {
        valid_ = false;
        if (generation_ != std::numeric_limits<std::uint64_t>::max()) ++generation_;
    }

    std::uint64_t begin(int slot, std::int64_t base, std::uint64_t binding) noexcept {
        invalidate();
        if (slot < 0 || base < 0 || base > std::numeric_limits<std::int64_t>::max() - 2
            || binding == 0 || generation_ == std::numeric_limits<std::uint64_t>::max()) return 0;
        slot_ = slot; base_ = base; binding_ = binding;
        return generation_;
    }

    bool publish(std::uint64_t generation, int slot, std::int64_t position,
                 std::uint64_t binding) noexcept {
        if (generation == 0 || generation != generation_ || slot != slot_
            || position != base_ + 2 || binding != binding_) return false;
        valid_ = true;
        return true;
    }

    bool can_commit(int slot, std::int64_t base, std::int64_t position,
                    int accepted_drafts, int verify_drafts, std::uint64_t binding) const noexcept {
        return valid_ && slot == slot_ && base == base_ && position == base_ + 2
            && accepted_drafts == 0 && verify_drafts == 1 && binding == binding_;
    }

private:
    std::uint64_t generation_ = 0, binding_ = 0;
    int slot_ = -1;
    std::int64_t base_ = 0;
    bool valid_ = false;
};

} // namespace tsg
