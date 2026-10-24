#include "../glm_verified_prefix_policy.h"
#include <cstdio>
#include <cstdlib>
#include <limits>

static int checks = 0;
static void require(bool value) { ++checks; if (!value) std::abort(); }

int main() {
    using tsg::GlmVerifiedPrefixWindow;
    for (int flags = 0; flags < 32; ++flags)
        for (int rows = 0; rows <= 8; ++rows)
            for (int ub = 0; ub <= 8; ++ub)
                require(GlmVerifiedPrefixWindow::capture_shape(flags & 1, flags & 2,
                    rows, ub, flags & 4, flags & 8, flags & 16)
                    == (flags == 31 && rows == 2 && ub >= 2));
    GlmVerifiedPrefixWindow state;
    require(!state.can_commit(0, 4, 6, 0, 1, 1));
    const auto first = state.begin(0, 4, 1);
    require(first != 0);
    require(!state.can_commit(0, 4, 6, 0, 1, 1)); // An incomplete GPU pass grants no authority.
    require(!state.publish(first, 1, 6, 1));
    require(!state.publish(first, 0, 5, 1));
    require(!state.publish(first, 0, 6, 2));
    require(state.publish(first, 0, 6, 1));
    for (int slot = -1; slot <= 2; ++slot)
        for (int base = 2; base <= 6; ++base)
            for (int position = 2; position <= 8; ++position)
                for (int accepted = -1; accepted <= 3; ++accepted)
                    for (int drafts = -1; drafts <= 3; ++drafts)
                        for (int binding = 0; binding <= 2; ++binding)
                            require(state.can_commit(slot, base, position, accepted, drafts, binding)
                                == (slot == 0 && base == 4 && position == 6 && accepted == 0
                                    && drafts == 1 && binding == 1));
    state.invalidate();
    require(!state.can_commit(0, 4, 6, 0, 1, 1));
    require(!state.publish(first, 0, 6, 1));
    const auto next = state.begin(0, 4, 1);
    require(next != first && next != 0);
    require(!state.publish(first, 0, 6, 1)); // Same slot/position cannot revive an older pass.
    require(state.publish(next, 0, 6, 1));
    require(state.begin(-1, 0, 1) == 0);
    require(state.begin(0, -1, 1) == 0);
    require(state.begin(0, std::numeric_limits<std::int64_t>::max() - 1, 1) == 0);
    require(state.begin(0, 0, 0) == 0);
    require(!state.can_commit(0, 4, 6, 0, 1, 1));
    std::printf("PASS %d verified-prefix authority checks; GPU copy and arithmetic tests are separate\n", checks);
}
