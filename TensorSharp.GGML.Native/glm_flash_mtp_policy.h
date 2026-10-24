// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <cstdint>
namespace tsg_glm {
// The first Flash draft adapter uses dense MLA only where sparse selection
// cannot remove a visible token. Longer contexts require a real draft indexer.
inline bool flash_mtp_eligible(bool enabled, int devices, int ranks, int context, int index_top_k) noexcept {
    return enabled && devices == 1 && ranks == 1 && context > 0 && index_top_k > 0 && context <= index_top_k;
}
inline bool flash_mtp_window(std::int64_t position, std::int64_t rows, int context, int index_top_k) noexcept {
    const auto limit = context < index_top_k ? context : index_top_k;
    return position >= 0 && rows > 0 && limit > 0 && position <= limit && rows <= limit - position;
}

struct FlashMtpInputWindow {
    std::int64_t skip_rows = 0;
    std::int64_t position = 0;
    std::int64_t rows = 0;
};

// The runtime supplies (token[p], hidden[p-1]) at trunk position p. Flash's
// reference proposer shifts tokens left, retaining the hidden state's position:
// (token[p], hidden[p-1]) belongs at draft position p-1. There is no hidden[-1],
// so the synthetic first catch-up row is omitted. In particular, draft position
// zero must contain hidden[0], with the token embedding zeroed by build_mtp.
// Validate the original trunk window first to retain the public context bound.
inline bool flash_mtp_map_inputs(std::int64_t position, std::int64_t rows,
        int context, int index_top_k, bool catch_up, FlashMtpInputWindow& out) noexcept {
    if (!flash_mtp_window(position, rows, context, index_top_k) || (!catch_up && position == 0))
        return false;
    const std::int64_t skip = position == 0 ? 1 : 0;
    out = {skip, position + skip - 1, rows - skip};
    return true;
}
} // namespace tsg_glm
