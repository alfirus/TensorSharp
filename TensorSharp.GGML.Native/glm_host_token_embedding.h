// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace tsg_glm {

// Raw-row geometry only. The native caller separately proves the immutable
// model profile, source lifetime, Q4_K layout and original CUDA GET_ROWS route.
struct host_token_embedding_layout {
    std::size_t hidden = 0;
    std::size_t vocabulary = 0;
    std::size_t row_bytes = 0;
    std::size_t table_bytes = 0;
    int type = -1;
};

inline constexpr std::size_t host_token_embedding_max_rows = 512;
inline constexpr std::size_t host_token_embedding_profile_hidden = 4096;
inline constexpr std::size_t host_token_embedding_profile_vocabulary = 154880;
inline constexpr std::size_t host_token_embedding_profile_row_bytes = 2304;
inline constexpr std::size_t host_token_embedding_profile_table_bytes = 356843520;
inline constexpr int host_token_embedding_profile_type = 12;

inline host_token_embedding_layout checked_host_token_embedding_layout(
    std::size_t source_bytes, std::size_t row_bytes, std::size_t hidden,
    std::size_t vocabulary, int type) {
    if (!row_bytes || !hidden || !vocabulary || type < 0 ||
        vocabulary > static_cast<std::size_t>((std::numeric_limits<std::int32_t>::max)()))
        throw std::invalid_argument("invalid host embedding geometry");
    if (row_bytes > (std::numeric_limits<std::size_t>::max)() / vocabulary)
        throw std::overflow_error("host embedding table overflow");
    const auto table_bytes = row_bytes * vocabulary;
    if (source_bytes != table_bytes)
        throw std::invalid_argument("host embedding extent does not match table");
    return {hidden, vocabulary, row_bytes, table_bytes, type};
}

// A layer-major caller validates its entire submission before advancing any
// recurrent state, even if the submission is later packed in <=512-row pieces.
inline bool validate_host_token_embedding_tokens(const std::int32_t* tokens,
    std::size_t count, std::size_t vocabulary) noexcept {
    if (!tokens || !count || !vocabulary ||
        vocabulary > static_cast<std::size_t>((std::numeric_limits<std::int32_t>::max)())) return false;
    if (count > (std::numeric_limits<std::size_t>::max)() / sizeof(*tokens)) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(tokens);
    if (begin > (std::numeric_limits<std::uintptr_t>::max)() - count * sizeof(*tokens)) return false;
    for (std::size_t i = 0; i < count; ++i)
        if (tokens[i] < 0 || static_cast<std::size_t>(tokens[i]) >= vocabulary) return false;
    return true;
}

// Copier is the caller's fault-containment seam (e.g. a Windows SEH leaf).
// Success alone publishes copied_bytes. A failed/throwing copier may have
// written a prefix; that destination must be discarded, never uploaded/reused.
template<class Copier>
bool pack_host_token_embedding_rows(const host_token_embedding_layout& layout,
    const std::uint8_t* source, std::size_t source_extent,
    const std::int32_t* tokens, std::size_t count,
    std::uint8_t* destination, std::size_t capacity, Copier copier,
    std::size_t* copied_bytes = nullptr) noexcept {
    if (copied_bytes) *copied_bytes = 0;
    const auto maximum = (std::numeric_limits<std::size_t>::max)();
    if (!source || !destination || !count || count > host_token_embedding_max_rows ||
        !layout.row_bytes || !layout.hidden || !layout.vocabulary || layout.type < 0 ||
        layout.row_bytes > maximum / layout.vocabulary ||
        layout.table_bytes != layout.row_bytes * layout.vocabulary ||
        source_extent < layout.table_bytes || layout.row_bytes > maximum / count ||
        !validate_host_token_embedding_tokens(tokens, count, layout.vocabulary)) return false;
    const auto packed_bytes = count * layout.row_bytes;
    if (capacity < packed_bytes) return false;
    const auto src = reinterpret_cast<std::uintptr_t>(source);
    const auto dst = reinterpret_cast<std::uintptr_t>(destination);
    const auto pointer_maximum = (std::numeric_limits<std::uintptr_t>::max)();
    if (src > pointer_maximum - layout.table_bytes || dst > pointer_maximum - packed_bytes) return false;
    if (src < dst + packed_bytes && dst < src + layout.table_bytes) return false;
    std::array<std::size_t, host_token_embedding_max_rows> offsets{};
    for (std::size_t i = 0; i < count; ++i) {
        // Capture all offsets before the copier can mutate caller-owned tokens.
        offsets[i] = static_cast<std::size_t>(tokens[i]) * layout.row_bytes;
        if (offsets[i] > layout.table_bytes - layout.row_bytes) return false;
    }
    try {
        for (std::size_t i = 0; i < count; ++i)
            if (!copier(destination + i * layout.row_bytes, source + offsets[i], layout.row_bytes)) return false;
    } catch (...) { return false; }
    if (copied_bytes) *copied_bytes = packed_bytes;
    return true;
}

} // namespace tsg_glm
