// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tsg {
inline bool host_moe_file_sort_flag(const char* value) {
    if (!value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::invalid_argument("TS_HOST_MOE_FILE_READ_SORT must be exactly 0 or 1");
}

// Selected slot/expert assignment is immutable input. Only the copy is sorted;
// no slot selection, route remapping, publication or recency operation lives here.
struct HostMoeMissReadOrder {
    static constexpr std::size_t maximum_misses = 288;
    std::vector<std::pair<int, int>> ordered;
    std::uint64_t inversions_before = 0;
    bool changed = false;
    HostMoeMissReadOrder(const std::vector<std::pair<int, int>>& selected,
                        int capacity, int experts) {
        if (capacity < 1 || experts < 1 || experts > 288 || selected.empty()
            || selected.size() > maximum_misses || selected.size() > static_cast<std::size_t>(experts))
            throw std::invalid_argument("File sort exceeds its bounded selected-union profile");
        for (std::size_t i = 0; i < selected.size(); ++i) {
            const auto& a = selected[i];
            if (a.first < 0 || a.first >= capacity || a.second < 0 || a.second >= experts)
                throw std::invalid_argument("File sort source/slot exceeds selected table");
            for (std::size_t j = 0; j < i; ++j) {
                const auto& b = selected[j];
                if (a.first == b.first || a.second == b.second)
                    throw std::invalid_argument("File sort repeats an assigned slot or source expert");
                inversions_before += b.second > a.second ? 1 : 0;
            }
        }
        ordered = selected;
        std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
            return a.second < b.second;
        });
        changed = inversions_before != 0;
    }
};
} // namespace tsg
