// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "../ggml_ops_host_moe_decode_staging.h"
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <stdexcept>

int main() {
    int checks = 0;
    const auto require = [&](bool value) { ++checks; if (!value) throw std::runtime_error("Scalar staging boundary changed"); };
    try {
        const char* invalid[] = {nullptr, "", "0", "01", "1 ", " 1", "true", "2", "-1"};
        for (const char* value : invalid)
            require(!tsg::host_moe_scalar_staging_flag(value));
        require(tsg::host_moe_scalar_staging_flag("1"));
        constexpr auto lo = tsg::kHostMoeFileStagingBytes, hi = tsg::kHostMoeScalarFileStagingBytes;
        for (bool enabled : {false, true}) for (bool pooled : {false, true})
            for (int rows : {-1, 0, 1, 2, 3, 8, 9, 32, 512}) for (int selected : {0, 1, 7, 8, 9, 16})
                for (std::size_t bytes : {std::size_t(0), lo - 1, lo, lo + 1, hi - 1, hi, hi + 1,
                    (std::numeric_limits<std::size_t>::max)()}) {
                    const bool expected = enabled && pooled && rows == 1 && selected == 8 && bytes > lo && bytes <= hi;
                    require(tsg::host_moe_scalar_staging_eligible(enabled, pooled, rows, selected, bytes) == expected);
                    require(tsg::host_moe_prefill_ring_staging_eligible(enabled, pooled, rows > 8, selected, bytes)
                        == (enabled && pooled && rows > 8 && selected == 8 && bytes > lo));
                }
        // Actual checkpoint top-eight unions fit one experimental window.
        for (std::size_t expert : {std::size_t(6488064), std::size_t(7536640), std::size_t(8781824)}) {
            require((8 * expert + lo - 1) / lo > 1);
            require(tsg::host_moe_scalar_staging_eligible(true, true, 1, 8, 8 * expert));
            require((8 * expert + hi - 1) / hi == 1);
        }
        std::printf("PASS: scalar staging exact flag, complete union and phase boundaries %d checks\n", checks);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1;
    }
}
