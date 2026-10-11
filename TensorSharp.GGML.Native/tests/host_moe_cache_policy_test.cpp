// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "../ggml_ops_host_moe_cache_policy.h"
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

int main()
{
    int checks = 0;
    const auto require = [&](bool condition) {
        ++checks;
        if (!condition) { std::fprintf(stderr, "FAIL bounded host MoE policy check %d\n", checks); std::exit(1); }
    };
    const char* invalid[] = {nullptr, "", "0", "1", "8", "129", "+128", " 128", "128 ", "-1",
                            "18446744073709551616", "128x"};
    for (const char* value : invalid)
        require(tsg::host_moe_cache_prefill_rows(value) == 8);
    for (int rows : {9, 31, 63, 64, 65, 88, 127, 128})
        require(tsg::host_moe_cache_prefill_rows(std::to_string(rows).c_str()) == rows);
    require(tsg::kHostMoeCacheMaxRouteElements * sizeof(float) == 16384);
    require(tsg::host_moe_cache_routes_fit(88, 10));
    require(tsg::host_moe_cache_routes_fit(128, 10));
    require(tsg::host_moe_cache_routes_fit(64, 64));
    require(tsg::host_moe_cache_routes_fit(128, 32));
    require(!tsg::host_moe_cache_routes_fit(65, 64));
    require(!tsg::host_moe_cache_routes_fit(128, 33));
    for (int invalid : {0, -1, (std::numeric_limits<int>::min)(), (std::numeric_limits<int>::max)()})
    {
        require(!tsg::host_moe_cache_routes_fit(invalid, 64));
        require(!tsg::host_moe_cache_routes_fit(64, invalid));
    }
    // Exercise every supported row/selected pair, including the exact boundary.
    for (int rows = 1; rows <= 129; ++rows) for (int selected = 1; selected <= 64; ++selected)
        require(tsg::host_moe_cache_routes_fit(rows, selected) == (rows * selected <= 4096));
    std::printf("PASS %d host MoE row/route policy checks; default8, explicit128, routing scratch16KiB\n", checks);
}
