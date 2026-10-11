// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include "ggml_ops_precision_policy.h"

namespace tsg
{
    constexpr int kHostMoeCacheMaxPrefillRows = 128;
    // The routing validation scratch stays 16 KiB independently of the
    // opt-in row ceiling. Larger selected-expert sets permit fewer rows.
    constexpr int kHostMoeCacheMaxRouteElements = 4096;

    inline bool host_moe_cache_routes_fit(int rows, int selected)
    {
        return rows > 0 && selected > 0 && rows <= kHostMoeCacheMaxRouteElements / selected;
    }

    // This opt-in only widens the compact expert helper, never the attention,
    // router, recurrent-state or whole-model precision policy. Invalid values
    // retain the existing decode-width behavior; they cannot enlarge a bound.
    inline int host_moe_cache_prefill_rows(const char* value)
    {
        const int fallback = static_cast<int>(TSG_PRECISION_DECODE_COLUMNS);
        if (value == nullptr || *value == '\0') return fallback;
        int result = 0;
        for (const char* p = value; *p; ++p)
        {
            if (*p < '0' || *p > '9') return fallback;
            result = result * 10 + (*p - '0');
            if (result > kHostMoeCacheMaxPrefillRows) return fallback;
        }
        return result > fallback ? result : fallback;
    }
}
