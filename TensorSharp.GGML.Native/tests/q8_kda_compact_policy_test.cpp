// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_q8_kda_compact_policy.h"
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace {
int checks = 0;
void require(bool value, const char * message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
}

int main() {
    try {
        struct flag_case { const char * value; bool enabled; };
        const flag_case flag_cases[] = {{nullptr,false}, {"0",false}, {"1",true}};
        for (const auto & test : flag_cases) {
            const auto result = tsg_q8_kda_parse_option(test.value);
            require(result.valid && result.enabled == test.enabled, "Valid immutable option rejected");
        }
        for (const char * text : {"", "00", "01", "true", "false", " 1", "1 ", "2", "-1", "0x1", "1\n"}) {
            const auto result = tsg_q8_kda_parse_option(text);
            require(!result.valid && !result.enabled, "Malformed immutable option accepted");
        }
        // Enumerate the exact finite N whitelist, including every non-target
        // narrow width and both edges. Neighbor K/M values must remain old.
        const int low = std::numeric_limits<int>::min(), high = std::numeric_limits<int>::max();
        for (int k : {low,-1,0,32,128,4064,4096,4128,16384,high})
        for (int m : {low,-1,0,1,24,63,64,65,127,128,129,8192,high})
        for (int n = -1; n <= 514; ++n) {
            bool member = false;
            if (k == 4096 && (m == 64 || m == 128)) {
                for (int admitted = 9; admitted <= 512; ++admitted)
                    if (admitted == n) member = true;
                if (n == 1) member = true;
            }
            require(tsg_q8_kda_target_shape(k,m,n) == member, "Shape whitelist widened or truncated");
            require(tsg_q8_kda_selected_variant(k,m,n,false) == 0, "Default/OFF selected a kernel");
            require(tsg_q8_kda_selected_variant(k,m,n,true) == (member ? (n == 1 ? 1 : 2) : 0),
                "Wrong ordered scalar/wide variant");
        }
        for (int m : {64,128}) for (int n : {low,high}) {
            require(!tsg_q8_kda_target_shape(4096,m,n)
                && tsg_q8_kda_selected_variant(4096,m,n,true) == 0,
                "Extreme N escaped bounded policy");
        }
        // Original b projections are actual Q8 but outside this experiment.
        for (int n : {1,9,33,336,512})
            require(tsg_q8_kda_selected_variant(128,8192,n,true) == 0,
                "Large-output b projection was included");
        std::printf("PASS q8-kda-compact-policy checks=%d\n",checks);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr,"FAIL q8-kda-compact-policy: %s\n",error.what());
        return 1;
    }
}
