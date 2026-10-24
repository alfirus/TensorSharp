// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "../glm_flash_mtp_policy.h"
#include <iostream>
#include <stdexcept>
#include <limits>

int main() {
    std::size_t checks = 0;
    auto check = [&](bool value) { ++checks; if (!value) throw std::runtime_error("Flash MTP policy boundary failed"); };
    try {
        using namespace tsg_glm;
        check(!flash_mtp_eligible(false, 1, 1, 2048, 2048));
        check(flash_mtp_eligible(true, 1, 1, 2048, 2048));
        check(!flash_mtp_eligible(true, 1, 1, 2049, 2048));
        check(!flash_mtp_eligible(true, 0, 1, 2048, 2048));
        check(!flash_mtp_eligible(true, 2, 1, 2048, 2048));
        check(!flash_mtp_eligible(true, 1, 2, 2048, 2048));
        check(!flash_mtp_eligible(true, 1, 1, 2048, 0));
        check(!flash_mtp_eligible(true, 1, 1, 0, 2048));
        check(!flash_mtp_window(-1, 1, 2048, 2048));
        check(!flash_mtp_window(0, 0, 2048, 2048));
        check(!flash_mtp_window(0, -1, 2048, 2048));
        check(!flash_mtp_window((std::numeric_limits<std::int64_t>::max)(), 1, 2048, 2048));
        check(!flash_mtp_window(1, (std::numeric_limits<std::int64_t>::max)(), 2048, 2048));
        check(flash_mtp_window(0, 2048, 2048, 2048));
        check(!flash_mtp_window(0, 2049, 2048, 2048));
        check(flash_mtp_window(2047, 1, 4096, 2048));
        check(!flash_mtp_window(2048, 1, 4096, 2048));
        for (int position = 0; position < 2048; ++position) {
            check(flash_mtp_window(position, 1, 2048, 2048));
            check(flash_mtp_window(position, 2048 - position, 2048, 2048));
            check(!flash_mtp_window(position, 2049 - position, 2048, 2048));
        }

        FlashMtpInputWindow input;
        check(flash_mtp_map_inputs(0, 1, 2048, 2048, true, input));
        check(input.skip_rows == 1 && input.position == 0 && input.rows == 0);
        check(!flash_mtp_map_inputs(0, 1, 2048, 2048, false, input));
        check(!flash_mtp_map_inputs(2048, 1, 2048, 2048, false, input));
        check(!flash_mtp_map_inputs(2047, 2, 2048, 2048, true, input));
        check(!flash_mtp_map_inputs(-1, 1, 2048, 2048, true, input));
        check(!flash_mtp_map_inputs(0, 0, 2048, 2048, true, input));
        check(!flash_mtp_map_inputs((std::numeric_limits<std::int64_t>::max)(), 1, 2048, 2048, true, input));
        check(input.skip_rows == 1 && input.position == 0 && input.rows == 0); // refusal leaves outputs unchanged

        // Compare any runtime chunking with the reference's left-shifted token
        // sequence. After prompt catch-up, DraftStep supplies the sampled tail:
        // draft row r always sees token[r+1] and target hidden[r]. The first
        // draft embedding is zeroed at r=0, never on a synthetic hidden[-1].
        const int chunks[] = {1, 2, 3, 7, 64, 511, 512, 2048};
        const int prompts[] = {1, 2, 3, 17, 512, 513, 1872, 2047};
        for (const int chunk : chunks) for (const int prompt : prompts) {
            std::int64_t next_draft_row = 0;
            for (int start = 0; start < prompt;) {
                const int count = chunk < prompt - start ? chunk : prompt - start;
                check(flash_mtp_map_inputs(start, count, 2048, 2048, true, input));
                check(input.position == next_draft_row);
                for (std::int64_t row = 0; row < input.rows; ++row) {
                    const auto trunk_token = start + input.skip_rows + row;
                    const auto preceding_hidden = trunk_token - 1;
                    check(input.position + row == next_draft_row);
                    check(trunk_token == next_draft_row + 1);
                    check(preceding_hidden == next_draft_row);
                    ++next_draft_row;
                }
                start += count;
            }
            check(next_draft_row == prompt - 1);
            check(flash_mtp_map_inputs(prompt, 1, 2048, 2048, false, input));
            check(input.skip_rows == 0 && input.rows == 1 && input.position == next_draft_row);
        }
        // Accepted catch-up overwrites the same coordinates used by drafting;
        // rejected later rows remain causally invisible until overwritten.
        for (int start = 1; start < 2046; ++start) {
            check(flash_mtp_map_inputs(start, 3, 2048, 2048, true, input));
            check(input.skip_rows == 0 && input.position == start - 1 && input.rows == 3);
            for (int row = 0; row < 3; ++row) {
                FlashMtpInputWindow step;
                check(flash_mtp_map_inputs(start + row, 1, 2048, 2048, false, step));
                check(step.position == input.position + row && step.rows == 1 && step.skip_rows == 0);
            }
        }
        std::cout << "glm_flash_mtp_policy_test: " << checks << " checks passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
