// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "../ggml_ops_host_moe_file_ring.h"
#include "../ggml_ops_host_moe_file_ring_coalesce.h"
#include "../ggml_ops_parallel_read_pipeline.h"
#include "../dsv41_engram_io.h"
#include <atomic>
#include <cstdio>
#include <future>
#include <map>
#include <thread>

static int checks;
static void require(bool value, const char* message) {
    ++checks; if (!value) throw std::runtime_error(message);
}
#include "host_moe_file_ring_coalesce_fixture.inc"

int main() {
    try {
        coalesced_ring_checks();
        tsg_dsv41::engram_io_pool pool(8);
        const std::array<std::size_t, 3> stride{7, 13, 19};
        const std::vector<std::pair<int, int>> misses{{5, 8}, {2, 1}, {9, 0}, {0, 2}, {7, 5}};
        for (std::size_t arena : {2, 7, 23, 64, 1024}) {
            const tsg::HostMoeFileRingPlan plan(misses, stride, arena);
            require(plan.slot_bytes * 2 <= arena, "ring exceeds charged arena");
            std::vector<std::uint8_t> staging(arena, 0xff);
            std::vector<std::size_t> counts;
            for (const auto& window : plan.windows) counts.push_back(window.count);
            for (int failure : {-1, 0, 1, 2}) {
                // A large arena has one window; still exercise each error stage.
                const auto fail_index = (std::min)(std::size_t(2), plan.windows.size() - 1);
                std::array<std::future<void>, 2> transfers;
                std::array<std::atomic<bool>, 2> in_flight{};
                std::atomic<int> active_reads{0};
                std::map<std::pair<int, int>, std::vector<std::uint8_t>> output;
                for (const auto& miss : misses) for (int part = 0; part < 3; ++part)
                    output[{miss.first, part}].resize(stride[part], 0xff);
                std::size_t copied = 0;
                bool threw = false;
                try {
                    tsg::run_bounded_parallel_read_pipeline(pool, counts, 2,
                        [&](std::size_t index, std::size_t slot) {
                            if (transfers[slot].valid()) transfers[slot].get();
                            if (failure == 0 && index == fail_index) throw std::runtime_error("fence failure");
                        }, [&](std::size_t index, std::size_t job, std::size_t slot) {
                            ++active_reads;
                            struct Exit { std::atomic<int>& count; ~Exit() { --count; } } exit{active_reads};
                            if (in_flight[slot]) throw std::runtime_error("DMA source overwritten before fence");
                            if (failure == 1 && index == fail_index && job == 0) throw std::runtime_error("read failure");
                            const auto& piece = plan.pieces[plan.windows[index].first + job];
                            auto* destination = staging.data() + slot * plan.slot_bytes + piece.staging_offset;
                            for (std::size_t i = 0; i < piece.bytes; ++i)
                                destination[i] = static_cast<std::uint8_t>(piece.part * 41 + piece.source_offset + i);
                        }, [&](std::size_t index, std::size_t slot) {
                            in_flight[slot] = true;
                            const auto& window = plan.windows[index];
                            // Resolve output pointers on the consumer, never concurrently mutate map metadata.
                            std::vector<std::uint8_t*> destinations;
                            for (std::size_t i = window.first; i < window.first + window.count; ++i) {
                                const auto& piece = plan.pieces[i];
                                destinations.push_back(output.at({piece.slot, piece.part}).data() + piece.expert_offset);
                            }
                            transfers[slot] = std::async(std::launch::async, [&, index, slot, destinations] {
                                for (int i = 0; i < 40; ++i) std::this_thread::yield();
                                const auto& transfer = plan.windows[index];
                                for (std::size_t job = 0; job < transfer.count; ++job) {
                                    const auto& piece = plan.pieces[transfer.first + job];
                                    const auto* source = staging.data() + slot * plan.slot_bytes + piece.staging_offset;
                                    for (std::size_t i = 0; i < piece.bytes; ++i) {
                                        if (source[i] != static_cast<std::uint8_t>(piece.part * 41 + piece.source_offset + i))
                                            throw std::runtime_error("DMA source changed while pending");
                                        destinations[job][i] = source[i];
                                    }
                                }
                                in_flight[slot] = false;
                            });
                            copied += window.bytes;
                            // Submission failure after actual asynchronous work has begun.
                            if (failure == 2 && index == fail_index) throw std::runtime_error("submission failure");
                        });
                } catch (const std::runtime_error&) { threw = true; }
                for (auto& transfer : transfers) if (transfer.valid()) transfer.get();
                require(active_reads == 0 && !in_flight[0] && !in_flight[1], "failure escaped live work");
                require(threw == (failure >= 0), "expected failure did not propagate");
                if (failure < 0) {
                    require(copied == misses.size() * (stride[0] + stride[1] + stride[2]), "source byte count changed");
                    for (const auto& miss : misses) for (int part = 0; part < 3; ++part)
                        for (std::size_t i = 0; i < stride[part]; ++i)
                            require(output.at({miss.first, part})[i]
                                == static_cast<std::uint8_t>(part * 41 + miss.second * stride[part] + i), "output byte changed");
                }
                int retry = 0;
                tsg::run_bounded_parallel_read_pipeline(pool, std::vector<std::size_t>{2, 3, 1}, 2,
                    [](auto, auto) {}, [](auto, auto, auto) {}, [&](auto, auto) { ++retry; });
                require(retry == 3, "pool cannot retry after failure");
            }
        }
        // Deterministically prove that window 1 is read during window 0's DMA.
        std::promise<void> submitted, read_during_transfer;
        auto submitted_future = submitted.get_future().share();
        auto read_future = read_during_transfer.get_future().share();
        std::future<void> transfer;
        tsg::run_bounded_parallel_read_pipeline(pool, std::vector<std::size_t>{1, 1}, 2,
            [](auto, auto) {}, [&](std::size_t index, auto, auto) {
                if (index == 1) { submitted_future.wait(); read_during_transfer.set_value(); }
            }, [&](std::size_t index, auto) {
                if (index == 0) transfer = std::async(std::launch::async, [&] {
                    submitted.set_value(); read_future.wait();
                });
            });
        transfer.get(); require(true, "overlap handshake completed");
        const auto rejects = [&](auto selected, auto extents, std::size_t arena) {
            bool rejected = false;
            try { tsg::HostMoeFileRingPlan plan(selected, extents, arena); }
            catch (const std::exception&) { rejected = true; }
            require(rejected, "invalid geometry accepted");
        };
        rejects(misses, stride, 0); rejects(misses, stride, 1);
        rejects(misses, std::array<std::size_t, 3>{0, 2, 3}, 32);
        rejects(std::vector<std::pair<int, int>>{{-1, 0}}, stride, 32);
        rejects(std::vector<std::pair<int, int>>{{0, -1}}, stride, 32);
        rejects(std::vector<std::pair<int, int>>{{0, 1}, {0, 2}}, stride, 32);
        rejects(std::vector<std::pair<int, int>>{{0, 1}, {1, 1}}, stride, 32);
        rejects(std::vector<std::pair<int, int>>{{0, 2}},
            std::array<std::size_t, 3>{(std::numeric_limits<std::size_t>::max)(), 1, 1}, 32);
        require(tsg::HostMoeFileRingPlan({}, stride, 32).windows.empty(), "empty miss set reads bytes");
        // Large plans allocate metadata only. Neither the non-power-of-two
        // 36 MiB slot nor a projection larger than that slot may change the
        // original expert/projection ordering or exact source coverage.
        constexpr std::size_t mib = std::size_t(1) << 20;
        const std::vector<std::pair<int, int>> wide_misses{{7, 11}, {1, 3}, {9, 0}, {0, 8}};
        for (auto large_stride : {std::array<std::size_t, 3>{2*mib+17, 2*mib+31, 3*mib+63},
                std::array<std::size_t, 3>{36*mib-1, 36*mib+1, 73*mib+3}}) {
            for (std::size_t arena : {32*mib, 64*mib, 72*mib}) {
                const tsg::HostMoeFileRingPlan plan(wide_misses, large_stride, arena);
                require(plan.slot_bytes == arena / 2, "large ring slot geometry changed");
                std::size_t miss = 0, part = 0, offset = 0, total = 0;
                for (const auto& piece : plan.pieces) {
                    require(miss < wide_misses.size() && piece.slot == wide_misses[miss].first
                        && piece.part == static_cast<int>(part) && piece.expert_offset == offset
                        && piece.source_offset == static_cast<std::size_t>(wide_misses[miss].second)*large_stride[part]+offset,
                        "large ring reordered or duplicated a source projection");
                    require(piece.bytes > 0 && piece.bytes <= large_stride[part]-offset
                        && piece.staging_offset+piece.bytes <= plan.slot_bytes,
                        "large ring piece exceeds its projection or slot");
                    total += piece.bytes; offset += piece.bytes;
                    if (offset == large_stride[part]) { offset = 0; if (++part == 3) { part = 0; ++miss; } }
                }
                require(miss == wide_misses.size() && part == 0 && offset == 0
                    && total == wide_misses.size()*(large_stride[0]+large_stride[1]+large_stride[2]),
                    "large ring did not cover its exact original bytes");
                require(plan.windows.size() == (total+plan.slot_bytes-1)/plan.slot_bytes,
                    "large ring introduced a spurious window");
            }
        }
        std::printf("PASS expert file ring: %d checks; exact bytes, bounded arena, real asynchronous overlap, joined failures and retries; no CUDA or model loaded\n", checks);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
