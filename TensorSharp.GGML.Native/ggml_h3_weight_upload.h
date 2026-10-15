// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace tsg {
namespace h3_upload_detail {

inline bool parse_memory_limit(const char * token, size_t & bytes) {
    if (std::strcmp(token, "max") == 0) {
        bytes = std::numeric_limits<size_t>::max();
        return true;
    }
    if (token[0] < '0' || token[0] > '9') return false;
    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(token, &end, 10);
    if (errno == ERANGE || end == token || *end != '\0' || parsed > std::numeric_limits<size_t>::max()) return false;
    bytes = static_cast<size_t>(parsed);
    return true;
}

inline void apply_memory_limit(const std::string & path, size_t & limit) {
    errno = 0;
    if (FILE * file = std::fopen(path.c_str(), "r")) {
        char token[64]{};
        size_t bytes = 0;
        const bool valid = std::fscanf(file, "%63s", token) == 1 && parse_memory_limit(token, bytes);
        std::fclose(file);
        // A present but unreadable/invalid limit is unknown, not unrestricted.
        limit = valid ? std::min(limit, bytes) : 0;
    } else if (errno != ENOENT && errno != ENOTDIR) {
        limit = 0;
    }
}

// Effective physical/container capacity, not current usage: file cache is
// reclaimable. Unknown capacity disables automatic warming.
inline size_t host_memory_limit() {
    size_t limit = 0;
#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
    const long pages = sysconf(_SC_PHYS_PAGES), page_bytes = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_bytes > 0 && static_cast<size_t>(pages) <= std::numeric_limits<size_t>::max() / static_cast<size_t>(page_bytes))
        limit = static_cast<size_t>(pages) * static_cast<size_t>(page_bytes);
#endif
#ifdef __linux__
    if (limit == 0) {
        if (FILE * file = std::fopen("/proc/meminfo", "r")) {
            unsigned long long kib = 0;
            if (std::fscanf(file, "MemTotal: %llu kB", &kib) == 1 && kib > 0 && kib <= std::numeric_limits<size_t>::max() / 1024)
                limit = static_cast<size_t>(kib) * 1024;
            std::fclose(file);
        }
    }
    if (limit == 0) return 0;
    apply_memory_limit("/sys/fs/cgroup/memory.max", limit);
    apply_memory_limit("/sys/fs/cgroup/memory/memory.limit_in_bytes", limit);
    // Also honor nested limits when the process sees a host cgroup hierarchy.
    // The root paths above cover a container's namespaced hierarchy.
    std::ifstream memberships("/proc/self/cgroup");
    std::string line;
    while (std::getline(memberships, line)) {
        const size_t first = line.find(':'), second = line.find(':', first == std::string::npos ? first : first + 1);
        if (first == std::string::npos || second == std::string::npos) continue;
        const std::string controllers = line.substr(first + 1, second - first - 1);
        const bool unified = controllers.empty();
        if (!unified && ("," + controllers + ",").find(",memory,") == std::string::npos) continue;
        std::string path = line.substr(second + 1);
        if (path.empty() || path[0] != '/' || path.find("..") != std::string::npos) continue;
        const std::string root = unified ? "/sys/fs/cgroup" : "/sys/fs/cgroup/memory";
        const char * suffix = unified ? "/memory.max" : "/memory.limit_in_bytes";
        while (path.size() > 1) {
            apply_memory_limit(root + path + suffix, limit);
            const size_t separator = path.find_last_of('/');
            path.resize(separator == 0 ? 1 : separator);
        }
    }
#endif
    return limit;
}

} // namespace h3_upload_detail

// Read-only CPU page warming before CUDA uploads. Workers never invoke backend
// APIs, and all join before the caller retains the original tensor_set calls.
class H3WeightUpload {
public:
    // The capacity override makes policy tests independent of the test host.
    H3WeightUpload(ggml_backend_t backend, const char * tag,
        size_t memory_limit_override = std::numeric_limits<size_t>::max())
        : backend_(backend), tag_(tag), memory_limit_override_(memory_limit_override) {
        const char * profile = std::getenv("TS_H3_UPLOAD_PROFILE");
        const char * prefault = std::getenv("TS_H3_UPLOAD_PREFAULT");
        profile_ = profile != nullptr && std::strcmp(profile, "1") == 0;
        mode_ = prefault != nullptr && std::strcmp(prefault, "0") == 0 ? Mode::Disabled :
            prefault != nullptr && std::strcmp(prefault, "1") == 0 ? Mode::Forced : Mode::Auto;
        if (const char * workers = std::getenv("TS_H3_UPLOAD_WORKERS")) {
            char * end = nullptr;
            const long count = std::strtol(workers, &end, 10);
            if (end != workers && *end == '\0' && count >= 1 && count <= 8)
                requested_workers_ = static_cast<size_t>(count);
        }
    }

    H3WeightUpload(const H3WeightUpload &) = delete;
    H3WeightUpload & operator=(const H3WeightUpload &) = delete;
    ~H3WeightUpload() { finish(); }

    template <typename Bindings>
    void prefault(const Bindings & bindings) {
        if (mode_ == Mode::Disabled) { decision_ = "disabled"; return; }
        if (!is_cuda()) { decision_ = "non-cuda"; return; }
        const auto begin = now();
        struct Range { const unsigned char * data; size_t bytes; };
        std::vector<Range> ranges;
        size_t aggregate = 0;
        try {
            ranges.reserve(bindings.size());
            for (const auto & binding : bindings) {
                if (binding.data == nullptr || binding.bytes < (1u << 20)) continue;
                if (binding.bytes > std::numeric_limits<size_t>::max() - aggregate) {
                    decision_ = "range-overflow";
                    prefault_seconds_ += seconds(begin);
                    return;
                }
                aggregate += binding.bytes;
                ranges.push_back({static_cast<const unsigned char *>(binding.data), binding.bytes});
            }
            candidate_bytes_ += aggregate;
            if (ranges.empty()) { decision_ = "no-large-ranges"; return; }
            memory_limit_bytes_ = memory_limit_override_ == std::numeric_limits<size_t>::max()
                ? h3_upload_detail::host_memory_limit() : memory_limit_override_;
            if (mode_ == Mode::Auto && (memory_limit_bytes_ == 0 || aggregate > memory_limit_bytes_ / 2)) {
                decision_ = memory_limit_bytes_ == 0 ? "unknown-memory" : "memory-limit";
                prefault_seconds_ += seconds(begin);
                return;
            }
            // Sorting changes only read-ahead order, never the upload order.
            std::sort(ranges.begin(), ranges.end(), [](const Range & a, const Range & b) {
                return reinterpret_cast<uintptr_t>(a.data) < reinterpret_cast<uintptr_t>(b.data);
            });
        } catch (...) {
            ++prefault_fallbacks_;
            decision_ = "metadata-fallback";
            prefault_seconds_ += seconds(begin);
            return;
        }
        decision_ = mode_ == Mode::Forced ? "forced" : "auto-allowed";
        prefault_bytes_ += aggregate;
        prefault_tensors_ += ranges.size();
        std::atomic<size_t> next{0}, reads{0};
        std::atomic<unsigned int> checksum{0};
        const auto touch = [&] {
            size_t local_reads = 0;
            unsigned int local_checksum = 0;
            for (;;) {
                const size_t index = next.fetch_add(1, std::memory_order_relaxed);
                if (index >= ranges.size()) break;
                const auto & range = ranges[index];
                const volatile unsigned char * source = range.data;
                constexpr size_t stride = 4096;
                for (size_t offset = 0; offset < range.bytes;) {
                    local_checksum ^= source[offset];
                    ++local_reads;
                    if (range.bytes - offset <= stride) break;
                    offset += stride;
                }
                // Include the last partial or unaligned source page.
                if ((range.bytes - 1) % stride != 0) {
                    local_checksum ^= source[range.bytes - 1];
                    ++local_reads;
                }
            }
            reads.fetch_add(local_reads, std::memory_order_relaxed);
            checksum.fetch_xor(local_checksum, std::memory_order_relaxed);
        };
        const size_t count = std::min(requested_workers_, ranges.size());
        std::array<std::thread, 8> workers;
        size_t started = 0;
        try {
            for (; started < count; ++started) workers[started] = std::thread(touch);
        } catch (...) {
            ++prefault_fallbacks_;
            touch();
        }
        for (size_t worker = 0; worker < started; ++worker) workers[worker].join();
        prefault_workers_ = std::max(prefault_workers_, started + (started < count ? size_t{1} : size_t{0}));
        prefault_reads_ += reads.load(std::memory_order_relaxed);
        prefault_checksum_ ^= checksum.load(std::memory_order_relaxed);
        prefault_seconds_ += seconds(begin);
    }

    size_t prefault_bytes() const { return prefault_bytes_; }
    size_t prefault_reads() const { return prefault_reads_; }
    size_t prefault_workers() const { return prefault_workers_; }
    unsigned int prefault_checksum() const { return prefault_checksum_; }
    const char * prefault_decision() const { return decision_; }

    void set(ggml_tensor * tensor, const void * source, size_t bytes) {
        const auto begin = now();
        ++tensors_;
        bytes_ += bytes;
        ggml_backend_tensor_set(tensor, source, 0, bytes);
        set_seconds_ += seconds(begin);
    }

    void finish() {
        if (finished_) return;
        finished_ = true;
        if (!profile_) return;
        std::fprintf(stderr,
            "[h3-upload] tag=%s prefault_mode=%s prefault_decision=%s memory_limit_bytes=%zu "
            "candidate_bytes=%zu workers_requested=%zu prefault_tensors=%zu prefault_bytes=%zu prefault_workers=%zu "
            "prefault_reads=%zu prefault_checksum=%u prefault_fallbacks=%zu prefault_seconds=%.6f "
            "tensors=%zu bytes=%zu tensor_set_seconds=%.6f upload_seconds=%.6f\n",
            tag_, mode_ == Mode::Auto ? "auto" : mode_ == Mode::Forced ? "forced" : "off", decision_, memory_limit_bytes_,
            candidate_bytes_, requested_workers_, prefault_tensors_, prefault_bytes_, prefault_workers_,
            prefault_reads_, prefault_checksum_, prefault_fallbacks_, prefault_seconds_,
            tensors_, bytes_, set_seconds_, prefault_seconds_ + set_seconds_);
        std::fflush(stderr);
    }

private:
    enum class Mode { Auto, Disabled, Forced };
    using Clock = std::chrono::steady_clock;
    Clock::time_point now() const { return profile_ ? Clock::now() : Clock::time_point{}; }
    double seconds(Clock::time_point begin) const {
        return profile_ ? std::chrono::duration<double>(Clock::now() - begin).count() : 0.0;
    }
    bool is_cuda() const {
        if (backend_ == nullptr) return false;
        const char * name = ggml_backend_name(backend_);
        return name != nullptr && std::strncmp(name, "CUDA", 4) == 0;
    }

    ggml_backend_t backend_;
    const char * tag_;
    const size_t memory_limit_override_;
    size_t memory_limit_bytes_ = 0, requested_workers_ = 8;
    size_t tensors_ = 0, bytes_ = 0, candidate_bytes_ = 0;
    size_t prefault_tensors_ = 0, prefault_bytes_ = 0, prefault_workers_ = 0, prefault_reads_ = 0, prefault_fallbacks_ = 0;
    unsigned int prefault_checksum_ = 0;
    double set_seconds_ = 0, prefault_seconds_ = 0;
    Mode mode_ = Mode::Auto;
    const char * decision_ = "not-requested";
    bool profile_ = false, finished_ = false;
};

} // namespace tsg
