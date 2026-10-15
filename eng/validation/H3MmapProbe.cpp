// CPU-only, read-only Linux/POSIX mapping comparison. Never drops caches or
// modifies the checkpoint. Compile: c++ -O2 -std=c++17 -pthread H3MmapProbe.cpp
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
static double seconds(timeval value) { return value.tv_sec + value.tv_usec * 1e-6; }
static uint64_t mix(uint64_t value) {
    value ^= value >> 30; value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27; value *= UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}
struct Case { bool shared, copy; size_t workers; };

int main(int argc, char ** argv) {
    try {
        if (argc < 2) throw std::runtime_error("Usage: H3MmapProbe FILE [--offset BYTES] [--bytes BYTES] [--trials4] [--workers8]");
        const char * path = argv[1];
        size_t offset = 0, bytes = size_t{2} << 30, trials = 4, parallel = 8;
        for (int i = 2; i + 1 < argc; i += 2) {
            const std::string key = argv[i];
            const size_t value = std::stoull(argv[i + 1]);
            if (key == "--offset") offset = value;
            else if (key == "--bytes") bytes = value;
            else if (key == "--trials") trials = value;
            else if (key == "--workers") parallel = value;
            else throw std::runtime_error("Unknown argument");
        }
        if (bytes == 0 || trials == 0 || trials > 16 || parallel == 0 || parallel > 8)
            throw std::runtime_error("Require positive byte count,1..16trials and1..8workers");
        int fd = open(path, O_RDONLY);
        if (fd < 0) throw std::runtime_error("Cannot open source file");
        struct stat statbuf{};
        if (fstat(fd, &statbuf) != 0 || offset > static_cast<size_t>(statbuf.st_size) ||
            bytes > static_cast<size_t>(statbuf.st_size) - offset)
            throw std::runtime_error("Requested range exceeds source file");
        const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
        const size_t aligned = offset / page * page, delta = offset - aligned, mapped = delta + bytes;
        constexpr size_t chunk = 8u << 20;
        const std::array<Case, 8> cases{{
            {true,false,1}, {false,false,1}, {false,true,parallel}, {true,true,parallel},
            {true,false,parallel}, {false,false,parallel}, {false,true,1}, {true,true,1}}};
        std::printf("{\"metadata\":{\"source_bytes\":%llu,\"device\":%llu,\"inode\":%llu,\"offset\":%zu,\"bytes\":%zu,"
                    "\"page_bytes\":%zu,\"trials\":%zu,\"maximum_copy_scratch_bytes\":%zu,"
                    "\"scope\":\"One open file descriptor;fresh munmap/remap every case;balanced reversed order;no cache eviction;sampled checksum\"}}\n",
                    static_cast<unsigned long long>(statbuf.st_size), static_cast<unsigned long long>(statbuf.st_dev),
                    static_cast<unsigned long long>(statbuf.st_ino), offset, bytes, page, trials, parallel * chunk);
        std::fflush(stdout);
        for (size_t trial = 0; trial < trials; ++trial) {
            for (size_t order = 0; order < cases.size(); ++order) {
                const Case c = cases[(trial % 2 == 0 ? order : cases.size() - 1 - order)];
                std::vector<std::vector<unsigned char>> buffers;
                if (c.copy) {
                    buffers.resize(c.workers);
                    for (auto & buffer : buffers) buffer.resize(chunk);
                }
                std::vector<uint64_t> checksums(c.workers, 0);
                rusage before{}, after{};
                getrusage(RUSAGE_SELF, &before);
                const auto map_begin = Clock::now();
                void * mapping = mmap(nullptr, mapped, PROT_READ, c.shared ? MAP_SHARED : MAP_PRIVATE, fd, static_cast<off_t>(aligned));
                if (mapping == MAP_FAILED) throw std::runtime_error("mmap failed");
                const double map_seconds = std::chrono::duration<double>(Clock::now() - map_begin).count();
                const auto * source = static_cast<const unsigned char *>(mapping) + delta;
                const size_t pages = 1 + (bytes - 1) / page;
                const auto begin = Clock::now();
                const auto work = [&](size_t worker) {
                    const size_t first_page = pages * worker / c.workers, last_page = pages * (worker + 1) / c.workers;
                    const size_t first = std::min(bytes, first_page * page), last = std::min(bytes, last_page * page);
                    uint64_t checksum = 0;
                    if (!c.copy) {
                        const volatile unsigned char * input = source;
                        for (size_t pos = first; pos < last; pos += page)
                            checksum ^= mix((pos / page << 8) ^ input[pos]);
                    } else {
                        for (size_t pos = first; pos < last;) {
                            const size_t count = std::min(chunk, last - pos);
                            std::memcpy(buffers[worker].data(), source + pos, count);
                            for (size_t inside = 0; inside < count; inside += page)
                                checksum ^= mix(((pos + inside) / page << 8) ^ buffers[worker][inside]);
                            pos += count;
                        }
                    }
                    if (worker + 1 == c.workers && (bytes - 1) % page != 0) {
                        const volatile unsigned char * input = source;
                        checksum ^= mix(((bytes - 1) / page << 8) ^ input[bytes - 1]);
                    }
                    checksums[worker] = checksum;
                };
                std::vector<std::thread> workers;
                size_t started = 1;
                try {
                    for (; started < c.workers; ++started) workers.emplace_back(work, started);
                } catch (...) {
                    for (size_t worker = started; worker < c.workers; ++worker) work(worker);
                }
                work(0);
                for (auto & worker : workers) worker.join();
                const double elapsed = std::chrono::duration<double>(Clock::now() - begin).count();
                getrusage(RUSAGE_SELF, &after);
                uint64_t checksum = 0;
                for (auto value : checksums) checksum ^= value;
                if (munmap(mapping, mapped) != 0) throw std::runtime_error("munmap failed");
                std::printf("{\"trial\":%zu,\"order\":%zu,\"mapping\":\"%s\",\"operation\":\"%s\",\"workers\":%zu,"
                            "\"wall_seconds\":%.9f,\"mmap_seconds\":%.9f,\"minor_faults\":%ld,\"major_faults\":%ld,"
                            "\"user_seconds\":%.9f,\"sys_seconds\":%.9f,\"checksum\":\"%016llx\"}\n",
                            trial, order, c.shared ? "shared" : "private", c.copy ? "copy" : "touch", workers.size()+1,
                            elapsed, map_seconds, after.ru_minflt-before.ru_minflt, after.ru_majflt-before.ru_majflt,
                            seconds(after.ru_utime)-seconds(before.ru_utime), seconds(after.ru_stime)-seconds(before.ru_stime),
                            static_cast<unsigned long long>(checksum));
                std::fflush(stdout);
            }
        }
        close(fd);
        return 0;
    } catch (const std::exception & error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
