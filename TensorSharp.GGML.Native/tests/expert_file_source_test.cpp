// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_file_source.h"
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <chrono>

namespace {
int checks = 0;
void require(bool value, const char* message) { ++checks; if (!value) throw std::runtime_error(message); }
template<class F> void rejected(F f, const char* message) {
    bool failed = false; try { f(); } catch (const std::exception&) { failed = true; }
    require(failed, message);
}
void random_access_hint_lifetime(const char* path, const std::array<std::uint8_t, 1031>& bytes) {
#if defined(_WIN32)
    struct RestoreEnvironment {
        const std::string original = [] {
            const char* value = std::getenv("TS_HOST_MOE_FILE_RANDOM_ACCESS");
            return value ? std::string(value) : std::string();
        }();
        ~RestoreEnvironment() { _putenv_s("TS_HOST_MOE_FILE_RANDOM_ACCESS", original.c_str()); }
    } restore;
    require(_putenv_s("TS_HOST_MOE_FILE_RANDOM_ACCESS", "0") == 0, "Cannot select unhinted source fixture");
    tsg::ExpertFileSource plain(path, 7, 1024);
    require(!plain.random_access_hint() && !plain.random_access_hint(true),
        "Unhinted buffered source claimed an unopened direct handle hint");
    require(_putenv_s("TS_HOST_MOE_FILE_RANDOM_ACCESS", "1") == 0, "Cannot select hinted source fixture");
    tsg::ExpertFileSource hinted(path, 7, 1024);
    require(hinted.random_access_hint() && !hinted.random_access_hint(true) && !plain.random_access_hint(),
        "Source hint did not follow its successfully opened handle independently of current environment");
    require(_putenv_s("TS_HOST_MOE_FILE_RANDOM_ACCESS", "0") == 0, "Cannot restore unhinted fixture selection");
    require(hinted.random_access_hint() && !plain.random_access_hint(),
        "An existing source changed its hint after a later environment mutation");
    std::array<std::uint8_t, 1024> plain_bytes{}, hinted_bytes{};
    plain.read(0, plain_bytes.data(), plain_bytes.size());
    hinted.read(0, hinted_bytes.data(), hinted_bytes.size());
    require(plain_bytes == hinted_bytes && std::equal(plain_bytes.begin(), plain_bytes.end(), bytes.begin() + 7),
        "Random-access handle selection changed exact registered payload bytes");
#else
    (void)bytes;
    tsg::ExpertFileSource source(path, 7, 1024);
    require(!source.random_access_hint() && !source.random_access_hint(true),
        "POSIX source claimed an available Windows random-access handle hint");
#endif
}
}
int main() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() / std::filesystem::u8path(
        u8"tensorsharp-专家-" + std::to_string(stamp) + ".bin");
    try {
        std::array<std::uint8_t, 1031> bytes{};
        for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<std::uint8_t>(i * 13);
        { std::ofstream f(path, std::ios::binary); f.write(reinterpret_cast<const char*>(bytes.data()), bytes.size()); }
        const auto name = path.u8string();
        random_access_hint_lifetime(name.c_str(), bytes);
        rejected([&] { tsg::ExpertFileSource s(name.c_str(), 7, 1025); }, "extent past EOF accepted");
        rejected([&] { tsg::ExpertFileSource s(name.c_str(), UINT64_MAX - 1, 8); }, "overflow accepted");
        rejected([&] { tsg::ExpertFileSource s(nullptr, 0, 1); }, "null path accepted");
#if !defined(_WIN32)
        int owned_fd = -1;
#endif
        {
            tsg::ExpertFileSource source(name.c_str(), 7, 1024);
#if defined(_WIN32)
            std::uint64_t registered_file_bytes = 0;
            require(source.readonly_mapping_file_size(registered_file_bytes) && registered_file_bytes == bytes.size(),
                "immutable mapping size lost the validated whole-file extent");
            HANDLE duplicate = source.duplicate_readonly_mapping_file();
            require(duplicate != INVALID_HANDLE_VALUE, "cannot duplicate immutable source handle");
            LARGE_INTEGER observed{};
            const bool matches = GetFileSizeEx(duplicate, &observed) && observed.QuadPart >= 0
                && static_cast<std::uint64_t>(observed.QuadPart) == registered_file_bytes;
            require(CloseHandle(duplicate) != FALSE, "immutable duplicate failed cleanup");
            require(matches, "immutable cached extent differs from its exact duplicate handle");
            HANDLE writer = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
            const auto writer_error = GetLastError();
            if (writer != INVALID_HANDLE_VALUE) CloseHandle(writer);
            require(writer == INVALID_HANDLE_VALUE && writer_error == ERROR_SHARING_VIOLATION,
                "live immutable source allowed write/truncate access");
            HANDLE remover = CreateFileW(path.c_str(), DELETE, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
            const auto remover_error = GetLastError();
            if (remover != INVALID_HANDLE_VALUE) CloseHandle(remover);
            require(remover == INVALID_HANDLE_VALUE && remover_error == ERROR_SHARING_VIOLATION,
                "live immutable source allowed delete/replacement access");
#endif
            std::array<std::uint8_t, 1030> output{};
            source.read(0, output.data() + 3, 1024);
            require(std::equal(bytes.begin() + 7, bytes.end(), output.begin() + 3), "exact offset read differs");
            require(output[0] == 0 && output[1029] == 0, "read overwrote guards");
            source.read(1024, nullptr, 0);
            rejected([&] { source.read(1024, output.data(), 1); }, "registered extent overread accepted");
            rejected([&] { source.read(1025, output.data(), 0); }, "empty out-of-extent read accepted");
            rejected([&] { source.read(0, nullptr, 1); }, "null payload accepted");
            // Decode bypasses the prefetch-only cache entirely; its original
            // task still reports exact completed source bytes. Wide prefill
            // probes the same canonical extent and reports zero on RAM hits.
            tsg::ExpertReadTask demand{&source, 19, output.data(), 37};
            demand.prepare();
            require(demand.source_read_bytes == 37 && !demand.payload_lease,
                "uncached demand task lost actual source-byte accounting");
            require(std::equal(bytes.begin() + 26, bytes.begin() + 63, output.begin()),
                "uncached demand task changed payload bytes");
            tsg::HostMoePayloadCache cache(128);
            auto prefetched = cache.acquire_cached(source.identity(), 0, 128,
                [&](auto offset, void* data, auto count) { source.read(offset, data, count); });
            require(bool(prefetched), "registered-file prefetch did not fill");
            demand.payload_cache = &cache; demand.cache_offset = 0; demand.cache_bytes = 128;
            demand.cache_direct_upload = true;
            std::fill(output.begin(), output.end(), 0);
            demand.prepare();
            require(demand.source_read_bytes == 0 && demand.payload_lease.cache_hit,
                "prefill RAM hit was counted as a file read");
            require(std::equal(bytes.begin() + 26, bytes.begin() + 63,
                static_cast<const std::uint8_t*>(demand.upload_data())), "prefill lease changed exact source slice");
            require(output[0] == 0, "direct payload lease copied into staging");
            demand.payload_cache = nullptr; demand.prepare();
            require(demand.source_read_bytes == 37 && !demand.payload_lease,
                "uncached decode reused stale lease or lost source-byte count");
#if !defined(_WIN32)
            struct stat expected{}; require(::stat(name.c_str(), &expected) == 0, "stat failed");
            for (int fd = 0; fd < 1024; ++fd) {
                struct stat actual{};
                if (::fstat(fd, &actual) == 0 && actual.st_dev == expected.st_dev && actual.st_ino == expected.st_ino) {
                    require(owned_fd == -1, "unexpected second source descriptor"); owned_fd = fd;
                }
            }
            require(owned_fd >= 0 && (::fcntl(owned_fd, F_GETFD) & FD_CLOEXEC), "source descriptor not close-on-exec");
            require(::truncate(name.c_str(), 11) == 0, "truncate fixture failed");
            rejected([&] { source.read(0, output.data(), 1024); }, "actual short file read accepted");
#endif
        }
#if !defined(_WIN32)
        errno = 0;
        require(::fcntl(owned_fd, F_GETFD) == -1 && errno == EBADF, "source descriptor leaked");
#endif
        require(std::filesystem::remove(path), "source handle still prevents cleanup");
#if defined(_WIN32)
        std::printf("PASS: %d Windows file-source checks; POSIX truncate/CLOEXEC not run\n", checks);
#else
        std::printf("PASS: %d POSIX file-source checks including real short-read/CLOEXEC/close\n", checks);
#endif
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        std::error_code ec; std::filesystem::remove(path, ec);
        return 1;
    }
}
