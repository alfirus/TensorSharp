// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace tsg {
// Exact loader-registered file extent. Never infer file offsets from mmap
// addresses: views may start at nonzero offsets or belong to different shards.
class ExpertFileSource {
#if defined(_WIN32)
    HANDLE file_ = INVALID_HANDLE_VALUE;
#else
    int file_ = -1;
#endif
    std::uint64_t offset_ = 0, bytes_ = 0;
public:
    ExpertFileSource(const char* utf8_path, std::uint64_t offset, std::uint64_t bytes)
        : offset_(offset), bytes_(bytes)
    {
        if (!utf8_path || !*utf8_path || !bytes || offset > UINT64_MAX - bytes)
            throw std::runtime_error("Invalid expert file extent");
#if defined(_WIN32)
        if (!utf8_path || !bytes || offset > UINT64_MAX - bytes) throw std::runtime_error("Invalid expert file extent");
        int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8_path, -1, nullptr, 0);
        if (count <= 1 || count > 32768) throw std::runtime_error("Invalid expert source path");
        std::vector<wchar_t> path(count);
        if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8_path, -1, path.data(), count))
            throw std::runtime_error("Cannot decode expert source path");
        file_ = CreateFileW(path.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
            FILE_FLAG_OVERLAPPED, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot open expert file source: " + std::to_string(GetLastError()));
        LARGE_INTEGER size{};
        if (!GetFileSizeEx(file_, &size) || size.QuadPart < 0
            || offset > static_cast<std::uint64_t>(size.QuadPart)
            || bytes > static_cast<std::uint64_t>(size.QuadPart) - offset) {
            CloseHandle(file_); file_ = INVALID_HANDLE_VALUE;
            throw std::runtime_error("Expert file extent exceeds its shard");
        }
#else
        if (offset + bytes > static_cast<std::uint64_t>((std::numeric_limits<off_t>::max)()))
            throw std::runtime_error("Expert file extent exceeds platform offsets");
        file_ = ::open(utf8_path, O_RDONLY | O_CLOEXEC);
        if (file_ < 0) throw std::runtime_error("Cannot open expert file source: " + std::to_string(errno));
        struct stat info{};
        if (::fstat(file_, &info) || info.st_size < 0 || offset > static_cast<std::uint64_t>(info.st_size)
            || bytes > static_cast<std::uint64_t>(info.st_size) - offset) {
            ::close(file_); file_ = -1;
            throw std::runtime_error("Expert file extent exceeds its shard");
        }
#endif
    }
    ~ExpertFileSource() {
#if defined(_WIN32)
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
#else
        if (file_ >= 0) ::close(file_);
#endif
    }
    ExpertFileSource(const ExpertFileSource&) = delete;
    ExpertFileSource& operator=(const ExpertFileSource&) = delete;
    std::uint64_t bytes() const { return bytes_; }
    void read(std::uint64_t relative, void* destination, std::size_t count) const {
        if (relative > bytes_ || count > bytes_ - relative || (!destination && count))
            throw std::runtime_error("Expert file read exceeds registered extent");
#if defined(_WIN32)
        struct Event {
            HANDLE handle = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            ~Event() { if (handle) CloseHandle(handle); }
        };
        // Each worker has at most one outstanding request. Reuse its event;
        // do not create/close a kernel object for every expert projection.
        thread_local Event event;
        if (!event.handle) throw std::runtime_error("Cannot create expert read event");
        auto* target = static_cast<std::uint8_t*>(destination);
        while (count) {
            const auto position = offset_ + relative;
            OVERLAPPED operation{};
            operation.Offset = static_cast<DWORD>(position);
            operation.OffsetHigh = static_cast<DWORD>(position >> 32);
            operation.hEvent = event.handle;
            ResetEvent(event.handle);
            DWORD got = 0, chunk = static_cast<DWORD>(std::min<std::size_t>(count, 1u << 30));
            BOOL ok = ReadFile(file_, target, chunk, &got, &operation);
            if (!ok && GetLastError() == ERROR_IO_PENDING)
                ok = GetOverlappedResult(file_, &operation, &got, TRUE);
            if (!ok || !got) throw std::runtime_error("Expert file read failed or truncated: " + std::to_string(GetLastError()));
            target += got; relative += got; count -= got;
        }
#else
        auto* target = static_cast<std::uint8_t*>(destination);
        while (count) {
            const auto chunk = std::min<std::size_t>(count, std::size_t(1) << 30);
            const auto got = ::pread(file_, target, chunk, static_cast<off_t>(offset_ + relative));
            if (got < 0 && errno == EINTR) continue;
            if (got <= 0) throw std::runtime_error("Expert file read failed or truncated: " + std::to_string(errno));
            target += got; relative += static_cast<std::size_t>(got); count -= static_cast<std::size_t>(got);
        }
#endif
    }
};

struct ExpertReadTask {
    const ExpertFileSource* source;
    std::uint64_t offset;
    std::uint8_t* destination;
    std::size_t bytes;
    void prepare() {
        source->read(offset, destination, bytes);
    }
};
void read_expert_files(std::vector<ExpertReadTask>& tasks,
    const std::function<void(std::size_t)>& consume);
}
