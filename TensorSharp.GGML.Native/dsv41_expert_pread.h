// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include "dsv4_exact_read.h"
#include <memory>
#include <stdexcept>
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#endif

namespace tsg_dsv41_pread {

inline bool enabled(const char * value) { return value && std::strcmp(value,"1") == 0; }

// Open each required shard once, before serving. In particular, reopening a
// FUSE file for every expert can invalidate pages populated by earlier reads.
// pread is offset based and permits concurrent requests on this descriptor.
class file {
    int fd_ = -1;
    uint64_t bytes_ = 0;
    std::string path_;
    file() = default;
public:
    file(const file &) = delete;
    file & operator=(const file &) = delete;
    ~file() {
#if !defined(_WIN32)
        if (fd_ >= 0) ::close(fd_);
#endif
    }
    static std::shared_ptr<file> open(const std::string & path) {
#if defined(_WIN32)
        (void)path;
        throw std::runtime_error("DeepSeek selected-expert pread requires Linux/POSIX");
#else
        auto result = std::shared_ptr<file>(new file);
        result->path_ = path;
        result->fd_ = ::open(path.c_str(),O_RDONLY | O_CLOEXEC);
        if (result->fd_ < 0)
            throw std::runtime_error("DeepSeek expert pread open " + path + ": " + std::strerror(errno));
        struct stat st;
        if (::fstat(result->fd_,&st) || st.st_size <= 0)
            throw std::runtime_error("DeepSeek expert pread has invalid file size: " + path);
        result->bytes_ = uint64_t(st.st_size);
        return result;
#endif
    }
    uint64_t bytes() const { return bytes_; }
    void read_exact(void * destination, size_t bytes, uint64_t offset) const {
        if (!destination || !bytes || offset > bytes_ || bytes > bytes_-offset)
            throw std::runtime_error("DeepSeek expert pread range is outside the verified shard: " + path_);
#if !defined(_WIN32)
        const auto r = tsg_dsv4::read_range_exact(destination,bytes,offset,
            [this](uint64_t off,void * dst,size_t n) { return tsg_dsv4::pread_at(fd_,off,dst,n); },
            [](const tsg_dsv4::exact_read_retry &) {});
        if (!r.ok) throw std::runtime_error("DeepSeek expert pread " + path_ + ": " +
            tsg_dsv4::exact_read_failure(r,bytes,offset));
#else
        (void)offset;
        throw std::runtime_error("DeepSeek selected-expert pread requires Linux/POSIX");
#endif
    }
};

struct file_range {
    std::shared_ptr<file> source;
    uint64_t offset = 0;
    size_t bytes = 0;
    bool valid(size_t expected) const {
        return source && bytes == expected && offset <= source->bytes() && bytes <= source->bytes()-offset;
    }
    void read_exact(void * destination,size_t count,size_t relative) const {
        if (relative > bytes || count > bytes-relative || !valid(bytes))
            throw std::runtime_error("DeepSeek expert pread tensor range is invalid");
        source->read_exact(destination,count,offset+relative);
    }
};

} // namespace tsg_dsv41_pread
