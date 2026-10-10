// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include "ggml_ops_shared_cache_budget.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace tsg_glm {

struct upload_allocator {
    static uint8_t * allocate(size_t bytes) { return new uint8_t[bytes]; }
    static void release(uint8_t * data) noexcept { delete[] data; }
};

// A worker's disposable read/upload scratch. Unlike vector growth, capacity is
// exact and charged BEFORE physical allocation. Contents are overwritten by the
// next file read, so growth can free the old buffer before reserving its
// replacement. The caller must finish the synchronous backend upload first.
// The charge covers payload capacity, not allocator bookkeeping, FILE buffers,
// thread stacks, driver staging or the model's pageable mappings.
template<class Allocator = upload_allocator>
class upload_buffer {
    struct deleter { void operator()(uint8_t * data) const noexcept { Allocator::release(data); } };
    // Destruction order also frees physical storage before returning credit.
    std::shared_ptr<tsg::SharedCacheCharge> charge_;
    std::unique_ptr<uint8_t[], deleter> data_;
    size_t size_ = 0;
public:
    upload_buffer() = default;
    upload_buffer(const upload_buffer &) = delete;
    upload_buffer & operator=(const upload_buffer &) = delete;
    size_t size() const noexcept { return size_; }
    uint8_t * data() noexcept { return data_.get(); }
    void clear() noexcept {
        data_.reset();
        size_ = 0;
        charge_.reset();
    }
    void resize(size_t bytes) {
        if (bytes <= size_) return;
        clear();
        auto charge = tsg::SharedCacheCharge::reserve(0, 3, bytes);
        if (!charge) throw std::runtime_error("GLM upload workspace exceeds shared host budget");
        std::unique_ptr<uint8_t[], deleter> data(Allocator::allocate(bytes));
        if (!data) throw std::bad_alloc();
        if (!charge->commit(bytes)) throw std::runtime_error("GLM upload workspace commit refused");
        charge_ = std::move(charge);
        data_ = std::move(data);
        size_ = bytes;
    }
};
} // namespace tsg_glm
