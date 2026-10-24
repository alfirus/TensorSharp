// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml_ops_shared_cache_budget.h"
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>

namespace tsg_glm {
inline bool device_baton_bytes(std::size_t hidden, std::size_t channels,
    std::size_t tokens, std::size_t& bytes) noexcept {
    bytes = 0;
    const auto limit = static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
    if (!hidden || !channels || !tokens || hidden > limit / sizeof(float)
        || channels > limit / (hidden * sizeof(float))
        || tokens > limit / (hidden * channels * sizeof(float))) return false;
    bytes = hidden * channels * tokens * sizeof(float); return true;
}
// The caller proves this is the real standard CUDA buft of its original backend.
// No graph captures this bank pointer. Every transfer clones the ORIGINAL
// endpoint descriptor: equal raw byte counts alone are not a layout proof.
class device_baton {
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_backend_buffer_type_t type_ = nullptr;
    std::shared_ptr<tsg::SharedCacheCharge> charge_;
    std::size_t bytes_ = 0;
public:
    std::uint64_t input_calls = 0, output_calls = 0, input_bytes = 0, output_bytes = 0;
    device_baton() = default;
    device_baton(const device_baton&) = delete;
    device_baton& operator=(const device_baton&) = delete;
    ~device_baton() noexcept {
        // GlmFreeChecked must retain the complete model/backend until close()
        // succeeds. Never let a destructor silently use the legacy void refund.
        if (owned()) std::terminate();
    }
    bool owned() const noexcept { return buffer_ || charge_; }
    std::size_t physical_bytes() const noexcept { return buffer_ ? bytes_ : 0; }
    std::size_t held_ticket_bytes() const noexcept { return charge_ ? charge_->reserved_bytes() : 0; }
    bool open(ggml_backend_buffer_type_t type, std::size_t bytes) {
        if (owned() || !type || !bytes || !tsg::SharedCacheCharge::graph_accounting_attached()
            || !tsg::SharedCacheCharge::checked_release_attached()) return false;
        input_calls = output_calls = input_bytes = output_bytes = 0;
        // Publish this exact charge into the already model-owned controller
        // BEFORE any physical allocation. Failed admission has no payload.
        charge_ = tsg::SharedCacheCharge::reserve(0, 5, bytes);
        if (!charge_ || !charge_->is_accounted()) return false;
        type_ = type; bytes_ = bytes;
        buffer_ = ggml_backend_buft_alloc_buffer(type, bytes);
        if (!buffer_) return false;
        if (ggml_backend_buffer_get_type(buffer_) != type
            || ggml_backend_buffer_get_size(buffer_) != bytes
            || !ggml_backend_buffer_get_base(buffer_) || !charge_->commit(bytes)) return false;
        return true;
    }
    bool transfer(ggml_tensor* original, std::size_t offset, bool to_bank) {
        if (!buffer_ || !charge_ || !charge_->is_committed() || !original
            || !original->buffer || !original->data || original->view_src
            || original->type != GGML_TYPE_F32 || !ggml_is_contiguous(original)
            || ggml_backend_buffer_get_type(original->buffer) != type_
            || original->buffer->iface.free_buffer != buffer_->iface.free_buffer) return false;
        const std::size_t count = ggml_nbytes(original);
        if (!count || offset > bytes_ || count > bytes_ - offset) return false;
        const auto base = reinterpret_cast<std::uintptr_t>(ggml_backend_buffer_get_base(original->buffer));
        const auto data = reinterpret_cast<std::uintptr_t>(original->data);
        const auto size = ggml_backend_buffer_get_size(original->buffer);
        if (data < base || data - base > size || count > size - (data - base)) return false;
        ggml_tensor slice = *original;
        slice.buffer = buffer_; slice.view_src = nullptr; slice.view_offs = 0;
        slice.data = static_cast<std::uint8_t*>(ggml_backend_buffer_get_base(buffer_)) + offset;
        // Call the destination's bool callback DIRECTLY through the public API.
        // Pinned standard CUDA implementation synchronizes its D2D stream.
        // Generic tensor_copy(_async) is forbidden: it can fall back to Host.
        const bool copied = to_bank ? ggml_backend_buffer_copy_tensor(original, &slice)
            : ggml_backend_buffer_copy_tensor(&slice, original);
        if (!copied) return false;
        if (to_bank) { ++output_calls; output_bytes += count; }
        else { ++input_calls; input_bytes += count; }
        return true;
    }
    template<class Drain> bool close(Drain&& drain) noexcept {
        try {
            if (buffer_) {
                if (!drain()) return false; // Preserve physical owner on uncertainty.
                // Stock CUDA free is void/CUDA_CHECK; a vendor free failure can
                // abort the process. No recoverable physical-free claim is made.
                ggml_backend_buffer_free(buffer_); buffer_ = nullptr;
            }
            if (charge_) {
                if (!charge_->release_checked()) return false; // SAME ticket retained.
                charge_.reset();
            }
            type_ = nullptr; bytes_ = 0; return true;
        } catch (...) { return false; }
    }
};
}
