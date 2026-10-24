// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include "ggml-backend-impl.h"
#include "glm_graph_backend.h"
#include "ggml_ops_shared_cache_budget.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace tsg_glm {

// A cached GLM graph owns one scheduler, allocated exactly once. Its original
// backend buffers keep their original type/context/interfaces; only allocation
// requests pass through these temporary buffer-type descriptors. Destroy the
// scheduler before this owner so every physical free precedes its credit refund.
// Unlike a growable shared scheduler, no partial-generation refund is inferred.
class graph_budget {
    // One shared allowance covers every wrapper and pending request in this
    // scheduler generation. Successful requests retain credit until this graph
    // owner dies, just like its tickets; individual scheduler frees do not lend it.
    std::size_t device_payload_limit_ = 0;
    ggml_backend_buffer_type_t strict_device_type_ = nullptr;
    std::atomic<std::size_t> device_payload_bytes_{0};
    std::atomic<bool> policy_refused_{false};
#ifdef TSG_GGML_TEST_HOOKS
    std::atomic<bool> fail_record_{false};
#endif
    void refuse_policy() noexcept { policy_refused_.store(true, std::memory_order_relaxed); }
    bool claim_payload(ggml_backend_buffer_type_t type, std::size_t bytes) noexcept {
        if (!device_payload_limit_) return true;
        if (policy_refused()) return false;
        if (bytes && type != strict_device_type_) { refuse_policy(); return false; }
        auto used = device_payload_bytes_.load(std::memory_order_relaxed);
        do {
            if (bytes > device_payload_limit_ - used) { refuse_policy(); return false; }
        } while (!device_payload_bytes_.compare_exchange_weak(used, used + bytes, std::memory_order_relaxed));
        return true;
    }
    struct payload_request {
        graph_budget& budget;
        std::size_t bytes;
        bool accepted, retained = false;
        payload_request(graph_budget& b, ggml_backend_buffer_type_t type, std::size_t count) noexcept
            : budget(b), bytes(count), accepted(b.claim_payload(type, count)) {}
        ~payload_request() {
            if (accepted && !retained && budget.device_payload_limit_)
                budget.device_payload_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
        }
    };
    struct buffer_type {
        ggml_backend_buffer_type descriptor{};
        ggml_backend_buffer_type_t original;
        int rank, kind;
        graph_budget& budget;
        std::mutex charges_mutex;
        std::vector<std::shared_ptr<tsg::SharedCacheCharge>> charges;
        static buffer_type& self(ggml_backend_buffer_type_t t) { return *static_cast<buffer_type*>(t->context); }
        static const char* name(ggml_backend_buffer_type_t t) { return ggml_backend_buft_name(self(t).original); }
        static size_t alignment(ggml_backend_buffer_type_t t) { return ggml_backend_buft_get_alignment(self(t).original); }
        static size_t maximum(ggml_backend_buffer_type_t t) { return ggml_backend_buft_get_max_size(self(t).original); }
        static size_t tensor_size(ggml_backend_buffer_type_t t, const ggml_tensor* v) { return ggml_backend_buft_get_alloc_size(self(t).original, v); }
        static size_t tensors_size(ggml_backend_buffer_type_t t, ggml_tensor** v, int n) { return ggml_backend_buft_get_alloc_size_n(self(t).original, v, n); }
        static bool host(ggml_backend_buffer_type_t t) { return ggml_backend_buft_is_host(self(t).original); }
        template<class Allocate> ggml_backend_buffer_t allocate(size_t bytes, Allocate allocate_original) noexcept {
            try {
                // Declaration order matters: failure frees the physical buffer,
                // then its ticket, then this pending cap reservation.
                payload_request request(budget, original, bytes);
                if (!request.accepted) return nullptr;
                auto charge = bytes ? tsg::SharedCacheCharge::reserve(rank, kind, bytes) : nullptr;
                if (bytes && !charge) return nullptr;
                if (bytes && budget.device_payload_limit_ && !charge->is_accounted()) {
                    budget.refuse_policy(); return nullptr;
                }
                std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(allocate_original(), ggml_backend_buffer_free);
                if (!buffer) return nullptr;
                const size_t actual = ggml_backend_buffer_get_size(buffer.get());
                if (budget.device_payload_limit_ && (actual != bytes || buffer->buft != original
                    || ggml_backend_buffer_is_multi_buffer(buffer.get()))) {
                    budget.refuse_policy(); return nullptr;
                }
                if (!charge && actual) return nullptr;
                if (charge) {
                    std::lock_guard<std::mutex> lock(charges_mutex);
#ifdef TSG_GGML_TEST_HOOKS
                    if (budget.fail_record_.exchange(false, std::memory_order_relaxed)) throw std::bad_alloc();
#endif
                    charges.push_back(charge);
                    try {
                        if (!charge->commit(actual)) { charges.pop_back(); return nullptr; }
                    } catch (...) { charges.pop_back(); throw; }
                }
                request.retained = true;
                return buffer.release();
            } catch (...) { return nullptr; }
        }
        static ggml_backend_buffer_t alloc(ggml_backend_buffer_type_t t, size_t bytes) {
            auto& owner = self(t);
            // The pinned ggml helper forwards nonzero requests directly. Call
            // that interface here so a C++ allocator exception is caught before
            // crossing ggml's extern-C bridge (MSVC /EHsc assumes C never throws).
            return owner.allocate(bytes, [&] { return bytes ? owner.original->iface.alloc_buffer(owner.original, bytes)
                : ggml_backend_buft_alloc_buffer(owner.original, bytes); });
        }
        static ggml_backend_buffer_t alloc_n(ggml_backend_buffer_type_t t, ggml_tensor** tensors, int count) {
            auto& owner = self(t);
            if (owner.budget.device_payload_limit_) { owner.budget.refuse_policy(); return nullptr; }
            try {
                return owner.allocate(ggml_backend_buft_get_alloc_size_n(owner.original, tensors, count),
                    [&] { return ggml_backend_buft_alloc_buffer_n(owner.original, tensors, count); });
            } catch (...) { return nullptr; }
        }
        buffer_type(graph_budget& scope, ggml_backend_buffer_type_t source, int device_rank, int allocation_kind)
            : original(source), rank(device_rank), kind(allocation_kind), budget(scope) {
            descriptor.iface = { name, alloc, (scope.device_payload_limit_ || source->iface.alloc_buffer_n) ? alloc_n : nullptr,
                alignment, maximum, tensor_size, source->iface.get_alloc_size_n ? tensors_size : nullptr, host };
            descriptor.device = source->device;
            descriptor.context = this;
        }
    };
    std::vector<std::unique_ptr<buffer_type>> owners_;
    std::vector<ggml_backend_buffer_type_t> types_;
    std::vector<std::unique_ptr<graph_backend>> backend_owners_;
    std::vector<ggml_backend_t> backends_;
    static ggml_backend_buffer_type_t unwrap(void* context, ggml_backend_buffer_type_t type) {
        const auto& owner = *static_cast<graph_budget*>(context);
        for (const auto& item : owner.owners_) if (&item->descriptor == type) return item->original;
        return type;
    }
public:
    graph_budget() = default;
    graph_budget(const graph_budget&) = delete;
    graph_budget& operator=(const graph_budget&) = delete;
    ggml_backend_buffer_type_t* prepare(ggml_backend_buffer_type_t* types, ggml_backend_t* backends,
        int count, int n_gpu, int kind, std::size_t device_payload_limit = 0,
        ggml_backend_buffer_type_t strict_device_type = nullptr) {
        if (!owners_.empty()) return nullptr; // a graph scheduler has a single allocation generation
        policy_refused_.store(false, std::memory_order_relaxed);
        device_payload_limit_ = device_payload_limit; strict_device_type_ = strict_device_type;
        device_payload_bytes_.store(0, std::memory_order_relaxed);
        if (!types || !backends || count <= 0 || n_gpu < 0 || n_gpu > count) {
            if (device_payload_limit_) refuse_policy();
            return nullptr;
        }
        if (device_payload_limit_ && (kind != 5 || n_gpu != 1 || !strict_device_type || !backends[0]
            || !strict_device_type->device || ggml_backend_buft_is_host(strict_device_type)
            || !strict_device_type->iface.alloc_buffer
            || ggml_backend_dev_type(strict_device_type->device) != GGML_BACKEND_DEVICE_TYPE_GPU
            || ggml_backend_get_default_buffer_type(backends[0]) != strict_device_type
            || strict_device_type->iface.alloc_buffer_n || strict_device_type->iface.get_alloc_size_n)) {
            refuse_policy(); return nullptr;
        }
        for (int i = 0; i < count; ++i) {
            if (!types[i] || !backends[i] || (device_payload_limit_
                && types[i] != strict_device_type && !ggml_backend_buft_is_host(types[i]))) {
                if (device_payload_limit_) refuse_policy();
                return nullptr;
            }
        }
        for (int i = 0; i < count; ++i) {
            const int rank = ggml_backend_buft_is_host(types[i]) ? n_gpu :
                [&] { for (int d = 0; d < n_gpu; ++d) if (ggml_backend_get_default_buffer_type(backends[d]) == types[i]) return d; return 0; }();
            auto owner = std::unique_ptr<buffer_type>(new buffer_type(*this, types[i], rank, kind));
            types_.push_back(&owner->descriptor);
            owners_.push_back(std::move(owner));
            auto backend = std::unique_ptr<graph_backend>(new graph_backend(backends[i], this, unwrap));
            backends_.push_back(backend->get());
            backend_owners_.push_back(std::move(backend));
        }
        return types_.data();
    }
    // The caller proves the supplied exact GPU type is the actual CUDA type.
    // A known scheduler facade can own a distinct device record while returning
    // that same physical type; its device pointer is not buffer ownership proof.
    // A policy refusal is sticky for this prepared generation; ordinary ticket
    // or physical allocation failures remain eligible for existing reclaim.
    bool policy_refused() const noexcept { return policy_refused_.load(std::memory_order_relaxed); }
    std::size_t device_payload_bytes() const noexcept { return device_payload_bytes_.load(std::memory_order_relaxed); }
#ifdef TSG_GGML_TEST_HOOKS
    void test_fail_next_charge_record() noexcept { fail_record_.store(true, std::memory_order_relaxed); }
#endif
    ggml_backend_t* scheduler_backends() { return backends_.data(); }
    ggml_backend_t scheduler_backend(ggml_backend_t original) {
        for (const auto& facade : backend_owners_) if (facade->original() == original) return facade->get();
        return nullptr;
    }
};
}
