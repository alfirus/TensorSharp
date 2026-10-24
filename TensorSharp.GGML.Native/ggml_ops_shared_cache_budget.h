// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>

namespace tsg
{
    // Callbacks only perform accounting. They must not throw or call native
    // cache APIs. The registry lock serializes callbacks and attach/detach;
    // cache locks may precede it, but this registry never takes a cache lock.
    class SharedCacheCharge
    {
    public:
        using Reserve = std::uint64_t (*)(void*, int, int, std::int64_t);
        using Commit = int (*)(void*, std::uint64_t);
        using Release = void (*)(void*, std::uint64_t);
        using ReleaseChecked = int (*)(void*, std::uint64_t);
    private:
        struct Registry {
            std::mutex mutex;
            void* context = nullptr;
            Reserve reserve = nullptr;
            Commit commit = nullptr;
            Release release = nullptr;
            ReleaseChecked release_checked = nullptr;
            // 0: device copies, 1: preload, 2: graph, 3: host payload,
            // 4: selected-expert cache (including its private graph allocation),
            // 5: Qwen4Exp joint-sequence arena (graph opt-in, like kind 2),
            // 6: persistent host expert payloads (host opt-in, separate owner
            //    from the transient request staging charged as kind 3),
            // 7: admitted speculative draft graphs and recurrent snapshots.
            //    Unlike legacy kinds, this owner always requires accounting.
            std::uint64_t live[8] = {};
            bool graph_buffers = false;
            bool host_buffers = false;
        };
        static Registry& registry()
        {
            // Native cache teardown can run from C runtime finalizers. Keep the
            // bookkeeping gate alive for as long as any cache entry can exist.
            static Registry* value = new Registry;
            return *value;
        }
        std::uint64_t token_ = 0;
        std::size_t reserved_ = 0;
        bool live_ = false;
        bool committed_ = false;
        int kind_ = 0;
        int rank_ = 0;
    public:
        SharedCacheCharge() = default;
        SharedCacheCharge(const SharedCacheCharge&) = delete;
        SharedCacheCharge& operator=(const SharedCacheCharge&) = delete;
        // Immutable after reserve(): this exact ticket, not an advisory registry
        // snapshot, tells a caller whether allocation is under shared accounting.
        bool is_accounted() const noexcept { return token_ != 0; }
        // Immutable reservation certificate. Callers still own the charge and
        // serialize its commit/release; these values do not prove physical use.
        int rank() const noexcept { return rank_; }
        int kind() const noexcept { return kind_; }
        std::size_t reserved_bytes() const noexcept { return reserved_; }
        bool is_committed() const {
            auto& r = registry();
            std::lock_guard<std::mutex> lock(r.mutex);
            return live_ && committed_;
        }
        static bool checked_release_attached() {
            auto& r = registry();
            std::lock_guard<std::mutex> lock(r.mutex);
            return r.reserve != nullptr && r.release_checked != nullptr;
        }
        static bool host_accounting_attached() {
            auto& r = registry();
            std::lock_guard<std::mutex> lock(r.mutex);
            return r.reserve != nullptr && r.host_buffers;
        }
        static bool graph_accounting_attached() {
            auto& r = registry();
            std::lock_guard<std::mutex> lock(r.mutex);
            return r.reserve != nullptr && r.graph_buffers;
        }

        static bool attach(void* context, Reserve reserve, Commit commit, Release release,
            bool graph_buffers = false, bool host_buffers = false, ReleaseChecked release_checked = nullptr)
        {
            if (context == nullptr || reserve == nullptr || commit == nullptr || release == nullptr) return false;
            auto& r = registry();
            std::lock_guard<std::mutex> lock(r.mutex);
            if (r.reserve != nullptr || r.live[0] != 0 || r.live[1] != 0 || r.live[4] != 0
                || r.live[7] != 0 || (graph_buffers && (r.live[2] != 0 || r.live[5] != 0)) || (host_buffers && (r.live[3] != 0 || r.live[6] != 0))) return false;
            r.context = context; r.reserve = reserve; r.commit = commit; r.release = release;
            r.release_checked = release_checked;
            r.graph_buffers = graph_buffers;
            r.host_buffers = host_buffers;
            return true;
        }
        static bool detach(void* context)
        {
            auto& r = registry();
            std::lock_guard<std::mutex> lock(r.mutex);
            if (r.context != context || r.reserve == nullptr || r.live[0] != 0 || r.live[1] != 0 || r.live[4] != 0
                || r.live[7] != 0 || (r.graph_buffers && (r.live[2] != 0 || r.live[5] != 0)) || (r.host_buffers && (r.live[3] != 0 || r.live[6] != 0))) return false;
            r.context = nullptr; r.reserve = nullptr; r.commit = nullptr; r.release = nullptr;
            r.release_checked = nullptr;
            r.graph_buffers = false;
            r.host_buffers = false;
            return true;
        }
        static std::shared_ptr<SharedCacheCharge> reserve(int rank, int kind, std::size_t bytes)
        {
            if (kind < 0 || kind > 7 || bytes == 0 || bytes > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) return {};
            auto charge = std::make_shared<SharedCacheCharge>();
            auto& r = registry();
            std::lock_guard<std::mutex> lock(r.mutex);
            if (kind == 7 && (r.reserve == nullptr || !r.graph_buffers)) return {};
            if (r.reserve != nullptr && ((kind != 2 && kind != 5 && kind != 7) || r.graph_buffers) && ((kind != 3 && kind != 6) || r.host_buffers)) {
                charge->token_ = r.reserve(r.context, rank, kind, static_cast<std::int64_t>(bytes));
                if (charge->token_ == 0) return {};
            }
            // Also count unconfigured allocations: an attach must never miss a
            // buffer that is being allocated, or adopt already-live cache bytes.
            ++r.live[kind];
            charge->kind_ = kind;
            charge->rank_ = rank;
            charge->live_ = true;
            charge->reserved_ = bytes;
            return charge;
        }
        bool commit(std::size_t actual)
        {
            auto& r = registry();
            std::lock_guard<std::mutex> lock(r.mutex);
            if (!live_ || (token_ != 0 && actual > reserved_)) return false;
            if (committed_) return true;
            if (token_ != 0 && !r.commit(r.context, token_)) return false;
            committed_ = true;
            return true;
        }
        // Caller has already freed the physical owner. A failed acknowledgment
        // keeps this exact ticket and registry lifetime for retry. The callback
        // may have completed ledger release before failing its acknowledgment;
        // it must therefore support retry of the same token without double credit.
        // Never substitute the legacy void callback for this confirmation.
        bool release_checked() {
            auto& r = registry();
            std::lock_guard<std::mutex> lock(r.mutex);
            if (!live_) return true;
            if (token_ != 0 && (r.release_checked == nullptr || r.release_checked(r.context, token_) != 1))
                return false;
            --r.live[kind_];
            live_ = false;
            return true;
        }
        ~SharedCacheCharge()
        {
            if (!live_) return;
            auto& r = registry();
            std::lock_guard<std::mutex> lock(r.mutex);
            if (token_ != 0) r.release(r.context, token_);
            --r.live[kind_];
        }
    };
}
