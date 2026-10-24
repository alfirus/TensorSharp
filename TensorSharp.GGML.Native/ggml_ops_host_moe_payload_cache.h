// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <iterator>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tsg {
// Source identities must outlive address reuse. A new registration receives a
// new identity even if its mapping or ExpertFileSource reuses an old address.
inline std::uint64_t next_host_moe_payload_source_id() {
    static std::atomic<std::uint64_t> next{1};
    auto value = next.load(std::memory_order_relaxed);
    for (;;) {
        if (value == (std::numeric_limits<std::uint64_t>::max)())
            throw std::overflow_error("Expert payload source identities exhausted");
        if (next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed)) return value;
    }
}

// A bounded cache of original, immutable expert projection bytes. Storage is
// pageable by default; the caller may provide an explicitly pinned allocator.
// Loader, allocation, reservation and commit callbacks run outside its mutex.
// Storage/charge-owner destruction may run under the mutex during eviction;
// those destructors must not reenter this cache. Join readers before destruction;
// returned leases may outlive the cache and retain physical storage and charges.
class HostMoePayloadCache {
public:
    // Reserve before allocation, commit after allocation. A null reservation or
    // false commit declines this optional admission. The allocation is freed
    // before destroying the owner and returning its physical charge.
    using ReserveCharge = std::function<std::shared_ptr<void>(std::size_t)>;
    using CommitCharge = std::function<bool(const std::shared_ptr<void>&, std::size_t)>;
    struct Storage {
        std::shared_ptr<void> owner;
        std::uint8_t* data = nullptr;
        std::size_t bytes = 0;
        bool pinned = false;
        explicit operator bool() const noexcept { return owner && data && bytes; }
    };
    // Storage must report exactly the requested allocation size. An empty or
    // differently sized result is freed and declines this optional admission.
    using AllocateStorage = std::function<Storage(std::size_t)>;
    struct Lease {
        std::shared_ptr<const void> owner;
        const std::uint8_t* data = nullptr;
        std::size_t bytes = 0;
        bool pinned = false, cache_hit = false;
        explicit operator bool() const noexcept { return owner != nullptr; }
    };
    struct Range { std::uint64_t source, offset; std::size_t bytes; };
    // Writable leases are unpublished and may outlive the cache. The caller
    // must fence queued writes before publishing or releasing this lease.
    struct ExternalWrite {
        std::shared_ptr<void> owner;
        std::uint8_t* data = nullptr;
        std::size_t bytes = 0;
        bool pinned = false;
        explicit operator bool() const noexcept { return owner != nullptr; }
    };
    struct Stats {
        std::size_t budget_bytes = 0, reserved_bytes = 0, resident_bytes = 0, entries = 0,
            frequency_counter_bytes = 0, pinned_bytes = 0, external_free_bytes = 0, external_free_entries = 0;
        std::uint64_t hits = 0, misses = 0, admitted = 0, bypasses = 0, waits = 0,
            evictions = 0, failures = 0, source_bytes = 0, hit_bytes = 0, bypass_bytes = 0, reuses = 0,
            frequency_updates = 0, frequency_ages = 0, frequency_rejections = 0, prefetch_hit_bytes = 0,
            external_bytes = 0, external_allocations = 0, external_allocation_bytes = 0,
            external_allocation_ns = 0, external_pool_reuses = 0, external_resident_reuses = 0,
            external_frees = 0, external_free_ns = 0, external_recycled_bytes = 0;
    };
private:
    struct Key {
        std::uint64_t source, offset;
        std::size_t bytes;
        bool operator==(const Key& other) const noexcept {
            return source == other.source && offset == other.offset && bytes == other.bytes;
        }
    };
    struct Hash {
        std::size_t operator()(const Key& key) const noexcept {
            auto result = std::hash<std::uint64_t>{}(key.source);
            result ^= std::hash<std::uint64_t>{}(key.offset) + 0x9e3779b9u + (result << 6) + (result >> 2);
            return result ^ (std::hash<std::size_t>{}(key.bytes) + 0x9e3779b9u + (result << 6) + (result >> 2));
        }
    };
    struct Usage {
        std::atomic<std::size_t> bytes{0}, pinned_bytes{0};
        std::atomic<std::uint64_t> external_frees{0}, external_free_ns{0};
    };
    enum class Status { loading, ready, bypass, failed };
    struct Entry {
        Key key;
        std::shared_ptr<Usage> usage;
        bool reserved = false, pinned_reserved = false, prefetched = false, external = false, retired = false;
        Status status = Status::loading; // Protected by the cache mutex.
        std::list<Key>::iterator recency;
        std::shared_ptr<void> charge;
        Storage storage;
        std::exception_ptr failure;
        Entry(Key value, std::shared_ptr<Usage> owner) : key(value), usage(std::move(owner)) {}
        void release_storage() noexcept {
            const bool measure = external && bool(storage);
            const auto started = measure ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            storage = {};
            if (measure) {
                usage->external_frees.fetch_add(1, std::memory_order_relaxed);
                usage->external_free_ns.fetch_add(static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()),
                    std::memory_order_relaxed);
            }
            charge.reset(); // Refund only after the allocation is physically freed.
            if (pinned_reserved) { usage->pinned_bytes.fetch_sub(key.bytes, std::memory_order_relaxed); pinned_reserved = false; }
            if (reserved) { usage->bytes.fetch_sub(key.bytes, std::memory_order_relaxed); reserved = false; }
        }
        ~Entry() { release_storage(); }
    };
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::size_t budget_;
    ReserveCharge reserve_;
    CommitCharge commit_;
    AllocateStorage allocate_;
    std::shared_ptr<Usage> usage_ = std::make_shared<Usage>();
    std::unordered_map<Key, std::shared_ptr<Entry>, Hash> entries_;
    std::list<Key> recency_; // Most recent decode read first; prefill never changes it.
    // Weak ownership: abandoning an unpublished write frees its allocation.
    // Maintenance retires live writers so late completion cannot resurrect data.
    std::vector<std::weak_ptr<Entry>> external_;
    // Completed exclusive promotions may return their already charged storage.
    // These entries have no source identity and are never visible to readers.
    // Physical and pinned bytes remain in Usage until reuse or real retirement.
    std::vector<std::shared_ptr<Entry>> external_free_;
    Stats counters_;
    // Fixed metadata, never one counter per encountered source/key. Collisions
    // can influence admission but never payload identity or returned bytes.
    static constexpr std::size_t frequency_slots_ = 65536;
    static constexpr unsigned frequency_epoch_ = 49152, frequency_sample_ = 64;
    const bool frequency_admission_;
    std::array<std::uint8_t, frequency_slots_> frequency_{};
    unsigned frequency_progress_ = 0;

    static Storage default_storage(std::size_t bytes) {
        auto* data = new std::uint8_t[bytes];
        // shared_ptr invokes the deleter if allocating its control block fails.
        return {std::shared_ptr<void>(data, [](void* pointer) { delete[] static_cast<std::uint8_t*>(pointer); }),
            data, bytes, false};
    }
    static Lease lease(const std::shared_ptr<Entry>& entry, std::size_t relative, std::size_t count, bool hit) {
        return {entry, entry->storage.data + relative, count, entry->storage.pinned, hit};
    }

    static std::uint64_t mix_frequency(std::uint64_t value) noexcept {
        value ^= value >> 30; value *= UINT64_C(0xbf58476d1ce4e5b9);
        value ^= value >> 27; value *= UINT64_C(0x94d049bb133111eb);
        return value ^ (value >> 31);
    }
    static std::array<std::size_t, 2> frequency_indexes(const Key& key) noexcept {
        const auto hash = mix_frequency(key.source ^ mix_frequency(key.offset)
            ^ mix_frequency(static_cast<std::uint64_t>(key.bytes) + UINT64_C(0x9e3779b97f4a7c15)));
        return {static_cast<std::size_t>(hash) & (frequency_slots_ - 1),
                static_cast<std::size_t>(hash >> 32) & (frequency_slots_ - 1)};
    }
    unsigned frequency_estimate(const Key& key) const noexcept {
        const auto indexes = frequency_indexes(key);
        return (std::min)(frequency_[indexes[0]], frequency_[indexes[1]]);
    }
    void observe_frequency(const Key& key) noexcept {
        ++counters_.frequency_updates;
        if (++frequency_progress_ == frequency_epoch_) {
            for (auto& value : frequency_) value >>= 1;
            frequency_progress_ = 0; ++counters_.frequency_ages;
        }
        const auto indexes = frequency_indexes(key);
        for (int i = 0; i < 2; ++i)
            if ((i == 0 || indexes[1] != indexes[0]) && frequency_[indexes[i]] < 255)
                ++frequency_[indexes[i]];
    }
#if defined(TSG_HOST_MOE_PAYLOAD_CACHE_TEST_HOOKS)
    inline static std::atomic<int> metadata_failure_{0};
    static void metadata_allocation(int stage) {
        int expected = stage;
        if (metadata_failure_.compare_exchange_strong(expected, 0)) throw std::bad_alloc();
    }
#else
    static void metadata_allocation(int) noexcept {}
#endif

    void remove_locked(const Key& key, const Entry* expected = nullptr) {
        const auto found = entries_.find(key);
        if (found == entries_.end() || (expected && found->second.get() != expected)) return;
        recency_.erase(found->second->recency);
        entries_.erase(found);
    }
    // In-flight readers retain the payload and its charge after eviction. Their
    // bytes still count, so a cache miss bypasses if they prevent admission.
    void trim_locked(std::size_t target, bool prefetch_only = false) {
        while (usage_->bytes.load(std::memory_order_relaxed) > target && !external_free_.empty())
            external_free_.pop_back();
        auto position = recency_.end();
        while (usage_->bytes.load(std::memory_order_relaxed) > target && position != recency_.begin()) {
            --position;
            const auto found = entries_.find(*position);
            if (found->second->status == Status::loading) continue;
            if (prefetch_only && (!found->second->prefetched || found->second.use_count() != 1)) continue;
            position = recency_.erase(position);
            entries_.erase(found);
            ++counters_.evictions;
        }
    }
    void finish(const std::shared_ptr<Entry>& entry, Status status, std::exception_ptr failure = {}) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            entry->failure = failure;
            entry->status = status;
            if (status != Status::ready) remove_locked(entry->key, entry.get());
            if (status == Status::failed) ++counters_.failures;
            if (status == Status::ready) ++counters_.admitted;
        }
        changed_.notify_all();
    }
public:
    explicit HostMoePayloadCache(std::size_t budget = 0, ReserveCharge reserve = {}, CommitCharge commit = {},
        bool frequency_admission = false, AllocateStorage allocate = {})
        : budget_(budget), reserve_(std::move(reserve)), commit_(std::move(commit)), allocate_(std::move(allocate)),
          frequency_admission_(frequency_admission) {
        if (commit_ && !reserve_) throw std::invalid_argument("Expert payload commit needs a reservation callback");
    }
    HostMoePayloadCache(const HostMoePayloadCache&) = delete;
    HostMoePayloadCache& operator=(const HostMoePayloadCache&) = delete;

    void set_budget(std::size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (bytes < budget_) for (auto& weak : external_) if (auto entry = weak.lock()) entry->retired = true;
        budget_ = bytes;
        trim_locked(bytes);
        // Loading entries cannot be freed yet. Unlink them under pressure so
        // completing a retired read cannot repopulate beyond the new budget.
        if (usage_->bytes.load(std::memory_order_relaxed) > bytes)
            for (auto it = entries_.begin(); it != entries_.end();) {
                if (it->second->status != Status::loading) { ++it; continue; }
                recency_.erase(it->second->recency); it = entries_.erase(it);
            }
    }
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& weak : external_) if (auto entry = weak.lock()) entry->retired = true;
        external_.clear();
        external_free_.clear();
        entries_.clear(); recency_.clear();
    }
    void erase_source(std::uint64_t source) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& weak : external_) if (auto entry = weak.lock())
            if (entry->key.source == source) entry->retired = true;
        for (auto it = entries_.begin(); it != entries_.end();) {
            if (it->first.source != source) { ++it; continue; }
            recency_.erase(it->second->recency);
            it = entries_.erase(it);
        }
    }
    // Request teardown runs this after joining workers and releasing its
    // leases. A concurrent reader still owns its bytes; demand-promoted data
    // belongs to the ordinary cache and is deliberately retained.
    void discard_prefetched() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = entries_.begin(); it != entries_.end();) {
            const auto& entry = it->second;
            if (!entry->prefetched || entry->status != Status::ready || entry.use_count() != 1) { ++it; continue; }
            recency_.erase(entry->recency);
            it = entries_.erase(it);
        }
    }
    Stats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto result = counters_;
        result.frequency_counter_bytes = frequency_.size();
        result.budget_bytes = budget_;
        result.reserved_bytes = usage_->bytes.load(std::memory_order_relaxed);
        result.pinned_bytes = usage_->pinned_bytes.load(std::memory_order_relaxed);
        result.external_frees = usage_->external_frees.load(std::memory_order_relaxed);
        result.external_free_ns = usage_->external_free_ns.load(std::memory_order_relaxed);
        result.external_free_entries = external_free_.size();
        for (const auto& entry : external_free_) result.external_free_bytes += entry->key.bytes;
        for (const auto& item : entries_) if (item.second->status == Status::ready) {
            result.resident_bytes += item.first.bytes; ++result.entries;
        }
        return result;
    }

    // Caller must first fence every DMA using these exclusive promotion leases.
    // A copied/retired lease cannot be reused. Pool retention never adds a new
    // reservation, returns credit early, or preserves a cacheable source key.
    std::size_t recycle_external(std::array<Lease, 3>& leases) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t recycled = 0;
        for (auto& value : leases) {
            if (!value || !value.cache_hit) continue;
            auto entry = std::const_pointer_cast<Entry>(std::static_pointer_cast<const Entry>(value.owner));
            if (entry.use_count() != 2 || !entry->external || entry->retired
                || entry->status != Status::ready || entry->usage != usage_ || !entry->storage
                || !budget_ || usage_->bytes.load(std::memory_order_relaxed) > budget_ || entries_.count(entry->key)) continue;
            try { metadata_allocation(6); external_free_.push_back(entry); }
            catch (const std::bad_alloc&) { continue; }
            entry->key.source = 0; entry->key.offset = 0;
            entry->status = Status::bypass; entry->retired = true;
            recycled += entry->key.bytes; counters_.external_recycled_bytes += entry->key.bytes;
            value = {};
        }
        return recycled;
    }

    // An exclusive promotion takes the complete triple atomically. Victim
    // callers discard incomplete remnants before a full file reload, otherwise
    // those leftovers would duplicate the newly resident GPU expert.
    std::array<Lease, 3> take_triple(const std::array<Range, 3>& ranges, bool discard_partial = false) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::array<std::shared_ptr<Entry>, 3> entries;
        for (int part = 0; part < 3; ++part) {
            const auto& range = ranges[part];
            if (!range.source || !range.bytes || range.offset > UINT64_MAX - range.bytes)
                throw std::invalid_argument("Invalid expert promotion range");
            for (int before = 0; before < part; ++before)
                if (ranges[before].source == range.source && ranges[before].offset == range.offset
                    && ranges[before].bytes == range.bytes)
                    throw std::invalid_argument("Expert promotion repeats a projection");
            const auto found = entries_.find({range.source, range.offset, range.bytes});
            if (found == entries_.end() || found->second->status != Status::ready) {
                if (discard_partial) for (const auto& partial : ranges)
                    remove_locked({partial.source, partial.offset, partial.bytes});
                ++counters_.misses; return {};
            }
            entries[part] = found->second;
        }
        std::array<Lease, 3> result;
        for (int part = 0; part < 3; ++part) {
            result[part] = lease(entries[part], 0, ranges[part].bytes, true);
            remove_locked(entries[part]->key, entries[part].get());
            ++counters_.hits; counters_.hit_bytes += ranges[part].bytes;
        }
        return result;
    }

    ExternalWrite begin_external(const Range& range, bool require_pinned = false) {
        if (!range.source || !range.bytes || range.offset > UINT64_MAX - range.bytes)
            throw std::invalid_argument("Invalid external expert payload range");
        const Key key{range.source, range.offset, range.bytes};
        std::shared_ptr<Entry> entry;
        bool reused = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            external_.erase(std::remove_if(external_.begin(), external_.end(), [](const auto& weak) {
                const auto value = weak.lock();
                return !value || value->retired || value->status != Status::loading;
            }), external_.end());
            if (entries_.count(key) || key.bytes > budget_) return {};
            for (const auto& weak : external_) if (auto value = weak.lock())
                if (value->key == key) return {};
            for (std::size_t index = 0; index < external_free_.size(); ++index) {
                const auto& available = external_free_[index];
                if (available->key.bytes != key.bytes || (require_pinned && !available->storage.pinned)) continue;
                entry = std::move(external_free_[index]);
                if (index + 1 < external_free_.size()) external_free_[index] = std::move(external_free_.back());
                external_free_.pop_back();
                entry->key = key; entry->status = Status::loading;
                ++counters_.reuses; ++counters_.external_pool_reuses; reused = true; break;
            }
            const auto used = usage_->bytes.load(std::memory_order_relaxed);
            if (!reused && used > budget_ - key.bytes) {
                for (auto position = recency_.rbegin(); position != recency_.rend(); ++position) {
                    const auto found = entries_.find(*position);
                    if (found->first.bytes != key.bytes || found->second->status != Status::ready
                        || found->second.use_count() != 1 || (require_pinned && !found->second->storage.pinned)) continue;
                    entry = std::move(found->second);
                    recency_.erase(entry->recency); entries_.erase(found);
                    entry->key = key; entry->status = Status::loading;
                    ++counters_.evictions; ++counters_.reuses; ++counters_.external_resident_reuses; reused = true; break;
                }
                if (!reused) trim_locked(budget_ - key.bytes);
            }
            if (!reused && usage_->bytes.load(std::memory_order_relaxed) > budget_ - key.bytes) return {};
            try {
                if (!reused) {
                    metadata_allocation(1);
                    entry = std::make_shared<Entry>(key, usage_);
                    usage_->bytes.fetch_add(key.bytes, std::memory_order_relaxed); entry->reserved = true;
                }
                entry->external = true; entry->retired = false; entry->prefetched = false;
                metadata_allocation(2); external_.push_back(entry);
            } catch (const std::bad_alloc&) { return {}; }
        }
        try {
            if (!reused) {
                if (reserve_) entry->charge = reserve_(key.bytes);
                if (!reserve_ || entry->charge) {
                    const auto started = std::chrono::steady_clock::now();
                    const auto record = [&] {
                        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - started).count();
                        std::lock_guard<std::mutex> lock(mutex_);
                        ++counters_.external_allocations; counters_.external_allocation_bytes += key.bytes;
                        counters_.external_allocation_ns += static_cast<std::uint64_t>(elapsed);
                    };
                    try { entry->storage = allocate_ ? allocate_(key.bytes) : default_storage(key.bytes); }
                    catch (...) { record(); throw; }
                    record();
                }
                if (!entry->storage || entry->storage.bytes != key.bytes) entry->storage = {};
                if (entry->storage && entry->storage.pinned) {
                    usage_->pinned_bytes.fetch_add(key.bytes, std::memory_order_relaxed); entry->pinned_reserved = true;
                }
                if (entry->storage && commit_ && !commit_(entry->charge, key.bytes)) entry->release_storage();
            }
            if (!entry->storage || (require_pinned && !entry->storage.pinned)) return {};
        } catch (const std::bad_alloc&) { return {}; }
        return {entry, entry->storage.data, key.bytes, entry->storage.pinned};
    }

    // A successful backend fence is a caller precondition. Publication has no
    // file provenance: D2H bytes must never inflate source-file read counters.
    bool publish_external(const ExternalWrite& write) {
        if (!write) return false;
        auto entry = std::static_pointer_cast<Entry>(write.owner);
        std::lock_guard<std::mutex> lock(mutex_);
        if (!entry->external || entry->retired || entry->status != Status::loading
            || !entry->storage || entry->usage != usage_ || entries_.count(entry->key)
            || usage_->bytes.load(std::memory_order_relaxed) > budget_) return false;
        try {
            metadata_allocation(3); recency_.push_front(entry->key);
            entry->recency = recency_.begin();
            try { entries_.emplace(entry->key, entry); }
            catch (...) { recency_.erase(entry->recency); throw; }
        } catch (const std::bad_alloc&) { return false; }
        entry->status = Status::ready;
        ++counters_.admitted; counters_.external_bytes += entry->key.bytes;
        return true;
    }
    bool publish_external_triple(const std::array<ExternalWrite, 3>& writes) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::array<std::shared_ptr<Entry>, 3> entries;
        for (int part = 0; part < 3; ++part) {
            if (!writes[part]) return false;
            auto entry = std::static_pointer_cast<Entry>(writes[part].owner);
            if (!entry->external || entry->retired || entry->status != Status::loading
                || !entry->storage || entry->usage != usage_ || entries_.count(entry->key)
                || usage_->bytes.load(std::memory_order_relaxed) > budget_) return false;
            for (int before = 0; before < part; ++before)
                if (entries[before]->key == entry->key) return false;
            entries[part] = std::move(entry);
        }
        try {
            for (int part = 0; part < 3; ++part) {
                const auto& entry = entries[part];
                metadata_allocation(3 + part); recency_.push_front(entry->key);
                entry->recency = recency_.begin();
                try { entries_.emplace(entry->key, entry); }
                catch (...) { recency_.erase(entry->recency); throw; }
            }
        } catch (const std::bad_alloc&) {
            for (const auto& entry : entries) {
                remove_locked(entry->key, entry.get()); entry->retired = true;
            }
            return false;
        }
        for (const auto& entry : entries) {
            entry->status = Status::ready;
            ++counters_.admitted; counters_.external_bytes += entry->key.bytes;
        }
        return true;
    }
#if defined(TSG_HOST_MOE_PAYLOAD_CACHE_TEST_HOOKS)
    static void test_fail_metadata_allocation(int stage) { metadata_failure_.store(stage); }
    unsigned test_frequency(std::uint64_t source, std::uint64_t offset, std::size_t bytes) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return frequency_estimate({source, offset, bytes});
    }
    // Hold the same immutable Entry lease a copying reader owns, so tests can
    // deterministically exercise eviction during a copy without timing memcpy.
    std::shared_ptr<const void> test_lease(std::uint64_t source, std::uint64_t offset, std::size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = entries_.find({source, offset, bytes});
        return found != entries_.end() && found->second->status == Status::ready
            ? found->second : std::shared_ptr<const void>{};
    }
#endif

private:
    // All admission paths share exact key validation and publication. A cache-
    // only request can never fall through to a discarded source read, and it
    // may replace only unleased speculative entries, preserving demand data.
    template<class Loader>
    Lease acquire_range_impl(std::uint64_t source, std::uint64_t base, std::size_t extent,
        std::size_t relative, void* destination, std::size_t count, bool admit, bool cache_only, Loader&& loader) {
        if (!source || !extent || base > (std::numeric_limits<std::uint64_t>::max)() - extent
            || relative > extent || count > extent - relative || (!cache_only && !destination && count))
            throw std::invalid_argument("Invalid expert payload cache range");
        if (!count) return {};
        const bool demand = admit && !cache_only;
        const Key key{source, base, extent};
        std::shared_ptr<Entry> entry;
        bool fill = false, reused = false;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (frequency_admission_ && demand) observe_frequency(key);
            const auto found = entries_.find(key);
            if (found != entries_.end()) {
                entry = found->second;
                if (demand) recency_.splice(recency_.begin(), recency_, entry->recency);
                if (entry->status == Status::loading) {
                    ++counters_.waits;
                    changed_.wait(lock, [&] { return entry->status != Status::loading; });
                }
                if (entry->status == Status::failed) std::rethrow_exception(entry->failure);
                if (entry->status == Status::ready) {
                    ++counters_.hits; counters_.hit_bytes += count;
                    if (!cache_only && entry->prefetched) counters_.prefetch_hit_bytes += count;
                    if (demand) entry->prefetched = false;
                    lock.unlock();
                    return lease(entry, relative, count, true);
                }
            } else {
                ++counters_.misses;
                if (admit && extent <= budget_) {
                    const auto used = usage_->bytes.load(std::memory_order_relaxed);
                    if (used <= budget_ && used > budget_ - extent) {
                        // A decode miss frequently replaces another equally
                        // sized expert. Reuse its allocation and charge
                        // only when no reader holds it. Never overwrite a lease.
                        auto selected = entries_.end();
                        unsigned score = (std::numeric_limits<unsigned>::max)(), sampled = 0;
                        for (auto position = recency_.rbegin(); position != recency_.rend(); ++position) {
                            if (frequency_admission_ && sampled++ == frequency_sample_) break;
                            const auto victim = entries_.find(*position);
                            if (victim->first.bytes != extent || victim->second->status != Status::ready
                                || victim->second.use_count() != 1
                                || (cache_only && !victim->second->prefetched)) continue;
                            const auto estimate = frequency_admission_ && !cache_only ? frequency_estimate(victim->first) : 0;
                            if (estimate < score) { selected = victim; score = estimate; }
                            if (!frequency_admission_ || cache_only) break;
                        }
                        if (selected != entries_.end()
                            && (cache_only || !frequency_admission_ || frequency_estimate(key) >= score)) {
                            entry = std::move(selected->second);
                            recency_.erase(entry->recency); entries_.erase(selected);
                            entry->key = key; entry->status = Status::loading;
                            ++counters_.evictions; ++counters_.reuses;
                            reused = true;
                        } else if (frequency_admission_ && !cache_only) ++counters_.frequency_rejections;
                    }
                    // Speculation may reclaim only its own unleased payloads.
                    // Demand still honors the optional frequency gate and can
                    // never circumvent it by trimming unrelated sizes.
                    if (!reused && (cache_only || !frequency_admission_)) trim_locked(budget_ - extent, cache_only);
                    if (reused || usage_->bytes.load(std::memory_order_relaxed) <= budget_ - extent) {
                        try {
                            if (!reused) {
                                metadata_allocation(1);
                                entry = std::make_shared<Entry>(key, usage_);
                            }
                            entry->prefetched = cache_only;
                            metadata_allocation(2);
                            if (cache_only) {
                                recency_.push_back(key);
                                entry->recency = std::prev(recency_.end());
                            } else {
                                recency_.push_front(key);
                                entry->recency = recency_.begin();
                            }
                            try { metadata_allocation(3); entries_.emplace(key, entry); }
                            catch (...) { recency_.erase(entry->recency); throw; }
                            if (!reused) {
                                usage_->bytes.fetch_add(extent, std::memory_order_relaxed);
                                entry->reserved = true;
                            }
                            fill = true;
                        } catch (const std::bad_alloc&) {
                            // Metadata is optional too. No map entry survives a
                            // failed insertion; any new LRU node was rolled back.
                            // Release a reused victim outside this mutex below.
                            reused = false;
                        }
                    }
                }
            }
        }
        if (!fill) entry.reset();
        if (fill) {
            try {
                if (!reused) {
                    if (reserve_) entry->charge = reserve_(extent);
                    if (!reserve_ || entry->charge) entry->storage = allocate_ ? allocate_(extent) : default_storage(extent);
                    if (!entry->storage || entry->storage.bytes != extent) entry->storage = {};
                    if (entry->storage && entry->storage.pinned) {
                        usage_->pinned_bytes.fetch_add(extent, std::memory_order_relaxed);
                        entry->pinned_reserved = true;
                    }
                    if (entry->storage && commit_ && !commit_(entry->charge, extent)) entry->release_storage();
                }
            } catch (const std::bad_alloc&) {
                // This optional cache may decline without rejecting inference.
                // Commit itself may allocate/throw after data exists: never
                // publish that uncommitted allocation as a successful fill.
                entry->release_storage();
            } catch (...) {
                entry->release_storage();
                finish(entry, Status::failed, std::current_exception());
                throw;
            }
            if (!entry->storage) {
                entry->release_storage();
                finish(entry, Status::bypass);
            } else {
                try { loader(base, entry->storage.data, extent); }
                catch (...) {
                    entry->release_storage();
                    finish(entry, Status::failed, std::current_exception());
                    throw;
                }
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    counters_.source_bytes += extent;
                }
                finish(entry, Status::ready);
                return lease(entry, relative, count, false);
            }
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++counters_.bypasses;
        }
        if (cache_only) return {};
        loader(base + relative, destination, count);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            counters_.source_bytes += count;
            counters_.bypass_bytes += count;
        }
        return {};
    }
public:
    // Best-effort whole-extent prefetch. A refusal allocates no fallback buffer
    // and performs no source read. Existing demand data is never displaced or
    // aged by this path; a demand hit promotes speculative data when needed.
    template<class Loader>
    Lease acquire_cached(std::uint64_t source, std::uint64_t base, std::size_t extent, Loader&& loader) {
        return acquire_range_impl(source, base, extent, 0, nullptr, extent, true, true, std::forward<Loader>(loader));
    }

    // The key covers an entire expert projection; arbitrary staging windows
    // read slices of that same extent, so changing ring boundaries cannot
    // fragment the cache. loader(offset, destination, bytes) reads source bytes.
    // admit=false probes existing data but neither fills nor ages the LRU: a
    // wide prefill scan must not evict the hot decode working set. A successful
    // acquisition leaves destination untouched and returns an immutable lease.
    // On a declined admission only the requested slice is read into destination.
    // Retain the lease through asynchronous upload completion, including failure.
    template<class Loader>
    Lease acquire_range(std::uint64_t source, std::uint64_t base, std::size_t extent,
        std::size_t relative, void* destination, std::size_t count, bool admit, Loader&& loader) {
        return acquire_range_impl(source, base, extent, relative, destination, count, admit, false,
            std::forward<Loader>(loader));
    }
    template<class Loader>
    bool read_range(std::uint64_t source, std::uint64_t base, std::size_t extent,
        std::size_t relative, void* destination, std::size_t count, bool admit, Loader&& loader) {
        const auto value = acquire_range(source, base, extent, relative, destination, count, admit,
            std::forward<Loader>(loader));
        if (value) std::memcpy(destination, value.data, value.bytes);
        return value.cache_hit;
    }
    template<class Loader>
    bool read(std::uint64_t source, std::uint64_t offset, void* destination,
        std::size_t bytes, bool admit, Loader&& loader) {
        return read_range(source, offset, bytes, 0, destination, bytes, admit, std::forward<Loader>(loader));
    }
};
} // namespace tsg
