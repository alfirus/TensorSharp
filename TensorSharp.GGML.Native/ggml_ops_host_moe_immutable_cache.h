// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace tsg {
// A separate source-keyed immutable cache. No writable loader, buffer reuse,
// external write, rekey, recycle, or direct-upload contract exists here.
// Join callers before destroying the cache. Leases and backend quarantines may
// outlive it and keep all Usage/external charges alive until physical cleanup.
class HostMoeImmutableCache {
public:
    static constexpr std::size_t entry_metadata_bytes = 4096;
    static constexpr std::size_t maximum_leases_per_entry = 16;
    struct Range { std::uint64_t source = 0, offset = 0; std::size_t bytes = 0; };
    struct Storage {
        std::shared_ptr<void> owner;
        const std::uint8_t* data = nullptr;
        std::size_t bytes = 0;
        explicit operator bool() const noexcept { return owner && data && bytes; }
    };
    struct Lease {
        std::shared_ptr<const void> owner;
        const std::uint8_t* data = nullptr;
        std::size_t bytes = 0;
        bool cache_hit = false;
        explicit operator bool() const noexcept { return owner && data && bytes; }
    };
    enum class ReadyStatus { not_ready, ready, lease_refused };
    struct ReadyLookup {
        ReadyStatus status = ReadyStatus::not_ready;
        Lease lease;
    };
    using ReserveCharge = std::function<std::shared_ptr<void>(std::size_t)>;
    using CommitCharge = std::function<bool(const std::shared_ptr<void>&, std::size_t)>;
    // The storage owner MUST retain lifetime through final physical/quota
    // cleanup, including cleanup quarantine after a failed/throwing factory.
    // A failed owner with null data may be returned to force ordinary fallback.
    using Factory = std::function<Storage(const Range&, const std::shared_ptr<void>& lifetime)>;
    struct Stats {
        std::size_t budget_bytes = 0, reserved_bytes = 0, resident_logical_bytes = 0,
            entries = 0, owned_entries = 0, pending_admissions = 0, active_leases = 0;
        std::uint64_t hits = 0, misses = 0, hit_bytes = 0, admitted = 0, bypasses = 0,
            evictions = 0, failures = 0, waits = 0, admitted_logical_bytes = 0, bypass_logical_bytes = 0;
    };
private:
    struct Key {
        Range range;
        bool operator<(const Key& other) const noexcept {
            if (range.source != other.range.source) return range.source < other.range.source;
            if (range.offset != other.range.offset) return range.offset < other.range.offset;
            return range.bytes < other.range.bytes;
        }
    };
    struct Usage {
        std::atomic<std::size_t> bytes{0}, owners{0}, pending{0}, leases{0};
    };
    struct Lifetime {
        std::shared_ptr<Usage> usage;
        std::shared_ptr<void> charge;
        std::size_t bytes;
        bool active = false;
        Lifetime(std::shared_ptr<Usage> u, std::size_t n) : usage(std::move(u)), bytes(n) {}
        ~Lifetime() {
            // The backend's last physical/quota owner must release its token
            // only after cleanup. No cache mutex is held at this point.
            charge.reset();
            if (active) { usage->bytes.fetch_sub(bytes); usage->owners.fetch_sub(1); }
        }
    };
    enum class Status { loading, ready, retired };
    struct Entry;
    using Recency = std::list<std::shared_ptr<Entry>>;
    struct Entry {
        Key key;
        std::size_t charge_bytes;
        // Reverse member destruction frees storage before releasing our token.
        std::shared_ptr<Lifetime> lifetime;
        Storage storage;
        Status status = Status::loading;
        std::atomic<std::size_t> readers{0};
        std::condition_variable changed;
        Recency::iterator recency;
        Entry(Key k, std::size_t n) : key(k), charge_bytes(n) {}
    };
    struct LeaseRecord {
        std::shared_ptr<Entry> entry;
        std::shared_ptr<Usage> usage;
        LeaseRecord(std::shared_ptr<Entry> e, std::shared_ptr<Usage> u) : entry(std::move(e)), usage(std::move(u)) {
            entry->readers.fetch_add(1); usage->leases.fetch_add(1);
        }
        ~LeaseRecord() { entry->readers.fetch_sub(1); usage->leases.fetch_sub(1); }
    };
    // map/list nodes have no retained bucket arrays. At most sixteen acquired
    // records per Entry fit the conservative metadata component; Lease copies
    // share one record and do not consume another active-lease count.
    static_assert(sizeof(Entry) + sizeof(Lifetime) + 16 * sizeof(LeaseRecord) + 1024 < entry_metadata_bytes,
        "Immutable entry metadata allowance is insufficient");
    using Entries = std::map<Key, std::shared_ptr<Entry>>;
    mutable std::mutex mutex_;
    std::size_t budget_;
    std::shared_ptr<Usage> usage_ = std::make_shared<Usage>();
    ReserveCharge reserve_;
    CommitCharge commit_;
    Entries entries_;
    Recency recency_;
    Stats counters_;
#ifdef TSG_HOST_MOE_IMMUTABLE_CACHE_TEST_HOOKS
    int metadata_failure_ = 0;
    void metadata_point(int stage) {
        if (metadata_failure_ == stage) { metadata_failure_ = 0; throw std::bad_alloc(); }
    }
#else
    void metadata_point(int) noexcept {}
#endif
    static void add(std::uint64_t& value, std::size_t amount = 1) noexcept {
        const auto maximum = (std::numeric_limits<std::uint64_t>::max)();
        value = amount > maximum - value ? maximum : value + amount;
    }
    static void validate_range(std::uint64_t source, std::uint64_t base, std::size_t extent,
        std::size_t relative, std::size_t count) {
        if (!source || !extent || !count || relative > extent || count > extent-relative
            || extent > UINT64_MAX-base)
            throw std::invalid_argument("Invalid immutable source/slice extent");
    }
    void bypass_locked(std::size_t count, bool failure = false) noexcept {
        add(counters_.bypasses); add(counters_.bypass_logical_bytes, count);
        if (failure) add(counters_.failures);
    }
    // Splice first so removing the map cannot destroy physical owners while
    // locked. Every caller destroys this retired list after releasing mutex_.
    void retire_locked(Entries::iterator found, Recency& retired, bool eviction) noexcept {
        const auto& entry = found->second;
        entry->status = Status::retired;
        retired.splice(retired.end(), recency_, entry->recency);
        entry->changed.notify_all();
        entries_.erase(found);
        if (eviction) add(counters_.evictions);
    }
    Lease lease_locked(const std::shared_ptr<Entry>& entry, std::size_t relative,
        std::size_t count, bool hit) {
        if (entry->readers.load() >= maximum_leases_per_entry) return {};
        metadata_point(5);
        auto record = std::make_shared<LeaseRecord>(entry, usage_);
        return {std::move(record), entry->storage.data + relative, count, hit};
    }
    bool linked_loading_locked(const std::shared_ptr<Entry>& entry) const {
        const auto found = entries_.find(entry->key);
        return entry->status == Status::loading && found != entries_.end() && found->second == entry;
    }
public:
    explicit HostMoeImmutableCache(std::size_t budget = 0, ReserveCharge reserve = {}, CommitCharge commit = {})
        : budget_(budget), reserve_(std::move(reserve)), commit_(std::move(commit)) {
        if (commit_ && !reserve_) throw std::invalid_argument("Immutable commit requires a reservation callback");
    }
    HostMoeImmutableCache(const HostMoeImmutableCache&) = delete;
    HostMoeImmutableCache& operator=(const HostMoeImmutableCache&) = delete;
    ~HostMoeImmutableCache() { clear(); }

    void clear() {
        Recency retired;
        { std::lock_guard<std::mutex> lock(mutex_);
            while (!entries_.empty()) retire_locked(entries_.begin(), retired, false);
        }
    }
    void erase_source(std::uint64_t source) {
        Recency retired;
        { std::lock_guard<std::mutex> lock(mutex_);
            for (auto it = entries_.begin(); it != entries_.end();) {
                auto selected = it++;
                if (selected->first.range.source == source) retire_locked(selected, retired, false);
            }
        }
    }
    void invalidate(const Range& range) {
        Recency retired;
        { std::lock_guard<std::mutex> lock(mutex_);
            const auto found = entries_.find(Key{range});
            if (found != entries_.end()) retire_locked(found, retired, false);
        }
    }
    void set_budget(std::size_t bytes) {
        Recency retired;
        { std::lock_guard<std::mutex> lock(mutex_);
            budget_ = bytes;
            // Even equal/growing updates retire old loading generations.
            for (auto it = entries_.begin(); it != entries_.end();) {
                auto selected = it++;
                if (selected->second->status == Status::loading) retire_locked(selected, retired, false);
            }
        }
        retired.clear();
        for (;;) {
            { std::lock_guard<std::mutex> lock(mutex_);
                if (usage_->bytes.load() <= budget_ || recency_.empty()) return;
                // Explicit pressure may retire leased entries, whose real
                // charge remains live. Admissions may evict only unleased ones.
                auto victim = entries_.find(recency_.back()->key);
                retire_locked(victim, retired, true);
            }
            retired.clear(); // Observe real final cleanup before more trim.
        }
    }
    Stats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto result = counters_;
        result.budget_bytes = budget_; result.reserved_bytes = usage_->bytes.load();
        result.owned_entries = usage_->owners.load(); result.pending_admissions = usage_->pending.load();
        result.active_leases = usage_->leases.load();
        for (const auto& item : entries_) if (item.second->status == Status::ready) {
            result.resident_logical_bytes += item.first.range.bytes; ++result.entries;
        }
        return result;
    }
    // Lookup is not a successful guarded CPU copy. Only the caller, after its
    // exact hit slice was copied successfully, may record those bytes here.
    void record_copy(std::size_t count) {
        std::lock_guard<std::mutex> lock(mutex_); add(counters_.hit_bytes, count);
    }

    // Logical lookup only: no physical charge, factory, callbacks or waiting.
    // Call with touch=false for prefill's hit-only/no-aging path. A not-ready
    // admission probe uses count_miss=false; acquire_range then accounts its
    // attempt, or record_miss_bypass accounts a declined physical plan once.
    // A ready-but-refused lease already counts hit+bypass and MUST NOT retry
    // admission; neither lease metadata exhaustion nor allocation failure is a
    // source miss. Returned leases retain precisely the same ownership rules.
    ReadyLookup find_ready_range(std::uint64_t source, std::uint64_t base, std::size_t extent,
        std::size_t relative, std::size_t count, bool touch, bool count_miss) {
        validate_range(source, base, extent, relative, count);
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = entries_.find(Key{{source, base, extent}});
        if (found == entries_.end() || found->second->status != Status::ready) {
            if (count_miss) { add(counters_.misses); bypass_locked(count); }
            return {};
        }
        const auto& entry = found->second;
        add(counters_.hits);
        try {
            auto lease = lease_locked(entry, relative, count, true);
            if (!lease) { bypass_locked(count); return {ReadyStatus::lease_refused, {}}; }
            if (touch) recency_.splice(recency_.begin(), recency_, entry->recency);
            return {ReadyStatus::ready, std::move(lease)};
        } catch (...) { bypass_locked(count, true); return {ReadyStatus::lease_refused, {}}; }
    }

    // The caller suppressed lookup accounting while considering admission,
    // then declined its geometry. Do not call after count_miss=true or after
    // acquire_range, which each already account the complete request attempt.
    void record_miss_bypass(std::size_t count) {
        std::lock_guard<std::mutex> lock(mutex_); add(counters_.misses); bypass_locked(count);
    }

    Lease acquire_range(std::uint64_t source, std::uint64_t base, std::size_t extent,
        std::size_t charge_bytes, std::size_t relative, std::size_t count, bool admit, const Factory& factory) {
        validate_range(source, base, extent, relative, count);
        if (extent > (std::numeric_limits<std::size_t>::max)()-entry_metadata_bytes
            || charge_bytes < extent+entry_metadata_bytes)
            throw std::invalid_argument("Invalid immutable source/slice/physical charge extent");
        const Key key{{source, base, extent}};
        std::shared_ptr<Entry> entry;
        Recency retired;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                auto found = entries_.find(key);
                if (found != entries_.end()) {
                    entry = found->second;
                    if (entry->status == Status::loading) {
                        if (!admit) { add(counters_.misses); bypass_locked(count); return {}; }
                        add(counters_.waits);
                        entry->changed.wait(lock, [&] { return entry->status != Status::loading; });
                    }
                    if (entry->status == Status::ready) {
                        add(counters_.hits);
                        try {
                            auto result = lease_locked(entry, relative, count, true);
                            if (!result) { bypass_locked(count); return {}; }
                            if (admit) recency_.splice(recency_.begin(), recency_, entry->recency);
                            return result;
                        } catch (...) { bypass_locked(count, true); return {}; }
                    }
                    add(counters_.misses); bypass_locked(count); return {};
                }
                if (!admit || !factory || charge_bytes > budget_) {
                    add(counters_.misses); bypass_locked(count); return {};
                }
                if (usage_->bytes.load() > budget_-charge_bytes) {
                    auto victim = recency_.end();
                    for (auto it = recency_.end(); it != recency_.begin();) {
                        --it;
                        if ((*it)->status == Status::ready && (*it)->readers.load() == 0) { victim = it; break; }
                    }
                    if (victim == recency_.end()) { add(counters_.misses); bypass_locked(count); return {}; }
                    retire_locked(entries_.find((*victim)->key), retired, true);
                } else {
                    // All metadata is installed before external reservation or
                    // physical factory work; failure cannot leave a loading key.
                    try {
                        metadata_point(1); entry = std::make_shared<Entry>(key, charge_bytes);
                        metadata_point(2); entry->lifetime = std::make_shared<Lifetime>(usage_, charge_bytes);
                        metadata_point(3); recency_.push_front(entry); entry->recency = recency_.begin();
                        try { metadata_point(4); entries_.emplace(key, entry); }
                        catch (...) { recency_.pop_front(); throw; }
                        entry->lifetime->active = true;
                        usage_->bytes.fetch_add(charge_bytes); usage_->owners.fetch_add(1); usage_->pending.fetch_add(1);
                        add(counters_.misses);
                        break;
                    } catch (...) { add(counters_.misses); bypass_locked(count, true); return {}; }
                }
            }
            retired.clear(); // Only actual free/token release may fund a new owner.
        }
        struct Pending {
            std::shared_ptr<Usage> usage;
            ~Pending() { usage->pending.fetch_sub(1); }
        } pending{usage_};
        Storage storage;
        bool valid = false, failed = false;
        try {
            if (reserve_) entry->lifetime->charge = reserve_(charge_bytes);
            if (!reserve_ || entry->lifetime->charge) {
                storage = factory(key.range, std::static_pointer_cast<void>(entry->lifetime));
                valid = storage && storage.bytes == extent
                    && extent <= (std::numeric_limits<std::uintptr_t>::max)()-reinterpret_cast<std::uintptr_t>(storage.data);
                if (valid && commit_) valid = commit_(entry->lifetime->charge, charge_bytes);
                failed = !valid;
            }
        } catch (...) { valid = false; failed = true; }
        Lease result;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (valid && linked_loading_locked(entry) && charge_bytes <= budget_ && usage_->bytes.load() <= budget_) {
                entry->storage = std::move(storage);
                try {
                    result = lease_locked(entry, relative, count, false);
                    if (result) {
                        entry->status = Status::ready;
                        recency_.splice(recency_.begin(), recency_, entry->recency);
                        add(counters_.admitted); add(counters_.admitted_logical_bytes, extent);
                        entry->changed.notify_all();
                    } else failed = true;
                } catch (...) { failed = true; }
            }
            if (!result) {
                if (linked_loading_locked(entry)) retire_locked(entries_.find(key), retired, false);
                bypass_locked(count, failed);
            }
        }
        // storage, retired Entry, and its token destruct only after unlock.
        // Backend quarantine can retain the token beyond this return.
        return result;
    }
#ifdef TSG_HOST_MOE_IMMUTABLE_CACHE_TEST_HOOKS
    void test_fail_metadata(int stage) { std::lock_guard<std::mutex> lock(mutex_); metadata_failure_ = stage; }
#endif
};
} // namespace tsg
