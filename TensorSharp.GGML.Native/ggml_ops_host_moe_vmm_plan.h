// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>

namespace tsg::host_moe_vmm {
// Inputs come from the actual tensor's get_alloc_size, never stride alone.
// Each pool has its own VA reservation and driver allocation granularity.
struct Projection {
    std::size_t stride = 0, allocation = 0, granule = 0, chunk = 0;
    std::size_t maximum_chunk = 0, maximum_buffer = 0, reserved_va = 0;
};

class Layout {
    std::array<Projection, 3> spec_{};
    std::array<std::size_t, 3> physical_{};
    std::size_t capacity_ = 0, total_ = 0;
    bool configured_ = false;
public:
    static constexpr std::size_t limit() noexcept {
        return (std::numeric_limits<std::size_t>::max)() < UINT64_C(0x7fffffffffffffff)
            ? (std::numeric_limits<std::size_t>::max)() : UINT64_C(0x7fffffffffffffff);
    }
    static std::size_t add(std::size_t a, std::size_t b) {
        if (a > limit() || b > limit() - a) throw std::overflow_error("VMM byte sum overflow");
        return a + b;
    }
    static std::size_t multiply(std::size_t a, std::size_t b) {
        if (a > limit() || b > limit() || (a && b > limit() / a))
            throw std::overflow_error("VMM byte product overflow");
        return a * b;
    }
    static std::size_t round(std::size_t bytes, std::size_t granule) {
        if (!granule || granule > limit()) throw std::invalid_argument("Invalid VMM granule");
        const auto remainder = bytes % granule;
        return remainder ? add(bytes, granule - remainder) : add(bytes, 0);
    }
    static Layout make(std::size_t capacity, const std::array<Projection, 3>& specs) {
        Layout result; result.capacity_ = capacity; result.spec_ = specs; result.configured_ = true;
        for (unsigned p = 0; p != 3; ++p) {
            const auto& s = specs[p];
            const auto logical = multiply(s.stride, capacity);
            if (!s.stride || !s.granule || !s.chunk || s.chunk % s.granule
                || s.chunk > s.maximum_chunk || s.maximum_chunk > limit()
                || !s.maximum_buffer || s.maximum_buffer > limit()
                || !s.reserved_va || s.reserved_va > limit() || s.reserved_va % s.granule
                || s.allocation < logical || (!capacity && s.allocation))
                throw std::invalid_argument("Invalid independent VMM projection geometry");
            result.physical_[p] = round(s.allocation, s.chunk);
            if (result.physical_[p] > s.maximum_buffer || result.physical_[p] > s.reserved_va)
                throw std::invalid_argument("VMM projection exceeds buffer or VA limit");
            result.total_ = add(result.total_, result.physical_[p]);
        }
        return result;
    }
    Layout empty() const {
        auto specs = spec_; for (auto& s : specs) s.allocation = 0;
        return configured_ ? make(0, specs) : Layout{};
    }
    bool configured() const noexcept { return configured_; }
    std::size_t capacity() const noexcept { return capacity_; }
    std::size_t total() const noexcept { return total_; }
    const Projection& projection(unsigned p) const { return spec_.at(p); }
    std::size_t physical(unsigned p) const { return physical_.at(p); }
    std::size_t chunks(unsigned p) const { return configured_ ? physical(p) / projection(p).chunk : 0; }
    bool compatible(const Layout& other) const noexcept {
        if (!configured_ || !other.configured_) return true;
        for (unsigned p = 0; p != 3; ++p) {
            const auto& a = spec_[p]; const auto& b = other.spec_[p];
            if (a.stride != b.stride || a.granule != b.granule || a.chunk != b.chunk
                || a.maximum_chunk != b.maximum_chunk || a.maximum_buffer != b.maximum_buffer
                || a.reserved_va != b.reserved_va) return false;
        }
        return true;
    }
    // Immutable additive delta: no future shrink/refund or cumulative grant.
    std::size_t growth_delta(const Layout& target) const {
        if (!target.configured_ || !compatible(target) || target.capacity_ < capacity_)
            throw std::invalid_argument("Incompatible VMM growth geometry");
        std::size_t bytes = 0;
        for (unsigned p = 0; p != 3; ++p) {
            if (target.physical(p) < physical(p)
                || (configured_ && target.projection(p).allocation < projection(p).allocation))
                throw std::invalid_argument("VMM growth cannot discard an allocation prefix");
            bytes = add(bytes, target.physical(p) - physical(p));
        }
        return bytes;
    }
};

class EmptyTail {
    std::size_t before_ = 0, after_ = 0;
    std::uint64_t generation_ = 0;
    EmptyTail(std::size_t a, std::size_t b, std::uint64_t g) : before_(a), after_(b), generation_(g) {}
public:
    // Caller supplies CacheEntry's COMPLETE expert_for_slot vector under its
    // mutation gate: -1 is empty and any nonnegative value is occupied.
    // This proves empty-tail shrink only; occupied tails require a different
    // compaction plan and cannot be made safe by discarding tags here.
    static std::optional<EmptyTail> prove(std::size_t before, std::size_t after,
        std::uint64_t generation, const int* expert_for_slot, std::size_t count) noexcept {
        if (!generation || !after || after >= before || count != before || !expert_for_slot) return {};
        for (std::size_t i = 0; i != count; ++i)
            if (expert_for_slot[i] < -1 || (i >= after && expert_for_slot[i] >= 0)) return {};
        return EmptyTail(before, after, generation);
    }
    bool matches(std::size_t before, std::size_t after, std::uint64_t generation) const noexcept {
        return before_ == before && after_ == after && generation_ == generation;
    }
};

// Later CUDA adapter contract: own three disjoint VA reservations, validated
// actual get_alloc_size inputs, and every shared charge behind its opaque ID.
// False/zero means the requested transition did not occur; implementations may
// not throw or hide a resource created before reporting failure. Metadata/VA
// owners and adapter ticket records need their own bounded charge and outlive
// this state. IDs are nonzero and unique while owned. drain covers every queue
// that could touch these mappings. No generic GGML copy fallback or graph
// publication belongs here; usable() describes resource geometry, not model
// error health or permission to reuse a stale graph/capacity/slot generation.
// Freeze slot/graph mutations through any retry and rebuild every capacity-
// dependent graph/descriptor with fresh UIDs before exposing changed geometry.
struct Adapter {
    using Id = std::uint64_t;
    virtual ~Adapter() = default;
    virtual Id reserve(std::size_t bytes) noexcept = 0;
    virtual bool commit(Id ticket) noexcept = 0;
    virtual bool refund(Id ticket) noexcept = 0;
    virtual Id create(unsigned pool, std::size_t bytes, Id ticket) noexcept = 0;
    virtual bool map(unsigned pool, std::size_t offset, Id resource, std::size_t bytes) noexcept = 0;
    virtual bool access(unsigned pool, std::size_t offset, std::size_t bytes) noexcept = 0;
    virtual bool drain() noexcept = 0;
    virtual bool unmap(unsigned pool, std::size_t offset, std::size_t bytes) noexcept = 0;
    virtual bool release(Id resource) noexcept = 0;
};

enum class Result { refused, unchanged, published, quarantined };
struct Usage {
    // pending/committed are HELD TICKET bytes, not physical observations.
    // Physical resource bytes are mapped + created_unmapped. A failed refund
    // may leave ticket_only bytes committed after the physical resource freed.
    std::size_t pending = 0, committed = 0, mapped = 0, created_unmapped = 0;
    std::size_t ticket_only = 0, quarantined = 0, records = 0;
};

template<std::size_t MaxChunks> class Owner {
    static_assert(MaxChunks > 0, "VMM owner needs a finite record bound");
    static_assert(MaxChunks <= (std::numeric_limits<std::size_t>::max)() / 3,
        "VMM owner record bound overflow");
    enum class Stage : unsigned char { empty, ticket, created, mapped, accessible };
    enum class Work : unsigned char { none, grow, shrink, close };
    struct Chunk { Adapter::Id resource = 0, ticket = 0; Stage stage = Stage::empty; bool committed = false; };
    std::array<std::array<Chunk, MaxChunks>, 3> chunks_{};
    Adapter* const adapter_;
    Layout current_, target_;
    Work work_ = Work::none;
    bool quarantined_ = false;

    bool bounded(const Layout& target) const {
        for (unsigned p = 0; p != 3; ++p) if (target.chunks(p) > MaxChunks) return false;
        return true;
    }
    std::size_t first(unsigned p) const { return work_ == Work::grow ? current_.chunks(p) : target_.chunks(p); }
    std::size_t last(unsigned p) const { return work_ == Work::grow ? target_.chunks(p) : current_.chunks(p); }
    const Layout& resources() const { return work_ == Work::grow ? target_ : current_; }
    bool drop(Adapter& adapter, unsigned p, std::size_t i) noexcept {
        auto& c = chunks_[p][i]; const auto bytes = resources().projection(p).chunk;
        if (c.stage == Stage::mapped || c.stage == Stage::accessible) {
            if (!adapter.unmap(p, i * bytes, bytes)) return false;
            c.stage = Stage::created;
        }
        if (c.stage == Stage::created) {
            if (!adapter.release(c.resource)) return false;
            c.resource = 0; c.stage = Stage::ticket;
        }
        if (c.stage == Stage::ticket) {
            if (!adapter.refund(c.ticket)) return false;
            c = {};
        }
        return true;
    }
    Result finish_cleanup(Adapter& adapter, bool drained = false) noexcept {
        bool physical = false;
        for (unsigned p = 0; p != 3; ++p) for (auto i = first(p); i != last(p); ++i)
            physical = physical || chunks_[p][i].resource != 0;
        if (physical && !drained && !adapter.drain()) { quarantined_ = true; return Result::quarantined; }
        for (unsigned n = 3; n != 0;) {
            const auto p = --n;
            for (auto i = last(p); i > first(p);) {
                if (!drop(adapter, p, --i)) { quarantined_ = true; return Result::quarantined; }
            }
        }
        const auto result = work_ == Work::grow ? Result::unchanged : Result::published;
        if (work_ != Work::grow) current_ = target_;
        target_ = {}; work_ = Work::none; quarantined_ = false;
        return result;
    }
public:
    // The same adapter and its immutable VA reservations outlive this owner,
    // including failed cleanup. Never reinterpret IDs in another adapter.
    explicit Owner(Adapter& adapter) noexcept : adapter_(&adapter) {}
    Owner(const Owner&) = delete; Owner& operator=(const Owner&) = delete;
    // No destructor refund: resource/charge lifetime cannot be guessed. Caller
    // must obtain checked close() and retain this bounded owner on failure.
    static constexpr std::size_t record_capacity() noexcept { return 3 * MaxChunks; }
    static constexpr std::size_t metadata_bytes() noexcept { return sizeof(Owner); }
    const Layout& current() const noexcept { return current_; }
    bool usable() const noexcept { return work_ == Work::none && current_.capacity() != 0; }
    bool clean() const noexcept { return work_ == Work::none && current_.capacity() == 0; }
    Usage usage() const noexcept {
        Usage u;
        for (unsigned p = 0; p != 3; ++p) for (std::size_t i = 0; i != MaxChunks; ++i) {
            const auto& c = chunks_[p][i]; if (c.stage == Stage::empty) continue;
            const auto bytes = (current_.configured() ? current_ : target_).projection(p).chunk;
            ++u.records; (c.committed ? u.committed : u.pending) += bytes;
            if (c.stage == Stage::ticket) u.ticket_only += bytes;
            else if (c.stage == Stage::created) u.created_unmapped += bytes;
            else u.mapped += bytes;
            if (quarantined_ && i >= first(p) && i < last(p)) u.quarantined += bytes;
        }
        return u;
    }
    Result grow(const Layout& target, Adapter& adapter) {
        if (&adapter != adapter_ || work_ != Work::none || !bounded(target)) return Result::refused;
        current_.growth_delta(target); // Validate before any adapter action.
        if (target.capacity() == current_.capacity()) {
            for (unsigned p = 0; p != 3; ++p)
                if (target.projection(p).allocation != current_.projection(p).allocation)
                    throw std::invalid_argument("Same-capacity VMM geometry changed allocation size");
            return Result::unchanged;
        }
        target_ = target; work_ = Work::grow;
        // All additive, immutable physical tickets precede the first create.
        for (unsigned p = 0; p != 3; ++p) for (auto i = first(p); i != last(p); ++i) {
            auto& c = chunks_[p][i]; c.ticket = adapter.reserve(target_.projection(p).chunk);
            if (!c.ticket) return finish_cleanup(adapter);
            c.stage = Stage::ticket;
        }
        for (unsigned p = 0; p != 3; ++p) for (auto i = first(p); i != last(p); ++i) {
            auto& c = chunks_[p][i]; const auto bytes = target_.projection(p).chunk;
            c.resource = adapter.create(p, bytes, c.ticket);
            if (!c.resource) return finish_cleanup(adapter);
            c.stage = Stage::created;
            if (!adapter.map(p, i * bytes, c.resource, bytes)) return finish_cleanup(adapter);
            c.stage = Stage::mapped;
            if (!adapter.access(p, i * bytes, bytes)) return finish_cleanup(adapter);
            c.stage = Stage::accessible;
        }
        if (!adapter.drain()) { quarantined_ = true; return Result::quarantined; }
        for (unsigned p = 0; p != 3; ++p) for (auto i = first(p); i != last(p); ++i) {
            auto& c = chunks_[p][i];
            if (!adapter.commit(c.ticket)) return finish_cleanup(adapter, true);
            c.committed = true;
        }
        current_ = target_; target_ = {}; work_ = Work::none;
        return Result::published;
    }
    Result shrink(const Layout& target, const EmptyTail& proof, std::uint64_t generation, Adapter& adapter) {
        if (&adapter != adapter_ || work_ != Work::none || !target.configured() || !current_.compatible(target)
            || !proof.matches(current_.capacity(), target.capacity(), generation)) return Result::refused;
        for (unsigned p = 0; p != 3; ++p)
            if (target.physical(p) > current_.physical(p)
                || target.projection(p).allocation > current_.projection(p).allocation) return Result::refused;
        // Caller first invalidates/drains all graph variants under its gate.
        // No fallback after a tail is physically removed; partial shrink is
        // unusable until retry succeeds, and every unreleased ticket is kept.
        target_ = target; work_ = Work::shrink;
        if (!adapter.drain()) { quarantined_ = true; return Result::quarantined; }
        return finish_cleanup(adapter, true);
    }
    Result retry(Adapter& adapter) noexcept {
        if (&adapter != adapter_) return Result::refused;
        return work_ == Work::none ? Result::unchanged : finish_cleanup(adapter);
    }
    bool close(Adapter& adapter) noexcept {
        if (&adapter != adapter_) return false;
        if (work_ != Work::none && retry(adapter) == Result::quarantined) return false;
        if (!current_.configured()) return true;
        target_ = current_.empty(); work_ = Work::close;
        return finish_cleanup(adapter) != Result::quarantined;
    }
};
} // namespace tsg::host_moe_vmm
