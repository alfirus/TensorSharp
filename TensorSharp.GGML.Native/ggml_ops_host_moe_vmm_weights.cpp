// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_host_moe_vmm_weights.h"
#include "ggml_ops_host_moe_vmm_plan.h"
#include "ggml_ops_shared_cache_budget.h"
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

namespace tsg::host_moe_vmm {
namespace {
// The additive checked budget ABI is deliberately a prerequisite. The legacy
// void Release callback cannot certify managed release by resetting a ptr.
template<class T, class = void> struct Charges {
    static constexpr bool available = false;
    static bool attached() noexcept { return false; }
    static bool verify(const T&, int, int, std::size_t, bool) noexcept { return false; }
    static bool release(T&) noexcept { return false; }
};
template<class T> struct Charges<T, std::void_t<decltype(std::declval<const T&>().rank()),
    decltype(std::declval<const T&>().kind()), decltype(std::declval<const T&>().reserved_bytes()),
    decltype(std::declval<const T&>().is_committed()), decltype(T::checked_release_attached()),
    decltype(std::declval<T&>().release_checked())>> {
    static constexpr bool available = true;
    static bool attached() noexcept { try { return T::checked_release_attached(); } catch (...) { return false; } }
    static bool verify(const T& c, int rank, int kind, std::size_t bytes, bool committed) noexcept {
        try { return c.is_accounted() && c.rank() == rank && c.kind() == kind
            && c.reserved_bytes() == bytes && c.is_committed() == committed; } catch (...) { return false; }
    }
    // False retains the native ticket and managed token owner for acknowledgment
    // retry; ledger release might already have happened. Never re-reserve or
    // refund twice. True confirms token retirement; destruction is then a no-op.
    static bool release(T& c) noexcept { try { return c.release_checked(); } catch (...) { return false; } }
};
using ChargeApi = Charges<tsg::SharedCacheCharge>;
}
bool weight_charge_certificate_available() noexcept { return ChargeApi::available; }
} // namespace tsg::host_moe_vmm

#if defined(TSG_GGML_USE_CUDA)
#include "ggml_ops_internal.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-cuda.h"
#include <cuda.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <new>

namespace tsg::host_moe_vmm {
namespace {
constexpr std::size_t kMiB = std::size_t(1) << 20;
constexpr std::size_t kDeviceHeadroom = 512 * kMiB;
constexpr std::size_t kChunks = 4096, kRecords = 3 * kChunks, kTables = 3;
constexpr std::size_t kTicketAllocationAllowance = 4096; // Policy allowance, not a portable malloc bound.
constexpr std::size_t kTensorMetadata = 16384;
using PhysicalOwner = Owner<kChunks>;
struct Table;
struct Storage;
}

// Process-owned shell survives slab deletion and a ticket-only Host release
// failure. Only destroy_weight_arena_checked may clear the retained arena.
struct WeightArena {
    WeightOptions options{};
    std::shared_ptr<tsg::SharedCacheCharge> host;
    Storage* storage = nullptr;
    std::size_t host_bytes = 0, live_tickets = 0;
    std::uint64_t generation = 0, next_table = 0;
    CUdevice device = 0;
    CUcontext primary = nullptr;
    bool primary_owned = false, active = false;
    std::atomic<bool> failed{false}, context_lost{false};
#if defined(TSG_GGML_TEST_HOOKS)
    WeightTestStats tests{};
    WeightTestFault test_fault = WeightTestFault::none;
    int test_countdown = 0;
#endif
};

namespace {
void fail(WeightArena& a) noexcept {
    if (!a.failed.exchange(true, std::memory_order_acq_rel))
        tsg::g_ggml_error_count.fetch_add(1, std::memory_order_acq_rel);
    tsg::g_backend_compute_failed.store(true, std::memory_order_release);
}
struct Current {
    WeightArena& arena; CUcontext previous = nullptr;
    bool pushed = false, attempted = false, restored = false;
    explicit Current(WeightArena& a) noexcept : arena(a) {
        pushed = !a.context_lost.load(std::memory_order_acquire) && a.primary
            && cuCtxGetCurrent(&previous) == CUDA_SUCCESS && cuCtxPushCurrent(a.primary) == CUDA_SUCCESS;
        if (!pushed) fail(a);
    }
    bool finish() noexcept {
        if (attempted) return restored;
        attempted=true;
        if (!pushed) return false;
        CUcontext popped=nullptr, now=nullptr;
        restored=cuCtxPopCurrent(&popped)==CUDA_SUCCESS && popped==arena.primary
            && cuCtxGetCurrent(&now)==CUDA_SUCCESS && now==previous;
        if (!restored) {
            // An uncertain old thread-local stack cannot be repaired by a new
            // thread's push/pop. Retain this entire arena permanently rather
            // than guessing a Pop or releasing its context/metadata/tickets.
            arena.context_lost.store(true,std::memory_order_release); fail(arena);
        }
        return restored;
    }
    ~Current() { if (!attempted && pushed) finish(); } // Failure/exception fallback only.
};
struct Ticket {
    std::shared_ptr<tsg::SharedCacheCharge> charge;
    std::uint64_t id = 0;
    CUmemGenericAllocationHandle resource = 0;
    std::size_t bytes = 0, offset = 0;
    unsigned pool = 0, next_free = 0;
    bool mapped = false;
};
struct Driver final : Adapter {
    WeightArena& arena; Table& table;
    std::array<Ticket, kRecords> records{};
    unsigned free_head = 0;
    std::uint64_t epoch = 0;
    explicit Driver(WeightArena& a, Table& t) noexcept : arena(a), table(t) {
        for (unsigned i = 0; i != kRecords; ++i) records[i].next_free = i + 1;
    }
    Ticket* find(Id id) noexcept {
        const auto index = static_cast<unsigned>(id & 0xffff);
        return index && index <= kRecords && records[index-1].id == id ? &records[index-1] : nullptr;
    }
    Id reserve(std::size_t) noexcept override;
    bool commit(Id) noexcept override;
    bool refund(Id) noexcept override;
    Id create(unsigned, std::size_t, Id) noexcept override;
    bool map(unsigned, std::size_t, Id, std::size_t) noexcept override;
    bool access(unsigned, std::size_t, std::size_t) noexcept override;
    bool drain() noexcept override;
    bool unmap(unsigned, std::size_t, std::size_t) noexcept override;
    bool release(Id) noexcept override;
};
struct Pool {
    Table* table = nullptr;
    unsigned index = 0;
    CUdeviceptr base = 0;
    std::size_t reservation = 0, granule = 0, chunk = 0, alignment = 0;
    std::atomic<ggml_backend_buffer_t> wrapper{nullptr};
};
bool valid_options(const WeightOptions& options) noexcept {
    return options.cuda_device>=0 && options.budget_rank>=0 && options.maximum_total_chunks
        && options.maximum_total_chunks<=8192
        && options.preferred_chunk_bytes<=Layout::limit()
        && (!options.preferred_chunk_bytes
            || !(options.preferred_chunk_bytes&(options.preferred_chunk_bytes-1)));
}
ggml_tensor shape_tensor(const ProjectionShape& s,std::size_t capacity,std::size_t maximum) {
    if (!capacity || capacity>maximum || maximum>INT32_MAX || s.type<0 || s.type>=GGML_TYPE_COUNT
        || s.columns<=0 || s.rows<=0) throw std::invalid_argument("Invalid VMM tensor geometry");
    const auto block=ggml_blck_size(s.type); const auto unit=ggml_type_size(s.type);
    if (!block || !unit || s.columns%block) throw std::invalid_argument("Invalid VMM quantized row");
    const auto row=Layout::multiply(static_cast<std::size_t>(s.columns/block),unit);
    Layout::multiply(s.columns,unit); // Upstream row_size multiplies before division.
    if (ggml_row_size(s.type,s.columns)!=row || Layout::multiply(row,s.rows)!=s.stride)
        throw std::invalid_argument("VMM projection is not exactly contiguous");
    Layout::multiply(s.stride,maximum);
    std::int64_t elements=s.columns;
    for (auto dimension:{static_cast<std::uint64_t>(s.rows),static_cast<std::uint64_t>(maximum)}) {
        if (dimension>1 && UINT64_C(0x7fffffffffffffff)/dimension<=static_cast<std::uint64_t>(elements))
            throw std::overflow_error("VMM element count violates upstream tensor bound");
        elements*=static_cast<std::int64_t>(dimension);
    }
    ggml_tensor result{}; result.type=s.type; result.flags=GGML_TENSOR_FLAG_INPUT|GGML_TENSOR_FLAG_OUTPUT;
    result.ne[0]=s.columns; result.ne[1]=s.rows; result.ne[2]=static_cast<std::int64_t>(capacity); result.ne[3]=1;
    result.nb[0]=unit; result.nb[1]=row; result.nb[2]=s.stride; result.nb[3]=Layout::multiply(s.stride,capacity);
    return result; // No borrowed data/buffer/src/view/extra or graph pointers.
}
Layout fixed_layout(ggml_backend_buffer_type_t type,const std::array<ProjectionShape,3>& shapes,
    std::size_t maximum,std::size_t capacity,std::array<Projection,3> specs) {
    for (unsigned p=0;p!=3;++p) {
        const auto tensor=shape_tensor(shapes[p],capacity,maximum);
        specs[p].stride=shapes[p].stride;
        specs[p].allocation=ggml_backend_buft_get_alloc_size(type,&tensor);
    }
    return Layout::make(capacity,specs);
}
struct DevicePlan { ggml_backend_buffer_type_t type=nullptr; Layout layout; WeightLayoutPlan report; };
WeightStatus device_plan(const WeightOptions& options,const std::array<ProjectionShape,3>& shapes,
    std::size_t maximum,std::size_t capacity,DevicePlan& out) {
    if (!valid_options(options)) return WeightStatus::refused;
    // Validate all caller geometry before asking the driver/backend for metadata.
    for (const auto& shape:shapes) shape_tensor(shape,capacity,maximum);
    CUdevice device=0; int supported=0;
    if (cuInit(0)!=CUDA_SUCCESS || cuDeviceGet(&device,options.cuda_device)!=CUDA_SUCCESS
        || cuDeviceGetAttribute(&supported,CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,device)!=CUDA_SUCCESS)
        return WeightStatus::failed;
    if (!supported) return WeightStatus::unsupported;
    out.type=ggml_backend_cuda_buffer_type(options.cuda_device);
    if (!out.type || ggml_backend_buft_is_host(out.type) || out.type->iface.alloc_buffer_n || out.type->iface.get_alloc_size_n)
        return WeightStatus::unsupported;
    std::array<Projection,3> specs{}; std::size_t reserved=0;
    const auto max_buffer=(std::min)(ggml_backend_buft_get_max_size(out.type),Layout::limit());
    const auto alignment=ggml_backend_buft_get_alignment(out.type);
    if (!alignment || (alignment&(alignment-1))) return WeightStatus::unsupported;
    for (unsigned p=0;p!=3;++p) {
        const auto tensor=shape_tensor(shapes[p],maximum,maximum);
        const auto allocation=ggml_backend_buft_get_alloc_size(out.type,&tensor);
        CUmemAllocationProp prop{}; prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id=device;
        std::size_t granule=0;
        if (cuMemGetAllocationGranularity(&granule,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM)!=CUDA_SUCCESS)
            return WeightStatus::failed;
        if (!granule || granule%alignment) return WeightStatus::unsupported;
        const auto chunk=options.preferred_chunk_bytes?options.preferred_chunk_bytes:granule;
        if (chunk%granule) return WeightStatus::unsupported;
        const auto va=Layout::round(allocation,chunk);
        reserved=Layout::add(reserved,va);
        specs[p]={shapes[p].stride,0,granule,chunk,chunk,max_buffer,va};
    }
    out.layout=fixed_layout(out.type,shapes,maximum,capacity,specs);
    std::size_t chunks=0;
    out.report.capacity=capacity; out.report.maximum_capacity=maximum;
    out.report.total=out.layout.total(); out.report.reserved_va_total=reserved;
    for (unsigned p=0;p!=3;++p) {
        if (out.layout.chunks(p)>kChunks) return WeightStatus::refused;
        chunks=Layout::add(chunks,out.layout.chunks(p));
        const auto& s=out.layout.projection(p);
        out.report.projections[p]={s.stride,s.allocation,s.granule,s.chunk,out.layout.physical(p),s.reserved_va,s.maximum_buffer,alignment};
    }
    return chunks<=options.maximum_total_chunks?WeightStatus::ready:WeightStatus::refused;
}
struct Table {
    WeightArena& arena;
    Driver driver;
    PhysicalOwner physical;
    std::array<Pool,3> pools{};
    std::array<ggml_tensor*,3> tensors{};
    alignas(GGML_MEM_ALIGN) std::array<unsigned char,kTensorMetadata> tensor_memory{};
    ggml_context* context = nullptr;
    ggml_backend_buffer_type_t type = nullptr;
    WeightSpec spec{};
    Layout pending;
    std::uint64_t generation = 0;
    bool init_pending = false, initialized = false, closing = false, detached = false, published = false;
    explicit Table(WeightArena& a) noexcept : arena(a), driver(a,*this), physical(driver) {
        for (unsigned p=0;p!=3;++p) { pools[p].table=this; pools[p].index=p; }
    }
    Layout plan(std::size_t capacity) const {
        std::array<Projection,3> projections{};
        for (unsigned p=0;p!=3;++p) {
            projections[p]={spec.projections[p].stride,0,pools[p].granule,pools[p].chunk,
                pools[p].chunk,(std::min)(ggml_backend_buft_get_max_size(type),Layout::limit()),pools[p].reservation};
        }
        return fixed_layout(type,spec.projections,spec.maximum_capacity,capacity,projections);
    }
    bool mapped_prefix(unsigned p,std::size_t bytes) const noexcept {
        std::size_t total=0;
        for (const auto& r:driver.records) if (r.id && r.mapped && r.pool==p) total+=r.bytes;
        return total==bytes; // Driver maps exact unique chunk offsets only.
    }
    bool initialize_padding() noexcept {
        if (!init_pending || initialized || arena.failed.load(std::memory_order_acquire)) return true;
        for (unsigned p=0;p!=3;++p) if (!mapped_prefix(p,pending.physical(p))) return true;
        for (unsigned p=0;p!=3;++p) {
            const auto logical=spec.projections[p].stride*pending.capacity();
            const auto allocation=pending.projection(p).allocation;
            if (allocation>logical) {
                if (cuMemsetD8(pools[p].base+logical,0,allocation-logical)!=CUDA_SUCCESS)
                    { fail(arena); return false; }
#if defined(TSG_GGML_TEST_HOOKS)
                ++arena.tests.padding_writes;
                if (arena.test_fault==WeightTestFault::padding_after_write) {
                    if (!arena.test_countdown) {
                        arena.test_fault=WeightTestFault::none; ++arena.tests.padding_failures;
                        fail(arena); return false; // drain() STILL fences this real queued write.
                    }
                    --arena.test_countdown;
                }
#endif
            }
        }
        initialized=true; return true;
    }
    bool valid(unsigned p,const ggml_tensor* tensor,std::size_t offset,std::size_t bytes) const noexcept {
        if (closing || detached || !physical.usable() || arena.failed.load(std::memory_order_acquire)
            || tensor!=tensors[p] || tensor->view_src || tensor->view_offs
            || tensor->data!=reinterpret_cast<void*>(pools[p].base)) return false;
        const auto logical=ggml_nbytes(tensor);
        return offset<=logical && bytes<=logical-offset
            && logical<=physical.current().physical(p);
    }
    bool publish() noexcept {
        if (arena.failed.load(std::memory_order_acquire) || !physical.usable()) return false;
        const auto& layout=physical.current();
        for (unsigned p=0;p!=3;++p) {
            auto* buffer=pools[p].wrapper.load(std::memory_order_acquire);
            if (!buffer) { fail(arena); return false; }
            tensors[p]->ne[2]=static_cast<std::int64_t>(layout.capacity());
            tensors[p]->nb[3]=spec.projections[p].stride*layout.capacity();
            buffer->size=layout.physical(p);
            if (!tensors[p]->buffer && ggml_backend_tensor_alloc(buffer,tensors[p],reinterpret_cast<void*>(pools[p].base))!=GGML_STATUS_SUCCESS)
                { fail(arena); return false; }
        }
        published=true; return !arena.failed.load(std::memory_order_acquire);
    }
};
struct Slot { alignas(Table) unsigned char bytes[sizeof(Table)]; bool live=false; };
struct Storage { std::array<Slot,kTables> slots{}; };
WeightHostPlan host_plan(const WeightOptions& options) {
    if (!valid_options(options)) throw std::invalid_argument("Invalid VMM arena options");
    WeightHostPlan result; result.slab_bytes=sizeof(Storage);
    result.maximum_live_ticket_records=options.maximum_total_chunks;
    result.declared_charge_bytes=Layout::add(sizeof(Storage),Layout::add(kTables*65536,
        Layout::multiply(options.maximum_total_chunks,kTicketAllocationAllowance)));
    return result;
}
struct Registry { std::mutex mutex; WeightArena arena; std::uint64_t epoch=0; };
Registry& registry() {
    // Fixed bootstrap control; no heap allocation and no finalizer refunds.
    static std::aligned_storage_t<sizeof(Registry),alignof(Registry)> memory;
    static Registry* value=new(&memory) Registry;
    return *value;
}
Table* slot(WeightArena& a,unsigned i) noexcept {
    return a.storage && i<kTables && a.storage->slots[i].live
        ? reinterpret_cast<Table*>(a.storage->slots[i].bytes) : nullptr;
}
bool active(WeightArena* a) noexcept { return a==&registry().arena && a->active; }
Table* find(WeightArena* a,WeightHandle h) noexcept {
    if (!active(a) || h.arena!=a->generation) return nullptr;
    auto* t=slot(*a,h.slot); return t && t->generation==h.table ? t : nullptr;
}
bool barrier(WeightArena& a,const WeightBarrier& b) noexcept {
    if (!b.freeze_join_invalidate) return false;
    if (!b.freeze_join_invalidate(b.context)) { fail(a); return false; }
    return true;
}

Adapter::Id Driver::reserve(std::size_t bytes) noexcept {
    if (arena.failed.load(std::memory_order_acquire) || free_head>=kRecords
        || arena.live_tickets>=arena.options.maximum_total_chunks || epoch>=UINT64_C(0xffffffffffff)) return 0;
    auto& r=records[free_head]; const auto index=free_head;
    try { r.charge=tsg::SharedCacheCharge::reserve(arena.options.budget_rank,4,bytes); }
    catch (...) { return 0; }
    if (!r.charge) return 0;
    free_head=r.next_free; r.id=(++epoch<<16)|(index+1); r.bytes=bytes; ++arena.live_tickets;
    if (!ChargeApi::verify(*r.charge,arena.options.budget_rank,4,bytes,false)) fail(arena);
    return r.id; // Even bad certificate retains a visible token for checked cleanup.
}
bool Driver::commit(Id id) noexcept {
    auto* r=find(id); if (!r || arena.failed.load(std::memory_order_acquire)) return false;
    bool committed=false; try { committed=r->charge->commit(r->bytes); } catch (...) { fail(arena); }
    if (committed && !ChargeApi::verify(*r->charge,arena.options.budget_rank,4,r->bytes,true)) fail(arena);
    return committed; // Preserve actual commit transition even if verification then failed.
}
bool Driver::refund(Id id) noexcept {
    auto* r=find(id); if (!r || r->resource || r->mapped) { fail(arena); return false; }
    if (arena.context_lost.load(std::memory_order_acquire)) return false;
    if (!ChargeApi::release(*r->charge)) return false;
    const auto index=static_cast<unsigned>(r-records.data());
    r->charge.reset(); r->id=0; r->bytes=r->offset=0; r->next_free=free_head;
    free_head=index; --arena.live_tickets; return true;
}
Adapter::Id Driver::create(unsigned p,std::size_t bytes,Id ticket) noexcept {
    auto* r=find(ticket);
    if (!r || p>=3 || r->bytes!=bytes || r->resource || arena.failed.load(std::memory_order_acquire)) return 0;
    Current current(arena); if (!current.pushed) return 0;
    std::size_t free=0,total=0;
    if (cuMemGetInfo(&free,&total)!=CUDA_SUCCESS || !total || free<=kDeviceHeadroom
        || bytes>free-kDeviceHeadroom) return 0;
    CUmemAllocationProp prop{}; prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id=arena.device;
#if defined(TSG_GGML_TEST_HOOKS)
    if (arena.test_fault==WeightTestFault::create_after_successes && !arena.test_countdown) {
        arena.test_fault=WeightTestFault::none; ++arena.tests.create_failures; return 0;
    }
#endif
    CUmemGenericAllocationHandle resource=0;
    if (cuMemCreate(&resource,bytes,&prop,0)!=CUDA_SUCCESS) return 0;
#if defined(TSG_GGML_TEST_HOOKS)
    ++arena.tests.successful_creates;
    if (arena.test_fault==WeightTestFault::create_after_successes) --arena.test_countdown;
#endif
    r->resource=resource; r->pool=p; current.finish();
    return ticket; // Actual create must stay visible even after failed restore.
}
bool Driver::map(unsigned p,std::size_t offset,Id id,std::size_t bytes) noexcept {
    auto* r=find(id);
    if (arena.context_lost.load(std::memory_order_acquire)
        || !r || p>=3 || r->pool!=p || r->bytes!=bytes || !r->resource || r->mapped
        || offset%table.pools[p].chunk || bytes!=table.pools[p].chunk
        || offset>table.pools[p].reservation || bytes>table.pools[p].reservation-offset) return false;
    for (const auto& old:records) if (old.id && old.mapped && old.pool==p && old.offset==offset) return false;
    Current current(arena); if (!current.pushed) return false;
    if (cuMemMap(table.pools[p].base+offset,bytes,0,r->resource,0)!=CUDA_SUCCESS) return false;
    r->offset=offset; r->mapped=true; current.finish(); return true;
}
bool Driver::access(unsigned p,std::size_t offset,std::size_t bytes) noexcept {
    Current current(arena); if (!current.pushed || p>=3) return false;
    CUmemAccessDesc desc{}; desc.location.type=CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id=arena.device; desc.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    const bool accessible=cuMemSetAccess(table.pools[p].base+offset,bytes,&desc,1)==CUDA_SUCCESS;
    current.finish(); return accessible;
}
bool Driver::drain() noexcept {
    Current current(arena); if (!current.pushed) return false;
    const bool initialized=table.initialize_padding();
    const bool drained=cuCtxSynchronize()==CUDA_SUCCESS;
    const bool restored=current.finish();
    if (!drained) fail(arena);
    return initialized && drained && restored;
}
bool Driver::unmap(unsigned p,std::size_t offset,std::size_t bytes) noexcept {
    Current current(arena); if (!current.pushed) return false;
    for (auto& r:records) if (r.id && r.mapped && r.pool==p && r.offset==offset && r.bytes==bytes) {
        if (cuMemUnmap(table.pools[p].base+offset,bytes)!=CUDA_SUCCESS) return false;
        r.mapped=false; current.finish(); return true;
    }
    return false;
}
bool Driver::release(Id id) noexcept {
    auto* r=find(id); if (!r || !r->resource || r->mapped) return false;
    Current current(arena); if (!current.pushed) return false;
    if (cuMemRelease(r->resource)!=CUDA_SUCCESS) return false;
    r->resource=0; current.finish(); return true;
}

Pool& pool(ggml_backend_buffer_t b) noexcept { return *static_cast<Pool*>(b->context); }
void free_buffer(ggml_backend_buffer_t b) noexcept {
    auto& p=pool(b); p.wrapper.store(nullptr,std::memory_order_release);
    if (!p.table->closing) { p.table->detached=true; fail(p.table->arena); }
    // GGML still owns/frees b after return. No wrapper pointer is retained for
    // retry and no arena/table/Host charge is destroyed from this callback.
}
void* base(ggml_backend_buffer_t b) noexcept { return reinterpret_cast<void*>(pool(b).base); }
ggml_status init_tensor(ggml_backend_buffer_t b,ggml_tensor* t) noexcept {
    auto& p=pool(b);
    if (!p.table->valid(p.index,t,0,ggml_nbytes(t))) { fail(p.table->arena); return GGML_STATUS_FAILED; }
    return GGML_STATUS_SUCCESS;
}
template<class Operation> void tensor_operation(ggml_backend_buffer_t b,const ggml_tensor* t,
    std::size_t offset,std::size_t bytes,Operation operation) noexcept {
    auto& p=pool(b); auto& table=*p.table;
    if (!table.valid(p.index,t,offset,bytes)) { fail(table.arena); return; }
    Current current(table.arena); if (!current.pushed) return;
    const bool copied=operation(p.base+offset);
    const bool drained=cuCtxSynchronize()==CUDA_SUCCESS;
    const bool restored=current.finish();
    if (!copied || !drained || !restored) fail(table.arena);
}
void set_tensor(ggml_backend_buffer_t b,ggml_tensor* t,const void* src,std::size_t o,std::size_t n) noexcept {
    tensor_operation(b,t,o,n,[&](CUdeviceptr dst) { return (!n || src) && cuMemcpyHtoD(dst,src,n)==CUDA_SUCCESS; });
}
void get_tensor(ggml_backend_buffer_t b,const ggml_tensor* t,void* dst,std::size_t o,std::size_t n) noexcept {
    tensor_operation(b,t,o,n,[&](CUdeviceptr src) { return (!n || dst) && cuMemcpyDtoH(dst,src,n)==CUDA_SUCCESS; });
}
void memset_tensor(ggml_backend_buffer_t b,ggml_tensor* t,std::uint8_t v,std::size_t o,std::size_t n) noexcept {
    tensor_operation(b,t,o,n,[&](CUdeviceptr dst) { return cuMemsetD8(dst,v,n)==CUDA_SUCCESS; });
}
bool copy_tensor(ggml_backend_buffer_t b,const ggml_tensor* src,ggml_tensor* dst) noexcept {
    auto& p=pool(b); auto& t=*p.table;
    const auto bytes=ggml_nbytes(src);
    if (!src->buffer || src->buffer->buft!=t.type || !ggml_are_same_layout(src,dst)
        || !t.valid(p.index,dst,0,bytes) || !src->data || src->view_src || src->view_offs)
        { fail(t.arena); return true; }
    if (src->buffer->iface.free_buffer==free_buffer) {
        const auto& source=pool(src->buffer);
        // VMM allocations are not attributed to a creating CUDA context by
        // CU_POINTER_ATTRIBUTE_CONTEXT. Prove our own same-device source by
        // its retained owner and actual mapped tensor span instead.
        if (&source.table->arena!=&t.arena || !source.table->published
            || !source.table->valid(source.index,src,0,bytes)) { fail(t.arena); return true; }
    } else {
        Current current(t.arena); if (!current.pushed) return true;
        CUcontext source_context=nullptr; unsigned memory_type=0;
        if (cuPointerGetAttribute(&source_context,CU_POINTER_ATTRIBUTE_CONTEXT,reinterpret_cast<CUdeviceptr>(src->data))!=CUDA_SUCCESS
            || cuPointerGetAttribute(&memory_type,CU_POINTER_ATTRIBUTE_MEMORY_TYPE,reinterpret_cast<CUdeviceptr>(src->data))!=CUDA_SUCCESS
            || source_context!=t.arena.primary || memory_type!=CU_MEMORYTYPE_DEVICE) { fail(t.arena); return true; }
        if (!current.finish()) return true;
    }
    const auto from=reinterpret_cast<CUdeviceptr>(src->data);
    const auto begin=reinterpret_cast<CUdeviceptr>(ggml_backend_buffer_get_base(src->buffer));
    const auto allocation=ggml_backend_buffer_get_size(src->buffer);
    if (from<begin || from-begin>allocation || bytes>allocation-(from-begin)) { fail(t.arena); return true; }
    if (from==p.base) return true;
    if ((from<p.base && bytes>p.base-from) || (p.base<from && bytes>from-p.base)) { fail(t.arena); return true; }
    tensor_operation(b,dst,0,bytes,[&](CUdeviceptr to) { return cuMemcpyDtoD(to,from,bytes)==CUDA_SUCCESS; });
    return true; // False would trigger forbidden whole-tensor Host fallback.
}
void clear_buffer(ggml_backend_buffer_t b,std::uint8_t value) noexcept {
    auto& p=pool(b); auto& t=*p.table;
    if (b->size!=t.physical.current().physical(p.index)) { fail(t.arena); return; }
    tensor_operation(b,t.tensors[p.index],0,ggml_nbytes(t.tensors[p.index]),
        [&](CUdeviceptr dst) { return cuMemsetD8(dst,value,b->size)==CUDA_SUCCESS; });
}
bool release_table(Table& t) noexcept {
    t.init_pending=false;
    if (t.arena.context_lost.load(std::memory_order_acquire)) return false;
    if (!t.physical.close(t.driver)) return false;
    t.closing=true;
    for (auto& p:t.pools) if (auto* b=p.wrapper.exchange(nullptr,std::memory_order_acq_rel)) ggml_backend_buffer_free(b);
    for (auto& p:t.pools) if (p.base) {
        Current current(t.arena); if (!current.pushed) return false;
        const bool released=cuMemAddressFree(p.base,p.reservation)==CUDA_SUCCESS;
        if (released) p.base=0; // Keep the actual transition on restore failure.
        const bool restored=current.finish();
        if (!released || !restored) return false;
    }
    if (t.context) { ggml_free(t.context); t.context=nullptr; }
    return true;
}
bool cleanup(WeightArena& a) noexcept {
    if (a.context_lost.load(std::memory_order_acquire)) return false;
    if (a.storage) {
        for (unsigned i=0;i!=kTables;++i) if (auto* t=slot(a,i)) {
            if (!release_table(*t)) return false;
            t->~Table(); a.storage->slots[i].live=false;
        }
        if (a.primary_owned) {
            if (cuDevicePrimaryCtxRelease(a.device)!=CUDA_SUCCESS) return false;
            a.primary_owned=false; a.primary=nullptr;
        }
        delete a.storage; a.storage=nullptr; // Real Host slab free before Host refund.
    }
    if (a.host) {
        if (!ChargeApi::release(*a.host)) return false;
        a.host.reset();
    }
    return true;
}
WeightStatus translate(Result r) noexcept {
    switch (r) {
        case Result::published:return WeightStatus::ready;
        case Result::unchanged:return WeightStatus::unchanged;
        case Result::quarantined:return WeightStatus::quarantined;
        default:return WeightStatus::refused;
    }
}
} // namespace

WeightStatus plan_weight_host(const WeightOptions& options,WeightHostPlan& out) noexcept {
    out={}; try { out=host_plan(options); return WeightStatus::ready; }
    catch (...) { return WeightStatus::refused; }
}
WeightStatus plan_weight_layout(const WeightOptions& options,const std::array<ProjectionShape,3>& shapes,
    std::size_t maximum,std::size_t capacity,WeightLayoutPlan& out) noexcept {
    out={};
    try {
        DevicePlan plan; const auto status=device_plan(options,shapes,maximum,capacity,plan);
        if (status==WeightStatus::ready) out=plan.report;
        return status;
    } catch (...) { return WeightStatus::refused; }
}
WeightStatus create_weight_arena(const WeightOptions& options,WeightArena*& out) noexcept {
    if (out || !ChargeApi::available || !ChargeApi::attached()) return WeightStatus::unsupported;
    try { if (!tsg::SharedCacheCharge::host_accounting_attached()) return WeightStatus::unsupported; }
    catch (...) { return WeightStatus::unsupported; }
    auto& r=registry(); std::lock_guard<std::mutex> lock(r.mutex); auto& a=r.arena;
    if (a.active || !valid_options(options) || r.epoch==UINT64_MAX) return WeightStatus::refused;
    a.options=options; a.generation=++r.epoch; a.next_table=0; a.live_tickets=0;
    a.failed.store(false,std::memory_order_release); a.context_lost.store(false,std::memory_order_release);
#if defined(TSG_GGML_TEST_HOOKS)
    a.tests={}; a.test_fault=WeightTestFault::none; a.test_countdown=0;
#endif
    try {
        a.host_bytes=host_plan(options).declared_charge_bytes;
        a.host=tsg::SharedCacheCharge::reserve(options.budget_rank,3,a.host_bytes);
        if (!a.host) return WeightStatus::refused;
        a.active=true; out=&a; // Entire shell now retained, even constructor rollback failure.
        if (!ChargeApi::verify(*a.host,options.budget_rank,3,a.host_bytes,false)) return WeightStatus::quarantined;
        a.storage=new Storage;
        if (!a.host->commit(a.host_bytes) || !ChargeApi::verify(*a.host,options.budget_rank,3,a.host_bytes,true))
            return WeightStatus::quarantined;
        int supported=0; CUcontext current=nullptr;
        if (cuInit(0)!=CUDA_SUCCESS || cuDeviceGet(&a.device,options.cuda_device)!=CUDA_SUCCESS
            || cuDeviceGetAttribute(&supported,CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,a.device)!=CUDA_SUCCESS
            || !supported || cuCtxGetCurrent(&current)!=CUDA_SUCCESS || !current) {
            if (cleanup(a)) { a.active=false; out=nullptr; return WeightStatus::unsupported; }
            return WeightStatus::quarantined;
        }
        CUcontext retained=nullptr;
        if (cuDevicePrimaryCtxRetain(&retained,a.device)!=CUDA_SUCCESS) return WeightStatus::quarantined;
        a.primary=retained; a.primary_owned=true;
        if (current!=a.primary) {
            if (cleanup(a)) { a.active=false; out=nullptr; return WeightStatus::unsupported; }
            return WeightStatus::quarantined;
        }
        return WeightStatus::ready;
    } catch (...) {
        if (!a.active) return WeightStatus::refused;
        return WeightStatus::quarantined;
    }
}

WeightStatus create_weight_table(WeightArena* a,unsigned index,const WeightSpec& spec,WeightHandle& handle) noexcept {
    auto& r=registry(); std::lock_guard<std::mutex> lock(r.mutex);
    if (!active(a) || !a->storage || a->failed.load(std::memory_order_acquire) || index>=kTables
        || slot(*a,index) || !spec.source_owner || !spec.initial_capacity || spec.initial_capacity>spec.maximum_capacity
        || spec.maximum_capacity>INT32_MAX || a->next_table==UINT64_MAX) return WeightStatus::refused;
    DevicePlan planned;
    try {
        const auto status=device_plan(a->options,spec.projections,spec.maximum_capacity,spec.initial_capacity,planned);
        if (status!=WeightStatus::ready) return status;
    } catch (...) { return WeightStatus::refused; }
    auto& storage=a->storage->slots[index]; auto* t=new(storage.bytes) Table(*a); storage.live=true;
    t->spec=spec; t->generation=++a->next_table;
    handle={a->generation,t->generation,index}; // Failed cleanup remains addressable.
    try {
        t->type=planned.type;
        t->context=ggml_init({t->tensor_memory.size(),t->tensor_memory.data(),true});
        if (!t->context) throw std::bad_alloc();
        if (Layout::multiply(ggml_tensor_overhead(),3)>t->tensor_memory.size())
            throw std::invalid_argument("VMM tensor metadata exceeds fixed slab");
        for (unsigned p=0;p!=3;++p) {
            const auto& s=spec.projections[p];
            const auto expected=shape_tensor(s,spec.initial_capacity,spec.maximum_capacity);
            auto* tensor=ggml_new_tensor_3d(t->context,s.type,s.columns,s.rows,spec.initial_capacity);
            t->tensors[p]=tensor; ggml_set_input(tensor); ggml_set_output(tensor);
            if (!ggml_are_same_layout(tensor,&expected)) throw std::runtime_error("VMM owned tensor differs from planned geometry");
            auto& pool=t->pools[p];
            const auto& projection=planned.layout.projection(p);
            pool.granule=projection.granule; pool.chunk=projection.chunk; pool.reservation=projection.reserved_va;
            pool.alignment=planned.report.projections[p].alignment;
            Current current(*a); if (!current.pushed) throw std::runtime_error("VMM context unavailable");
            if (cuMemAddressReserve(&pool.base,pool.reservation,pool.granule,0,0)!=CUDA_SUCCESS)
                throw std::runtime_error("VMM VA reserve failed");
            if (pool.base>UINT64_MAX-pool.reservation || pool.base%pool.alignment)
                throw std::overflow_error("VMM VA interval overflow");
            if (!current.finish()) throw std::runtime_error("VMM context restore failed");
        }
        for (unsigned p=0;p!=3;++p) for (unsigned q=0;q<p;++q)
            if (t->pools[p].base<t->pools[q].base+t->pools[q].reservation
                && t->pools[q].base<t->pools[p].base+t->pools[p].reservation)
                throw std::runtime_error("Independent VMM reservations overlap");
        ggml_backend_buffer_i iface{}; iface.free_buffer=free_buffer; iface.get_base=base; iface.init_tensor=init_tensor;
        iface.memset_tensor=memset_tensor; iface.set_tensor=set_tensor; iface.get_tensor=get_tensor;
        iface.cpy_tensor=copy_tensor; iface.clear=clear_buffer;
        for (auto& p:t->pools) {
            auto* buffer=ggml_backend_buffer_init(t->type,iface,&p,0);
            if (!buffer) throw std::bad_alloc();
            ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            p.wrapper.store(buffer,std::memory_order_release);
        }
        t->pending=planned.layout; t->init_pending=true;
        const auto result=t->physical.grow(t->pending,t->driver); t->init_pending=false;
        if (result==Result::published && t->publish()) return WeightStatus::ready;
        if (result==Result::quarantined) return WeightStatus::quarantined;
    } catch (...) { fail(*a); }
    if (!release_table(*t)) return WeightStatus::quarantined;
    t->~Table(); storage.live=false; handle={}; return WeightStatus::failed;
}

WeightStatus grow_weight_table(WeightArena* a,WeightHandle h,std::size_t capacity,const WeightBarrier& b) noexcept {
    auto& r=registry(); std::lock_guard<std::mutex> lock(r.mutex); auto* t=find(a,h);
    if (!t || !t->published || a->failed.load(std::memory_order_acquire)) return WeightStatus::refused;
    Layout target;
    try {
        target=t->plan(capacity); t->physical.current().growth_delta(target);
        for (unsigned p=0;p!=3;++p) if (target.chunks(p)>kChunks) return WeightStatus::refused;
    } catch (...) { return WeightStatus::refused; }
    if (capacity==t->physical.current().capacity()) return WeightStatus::unchanged;
    if (!barrier(*a,b)) return WeightStatus::failed;
    try {
        t->pending=target; t->init_pending=true; t->initialized=false;
        const auto result=t->physical.grow(t->pending,t->driver); t->init_pending=false;
        if (result==Result::published && !t->publish()) return WeightStatus::failed;
        return translate(result);
    } catch (...) { fail(*a); return WeightStatus::failed; }
}
WeightStatus shrink_weight_table_empty_tail(WeightArena* a,WeightHandle h,std::size_t capacity,
    const void* owner,std::uint64_t generation,const int* slots,std::size_t count,const WeightBarrier& b) noexcept {
    auto& r=registry(); std::lock_guard<std::mutex> lock(r.mutex); auto* t=find(a,h);
    if (!t || !t->published || owner!=t->spec.source_owner || a->failed.load(std::memory_order_acquire)) return WeightStatus::refused;
    if (!EmptyTail::prove(t->physical.current().capacity(),capacity,generation,slots,count))
        return WeightStatus::refused;
    Layout target;
    try { target=t->plan(capacity); } catch (...) { return WeightStatus::refused; }
    if (!barrier(*a,b)) return WeightStatus::failed;
    // Mint privately from this table AFTER the caller freezes/joins producers.
    // No externally transferable proof or borrowed slot span is retained.
    const auto proof=EmptyTail::prove(t->physical.current().capacity(),capacity,generation,slots,count);
    if (!proof) return WeightStatus::refused;
    try {
        t->pending=target; t->init_pending=true; t->initialized=false;
        const auto result=t->physical.shrink(t->pending,*proof,generation,t->driver);
        // A smaller quantized tail can begin in an old mapped chunk. Initialize
        // exact alloc-size padding after tail removal and fence before publish.
        if (result==Result::published && (!t->driver.drain() || !t->publish())) { t->init_pending=false; return WeightStatus::failed; }
        t->init_pending=false;
        return translate(result);
    } catch (...) { fail(*a); return WeightStatus::failed; }
}
WeightStatus copy_weight_slot(WeightArena* a,WeightHandle h,std::size_t from,std::size_t to,
    const void* owner,const int* slots,std::size_t count,const WeightBarrier& b) noexcept {
    auto& r=registry(); std::lock_guard<std::mutex> lock(r.mutex); auto* t=find(a,h);
    if (!t || !t->published || !t->physical.usable() || owner!=t->spec.source_owner || !slots
        || count!=t->physical.current().capacity() || from>=count || to>=count || slots[from]<0 || slots[to]!=-1
        || a->failed.load(std::memory_order_acquire)) return WeightStatus::refused;
    for (std::size_t i=0;i!=count;++i) if (slots[i]<-1) return WeightStatus::refused;
    if (!barrier(*a,b)) return WeightStatus::failed;
    if (slots[from]<0 || slots[to]!=-1) return WeightStatus::refused;
    Current current(*a); if (!current.pushed) return WeightStatus::failed;
    bool copied=true;
    for (unsigned p=0;p!=3;++p) {
        const auto bytes=t->spec.projections[p].stride;
        if (cuMemcpyDtoD(t->pools[p].base+to*bytes,t->pools[p].base+from*bytes,bytes)!=CUDA_SUCCESS)
            { copied=false; break; }
    }
    const bool drained=cuCtxSynchronize()==CUDA_SUCCESS;
    const bool restored=current.finish();
    if (!copied || !drained || !restored) { fail(*a); return WeightStatus::failed; }
    return WeightStatus::ready; // Caller alone publishes source-qualified slot tags after success.
}
ggml_tensor* weight_tensor(WeightArena* a,WeightHandle h,unsigned p) noexcept {
    auto& r=registry(); std::lock_guard<std::mutex> lock(r.mutex); auto* t=find(a,h);
    return t && p<3 && t->published && t->physical.usable() && !a->failed.load(std::memory_order_acquire) ? t->tensors[p] : nullptr;
}
WeightUsage weight_usage(WeightArena* a,WeightHandle h) noexcept {
    auto& r=registry(); std::lock_guard<std::mutex> lock(r.mutex); WeightUsage u;
    if (!active(a)) return u;
    u.declared_host_charge=a->host_bytes; u.portable_and_adapter_slab=a->storage?sizeof(Storage):0;
    u.failed=a->failed.load(std::memory_order_acquire); u.charge_certificate_api=ChargeApi::available;
    u.context_lost=a->context_lost.load(std::memory_order_acquire);
    u.actual_charge_certificate=!u.failed && a->host
        && ChargeApi::verify(*a->host,a->options.budget_rank,3,a->host_bytes,true);
    u.host_ticket_held=static_cast<bool>(a->host); u.host_ticket_only=a->host && !a->storage;
    u.primary_context_owned=a->primary_owned;
    if (auto* t=find(a,h)) {
        const auto physical=t->physical.usage(); u.capacity=t->physical.current().capacity();
        u.ticket_pending=physical.pending; u.ticket_committed=physical.committed; u.mapped=physical.mapped;
        u.created_unmapped=physical.created_unmapped; u.ticket_only=physical.ticket_only; u.quarantine=physical.quarantined;
        u.ticket_records=physical.records; u.usable=t->published && t->physical.usable() && !u.failed;
        for (const auto& ticket:t->driver.records) if (ticket.id) {
            try { u.actual_charge_certificate=u.actual_charge_certificate && ticket.charge
                && ChargeApi::verify(*ticket.charge,a->options.budget_rank,4,ticket.bytes,ticket.charge->is_committed()); }
            catch (...) { u.actual_charge_certificate=false; }
        }
        for (unsigned p=0;p!=3;++p) { u.physical_prefix[p]=t->physical.current().physical(p); u.reserved_va[p]=t->pools[p].reservation; }
    }
    return u;
}
bool close_weight_table(WeightArena* a,WeightHandle h,const WeightBarrier& b) noexcept {
    auto& r=registry(); std::lock_guard<std::mutex> lock(r.mutex); auto* t=find(a,h);
    if (!t || !barrier(*a,b) || !release_table(*t)) return false;
    t->~Table(); a->storage->slots[h.slot].live=false; return true;
}
bool retry_weight_cleanup(WeightArena* a,const WeightBarrier& b) noexcept {
    auto& r=registry(); std::lock_guard<std::mutex> lock(r.mutex);
    return active(a) && barrier(*a,b) && cleanup(*a);
}
bool destroy_weight_arena_checked(WeightArena*& a,const WeightBarrier& b) noexcept {
    auto& r=registry(); std::lock_guard<std::mutex> lock(r.mutex);
    if (!active(a) || !barrier(*a,b) || !cleanup(*a)) return false;
    a->active=false; a=nullptr; return true;
}
#if defined(TSG_GGML_TEST_HOOKS)
bool weight_test_fault(WeightArena* a,WeightTestFault fault,int countdown) noexcept {
    auto& r=registry(); std::lock_guard<std::mutex> lock(r.mutex);
    if (!active(a) || countdown<0 || (fault!=WeightTestFault::none
        && fault!=WeightTestFault::create_after_successes && fault!=WeightTestFault::padding_after_write)
        || (fault==WeightTestFault::none && countdown)) return false;
    a->test_fault=fault; a->test_countdown=countdown; return true;
}
WeightTestStats weight_test_stats(WeightArena* a) noexcept {
    auto& r=registry(); std::lock_guard<std::mutex> lock(r.mutex);
    return active(a)?a->tests:WeightTestStats{};
}
#endif
} // namespace tsg::host_moe_vmm

#else
namespace tsg::host_moe_vmm {
WeightStatus plan_weight_host(const WeightOptions&,WeightHostPlan& out) noexcept { out={}; return WeightStatus::unsupported; }
WeightStatus plan_weight_layout(const WeightOptions&,const std::array<ProjectionShape,3>&,std::size_t,std::size_t,
    WeightLayoutPlan& out) noexcept { out={}; return WeightStatus::unsupported; }
WeightStatus create_weight_arena(const WeightOptions&,WeightArena*&) noexcept { return WeightStatus::unsupported; }
WeightStatus create_weight_table(WeightArena*,unsigned,const WeightSpec&,WeightHandle&) noexcept { return WeightStatus::unsupported; }
WeightStatus grow_weight_table(WeightArena*,WeightHandle,std::size_t,const WeightBarrier&) noexcept { return WeightStatus::unsupported; }
WeightStatus shrink_weight_table_empty_tail(WeightArena*,WeightHandle,std::size_t,const void*,std::uint64_t,const int*,std::size_t,const WeightBarrier&) noexcept { return WeightStatus::unsupported; }
WeightStatus copy_weight_slot(WeightArena*,WeightHandle,std::size_t,std::size_t,const void*,const int*,std::size_t,const WeightBarrier&) noexcept { return WeightStatus::unsupported; }
ggml_tensor* weight_tensor(WeightArena*,WeightHandle,unsigned) noexcept { return nullptr; }
WeightUsage weight_usage(WeightArena*,WeightHandle) noexcept { return {}; }
bool close_weight_table(WeightArena*,WeightHandle,const WeightBarrier&) noexcept { return false; }
bool retry_weight_cleanup(WeightArena*,const WeightBarrier&) noexcept { return false; }
bool destroy_weight_arena_checked(WeightArena*&,const WeightBarrier&) noexcept { return false; }
#if defined(TSG_GGML_TEST_HOOKS)
bool weight_test_fault(WeightArena*,WeightTestFault,int) noexcept { return false; }
WeightTestStats weight_test_stats(WeightArena*) noexcept { return {}; }
#endif
}
#endif
