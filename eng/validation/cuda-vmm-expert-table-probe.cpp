// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Standalone, source-only experimental validation. No production allocator hooks.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-cuda.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::size_t mib = std::size_t(1) << 20;
constexpr int hidden = 512, intermediate = 512, maximum_slots = 32, selected = 2;
constexpr int initial_slots = 8, grown_slots = 24, shrunk_slots = 12;
bool callback_failed = false;
std::uint64_t next_uid = UINT64_C(0x564d4d5000000000);
std::uint64_t owned_copy_calls = 0;

void stage(const char* name, int capacity = 0, int width = 0, std::uint64_t uid = 0) noexcept {
    std::fprintf(stderr, "[VMM-PROBE-STAGE] stage=%s capacity=%d width=%d uid=%llu\n",
        name, capacity, width, static_cast<unsigned long long>(uid));
    std::fflush(stderr);
}

void require(bool yes, const char* message) { if (!yes) throw std::runtime_error(message); }
void driver(CUresult status, const char* operation) {
    if (status == CUDA_SUCCESS) return;
    const char* name = nullptr; cuGetErrorName(status, &name);
    throw std::runtime_error(std::string(operation) + ": " + (name ? name : "unknown driver error"));
}
void runtime(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
bool runtime_callback(cudaError_t status, const char* operation) noexcept {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "FAIL callback %s: %s\n", operation, cudaGetErrorString(status));
    callback_failed = true; return false;
}
std::size_t rounded(std::size_t bytes, std::size_t alignment) {
    require(alignment && bytes <= (std::numeric_limits<std::size_t>::max)() - (alignment - 1), "rounding overflow");
    return ((bytes + alignment - 1) / alignment) * alignment;
}

// Tickets count declared standard allocations and exact cuMemCreate bytes.
// Reserved VA, CUDA context/driver pages and upstream CUDA workspace pools are
// reported outside this payload ledger, never mislabelled as zero total VRAM.
struct Ledger {
    std::size_t limit = 128 * mib, pending = 0, committed = 0, peak = 0;
    bool reserve(std::size_t bytes) {
        if (!bytes || committed > limit || pending > limit - committed
            || bytes > limit - committed - pending) return false;
        pending += bytes; peak = (std::max)(peak, pending + committed); return true;
    }
    void commit(std::size_t bytes) { require(bytes <= pending, "commit without pending ticket"); pending -= bytes; committed += bytes; }
    void release(std::size_t bytes, bool was_committed) {
        auto& pool = was_committed ? committed : pending;
        require(bytes <= pool, "refund before ownership"); pool -= bytes;
    }
};
struct Chunk {
    CUdeviceptr address = 0;
    CUmemGenericAllocationHandle handle = 0;
    std::size_t bytes = 0;
    bool mapped = false, committed = false;
};
struct Projection;
std::array<Projection*, 3> quarantine{};

struct Projection {
    Ledger& ledger;
    ggml_backend_buffer_type_t type;
    ggml_tensor* tensor;
    ggml_backend_buffer_t buffer = nullptr;
    CUdeviceptr base = 0;
    std::size_t reservation = 0, chunk_bytes = 0, logical_bytes = 0;
    std::vector<Chunk> chunks;
    int device = 0;

    Projection(Ledger& ledger, ggml_backend_buffer_type_t type, ggml_tensor* tensor, std::size_t granularity)
        : ledger(ledger), type(type), tensor(tensor), chunk_bytes(rounded(2 * mib, granularity)) {
        reservation = rounded(payload(maximum_slots), chunk_bytes);
        chunks.reserve(reservation / chunk_bytes); // All growth metadata before physical allocation.
        driver(cuMemAddressReserve(&base, reservation, granularity, 0, 0), "cuMemAddressReserve");
    }
    std::size_t payload(int slots) const {
        auto descriptor = *tensor;
        descriptor.ne[2] = slots; descriptor.nb[3] = descriptor.nb[2] * slots;
        return ggml_backend_buft_get_alloc_size(type, &descriptor);
    }
    bool drop_chunk(Chunk& chunk) noexcept {
        // Retain the exact ticket and state after either cleanup failure.
        if (chunk.mapped) {
            if (cuMemUnmap(chunk.address, chunk.bytes) != CUDA_SUCCESS) return false;
            chunk.mapped = false;
        }
        if (chunk.handle) {
            if (cuMemRelease(chunk.handle) != CUDA_SUCCESS) return false;
            chunk.handle = 0;
        }
        try { ledger.release(chunk.bytes, chunk.committed); }
        catch (...) { callback_failed = true; return false; }
        chunk.bytes = 0; return true;
    }
    bool rollback(std::size_t old_count) noexcept {
        if (chunks.size() <= old_count) return true;
        stage("rollback-drain-before-unmap");
        // cudaMemset on device storage may return before its device work is
        // finished. Every rollback, including partial growth, must fence it
        // before unmapping ANY destination. Failed fences retain all records
        // and their exact tickets for close/quarantine; never refund here.
        if (!runtime_callback(cudaSetDevice(device), "rollback set device")
            || !runtime_callback(cudaDeviceSynchronize(), "rollback all queued chunk work")) return false;
        bool okay = true;
        for (std::size_t i = chunks.size(); i-- > old_count;) {
            if (drop_chunk(chunks[i])) chunks.erase(chunks.begin() + i);
            else okay = false;
        }
        return okay;
    }
    bool close() noexcept {
        if (!runtime_callback(cudaSetDevice(device), "close set device")
            || !runtime_callback(cudaDeviceSynchronize(), "close all streams")) return false;
        if (!rollback(0)) return false;
        if (base) {
            if (cuMemAddressFree(base, reservation) != CUDA_SUCCESS) return false;
            base = 0;
        }
        return true;
    }
    bool valid(const ggml_tensor* target, std::size_t offset, std::size_t bytes) const noexcept {
        const auto address = reinterpret_cast<std::uintptr_t>(target->data);
        const auto begin = static_cast<std::uintptr_t>(base);
        return address >= begin && address - begin <= logical_bytes && offset <= logical_bytes - (address - begin)
            && bytes <= logical_bytes - (address - begin) - offset;
    }
    static Projection& context(ggml_backend_buffer_t buffer) noexcept { return *static_cast<Projection*>(buffer->context); }
    static void retain_quarantine(Projection* owner) noexcept {
        callback_failed = true;
        for (auto& held : quarantine) if (!held) { held = owner; return; }
        std::fputs("FAIL bounded VMM cleanup quarantine overflow\n", stderr); std::abort();
    }
    static void free_buffer(ggml_backend_buffer_t buffer) noexcept {
        auto* owner = static_cast<Projection*>(buffer->context);
        if (owner->close()) { delete owner; return; }
        retain_quarantine(owner);
    }
    static void* get_base(ggml_backend_buffer_t buffer) noexcept { return reinterpret_cast<void*>(context(buffer).base); }
    static ggml_status init_tensor(ggml_backend_buffer_t buffer, ggml_tensor* tensor) noexcept {
        auto& owner = context(buffer);
        if (tensor->view_src || !owner.valid(tensor, 0, ggml_backend_buft_get_alloc_size(owner.type, tensor))) {
            callback_failed = true; return GGML_STATUS_FAILED;
        }
        // Exact standard CUDA allocation-size padding, not assumed row padding.
        const auto bytes = ggml_nbytes(tensor), allocation = ggml_backend_buft_get_alloc_size(owner.type, tensor);
        if (!runtime_callback(cudaSetDevice(owner.device), "init set device")) return GGML_STATUS_FAILED;
        if (allocation > bytes && !runtime_callback(cudaMemset(static_cast<char*>(tensor->data) + bytes, 0, allocation - bytes), "init padding"))
            return GGML_STATUS_FAILED;
        if (!runtime_callback(cudaDeviceSynchronize(), "init padding fence")) return GGML_STATUS_FAILED;
        return GGML_STATUS_SUCCESS;
    }
    static void memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor* tensor, std::uint8_t value, std::size_t offset, std::size_t bytes) noexcept {
        auto& owner = context(buffer);
        if (!owner.valid(tensor, offset, bytes)) { callback_failed = true; return; }
        if (runtime_callback(cudaSetDevice(owner.device), "memset set device"))
            runtime_callback(cudaMemset(static_cast<char*>(tensor->data) + offset, value, bytes), "memset tensor");
        runtime_callback(cudaDeviceSynchronize(), "memset tensor fence");
    }
    static void set_tensor(ggml_backend_buffer_t buffer, ggml_tensor* tensor, const void* data, std::size_t offset, std::size_t bytes) noexcept {
        auto& owner = context(buffer);
        if (!owner.valid(tensor, offset, bytes)) { callback_failed = true; return; }
        if (runtime_callback(cudaSetDevice(owner.device), "set device"))
            runtime_callback(cudaMemcpy(static_cast<char*>(tensor->data) + offset, data, bytes, cudaMemcpyHostToDevice), "set tensor");
        runtime_callback(cudaDeviceSynchronize(), "set tensor fence");
    }
    static void get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor* tensor, void* data, std::size_t offset, std::size_t bytes) noexcept {
        auto& owner = context(buffer);
        if (!owner.valid(tensor, offset, bytes)) { callback_failed = true; return; }
        if (runtime_callback(cudaSetDevice(owner.device), "get device"))
            runtime_callback(cudaMemcpy(data, static_cast<const char*>(tensor->data) + offset, bytes, cudaMemcpyDeviceToHost), "get tensor");
        runtime_callback(cudaDeviceSynchronize(), "get tensor fence");
        if (callback_failed) {
            std::fprintf(stderr, "[VMM-PROBE-GET-FAILURE] capacity=%lld offset=%zu bytes=%zu base=%llu mapped_prefix=%zu\n",
                static_cast<long long>(tensor->ne[2]), offset, bytes, static_cast<unsigned long long>(owner.base), owner.logical_bytes);
            std::fflush(stderr);
        }
    }
    static bool copy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor* source, ggml_tensor* target) noexcept {
        auto& owner = context(buffer);
        cudaPointerAttributes source_info{};
        // A failed callback must never request ggml's unaccounted full-tensor
        // malloc fallback. This prototype permits same-device copies only.
        if (source->buffer->buft != owner.type || !ggml_are_same_layout(source, target)
            || !owner.valid(target, 0, ggml_nbytes(source))
            || cudaPointerGetAttributes(&source_info, source->data) != cudaSuccess
            || source_info.type != cudaMemoryTypeDevice || source_info.device != owner.device) {
            std::fputs("FAIL unsupported VMM copy; host staging forbidden\n", stderr); std::abort();
        }
        if (runtime_callback(cudaSetDevice(owner.device), "copy set device"))
            runtime_callback(cudaMemcpy(target->data, source->data, ggml_nbytes(source), cudaMemcpyDeviceToDevice), "same-device copy");
        runtime_callback(cudaDeviceSynchronize(), "owned D2D copy fence");
        ++owned_copy_calls;
        // Returning false here would invoke the forbidden full-tensor staging
        // fallback. Callback failure is sticky and fails the next probe check.
        return true;
    }
    static void clear(ggml_backend_buffer_t buffer, std::uint8_t value) noexcept {
        auto& owner = context(buffer);
        if (runtime_callback(cudaSetDevice(owner.device), "clear set device"))
            runtime_callback(cudaMemset(reinterpret_cast<void*>(owner.base), value, owner.logical_bytes), "clear mapped prefix");
        runtime_callback(cudaDeviceSynchronize(), "clear mapped prefix fence");
    }
    void make_buffer() {
        ggml_backend_buffer_i iface{};
        iface.free_buffer = free_buffer; iface.get_base = get_base; iface.init_tensor = init_tensor;
        iface.memset_tensor = memset_tensor; iface.set_tensor = set_tensor; iface.get_tensor = get_tensor;
        iface.cpy_tensor = copy_tensor; iface.clear = clear;
        // Omitted 2D callbacks use ggml's bounded row loops, not a full copy.
        buffer = ggml_backend_buffer_init(type, iface, this, 0);
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    }
};

struct Table {
    Ledger& ledger;
    ggml_backend_t backend;
    ggml_context* ctx = nullptr;
    std::array<Projection*, 3> projections{};
    std::array<ggml_tensor*, 3> weights{};
    std::array<int, maximum_slots> source_ids{};
    int capacity = 0, last_created = 0;
    Table(Ledger& ledger, ggml_backend_t backend, std::size_t granularity) : ledger(ledger), backend(backend) {
        ctx = ggml_init({64 * 1024, nullptr, true}); require(ctx != nullptr, "weight metadata context");
        source_ids.fill(-1);
        try {
            for (int p = 0; p < 3; ++p) {
                weights[p] = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, p == 2 ? intermediate : hidden,
                    p == 2 ? hidden : intermediate, initial_slots);
                ggml_set_input(weights[p]); ggml_set_output(weights[p]);
                auto* owner = new Projection(ledger, ggml_backend_get_default_buffer_type(backend), weights[p], granularity);
                try { owner->make_buffer(); }
                catch (...) {
                    // Publication guard covers reserved VA even when creating
                    // the ggml buffer object throws. Failed cleanup retains its
                    // owner for retry rather than losing the VA address.
                    if (owner->close()) delete owner;
                    else Projection::retain_quarantine(owner);
                    throw;
                }
                projections[p] = owner;
            }
            for (int a = 0; a < 3; ++a) for (int b = 0; b < a; ++b)
                require(projections[a]->base + projections[a]->reservation <= projections[b]->base
                    || projections[b]->base + projections[b]->reservation <= projections[a]->base, "projection VA reservations overlap");
            require(grow(initial_slots), "initial physical VMM mapping");
            for (int p = 0; p < 3; ++p)
                require(ggml_backend_tensor_alloc(projections[p]->buffer, weights[p], reinterpret_cast<void*>(projections[p]->base)) == GGML_STATUS_SUCCESS,
                    "bind owned VMM tensor");
        } catch (...) { close(); throw; }
    }
    ~Table() { close(); }
    void close() noexcept {
        if (backend) ggml_backend_synchronize(backend);
        for (auto& owner : projections) if (owner) { auto buffer = owner->buffer; owner = nullptr; ggml_backend_buffer_free(buffer); }
        if (ctx) { ggml_free(ctx); ctx = nullptr; }
    }
    bool grow(int slots, int fail_create_after = -1) {
        require(slots > capacity && slots <= maximum_slots, "growth shape");
        stage(fail_create_after < 0 ? "grow-begin" : "grow-injected-begin", capacity, slots);
        last_created = 0;
        ggml_backend_synchronize(backend);
        std::array<std::size_t, 3> old_counts{};
        for (int p = 0; p < 3; ++p) old_counts[p] = projections[p]->chunks.size();
        auto rollback = [&] {
            bool okay = true;
            for (int p = 2; p >= 0; --p) okay = projections[p]->rollback(old_counts[p]) && okay;
            require(okay, "growth rollback retained quarantined physical ownership");
        };
        try {
            // Reserve every delta ticket before any cuMemCreate.
            for (auto* owner : projections) {
                const auto count = rounded(owner->payload(slots), owner->chunk_bytes) / owner->chunk_bytes;
                while (owner->chunks.size() < count) {
                    if (!ledger.reserve(owner->chunk_bytes)) { rollback(); return false; }
                    owner->chunks.push_back({owner->base + owner->chunks.size() * owner->chunk_bytes, 0, owner->chunk_bytes, false, false});
                }
            }
            for (int p = 0; p < 3; ++p) {
                auto& owner = *projections[p];
                for (std::size_t i = old_counts[p]; i < owner.chunks.size(); ++i) {
                    if (last_created == fail_create_after) { rollback(); return false; }
                    auto& chunk = owner.chunks[i];
                    std::fprintf(stderr, "[VMM-PROBE-CHUNK] part=%d index=%zu address=%llu bytes=%zu\n", p, i,
                        static_cast<unsigned long long>(chunk.address), chunk.bytes); std::fflush(stderr);
                    CUmemAllocationProp prop{}; prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
                    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id = owner.device;
                    driver(cuMemCreate(&chunk.handle, chunk.bytes, &prop, 0), "cuMemCreate"); ++last_created;
                    driver(cuMemMap(chunk.address, chunk.bytes, 0, chunk.handle, 0), "cuMemMap"); chunk.mapped = true;
                    CUmemAccessDesc access{}; access.location = prop.location; access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                    driver(cuMemSetAccess(chunk.address, chunk.bytes, &access, 1), "cuMemSetAccess");
                    runtime(cudaMemset(reinterpret_cast<void*>(chunk.address), 0, chunk.bytes), "zero new physical chunk");
                }
            }
            stage("grow-initialization-fence", capacity, slots);
            // Complete new-chunk zeroing before publishing geometry or using
            // the independent ggml backend stream to upload/compute weights.
            runtime(cudaDeviceSynchronize(), "new chunk initialization fence");
            for (int p = 0; p < 3; ++p) for (std::size_t i = old_counts[p]; i < projections[p]->chunks.size(); ++i) {
                auto& chunk = projections[p]->chunks[i]; ledger.commit(chunk.bytes); chunk.committed = true;
            }
            publish(slots); stage("grow-published", capacity); return true;
        } catch (...) { rollback(); throw; }
    }
    void publish(int slots) noexcept {
        for (int p = 0; p < 3; ++p) {
            auto& owner = *projections[p];
            owner.logical_bytes = rounded(owner.payload(slots), owner.chunk_bytes);
            owner.buffer->size = owner.logical_bytes; // Only a fully mapped prefix, never reserved VA.
            weights[p]->ne[2] = slots; weights[p]->nb[3] = weights[p]->nb[2] * slots;
        }
        capacity = slots;
    }
    void copy_slot(int source, int destination, cudaStream_t stream) {
        require(source >= 0 && source < capacity && destination >= 0 && destination < capacity && source != destination, "D2D slot bounds");
        ggml_backend_synchronize(backend);
        for (auto* weight : weights)
            runtime(cudaMemcpyAsync(static_cast<char*>(weight->data) + destination * weight->nb[2],
                static_cast<const char*>(weight->data) + source * weight->nb[2], weight->nb[2], cudaMemcpyDeviceToDevice, stream), "D2D hot-slot compaction");
        runtime(cudaStreamSynchronize(stream), "fence D2D slot compaction");
        source_ids[destination] = source_ids[source];
    }
    void shrink(int slots) {
        stage("shrink-begin", capacity, slots);
        require(slots > 0 && slots < capacity, "shrink shape"); ggml_backend_synchronize(backend);
        // Logical publication first prevents any future ID from naming an
        // unmapped tail. Failed reclamation stays charged and fails the probe.
        publish(slots);
        for (auto* owner : projections) {
            const auto keep = owner->logical_bytes / owner->chunk_bytes;
            require(owner->rollback(keep), "tail reclamation failed; physical ticket retained");
        }
        for (int i = slots; i < maximum_slots; ++i) source_ids[i] = -1;
        stage("shrink-complete", capacity);
    }
    std::size_t physical_bytes() const noexcept {
        std::size_t bytes = 0;
        for (const auto* owner : projections) for (const auto& chunk : owner->chunks)
            if (chunk.handle || chunk.mapped) bytes += chunk.bytes;
        return bytes;
    }
};

struct StandardWeights {
    Ledger& ledger;
    ggml_backend_t backend;
    ggml_context* ctx = nullptr;
    std::array<ggml_tensor*, 3> weights{};
    std::array<ggml_backend_buffer_t, 3> buffers{};
    std::array<std::size_t, 3> charges{};
    StandardWeights(Ledger& ledger, ggml_backend_t backend, const Table& table,
        const std::array<std::vector<std::uint8_t>, 3>& raw) : ledger(ledger), backend(backend) {
        ctx = ggml_init({64 * 1024, nullptr, true}); require(ctx != nullptr, "oracle metadata");
        try {
            const auto type = ggml_backend_get_default_buffer_type(backend);
            for (int p = 0; p < 3; ++p) {
                weights[p] = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, table.weights[p]->ne[0], table.weights[p]->ne[1], table.capacity);
                ggml_set_input(weights[p]); ggml_set_output(weights[p]);
                const auto bytes = ggml_backend_buft_get_alloc_size(type, weights[p]);
                require(ledger.reserve(bytes), "oracle prephysical ticket"); charges[p] = bytes;
                buffers[p] = ggml_backend_buft_alloc_buffer(type, bytes); require(buffers[p] != nullptr, "ordinary CUDA oracle buffer");
                ledger.commit(bytes);
                ggml_backend_buffer_set_usage(buffers[p], GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                require(ggml_backend_tensor_alloc(buffers[p], weights[p], ggml_backend_buffer_get_base(buffers[p])) == GGML_STATUS_SUCCESS, "oracle tensor bind");
                for (int slot = 0; slot < table.capacity; ++slot)
                    ggml_backend_tensor_set(weights[p], raw[p].data() + table.source_ids[slot] * weights[p]->nb[2], slot * weights[p]->nb[2], weights[p]->nb[2]);
            }
        } catch (...) { close(); throw; }
    }
    ~StandardWeights() { close(); }
    void close() noexcept {
        ggml_backend_synchronize(backend);
        for (int p = 0; p < 3; ++p) if (charges[p]) {
            const bool committed = buffers[p] != nullptr;
            ggml_backend_buffer_free(buffers[p]); buffers[p] = nullptr;
            try { ledger.release(charges[p], committed); } catch (...) { callback_failed = true; }
            charges[p] = 0;
        }
        if (ctx) { ggml_free(ctx); ctx = nullptr; }
    }
};

struct Graph {
    Ledger& ledger;
    ggml_backend_t backend;
    ggml_context* ctx = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_tensor *input = nullptr, *ids = nullptr, *routes = nullptr, *output = nullptr;
    int rows;
    std::size_t charge = 0;
    bool committed = false;
    Graph(Ledger& ledger, ggml_backend_t backend, const std::array<ggml_tensor*, 3>& weights, int rows)
        : ledger(ledger), backend(backend), rows(rows) {
        try {
            ctx = ggml_init({2 * mib, nullptr, true}); require(ctx != nullptr, "graph metadata");
            graph = ggml_new_graph_custom(ctx, 128, false); refresh();
            input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden, 1, rows);
            ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, selected, rows);
            routes = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, selected, rows);
            ggml_set_input(input); ggml_set_input(ids); ggml_set_input(routes);
            auto* gate = ggml_mul_mat_id(ctx, weights[0], input, ids);
            auto* up = ggml_mul_mat_id(ctx, weights[1], input, ids);
            auto* activation = ggml_mul(ctx, ggml_silu(ctx, gate), up);
            auto* down = ggml_mul_mat_id(ctx, weights[2], activation, ids);
            auto* weighted = ggml_mul(ctx, down, routes);
            auto* sum = ggml_view_2d(ctx, weighted, hidden, rows, weighted->nb[2], 0);
            for (int k = 1; k < selected; ++k) sum = ggml_add(ctx, sum,
                ggml_view_2d(ctx, weighted, hidden, rows, weighted->nb[2], k * weighted->nb[1]));
            output = ggml_cont(ctx, sum); ggml_set_output(output); ggml_build_forward_expand(graph, output);
            allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)); require(allocator != nullptr, "graph allocator");
            std::size_t bytes = 0; ggml_gallocr_reserve_n_size(allocator, graph, nullptr, nullptr, &bytes);
            require(bytes && ledger.reserve(bytes), "scratch prephysical ticket"); charge = bytes;
            require(ggml_gallocr_reserve(allocator, graph), "scratch allocation");
            require(ggml_gallocr_get_buffer_size(allocator, 0) == bytes && ggml_gallocr_alloc_graph(allocator, graph), "scratch exact plan");
            ledger.commit(bytes); committed = true;
        } catch (...) { close(); throw; }
    }
    ~Graph() { close(); }
    void refresh() noexcept { graph->uid = next_uid++; }
    void close() noexcept {
        if (backend) ggml_backend_synchronize(backend);
        if (allocator) { ggml_gallocr_free(allocator); allocator = nullptr; }
        if (charge) { try { ledger.release(charge, committed); } catch (...) { callback_failed = true; } charge = 0; }
        if (ctx) { ggml_free(ctx); ctx = nullptr; }
    }
    std::vector<float> run(int capacity, const char* phase) {
        stage(phase, capacity, rows, graph->uid);
        std::vector<float> x(hidden * rows), probabilities(selected * rows), result(hidden * rows);
        std::vector<std::int32_t> indices(selected * rows);
        for (int r = 0; r < rows; ++r) {
            for (int i = 0; i < hidden; ++i) x[r * hidden + i] = .2f * std::cos(.019f * i + .031f * r);
            for (int k = 0; k < selected; ++k) { indices[r * selected + k] = k ? capacity - 1 : 2; probabilities[r * selected + k] = k ? .375f : .625f; }
        }
        ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
        ggml_backend_tensor_set(ids, indices.data(), 0, indices.size() * sizeof(std::int32_t));
        ggml_backend_tensor_set(routes, probabilities.data(), 0, probabilities.size() * sizeof(float));
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "ordinary CUDA graph compute");
        stage("graph-computed-before-output-get", capacity, rows, graph->uid);
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
        require(!callback_failed, "owned callback failed"); return result;
    }
};

bool retry_quarantine() noexcept {
    bool okay = true;
    for (auto& held : quarantine) if (held) {
        if (held->close()) { delete held; held = nullptr; }
        else okay = false;
    }
    return okay;
}

void exercise(ggml_backend_t backend, std::size_t granularity, Ledger& ledger) {
    std::array<std::vector<std::uint8_t>, 3> raw;
    const auto row_bytes = ggml_row_size(GGML_TYPE_Q8_0, hidden);
    const auto stride = row_bytes * intermediate;
    std::vector<float> row(hidden);
    for (int p = 0; p < 3; ++p) {
        raw[p].resize(stride * maximum_slots);
        for (int slot = 0; slot < maximum_slots; ++slot) for (int r = 0; r < intermediate; ++r) {
            for (int c = 0; c < hidden; ++c) row[c] = .025f * std::sin(.071f * c + .037f * r + .13f * slot + .23f * p);
            require(ggml_quantize_chunk(GGML_TYPE_Q8_0, row.data(), raw[p].data() + slot * stride + r * row_bytes,
                0, 1, hidden, nullptr) == row_bytes, "Q8 row quantization");
        }
    }
    Table table(ledger, backend, granularity);
    auto fill = [&](int first, int end) {
        for (int p = 0; p < 3; ++p) for (int slot = first; slot < end; ++slot)
            ggml_backend_tensor_set_async(backend, table.weights[p], raw[p].data() + slot * stride, slot * stride, stride);
        ggml_backend_synchronize(backend);
        for (int slot = first; slot < end; ++slot) table.source_ids[slot] = slot;
    };
    auto verify = [&] {
        stage("verify-begin", table.capacity);
        std::vector<std::uint8_t> scratch(stride); // One bounded expert, never a full tensor staging copy.
        for (int p = 0; p < 3; ++p) {
            std::fprintf(stderr, "[VMM-PROBE-VERIFY] part=%d capacity=%d base=%llu mapped_prefix=%zu\n", p, table.capacity,
                static_cast<unsigned long long>(table.projections[p]->base), table.projections[p]->logical_bytes); std::fflush(stderr);
            for (int slot = 0; slot < table.capacity; ++slot) {
                if (p == 0 && slot == 0)
                    ggml_backend_tensor_get(table.weights[p], scratch.data(), slot * stride, stride); // Owned synchronous callback.
                else ggml_backend_tensor_get_async(backend, table.weights[p], scratch.data(), slot * stride, stride);
                ggml_backend_synchronize(backend);
                require(std::memcmp(scratch.data(), raw[p].data() + table.source_ids[slot] * stride, stride) == 0, "VMM raw bytes changed");
            }
        }
        stage("verify-complete", table.capacity);
    };
    fill(0, initial_slots); verify();
    Graph narrow(ledger, backend, table.weights, 1), wide(ledger, backend, table.weights, 9);
    std::uint64_t compared = 0;
    auto compare = [&] {
        stage("compare-begin", table.capacity);
        StandardWeights oracle(ledger, backend, table, raw);
        const auto before_copy = owned_copy_calls;
        std::vector<std::uint8_t> poisoned(stride);
        ggml_backend_tensor_memset(table.weights[0], 0xa5, 2 * stride, stride);
        ggml_backend_tensor_get(table.weights[0], poisoned.data(), 2 * stride, stride);
        require(!callback_failed && std::all_of(poisoned.begin(), poisoned.end(), [](std::uint8_t byte) { return byte == 0xa5; })
            && std::memcmp(poisoned.data(), raw[0].data() + table.source_ids[2] * stride, stride) != 0,
            "owned copy destination was not actually poisoned");
        // Directly invoke the supported owned-destination copy callback. Never
        // call generic tensor_copy[_async], whose stock-destination fallback
        // would allocate a full unaccounted Host staging tensor.
        require(ggml_backend_buffer_copy_tensor(oracle.weights[0], table.weights[0])
            && owned_copy_calls == before_copy + 1 && !callback_failed, "owned same-device copy callback");
        verify(); // A no-op or partial owned callback cannot satisfy parity.
        for (auto* retained : {&narrow, &wide}) {
            Graph standard(ledger, backend, oracle.weights, retained->rows);
            const auto expected = standard.run(table.capacity, "oracle-graph"), actual = retained->run(table.capacity, "retained-graph-first");
            require(std::all_of(expected.begin(), expected.end(), [](float v) { return std::isfinite(v); }), "nonfinite oracle");
            require(std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float)) == 0, "VMM changed original MMID/SwiGLU/Down F32 bits");
            // At least three submissions at each geometry cover warmup,
            // capture and a following reuse when upstream enables capture.
            for (int repeat = 0; repeat < 2; ++repeat) {
                const auto repeated = retained->run(table.capacity, repeat ? "retained-graph-third" : "retained-graph-second");
                require(std::memcmp(actual.data(), repeated.data(), actual.size() * sizeof(float)) == 0, "VMM graph reuse changed bits");
            }
            compared += actual.size() * 3;
        }
        stage("compare-complete", table.capacity);
    };
    compare();
    const auto before_bytes = ledger.committed, before_pending = ledger.pending;
    const auto before_narrow_uid = narrow.graph->uid, before_wide_uid = wide.graph->uid;
    const auto original_limit = ledger.limit;
    ledger.limit = ledger.committed;
    stage("prephysical-refusal-test", table.capacity);
    require(!table.grow(grown_slots) && table.last_created == 0, "prephysical ticket refusal allocated a chunk");
    ledger.limit = original_limit;
    stage("partial-growth-refusal-test", table.capacity);
    require(!table.grow(grown_slots, 2), "injected partial growth did not refuse");
    require(table.last_created == 2, "partial-growth fault was not reached after two real physical allocations");
    require(table.capacity == initial_slots && ledger.committed == before_bytes && ledger.pending == before_pending
        && narrow.graph->uid == before_narrow_uid && wide.graph->uid == before_wide_uid, "refusal published geometry or leaked physical ticket");
    stage("partial-growth-rollback-complete", table.capacity);
    verify(); compare();
    const auto initial_physical = table.physical_bytes();
    require(table.grow(grown_slots), "VMM delta growth"); fill(initial_slots, grown_slots);
    const auto grown_physical = table.physical_bytes(); require(grown_physical > initial_physical, "growth mapped no extra physical pages");
    narrow.refresh(); wide.refresh(); // Required even though all tensor/data addresses remain stable.
    verify(); compare();
    cudaStream_t compact = nullptr; runtime(cudaStreamCreateWithFlags(&compact, cudaStreamNonBlocking), "compaction stream");
    try { table.copy_slot(23, 2, compact); runtime(cudaStreamDestroy(compact), "compaction stream free"); compact = nullptr; }
    catch (...) { if (compact) cudaStreamDestroy(compact); throw; }
    const auto before_shrink = ledger.committed;
    table.shrink(shrunk_slots);
    const auto shrunk_physical = table.physical_bytes();
    require(shrunk_physical < grown_physical && ledger.committed == before_shrink - (grown_physical - shrunk_physical), "tail unmap did not refund exact physical chunks");
    narrow.refresh(); wide.refresh(); verify(); compare();
    std::printf("PASS VMM payload prototype initial=%d grown=%d shrunk=%d rows=1,9 bit_comparisons=%llu granule=%zu chunk=%zu "
        "raw_host_source_bytes=%zu declared_payload_peak=%zu vmm_physical_initial=%zu vmm_physical_grown=%zu vmm_physical_shrunk=%zu "
        "owned_d2d_callback_calls=%llu rollback_after_real_creates=2 tail_compaction=1 no_full_tensor_host_staging=1 "
        "scope=owned_weight_payload_and_ordinary_GGML_math_not_total_VRAM_or_GLM_performance\n",
        initial_slots, grown_slots, shrunk_slots, static_cast<unsigned long long>(compared), granularity,
        table.projections[0]->chunk_bytes, raw[0].size() * 3, ledger.peak, initial_physical, grown_physical, shrunk_physical,
        static_cast<unsigned long long>(owned_copy_calls));
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2 || std::strcmp(argv[1], "--execute") != 0) {
        std::fputs("Usage: cuda-vmm-expert-table-probe --execute (GPU work; exclusive slot required). No CUDA initialization performed.\n", stderr); return 2;
    }
    ggml_backend_t backend = nullptr;
    Ledger ledger;
    int outcome = 1;
    try {
        if (const auto* emulation = std::getenv("GGML_CUDA_DEVICES"); emulation && *emulation) {
            std::puts("UNAVAILABLE: virtual CUDA device mapping is outside this prototype"); return 77;
        }
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count != 1) { std::puts("UNAVAILABLE: requires one real CUDA device"); return 77; }
        runtime(cudaSetDevice(0), "select CUDA device"); driver(cuInit(0), "cuInit");
        int supported = 0;
        driver(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, 0), "query VMM capability");
        if (!supported) { std::puts("UNAVAILABLE: CUDA VMM is unsupported"); return 77; }
        CUmemAllocationProp prop{}; prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id = 0;
        std::size_t granularity = 0;
        driver(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM), "query minimum VMM granularity");
        if (!granularity || granularity > 2 * mib || (granularity & (granularity - 1)) != 0) {
            std::puts("UNAVAILABLE: VMM granularity is outside this bounded prototype"); return 77;
        }
        backend = ggml_backend_cuda_init(0); require(backend != nullptr, "ggml CUDA backend unavailable after positive capability");
        require(ggml_backend_cuda_get_device_count() == 1, "ggml device topology differs from CUDA runtime");
        const auto type = ggml_backend_get_default_buffer_type(backend);
        require(type == ggml_backend_cuda_buffer_type(0) && ggml_backend_supports_buft(backend, type), "exact standard CUDA buft handshake");
        exercise(backend, granularity, ledger);
        require(retry_quarantine(), "VMM cleanup remains quarantined");
        require(!callback_failed && ledger.pending == 0 && ledger.committed == 0, "owned callback or physical ledger did not clean up");
        std::puts("CLEANUP pending=0 committed=0 quarantined=0; upstream CUDA workspace/context excluded"); outcome = 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); }
    if (backend) { ggml_backend_synchronize(backend); retry_quarantine(); ggml_backend_free(backend); }
    return outcome;
}
