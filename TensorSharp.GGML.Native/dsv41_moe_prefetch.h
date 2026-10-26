// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include "ggml_ops_internal.h"

namespace tsg_dsv41_prefetch {

inline bool enabled(const char * value) {
    if (!value || !*value || std::strcmp(value,"0")==0) return false;
    if (std::strcmp(value,"1")==0) return true;
    throw std::runtime_error("TS_DSV41_CPU_MOE_PREFETCH must be 0 or 1");
}

inline void fail(std::string & message, bool & failed, const char * reason) noexcept {
    failed = true;
    try { if (message.empty()) message = reason; } catch (...) { /* failed remains set even without memory */ }
}

// Only selected expert ranges are touched, on the existing bounded reader
// pool. This helper parallelizes mmap page faults; it does not create a weight
// copy or change the selected IDs, their ordering, or the arithmetic.
inline void selected(ggml_tensor * const * tensors, int experts, const int32_t * ids, size_t count) {
    if (!ids || !count || experts < 1) throw std::runtime_error("Invalid CPU expert prefetch inputs");
    std::vector<int32_t> unique(ids,ids+count);
    for (int32_t id : unique)
        if (id < 0 || id >= experts) throw std::runtime_error("CPU expert prefetch ID is out of bounds");
    std::sort(unique.begin(),unique.end());
    unique.erase(std::unique(unique.begin(),unique.end()),unique.end());
    for (int part = 0; part < 3; ++part) {
        const auto * tensor = tensors[part];
        if (!tensor || !tensor->data || !tensor->buffer || !ggml_backend_buffer_is_host(tensor->buffer) ||
            tensor->ne[2] != experts || tensor->nb[2] == 0 ||
            size_t(experts) > SIZE_MAX/tensor->nb[2] || ggml_nbytes(tensor) != size_t(experts)*tensor->nb[2])
            throw std::runtime_error("CPU expert prefetch requires complete host-resident expert stacks");
        std::vector<std::pair<size_t,size_t>> ranges;
        for (int32_t id : unique) {
            const size_t offset = size_t(id)*tensor->nb[2], bytes = tensor->nb[2];
            if (!ranges.empty() && ranges.back().first+ranges.back().second == offset) ranges.back().second += bytes;
            else ranges.emplace_back(offset,bytes);
        }
        tsg::prefetch_mapped_ranges(static_cast<const uint8_t *>(tensor->data),ranges);
    }
}

class node {
    ggml_tensor * tensors[3];
    int experts, used;
    std::string & error;
    bool & failed;
    static void callback(ggml_tensor * dst, int ith, int, void * opaque) noexcept {
        if (ith) return;
        auto & self = *static_cast<node *>(opaque);
        try {
            const auto * source = dst->src[0];
            const int64_t count = ggml_nelements(dst);
            if (!source || source->type != GGML_TYPE_I32 || !source->data || !ggml_is_contiguous(source) ||
                count <= 0 || count > self.used*8 || count != ggml_nelements(source))
                throw std::runtime_error("Invalid CPU expert prefetch ID tensor");
            selected(self.tensors,self.experts,static_cast<const int32_t *>(source->data),size_t(count));
            std::memcpy(dst->data,source->data,size_t(count)*sizeof(int32_t));
        } catch (const std::exception & problem) {
            fail(self.error,self.failed,problem.what());
            std::memset(dst->data,0,ggml_nbytes(dst)); // valid safe IDs until the caller rejects the failed graph
        } catch (...) {
            fail(self.error,self.failed,"CPU expert prefetch callback failed");
            std::memset(dst->data,0,ggml_nbytes(dst));
        }
    }
public:
    node(ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down, int total, int selected_count,
         std::string & message, bool & failure)
        : tensors{gate,up,down},experts(total),used(selected_count),error(message),failed(failure) {}
    ggml_tensor * build(ggml_context * ctx, ggml_tensor * ids) {
        ggml_tensor * sources[] = {ids};
        auto * out = ggml_custom_4d(ctx,GGML_TYPE_I32,ids->ne[0],ids->ne[1],ids->ne[2],ids->ne[3],
            sources,1,callback,1,this);
        ggml_set_name(out,"dsv41_prefetched_expert_ids");
        return out;
    }
};

} // namespace tsg_dsv41_prefetch
