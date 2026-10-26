// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include "ggml_ops_internal.h"
#include "dsv41_moe_prefetch.h"
#include "ggml_ops_dsv4_fused.h"
#if defined(TSG_GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif

namespace tsg_dsv41_cache {

class cpu_lane {
    ggml_backend_t backend = nullptr;
    ggml_threadpool_t pool = nullptr;
    int threads;
public:
    explicit cpu_lane(int count) : threads(count) {}
    ggml_backend_t get() {
        if (backend) return backend;
        backend = ggml_backend_cpu_init();
        if (!backend) throw std::runtime_error("DeepSeek cache CPU fallback backend allocation failed");
        ggml_backend_cpu_set_n_threads(backend,threads);
        auto params = ggml_threadpool_params_default(threads);
        pool = ggml_threadpool_new(&params);
        if (!pool) {
            ggml_backend_free(backend);backend=nullptr;
            throw std::runtime_error("DeepSeek cache CPU fallback worker allocation failed");
        }
        ggml_backend_cpu_set_threadpool(backend,pool);
        return backend;
    }
    ~cpu_lane() {
        if (backend) ggml_backend_free(backend);
        if (pool) ggml_threadpool_free(pool);
    }
};

// A CPU scheduler node owns this bridge. Its parent CPU backend is already
// executing the callback, so fallback uses an independent backend/thread pool;
// recursively submitting to the parent's pool would deadlock its workers.
class state {
    tsg::HostMoeSegment segment;
    ggml_backend_t device;
    size_t budget, layer_count;
    std::shared_ptr<cpu_lane> fallback_lane;
    std::string * failure;
    bool * failed;
    bool cpu_prefetch;
    ggml_tensor * source_weights[3];
    ggml_backend_t cpu = nullptr;
    ggml_context * context = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr, * ids = nullptr, * routes = nullptr, * output = nullptr;
    uint64_t cached_calls = 0, fallback_calls = 0;
    bool fallback_ready = false;
    tsg_dsv4_fused_desc device_descriptor;
#if defined(TSG_GGML_USE_CUDA)
    ggml_context * device_context = nullptr;
    ggml_backend_buffer_t device_buffer = nullptr;
    ggml_tensor * finite_flag = nullptr;
    std::vector<int32_t> selected_host;
    std::vector<float> fallback_input, fallback_routes, fallback_output;
#endif

    void clear_fallback() noexcept {
        if (cpu) ggml_backend_synchronize(cpu);
        if (allocator) ggml_gallocr_free(allocator);
        if (context) ggml_free(context);
        cpu = nullptr; context = nullptr; allocator = nullptr; graph = nullptr;
        input = ids = routes = output = nullptr;
        fallback_ready = false;
    }

    void make_fallback() {
        if (fallback_ready) return;
        clear_fallback();
        cpu = fallback_lane->get();
        const size_t bytes = 128 * ggml_tensor_overhead() + ggml_graph_overhead_custom(256,false);
        context = ggml_init({bytes,nullptr,true});
        if (!context) throw std::runtime_error("DeepSeek cache CPU fallback metadata allocation failed");
        input = ggml_new_tensor_3d(context,GGML_TYPE_F32,segment.hidden,1,segment.seq_len);
        ids = ggml_new_tensor_2d(context,GGML_TYPE_I32,segment.n_used,segment.seq_len);
        routes = ggml_new_tensor_3d(context,GGML_TYPE_F32,1,segment.n_used,segment.seq_len);
        for (auto * tensor : {input,ids,routes}) ggml_set_input(tensor);
        auto * gate = ggml_mul_mat_id(context,source_weights[0],input,ids);
        auto * up = ggml_mul_mat_id(context,source_weights[1],input,ids);
        if (segment.clamp_limit > 1e-6f) {
            gate = ggml_clamp(context,gate,-INFINITY,segment.clamp_limit);
            up = ggml_clamp(context,up,-segment.clamp_limit,segment.clamp_limit);
        }
        auto * activation = ggml_swiglu_split(context,gate,up);
        auto * down = ggml_mul_mat_id(context,source_weights[2],activation,ids);
        auto * weighted = ggml_mul(context,down,routes);
        auto * sum = ggml_view_2d(context,weighted,segment.hidden,segment.seq_len,weighted->nb[2],0);
        for (int k = 1; k < segment.n_used; ++k)
            sum = ggml_add(context,sum,ggml_view_2d(context,weighted,segment.hidden,segment.seq_len,
                weighted->nb[2],size_t(k)*weighted->nb[1]));
        output = ggml_cont(context,sum);
        ggml_set_output(output);
        graph = ggml_new_graph_custom(context,256,false);
        ggml_build_forward_expand(graph,output);
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
        if (!allocator || !ggml_gallocr_alloc_graph(allocator,graph))
            throw std::runtime_error("DeepSeek cache CPU fallback graph allocation failed");
        fallback_ready = true;
    }

    void validate(ggml_tensor * destination) const {
        if (!destination || destination->type != GGML_TYPE_F32 || !ggml_is_contiguous(destination) || !destination->data)
            throw std::runtime_error("DeepSeek expert cache destination is invalid");
        for (int i = 0; i < 3; ++i)
            if (!destination->src[i] || !ggml_is_contiguous(destination->src[i]) || !destination->src[i]->data)
                throw std::runtime_error("DeepSeek expert cache requires contiguous inputs");
        if (destination->src[0]->type != GGML_TYPE_F32 || destination->src[1]->type != GGML_TYPE_I32 ||
            destination->src[2]->type != GGML_TYPE_F32)
            throw std::runtime_error("DeepSeek expert cache input types are invalid");
        if (ggml_nelements(destination->src[0]) != (int64_t)segment.seq_len*segment.hidden ||
            ggml_nelements(destination->src[1]) != (int64_t)segment.seq_len*segment.n_used ||
            ggml_nelements(destination->src[2]) != (int64_t)segment.seq_len*segment.n_used ||
            ggml_nelements(destination) != (int64_t)segment.seq_len*segment.hidden)
            throw std::runtime_error("DeepSeek expert cache input shapes are invalid");
    }

    void run_fallback(const float * x, const int32_t * selected, const float * weights, float * result) {
        const size_t row_values = size_t(segment.seq_len)*segment.hidden;
        if (cpu_prefetch) tsg_dsv41_prefetch::selected(source_weights,segment.num_experts,selected,
            size_t(segment.seq_len)*segment.n_used);
        make_fallback();
        ggml_backend_tensor_set(input,x,0,row_values*sizeof(float));
        ggml_backend_tensor_set(ids,selected,0,size_t(segment.seq_len)*segment.n_used*sizeof(int32_t));
        ggml_backend_tensor_set(routes,weights,0,size_t(segment.seq_len)*segment.n_used*sizeof(float));
        if (tsg::compute_graph(cpu,graph) != GGML_STATUS_SUCCESS ||
            tsg::g_backend_compute_failed.load(std::memory_order_acquire))
            throw std::runtime_error("DeepSeek expert cache CPU fallback graph failed");
        ggml_backend_tensor_get(output,result,0,row_values*sizeof(float));
        if (!fallback_calls++) std::fprintf(stderr,
            "[dsv41-cache] layer=%d rows=%d GPU cache declined; complete CPU fallback\n",segment.layer,segment.seq_len);
    }

    void run(ggml_tensor * destination) {
        validate(destination);
        const auto * x = static_cast<const float *>(destination->src[0]->data);
        const auto * selected = static_cast<const int32_t *>(destination->src[1]->data);
        const auto * weights = static_cast<const float *>(destination->src[2]->data);
        auto * result = static_cast<float *>(destination->data);
        const size_t row_values = size_t(segment.seq_len)*segment.hidden;
        for (size_t i = 0; i < row_values; ++i)
            if (!std::isfinite(x[i])) throw std::runtime_error("DeepSeek expert cache input is nonfinite");
        for (int i = 0; i < segment.seq_len*segment.n_used; ++i)
            if (selected[i] < 0 || selected[i] >= segment.num_experts || !std::isfinite(weights[i]))
                throw std::runtime_error("DeepSeek expert cache route is invalid");
        const int status = tsg::host_moe_cached_experts(segment,x,selected,weights,result,
            "deepseek41 native",device,budget,layer_count);
        if (status < 0) throw std::runtime_error(tsg::g_last_error.empty()
            ? "DeepSeek expert cache GPU graph or source read failed" : tsg::g_last_error);
        if (status == 0) {
            run_fallback(x,selected,weights,result);
        } else {
            if (!cached_calls++) std::fprintf(stderr,
                "[dsv41-cache] layer=%d rows=%d complete GPU expert graph active\n",segment.layer,segment.seq_len);
        }
        for (size_t i = 0; i < row_values; ++i)
            if (!std::isfinite(result[i])) throw std::runtime_error("DeepSeek expert cache result is nonfinite");
    }

#if defined(TSG_GGML_USE_CUDA)
    static ggml_backend_buffer_t storage(const ggml_tensor * tensor) {
        while (tensor && tensor->view_src) tensor = tensor->view_src;
        return tensor ? tensor->buffer : nullptr;
    }

    void make_device_scratch() {
        selected_host.resize(size_t(segment.seq_len)*segment.n_used);
        if (device_buffer) return;
        if (device_context) { ggml_free(device_context); device_context = nullptr; }
        device_context = ggml_init({ggml_tensor_overhead(),nullptr,true});
        if (!device_context) throw std::runtime_error("DeepSeek expert finite-status metadata allocation failed");
        finite_flag = ggml_new_tensor_1d(device_context,GGML_TYPE_I32,1);
        device_buffer = ggml_backend_alloc_ctx_tensors(device_context,device);
        if (!device_buffer) throw std::runtime_error("DeepSeek expert finite-status device allocation failed");
    }

    void run_device(ggml_tensor * destination) {
        validate(destination);
        if (!device || !ggml_backend_is_cuda(device))
            throw std::runtime_error("DeepSeek expert device bridge requires its owning CUDA backend");
        const auto buft = ggml_backend_get_default_buffer_type(device);
        for (const auto * tensor : {destination->src[0],destination->src[1],destination->src[2],destination}) {
            auto buffer = storage(tensor);
            if (!buffer || ggml_backend_buffer_get_type(buffer) != buft)
                throw std::runtime_error("DeepSeek expert device bridge tensors belong to a different backend");
        }
        make_device_scratch();
        int32_t nonfinite = 0;
        tsg_dsv4_cuda_expert_finite_scan(device,destination->src[0],destination->src[2],finite_flag);
        ggml_backend_tensor_get_async(device,destination->src[1],selected_host.data(),0,selected_host.size()*sizeof(int32_t));
        ggml_backend_tensor_get_async(device,finite_flag,&nonfinite,0,sizeof(nonfinite));
        tsg::sync_backend(device);
        if (tsg::g_backend_compute_failed.load(std::memory_order_acquire))
            throw std::runtime_error("DeepSeek expert device input transfer failed");
        if (nonfinite) throw std::runtime_error("DeepSeek expert device input or route is nonfinite");
        for (int32_t id : selected_host)
            if (id < 0 || id >= segment.num_experts)
                throw std::runtime_error("DeepSeek expert device route is invalid");
        auto descriptor = segment;
        descriptor.moe_in = destination->src[0];
        descriptor.weights = destination->src[2];
        descriptor.moe_out = destination;
        const int status = tsg::host_moe_cached_experts(descriptor,nullptr,selected_host.data(),nullptr,nullptr,
            "deepseek41 native",device,budget,layer_count);
        if (status < 0) throw std::runtime_error(tsg::g_last_error.empty()
            ? "DeepSeek expert device cache graph or source read failed" : tsg::g_last_error);
        if (status == 0) {
            // A decline replays the whole span on the independent CPU lane.
            // No partially computed GPU experts are mixed into that result.
            const size_t row_values = size_t(segment.seq_len)*segment.hidden;
            fallback_input.resize(row_values); fallback_routes.resize(selected_host.size()); fallback_output.resize(row_values);
            ggml_backend_tensor_get_async(device,destination->src[0],fallback_input.data(),0,row_values*sizeof(float));
            ggml_backend_tensor_get_async(device,destination->src[2],fallback_routes.data(),0,fallback_routes.size()*sizeof(float));
            tsg::sync_backend(device);
            if (tsg::g_backend_compute_failed.load(std::memory_order_acquire))
                throw std::runtime_error("DeepSeek expert device CPU-fallback transfer failed");
            run_fallback(fallback_input.data(),selected_host.data(),fallback_routes.data(),fallback_output.data());
            ggml_backend_tensor_set_async(device,destination,fallback_output.data(),0,row_values*sizeof(float));
        } else if (!cached_calls++) {
            std::fprintf(stderr,"[dsv41-cache] layer=%d rows=%d complete GPU expert graph active; device bridge\n",
                segment.layer,segment.seq_len);
        }
        tsg_dsv4_cuda_expert_finite_scan(device,destination,nullptr,finite_flag);
        ggml_backend_tensor_get_async(device,finite_flag,&nonfinite,0,sizeof(nonfinite));
        // Drain the helper's queued final D2D copy (or full fallback upload)
        // before accepting this span or reusing any graph/source buffers.
        tsg::sync_backend(device);
        if (tsg::g_backend_compute_failed.load(std::memory_order_acquire) || nonfinite)
            throw std::runtime_error("DeepSeek expert device result is nonfinite or its transfer failed");
    }

    static void device_callback(void * opaque, ggml_tensor * destination, void *, int) noexcept {
        auto & self = *static_cast<state *>(opaque);
        try { self.run_device(destination); }
        catch (const std::exception & error) {
            tsg_dsv41_prefetch::fail(*self.failure,*self.failed,error.what());
            tsg_dsv4_cuda_expert_poison(self.device,destination);
        }
        catch (...) {
            tsg_dsv41_prefetch::fail(*self.failure,*self.failed,"DeepSeek expert device cache callback failed");
            tsg_dsv4_cuda_expert_poison(self.device,destination);
        }
    }
#endif

    static void host_callback(void * opaque, ggml_tensor * destination) noexcept {
        callback(destination,0,1,opaque);
    }

    static void callback(ggml_tensor * destination, int ith, int, void * opaque) noexcept {
        if (ith) return;
        auto & self = *static_cast<state *>(opaque);
        try { self.run(destination); }
        catch (const std::exception & error) {
            tsg_dsv41_prefetch::fail(*self.failure,*self.failed,error.what());
            std::fill_n(static_cast<float *>(destination->data),size_t(ggml_nelements(destination)),
                std::numeric_limits<float>::quiet_NaN());
        }
        catch (...) {
            tsg_dsv41_prefetch::fail(*self.failure,*self.failed,"DeepSeek expert cache callback failed");
            std::fill_n(static_cast<float *>(destination->data),size_t(ggml_nelements(destination)),
                std::numeric_limits<float>::quiet_NaN());
        }
    }

public:
    state(tsg::HostMoeSegment descriptor, ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down,
          ggml_backend_t backend, size_t bytes, size_t layers, int threads, std::string & error, bool & failure_flag,
          std::shared_ptr<cpu_lane> lane = nullptr, bool prefetch = false)
        : segment(descriptor),device(backend),budget(bytes),layer_count(layers),
          fallback_lane(lane ? std::move(lane) : std::make_shared<cpu_lane>(threads)),failure(&error),failed(&failure_flag),cpu_prefetch(prefetch),
          source_weights{gate,up,down} {}
    ~state() {
        clear_fallback();
#if defined(TSG_GGML_USE_CUDA)
        if (device_buffer) { ggml_backend_synchronize(device); ggml_backend_buffer_free(device_buffer); }
        if (device_context) ggml_free(device_context);
#endif
    }
    ggml_tensor * build(ggml_context * ctx, ggml_tensor * x, ggml_tensor * selected, ggml_tensor * weights,
                        bool device_bridge = false) {
        ggml_tensor * sources[] = {x,selected,weights};
        if (device_bridge) {
#if defined(TSG_GGML_USE_CUDA)
            device_descriptor.kind = TSG_DSV41_CACHED_MOE;
            device_descriptor.object = this;
            device_descriptor.host_compute = host_callback;
            device_descriptor.device_launch = device_callback;
            auto * result = ggml_custom_4d(ctx,GGML_TYPE_F32,segment.hidden,segment.seq_len,1,1,
                sources,3,tsg_dsv4_fused_cpu,1,&device_descriptor);
            ggml_format_name(result,"dsv41_device_cached_experts.%d",segment.layer);
            return result;
#else
            throw std::runtime_error("DeepSeek expert device bridge requires a CUDA-enabled native build");
#endif
        }
        auto * result = ggml_custom_4d(ctx,GGML_TYPE_F32,segment.hidden,segment.seq_len,1,1,
            sources,3,callback,1,this);
        ggml_format_name(result,"dsv41_cached_experts.%d",segment.layer);
        return result;
    }
};

inline size_t configured_bytes(const char * value, const char * name = "TS_DSV41_EXPERT_CACHE_MB") {
    if (!value || !*value) return 0;
    size_t megabytes = 0;
    for (const unsigned char * p = reinterpret_cast<const unsigned char *>(value); *p; ++p) {
        if (*p < '0' || *p > '9' || megabytes > (SIZE_MAX/1048576 - (*p-'0'))/10)
            throw std::runtime_error(std::string(name)+" must be an unsigned MiB count");
        megabytes = megabytes*10 + (*p-'0');
    }
    return megabytes*1048576;
}

inline bool configured_device_bridge(const char * value) {
    if (!value || !*value || std::strcmp(value,"0")==0) return false;
    if (std::strcmp(value,"1")==0) return true;
    throw std::runtime_error("TS_DSV41_EXPERT_CACHE_DEVICE_BRIDGE must be 0 or 1");
}

} // namespace tsg_dsv41_cache
