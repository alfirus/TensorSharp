// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Validation only: the SiLU implementation is linked unchanged from ggml.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "gguf.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// The declaration matches unchanged src/ggml-cpu/vec.h. Do not duplicate its
// architecture-specific vector exponential or replace it with scalar SiLU.
extern "C" void ggml_vec_silu_f32(int n, float * y, const float * x);
namespace {
using clock_type = std::chrono::steady_clock;
constexpr int width = 4096, tile = 64, chains = 24;
constexpr const char * revision = "3612c0af93f2cfc8f048295300a61187723b12ae";
void require(bool b, const char * m) { if (!b) throw std::runtime_error(m); }
std::string quote(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '\\' || c == '"') { out += '\\'; out += char(c); }
        else if (c < 32) { char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", c); out += b; }
        else out += char(c);
    }
    return out + '"';
}
uint32_t bits(float x) { uint32_t v; std::memcpy(&v, &x, 4); return v; }
float from_bits(uint32_t x) { float v; std::memcpy(&v, &x, 4); return v; }
// Exactly MAX(MIN(x,max),min), including NaN and signed-zero branch choices.
float clip(float x, float lo, float hi) { const float a = x < hi ? x : hi; return a > lo ? a : lo; }
struct bounds { std::string up, down; float lo, hi, after_lo, after_hi; };
struct checkpoint {
    gguf_context * metadata = nullptr;
    std::ifstream input;
    explicit checkpoint(const std::string& path) : input(path, std::ios::binary) {
        metadata = gguf_init_from_file(path.c_str(), {true, nullptr});
        if (!metadata || !input) { if (metadata) gguf_free(metadata); metadata = nullptr; throw std::runtime_error("Cannot read projector GGUF"); }
    }
    ~checkpoint() { if (metadata) gguf_free(metadata); }
    float scalar(const std::string& name) {
        const auto id = gguf_find_tensor(metadata, name.c_str());
        require(id >= 0, "Required learned clamp tensor missing");
        const auto * ne = gguf_get_tensor_ne(metadata, id);
        require(gguf_get_tensor_type(metadata, id) == GGML_TYPE_F32 && gguf_get_tensor_size(metadata, id) == 4 &&
            ne[0] == 1 && ne[1] == 1 && ne[2] == 1 && ne[3] == 1, "Require exact learned F32 scalar clamp");
        float v = 0; input.seekg(std::streamoff(gguf_get_data_offset(metadata) + gguf_get_tensor_offset(metadata, id)));
        require(bool(input.read(reinterpret_cast<char *>(&v), 4)), "Truncated learned clamp scalar");
        require(std::isfinite(v), "Nonfinite learned clamp"); return v;
    }
    void matrix(const std::string& name, int k, int m) {
        const auto id = gguf_find_tensor(metadata, name.c_str()); require(id >= 0, "Required audio FFN matrix missing");
        const auto * ne = gguf_get_tensor_ne(metadata, id);
        require(ne[0] == k && ne[1] == m && ne[2] == 1 && ne[3] == 1, "Unsupported learned audio FFN matrix geometry");
    }
    std::array<bounds, chains> load() {
        std::array<bounds, chains> result;
        for (int layer = 0; layer < 12; ++layer) for (int branch = 0; branch < 2; ++branch) {
            auto& b = result[layer * 2 + branch]; const auto suffix = branch ? "_1" : "";
            const auto prefix = "a.blk." + std::to_string(layer);
            b.up = prefix + ".ffn_up" + suffix; b.down = prefix + ".ffn_down" + suffix;
            matrix(b.up + ".weight", 1024, width); matrix(b.down + ".weight", width, 1024);
            b.lo = scalar(b.up + ".output_min"); b.hi = scalar(b.up + ".output_max");
            b.after_lo = scalar(b.down + ".input_min"); b.after_hi = scalar(b.down + ".input_max");
            require(b.lo <= b.hi && b.after_lo <= b.after_hi, "Reversed learned clamps");
        }
        return result;
    }
};
struct options {
    std::string projector, helper_library, output, dependency_revision;
    std::vector<int> lengths{25,48,96,500};
    int threads = 8, warmup = 3, repeats = 20;
    double seconds = 1;
};
int integer(const std::string& s) { size_t n = 0; int v = std::stoi(s, &n); require(n == s.size(), "Invalid integer suffix"); return v; }
options parse(int argc, char ** argv) {
    options o; std::vector<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        require(i + 1 < argc, "Each option requires one value"); std::string k = argv[i], v = argv[++i];
        require(std::find(seen.begin(), seen.end(), k) == seen.end(), "Duplicate option"); seen.push_back(k);
        if (k == "--projector") o.projector = v;
        else if (k == "--helper-library") o.helper_library = v;
        else if (k == "--ggml-revision") o.dependency_revision = v;
        else if (k == "--output") o.output = v;
        else if (k == "--threads") o.threads = integer(v);
        else if (k == "--warmup") o.warmup = integer(v);
        else if (k == "--repeats") o.repeats = integer(v);
        else if (k == "--minimum-seconds") { size_t n = 0; o.seconds = std::stod(v, &n); require(n == v.size(), "Invalid duration suffix"); }
        else if (k == "--lengths") {
            o.lengths.clear(); std::istringstream in(v); std::string part;
            while (std::getline(in, part, ',')) { const int n = integer(part); require(n >= 1 && n <= 1024 && std::find(o.lengths.begin(), o.lengths.end(), n) == o.lengths.end(), "Invalid/duplicate length"); o.lengths.push_back(n); }
            require(!o.lengths.empty() && v.back() != ',', "Empty length");
        } else throw std::runtime_error("Unknown option " + k);
    }
    require(!o.projector.empty() && !o.helper_library.empty() && o.dependency_revision == revision && !o.output.empty(), "Require projector/helper-library/exact ggml-revision/output");
    require((o.threads == 7 || o.threads == 8) && o.warmup >= 1 && o.warmup <= 100 && o.repeats >= 1 && o.repeats <= 10000 && std::isfinite(o.seconds) && o.seconds >= 0 && o.seconds <= 60, "Invalid thread/timing policy");
    require(std::getenv("GGML_CPU_DISABLE_FUSION") == nullptr, "Do not change baseline fusion policy");
    return o;
}
struct guarded {
    static constexpr size_t border = 64;
    static constexpr float canary = 9876.5f;
    size_t count, allocated; std::vector<float> backing; float * storage;
    explicit guarded(size_t n) : count(n), allocated((n + 2 * border + 15) / 16 * 16), backing(allocated + 15),
        storage(reinterpret_cast<float *>((reinterpret_cast<uintptr_t>(backing.data()) + 63) & ~uintptr_t(63))) { std::fill(storage, storage + allocated, canary); }
    guarded(const guarded&) = delete;
    float * data() const { return storage + border; }
    bool guards() const { for (size_t i = 0; i < border; ++i) if (storage[i] != canary) return false; for (size_t i = border + count; i < allocated; ++i) if (storage[i] != canary) return false; return true; }
    uint64_t fingerprint() const { uint64_t h = 14695981039346656037ULL; const auto * p = reinterpret_cast<const unsigned char *>(storage); for (size_t i = 0; i < allocated * 4; ++i) h = (h ^ p[i]) * 1099511628211ULL; return h; }
};
struct session {
    ggml_backend_t backend = nullptr; ggml_threadpool_t pool = nullptr;
    explicit session(int threads) {
        backend = ggml_backend_cpu_init(); require(backend, "Cannot create CPU backend");
        auto params = ggml_threadpool_params_default(threads); pool = ggml_threadpool_new(&params);
        if (!pool) { ggml_backend_free(backend); backend = nullptr; throw std::runtime_error("Cannot create CPU pool"); }
        ggml_backend_cpu_set_n_threads(backend, threads); ggml_backend_cpu_set_threadpool(backend, pool); ggml_backend_cpu_set_use_ref(backend, false);
    }
    ~session() { if (backend) ggml_backend_free(backend); if (pool) ggml_threadpool_free(pool); }
};
struct worker_counts {
    bool instrument = false;
    std::array<std::atomic<uint64_t>, 8> rows{};
};
struct callback_state { const bounds * limits = nullptr; worker_counts * counts = nullptr; };
void fused(ggml_tensor * dst, const ggml_tensor * src, int ith, int nth, void * data) {
    const auto& state = *static_cast<const callback_state *>(data); const auto& b = *state.limits;
    uint64_t rows = 0;
    for (int64_t row = ith; row < src->ne[1]; row += nth) {
        const auto * input = reinterpret_cast<const float *>(static_cast<const char *>(src->data) + row * src->nb[1]);
        auto * output = reinterpret_cast<float *>(static_cast<char *>(dst->data) + row * dst->nb[1]);
        alignas(64) float temporary[tile];
        for (int col = 0; col < width; col += tile) {
            for (int i = 0; i < tile; ++i) temporary[i] = clip(input[col + i], b.lo, b.hi);
            ggml_vec_silu_f32(tile, temporary, temporary);
            for (int i = 0; i < tile; ++i) output[col + i] = clip(temporary[i], b.after_lo, b.after_hi);
        }
        ++rows;
    }
    if (state.counts->instrument) state.counts->rows[size_t(ith)].fetch_add(rows, std::memory_order_relaxed);
}
bool eligible(const ggml_tensor * x, const callback_state * state, bool enabled = true, bool both_learned_clamps = true) {
    if (!enabled || !both_learned_clamps || !x || !state || !state->limits || !state->counts || x->type != GGML_TYPE_F32 ||
        x->ne[0] != width || x->ne[1] <= 0 || x->ne[2] != 1 || x->ne[3] != 1 || !ggml_is_contiguous(x)) return false;
    const auto& b = *state->limits;
    if (!std::isfinite(b.lo) || !std::isfinite(b.hi) || !std::isfinite(b.after_lo) || !std::isfinite(b.after_hi) || b.lo > b.hi || b.after_lo > b.after_hi) return false;
    const float low = std::numeric_limits<float>::lowest(), high = std::numeric_limits<float>::max();
    return !(b.lo == low && b.hi == high && b.after_lo == low && b.after_hi == high);
}
ggml_tensor * make_fused(ggml_context * ctx, ggml_tensor * x, callback_state * state, bool enabled = true, bool both_learned_clamps = true) {
    return eligible(x, state, enabled, both_learned_clamps) ? ggml_map_custom1(ctx, x, fused, GGML_N_TASKS_MAX, state) : nullptr;
}
double oracle(float value, const bounds& b) {
    // Independent Double arithmetic; NaN input follows the documented clamp
    // branch choice (upper bound), not std::fmin/std::clamp substitution.
    double x = std::isnan(value) ? double(b.hi) : std::max(double(b.lo), std::min(double(value), double(b.hi)));
    x = x / (1.0 + std::exp(-x));
    return std::max(double(b.after_lo), std::min(x, double(b.after_hi)));
}
struct problem {
    int tokens; const std::array<bounds, chains>& limits; guarded inputs; uint64_t input_hash; std::vector<double> expected;
    problem(int n, const std::array<bounds, chains>& b) : tokens(n), limits(b), inputs(size_t(chains) * width * n), expected(inputs.count) {
        uint32_t random = 1729;
        for (int chain = 0; chain < chains; ++chain) {
            const auto& c = b[chain];
            std::vector<float> directed{0.f, -0.f, std::numeric_limits<float>::denorm_min(), -std::numeric_limits<float>::denorm_min(),
                std::numeric_limits<float>::min(), -std::numeric_limits<float>::min(), std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
                INFINITY, -INFINITY, from_bits(0x7fc00123), from_bits(0xffc00456), -1.f, 1.f, -88.f, 88.f};
            for (float threshold : {c.lo, c.hi, c.after_lo, c.after_hi}) {
                directed.push_back(std::nextafter(threshold, -INFINITY)); directed.push_back(threshold); directed.push_back(std::nextafter(threshold, INFINITY));
            }
            // Positive inverse SiLU targets the reachable downstream upper
            // threshold. Negative learned bounds lie below SiLU's minimum;
            // direct clamp controls cover their otherwise unreachable branches.
            if (c.after_hi > 0) {
                double lo = 0, hi = std::max(100., double(c.after_hi) * 2);
                for (int i = 0; i < 100; ++i) { double m = (lo + hi) / 2; if (m / (1 + std::exp(-m)) < c.after_hi) lo = m; else hi = m; }
                const float inverse = float((lo + hi) / 2);
                directed.push_back(std::nextafter(inverse, -INFINITY)); directed.push_back(inverse); directed.push_back(std::nextafter(inverse, INFINITY));
            }
            for (size_t i = 0; i < size_t(width) * n; ++i) {
                random = random * 1664525U + 1013904223U;
                float x = float(int(random >> 8) - 8388608) / 8388608.f * std::max(std::abs(c.lo), std::abs(c.hi)) * 1.5f;
                if (i % 128 < directed.size()) x = directed[i % 128];
                const size_t at = size_t(chain) * width * n + i; inputs.data()[at] = x; expected[at] = oracle(x, c);
            }
        }
        input_hash = inputs.fingerprint();
    }
    bool immutable() const { return inputs.guards() && inputs.fingerprint() == input_hash; }
};
struct route {
    guarded outputs; ggml_backend_t backend; ggml_context * ctx = nullptr; ggml_backend_graph_plan_t plan = nullptr;
    ggml_backend_buffer_t input_buffer = nullptr, output_buffer = nullptr, intermediates = nullptr;
    std::array<callback_state, chains> states{}; worker_counts counters;
    bool candidate; int graph_nodes = 0;
    route(session& owner, problem& p, bool use_fused) : outputs(p.inputs.count), backend(owner.backend), candidate(use_fused) {
        try {
            ctx = ggml_init({1024 * 1024, nullptr, true}); require(ctx, "Cannot create graph context");
            auto * all = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, int64_t(chains) * p.tokens);
            input_buffer = ggml_backend_cpu_buffer_from_ptr(p.inputs.storage, p.inputs.allocated * 4);
            output_buffer = ggml_backend_cpu_buffer_from_ptr(outputs.storage, outputs.allocated * 4);
            require(input_buffer && output_buffer && ggml_backend_tensor_alloc(input_buffer, all, p.inputs.data()) == GGML_STATUS_SUCCESS, "Cannot bind input/output buffers");
            ggml_set_input(all); auto * graph = ggml_new_graph_custom(ctx, 256, false);
            for (int i = 0; i < chains; ++i) {
                const size_t offset = size_t(i) * width * p.tokens * 4;
                auto * x = ggml_view_2d(ctx, all, width, p.tokens, width * 4, offset); const auto& b = p.limits[i];
                states[i] = {&b, &counters};
                auto * y = use_fused ? make_fused(ctx, x, &states[i]) : ggml_clamp(ctx, ggml_silu(ctx, ggml_clamp(ctx, x, b.lo, b.hi)), b.after_lo, b.after_hi);
                require(y && y->view_src == nullptr && ggml_is_contiguous(y), "Output must be an eligible non-view contiguous node");
                require(ggml_backend_tensor_alloc(output_buffer, y, outputs.data() + offset / 4) == GGML_STATUS_SUCCESS, "Cannot bind guarded output segment");
                ggml_set_output(y); ggml_build_forward_expand(graph, y);
            }
            graph_nodes = graph->n_nodes;
            intermediates = ggml_backend_alloc_ctx_tensors(ctx, backend);
            // The candidate has only prebound input/output storage and views.
            // A null allocation is normal when no intermediate needs storage.
            require(use_fused || intermediates, "Cannot allocate baseline graph intermediates");
            plan = ggml_backend_graph_plan_create(backend, graph); require(plan, "Cannot create persistent CPU plan");
        } catch (...) { release(); throw; }
    }
    route(const route&) = delete;
    void release() { if (plan) { ggml_backend_graph_plan_free(backend, plan); plan = nullptr; } ggml_backend_buffer_free(intermediates); intermediates = nullptr; ggml_backend_buffer_free(output_buffer); output_buffer = nullptr; ggml_backend_buffer_free(input_buffer); input_buffer = nullptr; if (ctx) { ggml_free(ctx); ctx = nullptr; } }
    ~route() { release(); }
    void compute() { require(ggml_backend_graph_plan_compute(backend, plan) == GGML_STATUS_SUCCESS, "CPU graph compute failed"); }
};
struct quality { bool passed = true, bits = true, finite = true, guards = true; double max_error = 0, max_ratio = 0; };
quality check(problem& p, route& r, const std::vector<float>& original) {
    quality q; q.guards = p.immutable() && r.outputs.guards(); q.bits = std::memcmp(r.outputs.data(), original.data(), original.size() * 4) == 0;
    for (size_t i = 0; i < p.expected.size(); ++i) {
        const double actual = r.outputs.data()[i]; q.finite &= std::isfinite(actual) && actual != guarded::canary;
        if (!std::isfinite(actual)) { q.passed = false; continue; }
        const double error = std::abs(actual - p.expected[i]), bound = 2e-6 + 2e-6 * std::abs(p.expected[i]);
        q.max_error = std::max(q.max_error, error); q.max_ratio = std::max(q.max_ratio, error / bound); q.passed &= error <= bound;
    }
    q.passed &= q.bits && q.finite && q.guards; return q;
}
void combine(quality& a, const quality& b) { a.passed &= b.passed; a.bits &= b.bits; a.finite &= b.finite; a.guards &= b.guards; a.max_error = std::max(a.max_error, b.max_error); a.max_ratio = std::max(a.max_ratio, b.max_ratio); }
struct timing { std::vector<double> samples; double median = 0; quality checks; int checked = 0; };
timing measure(problem& p, route& r, const std::vector<float>& original, const options& o) {
    timing t; require(!r.counters.instrument, "Worker instrumentation must be disabled during timing");
    for (int i = 0; i < o.warmup; ++i) { r.compute(); combine(t.checks, check(p, r, original)); ++t.checked; }
    double milliseconds = 0;
    do { const auto start = clock_type::now(); r.compute(); const double ms = std::chrono::duration<double, std::milli>(clock_type::now() - start).count(); t.samples.push_back(ms); milliseconds += ms; }
    while (t.samples.size() < size_t(o.repeats) || milliseconds < o.seconds * 1000);
    combine(t.checks, check(p, r, original)); ++t.checked;
    auto s = t.samples; std::sort(s.begin(), s.end()); t.median = (s[(s.size() - 1) / 2] + s[s.size() / 2]) / 2; return t;
}
int controls(const bounds& actual) {
    auto * ctx = ggml_init({1024 * 1024, nullptr, true}); require(ctx, "Cannot create controls context");
    int checked = 0;
    try {
        worker_counts counts; callback_state state{&actual, &counts};
        auto * valid = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, 25);
        auto refuse = [&](ggml_tensor * x, callback_state * s, bool enabled = true, bool learned = true) {
            const size_t before = ggml_used_mem(ctx); require(make_fused(ctx, x, s, enabled, learned) == nullptr && ggml_used_mem(ctx) == before, "Ineligible factory must refuse without allocating a node"); ++checked;
        };
        refuse(valid, &state, false); refuse(valid, &state, true, false); refuse(nullptr, &state); refuse(valid, nullptr);
        refuse(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width - 1, 25), &state);
        refuse(ggml_new_tensor_2d(ctx, GGML_TYPE_F16, width, 25), &state);
        refuse(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, 0), &state);
        refuse(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, 25, 2), &state);
        auto * strided = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, 25); strided->nb[1] += 64; refuse(strided, &state);
        for (int which = 0; which < 4; ++which) {
            bounds bad = actual; if (which == 0) bad.lo = INFINITY; if (which == 1) bad.after_hi = NAN;
            if (which == 2) { bad.lo = bad.hi + 1; }
            if (which == 3) { bad.lo = bad.after_lo = std::numeric_limits<float>::lowest(); bad.hi = bad.after_hi = std::numeric_limits<float>::max(); }
            callback_state s{&bad, &counts}; refuse(valid, &s);
        }
        // Direct threshold/extreme semantics for BOTH learned clamps, including
        // downstream negative thresholds unreachable after ordinary SiLU.
        for (const auto ends : {std::array<float,2>{actual.lo,actual.hi}, std::array<float,2>{actual.after_lo,actual.after_hi}}) {
            for (float x : {std::nextafter(ends[0],-INFINITY), ends[0], std::nextafter(ends[0],INFINITY), std::nextafter(ends[1],-INFINITY), ends[1], std::nextafter(ends[1],INFINITY), -0.f, 0.f, -INFINITY, INFINITY, from_bits(0x7fc00123)}) {
                const float expected = MAX(MIN(x,ends[1]),ends[0]); require(bits(clip(x,ends[0],ends[1])) == bits(expected), "Clamp branch/signed-zero/NaN mismatch"); ++checked;
            }
        }
        ggml_free(ctx); return checked;
    } catch (...) { ggml_free(ctx); throw; }
}
void phase(std::ostream& out, const char * name, const timing& t) {
    out << "{\"name\":" << quote(name) << ",\"median_ms\":" << t.median << ",\"repeats\":" << t.samples.size()
        << ",\"warm_and_final_checks\":" << t.checked << ",\"passed\":" << (t.checks.passed ? "true" : "false") << ",\"complete_baseline_bits\":" << (t.checks.bits ? "true" : "false")
        << ",\"finite\":" << (t.checks.finite ? "true" : "false") << ",\"guards_and_immutability\":" << (t.checks.guards ? "true" : "false") << ",\"double_max_error\":" << t.checks.max_error << ",\"double_max_error_over_bound\":" << t.checks.max_ratio << ",\"samples_ms\":[";
    for (size_t i = 0; i < t.samples.size(); ++i) { out << (i ? "," : "") << t.samples[i]; }
    out << "]}";
}
} // namespace
int main(int argc, char ** argv) {
    FILE * reserved = nullptr;
    try {
        const auto o = parse(argc, argv);
        reserved = std::fopen(o.output.c_str(), "wx"); require(reserved, "Require fresh writable output path");
        require(!ggml_cpu_has_sve() && !ggml_cpu_has_riscv_v(), "Runtime-vector-length helpers are outside tile64 equivalence scope");
        void * symbol = dlsym(RTLD_DEFAULT, "ggml_vec_silu_f32"); Dl_info helper{};
        require(symbol && symbol == reinterpret_cast<void *>(&ggml_vec_silu_f32) && dladdr(symbol, &helper) && helper.dli_fname,
            "Cannot identify actual linked unchanged ggml SiLU helper");
        require(std::filesystem::canonical(helper.dli_fname) == std::filesystem::canonical(o.helper_library), "Actual helper library differs from guarded linked library");
        checkpoint file(o.projector); const auto limits = file.load(); int control_count = 0; for (const auto& b : limits) control_count += controls(b);
        session owner(o.threads); bool passed = true; std::ostringstream out; out << std::setprecision(17);
        out << "{\"ggml_revision\":" << quote(revision) << ",\"projector\":" << quote(std::filesystem::canonical(o.projector).string())
            << ",\"unchanged_silu_symbol\":\"ggml_vec_silu_f32\",\"actual_helper_library\":" << quote(std::filesystem::canonical(helper.dli_fname).string())
            << ",\"threads\":" << o.threads << ",\"tile\":" << tile << ",\"width\":" << width << ",\"learned_chains\":[";
        for (int i = 0; i < chains; ++i) { const auto& b = limits[i]; out << (i ? "," : "") << "{\"up\":" << quote(b.up) << ",\"down\":" << quote(b.down) << ",\"bounds\":[" << b.lo << ',' << b.hi << ',' << b.after_lo << ',' << b.after_hi << "],\"bound_f32_bits\":[" << bits(b.lo) << ',' << bits(b.hi) << ',' << bits(b.after_lo) << ',' << bits(b.after_hi) << "]}"; }
        out << "],\"control_checks\":" << control_count << ",\"warmup\":" << o.warmup << ",\"minimum_repeats\":" << o.repeats << ",\"minimum_seconds\":" << o.seconds << ",\"persistent_graph_plans\":true,\"persistent_pool\":true,\"cases\":[";
        bool comma = false;
        for (int n : o.lengths) {
            problem p(n, limits); route baseline(owner, p, false), candidate(owner, p, true);
            baseline.compute(); const std::vector<float> original(baseline.outputs.data(), baseline.outputs.data() + baseline.outputs.count);
            const auto initial = check(p, baseline, original); passed &= initial.passed;
            candidate.counters.instrument = true; candidate.compute(); candidate.counters.instrument = false;
            const auto instrumented = check(p, candidate, original); passed &= instrumented.passed;
            uint64_t rows = 0; bool full_workers = true;
            for (int i = 0; i < o.threads; ++i) { const auto count = candidate.counters.rows[size_t(i)].load(); rows += count; full_workers &= count > 0; }
            const bool coverage = rows == uint64_t(chains) * n && full_workers; passed &= coverage;
            const auto a = measure(p, baseline, original, o), b = measure(p, candidate, original, o), c = measure(p, candidate, original, o), d = measure(p, baseline, original, o);
            passed &= a.checks.passed && b.checks.passed && c.checks.passed && d.checks.passed;
            out << (comma ? "," : "") << "{\"tokens_per_chain\":" << n << ",\"complete_elements\":" << p.inputs.count << ",\"initial_baseline_passed\":" << (initial.passed ? "true" : "false")
                << ",\"instrumented_candidate_passed\":" << (instrumented.passed ? "true" : "false") << ",\"baseline_graph_nodes\":" << baseline.graph_nodes << ",\"candidate_graph_nodes\":" << candidate.graph_nodes
                << ",\"worker_coverage_outside_timing_passed\":" << (coverage ? "true" : "false") << ",\"instrumented_rows\":" << rows << ",\"worker_rows\":[";
            for (int i = 0; i < o.threads; ++i) out << (i ? "," : "") << candidate.counters.rows[size_t(i)].load();
            out << "],\"candidate_over_baseline_ratios\":[" << b.median/a.median << ',' << c.median/d.median << "],\"phases\":[";
            phase(out,"baseline-a",a); out << ','; phase(out,"candidate-a",b); out << ','; phase(out,"candidate-b",c); out << ','; phase(out,"baseline-b",d); out << "]}"; comma = true;
        }
        out << "],\"completed\":true,\"passed\":" << (passed ? "true" : "false")
            << ",\"model_qualified\":false,\"production_promoted\":false,\"double_bound\":\"2e-6+2e-6*abs(reference)\",\"limitations\":[\"Synthetic projected activations use all24 checkpoint clamp pairs; no learned matrix products/full tower/frontend/text/HTTP execution.\",\"All warm outputs plus final timed outputs are checked completely; intermediate timed outputs are not separately copied or validated.\",\"Both routes call the same linked upstream vector SiLU. Tile64 dispatch is qualified only for fixed4/8/16-lane or scalar helpers; SVE/RVV refuse.\",\"Graph creation/allocation, oracle/checks, fingerprints and instrumented worker proof are excluded from timing.\",\"Node counts include input views; topology and logical storage are not measured DRAM traffic or service gains.\"]}\n";
        const auto report = out.str(); require(std::fwrite(report.data(), 1, report.size(), reserved) == report.size(), "Report write failed");
        const int close_status = std::fclose(reserved); reserved = nullptr;
        require(close_status == 0, "Report close failed"); return passed ? 0 : 1;
    } catch (const std::exception& e) {
        if (reserved) { const auto failure = std::string("{\"completed\":false,\"passed\":false,\"error\":") + quote(e.what()) + "}\n"; (void)std::fwrite(failure.data(), 1, failure.size(), reserved); std::fclose(reserved); }
        std::cerr << e.what() << '\n'; return 2;
    }
}
