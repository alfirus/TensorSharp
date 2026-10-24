// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Source-service experiment only. No CUDA, pinning, file writes or model load.
// Build on Windows/MSVC with /std:c++17 /EHsc /O2 and /I TensorSharp.GGML.Native.
// ReadFile remains the authoritative byte oracle and full-task fault fallback.
#if !defined(_WIN32) || !defined(_MSC_VER)
#error This source-service probe requires Windows and MSVC structured exceptions.
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "ggml_ops_file_source.h"
#include "dsv41_engram_io.h"
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t MiB = std::size_t(1) << 20;
constexpr std::size_t maximum_tasks = 65536, maximum_spans = 4096;
constexpr unsigned prepare_workers = 15, pool_slots = prepare_workers + 1;
double ms(Clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
std::runtime_error windows_error(const char* action) {
    return std::runtime_error(std::string(action) + ": Win32=" + std::to_string(GetLastError()));
}
std::uint64_t number(const std::wstring& text) {
    require(!text.empty(), "Empty integer option");
    std::uint64_t value = 0;
    for (const auto c : text) {
        require(c >= L'0' && c <= L'9', "Integers must contain decimal digits only");
        const auto digit = static_cast<unsigned>(c - L'0');
        require(value <= (UINT64_MAX - digit) / 10, "Integer option overflow");
        value = value * 10 + digit;
    }
    return value;
}
std::string utf8(const std::wstring& value) {
    require(!value.empty() && value.size() < 32768, "Invalid path length");
    const int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
        static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
    require(size > 0, "Cannot encode path as UTF-8");
    std::string result(static_cast<std::size_t>(size), '\0');
    require(WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
        static_cast<int>(value.size()), result.data(), size, nullptr, nullptr) == size, "Cannot encode path");
    return result;
}
struct Span { std::uint64_t relative, bytes; };
enum class Fault { none, partial, in_page };
struct Options {
    std::string path;
    std::uint64_t offset = 0, bytes = 0;
    std::size_t arena_bytes = 72 * MiB, task_bytes = 4 * MiB;
    unsigned blocks = 2, warmups = 1;
    Fault fault = Fault::none;
    std::vector<Span> spans;
};
Options options(int argc, wchar_t** argv) {
    Options o;
    bool have_offset = false, have_bytes = false;
    for (int i = 1; i < argc; ++i) {
        const std::wstring name = argv[i];
        require(i + 1 < argc, "Every option requires a value");
        const std::wstring value = argv[++i];
        if (name == L"--path") { require(o.path.empty(), "Repeated path"); o.path = utf8(value); }
        else if (name == L"--offset") { require(!have_offset, "Repeated offset"); have_offset = true; o.offset = number(value); }
        else if (name == L"--bytes") { require(!have_bytes, "Repeated bytes"); have_bytes = true; o.bytes = number(value); }
        else if (name == L"--arena-mib") {
            const auto size = number(value); require(size == 32 || size == 72, "Arena must be 32 or 72 MiB");
            o.arena_bytes = static_cast<std::size_t>(size) * MiB;
        } else if (name == L"--task-bytes") {
            const auto size = number(value); require(size >= 2 && size <= 4 * MiB, "Task bytes must be in [2,4194304]");
            o.task_bytes = static_cast<std::size_t>(size);
        } else if (name == L"--blocks" || name == L"--warmups") {
            const auto count = number(value); require(count >= 1 && count <= 8, "Blocks/warmups must be in [1,8]");
            if (name == L"--blocks") o.blocks = static_cast<unsigned>(count); else o.warmups = static_cast<unsigned>(count);
        } else if (name == L"--span") {
            require(o.spans.size() < maximum_spans, "Too many spans");
            const auto separator = value.find(L':');
            require(separator != std::wstring::npos && value.find(L':', separator + 1) == std::wstring::npos,
                "Span syntax is relative_offset:bytes");
            o.spans.push_back({number(value.substr(0, separator)), number(value.substr(separator + 1))});
        } else if (name == L"--fault") {
            require(value == L"none" || value == L"partial" || value == L"inpage", "Fault must be none, partial or inpage");
            o.fault = value == L"none" ? Fault::none : value == L"partial" ? Fault::partial : Fault::in_page;
        } else throw std::runtime_error("Unknown option");
    }
    require(!o.path.empty() && have_offset && have_bytes && o.bytes > 0 && o.offset <= UINT64_MAX - o.bytes,
        "Required: --path PATH --offset DECIMAL --bytes DECIMAL (exact registered extent)");
    if (o.spans.empty()) o.spans.push_back({0, o.bytes});
    std::uint64_t total = 0;
    for (const auto& span : o.spans) {
        require(span.bytes && span.relative <= o.bytes && span.bytes <= o.bytes - span.relative,
            "Requested span exceeds the registered source extent");
        require(span.bytes <= (std::uint64_t(4) << 30) - total, "Requested logical bytes exceed 4 GiB probe bound");
        total += span.bytes;
    }
    return o;
}

// This is an entirely local probe owner, not a production mapping service.
// The duplicate keeps the exact source file object/generation and restrictive
// readonly sharing alive independently of the original ExpertFileSource.
// There is no destructor cleanup: main must check close, retaining this entire
// owner in its bounded quarantine slot if any OS release fails.
class ReadView {
    HANDLE file_ = INVALID_HANDLE_VALUE, section_ = nullptr;
    const std::uint8_t* base_ = nullptr;
    std::size_t map_bytes_ = 0, delta_ = 0;
    std::uint64_t bytes_ = 0, source_identity_ = 0;
public:
    DWORD close_error = ERROR_SUCCESS;
    ReadView() = default;
    ReadView(const ReadView&) = delete;
    ReadView& operator=(const ReadView&) = delete;
    void open(const tsg::ExpertFileSource& source) {
        require(!base_ && !section_ && file_ == INVALID_HANDLE_VALUE, "View owner already used");
        std::uint64_t file_bytes = 0;
        require(source.readonly_mapping_file_size(file_bytes), "Source lacks an exact readonly mapping handle");
        SYSTEM_INFO system{}; GetSystemInfo(&system);
        const auto granule = static_cast<std::uint64_t>(system.dwAllocationGranularity);
        require(granule && source.offset() <= file_bytes && source.bytes() <= file_bytes - source.offset(),
            "Invalid readonly source geometry");
        const auto begin = source.offset() / granule * granule;
        const auto delta = source.offset() - begin;
        require(delta <= SIZE_MAX && source.bytes() <= SIZE_MAX - delta, "View exceeds process address size");
        delta_ = static_cast<std::size_t>(delta); map_bytes_ = delta_ + static_cast<std::size_t>(source.bytes());
        bytes_ = source.bytes(); source_identity_ = source.identity();
        file_ = source.duplicate_readonly_mapping_file();
        if (file_ == INVALID_HANDLE_VALUE) throw windows_error("Duplicate exact readonly source");
        section_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!section_) throw windows_error("Create readonly section");
        base_ = static_cast<const std::uint8_t*>(MapViewOfFile(section_, FILE_MAP_READ,
            static_cast<DWORD>(begin >> 32), static_cast<DWORD>(begin), map_bytes_));
        if (!base_) throw windows_error("Map exact registered source extent");
        MEMORY_BASIC_INFORMATION info{};
        require(VirtualQuery(base_, &info, sizeof(info)) == sizeof(info) && info.Type == MEM_MAPPED
            && info.AllocationBase == base_ && info.Protect == PAGE_READONLY, "View is not readonly mapped storage");
    }
    const std::uint8_t* range(std::uint64_t relative, std::size_t count) const {
        require(base_ && relative <= bytes_ && count <= bytes_ - relative, "Mapped request exceeds source extent");
        return base_ + delta_ + static_cast<std::size_t>(relative);
    }
    bool close() noexcept {
        close_error = ERROR_SUCCESS;
        if (base_) {
            if (!UnmapViewOfFile(base_)) { close_error = GetLastError(); return false; }
            base_ = nullptr; map_bytes_ = delta_ = 0;
        }
        if (section_) {
            if (!CloseHandle(section_)) { close_error = GetLastError(); return false; }
            section_ = nullptr;
        }
        if (file_ != INVALID_HANDLE_VALUE) {
            if (!CloseHandle(file_)) { close_error = GetLastError(); return false; }
            file_ = INVALID_HANDLE_VALUE;
        }
        return true;
    }
    unsigned owners() const noexcept { return unsigned(base_ != nullptr) + unsigned(section_ != nullptr) + unsigned(file_ != INVALID_HANDLE_VALUE); }
    std::uint64_t identity() const noexcept { return source_identity_; }
};

// Trivial leaf frame only: no C++ object requiring unwinding inside __try.
// The injected modes first change a strict destination prefix, then fail. The
// caller must overwrite the WHOLE logical task through authoritative read().
__declspec(noinline) bool guarded_copy(void* destination, const void* source, std::size_t bytes, Fault fault) noexcept {
    __try {
        if (fault != Fault::none) {
            if (bytes > 1) std::memcpy(destination, source, bytes / 2);
            if (fault == Fault::in_page) RaiseException(EXCEPTION_IN_PAGE_ERROR, 0, 0, nullptr);
            return false;
        }
        std::memcpy(destination, source, bytes);
        return true;
    } __except ((GetExceptionCode() == EXCEPTION_IN_PAGE_ERROR || GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION)
        ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) { return false; }
}
struct Task { std::uint64_t relative; std::size_t destination, bytes; };
struct Window { std::size_t first, count, bytes, slot; };
struct Plan { std::vector<Task> tasks; std::vector<Window> windows; std::uint64_t bytes = 0; };
Plan plan(const Options& o) {
    Plan p; const auto slot_bytes = o.arena_bytes / 2;
    for (const auto& span : o.spans) for (std::uint64_t offset = 0; offset < span.bytes;) {
        if (p.windows.empty() || p.windows.back().bytes == slot_bytes)
            p.windows.push_back({p.tasks.size(), 0, 0, p.windows.size() % 2});
        auto& window = p.windows.back();
        const auto bytes = static_cast<std::size_t>((std::min)({span.bytes - offset,
            static_cast<std::uint64_t>(o.task_bytes), static_cast<std::uint64_t>(slot_bytes - window.bytes)}));
        require(p.tasks.size() < maximum_tasks, "Task metadata exceeds the bounded probe plan");
        p.tasks.push_back({span.relative + offset, window.slot * slot_bytes + window.bytes, bytes});
        ++window.count; window.bytes += bytes; p.bytes += bytes; offset += bytes;
    }
    return p;
}
struct Buffers {
    std::uint8_t* arena = nullptr;
    std::vector<std::uint8_t> oracle;
    explicit Buffers(const Options& o) : oracle(o.task_bytes) {
        arena = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, o.arena_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (!arena) throw windows_error("Allocate bounded ordinary transfer arena");
    }
    Buffers(const Buffers&) = delete;
    Buffers& operator=(const Buffers&) = delete;
    bool close() noexcept {
        if (arena && !VirtualFree(arena, 0, MEM_RELEASE)) return false;
        arena = nullptr; return true;
    }
};
struct TaskResult { std::uint64_t read_bytes = 0, view_bytes = 0; bool fallback = false, injected = false, consumed = false; };
struct Result {
    double service_ms = 0, validation_ms = 0;
    std::uint64_t checksum = 14695981039346656037ULL, bytes = 0, read_bytes = 0, view_bytes = 0;
    std::size_t tasks = 0, fallbacks = 0, injected = 0;
};
Result run(tsg_dsv41::engram_io_pool& pool, const tsg::ExpertFileSource& source, const ReadView& view,
    Buffers& buffers, const Plan& p, bool mapped, std::atomic<bool>& disabled, Fault fault = Fault::none) {
    Result result;
    std::vector<TaskResult> records(p.tasks.size());
    for (const auto& window : p.windows) {
        // Each slot is reused only after the previous window is fully joined
        // and validated. No asynchronous consumer or H2D exists in this probe.
        std::memset(buffers.arena + p.tasks[window.first].destination, 0xA5, window.bytes);
        const auto started = Clock::now();
        pool.run_pipelined(window.count, [&](std::size_t local) {
            const auto index = window.first + local;
            const auto& task = p.tasks[index]; auto& record = records[index];
            auto* destination = buffers.arena + task.destination;
            if (!mapped || disabled.load(std::memory_order_relaxed)) {
                source.read(task.relative, destination, task.bytes); record.read_bytes = task.bytes;
                return;
            }
            const auto selected_fault = index == 0 ? fault : Fault::none;
            if (guarded_copy(destination, view.range(task.relative, task.bytes), task.bytes, selected_fault))
                record.view_bytes = task.bytes;
            else {
                disabled.store(true, std::memory_order_relaxed);
                // A failed copy may have changed any prefix. Never publish it:
                // the original read overwrites exactly the ENTIRE task first.
                source.read(task.relative, destination, task.bytes);
                record.read_bytes = task.bytes; record.fallback = true;
                record.injected = selected_fault != Fault::none;
            }
        }, [&](std::size_t local) { records[window.first + local].consumed = true; });
        result.service_ms += ms(started);
        const auto validation = Clock::now();
        for (std::size_t i = window.first; i < window.first + window.count; ++i) {
            const auto& task = p.tasks[i]; const auto& record = records[i];
            require(record.consumed && record.read_bytes + record.view_bytes == task.bytes,
                "Partial task or missing consumer publication");
            source.read(task.relative, buffers.oracle.data(), task.bytes);
            require(std::memcmp(buffers.oracle.data(), buffers.arena + task.destination, task.bytes) == 0,
                "Source-view bytes differ from authoritative full-task ReadFile");
            for (std::size_t n = 0; n < task.bytes; ++n) {
                result.checksum ^= buffers.arena[task.destination + n]; result.checksum *= 1099511628211ULL;
            }
            result.bytes += task.bytes; result.read_bytes += record.read_bytes; result.view_bytes += record.view_bytes;
            result.fallbacks += record.fallback; result.injected += record.injected; ++result.tasks;
        }
        result.validation_ms += ms(validation);
    }
    require(result.bytes == p.bytes && result.tasks == p.tasks.size(), "Incomplete source-service coverage");
    return result;
}
void print(const char* phase, unsigned sequence, bool mapped, const Result& r, const Plan& p) {
    std::cout << std::fixed << std::setprecision(6) << "{\"event\":\"round\",\"phase\":\"" << phase
        << "\",\"sequence\":" << sequence << ",\"mode\":\"" << (mapped ? "view" : "read")
        << "\",\"service_ms\":" << r.service_ms << ",\"validation_ms\":" << r.validation_ms
        << ",\"tasks\":" << r.tasks << ",\"windows\":" << p.windows.size() << ",\"logical_bytes\":" << r.bytes
        << ",\"literal_read_bytes\":" << r.read_bytes << ",\"view_copy_bytes\":" << r.view_bytes
        << ",\"fallback_tasks\":" << r.fallbacks << ",\"injected_faults\":" << r.injected
        << ",\"fnv1a64\":" << r.checksum << "}" << std::endl;
}
ReadView* retained_view = nullptr;
Buffers* retained_buffers = nullptr;
}

int wmain(int argc, wchar_t** argv) {
    std::unique_ptr<ReadView> view;
    std::unique_ptr<Buffers> buffers;
    std::unique_ptr<tsg_dsv41::engram_io_pool> pool;
    std::shared_ptr<tsg::ExpertFileSource> source;
    int status = 1;
    try {
        const auto o = options(argc, argv);
        const auto layout = plan(o);
        require(o.fault == Fault::none || layout.tasks.front().bytes >= 2,
            "Partial-copy qualification requires at least two bytes in the first task");
        const auto source_started = Clock::now();
        source = std::make_shared<tsg::ExpertFileSource>(o.path.c_str(), o.offset, o.bytes, false);
        const auto source_ms = ms(source_started);
        view.reset(new ReadView());
        const auto view_started = Clock::now(); view->open(*source); const auto view_ms = ms(view_started);
        require(view->identity() == source->identity(), "Source generation changed");
        const auto setup_started = Clock::now();
        buffers.reset(new Buffers(o)); pool.reset(new tsg_dsv41::engram_io_pool(pool_slots));
        const auto setup_ms = ms(setup_started);
        bool rejected = false;
        try { (void)view->range(o.bytes, 1); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "Mapped bounds rejection was not enforced");
        rejected = false;
        try { source->read(o.bytes, buffers->oracle.data(), 1); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "Original source bounds rejection was not enforced");
        std::cout << std::fixed << std::setprecision(6)
            << "{\"event\":\"startup\",\"source_identity\":" << source->identity() << ",\"source_offset\":" << o.offset
            << ",\"source_bytes\":" << o.bytes << ",\"span_count\":" << o.spans.size()
            << ",\"arena_bytes\":" << o.arena_bytes << ",\"slot_bytes\":" << o.arena_bytes / 2
            << ",\"maximum_task_bytes\":" << o.task_bytes << ",\"configured_prepare_workers\":" << prepare_workers
            << ",\"pool_slots\":" << pool_slots << ",\"source_open_ms\":" << source_ms << ",\"view_open_ms\":" << view_ms
            << ",\"arena_pool_setup_ms\":" << setup_ms << ",\"source_random_access_hint\":" << unsigned(source->random_access_hint())
            << ",\"unlocked\":true,\"pinning\":false,\"cuda\":false,\"transfer_geometry\":\"two_reused_half_arena_windows_joined_before_reuse_no_H2D_overlap\"}"
            << std::endl;
        std::atomic<bool> disabled{false};
        const auto first_read = run(*pool, *source, *view, *buffers, layout, false, disabled);
        print("first_touch_read_then_view_not_cold_comparison", 0, false, first_read, layout);
        const auto first_view = run(*pool, *source, *view, *buffers, layout, true, disabled, o.fault);
        print("first_touch_read_then_view_not_cold_comparison", 1, true, first_view, layout);
        require(first_read.checksum == first_view.checksum, "First-touch checksum mismatch");
        if (o.fault != Fault::none) {
            require(first_view.injected == 1 && first_view.fallbacks >= 1 && disabled.load(),
                "Injected partial-copy fault was not consumed and fully overwritten");
            std::cout << "{\"event\":\"qualification\",\"fault_fallback_passed\":true,\"warmed_timing_qualified\":false,\"scope\":\"synthetic_partial_copy_full_task_ReadFile_overwrite_only\"}" << std::endl;
        } else {
            require(!disabled.load() && first_view.fallbacks == 0, "Actual mapped fault; warmed timing is unqualified");
            for (unsigned i = 0; i < o.warmups * 2; ++i) {
                const bool mapped = i % 2 != 0;
                const auto r = run(*pool, *source, *view, *buffers, layout, mapped, disabled);
                print("warmup", i, mapped, r, layout);
                require(r.checksum == first_read.checksum && !disabled.load(), "Warmup failed byte/fault qualification");
            }
            for (unsigned i = 0; i < o.blocks * 4; ++i) {
                const bool mapped = i % 4 == 1 || i % 4 == 2; // Literal Read / View / View / Literal Read.
                const auto r = run(*pool, *source, *view, *buffers, layout, mapped, disabled);
                print("measured_warmed_ABBA", i, mapped, r, layout);
                require(r.checksum == first_read.checksum && !disabled.load() && r.fallbacks == 0,
                    "Mapped fault or byte mismatch invalidates warmed comparison");
            }
            std::cout << "{\"event\":\"qualification\",\"all_requested_bytes_memcmp\":true,\"warmed_timing_qualified\":true,\"scope\":\"joined_CPU_source_service_only_validation_ReadFile_warms_cache_no_H2D_no_cold_SSD_model_speed_or_guaranteed_residency\"}" << std::endl;
        }
        // Show that the mapping lease survives destruction of its original
        // Source owner. Only two saved bytes are retained, not another payload.
        std::uint8_t expected[2]{}, actual[2]{};
        source->read(0, expected, 1); source->read(o.bytes - 1, expected + 1, 1);
        pool.reset(); source.reset();
        require(guarded_copy(actual, view->range(0, 1), 1, Fault::none)
            && guarded_copy(actual + 1, view->range(o.bytes - 1, 1), 1, Fault::none)
            && std::memcmp(actual, expected, 2) == 0, "Independent exact-handle view lease failed");
        std::cout << "{\"event\":\"independent_lease\",\"source_destroyed\":true,\"exact_endpoint_bytes\":true}" << std::endl;
        status = 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL source-service probe: " << error.what() << std::endl;
    }
    // All preparations/consumers must be joined before releasing view or arena.
    pool.reset(); source.reset();
    bool clean = true;
    if (view) {
        bool closed = false;
        for (unsigned attempt = 0; attempt < 3 && !closed; ++attempt) closed = view->close();
        if (!closed) {
            std::cerr << "FAIL checked view cleanup: Win32=" << view->close_error << " owners=" << view->owners() << std::endl;
            retained_view = view.release(); clean = false;
        }
    }
    if (buffers && !buffers->close()) {
        const auto error = GetLastError();
        std::cerr << "FAIL checked arena cleanup: Win32=" << error << std::endl;
        retained_buffers = buffers.release(); clean = false;
    }
    std::cout << "{\"event\":\"cleanup\",\"view_owners\":" << (retained_view ? retained_view->owners() : 0)
        << ",\"arena_owners\":" << unsigned(retained_buffers != nullptr) << ",\"checked_clean\":" << (clean ? "true" : "false")
        << ",\"unified_budget_accounting_qualified\":false}" << std::endl;
    return clean ? status : 1;
}
