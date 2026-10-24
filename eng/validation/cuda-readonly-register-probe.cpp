// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Standalone Windows capability/cost probe; no model or TensorSharp native code.
#define NOMINMAX
#include <windows.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
std::string quote(const std::string& value) {
    std::ostringstream out; out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
        else out << c;
    }
    return out.str() + '"';
}
std::string utf8(const std::wstring& value) {
    if (value.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, value.data(), int(value.size()), nullptr, 0, nullptr, nullptr);
    std::string result(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, value.data(), int(value.size()), result.data(), n, nullptr, nullptr);
    return result;
}
struct ProbeError : std::runtime_error {
    bool unavailable;
    ProbeError(const std::string& message, bool missing = false) : std::runtime_error(message), unavailable(missing) {}
};
void wincheck(bool ok, const char* action) {
    if (!ok) throw ProbeError(std::string(action) + ": Win32 error " + std::to_string(GetLastError()));
}
std::string cuda_error(cudaError_t code, const char* action) {
    return std::string(action) + ": " + cudaGetErrorName(code) + " (" + std::to_string(int(code)) + "): " + cudaGetErrorString(code);
}
void check(cudaError_t code, const char* action, bool unavailable = false) {
    if (code != cudaSuccess) throw ProbeError(cuda_error(code, action), unavailable);
}
size_t round_up(size_t value, size_t alignment) {
    if (!alignment || value > SIZE_MAX - (alignment - 1)) throw ProbeError("Registration range overflow");
    return (value + alignment - 1) / alignment * alignment;
}
struct Range { size_t offset, bytes, begin, registered_bytes; };
Range range(size_t offset, size_t bytes, size_t file_bytes, size_t page) {
    if (!bytes || offset > file_bytes || bytes > file_bytes - offset) throw ProbeError("Payload range outside fixture");
    size_t begin = offset / page * page, end = round_up(offset + bytes, page);
    if (end > file_bytes) throw ProbeError("Rounded registration envelope outside mapped fixture");
    return { offset, bytes, begin, end - begin };
}
uint64_t fingerprint(const unsigned char* bytes, size_t count) {
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < count; ++i) { hash ^= bytes[i]; hash *= 1099511628211ULL; }
    return hash;
}
struct Options {
    std::filesystem::path fixture, output;
    size_t payload_bytes = 16 * 1024 * 1024;
    int rounds = 4, device = 0;
    bool self_test = false;
};
Options options(int argc, wchar_t** argv) {
    Options result;
    for (int i = 1; i < argc; ++i) {
        std::wstring key = argv[i];
        if (key == L"--self-test") { result.self_test = true; continue; }
        if (++i == argc) throw ProbeError("Option requires a value");
        std::wstring value = argv[i];
        if (key == L"--fixture") result.fixture = std::filesystem::absolute(value);
        else if (key == L"--output") result.output = std::filesystem::absolute(value);
        else if (key == L"--bytes") result.payload_bytes = std::stoull(value);
        else if (key == L"--rounds") result.rounds = std::stoi(value);
        else if (key == L"--device") result.device = std::stoi(value);
        else throw ProbeError("Unknown option: " + utf8(key));
    }
    if (result.fixture.empty() || result.output.empty() || result.payload_bytes < 4096 ||
        result.payload_bytes > 64 * 1024 * 1024 || result.rounds < 1 || result.rounds > 8 || result.device < 0)
        throw ProbeError("Use --fixture NEW_FILE --output NEW_JSON [--bytes 4096..67108864] [--rounds 1..8] [--device N] [--self-test]");
    // Never silently replace evidence, even if an invocation was interrupted.
    if (result.fixture == result.output || std::filesystem::exists(result.fixture) || std::filesystem::exists(result.output))
        throw ProbeError("Fixture and report paths must both be new");
    return result;
}
struct Trial {
    size_t offset = 0, bytes = 0, registration_offset = 0, registration_bytes = 0;
    int round = 0;
    std::string variant;
    double register_ms = 0, h2d_submit_ms = 0, h2d_wall_ms = 0, h2d_event_ms = 0, unregister_ms = 0;
    bool exact = false, fence_before_unregister = false;
};
struct Report {
    std::string status = "failed", reason, device_name;
    int runtime = 0, driver = 0, device_count = 0, major = 0, minor = 0;
    size_t page = 0, allocation_granularity = 0, file_bytes = 0;
    std::vector<std::pair<std::string, int>> attributes;
    std::vector<std::string> cleanup_errors;
    std::vector<Trial> trials;
    uint64_t reference_fingerprint = 0;
    bool mapping_readonly = false, final_fence = false, final_unregistered = false, unmapped = false, self_test = false;
};
void write_report(const Options& o, const Report& r) {
    std::ostringstream s; s << std::setprecision(10);
    s << "{\n  \"status\": " << quote(r.status) << ",\n  \"reason\": " << quote(r.reason)
      << ",\n  \"scope\": \"Windows read-only generated-file registration microprobe; warm/repeated mappings, no model execution, no SSD-cache or inference-speed claim\""
      << ",\n  \"cpu_only_self_test\": " << (r.self_test ? "true" : "false")
      << ",\n  \"fixture\": " << quote(utf8(o.fixture.wstring()))
      << ",\n  \"file_bytes\": " << r.file_bytes << ",\n  \"page_bytes\": " << r.page
      << ",\n  \"allocation_granularity_bytes\": " << r.allocation_granularity
      << ",\n  \"mapping_protection\": \"PAGE_READONLY / FILE_MAP_READ\""
      << ",\n  \"mapping_protection_verified\": " << (r.mapping_readonly ? "true" : "false")
      << ",\n  \"registration_flags\": \"cudaHostRegisterReadOnly (0x08); no device mapping requested\""
      << ",\n  \"device_ordinal\": " << o.device << ",\n  \"device_count\": " << r.device_count
      << ",\n  \"device_name\": " << quote(r.device_name) << ",\n  \"compute_capability\": " << quote(std::to_string(r.major) + "." + std::to_string(r.minor))
      << ",\n  \"cuda_runtime_version\": " << r.runtime << ",\n  \"cuda_driver_version\": " << r.driver
      << ",\n  \"reference_fnv1a64\": " << quote(std::to_string(r.reference_fingerprint))
      << ",\n  \"verification\": " << quote(r.self_test
            ? "CPU mapping bytes compared against buffered ReadFile reference; no CUDA transfer tested"
            : "Every payload byte compared against buffered ReadFile reference after H2D/D2H; FNV is only a report fingerprint")
      << ",\n  \"capabilities\": {";
    for (size_t i = 0; i < r.attributes.size(); ++i) s << (i ? ", " : "") << quote(r.attributes[i].first) << ": " << r.attributes[i].second;
    s << "},\n  \"trials\": [";
    for (size_t i = 0; i < r.trials.size(); ++i) {
        const auto& t = r.trials[i];
        s << (i ? ",\n" : "\n") << "    {\"round\": " << t.round << ", \"variant\": " << quote(t.variant)
          << ", \"payload_offset\": " << t.offset << ", \"payload_bytes\": " << t.bytes
          << ", \"registration_offset\": " << t.registration_offset << ", \"registration_bytes\": " << t.registration_bytes
          << ", \"register_ms\": " << t.register_ms << ", \"h2d_submit_ms\": " << t.h2d_submit_ms
          << ", \"h2d_wall_ms\": " << t.h2d_wall_ms << ", \"h2d_cuda_event_ms\": " << t.h2d_event_ms
          << ", \"unregister_ms\": " << t.unregister_ms << ", \"exact_bytes\": " << (t.exact ? "true" : "false")
          << ", \"fence_before_unregister\": " << (t.fence_before_unregister ? "true" : "false") << "}";
    }
    s << "\n  ],\n  \"cleanup\": {\"stream_fence_succeeded\": " << (r.final_fence ? "true" : "false")
      << ", \"registration_released\": " << (r.final_unregistered ? "true" : "false")
      << ", \"mapping_unmapped_after_fence\": " << (r.unmapped ? "true" : "false") << ", \"errors\": [";
    for (size_t i = 0; i < r.cleanup_errors.size(); ++i) s << (i ? ", " : "") << quote(r.cleanup_errors[i]);
    s << "]}\n}\n";
    if (!o.output.parent_path().empty()) std::filesystem::create_directories(o.output.parent_path());
    // CREATE_NEW avoids truncating a prior report in an accidental repeated run.
    HANDLE file = CreateFileW(o.output.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    wincheck(file != INVALID_HANDLE_VALUE, "Create report");
    std::string text = s.str(); DWORD written = 0;
    bool ok = WriteFile(file, text.data(), DWORD(text.size()), &written, nullptr) && written == text.size();
    DWORD error = GetLastError(); CloseHandle(file); SetLastError(error); wincheck(ok, "Write report");
    std::cout << text;
}
struct Resources {
    HANDLE file = INVALID_HANDLE_VALUE, mapping = nullptr;
    unsigned char* view = nullptr;
    void* registered = nullptr;
    void* device = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    // Explicit cleanup keeps errors visible and never unmaps a possibly active DMA source.
    void cleanup(Report& r) {
        // No stream means this probe has submitted no asynchronous work. Capability
        // failures must remain unavailable without a spurious device-sync failure.
        bool fenced = stream == nullptr;
        auto remember = [&](cudaError_t code, const char* action) {
            if (code != cudaSuccess) r.cleanup_errors.push_back(cuda_error(code, action));
            return code == cudaSuccess;
        };
        if (stream) fenced = remember(cudaStreamSynchronize(stream), "Final stream fence");
        r.final_fence = fenced;
        if (!fenced) {
            r.cleanup_errors.push_back("DMA completion unconfirmed: explicit unregister/unmap skipped; process teardown owns remaining resources");
            return;
        }
        if (registered && remember(cudaHostUnregister(registered), "Cleanup unregister")) registered = nullptr;
        r.final_unregistered = registered == nullptr;
        if (start) { remember(cudaEventDestroy(start), "Destroy start event"); start = nullptr; }
        if (stop) { remember(cudaEventDestroy(stop), "Destroy stop event"); stop = nullptr; }
        if (device) { remember(cudaFree(device), "Free destination"); device = nullptr; }
        if (stream) { remember(cudaStreamDestroy(stream), "Destroy stream"); stream = nullptr; }
        if (view && !registered) {
            if (!UnmapViewOfFile(view)) r.cleanup_errors.push_back("UnmapViewOfFile failed: " + std::to_string(GetLastError()));
            else { view = nullptr; r.unmapped = true; }
        }
        if (mapping && !view) { if (!CloseHandle(mapping)) r.cleanup_errors.push_back("Close mapping failed"); mapping = nullptr; }
        if (file != INVALID_HANDLE_VALUE && !view) { if (!CloseHandle(file)) r.cleanup_errors.push_back("Close file failed"); file = INVALID_HANDLE_VALUE; }
    }
};
void create_fixture(const Options& o, size_t bytes) {
    if (!o.fixture.parent_path().empty()) std::filesystem::create_directories(o.fixture.parent_path());
    HANDLE file = CreateFileW(o.fixture.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    wincheck(file != INVALID_HANDLE_VALUE, "Create new fixture");
    try {
        std::vector<unsigned char> buffer(64 * 1024);
        for (size_t done = 0; done < bytes;) {
            DWORD count = DWORD(std::min(buffer.size(), bytes - done));
            for (DWORD i = 0; i < count; ++i) {
                uint64_t index = done + i;
                buffer[i] = static_cast<unsigned char>((index * 1315423911ULL) ^ (index >> 7) ^ (index >> 19) ^ 0xA5);
            }
            DWORD written = 0;
            wincheck(WriteFile(file, buffer.data(), count, &written, nullptr) && written == count, "Write fixture");
            done += count;
        }
        wincheck(FlushFileBuffers(file) != 0, "Flush fixture");
    } catch (...) { CloseHandle(file); throw; }
    wincheck(CloseHandle(file) != 0, "Close writable fixture handle");
}
void mapping_setup(const Options& o, Report& r, Resources& owner, std::vector<unsigned char>& reference) {
    SYSTEM_INFO info{}; GetSystemInfo(&info); r.page = info.dwPageSize; r.allocation_granularity = info.dwAllocationGranularity;
    if (r.page != 4096) throw ProbeError("This bounded Windows probe requires the expected 4 KiB system page size", true);
    r.file_bytes = round_up(o.payload_bytes + 2 * r.page, r.allocation_granularity);
    create_fixture(o, r.file_bytes);
    owner.file = CreateFileW(o.fixture.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    wincheck(owner.file != INVALID_HANDLE_VALUE, "Open read-only fixture");
    reference.resize(r.file_bytes); size_t done = 0;
    while (done < reference.size()) {
        DWORD read = 0, count = DWORD(std::min<size_t>(reference.size() - done, 1024 * 1024));
        wincheck(ReadFile(owner.file, reference.data() + done, count, &read, nullptr) && read == count, "Buffered reference ReadFile");
        done += read;
    }
    r.reference_fingerprint = fingerprint(reference.data(), reference.size());
    owner.mapping = CreateFileMappingW(owner.file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    wincheck(owner.mapping != nullptr, "Create read-only mapping");
    owner.view = static_cast<unsigned char*>(MapViewOfFile(owner.mapping, FILE_MAP_READ, 0, 0, r.file_bytes));
    wincheck(owner.view != nullptr, "Map read-only file view");
    size_t checked = 0;
    while (checked < r.file_bytes) {
        MEMORY_BASIC_INFORMATION region{};
        wincheck(VirtualQuery(owner.view + checked, &region, sizeof(region)) == sizeof(region), "Query mapped protection");
        if (region.State != MEM_COMMIT || region.Type != MEM_MAPPED || region.Protect != PAGE_READONLY || !region.RegionSize)
            throw ProbeError("Mapped fixture is not committed read-only file-backed memory");
        checked += std::min<size_t>(region.RegionSize, r.file_bytes - checked);
    }
    r.mapping_readonly = true;
    if (reinterpret_cast<uintptr_t>(owner.view) % r.page) throw ProbeError("Mapped base is not page aligned");
    if (!std::equal(reference.begin(), reference.end(), owner.view)) throw ProbeError("Mapped bytes differ from buffered reference");
    // CPU-only coverage of both exact page range and the unaligned expert-like payload envelope.
    auto aligned = range(0, o.payload_bytes, r.file_bytes, r.page);
    auto unaligned = range(139, o.payload_bytes - 321, r.file_bytes, r.page);
    if (aligned.begin || aligned.registered_bytes != round_up(o.payload_bytes, r.page) ||
        unaligned.begin || unaligned.registered_bytes != round_up(139 + o.payload_bytes - 321, r.page))
        throw ProbeError("Registration envelope self-test failed");
    auto later = range(r.page + 139, r.page, r.file_bytes, r.page);
    if (later.begin != r.page || later.registered_bytes != 2 * r.page) throw ProbeError("Nonzero registration base self-test failed");
}
void probe(const Options& o, Report& r, Resources& owner, const std::vector<unsigned char>& reference) {
    check(cudaRuntimeGetVersion(&r.runtime), "Query runtime version", true);
    check(cudaDriverGetVersion(&r.driver), "Query driver version", true);
    check(cudaGetDeviceCount(&r.device_count), "Enumerate CUDA devices", true);
    if (o.device >= r.device_count) throw ProbeError("Requested CUDA device unavailable", true);
    check(cudaSetDevice(o.device), "Select CUDA device", true);
    cudaDeviceProp properties{}; check(cudaGetDeviceProperties(&properties, o.device), "Query device properties", true);
    r.device_name = properties.name; r.major = properties.major; r.minor = properties.minor;
    auto attribute = [&](const char* name, cudaDeviceAttr attr) {
        int value = 0; check(cudaDeviceGetAttribute(&value, attr, o.device), name, true);
        r.attributes.emplace_back(name, value); return value;
    };
    int host_register = attribute("host_register_supported", cudaDevAttrHostRegisterSupported);
    int readonly_register = attribute("host_register_readonly_supported", cudaDevAttrHostRegisterReadOnlySupported);
    attribute("pageable_memory_access", cudaDevAttrPageableMemoryAccess);
    attribute("pageable_memory_access_uses_host_page_tables", cudaDevAttrPageableMemoryAccessUsesHostPageTables);
    attribute("unified_addressing", cudaDevAttrUnifiedAddressing);
    attribute("async_engine_count", cudaDevAttrAsyncEngineCount);
    if (!host_register || !readonly_register)
        throw ProbeError("Required read-only host registration capability is absent; stop registered-file-page tier design", true);
    check(cudaStreamCreateWithFlags(&owner.stream, cudaStreamNonBlocking), "Create copy stream");
    check(cudaEventCreate(&owner.start), "Create start event"); check(cudaEventCreate(&owner.stop), "Create stop event");
    check(cudaMalloc(&owner.device, o.payload_bytes), "Allocate bounded device destination");
    std::vector<unsigned char> actual(o.payload_bytes);
    std::vector<Range> ranges{range(0, o.payload_bytes, r.file_bytes, r.page),
        range(r.page + 139, o.payload_bytes - 321, r.file_bytes, r.page)};
    for (const auto& selection : ranges) for (int round = 0; round < o.rounds; ++round) {
        // Reverse the order across rounds. The buffered source and fixture creation warm file data;
        // these measurements make no cold-storage or OS cache residency claim.
        for (int step = 0; step < 3; ++step) {
            int variant = round % 2 ? 2 - step : step;
            Trial t; t.round = round; t.offset = selection.offset; t.bytes = selection.bytes;
            t.registration_offset = selection.begin; t.registration_bytes = selection.registered_bytes;
            t.variant = variant == 0 ? "buffered_reference" : variant == 1 ? "pageable_readonly_mapping" : "registered_readonly_mapping";
            r.trials.push_back(t); Trial& result = r.trials.back();
            check(cudaStreamSynchronize(owner.stream), "Fence before trial source change");
            const unsigned char* source = variant == 0 ? reference.data() + selection.offset : owner.view + selection.offset;
            if (variant == 2) {
                void* base = owner.view + selection.begin;
                if (reinterpret_cast<uintptr_t>(base) % r.page || selection.registered_bytes % r.page)
                    throw ProbeError("Registration address/length is not 4 KiB aligned");
                auto start = Clock::now();
                cudaError_t status = cudaHostRegister(base, selection.registered_bytes, cudaHostRegisterReadOnly);
                result.register_ms = milliseconds(start);
                check(status, "Register read-only file pages", true); owner.registered = base;
            }
            check(cudaEventRecord(owner.start, owner.stream), "Record H2D start");
            auto transfer_start = Clock::now();
            check(cudaMemcpyAsync(owner.device, source, selection.bytes, cudaMemcpyHostToDevice, owner.stream), "Submit exact H2D payload");
            result.h2d_submit_ms = milliseconds(transfer_start);
            check(cudaEventRecord(owner.stop, owner.stream), "Record H2D completion");
            check(cudaStreamSynchronize(owner.stream), "Fence H2D before verification/unregister");
            result.h2d_wall_ms = milliseconds(transfer_start); result.fence_before_unregister = true;
            float event_ms = 0; check(cudaEventElapsedTime(&event_ms, owner.start, owner.stop), "Read H2D event timing");
            result.h2d_event_ms = event_ms;
            check(cudaMemcpy(actual.data(), owner.device, selection.bytes, cudaMemcpyDeviceToHost), "Read exact bytes for verification");
            result.exact = std::equal(actual.begin(), actual.begin() + selection.bytes, reference.begin() + selection.offset);
            if (!result.exact) throw ProbeError("H2D/D2H payload differs from buffered reference");
            if (owner.registered) {
                // H2D has explicitly completed before unregister; the following D2H does not use the mapping.
                auto start = Clock::now(); check(cudaHostUnregister(owner.registered), "Unregister read-only file pages");
                result.unregister_ms = milliseconds(start); owner.registered = nullptr;
            }
        }
    }
    if (!std::equal(reference.begin(), reference.end(), owner.view)) throw ProbeError("Read-only fixture changed during probe");
    r.status = "complete"; r.reason = "Required capabilities present; every copied payload matched buffered reference exactly";
}
} // namespace

int wmain(int argc, wchar_t** argv) {
    Options o; Report report; Resources owner; bool parsed = false;
    try {
        o = options(argc, argv); parsed = true; report.self_test = o.self_test;
        std::vector<unsigned char> reference;
        mapping_setup(o, report, owner, reference);
        if (o.self_test) { report.status = "cpu_self_test_complete"; report.reason = "Read-only mapping, buffered reference and page-envelope checks passed; zero CUDA API calls"; }
        else probe(o, report, owner, reference);
    } catch (const ProbeError& error) { report.status = error.unavailable ? "unavailable" : "failed"; report.reason = error.what(); }
    catch (const std::exception& error) { report.status = "failed"; report.reason = error.what(); }
    owner.cleanup(report);
    if (!report.cleanup_errors.empty()) { report.status = "failed"; report.reason += "; cleanup failure"; }
    try { if (parsed) write_report(o, report); else std::cerr << report.reason << '\n'; }
    catch (const std::exception& error) { std::cerr << "Report failure: " << error.what() << '\n'; return 1; }
    return report.status == "complete" || report.status == "cpu_self_test_complete" ? 0 : report.status == "unavailable" ? 77 : 1;
}
