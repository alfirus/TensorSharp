// Standalone Windows service/ratio probe. No TensorSharp/ggml or model mutation.
// Compile with x64 MSVC and the Windows SDK; Cabinet/BCrypt links are embedded below.
#ifndef _WIN32
#error This probe requires the Windows Cabinet Compression API.
#endif
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0602
#endif
#include <windows.h>
#include <compressapi.h>
#include <bcrypt.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#pragma comment(lib, "Cabinet.lib")
#pragma comment(lib, "Bcrypt.lib")

namespace {
constexpr std::size_t MiB = 1024u * 1024u;
constexpr std::size_t max_input = 64u * MiB;
constexpr std::size_t max_encoded = 129u * MiB;
constexpr std::size_t workspace_limit = 64u * MiB; // Both handles combined.
using Clock = std::chrono::steady_clock;
[[noreturn]] void fail(const std::string& text) { throw std::runtime_error(text); }
void require(bool condition, const std::string& text) { if (!condition) fail(text); }
void win(bool condition, const char* operation) {
    if (!condition) {
        const auto error = GetLastError();
        fail(std::string(operation) + " Win32=" + std::to_string(error));
    }
}
double millis(Clock::time_point from) {
    return std::chrono::duration<double, std::milli>(Clock::now() - from).count();
}
std::string utf8(const std::wstring& value) {
    if (value.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
        static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
    win(n > 0, "WideCharToMultiByte size");
    std::string result(static_cast<std::size_t>(n), '\0');
    win(WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
        static_cast<int>(value.size()), result.data(), n, nullptr, nullptr) == n,
        "WideCharToMultiByte");
    return result;
}
std::string json(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << static_cast<char>(c);
        else if (c < 0x20) out << "\\u" << std::hex << std::setw(4)
            << std::setfill('0') << static_cast<unsigned>(c) << std::dec;
        else out << static_cast<char>(c);
    }
    return out.str() + '"';
}
std::uint64_t decimal(const std::wstring& text) {
    require(!text.empty(), "Empty unsigned decimal argument");
    require(text.size() == 1 || text[0] != L'0', "Unsigned decimal must not have leading zeros");
    std::uint64_t result = 0;
    for (wchar_t c : text) {
        require(c >= L'0' && c <= L'9', "Expected canonical unsigned decimal argument");
        auto digit = static_cast<unsigned>(c - L'0');
        require(result <= (UINT64_MAX - digit) / 10, "Unsigned decimal overflow");
        result = result * 10 + digit;
    }
    return result;
}
struct Args {
    std::wstring file, label;
    std::uint64_t offset = 0, bytes = 0;
    unsigned repeat = 8, warmup = 1;
};
Args arguments(int argc, wchar_t** argv) {
    Args a;
    std::set<std::wstring> seen;
    for (int i = 1; i < argc; ++i) {
        std::wstring key = argv[i];
        require(seen.insert(key).second && i + 1 < argc, "Duplicate argument or missing value");
        std::wstring value = argv[++i];
        if (key == L"--file") a.file = value;
        else if (key == L"--offset") a.offset = decimal(value);
        else if (key == L"--bytes") a.bytes = decimal(value);
        else if (key == L"--type") a.label = value;
        else if (key == L"--repeat") {
            auto n = decimal(value); require(n >= 1 && n <= 64, "repeat must be 1..64");
            a.repeat = static_cast<unsigned>(n);
        } else if (key == L"--warmup") {
            auto n = decimal(value); require(n <= 4, "warmup must be 0..4");
            a.warmup = static_cast<unsigned>(n);
        } else fail("Unknown command line argument");
    }
    require(seen.count(L"--file") && seen.count(L"--offset") && seen.count(L"--bytes"),
        "Required: --file PATH --offset DECIMAL --bytes DECIMAL [--type LABEL] [--repeat 1..64] [--warmup 0..4]");
    require(!a.file.empty() && a.label.size() <= 128, "Invalid path or excessive type label");
    require(a.bytes > 0 && a.bytes <= max_input, "Input slice must be 1..64MiB");
    return a;
}
struct File {
    HANDLE handle = INVALID_HANDLE_VALUE;
    ~File() {
        if (handle != INVALID_HANDLE_VALUE && !CloseHandle(handle)) {
            std::cerr << "FAIL cleanup file handle could not close; no cleanup qualification\n";
            ExitProcess(2); // Do not discard an uncertain live owner then continue.
        }
    }
    void close() {
        if (handle != INVALID_HANDLE_VALUE) {
            win(CloseHandle(handle) != FALSE, "CloseHandle input");
            handle = INVALID_HANDLE_VALUE;
        }
    }
};
BY_HANDLE_FILE_INFORMATION information(HANDLE file) {
    BY_HANDLE_FILE_INFORMATION i{};
    win(GetFileInformationByHandle(file, &i) != FALSE, "GetFileInformationByHandle");
    require(!(i.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY), "Input must be a regular file");
    return i;
}
bool same_file(const BY_HANDLE_FILE_INFORMATION& a, const BY_HANDLE_FILE_INFORMATION& b) {
    return a.dwVolumeSerialNumber == b.dwVolumeSerialNumber && a.nFileIndexHigh == b.nFileIndexHigh
        && a.nFileIndexLow == b.nFileIndexLow && a.nFileSizeHigh == b.nFileSizeHigh
        && a.nFileSizeLow == b.nFileSizeLow
        && CompareFileTime(&a.ftLastWriteTime, &b.ftLastWriteTime) == 0;
}
std::string final_path(HANDLE file) {
    DWORD required = GetFinalPathNameByHandleW(file, nullptr, 0, FILE_NAME_NORMALIZED);
    win(required > 0 && required < 32768, "GetFinalPathNameByHandleW size");
    std::vector<wchar_t> path(static_cast<std::size_t>(required) + 1);
    DWORD n = GetFinalPathNameByHandleW(file, path.data(), static_cast<DWORD>(path.size()), FILE_NAME_NORMALIZED);
    win(n > 0 && n < path.size(), "GetFinalPathNameByHandleW");
    return utf8(std::wstring(path.data(), n));
}
struct Hash {
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    ~Hash() {
        if (hash && BCryptDestroyHash(hash) < 0) ExitProcess(2);
        if (algorithm && BCryptCloseAlgorithmProvider(algorithm, 0) < 0) ExitProcess(2);
    }
};
std::string sha256(const unsigned char* data, std::size_t bytes) {
    std::vector<unsigned char> object; // Must outlive Hash on all exception paths.
    Hash h;
    require(BCryptOpenAlgorithmProvider(&h.algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) >= 0, "BCryptOpenAlgorithmProvider");
    DWORD object_bytes = 0, returned = 0, hash_bytes = 0;
    require(BCryptGetProperty(h.algorithm, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&object_bytes),
        sizeof(object_bytes), &returned, 0) >= 0 && returned == sizeof(object_bytes), "SHA256 object length");
    require(object_bytes > 0 && object_bytes <= MiB, "SHA256 object workspace exceeds bound");
    require(BCryptGetProperty(h.algorithm, BCRYPT_HASH_LENGTH, reinterpret_cast<PUCHAR>(&hash_bytes),
        sizeof(hash_bytes), &returned, 0) >= 0 && hash_bytes == 32, "SHA256 digest length");
    object.resize(object_bytes);
    require(BCryptCreateHash(h.algorithm, &h.hash, object.data(), object_bytes, nullptr, 0, 0) >= 0, "BCryptCreateHash");
    require(BCryptHashData(h.hash, const_cast<PUCHAR>(data), static_cast<ULONG>(bytes), 0) >= 0, "BCryptHashData");
    unsigned char digest[32]{};
    require(BCryptFinishHash(h.hash, digest, sizeof(digest), 0) >= 0, "BCryptFinishHash");
    std::ostringstream out;
    for (auto c : digest) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(c);
    // Hash needs its backing object until DestroyHash, before vector destruction.
    require(BCryptDestroyHash(h.hash) >= 0, "BCryptDestroyHash"); h.hash = nullptr;
    require(BCryptCloseAlgorithmProvider(h.algorithm, 0) >= 0, "BCryptCloseAlgorithmProvider"); h.algorithm = nullptr;
    return out.str();
}
struct Workspace {
    struct alignas(std::max_align_t) Header { Workspace* owner; std::size_t charge; };
    static_assert(alignof(Header) <= MEMORY_ALLOCATION_ALIGNMENT, "Heap alignment is insufficient");
    std::mutex gate;
    std::size_t live = 0, peak = 0, allocations = 0, frees = 0, refusals = 0;
    std::atomic<bool> failed{false};
    static PVOID __cdecl allocate(PVOID context, SIZE_T bytes) noexcept {
        auto& w = *static_cast<Workspace*>(context);
        try {
            std::lock_guard<std::mutex> lock(w.gate);
            if (bytes > workspace_limit - sizeof(Header) || w.live > workspace_limit - bytes - sizeof(Header)) {
                ++w.refusals; return nullptr;
            }
            auto charge = bytes + sizeof(Header);
            auto* h = static_cast<Header*>(HeapAlloc(GetProcessHeap(), 0, charge));
            if (!h) { ++w.refusals; return nullptr; }
            h->owner = &w; h->charge = charge;
            w.live += charge; w.peak = (std::max)(w.peak, w.live); ++w.allocations;
            return h + 1;
        } catch (...) { w.failed.store(true); return nullptr; }
    }
    static VOID __cdecl release(PVOID context, PVOID memory) noexcept {
        if (!memory) return;
        auto& w = *static_cast<Workspace*>(context);
        try {
            std::lock_guard<std::mutex> lock(w.gate);
            auto* h = static_cast<Header*>(memory) - 1;
            if (h->owner != &w || h->charge > w.live) { w.failed.store(true); return; }
            auto charge = h->charge;
            if (!HeapFree(GetProcessHeap(), 0, h)) { w.failed.store(true); return; }
            w.live -= charge; ++w.frees;
        } catch (...) { w.failed.store(true); }
    }
};
struct Codec {
    Workspace workspace; // Outlives both handles and their callback invocations.
    COMPRESSOR_HANDLE compressor = nullptr;
    DECOMPRESSOR_HANDLE decompressor = nullptr;
    void close() noexcept {
        if (decompressor) {
            if (!CloseDecompressor(decompressor)) {
                std::cerr << "FAIL CloseDecompressor; owners retained until process exit\n"; ExitProcess(2);
            }
            decompressor = nullptr;
        }
        if (compressor) {
            if (!CloseCompressor(compressor)) {
                std::cerr << "FAIL CloseCompressor; owners retained until process exit\n"; ExitProcess(2);
            }
            compressor = nullptr;
        }
        if (workspace.failed.load() || workspace.live != 0 || workspace.allocations != workspace.frees) {
            std::cerr << "FAIL Codec workspace cleanup incomplete\n"; ExitProcess(2);
        }
    }
    ~Codec() { close(); }
    void open(DWORD algorithm) {
        COMPRESS_ALLOCATION_ROUTINES allocation{Workspace::allocate, Workspace::release, &workspace};
        win(CreateCompressor(algorithm, &allocation, &compressor) != FALSE, "CreateCompressor");
        win(CreateDecompressor(algorithm, &allocation, &decompressor) != FALSE, "CreateDecompressor");
    }
};
struct Timing {
    double sum = 0, minimum = (std::numeric_limits<double>::max)(), maximum = 0;
    unsigned count = 0;
    void add(double ms) { sum += ms; minimum = (std::min)(minimum, ms); maximum = (std::max)(maximum, ms); ++count; }
    void print(const char* name) const {
        std::cout << ',' << json(name) << ":{\"count\":" << count << ",\"mean_ms\":" << sum / count
            << ",\"min_ms\":" << minimum << ",\"max_ms\":" << maximum << '}';
    }
};
__declspec(noinline) void raw_copy(void* to, const void* from, std::size_t bytes) { std::memcpy(to, from, bytes); }
void run_codec(DWORD algorithm, const char* name, const Args& args, const unsigned char* raw) {
    std::unique_ptr<unsigned char[]> encoded, decoded; // Outlive Codec during exception cleanup.
    Codec codec;
    auto started = Clock::now(); codec.open(algorithm); double create_ms = millis(started);
    SIZE_T capacity = 0;
    started = Clock::now();
    BOOL queried = Compress(codec.compressor, raw, static_cast<SIZE_T>(args.bytes), nullptr, 0, &capacity);
    DWORD query_error = queried ? ERROR_SUCCESS : GetLastError();
    double query_ms = millis(started);
    require(!queried && query_error == ERROR_INSUFFICIENT_BUFFER && capacity > 0 && capacity <= max_encoded,
        "Compress size query failed or exceeded 129MiB encoded capacity bound");
    encoded = std::make_unique<unsigned char[]>(capacity);
    decoded = std::make_unique<unsigned char[]>(static_cast<std::size_t>(args.bytes));
    Timing compress, decompress, compare, copy;
    double first_compress = 0, first_decompress = 0;
    SIZE_T encoded_min = SIZE_MAX, encoded_max = 0;
    for (unsigned iteration = 0; iteration < 1 + args.warmup + args.repeat; ++iteration) {
        SIZE_T encoded_bytes = 0, decoded_bytes = 0;
        started = Clock::now();
        win(ResetCompressor(codec.compressor) != FALSE, "ResetCompressor independent slice");
        win(Compress(codec.compressor, raw, static_cast<SIZE_T>(args.bytes), encoded.get(), capacity, &encoded_bytes) != FALSE,
            "Compress actual buffer");
        double compression_ms = millis(started);
        require(encoded_bytes > 0 && encoded_bytes <= capacity, "Compressed output length outside allocated capacity");
        std::memset(decoded.get(), 0xa5, static_cast<std::size_t>(args.bytes)); // Untimed output poison.
        started = Clock::now();
        win(ResetDecompressor(codec.decompressor) != FALSE, "ResetDecompressor independent slice");
        win(Decompress(codec.decompressor, encoded.get(), encoded_bytes, decoded.get(),
            static_cast<SIZE_T>(args.bytes), &decoded_bytes) != FALSE, "Decompress");
        double decompression_ms = millis(started);
        require(decoded_bytes == args.bytes, "Decompress returned a different raw byte length");
        started = Clock::now();
        require(std::memcmp(raw, decoded.get(), static_cast<std::size_t>(args.bytes)) == 0,
            "Decompressed bytes differ from original exact slice");
        double comparison_ms = millis(started);
        require(!codec.workspace.failed.load() && codec.workspace.refusals == 0, "Codec allocation callback failed/refused");
        encoded_min = (std::min)(encoded_min, encoded_bytes); encoded_max = (std::max)(encoded_max, encoded_bytes);
        if (iteration == 0) { first_compress = compression_ms; first_decompress = decompression_ms; }
        if (iteration >= 1 + args.warmup) {
            compress.add(compression_ms); decompress.add(decompression_ms); compare.add(comparison_ms);
            std::memset(decoded.get(), 0xa5, static_cast<std::size_t>(args.bytes));
            started = Clock::now(); raw_copy(decoded.get(), raw, static_cast<std::size_t>(args.bytes)); copy.add(millis(started));
            require(std::memcmp(raw, decoded.get(), static_cast<std::size_t>(args.bytes)) == 0, "Raw copy baseline differs");
        }
    }
    codec.close();
    std::cout << "{\"event\":\"codec\",\"name\":" << json(name) << ",\"mode\":\"buffer_default_level\",\"includes_stream_reset\":true"
        << ",\"raw_bytes\":" << args.bytes << ",\"encoded_bytes_min\":" << encoded_min
        << ",\"encoded_bytes_max\":" << encoded_max << ",\"encoded_capacity_bytes\":" << capacity
        << ",\"encoded_max_over_raw\":" << double(encoded_max) / double(args.bytes)
        << ",\"declared_buffer_bytes_with_raw\":" << args.bytes * 2 + capacity
        << ",\"workspace_limit_bytes\":" << workspace_limit << ",\"workspace_peak_requested_with_headers\":" << codec.workspace.peak
        << ",\"workspace_live_after_close\":" << codec.workspace.live << ",\"workspace_allocations\":" << codec.workspace.allocations
        << ",\"workspace_frees\":" << codec.workspace.frees << ",\"workspace_refusals\":" << codec.workspace.refusals
        << ",\"create_ms\":" << create_ms << ",\"size_query_ms\":" << query_ms
        << ",\"first_compress_ms\":" << first_compress << ",\"first_decompress_ms\":" << first_decompress
        << ",\"warmup_pairs_excluded\":" << args.warmup << ",\"exact_roundtrip_pairs\":" << 1 + args.warmup + args.repeat;
    compress.print("warm_compress"); decompress.print("warm_decompress"); compare.print("warm_compare"); copy.print("warm_raw_memcpy");
    std::cout << ",\"cleanup_checked\":true,\"scope\":\"one_exact_slice_CPU_service_not_inference_or_SSD\"}\n" << std::flush;
}
} // namespace

int wmain(int argc, wchar_t** argv) {
    try {
        const auto args = arguments(argc, argv);
        std::cout << std::fixed << std::setprecision(9);
        File file;
        file.handle = CreateFileW(args.file.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
        win(file.handle != INVALID_HANDLE_VALUE, "CreateFileW readonly restricted input");
        require(GetFileType(file.handle) == FILE_TYPE_DISK, "Input must be a disk file");
        const auto before = information(file.handle);
        const std::uint64_t file_bytes = (std::uint64_t(before.nFileSizeHigh) << 32) | before.nFileSizeLow;
        require(args.offset <= file_bytes && args.bytes <= file_bytes - args.offset
            && args.offset <= static_cast<std::uint64_t>((std::numeric_limits<LONGLONG>::max)()),
            "Requested slice exceeds exact opened file bounds");
        LARGE_INTEGER offset{}; offset.QuadPart = static_cast<LONGLONG>(args.offset);
        win(SetFilePointerEx(file.handle, offset, nullptr, FILE_BEGIN) != FALSE, "SetFilePointerEx");
        auto raw = std::make_unique<unsigned char[]>(static_cast<std::size_t>(args.bytes));
        auto started = Clock::now();
        for (std::size_t done = 0; done < args.bytes;) {
            DWORD count = static_cast<DWORD>((std::min)(args.bytes - done, std::uint64_t(MiB)));
            DWORD read = 0;
            win(ReadFile(file.handle, raw.get() + done, count, &read, nullptr) != FALSE, "ReadFile exact slice");
            require(read == count, "Short read from exact slice"); done += read;
        }
        double read_ms = millis(started);
        const auto hash = sha256(raw.get(), static_cast<std::size_t>(args.bytes));
        std::cout << "{\"event\":\"input\",\"file\":" << json(final_path(file.handle)) << ",\"type_label\":" << json(utf8(args.label))
            << ",\"offset\":" << args.offset << ",\"raw_bytes\":" << args.bytes << ",\"file_bytes\":" << file_bytes
            << ",\"volume_serial\":" << before.dwVolumeSerialNumber << ",\"file_index_high\":" << before.nFileIndexHigh
            << ",\"file_index_low\":" << before.nFileIndexLow << ",\"last_write_high\":" << before.ftLastWriteTime.dwHighDateTime
            << ",\"last_write_low\":" << before.ftLastWriteTime.dwLowDateTime << ",\"slice_sha256\":" << json(hash)
            << ",\"repeat\":" << args.repeat << ",\"warmup\":" << args.warmup << ",\"first_read_ms\":" << read_ms
            << ",\"provenance\":\"partial_exact_opened_slice_not_whole_model_SHA\"}\n" << std::flush;
        run_codec(COMPRESS_ALGORITHM_XPRESS, "xpress", args, raw.get());
        run_codec(COMPRESS_ALGORITHM_XPRESS_HUFF, "xpress_huff", args, raw.get());
        require(same_file(before, information(file.handle)), "Input identity/size/write time changed while readonly handle was held");
        file.close(); raw.reset();
        std::cout << "{\"event\":\"complete\",\"passed\":true,\"codecs\":2,\"exact_input_handle_closed\":true,"
            "\"workspace_live\":0,\"limitations\":\"No model/GPU/H2D/physical_IO/residency/capacity/speedup proof; "
            "no cold-cache isolation; callback bytes exclude Heap allocator overhead and OS/internal global allocations; "
            "first API size query may warm compressor; warm memcpy/decode order fixed; no fairness claim; exit0 required\"}\n" << std::flush;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL expert lossless compression probe: " << e.what() << '\n';
        return 1;
    }
}
