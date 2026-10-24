// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Standalone CPU fixture; Windows quota changes run only in its own child.
#include "ggml_ops_host_moe_mapped_ram.h"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

namespace {
using Backend = tsg::HostMoeMappedRam;
using Cache = tsg::HostMoeImmutableCache;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
int checks = 0;
void geometry_checks() {
    auto plan = [](std::uint64_t base, std::uint64_t extent, std::uint64_t file,
                   std::uint64_t relative, std::size_t count) {
        ++checks; return Backend::checked_geometry(base, extent, file, relative, count, 4096, 65536);
    };
    auto g = plan(17, 200000, 262144, 65535, 4097);
    require(g.valid && g.map_offset == 65536 && g.data_delta == 16 && g.lock_delta == 0
        && g.map_bytes == 4113 && g.pin_bytes == 8192 && g.charge_bytes == 12288, "Unaligned geometry differs");
    g = plan((std::uint64_t(1) << 32) + 17, 65536, (std::uint64_t(1) << 32) + 131072, 0, 8192);
    require(g.valid && g.map_offset == (std::uint64_t(1) << 32) && g.data_delta == 17
        && g.pin_bytes == 12288, "64-bit file offset was narrowed");
    require(!plan(0, 8192, 8192, 8192, 1).valid, "Source overrun admitted");
    require(!plan(0, 8192, 8192, 0, 0).valid, "Zero extent admitted");
    require(!plan(UINT64_MAX - 1, 8, UINT64_MAX, 0, 1).valid, "Registered overflow admitted");
    require(!plan(17, 8192, 8209, 0, 8192).valid, "Unqualified EOF padding admitted");
    require(!plan(0, 8192, 4096, 0, 1).valid, "File overrun admitted");
    auto a = plan(17, 8192, 16384, 0, 4096), b = plan(17, 8192, 16384, 4096, 4096);
    require(a.valid && b.valid && a.pin_bytes == 8192 && b.pin_bytes == 8192
        && a.charge_bytes + b.charge_bytes == 24576, "Adjacent file-page overlap deducted guessed sharing");
    require(!Backend::checked_geometry(0, 8192, 8192, 0, 1, 3000, 65536).valid, "Invalid page geometry admitted");
}
#if defined(_WIN32) && defined(_MSC_VER)
constexpr std::size_t payload = std::size_t(16) << 20, file_bytes = payload + 4096;
std::uint8_t pattern(std::uint64_t i) { return static_cast<std::uint8_t>((i * 37 + (i >> 12) * 13 + 7) % 251); }
std::shared_ptr<void> token(int& live) {
    ++live; return std::shared_ptr<void>(new int(1), [&live](void* p) { delete static_cast<int*>(p); --live; });
}
struct Quota { SIZE_T low = 0, high = 0; DWORD flags = 0; };
Quota current_quota() {
    Quota q; require(GetProcessWorkingSetSizeEx(GetCurrentProcess(), &q.low, &q.high, &q.flags) != 0, "Cannot query child quota"); return q;
}
bool same(const Quota& a, const Quota& b) { return a.low == b.low && a.high == b.high && a.flags == b.flags; }
void set_quota(const Quota& q) {
    require(SetProcessWorkingSetSizeEx(GetCurrentProcess(), q.low, q.high, q.flags) != 0, "Cannot change own child quota");
}
void exact(const Cache::Storage& storage) {
    std::array<std::uint8_t, 8192> bytes{};
    for (std::size_t offset = 0; offset < storage.bytes;) {
        const auto count = (std::min)(bytes.size(), storage.bytes - offset);
        require(Backend::guarded_copy(bytes.data(), storage.data + offset, count), "Guarded mapped copy failed");
        for (std::size_t i = 0; i < count; ++i) require(bytes[i] == pattern(17 + offset + i), "Mapped immutable bytes differ");
        offset += count;
    }
    require(Backend::test_storage_locked(storage), "Storage lost its complete locked envelope");
}
void windows_child(const std::filesystem::path& path) {
    auto& backend = Backend::instance();
    const auto original = current_quota();
    Backend::Options options; options.exclusive_quota = true; options.quota_ceiling_bytes = std::size_t(512) << 20;
    auto source = std::make_unique<tsg::ExpertFileSource>(path.u8string().c_str(), 17, payload);
    const Cache::Range range{source->identity(), 0, payload};
    const auto geometry = backend.plan(*source, 0, payload);
    require(geometry.valid && geometry.pin_bytes == payload + 4096 && geometry.charge_bytes == payload + 8192,
        "Actual registered source geometry differs");
    {
        int control_live = 0, entry_live = 0;
        auto control = token(control_live); require(backend.configure({}, control), "Existing-quota mode unavailable");
        auto life = token(entry_live); auto storage = backend.create(*source, range, life); life.reset();
        std::printf("default quota: min=%llu max=%llu full lock=%s (refusal is not residency)\n",
            static_cast<unsigned long long>(original.low), static_cast<unsigned long long>(original.high), storage.owner ? "supported" : "refused");
        if (storage.owner) exact(storage);
        storage = {}; require(entry_live == 0 && backend.finish_zero(), "Default lock cleanup retained ownership");
        control.reset(); require(control_live == 0 && same(original, current_quota()), "Existing-quota mode changed limits");
    }
    for (auto fault : {Backend::TestFault::allocation, Backend::TestFault::duplicate, Backend::TestFault::section,
        Backend::TestFault::map, Backend::TestFault::query, Backend::TestFault::lock,
        Backend::TestFault::locked_query, Backend::TestFault::quota_set}) {
        int controls = 0, entries = 0; auto control = token(controls), life = token(entries);
        require(backend.configure(options, control), "Cannot configure fault fixture");
        Backend::test_fail_next(fault);
        auto storage = backend.create(*source, range, life); life.reset();
        require(!storage.owner && entries == 0 && backend.stats().owned_entries == 0, "Refused factory published or retained a cleanable charge");
        require(backend.finish_zero() && same(original, current_quota()), "Factory refusal failed quota cleanup");
        control.reset(); require(controls == 0, "Factory refusal retained control charge"); ++checks;
    }
    {
        // Keep an anchor claim alive while replacing a large view with a smaller
        // one. Only an actual historical Host-backed claim may raise the floor.
        int controls = 0, entries = 0; auto control = token(controls), anchor_life = token(entries);
        require(backend.configure(options, control), "Cannot configure high-water fixture");
        const Cache::Range anchor_range{source->identity(), 0, 4096};
        auto anchor = backend.create(*source, anchor_range, anchor_life); anchor_life.reset();
        require(bool(anchor.owner), "High-water anchor lock unavailable");
        const auto before = backend.stats();
        auto large_life = token(entries); auto large = backend.create(*source, range, large_life); large_life.reset();
        require(bool(large.owner), "High-water material claim unavailable");
        const auto high = current_quota(); const auto after = backend.stats();
        const auto expected_peak = geometry.pin_bytes + backend.plan(*source, 0, 4096).pin_bytes;
        require(high.low == (std::max)(original.low, expected_peak + options.quota_headroom_bytes)
            && high.high <= options.quota_ceiling_bytes, "High-water quota exceeded actual past claims/headroom or ceiling");
        require(after.admission_calls == before.admission_calls + 1
            && after.admission_successes == before.admission_successes + 1
            && after.map_calls == before.map_calls + 1 && after.lock_calls == before.lock_calls + 1
            && after.duplicate_calls == before.duplicate_calls + 1 && after.section_calls == before.section_calls + 1
            && after.page_query_calls == before.page_query_calls + (geometry.pin_bytes / 4096 + 63) / 64
            && after.admission_nanoseconds > before.admission_nanoseconds
            && after.lock_nanoseconds > before.lock_nanoseconds && after.page_query_nanoseconds > before.page_query_nanoseconds,
            "Admission telemetry omitted or duplicated actual Windows API work");
        large = {}; require(entries == 1 && current_quota().low == high.low, "Nonfinal retirement shrank verified quota");
        const auto smaller_before = backend.stats();
        auto small_life = token(entries);
        auto small = backend.create(*source, {source->identity(), 4096, 4096}, small_life); small_life.reset();
        const auto smaller_after = backend.stats();
        require(bool(small.owner) && current_quota().low == high.low
            && smaller_after.quota_write_calls == smaller_before.quota_write_calls
            && smaller_after.quota_reuse_claims == smaller_before.quota_reuse_claims + 1,
            "Smaller replacement rewrote a sufficient verified quota high-water");
        require(smaller_after.quota_min_high_water_bytes <= (std::max)(static_cast<std::uint64_t>(original.low),
            smaller_after.peak_claim_bytes + options.quota_headroom_bytes), "Telemetry high-water gained unbacked slack");
        small = {}; require(current_quota().low == high.low && entries == 1, "Remaining anchor lost high-water entitlement");
        anchor = {}; require(entries == 0 && same(original, current_quota()) && backend.finish_zero(), "Last claim did not restore original quota");
        const auto finished = backend.stats(); control.reset(); require(controls == 0, "High-water scope retained control charge");
        require(finished.admission_calls >= smaller_after.admission_calls
            && finished.quota_restore_writes > smaller_after.quota_restore_writes, "Final cleanup reset cumulative telemetry");
        // A smaller explicit ceiling must refuse before writing, never repurpose
        // the previous scope's historical entitlement or reset its counters.
        auto limited = options; limited.quota_ceiling_bytes = limited.quota_headroom_bytes;
        control = token(controls); auto life = token(entries);
        require(backend.configure(limited, control), "Cannot configure bounded-ceiling refusal");
        require(backend.stats().admission_calls == finished.admission_calls, "New scope reset process-lifetime telemetry");
        require(!backend.create(*source, range, life).owner, "Explicit ceiling admitted an oversized claim"); life.reset();
        require(entries == 0 && backend.stats().quota_write_calls == finished.quota_write_calls
            && same(original, current_quota()) && backend.finish_zero(), "Ceiling refusal changed quota or retained ownership");
        control.reset(); require(controls == 0, "Ceiling refusal retained control metadata"); ++checks;
    }
    {
        int controls = 0, entries = 0; auto control = token(controls), life = token(entries);
        require(backend.configure(options, control) && backend.configure(options, control), "Same-owner configure is not idempotent");
        auto foreign = token(controls); require(!backend.configure(options, foreign), "Second control owner admitted"); foreign.reset();
        auto changed = options; ++changed.quota_ceiling_bytes;
        require(!backend.configure(changed, control), "Changed options admitted into active broker");
        auto storage = backend.create(*source, range, life); life.reset(); require(bool(storage.owner), "Bounded quota lock unavailable");
        exact(storage);
        // A separate model-like view may trim/unmap while the cache view remains leased.
        HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        require(file != INVALID_HANDLE_VALUE, "Cannot open model-like view");
        HANDLE section = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        require(section != nullptr, "Cannot create model-like section");
        void* model = MapViewOfFile(section, FILE_MAP_READ, 0, 0, file_bytes);
        require(model != nullptr, "Cannot map model-like view");
        SetLastError(0); require(!VirtualUnlock(model, file_bytes) && GetLastError() == ERROR_NOT_LOCKED,
            "Unlocked original view did not report ERROR_NOT_LOCKED"); exact(storage);
        require(UnmapViewOfFile(model) && CloseHandle(section) && CloseHandle(file), "Original-view cleanup failed"); exact(storage);
        std::array<std::uint8_t, 64> destination{};
        const auto copy_before = backend.stats();
        Backend::test_fail_next(Backend::TestFault::copy);
        require(!Backend::guarded_copy(destination.data(), storage.data, destination.size()), "Injected partial copy did not refuse");
        source->read(0, destination.data(), destination.size());
        for (std::size_t i = 0; i < destination.size(); ++i) require(destination[i] == pattern(17 + i), "Exact buffered fallback did not overwrite partial copy");
        Backend::test_fail_next(Backend::TestFault::in_page);
        require(!Backend::guarded_copy(destination.data(), storage.data, destination.size()), "In-page SEH was not contained");
        require(!Backend::guarded_copy(destination.data(), reinterpret_cast<void*>(1), destination.size()), "Access violation escaped guarded copy");
        const auto copy_after = backend.stats();
        require(copy_after.guarded_copy_calls == copy_before.guarded_copy_calls + 3
            && copy_after.guarded_copy_failures == copy_before.guarded_copy_failures + 3
            && copy_after.guarded_copy_bytes == copy_before.guarded_copy_bytes
            && copy_after.guarded_copy_nanoseconds > copy_before.guarded_copy_nanoseconds,
            "Failed/partial guarded-copy telemetry claimed successful bytes or omitted cost");
        auto lease = storage; storage = {}; control.reset();
        require(!backend.finish_zero() && entries == 1 && controls == 1, "Live lease allowed quota/control refund");
        require(!backend.configure(options, lease.owner), "Stopped broker resumed admissions");
        lease = {}; require(entries == 0 && backend.finish_zero() && controls == 0, "Last lease did not retire before refund");
        require(same(original, current_quota()), "Lease teardown did not restore quota flags/limits"); ++checks;
    }
    for (auto fault : {Backend::TestFault::unmap, Backend::TestFault::close_section,
                       Backend::TestFault::close_file, Backend::TestFault::restore, Backend::TestFault::restore_query}) {
        int controls = 0, entries = 0; auto control = token(controls), life = token(entries);
        require(backend.configure(options, control), "Cannot configure quarantine fixture");
        auto storage = backend.create(*source, range, life); life.reset(); require(bool(storage.owner), "Quarantine setup lock failed");
        Backend::test_fail_next(fault); storage = {};
        auto stats = backend.stats(); require(entries == 1 && stats.quarantine_charge_bytes == geometry.charge_bytes
            && stats.owned_entries == 1, "Failed cleanup refunded its resource/quota charge");
        if (fault == Backend::TestFault::restore_query)
            require(same(original, current_quota()) && stats.quota_restore_pending,
                "Successful restore write/readback failure did not retain retryable ownership");
        auto bypass_life = token(entries);
        require(!backend.create(*source, range, bypass_life).owner, "Quarantine admitted a new owner"); bypass_life.reset();
        require(backend.retry_cleanup() && entries == 0 && backend.finish_zero(), "Quarantine was not retryable");
        control.reset(); require(controls == 0 && same(original, current_quota()), "Quarantine cleanup retained/overwrote quota"); ++checks;
    }
    {
        int controls = 0, entries = 0; auto control = token(controls), life = token(entries);
        require(backend.configure(options, control), "Cannot configure conflict fixture");
        auto storage = backend.create(*source, range, life); life.reset(); require(bool(storage.owner), "Conflict setup lock failed");
        const auto expected = current_quota(); auto foreign = expected; foreign.high += 4096; set_quota(foreign);
        auto another = token(entries); require(!backend.create(*source, range, another).owner, "Foreign quota write was overwritten"); another.reset();
        require(backend.stats().quota_conflict && same(foreign, current_quota()), "Foreign quota conflict was not preserved");
        storage = {}; require(entries == 1 && !backend.finish_zero(), "Conflicted restore refunded its charge");
        // Explicit child test restores the broker's exact last write; retry may now finish cleanup.
        set_quota(expected); require(backend.retry_cleanup() && entries == 0 && backend.finish_zero(), "Matched retry did not restore baseline");
        control.reset(); require(controls == 0 && same(original, current_quota()), "Conflict cleanup lost original flags"); ++checks;
    }
    {
        int controls = 0, entries = 0; auto control = token(controls), life = token(entries);
        require(backend.configure(options, control), "Cannot configure duplicate-source fixture");
        auto storage = backend.create(*source, range, life); life.reset(); require(bool(storage.owner), "Duplicate-source lock failed");
        source.reset(); // Only cache's duplicated original file object now protects its immutable bytes.
        HANDLE writer = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        require(writer == INVALID_HANDLE_VALUE && GetLastError() == ERROR_SHARING_VIOLATION, "Source close allowed mutation of leased cache view");
        require(!DeleteFileW(path.c_str()) && GetLastError() == ERROR_SHARING_VIOLATION, "Source close allowed replacement of leased file");
        exact(storage); Backend::test_fail_next(Backend::TestFault::unlock); storage = {};
        require(entries == 0 && backend.stats().cleanup_failures > 0 && backend.finish_zero(), "Successful unmap did not retire failed unlock safely");
        control.reset(); require(controls == 0 && same(original, current_quota()), "Final mapping cleanup retained ownership"); ++checks;
    }
    const auto costs = backend.stats();
    require(costs.admission_calls >= costs.admission_successes && costs.admission_successes > 0
        && costs.quota_write_calls >= costs.quota_expand_writes + costs.quota_restore_writes
        && costs.quota_read_calls > 0 && costs.retire_calls > 0 && costs.retire_nanoseconds > 0
        && costs.guarded_copy_bytes > 0, "Cumulative lifecycle cost telemetry is inconsistent");
    std::printf("COST: admissions=%llu successes=%llu quota_writes=%llu expand=%llu restore=%llu reuse=%llu lock_ns=%llu page_query_ns=%llu copy_ns=%llu retire_ns=%llu\n",
        static_cast<unsigned long long>(costs.admission_calls), static_cast<unsigned long long>(costs.admission_successes),
        static_cast<unsigned long long>(costs.quota_write_calls), static_cast<unsigned long long>(costs.quota_expand_writes),
        static_cast<unsigned long long>(costs.quota_restore_writes), static_cast<unsigned long long>(costs.quota_reuse_claims),
        static_cast<unsigned long long>(costs.lock_nanoseconds), static_cast<unsigned long long>(costs.page_query_nanoseconds),
        static_cast<unsigned long long>(costs.guarded_copy_nanoseconds), static_cast<unsigned long long>(costs.retire_nanoseconds));
    std::printf("PASS: mapped RAM Windows child lifecycle; %d checks, exact bytes, same source, trim isolation, guarded faults, lease/quarantine, quota flags/conflict/restoration\n", checks);
}
int run_child(const std::filesystem::path& fixture) {
    wchar_t executable[32768]{};
    require(GetModuleFileNameW(nullptr, executable, 32768) != 0, "Cannot find child executable");
    std::wstring command = L"\"" + std::wstring(executable) + L"\" --child \"" + fixture.wstring() + L"\"";
    STARTUPINFOW startup{}; startup.cb = sizeof(startup); PROCESS_INFORMATION process{};
    HANDLE output = nullptr, error = nullptr;
    require(DuplicateHandle(GetCurrentProcess(), GetStdHandle(STD_OUTPUT_HANDLE), GetCurrentProcess(),
        &output, 0, TRUE, DUPLICATE_SAME_ACCESS) != 0, "Cannot inherit child output");
    if (!DuplicateHandle(GetCurrentProcess(), GetStdHandle(STD_ERROR_HANDLE), GetCurrentProcess(),
        &error, 0, TRUE, DUPLICATE_SAME_ACCESS)) { CloseHandle(output); throw std::runtime_error("Cannot inherit child error output"); }
    startup.dwFlags = STARTF_USESTDHANDLES; startup.hStdOutput = output; startup.hStdError = error;
    const bool created = CreateProcessW(executable, command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW,
        nullptr, nullptr, &startup, &process) != 0;
    CloseHandle(output); CloseHandle(error);
    require(created, "Cannot spawn own quota child");
    CloseHandle(process.hThread);
    const auto wait = WaitForSingleObject(process.hProcess, 60000);
    if (wait != WAIT_OBJECT_0) { TerminateProcess(process.hProcess, 1); WaitForSingleObject(process.hProcess, 10000); }
    DWORD exit = 1; GetExitCodeProcess(process.hProcess, &exit); CloseHandle(process.hProcess);
    require(wait == WAIT_OBJECT_0, "Quota child exceeded bounded wall time"); return static_cast<int>(exit);
}
#endif
}

int main(int argc, char** argv) {
    try {
        geometry_checks();
#if defined(_WIN32) && defined(_MSC_VER)
        if (argc == 3 && std::strcmp(argv[1], "--child") == 0) { windows_child(std::filesystem::u8path(argv[2])); return 0; }
        std::filesystem::path directory = "artifacts/native-mapped-ram-tests";
        if (argc == 3 && std::strcmp(argv[1], "--fixture-dir") == 0) directory = std::filesystem::u8path(argv[2]);
        else require(argc == 1, "Usage: host_moe_mapped_ram_test [--fixture-dir ignored-artifact-directory]");
        std::filesystem::create_directories(directory);
        const auto fixture = std::filesystem::absolute(directory / ("mapped-ram-" + std::to_string(GetCurrentProcessId()) + ".bin"));
        HANDLE file = CreateFileW(fixture.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        require(file != INVALID_HANDLE_VALUE, "Generated fixture already exists or cannot be created");
        bool written = true; std::array<std::uint8_t, 65536> block{};
        for (std::size_t offset = 0; offset < file_bytes && written;) {
            const auto count = (std::min)(block.size(), file_bytes - offset);
            for (std::size_t i = 0; i < count; ++i) block[i] = pattern(offset + i);
            DWORD done = 0; written = WriteFile(file, block.data(), static_cast<DWORD>(count), &done, nullptr) && done == count;
            offset += count;
        }
        written = CloseHandle(file) && written;
        struct Remove { std::filesystem::path path; ~Remove() { DeleteFileW(path.c_str()); } } remove{fixture};
        require(written, "Cannot write generated immutable fixture");
        const auto before = current_quota(); const int exit = run_child(fixture);
        require(same(before, current_quota()), "Parent working-set limits changed");
        require(exit == 0, "Mapped RAM child failed; no unavailable scenario counts as passing");
        require(DeleteFileW(fixture.c_str()) != 0, "Generated fixture cleanup failed");
        std::puts("PASS: quota changes confined to owned child; parent limits/flags unchanged; generated fixture cleanup");
#else
        (void)argc; (void)argv;
        std::printf("PASS: %d portable mapped geometry checks; Windows lifecycle unavailable, not validated here\n", checks);
#endif
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
}
