#!/usr/bin/env python3
"""Bounded Windows read-only mapped-page lock probe; no CUDA or model access."""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

MIB = 1 << 20
WORKING_SET_CAP = 512 * MIB
PAGE_READONLY, MEM_COMMIT, MEM_MAPPED = 2, 0x1000, 0x40000
SOURCES = {
    "VirtualLock": "https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtuallock",
    "VirtualUnlock": "https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualunlock",
    "working_set": "https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-setprocessworkingsetsize",
    "locked_observation": "https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-psapi_working_set_ex_block",
}


class Unavailable(RuntimeError):
    pass


class InjectedFailure(RuntimeError):
    pass


def envelope(offset: int, length: int, mapped_bytes: int, page: int) -> tuple[int, int]:
    if page <= 0 or page & (page - 1) or offset < 0 or length <= 0:
        raise ValueError("Invalid page envelope input")
    if offset >= mapped_bytes or length > mapped_bytes - offset:
        raise ValueError("Payload escapes its mapped owner")
    begin, end = offset & -page, (offset + length + page - 1) & -page
    if end > mapped_bytes:
        raise ValueError("Rounded envelope escapes its mapped owner")
    return begin, end - begin


def working_set_target(original: tuple[int, int], payload: int) -> tuple[int, int]:
    minimum = max(original[0], payload + 16 * MIB)
    maximum = max(original[1], minimum + 16 * MIB)
    if maximum > WORKING_SET_CAP:
        raise Unavailable("Required working-set request exceeds the 512MiB ceiling")
    return minimum, maximum


def locked_intervals(flags: list[int], page: int) -> list[tuple[int, int]]:
    # Ignore members of an invalid PSAPI_WORKING_SET_EX_BLOCK.
    indices = [i for i, value in enumerate(flags) if value & 1 and value & (1 << 22)]
    result: list[tuple[int, int]] = []
    for i in indices:
        if result and result[-1][0] + result[-1][1] == i * page:
            result[-1] = result[-1][0], result[-1][1] + page
        else:
            result.append((i * page, page))
    return result


class Windows:
    def __init__(self) -> None:
        if sys.platform != "win32" or C.sizeof(C.c_void_p) != 8:
            raise Unavailable("This probe requires 64-bit Windows Python")
        from ctypes import wintypes as W

        class SystemInfo(C.Structure):
            _fields_ = [("oem", W.DWORD), ("page", W.DWORD), ("low", C.c_void_p),
                        ("high", C.c_void_p), ("mask", C.c_size_t), ("cpus", W.DWORD),
                        ("type", W.DWORD), ("granularity", W.DWORD),
                        ("level", W.WORD), ("revision", W.WORD)]

        class MemoryInfo(C.Structure):
            _fields_ = [("base", C.c_void_p), ("allocation", C.c_void_p),
                        ("allocation_protect", W.DWORD), ("partition", W.WORD),
                        ("bytes", C.c_size_t), ("state", W.DWORD),
                        ("protect", W.DWORD), ("type", W.DWORD)]

        class PageInfo(C.Structure):
            _fields_ = [("address", C.c_void_p), ("flags", C.c_size_t)]

        self.kernel = C.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "GetCurrentProcess": (C.c_void_p, []),
            "GetSystemInfo": (None, [C.POINTER(SystemInfo)]),
            "GetProcessWorkingSetSize": (W.BOOL, [C.c_void_p, C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)]),
            "SetProcessWorkingSetSize": (W.BOOL, [C.c_void_p, C.c_size_t, C.c_size_t]),
            "CreateFileW": (C.c_void_p, [W.LPCWSTR, W.DWORD, W.DWORD, C.c_void_p, W.DWORD, W.DWORD, C.c_void_p]),
            "CreateFileMappingW": (C.c_void_p, [C.c_void_p, C.c_void_p, W.DWORD, W.DWORD, W.DWORD, W.LPCWSTR]),
            "MapViewOfFile": (C.c_void_p, [C.c_void_p, W.DWORD, W.DWORD, W.DWORD, C.c_size_t]),
            "VirtualQuery": (C.c_size_t, [C.c_void_p, C.POINTER(MemoryInfo), C.c_size_t]),
            "VirtualLock": (W.BOOL, [C.c_void_p, C.c_size_t]),
            "VirtualUnlock": (W.BOOL, [C.c_void_p, C.c_size_t]),
            "UnmapViewOfFile": (W.BOOL, [C.c_void_p]),
            "CloseHandle": (W.BOOL, [C.c_void_p]),
            "SetFilePointerEx": (W.BOOL, [C.c_void_p, C.c_int64, C.POINTER(C.c_int64), W.DWORD]),
            "ReadFile": (W.BOOL, [C.c_void_p, C.c_void_p, W.DWORD, C.POINTER(W.DWORD), C.c_void_p]),
        }
        for name, (restype, argtypes) in signatures.items():
            function = getattr(self.kernel, name)
            function.restype, function.argtypes = restype, argtypes
        self.psapi = C.WinDLL("psapi", use_last_error=True)
        self.query = self.psapi.QueryWorkingSetEx
        self.query.restype = W.BOOL
        self.query.argtypes = [C.c_void_p, C.c_void_p, W.DWORD]
        self.MemoryInfo, self.PageInfo, self.DWORD = MemoryInfo, PageInfo, W.DWORD
        self.process = self.kernel.GetCurrentProcess()  # No foreign process handle is accepted.
        system = SystemInfo()
        self.kernel.GetSystemInfo(C.byref(system))
        self.page = system.page
        if self.page != 4096:
            raise Unavailable("This qualification requires actual 4KiB Windows pages")

    @staticmethod
    def check(ok: object, name: str) -> None:
        if not ok:
            raise OSError(C.get_last_error(), name)

    def working_set(self) -> tuple[int, int]:
        low, high = C.c_size_t(), C.c_size_t()
        self.check(self.kernel.GetProcessWorkingSetSize(self.process, C.byref(low), C.byref(high)),
                   "GetProcessWorkingSetSize")
        return low.value, high.value


class Mapping:
    def __init__(self, api: Windows, path: Path, size: int, events: list[dict], label: str = "main") -> None:
        self.api, self.size, self.events = api, size, events
        self.label = label
        self.file = self.section = self.base = None
        self.owned: list[tuple[int, int]] = []
        self.uncertain_lock = False
        self.file = api.kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 0x80, None)
        if self.file == C.c_void_p(-1).value:
            self.file = None
            raise OSError(C.get_last_error(), "CreateFileW(GENERIC_READ, FILE_SHARE_READ)")
        try:
            self.section = api.kernel.CreateFileMappingW(self.file, None, PAGE_READONLY, 0, 0, None)
            api.check(self.section, "CreateFileMappingW(PAGE_READONLY)")
            self.base = api.kernel.MapViewOfFile(self.section, 4, 0, 0, size)
            api.check(self.base, "MapViewOfFile(FILE_MAP_READ)")
            events.append({"event": "read_only_view_created", "view": self.label, "bytes": size})
        except BaseException:
            self.close()
            raise

    def protection(self) -> list[dict]:
        result, cursor = [], self.base
        while cursor < self.base + self.size:
            info = self.api.MemoryInfo()
            self.api.check(self.api.kernel.VirtualQuery(cursor, C.byref(info), C.sizeof(info)), "VirtualQuery")
            if info.state != MEM_COMMIT or info.protect != PAGE_READONLY or info.type != MEM_MAPPED:
                raise RuntimeError("View is not committed, mapped, exactly PAGE_READONLY")
            end = min(info.base + info.bytes, self.base + self.size)
            if end <= cursor:
                raise RuntimeError("VirtualQuery made no forward progress")
            result.append({"offset": cursor - self.base, "bytes": end - cursor,
                           "state": info.state, "protect": info.protect, "type": info.type})
            cursor = end
        return result

    def observe(self) -> tuple[dict, list[int]]:
        count = self.size // self.api.page
        entries = (self.api.PageInfo * count)()
        for i in range(count):
            entries[i].address = self.base + i * self.api.page
        self.api.check(self.api.query(self.api.process, entries, C.sizeof(entries)), "QueryWorkingSetEx")
        flags = [entry.flags for entry in entries]
        valid = [value for value in flags if value & 1]
        return {"pages": count, "valid_pages": len(valid),
                "locked_pages": sum(bool(value & (1 << 22)) for value in valid),
                "shareable_valid_pages": sum(bool(value & (1 << 15)) for value in valid),
                "readonly_valid_pages": sum(((value >> 4) & 0x7FF) == PAGE_READONLY for value in valid)}, flags

    def verify_buffered(self, expected_sha: str) -> dict:
        self.api.check(self.api.kernel.SetFilePointerEx(self.file, 0, None, 0), "SetFilePointerEx")
        buffer = C.create_string_buffer(65536)
        buffered, mapped, offset = hashlib.sha256(), hashlib.sha256(), 0
        while offset < self.size:
            length, done = min(len(buffer), self.size - offset), self.api.DWORD()
            self.api.check(self.api.kernel.ReadFile(self.file, buffer, length, C.byref(done), None), "ReadFile")
            if done.value != length:
                raise RuntimeError("Short generated-fixture buffered read")
            reference, view = buffer.raw[:length], C.string_at(self.base + offset, length)
            if reference != view:
                raise RuntimeError("Mapped bytes differ from buffered ReadFile bytes")
            buffered.update(reference)
            mapped.update(view)
            offset += length
        if buffered.hexdigest() != expected_sha or mapped.hexdigest() != expected_sha:
            raise RuntimeError("Fixture SHA256 differs from generated immutable source")
        return {"exact_bytes": self.size, "sha256": expected_sha,
                "method": "buffered ReadFile vs FILE_MAP_READ, every byte; 64KiB transient buffers"}

    def unlock_owned(self) -> None:
        while self.owned:
            offset, length = self.owned[-1]
            self.api.check(self.api.kernel.VirtualUnlock(self.base + offset, length), "VirtualUnlock")
            self.events.append({"event": "unlock_succeeded", "view": self.label, "offset": offset, "bytes": length})
            self.owned.pop()

    def trial(self, name: str, offset: int, length: int, expected_sha: str,
              inject_failure: bool = False) -> dict:
        begin, span = envelope(offset, length, self.size, self.api.page)
        before, _ = self.observe()
        if before["locked_pages"]:
            raise RuntimeError("Previous trial leaked a locked page")
        started = time.perf_counter()
        self.uncertain_lock = True
        C.set_last_error(0)
        ok = bool(self.api.kernel.VirtualLock(self.base + begin, span))
        error = 0 if ok else C.get_last_error()
        elapsed = (time.perf_counter() - started) * 1000
        if ok:
            self.owned.append((begin, span))
            self.uncertain_lock = False
        result = {"name": name, "payload_offset": offset, "payload_bytes": length,
                  "envelope_offset": begin, "envelope_bytes": span, "lock_succeeded": ok,
                  "win32_error": error, "lock_wall_ms": elapsed, "before": before}
        try:
            during, flags = self.observe()
            result["during"] = during
            if not ok:
                # Do not assume failed VirtualLock is atomic. Release any observed partial ownership.
                self.owned.extend(locked_intervals(flags, self.api.page))
                self.uncertain_lock = False
            if ok and (locked_intervals(flags, self.api.page) != [(begin, span)]
                       or during["readonly_valid_pages"] < during["locked_pages"]):
                raise RuntimeError("Successful lock lacks its exact observed read-only page envelope")
            if ok:
                result["verification"] = self.verify_buffered(expected_sha)
                if inject_failure:
                    self.events.append({"event": "injected_exception_while_locked", "bytes": span})
                    raise InjectedFailure("Expected probe failure after observed lock and exact bytes")
        finally:
            self.unlock_owned()
        after, _ = self.observe()
        result["after_unlock"] = after
        if after["locked_pages"]:
            raise RuntimeError("VirtualUnlock left observed locked pages")
        return result

    def close(self) -> None:
        self.unlock_owned()
        if self.uncertain_lock:
            raise RuntimeError("Unknown partial lock state: keep mapping alive until child exit")
        if self.base:
            self.api.check(self.api.kernel.UnmapViewOfFile(self.base), "UnmapViewOfFile")
            self.events.append({"event": "view_unmapped_after_unlock", "view": self.label})
            self.base = None
        for name in ("section", "file"):
            handle = getattr(self, name)
            if handle:
                self.api.check(self.api.kernel.CloseHandle(handle), "CloseHandle(" + name + ")")
                setattr(self, name, None)
                self.events.append({"event": name + "_handle_closed", "view": self.label})


def two_view_trim_check(a: Mapping, path: Path, expected_sha: str, result: dict) -> None:
    """Test virtual-view lock isolation; physical file-cache identity remains unproven."""
    a.label = "owned_cache_A"
    b = Mapping(a.api, path, a.size, a.events, label="model_like_B")
    try:
        result["distinct_virtual_views"] = a.base != b.base
        if a.base == b.base:
            raise RuntimeError("Two live views unexpectedly have the same virtual address")
        result["b_protection"] = b.protection()
        result["b_initial_verification"] = b.verify_buffered(expected_sha)
        before_a, _ = a.observe()
        before_b, _ = b.observe()
        result.update(before_a=before_a, before_b=before_b)
        if before_a["locked_pages"] or before_b["locked_pages"]:
            raise RuntimeError("Alias isolation requires both views initially unlocked")
        a.uncertain_lock = True
        C.set_last_error(0)
        locked = bool(a.api.kernel.VirtualLock(a.base, a.size))
        error = 0 if locked else C.get_last_error()
        result.update(a_lock_succeeded=locked, a_lock_error=error)
        if locked:
            a.owned.append((0, a.size))
            a.uncertain_lock = False
        observed_a, flags = a.observe()
        result["a_after_lock"] = observed_a
        if not locked:
            a.owned.extend(locked_intervals(flags, a.api.page))
            a.uncertain_lock = False
            raise Unavailable("Owned-cache view lock refused, Win32 error " + str(error))
        if locked_intervals(flags, a.api.page) != [(0, a.size)]:
            raise RuntimeError("Owned-cache view does not have its complete observed lock")
        observed_b, _ = b.observe()
        result["b_while_a_locked"] = observed_b
        if observed_b["locked_pages"]:
            raise RuntimeError("Unowned model-like view unexpectedly reports locked pages")
        C.set_last_error(0)
        unlocked_b = bool(a.api.kernel.VirtualUnlock(b.base, b.size))
        trim_error = 0 if unlocked_b else C.get_last_error()
        result["b_trim_like_virtual_unlock"] = {"returned_success": unlocked_b, "win32_error": trim_error,
                                                "expected_error_not_locked": 158}
        a.events.append({"event": "unlocked_B_trim_attempt", "view": b.label,
                         "returned_success": unlocked_b, "win32_error": trim_error})
        if unlocked_b or trim_error != 158:
            raise RuntimeError("Unlocked B did not return expected ERROR_NOT_LOCKED(158)")
        after_trim_a, flags = a.observe()
        result["a_after_b_trim"] = after_trim_a
        result["b_after_trim"], _ = b.observe()
        if locked_intervals(flags, a.api.page) != [(0, a.size)]:
            raise RuntimeError("Trimming separate B changed A's complete lock")
        result["a_exact_after_b_trim"] = a.verify_buffered(expected_sha)
        b.close()
        result["b_unmapped_while_a_locked"] = True
        after_unmap_a, flags = a.observe()
        result["a_after_b_unmap"] = after_unmap_a
        if locked_intervals(flags, a.api.page) != [(0, a.size)]:
            raise RuntimeError("Unmapping separate B changed A's complete lock")
        result["a_protection_after_b_unmap"] = a.protection()
        result["a_exact_after_b_unmap"] = a.verify_buffered(expected_sha)
    finally:
        try:
            b.close()
        finally:
            a.unlock_owned()
    after_unlock_a, _ = a.observe()
    result["a_after_final_unlock"] = after_unlock_a
    if after_unlock_a["locked_pages"]:
        raise RuntimeError("Final owned-cache unlock left locked pages")
    result["status"] = "complete"


def child(path: Path, expected_sha: str, alias_trim: bool = False) -> tuple[dict, int]:
    report: dict = {"pid": os.getpid(), "parent_pid": os.getppid(), "events": [], "trials": []}
    api = owner = original = None
    changed = False
    status, exit_code = "complete", 0
    try:
        api = Windows()
        size = path.stat().st_size
        if size < 8192 or size > 64 * MIB or size % api.page:
            raise ValueError("Child fixture must be 4KiB aligned, 8KiB through 64MiB")
        if len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
            raise ValueError("Child requires the generated fixture SHA256")
        original = api.working_set()
        report.update(page_bytes=api.page, original_working_set_bytes=list(original),
                      working_set_request_cap_bytes=WORKING_SET_CAP)
        owner = Mapping(api, path, size, report["events"])
        report["mapping_protection"] = owner.protection()
        report["initial_verification"] = owner.verify_buffered(expected_sha)
        size = owner.size
        cases = [("unaligned", 17, 4097), ("page_crossing", 4095, 2), ("tail", size - 17, 17),
                 ("whole_fixture", 0, size)]
        default = [owner.trial("default_" + name, start, count, expected_sha) for name, start, count in cases]
        report["trials"].extend(default)
        if not all(trial["lock_succeeded"] for trial in default):
            target = working_set_target(original, size)
            report["requested_working_set_bytes"] = list(target)
            C.set_last_error(0)
            if not api.kernel.SetProcessWorkingSetSize(api.process, *target):
                raise Unavailable("Own child working-set increase refused, Win32 error " + str(C.get_last_error()))
            changed = True
            report["adjusted_working_set_bytes"] = list(api.working_set())
            for name, start, count in cases + [("unaligned_whole_fixture", 17, size - 34)]:
                trial = owner.trial("adjusted_" + name, start, count, expected_sha)
                report["trials"].append(trial)
                if not trial["lock_succeeded"]:
                    raise Unavailable("Bounded lock refused after child-only quota increase, Win32 error "
                                      + str(trial["win32_error"]))
        injected = False
        try:
            owner.trial("injected_after_successful_lock", 4095, 2, expected_sha, inject_failure=True)
        except InjectedFailure:
            injected = True
        if not injected:
            raise Unavailable("No successful lock available for exception-cleanup qualification")
        after_failure, _ = owner.observe()
        if owner.owned or after_failure["locked_pages"]:
            raise RuntimeError("Injected exception retained page ownership")
        report["injected_exception_cleanup"] = {"exercised": True, "after_unlock": after_failure}
        report["final_protection"] = owner.protection()
        report["final_verification"] = owner.verify_buffered(expected_sha)
        if alias_trim:
            report["two_view_trim_isolation"] = {}
            two_view_trim_check(owner, path, expected_sha, report["two_view_trim_isolation"])
    except Unavailable as error:
        status, exit_code, report["error"] = "unavailable", 77, str(error)
    except Exception as error:
        status, exit_code, report["error"] = "failed", 1, repr(error)
    finally:
        errors = []
        if owner:
            try:
                owner.close()
            except Exception as error:
                errors.append(repr(error))
        if changed and api and original:
            try:
                api.check(api.kernel.SetProcessWorkingSetSize(api.process, *original), "Restore own child working set")
                report["restored_working_set_bytes"] = list(api.working_set())
                if api.working_set() != original:
                    raise RuntimeError("Restored working-set limits differ from originals")
                report["events"].append({"event": "child_working_set_restored"})
            except Exception as error:
                errors.append(repr(error))
        report["cleanup_errors"] = errors
        if errors:
            status, exit_code = "failed", 1
    report.update(status=status, exit_code=exit_code, working_set_changed=changed)
    return report, exit_code


def self_test() -> None:
    assert envelope(17, 4097, 16384, 4096) == (0, 8192)
    assert envelope(4095, 2, 16384, 4096) == (0, 8192)
    assert envelope(16383, 1, 16384, 4096) == (12288, 4096)
    for values in [(0, 0, 4096, 4096), (-1, 1, 4096, 4096), (4096, 1, 4096, 4096),
                   (0, 4097, 4096, 4096), (0, 1, 4095, 4096), (0, 1, 4096, 3000)]:
        try:
            envelope(*values)
        except ValueError:
            pass
        else:
            raise AssertionError(values)
    assert locked_intervals([1 | (1 << 22), 1 | (1 << 22), 1, 1 << 22], 4096) == [(0, 8192)]
    assert working_set_target((204800, 1413120), 16 * MIB) == (32 * MIB, 48 * MIB)
    try:
        working_set_target((MIB, WORKING_SET_CAP + 1), 16 * MIB)
    except Unavailable:
        pass
    else:
        raise AssertionError("Working-set ceiling escaped")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bytes", type=int, default=16 * MIB)
    parser.add_argument("--keep-fixture", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--alias-trim", action="store_true", help="Also qualify separate readonly-view trim isolation")
    parser.add_argument("--child-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("PASS: page envelopes, invalid boundaries, valid/locked bit decoding, bounded quota planning")
        return 0
    if args.child_sha256:
        if not args.fixture:
            parser.error("Child requires its generated fixture")
        report, code = child(args.fixture.resolve(), args.child_sha256, args.alias_trim)
        print(json.dumps(report))
        return code
    if not args.fixture or not args.output:
        parser.error("Provide new --fixture and --output paths under ignored artifacts/")
    if args.bytes < 8192 or args.bytes > 64 * MIB or args.bytes % 4096:
        parser.error("--bytes must be 4KiB aligned, 8KiB through 64MiB")
    if args.fixture.exists() or args.output.exists():
        parser.error("Fixture and output must not already exist")
    if not args.fixture.parent.is_dir() or not args.output.parent.is_dir():
        parser.error("Fixture/output parent directories must already exist")
    self_test()
    report: dict = {"kind": "windows_readonly_mapped_VirtualLock_capability", "fixture_bytes": args.bytes,
                    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "sources": SOURCES, "limitations": [
                        "No CUDA, model, memory-pressure scan, cache trim, or other-process quota changes.",
                        "Fresh fixture and full-byte verification warm file pages; API wall latency is not inference timing.",
                        "QueryWorkingSetEx Shared means shareable; it does not identify Cache Manager/ReadFile physical pages.",
                        "Exact buffered ReadFile bytes prove coherence only; no physical SSD-sharing, eviction or speed claim."]}
    code, created = 1, False
    try:
        api = Windows()
        parent_before = api.working_set()
        report["parent_working_set_before_bytes"] = list(parent_before)
        digest, pattern = hashlib.sha256(), bytes(range(256)) * 16
        with args.fixture.open("xb") as fixture:
            created = True
            for page in range(args.bytes // 4096):
                shift = page % 256
                block = struct.pack("<QQ", page, page ^ 0x9E3779B97F4A7C15) + (pattern + pattern)[shift:shift + 4080]
                fixture.write(block)
                digest.update(block)
        report["fixture_sha256"] = digest.hexdigest()
        command = [sys.executable, str(Path(__file__).resolve()), "--fixture", str(args.fixture.resolve()),
                   "--child-sha256", digest.hexdigest()]
        if args.alias_trim:
            command.append("--alias-trim")
        result = subprocess.run(command, capture_output=True, text=True, timeout=45)
        report["child"] = json.loads(result.stdout)
        code = result.returncode
        if code != report["child"]["exit_code"] or report["child"]["parent_pid"] != os.getpid():
            raise RuntimeError("Child result/provenance mismatch")
        parent_after = api.working_set()
        report["parent_working_set_after_bytes"] = list(parent_after)
        report["parent_working_set_unchanged"] = parent_before == parent_after
        if parent_before != parent_after:
            raise RuntimeError("Parent process working-set settings changed")
        report["status"] = report["child"]["status"]
    except Unavailable as error:
        report["status"], report["error"], code = "unavailable", str(error), 77
    except Exception as error:
        report["status"], report["error"], code = "failed", repr(error), 1
    finally:
        if created and not args.keep_fixture:
            try:
                args.fixture.unlink()
                report["fixture_removed_after_child_exit"] = True
            except Exception as error:
                report["cleanup_error"], report["status"], code = repr(error), "failed", 1
        report["exit_code"] = code
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2)
            output.write("\n")
    print(json.dumps({"status": report["status"], "exit_code": code, "report": str(args.output.resolve())}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
