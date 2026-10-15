#!/usr/bin/env python3
"""Read-only Linux mmap/mincore diagnostic for model-file reopen boundaries.

No cache eviction, mmap writes, GPU work, or model execution. Each round verifies
the same selected bytes and compares extra read-only opens while a mapping stays
alive with closing every owned mapping/descriptor before reopening the path.
"""

import argparse
import ctypes
import hashlib
import json
import os
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    parser.add_argument("--offset", type=int, default=1 << 20)
    parser.add_argument("--bytes", type=int, default=32 << 20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args()
    page = os.sysconf("SC_PAGE_SIZE")
    if args.offset % page or args.bytes < 1 or args.rounds < 1:
        parser.error("offset must be page aligned; bytes and rounds must be positive")
    stat = os.stat(args.file)
    if args.offset < 0 or args.offset > stat.st_size - args.bytes:
        parser.error("selected range does not fit the file")

    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mmap.restype = ctypes.c_void_p
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    vectors = (ctypes.c_ubyte * ((args.bytes + page - 1) // page))()
    active_fds, active_maps = set(), set()
    results = []
    expected = None

    def opened():
        fd = os.open(args.file, os.O_RDONLY)
        active_fds.add(fd)
        return fd

    def closed(fd):
        os.close(fd)
        active_fds.remove(fd)

    def mapped(fd):
        # PROT_READ, MAP_SHARED: no byte can be changed through the view.
        ptr = libc.mmap(None, args.bytes, 1, 1, fd, args.offset)
        if ptr == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), "mmap")
        active_maps.add(ptr)
        return ptr

    def unmapped(ptr):
        if libc.munmap(ptr, args.bytes):
            raise OSError(ctypes.get_errno(), "munmap")
        active_maps.remove(ptr)

    def resident(ptr):
        if libc.mincore(ptr, args.bytes, vectors):
            raise OSError(ctypes.get_errno(), "mincore")
        return sum(bool(v & 1) for v in vectors) / len(vectors)

    def record(round_index, phase, ptr):
        row = {"round": round_index, "phase": phase,
               "residentFraction": resident(ptr),
               "ownedDescriptors": len(active_fds), "ownedMappings": len(active_maps)}
        results.append(row)
        print(json.dumps(row), flush=True)

    def warm(ptr):
        nonlocal expected
        start = time.perf_counter()
        sha = hashlib.sha256()
        for done in range(0, args.bytes, 1 << 20):
            sha.update(ctypes.string_at(ptr + done, min(1 << 20, args.bytes - done)))
        actual = sha.hexdigest()
        if expected is not None and expected != actual:
            raise RuntimeError("mapped bytes changed between rounds")
        expected = actual
        return (time.perf_counter() - start) * 1000

    try:
        for round_index in range(args.rounds):
            fd = opened()
            ptr = mapped(fd)
            record(round_index, "initial_mapping", ptr)
            warm_ms = warm(ptr)
            record(round_index, "mapped_hash_warm", ptr)
            results[-1]["warmMilliseconds"] = warm_ms
            extra = opened()
            record(round_index, "extra_readonly_open_existing_view", ptr)
            fresh = mapped(extra)
            record(round_index, "extra_readonly_open_fresh_view", fresh)
            os.pread(extra, 1, args.offset)
            record(round_index, "extra_pread_one_byte", ptr)
            unmapped(fresh)
            closed(extra)
            record(round_index, "extra_readonly_close_existing_view", ptr)
            unmapped(ptr)
            ptr = mapped(fd)
            record(round_index, "view_replaced_same_held_descriptor", ptr)
            closed(fd)
            record(round_index, "descriptor_closed_mapping_held", ptr)
            extra = opened()
            fresh = mapped(extra)
            record(round_index, "reopen_while_old_mapping_held", fresh)
            unmapped(ptr)
            unmapped(fresh)
            closed(extra)
            # All of this process's references to the inode are now gone.
            extra = opened()
            fresh = mapped(extra)
            record(round_index, "reopen_after_all_owned_references_closed", fresh)
            unmapped(fresh)
            closed(extra)

        report = {"file": os.path.abspath(args.file), "fileBytes": stat.st_size,
                  "offset": args.offset, "bytes": args.bytes, "rounds": args.rounds,
                  "sha256": expected, "results": results,
                  "scope": "read-only Linux mmap/mincore open/close diagnostic; no cache eviction or GPU"}
        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as output:
                json.dump(report, output, indent=2)
                output.write("\n")
        else:
            print(json.dumps(report, indent=2))
    finally:
        for ptr in active_maps:
            libc.munmap(ptr, args.bytes)
        for fd in active_fds:
            os.close(fd)


if __name__ == "__main__":
    main()
