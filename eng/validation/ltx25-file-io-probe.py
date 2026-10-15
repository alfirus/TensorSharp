#!/usr/bin/env python3
"""Read-only NFS diagnostic: one handle, eight MiB total buffer, disjoint ranges.

This tool never changes inference policy or evicts OS caches. It reads unused
embedding rows and alternates sequential and eight-worker positional reads.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import time


def process_io():
    path = Path('/proc/self/io')
    return {key: int(value) for key, value in
            (line.split(':') for line in path.read_text().splitlines())} if path.exists() else {}


def rss():
    return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mib', type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists(): raise SystemExit('Refusing to replace report')
    if not 128 <= args.mib <= 1536 or args.mib % 128: raise SystemExit('Expected128..1536MiB in128MiB increments')
    fd = os.open(args.encoder, os.O_RDONLY)
    try:
        identity = os.fstat(fd)
        length = struct.unpack('<Q', os.read(fd, 8))[0]
        header = os.read(fd, length)
        if len(header) != length: raise RuntimeError('Short safetensors header')
        tensor = json.loads(header)['model.embed_tokens.weight']
        if tensor['dtype'] != 'BF16': raise RuntimeError('Expected actual BF16 embedding table')
        # Skip the low256MiB containing ordinary short-prompt tokens and the
        # high128MiB containing the vocabulary's common byte/punctuation IDs.
        offset = 8 + length + tensor['data_offsets'][0] + (256 << 20)
        end = 8 + length + tensor['data_offsets'][1] - (128 << 20)
        total = args.mib << 20
        if offset + total > end: raise RuntimeError('Embedding range is too small')
        advice = None
        if hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
            advice = 'POSIX_FADV_SEQUENTIAL; no cache eviction'
        buffer = bytearray(8 << 20)
        view = memoryview(buffer)
        slices = [view[i << 20:(i + 1) << 20] for i in range(8)]
        def read_into(destination, position):
            done = 0
            while done < len(destination):
                count = os.preadv(fd, [destination[done:]], position + done)
                if not count: raise RuntimeError('Unexpected EOF')
                done += count
            return done
        report = dict(completed=False, encoder=str(args.encoder), encoder_bytes=identity.st_size,
            embedding_shape=tensor['shape'], header_sha256=hashlib.sha256(header).hexdigest(),
            offset=offset, bytes=total, buffer_bytes=len(buffer), advice=advice,
            scope='Alternating disjoint64MiB embedding ranges, no inference, no cache eviction', samples=[])
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as pool:
            for block in range(total // (64 << 20)):
                workers = 1 if block % 2 == 0 else 8
                begin = offset + block * (64 << 20)
                before = process_io(); resident = rss(); start = time.perf_counter()
                for step in range(8):
                    position = begin + step * len(buffer)
                    if workers == 1:
                        read_into(view, position)
                    else:
                        futures = [pool.submit(read_into, slices[i], position + (i << 20)) for i in range(8)]
                        for future in futures: future.result()
                elapsed = time.perf_counter() - start; after = process_io()
                report['samples'].append(dict(block=block, workers=workers, offset=begin, bytes=64 << 20,
                    seconds=elapsed, mib_per_second=64 / elapsed,
                    process_io_delta={key: after[key] - before.get(key, 0) for key in after},
                    rss_before=resident, rss_after=rss(), checksum=int(buffer[0]) + int(buffer[-1])))
        report['elapsed_seconds'] = time.perf_counter() - started
        report['rss_peak_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report['identity_unchanged'] = all(getattr(identity, field) == getattr(os.fstat(fd), field)
            for field in ['st_dev', 'st_ino', 'st_size', 'st_mtime_ns'])
        report['completed'] = True
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({key: sum(row['bytes'] for row in report['samples'] if row['workers'] == key) /
            sum(row['seconds'] for row in report['samples'] if row['workers'] == key) / (1 << 20) for key in [1, 8]}))
    finally:
        os.close(fd)


if __name__ == '__main__':
    main()
