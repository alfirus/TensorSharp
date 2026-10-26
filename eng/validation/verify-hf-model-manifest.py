#!/usr/bin/env python3
"""Verify downloaded files concurrently into a separate pinned manifest.

Coordinate with existing download/verifier processes before use. The output is
always separate from the input, avoiding concurrent evidence writers. Reusing
existing verified records is explicit and requires unchanged size and a file
mtime no later than the source manifest. No payload files are modified.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time


def save(output: Path, manifest: dict):
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def verify_file(record: dict, directory: Path, source_mtime_ns: int, reuse: bool) -> dict:
    record = dict(record)
    base = directory.resolve()
    path = (base / record["path"]).resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"Manifest path is outside the model directory: {record['path']}")
    expected = record["sha256"]
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError(f"Invalid SHA-256 record: {record['path']}")
    before = path.stat()
    if before.st_size != record["size"]:
        raise ValueError(f"Wrong size: {record['path']}")
    previously_verified = record.get("verified") is True
    record["verified"] = False
    if (reuse and previously_verified and record.get("actual_sha256") == expected
            and before.st_mtime_ns <= source_mtime_ns):
        record["verified"] = True
        record["verification_reused"] = True
    else:
        start = time.monotonic()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f"File changed while hashing: {record['path']}")
        record["actual_sha256"] = digest.hexdigest()
        record["verification_seconds"] = time.monotonic() - start
        record["verification_reused"] = False
        if record["actual_sha256"] != expected:
            raise ValueError(f"SHA-256 mismatch: {record['path']}")
        record["verified"] = True
    record["verified_mtime_ns"] = before.st_mtime_ns
    record["verified_inode"] = before.st_ino
    return record


def verify_manifest(source: Path, directory: Path, output: Path, workers: int = 4, reuse: bool = False) -> dict:
    if source.resolve() == output.resolve():
        raise ValueError("Use a separate output path; the original verifier may still own the source manifest")
    initial_stat = source.stat()
    source_bytes = source.read_bytes()
    current_stat = source.stat()
    if (initial_stat.st_ino, initial_stat.st_mtime_ns) != (current_stat.st_ino, current_stat.st_mtime_ns):
        raise ValueError("Source manifest changed while reading; coordinate with its current writer")
    manifest = json.loads(source_bytes.decode("utf-8-sig"))
    revision = manifest["revision"]
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("The manifest must pin a lowercase 40-character revision")
    records = manifest["files"]
    if not records or len({record["path"] for record in records}) != len(records):
        raise ValueError("The manifest contains no files or duplicate paths")
    if workers < 1:
        raise ValueError("workers must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.update(complete=False, verification_mode="parallel_sha256", verification_workers=workers,
                    source_manifest_sha256=hashlib.sha256(source_bytes).hexdigest(),
                    verification_started_unix=time.time())
    manifest.pop("finished_unix", None)
    work = [dict(record) for record in records]
    save(output, manifest)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Start the huge Engram shard immediately; SHA-256 of one file is
            # sequential, so queueing it behind smaller shards extends the run.
            order = sorted(range(len(work)), key=lambda i: work[i]["size"], reverse=True)
            futures = {pool.submit(verify_file, work[i], directory, initial_stat.st_mtime_ns, reuse): i
                       for i in order}
            for future in as_completed(futures):
                record = future.result()
                records[futures[future]] = record
                save(output, manifest)
                print(json.dumps(record), flush=True)
        for record in records:
            current = (directory / record["path"]).stat()
            if (current.st_size, current.st_ino, current.st_mtime_ns) != (
                    record["size"], record["verified_inode"], record["verified_mtime_ns"]):
                raise ValueError(f"File changed after verification: {record['path']}")
        manifest["complete"] = True
        manifest["finished_unix"] = time.time()
        return manifest
    except Exception as error:
        manifest["verification_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save(output, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse-verified", action="store_true")
    args = parser.parse_args()
    verify_manifest(args.manifest, args.directory, args.output, args.workers, args.reuse_verified)


if __name__ == "__main__":
    main()
