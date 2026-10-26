#!/usr/bin/env python3
"""Download a disjoint subset of a pinned Hugging Face manifest with Xet.

This supplements an active download without restarting or modifying its partial
files. The original manifest verifier must still check every published SHA-256.
Requires huggingface_hub with hf-xet installed. --verify additionally hashes this
subset; otherwise completion means size-checked download, not SHA validation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    selected = [dict(record) for record in manifest["files"] if record["path"] in args.names]
    if {record["path"] for record in selected} != set(args.names):
        parser.error("A requested name is absent from the pinned manifest")
    revision = manifest["revision"]
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        parser.error("The manifest must identify an exact lowercase commit revision")
    # Read by huggingface_hub at import time. High performance remains opt-in;
    # these VMs share a tight memory cgroup with native compilation and serving.
    os.environ["HF_HUB_DISABLE_XET"] = "0"
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "0")
    from huggingface_hub import hf_hub_download
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence = {"repo": manifest["repo"], "revision": revision, "started_unix": time.time(),
                "complete": False, "files": selected, "mode": "xet", "workers": args.workers,
                "high_performance": os.environ["HF_XET_HIGH_PERFORMANCE"]}
    lock = threading.Lock()

    def save():
        with lock:
            temporary = args.evidence.with_suffix(".tmp")
            temporary.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
            temporary.replace(args.evidence)

    def download(record):
        start = time.monotonic()
        path = Path(hf_hub_download(manifest["repo"], record["path"], revision=revision,
                                    local_dir=args.directory))
        if path.stat().st_size != record["size"]:
            raise RuntimeError(f"Wrong downloaded size: {record['path']}")
        record["downloaded"] = True
        record["download_seconds"] = time.monotonic() - start
        record["verified"] = False
        if args.verify:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                    digest.update(block)
            record["actual_sha256"] = digest.hexdigest()
            record["verified"] = record["actual_sha256"] == record["sha256"]
            if not record["verified"]:
                raise RuntimeError(f"SHA-256 mismatch: {record['path']}")
        save()
        print(json.dumps(record), flush=True)

    save()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(download, selected))
        evidence["complete"] = True
    except Exception as error:
        evidence["error_type"] = type(error).__name__
        raise
    finally:
        evidence["finished_unix"] = time.time()
        save()


if __name__ == "__main__":
    main()
