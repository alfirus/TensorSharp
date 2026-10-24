#!/usr/bin/env python3
"""Read bounded local GGUF headers and save tensor types/shapes without loading weights.

Header hashes identify the inspected metadata only; they are not full model
integrity checks. Uses the existing repository GGUF header parser.
"""
import argparse
import collections
import fnmatch
import hashlib
import importlib.util
import json
from pathlib import Path
import struct


def inspect_file(module, path):
    before = path.stat()
    limit = min(before.st_size, 8 << 20)
    while True:
        with path.open("rb") as source:
            data = source.read(limit)
        try:
            header = module.inspect(data, before.st_size)
            break
        except (EOFError, struct.error):
            if limit >= min(before.st_size, 64 << 20):
                raise ValueError("Header unavailable within 64 MiB: " + str(path))
            limit = min(before.st_size, 64 << 20, limit * 2)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("File changed during inspection: " + str(path))
    if header["version"] not in (2, 3) or not 0 < header["header_bytes"] <= len(data):
        raise ValueError("Unsupported or truncated GGUF header")
    seen = set()
    for tensor in header["tensors"]:
        if (tensor["name"] in seen or not tensor["dimensions"]
                or any(n <= 0 for n in tensor["dimensions"])
                or tensor["stored_bytes"] <= 0
                or tensor["file_offset"] < header["header_bytes"]
                or tensor["file_offset"] + tensor["stored_bytes"] > before.st_size):
            raise ValueError("Invalid tensor metadata: " + str(tensor))
        seen.add(tensor["name"])
    return dict(path=str(path), bytes=before.st_size, mtime_ns=before.st_mtime_ns,
                header_bytes=header["header_bytes"],
                header_sha256=hashlib.sha256(data[:header["header_bytes"]]).hexdigest().upper(),
                bounded_bytes_read=len(data), version=header["version"],
                split={k: v for k, v in header["metadata"].items() if k.startswith("split.")},
                architecture=header["metadata"].get("general.architecture"),
                tensor_count=len(header["tensors"]), tensors=header["tensors"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--match", action="append", default=[], help="Tensor name glob; repeatable")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    helper = Path(__file__).with_name("inspect-remote-gguf.py")
    spec = importlib.util.spec_from_file_location("local_gguf_header_parser", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shards = [inspect_file(module, p.resolve()) for p in args.files]
    names = [t["name"] for s in shards for t in s["tensors"]]
    if len(set(names)) != len(names):
        parser.error("Duplicate tensor names across input files")
    histogram = collections.Counter(str(t["type"]) for s in shards for t in s["tensors"])
    if args.match:
        for shard in shards:
            shard["tensors"] = [t for t in shard["tensors"]
                                if any(fnmatch.fnmatchcase(t["name"], p) for p in args.match)]
    result = dict(schema=1, header_only=True, whole_file_integrity_qualified=False,
                  parser_sha256=hashlib.sha256(helper.read_bytes()).hexdigest().upper(),
                  tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper(),
                  total_tensor_count=len(names), tensor_type_histogram=histogram,
                  name_filters=args.match, shards=shards)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(dict(shards=len(shards), tensors=len(names), type_histogram=histogram)))


if __name__ == "__main__":
    main()
