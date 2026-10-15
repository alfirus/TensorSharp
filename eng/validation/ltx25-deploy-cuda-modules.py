#!/usr/bin/env python3
"""Install or verify explicitly pinned CUDA PTX modules after managed builds.

Each --module is filename:sha256:target (for example kernels.ptx:...:sm_86).
Validate source bytes and targets before copying anything, then assert every
destination byte hash and target. This never edits the upstream/native sources.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def inspect(path):
    data = path.read_bytes()
    match = re.search(rb"^\s*\.target\s+(\w+)", data, re.MULTILINE)
    if not match:
        raise ValueError(f"PTX target missing: {path}")
    return {"path": str(path.resolve()), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "target": match[1].decode()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, action="append", required=True)
    parser.add_argument("--module", action="append", required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = []
    for specification in args.module:
        name, expected_hash, expected_target = specification.split(":")
        if Path(name).name != name or not re.fullmatch(r"[a-f0-9]{64}", expected_hash):
            raise ValueError("Expected a filename and lowercase SHA256.")
        source = inspect(args.source / name)
        if source["sha256"] != expected_hash or source["target"] != expected_target:
            raise ValueError(f"Pinned source module mismatch: {source}")
        sources.append((name, source))
    destinations = []
    for directory in args.destination:
        if not args.verify_only:
            directory.mkdir(parents=True, exist_ok=True)
        for name, source in sources:
            path = directory / name
            if not args.verify_only:
                shutil.copyfile(args.source / name, path)
            actual = inspect(path)
            if any(actual[key] != source[key] for key in ("bytes", "sha256", "target")):
                raise ValueError(f"Deployed module mismatch: {actual}")
            destinations.append(actual)
    report = {"passed": True, "verify_only": args.verify_only,
              "scope": "Explicit PTX byte hashes and target compatibility; not native-source or inference validation.",
              "sources": [source for _, source in sources], "destinations": destinations}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "modules": len(sources), "destinations": len(args.destination)}))


if __name__ == "__main__":
    main()
