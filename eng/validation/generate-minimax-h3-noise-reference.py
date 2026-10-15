#!/usr/bin/env python3
"""Generate H3 noise goldens directly from a pinned, unchanged sd.cpp header."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

REVISION = "d637cfb935feb2254898a8f1524046c958f3df80"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise SystemExit("Expected pinned stable-diffusion.cpp revision " + REVISION)
    paths = ["src/core/rng.hpp", "src/core/rng_philox.hpp", "src/pipeline/video.cpp"]
    if subprocess.check_output(["git", "-C", str(args.reference), "status", "--porcelain", "--", *paths], text=True).strip():
        raise SystemExit("Reference RNG/pipeline sources are modified")
    generator = Path(__file__).with_name("minimax-h3-noise-reference.cpp")
    with tempfile.TemporaryDirectory(prefix="h3-noise-reference-") as temporary:
        executable = Path(temporary) / "reference"
        subprocess.run(["c++", "-std=c++17", "-O2", "-ffp-contract=off", "-I", str(args.reference / "src"), str(generator), "-o", str(executable)], check=True)
        result = json.loads(subprocess.check_output([str(executable)], text=True))
    result["reference"] = {
        "repository": "https://github.com/leejet/stable-diffusion.cpp",
        "revision": revision,
        "rng": "cuda (Philox4x32-10)",
        "sourceSha256": {path: hashlib.sha256((args.reference / path).read_bytes()).hexdigest() for path in paths},
        "generatorSha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
        "layout": "video CTHW to [T,H/2,W/2,C,2,2]; audio [32,2,T] to [2,T,32]",
        "absoluteTolerance": 1e-6,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(json.dumps({"cases": len(result["cases"]), "output": str(args.output), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
