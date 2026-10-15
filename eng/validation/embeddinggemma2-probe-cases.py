#!/usr/bin/env python3
"""Export exact oracle token inputs for the EmbeddingGemma2Probe manifest."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--oracle", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
data = args.oracle.read_bytes()
oracle = json.loads(data)
manifest = {"oracle_file": str(args.oracle.resolve()), "oracle_sha256": hashlib.sha256(data).hexdigest(),
            "cases": [{"name": "oracle_" + str(item["index"]), "label": item["text"], "tokens": item["tokens"]}
                      for item in oracle["cases"]]}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
print(str(args.output.resolve()))
