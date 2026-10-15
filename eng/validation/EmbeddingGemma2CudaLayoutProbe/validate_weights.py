#!/usr/bin/env python3
"""Cross-check --weights-only output against unchanged gguf-py and NumPy.

Set PYTHONPATH to llama.cpp/gguf-py. This tool makes no CUDA calls and verifies
bit-identical decoded and transposed matrices, not model or device quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
import gguf.gguf_reader
import gguf.quants


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to replace an existing report")
    report = json.loads(args.candidate.read_text())
    if report.get("cuda_executed") or not report.get("completed"):
        raise ValueError("Expected a completed read-only --weights-only report")
    if digest(report["model"]) != report["model_sha256"]:
        raise ValueError("Model hash differs from candidate")
    reader = GGUFReader(report["model"], mode="r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    results = []
    for row in report["measurements"]:
        parts = []
        for name in row["Weights"]:
            tensor = tensors[name]
            values = np.asarray(dequantize(tensor.data, tensor.tensor_type), dtype=np.float32)
            parts.append(values.reshape(int(tensor.shape[1]), int(tensor.shape[0])))
        values = np.ascontiguousarray(np.concatenate(parts, axis=0), dtype="<f4")
        transposed = np.ascontiguousarray(values.T, dtype="<f4")
        original_hash = hashlib.sha256(values.tobytes()).hexdigest()
        transposed_hash = hashlib.sha256(transposed.tobytes()).hexdigest()
        results.append(dict(name=row["Name"], shape=list(values.shape),
                            row_major_match=original_hash == row["row_major_f32_sha256"],
                            transposed_match=transposed_hash == row["transposed_f32_sha256"],
                            row_major_sha256=original_hash, transposed_sha256=transposed_hash))
    passed = bool(results) and all(row["row_major_match"] and row["transposed_match"] for row in results)
    output = dict(scope="Independent gguf-py/NumPy weight-decoder cross-check; zero CUDA calls",
                  numpy=np.__version__, platform=platform.platform(),
                  source_report_sha256=digest(args.candidate), model_sha256=report["model_sha256"],
                  reader_source_sha256=digest(gguf.gguf_reader.__file__),
                  dequant_source_sha256=digest(gguf.quants.__file__), matrices=results, passed=passed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(dict(matrix_count=len(results), passed=passed, cuda_runs=0)))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
