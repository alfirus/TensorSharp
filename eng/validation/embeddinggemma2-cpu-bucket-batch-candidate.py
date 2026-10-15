#!/usr/bin/env python3
"""Stage an isolated CPU same-padding-bucket attention batching experiment."""
import argparse
import importlib.util
import json
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "embeddinggemma2_cpu_candidate_build",
    Path(__file__).with_name("embeddinggemma2-cpu-alignment-candidate.py"))
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)

DECLARATION = "        bool same_length = true;\n"
BUCKET_DECLARATION = DECLARATION + """        // Validation candidate: independent streams for mixed actual lengths
        // with identical existing CPU padding. Flat learned rows stay unchanged.
        bool same_cpu_bucket = false;
#if defined(__linux__) && defined(__x86_64__)
        same_cpu_bucket = cpu_flat_qkv_rms && cpu_float_linears && cpu_alignment == 16 && pooling == 1;
#endif
        auto cpu_bucket = [&](int size) {
            return std::min(context, ((size + cpu_alignment - 1) / cpu_alignment) * cpu_alignment);
        };
"""
LENGTH_UPDATE = "            same_length = same_length && lengths[i] == lengths[0];\n"
BUCKET_UPDATE = LENGTH_UPDATE + "            same_cpu_bucket = same_cpu_bucket && cpu_bucket(lengths[i]) == cpu_bucket(lengths[0]);\n"
PACKED = "        const bool packed = batch > 1 && (!same_length ||\n"
BUCKET_PACKED = "        const bool packed = batch > 1 && (!(same_length || same_cpu_bucket) ||\n"
SCOPE = ("Validation-only same-existing-padding-bucket batching for the qualified "
         "Gemma2 CPU F32 RMS geometry, alignment16 and mean pooling. "
         "Attention uses independent batch streams; mean pooling uses per-stream "
         "rows rather than the old zero-weight block matrix. No precision, padding, "
         "token order, CPU worker count or upstream source change. Full-model "
         "batch/single numerical validation is required; arithmetic identity is not assumed.")


def scoped_source(source):
    BUILD.require("same_cpu_bucket" not in source, "Bucket candidate already present")
    BUILD.require("cpu_flat_qkv_rms = validated_cpu_geometry" in source,
                  "Require the qualified CPU geometry/worker predicate")
    for anchor in (DECLARATION, LENGTH_UPDATE, PACKED):
        BUILD.require(source.count(anchor) == 1, "Require exactly one known bucket insertion anchor")
    start = source.index(DECLARATION)
    end = source.index(PACKED)
    BUILD.require(start < source.index(LENGTH_UPDATE) < end,
                  "Length validation and bucket updates must precede packing")
    loop = source[start:end]
    BUILD.require('require(lengths[i] > 0 && lengths[i] <= context,' in loop,
                  "Require validated positive bounded lengths before bucket arithmetic")
    BUILD.require(loop.index('require(lengths[i] > 0') < loop.index(LENGTH_UPDATE),
                  "Length validation must execute before bucket arithmetic")
    return source.replace(DECLARATION, BUCKET_DECLARATION).replace(
        LENGTH_UPDATE, BUCKET_UPDATE).replace(PACKED, BUCKET_PACKED)


def build_candidate(args):
    return BUILD.build_candidate(args, source_transform=scoped_source, scope=SCOPE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("build-dir", "source", "native-library", "ggml-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-source-sha256", "expected-native-sha256", "expected-ggml-revision"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--execute", action="store_true",
                        help="Otherwise prepare/hash/record only; no compile/link")
    report = build_candidate(parser.parse_args())
    print(json.dumps({k: report[k] for k in (
        "completed", "built", "guarded_inputs_unchanged", "ordinary_native_unchanged", "model_qualified")}))


if __name__ == "__main__":
    main()
