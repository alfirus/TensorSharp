#!/usr/bin/env python3
"""Capture the pinned HF mask's actual scalar truth table, without torch.

Only the inspected sliding_window_mask_function definition is compiled. The
fixture records the complete source hash and distinguishes the model config
left context from its exclusive comparison argument.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Callable
import urllib.request

REVISION = "976e7a41c6615559f7a2ccb79fd5212c378b2c95"
SOURCE = "src/transformers/models/gemma4/modeling_gemma4.py"
URL = f"https://raw.githubusercontent.com/huggingface/transformers/{REVISION}/{SOURCE}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/embeddinggemma2/hf-modeling_gemma4-pinned.py"))
    parser.add_argument("--output", type=Path, default=Path("InferenceWeb.Tests/Fixtures/EmbeddingGemma2/audio-attention-mask.json"))
    args = parser.parse_args()
    if not args.source.exists():
        args.source.parent.mkdir(parents=True, exist_ok=True)
        args.source.write_bytes(urllib.request.urlopen(URL, timeout=60).read())
    source = args.source.read_bytes()
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "sliding_window_mask_function")
    # Fail rather than execute a future implementation that grows imports,
    # calls, decorators or attribute accesses beyond this scalar definition.
    for node in ast.walk(function):
        if isinstance(node, (ast.Call, ast.Import, ast.ImportFrom, ast.Attribute)):
            raise ValueError("Inspect a changed source definition before updating this scalar fixture")
    scope = {"Callable": Callable}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(args.source), "exec"), scope)
    mask = scope[function.name]((12, 0))
    cases = []
    for query in [0, 11, 12, 23, 24, 25, 95, 96, 127, 128]:
        for key in sorted({0, query, query + 1, max(0, query - 11), max(0, query - 12), max(0, query - 13)}):
            cases.append(dict(query=query, key=key, expected=bool(mask(0, 0, query, key))))
    report = dict(source_revision=REVISION, source_path=SOURCE, source_url=URL,
        source_sha256=hashlib.sha256(source).hexdigest(), function=function.name,
        function_line=function.lineno, config_attention_context_left=13,
        left_window_argument=12, right_window_argument=0,
        effective_maximum_past_distance=11,
        method="Truth table executed directly from the inspected pinned HF scalar mask definition; no TensorSharp or NumPy encoder output",
        regeneration="python3 eng/validation/generate-gemma4-audio-mask.py", cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(source_sha256=report["source_sha256"], function_line=function.lineno, cases=len(cases))))


if __name__ == "__main__":
    main()
