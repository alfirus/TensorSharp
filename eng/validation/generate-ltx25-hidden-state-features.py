#!/usr/bin/env python3
"""Generate batch-one Float32 hidden-state feature fixtures from unchanged LTX.

This executes the original V2 normalization/rescaling functions and original
stable right-pad helpers. It uses synthetic precomputed hidden states, not a
Gemma model, learned aggregate projections, connectors or BF16 inference.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess


REVISION = "be6d6d0a69d11a0a399c462ed92606692e5f8697"
FEATURE_PATH = "packages/ltx-core/src/ltx_core/text_encoders/gemma/feature_extractor.py"
PROCESSOR_PATH = "packages/ltx-core/src/ltx_core/text_encoders/gemma/embeddings_processor.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise SystemExit("Unexpected original LTX revision: " + revision)
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        raise SystemExit("Original LTX source must remain unchanged")
    import torch
    torch.set_num_threads(1)

    # Loading this leaf directly avoids the Gemma package's unrelated optional
    # transformers dependency. Its original functions execute unchanged.
    spec = importlib.util.spec_from_file_location("original_ltx_feature_extractor", source / FEATURE_PATH)
    feature = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(feature)

    # These standalone helpers require only torch. Compile their original AST
    # nodes verbatim, avoiding eager connector/model imports. No source rewrite
    # or reimplementation of their bodies is involved.
    helper_names = {"convert_to_additive_mask", "_compute_right_pad_order", "_apply_right_pad_order"}
    tree = ast.parse((source / PROCESSOR_PATH).read_text())
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in helper_names]
    if {node.name for node in helpers} != helper_names:
        raise SystemExit("Pinned original right-pad helpers are missing")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(source / PROCESSOR_PATH), "exec"), namespace)

    tokens, hidden, layers = 5, 7, 4
    states = []
    for layer in range(layers):
        values = [math.sin((token + 1) * (index + 2) * (layer + 1) * .19) * (layer + 1)
                  + (index - 2) * .03125 - token * .125
                  for token in range(tokens) for index in range(hidden)]
        state = torch.tensor(values, dtype=torch.float32).reshape(1, tokens, hidden)
        if layer == 1:
            state[:, 3, :] = 0  # epsilon must keep an all-zero state finite.
        states.append(state)
    encoded = torch.stack(states, dim=-1)
    # Deliberately nonzero padded projection rows model the presence of a bias.
    video_rows = torch.arange(tokens * 3, dtype=torch.float32).reshape(1, tokens, 3) * .1875 - 2
    audio_rows = torch.arange(tokens * 2, dtype=torch.float32).reshape(1, tokens, 2) * -.3125 + 4
    fixture = {
        "source_revision": revision, "torch_version": torch.__version__, "dtype": "float32",
        "source_sha256": {path: hashlib.sha256((source / path).read_bytes()).hexdigest()
                          for path in [FEATURE_PATH, PROCESSOR_PATH]},
        "tokens": tokens, "hidden_size": hidden, "state_count": layers,
        "hidden_states": [state.flatten().tolist() for state in states],
        "video_projection_width": 11, "audio_projection_width": 5,
        "video_reorder_width": 3, "audio_reorder_width": 2,
        "video_reorder_input": video_rows.flatten().tolist(),
        "audio_reorder_input": audio_rows.flatten().tolist(),
        "limitations": "Synthetic precomputed Float32 hidden states only. No Gemma execution, learned projection, connector, BF16 parity, denoiser context or generation.",
        "cases": [],
    }
    masks = {"left": [0, 0, 1, 1, 1], "right": [1, 1, 1, 0, 0],
             "scattered": [0, 1, 0, 1, 0], "all_pad": [0, 0, 0, 0, 0], "all_valid": [1, 1, 1, 1, 1]}
    for name, values in masks.items():
        mask = torch.tensor([values], dtype=torch.int64)
        normalized = feature.norm_and_concat_per_token_rms(encoded, mask)
        scaled_video = feature._rescale_norm(normalized, fixture["video_projection_width"], hidden)
        scaled_audio = feature._rescale_norm(normalized, fixture["audio_projection_width"], hidden)
        order, reordered_additive = namespace["_compute_right_pad_order"](
            namespace["convert_to_additive_mask"](mask, torch.float32))
        fixture["cases"].append({
            "name": name, "attention_mask": values,
            "normalized": normalized.flatten().tolist(),
            "video_scaled": scaled_video.flatten().tolist(), "audio_scaled": scaled_audio.flatten().tolist(),
            "source_token_order": order.flatten().tolist(),
            "right_padded_mask": (reordered_additive >= 0).to(torch.int64).flatten().tolist(),
            "video_reordered": namespace["_apply_right_pad_order"](video_rows, order).flatten().tolist(),
            "audio_reordered": namespace["_apply_right_pad_order"](audio_rows, order).flatten().tolist(),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
