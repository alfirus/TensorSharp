#!/usr/bin/env python3
"""Stream authentic Gemma4 Unified BF16 weights through the original HF forward.

Only one transformer layer is resident. No production runtime depends on this
tool. Feature extraction executes unchanged original LTX on the selected GPU.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

REVISION = "be6d6d0a69d11a0a399c462ed92606692e5f8697"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--prompt", default="A red ball rolls across a wooden table.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--stage", choices=["hidden", "features"], default="hidden")
    parser.add_argument("--feature-device", choices=["mps", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.output.exists(): raise SystemExit("Refusing to replace a report")
    revision = subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip()
    assert revision == REVISION
    assert not subprocess.check_output(["git", "-C", str(args.reference), "status", "--porcelain"], text=True).strip()
    sys.path.insert(0, str(args.reference.resolve() / "packages/ltx-core/src"))
    import torch
    import transformers
    from safetensors import safe_open
    from tokenizers import Tokenizer
    from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedTextConfig
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedTextModel
    from ltx_core.text_encoders.gemma.encoders.encoder_configurator import _populate_rotary_v5
    from ltx_core.text_encoders.gemma.feature_extractor import FeatureExtractorV2
    if transformers.__version__ != "5.10.1": raise SystemExit("Reference requires transformers5.10.1")
    torch.set_num_threads(args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(completed=False, source_revision=revision, transformers_version=transformers.__version__,
        torch_version=torch.__version__, encoder=str(args.encoder), encoder_bytes=args.encoder.stat().st_size,
        prompt=args.prompt, stage=args.stage, feature_device=args.feature_device, dtype="bfloat16",
        attention_backend="torch_sdpa_math_float32_accumulation", residency="one BF16 layer; original forward unchanged")
    def save(): args.output.write_text(json.dumps(report, indent=2) + "\n")
    def store(name, value):
        raw = value.detach().float().contiguous().cpu().numpy().astype("<f4").tobytes()
        path = Path(str(args.output) + "." + name + ".f32")
        if path.exists(): raise RuntimeError("Refusing to replace array")
        path.write_bytes(raw)
        return dict(path=str(path), elements=len(raw) // 4, sha256=hashlib.sha256(raw).hexdigest())
    save()
    try:
        with safe_open(args.encoder, framework="pt", device="cpu") as file:
            metadata = json.loads(file.metadata()["gemma_config"])
            assert metadata["model_type"] == "gemma4_unified"
            config = Gemma4UnifiedTextConfig(**metadata["text_config"])
            config._attn_implementation = "sdpa"; config.use_cache = False
            tokenizer = Tokenizer.from_str(bytes(file.get_tensor("tokenizer_json").tolist()).decode())
            ids = tokenizer.encode(args.prompt.strip(), add_special_tokens=False).ids
            if not ids or ids[0] != 2: ids.insert(0, 2)
            ids = ids[:1024]; report["token_ids"] = ids; save()
            with torch.device("meta"): model = Gemma4UnifiedTextModel(config)
            model.eval(); _populate_rotary_v5(model, config)
            model.embed_tokens.register_buffer("embed_scale", torch.tensor(config.hidden_size ** .5), persistent=False)
            model.embed_tokens.weight = torch.nn.Parameter(file.get_tensor("model.embed_tokens.weight"), requires_grad=False)
            model.norm.weight = torch.nn.Parameter(file.get_tensor("model.norm.weight"), requires_grad=False)
            keys = list(file.keys())
            for index, layer in enumerate(model.layers):
                prefix = "model.layers." + str(index) + "."
                names = [key for key in keys if key.startswith(prefix)]
                def before(module, inputs, prefix=prefix, names=names, index=index):
                    module.load_state_dict({key[len(prefix):]: file.get_tensor(key) for key in names}, strict=True, assign=True)
                    print("loaded layer", index + 1, "/", config.num_hidden_layers, flush=True)
                def after(module, inputs, output):
                    module.to_empty(device="meta")
                layer.register_forward_pre_hook(before); layer.register_forward_hook(after)
            start = time.perf_counter()
            with torch.inference_mode(), torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
                result = model(torch.tensor([ids]), position_ids=torch.arange(1024 - len(ids), 1024)[None],
                    output_hidden_states=True, use_cache=False)
            report["hidden_states"] = [store("hidden_" + str(i), value) for i, value in enumerate(result.hidden_states)]
            report["hidden_elapsed_ms"] = (time.perf_counter() - start) * 1000; save()
            if args.stage == "features":
                # Release Gemma's embedding/module descriptors before loading
                # large learned projection matrices on the selected accelerator.
                states = result.hidden_states; del result, model
                projections = {}
                for stream in ["video", "audio"]:
                    prefix = "text_embedding_projection." + stream + "_aggregate_embed."
                    weight = file.get_tensor(prefix + "weight")
                    with torch.device("meta"): linear = torch.nn.Linear(weight.shape[1], weight.shape[0])
                    linear.load_state_dict({"weight": weight.to(args.feature_device), "bias": file.get_tensor(prefix + "bias").to(args.feature_device)}, assign=True)
                    projections[stream] = linear
                feature = FeatureExtractorV2(projections["video"], config.hidden_size, projections["audio"]).eval()
                with torch.inference_mode():
                    video, audio = feature(tuple(value.to(args.feature_device) for value in states),
                        torch.ones(1, len(ids), dtype=torch.int64, device=args.feature_device))
                report["video_aggregate"] = store("video_aggregate", video)
                report["audio_aggregate"] = store("audio_aggregate", audio)
            report["completed"] = True; save()
    except Exception as error:
        report["error"] = repr(error); save(); raise
    print(json.dumps(dict(completed=True, states=len(report["hidden_states"]), output=str(args.output))))


if __name__ == "__main__": main()
