#!/usr/bin/env python3
"""Original BF16 feature extraction and real quantized connector oracle.

Input is a hash-verified hidden-state report from the independent Gemma oracle;
the optional aggregate report isolates connectors with identical input features.
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True); p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True); p.add_argument("--hidden-report", type=Path, required=True)
    p.add_argument("--aggregate-report", type=Path); p.add_argument("--device", choices=["mps", "cuda"], required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise SystemExit("Refusing to replace report")
    assert subprocess.check_output(["git", "-C", str(a.reference), "rev-parse", "HEAD"], text=True).strip() == REVISION
    assert not subprocess.check_output(["git", "-C", str(a.reference), "status", "--porcelain"], text=True).strip()
    sys.path.insert(0, str(a.reference.resolve() / "packages/ltx-core/src"))
    import numpy as np
    import torch
    import gguf
    from safetensors import safe_open
    from ltx_core.text_encoders.gemma.feature_extractor import FeatureExtractorV2
    from ltx_core.text_encoders.gemma.embeddings_connector import Embeddings1DConnector
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    report = dict(completed=False, source_revision=REVISION, device=a.device, encoder=str(a.encoder),
        model=str(a.model), hidden_report=str(a.hidden_report), hidden_report_sha256=hashlib.sha256(a.hidden_report.read_bytes()).hexdigest(),
        feature_dtype="bfloat16", connector_dtype="float32", attention_backend="torch_sdpa_math_float32_accumulation")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    def save(): a.output.write_text(json.dumps(report, indent=2) + "\n")
    def load(parent, descriptor):
        path = parent / Path(descriptor["path"]).name; raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == descriptor["sha256"]
        assert len(raw) == descriptor["elements"] * 4
        return torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy())
    def store(name, tensor):
        raw = tensor.detach().float().cpu().contiguous().numpy().astype("<f4").tobytes(); path = Path(str(a.output) + "." + name + ".f32")
        if path.exists(): raise RuntimeError("Refusing to replace array")
        path.write_bytes(raw); return dict(path=str(path), elements=len(raw) // 4, sha256=hashlib.sha256(raw).hexdigest())
    save()
    try:
        hidden = json.loads(a.hidden_report.read_text()); assert hidden["completed"] and hidden["source_revision"] == REVISION
        ids = hidden["token_ids"]; report["token_ids"] = ids
        with safe_open(a.encoder, framework="pt", device="cpu") as file:
            c = json.loads(file.metadata()["gemma_config"])["text_config"]
            assert len(hidden["hidden_states"]) == c["num_hidden_layers"] + 1
            if a.aggregate_report:
                aggregate = json.loads(a.aggregate_report.read_text()); assert aggregate["completed"] and aggregate["token_ids"] == ids
                features = {stream: load(a.aggregate_report.parent, aggregate[stream + "_aggregate"]).reshape(1, len(ids), -1).to(a.device)
                    for stream in ["video", "audio"]}
                report["aggregate_report_sha256"] = hashlib.sha256(a.aggregate_report.read_bytes()).hexdigest()
            else:
                states = tuple(load(a.hidden_report.parent, descriptor).reshape(1, len(ids), c["hidden_size"]).bfloat16().to(a.device)
                    for descriptor in hidden["hidden_states"])
                projections = {}
                for stream in ["video", "audio"]:
                    prefix = "text_embedding_projection." + stream + "_aggregate_embed."
                    w = file.get_tensor(prefix + "weight")
                    with torch.device("meta"): linear = torch.nn.Linear(w.shape[1], w.shape[0])
                    linear.load_state_dict({"weight": w.to(a.device), "bias": file.get_tensor(prefix + "bias").to(a.device)}, assign=True)
                    projections[stream] = linear
                extractor = FeatureExtractorV2(projections["video"], c["hidden_size"], projections["audio"]).eval()
                with torch.inference_mode():
                    video, audio = extractor(states, torch.ones(1, len(ids), dtype=torch.int64, device=a.device))
                features = dict(video=video.float(), audio=audio.float())
                del extractor, projections, linear, states, w
                for stream in ["video", "audio"]: report[stream + "_aggregate"] = store(stream + "_aggregate", features[stream])
                save()
        reader = gguf.GGUFReader(a.model); tensors = {tensor.name: tensor for tensor in reader.tensors}
        with torch.inference_mode(), torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            for stream in ["video", "audio"]:
                prefix = stream + "_embeddings_connector."
                selected = {name[len(prefix):]: value for name, value in tensors.items() if name.startswith(prefix)}
                dim, registers = map(int, selected["learnable_registers"].shape)
                heads = int(selected["transformer_1d_blocks.0.attn1.to_gate_logits.bias"].shape[0]); layers = 0
                while f"transformer_1d_blocks.{layers}.attn1.to_q.weight" in selected: layers += 1
                with torch.device("meta"):
                    connector = Embeddings1DConnector(attention_head_dim=dim // heads, num_attention_heads=heads,
                        num_layers=layers, num_learnable_registers=registers, positional_embedding_max_pos=[4096],
                        double_precision_rope=True, apply_gated_attention=True).float().eval()
                for name, value in selected.items():
                    array = gguf.quants.dequantize(value.data, value.tensor_type).reshape(tuple(reversed(value.shape))).astype(np.float32, copy=False)
                    target = connector
                    parts = name.split(".")
                    for part in parts[:-1]: target = getattr(target, part)
                    setattr(target, parts[-1], torch.nn.Parameter(torch.from_numpy(array.copy()).to(a.device), requires_grad=False))
                print("loaded", stream, layers, "connector layers", flush=True)
                values = torch.zeros(1, 1024, dim, device=a.device); values[:, :len(ids)] = features[stream].float()
                mask = torch.full((1, 1, 1, 1024), -torch.finfo(torch.float32).max, device=a.device); mask[..., :len(ids)] = 0
                start = time.perf_counter(); result, output_mask = connector(values, mask)
                assert torch.all(output_mask == 0)
                report[stream + "_context"] = store(stream + "_context", result)
                report[stream + "_layers"] = layers; report[stream + "_connector_elapsed_ms"] = (time.perf_counter() - start) * 1000
                del connector, result; save()
        report["completed"] = True; save()
    except Exception as error: report["error"] = repr(error); save(); raise
    print(json.dumps(dict(completed=True, output=str(a.output))))


if __name__ == "__main__": main()
