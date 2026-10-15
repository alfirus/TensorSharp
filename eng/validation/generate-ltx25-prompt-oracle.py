#!/usr/bin/env python3
"""Required tiny native-conditioning fixtures from HF Gemma4 and unchanged LTX."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

REVISION = "be6d6d0a69d11a0a399c462ed92606692e5f8697"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("InferenceWeb.Tests/Fixtures/Ltx25"))
    parser.add_argument("--feature-device", choices=["mps", "cuda"], required=True,
        help="BF16 rsqrt follows accelerator semantics; PyTorch CPU rounds through a BF16 sqrt.")
    args = parser.parse_args()
    assert subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip() == REVISION
    assert not subprocess.check_output(["git", "-C", str(args.reference), "status", "--porcelain"], text=True).strip()
    sys.path.insert(0, str(args.reference.resolve() / "packages/ltx-core/src"))
    import torch
    import transformers
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, AddedToken
    from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedTextConfig
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedTextModel
    from ltx_core.text_encoders.gemma.embeddings_connector import Embeddings1DConnector
    from ltx_core.text_encoders.gemma.feature_extractor import FeatureExtractorV2, norm_and_concat_per_token_rms, _rescale_norm
    from ltx_core.text_encoders.gemma.encoders.encoder_configurator import _populate_rotary_v5

    spec = importlib.util.spec_from_file_location("denoiser_fixture", Path(__file__).with_name("generate-ltx25-denoiser-oracle.py"))
    writer = importlib.util.module_from_spec(spec); spec.loader.exec_module(writer)
    torch.set_num_threads(1); torch.manual_seed(71325)
    chars = ["<pad>", "<eos>", "<bos>", "<unk>", "▁"] + list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?\n")
    vocab = {v: i for i, v in enumerate(chars)}
    for i in range(len(vocab), 512): vocab[f"<unused{i}>"] = i
    tokenizer = Tokenizer(models.BPE(vocab, [], unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Replace(" ", "▁")
    tokenizer.pre_tokenizer = pre_tokenizers.Split(" ", "merged_with_previous")
    tokenizer.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()])
    tokenizer.add_special_tokens([AddedToken(t, special=True) for t in chars[:4]])
    config = Gemma4UnifiedTextConfig(vocab_size=512, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, global_head_dim=16, num_global_key_value_heads=1, attention_k_eq_v=True,
        layer_types=["sliding_attention", "full_attention"], sliding_window=1024,
        rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0, "partial_rotary_factor": .25}},
        use_cache=False)
    config._attn_implementation = "sdpa"
    model = Gemma4UnifiedTextModel(config).to(torch.bfloat16).eval()
    # The original loader repopulates nonpersistent RoPE buffers in F32 after
    # assigning BF16 parameters. Casting the entire test model would otherwise
    # silently round its inverse frequencies and create a different oracle.
    _populate_rotary_v5(model, config)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(torch.randn_like(parameter) * .15)
            if "norm.weight" in name: parameter.add_(1)
    vproj = torch.nn.Linear(48, 16).to(torch.bfloat16).eval()
    aproj = torch.nn.Linear(48, 8).to(torch.bfloat16).eval()
    feature = FeatureExtractorV2(vproj, 16, aproj)
    connectors = {}
    for stream, dim in [("video", 16), ("audio", 8)]:
        connector = Embeddings1DConnector(attention_head_dim=dim // 2, num_attention_heads=2,
            num_layers=2, num_learnable_registers=4, positional_embedding_max_pos=[4096],
            double_precision_rope=True, apply_gated_attention=True).float().eval()
        with torch.no_grad():
            for name, parameter in connector.named_parameters():
                parameter.copy_(torch.randn_like(parameter) * .09)
                if "norm.weight" in name: parameter.add_(1)
        connectors[stream] = connector
    args.output.mkdir(parents=True, exist_ok=True)
    state = {"model." + k: v.contiguous() for k, v in model.state_dict().items()}
    for stream, projection in [("video", vproj), ("audio", aproj)]:
        state.update({"text_embedding_projection." + stream + "_aggregate_embed." + k: v.contiguous() for k, v in projection.state_dict().items()})
    state["tokenizer_json"] = torch.tensor(list(tokenizer.to_str().encode()), dtype=torch.uint8)
    text = config.to_dict(); text.update(hidden_size_per_layer_input=0, enable_moe_block=False)
    save_file(state, args.output / "prompt-encoder.bin", metadata={"gemma_config": json.dumps({"model_type": "gemma4_unified", "text_config": text})})
    writer.write_gguf(args.output / "prompt-connectors.bin", {stream + "_embeddings_connector." + k: v for stream, connector in connectors.items() for k, v in connector.state_dict().items()})
    feature = feature.to(args.feature_device)
    cases = []
    trace = {}
    def hook(name):
        def capture(module, inputs, output):
            if isinstance(inputs[0], torch.Tensor): trace["model." + name + ".input"] = inputs[0].float().flatten().tolist()
            if isinstance(output, torch.Tensor): trace["model." + name + ".output"] = output.float().flatten().tolist()
        return capture
    for name, module in model.named_modules():
        if name.endswith(("proj", "norm")): module.register_forward_hook(hook(name))
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        for prompt in ["", "cat", " A cat.\nHello! ", "<bos>cat"]:
            ids = tokenizer.encode(prompt.strip(), add_special_tokens=False).ids
            if not ids or ids[0] != 2: ids.insert(0, 2)
            ids = ids[:1024]
            positions = torch.arange(1024 - len(ids), 1024)[None]
            result = model(torch.tensor([ids]), position_ids=positions, output_hidden_states=True, use_cache=False)
            feature_states = tuple(value.to(args.feature_device) for value in result.hidden_states)
            feature_mask = torch.ones(1, len(ids), dtype=torch.int64, device=args.feature_device)
            video, audio = feature(feature_states, feature_mask)
            normalized = norm_and_concat_per_token_rms(torch.stack(feature_states, dim=-1), feature_mask)
            case = dict(prompt=prompt, ids=ids, hidden_states=[v.float().flatten().tolist() for v in result.hidden_states],
                video_aggregate=video.float().flatten().tolist(), audio_aggregate=audio.float().flatten().tolist(), trace=dict(trace),
                normalized=normalized.float().flatten().tolist(), video_scaled=_rescale_norm(normalized, 16, 16).float().flatten().tolist(),
                audio_scaled=_rescale_norm(normalized, 8, 16).float().flatten().tolist())
            # A short connector oracle exercises learned-register replacement;
            # the full 1024 length is independently checked in a single case.
            for stream, valid in [("video", video), ("audio", audio)]:
                length = 16 if len(ids) <= 16 else 32
                values = torch.zeros(1, length, valid.shape[-1]); values[:, :len(ids)] = valid.float().cpu()
                mask = torch.full((1, 1, 1, length), -torch.finfo(torch.float32).max); mask[..., :len(ids)] = 0
                output, output_mask = connectors[stream](values, mask)
                case[stream + "_connector"] = output.flatten().tolist()
                case[stream + "_connector_tokens"] = length
            cases.append(case)
    (args.output / "prompt-oracle.json").write_text(json.dumps(dict(source_revision=REVISION,
        transformers_version=transformers.__version__, torch_version=torch.__version__, dtype="bfloat16",
        attention_backend="torch_sdpa_math_float32_accumulation", rope_buffers_dtype="float32",
        feature_device=args.feature_device, cases=cases), indent=2) + "\n")
    print(json.dumps(dict(cases=len(cases), tensors=len(state), source_revision=REVISION)))


if __name__ == "__main__": main()
