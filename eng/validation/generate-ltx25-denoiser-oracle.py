#!/usr/bin/env python3
"""Generate a tiny complete AV checkpoint and expected outputs with official LTX.

Requires torch, numpy and einops; run against an unchanged pinned LTX-2 checkout.
The binary GGUF and JSON are required test fixtures, not benchmark evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

REVISION = "be6d6d0a69d11a0a399c462ed92606692e5f8697"


# The published LTX 2.5 checkpoints declare both multipliers as 1000. The original
# LTXModelConfigurator.from_metadata reads them from this config; constructing
# LTXModel with keyword defaults would silently use av_ca_timestep_scale_multiplier=1.
TRANSFORMER_CONFIG = {"timestep_scale_multiplier": 1000, "av_ca_timestep_scale_multiplier": 1000.0}


def write_gguf(path, state, config):
    def string(value):
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    header = bytearray(struct.pack("<IIQQ", 0x46554747, 3, len(state), 2))
    header += string("general.architecture") + struct.pack("<I", 8) + string("ltxv")
    header += string("config") + struct.pack("<I", 8) + string(json.dumps(config))
    data = bytearray()
    for name, tensor in state.items():
        raw = tensor.contiguous().numpy().astype("<f4").tobytes()
        shape = tuple(reversed(tensor.shape))
        header += string(name) + struct.pack("<I", len(shape))
        header += struct.pack("<" + "Q" * len(shape), *shape)
        header += struct.pack("<IQ", 0, len(data))
        data += raw + bytes((-len(raw)) % 32)
    path.write_bytes(header + bytes((-len(header)) % 32) + data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("InferenceWeb.Tests/Fixtures/Ltx25"))
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise SystemExit(f"Expected official reference {REVISION}, found {revision}")
    if subprocess.check_output(["git", "-C", str(args.reference), "status", "--porcelain"], text=True).strip():
        raise SystemExit("Official reference must remain unchanged")
    sys.path.insert(0, str(args.reference.resolve() / "packages/ltx-core/src"))
    import torch
    from ltx_core.model.transformer.attention import AttentionOps, PytorchAttention
    from ltx_core.model.transformer.model import LTXModel
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.model.transformer.rope import precompute_freqs_cis, generate_freq_grid_np, generate_freq_grid_pytorch
    from ltx_core.model.transformer.timestep_embedding import get_timestep_embedding

    torch.set_num_threads(1)
    torch.manual_seed(913)
    attention = PytorchAttention([torch.nn.attention.SDPBackend.MATH])
    model = LTXModel(num_attention_heads=2, attention_head_dim=8, in_channels=3, out_channels=3,
                     num_layers=2, cross_attention_dim=16, audio_num_attention_heads=2,
                     audio_attention_head_dim=4, audio_in_channels=4, audio_out_channels=4,
                     audio_cross_attention_dim=8, apply_gated_attention=True, cross_attention_adaln=True,
                     ff_bias=False, audio_ff_bias=True, use_keyframes_abs_pos_embedding=True,
                     timestep_scale_multiplier=TRANSFORMER_CONFIG["timestep_scale_multiplier"],
                     av_ca_timestep_scale_multiplier=TRANSFORMER_CONFIG["av_ca_timestep_scale_multiplier"],
                     attention_ops=AttentionOps(attention, attention)).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(torch.randn_like(parameter) * (0.13 if "scale_shift_table" in name else 0.09))
            if name.endswith("_norm.weight"):
                parameter.add_(1.0)
    args.output.mkdir(parents=True, exist_ok=True)
    # .gguf is globally ignored for downloaded model weights. The tiny required
    # checkpoint fixture uses .bin so it is retained by ordinary Git adds.
    checkpoint = args.output / "denoiser.bin"
    write_gguf(checkpoint, model.state_dict(), {"transformer": TRANSFORMER_CONFIG})

    video = torch.randn(1, 4, 3)
    audio = torch.randn(1, 3, 4)
    video_context = torch.randn(1, 3, 16)
    audio_context = torch.randn(1, 2, 8)
    # Causal VAE patch bounds: two temporal patches, one row and two columns.
    video_bounds = torch.tensor([[[0, 1 / 24, 0, 32, 0, 32], [0, 1 / 24, 0, 32, 32, 64],
                                  [1 / 24, 9 / 24, 0, 32, 0, 32], [1 / 24, 9 / 24, 0, 32, 32, 64]]])
    audio_bounds = torch.tensor([[[0, .01], [.01, .05], [.05, .09]]])
    positions_v = video_bounds.reshape(1, 4, 3, 2).transpose(1, 2)
    positions_a = audio_bounds.reshape(1, 3, 1, 2).transpose(1, 2)
    cases = []
    def flat(tensor):
        return tensor.detach().flatten().tolist()
    for name, precision, vs, aus, vt, at, keyframes in [
        ("shared-sigma", "Float64", .725, .725, [.725] * 4, [.725] * 3, None),
        ("masked-different-sigma", "Float64", .421875, .909375, [0, .2, .421875, .1], [.909375, 0, .3], [1, 0, -1, 2]),
        ("float32-rope", "Float32", .99375, .725, [.99375] * 4, [.725] * 3, [1, 0, 0, 0]),
        ("shifted-positions", "Float64", .725, .725, [.725] * 4, [.725] * 3, None),
    ]:
        model.video_args_preprocessor.simple_preprocessor.double_precision_rope = precision == "Float64"
        model.audio_args_preprocessor.simple_preprocessor.double_precision_rope = precision == "Float64"
        current_v = positions_v.clone()
        if name == "shifted-positions":
            current_v[:, 0] += .375
        vm = Modality(video, torch.tensor([vs]), torch.tensor([vt]), current_v, video_context,
                      keyframes_mask=None if keyframes is None else torch.tensor(keyframes).reshape(1, 4, 1))
        am = Modality(audio, torch.tensor([aus]), torch.tensor([at]), positions_a, audio_context)
        with torch.inference_mode():
            vo, ao = model(vm, am, None)
            vp = model.video_args_preprocessor.prepare(vm, am).positional_embeddings
            ap = model.audio_args_preprocessor.prepare(am, vm).positional_embeddings
        cases.append(dict(name=name, precision=precision, video=flat(video), audio=flat(audio),
                          videoTokens=4, audioTokens=3, videoContext=flat(video_context), audioContext=flat(audio_context),
                          videoContextTokens=3, audioContextTokens=2,
                          videoPositions=flat(current_v.transpose(1, 2)), audioPositions=flat(audio_bounds),
                          videoSigma=vs, audioSigma=aus, videoTimesteps=vt, audioTimesteps=at, keyframeMask=keyframes,
                          videoVelocity=flat(vo), audioVelocity=flat(ao),
                          videoRopeCos=flat(vp[0].transpose(1, 2)), videoRopeSin=flat(vp[1].transpose(1, 2)),
                          audioRopeCos=flat(ap[0].transpose(1, 2)), audioRopeSin=flat(ap[1].transpose(1, 2))))
    times = torch.tensor([0, .421875, .99375, 1])
    features = get_timestep_embedding(times * 1000, 256, flip_sin_to_cos=True, downscale_freq_shift=0)
    rotary_samples = []
    for precision in ("Float32", "Float64"):
        for label, positions, dim in (("video", positions_v, 4096), ("audio", positions_a, 2048)):
            cos, sin = precompute_freqs_cis(positions, dim, torch.float32, max_pos=[20, 2048, 2048] if label == "video" else [20],
                                          use_middle_indices_grid=True, num_attention_heads=32,
                                          freq_grid_generator=generate_freq_grid_np if precision == "Float64" else generate_freq_grid_pytorch)
            cosine = cos.transpose(1, 2).flatten()
            sine = sin.transpose(1, 2).flatten()
            indices = sorted(set([0, 1, len(cosine) - 1] + list(range(0, len(cosine), 37))))
            rotary_samples.append(dict(stream=label, dim=dim, heads=32, precision=precision, indices=indices,
                                       cosine=cosine[indices].tolist(), sine=sine[indices].tolist()))
    result = dict(reference=dict(repository="https://github.com/Lightricks/LTX-2", revision=revision,
                                 torch=torch.__version__, checkpointSha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                 transformerConfig=TRANSFORMER_CONFIG),
                  cases=cases, rotarySamples=rotary_samples, timestepInputs=flat(times), timestepFeatures=flat(features))
    (args.output / "oracle.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(checkpoint=str(checkpoint), tensors=len(model.state_dict()), cases=len(cases), revision=revision)))


if __name__ == "__main__":
    main()
