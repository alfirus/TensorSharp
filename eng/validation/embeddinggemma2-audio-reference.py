#!/usr/bin/env python3
"""Independent F32 NumPy audio tower oracle for the EmbeddingGemma 2 GGUF.

Uses gguf-py only to read/dequantize weights. The full SSCP, twelve conformer
layers, output projection, and audio-to-text soft projection run in NumPy.
The frontend follows the pinned HF definition in generate-gemma4-audio-frontend.py.
Optional --candidate accepts raw little-endian F32 soft rows for comparison.
Generated states/reports belong under ignored docs/validation/ or artifacts/.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import wave

import numpy as np

HF_REVISION = "976e7a41c6615559f7a2ccb79fd5212c378b2c95"
HF_SOURCE = "src/transformers/models/gemma4/modeling_gemma4.py"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rms(x, weight=None, epsilon=1e-6):
    result = x * np.power(np.mean(x * x, axis=-1, keepdims=True) + np.float32(epsilon), np.float32(-.5))
    return result if weight is None else result * weight


def sigmoid(x):
    # Stable positive/negative forms avoid overflow in clipped GLU/SiLU values.
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + e), e / (1 + e)).astype(np.float32)


class AudioEncoder:
    def __init__(self, model):
        from gguf import GGUFReader
        from gguf.quants import dequantize
        self.reader = GGUFReader(str(model), mode="r")
        self.dequantize = dequantize
        self.tensors = {tensor.name: tensor for tensor in self.reader.tensors}
        self.cache = {}
        self.hidden = int(self.metadata("clip.audio.embedding_length", 1024))
        self.heads = int(self.metadata("clip.audio.attention.head_count", 8))
        self.layers = int(self.metadata("clip.audio.block_count", 12))
        self.epsilon = float(self.metadata("clip.audio.attention.layer_norm_epsilon", 1e-6))
        self.dimension = self.hidden // self.heads
        self.states = {}

    def metadata(self, name, default):
        field = self.reader.fields.get(name)
        return default if field is None else field.contents()

    def weight(self, name):
        if name not in self.cache:
            tensor = self.tensors[name]
            self.cache[name] = np.asarray(self.dequantize(tensor.data, tensor.tensor_type), dtype=np.float32).reshape(tuple(reversed(tensor.shape)))
        return self.cache[name]

    def linear(self, x, prefix):
        # GGUF stores the converter's HF ClippableLinear scalar bounds.
        low = float(self.weight(prefix + ".input_min")[0]) if prefix + ".input_min" in self.tensors else -np.inf
        high = float(self.weight(prefix + ".input_max")[0]) if prefix + ".input_max" in self.tensors else np.inf
        result = np.clip(x, low, high) @ self.weight(prefix + ".weight").T
        if prefix + ".bias" in self.tensors:
            result += self.weight(prefix + ".bias")
        low = float(self.weight(prefix + ".output_min")[0]) if prefix + ".output_min" in self.tensors else -np.inf
        high = float(self.weight(prefix + ".output_max")[0]) if prefix + ".output_max" in self.tensors else np.inf
        return np.clip(result, low, high).astype(np.float32)

    def normalize(self, x, name):
        return rms(x, self.weight(name + ".weight"), self.epsilon)

    def sscp(self, x, layer):
        prefix = f"a.conv1d.{layer}"
        kernel = self.weight(prefix + ".weight")  # [out,in,time,frequency]
        padded = np.pad(x, ((1, 1), (1, 1), (0, 0)))
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(0, 1))[::2, ::2]
        result = np.einsum("tfcij,ocij->tfo", windows, kernel, optimize=True, dtype=np.float32)
        centered = result - np.mean(result, axis=-1, keepdims=True)
        result = centered / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + np.float32(self.epsilon))
        result *= self.weight(prefix + ".norm.weight")
        if prefix + ".norm.bias" in self.tensors:
            result += self.weight(prefix + ".norm.bias")
        return np.maximum(result, np.float32(0))

    def feed_forward(self, x, prefix, suffix):
        residual = x
        x = self.normalize(np.clip(x, -1e10, 1e10), prefix + ".ffn_norm" + suffix)
        x = self.linear(x, prefix + ".ffn_up" + suffix)
        x *= sigmoid(x)
        x = self.linear(x, prefix + ".ffn_down" + suffix)
        x = self.normalize(np.clip(x, -1e10, 1e10), prefix + ".ffn_post_norm" + suffix)
        return residual + np.float32(.5) * x

    def attention(self, x, prefix, past):
        residual = x
        x = self.normalize(np.clip(x, -1e10, 1e10), prefix + ".attn_pre_norm")
        length = x.shape[0]
        q = self.linear(x, prefix + ".attn_q").reshape(length, self.heads, self.dimension)
        k = self.linear(x, prefix + ".attn_k").reshape(q.shape)
        v = self.linear(x, prefix + ".attn_v").reshape(q.shape)
        q *= np.float32(self.dimension ** -.5 / math.log(2))
        # The GGUF converter already applies softplus to the HF parameter.
        q *= self.weight(prefix + ".per_dim_scale.weight")
        k *= np.float32(math.log(1 + math.e) / math.log(2))
        half = self.hidden // 2
        frequencies = np.exp(np.arange(half, dtype=np.float32) * np.float32(-math.log(10000) / (half - 1)))
        positions = np.arange(past, -1, -1, dtype=np.float32)
        angles = positions[:, None] * frequencies[None, :]
        signals = np.concatenate((np.sin(angles), np.cos(angles)), axis=-1)
        relative = self.linear(signals, prefix + ".attn_k_rel").reshape(past + 1, self.heads, self.dimension)
        result = np.empty_like(q)
        # Dense per-query local attention is independent of HF's blocked
        # relative-shift indexing and includes the distance-past boundary.
        for row in range(length):
            keys = np.arange(max(0, row - past), row + 1)
            distance = row - keys
            logits = np.einsum("hd,khd->hk", q[row], k[keys], dtype=np.float32)
            logits += np.einsum("hd,khd->hk", q[row], relative[past - distance], dtype=np.float32)
            logits = np.tanh(logits / np.float32(50)) * np.float32(50)
            probabilities = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
            result[row] = np.einsum("hk,khd->hd", probabilities, v[keys], dtype=np.float32)
        x = self.linear(result.reshape(length, self.hidden), prefix + ".attn_out")
        return residual + self.normalize(np.clip(x, -1e10, 1e10), prefix + ".attn_post_norm")

    def light_convolution(self, x, prefix):
        residual = x
        x = self.linear(self.normalize(x, prefix + ".conv_norm"), prefix + ".conv_pw1")
        data, gate = np.split(x, 2, axis=-1)
        x = data * sigmoid(gate)
        kernel = self.weight(prefix + ".conv_dw.weight")  # [channel,kernel]
        width = kernel.shape[1]
        padded = np.pad(x, ((width - 1, 0), (0, 0)))
        windows = np.lib.stride_tricks.sliding_window_view(padded, width, axis=0)
        x = np.einsum("tck,ck->tc", windows, kernel, dtype=np.float32)
        x = self.normalize(np.clip(x, -1e10, 1e10), prefix + ".norm_conv")
        x *= sigmoid(x)
        return residual + self.linear(x, prefix + ".conv_pw2")

    def encode(self, mel, past=11):
        x = self.sscp(mel[..., None], 0)
        self.states["sscp0"] = x.copy()
        x = self.sscp(x, 1)
        self.states["sscp1"] = x.copy()
        x = self.linear(x.reshape(x.shape[0], -1), "a.input_projection")
        self.states["input_projection"] = x.copy()
        for layer in range(self.layers):
            prefix = f"a.blk.{layer}"
            x = self.feed_forward(x, prefix, "")
            self.states[f"layer{layer}_ffn0"] = x.copy()
            x = self.attention(x, prefix, past)
            self.states[f"layer{layer}_attention"] = x.copy()
            x = self.light_convolution(x, prefix)
            self.states[f"layer{layer}_convolution"] = x.copy()
            x = self.feed_forward(x, prefix, "_1")
            x = self.normalize(np.clip(x, -1e10, 1e10), prefix + ".ln2")
            self.states[f"layer{layer}"] = x.copy()
        x = self.linear(x, "a.pre_encode.out")
        self.states["encoder_output_projection"] = x.copy()
        x = rms(x, epsilon=self.epsilon)
        self.states["embedding_norm"] = x.copy()
        x = self.linear(x, "mm.a.input_projection")
        self.states["soft_rows"] = x.copy()
        return x


def load_waveform(path):
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getframerate() != 16000 or audio.getsampwidth() != 2:
            raise ValueError("This numerical oracle requires mono 16kHz PCM16 WAV input")
        return np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32) / np.float32(32768)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=Path, help="Optional compressed NPZ containing intermediate tensors")
    parser.add_argument("--candidate", type=Path, help="Little-endian F32 [tokens,512] soft rows")
    parser.add_argument("--min-cosine", type=float, default=.9999)
    parser.add_argument("--max-absolute-error", type=float, default=.01)
    parser.add_argument("--past-distance", type=int, default=11, help="HF context-left13 passes window12 with exclusive comparison: maximum admitted distance11")
    parser.add_argument("--log-policy", choices=["huggingface_add", "llama_clamp"], default="huggingface_add")
    args = parser.parse_args()
    if args.past_distance < 0:
        parser.error("past-distance must be nonnegative")
    spec = importlib.util.spec_from_file_location("audio_frontend", Path(__file__).with_name("generate-gemma4-audio-frontend.py"))
    frontend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frontend)
    samples = load_waveform(args.wav)
    mel = frontend.spectrogram(np.pad(samples, (0, (-samples.size) % 128)), args.log_policy)
    if mel.shape[0] == 0:
        raise ValueError("Audio must contain at least one valid mel frame")
    encoder = AudioEncoder(args.mmproj)
    output = encoder.encode(mel, args.past_distance)
    if not np.all(np.isfinite(output)):
        raise ValueError("Oracle returned nonfinite soft rows")
    report = dict(projector=str(args.mmproj.resolve()), projector_sha256=sha256(args.mmproj),
                  waveform=str(args.wav.resolve()), waveform_sha256=sha256(args.wav),
                  source_revision=HF_REVISION, source_path=HF_SOURCE,
                  source_url=f"https://github.com/huggingface/transformers/blob/{HF_REVISION}/{HF_SOURCE}",
                  method="Independent NumPy F32 SSCP/complete conformer, " +
                    ("HF log(mel+0.001)" if args.log_policy == "huggingface_add" else "explicit llama log(max(mel,0.001))") +
                    ", HF exclusive-left attention window12 (maximum distance11)",
                  audio_log_policy=args.log_policy, left_window_size=12,
                  past_distance_inclusive=args.past_distance, numpy=np.__version__,
                  input_mel_frames=mel.shape[0], shape=list(output.shape), soft_rows=output.tolist(),
                  limitations="Architecture parity for the recorded quantized projector and waveform; no speech-quality leaderboard or performance claim.")
    if args.states:
        args.states.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.states, mel=mel, **encoder.states)
        report["states"] = str(args.states.resolve())
    if args.candidate:
        candidate = np.fromfile(args.candidate, dtype="<f4").reshape(output.shape)
        if not np.all(np.isfinite(candidate)):
            raise ValueError("Candidate returned nonfinite soft rows")
        left, right = output.astype(np.float64), candidate.astype(np.float64)
        cosine = np.sum(left * right, axis=-1) / np.sqrt(np.sum(left * left, axis=-1) * np.sum(right * right, axis=-1))
        error = float(np.max(np.abs(left - right)))
        report["comparison"] = dict(candidate=str(args.candidate.resolve()), candidate_sha256=sha256(args.candidate),
                                    min_row_cosine=float(np.min(cosine)), max_absolute_error=error,
                                    passed=bool(np.min(cosine) >= args.min_cosine and error <= args.max_absolute_error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("shape", "input_mel_frames", "comparison") if key in report}, indent=2))
    return 0 if report.get("comparison", {}).get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
