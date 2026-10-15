# MiniMax-H3

MiniMax-H3 generates video and native 32 kHz stereo audio in one packed diffusion
sequence. TensorSharp exposes it through `MiniMaxH3Model.GenerateVideo`, the CLI
and the video HTTP routes. FL2VA handles text and keyframes; Ref2VA handles text
and reference media.

Six backends have separate execution paths: `ggml_metal`, `ggml_cuda` and
`ggml_cpu` use TensorSharp-owned native graphs against unchanged ggml;
`mlx`, `cuda` and `cpu` use direct tensor implementations. CUDA and MLX execute
DiT, text/vision encoding, audio VAEs and the causal 3-D video encoder on their
selected devices. The `cpu` forward path uses managed C# arithmetic.

```sh
TensorSharp.Cli --model minimax_h3_fl2va_pruned-Q4_K.gguf --backend cuda \
  --prompt "A red fox walks through a quiet forest. Forest birds chirp." \
  --width 256 --height 160 --video-frames 5 --diffusion-steps 1 --cfg 1.0 \
  --diffusion-seed 42 --output fox.mp4
```

This is a small execution smoke example, not a quality preset. It writes video
and a stereo WAV sidecar. The larger 640×384×22, eight-step workload is a separate
validation case.

## Current validation

Validation recorded on 2026-10-13–14 uses an Apple M5 Pro with 48 GB for the local
Metal/MLX lanes and an NVIDIA A40 with about 46 GB VRAM for the remote CUDA/CPU
lanes. The remote worker has about 46.5 GB RAM and a 7.65-core CPU quota; CPU
runs use eight threads.

The dependency is unchanged ggml
`3612c0af93f2cfc8f048295300a61187723b12ae`. The independent clean
stable-diffusion.cpp baseline is
`d637cfb935feb2254898a8f1524046c958f3df80`, with its official pinned ggml
`d25b121ce1d6ae0c7e62ad080460d9fc1e46dd93`. The five weight files and tokenizer
were hash-verified from `unsloth/MiniMax-H3-GGUF` revision
`d629413c2e5b51b38c453668b75ca3b06ca92703`.

| Backend | Actual current component coverage | Whole-generation coverage |
|---|---|---|
| `ggml_metal` | Corrected DiT oracle 1/1, zero skips; other unchanged component gates exercised | Matched 640×384×22, eight-step generation and sampled-frame inspection completed |
| `mlx` | Final 39-case device/cache/pipeline suite, post-repacking text/DiT oracles and three final reader/VAE oracles; zero skips | Final matched eight-step generation; RGB and WAV bit-identical across optimization versions; I2V technical smoke completed |
| `ggml_cuda` | Nine final component oracle cases with automatic CUDA prefault active; zero skips | Final default: two alternating 640×384×22, 20-step pairs; exact decoded RGB/WAV preservation; synthetic FL2VA and canonical mixed Ref2VA completed |
| `cuda` | Nine component oracle cases, 16 device primitives, 30 cache/pipeline/argument regressions and four final reader/VAE cases; zero skips | Matched 640×384×22, 20-step generation completed; whole-process performance remains behind SD |
| `ggml_cpu` | Corrected DiT oracle 1/1, zero skips; unchanged component gates exercised | Corrected paired generation pending |
| `cpu` | Corrected DiT oracle 1/1, zero skips; lazy decoder encode/decode gates 4/4, zero skips | Corrected paired generation pending |

Component gates cover a two-layer text slice, two-block DiT slice, full vision
tower, video/audio VAE encode and decode, and five-/17-frame causal video encode.
These small fixtures do not establish all resolutions, conditioning routes or
whole-generation performance. The corrected DiT fixtures replaced outputs from
the earlier oracle that also incorrectly applied refiner RoPE. Skips, absent
models and unavailable devices are not passes.

Three whole-path corrections invalidate the earlier generation benchmarks:
text refinement has no RoPE; the sampler includes audio carrier drift; and the
final audio carrier is scaled by `audioShift/videoShift`. Earlier 2.4×/1.7× Metal
claims and historical generation-quality comparisons are superseded.

### Independent full-depth F32 reference

A streaming PyTorch oracle dequantizes the **same GGUF weights**, uses IEEE F32
with TF32 disabled and math SDPA, and runs all 50 blocks. Inputs are the exact
projection inputs captured through the unchanged SD CLI's public eval callback:
256×160×5, 22 text tokens, 118 packed tokens, sigma 1. This does not independently
validate the whole text encoder, every sampler step or the original dense weights
before quantization.

| Raw output versus independent F32 | Current direct CUDA relative L2 / cosine | Clean SD relative L2 / cosine |
|---|---|---|
| Audio prediction | 0.00007429 / 0.999999997 | 0.0177564 / 0.9998424 |
| Video prediction | 0.00031502 / 0.999999960 | 0.0199001 / 0.9998021 |

The corrected text-refiner output has relative L2 0.0001601 against F32 truth,
versus 0.0110527 for the clean baseline on the same input. Matching the baseline's
approximation would reduce this numerical accuracy.

The one-step physical endpoint `input + raw_prediction` was passed through the
real public stereo VAE. Direct CUDA versus the independent full-depth endpoint
has waveform cosine 0.999999745 and relative L2 0.0007782 before encoding. After
both outputs pass through PCM16, cosine is 0.99998837 and the maximum difference
is one PCM16 LSB. Clean SD PCM16 versus that same reference has cosine 0.686735.
This is a matched-input **one-denoiser-step** waveform check, not an oracle for
an eight-step clip or perceptual quality.

### Actual whole-process smoke

The corrected default CUDA CLI and clean SD CLI both exited zero and produced
256×160, five-frame video with finite 32 kHz stereo at seed 42, one Euler step,
shift 12, CFG 1 and independently verified CUDA Philox noise. Decoded video PSNR
was 34.71 dB; waveform cosine between engines was 0.69447. Clipping is fixed and
levels are close, but cross-engine waveform parity is not established. The
independent F32 tests above explain why agreement with SD alone is insufficient
as a correctness criterion.

The larger eight-step Metal pair has decoded RGB PSNR 35.54 dB and shared-prefix
stereo waveform cosine 0.99814. Sampled frames show matching fox/forest composition,
motion and tree occlusion without obvious seams or black/saturated output. In
the earlier 20-step A40 runs, native CUDA versus SD has RGB PSNR 35.01 dB and waveform
cosine 0.95177; direct CUDA has RGB PSNR 29.88 dB and waveform cosine 0.95102.
Those waveform differences are not an independent multi-step correctness oracle.

Structural validity, independent numerical agreement and perceptual quality are
reported separately. Tiny 256×160 two-step runs produce coarse texture in all
three compared engines; successful smoke execution is not a quality preset.
Long clips, multiple natural prompts/seeds and a formal perceptual audio/video
evaluation remain unestablished. The inspected llama.cpp and vLLM revisions have
MiniMax language models but no identified H3 AV diffusion route; neither supplies
an executable H3-generation performance baseline here. SGLang's base H3 and
separately trained eight-step FastH3 checkpoints must not be conflated.

## Performance and precision

Actual new-process measurements include loading and output encoding. They do
not control the OS file cache or establish warm-model service latency.

| Device/backend | Geometry / steps | TensorSharp process | Clean SD process |
|---|---|---:|---:|
| M5 Pro `ggml_metal` | 640×384×22 / 8 | 70.019 s | 102.262 s |
| M5 Pro `mlx`, final | 640×384×22 / 8 | 87.054 s | 102.262 s |
| A40 `ggml_cuda`, initial pair | 640×384×22 / 20 | 110.93 s | 112.21 s |
| A40 `ggml_cuda`, earlier binary, warm-cache repeat | 640×384×22 / 20 | 108.299 s | 47.932 s |
| A40 `ggml_cuda`, final automatic-prefault default | 640×384×22 / 20 | 71.899 / 64.904 s | 145.020 / 50.238 s, corresponding alternating rounds |
| A40 `cuda` | 640×384×22 / 20 | 228.98 s | 112.21 s |

The native Metal pair is about 1.46× faster. Final MLX uses the same inputs as
that saved SD run, but its timing was measured later, not in a new alternating
pair. The initial native CUDA results were near parity, but the alternating
two-round run included SD at 47.932 s against TensorSharp at 108.299 s. This
supersedes an inference of general CUDA parity from the initial result.
Direct CUDA's strict F32 arithmetic also remains slower at the larger token
count despite the projection improvement below.

Native CUDA now reads immutable pending weight pages in parallel before the
original calling-thread uploads. Automatic warming selects bindings of at least
1 MiB only when their aggregate fits within half the smaller physical/cgroup
memory capacity. Unknown capacity disables it. `TS_H3_UPLOAD_PREFAULT=0` disables
warming, `1` forces it, and `TS_H3_UPLOAD_WORKERS` accepts 1–8 (default 8).
The independent managed denoiser read retains its existing default policy.
This TensorSharp-owned code uses public ggml backend APIs and never registers,
rewrites or copies the complete mapped source into private storage.

The final default passed nine component oracles and both alternating generation
pairs. Its 64.904–71.899 s processes improve on the earlier 108.299–109.218 s
TensorSharp repetitions, with decoded RGB and WAV identical across all eight
before/experimental/final runs. File-cache states were uncontrolled: the final
warm SD repetition still took 50.238 s against TensorSharp's 64.904 s, so general
CUDA whole-process parity remains unestablished. Upload profiling is opt-in via
`TS_H3_UPLOAD_PROFILE=1` and was excluded from these generation timings.

On the direct `cuda` backend, H3 quantized projections default to bounded
Q4_K/Q6_K/Q8_0/Q4_0 conversion followed by
per-call **pedantic F32** GEMM (69). Raw quantized weights retain lease-owned
device residency; converted scratch is capped at 64 MiB. Inputs and outputs
retain F32 range, and the near-cancellation regression also checks default
dispatch. `TS_H3_CUDA_F32_BANDS=0` restores the original F32 kernels;
`1` or `tf32` explicitly selects approximate TF32 products for large matrices.
Other model dispatch and the shared cuBLAS math mode are unchanged. Mapped dense
projections retain their previous size-dependent precision policy.

On an exclusive A40, identical 118-token inputs through the same frozen full50
probe gave these actual trace-free forwards, with bit-identical repeats per mode:

| CUDA DiT mode | First forward | Warm median, three repeats |
|---|---:|---:|
| Original quantized F32 kernels | 51.092 s | 5.515 s |
| Default bounded, pedantic F32 bands | 33.199 s | 0.604 s |

The roughly 9.1× warm gain covers DiT `Forward`, including returned host arrays.
It excludes text encoding, the sampler loop, VAEs and codecs. File-cache state
was not controlled, and the first forward includes device weight upload.

The default's same-input full50 errors against saved independent F32 truth are
0.00007429 audio / 0.00031502 video relative L2, essentially unchanged from the
original F32 kernels. The comparator verifies input hashes and the probe's
saved velocity digest. The probe rejects inherited layer-truncation overrides.
The 16 primitive gates include quantization types, rows 1/16/33/256, tiny/large
activations, partial bands and near cancellation. The optional TF32 mode's prior
full50 errors were 0.0003162 / 0.0007305; it can lose adversarial cancellation
residuals and is not the default.

Direct DiT now retains one exact, device-resident refined prompt, following
SGLang's separation of request-static text refinement. Byte comparison detects
caller mutation; conditioning changes, residency release and disposal invalidate
it. Tracing deliberately executes every refiner boundary. Real checkpoint
miss/hit tests preserve every F32 output value. MLX also fuses per-frame centered
GroupNorm, affine and SiLU in the causal reference encoder: representative
operator medians improve 1.25–3.00×, with five-/17-frame independent encoder
oracles passing. These operator/refiner figures are not whole-generation gains.

MLX Q4_K loading now repacks GGML nibbles directly, in independent rows, avoiding
generic bit-pack loops and redundant zero initialization. Independent CPU tests
compare every packed byte and affine-parameter bit, and actual text/DiT device
oracles pass. Full MLX generation took 158.333 s before packing changes, 86.318 s
after packing and 87.054 s after the final reader change. Decoded RGB **and WAV
are bit-identical across all three frozen versions**, an observed 1.82×
before-to-final process improvement on this input. File/JIT caches were
uncontrolled, so every stage-time reduction cannot be attributed solely to
packing, and the reader change alone did not establish another process gain.

Safetensors F16 conversion uses vectorized widening in bounded spans. All 65,536
Half encodings match the original scalar F32 bits, including signed zero,
subnormals and NaN payloads. Hot CPU conversion medians improve 2.62–2.65×;
actual decoder and five-/17-frame encoder oracles pass. This conversion alone
does not establish another whole-generation speedup.

Reference audio comparisons require preprocessing care: TensorSharp caps inputs
to the selected output duration and a whole 800-sample hop; the inspected SD
version retains the complete standalone reference soundtrack. The original
one-second/five-frame mixed-reference pair therefore compared different packed
contexts and is not parity evidence. With an explicit 6,400-sample canonical
reference in both engines, Ref2VA achieved RGB PSNR 37.674 dB and stereo waveform
cosine 0.999423. The five-frame synthetic FL2VA first/last-frame and mixed Ref2VA
requests both completed; these are conditioning execution/diagnostic cases,
not a natural-media perceptual benchmark.

The CUDA mapped-F16 VAE option `TS_H3_CUDA_MAPPED_VAE_WEIGHTS=1` also remains
opt-in: measured cold construction plus first decode improved from about 31.5 s
to 20.9 s, while warm decode regressed from about 0.213 s to 0.253 s in two
reversed pairs. The default keeps expanded F32 decoder weights. Decoder loading
is lazy on direct backends, so encoder-only requests avoid that expansion; the
first decode's deferred loading must be included in cold timing.

The repository CUDA PTX snapshot was built with CUDA 12.8 for sm120; actual A40 execution
uses locally compiled sm86 PTX. The latest addition retains all 160 preceding
entry bodies unchanged and appends one H3 conversion entry. No ggml patches
are applied.

## Files

Four networks cooperate, so a run needs four files. The two denoisers are
**separate checkpoints, not settings** — which one you load decides what
conditioning it accepts.

| File | Size | Source |
|---|---|---|
| `minimax_h3_fl2va_pruned-Q4_K.gguf` | 10.64 GiB | [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF) — text + keyframes |
| `minimax_h3_ref2va_pruned-Q4_K.gguf` | 10.60 GiB | same repo — text + references |
| `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` | 16.97 GiB | same repo — the text encoder, shared (`-Q2_K_M.gguf`, 12.20 GiB, pairs with a smaller denoiser) |
| `minimax_h3_video_vae_fp16.safetensors` | 5.21 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) (also mirrored under `vae/` in the unsloth repo) |
| `minimax_h3_audio_vae_fp32.safetensors` | 0.61 GB | same repo — omit for silent video |

One model set occupies about 35.5 GB on disk. The second denoiser shares the
text encoder, VAEs and tokenizer. Device residency is handed off between stages
when needed; activation buffers, temporary weights and other live contexts also
contribute to memory use.

Companions resolve automatically next to the denoiser, or with `--video-vae`,
`--video-text-encoder`, `--audio-vae`.

Two shipped configs download the lot on first run and only differ in their `model`
entry, so the encoder and the VAEs are fetched once:

```sh
TensorSharp.Server.Host --config config/minimax-h3-fl2va.json # keyframes
TensorSharp.Cli         --config config/minimax-h3-ref2va.json \
  --ref-image person.png --prompt "…" --output out.mp4        # references
```

Both name `"backend": "ggml_cuda"` (a `--backend` on the command line wins) and set
width 640, height 384, 22 frames, 24 fps. Neither sets steps or guidance: a shipped
config has to parse on both hosts, and the two spell steps differently
(`--video-steps N` on the server, `--diffusion-steps N` on the CLI), while the
server takes no `--cfg` at all — so the model's own defaults apply.

> **The text-encoder GGUF carries no tokenizer**, and that is the one thing a config
> cannot fetch for you: auto-download fills in options that are **flags**, and the
> tokenizer is not one. Put three files from the `processor/` folder of
> [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe/processor)
> beside it — `vocab.json`, `merges.txt` and `tokenizer_config.json` — or point
> `TS_VIDEO_TOKENIZER` at the folder holding them. (The TensorAgent Mac app's catalog
> entries download all three; see [TensorAgent](../../TensorAgent/README.md).) Neither
> published GGUF has any metadata at all, so TensorSharp identifies H3 by its tensors
> rather than by an architecture string (arch keys `minimax-h3` / `minimax_h3` are
> accepted when a file does declare one).

> **`tokenizer_config.json` is the easy one to leave out, and pictures need it.** It
> alone defines `<|vision_start|>`, `<|image_pad|>` and `<|vision_end|>`; they are not
> in `vocab.json`. Without it each marker is split into ordinary pieces, the prompt
> comes out *longer* than the placeholders it was built from, and the one-to-one check
> after tokenizing — which only caught a shorter prompt — passed, so the vision features
> were written over the wrong positions and the clip quietly ignored the picture. A
> keyframe request, or a reference request with a photo or a clip in it, is now refused
> before the vision tower runs, with an error naming the file. Text-to-video never
> needed it, and neither do references that are all sound recordings: a soundtrack
> reaches the prompt only as its `<Audio N>:` label.

> **`--cfg 1.0` is required.** H3 is CFG-distilled and TensorSharp refuses other
> values. The model defaults to 20 steps. The smaller examples below describe
> supported requests; their presence is not a current perceptual-quality gate.

## Conditioning modes

| Inputs | Mode | Checkpoint |
|---|---|---|
| Text | `t2v` | FL2VA or Ref2VA |
| First photo | `i2v` | FL2VA |
| First and/or last photo | `fl2v` | FL2VA |
| Reference images, clips or soundtracks | `ref` | Ref2VA |

`--video-mode` makes the choice explicit; otherwise inputs select it. Keep
`ref2va` in a reference checkpoint's filename. Requests mixing keyframes and
named references, or using the wrong checkpoint partition, are rejected.
Synthetic first/last-frame and canonical image-plus-audio reference runs have
completed. Natural-media reference quality, reference clips and broader prompt/
seed coverage remain unestablished.

```sh
TensorSharp.Cli --model minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_metal \
  --image start.png --end-image end.png --video-mode fl2v \
  --prompt "A slow cinematic push-in" --diffusion-seed 42 \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 20 --cfg 1.0 \
  --output morph.mp4

TensorSharp.Cli --model minimax_h3_ref2va_pruned-Q4_K.gguf --backend ggml_metal \
  --ref-image person.jpg --ref-audio song.wav --video-mode ref \
  --prompt "She performs on a small club stage" --diffusion-seed 42 \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 20 --cfg 1.0 \
  --output stage.mp4
```

A keyframe is encoded both as a video latent and as Qwen3-VL vision features.
When the output aspect differs, it is centre-cropped to fit; leaving size unset
uses its aspect. Ref2VA references keep their own aspect and do not choose the
output canvas. `--ref-image` can repeat up to nine times. Each reference adds
vision/prompt or latent tokens to the same attended sequence.

`--ref-video` accepts a file or a frame directory. Pair its soundtrack separately
with `--ref-video-audio`, by index; use `--ref-audio` for standalone soundtracks.
Clips are sampled at 24 fps, fitted to the 17k+5 frame grid and capped by output
length. Standalone sound is resampled to 32 kHz stereo, capped or zero-padded to
the output duration, then floored to a whole 800-sample VAE hop. A reference
video's paired soundtrack uses that reference clip's duration. The inspected
SD baseline instead retains the full standalone reference soundtrack; identical
input files therefore do not automatically give matched reference context.
Temporary extracted frames are removed; supplied frame directories are preserved.

## Sizes and settings

- Spatial dimensions round up to multiples of 32. The default is 640×384, or
  that area at a conditioning image's aspect.
- Frame counts round up to 5, 22, 39, 56, … (`17k+5`); the default is 22.
- H3 uses 24 fps, CFG 1 and its own Euler flow schedule. The default is 20 steps.
  Wan-specific negative-prompt, sampler and guidance-cache options do not change H3.
- `--diffusion-seed` selects generation noise. `--seed` controls text sampling
  and is not the H3 noise seed.
- Video decode is spatially tiled and temporally chunked. Long-clip finiteness
  and perceptual quality still require current full-generation gates.

## HTTP API

Three routes share one parser: `POST /api/video-generate`,
`POST /api/video-generate/stream` (same body, SSE progress ticks) and the
OpenAI-shaped `POST /v1/videos/generations`.

```sh
curl -s localhost:5000/api/video-generate -H 'content-type: application/json' -d '{
  "prompt": "a red fox trotting through falling snow, cinematic",
  "width": 640, "height": 384, "frames": 22, "steps": 8, "cfg": 1.0,
  "imagePath": "card.jpeg", "videoMode": "i2v"
}'
```

Reference conditioning uses the same route against the Ref2VA checkpoint:

```sh
curl -s localhost:5000/api/video-generate -H 'content-type: application/json' -d '{
  "prompt": "the same woman sits at a table in a sunlit cafe by a window, wide shot",
  "width": 640, "height": 384, "frames": 22, "steps": 20, "cfg": 1.0,
  "referenceImages": ["person.jpg", "bottle.png"], "videoMode": "ref"
}'
```

Returns `{ ok, url, audioUrl, audioMuxed, width, height, frames, fps, seed, codec, elapsedSeconds }`.
`audioUrl` is null when the model produced no track; `audioMuxed` says whether the MP4
carries the soundtrack as well ([below](#audio-output)). The full field set is `prompt`,
`width`, `height`, `frames`, `steps`, `cfg`, `fps`, `seed`, `flowShift`,
`imagePath` (or inline base64 `image`), `videoMode`, `generateAudio`, `endImage`,
`referenceImages`, `referenceVideos`, `referenceAudios`, `referenceVideoAudios`;
the Wan fields `negativePrompt`, `sampler`, `cfgCacheStride` and `cfg2` are
accepted and ignored by H3. Most of them are **camelCase only** —
`videoMode`, `generateAudio`, `endImage` and the four `reference*` lists are the
only ones that also accept a snake_case spelling (`video_mode`, `generate_audio`,
`end_image`, `reference_images`, …), and camelCase wins when both are present.
Sending `image_path` or `flow_shift` is silently ignored rather than rejected, so
prefer camelCase everywhere. `referenceVideoAudios` pairs **by index** with
`referenceVideos`, the same positional rule `--ref-video-audio` follows. `imagePath` / `endImage` /
`referenceImages` name files previously uploaded through `/api/upload`; paths
outside the upload directory are rejected. `/v1/videos/generations` shares the same
parser, so the same seven fields accept snake_case there too (`video_mode`,
`reference_images`, …); it returns `audio_url` rather than `audioUrl`, and no mux flag.

A model rejection — the wrong checkpoint for the mode, or keyframes together with
references — comes back as a **400 carrying the model's own message**, not a generic
500, so the file to load instead is in the response body.

`GET /api/models` answers with a `video` object (null for every non-video model)
carrying `family` (`minimax-h3`), `supportsAudio`, `supportsImageConditioning`,
`supportsEndImageConditioning`, `supportsReferenceConditioning` and
`maxReferenceImages`. That is how the Web UI decides whether to offer a first frame,
a last frame or up to nine references, instead of pattern-matching an architecture
string.

### Audio output

The soundtrack is always written as a sidecar WAV beside the MP4. The desktop
provider muxes audio when FFmpeg is available; its OpenCV/OS fallback writes
video alone and reports `audioMuxed: false`. To combine the sidecar manually:

```sh
ffmpeg -i fox.mp4 -i fox.wav -c:v copy -c:a aac fox_with_audio.mp4
```

Where the encoder can mux — AVAssetWriter on Apple, which is what the
[TensorAgent Mac app](../../TensorAgent/README.md) writes with — the MP4 also carries the
soundtrack as an AAC track (2 channels, 32 kHz, 128 kbit/s) with the index at the front
of the file, and the reply says `audioMuxed: true`. Codec differences are included when comparing exported soundtracks.

`--no-audio` (`"generateAudio": false`) skips the audio decode entirely, saving the
audio VAE's time and memory; video-only models ignore it.

## Architecture

Recorded here because none of it is inferable from the checkpoints, which carry no
metadata. Cross-checked against stable-diffusion.cpp, the upstream reference
PyTorch, and the GGUF tensor tables.

### Diffusion transformer

50 blocks, ~19.3 B parameters, single-stream: text, conditioning frames, target
audio and target video are ONE sequence under full bidirectional attention. There
is **no cross-attention**.

| | |
|---|---|
| hidden | 5376 |
| attention | 56 heads × 128 = **7168 inner**, wider than the 5376 residual stream |
| MLP | SwiGLU, gate first |
| patch | (t, h, w) = (1, 2, 2); video 24 ch → 96-element patch |
| audio | 32 ch, one token per latent frame per stereo channel |

Within a token the 96 video values are **channel-major, patch-minor**
(`c*4 + ky*2 + kx`). The transposed order has identical statistics and silently
scrambles every token.

**AdaLN** uses a learned curve table, not a timestep MLP: `adaln_t_table` is
`[8, 1025]`, interpolated at the timestep, and one `Linear(8 → 96768)` per block
emits 6 modulation vectors for each of 3 modalities (0 = visual, 1 = text,
2 = audio). A token run picks `timestep*3 + modality`.

**RoPE** is 3-axis with a learned 16-entry `inv_freq`, so 96 of 128 head dims
rotate. Positions are **continuous floats**: time is measured in audio-latent units
(1/40 s) so both streams share one timeline, and the spatial axes are normalized by
the frame's geometric mean extent.

### Text refinement and the dual audio schedule

The two text-refiner blocks use bidirectional QK-normalized attention **without
RoPE**. The later packed trunk applies the continuous three-axis positional table.
Earlier TensorSharp and fixture-oracle code incorrectly rotated the refiner;
those DiT outputs are superseded.

Video uses shift 12 and audio shift 3. The sampler carries
`carrier = (sigma_video / sigma_audio) * physical_audio`. Before DiT, the pipeline
restores physical audio. The DiT component still returns the raw prediction
multiplied by the negative shift derivative, as its component oracle specifies.
The sampler converts that prediction into a carrier velocity, including the
noise-cancelling drift term. At the final endpoint it multiplies the carrier by
`audioShift / videoShift` before the VAE's channel mean/std normalization.

At sigma 1 with shifts 12/3, one Euler step simplifies to
`physical_clean = physical_input + raw_prediction`. Independent trajectory,
custom-shift and nonzero clean-endpoint tests cover these conversions.

### Text encoder

Qwen3-VL-32B truncated to 50 language layers **with the final norm removed** — the
DiT consumes the raw layer-50 hidden state, whose massive-activation outliers
(absmax ~15 000) are load-bearing rather than a bug. **There is no chat template**:
the prompt is the raw text. Applying one shifts every hidden state.

### Video VAE

16× spatial, 4× temporal. The decoder is a **pure 36-layer transformer** — no
deconvolutions. Each latent voxel is one token, and a final `Linear(2048 → 3072)`
plus depth-to-space does the whole upsample in one step.

Decoding is **tiled at 256 px, and that is a correctness requirement, not an
optimization**: the decoder's RoPE coordinates are length-normalized over whatever
extent it is handed, and it only works over the extent it was trained on. Decoding
a 640×384 frame in one call produces visibly sheared output even though every
individual tile is exact.

The encoder is a causal 3-D CNN, but for the single frame image conditioning needs,
the causal padding is two leading *zero* frames — so only the last temporal slice
of each kernel contributes and the network reduces **exactly** to 2-D convolutions.

### Audio VAE

DAC/BigVGAN, 32 latent channels → stereo 32 kHz. Upsample rates {5,5,2,2,2,2,2}
multiply to 800, and 32000/800 = the 40 Hz latent rate. Uses alias-free snake
activations: every nonlinearity is wrapped in a 2× upsample / activate / 2×
downsample sandwich so it cannot fold high frequencies back into the band.

Like the video latent, the audio latent is **un-normalized before it is decoded**. The
denoiser works in a whitened space, so each channel plane is mapped back with
`x * latents_std + latents_mean` on its way into the decoder, and a reference
soundtrack's encoded latent is whitened the opposite way before the denoiser sees it.
`latents_std` averages about 1.9, so skipping the step gives neither silence nor noise
but a track that is spectrally plausible and merely wrong.

## Environment variables

| Variable | Scope |
|---|---|
| `TS_H3_CUDA_F32_BANDS` | Direct CUDA H3 quantized projections: default pedantic F32 bands; `0` selects legacy F32 kernels, `1`/`tf32` opts into approximate large TF32 products |
| `TS_H3_CUDA_MAPPED_VAE_WEIGHTS=1` | Opt-in direct CUDA mapped F16 VAE weights; cold/warm tradeoff above |
| `TS_H3_CUDA_BF16_PEDANTIC=1` | Diagnostic pedantic compute for H3 BF16 dense projections |
| `TS_H3_PHASE=1` | Stage timings including text, denoiser and VAEs |
| `TS_H3_TRACE=1` | Latent/velocity magnitude diagnostics |
| `TS_H3_DIT_TRACE_DIR` / `TS_H3_LINEAR_TRACE_FILE` | Synchronizing boundary/projection diagnostics; exclude from formal timings |
| `TS_H3_PREFAULT` | Denoiser file reading: 0 off, 1 serial, 2 early overlap, 3 default pipelined |
| `TS_H3_PREFAULT_THREADS` | Prefault streams; default 1 |
| `TS_H3_TE_GROUP` | Optional grouped native text-encoder trunk; off by default |
| `TS_H3_KEEP_RESIDENT=1` | Optional Metal residency policy between clips |
| `TS_H3_UPLOAD_PREFAULT` | Native CUDA bulk page-prefault policy: automatic half-memory/cgroup guard; `0` disables, `1` forces. Final default validation is pending above |
| `TS_H3_UPLOAD_WORKERS` | Native bulk-prefault workers (1–8, default 8); distinct from the pipeline's denoiser file-read streams |
| `TS_H3_UPLOAD_PROFILE=1` | Native weight-upload and graph-compute diagnostics; record separately from trace-free performance runs |

## Reproducing validation

Use `eng/validation/minimax-h3-oracles.py` to select an actual backend and reject
missing inputs, skipped cases or incomplete counts:

```sh
python3 eng/validation/minimax-h3-oracles.py --backend cuda \
  --models /path/to/minimax-h3 --primitives --output artifacts/h3/cuda-primitives
python3 eng/validation/minimax-h3-oracles.py --backend ggml_cpu \
  --models /path/to/minimax-h3 --output artifacts/h3/ggml-cpu-components
```

Regenerate independent component fixtures with the scripts in
`InferenceWeb.Tests/tools/`. DiT fixture generation must use the corrected
no-RoPE refiner. `MiniMaxH3GeometryTests` and `MiniMaxH3NoiseTests` cover geometry,
carrier conversions and independently generated SD Philox vectors.

`eng/validation/minimax-h3-generation.py` runs matched CLI inputs with explicit
seed, sigma list and SD CUDA RNG, records dependency revisions, binary hashes,
actual hardware and source cleanliness, and checks geometry, samples and finite
media. Its structural pass is **not** a quality-parity pass. It supports the small
smoke and a separate 640×384×22/eight-step case; use `--steps 20` for the default
schedule. `minimax-h3-canonicalize-reference-audio.py` creates an explicit common
PCM16 reference and provenance manifest for conditioning comparisons. A short
source can still be padded differently by the engines, so inspect effective
reference lengths as well as file hashes.

`minimax-h3-sdcpp-trace.cpp` uses the public eval callback and links unchanged
baseline objects. `minimax-h3-refiner-reference.py` and
`minimax-h3-full-dit-reference.py` arbitrate exact saved inputs with PyTorch F32;
`minimax-h3-endpoint-arbitration.py` runs those endpoints through the real stereo
VAE and PCM16. Trace downloads/callbacks are excluded from formal timings.

Logs, reports and snapshots stay in ignored `docs/validation/minimax-h3/` or
`artifacts/minimax-h3/`. Reusable tools and required test fixtures remain in
`eng/` and `InferenceWeb.Tests/Fixtures/`. Unrun workloads are recorded as pending,
not as passing coverage. See [the validation workflow](../../eng/validation/README-minimax-h3.md)
for reproduction commands and the current evidence index.
