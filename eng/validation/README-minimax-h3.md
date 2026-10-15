# MiniMax-H3 validation

Use the same checkpoint files and tokenizer for both engines. Keep upstream
ggml and the stable-diffusion.cpp baseline unchanged; the paired runner rejects
dirty or unavailable dependency checkouts. Put outputs in ignored `artifacts/`
or `docs/validation/`. A structural media pass does not establish perceptual
quality or benchmark parity.

The recorded 2026-10-13–14 evidence uses an M5 Pro with 48 GB locally and an A40
remotely. ggml is unchanged at `3612c0af93f2cfc8f048295300a61187723b12ae`;
the clean SD baseline is `d637cfb935feb2254898a8f1524046c958f3df80`, with pinned
ggml `d25b121ce1d6ae0c7e62ad080460d9fc1e46dd93`.

| Recorded case | TensorSharp | SD / preservation evidence |
|---|---:|---|
| Native Metal, 640×384×22, eight steps | 70.019 s | 102.262 s in the paired run |
| Direct MLX, same request, before packing → final | 158.333 → 87.054 s | Decoded RGB and WAV bit-identical across before-packing, after-packing and final-reader binaries |
| Earlier native CUDA binary, 640×384×22, 20 steps, warm-cache repeat | 108.299 s | SD 47.932 s; initial near-parity inference is superseded |
| Earlier bulk-prefault experiment, same 20-step request | about 66–69 s | SD comparison runs ranged 47.92–79.08 s; no general CUDA parity established |
| Final automatic-prefault native CUDA default, same request | 71.899 / 64.904 s | Corresponding alternating SD rounds: 145.020 / 50.238 s; exact RGB/WAV preservation across earlier and final binaries |
| Direct CUDA, same 20-step request | 228.98 s | Earlier same-input SD process 112.21 s |

These are new-process measurements with uncontrolled file/JIT caches, not
warm-model service benchmarks. Final MLX versus saved SD uses the same inputs
but was not a new alternating timing pair. Reader widening alone did not
establish an additional process improvement: 87.054 s final versus 86.318 s after
packing. Tiny smoke settings and successful file exports do not establish
perceptual quality. The inspected llama.cpp and vLLM trees have no identified
H3 AV diffusion execution route, so they are not executable generation baselines.

The selected native CUDA helper uses CPU-only bulk prefault of immutable pending
weights, automatically bounded by half the smaller of physical RAM and the
cgroup memory limit. Unknown capacity disables automatic warming.
`TS_H3_UPLOAD_PREFAULT=0` disables it and `1` forces it;
`TS_H3_UPLOAD_WORKERS` selects 1–8 workers (default 8). The independent managed denoiser read
retains `TS_H3_PREFAULT=3`; removing it did not establish a repeatable process
gain. Retain the exact policy/environment in each frozen-binary report.

The final native default passed all nine actual component oracle cases and
both alternating generation pairs with no skips. Eight before/experimental/final
TensorSharp generations have identical decoded RGB and WAV hashes. The final
warm SD repetition remains faster, so CUDA whole-process parity is not claimed.
`TS_H3_UPLOAD_PROFILE=1` reports the memory decision and separate prefault/upload
costs; it was enabled only for diagnostic oracle verification, not the final
generation timings.

Build the Release CLI and tests first. On the A40 worker, expose the installed
CUDA toolkit and compile compatible PTX rather than using the committed sm120
baseline:

```sh
PATH=/usr/local/cuda/bin:$PATH dotnet build InferenceWeb.Tests/InferenceWeb.Tests.csproj \
  -c Release -p:CudaArch=compute_86
```

Freeze the resulting CLI directory for each before/after comparison. Record
the native library, managed assemblies and PTX from that same directory. The
paired runner records those binary hashes, hardware, dependency revisions,
explicit sigma schedule, seed, conditioning hashes, media shape and waveform
finiteness. It alternates engine order across rounds. Whole-process timing
includes model loading and codecs; neither file-cache coldness nor warm-model
service latency is established by this runner.

```sh
python3 eng/validation/minimax-h3-oracles.py --backend cuda \
  --models /workspace/models/minimax-h3 --primitives \
  --output artifacts/h3/cuda-primitives
python3 eng/validation/minimax-h3-oracles.py --backend cuda \
  --models /workspace/models/minimax-h3 --output artifacts/h3/cuda-oracles
python3 eng/validation/minimax-h3-generation.py --backend ggml_cuda \
  --models /workspace/models/minimax-h3 --case quality --steps 20 --rounds 2 \
  --tensorsharp-cli /path/to/frozen/TensorSharp.Cli.dll \
  --sdcpp-cli /path/to/clean-sd-build/bin/sd-cli \
  --sdcpp-source /path/to/clean-sd-source --output artifacts/h3/paired-20step
```

The oracle runner rejects missing fixtures, skips and unexpected case counts.
Its text/DiT fixtures cover slices; the vision tower and VAEs have separate
coverage. `--case quality` defaults to eight steps on the base checkpoint;
specify `--steps 20` for the TensorSharp default. Dedicated FastH3 checkpoints
have separate trained schedules and are not interchangeable with base H3.

For synthetic conditioning coverage:

```sh
python3 eng/validation/generate-minimax-h3-conditioning.py \
  --output artifacts/h3/conditioning
```

Pass `--image` and `--end-image` for FL2VA, or `--checkpoint ref2va` together
with repeatable `--ref-image`/`--ref-audio` for Ref2VA. These small synthetic cases
exercise routing and output validity; they are not a perceptual dataset.
`minimax-h3-compare.py` rejects mismatched checkpoint, prompt, schedule, seed or
conditioning identities before comparing saved media. Conditioning paths may
differ across hosts if their roles/order and content hashes agree.

Reference audio needs an additional preprocessing check. TensorSharp caps or
pads standalone audio to the generated duration and floors to an 800-sample
hop; the inspected SD version preserves the full file and ceil-pads it. The
original one-second soundtrack versus a five-frame output therefore supplied
8 reference latents to TensorSharp and 40 to SD. Its output difference is not
evidence of an inference arithmetic bug or a matched-input parity failure.
SGLang also caps reference material by the output duration.

Canonicalize a common 32 kHz stereo PCM16 file explicitly for the five-frame
comparison; leave the original file and report intact:

```sh
python3 eng/validation/minimax-h3-canonicalize-reference-audio.py \
  --input artifacts/h3/conditioning/reference.wav \
  --output artifacts/h3/conditioning/reference-canonical-6400.wav \
  --frames 5 --fps 24
python3 eng/validation/minimax-h3-generation.py --backend ggml_cuda \
  --models /workspace/models/minimax-h3 --checkpoint ref2va --case smoke \
  --steps 8 --seed 44 --ref-image artifacts/h3/conditioning/first.png \
  --ref-audio artifacts/h3/conditioning/reference-canonical-6400.wav \
  --prompt "A glossy red ball sits on a blue table. The ball slowly rolls to the right. A soft rolling sound." \
  --tensorsharp-cli /path/to/frozen/TensorSharp.Cli.dll \
  --sdcpp-cli /path/to/clean-sd-build/bin/sd-cli \
  --sdcpp-source /path/to/clean-sd-source --output artifacts/h3/ref2va-canonical
```

The canonicalizer refuses existing outputs and empty/sub-hop results, preserves
the selected PCM bytes, and writes an adjacent `.wav.json` manifest containing
source/output hashes, sample counts and the scope **explicit benchmark
canonicalization, not pipeline change**. A source shorter than the selected
duration can still be padded differently by the engines; inspect effective
lengths rather than treating matching file hashes as sufficient.

The actual image-plus-canonical-audio eight-step pair achieved RGB PSNR
**37.674 dB** and stereo waveform cosine **0.999423**. This five-frame synthetic
request and the FL2VA first/last-frame case establish conditioning execution
and diagnostic agreement, not natural-media perceptual quality or speed parity.

`MiniMaxH3DitProbe` rejects inherited `TS_H3_DIT_LAYERS`. Its trace-free mode
measures complete DiT forwards including returned host arrays, separately from
text encoding, sampling and VAEs. Use one warmup and at least three repeats.
`minimax-h3-prediction-compare.py` binds saved velocities to the probe's digest,
checks exact input hashes against saved independent full50 IEEE F32 truth, and
compares the recovered sigma-one raw predictions. This does not independently
validate a multi-step generation. Trace callbacks/downloads change execution
synchronization and must be excluded from performance claims.

Pure CPU evidence-gate regressions:

```sh
python3 -m unittest discover -s eng/validation/tests \
  -p test_minimax_h3_prediction_compare.py -v
```

`MiniMaxH3MlxGroupNormProbe` measures the prior expression versus the fused
per-frame normalization with synchronized device evaluation.
`MiniMaxH3MlxQ4KPackProbe` compares lossless CPU packing independently of paging,
device upload and inference. Both are stage measurements. Full-generation
quality preservation additionally requires decoded RGB and waveform identity
for before/after frozen binaries when the intended change is bit preserving.

`H3MmapProbe.cpp` compares read-only shared/private mappings on one opened file,
with fresh mappings, reversed trial order, page touches or bounded copies, and
fault/CPU counters. `H3MmapNetProbe` inspects the actual .NET mapping flags through
Linux `smaps`. The measured cached 2 GiB NFS slice showed no meaningful mapping
flag benefit; production mapping semantics remain unchanged. These CPU probes
neither drop caches nor establish cold-file or GPU performance.

```sh
mkdir -p artifacts/h3
c++ -O2 -std=c++17 -pthread eng/validation/H3MmapProbe.cpp -o artifacts/h3/mmap-probe
artifacts/h3/mmap-probe /workspace/models/minimax-h3/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --offset 65184 --bytes 2147483648 --trials 4 --workers 8
dotnet run -c Release --project eng/validation/H3MmapNetProbe -- \
  /workspace/models/minimax-h3/qwen3vl_32b_minimax_h3-Q4_K_M.gguf
```

Current ignored evidence lives under `docs/validation/minimax-h3/optimization-current/`:

- `local-ggml-metal-quality/report.json` and `inspection.json`: actual native Metal pair and sampled-frame inspection.
- `local-mlx-quality-final/three-version-identity.json`: 158.333 / 86.318 / 87.054 s processes and exact decoded RGB/WAV hashes.
- `remote/generation-ggml-cuda-20step-balanced/report.json`: alternating two-round timing and matched-output comparisons, including the 47.932 versus 108.299 s result.
- `remote/generation-ref2va-canonical/report.json`: actual common 6,400-sample mixed-reference generation; `ref2va-reference-preparation-audit.json` explains the original mismatched context.
- `remote/generation-ggml-cuda-final-default-20step/report.json` and `remote/native-prefault-version-identity.json`: final automatic policy, both alternating pairs, unchanged dependency checks and exact before/after media preservation.
- `remote/ggml-cuda-final-oracles/report.json`: nine final native CUDA component cases; `remote/cuda-reader-final/report.json`: four affected direct CUDA VAE cases after vectorized reader conversion.
- `canonicalize-reference-audio-validation.json`: eight functional checks for exact PCM preservation, format/duration rejection, fresh-output protection and manifest integrity.

Skipped, unavailable model/device scenarios, long clips and unrun perceptual workloads must remain
explicit limitations, not passing coverage. For architecture and API details,
see [the model documentation](../../docs/models/minimax-h3.md).
