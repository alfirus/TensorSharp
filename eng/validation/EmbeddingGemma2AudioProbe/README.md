# Independent audio tower checks

The validated EmbeddingGemma 2 GGML CPU instance now selects complete F32 SSCP
and conformer graphs by default. Explicit `TS_GEMMA4_AUDIO_SSCP_GRAPH=0` and
`TS_GEMMA4_AUDIO_BLOCK_GRAPH=0` retain the earlier route for diagnostics. Use
`--require-embedding-default true` for a no-override quality run. Set
`TS_GGML_CPU_THREADS` as well as the managed CPU thread budget when comparing
native tower latency; the media context has its own native CPU pool.

The optional existing `TSG_EMBEDDING_LINUX_BLAS=1` route also applies to the
validated CPU audio instance's measured learned projection shapes when a
pthread LP64 OpenBLAS provider reports exactly one worker. Set
`OPENBLAS_NUM_THREADS=1` before process start and optionally select its absolute
path with `TSG_EMBEDDING_BLAS_LIBRARY`. The ordinary constructor, unavailable
providers, other platforms and unsupported matrices keep their prior route.
Convolution, convolution GLU and attention arithmetic are unchanged. Run this
experiment with `--validated-precision true`; the no-override embedding-default
gate rejects these optional BLAS selectors. Existing
`TSG_EMBEDDING_CPU_PROFILE=1` additionally prints per-graph audio BLAS node/call,
worker and rejection counters for a quality diagnostic; omit profiling when
timing. Projection microbenchmarks alone do not establish tower/service gains.

Optional `--profile-phases true` records elapsed times at validation trace
boundaries, including QKV projection and the attention core. It adds tensor
readbacks and is a diagnostic, not the timing lane. Compare ordinary runs with
profiling absent for performance claims.

Non-16kHz decoding uses TensorSharp-owned rational polyphase Hann sinc
resampling (64 zero crossings, rolloff 0.99), with bounded cached kernels and a
fallback for unusual coprime rates. Native 16 kHz input is unchanged. The
independent fixture generator `eng/validation/generate-gemma4-audio-resampler.py`
executes `torchaudio.transforms.Resample` with these parameters; unit checks
cover random/impulse signals at eight rates, short boundaries, passband gain
and alias rejection. This is a wider filter than vLLM/SGLang's default
torchaudio configuration. HF file loading may select torchcodec or
librosa/soxr, so matching this selected bandlimited algorithm does not claim
bit identity with every HF codec pipeline.

For complete source-rate integration, compare the original 44.1/48 kHz WAVs with
the independently resampled Float32 WAVs, and check canonical 16 kHz audio
separately against the full HF NumPy model golden using
`eng/validation/embeddinggemma2-resampling-compare.py`. These integration checks
share the actual model for the resampling comparison; the canonical complete
golden remains the independent model gate.

This probe consumes the independent oracle's `[frames,128]` little-endian F32
mel array and exports raw audio soft rows before text pooling. Build with
`BuildProjectReferences=false` against a current Models build to keep an active
benchmark's shared binaries unchanged.

```sh
dotnet build eng/validation/EmbeddingGemma2AudioProbe -c Release \
  -p:BuildProjectReferences=false -p:TensorSharpSkipGgmlNative=true \
  -p:TensorSharpSkipMlxNative=true
dotnet eng/validation/EmbeddingGemma2AudioProbe/bin/Release/net10.0/EmbeddingGemma2AudioProbe.dll \
  --projector /absolute/path/embeddinggemma-2-mmproj-Q8_0.gguf \
  --mel docs/validation/embeddinggemma2-reference/audio-numpy-mel.f32 \
  --frames 99 --backend GGML_METAL \
  --output docs/validation/embeddinggemma2-reference/audio-metal-soft.f32 \
  --report docs/validation/embeddinggemma2-reference/audio-metal-soft.json
```

Backends are actual CPU, MLX, CUDA, GGML_CPU, GGML_METAL, or GGML_CUDA. Use
`embeddinggemma2-audio-reference.py --candidate <output.f32>` for a complete
NumPy comparison against the published Hugging Face architecture, including
`log(mel+0.001)` and maximum admitted past distance 11. The pinned Hugging Face
mask takes a left-window argument of 12 and uses the strict inequality
`query-key < 12`; earlier inclusive-distance-12 reports are superseded.
The independent source truth table is in
`InferenceWeb.Tests/Fixtures/EmbeddingGemma2/audio-attention-mask.json`.
Reports bind projector, mel,
binary and nested CUDA runtime PTX hashes. Verify both the report and process
exit status. Raw rows are unnormalized; inspect cosine and absolute differences
with their magnitude before attributing a backend precision error to model
architecture. Synthetic audio does not establish speech retrieval quality.

Optional `--warmup 3 --repeats 20 --minimum-duration-seconds 1` records warmed
tower timings, including device completion, F32 readback, and output tensor
disposal. Loading, media decoding, mel extraction, text pooling, and HTTP are
excluded. Freeze all runtime binaries before comparing kernels, vary only the
intended binary, run both orders, and keep other inference and builds idle.
These timings do not establish end-to-end performance parity with llama.cpp.

`--validated-precision true` requests the same per-instance F32 precision policy
as the EmbeddingGemma 2 wrapper. `--require-embedding-default true` additionally
rejects all audio precision/graph environment overrides, including explicit
`0` values, before allocating a backend. This gate checks the scoped default
without changing process-wide CUDA math modes. Omit both options to preserve
the existing Gemma chat constructor policy for a diagnostic comparison. Pair a
fresh Models/bridge build with its matching native library: conformer descriptors
are versioned and old native snapshots reject incompatible argument layouts.

Optional `--trace-directory <ignored-directory>` exports intermediate F32
stages for `embeddinggemma2-compare-audio-stages.py`. Stage tracing adds device
readbacks and file I/O and must not be used for performance measurements.

## Check the complete embedding, too

A passing audio tower comparison does not establish a passing embedding.
`embeddinggemma2-audio-embedding-reference.py` independently combines the
published audio soft rows with the verified Gemma BPE token rows, BOS, audio
delimiters, EOS, and optional task prefix, then executes the dense NumPy text
encoder. Compare the actual `EmbeddingGemma2Probe` or HTTP output with the
unchanged complete-embedding gates: cosine at least `.9999` and maximum
component error at most `.002`.

```sh
PYTHONPATH=/absolute/path/llama.cpp/gguf-py \
python3 eng/validation/embeddinggemma2-audio-embedding-reference.py \
  --model /absolute/path/embeddinggemma-2-Q8_0.gguf \
  --audio-oracle docs/validation/embeddinggemma2-reference/audio-numpy-hf-exclusive-window-oracle.json \
  --candidate docs/validation/embeddinggemma2/media-default-hf.json \
  --output docs/validation/embeddinggemma2/media-default-hf-audio-comparison.json
```

The GGML CPU investigation demonstrated why both gates are necessary. Its
canonical audio tower passed the original raw-row gate, but the first complete
default embedding failed: cosine `.9998302907445553`, maximum component error
`.0025018822964077`. The text encoder's Q8 activation approximation caused the
remaining error. The existing diagnostic `TSG_EMBEDDING_F32_LINEAR=1` expanded
learned matrices to F32 and passed the same input without changing thresholds:
cosine `.9999999427362855`, maximum error `.0000407433593321`.

The qualified EmbeddingGemma 2 GGML CPU geometry now selects that F32 learned
projection path by default. Fresh validation with no precision overrides passed
the eight independent text cases, the independent 8,192-token case, canonical
raw audio, the complete HF audio embedding, and all seven media cases through
both HTTP protocols with successful process cleanup. `TSG_EMBEDDING_F32_LINEAR=0`
retains the earlier approximate path for diagnostics. Other architectures and
GPU backends retain their separate policies. Expanded learned matrices consume
more memory; these correctness results do not establish CPU latency parity.

Keep the first failed default report and the explicit diagnostic report beside
the fixed-default evidence. Never substitute a relaxed threshold or a passing
raw-row result for the failed complete embedding. The default HF frontend uses
`log(mel + .001)`; explicitly selecting `llama_clamp` is a separate controlled
comparison with llama.cpp and does not establish default frontend parity.

For an explicitly clamped controlled pair,
`embeddinggemma2-controlled-audio-reference.py` checks both saved engine outputs
against the complete independent F32 audio/text model. Generate its audio oracle
with `embeddinggemma2-audio-reference.py --log-policy llama_clamp --past-distance 11`.
The diagnostic requires exact model, projector, waveform, input sequence, and
successful process-cleanup metadata. It uses the same `.9999`/`.002` complete
embedding gates and leaves the canonical HF tool unchanged.

The frozen A40 controlled run illustrates the distinction: direct CUDA passed
at cosine `.9999998558` and maximum error `.00006993`; scoped GGML CUDA passed
at `.9999978374` and `.00024398`. The fast llama CUDA reference failed at
`.9986395569` and `.00685544`. Both original paired reports still fail their
cross-engine gate, and the separate diagnostic exits unsuccessfully because
the reference fails. This result establishes candidate accuracy for the saved
waveform/model; it does not qualify media latency or other image/video inputs.

The fresh GGML CPU default F32 controlled run similarly passes the independent
complete clamped-audio oracle at cosine `.9999999463` and maximum error
`.00003923`. Its fast llama CPU reference fails at `.9995239695` and
`.00373045`. Preserve the failed pair and diagnostic process result: the
candidate-to-truth pass does not turn the cross-engine failure into a pass.
This controlled media pair matches seven token counts, passes three numerical
comparisons and one latency comparison, and measures audio at 259.6 ms versus
110.4 ms. The separate default-HF quality checks remain the frontend contract.

The default direct CPU controlled pair separately passes the complete audio
oracle at cosine `.999999999997` and maximum error `.0000003113`; the same fast
llama CPU reference still fails. Its full media pair meets the latency margin
in six cases, while audio takes 270.5 ms versus 108.9 ms. Matching all seven
token counts and successful server exits do not turn the three-of-seven
cross-engine quality result into an overall pass. Its after-work main-server
RSS is 7186.7 MiB versus 2939.8 MiB for llama, with the same whole-process memory
limitations recorded by the runner.
