# EmbeddingGemma 2 validation

`EmbeddingGemma2Probe` runs direct text and ordered multimodal sequences. A case
manifest contains `{"cases":[{"name":"example","parts":[{"kind":"text","value":"task: search result | query: example"}]}]}`.
Part kinds are `text`, `image`, `audio`, and `video`; media values are local paths
relative to the manifest. Add `--mmproj <gguf>` for media cases.
For exact numerical oracle inputs, a case may instead contain a `tokens` array
and an optional human-readable `label`. Token IDs override text only and cannot
be combined with media parts. `../embeddinggemma2-probe-cases.py` exports this
format directly from the NumPy report, avoiding retokenization of diagnostic
labels for long synthetic sequences.

```sh
dotnet run --project eng/validation/EmbeddingGemma2Probe -c Release \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true -- \
  --model /absolute/path/embeddinggemma-2-BF16.gguf \
  --cases docs/validation/embeddinggemma2/bf16-cases.json \
  --output docs/validation/embeddinggemma2/direct-bf16-cpu.json \
  --backend GGML_CPU --threads 8 --warmup 1 --repeats 3
```

Reports retain model/projector hashes, binary hashes, inputs, normalized vectors,
raw-token parity for text, actual expanded token counts, and every latency
sample. Initial model loading is excluded; media decoding and preprocessing are
included. Run engines sequentially on an idle machine when comparing timings.
Success is recorded after owned models are disposed. The standalone application
then shuts down its global GGML registry; `--shutdown-ggml false` isolates model
cleanup for lifecycle regression checks. Always verify the process exit status
as well as the report's completion field.

For an isolated native experiment, copy this probe project/source into an
ignored evidence directory and build with
`-p:TensorSharpEmbeddingProbeSnapshot=/absolute/path/to/complete/frozen/runtime`.
This opt-in uses assembly references and copied runtime assets instead of
rebuilding production projects or copying the ordinary native library. Hash the
source, selected SDK, frozen inputs and copied outputs before and after the run;
the property itself is not an artifact identity check. The default build is
unchanged. Replace a native library only in a fresh private snapshot, preserving
the baseline and recording the exact replacement hash.

`../embeddinggemma2-compare.py` compares this probe with
`eng/embeddinggemma2-reference.py`'s independent NumPy oracle. Name each probe
case `oracle_<index>` and use the exact corresponding oracle text. It verifies
model and manifest hashes, token counts, dimensions, finite outputs, cosine, and
absolute error, writing a report even when a numerical gate fails.

`../embeddinggemma2-http.py` extends the existing sequential HTTP benchmark with
task prompts, the committed tokenizer oracle, raw-token parity, all four MRL
dimensions, and exact 511/512/513/1023/1024/1025/2048-token cases. Add
`--include-full-context` for 8192 tokens. Use `--help` for all shared runner
options, `--llama-build-dir` for the actual baseline CMake build, and
`--environment-note` to record material benchmark conditions. Both upstream
dependency revisions and ggml's unchanged tracked sources are checked and
recorded before and after a run.

The optional `--measure-reference-after-quality-failure` continues timing after
a reference-only consistency failure. Candidate gates retain their original
thresholds. The report still records `strict_pass: false`, failed reference
cosines, and a failing process exit, even if all timings are collected. This is
useful when the fast llama CUDA implementation loses precision on an 8,192-token
input that the independently checked TensorSharp implementation handles more
accurately. It is a diagnostic measurement, not a relaxed quality pass.

`TS_EMBEDDING_CUDA_MATERIALIZED_ATTENTION=1` selects an experimental per-encoder
CUDA F32 attention plan. Both matrix products use per-call cuBLAS compute type
69; score scratch is capped at 512 MiB. Global attention and local windows for
lengths 256 through 2,048 can use this plan. Short inputs and longer local
windows retain the streaming implementation. The inclusive local mask cache is
bounded to four entries and 64 MiB. Its uploads are acknowledged only after the
existing output readback; exceptions drain pending work before disposal. The
plan remains disabled by default until its complete performance strategy is
qualified. Record actual plan counters and all sixteen timing cases; a faster
8,192-token result alone does not establish parity for short or mixed batches.

`TS_EMBEDDING_CUDA_NORMALIZED_HALF=1` is a separate experimental projection
policy, disabled by default. Its encoder-owned plan prepares immutable norm
gains and joined QKV/gate-up weights with exponent and partial-dot bounds, then
owns the actual F32 RMSNorm-to-Half conversion boundary. Hidden state, raw PLE,
input gates, down/output projections and pooling retain their existing F32
routes. cuBLAS compute type 68 accumulates into F32 without changing the shared
handle's math mode. Static range proofs do not establish model accuracy; run
the dedicated actual-device operators, unchanged strong nine NumPy cases and
all sixteen HTTP scenarios before claiming a quality or performance benefit.
Record positive plan calls and its actual workspace/weight capacity at cleanup.

`TS_EMBEDDING_CUDA_MATERIALIZED_ATTENTION_TF32=1` is another separately named
approximation diagnostic and requires `TS_EMBEDDING_CUDA_MATERIALIZED_ATTENTION=1`.
It changes only that plan's per-call QK/AV compute policy from pedantic 69 to
FAST_TF32 77. Scores, values and results remain in F32 storage, but TF32 products
reduce operand mantissa precision. The original pedantic operators and default
remain unchanged. Record the actual instance policy and positive calls; the
separate approximate-mode operators, strong nine model gates and complete
sixteen-case comparison must pass before any model/performance qualification.

`../embeddinggemma2-stsb.py` runs both engines sequentially against the original
GLUE STS-B development gold scores. The official archive and TSV hashes are
pinned. The default first 200 pairs provide an English similarity regression at
all four MRL dimensions; the output records scores, Spearman correlations, and
the allowed difference from llama.cpp. `--expected-tensorsharp-backend` rejects
a fallback implementation, and `--environment-note` records hardware or CPU
quota constraints. MLX libraries, Metal libraries, and CUDA PTX files beside the
server are hashed outside the measured work.

`../embeddinggemma2-media-http.py` starts the real CLI with a projector and runs
the manifest through OpenAI and Ollama media content arrays. It checks vector
and token parity, base64/MRL output, malformed media requests, temporary file
cleanup, and a graceful server exit after the dependency-injection container
disposes the model and its towers. These are protocol and lifecycle checks;
the report does not claim independent multimodal quality or comparative speed.

`../embeddinggemma2-media-pair.py` separately measures sequential real HTTP
engines with identical prepared media bytes, an idle device, three warmups,
at least twenty samples and one measured second per case, and ten seconds of
engine prewarming. File/base64 preparation and model loading are excluded;
server decoding, preprocessing, inference, and response serialization are
included. The controlled image fixtures match pixel geometry, and every case
must have the same expanded token count. `--controlled-video-prefix` inserts
llama.cpp's automatic `Video:` text into the candidate sequence explicitly.
Setting `TS_EMBEDDING_AUDIO_LOG_POLICY=llama_clamp` additionally matches the
reference's mel log floor for this controlled comparison. Default HF frontend
and video policies must be checked in separate reports. Preserve failed cosine,
latency, and process-cleanup gates; matching media outputs does not imply a
speech/image retrieval leaderboard result.

On Linux, the text and media runners record main-server resident memory and
resident high-water marks from `/proc/<pid>/status` outside timed samples.
High-water marks include startup, and these observations exclude child-process
RSS and GPU device memory. Unsupported or missing fields remain unavailable,
rather than being reported as zero. Both runners require a zero graceful server
exit; a successful output followed by a disposal crash is a failed run.

The fresh eight-thread GGML CPU F32-default run passes independent text and
complete HF audio checks, but meets the 5% latency margin in zero of seven text
cases and one of seven controlled media cases. Both servers exit successfully
in each pair; strict reference consistency and paired numerical failures remain
recorded. The controlled audio comparison against independent truth passes
TensorSharp and fails the original fast llama CPU reference. That diagnostic
does not establish cross-engine quality or all-media performance parity.
After-work RSS is 1214.5/2202.0 MiB for the text TS/llama processes and
4124.3/2928.3 MiB for media. Retain the original context/slot arguments and the
memory-scope limitations when interpreting these values.

The separate default direct CPU media pair meets the latency margin in six of
seven cases. Audio remains slower (270.5/108.9 ms), and the strict cross-engine
numerical result passes only three cases despite all token counts matching.
Its complete controlled audio embedding independently passes at cosine
`.999999999997` and maximum error `.0000003113`, while the fast llama CPU
reference fails the original complete-embedding gate. Both servers exit 0;
the overall report remains failed. After-work main-server RSS is
7186.7/2939.8 MiB for TensorSharp/llama. These results do not establish CPU
text or default-HF multimodal performance parity.

`../embeddinggemma2-cuda-mixed-phase-probe.py` separately checks the exact
31/511/513/1025-token ragged input, single-versus-batch outputs, reversal, and
repetition while collecting opt-in CUDA phase events. It requires actual CUDA,
the verified tokenizer/model hash, and explicit materialized-attention/profile
flags. This is an untimed diagnostic. Summarize the original log with discard
zero when single-run geometries are present; to discard the cold mixed step,
filter that exact geometry into a separately hashed log first. Do not treat GPU
event intervals, which include host submission idle, as isolated kernel time or
as a replacement for a quiet paired benchmark.

The small corpus and synthetic media provide architecture and retrieval
regression checks. They do not establish MTEB, multimodal leaderboard scores, or
performance on devices/model files absent from the recorded runs. Generated
evidence belongs in ignored `docs/validation/` or `artifacts/`.
