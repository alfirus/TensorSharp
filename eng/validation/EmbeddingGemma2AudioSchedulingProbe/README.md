# Frozen audio scheduling screen

This standalone Linux probe compares one persistent master thread for production
audio preparation and native text encoding with two persistent master threads for
those stages. It does not alter production code, the frozen runtime, environment
variables, or OpenMP policy. It uses one exclusively owned audio-only model,
reflects the frozen private `Prepare` method/native handle, and calls the existing
public `GgmlEmbeddingNative.EncodeInputs` pinning wrapper while retaining the model
and its gates. Preparation may acquire the text gate internally; the probe only
acquires that gate separately around native encoding.

Build into a new ignored directory against an immutable runtime snapshot:

```sh
dotnet build eng/validation/EmbeddingGemma2AudioSchedulingProbe -c Release \
  -p:TensorSharpSchedulingProbeSnapshot=/absolute/frozen-host \
  -o /absolute/docs/validation/scheduling-probe
```

Run sequentially in a quiet CPU window. Supply independently verified frozen
assembly/native hashes and the original complete pooled/audio oracle files. Use
`--policy hf` with the canonical HF oracle or `--policy llama_clamp` with the
matching clamp oracle; these are separate invocations and reports.

```sh
TS_GGML_CPU_THREADS=8 OPENBLAS_NUM_THREADS=1 \
dotnet /absolute/docs/validation/scheduling-probe/EmbeddingGemma2AudioSchedulingProbe.dll \
  --model /workspace/models/embeddinggemma-2-Q8_0.gguf \
  --projector /workspace/models/mmproj-F32.gguf \
  --wav /absolute/tone.wav \
  --oracle /absolute/pooled-audio-independent.json \
  --oracle-sha256 POOLED_ORACLE_SHA256 \
  --audio-oracle /absolute/full-audio-independent.json \
  --policy hf --output /absolute/new-scheduling-report.json \
  --expected-models-sha256 MODELS_SHA256 \
  --expected-runtime-sha256 RUNTIME_SHA256 \
  --expected-native-sha256 NATIVE_SHA256
```

The ordering is same-a/split-a/split-b/same-b, with three warmups and at least
twenty timed calls plus one cumulative second per phase. Total timing starts
before dispatch and stops on the caller after completion; component timing occurs
inside workers. All timing samples are retained. Every warmup and sample checks
the complete normalized 768-vector against the independent oracle at cosine
0.9999/max absolute error 0.002, and additionally requires bit-identical prepared
rows and pooled outputs to the initial same-thread run. Model, waveform, oracle,
copied frozen dependency, loaded native library, and before/after file identities
are bound in the report. Worker callback failures propagate; queued work is
drained and joined before completion.

This one-waveform resident scheduling experiment does not measure HTTP or actual
production `Task.Run` scheduling. In particular, it cannot establish that a
dedicated owner is faster than one whole multimodal ThreadPool callback. Direct
native calls omit the production asynchronous input clone equally in both routes.
No service speedup or production change follows solely from this screen.

Pure worker lifecycle controls require no model or frozen assemblies:

```sh
dotnet run --project eng/validation/EmbeddingGemma2AudioSchedulingProbe.Tests -c Release
```

## Whole ThreadPool callback versus resident caller

Optional `--mode threadpool-resident` selects a separate experiment. Omitting
`--mode`, or passing `--mode same-split`, retains the original same/split
ordering and report. Preserve previous executed source/reports before building
the extension into a fresh ignored output directory against frozen9 or frozen11.

The new ABBA order is whole-taskrun-a/resident-a/resident-b/whole-taskrun-b.
Both routes invoke one identical synchronous callback: production preparation,
an explicit input clone, finite validation and the existing native encoding
wrapper. One route dispatches through `Task.Run(..., CancellationToken.None)`;
the other uses the existing persistent owner. Both wait for complete callbacks
before returning or propagating an error. Caller-side total timing includes
dispatch/completion; preparation, cloning and text times are separately recorded.
The same MM and text gate boundaries, complete oracle limits, bit identity,
loaded native binding and before/after closure guards remain mandatory.

Every warmup and timed call records its natural managed/native caller IDs and
`IsThreadPoolThread` status. Resident calls must retain the owner identity;
ThreadPool calls must actually run on ThreadPool threads, but need not change
identity. No worker churn is forced, no ThreadPool size is set, and no OpenMP
policy is changed. Multiple caller IDs alone do not prove OpenMP team creation
or physical worker reuse. A naturally stable ThreadPool caller is a valid result.

Initial runs should remain audio-only. A separate invocation may additionally
pass `--prewarm-image /absolute/image.png`. This explicitly loads vision and
executes one identical restored public `EmbedAsync` image request before each
of the four phases, outside audio timing and without holding the MM gate.
The common public prewarm is used for both routes, rather than a route-specific
owner prewarm. Its path/hash, output hash, token count and elapsed time are
recorded. Image outputs must be finite and normalized; this is **not** an
independent image quality qualification. The complete audio oracle still applies
to every subsequent audio vector. Prewarm and audio-only results must remain
separate reports.

The production scheduling implementation is unchanged, including the restored
two-`Task.Run` path. The probe directly controls the complete synchronous callback
and omits public asynchronous gate/continuation overhead. No production owner,
HTTP improvement or service parity follows from this experiment alone.

## Restored public API versus resident callback

`--mode public-resident` is a separate audio-only comparison on the same frozen
model and independent oracle. Its ABBA order is
public-a/resident-a/resident-b/public-b. The public route calls the actual restored
`model.EmbedAsync(parts, CancellationToken.None).GetAwaiter().GetResult()` on the
probe caller. The resident route uses the existing synchronous production
`Prepare` → input clone → native encoding callback. These are different routes,
including real public validation, cloning, gate and continuation work. Caller
wall timing starts before route gates/API entry and ends after completion; no
concurrent model requests or forced ThreadPool churn are introduced.

Every public warmup and timed call must report exactly 29 prompt tokens and one
finite normalized 768-vector, pass the unchanged independent .9999/.002 pooled
gates and retain complete output bit identity to the initial resident baseline.
Public prepared input bytes, native caller IDs, and preparation/clone/text timing
breakdowns are unobserved and serialize as `null`, with explicit unavailable
observation labels. The report refuses accidentally populated public internal
fields. Resident prepared rows and callback identities/breakdowns remain fully
observed and gated. No public native caller identity or input bit identity is
inferred from its output.

Use the same command above with `--mode public-resident`, a fresh output path,
and exact frozen11 hashes. Start audio-only. An optional separate
`--prewarm-image` invocation uses the identical restored public image request
before every phase, outside audio timing. Existing full dependency/oracle guards,
owner-thread model disposal and worker join remain mandatory. This component
comparison does not qualify HTTP performance or justify a production scheduler.
The omitted/default `same-split` and explicit `threadpool-resident` modes retain
their prior phase labels and valid-route report fields.
