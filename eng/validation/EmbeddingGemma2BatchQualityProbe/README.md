# Frozen CPU batch quality probe

This validation-only executable loads an immutable TensorSharp runtime. It has no
production project references and never builds native code, ggml, or the models
project. Local ARM compilation and pure controls do not qualify the Linux x64
candidate. Generated reports belong in ignored `docs/validation/` or `artifacts/`.

Build against the exact complete frozen runtime being tested:

```sh
dotnet build eng/validation/EmbeddingGemma2BatchQualityProbe/EmbeddingGemma2BatchQualityProbe.csproj \
  -c Release -p:TensorSharpBatchQualitySnapshot=/absolute/frozen/runtime
dotnet run --project eng/validation/EmbeddingGemma2BatchQualityProbe.Tests -c Release
```

Prefer copying these sources and the linked `DefaultHttpSource.cs` into an ignored
build directory before compilation. Keep a before/after hash manifest for the
source copy, compiler, executable, frozen runtime, native dependency and ggml
revision. The executable also binds the actual model, copied dependencies, frozen
dependencies and loaded native library; caller-supplied expected hashes must come
from the independently preserved runtime manifest. A report alone does not prove
that the runtime was an approved baseline or candidate.

Every model invocation requires these arguments (all hashes are lowercase SHA256):

```text
--model /absolute/model.gguf --output /absolute/fresh-report.json
--workers 7 --policy default
--expected-model-sha256 MODEL_SHA
--expected-models-sha256 MODELS_DLL_SHA
--expected-runtime-sha256 RUNTIME_DLL_SHA
--expected-server-sha256 SERVER_DLL_SHA
--expected-native-sha256 LIBGGMLOPS_SHA
```

Run separate serialized invocations for workers 7 and 8 and each of `default` and
`optional`. Default requires all recorded optimization and instrumentation
selectors absent. Optional requires `TSG_EMBEDDING_LINUX_BLAS=1`,
`TSG_EMBEDDING_CPU_ATTENTION=1`, an explicit
`TSG_EMBEDDING_BLAS_LIBRARY=/absolute/qualified/provider.so`, and
`OPENBLAS_NUM_THREADS=1`; other optimization selectors must be absent. The provider
is hash-bound. Successful optional execution must additionally be established
from the existing native provider/disposal logs or a separate diagnostic run;
the quality tool does not manufacture operator counters. `TS_GGML_CPU_THREADS`
must be absent or agree with the explicit worker argument. Other OpenMP and
thread controls are recorded, not modified by the tool.

The seven default cases contain same padded buckets 16, 32, 512, and 528 (the
last crosses the published sliding-window radius), equal-length and multi-bucket
controls, and the observed batch32 geometry with ten lengths15 and twenty-two
lengths16. Token content is generated deterministically from a production-tokenized
seed, retaining its BOS/EOS and vocabulary IDs. Every case includes duplicates.
Full token IDs and byte hashes are saved. These are synthetic raw-token fixtures,
not a natural-language semantic benchmark.

For each unique sequence, the tool obtains a public raw-token singleton, actual
scaled token rows, and a public external-input singleton. It then compares public
sorted token batching and whole native token/external batches, including reversed
order, against the public singleton results. Whole native calls use the existing
public pinned `GgmlEmbeddingNative.Encode`/`EncodeInputs` wrappers under the actual
model's semaphore; there is no new native ABI or production execution policy.
External inputs here are complete production scaled token rows, not new media
tower inputs. Existing complete media oracles remain separate.

All output vectors must contain 768 finite values and have unit norm within
`1e-5`. Every batch/singleton, duplicate, and raw-token/external comparison keeps
cosine `>=0.99999` and maximum absolute error `<=0.0005`. MRL 128/256/512 uses the
actual frozen server helper, checks an independently normalized prefix within
`1e-6`, and keeps the same comparison bounds. Complete output and MRL vectors,
raw float byte hashes, actual IDs/lengths, policy and worker settings are retained
for an independent baseline/candidate report comparison. This tool compares
execution routes; it is not an independent learned-model arithmetic oracle.

`--cases bucket16,bucket512` selects distinct documented case names in that order.
`all` is the default. A subset qualifies only that subset. Case names are:

```text
bucket16, bucket32, bucket512, bucket528-window-cutoff,
equal32-control, multiple-buckets-control, observed-batch32-short-geometry,
bucket8192-boundary
```

The full core is normally 75 learned forwards plus 20 token-row graphs. The exact
budget is calculated from the actual distinct token content before the first
forward, so a tokenizer seed with repeated content can change those totals.
Observed API/wrapper call attempts are reported separately from source-derived
public internal microbatch counts. There are no quality-tool timing metrics.
Partial case/route evidence and failed numerical gates are retained; errors and
unavailable model/device routes cannot pass qualification. Output paths are
reserved with `CreateNew` so existing reports are preserved.

For one inexpensive admission diagnostic, use:

```text
--cases observed-batch32-short-geometry --diagnostic true --route-set native-token-only
```

Enable the existing native profile selectors in that separate process. This runs
exactly one whole token batch, saving finite/unit-norm/duplicate/MRL checks while
omitting singleton comparisons and token-row graphs. It explicitly sets
`quality_qualified=false`. Its externally captured native profile can prove the
actual graph layout; source-derived attention/concat counts are not observed
kernel counters. Compare the baseline profile's packed length512/batch1/32
segments with the candidate's length16/batch32/no segments. No performance claim
may be drawn from an instrumented quality diagnostic.

Long coverage is an explicit resource-bounded opt-in:

```text
--include-long true --long-token-fixture /absolute/original8192-report.json
--expected-long-fixture-sha256 ORIGINAL_REPORT_SHA
```

The fixture may contain a top-level `tokens` array or exactly one `cases[]` entry
with 8192 `input_tokens`, as in the preserved independent text-golden report. The
8192 IDs are unchanged; the 8191 row drops the original penultimate ID and retains
the original EOS. Public batching splits these two sequences at its 4096-token
budget. Therefore the long case deliberately uses two public singletons, a whole
native token batch, its reversal, and a whole native external-input batch. Public
long batching, external singletons and external reversal are explicitly omitted.
This is five learned forwards (65,532 actual learned token rows) plus two token-row
graphs, with padded native batch volume16,384 <=65,536. The two logical F16
attention mask planes alone occupy about512 MiB for either old segmented or new
independent layout, before model, PLE and activation memory. This is not a runtime
or peak-memory forecast. Execute this branch at both7/8 workers and default/optional
policies to qualify the broad candidate, or label the long branch unqualified.

Do not count a diagnostic, a subset, a skipped Linux model route, or pure tests as
full candidate quality coverage. The probe does not run HTTP checks, STS-B, or
performance comparisons and does not establish parity with llama.cpp.
