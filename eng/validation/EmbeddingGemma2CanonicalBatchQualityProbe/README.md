# Frozen canonical32 batch quality

This separate validation tool checks the exact32 **distinct** `batch32_short`
strings saved by the standard text benchmark. It preserves the active
`EmbeddingGemma2BatchQualityProbe` and its original seven synthetic fixtures.
It has no production project references, native build or timing loop.

`Program.cs` is an explicit copy of the reviewed original runner SHA256
`1d41b7f9ca32988f1bb5dcf869871eab88d76c740115a1a7e625c7224554a769`.
Only CLI/input binding, the two case sources and scope/qualification labels
change. Native handle/semaphore ownership, token/external singleton and whole
batch/reverse/public routes, vector/MRL gates, disposal and guard finalization
remain the original code. The project links unchanged `BatchControls.cs` and
`DefaultHttpSource.cs`. Preserve an exact original/copy diff in ignored evidence
and bind all linked sources, frozen runtime/compiler/executable and generated
outputs before/after an actual run.

Build with the same frozen-only property as the original probe:

```sh
dotnet build eng/validation/EmbeddingGemma2CanonicalBatchQualityProbe/EmbeddingGemma2CanonicalBatchQualityProbe.csproj \
  -c Release -p:TensorSharpBatchQualitySnapshot=/absolute/frozen/runtime
dotnet run --project eng/validation/EmbeddingGemma2CanonicalBatchQualityProbe.Tests -c Release
```

Prefer an ignored private project/source copy with isolated build outputs.
Explicitly use the selected frozen12 baseline or candidate13 runtime, with only
the approved native library difference and identical Models/Runtime/Server.
All original required arguments remain:

```text
--model /absolute/embeddinggemma-2-Q8_0.gguf --output /absolute/fresh-report.json
--workers 7 --policy default
--expected-model-sha256 MODEL_SHA
--expected-models-sha256 MODELS_DLL_SHA
--expected-runtime-sha256 RUNTIME_DLL_SHA
--expected-server-sha256 SERVER_DLL_SHA
--expected-native-sha256 SELECTED_NATIVE_SHA
--benchmark-report /absolute/preserved-standard-text/results.json
--expected-benchmark-report-sha256 PRESERVED_REPORT_SHA
```

No case, long, diagnostic or reduced-route options are accepted. Run only after
the coordinator closes the active whole guard and the source review is clear.
Use separate serialized processes for7/8 workers and default/qualified optional
policy. Thread/optimization audit and optional provider policy are unchanged
from the original probe. Optional execution/admission still needs separate
native counter/profile evidence. No model/device unavailability can pass.

The original report must be hash-bound, have stable recorded before/after
runtime artifacts, and match the actual model and frozen Models/Runtime/Server.
Its original `strict_pass` result is recorded, including false performance
outcomes. The original native identity is informational; the selected candidate
native is separately validated and must not be confused with that old source
report's library.

The tool saves all32 original strings and observes their IDs through the actual
frozen `Tokenize` API. It requires32 distinct token sequences in original input
order: ten lengths15 followed by twenty-two lengths16, all padding to16. The
first case retains all32 distinct rows. A separate three-row case uses exact
source rows0,10,0, preserving a mixed-length duplicate control. There is no
claim of independent tokenizer validation or public text/HTTP qualification.

Each case preserves public token/external singletons, whole native token and
external batches in both orders, and public sorted token batching. Every full
output has768 finite values and unit norm within1e-5. Original batch/single,
raw/external and duplicate gates remain cosine>=.99999 and maxerror<=.0005.
MRL128/256/512 still invokes the frozen server helper and checks its independently
normalized prefix within1e-6, with the same comparison gates. Complete full/MRL
vectors and byte hashes, IDs/row hashes, call attempts and failures are saved.

The exact source-derived budget is78 learned forwards and34 token-row graphs,
3806 actual learned token rows per process. Observed calls must be public-token36,
public-external-singleton34, native-token4, native-external4 and token-row34;
public internal microbatch counts remain source-derived. All canonical native
batches contain<=512 padded rows. Quality flags qualify only these two cases;
`full_core_qualified` and `long_branch_qualified` remain false. No timing,
independent learned-model oracle, native admission or llama parity follows.

The historical eight-case saved-report checker intentionally rejects these new
case names. A separately reviewed canonical comparison mode/tool must recompute
all saved original gates and cross-baseline vector comparisons before this
additional lane qualifies a candidate. Do not rename cases or weaken the old
checker to make them fit.
