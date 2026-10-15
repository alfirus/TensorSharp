# Saved CPU batch quality audit

`embeddinggemma2-cpu-batch-quality-check.py` independently recomputes complete
F32 saved vectors from `EmbeddingGemma2BatchQualityProbe`. It runs no model and
does not use the producer's numerical summaries as its result. Both reports
must cover the selected complete `all`, `core` or `long` scope with the same
worker count and compute policy. Core covers seven cases; long covers the
original hash-bound 8192-token sequence and its specified 8191-token variant.

Supply an identity JSON with exactly six lowercase SHA-256 strings:
`model`, `models`, `runtime`, `server`, `baseline_native`, `candidate_native`.
Use the independently preserved build/runtime manifest for these values. Pass
each input file's expected SHA-256 separately:

```sh
python3 eng/validation/embeddinggemma2-cpu-batch-quality-check.py \
  --baseline /absolute/baseline.json --baseline-sha256 BASELINE_REPORT_SHA \
  --candidate /absolute/candidate.json --candidate-sha256 CANDIDATE_REPORT_SHA \
  --expected-identities /absolute/identities.json \
  --expected-identities-sha256 IDENTITY_MANIFEST_SHA \
  --scope all --output docs/validation/cpu-batch-independent.json
```

The checker verifies complete token IDs, token-row hashes, F32 vector hashes,
unit norms, singleton/raw-input agreement, duplicates, reversed/API/native
routes, actual saved MRL vectors and independently normalized prefixes. It
compares every corresponding baseline/candidate vector at the unchanged
`.99999` minimum cosine and `.0005` maximum absolute error. Missing cases,
routes, selectors, hashes, call counts or long input bindings fail closed.
Duplicate JSON keys, nonfinite values and diagnostic-only reports are refused.
Outputs use exclusive creation; original failed reports remain untouched.

For the separate canonical32 probe, use `--scope canonical` and supply
`--benchmark /absolute/original/results.json --benchmark-sha256 ORIGINAL_SHA`.
This scope requires the actual 32 distinct benchmark strings/token sequences
and the separate original-row `0/10/0` duplicate control. It checks the original
runtime identities, retained failed outcome and input guard independently;
it cannot qualify or replace the seven synthetic cases or the long branch.
The reported learned-forward/token-row budgets are recomputed for every scope.

Recorded report guards are checked for equality, and local report/checker
inputs are rehashed. VM files are not rehashed by this offline tool. Actual
operator/provider admission, compiler/runtime closure, independent learned
model truth, public HTTP behavior and latency require separate evidence.
Synthetic unit controls are validator tests, never model validation.
