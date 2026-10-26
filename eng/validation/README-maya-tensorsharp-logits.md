# Bounded Maya / TensorSharp full-vocabulary comparison

`compare-maya-tensorsharp-logits.py` prepares a teacher-forced continuation and
compares every captured logit row. It requires NumPy and reads artifacts only.
Capture runs are separate from performance runs because logit readback and file
writes alter timings.

First run the unchanged Maya engine through `run-maya-engine-benchmark.py` with
the original matched-token cases and `--capture-logits PREFIX`, using fresh KV,
no speculation, explicit EOS IDs, and one measured repeat. The runner labels
this `quality_only` and records the exact input/output IDs, request index,
capture size, vocabulary and SHA-256. Successful warmup and reset-guard requests
also consume Maya dump indices; the comparator uses the explicit recorded
index instead of guessing from case order. Maya's file `PREFIX.INDEX.f32`
contains little-endian contiguous float32 rows. Row zero predicts the first
output token after the prompt; row `i` predicts output `i` after the prompt and
earlier teacher output tokens. A final emitted EOS has a prediction row too.

Prepare the replay from that quality report and the unchanged original cases:

```sh
python3 eng/validation/compare-maya-tensorsharp-logits.py prepare \
  --cases artifacts/deepseek-maya/original-cases.json \
  --teacher-report artifacts/deepseek-maya/maya-quality.json \
  --output artifacts/deepseek-maya/teacher-replay.json
```

This freezes the selected measured teacher repeat (default `--teacher-repeat
0`), all original cases, prompt IDs, actual generated continuation, dump
identity and first prediction position. Each replay case's `max_new` becomes
the actual teacher continuation length; `forced_ids` contains those exact IDs.
No missing case or partial row intersection is accepted. The original cases
must declare `vocab_size`, `eos_ids`, `temperature: 0` and `thinking: false`, as
exported by the current `DeepSeekEngineBench --export-cases` command.

Run TensorSharp's `DeepSeekEngineBench` with this replay file, one measured
repeat, no warmups and `--capture-logits DIRECTORY`. It records both its own
argmax and the forced token, and feeds the teacher token into the next step.
This keeps later distributions conditioned on the same history even when the
two engines disagree. The report and each row record their exact positions,
vocabulary and hashes; cache reset/prefill/final positions are checked too.

Compare the completed captures:

```sh
python3 eng/validation/compare-maya-tensorsharp-logits.py compare \
  --replay-cases artifacts/deepseek-maya/teacher-replay.json \
  --teacher-report artifacts/deepseek-maya/maya-quality.json \
  --candidate-report artifacts/deepseek-maya/tensorsharp-quality.json \
  --output artifacts/deepseek-maya/logit-comparison.json
```

If remote files were copied into local evidence directories, pass
`--teacher-root DIRECTORY` and/or `--candidate-root DIRECTORY`. These explicitly
remap recorded paths by basename. An optional `--provenance FILE.json` preserves
additional declared model-shard hashes, build identities, device/node evidence
and launch settings; it does not attest those declarations. Existing source,
command, environment and option fields are preserved from the capture reports.
Missing provenance fields remain null rather than being filled with guesses.

The report contains per-row and per-case metrics, mean/max
`KL(teacher || TensorSharp)`, exact argmax agreement, logit differences, and
both engines' perplexity on the identical teacher-generated continuation.
Perplexity includes every observed continuation prediction, including a final
EOS. It does not measure held-out corpus perplexity or factual/task accuracy.
An overflowing perplexity is null with an explicit overflow flag; its mean NLL
remains available. Softmax/KL use stable float64 computation over the complete
vocabulary, without a top-k approximation.

Incomplete reports, changed hashes, wrong prompt/forced IDs, extra/missing
rows, wrong vocabulary/positions, reused row paths and nonfinite coordinates
produce a failed comparison and exit code 1. Nonfinite coordinates are retained
as diagnostics. Valid finite comparisons return exit code 0 and descriptive
metrics even when numerical agreement is poor. No numerical acceptance
threshold or automatic model-quality/performance qualification is applied.
Generated plans/reports must remain in the checkout's ignored `artifacts/` or
`docs/validation/`, and existing evidence files are never overwritten.

```sh
python3 -m unittest discover -s eng/validation/tests \
  -p test_maya_tensorsharp_logits.py -v
```
