# Flat logical rows for CPU Q/K/V RMS normalization

This isolated screen compares unchanged ggml RMS normalization (including its
existing RMS+MUL fusion for Q/K) against one owned custom callback. It modifies
no production or upstream code and links an existing unchanged ggml CPU/base
library. The callback distributes logical head×token rows across the same
eight ggml workers, preserving the original input view and output shape.

The default token counts remain `113,311,912,8192`. Optional
`--rows 1,8,16,32,64,128` selects a separate short-context crossover screen;
the JSON records the selected counts. Numerical bounds and ABBA phases remain
unchanged. Each candidate worker must complete exactly its assigned logical
rows, including an expected zero count when fewer rows exist than workers.

In upstream revision `3612c0af93f2cfc8f048295300a61187723b12ae`,
`src/ggml-cpu/ops.cpp` assigns only dimension1 rows to workers, outside the
dimension2/3 loops. Native EmbeddingGemma2 Q/K/V inputs are
`[head_width, heads, tokens]`, so only 1–4 workers perform rows even when the
graph requests eight tasks. The existing fused RMS+MUL implementation uses
that same loop. The screen checks fusion eligibility using the unchanged
graph helper and explicitly disables reference mode. It rejects any
`GGML_CPU_DISABLE_FUSION` setting before backend initialization and records
other relevant thread/dependency selectors.

The callback preserves each row's sequential Double sum of **F32-rounded
squares**, then the float mean, `1/sqrtf(mean+epsilon)`, and F32 multiplication
order. Weighted Q/K retains `(input*scale)*weight`, including per-axis
broadcast strides; V has no weight. The independent oracle uses true Double
squares and normalization. Every output must match the original baseline
bit for bit and meet `2e-6+2e-6*abs(reference)` against the complete independent
Double output. Finite/written values, complete immutable source fingerprints,
aligned prefix/suffix guards and warm/final checks are mandatory.

The 24 cases cover head widths 256/512, Q with 4 heads, sliding K/V with 2 heads
and global K/V with 1 head, at token counts 113/311/912/8192. Source views retain the actual
packed-QKV token strides and Q/K/V offsets. Inputs and nonuniform weights are
synthetic; these are checkpoint dimensions, not learned model execution.
There is no candidate packing, permutation, contiguous-copy or row scratch.
Executed fixtures have one batch dimension, contiguous destinations and shared
one-dimensional weight vectors. Four-dimensional batches and weight variation
on other axes are not qualified by this screen.

Example on Linux using a separately preserved existing ggml library:

```sh
g++ -std=c++17 -O3 -march=native -pthread \
  -I ExternalProjects/ggml/include -I ExternalProjects/ggml/src \
  eng/validation/embeddinggemma2-cpu-flat-rms-probe.cpp \
  -L /absolute/existing-unchanged-ggml/lib \
  -Wl,-rpath,/absolute/existing-unchanged-ggml/lib \
  -lggml-cpu -lggml-base -ldl \
  -o /absolute/ignored-evidence/flat-rms-probe
OMP_NUM_THREADS=8 TS_GGML_CPU_THREADS=8 \
  /absolute/ignored-evidence/flat-rms-probe \
  --threads 8 --warmup 3 --repeats 20 --minimum-seconds 0.1 \
  --output /absolute/ignored-evidence/new-flat-rms-report.json
```

Run in a serialized CPU window.
The ABBA order is baseline-a/candidate-a/candidate-b/baseline-b, with a
persistent backend, worker pool and graph plans. Graph construction,
allocation, oracle calculation and quality checks are outside timing.
Warmup worker counters prove complete row coverage on all requested workers;
counter increments are disabled during timing, with one relaxed enable load
remaining per callback.

Preserve source/compiler/executable/library identities, flags and dependency
revision in an execution wrapper. Full bit-identity or Double gate failures
remain failed even when a candidate is faster. No model, HTTP, semantic
quality or service performance follows from this component screen.
