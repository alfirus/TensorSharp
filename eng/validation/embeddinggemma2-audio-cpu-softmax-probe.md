# Shape-only CPU audio softmax screen

This validation-only tool compares the current audio attention graph prefix
against the same unchanged ggml operations with two shape-only views. It
modifies no production or upstream code. Inputs are synthetic F32 attention
scores at the published audio window12/head8 geometry, not learned forwards.

The current prefix is `scale(1/50) -> tanh -> scale(50) -> ADD(mask) -> softmax`.
Original softmax input is `[12,1,8,N]`. The candidate views the already
contiguous masked scores as `[12,8*N]`, calls the same `ggml_soft_max`, and
views its result back to `[12,1,8,N]`. No new arithmetic, packing or copy is
introduced. There are no ALiBi biases or sink inputs. Mask application stays
after the soft cap and broadcasts unchanged across heads.

At unchanged upstream revision
`3612c0af93f2cfc8f048295300a61187723b12ae`, the CPU softmax kernel in
`src/ggml-cpu/ops.cpp` loops over dimensions3/2 outside and distributes only
dimension1 to workers. Original `ne1=1` therefore gives all softmax rows to
worker0. The graph task count still requests `min(threads,nrows)`. The
candidate's `ne1=8*N` exposes all rows to that existing worker loop. The JSON
labels active worker bounds as **source-derived**, not measured worker counts.

Default lengths are `9,25,100,500,1500`; run separately with `--threads 7` and
`--threads 8`. Optional `--lengths` selects an explicit separate screen.
Fixtures include deterministic mixed scores, tiny scores, positive/negative
soft-cap saturation, near ties, signed zeros, causal prefix padding and
fully valid later windows. Every row has at least one permitted key. The
independent Double oracle uses complete Double soft-cap and normalized
exponentials from the original F32 inputs, rather than reproducing the F32
intermediate rounding or upstream softmax implementation.

Every warm/final output must match the complete original4D baseline **bit for
bit**, remain finite, satisfy `2e-6+2e-6*abs(reference)` against the complete
Double probabilities, lie in `[0,1]`, sum to1 within `5e-6`, and contain exact
zero at every masked entry. Input/mask fingerprints and aligned prefix/suffix
output guards must remain intact. Warm failures remain failed even if final
checks pass. Failure does not suppress diagnostic timing. Baseline initial
Double quality is also retained in the aggregate result.

Example against separately preserved unchanged libraries:

```sh
g++ -std=c++17 -O3 -march=native -pthread \
  -isystem ExternalProjects/ggml/include -isystem ExternalProjects/ggml/src \
  eng/validation/embeddinggemma2-audio-cpu-softmax-probe.cpp \
  -L /absolute/existing-unchanged-ggml/lib \
  -Wl,-rpath,/absolute/existing-unchanged-ggml/lib \
  -lggml-cpu -lggml-base -ldl \
  -o /absolute/ignored-evidence/audio-softmax-probe
OMP_NUM_THREADS=8 TS_GGML_CPU_THREADS=8 \
  /absolute/ignored-evidence/audio-softmax-probe \
  --threads 8 --warmup 3 --repeats 20 --minimum-seconds 0.1 \
  --output /absolute/ignored-evidence/new-softmax-report.json
```

Run in a serialized CPU window and preserve actual source/compiler/executable/
library hashes, compiler flags and unchanged dependency revision externally.
The ABBA order is original-a/reshaped-a/reshaped-b/original-b with a persistent
backend, thread pool and graph plans. Graph construction/allocation, oracles,
quality checks, fingerprints and guards are outside timing. The timed prefix
includes the unchanged TANH and scales; upstream schedules TANH, sigmoid,
ReLU, clamp and scales as single-task nodes. This screen does not parallelize
those operations or attribute their individual time. A component gain does
not establish whole-tower or HTTP benefit. There is no model, semantic-quality
or service claim.
