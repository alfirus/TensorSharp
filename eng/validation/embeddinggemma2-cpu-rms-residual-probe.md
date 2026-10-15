# CPU RMS, gain and residual addition

This standalone tool compares the unchanged ggml CPU graph
`RMS_NORM → MUL(gain) → ADD(residual)` with one owned custom3 node. It edits no
production/native build or upstream source. It does not repeat the separately
qualified flat Q/K/V normalization optimization. Common two-dimensional RMS
already partitions tokens correctly and upstream already fuses RMS with gain;
the distinct candidate includes the subsequent residual addition.

The candidate keeps upstream's sequential Double sum of **F32-rounded squares**,
the float mean, `1/sqrtf(mean+epsilon)`, ordered F32 input-times-scale, gain
multiply, then residual add. Compile with `-ffp-contract=off` and no fast-math.
It uses no approximate reciprocal, rsqrt, SIMD reduction, optional layer scale,
GELU or activation table change. The independent oracle uses true Double squares
and normalization. Full baseline bit identity is mandatory in addition to the
complete `2e-6 + 2e-6*abs(DoubleReference)` bound.

The baseline links an existing unchanged dependency at revision
`3612c0af93f2cfc8f048295300a61187723b12ae`. Before initialization the tool rejects
`GGML_CPU_DISABLE_FUSION`; it explicitly selects non-reference CPU execution and
verifies the actual graph's RMS+MUL fusion eligibility. This is not a captured
runtime fusion counter or measured barrier attribution. The candidate retains
all three explicit input dependencies. Its state and immutable epsilon outlive
the persistent graph plan, and the plan is freed before buffers/context. No
pointer cache, persistent global handle or upstream patch is introduced.

The 48 cases use width512 and N1/8/16/32/128/320/1024/8192, with six families:
ordinary shared gain, padded inputs with per-row gain, signed zeros, subnormals,
large finite values, and near cancellation. Shared and per-row gain broadcasts
are checked; input/gain/residual row strides can differ. Outputs are contiguous.
Gain is a directly bound strided leaf, avoiding a weight VIEW node between RMS
and MUL that would make the baseline fusion ineligible. Input/residual use
padded views over guarded backing tensors in both routes.
Higher axes, quantized operands, other widths and invalid epsilon/shape/stride
layouts are refused by the owned factory, with host controls for admission.
Those refusals are controls, not component arithmetic passes. No other model
geometry is qualified.

Every baseline/candidate initial, instrumented, warm and final numerical result
contributes to the aggregate. Complete immutable operand fingerprints and aligned
prefix/suffix canaries cover padded gaps too. A separate untimed candidate call
must prove every assigned row on the requested seven/eight workers; N1 correctly
has zero work on other workers. Timed calls disable counter increments but retain
the relaxed enable check and local row counter, so instrumentation overhead is
not claimed zero. Input allocation/fill, plan creation, full output checks and
oracles remain outside timing.

Start with host-only controls; these compile no native graph code and establish
no native/model/device or performance qualification:

```sh
clang++ -x c++ -std=c++17 -O2 -ffp-contract=off -Wall -Wextra -Werror \
  -DTSG_RMS_RESIDUAL_HOST_ONLY \
  eng/validation/embeddinggemma2-cpu-rms-residual-probe.cpp \
  -o /absolute/ignored/host-controls
/absolute/ignored/host-controls --host-controls
```

After independent source review, compile against the existing coherent native
library without rebuilding production or ggml. A Linux example is:

```sh
g++ -std=c++17 -O3 -ffp-contract=off -Wall -Wextra -Werror -pthread \
  -isystem ExternalProjects/ggml/include -isystem ExternalProjects/ggml/src \
  eng/validation/embeddinggemma2-cpu-rms-residual-probe.cpp \
  -L /absolute/frozen/native -lGgmlOps \
  -Wl,-rpath,/absolute/frozen/native -ldl \
  -o /absolute/ignored/rms-residual-probe
OMP_NUM_THREADS=8 TS_GGML_CPU_THREADS=8 \
  /absolute/ignored/rms-residual-probe --mode quality --threads 8 \
  --output /absolute/ignored/fresh-quality.json
```

Mac uses the selected unchanged dylib and its matching dependency headers;
retain the same `-ffp-contract=off` and strict numerical gates. Use fresh output
paths: the program reserves them with `O_EXCL`. Setup failures and produced
negative reports must remain preserved. The host-only binary's normal quality
mode returns a completed=false/passed=false unavailable report, never a native
pass.

Only after review and numerical qualification may the coordinator run
`--mode benchmark` in a quiet window. It performs a complete untimed 48-case
preflight before any timing, then baseline-a/candidate-a/candidate-b/baseline-b
with W3/R20/minimum0.1seconds and raw samples. Default CLI admits only7/8 workers
and that bounded timing policy. This is resident component timing, not learned
model, HTTP, semantic quality, service latency or llama parity.

An execution wrapper must hash-bind source/README/compiler/executable, actual
compile flags, dependency headers/revision/clean status, coherent existing native
and its runtime library closure before/after; preserve aliases and derived build
outputs without rebasing. No timing is authorized solely by compiling this file.
An eventual production candidate would still need strict8/8192, batching and
duplicates, complete HF/media vectors, protocol/MRL/lifetime checks and matched
service measurements. Logical intermediate and graph-boundary savings supply
no wall-time prediction.
