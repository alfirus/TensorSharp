# Resident F32 projection component experiment

This standalone tool tests a TensorSharp-owned, validation-only layout and
AVX512 kernel. It does not edit ggml, change production dispatch, quantize
activations, or qualify a model. The reference is the already optimized
unchanged GGML F32 projection from frozen Linux native638 and clean ggml
3612c0af93f2cfc8f048295300a61187723b12ae. A separate optional baseline uses
the existing qualified one-worker LP64 OpenBLAS provider and its current row
partition for N≥128. Provider refusal propagates; it never silently substitutes
a different projection.

Original checkpoint Q8_0 bytes are read once per matrix and SHA256-bound.
Unchanged ggml_get_type_traits(Q8_0)->to_float produces original F32 weights.
A second F32 allocation copies every weight bit into
packed[(m/16)*K*16+k*16+m%16]. The packed graph leaf truthfully has dimensions
[16,K,M/16]; it is an explicit dependency alongside the original contiguous
F32 input. No raw-pointer cache or model ownership is introduced.

Each tile owns 16 output channels and up to eight token rows. Eight named ZMM
accumulators begin at +0. For ascending K, one packed weight vector feeds each
row's original scalar input through an explicit F32 FMA. There is no reduction
over vector lanes, half/integer math, or compensated accumulation. F32 operands
and accumulation do **not** promise exact-real arithmetic or bit identity with
the upstream reduction order. Sequential FMA can fail cancellation-sensitive
quality gates.

Workers own disjoint consecutive 16-channel blocks and then traverse eight-row
tiles, retaining each weight panel across rows. The initial three matrices need
10MiB original F32 plus 10MiB packed storage. The three N values share those
matrices; all original inputs are retained separately. Packing is cold, outside
resident timing, and its observed duration/bytes are reported. Plans are freed
before external buffer wrappers and metadata. Matrix/input/parameter owners and
the persistent threadpool outlive their plans.

## Initial quality scope

Actual components are down [K2048,M512], SWA attention-output [K1024,M512],
and global attention-output [K2048,M512], each at N128/800/1024. Run separately
with seven and eight existing GGML workers: 18 actual cases. Inputs reproduce
the prior projection tool's seed 101+N, LCG, and alternating cancellation-sensitive
first row. They are synthetic inputs with real checkpoint weights, not captured
learned activations.

Each process also executes 28 complete small-matrix controls: K1/17/31/33 and
N1/7/8/9/15/16/17, including signed zero, subnormal weights, and cancellation.
Full baseline and candidate outputs must be finite and written; every
prefix/suffix canary, original Q8/F32/input fingerprint, and full weight-pack bit
comparison must remain intact. Complete small outputs and original actual
row/column/worker-boundary samples plus tile boundaries use the unchanged bound
max(2e-5,1e-6*sum(abs(original_F32_W*original_F32_X))) against independent
Double products. Candidate samples also match a separate scalar sequential
std::fma bit oracle; baseline entries for that extra oracle are null.
Complete baseline/candidate max differences are reported without an invented
cross-implementation bit-identity requirement.

An additional untimed instrumented candidate graph proves exact worker tiles,
output element coverage, callback execution, and round-to-nearest mode.
Actual M512 supplies work to all seven/eight workers; small M32 controls
honestly leave most workers empty. Coverage increments are disabled in timed
graphs; their optional-flag checks remain part of the candidate.

All 37 initial cases per process must pass before any timing. Every warmup
is checked immediately outside timing, ANDing pass/finite/bit flags and retaining
maxima and total counts. Timed runs refill output outside the individual clock,
then check the final output. Four phases are baseline-a, candidate-a,
candidate-b, baseline-b for each of nine actual shapes; raw caller wall samples
are retained. This is a component comparison, not HTTP or model latency.

## Build and execution boundaries

Host-only controls are portable on supported POSIX validation hosts and
never enter the target kernel:

    clang++ -std=c++17 -O3 -Wall -Wextra -Werror \
      -fno-fast-math -ffp-contract=off \
      eng/validation/embeddinggemma2-f32-prepacked-projection-probe.cpp \
      -o /absolute/ignored/host-probe
    /absolute/ignored/host-probe --host-controls

The target function alone has target("avx512f,fma"); compiler runtime feature
queries include OS extended-state support. ARM and other unqualified platforms
return an explicit completed=false,passed=false report with zero target/native
graph calls, empty cases/phases, and no model read. A host-control pass on ARM is
not target arithmetic validation. Compile without fast-math or contraction;
the target uses explicit FMA intrinsics. Fast-math builds are rejected.

Actual Linux compilation must use -DTSG_PREPACKED_WITH_GGML=1, the **unchanged**
owned embedding_linux_blas.h and exact frozen638 GGML headers, and link captured
frozen library638b6206de314b99386d7eb66fed968cf1c0877af6483ef612689dfdf35d71d4.
Do not use the current ordinary Mac binary as that baseline. Bind source/header,
compiler driver and dependency closure, model, original native/runtime/provider,
linked libraries and aliases, executable, generated dependency/assembly files
before/after without rebasing. Verify clean upstream revision. Use a fresh
ignored evidence directory. The tool's expected native digest is a declared
contract, not an observation of a loaded library; the outer wrapper must prove it.

Quality-only default:

    /absolute/ignored/probe --model /workspace/models/embeddinggemma-2-Q8_0.gguf \
      --threads 8 --baseline ggml-f32 --output /absolute/ignored/quality-cpu8.json

For the separately labeled optional provider use --baseline blas-policy and
--library /absolute/libopenblas.so.0, OPENBLAS_NUM_THREADS=1, the same workers,
and all unrelated selectors absent. Actual library/configuration and worker1
qualification are reported. Preserve baseline and candidate numerical failures;
exit1 is a completed negative result when every case ran.

Only after independent source review, actual Linux quality closure and compiled
instruction/register-spill review may the coordinator authorize --benchmark
(default W3/R20/min.1). Confirm F32 ZMM FMA, no integer/half approximation and
whether accumulators spill. Timing permission is an external workflow boundary,
not established by the flag itself. No VM stage or execution is authorized here.

Original Q16 product failures, residual-Q8 slowness and rejected alignment/TF32
experiments remain separate negative evidence. A hot-matrix win would justify a
later rotated 24-layer working-set experiment before isolated full-model quality
and service tests. No throughput forecast, promotion, or parity claim follows
from this source or these components.
