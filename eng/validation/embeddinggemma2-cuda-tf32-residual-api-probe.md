# Residual products with API execution and separate attribution

This standalone diagnostic is derived from the preserved
`embeddinggemma2-cuda-tf32-residual-probe.cu` source with SHA256
`4f7dc84f567777b794500c08f4107a1de5c3882d9ac43396e074dbb222cb01b2`.
It changes no production or upstream code and does not replace the earlier
source or its reports.

The earlier screen required FAST_TF32 on original operands to agree closely
with a separate pedantic product on explicitly RN-even rounded high operands:
the L2 distance had to be at most 5% of the distance from the original pedantic
F32 product. On the actual A40, API calls succeeded but the three large shapes
did not satisfy that heuristic. The observed mismatch does not establish its
cause, or the quality of products whose operands have already been rounded.
The [cuBLAS 12.8 compute contract](https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#cublascomputetype-t)
permits TF32 use; it does not promise equality with a separate pedantic
reduction. [PTX TF32 matrix arithmetic](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma)
leaves accumulation order and rounding unspecified.

This separate screen preflights the actual high×low, low×high, high×high and
low×low products using the existing private FAST_TF32 handle and stream. Three
products execute when the first three calls succeed; four products require all
four to succeed. `NOT_SUPPORTED` remains explicitly unavailable; other API or
CUDA errors propagate. The original canary values and boolean remain recorded.
A failed canary is labeled `unknown-canary-not-qualified` and does not prevent
these diagnostic computations. API success supplies no TF32 or Tensor Core
attribution; `tensor_core_instructions_proven` is always false.

All numerical and integrity checks remain unchanged: the same six shapes and
original F32 inputs, RN-even bit decomposition and 24 conversion fixtures,
independent sampled Double products, complete pedantic baseline comparison at
`2e-6 + 2e-6 * abs(reference)`, warm/final finite and written checks, private
buffer guards, source immutability and exact independent split matching.
Three products add high×low, low×high and high×high with explicit F32 addition;
four include low×low. No precision gate is waived when attribution is unknown.

```sh
nvcc -O3 -std=c++17 -arch=sm_86 --ftz=false --prec-div=true --prec-sqrt=true \
  eng/validation/embeddinggemma2-cuda-tf32-residual-api-probe.cu \
  -lcublas -o /absolute/ignored-evidence/tf32-residual-api-probe
CUDA_VISIBLE_DEVICES=0 /absolute/ignored-evidence/tf32-residual-api-probe \
  --output /absolute/ignored-evidence/new-api-screen.json
```

Use a new evidence path: an existing output is rejected. An explicit
`NVIDIA_TF32_OVERRIDE=0` or device below SM80 is unavailable. ABBA timing retains
three warmups and at least twenty repeats plus 0.1 seconds per phase. Activation
splitting, separate GEMM submissions, F32 addition and completion are timed;
weight splitting is recorded cold and excluded from warm timing. Allocation,
transfer and qualification are outside timing.

Candidate and baseline quality counts are reported separately. In particular,
the known large-cancellation pedantic baseline failure must not be attributed
to, or hidden by, residual candidate results. Exit zero means the diagnostic
completed, not that its numerical gates passed. Source, compiler, executable,
library and report identities must be preserved by the execution wrapper.
This synthetic operator screen supplies no model, HTTP, service latency or
hardware-instruction qualification. Any promising result still needs a separate
trace and complete unchanged model gates before production consideration.
