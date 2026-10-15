# CPU audio FFN activation screen

This standalone validation tool compares the original ggml F32 graph
`clamp(up-output bounds) -> SiLU -> clamp(down-input bounds)` with one owned
custom node. It changes no production or upstream source. It performs no
learned matrix products, full audio tower, preprocessing, text pooling or HTTP.

The probe reads all 24 FFN chains' four scalar clamp bounds directly from the
original projector GGUF. It requires each scalar's original F32 storage and
the actual `1024 -> 4096 -> 1024` matrix metadata. The report retains all tensor
names, values and F32 bit patterns. Bounds are never invented. Preserve the
projector SHA256 before/after execution alongside the actual command manifest.

Both paths call the **same linked** `ggml_vec_silu_f32` from unchanged ggml
revision `3612c0af93f2cfc8f048295300a61187723b12ae`. The probe checks its dynamic
symbol/address and resolved library against `--helper-library`. It does not
copy the SiLU implementation or substitute scalar/vector exponential math.
Bind the library and actual ggml sources/headers/toolchain with before/after
SHA256 guards; the revision CLI alone does not establish library provenance.

The candidate partitions token rows using the existing GGML workers. Each row
is processed in tiles of 64 F32 values: exact upstream clamp branch choices,
the unchanged linked vector SiLU in place on a small stack tile, then the
second exact clamp into output. Fixed SIMD widths 4/8/16 divide both 64 and
4096, preserving the original vector/tail dispatch. SVE and RVV helpers are
explicitly refused because their runtime vector lengths need not divide 64.
The original graph retains its ordinary one-task clamps and parallel SiLU.

Admission controls require enabled use, both learned clamp descriptors,
contiguous 2D F32 width4096, positive row count and finite ordered bounds.
Missing/disabled/strided/other-width/other-type/batched or fully no-op clamp
cases return null **before** allocating a custom node. These are constraints
for an eventual integration, not a promoted production route. Unchanged
production fallback remains outside this tool. Direct control tests cover
both clamp thresholds with nextafter values, NaNs, infinities and signed zero.
The expression `MAX(MIN(x,max),min)` has different NaN behavior from fmin/fmax
and must not be replaced by them.

Every case executes all 24 independent chains in one persistent graph plan.
Default lengths are 25/48/96/500. Run separately with 7 and 8 workers, ABBA
original-a/candidate-a/candidate-b/original-b, three warmups and at least 20
timed calls plus one cumulative timed second per phase. The baseline and
candidate share guarded read-only inputs but have separate guarded outputs.
No graph construction or allocation occurs during timing.

Fixtures include deterministic synthetic projected values, all learned
thresholds and their adjacent F32 values, positive inverse-SiLU values near
the reachable downstream upper threshold, signed zeros, subnormals, large
finite values, infinities and NaN payloads. The first learned clamp makes the
chain finite. Negative downstream bounds below SiLU's minimum are unreachable
inside the chain; direct clamp controls cover those branches separately.

Initial baseline, an explicitly instrumented candidate call, every warm output
and the final timed output must match the complete baseline **bit for bit**,
remain finite and satisfy `2e-6+2e-6*abs(reference)` against independent Double
clamp/SiLU/clamp arithmetic. Guards and complete input fingerprints must stay
intact. All failures remain failed. Intermediate timed outputs are not checked
separately. Actual worker row counts are collected only by the instrumented
call outside timing; timed callbacks have instrumentation disabled.

Example using an existing immutable native library containing unchanged ggml:

```sh
g++ -std=c++17 -O3 -Wall -Wextra -Werror -pthread \
  -isystem ExternalProjects/ggml/include -isystem ExternalProjects/ggml/src \
  eng/validation/embeddinggemma2-audio-cpu-ffn-activation-probe.cpp \
  -L /absolute/frozen-native -Wl,-rpath,/absolute/frozen-native \
  -lGgmlOps -ldl -o /absolute/ignored-evidence/ffn-activation-probe
OMP_NUM_THREADS=8 TS_GGML_CPU_THREADS=8 OPENBLAS_NUM_THREADS=1 \
  /absolute/ignored-evidence/ffn-activation-probe \
  --projector /workspace/models/embeddinggemma-2-mmproj-Q8_0.gguf \
  --helper-library /absolute/frozen-native/libGgmlOps.so \
  --ggml-revision 3612c0af93f2cfc8f048295300a61187723b12ae \
  --threads 8 --warmup 3 --repeats 20 --minimum-seconds 1 \
  --output /absolute/ignored-evidence/new-threads8-report.json
```

On macOS use `clang++` and the frozen `.dylib`; `-ldl` is unnecessary. A bounded
local correctness run may explicitly use `--lengths 25 --warmup 1 --repeats 1
--minimum-seconds 0`; it is not qualified timing. Preserve the exact command,
compiler version/hash, executable/source/library/projector hashes, dependency
revision/clean-tree state and selected nonsecret thread/optimization environment
before/after. Reports use exclusive creation and never overwrite prior evidence.
All generated outputs belong in ignored `docs/validation/` or `artifacts/`.

Component timing excludes projections, model loading and actual learned
activation distributions. Node count and local storage reductions do not
measure DRAM traffic. A win here requires subsequent original-bound full raw
audio, pooled HF/clamp/resampling and matched tower/service validation before
any production change or end-to-end performance claim.
