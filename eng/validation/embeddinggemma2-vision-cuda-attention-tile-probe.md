# CUDA vision attention tile screen

This standalone CUDA tool copies the TensorSharp-owned head-64 vision kernel
from the source hash recorded in its header. The primary variants make only its
query tile, key tile and launch bounds template parameters. It does not edit
production dispatch or upstream ggml. It compares Q128/K16 with Q128/K32,
Q128/K48, Q64/K16 and Q64/K32; all operands, products, online softmax state and
accumulators remain F32. These five variants keep normal `expf`.

Three separately labeled `bounded-exp` variants repeat Q128/K16, Q128/K32 and
Q64/K32 with a compile-time selector using `__expf` only for arguments in
`[-80,0]`; other arguments retain `expf`. The same numerical gates apply to
every variant. No global fast-math mode is enabled. IEEE division and subnormals
remain enabled; the head-72 production specialization is unchanged.

```bash
nvcc -O3 -std=c++17 -arch=sm_86 -Xptxas=-v \
  --ftz=false --prec-div=true --prec-sqrt=true \
  -I /workspace/TensorSharp/ExternalProjects/ggml/include \
  eng/validation/embeddinggemma2-vision-cuda-attention-tile-probe.cu \
  -o /workspace/embeddinggemma2-cpu-media-20261014/vision-attention-tile-probe
CUDA_VISIBLE_DEVICES=0 \
  /workspace/embeddinggemma2-cpu-media-20261014/vision-attention-tile-probe \
  --output /workspace/embeddinggemma2-cpu-media-20261014/vision-attention-tiles.json
```

The eight operator fixtures cover query/key tails, 8193 keys, padded/interleaved
head/batch strides, exact 80000.125 values and query bits below F16 precision.
Every output in these fixtures is independently checked using two-pass Double
softmax. Exact 80000.125 outputs must remain exact. Source-bit tests use the
original 2e-6 absolute gate; other outputs use `8e-5 + 2e-5 * abs(Double)`.
Unlike the native test's downstream half-scale node, this tool checks unscaled
outputs, making the absolute part at least as strict. It checks complete input
bytes including NaN padding, and output guards and finiteness.

Three timed shapes match the actual tower's N2304, N2340 and N1089 with twelve
64-wide heads. Their inputs are deterministic synthetic F32 values, rather than
captured activations. Double checks cover complete query vectors at query-tile
boundaries and head edges; every output also compares with the unchanged copied
Q128/K16 baseline. Both warmup and final gates must pass. Each candidate uses
ABBA with W3/R20 and a minimum 0.25 seconds per phase. All samples, numerical
failures, register/local/shared-memory usage and occupancy estimates are saved.

Run only in a coordinated quiet GPU window. Timings cover launch and private
stream completion; loading, allocation, transfers and oracle checks are
excluded. Record source/executable/production-kernel hashes and compile flags
alongside the report. Neither a kernel speedup nor a passing synthetic screen
qualifies the full vision tower or HTTP. A winner requires a separate production
candidate with the original complete raw-row/model gates, cache/lifetime tests,
and fresh immutable paired media measurements. `model_qualified` and
`production_promoted` remain false in this screen.
