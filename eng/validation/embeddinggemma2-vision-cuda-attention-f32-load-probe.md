# CUDA vision attention F32 load screen

This standalone experiment reuses the corrected private-stream harness from
the attention tile probe. It compares the unchanged generic-load Q128/K16
arithmetic with a direct F32 load helper, a compact view carrying only row/head
counts and byte strides, and separately labeled `__launch_bounds__(128,3)`
register limits. The head-64 production constructor already requires F32
inputs, but this tool does not change production dispatch or the head-72 body.

The direct loads retain arbitrary padded/interleaved byte strides. Every
variant keeps the same F32 dot, online softmax and accumulator equations,
ordinary `expf`, IEEE division and subnormal behavior. No global fast-math or
cuBLAS policy changes are made. Kernel resource metadata reports actual
register/local-memory use and theoretical occupancy; a higher block count is
not itself evidence of a performance gain.

```bash
nvcc -O3 -std=c++17 -arch=sm_86 -Xptxas=-v \
  --ftz=false --prec-div=true --prec-sqrt=true \
  -I /workspace/TensorSharp/ExternalProjects/ggml/include \
  eng/validation/embeddinggemma2-vision-cuda-attention-f32-load-probe.cu \
  -o /workspace/embeddinggemma2-cpu-media-20261014/vision-attention-f32-load-probe
CUDA_VISIBLE_DEVICES=0 \
  /workspace/embeddinggemma2-cpu-media-20261014/vision-attention-f32-load-probe \
  --output /workspace/embeddinggemma2-cpu-media-20261014/vision-attention-f32-load.json
```

All eight existing operator fixtures retain complete independent Double
checks, including padded/interleaved batches, partial tiles, 8193 keys, exact
80000.125 values, and source bits below F16 precision. The three actual-grid
shapes use N2304, N2340 and N1089 with twelve 64-wide heads and synthetic F32
inputs. They retain sampled complete Double rows, complete baseline-output
comparisons, input byte immutability, output canaries and finite checks. The
original bounds remain `8e-5 + 2e-5 * abs(Double)`, with 2e-6 for source-bit
fixtures and exact equality for the exact-large fixture. Both warmup and final
checks must pass.

Each candidate uses ABBA W3/R20 with at least 0.25 seconds per phase. All input
and canary-reset transfers enqueue on the same private stream and synchronize
before timing. Timings include kernel submission and completion, excluding
transfers, allocation and quality checks. Run only in a coordinated quiet GPU
window and preserve source, executable, compiler and production binary hashes.

This screen cannot establish a full-tower or service gain. A clear winner still
requires a separate production candidate, complete independent raw-row/model
gates, cache/lifetime tests and fresh immutable paired media measurements.
`model_qualified` and `production_promoted` remain false.
