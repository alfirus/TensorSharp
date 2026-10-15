# CUDA vision attention cooperative-query screen

This standalone experiment compares the unchanged generic-load Q128/K16
baseline with four or eight threads per query. Each lane retains sixteen or
eight query/output components, while all query groups share a sixteen-key K/V
tile. Four-thread candidates use 32/64/128 queries per block; eight-thread
candidates use 16/32/64. Production dispatch and the head-72 kernel are unchanged.

The four-thread lane0 dot follows the original four x-mod4 FMA sequences and
`(dot0+dot1)+(dot2+dot3)` sum, using subgroup XOR1 then XOR2. Eight-thread
candidates change summation order. Both retain F32 operands/accumulators,
ordinary `expf`, IEEE division and the original numerical gates. Only subgroup
lane0 computes softmax. An explicit subgroup mask excludes inactive tail query
groups from shuffles; `__syncwarp(mask)` publishes lane0 probabilities before
other lanes read them. Every block thread joins both K/V barriers, including
inactive query groups. Resource metadata and compiler logs retain register,
local-memory and spill information.

```bash
nvcc -O3 -std=c++17 -arch=sm_86 -Xptxas=-v \
  --ftz=false --prec-div=true --prec-sqrt=true \
  -I /workspace/TensorSharp/ExternalProjects/ggml/include \
  eng/validation/embeddinggemma2-vision-cuda-attention-cooperative-probe.cu \
  -o /workspace/embeddinggemma2-cpu-media-20261014/vision-attention-cooperative-probe
CUDA_VISIBLE_DEVICES=0 \
  /workspace/embeddinggemma2-cpu-media-20261014/vision-attention-cooperative-probe \
  --output /workspace/embeddinggemma2-cpu-media-20261014/vision-attention-cooperative.json
```

The corrected private-stream harness retains all eight complete independent
Double operator fixtures: batches, padded/interleaved strides, partial query
groups/key tiles, 8193 keys, exact 80000.125 and source bits below F16 precision.
The original bounds remain `8e-5 + 2e-5 * abs(Double)`, with 2e-6 for source-bit
fixtures and exact equality for exact-large outputs. Guards, complete input
bytes and finiteness are checked. Both warmup and final gates contribute to the
overall result.

N2304/N2340/N1089 with twelve 64-wide heads use synthetic F32 inputs. Double
checks sample complete query/head vectors, including 16/32/64/128-query tile
boundaries; every output also compares with the generic baseline. Each
candidate runs ABBA W3/R20 for at least 0.25 seconds per phase. Private-stream
uploads/reset complete before timing; allocation, transfers and quality checks
are excluded. Run only in a coordinated quiet window without competing CPU
performance jobs, and record source/executable/runtime hashes.

A passing synthetic screen or faster kernel does not qualify the full tower or
service. Promotion needs a separate production candidate, unchanged complete
independent raw-row/model gates, cache/lifetime tests and fresh immutable media
measurements. `model_qualified` and `production_promoted` remain false.
