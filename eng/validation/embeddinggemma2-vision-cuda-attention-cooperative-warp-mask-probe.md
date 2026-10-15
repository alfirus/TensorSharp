# CUDA cooperative-query warp mask follow-up

This standalone source preserves the cooperative probe's numerical equations,
variants, private-stream harness and unchanged quality gates. The only kernel
change replaces distinct query-subgroup masks with one warp-uniform
`__ballot_sync(0xffffffffu, active)` before the active-query branch. Every block
size is a multiple of 32. Active queries occupy complete four/eight-lane groups;
shuffle width still confines values to each group. Fully inactive tail warps
skip masked operations and still reach both shared K/V block barriers.

All participating lanes use the same ballot mask for shuffles and probability
publication via `__syncwarp(mask)`. The hypothesis is that distinct per-group
mask arguments may serialize subgroup rendezvous; this is a performance
hypothesis, not a diagnosed correctness failure in the prior passing screen.
The prior slower source/report remain immutable.

```bash
nvcc -O3 -std=c++17 -arch=sm_86 -Xptxas=-v \
  --ftz=false --prec-div=true --prec-sqrt=true \
  -I /workspace/TensorSharp/ExternalProjects/ggml/include \
  eng/validation/embeddinggemma2-vision-cuda-attention-cooperative-warp-mask-probe.cu \
  -o /workspace/embeddinggemma2-cpu-media-20261014/vision-attention-warp-mask-probe
CUDA_VISIBLE_DEVICES=0 \
  /workspace/embeddinggemma2-cpu-media-20261014/vision-attention-warp-mask-probe \
  --output /workspace/embeddinggemma2-cpu-media-20261014/vision-attention-warp-mask.json
```

See the [cooperative screen instructions](embeddinggemma2-vision-cuda-attention-cooperative-probe.md)
for all fixtures and limitations. This follow-up still checks all eight
complete independent Double operator fixtures and N2304/N2340/N1089 synthetic
grids with sampled Double rows and complete baseline comparisons, including
tail groups and exact source bits. ABBA W3/R20 has a minimum 0.25 seconds per
phase. No production change or model qualification is made. Run only after
coordinating a quiet slot; do not overlap or delay final service measurements.
