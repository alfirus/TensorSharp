This standalone CUDA diagnostic compares pedantic F32 cuBLAS with explicit
BF16 residual decomposition using three, six, or nine products. It changes no
production dispatch or upstream source. The approach is motivated by
[Henry et al.](https://arxiv.org/abs/1904.06376); API policies follow the
[NVIDIA cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/index.html).
BF16 operands are explicitly formed; accumulators/output stay F32 and reduced
precision reductions are disabled on a private handle.
The `candidate_terms=-1` control also compares batch-one
`cublasGemmStridedBatchedEx` with ordinary `cublasGemmEx`, preserving pedantic
F32 arithmetic. This tests the actual private backend's submission form.

```sh
nvcc -O3 -std=c++17 -arch=sm_86 \
  eng/validation/embeddinggemma2-cuda-compensated-projection-probe.cu \
  -lcublas -o artifacts/embeddinggemma2-cuda-compensated-projection-probe
artifacts/embeddinggemma2-cuda-compensated-projection-probe > artifacts/compensated-projection.json
```

The first four shapes match actual vision projection dimensions; inputs are
deterministic synthetic values, not captured activations or checkpoint weights.
Two adversarial shapes exercise unrounded F32 bits, values outside the F16
range, and cancellation. Each candidate uses a baseline/candidate/candidate/
baseline order with three warmups and at least twenty samples and 0.25 seconds
per phase. Timings include input splitting and synchronized completion, but
exclude resident weight decomposition, allocations, and handle setup.

Every output is checked against the full pedantic product; up to 4096 positions
also compare with independently accumulated Double products. The diagnostic
gate is `2e-6 + 2e-6 * abs(Double result)`, and reports output canaries,
finiteness, and bitwise input immutability. A failed candidate remains failed.
This gate is not an EmbeddingGemma model quality gate. No approximation may be
promoted from this shape screen alone: complete tower/model numerical gates,
range/underflow/failure handling, ownership, and actual HTTP comparisons are
still required. Device zero must support native BF16; missing hardware fails.
`completed` means the experiment ran; top-level `model_qualified=false`
and `production_promoted=false` explicitly avoid a qualification claim.
