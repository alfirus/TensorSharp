# Isolated TF32 residual projection screen

This validation-only executable compares the existing classic pedantic F32
strided-batch-one projection with three and four explicit TF32 products. It
changes no production dispatch, upstream source, process environment or shared
cuBLAS handle. Private handles share one owned nonblocking stream. Every CUDA
submission completes before host reads, timing completion or buffer destruction.

For both original F32 weights and activations, `high` is rounded to ten explicit
mantissa bits with round-to-nearest, ties-to-even. `low` is the same rounding of
the explicitly F32 residual `source - high`. The three-product route computes
`highWeight*lowInput + lowWeight*highInput + highWeight*highInput`; the fourth
adds `lowWeight*lowInput` before the final high/high addition. Products use
`CUBLAS_COMPUTE_32F_FAST_TF32`, while separate owned kernels perform explicit
F32 subtraction and addition. No exact-F32 arithmetic guarantee is inferred.

Integer bit rounding avoids an unsupported instruction on the A40: NVIDIA's
[PTX target notes](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt)
place `cvt.rn.tf32.f32` on SM90 or higher. The independent host reference chooses
adjacent TF32 values by Double distance rather than reusing the device's bit
formula. Twenty-four device fixtures cover even/odd ties, signs, signed zero,
subnormals, exponent boundaries, finite overflow, infinities and NaN payloads.
Nonfinite values and finite rounding overflow preserve classification and get a
zero residual without evaluating `inf - inf`; matrix decompositions containing
them are unavailable. Subnormal conversion checks do not qualify hardware GEMM
subnormal behavior.

[cuBLAS 12.8 compute documentation](https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#cublascomputetype-t)
allows TF32 arithmetic for the selected FAST compute mode. Its
[override documentation](https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#cublascomputetype-t)
states that `NVIDIA_TF32_OVERRIDE=0` disables TF32 acceleration. The executable
records the existing value and rejects explicit zero before device work; it
does not change the variable. Devices below SM80 are explicitly unavailable.

An untimed one-product check compares FAST computation on original unrounded
inputs with pedantic F32 and with pedantic computation on explicit TF32-high
inputs. A shape must show an observable arithmetic difference from original
F32 and be substantially closer to the TF32-high result. Otherwise its TF32
candidate is unavailable and only the baseline is timed. This is evidence of
effective TF32 arithmetic, not proof of actual Tensor Core instructions. A
separate kernel trace is required before attributing a win to that hardware.

The six synthetic cases and precision gates match the standalone Lt screen:
four actual projection geometries plus unrounded and large-cancellation cases.
Every output compares with the complete pedantic baseline, and at most 4096
positions also use independent Double products. Both retain the unchanged
`2e-6 + 2e-6 * abs(reference)` bounds. Warmup and final checks include every
source/split/product/output buffer's surrounding canaries, immutable original
inputs and weights, exact split agreement, complete finite/written outputs and
the two numerical gates. A failed quality phase remains timed and recorded.

```bash
nvcc -O3 -std=c++17 -arch=sm_86 \
  --ftz=false --prec-div=true --prec-sqrt=true \
  eng/validation/embeddinggemma2-cuda-tf32-residual-probe.cu \
  -lcublas -o /workspace/embeddinggemma2-cpu-media-20261014/tf32-residual-probe
CUDA_VISIBLE_DEVICES=0 \
  /workspace/embeddinggemma2-cpu-media-20261014/tf32-residual-probe \
  --output /workspace/embeddinggemma2-cpu-media-20261014/tf32-residual-screen.json
```

Each route uses ABBA W3/R20 and at least 0.1 seconds per phase. Timings include
activation splitting, three/four product submissions, F32 combination and
private-stream completion. Weight splitting is separately recorded as a cold
measurement and excluded from warmed phases. Uploads, resets, allocation,
arithmetic qualification and oracle checks are excluded. Original samples and
both orders are retained. Exit zero means execution completed; quality counters
and unavailable shapes must be read separately. No skipped shape is passing
coverage. Preserve source/compiler/executable/library hashes around the run.

This operator screen does not qualify the model or HTTP performance. A promising
result still requires complete independent raw-row/model and full-embedding
gates plus fresh immutable service comparisons. `model_qualified` and
`production_promoted` remain false.
