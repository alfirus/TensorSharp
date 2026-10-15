# Pedantic F32 cuBLASLt projection screen

This isolated tool compares cuBLASLt heuristic algorithms with the current
batch-one strided cuBLAS projection API. Both routes require
`CUBLAS_COMPUTE_32F_PEDANTIC`; no TF32, F16 or BF16 candidate is included.
It does not change TensorSharp production dispatch.

The descriptor uses column-major `[K,M]` weights with transpose A and
column-major `[K,N]` inputs, producing `[M,N]`. A private handle and nonblocking
stream own each operation. All transfers and canary resets precede execution
on that stream, with explicit completion before observation. Heuristic setup
is outside timing, requests at most 16 algorithms, and permits at most 32 MiB
of workspace. Alignment preferences account for the output's 16-float guard.
These APIs and the pedantic compute contract are documented by
[NVIDIA cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html).

```bash
nvcc -O3 -std=c++17 -arch=sm_86 --ftz=false --prec-div=true --prec-sqrt=true \
  eng/validation/embeddinggemma2-cuda-lt-projection-probe.cu \
  -lcublas -lcublasLt -o /workspace/embeddinggemma2-cpu-media-20261014/lt-projection-probe
/workspace/embeddinggemma2-cpu-media-20261014/lt-projection-probe \
  --output /workspace/embeddinggemma2-cpu-media-20261014/lt-projection.json
```

The six synthetic-input cases retain the compensated projection tool's
actual-size geometries, unrounded source bits and large cancellation fixture.
Every warm and final result checks all outputs for finiteness, both output
canaries, byte-exact input immutability, full classic-baseline differences and
up to 4096 independently evaluated Double positions. The unchanged Double
bound is `2e-6 + 2e-6 * abs(expected)`. Every output also gates against the full
classic baseline using that bound with the baseline value. Failed arithmetic is retained, including
any classic baseline failure on the cancellation fixture. An unavailable
algorithm is explicitly unavailable and cannot count as passing coverage.
Exit zero means execution completed; the report separately counts passed and
failed phases and records `quality_all_executed_phases_passed`.

Each available candidate runs ABBA against the classic baseline, with three
warmups, at least twenty samples and at least 0.1 seconds per phase. Timing
includes submission and private-stream completion, excluding transfers,
allocation, heuristic setup and validation. Preserve source, compiler,
executable and production binary hashes, and run only in a coordinated quiet
GPU slot. The source is not device-qualified until an actual compile and run
complete. A clear component winner still needs a separate production candidate
and complete model/raw-row/service validation. This is not model or latency
parity evidence.
