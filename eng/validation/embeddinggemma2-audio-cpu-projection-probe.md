# Audio CPU projection probe

This standalone executable compares unchanged upstream CPU `ggml_mul_mat` with
`GGML_PREC_F32` against TensorSharp's existing optional LP64 OpenBLAS
`cpu_matmul(minimum_columns=1)` callback. It does not edit production sources or
build ggml. Both graphs use the same persistent eight-thread CPU pool and retain
their graph plans. A provider that does not report pthread OpenBLAS with exactly
one BLAS worker is unavailable, not passing.
The ggml threadpool object remains alive across submissions. In an OpenMP ggml
build, unchanged upstream code manages its actual OpenMP teams at each
submission; this does not prove a fixed set of eight OS worker threads.

Classic audio geometry is read from `ggml_ops_gemma4_audio.cpp`:
`hidden = 8 * 128 = 1024`, FFN width 4096, and SSCP's two stride-two convolutions
reduce 128 mel bins to 32 frequency bins with 128 channels (4096 projection
columns). The probe covers FFN up/down, Q/K/V/attention output and SSCP projection
at 25, 48 and 96 rows. FFN down and SSCP have separate synthetic operands despite
sharing a matrix shape. This does not cover the 1024-to-2048 convolution GLU.

From a configured, already built unchanged upstream checkout on Linux:

```bash
g++ -std=c++17 -O3 -pthread \
  -I TensorSharp.GGML.Native -I ExternalProjects/ggml/include \
  eng/validation/embeddinggemma2-audio-cpu-projection-probe.cpp \
  -L TensorSharp.GGML.Native/build/g/src/ggml-cpu \
  -L TensorSharp.GGML.Native/build/g/src \
  -Wl,-rpath,"$PWD/TensorSharp.GGML.Native/build/g/src/ggml-cpu" \
  -Wl,-rpath,"$PWD/TensorSharp.GGML.Native/build/g/src" \
  -lggml-cpu -lggml-base -ldl \
  -o docs/validation/embeddinggemma2-cpu-media-20261014/audio-cpu-projection-probe

TS_GGML_CPU_THREADS=8 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
  docs/validation/embeddinggemma2-cpu-media-20261014/audio-cpu-projection-probe \
  --library /absolute/path/to/libopenblas.so.0 --threads 8 \
  --warmup 3 --repeats 20 --minimum-seconds 1 \
  --output docs/validation/embeddinggemma2-cpu-media-20261014/audio-cpu-projection.json
```

Use the actual existing build's library paths; some configurations place shared
libraries at its root. Compile and run in a coordinated quiet window after other
timings. Record the compiler command, executable/header/provider/shared-library
hashes and unchanged ggml revision alongside the report. The probe observes
thread policy; it does not call a BLAS setter or modify process environment.

The qualified remote build bundles unchanged ggml statically inside
`libGgmlOps.so`, and exports the same CPU/base API. Link that exact coherent
bundle rather than a different llama.cpp ggml revision:

```bash
g++ -std=c++17 -O3 -pthread \
  -I /workspace/TensorSharp/TensorSharp.GGML.Native \
  -I /workspace/TensorSharp/ExternalProjects/ggml/include \
  /workspace/TensorSharp/eng/validation/embeddinggemma2-audio-cpu-projection-probe.cpp \
  -L /workspace/TensorSharp/TensorSharp.GGML.Native/build \
  -Wl,-rpath,/workspace/TensorSharp/TensorSharp.GGML.Native/build \
  -lGgmlOps -ldl \
  -o /workspace/embeddinggemma2-cpu-media-20261014/audio-cpu-projection-probe

TS_GGML_CPU_THREADS=8 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
  /workspace/embeddinggemma2-cpu-media-20261014/audio-cpu-projection-probe \
  --library /lib/x86_64-linux-gnu/libopenblas.so.0 --threads 8 \
  --warmup 3 --repeats 20 --minimum-seconds 1 \
  --output /workspace/embeddinggemma2-cpu-media-20261014/audio-cpu-projection.json
```

The native library must stay immutable during this execution. This direct CPU
probe does not initialize the bundled CUDA runtime or run production model code.

Each shape runs baseline/BLAS/BLAS/baseline. Each phase warms up three times and
measures at least 20 calls and one second. Guard regions surround every operand
and output, all output elements must be finite and written, and immutable source
fingerprints must remain unchanged. Sampled edges, every worker partition
boundary and deterministic interior columns are checked against independent
double products with `max(2e-5, 1e-6 * sum(abs(products)))` forward-error bounds.
Each phase refills its output canary before warmup. BLAS phases must record
exactly eight nonempty channel partitions per call, the expected complete output
element count, all eight worker bits, and no rejection or scalar fallback.

The same synthetic weight matrix is reused throughout each shape, which is a
favorable cache regime. Timings include graph-plan submission and completion,
exclude construction, input uploads, output readback and oracle checks, and do
not establish complete audio tower or service speedups. The report retains all
samples and both order ratios. Production integration still requires complete
raw-tower, pooled HF/resampled quality and end-to-end latency qualification.
