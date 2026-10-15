# Text CPU projection experiment

This standalone probe reads original checkpoint matrices and compares the
existing F32 OpenBLAS projection policy, forced channel/token partitions, ordinary
upstream Q8 products, and an explicitly residual-corrected Q8 graph. It changes
no production source or upstream ggml file. Ordinary Q8 products are a precision
diagnostic, not a passing quality lane.

For correction, `r = x - dequantize(quantize_q8(x))` is computed by an owned F32
custom node. The candidate graph is `mul_mat(W, x) + mul_mat(W, r)`, with F32
output addition. `--passes 3` adds one further residual and product; the default
is two passes. Quantization and dequantization call the unchanged public ggml
CPU/base type-traits APIs. Every Q8 product retains the original checkpoint Q8 weight bytes. Non-Q8 source matrices (including the published wide PLE) retain exact dequantized F32 comparisons, and Q8 routes are explicitly marked unavailable; weights are never requantized. The graph does not claim exact F32 arithmetic: it measures its
remaining error against products of original F32 activations and the checkpoint's
exact dequantized F32 weights, accumulated independently in double.

Actual tensor families include fused sliding/global QKV, fused FFN gate/up,
FFN down, attention output, per-layer gates/projections and the wide PLE
projection. Fusion concatenates original checkpoint rows without requantization.
Activations are deterministic synthetic F32 rows with one cancellation-sensitive
row; these are not captured hidden states. All backend buffer bases and interior views use 64-byte aligned guarded allocations, including original Q8 bytes. The weight bytes, dequantized weights
and inputs have immutable fingerprints. All outputs must be finite, written and
within surrounding canaries. Sampled edges, partition boundaries and deterministic
interior columns use `max(2e-5, 1e-6 * sum(abs(original products)))` forward-error
bounds, with both warmup and final-check failures/maxima retained per phase. Every BLAS phase verifies complete output coverage, configured workers,
and absence of fallback. The JSON keeps negative one-pass results and both ABBA
order ratios; the overall result requires at least one executed precise comparison and all its product gates to pass. Unavailable Q8 lanes are recorded separately and never counted as executed or passing coverage.

Compile against the actual coherent TensorSharp native bundle exporting unchanged
ggml CPU/base APIs; do not build a different ggml for this probe:

```bash
g++ -std=c++17 -O3 -pthread \
  -I /workspace/TensorSharp/TensorSharp.GGML.Native \
  -I /workspace/TensorSharp/ExternalProjects/ggml/include \
  /workspace/TensorSharp/eng/validation/embeddinggemma2-text-cpu-projection-probe.cpp \
  /workspace/TensorSharp/TensorSharp.Cpu.Native/embedding_q16_vnni.cpp \
  -L /workspace/TensorSharp/TensorSharp.GGML.Native/build \
  -Wl,-rpath,/workspace/TensorSharp/TensorSharp.GGML.Native/build \
  -lGgmlOps -ldl \
  -o /workspace/embeddinggemma2-cpu-media-20261014/text-cpu-projection-probe

TS_GGML_CPU_THREADS=8 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
  /workspace/embeddinggemma2-cpu-media-20261014/text-cpu-projection-probe \
  --model /workspace/models/embeddinggemma-2-Q8_0.gguf \
  --library /lib/x86_64-linux-gnu/libopenblas.so.0 --threads 8 \
  --families qkv-swa,gate-up,down,ple --rows 128,1024 --passes 2 \
  --warmup 3 --repeats 10 --minimum-seconds 0.15 \
  --output /workspace/embeddinggemma2-cpu-media-20261014/text-cpu-projection.json
```

The specified library must report pthread LP64 OpenBLAS with one worker. The
probe sets no thread policy. Record source/header/executable/native bundle,
provider, model SHA-256 and clean upstream revision alongside the JSON. Original
matrix-byte FNV fingerprints in the JSON also check immutability each phase;
they are not substitutes for the outer SHA-256 artifact bindings.

The optional `--routes blas-channels,blas-rows,q8-one-pass,q8-residual-corrected,ggml-f32` list selects comparisons (all five by default).

Run in a coordinated quiet window. Graph plans and the eight-worker CPU pool
persist; an OpenMP ggml build may still create its actual teams per submission.
Each comparison uses baseline/candidate/candidate/baseline, with separate warmup
and at least the configured count and duration in each phase. Timings include
quantization, correction, F32 addition, graph submission and completion. Loading,
graph creation and oracle checks are excluded. Repeating one weight matrix is a
favorable cache regime; there is no layer rotation. Neither complete model quality
nor service latency is established. A promising result still requires unchanged
strict eight-case and 8192-token independent model oracles, batch/single gates,
full development-set retrieval quality, complete media composition where relevant,
and fresh paired service timings before production use.

## Matched original-Q8 × Q16 VNNI screen

The explicitly selected `q8-q16-vnni` route links the unchanged TensorSharp-owned
two-dot AVX512 VNNI kernel. It retains original signed Q8 checkpoint bytes,
prepacks their lanes/scales once outside timing, and uses an owned validation
adapter for the same AVX2 Q16 activation quantizer. Two dependent custom nodes
prepare/pack activations and execute the retained 16-row × 64-column native
groups using the existing GGML worker team. Scratch, immutable prepacked weights
and graph plans persist across calls. Preparation, packing, precise subnormal-row
fallback, group dispatch and graph submission/completion are timed; this route
creates no managed graph or additional native workers.

An independent scalar oracle derives Q16 values from original F32 activations,
then multiplies the original checkpoint Q8 integers and scales. Sampled outputs
must agree bit for bit with that integer/F32 arithmetic. This additional check
does not replace or relax the existing independent original-F32 product bound.
All private scratch canaries and prepacked-weight fingerprints are checked,
alongside original input/weight immutability. Row/group counters prove complete
preparation and output work. Unsupported device/build and non-Q8 source cases are
marked unavailable and never counted as passed. Numerical failures retain all
ABBA timings and return exit 1 with `completed:true` and `passed:false`.

For a matched comparison against an independently built, unchanged-upstream
LLAMAFILE-ON library, compile the same sources against that library and bind its
binary/source hashes. The probe does not enable or change any production option.
Set `--baseline ggml-f32` explicitly; default baseline remains `blas-policy`.
The F32 baseline needs no OpenBLAS provider unless a BLAS route is also selected.

```bash
TS_GGML_CPU_THREADS=8 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
  /workspace/embeddinggemma2-cpu-media-20261014/text-cpu-projection-probe \
  --model /workspace/models/embeddinggemma-2-Q8_0.gguf --threads 8 \
  --baseline ggml-f32 --routes q8-q16-vnni \
  --families qkv-swa,qkv-global,gate-up,down,attn-swa,attn-global,inp-gate,proj,ple \
  --rows 32,128,256,512,1024 --warmup 3 --repeats 10 \
  --minimum-seconds 0.05 --output /workspace/embeddinggemma2-cpu-media-20261014/text-q16-matched.json
```

The published BF16 PLE is explicitly unavailable for the Q16 route. Including
the Q8 input-gate shape is an arithmetic screen, not permission to change the
existing intentional F32 model policy. A passing screen would still require
complete strict model and batch quality qualification before integration.
