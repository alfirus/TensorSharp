# Isolated SM86 vision launch-bound candidate

This tool creates one staged copy of `ggml_ops_attention_precision.cu`, compiles
only its owned CUDA object and relinks a separate `libGgmlOps.so` from the actual
captured native build commands. It never invokes the ordinary build or changes
production sources, upstream ggml, original objects, static libraries or runtime
snapshots. Run only in a serialized validation window.

The staged source duplicates the original generic-view head64 kernel with
`__launch_bounds__(128, 3)`. Its body, Q128/K16 tile, F32 arithmetic, `expf`, strides,
barriers and output layout are identical. Dispatch selects it only for actual
SM86, F32 width64 self-attention, query/key/value rows2304 or2340,12heads,batch1,
scale1. The original kernel and every other geometry, including width72 and
the1089-row video grid, remain unchanged.

```sh
python3 eng/validation/embeddinggemma2-cuda-launch-bound-candidate.py \
  --build-dir /workspace/TensorSharp/TensorSharp.GGML.Native/build \
  --source /workspace/TensorSharp/TensorSharp.GGML.Native/ggml_ops_attention_precision.cu \
  --native-library /workspace/TensorSharp/TensorSharp.GGML.Native/build/libGgmlOps.so \
  --ggml-dir /workspace/TensorSharp/ExternalProjects/ggml \
  --output-dir /workspace/TensorSharp/docs/validation/new-cuda-launch-bound-candidate \
  --expected-source-sha256 VERIFIED_OWNED_SOURCE_SHA256 \
  --expected-native-sha256 VERIFIED_BASELINE_LIBRARY_SHA256 \
  --expected-ggml-revision 3612c0af93f2cfc8f048295300a61187723b12ae \
  --execute
```

Without `--execute`, the tool only stages source/commands and hashes inputs.
The original object dependency file, compile database and direct shared-library
link recipe are required. Only the exact GgmlOps object entry is selected among
test-target duplicates. CMake's specific CUDA include response is expanded only
after its contents are verified to contain existing include paths. Other response
files, compiler side outputs, unknown CUDA options, host overrides and implicit
NVCC flag injection are rejected. No shell is used. The tool binds and rechecks
original dependency closure, owned/compile-database sources, original native
binary and link objects/static libraries, resolved dynamic/standard system
libraries, CUDA tools/profile/libdevice, host compiler tools and command metadata.
New dependency discovery must fit the captured closure before relink.

The manifest records source/library hashes, exact commands, logs, before/after
guards and unchanged upstream revision/status. Successful compilation does not
qualify numerical behavior or performance. Preserve the baseline native/runtime
snapshot and copy the candidate library into a new validation snapshot before
running unchanged required CUDA operators, all three full raw vision oracles,
complete default-HF/composite pooled checks and matched ABBA tower/service timings.
Earlier synthetic-grid gains are not an end-to-end result. The larger register
limit spills on A40 and regresses the1089 grid; dispatch must not be broadened
without new evidence. No production promotion is performed by this tool.

Pure command/isolation controls (no CUDA or model execution):

```sh
python3 -m unittest discover -s eng/validation \
  -p test_embeddinggemma2_cuda_launch_bound_candidate.py -v
```
