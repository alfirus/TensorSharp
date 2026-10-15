# Isolated macOS ARM Q/K/V RMS candidate

This eng-only builder makes an unmodified baseline source copy and a candidate
source copy, recompiles just the owned embedding encoder, and links each against
the same captured nonencoder objects/static archives/framework interfaces. It
does not run a normal CMake build or modify production/native/upstream files.
Keep all outputs in a fresh ignored `docs/validation/` or `artifacts/` directory.

The only candidate change extends the existing Linux x64 dispatch preprocessor
guard to macOS ARM64. The unchanged validated Gemma2 CPU geometry, effective
seven/eight-worker predicate, owned RMS header and minimum8-token crossover all
remain intact. Neither arithmetic nor the upstream dependency is edited. CPU
batch-bucket experiments must retain their independent Linux x64 guard; this RMS
extension alone does not qualify an ARM batching change.

```sh
python3 eng/validation/embeddinggemma2-cpu-rms-arm-candidate.py \
  --build-dir /absolute/TensorSharp.GGML.Native/build \
  --source /absolute/TensorSharp.GGML.Native/ggml_ops_embeddings.cpp \
  --native-library /absolute/TensorSharp.GGML.Native/build/libGgmlOps.dylib \
  --ggml-dir /absolute/ExternalProjects/ggml \
  --expected-source-sha256 PRESERVED_ENCODER_SHA \
  --expected-native-sha256 ACTUAL_ORDINARY_LIBRARY_SHA \
  --expected-ggml-revision 3612c0af93f2cfc8f048295300a61187723b12ae \
  --output-dir /absolute/repository/docs/validation/rms-arm-private-new \
  --execute
```

Without `--execute`, it validates/captures recipes and stages sources without
compiling or linking. Actual execution is restricted to macOS ARM64. Copied
source roots without Git metadata are unsupported: the output must pass the
actual repository's ignore check.

The existing captured compile helper is hash-pinned. Only the encoder source,
object, dependency-file and native output paths are redirected. Compilation and
linking preserve the actual captured C++ driver's **spelling** (for example
`/usr/bin/c++` or `clang++`): executing its symlink target as `clang` can disable
automatic C++ runtime linking. Compiler files/targets, selected Xcode tools/SDK,
recipes, original dependency closure, original objects/archives, owned source
headers and complete tracked upstream checkout are hash-bound before/after.
Uncaptured compiler/include/library/SDK/DYLD overrides, response files, side
outputs and unsafe floating-point flags are rejected.

Outputs are separate `reference`, `baseline`, and `candidate` directories. The
reference is the exact copied ordinary library. Baseline and candidate use the
same original nonencoder objects and identical captured compile/link flags,
apart from source/output/dependency path redirection and the common quote include
path to the unchanged owned headers. Fresh dependency files may not introduce
unbound headers. Original objects and ordinary library remain unchanged.

`manifest.json` records whether the recompiled baseline library's byte hash equals
the supplied ordinary hash. If it differs, root must first qualify that coherent
rebuilt baseline against the actual frozen reference before attributing any
candidate comparison to it. Matching hashes establish byte identity, not a
candidate model or performance result. System SDK `.tbd` files bind link-time
interfaces; Apple dyld shared-cache implementation bytes are not file-hashed.

The separate ARM component evidence covers complete synthetic Q/K/V output bits,
Double bounds, guards and measured worker rows at7/8 workers. It is not a full-model
or service result. This builder runs zero learned forwards. Independent source
review plus actual model/raw/batch/MRL/multimodal/HTTP and unprofiled matched timing
gates remain prerequisites for promotion. Unavailable or failed setups cannot
count as model passes.
