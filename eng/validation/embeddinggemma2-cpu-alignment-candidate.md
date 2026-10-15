# Isolated CPU alignment1 candidate

This tool makes exactly one insertion in a copied TensorSharp-owned encoder
source, after its optimized-weight loop:

```cpp
if (validated_cpu_geometry && e->cpu_float_linears) e->cpu_alignment = 1;
```

The insertion keeps quantized and all other model/backend geometries on their
existing policy. It does not change upstream ggml, production sources, CMake,
ordinary objects/libraries or frozen snapshots. The experiment is limited to
whether validated F32 CPU execution needs historical sequence padding.

The tool reads the actual CMake compile database. If that database omits the
owned source, it accepts the single known Makefile compile recipe and expands
only `CXX_DEFINES`, `CXX_INCLUDES`, and `CXX_FLAGS`. It uses `shlex` and direct
subprocess argv, never a shell or Make/CMake invocation. Only source/object,
dependency outputs and quote-header lookup are changed. The direct Linux ELF
link command replaces exactly one encoder object and its output path; response
files and unsupported linker side outputs/scripts are rejected.

The original `.o.d` file is required to bind actual included headers. Explicit
link objects/static/shared libraries, `-l` resolution, system linker scripts,
common compiler implicit libraries, compiler/link tools, source/headers,
metadata, the original object and ordinary library are hashed before and after.
Fresh candidate dependencies must stay within that captured header closure.
The unchanged tracked ggml checkout and revision are checked before and after.
Only a new ignored `docs/validation/` or `artifacts/` directory is accepted.

```bash
python3 eng/validation/embeddinggemma2-cpu-alignment-candidate.py \
  --build-dir /workspace/TensorSharp/TensorSharp.GGML.Native/build \
  --source /workspace/TensorSharp/TensorSharp.GGML.Native/ggml_ops_embeddings.cpp \
  --native-library /workspace/TensorSharp/TensorSharp.GGML.Native/build/libGgmlOps.so \
  --ggml-dir /workspace/TensorSharp/ExternalProjects/ggml \
  --expected-source-sha256 REVIEWED_CURRENT_SOURCE_HASH \
  --expected-native-sha256 FROZEN_ORDINARY_NATIVE_HASH \
  --expected-ggml-revision 3612c0af93f2cfc8f048295300a61187723b12ae \
  --output-dir /workspace/embeddinggemma2-20261014/docs/validation/embeddinggemma2-cpu-media-20261014/alignment1-candidate \
  --execute
```

Without `--execute`, the tool stages source and a command/hash manifest only.
With it, compilation/linking write isolated logs/object/dependency/library and
the manifest. Neither mode runs numerical tests or qualifies a model.

Run only in a coordinated post-media slot. Copy the candidate library into a
**new** immutable host/probe snapshot; retain the complete original snapshot.
Bind source/runtime/model/preprocessing hashes to every result. Before timing,
require the unchanged eight independent text cases plus 8192-token case,
batch/duplicate consistency checks, and full HF pooled media quality. Padding
changes can alter F32 reductions and cross F16 GELU-table discontinuities, so
strict original tolerances apply. A numerical failure rejects the candidate;
no inferred padding gain or production promotion is allowed.

Build-isolation controls run with:

```bash
python3 -m unittest eng/validation/test_embeddinggemma2_cpu_alignment_candidate.py -v
```

Their synthetic compiler tests only verify isolation/rejection/hash controls.
They are not native arithmetic or device/model coverage.
