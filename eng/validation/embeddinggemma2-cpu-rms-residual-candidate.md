# Isolated whole-encoder RMS and residual candidate

This eng-only tool prepares a separate owned encoder source and captured build
plan. It never edits production/upstream sources or compiles a native library.
The original production source is pinned to `a949b81e…`, the supplied coherent
native to `936168c6…`, and unchanged ggml to
`3612c0af93f2cfc8f048295300a61187723b12ae`.

The transform replaces exactly three Gemma sites: post-attention normalization
plus residual ADD, post-FFN normalization plus ADD, and post-PLE normalization
plus ADD. All other operations, including learned layer scale and GELU, remain
byte-for-byte unchanged. It injects owned helper text directly in the private
encoder source, with no new include or upstream dependency.

The row arithmetic is extracted from the actual qualified standalone component
source SHA `37516bef…`: F32 squares, sequential Double sum, float mean, sqrtf,
ordered F32 input-times-scale, gain multiply, then residual add. Only helper
identifiers change. Function-local Clang `fp contract(off)` preserves the ADD
rounding without changing compilation of existing encoder functions. A GCC
function-specific attribute is retained as source portability support; that
branch is not qualified by the Mac experiment. The captured native compile
recipe must continue rejecting fast-math/unsafe floating-point overrides.

Initial admission is Apple ARM only, validated EmbeddingGemma2 CPU geometry,
all learned linears F32, and seven/eight configured workers. The predicate is
set after the optimized weight loop. Each factory also requires F32 2D width512
inputs/residuals, matching row counts1..8192, a shared512-element gain, contiguous
components, nonaliasing rows with checked span arithmetic, and positive finite
immutable encoder epsilon. Other platforms/geometries/workers, GPU, mixed
precision, larger batches and ineligible layouts retain the original graph.

The custom node has three explicit dependencies and no in-place output or
pointer cache. Its only borrowed userdata is the encoder's immutable epsilon;
the existing destructor frees both cached graph plans before encoder ownership
ends. There are no atomic or timed worker counters. A graph-build count and
node names expose actual admitted nodes when existing CPU profiling is enabled;
the planned upper bound is72 (three sites×24layers). Refusal can reduce that
count, so source availability or the upper bound is never dispatch proof.

Without arguments the tool prints the source-only transform identity. With
`--stage --output-dir /absolute/fresh/ignored/directory`, it invokes the reviewed
Mac builder with execute=false, creating private original/candidate sources,
an exact reference-native copy and guarded compile/link recipes. There is no
`--execute` option. The builder's optional source-transform/scope/additional
guard arguments preserve its original default behavior. Additional guards bind
this transform, its README, and the exact qualified component source. New row
text is normalized back to the component function in pure controls.

Generated output must be fresh and ignored; the reused builder checks logical
and physical containment, actual Git ignore rules, original source/build
separation, compiler spelling, compile/link metadata and captured input closure.
Both future libraries must share all unchanged nonencoder objects/static
archives. Rebuilt baseline must equal native936 or separately qualify against
it. Before/after and derived artifact guards must remain intact. Preparing a
recipe is not a successful compilation or native scenario.

Before any later build, independently review the exact staged source/diff and
captured plan. Then qualify strict8/8192, batch/reverse/duplicates/raw external
inputs, MRL, complete HF/media/protocol/lifetime outputs and actual diagnostic
node counts on the new immutable runtime. Only matched complete model/API
measurements can establish a gain. The component's small per-node savings make
no prediction of service gap closure, and no production promotion is automatic.
