# CPU attention batching at the existing padding boundary

This validation-only tool changes an isolated copy of the owned embedding
encoder, then reuses the guarded compiler/linker isolation in
`embeddinggemma2-cpu-alignment-candidate.py`. The ordinary native library and
upstream ggml sources remain unchanged. It requires explicit source, native
library and unchanged ggml revision hashes, and a new ignored output directory.
Without `--execute`, no compiler or linker is run.

The candidate only admits the currently qualified Linux x64 Gemma2 F32
seven/eight-worker geometry, alignment16 and mean pooling. A batch with mixed
actual lengths may use independent attention batch streams when every length
has the same existing padded length. Token order, token positions, padding,
flattened learned projection rows and worker count stay the same. Unlike the
old packed graph, it has one attention operation per layer and no segmented
attention concatenation tree. Mean pooling also changes from a zero-weight
block matrix to independent per-stream rows. Floating-point identity is not
assumed: the original full-model batch/single/duplicate gates must pass.

The eight pure controls cover exact source scope, admission prerequisites,
validation order, guarded-builder reuse, and exact layout/real-number mean
algebra. They do not execute native kernels or qualify a model/device.

Qualification must compare against the current frozen CPU build with identical
models, workers and compute policy. Include naturally mixed same-bucket short
inputs, original batch32-short, the original strict batch/single, duplicate,
raw-token and MRL gates, and unaffected single/equal-length/multiple-bucket
controls. Run both default compute and the explicit optional BLAS/attention
policy. Preserve failures and independently check complete saved vectors. Use
separate diagnostic graph profiles to prove admission and reduced attention/
concat node counts; diagnostic timings are not service timings. Only after
quality passes, run unprofiled ABBA HTTP measurements with original latency
gates and full closure guards. No speedup is established by this source tool.
