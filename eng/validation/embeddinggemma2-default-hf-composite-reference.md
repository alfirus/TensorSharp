# Complete default-HF multimodal reference

This tool checks the original seven default-HF HTTP cases through both APIs
against independent complete F32 vision and text math. It uses a fresh
`EmbeddingGemma2CompositeInputProbe --http` observation, tied to exact frozen
Models/Runtime assemblies and the actual ordered HTTP inputs. A successful
before/after runtime guard must bind every HTTP binary identity to one complete
unchanged snapshot. All arithmetic/instrumentation selectors must be absent;
explicit thread policy remains recorded separately.

The input proof observes the production tokenizer and codec. The Python tool
then independently decodes opaque PNGs and video through FFmpeg in the observed
channel format, verifies RGBA bytes, checks HF sizing and sampling, and compares
float bicubic pixels at a maximum absolute difference of `2e-6`. PIL F-channel
bicubic is a bounded approximation to the HF torchvision float resize, not a
claim of identical behavior across all HF processor/codec variants. Alpha
compositing and alternate codecs are outside this fixture's qualification.

All unique image/video vision forwards and all seven complete text forwards
execute fresh. The preserved canonical HF audio soft rows are explicitly
reused and are not counted as fresh audio-model math. Their source, waveform,
exclusive-left window policy and projector are hash-bound to the previous
complete independent HF audio anchor. Four unchanged independent arithmetic
sources are bound to the explicitly supplied complete composite anchor; no
TensorSharp learned-model implementation is imported.

```bash
PYTHONPATH=/Users/zhongkaifu/work/llama.cpp/gguf-py \
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
artifacts/embeddinggemma2/python313/bin/python \
  eng/validation/embeddinggemma2-default-hf-composite-reference.py \
  --model /Users/zhongkaifu/work/models/embeddinggemma-2-Q8_0.gguf \
  --projector /Users/zhongkaifu/work/models/embeddinggemma-2-mmproj-Q8_0.gguf \
  --input-proof /absolute/path/fresh-http-input-proof/composite-input-proof.json \
  --media-root /absolute/path/actual-media-files \
  --anchor-oracle /absolute/path/composite-oracle.json \
  --anchor-sha256 EXPLICIT_COMPLETE_ORACLE_SHA256 \
  --hf-audio-embedding-oracle /absolute/path/pooled-audio-independent-frozen-final-hf.json \
  --hf-audio-embedding-oracle-sha256 EXPLICIT_HF_AUDIO_ANCHOR_SHA256 \
  --runtime-guard /absolute/path/v9-gates-artifact-guard.json \
  --ffmpeg /absolute/path/verified/ffmpeg \
  --http /absolute/path/cpu/http-media.json \
  --http /absolute/path/cuda/http-media.json \
  --output /absolute/path/new-ignored-evidence/default-hf-complete.json
```

Both protocols require finite unit 768-dimensional embeddings, exact ordered
parts/token accounting, cosine at least `0.9999` and maximum absolute error at
most `0.002`. The report preserves every comparison and any failure. Parsed
JSON bytes and all sources, weights, media, observed buffers and FFmpeg binary
have before/after hash guards. A failed decoder/pixel/source guard does not
qualify a model scenario. Existing evidence is never overwritten.

The pure evidence-control tests reject stale assemblies, changed runtime
snapshots, instrumentation, incorrect policy, malformed pixels and geometry
cache aliasing. They do not count as model execution. These numerical checks
also make no latency, broad retrieval or llama pipeline equivalence claim.
