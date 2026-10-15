# Saved complete default-HF regression

This tool checks new frozen CPU/CUDA HTTP output against the seven complete
independent vectors produced by `embeddinggemma2-default-hf-composite-reference.py`.
It performs no fresh learned model computation. It preserves the original
0.9999 cosine / 0.002 component limits and both protocol comparisons, exact
weights/media/ordered input sequences, original successful source evidence,
and a complete unchanged runtime guard covering every candidate HTTP binary.

When Models/Runtime preprocessing assemblies are identical, the original exact
input observations can be reused. A changed preprocessing assembly requires
`--input-proof` from the frozen candidate, bound to its actual HTTP report;
every observed tokenizer, resize, codec and sampling result must be identical
to the original. Native-only changes still require their own complete runtime
guard. A model/source/media mismatch is an error, never a numerical pass.

```sh
python3 eng/validation/embeddinggemma2-default-hf-composite-recheck.py \
  --oracle /absolute/default-hf-complete.json --oracle-sha256 ORIGINAL_SHA256 \
  --model /absolute/embeddinggemma-2-Q8_0.gguf \
  --projector /absolute/embeddinggemma-2-mmproj-Q8_0.gguf \
  --http /absolute/candidate/http-media.json \
  --runtime-guard /absolute/candidate/complete-runtime-guard.json \
  --output /absolute/new-recheck.json
```

Repeated backends/reports are rejected rather than counted as extra coverage.
Existing output paths are rejected to preserve failed or incomplete evidence.
The original synthetic opaque-media, observed tokenizer, approximate float
bicubic and reused independent audio-row limitations still apply. This does
not qualify service latency, a retrieval leaderboard or default llama/HF
pipeline equivalence.
