# EmbeddingGemma 2 audio frontend probe

Measures the managed 16 kHz PCM-to-mel frontend on deterministic 0.1, 1, 5,
and 30 second waveforms, separately for the default HF log-add policy and the
explicit llama.cpp clamp policy. Output includes every timing sample, warmup
count, assembly hash, and mel byte hash. Build and preserve a baseline directory
before changing the frontend, then run baseline and candidate sequentially on
an otherwise idle machine. Matching mel hashes test bit identity for these
inputs; the independent audio frontend fixtures remain the numerical gate.

```bash
dotnet build eng/validation/EmbeddingGemma2FrontendProbe -c Release \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
dotnet eng/validation/EmbeddingGemma2FrontendProbe/bin/Release/net10.0/EmbeddingGemma2FrontendProbe.dll \
  --report docs/validation/embeddinggemma2/frontend.json --warmup 8 --repeats 50
```

Optional `--samples` overrides the comma-separated waveform sample counts.
Generated reports and binary snapshots belong in ignored `docs/validation/` or
`artifacts/`. This probe does not run a model or native backend. Its timing
excludes file decoding, resampling, the audio tower, text pooling, HTTP, and
loading, and cannot establish end-to-end performance parity with llama.cpp.
