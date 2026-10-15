Build against an exact frozen host, without rebuilding production dependencies:

```sh
dotnet build eng/validation/EmbeddingGemma2CompositeInputProbe -c Release \
  -p:TensorSharpCompositeProbeSnapshot=/absolute/frozen-host
dotnet eng/validation/EmbeddingGemma2CompositeInputProbe/bin/Release/net10.0/EmbeddingGemma2CompositeInputProbe.dll \
  --model /absolute/embeddinggemma-2-Q8_0.gguf \
  --pair /absolute/completed-controlled-pair/results.json \
  --output /absolute/ignored/empty-proof-directory
```

The probe rejects mismatched Models/Runtime or model hashes, incomplete pairs,
changed media, and nonempty output directories. It executes the frozen
production tokenizer and image/video codec/preprocessing helpers, without learned
inference. It records contiguous text runs including the controlled candidate
HTTP `Video:` insertion, selected frame indices, decoded RGBA and final CHW
pixels, codec/runtime identities, and relative artifact paths. Still images use
the actual 280-token reference-sizing processor. Audio files are identified by
SHA256 for binding to an existing independent waveform oracle.

The snapshot path is embedded at build time. The probe explicitly resolves the
actual platform's OpenCV native library from that frozen snapshot, since direct
DLL references do not reproduce NuGet runtime-target entries in its deps file.

Execute on the platform whose codec behavior is being qualified. Copying a
foreign managed assembly onto another OS does not validate its native decoder.
This is input evidence for an independent full-model oracle, not an independent
tokenizer/decoder or a performance benchmark.

For a canonical default-HF protocol report, replace `--pair` with the mutually
exclusive `--http /absolute/http-media.json`. This mode requires a completed,
passing report, successful server exit, the embedding-default guard, canonical
HF audio policy, and a resolved direct-host audit with the complete current set
of46 selectors and selector sources absent, including CUPTI. It binds the
manifest's hash and exact ordered cases/parts, model/media bytes, and loaded
Models/Runtime assembly hashes. It observes frozen multimodal option defaults
and rejects nonempty video prefixes, timestamps or different sampling/token
policies. Production decoding/preprocessing/tokenization remains the same.

Default-HF proofs use `source_report_kind="default_hf_http"`,
`source_report_path`, and `source_report_sha256`; they do not invent a paired
report. Legacy `--pair` fields and serialization are retained. The new mode
emits actual resized image/video pixels, which require fresh independent HF
math rather than reusing identity-size controlled-image arithmetic.

Pure report-audit/manifest rejection controls run without model inference:

```sh
dotnet run --project eng/validation/EmbeddingGemma2CompositeInputProbe.Tests -c Release
```

`embeddinggemma2-controlled-composite-reference.py` consumes this proof, checks
the identity-size PNG pixels and native-layout FFmpeg decode independently,
then executes four fresh video-frame vision forwards and three complete F32
text forwards. It preserves the original 0.9999 cosine / 0.002 component gates.
Only its recorded controlled inputs are qualified.

After a native-only optimization, use
`embeddinggemma2-controlled-composite-recheck.py --oracle ORIGINAL.json
--oracle-sha256 SHA256 --model MODEL.gguf --projector MMPROJ.gguf
--pair NEW/results.json --output NEW-RECHECK.json` to reuse that complete oracle.
The recheck requires unchanged numerical sources, weights, media, and ordered
input sequences. A rebuilt managed assembly requires `--input-proof NEW.json`
with fresh actual observations identical to the original proof, including every
token, pixel byte, sampling policy and codec identity. It records zero fresh model
math; original failed pairs remain unchanged. Both tools refuse to overwrite
their output reports.
