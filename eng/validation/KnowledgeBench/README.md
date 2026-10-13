# Knowledge ingestion and retrieval benchmark

Run from the repository root:

```bash
dotnet run --project eng/validation/KnowledgeBench -c Release -- --output artifacts/knowledge/benchmark.json
```

The default run imports 1,200 deterministic Markdown documents with twelve sections
each, records durable ingestion throughput, 200 warm sequential retrieval samples,
restart latency, memory, and actual document/chunk counts. It verifies exact identifier
ranking, collection isolation, deletion and restart behavior. It also records retrieval
recall and reciprocal rank on `TensorSharp.Knowledge.Tests/Fixtures/retrieval-fixture.json`, a small manually authored set
with exact, paraphrase and Chinese queries. The fixture is a regression signal, not a
representative RAG quality evaluation.

To add actual GGUF embedding inference on the same quality fixture and a separate
1,200-document dense retrieval performance case:

```bash
dotnet run --project eng/validation/KnowledgeBench -c Release -- \
  --embedding-model /absolute/path/to/embedding-model.gguf \
  --embedding-backend CPU \
  --output artifacts/knowledge/benchmark-embedding.json
```

Use `--query-prefix` and `--document-prefix` when required by the selected model.
`--documents` and `--queries` control the lexical scale run; smaller diagnostic runs
record that they did not meet the 1,000 document / 10,000 chunk scale target.
`--embedding-documents` (default 1,200) and `--embedding-queries` (default 100)
control the separate dense-scale run. It embeds every unique short record using the
actual selected GGUF, persists and reloads its vectors, and exercises exact identifiers,
paraphrases, Chinese queries and broad terms after 20 warmup calls. The report includes
combined and per-query-kind latency, encoder-plus-commit ingestion throughput, actual
dimensions/model identity, and scope/deletion/restart checks. Model loading and store
reload timings are separate. This smaller chunk corpus must not be confused with the
28,800-chunk lexical run or the 14-query hand-authored quality screen.

The report separates lexical fallback from real embedding results. Without a model,
embedding validation is explicitly `not_run`. Dense retrieval is exact cosine with
reciprocal rank fusion, not ANN. The measured path includes parsing, chunking,
embedding when selected, persistence and retrieval. It excludes HTTP/UI latency,
generated answer correctness and actual media understanding. No external system,
concurrent-client or GPU comparison is implied. Temporary corpora are deleted; keep
generated JSON reports in the ignored `artifacts/` directory.
