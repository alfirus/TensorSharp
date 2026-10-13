# Existing-store retrieval probe

Run a lexical retrieval regression against a disposable snapshot of a local
knowledge store. The probe never opens the input with `KnowledgeStore`, whose
constructor can perform recovery and garbage collection. It copies manifests and
originals to a temporary directory and deletes that directory after validation.

```sh
dotnet run --project eng/validation/KnowledgeRetrievalProbe -- \
  /path/to/knowledge 'My collection' 'Riley的成绩和老师评语' \
  artifacts/knowledge/retrieval-report.json 200 6
```

Arguments are store path, collection name, query, report path, optional iteration
count (default 200), and optional result limit (default 8). Reports must be outside
the input store. They include document ordinals, chunk indices, pages, text
lengths, truncation flags and warm p50/p95 latency. Source text, filenames,
identifiers and the query are omitted from the report. Document ordinals follow
the store's filename-sorted document listing.

Keep generated reports under ignored `artifacts/` or `docs/validation/`. This
measures in-process lexical retrieval after 20 warmups. It does not measure OCR,
embedding inference, HTTP/UI latency or generated-answer quality. Copying a live
store is not an atomic snapshot; avoid concurrent imports or removals during the
probe.
