# PDF OCR validation

This tool runs actual Apple Vision OCR against the authored PDF fixture, then
checks durable knowledge-store import and page-specific retrieval. It does not
load a chat model. Unsupported platforms fail explicitly.

```sh
dotnet run --project eng/validation/KnowledgePdfBench -- \
  InferenceWeb.Tests/Fixtures/Knowledge/scanned-four-pages.pdf \
  artifacts/knowledge/pdf-ocr/benchmark
```

The first OCR pass is separate from three warm passes. The durable import runs
after OCR has warmed up and must not be reported as a cold upload measurement.

For fresh-process stage timings, choose `import`, `extract`, `parse`,
`native-main`, or `native-worker`:

```sh
dotnet run --project eng/validation/KnowledgePdfBench -- --cold-profile import \
  InferenceWeb.Tests/Fixtures/Knowledge/scanned-four-pages.pdf \
  artifacts/knowledge/pdf-ocr/cold-import.json
```

`import` times store creation, selectable-text parsing and image-header
preflight, each page OCR, durable import completion, and a warm reindex. The
other modes isolate parsing or native OCR on the caller versus worker thread.
JSON reports include elapsed stage time, process CPU time and worker availability.

A fresh process is not necessarily an OS-cold OCR run. Apple Vision can cache
compiled recognition models across processes for an executable identity. Report
first use of a new apphost separately from repeated launches of the same
apphost, and retain slow first-use measurements. These diagnostics do not clear
system caches or move initialization work into product startup. Keep all output,
native samples and exploratory probe files in ignored `artifacts/` directories.
