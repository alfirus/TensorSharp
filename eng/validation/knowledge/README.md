# Knowledge HTTP and OCR validation

These scripts connect to an already-running isolated test host. They never start
a model, change the loaded model, or download weights. Coordinate model/host use
with other validation work: GPU/model execution may share a serial scheduler.
Generated reports, exact synthetic sources, raw protocol transcripts, screenshots,
and local application data belong in ignored `artifacts/` or `docs/validation/`.

## TensorAgent answer quality and retrieval loops

`answer_quality_probe.py` exercises the real `/api/chat` stream with an existing
collection. Start `eng/validation/TensorAgentHost` with a dedicated ignored root;
copy any library being diagnosed to that root's `data/knowledge` **before** starting
the host. Do not point the host at the live application's library. The probe creates
a chat in the isolated host but does not reindex, delete, or import knowledge.

```sh
python3 eng/validation/knowledge/answer_quality_probe.py \
  --connection artifacts/knowledge-answer-quality/host/connection.json \
  --collection test --think \
  --prompt 'Compare the grades and teacher comments for the student in this collection.' \
  --timeout 300 --max-tokens 2048 \
  --output artifacts/knowledge-answer-quality/answer
```

The report records total time, time to visible answer, tool steps, completion,
answer text, and whether displayed sources appear in the answer and resolve.
On its deadline it requests cancellation of the isolated turn. Raw transcripts
may contain private source text: keep them in ignored evidence directories.
Review factual accuracy and coverage against the original separately; passing
citation checks alone does not establish answer quality. Record the model,
thinking mode, token/context budgets, and whether a baseline used older binaries.
One run is diagnostic evidence, not a general performance benchmark.

For Muse, budgeted knowledge planning uses medium reasoning effort only when the
request leaves effort unset; explicit effort is preserved. Validate both factual
accuracy and latency when comparing effort settings. Delivered evidence issues
stable document markers such as `[K1]` and `[K2]`; final text expands cited markers
to current canonical source links, while structured JSON preserves them with
separate source metadata. Candidates excluded by the evidence budget do not issue
markers. Unused, unknown and stale markers never add source chips; external URLs
must remain unchanged.

## Image import, reindex and retrieval

`media_retrieval_probe.py` imports an image and reindexes it with the loaded
vision model, then verifies original bytes, idempotent imports, visible text,
filename lookup, query ranking and empty-scope isolation. It mutates the isolated
library copy. Copy the library before starting `TensorAgentHost` with a dedicated
root under `artifacts/` or `docs/validation/`; never use the live app's data root.
The probe requires the current launcher's `dataRoot` connection metadata.

```sh
python3 eng/validation/knowledge/media_retrieval_probe.py \
  --connection artifacts/knowledge-banner/host/connection.json \
  --collection test --image TensorSharp.Server.Host/wwwroot/images/banner_1.png \
  --query '请到知识库中寻找tensorsharp的宣传画' --expected-text TensorSharp \
  --output artifacts/knowledge-banner/media
```

Run `answer_quality_probe.py` separately with the same query and collection to
check the generated answer and authoritative citations. Record the actual
model/backend and coverage. The image probe does not validate caption accuracy,
native media embeddings, audio/video extraction, or other devices.

## Protocol matrix

The standalone `TensorSharp.Server.Host` exposes OpenAI Chat Completions,
Responses, and Ollama routes. Start it with a dedicated
`TS_KNOWLEDGE_DIRECTORY`, a compatible local model, and a configured code execution
backend when testing the agent path. Optional `TS_KNOWLEDGE_EMBEDDING_MODEL` selects
a local embedding GGUF. The ordinary TensorAgent loopback host is a WebUI host;
do not assume it also exposes all standalone compatibility routes.

```sh
python3 eng/validation/knowledge/protocol_probe.py \
  --base-url http://127.0.0.1:8099 --model model.gguf \
  --agent --require-chinese --samples 3 \
  --output artifacts/knowledge/protocol-matrix
```

Alternatively `--connection path/to/connection.json` accepts `baseUrl`, `model`,
and an optional loopback `cookie`; `--base-url`/`--model` override it. An authenticated
remote test host can use `TS_PROBE_BEARER_TOKEN`; it is sent only to the supplied
host and is not saved in reports. Use a dedicated test deployment, not a production
library. The probe creates and cleans up only its own collections.

By default the probe exercises both stream and JSON responses from:

| Option | Endpoint | Agent loop |
| --- | --- | --- |
| `chat` | `/v1/chat/completions` | With `--agent` and an available code backend |
| `responses` | `/v1/responses` | With `--agent` and an available code backend |
| `ollama` | `/api/chat/ollama` | With `--agent` and an available code backend |
| `generate` | `/api/generate` | Not supported; retrieval and citations only |

`/api/chat` remains the TensorSharp WebUI route. Do not send Ollama NDJSON requests
to it. Restrict diagnostics with `--protocols chat responses` and/or
`--modes stream`; unrequested protocols are not validated.

The probe checks malformed scope rejection before streaming, a random uploaded
fact with selected versus empty and unrelated-collection scopes, authoritative `knowledge_sources`, actual
versioned source downloads, and citations in model-written text. Knowledge-enabled
protocols may buffer answer deltas for final citation binding; measured latency
includes retrieval and generation, and first-byte time is not token-generation
speed. `--samples` controls sequential selected-fact samples; small counts are
only diagnostics. There is no concurrent or cross-product comparison.

PDF checks import the committed fixtures and require every expected page code and
table total in retrieved text at the correct page/document, then read/download the
references and ask the model about a scanned-page fact through every selected
protocol. Chinese phrases are reported separately; `--require-chinese` makes them
an explicit pass requirement. Phrase checks report raw exact and whitespace-normalized
matches separately, since some OCR engines insert spaces between Chinese characters;
different or missing characters still fail. `--skip-pdf` records `not_run`, never OCR success.

With `--agent`, the probe imports an original 500-row integer CSV, directs the model
to open the complete source and run Python, and requires authoritative successful
`knowledge_open` and `shell` steps, exact printed row count/sum, and the same values
in the final answer. Bounded `knowledge_tool_steps` metadata supplies execution
proof. Missing tools, unavailable execution, or truncated/unverifiable output
fails the case. A correct answer alone cannot pass. Without `--agent`, that
category is recorded `not_run`.

By default the model authors the Python. `--agent-recipe` supplies a safe CSV
reading recipe and records `promptMode: provided-python-recipe`; this measures
tool integration with a supplied algorithm, not autonomous code-writing quality.
Keep any failed unguided run when investigating a guided run. The final answer's
source URLs must also match retrieved identities and byte-verified originals.
`--agent-only` runs only this category and records scope/PDF checks as `not_run`;
it cannot substitute for the full protocol/OCR matrix.

The final report's `passed` concerns executed required cases; read its `coverage`
and per-category statuses before describing coverage. Raw replies are retained
for diagnosis. No retry hides an initial failure. Cleanup occurs after evidence
is saved, so source URLs from the temporary test collections intentionally become
unavailable after the probe exits.

## PDF fixtures and hermetic checks

See [fixture provenance and regeneration](../../../InferenceWeb.Tests/Fixtures/Knowledge/README.md).
The generator needs Pillow, ReportLab, pypdf and explicit font paths only during
fixture authoring. These packages are not product dependencies. Final committed
PDFs have exact hashes in the manifest; the image-only pages contain no text layer.
The mixed PDF includes a selectable heading over two separate image blocks.

```sh
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj \
  --filter FullyQualifiedName~KnowledgePdfFixtureTests
python3 -m unittest discover -s eng/validation/knowledge -p test_protocol_probe.py
```

The first independently inspects PDF structure with PdfPig; the second checks the
probe's SSE/NDJSON observers using canned protocol data. Neither establishes
native OCR or real-model inference success. Render all fixture pages and inspect
the previews after changes. Native OCR tests and live HTTP runs provide separate
coverage; record unavailable platforms/models as unvalidated.

## OCR-only host checks and timing

`pdf_store_probe.py` exercises PDF upload, page retrieval, original downloads,
reference reads, reindexing, stale versions, and empty/unrelated collection scope
through an existing TensorAgent or standalone host. It requires the Chinese
fixture phrases and performs no model inference:

```sh
python3 eng/validation/knowledge/pdf_store_probe.py \
  --connection artifacts/knowledge/pdf-host/connection.json \
  --output artifacts/knowledge/pdf-http
dotnet run --project eng/validation/KnowledgePdfBench/KnowledgePdfBench.csproj -- \
  InferenceWeb.Tests/Fixtures/Knowledge/scanned-four-pages.pdf \
  artifacts/knowledge/pdf-ocr/benchmark
```

The native benchmark requires supported Apple Vision OCR. It separately reports
the first pass and three warm passes over four pages, plus durable import and
retrieval. It uses no chat model or network and fails on unsupported platforms;
the clean synthetic pages do not establish general OCR accuracy or concurrency
performance. Preserve its runtime, operating system, fixture hash, and method
alongside any quoted latency.

For actual iOS execution, use the separate
[Apple PDF OCR probe](../ApplePdfOcrProbe/README.md). It builds the production
Models assembly into a disposable simulator app and checks provider routing,
scanned/mixed pages, Chinese, crop geometry, limits and the durable store. It
records first-class runtime evidence and warm page timings. Simulator results
describe the Mac host; an unsigned `ios-arm64` AOT/link build is only build
coverage and does not establish physical-device performance.

## Portable PDF OCR on Linux and Windows

TensorAgent's default Windows/Linux path prepares tools and language packs
automatically in its private cache. Use the
[managed first-use validation](../../ocr/README.md#validate-actual-first-use) to
prove an empty-cache download, private install, and offline reuse without system
OCR tools. `PortablePdfBench --managed true --cache PATH` exercises this path;
the instructions below exercise optional, explicitly configured installations.

The separate `eng/validation/PortablePdfBench` uses the actual Poppler/Tesseract
page provider and store. It requires functioning local executables and language
data: missing tools, failed OCR, missing fixture facts, or unavailable languages
fail validation. No chat model or ggml native execution is needed. The portable
wrapper skips building ggml explicitly and records that this is not ggml coverage.

On an isolated Ubuntu/Debian validation machine, provision the tools explicitly:

```sh
sudo apt-get update
sudo apt-get install -y poppler-utils tesseract-ocr
python3 eng/validation/knowledge/prepare_tessdata.py \
  --output artifacts/knowledge/portable/tessdata --data-profile best
python3 eng/validation/knowledge/run_portable_pdf_validation.py \
  --output artifacts/knowledge/portable/results \
  --tessdata artifacts/knowledge/portable/tessdata \
  --data-profile best --languages eng+chi_sim --samples 3
```

Install .NET SDK 10 before running the wrapper. It accepts `--dotnet`,
`--pdftoppm`, and `--tesseract` for explicit executable paths. The language-data
preparation is an explicit validation utility, not a product startup/download
hook. The validation default is `best`, using the official
[tessdata_best models](https://github.com/tesseract-ocr/tessdata_best); `fast` uses
[tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast). Upstream describes
these as an [accuracy versus speed choice](https://tesseract-ocr.github.io/tessdoc/Data-Files.html).
Both profiles download English, simplified/traditional Chinese, and the two
Chinese vertical companion packs. Each profile is pinned by commit, byte count,
and SHA-256 in `tessdata-best.json` or `tessdata-fast.json`. The wrapper checks all
five files, including companions that a model can load implicitly. Use separate
directories/output folders for profile comparisons and preserve their recorded
hashes. This custom-provider validation does not change product settings.
TensorAgent's managed default provisions the same five best-profile packs;
explicit executable/data paths select the user's existing installation instead.

On Windows, install Poppler's `pdftoppm.exe` and Tesseract 5 first, then invoke
the same scripts with `python` and explicit executable paths if needed. The
[upstream Tesseract documentation](https://tesseract-ocr.github.io/tessdoc/Downloads.html)
identifies modern Windows installers as third-party builds. The manual
[portable OCR CI workflow](../../../.github/workflows/knowledge-pdf-portable.yml)
uses pinned Chocolatey community packages
[Poppler 26.7.0](https://community.chocolatey.org/packages/poppler/26.7.0) and
[Tesseract 5.5.3.20260724](https://community.chocolatey.org/packages/tesseract/5.5.3.20260724),
then the same pinned official language data. The dispatch input defaults to
`best` and also supports `fast`. It runs Windows and Linux jobs and
retains failures as artifacts. It does not publish releases or require secrets.

The workflow must be dispatched on an available runner to establish Windows
coverage. Writing it, linting YAML, compiling on macOS, or exercising Linux does
not count as a Windows run. The wrapper records actual operating system,
architecture, engine versions, fixture/data hashes, test counters, explicit
`skippedTests` from TRX result outcomes, logs, and benchmark output. Some test
adapters leave the aggregate `notExecuted` counter at zero even when individual
tests were skipped; use the recorded outcomes. A skipped native scenario is never a pass.
Warm timing on one CPU/OS and clean synthetic scans is not a cross-platform
performance or general OCR accuracy guarantee.
The benchmark retains raw OCR and reports exact versus whitespace-normalized code
matches. Its store retrieval checks still require the literal source code at the
expected page; OCR normalization does not substitute for a working indexed source.
Wrong characters fail the quality check even when rendering, indexing, and
document lifecycle checks succeed. Rasterization, fonts, engine builds, and
language data can affect recognition; retain the failed text and report quality
separately from integration coverage. Do not change the expected text or accept
character substitutions to turn an OCR error into a pass.

## Existing WebUI probes

- `http_probe.py`: actual TensorAgent uploads, media-derived evidence, selected
  inference, scope isolation, citations, and document lifecycle.
- `agent_probe.py`: actual WebUI `knowledge_open` → Python aggregation over a
  complete CSV, requiring executed tools and exact stdout/answer values.

These accept the connection file written by `eng/validation/TensorAgentHost`.
Their small clean media fixtures are correctness checks, not a representative
speech/vision/OCR accuracy benchmark. Core ingestion/retrieval performance is
measured independently by [KnowledgeBench](../KnowledgeBench/README.md).
