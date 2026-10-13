# Personal multimodal knowledge

[English](knowledge-system.md) | [中文](knowledge-system_zh-cn.md)

TensorSharp.Knowledge stores original files and searchable, cited representations
of their contents. TensorAgent adds collection management, uploads, saved-chat imports, search, and
per-chat selection. The same store and retrieval contracts serve the standalone
TensorSharp server and agent tools.

This guide describes current source support added after `v2026.10.10`; use a
build containing these changes. TensorAgent provides the library management UI.
The standalone server exposes management through `/api/knowledge` when configured;
its bundled browser chat does not provide TensorAgent's Knowledge menu. The CLI
has no collection-selection option. SDK callers can use the store directly.

The default is local BM25 text retrieval. Optional GGUF embeddings add semantic
retrieval; compatible loaded media models can derive text from images, audio, and
sampled video frames. TensorAgent automatically prepares local PDF OCR tools and
language packs when needed on Windows/Linux; chat and embedding models remain
explicit choices. See
[research and design rationale](knowledge-system-research.md) for the primary
open-source references that informed these independently implemented patterns.

## Using the library in TensorAgent

1. Open **Menu → Knowledge**, name a collection, and select **Create**.
2. Browse that collection and choose **Upload files** or **Upload folder**.
   Folder upload includes files in every nested folder and retains their relative
   paths, so identical filenames in different folders remain separate sources.
   TensorAgent uses the native folder picker inside the app and the browser's
   directory picker on the Web UI. The library retains originals after the
   upload cache or chat is cleared.
   **Add chats** opens a searchable, paginated history picker. Select up to 32
   saved chats across pages, then choose **Add and index selected chats**. The
   collection stores each transcript and available attachments or produced files;
   both user and assistant messages become searchable. Missing files are reported,
   and incomplete chats stay selected for retry. Private reasoning is excluded.
   Adding a chat again refreshes its transcript and files at stable source paths
   without duplicating unchanged sources. The snapshot does not update automatically
   when the conversation changes.
3. Read each document's extraction status and warnings. A batch can succeed for
   some files and fail for others; both outcomes are shown.
4. Use the collection counts, document/media/dataset totals, filename or path
   filter, type filter, sort order and page controls to browse a large library.
   **Rename** changes a collection's display name without changing its identity
   or the chats that select it. Search the collection's contents to inspect
   excerpts and source locations; **Read surrounding context** expands a result.
   Source links open a preview with indexed text, supported original media or
   PDFs, and **Download original**. PDF previews show indexed text first; native
   apps present the original through the platform file viewer. Long text previews
   offer excerpt navigation.
5. Check **Use {collection} in this chat** beside the upload controls, or select
   the collection's checkbox in the catalog. The upload panel shows whether the
   collection is available to this chat. Browsing, importing and reindexing do not
   select it for inference. Selected collection chips appear above the composer;
   up to 32 collections can be selected. A new chat starts with no selection, and
   reopening a saved chat restores its selection.
6. Ask a question or start agent work. Relevant evidence is retrieved before the
   model responds. When a model supports tools, it can discover sources, refine
   its search, expand evidence, and open originals for analysis.

Assistant turns show **Retrieved sources** only for sources explicitly cited in
the final answer, using an issued citation marker, an authoritative source URL,
or an exact retrieved `kb://` reference. Unused search hits and neighboring chunks
are omitted; an answer with no source citations has no source list. Links use host-verified identities and
survive saved-chat restoration and stream replay. Document citations do not claim
every retrieved page was used. Citation identity does not guarantee that every
statement in the answer is supported. Source text and titles are rendered as text.
Clicking either an answer's source link or a **Retrieved sources** link opens the
same preview, starting at the cited chunk when one is available. Stale or deleted
sources show an error instead of silently opening another version. Native
downloads present the original in the platform file viewer.
Image and video sources show thumbnails in search results, expanded excerpts, and
cited-source links. Supported video previews seek to the retrieved timestamp.
Audio, unsupported formats, and unavailable previews show media icons. Thumbnails
use the same authenticated, versioned originals as source previews; they are not
separate resized image files. The knowledge menu scrolls within the visible
viewport while keeping **Done** accessible, including at reduced viewport heights.

Folder uploads are processed in bounded batches with visible progress and
per-file failures. Unsupported formats remain visibly metadata-only; failed or
unreadable files are reported. Native imports report symbolic links instead of
following them outside the selected tree. Empty folders contain no indexable source.

Ordinary retrieval suppresses repeated evidence and stops after three consecutive
lookups without new evidence, 16 knowledge calls, or a cumulative evidence budget
(twice the response budget, with a 4,096-byte floor and 24,000-byte ceiling) per
agent conversation.
The budget includes eager evidence, full returned excerpts and citation metadata;
expanded chunks are charged in full when sent again. It is checked between calls,
so the last response can cross it. Repeated requests for disabled retrieval tools
end exploration instead of starting another planning round.
The model then answers from a bounded evidence snapshot; code tools remain
available for requested computations and artifacts. TensorAgent also limits
knowledge planning to at most 512 thinking tokens per generation by default,
within the generation's total token allowance. Smaller allowances below 512 tokens
are left unchanged, and explicit `TS_THINKING_BUDGET` overrides are respected.
Child agents inherit the request's thinking-token limit while keeping their own
transcript and cache scope; delegation does not reset that limit to a larger default.
When Muse's trained transition tokens are available, reaching the reasoning cap
closes that channel within the same generation and retains its KV state. A
tool-free turn opens the answer channel; a turn with tools preserves recipient
selection. At most three transitions are attempted. Unsupported framing or
continued reasoning retains the bounded thinking-off fallback, with collected
evidence and one final pass without host tools; verified-artifact workflows retain
their completion guard. This fallback is separate from normal retrieval completion,
which keeps code available. These bounds do not guarantee answer quality or latency.
An explicitly approved complete-document snapshot is a separate bounded path:
it can exceed the ordinary evidence ceiling after passing the model-context check
described below, then closes retrieval without increasing autonomous tool budgets.
Its exact payload, including middle sections, is retained for synthesis and recovery.
When the required evidence already fits in context, guidance favors direct synthesis
and discourages creating subagents solely to partition that evidence. Explicit
delegation requests and substantial independent tool, source, verification or
workspace work remain supported.
Muse-Glimmer's reasoning budget follows its message channels, and its tool-free,
thinking-off pass starts in the user-answer channel.
Budgeted Muse knowledge planning defaults to medium reasoning effort when unset;
explicit user effort is preserved. Other Muse requests retain their template default.

Delivered excerpts and approved complete-document snapshots supply short citation
markers such as `[K1]` and `[K2]`, stable per document/version within the request.
Only excerpts included in the serialized evidence budget issue markers; omitted
candidates do not authorize citations. The host expands an issued marker to its
canonical document link only while the selected collection and version remain
valid. Localized bracket forms such as `【K1】` and `［K1］` resolve only to an
actually issued marker; bracket-like text inside URLs, images or invented Markdown
links does not authorize a source. Structured JSON retains markers and carries
canonical sources separately in metadata. Unused or unissued markers do not produce source chips. Unknown
markers, external URL origins, and stale sources are never rebound.

At the end of a response, the host can also repair one narrow citation mistake:
a filename used where a document ID belongs in a local source URL. It substitutes
the ID only when that filename uniquely identifies a source already retrieved
for this request, in the URL's matching collection and version, and that version
is still current. On the WebUI path, a repair emits a complete-answer SSE
`replace` frame followed by `knowledgeCitationRepairs` with the repair count.
OpenAI/Ollama adapters normalize before emitting answer text and report the count
as `knowledge_citation_repairs`. Unknown or unretrieved links,
ambiguous filenames, other origins/collections, and stale versions remain
unchanged. This normalization neither retrieves new evidence nor proves generated
claims; the separate **Retrieved sources** links remain the authoritative source
identities supplied by the host.

Documents load in pages of 100 with **Previous files** and **Next files**. Refresh reloads the
current catalog. **Reindex** processes the retained original again using the
currently available extractors and embedding provider. This is useful after
loading a media model or changing an embedding model. Removing a document or
collection requires a second inline confirmation; removing a selected collection
also removes it from the current chat's selection.

## Extraction and provenance

| Input | Searchable representation and location | Coverage |
| --- | --- | --- |
| Text, Markdown, code, logs, XML, YAML and related text formats | Text segments; line range and Markdown section when available | Strict text decoding with BOM detection; binary data is rejected |
| CSV/TSV | Header/value records, including quoted multiline fields; record row | Searchable records, not a database query engine |
| JSON arrays, JSONL/NDJSON | Array item or line records; row and JSON item label | Bounded parsing; other JSON values remain text |
| HTML | Text with script/style/noscript removed | No script execution, rendering, or remote resource loading |
| SRT/VTT | Subtitle cues with start/end times | Supplied subtitles only |
| PDF | Selectable text plus OCR of complete scanned/image-bearing pages; original page number | Apple Vision on supported Apple hosts; app-managed Poppler/Tesseract in TensorAgent on Windows and Linux; no chat model required. Partial because recognition can misread text and does not reconstruct exact layout |
| DOCX | Paragraphs, tables, referenced notes | Partial: no rendered pagination, embedded images, or page layout |
| PPTX | Slide text in presentation order | Partial: no image/chart interpretation, speaker notes, or rendered layout |
| XLSX | Worksheet rows, cell coordinates, cached values | Partial: no formula recalculation or chart/image interpretation |
| Images | Generated description and visible-text transcription using a loaded vision model/projector | Partial: model output can omit or misread details; no native image embeddings |
| Audio | Generated transcript using a loaded compatible audio tower or an explicitly configured transcription companion | Partial: no diarization, word timestamps, or non-speech event extraction |
| Video | Up to eight described frames, with sample timestamps | Partial: no soundtrack indexing or coverage of events between samples; variable-frame-rate times are approximate |
| Unsupported formats or unavailable media capabilities | Original file and searchable filename metadata | Metadata only; content has not been understood |

`Ready` means the selected extractor produced content without a declared coverage
warning. `Partial` means content was extracted with limitations. `MetadataOnly`
means only the filename is indexed; its metadata chunk must not be mistaken for
extracted content. Document records include extractor identity, warnings, content
hash, version, and embedding identity where applicable. Generated descriptions
and transcripts always remain marked partial.

PDF extraction preserves the selectable text layer and also submits each page
containing raster images to OCR, so a heading or footer does not hide a scanned
body. Both default adapters render the complete crop box with its page rotation;
it does not select only the largest embedded image. OCR text carries the `ocr`
modality and original one-based page number. Extractor identity records the OCR
engine and operating system. The default Vision language set is Simplified Chinese,
Traditional Chinese and English, ordered as required by Vision. Pages with no recognized text, unavailable OCR,
timeouts, or resource limits appear explicitly in the document's warnings.
Existing PDFs can use **Reindex** to derive this content from the retained original.

The default PDF OCR budget is 128 pages per document and 16,777,216 rendered
pixels per page. Apple Vision rendering is capped at 200 DPI; the portable
Poppler/Tesseract provider requests 300 DPI, reduced when needed to fit the pixel
budget. The core checks 64 million embedded-image pixels in
total per page before rendering, with at most 4,096 image placements (repeated
placements count again). Each OCR
page has a 60-second cancellation deadline; native
CoreGraphics drawing checks cancellation between stages rather than being
forcibly interrupted. Total extracted text remains subject to the store's
character and segment limits. `IKnowledgePdfPageExtractor` is the portable core
extension point; the host uses local Apple OCR on macOS 13+, iOS 16+ and
Mac Catalyst 16+. Windows and Linux desktop hosts select the portable provider.
TensorAgent prepares its private Poppler/Tesseract installation automatically at
the first page requiring OCR, before starting that page's deadline. Selectable
text-only PDFs do not trigger a download. Dependency setup failure prevents the
import/reindex from committing, preserving any existing original and citations.
Standalone hosts can opt into the same provisioner or configure their own tools;
an unavailable manually configured provider reports scanned-page gaps.
The portable provider renders the full crop box with rotation and invokes tools
with argument arrays, bounded output, a private temporary directory, and a
cancellation deadline. Its extractor identity records engine versions, resolution,
selected languages, and language-data hashes when the data directory is resolved.

| Host | Default PDF OCR | Requirements and coverage boundaries |
| --- | --- | --- |
| macOS 13+, iOS 16+, Mac Catalyst 16+ | CoreGraphics + Apple Vision | Uses system frameworks; no separate OCR installation. iOS does not launch desktop utilities. |
| TensorAgent on Windows x64 / Windows 11 ARM64 | App-managed Poppler + Tesseract | Automatic first-use download; Windows ARM64 uses Windows 11 x64 emulation. Windows runtime validation requires an actual Windows runner. |
| TensorAgent core host on Linux x64 / ARM64 | App-managed Poppler + Tesseract | glibc runtime; automatic first-use download. No Linux MAUI desktop head is currently shipped. musl distributions require a host-provided backend. |
| Standalone Windows/Linux hosts | Poppler + Tesseract | Pass a `PdfOcrProvisioner` to `KnowledgeRuntime.CreateStore` for the same automatic setup, or configure local tools. |
| Other desktop hosts | Portable provider where process execution is supported | FreeBSD is eligible for this provider, but has not been runtime-validated. |
| Browser UI | The server host's provider | OCR runs on the connected TensorSharp/TensorAgent host, not in browser JavaScript. |
| Android and browser WebAssembly SDK hosts | No bundled provider | This repository has no corresponding application head; hosts can supply `IKnowledgePdfPageExtractor`. |

Executable paths and language configuration are administrator-controlled, never
document-supplied. The process backend does not provide an OS filesystem or hard
memory sandbox. Tools and fonts retain their separate upstream licenses, including
Poppler's GPL license; the lock, notices, and source recipe references accompany
the managed runtime. Current validation and its
Windows/device limitations are described in the
[portable and Apple validation instructions](../eng/validation/knowledge/README.md).

The media adapter uses TensorSharp's existing model scheduler. Images are limited
to 16,777,216 pixels. Local audio accepts PCM WAV up to 120 seconds; other formats
need a compatible `TS_JEV_TRANSCRIPTION_URL` companion or conversion before
reindexing. The companion has its own format and duration limits. Video requires
an installed compatible codec and vision model, with limits of ten minutes,
36,000 frames, and 16,777,216 pixels per frame. Each local media generation has a
three-minute deadline, a 768-token output budget, and a 16,000-character cap.
File extensions alone do not establish codec or model support.
Supported media MIME types can also identify uploads whose names have no
recognized extension. Codec readers receive a private copy with a known suffix;
the retained original name and bytes stay intact. Media generation is parsed
before indexing: protocol framing, private reasoning and tool calls do not become
source evidence. Image extraction requests visible text first, then a factual
description, to preserve names within the output budget.

Filenames and folder paths remain searchable after successful extraction without
being inserted into the returned source text. Explicit image, audio and video
lookups prefer matching original media of the requested type, preventing many
short code chunks mentioning the same name from hiding a matching asset. This
preference still requires a query match and respects selected collections. PDF
OCR stays document evidence. Optional model reranking receives the source
modality and both ends of long image/frame captions so trailing visible text is
available within its bounded excerpt budget. These are retrieval aids; the
default index still searches derived text rather than native media embeddings.

## Store lifecycle and retrieval

Each document is an atomic durable unit containing metadata, derived chunks, and
optional vectors. Original bytes are immutable SHA-256-addressed objects;
document manifests reference them. A single-writer lock protects the store.
Startup loads validated manifests, rebuilds the in-memory inverted index, and
cleans abandoned staging files and unreferenced originals. This is an embedded
personal store, not a distributed or multi-writer service.

Within a collection, the filename identifies an imported document. Uploading the
same filename and identical bytes is idempotent. Changed bytes replace that
document's derived content and advance its version. Reindexing also advances the
version. Failed extraction or embedding leaves the previous committed document
available. Deletion removes derived records and any original no longer referenced
by another document. Only the current version is retained: old citation reads
fail as stale instead of silently returning different content.

Chunks retain their source segment's location, modality, document ID, and
version. References have the form
`kb://{collectionId}/{documentId}@{version}/{chunkIndex}`. A location may describe
the whole segment containing a chunk rather than precise coordinates for each
character. Adjacent reads expand within the same document and version.

Collection filtering occurs before ranking. BM25 uses an inverted index and
Unicode-aware terms, including CJK n-grams. When embeddings are configured,
normalized exact cosine search considers only vectors with matching model
identity and dimensions. Reciprocal rank fusion combines the independent lexical
and dense rankings with deterministic tie breaking. Relevance chooses the first
result. For ambiguous broad questions, a modest repeated-page penalty encourages
diverse evidence within the bounded candidate pool; page numbers do not determine
relevance. Focused queries can return multiple strong excerpts from one page.
Returned scores include that selection penalty and follow the resulting order.
They are relative ranking signals, not calibrated relevance probabilities, and
should not be compared across queries or collections. Dense search scans compatible
vectors in the selected scope; there is no approximate nearest-neighbor index.

Explicit requests to find an image, audio file or video prefer matching originals
of that media type with matching query anchors. This avoids filling an image
lookup with source-code mentions of the same name. It does not turn PDF OCR into
an image source or infer a media type for every question about a format. Generated
media descriptions and visible text remain partial evidence; inspect the original
to verify details.

Chat hosts can then apply one bounded LLM relevance pass to the first eight
candidates, before clipping the evidence shown to the answering model. This pass
is enabled by default when a chat model is loaded; `TS_KNOWLEDGE_RERANK=0` disables
it. It is skipped for fewer than three candidates and for an approved complete
document snapshot. Eager retrieval and subsequent `knowledge_search` calls share
one pass per request, including child agents. The internal candidate pool is
capped at 12,000 text characters; the separate ranking input is capped at 12,000
serialized UTF-8 bytes with excerpts initially limited to 800 characters each.
The full bounded search query is preserved; an input that cannot fit falls back.
The pass uses a private session, no tools or thinking, at most 96 output tokens,
an actual tokenizer fit check, and a 10-second cancellation deadline. Cancellation
waits for inference cleanup before answer generation resumes. Invalid output,
model unavailability or timeout preserves the original retrieval order. The only
accepted model result assigns one integer usefulness rating from 0 to 3 to every
offered candidate ID exactly once. A private JSON grammar constrains the response
shape; the host validates all IDs and sorts by rating with baseline order for ties.
Ratings are relative to the information requested, not confidence probabilities;
source content, references and locations are retained and current versions are
checked after the await. No candidate is discarded solely for a low rating.
Result envelopes distinguish `order: llm_relevance`
from `order: retrieval`; `scoreKind: retrieval` identifies the unchanged first-stage
scores, which may differ from the final ordering. This pass cannot discover
evidence absent from the candidate pool or guarantee answer quality. Diagnostics
record outcome, candidate count and elapsed time without queries or source text.

The GGUF provider owns an encoder separately from the chat model. Its vector-space
identity includes the weights' SHA-256, query/document prefixes, and aggregation
version. Input longer than the encoder's token window is split and combined by a
normalized, token-count-weighted mean. Changing model or prefixes does not mix
old and new vector spaces: incompatible documents retain lexical search and need
reindexing for semantic retrieval.

Default store limits are 64 MiB per file, 2 GiB of logical source storage, 100
collections, 10,000 documents, 2,000,000 extracted characters per document,
10,000 segments/chunks per document, and 100,000 total chunks. Chunk size is 1,200
characters with 120 characters of overlap. Search is bounded to 50 results,
32,000 returned text characters, and 4,000 query characters. Complete-document
context expansion has a separate default cap of 64,000 indexed text characters;
it returns the entire conservatively deduplicated document or no candidate.
SDK hosts can supply `KnowledgeOptions` to set appropriate limits. Transient ingestion files and
derived indexes require additional disk and memory beyond source bytes.

## Inference and agent behavior

Automatic retrieval and knowledge-tool integration apply to the WebUI
`POST /api/chat` path in TensorAgent and the configured standalone server, and to
the standalone TensorSharp server's
`POST /v1/chat/completions`, `POST /v1/responses`, and `POST /api/chat/ollama`
adapters. Ollama `POST /api/generate` supports retrieval and citations without an
agent tool loop. `/api/chat` remains the WebUI route; the Ollama chat route is
explicitly `/api/chat/ollama`. These features are not implicitly added to every
`/v1` endpoint. Other applications can also use the core SDK or `/api/knowledge`
API to compose their own inference workflows.

`knowledgeCollections` in a supported inference request explicitly grants a
retrieval scope for that request. The field uses this same camel case in all
supported protocols. Missing or empty selection supplies no knowledge. Invalid or deleted
selections are rejected. Chat hosts with a tokenizer fit check can expand a strongly identified
source before generation: the initial search must return at least three hits,
all from the same document across the gathered candidates (up to eight with
reranking available, otherwise six), or the selected collections
must contain only that document. The default store cap is 64,000 indexed text
characters; the host also caps the candidate at 96,000 serialized UTF-8 bytes. It
then renders the actual prompt and uses the loaded model's tokenizer to check that the prompt, requested
answer allowance, and a 1,024-token margin fit both the model and engine capacity. History containing
media skips this expansion. A missing fit check, mixed-source hits, or a size/token
failure falls back to ordinary retrieval; no partial document is labeled complete.
This provides coverage of one source's indexed text, not proof that extraction is
complete, other relevant documents were searched, or the generated answer is correct.
Accepted expansion ends knowledge exploration; code and artifact tools remain
available under the normal completion rules.

Without an accepted complete-document candidate, tool-capable requests start with
up to two excerpts in a 4,000-byte envelope, leaving room for targeted follow-up searches. Tool-free
requests receive up to six excerpts within the full response budget. The host adds fixed
trust/citation guidance and presents source contents as untrusted reference data
in the current user message or a host continuation. Extracted instructions cannot
change the allowed collection set.
Knowledge guidance governs evidence handling, scope, uncertainty and citations.
It asks the model to preserve definitions, units, comparison criteria and relevant
qualifiers, and distinguish observations, opinions and measured outcomes without
generalizing from an isolated statement. Response structure, language and level of
detail follow the user's task. Guidance asks the model to explain relevant evidence
without unnecessary quotation or repetition. It is not a semantic verification of
the answer.

OpenAI/Ollama non-streaming responses add `knowledge_sources`,
`knowledge_citation_repairs`, and bounded `knowledge_tool_steps` metadata when
knowledge is selected. Streaming responses attach these to the terminal Chat
Completions chunk, `response.completed.response`, or Ollama `done: true` line.
The source objects retain camel-case identity/location fields. Tool steps record
actual host execution, up to 16 invocations with at most 512 UTF-8 bytes of stdout
per step and explicit truncation flags. They do not substitute for full execution
logs. `tool_choice: "none"` suppresses agent tools while preserving retrieval.

Knowledge-enabled compatibility adapters buffer final answer text until citation
normalization completes, then emit standard append-only text deltas. Reasoning
and tool activity may arrive earlier. This increases time to the first answer
delta; it does not imply that inference was idle. Structured-output requests keep
their schema-constrained content unchanged and expose authoritative sources as
separate metadata.

For models with tool support, the host provides:

| Tool | Behavior |
| --- | --- |
| `knowledge_list` | Discover selected sources in pages of up to 20 records |
| `knowledge_search` | Retrieve bounded, cited evidence within the selected scope |
| `knowledge_read` | Expand an exact reference with zero to two neighboring chunks on either side |
| `knowledge_open` | Retrieve the original's versioned source link and, when code tools have a private workspace, copy the complete original there for analysis |

The tools cannot modify the library or enlarge their scope. Subagents inherit the
same scope; original-file copies go into the invoking agent's private workspace.
This allows dataset calculations on complete originals rather than assuming that
the top retrieved rows represent a whole dataset. Retrieval response budgets
include serialized UTF-8 metadata and text, with an explicit truncation signal.

Fresh retrieval is rebuilt for each turn. Automatically inserted excerpts are not
stored as chat history. Source metadata is stored for citation display, but it
does not make an unselected collection accessible. A prior assistant answer is
still part of ordinary conversation history and may itself repeat source facts.

## Host setup and HTTP API

TensorAgent creates its library under `AgentPaths.DataRoot/knowledge` and exposes
management through its existing authenticated loopback server. The standalone
`TensorSharp.Server.Host` enables the library when `TS_KNOWLEDGE_DIRECTORY` names
a store directory. Both use the same store implementation. Collection IDs are
scope identifiers, not access credentials: a remote deployment must put all
management, search, source-download, and chat endpoints behind the application's
authentication and authorization boundary. The standalone library does not add a
multi-user permission model.

PDF OCR configuration is read when the host creates its store:

| Variable | Meaning |
| --- | --- |
| `TS_KNOWLEDGE_PDF_OCR` | `auto` (Apple Vision when supported, otherwise portable desktop OCR), `apple`, `tesseract`, or `none` |
| `TS_KNOWLEDGE_PDFTOPPM_PATH` | Optional absolute path to `pdftoppm` / `pdftoppm.exe`; otherwise resolve from absolute `PATH` entries |
| `TS_KNOWLEDGE_TESSERACT_PATH` | Optional absolute path to `tesseract` / `tesseract.exe` |
| `TS_KNOWLEDGE_TESSDATA_DIRECTORY` | Optional directory containing the chosen `.traineddata` files |
| `TS_KNOWLEDGE_OCR_LANGUAGES` | Explicit installed Tesseract names. Managed default: `eng+chi_sim`; `eng+chi_tra` selects the already installed Traditional Chinese pack. Custom system installations select their installed English/Chinese packs by default. |

TensorAgent needs no manual OCR installation with its default configuration.
It downloads pinned Poppler 26.09.0, Tesseract 5.5.3, native dependencies, and
official `tessdata_best` English, Simplified/Traditional Chinese, and both Chinese
vertical companion packs. Initial setup needs network access and approximately
335 MB of downloads on Linux ARM64; other platform sizes are recorded in the
embedded manifest. Subsequent imports and reindexing use the local installation
offline. OCR language models are downloaded; document bytes never leave the host
for recognition. The UI reports setup progress and allows retry after failure.

Downloads are verified by size and SHA-256. A pinned micromamba bootstrap installs
an explicit offline package list into a private cache, with configuration files,
environment-based package settings, and link scripts disabled. Concurrent callers
share setup; abandoning one request does not interrupt another. Failed installs
are never published, and verified archives survive cancellation for retry.
No administrator rights, system package manager, or global PATH change is needed.
The cache lives under `AgentPaths.PdfOcrDirectory`; original documents remain in
the separate knowledge data directory. An optional adjacent `ocr-assets` seed
directory allows offline first setup; see [dependency maintenance](../eng/ocr/README.md)
for verification and redistribution requirements.

Setting either executable path or the language-data directory opts out of managed
installation and uses that explicit local configuration. Restart after changing
configuration and reindex existing PDFs. The
portable extractor fingerprints the selected language models and installed
`_vert` companion models that Tesseract may load implicitly; companion additions,
removals, and file changes invalidate its cached model identity. The
[portable validation guide](../eng/validation/knowledge/README.md#portable-pdf-ocr-on-linux-and-windows)
provides managed setup benchmarks and optional custom-install instructions. Choosing
`none` disables OCR while preserving selectable PDF text and retained originals.

For portable OCR, the installed trained data also controls the accuracy/speed
tradeoff. Official `tessdata_best` favors accuracy; `tessdata_fast` favors speed.
The validation guide pins both profiles and includes Chinese vertical companion
packs. Managed setup selects the accuracy profile. Advanced deployments can point
`TS_KNOWLEDGE_TESSDATA_DIRECTORY` to another local profile. Keep source
documents available to verify uncertain characters even with the accuracy profile.

Optional embedding configuration, set before starting the host:

```sh
export TS_KNOWLEDGE_EMBEDDING_MODEL=/absolute/path/to/embedding-model.gguf
export TS_KNOWLEDGE_EMBEDDING_BACKEND=CPU
export TS_KNOWLEDGE_QUERY_PREFIX=''
export TS_KNOWLEDGE_DOCUMENT_PREFIX=''
```

Choose a supported sentence-embedding GGUF and use the prefixes required by that
model. Omit the embedding model variable for lexical retrieval. Media processing
uses the currently loaded compatible model; upload does not choose or download
one. A configured transcription companion is an external service and receives
the recording as part of the user's explicitly configured workflow.

Both hosts expose these routes below `/api/knowledge`:

| Method and path | Request / response |
| --- | --- |
| `GET /collections` | `{ collections: [...] }` |
| `POST /collections` | `{ name, description? }` → `{ collection }` |
| `DELETE /collections/{collection}` | Delete the collection and its sources |
| `GET /collections/{collection}/documents?offset=0&limit=100` | `{ documents, total, nextOffset }`; default 100, maximum 200 per page |
| `POST /collections/{collection}/documents` | Multipart upload → `{ documents }`; TensorAgent also returns per-file `errors` |
| `DELETE /collections/{collection}/documents/{document}` | Delete one source |
| `POST /collections/{collection}/documents/{document}/reindex` | `{ document }` with a new version |
| `GET /collections/{collection}/documents/{document}/source?version=1` | Original bytes as an attachment; stale version returns 409 |
| `POST /search` | `{ query, collectionIds, limit?, maxCharacters? }` → `{ results }` |
| `POST /read` | `{ reference, collectionIds, adjacentChunks?, maxCharacters? }` → `{ results }` |

TensorAgent accepts up to 16 files and 128 MiB per upload batch; per-file limits
still apply. The standalone endpoint accepts one file per request and validates
the multipart envelope before committing it. Source responses are downloads with
`nosniff` and `no-store`, preventing uploaded HTML from being served as active UI.
Ordinary argument/format errors use 400, missing sources use 404, stale references
use 409, and exceeded quotas use 413. Chat requests use `knowledgeCollections`;
search/read bodies use `collectionIds`.

TensorAgent additionally supports `POST /collections/{collection}/rename` with
`{ name }`, and `GET /collections/{collection}/documents/{document}/preview`
with a required `version` and optional `chunkIndex`. Preview responses contain
bounded indexed text, source metadata, a safe `inlineContentType`, and neighboring
excerpt positions. `source?version=1&inline=1` serves only allowlisted file
extensions with matching media/PDF MIME types; all other originals remain
attachments. TensorAgent's document listing accepts `q`,
`type=all|document|media|dataset`, and `sort=name|updated|size`, returning filtered
totals and an unfiltered collection summary. Folder multipart batches pair each
file with `relativePath0`, `relativePath1`, etc.; these paths are metadata and
never storage destinations. The store's original blobs remain content-addressed.

TensorAgent's authenticated `GET /chats?q=&offset=0&limit=100` lists nonempty
saved chats with `chats`, `total`, and `nextOffset`. Import selected history with
`POST /collections/{collection}/chats` and `{ conversationIds: [...] }` (1–32
IDs). The response separates committed `documents`, successful transcript
`importedChats`, and per-chat or per-file `errors`. Transcripts use
`Chats/{conversationId}/conversation.md`; copied attachments and produced files
live under the same prefix. Media extraction follows the collection's usual
model availability and coverage limits. Load a compatible model and **Reindex**
metadata-only sources to extract their content.

## Embedding in a .NET application

The core project has no dependency on TensorAgent or a running inference model:

```csharp
using TensorSharp.Knowledge;

var options = new KnowledgeOptions();
using var store = new KnowledgeStore(
    "/absolute/path/to/library", new BuiltInKnowledgeExtractor(options), options: options);
var collection = await store.CreateCollectionAsync("Project research");
await using var input = File.OpenRead("notes.md");
var document = await store.ImportAsync(collection.Id, "notes.md", input, "text/markdown");
var hits = await store.SearchAsync("launch requirements", new KnowledgeSearchOptions
{
    CollectionIds = new[] { collection.Id }, Limit = 6, MaxCharacters = 8000
});
if (hits.Count > 0)
{
    var context = store.Read(hits[0].Reference, maxCharacters: 8000,
        adjacentChunks: 1, collectionIds: new[] { collection.Id });
}
```

Custom extractors implement `IKnowledgeExtractor` and return ordered
`KnowledgeSegment` values with provenance, status, warnings, and extractor ID.
Custom embedding providers implement `IKnowledgeEmbeddingProvider` with a stable
model identity, dimensions, and batched query/document embeddings. The core SDK's
`Read` method allows an omitted scope for trusted local callers; HTTP and agent
adapters always supply explicit scope. Applications are responsible for their
own caller authorization.

For scanned PDFs, pass an `IKnowledgePdfPageExtractor` to
`BuiltInKnowledgeExtractor` as `pdfPageExtractor`. The portable core does not
implicitly load platform OCR; `KnowledgeRuntime.CreateStore` wires the default
host provider into TensorSharp/TensorAgent hosts. A .NET host can construct
`PdfKnowledgePageExtractor` with `PdfOcrOptions` to choose the provider, executable
paths, languages, data directory, and portable render DPI. Providers receive the original
path, one-based page number, remaining text budget, pixel limit, and cancellation
token, and return recognized text plus extractor identity and warnings.
The optional `PrepareAsync` hook runs once at the first eligible OCR page, outside
the per-page timeout; its default implementation does nothing. To enable managed
tools in another .NET host, own a `PdfOcrProvisioner` for that host's cache and pass
it as `managedPdfOcr` to `KnowledgeRuntime.CreateStore`. Dispose it at shutdown.

## Validation and current boundaries

Reusable tests live in `TensorSharp.Knowledge.Tests`, the knowledge tests in
`InferenceWeb.Tests`, and `TensorAgent/tests/TensorAgent.Tests`. They exercise
format extraction, provenance, durable lifecycle, stale references, model-space
compatibility, bounds, request scope, tool behavior, transport, saved-chat
metadata, and executable UI workflows. Optional live-model tests require an
explicitly configured model and must be reported as not run when unavailable.
The [TensorAgent browser probe](../eng/validation/tensoragent-knowledge-layout.py)
uses the real HTML/CSS and JavaScript with explicit API fixtures to check chat
selection, importing, scoped search, thumbnails, and preview navigation. Its
geometry checks cover long content, short and touch viewports, and a simulated
keyboard height. It does not exercise a native keyboard, MAUI device, or inference.

The [knowledge benchmark](../eng/validation/KnowledgeBench/README.md) measures
deterministic corpus ingestion, warm retrieval, restart, memory, and a small
authored retrieval fixture. Optional real embedding runs are reported separately
from lexical runs. The [end-to-end probe](../eng/validation/knowledge/http_probe.py)
exercises the real host. The [full-CSV agent probe](../eng/validation/knowledge/agent_probe.py)
checks a real model opening the complete uploaded CSV with `knowledge_open`,
executing Python aggregation, and returning the expected result. It records the
actual tool trace and does not accept an unsupported model success claim as
passing. This is one synthetic calculation, not a general agent benchmark.
The [protocol/OCR probe](../eng/validation/knowledge/README.md) exercises
streamed and non-streamed Chat Completions, Responses, Ollama chat/generate,
scope rejection, selected versus unselected evidence, downloadable citations,
multi-page scanned/mixed PDF retrieval, and optional complete-CSV tool execution.
Its fixture manifest and PdfPig checks distinguish actual raster-only content
from selectable text. Chinese OCR has its own result and optional required gate.
Agent checks require successful recorded source-open/code steps, exact output,
and valid answer citations; the optional supplied Python recipe is reported as
guided integration rather than autonomous code-writing validation. The
[OCR-only HTTP probe](../eng/validation/knowledge/pdf_store_probe.py) also checks
reindexing and stale versions without inference. The
[native PDF benchmark](../eng/validation/KnowledgePdfBench/Program.cs) separately
measures first-pass and warm full-page OCR, durable import, and retrieval on the
committed synthetic fixture. Supported Apple OCR is required for that benchmark.
The [portable OCR validation](../eng/validation/knowledge/README.md#portable-pdf-ocr-on-linux-and-windows)
uses real Poppler/Tesseract processes, pinned language data, and a separate
`PortablePdfBench` on each tested operating system. Its `--managed true --cache PATH`
mode measures cold setup separately from page OCR and records setup events; a
second process can prove cache reuse. Its Windows workflow and
scripts provide a reproducible lane; a configured workflow is not evidence that
Windows tests ran. Missing dependencies fail the strict benchmark, and any
platform skips remain separate from passes.
Generated logs, screenshots, and actual measurements
belong in ignored `docs/validation/` or `artifacts/`, not this design document.
Tiny media samples and synthetic queries establish only the scenarios exercised;
they do not demonstrate general document-understanding or RAG quality parity with
the researched systems.

Without an embedding provider, retrieval uses lexical matches. A broad request
such as “summarize this collection” can return no excerpts for a model without
tools; arbitrary source chunks are not inserted as a substitute. Ask about
specific terms, configure a supported embedding model, or use a tool-capable model
to list and open the selected sources for a complete review.

This implementation does not provide graph/temporal memory, connector
synchronization, ANN/vector-database scaling, dedicated cross-encoder rerankers, universal OCR,
native joint image/audio/video embeddings, long-video understanding, background
ingestion jobs, or multi-tenant ACLs. The extractor and embedding boundaries allow
future providers without changing original-file ownership or source citations.
