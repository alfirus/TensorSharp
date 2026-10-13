# Multimodal knowledge system: research and design rationale

Research accessed **2026-10-10** (America/Los_Angeles). This is a design reference,
not a benchmark report or a claim that TensorSharp implements every capability
described below. The starting discussion was [the shared knowledge-base
discussion](https://chatgpt.com/share/6aca8594-5ce4-83e8-b268-baf0a14a5cab).
The shared page is not a primary technical specification; the project-owned
documentation and source linked here provide the implementation guidance.

## Lessons from the referenced projects

| Project and primary source | Observed pattern | TensorSharp / TensorAgent design implication |
| --- | --- | --- |
| [Open WebUI knowledge tools](https://docs.openwebui.com/features/workspace/knowledge/) | Separate collection discovery, file-name lookup, exact content search, semantic retrieval, and bounded file reads. Attached knowledge scopes agent tools. Directory synchronization compares content hashes. | Expose small, composable knowledge tools. Discover a collection, search within it, then read the cited chunk or document. Do not insert the entire library into every prompt. Scope must be enforced by application code. |
| [LlamaIndex nodes](https://developers.llamaindex.ai/python/framework/module_guides/loading/documents_and_nodes/usage_nodes/) and [auto-merging retriever](https://developers.llamaindex.ai/python/framework/integrations/retrievers/auto_merging_retriever/) | Chunks retain document metadata and relationships. Auto-merging can replace related retrieved children with their parent context. | Keep stable source and chunk identities, chunk order, section paths, and source locations. Allow bounded adjacent or parent-context expansion without conflating the retrieval chunk with the source document. |
| [RAGFlow](https://github.com/infiniflow/ragflow) | Parsing quality, format-dependent chunking, visible references, and combined retrieval are first-class parts of the system. | Extraction diagnostics and citations belong in the ingestion/retrieval contract. A file that uploaded successfully is not necessarily a file whose contents were successfully indexed. |
| [Onyx connectors](https://docs.onyx.app/overview/core_features/connectors) and [project overview](https://github.com/onyx-dot-app/onyx) | Ingestion carries metadata and permissions into a searchable internal representation. Document indexing is a substantial service, separate from a lightweight chat interface. | Share a knowledge service across hosts and clients. Apply collection/document scope before search, listing, reads, and mutation. Local collection selection alone is not a multi-user authorization system. |
| [Dify retrieval service](https://github.com/langgenius/dify/blob/main/api/core/rag/datasource/retrieval_service.py) | Retrieval supports bounded results, document/metadata filtering, optional reranking, score thresholds, and duplicate removal using provider/document identity. | Make search options explicit, deduplicate by stable identity, and provide component scores/citations so retrieval behavior can be tested independently of generated answers. |
| [Docling document model](https://docling-project.github.io/docling/concepts/docling_document/) | One representation preserves text, tables, pictures, hierarchy, reading order, provenance, and layout where available. | Extractors should return ordered typed content segments with optional page, row, slide, bounding-box, and time provenance. Markdown/plain text is a searchable representation, not a replacement for the original asset. |
| [MinerU](https://github.com/opendatalab/MinerU) and [output contract](https://github.com/opendatalab/MinerU/blob/master/docs/en/reference/output_files.md) | Parsing results support structured content and stable page/block locations. Different parsing tiers trade speed and extraction quality. | Preserve source locators and record the extractor/version used. Select expensive OCR/layout work explicitly and report unavailable capabilities. Avoid promising equivalent quality from a basic text extractor. |
| [MarkItDown](https://github.com/microsoft/markitdown) | Format converters normalize documents to Markdown, with optional dependencies and plugins for additional capabilities. Image descriptions and richer audio/video processing depend on configured services/models. | Use an extractor interface and explicit capability detection. Keep the dependency-light .NET path usable; add native or external adapters without coupling the knowledge store to a Python service or requiring network access. |
| [VideoRAG](https://github.com/HKUDS/VideoRAG) | The architecture combines visual and audio evidence, temporal context, and graph-oriented indexing for long videos. | Represent video evidence as time-bounded transcript/frame segments linked to the original video. Transcript import alone does not establish visual understanding or long-video quality parity. |
| [LightRAG](https://github.com/HKUDS/LightRAG) and [lifecycle operations](https://github.com/HKUDS/LightRAG/blob/main/docs/ProgramingWithCore.md) | Incremental insertion/deletion must maintain derived chunks, vectors, and graph provenance. Indexing and querying must use a compatible embedding model; changing it requires re-embedding. | Store embedding identity alongside vectors and reject incompatible spaces even when dimensions match. Replace/delete all derived artifacts together; graph layers, if added, must preserve evidence ownership. |
| [Graphiti](https://github.com/getzep/graphiti) | Facts have temporal validity and source episodes. Retrieval combines semantic, lexical, and graph information. | Preserve creation/update time and source lineage now. A future temporal-memory layer should retain contradictory/superseded evidence explicitly; a flat document index must not claim to implement temporal reasoning. |
| [Pathway document store](https://pathway.com/developers/api-docs/pathway-xpacks-llm/document_store) and [connector semantics](https://pathway.com/developers/user-guide/connect/connectors-overview/) | Parsing, splitting, metadata, and retrieval are separate stages. Streaming changes include additions and retractions, and a replacement is a deletion plus addition within one update. | Stable identity, content hashing, atomic replacement, and removal of obsolete chunks are prerequisites for future synchronization. Do not append duplicate vectors when a source is updated. |
| [Khoj query filters](https://docs.khoj.dev/miscellaneous/query-filters/) | Natural-language search can be constrained by file, terms, and dates. | Provide structured filters alongside query text. A bounded exact read is useful after semantic discovery; identifiers and code should remain searchable without an embedding model. |
| [Qdrant collections](https://qdrant.tech/documentation/manage-data/collections/) and [hybrid queries](https://qdrant.tech/documentation/search/hybrid-queries/) | Vector spaces specify dimensions and metrics; named vectors distinguish representations. Hybrid retrieval fuses separate candidate rankings. | Persist a vector-space descriptor including model identity, dimension, and normalization/metric. Keep lexical and vector scores separate, and combine ranks rather than directly adding incompatible score scales. |
| [LanceDB hybrid search](https://docs.lancedb.com/search/hybrid-search) | Full-text and vector retrieval can be fused with reciprocal rank fusion; explicit limits and prefilters control the candidate set. | Use bounded candidate pools, deterministic tie breaking, and scope filtering before ranking. An embedded exact-search baseline can serve small local libraries while preserving a future indexed-storage boundary. |

These are independently implemented architectural patterns. No source code from
these projects is copied, and this research does not introduce them as runtime
dependencies. Product documentation and default-branch source may change after
the access date; no upstream benchmark numbers are used as TensorSharp results.

## Proposed system boundaries

1. **Source assets and ingestion.** A collection contains stable document records.
   Retain original bytes, content hash, display name, media type, ingestion status,
   and extractor diagnostics. An extractor returns ordered segments and provenance;
   chunking and embedding are independent stages. Limits apply to bytes read,
   decompressed content, segment count, chunk size, and provider work. Failed or
   cancelled extraction must not publish a partially replaced searchable document.
2. **Persistent knowledge index.** Publish document metadata, chunks, lexical
   information, and optional vectors together. Reopening the service must preserve
   search results. Duplicate input must not multiply chunks. Reindexing a document
   must remove obsolete chunks, and deletion must remove the source and derived
   records without affecting unrelated collections.
3. **Retrieval.** A lexical path works without a loaded embedding model. An optional
   embedding provider supplies vectors in one explicitly identified space. Rank
   fusion combines independently retrieved lexical and dense candidates; scores
   are relevance signals, not probabilities. Enforce collection/document filters,
   result count, and returned-text budgets before returning evidence to the caller.
4. **Inference and agent integration.** Users attach selected collections to a
   conversation or agent run. Search returns short excerpts and stable citations;
   read tools expand only the relevant evidence. Retrieved text is untrusted source
   material, never an instruction to execute tools or change scope. Existing chat
   without selected knowledge should continue to work.
5. **TensorAgent management.** The library surface should expose collection creation
   and selection, upload, processing status, retrieval preview, source/chunk reading,
   reindexing, and deletion. It should state whether a file is fully extracted,
   represented by supplied text, or awaiting an extractor/model.

The .NET service owns the data contract and lifecycle. Native inference remains a
provider behind that contract. OCR, transcription, captioning, and joint multimodal
embeddings are different capabilities: accepting a media file is not evidence
that all four ran successfully. Supplied captions/transcripts are useful evidence,
but must remain distinguishable from model-extracted observations.

## Identity, scope, and performance decisions

### Answer completion and evidence selection

Follow-up review on **2026-10-11**: [Open WebUI's knowledge guide](https://docs.openwebui.com/features/workspace/knowledge/)
documents bounded reads and notes that smaller models can struggle with autonomous
multi-step retrieval. TensorSharp therefore tracks repeated calls and new evidence
within a request, and switches stalled retrieval toward synthesis. Requested code
and artifact work retains its tools; repeated requests for disabled retrieval
trigger a terminal answer pass. The ceiling is a termination safeguard, not a claim
that all evidence has been read. Missing evidence must remain explicit in the answer.

For a strongly identified source, bounded document expansion can keep tables,
headings and related sections together. Chat hosts with a tokenizer fit check consider this only when
the initial search has at least three hits and all gathered candidates (up to
eight when reranking is available, otherwise six) belong
to one document, or that is the sole document in the selected collections. The
candidate must include all conservatively deduplicated indexed text within the
default 64,000-character store limit and 96,000 serialized UTF-8 bytes. Approval
additionally requires the actual rendered prompt's tokenizer count, the requested answer-token reserve,
and a 1,024-token margin to fit both the model and engine capacity. Media-bearing history
skips expansion. Mixed-source results, unavailable fit checks and any failed bound
use the ordinary excerpt path; the host never calls a clipped document complete.

An approved complete-source payload closes retrieval and is replayed exactly during
synthesis or thinking recovery, preserving middle sections and citations. This is
an explicit exception to the ordinary evidence ceiling, not a larger autonomous
search budget. It represents coverage of that source's indexed text and a source
selection assumption; it does not establish complete extraction, library-wide
coverage, answer correctness or benchmarked latency gains. Normal mixed code work
and artifact completion guards remain available.

Without an accepted complete-source candidate, tool-capable requests begin with
two short excerpts (at most 4,000 UTF-8 bytes), then retrieve against concrete gaps.
Tool-free requests keep the larger six-excerpt window. Query guidance encourages
distinctive terms from the source text. Synthesis guidance asks the model to preserve
definitions, units, comparison criteria and relevant qualifiers, distinguish
observations from opinions and measured outcomes, and avoid unsupported broader
conclusions from isolated statements. It follows the user's requested format,
language and level of detail. These rules apply across knowledge domains;
lexical relevance alone does not establish that a generated answer is accurate.

Ordinary retrieval stops after 16 knowledge calls, three consecutive calls without
new evidence, or a cumulative per-conversation evidence budget (twice the response
budget, with a 4,096-byte floor and 24,000-byte ceiling), so a batch of individually
bounded searches cannot keep growing the prompt. Eager evidence, citation metadata and full retransmitted expansions count
toward it; compact duplicate notices do not. Novel text remains a separate progress
signal. This bounds model prefill work as well as retrieval calls. A last response
may cross the ceiling before the next call is suppressed. A model requesting
disabled retrieval tools proceeds to the terminal synthesis pass.

TensorAgent normally bounds knowledge-turn planning to at most 512 thinking tokens
within each generation's total token allowance; allowances below 512 tokens stay
unchanged. An explicit `TS_THINKING_BUDGET` overrides that default. Child agents
inherit the request's thinking-token limit while retaining private
transcripts and cache scopes. Delegation does not restore a larger default limit.
Muse-Glimmer uses message channels rather than `</think>`. When its trained transition tokens
are available, the sampler closes reasoning within the same request and KV state,
opening the answer channel for tool-free turns or preserving recipient selection
when tools remain available. At most three transitions are attempted. Unsupported
framing or continued reasoning retains the thinking-off fallback: bounded prior
evidence and one final pass without host tools (verified-artifact workflows retain
their guard). A tool-free, thinking-off prompt explicitly opens the user-answer
channel. This design does not by itself establish a latency or semantic-quality gain.
Its template reads `reasoning_strength`, so explicit
reasoning effort is mapped to that field. Budgeted Muse knowledge planning defaults
to medium effort only when none was supplied; explicit choices and other requests
retain their existing behavior. Host continuation messages retain the original user
question during context compaction. Code tools needed for an unfinished requested
computation remain available within the normal tool-loop limit.
General delegation and completed-retrieval guidance favor direct synthesis when
the needed evidence already fits in context. They discourage partitioning that
evidence solely to create subagents, while preserving explicit delegation requests
and substantial independent tool, source, verification or workspace work. This is
guidance rather than a hard ban or a measured performance guarantee.

[Dify's retrieval service](https://github.com/langgenius/dify/blob/main/api/core/rag/datasource/retrieval_service.py)
deduplicates retrieved material while preserving source identity. TensorSharp also
suppresses overlapping selectable-text/OCR candidates so they do not consume the
search result budget twice. Its lexical path retains names in mixed-language
queries and uses CJK word fragments instead of allowing common individual
characters to outweigh a specific name. This does not provide general cross-language
semantic retrieval without compatible embeddings.

Retrieval candidates and final-answer sources are separate sets. Only explicit
citations in the completed answer are bound to current, retrieved source identities;
localized bracket variants of an actually issued marker are recognized without
repairing unknown markers or external URL origins.
Unused candidates do not become source chips. A document URL cites the document,
while an exact chunk reference can retain its page/row location. This checks
provenance and citation identity, not whether every generated claim is entailed
by the source.

- **Identity:** source identity, content identity, chunk identity, and embedding
  identity are separate. A rename need not alter content; identical text in two
  user-authorized collections must retain independent ownership. A parser change
  can require reindexing even if source bytes did not change.
- **Scope:** selected collection IDs provide retrieval scope. Hosted multi-user
  deployments additionally need authenticated ownership/ACL checks; an arbitrary
  client-supplied user ID or collection ID is not authorization. Every read and
  mutation needs the same policy as search.
- **Bounds:** cap ingestion and response sizes, use cancellation, and batch
  embeddings. Progressive search/read reduces prompt usage, but it does not
  justify unbounded internal scans or returning every chunk.
- **Storage:** a portable local store and exact vector search are a reasonable
  measurable baseline. They do not imply million-document scalability. Keep store
  and provider boundaries suitable for an inverted index, ANN backend, or server
  implementation when corpus measurements justify one.
- **Compatibility:** vector dimensionality is necessary but insufficient.
  Identical dimensions from different models or preprocessing settings do not
  produce a shared semantic space. Incompatible vectors must not be silently
  compared, and reindexing must have an explicit outcome.
- **Multimodal evidence:** preserve modality and source position. Time ranges
  identify audio/video moments; page/slide/row ranges identify document locations.
  A generated caption is a derived interpretation and should not be presented as
  verbatim source text.

### Bounded query-aware reranking

Additional primary-source review accessed **2026-10-11**. Reranking should remain
a separate relevance-selection stage, without prescribing the user's answer format
or a subject-specific interpretation of the evidence. TensorSharp now implements
a bounded optional LLM pass informed by these patterns. The references explain
the design; they do not establish TensorSharp quality or latency gains.

- [LlamaIndex's LLMRerank implementation](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/postprocessor/llm_rerank.py)
  supplies the query with bounded candidate batches, parses selected candidate
  positions and scores, maps them back to original nodes, and returns a separate
  `top_n`. The applicable pattern is a bounded model judgment over already retrieved
  candidates. TensorSharp should validate every returned identifier against the
  offered set; a model must not create source identities or rewrite excerpts.
- [Dify's reranker](https://github.com/langgenius/dify/blob/main/api/core/rag/rerank/rerank_model.py)
  deduplicates candidates before scoring, retains original metadata/provider
  identity through returned indices, and applies score and result-count limits.
  TensorSharp should preserve document/version/location ownership through reranking
  and keep original retrieval scores distinct from model relevance scores. A
  relevance judgment does not establish that the final answer used that source.
- [Open WebUI's reranking configuration](https://docs.openwebui.com/reference/env-configuration/)
  separates retrieval and reranking result limits, exposes local scoring batch
  size, and supports an external-reranker timeout. TensorSharp should separately
  bound candidate count, serialized prompt size, model output and elapsed work;
  adding a reranker must not recreate an unbounded retrieval loop.

Integration safeguards here are TensorSharp design conclusions, not
claims that each upstream implements identical policies. Filter scope before
candidate generation, keep the original user query available, and perform ranking
before clipping to the final evidence budget. Reject unknown/duplicate identifiers
and malformed output; on timeout or invalid output, preserve the original candidate
order. Cancellation must propagate. A reranker using the loaded generation model
must run outside an already-held generation lease, without recursive tool calls or
conversation-history mutation. Otherwise it can deadlock or consume more prefill
time than it saves. An optional model stage cannot recover evidence absent from the
candidate pool, and its scores should not be treated as calibrated probabilities.

The implemented pass is shared across eager and tool retrieval and attempted at
most once per request. It offers at most eight candidates, caps its serialized
input at 12,000 UTF-8 bytes, preserves the full bounded search query, and reserves
96 output tokens after checking the active model's actual prompt capacity. An
approved complete-source snapshot bypasses the pass. Generation has no tools or
thinking, uses an isolated session through the shared scheduler, and observes a
10-second cancellation deadline followed by awaited cleanup. Only a complete
set of offered IDs is accepted, each paired with one query-relative integer
usefulness rating from 0 to 3. A private JSON grammar constrains response shape;
the host checks membership and uniqueness and sorts ratings with baseline ties.
Ratings are ordinal, not confidence probabilities, and never threshold-drop a
candidate. Malformed output and timeouts retain the baseline order. Excerpts
remain original source objects, versions are revalidated
after the model await, and final evidence serialization still enforces the user's
scope budget. The envelope identifies model ordering separately from original
retrieval scores. `TS_KNOWLEDGE_RERANK=0` opts out. Focused managed tests cover the
failure paths and integration; live outcome and latency evidence must be recorded
separately rather than counted as success when the model falls back.

Evaluate relevance and answer coverage on generic positive, negative, identifier,
multilingual and multi-document cases. Report retrieval and reranking latency
separately, including fallback rates; do not infer general gains from one document
or one completed answer. Domain-specific evaluation fixtures may measure factual
accuracy, but should not become runtime prompt rules.

## Validation required for implementation claims

| Area | Required evidence |
| --- | --- |
| End-to-end lifecycle | Create collection → upload → extract/index → search → read citation → reopen store → reindex → delete; verify unrelated content remains available. |
| Grounding | Known queries resolve to expected document and source location; returned citations/read results agree with stored source text. |
| Scope | Search, listing, direct reads, and mutation cannot escape the allowed collection set; empty selection semantics are explicit. |
| Multimodal ingestion | Separate coverage for text/structured documents, images, audio, and video. Fake extractors validate orchestration only; model-backed extraction requires actual media/model runs. |
| Vector compatibility | Reject wrong dimension, non-finite values, and mismatched model identity; cover lexical fallback and reindexing after a model change. |
| Resource limits | Reject oversized inputs; cover cancellation, malformed containers, decompression bounds, empty documents, interrupted persistence, and repeated updates. |
| Search quality | Report recall or expected-source hit rate on a fixed labeled corpus, including lexical identifiers, paraphrases, multilingual text, and negatives where supported. |
| Performance | Record corpus/chunk count, text volume, embedding dimensions/provider, hardware/runtime, warm-up, ingestion rate, search p50/p95, and memory. Separate extraction/model cost from index lookup cost. |

Benchmarks using generated text, supplied transcripts, or deterministic fake
embeddings measure the exercised storage/retrieval/orchestration path. They do not
establish OCR accuracy, speech recognition quality, visual/video understanding,
real-model semantic recall, generation quality, or mobile-device performance.
Unavailable model/device scenarios must be reported as unvalidated, not passing.
Graph construction, temporal reasoning, external connector synchronization, and
distributed indexing need their own implementation and evaluation before being
advertised.

Reusable validation programs belong in `eng/`; required fixtures belong in the
test project. Generated measurements and logs belong in ignored
`docs/validation/` or `artifacts/`. This managed knowledge feature does not require
ggml changes. Any subsequent native work must record the unchanged upstream
dependency revision and actual device/model coverage.
