using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using TensorSharp.Knowledge;
using TensorSharp.Server.Knowledge;

static string? Arg(string[] arguments, string flag)
{
    int index = Array.IndexOf(arguments, flag);
    return index >= 0 && index + 1 < arguments.Length ? arguments[index + 1] : null;
}

int documentCount = int.Parse(Arg(args, "--documents") ?? "1200");
int queryCount = int.Parse(Arg(args, "--queries") ?? "200");
int embeddingDocuments = int.Parse(Arg(args, "--embedding-documents") ?? "1200");
int embeddingQueries = int.Parse(Arg(args, "--embedding-queries") ?? "100");
if (documentCount < 1 || queryCount < 1 || embeddingDocuments < 1 || embeddingQueries < 1)
    throw new ArgumentException("Document and query counts must be positive.");
string outputPath = Path.GetFullPath(Arg(args, "--output") ?? "artifacts/knowledge/benchmark.json");
string? modelPath = Arg(args, "--embedding-model");
string backend = Arg(args, "--embedding-backend") ?? "CPU";
string queryPrefix = Arg(args, "--query-prefix") ?? "";
string documentPrefix = Arg(args, "--document-prefix") ?? "";
string temporary = Path.Combine(Path.GetTempPath(), "tensorsharp-knowledge-bench-" + Guid.NewGuid().ToString("N"));
Directory.CreateDirectory(temporary);
var jsonOptions = new JsonSerializerOptions { PropertyNameCaseInsensitive = true, WriteIndented = true };
var fixture = JsonSerializer.Deserialize<RetrievalFixture>(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "retrieval-fixture.json")), jsonOptions)!;
var process = Process.GetCurrentProcess();
var total = Stopwatch.StartNew();
try
{
    object lexicalScale = await ScaleAsync(Path.Combine(temporary, "scale"), documentCount, queryCount);
    object lexicalQuality = await QualityAsync(Path.Combine(temporary, "quality-lexical"), fixture, null);
    object embeddingQuality;
    object embeddingScale;
    if (modelPath is null)
    {
        embeddingQuality = new { status = "not_run", reason = "No --embedding-model supplied. Lexical results do not validate semantic or model quality." };
        embeddingScale = new { status = "not_run", reason = "No --embedding-model supplied; dense-scale performance was not measured." };
    }
    else
    {
        var load = Stopwatch.StartNew();
        var provider = new GgufKnowledgeEmbeddingProvider(Path.GetFullPath(modelPath), backend, queryPrefix, documentPrefix);
        double loadMs = load.Elapsed.TotalMilliseconds;
        string identity = provider.ModelId;
        int dimensions = provider.Dimensions;
        object quality = await QualityAsync(Path.Combine(temporary, "quality-hybrid"), fixture, provider);
        embeddingQuality = new { status = "completed", backend, modelPath = Path.GetFullPath(modelPath), identity, dimensions, loadMs, quality };
        embeddingScale = await EmbeddingScaleAsync(Path.Combine(temporary, "scale-hybrid"), fixture,
            embeddingDocuments, embeddingQueries,
            () => new GgufKnowledgeEmbeddingProvider(Path.GetFullPath(modelPath), backend, queryPrefix, documentPrefix));
    }
    process.Refresh();
    var report = new
    {
        schemaVersion = 2,
        createdAtUtc = DateTimeOffset.UtcNow,
        environment = new { framework = RuntimeInformation.FrameworkDescription, os = RuntimeInformation.OSDescription,
            architecture = RuntimeInformation.ProcessArchitecture.ToString(), processors = Environment.ProcessorCount,
            processPeakWorkingSetBytes = process.PeakWorkingSet64 > 0 ? (long?)process.PeakWorkingSet64 : null,
            processWorkingSetBytes = process.WorkingSet64 },
        methodology = new
        {
            lexical = "Managed BM25 inverted postings; durable per-document JSON commits include fsync; ingestion includes extraction and indexing.",
            dense = "Optional actual GGUF inference plus exact normalized cosine search and reciprocal rank fusion. Separate short-document dense-scale corpus and small hand-authored quality fixture. No ANN index.",
            quality = "14 manually authored queries, 12 documents; document-level recall@1, recall@5 and MRR. No generation or answer-quality validation.",
            latency = "Sequential in-process calls after warmup; retrieval excludes HTTP and UI. Fixture quality queries include query embedding when configured.",
            limitations = new[] { "Synthetic scale data is not a representative customer corpus.", "No comparisons against other RAG systems or ANN engines.",
                "No concurrent-client, multi-process, GPU, real image/audio/video understanding or generated-answer benchmark here.",
                "Per-file fsync does not claim database-grade power-loss recovery for directory metadata on all filesystems.",
                "Results are environment-specific measurements, not performance guarantees." },
        },
        lexicalScale, lexicalQuality, embeddingQuality, embeddingScale,
        totalSeconds = total.Elapsed.TotalSeconds,
    };
    Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
    await File.WriteAllTextAsync(outputPath, JsonSerializer.Serialize(report, jsonOptions));
    Console.WriteLine(JsonSerializer.Serialize(new { output = outputPath, totalSeconds = total.Elapsed.TotalSeconds,
        workingSetMiB = process.WorkingSet64 / 1048576.0,
        peakWorkingSetMiB = process.PeakWorkingSet64 > 0 ? (double?)(process.PeakWorkingSet64 / 1048576.0) : null,
        embedding = modelPath is null ? "not_run" : "completed" }));
}
finally { Directory.Delete(temporary, true); }

static async Task<object> ScaleAsync(string root, int documents, int queries)
{
    var options = new KnowledgeOptions { ChunkCharacters = 256, ChunkOverlapCharacters = 32, MaxDocuments = Math.Max(10000, documents + 10), MaxTotalChunks = Math.Max(100000, documents * 40) };
    var latencies = new List<double>();
    string collectionId;
    string isolatedId;
    string deletedDocument;
    int chunks;
    double ingestSeconds;
    long bytes = 0;
    var stopwatch = Stopwatch.StartNew();
    using (var store = new KnowledgeStore(root, new BuiltInKnowledgeExtractor(options), options: options))
    {
        collectionId = (await store.CreateCollectionAsync("Synthetic scale corpus")).Id;
        isolatedId = (await store.CreateCollectionAsync("Isolation canary")).Id;
        for (int i = 0; i < documents; i++)
        {
            // Twelve independently traceable source sections per document; repeated words
            // deliberately exercise large postings as well as selective identifier queries.
            var text = new StringBuilder();
            for (int section = 0; section < 12; section++)
                text.Append($"## Section {section}\nInstrument instrument{i:D6} uses calibration spectrum band{section}. The maintenance log records pressure, temperature, drift and optical alignment. Review the sensor housing before collecting a new reference exposure. Record ticket ticket{i:D6} after each adjustment.\n");
            string source = text.ToString(); bytes += Encoding.UTF8.GetByteCount(source);
            await Import(store, collectionId, $"instrument-{i:D6}.md", source);
        }
        await Import(store, isolatedId, "canary.txt", "Private isolatedcanaryvalue99271");
        ingestSeconds = stopwatch.Elapsed.TotalSeconds;
        chunks = store.ListDocuments(collectionId).Sum(d => d.ChunkCount);
        deletedDocument = store.ListDocuments(collectionId)[0].Id;
        if ((await Search(store, collectionId, "isolatedcanaryvalue99271")).Count != 0) throw new Exception("Collection isolation failed.");
        for (int warmup = 0; warmup < 20; warmup++) await Search(store, collectionId, "calibration spectrum");
        for (int i = 0; i < queries; i++)
        {
            string query = i % 3 == 0 ? "calibration spectrum optical" : i % 3 == 1 ? $"instrument{i % documents:D6}" : $"ticket{i % documents:D6}";
            stopwatch.Restart(); var result = await Search(store, collectionId, query);
            latencies.Add(stopwatch.Elapsed.TotalMilliseconds);
            if (result.Count == 0) throw new Exception("A synthetic retrieval returned no matches.");
            if (i % 3 != 0 && result[0].FileName != $"instrument-{i % documents:D6}.md") throw new Exception("Exact identifier retrieval failed.");
        }
        if (!await store.DeleteDocumentAsync(collectionId, deletedDocument)) throw new Exception("Deletion failed.");
        if ((await Search(store, collectionId, "instrument000000")).Count != 0) throw new Exception("Deleted content is still searchable.");
    }
    stopwatch.Restart();
    using var reopened = new KnowledgeStore(root, new BuiltInKnowledgeExtractor(options), options: options);
    double reloadMs = stopwatch.Elapsed.TotalMilliseconds;
    if (reopened.ListDocuments(collectionId).Count != documents - 1 || (await Search(reopened, collectionId, "instrument000000")).Count != 0)
        throw new Exception("Restart durability/deletion failed.");
    return new
    {
        mode = "lexical", documents, chunks, sourceBytes = bytes,
        chunkCharacters = options.ChunkCharacters, overlapCharacters = options.ChunkOverlapCharacters,
        ingestSeconds, documentsPerSecond = documents / ingestSeconds, chunksPerSecond = chunks / ingestSeconds,
        reloadMs, retrieval = Latency(latencies), correctness = new { exactIdentifierRanking = true, collectionIsolation = true, deletion = true, restart = true },
        scaleTargetMet = documents >= 1000 && chunks >= 10000,
    };
}

static async Task<object> QualityAsync(string root, RetrievalFixture fixture, IKnowledgeEmbeddingProvider? provider)
{
    var stopwatch = Stopwatch.StartNew();
    using var store = new KnowledgeStore(root, new BuiltInKnowledgeExtractor(), provider);
    var collection = await store.CreateCollectionAsync("Retrieval quality fixture");
    foreach (var document in fixture.Documents) await Import(store, collection.Id, document.Name, document.Text);
    double ingestSeconds = stopwatch.Elapsed.TotalSeconds;
    var results = new List<QueryResult>();
    // A distinct warmup avoids reporting the cold JIT path as steady-state retrieval.
    await Search(store, collection.Id, "verification unrelated warmup");
    foreach (var query in fixture.Queries)
    {
        stopwatch.Restart();
        var hits = await store.SearchAsync(query.Query, new() { CollectionIds = [collection.Id], Limit = 10, MaxCharacters = 12000 });
        double elapsed = stopwatch.Elapsed.TotalMilliseconds;
        string[] ranked = hits.Select(h => h.FileName).Distinct(StringComparer.Ordinal).Take(5).ToArray();
        int index = Array.IndexOf(ranked, query.Expected);
        results.Add(new(query.Kind, query.Query, query.Expected, ranked, index < 0 ? 0 : index + 1, elapsed));
    }
    return new
    {
        mode = provider is null ? "lexical" : "hybrid", fixture.Description,
        documents = fixture.Documents.Length, chunks = store.ListDocuments(collection.Id).Sum(d => d.ChunkCount),
        ingestSeconds, retrieval = Latency(results.Select(r => r.Milliseconds).ToList()),
        overall = Metrics(results), byKind = results.GroupBy(r => r.Kind).ToDictionary(g => g.Key, g => Metrics(g.ToList())),
        queries = results,
    };
}

static async Task<object> EmbeddingScaleAsync(string root, RetrievalFixture fixture, int documents, int queries,
    Func<IKnowledgeEmbeddingProvider> createProvider)
{
    var options = new KnowledgeOptions { MaxDocuments = Math.Max(10000, documents + 10), MaxTotalChunks = Math.Max(100000, documents * 4) };
    var stopwatch = Stopwatch.StartNew();
    var provider = createProvider();
    double modelLoadMs = stopwatch.Elapsed.TotalMilliseconds;
    string identity = provider.ModelId;
    int dimensions = provider.Dimensions;
    string collectionId;
    string deletedDocument;
    int chunks;
    double ingestSeconds;
    long sourceBytes = 0;
    var samples = new List<(string Kind, double Milliseconds)>();
    using (var store = new KnowledgeStore(root, new BuiltInKnowledgeExtractor(options), provider, options))
    {
        collectionId = (await store.CreateCollectionAsync("Actual embedding scale corpus")).Id;
        string isolatedId = (await store.CreateCollectionAsync("Separate scope")).Id;
        stopwatch.Restart();
        for (int i = 0; i < documents; i++)
        {
            // Actual encoder inference runs for every unique document; no fake vectors,
            // duplicate-hash shortcut or persisted cache is used to populate the corpus.
            var template = fixture.Documents[i % fixture.Documents.Length];
            string text = $"Record item{i:D6}. Region zone{i % 37:D3}. Revision {i % 19}. {template.Text}";
            sourceBytes += Encoding.UTF8.GetByteCount(text);
            await Import(store, collectionId, $"record-{i:D6}.txt", text);
            if ((i + 1) % 200 == 0) Console.WriteLine($"Dense scale: embedded and committed {i + 1}/{documents} documents.");
        }
        ingestSeconds = stopwatch.Elapsed.TotalSeconds;
        await Import(store, isolatedId, "scope-canary.txt", "Private isolatedcanaryvalue99271. This document belongs to a different collection.");
        var stored = store.ListDocuments(collectionId);
        chunks = stored.Sum(d => d.ChunkCount);
        if (stored.Any(d => d.EmbeddingModelId != identity)) throw new Exception("Dense-scale document lacks the expected real-model embedding identity.");
        deletedDocument = stored.First(d => d.FileName == "record-000000.txt").Id;
        var scopeResults = await Search(store, collectionId, "isolatedcanaryvalue99271");
        if (scopeResults.Any(h => h.CollectionId != collectionId || h.FileName == "scope-canary.txt")) throw new Exception("Dense collection isolation failed.");
        for (int warmup = 0; warmup < 20; warmup++)
            await Search(store, collectionId, fixture.Queries[warmup % fixture.Queries.Length].Query);
        var paraphrases = fixture.Queries.Where(q => q.Kind == "paraphrase").ToArray();
        var cjk = fixture.Queries.Where(q => q.Kind == "cjk").ToArray();
        for (int i = 0; i < queries; i++)
        {
            string kind = (i % 4) switch { 0 => "exact_identifier", 1 => "paraphrase", 2 => "cjk", _ => "broad_terms" };
            string query = (i % 4) switch
            {
                0 => $"item{i % documents:D6}",
                1 => paraphrases[(i / 4) % paraphrases.Length].Query,
                2 => cjk[(i / 4) % cjk.Length].Query,
                _ => "maintenance procedure storage record",
            };
            stopwatch.Restart();
            var result = await Search(store, collectionId, query);
            samples.Add((kind, stopwatch.Elapsed.TotalMilliseconds));
            if (result.Count == 0 || result.Any(h => h.CollectionId != collectionId)) throw new Exception("Dense-scale query produced empty or out-of-scope results.");
            if (kind == "exact_identifier" && result[0].FileName != $"record-{i % documents:D6}.txt")
                throw new Exception("Dense-scale exact identifier retrieval did not preserve its lexical anchor.");
        }
        if (!await store.DeleteDocumentAsync(collectionId, deletedDocument)) throw new Exception("Dense-scale deletion failed.");
        if ((await Search(store, collectionId, "item000000")).Any(h => h.DocumentId == deletedDocument)) throw new Exception("Dense-scale deletion left a retrievable source.");
    }
    stopwatch.Restart();
    var reloadProvider = createProvider();
    double modelReloadMs = stopwatch.Elapsed.TotalMilliseconds;
    stopwatch.Restart();
    using var reopened = new KnowledgeStore(root, new BuiltInKnowledgeExtractor(options), reloadProvider, options);
    double storeReloadMs = stopwatch.Elapsed.TotalMilliseconds;
    if (reopened.ListDocuments(collectionId).Count != documents - 1
        || (await Search(reopened, collectionId, "item000000")).Any(h => h.DocumentId == deletedDocument))
        throw new Exception("Dense-scale restart durability/deletion failed.");
    if (documents > 1 && (await Search(reopened, collectionId, "item000001"))[0].FileName != "record-000001.txt")
        throw new Exception("Persisted embeddings and index failed retrieval after restart.");
    return new
    {
        status = "completed", mode = "hybrid_exact_cosine", identity, dimensions,
        corpus = "Unique short records generated from 12 topical English/Chinese text templates; actual encoder inference and durable commit for every record. This scale corpus is not the 14-query quality evaluation.",
        documents, chunks, sourceBytes, chunkCharacters = options.ChunkCharacters, overlapCharacters = options.ChunkOverlapCharacters,
        modelLoadMs, ingestSeconds, documentsPerSecond = documents / ingestSeconds, chunksPerSecond = chunks / ingestSeconds,
        warmupQueries = 20, retrieval = Latency(samples.Select(s => s.Milliseconds).ToList()),
        retrievalByKind = samples.GroupBy(s => s.Kind).ToDictionary(g => g.Key, g => Latency(g.Select(s => s.Milliseconds).ToList())),
        modelReloadMs, storeReloadMs,
        correctness = new { actualModelIdentity = true, exactIdentifierRanking = true, collectionIsolation = true, deletion = true, restart = true },
        scaleTargetMet = documents >= 1200 && chunks >= 1200,
    };
}

static object Metrics(IReadOnlyList<QueryResult> values) => new
{
    count = values.Count, recallAt1 = values.Count(v => v.Rank == 1) / (double)values.Count,
    recallAt5 = values.Count(v => v.Rank > 0) / (double)values.Count,
    mrr = values.Average(v => v.Rank == 0 ? 0 : 1.0 / v.Rank),
};
static object Latency(List<double> values)
{
    double[] sorted = values.Order().ToArray();
    return new { count = values.Count, p50Ms = sorted[(int)Math.Ceiling(sorted.Length * .50) - 1], p95Ms = sorted[(int)Math.Ceiling(sorted.Length * .95) - 1],
        meanMs = values.Average(), queriesPerSecond = values.Count * 1000 / values.Sum(), samplesMs = values };
}
static async Task Import(KnowledgeStore store, string collection, string name, string text)
{
    using var stream = new MemoryStream(Encoding.UTF8.GetBytes(text));
    await store.ImportAsync(collection, name, stream, "text/plain");
}
static Task<IReadOnlyList<KnowledgeSearchResult>> Search(KnowledgeStore store, string collection, string query) =>
    store.SearchAsync(query, new() { CollectionIds = [collection], Limit = 8, MaxCharacters = 8000 });

sealed record RetrievalFixture(string Description, FixtureDocument[] Documents, FixtureQuery[] Queries);
sealed record FixtureDocument(string Name, string Text);
sealed record FixtureQuery(string Kind, string Query, string Expected);
sealed record QueryResult(string Kind, string Query, string Expected, string[] Retrieved, int Rank, double Milliseconds);
