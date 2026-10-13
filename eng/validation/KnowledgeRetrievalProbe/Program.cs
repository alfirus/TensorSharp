using System.Diagnostics;
using System.Text.Json;
using TensorSharp.Knowledge;

// Run against a disposable copy: KnowledgeStore is a writer and can perform
// recovery/garbage collection during construction. Never open the user's store.
if (args.Length < 4)
    throw new ArgumentException("Usage: KnowledgeRetrievalProbe <store> <collection name> <query> <report.json> [iterations=200] [limit=8]");
string source = Path.GetFullPath(args[0]);
string destination = Path.GetFullPath(args[3]);
if (destination.StartsWith(Path.TrimEndingDirectorySeparator(source) + Path.DirectorySeparatorChar, StringComparison.Ordinal))
    throw new ArgumentException("The report must be written outside the source store.");
int iterations = args.Length > 4 ? int.Parse(args[4]) : 200;
int limit = args.Length > 5 ? int.Parse(args[5]) : 8;
if (iterations is < 1 or > 10000) throw new ArgumentOutOfRangeException(nameof(iterations));
string snapshot = Path.Combine(Path.GetTempPath(), "knowledge-retrieval-probe-" + Guid.NewGuid().ToString("N"));
try
{
    foreach (string directory in new[] { "collections", "documents", "objects" })
    {
        string copy = Path.Combine(snapshot, directory);
        Directory.CreateDirectory(copy);
        foreach (string file in Directory.EnumerateFiles(Path.Combine(source, directory)))
            File.Copy(file, Path.Combine(copy, Path.GetFileName(file)));
    }
    using var store = new KnowledgeStore(snapshot, new NoExtraction());
    var collection = store.ListCollections().Single(collection => collection.Name == args[1]);
    var options = new KnowledgeSearchOptions { CollectionIds = [collection.Id], Limit = limit, MaxCharacters = 12000 };
    for (int i = 0; i < 20; i++) await store.SearchAsync(args[2], options);
    var times = new List<double>();
    IReadOnlyList<KnowledgeSearchResult> hits = [];
    for (int i = 0; i < iterations; i++)
    {
        long started = Stopwatch.GetTimestamp();
        hits = await store.SearchAsync(args[2], options);
        times.Add(Stopwatch.GetElapsedTime(started).TotalMilliseconds);
    }
    var sorted = times.Order().ToArray();
    // No source text, names, identifiers or query are persisted in the report.
    var documents = store.ListDocuments(collection.Id);
    var report = new
    {
        mode = "lexical", sourceUntouched = true, documents = documents.Count,
        chunks = documents.Sum(document => document.ChunkCount), iterations, limit,
        p50Ms = sorted[(int)Math.Ceiling(iterations * .50) - 1], p95Ms = sorted[(int)Math.Ceiling(iterations * .95) - 1],
        hits = hits.Select(hit => new { documentOrdinal = documents.ToList().FindIndex(document => document.Id == hit.DocumentId),
            hit.ChunkIndex, page = hit.Location?.Page, hit.Modality, characters = hit.Text.Length, hit.Truncated }),
        limitations = "Read-only snapshot; lexical retrieval only. No extraction, embeddings, model inference, generated-answer quality, HTTP or UI timing. Concurrent changes in the source store are not an atomic snapshot.",
    };
    Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
    string json = JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true });
    await File.WriteAllTextAsync(destination, json);
    Console.WriteLine(json);
}
finally { if (Directory.Exists(snapshot)) Directory.Delete(snapshot, true); }

sealed class NoExtraction : IKnowledgeExtractor
{
    public Task<KnowledgeExtraction> ExtractAsync(KnowledgeExtractionContext context, CancellationToken cancellationToken = default) =>
        throw new InvalidOperationException("The snapshot probe must not extract or reindex sources.");
}
