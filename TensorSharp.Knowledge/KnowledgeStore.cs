using System.Globalization;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace TensorSharp.Knowledge;

/// <summary>
/// Single-writer local knowledge store. Each document is its own atomic commit; the
/// inverted index is rebuilt from durable manifests at startup. Originals are immutable
/// content-addressed blobs. Collections are scopes, not authentication credentials.
/// </summary>
public sealed class KnowledgeStore : IDisposable
{
    private const int SchemaVersion = 1;
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = false };
    // Only query-side function words are ignored, and only when the query also has
    // content terms. Thus an exact all-function-word phrase remains searchable.
    private static readonly HashSet<string> QueryFunctionWords = new(
        "a an and are as at be been being but by can could did do does doing for from had has have how i if in into is it its may me might my of on or our ours should so than that the their them then there these they this those to too us was we were what when where which who why will with would you your 的 了 和 是 在 有 与 及 哪个 什么 如何 怎么 其中 这个 那个 一个 以及 还有 是否 我们 你们 他们 它们 请问 比较".Split(' '), StringComparer.Ordinal);
    private static readonly HashSet<string> ImageQueryTerms = new(
        "image images picture pictures photo photos photograph photographs poster posters banner banners illustration illustrations screenshot screenshots artwork".Split(' '), StringComparer.Ordinal);
    private static readonly HashSet<string> AudioQueryTerms = new(
        "audio sound sounds podcast podcasts music song songs".Split(' '), StringComparer.Ordinal);
    private static readonly HashSet<string> VideoQueryTerms = new(
        "video videos movie movies footage".Split(' '), StringComparer.Ordinal);
    private static readonly HashSet<string> QueryRequestWords = new(
        "please find search looking look locate show fetch retrieve".Split(' '), StringComparer.Ordinal);
    private static readonly HashSet<string> MediaConceptTerms = new(
        "encode encoding encoder encoders decode decoding decoder decoders generation generator generators model models api implementation documentation docs guide manual tutorial processing inference".Split(' '), StringComparer.Ordinal);
    private static readonly HashSet<string> QueryQuestionWords = new(
        "how why what explain describe support supports use uses work works".Split(' '), StringComparer.Ordinal);
    private readonly string _root;
    private readonly IKnowledgeExtractor _extractor;
    private readonly IKnowledgeEmbeddingProvider? _embeddings;
    private readonly string? _embeddingModelId;
    private readonly int _embeddingDimensions;
    private readonly KnowledgeOptions _options;
    private readonly FileStream _writerLock;
    private readonly SemaphoreSlim _mutations = new(1, 1);
    private readonly object _sync = new();
    private readonly Dictionary<string, KnowledgeCollection> _collections = new(StringComparer.Ordinal);
    private readonly Dictionary<string, StoredDocument> _documents = new(StringComparer.Ordinal);
    private readonly Dictionary<(string Collection, string Name), StoredDocument> _documentsByName = new();
    private readonly Dictionary<string, HashSet<string>> _collectionChunks = new(StringComparer.Ordinal);
    private readonly Dictionary<string, long> _collectionTokenCounts = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _objectReferences = new(StringComparer.Ordinal);
    private readonly Dictionary<string, IndexedChunk> _chunks = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Dictionary<string, int>> _postings = new(StringComparer.Ordinal);
    private bool _disposed;
    private int _activeSearches;
    private long _storedBytes;

    public KnowledgeOptions Options => _options;
    public bool IsEmbeddingEnabled => _embeddings is not null;
    public string? EmbeddingModelId => _embeddingModelId;

    public KnowledgeStore(string root, IKnowledgeExtractor extractor,
        IKnowledgeEmbeddingProvider? embeddings = null, KnowledgeOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(root);
        _extractor = extractor ?? throw new ArgumentNullException(nameof(extractor));
        _embeddings = embeddings;
        _options = options ?? new();
        ValidateOptions(_options);
        if (embeddings is not null && (string.IsNullOrWhiteSpace(embeddings.ModelId) || embeddings.ModelId.Length > 1000 || embeddings.Dimensions is <= 0 or > 65536))
            throw new ArgumentException("Embedding providers need a stable model identity and 1–65536 dimensions.", nameof(embeddings));
        _embeddingModelId = embeddings?.ModelId;
        _embeddingDimensions = embeddings?.Dimensions ?? 0;
        _root = Path.GetFullPath(root);
        Directory.CreateDirectory(_root);
        RejectLink(_root);
        string lockPath = Path.Combine(_root, "writer.lock");
        if (File.Exists(lockPath)) RejectLink(lockPath);
        _writerLock = new FileStream(lockPath, FileMode.OpenOrCreate,
            FileAccess.ReadWrite, FileShare.None);
        try
        {
            foreach (string directory in new[] { "collections", "documents", "objects", "staging" })
            {
                string path = Path.Combine(_root, directory);
                Directory.CreateDirectory(path);
                RejectLink(path);
            }
            Load();
            // Only this process can write here. Abandoned staged writes were never committed.
            foreach (string path in Directory.EnumerateFiles(Path.Combine(_root, "staging"))) File.Delete(path);
            CollectUnreferencedObjects();
        }
        catch { _writerLock.Dispose(); throw; }
    }

    public async Task<KnowledgeCollection> CreateCollectionAsync(string name, string? description = null,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(name) || name.Length > 200) throw new ArgumentException("Collection name must contain 1–200 characters.", nameof(name));
        if (description?.Length > 4000) throw new ArgumentException("Collection description exceeds 4000 characters.", nameof(description));
        await _mutations.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ThrowIfDisposed();
            lock (_sync)
                if (_collections.Count >= _options.MaxCollections) throw new KnowledgeLimitException("Collection limit reached.");
            var collection = new KnowledgeCollection(Guid.NewGuid().ToString("N"), name.Trim(), description, DateTimeOffset.UtcNow);
            await WriteJsonAtomicAsync(CollectionPath(collection.Id), collection, cancellationToken).ConfigureAwait(false);
            lock (_sync) _collections.Add(collection.Id, collection);
            return collection;
        }
        finally { _mutations.Release(); }
    }

    public IReadOnlyList<KnowledgeCollection> ListCollections()
    {
        lock (_sync) { ThrowIfDisposed(); return _collections.Values.OrderBy(c => c.CreatedAt).ToArray(); }
    }

    public async Task<KnowledgeCollection> RenameCollectionAsync(string collectionId, string name,
        CancellationToken cancellationToken = default)
    {
        ValidateId(collectionId, nameof(collectionId));
        if (string.IsNullOrWhiteSpace(name) || name.Length > 200)
            throw new ArgumentException("Collection name must contain 1–200 characters.", nameof(name));
        await _mutations.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            KnowledgeCollection renamed;
            lock (_sync)
            {
                ThrowIfDisposed(); RequireCollection(collectionId);
                renamed = _collections[collectionId] with { Name = name.Trim() };
            }
            await WriteJsonAtomicAsync(CollectionPath(collectionId), renamed, cancellationToken).ConfigureAwait(false);
            lock (_sync) _collections[collectionId] = renamed;
            return renamed;
        }
        finally { _mutations.Release(); }
    }

    public IReadOnlyList<KnowledgeDocument> ListDocuments(string collectionId)
    {
        ValidateId(collectionId, nameof(collectionId));
        lock (_sync)
        {
            ThrowIfDisposed(); RequireCollection(collectionId);
            return _documents.Values.Where(d => d.Document.CollectionId == collectionId)
                .OrderBy(d => DocumentName(d.Document), StringComparer.Ordinal).Select(d => PublicDocument(d.Document)).ToArray();
        }
    }

    public Task<KnowledgeDocument> ImportAsync(string collectionId, string fileName, Stream content,
        string? contentType = null, CancellationToken cancellationToken = default)
        => ImportAsync(collectionId, fileName, content, contentType, relativePath: null, cancellationToken);

    public async Task<KnowledgeDocument> ImportAsync(string collectionId, string fileName, Stream content,
        string? contentType, string? relativePath, CancellationToken cancellationToken = default)
    {
        ValidateId(collectionId, nameof(collectionId));
        ValidateFileName(fileName);
        ValidateRelativePath(relativePath, fileName);
        ArgumentNullException.ThrowIfNull(content);
        if (contentType?.Length > 200) throw new ArgumentException("Content type exceeds 200 characters.", nameof(contentType));
        await _mutations.WaitAsync(cancellationToken).ConfigureAwait(false);
        string? staged = null;
        string? uploadedHash = null;
        try
        {
            ThrowIfDisposed();
            StoredDocument? previous;
            lock (_sync)
            {
                RequireCollection(collectionId);
                _documentsByName.TryGetValue((collectionId, relativePath ?? fileName), out previous);
                if (previous is null && _documents.Count >= _options.MaxDocuments) throw new KnowledgeLimitException("Document limit reached.");
            }
            staged = Path.Combine(_root, "staging", Guid.NewGuid().ToString("N") + ".upload");
            (long length, string hash) = await CopyBoundedAsync(content, staged, cancellationToken).ConfigureAwait(false);
            uploadedHash = hash;
            if (previous?.Document.ContentHash == hash) return PublicDocument(previous.Document);
            lock (_sync)
                if (_storedBytes - (previous?.Document.SizeBytes ?? 0) + length > _options.MaxStoredBytes)
                    throw new KnowledgeLimitException("Knowledge storage quota exceeded.");
            var extraction = await _extractor.ExtractAsync(new(staged, fileName, contentType), cancellationToken).ConfigureAwait(false);
            var stored = await PrepareAsync(collectionId, fileName, contentType, relativePath, hash, length, previous, extraction, cancellationToken).ConfigureAwait(false);
            string blob = ObjectPath(hash);
            if (!File.Exists(blob)) File.Move(staged, blob);
            else
            {
                RejectLink(blob);
                if (new FileInfo(blob).Length != length) throw new InvalidDataException("An existing content-addressed object has an invalid size.");
            }
            await CommitAsync(stored, previous, cancellationToken).ConfigureAwait(false);
            return PublicDocument(stored.Document);
        }
        finally
        {
            if (staged is not null) TryDelete(staged);
            if (uploadedHash is not null) lock (_sync) DeleteUnusedObject(uploadedHash);
            _mutations.Release();
        }
    }

    /// <summary>Re-extract and re-embed the current original; advances the version and invalidates old citations.</summary>
    public async Task<KnowledgeDocument> ReindexAsync(string collectionId, string documentId, CancellationToken cancellationToken = default)
    {
        ValidateId(collectionId, nameof(collectionId)); ValidateId(documentId, nameof(documentId));
        await _mutations.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ThrowIfDisposed();
            StoredDocument previous;
            lock (_sync) previous = RequireDocument(collectionId, documentId);
            KnowledgeDocument doc = previous.Document;
            var extraction = await _extractor.ExtractAsync(new(ObjectPath(doc.ContentHash), doc.FileName, doc.ContentType), cancellationToken).ConfigureAwait(false);
            var stored = await PrepareAsync(collectionId, doc.FileName, doc.ContentType, doc.RelativePath, doc.ContentHash, doc.SizeBytes, previous, extraction, cancellationToken).ConfigureAwait(false);
            await CommitAsync(stored, previous, cancellationToken).ConfigureAwait(false);
            return PublicDocument(stored.Document);
        }
        finally { _mutations.Release(); }
    }

    public async Task<bool> DeleteDocumentAsync(string collectionId, string documentId, CancellationToken cancellationToken = default)
    {
        ValidateId(collectionId, nameof(collectionId)); ValidateId(documentId, nameof(documentId));
        await _mutations.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ThrowIfDisposed(); cancellationToken.ThrowIfCancellationRequested();
            lock (_sync)
            {
                RequireCollection(collectionId);
                if (!_documents.TryGetValue(documentId, out var stored) || stored.Document.CollectionId != collectionId) return false;
                // The disk delete is the commit. Search cannot observe a half-removed document.
                File.Delete(DocumentPath(documentId));
                RemoveIndex(stored); _documents.Remove(documentId);
                DeleteUnusedObject(stored.Document.ContentHash);
                return true;
            }
        }
        finally { _mutations.Release(); }
    }

    public async Task<bool> DeleteCollectionAsync(string collectionId, CancellationToken cancellationToken = default)
    {
        ValidateId(collectionId, nameof(collectionId));
        await _mutations.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ThrowIfDisposed(); cancellationToken.ThrowIfCancellationRequested();
            lock (_sync)
            {
                if (!_collections.ContainsKey(collectionId)) return false;
                // The collection manifest is the commit point; startup discards documents
                // without a collection after an interruption during the following cleanup.
                File.Delete(CollectionPath(collectionId));
                _collections.Remove(collectionId);
                foreach (var stored in _documents.Values.Where(d => d.Document.CollectionId == collectionId).ToArray())
                {
                    RemoveIndex(stored); _documents.Remove(stored.Document.Id);
                    TryDelete(DocumentPath(stored.Document.Id));
                    DeleteUnusedObject(stored.Document.ContentHash);
                }
                return true;
            }
        }
        finally { _mutations.Release(); }
    }

    /// <summary>BM25 inverted-index retrieval, optionally fused with exact cosine ranks using reciprocal rank fusion.
    /// Result scores include a modest repeated-page penalty for ambiguous broad queries; they are ranking scores, not probabilities.</summary>
    public async Task<IReadOnlyList<KnowledgeSearchResult>> SearchAsync(string query, KnowledgeSearchOptions options,
        CancellationToken cancellationToken = default)
    {
        lock (_sync) { ThrowIfDisposed(); _activeSearches++; }
        try { return await SearchCoreAsync(query, options, cancellationToken).ConfigureAwait(false); }
        finally { lock (_sync) { _activeSearches--; Monitor.PulseAll(_sync); } }
    }

    private async Task<IReadOnlyList<KnowledgeSearchResult>> SearchCoreAsync(string query, KnowledgeSearchOptions options,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(options);
        cancellationToken.ThrowIfCancellationRequested();
        if (string.IsNullOrWhiteSpace(query)) return Array.Empty<KnowledgeSearchResult>();
        if (query.Length > _options.MaxQueryCharacters) throw new KnowledgeLimitException("Search query is too long.");
        ValidateSearchBounds(options.Limit, options.MaxCharacters);
        var scope = Scope(options.CollectionIds);
        cancellationToken.ThrowIfCancellationRequested();
        Dictionary<string, double> lexical;
        var denseScores = new Dictionary<string, double>(StringComparer.Ordinal);
        IndexedChunk[] denseCandidates;
        Dictionary<string, IndexedChunk> candidates;
        var anchoredCandidates = new HashSet<string>(StringComparer.Ordinal);
        var preferredMediaCandidates = new HashSet<string>(StringComparer.Ordinal);
        HashSet<string> coverageDocuments;
        string? requestedMedia;
        lock (_sync)
        {
            ThrowIfDisposed();
            foreach (string id in scope) RequireCollection(id);
            int scopedCount = scope.Sum(id => _collectionChunks.TryGetValue(id, out var keys) ? keys.Count : 0);
            if (scopedCount == 0) return Array.Empty<KnowledgeSearchResult>();
            double averageLength = Math.Max(1, scope.Sum(id => _collectionTokenCounts.GetValueOrDefault(id)) / (double)scopedCount);
            lexical = new(StringComparer.Ordinal);
            candidates = new(StringComparer.Ordinal);
            var queryTerms = QueryTerms(query);
            requestedMedia = RequestedMedia(query, queryTerms);
            var contentTerms = queryTerms.Where(term => !QueryFunctionWords.Contains(term)).ToArray();
            var terms = contentTerms.Length > 0 ? contentTerms : queryTerms;
            // In mixed-script questions, a source-language name or identifier is
            // a stronger anchor than coincidental CJK wording in unrelated sources.
            var anchors = queryTerms.Any(IsCjkTerm) || requestedMedia is not null
                ? terms.Where(term => !IsCjkTerm(term) && (requestedMedia is null
                    || !ImageQueryTerms.Contains(term) && !AudioQueryTerms.Contains(term)
                        && !VideoQueryTerms.Contains(term) && !QueryRequestWords.Contains(term)))
                    .ToHashSet(StringComparer.Ordinal) : [];
            var matchedDocumentTerms = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
            foreach (string term in terms)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (!_postings.TryGetValue(term, out var posting)) continue;
                var matches = posting.Where(p => scope.Contains(_chunks[p.Key].Document.CollectionId)).ToArray();
                double idf = Math.Log(1 + (scopedCount - matches.Length + .5) / (matches.Length + .5));
                foreach (var match in matches)
                {
                    IndexedChunk chunk = _chunks[match.Key];
                    double score = idf * match.Value * 2.2 / (match.Value + 1.2 * (.25 + .75 * chunk.Length / averageLength));
                    lexical[match.Key] = lexical.GetValueOrDefault(match.Key) + score;
                    candidates[match.Key] = chunk;
                    if (anchors.Contains(term)) anchoredCandidates.Add(match.Key);
                    if (!matchedDocumentTerms.TryGetValue(chunk.Document.Id, out var documentTerms))
                        matchedDocumentTerms[chunk.Document.Id] = documentTerms = new(StringComparer.Ordinal);
                    documentTerms.Add(term);
                }
            }
            // A named image can have only one occurrence in a long generated
            // caption while hundreds of short source-code chunks repeat the same
            // name. Explicit asset requests prefer matching originals of that
            // media type, without treating PDF OCR or unrelated media as matches.
            if (requestedMedia is not null)
                foreach (var candidate in candidates)
                    if (MatchesMedia(candidate.Value.Document, requestedMedia)
                        && (anchors.Count == 0 || anchoredCandidates.Contains(candidate.Key)))
                        preferredMediaCandidates.Add(candidate.Key);
            // A focused one-term lookup keeps BM25 relevance: a later page may
            // explain the topic while earlier pages only mention it in passing.
            // Coverage is useful when a broader question reduces to one indexed
            // term, or its only source-language matches are mixed-script anchors.
            coverageDocuments = matchedDocumentTerms.Where(pair => terms.Length > 1 && pair.Value.Count == 1
                || anchors.Count > 0 && pair.Value.All(anchors.Contains)).Select(pair => pair.Key).ToHashSet(StringComparer.Ordinal);
            denseCandidates = _embeddings is null ? Array.Empty<IndexedChunk>() : scope
                .SelectMany(id => _collectionChunks.TryGetValue(id, out var keys) ? (IEnumerable<string>)keys : Array.Empty<string>())
                .Select(key => _chunks[key]).Where(c => c.Chunk.Vector is not null
                    && c.Document.EmbeddingModelId == _embeddingModelId && c.Chunk.Vector.Length == _embeddingDimensions).ToArray();
        }
        var ranked = new Dictionary<string, double>(StringComparer.Ordinal);
        int rank = 0;
        int candidateLimit = Math.Max(options.Limit * 8, 64);
        var lexicalOrder = lexical.OrderByDescending(p => preferredMediaCandidates.Contains(p.Key))
            .ThenByDescending(p => anchoredCandidates.Contains(p.Key))
            .ThenByDescending(p => p.Value).ThenBy(p => p.Key, StringComparer.Ordinal);
        IEnumerable<KeyValuePair<string, double>> lexicalCandidates = lexicalOrder;
        if (requestedMedia is not null)
        {
            // Keep the best match from each source in the candidate budget before
            // adding more excerpts from any one large source. Otherwise pruning
            // can discard an asset or complementary source before diversification.
            var documents = new HashSet<string>(StringComparer.Ordinal);
            var firstChunks = lexicalOrder.Where(item => documents.Add(candidates[item.Key].Document.Id))
                .Select(item => item.Key).ToHashSet(StringComparer.Ordinal);
            lexicalCandidates = lexicalOrder.OrderByDescending(item => firstChunks.Contains(item.Key));
        }
        foreach (var item in lexicalCandidates.Take(candidateLimit))
            ranked[item.Key] = 1.0 / (60 + ++rank);
        if (_embeddings is not null && denseCandidates.Length > 0)
        {
            VerifyEmbeddingIdentity();
            float[][] batch = await _embeddings.EmbedAsync(new[] { query }, true, cancellationToken).ConfigureAwait(false);
            ValidateVectors(batch, 1, _embeddingDimensions);
            float[] vector = Normalize(batch[0]);
            var dense = new List<(IndexedChunk Chunk, double Score)>(denseCandidates.Length);
            foreach (var candidate in denseCandidates)
            {
                cancellationToken.ThrowIfCancellationRequested();
                double cosine = Dot(vector, candidate.Chunk.Vector!);
                if (cosine > 0) dense.Add((candidate, cosine));
            }
            rank = 0;
            foreach (var item in dense.OrderByDescending(d => d.Score).ThenBy(d => d.Chunk.Key, StringComparer.Ordinal).Take(candidateLimit))
            {
                ranked[item.Chunk.Key] = ranked.GetValueOrDefault(item.Chunk.Key) + 1.0 / (60 + ++rank);
                denseScores[item.Chunk.Key] = item.Score;
                candidates[item.Chunk.Key] = item.Chunk;
            }
        }
        var results = new List<KnowledgeSearchResult>();
        var selected = new List<IndexedChunk>();
        var fingerprints = new Dictionary<string, DuplicateFingerprint>(StringComparer.Ordinal);
        int remaining = options.MaxCharacters;
        lock (_sync)
        {
            var order = ranked.OrderByDescending(p => p.Value)
                .ThenByDescending(p => lexical.GetValueOrDefault(p.Key))
                .ThenByDescending(p => denseScores.GetValueOrDefault(p.Key))
                .ThenBy(p => candidates[p.Key].Document.FileName, StringComparer.Ordinal)
                .ThenBy(p => p.Key, StringComparer.Ordinal);
            var remainingCandidates = order.ToList();
            var selectedPages = new Dictionary<(string Document, int Page), int>();
            var selectedDocuments = new Dictionary<string, int>(StringComparer.Ordinal);
            while (remainingCandidates.Count > 0)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (remaining <= 0 || results.Count >= options.Limit) break;
                int bestIndex = 0;
                double bestScore = double.NegativeInfinity;
                bool bestPreferredMedia = false;
                for (int i = 0; i < remainingCandidates.Count; i++)
                {
                    var candidate = remainingCandidates[i];
                    var chunk = candidates[candidate.Key];
                    int repeats = coverageDocuments.Contains(chunk.Document.Id) && chunk.Chunk.Location?.Page is int page
                        ? selectedPages.GetValueOrDefault((chunk.Document.Id, page)) : 0;
                    if (requestedMedia is not null)
                        repeats = Math.Max(repeats, selectedDocuments.GetValueOrDefault(chunk.Document.Id));
                    // Relevance always chooses the first hit. For an ambiguous
                    // broad query, repeated pages need proportionally stronger
                    // evidence; page numbers never substitute for query relevance.
                    double score = candidate.Value / (1 + .25 * repeats);
                    bool preferredMedia = preferredMediaCandidates.Contains(candidate.Key);
                    if ((preferredMedia && !bestPreferredMedia) || (preferredMedia == bestPreferredMedia && score > bestScore))
                    { bestIndex = i; bestScore = score; bestPreferredMedia = preferredMedia; }
                }
                var item = remainingCandidates[bestIndex];
                remainingCandidates.RemoveAt(bestIndex);
                // Embedding can take appreciable time. A deletion or replacement during
                // that await revokes the snapshot: never disclose stale source text.
                if (!_chunks.TryGetValue(item.Key, out var current) || !ReferenceEquals(current, candidates[item.Key])
                    || !_collections.ContainsKey(current.Document.CollectionId)) continue;
                // OCR and selectable PDF text can represent the same paragraph.
                // Keep complementary evidence, but do not spend the result budget twice.
                if (selected.Any(previous => IsDuplicateEvidence(previous, current, fingerprints))) continue;
                var result = Result(current, bestScore, remaining);
                results.Add(result); selected.Add(current); remaining -= result.Text.Length;
                selectedDocuments[current.Document.Id] = selectedDocuments.GetValueOrDefault(current.Document.Id) + 1;
                if (current.Chunk.Location?.Page is int selectedPage)
                {
                    var pageKey = (current.Document.Id, selectedPage);
                    selectedPages[pageKey] = selectedPages.GetValueOrDefault(pageKey) + 1;
                }
            }
        }
        return results;
    }

    public IReadOnlyList<KnowledgeSearchResult> Read(string reference, int maxCharacters = 12000,
        int adjacentChunks = 0, IReadOnlyList<string>? collectionIds = null)
    {
        if (maxCharacters <= 0 || maxCharacters > _options.MaxContextCharacters) throw new ArgumentOutOfRangeException(nameof(maxCharacters));
        if (adjacentChunks is < 0 or > 5) throw new ArgumentOutOfRangeException(nameof(adjacentChunks));
        var parsed = ParseReference(reference);
        var scope = collectionIds is null ? null : Scope(collectionIds);
        if (scope is not null && !scope.Contains(parsed.CollectionId)) throw new KeyNotFoundException("Source is outside the selected collections.");
        lock (_sync)
        {
            ThrowIfDisposed();
            var document = RequireDocument(parsed.CollectionId, parsed.DocumentId);
            if (document.Document.Version != parsed.Version) throw new KnowledgeStaleReferenceException("This source version is stale. Search again for the current document.");
            if (parsed.ChunkIndex >= document.Chunks.Length) throw new KeyNotFoundException("Knowledge chunk does not exist.");
            var result = new List<KnowledgeSearchResult>();
            int remaining = maxCharacters;
            var indices = new List<int> { parsed.ChunkIndex };
            for (int offset = 1; offset <= adjacentChunks; offset++)
            {
                if (parsed.ChunkIndex - offset >= 0) indices.Add(parsed.ChunkIndex - offset);
                if (parsed.ChunkIndex + offset < document.Chunks.Length) indices.Add(parsed.ChunkIndex + offset);
            }
            foreach (int i in indices)
            {
                if (remaining <= 0) break;
                var item = Result(_chunks[Reference(document.Document, i)], 0, remaining);
                result.Add(item); remaining -= item.Text.Length;
            }
            // Keep the requested chunk first when a downstream tool must apply a
            // smaller serialization budget; neighbors carry their own positions.
            return result;
        }
    }

    /// <summary>
    /// Expands a current reference to all of its document's indexed excerpts, in
    /// source order. Returns null if the complete deduplicated result exceeds the
    /// budget; it never returns a partially expanded document. Source status still
    /// describes extraction completeness. Returned chunks retain their exact text
    /// and provenance. Callers must separately budget serialization and model tokens.
    /// </summary>
    public IReadOnlyList<KnowledgeSearchResult>? ReadDocumentContext(string reference, int maxCharacters = 32000,
        IReadOnlyList<string>? collectionIds = null)
    {
        if (maxCharacters <= 0 || maxCharacters > _options.MaxDocumentContextCharacters)
            throw new ArgumentOutOfRangeException(nameof(maxCharacters));
        var parsed = ParseReference(reference);
        var scope = collectionIds is null ? null : Scope(collectionIds);
        if (scope is not null && !scope.Contains(parsed.CollectionId))
            throw new KeyNotFoundException("Source is outside the selected collections.");
        lock (_sync)
        {
            ThrowIfDisposed();
            var document = RequireDocument(parsed.CollectionId, parsed.DocumentId);
            if (document.Document.Version != parsed.Version)
                throw new KnowledgeStaleReferenceException("This source version is stale. Search again for the current document.");
            if (parsed.ChunkIndex >= document.Chunks.Length) throw new KeyNotFoundException("Knowledge chunk does not exist.");
            var chunks = Enumerable.Range(0, document.Chunks.Length)
                .Select(index => _chunks[Reference(document.Document, index)]).ToArray();
            var redundant = new HashSet<int>();
            foreach (var page in chunks.Where(chunk => chunk.Chunk.Location?.Page is not null)
                .GroupBy(chunk => chunk.Chunk.Location))
            {
                var native = page.Where(chunk => chunk.Chunk.Modality == "text").ToArray();
                var ocr = page.Where(chunk => chunk.Chunk.Modality == "ocr").ToArray();
                if (native.Length == 0 || ocr.Length == 0) continue;
                // Page representations can have different chunk boundaries. Compare
                // their reconstructed text, but emit only untouched original chunks.
                // Complete context is stricter than search's near-duplicate filter:
                // even a changed number, punctuation or added OCR sentence is kept.
                string nativeText = NormalizeEvidenceWhitespace(StitchForComparison(native));
                string ocrText = NormalizeEvidenceWhitespace(StitchForComparison(ocr));
                if (nativeText == ocrText)
                    foreach (var chunk in ocr) redundant.Add(chunk.Index);
                else
                    foreach (var chunk in ocr)
                        if (ContainsExactEvidenceExcerpt(nativeText, ocrText, NormalizeEvidenceWhitespace(chunk.Chunk.Text)))
                            redundant.Add(chunk.Index);
            }

            var result = new List<KnowledgeSearchResult>();
            var seen = new HashSet<(KnowledgeLocation? Location, string Text)>();
            int remaining = maxCharacters;
            foreach (var chunk in chunks)
            {
                if (redundant.Contains(chunk.Index)
                    || !seen.Add((chunk.Chunk.Location, NormalizeEvidenceWhitespace(chunk.Chunk.Text)))) continue;
                if (chunk.Chunk.Text.Length > remaining) return null;
                result.Add(Result(chunk, 0, chunk.Chunk.Text.Length));
                remaining -= chunk.Chunk.Text.Length;
            }
            return result;
        }
    }

    private string StitchForComparison(IReadOnlyList<IndexedChunk> chunks)
    {
        var text = new StringBuilder(chunks[0].Chunk.Text);
        for (int i = 1; i < chunks.Count; i++)
        {
            string previous = chunks[i - 1].Chunk.Text, current = chunks[i].Chunk.Text;
            int overlap = _options.ChunkOverlapCharacters;
            if (overlap > previous.Length || overlap > current.Length) overlap = 0;
            if (overlap > 0 && !previous.AsSpan(previous.Length - overlap).SequenceEqual(current.AsSpan(0, overlap)))
            {
                // Chunking advances past a low surrogate at an overlap boundary.
                // Do not guess shorter overlaps between independently extracted
                // segments: repeated words may represent distinct source content.
                if (overlap > 1 && char.IsLowSurrogate(previous[previous.Length - overlap])
                    && previous.AsSpan(previous.Length - overlap + 1).SequenceEqual(current.AsSpan(0, overlap - 1))) overlap--;
                else overlap = 0;
            }
            if (overlap == 0) text.Append('\n');
            text.Append(current.AsSpan(overlap));
        }
        return text.ToString();
    }

    private static string NormalizeEvidenceWhitespace(string text) => string.Join(' ',
        text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));

    private static bool ContainsExactEvidenceExcerpt(string nativeText, string ocrText, string excerpt)
    {
        if (excerpt.Length == 0) return false;
        // Match against the whole native page so different chunk boundaries do not
        // hide exact duplicates. Include complete boundary tokens from the OCR page:
        // an overlap may start mid-word, but "adequate" must not match "inadequate",
        // nor "4.5" match "14.5". Punctuation, signs and numeric spacing stay exact.
        for (int start = ocrText.IndexOf(excerpt, StringComparison.Ordinal); start >= 0;
            start = ocrText.IndexOf(excerpt, start + 1, StringComparison.Ordinal))
        {
            int left = start, right = start + excerpt.Length;
            while (left > 0 && ocrText[left - 1] != ' ') left--;
            while (right < ocrText.Length && ocrText[right] != ' ') right++;
            string completeTokens = ocrText[left..right];
            for (int match = nativeText.IndexOf(completeTokens, StringComparison.Ordinal); match >= 0;
                match = nativeText.IndexOf(completeTokens, match + 1, StringComparison.Ordinal))
            {
                if ((match == 0 || nativeText[match - 1] == ' ')
                    && (match + completeTokens.Length == nativeText.Length || nativeText[match + completeTokens.Length] == ' '))
                    return true;
            }
        }
        return false;
    }

    public Stream OpenOriginal(string collectionId, string documentId, int version)
    {
        ValidateId(collectionId, nameof(collectionId)); ValidateId(documentId, nameof(documentId));
        lock (_sync)
        {
            ThrowIfDisposed();
            var stored = RequireDocument(collectionId, documentId);
            if (stored.Document.Version != version) throw new KnowledgeStaleReferenceException("This source version is stale.");
            string path = ObjectPath(stored.Document.ContentHash);
            RejectLink(path);
            return new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read | FileShare.Delete);
        }
    }

    public KnowledgeDocumentPreview PreviewDocument(string collectionId, string documentId, int version,
        int maxCharacters = 32000, int chunkIndex = 0)
    {
        ValidateId(collectionId, nameof(collectionId)); ValidateId(documentId, nameof(documentId));
        if (maxCharacters is < 1 or > 32000) throw new ArgumentOutOfRangeException(nameof(maxCharacters));
        if (chunkIndex < 0) throw new ArgumentOutOfRangeException(nameof(chunkIndex));
        lock (_sync)
        {
            ThrowIfDisposed();
            var stored = RequireDocument(collectionId, documentId);
            if (stored.Document.Version != version) throw new KnowledgeStaleReferenceException("This source version is stale.");
            if (chunkIndex >= stored.Chunks.Length) throw new KeyNotFoundException("Knowledge chunk does not exist.");
            var text = new StringBuilder();
            bool truncated = false;
            int? nextChunkIndex = null;
            int emitted = 0;
            StoredChunk? previous = null;
            for (int index = chunkIndex; index < stored.Chunks.Length; index++)
            {
                var chunk = stored.Chunks[index];
                if (chunk.Modality == "metadata") continue;
                int overlap = previous is not null && previous.Location == chunk.Location && previous.Modality == chunk.Modality
                    ? Math.Min(_options.ChunkOverlapCharacters, Math.Min(previous.Text.Length, chunk.Text.Length)) : 0;
                if (overlap > 0 && !previous!.Text.AsSpan(previous.Text.Length - overlap).SequenceEqual(chunk.Text.AsSpan(0, overlap))) overlap = 0;
                int separator = overlap == 0 && text.Length > 0 ? 1 : 0;
                // End on a chunk boundary when another complete excerpt cannot fit.
                // A caller choosing a smaller-than-chunk budget still gets a bounded
                // first excerpt, and can reopen that chunk with the standard budget.
                if (emitted > 0 && chunk.Text.Length - overlap + separator > maxCharacters - text.Length)
                { truncated = true; nextChunkIndex = index; break; }
                if (overlap == 0 && text.Length > 0)
                {
                    if (text.Length >= maxCharacters) { truncated = true; nextChunkIndex = index; break; }
                    text.Append('\n');
                }
                int count = Math.Min(maxCharacters - text.Length, chunk.Text.Length - overlap);
                if (count > 0 && count < chunk.Text.Length - overlap && char.IsHighSurrogate(chunk.Text[overlap + count - 1])) count--;
                text.Append(chunk.Text.AsSpan(overlap, count));
                emitted++;
                if (count < chunk.Text.Length - overlap) { truncated = true; nextChunkIndex = index; break; }
                previous = chunk;
            }
            return new(PublicDocument(stored.Document), text.ToString(), truncated, chunkIndex, stored.Chunks[chunkIndex].Location,
                chunkIndex == 0 ? null : Math.Max(0, chunkIndex - Math.Max(1, emitted)), nextChunkIndex);
        }
    }

    private async Task<StoredDocument> PrepareAsync(string collectionId, string fileName, string? contentType, string? relativePath,
        string hash, long size, StoredDocument? previous, KnowledgeExtraction extraction, CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(extraction);
        if (extraction.Segments is null || extraction.Segments.Count > _options.MaxSegments) throw new KnowledgeLimitException("Extraction segment limit exceeded.");
        if (extraction.Warnings is null || extraction.Warnings.Count > 100 || extraction.Warnings.Any(w => w is null || w.Length > 4000))
            throw new KnowledgeLimitException("Extraction warnings exceed the metadata limit.");
        if (!Enum.IsDefined(extraction.Status)) throw new InvalidDataException("Unknown extraction status.");
        if (extraction.ExtractorId?.Length > 1000) throw new KnowledgeLimitException("Extractor identity exceeds the metadata limit.");
        var chunks = new List<StoredChunk>();
        int characters = 0;
        foreach (var segment in extraction.Segments)
        {
            ct.ThrowIfCancellationRequested();
            if (segment?.Text is null) throw new InvalidDataException("Extracted segments must contain text.");
            ValidateSourceMetadata(segment.Location, segment.Modality);
            characters = checked(characters + segment.Text.Length);
            if (characters > _options.MaxExtractedCharacters) throw new KnowledgeLimitException("Extracted text exceeds the configured character limit.");
            string text = segment.Text.Trim();
            for (int start = 0; start < text.Length;)
            {
                int length = Math.Min(_options.ChunkCharacters, text.Length - start);
                int end = start + length;
                if (end < text.Length && end > start && char.IsHighSurrogate(text[end - 1])) end--;
                if (end < text.Length)
                {
                    int boundary = text.LastIndexOfAny(['\n', ' ', '\t'], end - 1, Math.Max(1, (end - start) / 4));
                    if (boundary > start) end = boundary + 1;
                }
                chunks.Add(new StoredChunk { Text = text[start..end], Location = segment.Location, Modality = segment.Modality });
                if (chunks.Count > _options.MaxChunksPerDocument) throw new KnowledgeLimitException("Document chunk limit exceeded.");
                if (end == text.Length) break;
                start = Math.Max(start + 1, end - _options.ChunkOverlapCharacters);
                if (start < text.Length && char.IsLowSurrogate(text[start])) start++;
            }
        }
        // Assets without a configured semantic extractor remain explicitly metadata-only.
        var status = extraction.Status;
        if (chunks.Count == 0)
        {
            status = KnowledgeDocumentStatus.MetadataOnly;
            chunks.Add(new StoredChunk { Text = fileName, Modality = "metadata" });
        }
        lock (_sync)
            if (_chunks.Count - (previous?.Chunks.Length ?? 0) + chunks.Count > _options.MaxTotalChunks)
                throw new KnowledgeLimitException("Knowledge chunk capacity exceeded.");
        if (_embeddings is not null && status != KnowledgeDocumentStatus.MetadataOnly)
        {
            for (int start = 0; start < chunks.Count; start += _options.EmbeddingBatchSize)
            {
                ct.ThrowIfCancellationRequested();
                VerifyEmbeddingIdentity();
                var batch = chunks.Skip(start).Take(_options.EmbeddingBatchSize).ToArray();
                var vectors = await _embeddings.EmbedAsync(batch.Select(c => c.Text).ToArray(), false, ct).ConfigureAwait(false);
                ValidateVectors(vectors, batch.Length, _embeddingDimensions);
                for (int i = 0; i < batch.Length; i++) batch[i].Vector = Normalize(vectors[i]);
            }
        }
        var now = DateTimeOffset.UtcNow;
        return new StoredDocument
        {
            Schema = SchemaVersion,
            Document = new KnowledgeDocument
            {
                Id = previous?.Document.Id ?? Guid.NewGuid().ToString("N"), CollectionId = collectionId,
                FileName = fileName, RelativePath = relativePath, ContentType = contentType, ContentHash = hash, SizeBytes = size,
                Version = checked((previous?.Document.Version ?? 0) + 1), ChunkCount = chunks.Count, Status = status,
                Warnings = extraction.Warnings.ToArray(), EmbeddingModelId = chunks.Any(c => c.Vector is not null) ? _embeddingModelId : null,
                ExtractorId = extraction.ExtractorId,
                CreatedAt = previous?.Document.CreatedAt ?? now, UpdatedAt = now,
            },
            Chunks = chunks.ToArray(),
        };
    }

    private async Task CommitAsync(StoredDocument stored, StoredDocument? previous, CancellationToken ct)
    {
        await WriteJsonAtomicAsync(DocumentPath(stored.Document.Id), stored, ct).ConfigureAwait(false);
        lock (_sync)
        {
            if (previous is not null) RemoveIndex(previous);
            _documents[stored.Document.Id] = stored;
            AddIndex(stored);
            if (previous is not null && previous.Document.ContentHash != stored.Document.ContentHash) DeleteUnusedObject(previous.Document.ContentHash);
        }
    }

    private async Task<(long Length, string Hash)> CopyBoundedAsync(Stream source, string path, CancellationToken ct)
    {
        await using var destination = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None, 81920, true);
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        byte[] buffer = new byte[81920];
        long length = 0;
        int read;
        while ((read = await source.ReadAsync(buffer, ct).ConfigureAwait(false)) > 0)
        {
            length = checked(length + read);
            if (length > _options.MaxFileBytes) throw new KnowledgeLimitException("File exceeds the configured upload limit.");
            hash.AppendData(buffer, 0, read);
            await destination.WriteAsync(buffer.AsMemory(0, read), ct).ConfigureAwait(false);
        }
        await destination.FlushAsync(ct).ConfigureAwait(false);
        destination.Flush(true);
        return (length, Convert.ToHexStringLower(hash.GetHashAndReset()));
    }

    private async Task WriteJsonAtomicAsync<T>(string destination, T value, CancellationToken ct)
    {
        string temporary = Path.Combine(_root, "staging", Guid.NewGuid().ToString("N") + ".json");
        try
        {
            await using (var stream = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None, 16384, true))
            {
                await JsonSerializer.SerializeAsync(stream, value, Json, ct).ConfigureAwait(false);
                await stream.FlushAsync(ct).ConfigureAwait(false); stream.Flush(true);
            }
            ct.ThrowIfCancellationRequested();
            File.Move(temporary, destination, true);
        }
        finally { TryDelete(temporary); }
    }

    private void Load()
    {
        foreach (string path in Directory.EnumerateFiles(Path.Combine(_root, "collections"), "*.json"))
        {
            var collection = ReadJson<KnowledgeCollection>(path, 16384);
            ValidateId(collection.Id, "stored collection ID");
            if (Path.GetFileNameWithoutExtension(path) != collection.Id || string.IsNullOrWhiteSpace(collection.Name)) throw new InvalidDataException("Invalid collection manifest.");
            if (_collections.Count >= _options.MaxCollections) throw new KnowledgeLimitException("Stored collections exceed configured capacity.");
            _collections.Add(collection.Id, collection);
        }
        long storedBytes = 0;
        foreach (string path in Directory.EnumerateFiles(Path.Combine(_root, "documents"), "*.json"))
        {
            // JSON float vectors can be larger than source text; still cap each manifest
            // before deserializing so an accidentally corrupt file cannot allocate unbounded memory.
            long maxManifest = Math.Min(512L * 1024 * 1024, _options.MaxExtractedCharacters * 8L + _options.MaxChunksPerDocument * 30000L);
            var stored = ReadJson<StoredDocument>(path, maxManifest);
            var doc = stored.Document ?? throw new InvalidDataException("Document metadata missing.");
            ValidateId(doc.Id, "stored document ID"); ValidateId(doc.CollectionId, "stored collection ID");
            if (!_collections.ContainsKey(doc.CollectionId)) { File.Delete(path); continue; }
            if (stored.Schema != SchemaVersion || Path.GetFileNameWithoutExtension(path) != doc.Id || doc.Version <= 0
                || doc.ContentHash is null || doc.ContentHash.Length != 64 || doc.ContentHash.Any(c => !char.IsAsciiHexDigitLower(c))
                || doc.SizeBytes < 0 || doc.SizeBytes > _options.MaxFileBytes || stored.Chunks is null || stored.Chunks.Length == 0
                || stored.Chunks.Length != doc.ChunkCount || stored.Chunks.Length > _options.MaxChunksPerDocument
                || !Enum.IsDefined(doc.Status) || doc.ContentType?.Length > 200 || doc.ExtractorId?.Length > 1000
                || doc.EmbeddingModelId?.Length > 1000 || doc.Warnings is null || doc.Warnings.Count > 100
                || doc.Warnings.Any(w => w is null || w.Length > 4000))
                throw new InvalidDataException("Invalid knowledge document manifest.");
            ValidateFileName(doc.FileName);
            var original = new FileInfo(ObjectPath(doc.ContentHash));
            RejectLink(original.FullName);
            if (!original.Exists || original.Length != doc.SizeBytes) throw new InvalidDataException("Knowledge original is missing or has an invalid size.");
            long chunkCharacters = 0;
            int? vectorDimensions = null;
            foreach (var chunk in stored.Chunks)
            {
                if (chunk is null || chunk.Text is null || chunk.Text.Length > Math.Max(_options.ChunkCharacters, 512)) throw new InvalidDataException("Invalid stored chunk.");
                ValidateSourceMetadata(chunk.Location, chunk.Modality);
                chunkCharacters += chunk.Text.Length;
                if (chunkCharacters > Math.Max(512, _options.MaxExtractedCharacters * 2L)) throw new InvalidDataException("Stored chunk text exceeds the extraction limit.");
                if (chunk.Vector is not null)
                {
                    if (string.IsNullOrWhiteSpace(doc.EmbeddingModelId) || chunk.Vector.Length is <= 0 or > 65536 || chunk.Vector.Any(v => !float.IsFinite(v)))
                        throw new InvalidDataException("Invalid stored embedding.");
                    if (vectorDimensions is not null && vectorDimensions != chunk.Vector.Length) throw new InvalidDataException("Stored embedding dimensions differ within a document.");
                    vectorDimensions = chunk.Vector.Length;
                    chunk.Vector = Normalize(chunk.Vector);
                }
                else if (doc.EmbeddingModelId is not null) throw new InvalidDataException("Stored embedding is missing from an embedded document.");
            }
            storedBytes = checked(storedBytes + doc.SizeBytes);
            if (_documents.Count >= _options.MaxDocuments || _chunks.Count + stored.Chunks.Length > _options.MaxTotalChunks || storedBytes > _options.MaxStoredBytes)
                throw new KnowledgeLimitException("Stored knowledge exceeds configured capacity.");
            ValidateRelativePath(doc.RelativePath, doc.FileName);
            if (_documentsByName.ContainsKey((doc.CollectionId, DocumentName(doc))))
                throw new InvalidDataException("Duplicate knowledge document name.");
            _documents.Add(doc.Id, stored); AddIndex(stored);
        }
    }

    private static T ReadJson<T>(string path, long maxBytes)
    {
        RejectLink(path);
        if (new FileInfo(path).Length > maxBytes) throw new InvalidDataException("Knowledge manifest exceeds the configured limit.");
        using var stream = File.OpenRead(path);
        return JsonSerializer.Deserialize<T>(stream, Json) ?? throw new InvalidDataException("Empty knowledge manifest.");
    }

    private void AddIndex(StoredDocument stored)
    {
        var document = stored.Document;
        _documentsByName.Add((document.CollectionId, DocumentName(document)), stored);
        _storedBytes += document.SizeBytes;
        _objectReferences[document.ContentHash] = _objectReferences.GetValueOrDefault(document.ContentHash) + 1;
        if (!_collectionChunks.TryGetValue(document.CollectionId, out var collectionChunks))
            _collectionChunks[document.CollectionId] = collectionChunks = new(StringComparer.Ordinal);
        for (int i = 0; i < stored.Chunks.Length; i++)
        {
            var chunk = stored.Chunks[i];
            var terms = Tokenize(chunk.Text).GroupBy(t => t, StringComparer.Ordinal).ToDictionary(g => g.Key, g => g.Count(), StringComparer.Ordinal);
            // Names and folder paths remain searchable after OCR/captioning.
            // Index them once, without inserting metadata into source evidence or
            // amplifying a filename across every chunk of a large document.
            if (i == 0)
                foreach (string term in Tokenize(DocumentName(document))) terms.TryAdd(term, 1);
            string key = Reference(stored.Document, i);
            int length = Math.Max(1, terms.Values.Sum());
            _chunks.Add(key, new IndexedChunk(key, stored.Document, chunk, i, length, terms.Keys.ToArray()));
            collectionChunks.Add(key);
            _collectionTokenCounts[document.CollectionId] = _collectionTokenCounts.GetValueOrDefault(document.CollectionId) + length;
            foreach (var term in terms)
            {
                if (!_postings.TryGetValue(term.Key, out var posting)) _postings[term.Key] = posting = new(StringComparer.Ordinal);
                posting.Add(key, term.Value);
            }
        }
    }

    private void RemoveIndex(StoredDocument stored)
    {
        var document = stored.Document;
        _documentsByName.Remove((document.CollectionId, DocumentName(document)));
        _storedBytes -= document.SizeBytes;
        int references = _objectReferences[document.ContentHash] - 1;
        if (references == 0) _objectReferences.Remove(document.ContentHash);
        else _objectReferences[document.ContentHash] = references;
        for (int i = 0; i < stored.Chunks.Length; i++)
        {
            string key = Reference(stored.Document, i);
            if (!_chunks.Remove(key, out var chunk)) continue;
            _collectionChunks[document.CollectionId].Remove(key);
            _collectionTokenCounts[document.CollectionId] -= chunk.Length;
            foreach (string term in chunk.Terms)
                if (_postings.TryGetValue(term, out var posting)) { posting.Remove(key); if (posting.Count == 0) _postings.Remove(term); }
        }
    }

    private static IEnumerable<string> Tokenize(string text)
    {
        var current = new StringBuilder();
        string? previousCjk = null;
        foreach (Rune rune in text.EnumerateRunes())
        {
            bool cjk = IsCjk(rune);
            if (cjk || !Rune.IsLetterOrDigit(rune))
            {
                if (current.Length > 0) { yield return current.ToString(); current.Clear(); }
                if (cjk)
                {
                    string token = rune.ToString();
                    yield return token;
                    if (previousCjk is not null) yield return previousCjk + token;
                    previousCjk = token;
                }
                else previousCjk = null;
            }
            else
            {
                previousCjk = null;
                if (current.Length < 128) current.Append(Rune.ToLowerInvariant(rune).ToString());
            }
        }
        if (current.Length > 0) yield return current.ToString();
    }

    private static bool IsCjk(Rune rune) => rune.Value is >= 0x3400 and <= 0x9FFF or >= 0x20000 and <= 0x3134F or >= 0x3040 and <= 0x30FF;
    private static bool IsCjkTerm(string term) => IsCjk(Rune.GetRuneAt(term, 0));
    private static string? RequestedMedia(string query, string[] terms)
    {
        // A question about image encoding, an audio model, or video APIs needs
        // documentation. A modality noun alone is insufficient asset intent.
        if (terms.Any(MediaConceptTerms.Contains)
            || ContainsPhrase(query, "编码", "解码", "模型", "接口", "原理", "算法", "文档", "教程", "实现")) return null;
        bool lookup = terms.Any(QueryRequestWords.Contains) || ContainsPhrase(query, "寻找", "查找", "搜索", "检索", "找到", "展示", "显示", "获取");
        if (!lookup && (terms.Any(QueryQuestionWords.Contains) || ContainsPhrase(query, "如何", "怎么", "为什么", "原理", "支持"))) return null;
        bool image = terms.Any(ImageQueryTerms.Contains) || ContainsPhrase(query, "图片", "图像", "照片", "相片", "海报", "宣传画", "宣传图", "插画", "截图");
        bool audio = terms.Any(AudioQueryTerms.Contains) || ContainsPhrase(query, "音频", "声音", "录音", "播客", "歌曲", "音乐");
        bool video = terms.Any(VideoQueryTerms.Contains) || ContainsPhrase(query, "视频", "影片", "录像");
        // Requests discussing several modalities retain ordinary relevance.
        return (image ? 1 : 0) + (audio ? 1 : 0) + (video ? 1 : 0) == 1
            ? image ? "image" : audio ? "audio" : "video" : null;

        static bool ContainsPhrase(string text, params string[] phrases) => phrases.Any(phrase => text.Contains(phrase, StringComparison.Ordinal));
    }
    private static bool MatchesMedia(KnowledgeDocument document, string requested)
    {
        if (document.ContentType?.StartsWith(requested + "/", StringComparison.OrdinalIgnoreCase) == true) return true;
        string? kind = Path.GetExtension(document.FileName).ToLowerInvariant() switch
        {
            ".png" or ".jpg" or ".jpeg" or ".gif" or ".webp" or ".bmp" or ".tif" or ".tiff"
                or ".heic" or ".heif" or ".avif" or ".svg" => "image",
            ".mp3" or ".wav" or ".m4a" or ".aac" or ".flac" or ".ogg" or ".opus" => "audio",
            ".mp4" or ".mov" or ".webm" or ".mkv" or ".avi" => "video",
            _ => null,
        };
        return kind == requested;
    }
    private static string[] QueryTerms(string query)
    {
        var terms = Tokenize(query).Distinct(StringComparer.Ordinal).ToArray();
        // Unigrams are retained in the index for one-character queries. Longer
        // CJK questions use bigrams so unrelated characters do not overwhelm names
        // and meaningful phrases. This index is rebuilt on load, without reimport.
        bool hasCjkPairs = terms.Any(term => IsCjkTerm(term) && term.EnumerateRunes().Count() == 2);
        return hasCjkPairs ? terms.Where(term => !IsCjkTerm(term) || term.EnumerateRunes().Count() > 1).ToArray() : terms;
    }

    private static bool IsDuplicateEvidence(IndexedChunk first, IndexedChunk second,
        Dictionary<string, DuplicateFingerprint> fingerprints)
    {
        if (first.Document.Id != second.Document.Id || first.Chunk.Location != second.Chunk.Location) return false;
        if (first.Chunk.Text == second.Chunk.Text) return true;
        // Approximate suppression is restricted to two representations of the same
        // PDF page. Different pages, same-modality passages and changed numbers
        // remain independently retrievable.
        if (first.Chunk.Location?.Page is null || first.Chunk.Modality == second.Chunk.Modality
            || !(first.Chunk.Modality is "ocr" or "text") || !(second.Chunk.Modality is "ocr" or "text")) return false;
        var a = Fingerprint(first); var b = Fingerprint(second);
        if (a.Shingles.Count < 12 || b.Shingles.Count < 12 || a.Numbers != b.Numbers) return false;
        int intersection = a.Shingles.Count(value => b.Shingles.Contains(value));
        return intersection >= .9 * (a.Shingles.Count + b.Shingles.Count - intersection);

        DuplicateFingerprint Fingerprint(IndexedChunk chunk)
        {
            if (fingerprints.TryGetValue(chunk.Key, out var value)) return value;
            var tokens = Tokenize(chunk.Chunk.Text).ToArray();
            var shingles = new HashSet<int>();
            for (int i = 2; i < tokens.Length; i++) shingles.Add(HashCode.Combine(tokens[i - 2], tokens[i - 1], tokens[i]));
            return fingerprints[chunk.Key] = new(shingles, string.Join('\0', tokens.Where(token => token.Any(char.IsDigit))));
        }
    }

    private static KnowledgeSearchResult Result(IndexedChunk chunk, double score, int maximum)
    {
        int count = Math.Min(maximum, chunk.Chunk.Text.Length);
        if (count > 0 && count < chunk.Chunk.Text.Length && char.IsHighSurrogate(chunk.Chunk.Text[count - 1])) count--;
        return new KnowledgeSearchResult
        {
            Reference = chunk.Key, CollectionId = chunk.Document.CollectionId, DocumentId = chunk.Document.Id,
            FileName = chunk.Document.FileName, RelativePath = chunk.Document.RelativePath, Version = chunk.Document.Version, ChunkIndex = chunk.Index,
            DocumentChunkCount = chunk.Document.ChunkCount,
            Text = chunk.Chunk.Text[..count], Location = chunk.Chunk.Location, Modality = chunk.Chunk.Modality,
            Score = score, Status = chunk.Document.Status, Truncated = count < chunk.Chunk.Text.Length,
        };
    }

    private static string Reference(KnowledgeDocument document, int index) => $"kb://{document.CollectionId}/{document.Id}@{document.Version}/{index}";
    private static (string CollectionId, string DocumentId, int Version, int ChunkIndex) ParseReference(string reference)
    {
        if (reference is null || !reference.StartsWith("kb://", StringComparison.Ordinal) || reference.Length > 160)
            throw new ArgumentException("Invalid knowledge reference.", nameof(reference));
        string[] parts = reference[5..].Split('/');
        if (parts.Length != 3) throw new ArgumentException("Invalid knowledge reference.", nameof(reference));
        string[] document = parts[1].Split('@');
        ValidateId(parts[0], nameof(reference));
        if (document.Length != 2) throw new ArgumentException("Invalid knowledge reference.", nameof(reference));
        ValidateId(document[0], nameof(reference));
        if (!int.TryParse(document[1], NumberStyles.None, CultureInfo.InvariantCulture, out int version) || version <= 0
            || !int.TryParse(parts[2], NumberStyles.None, CultureInfo.InvariantCulture, out int chunk) || chunk < 0)
            throw new ArgumentException("Invalid knowledge reference.", nameof(reference));
        return (parts[0], document[0], version, chunk);
    }

    private static HashSet<string> Scope(IReadOnlyList<string> collectionIds)
    {
        ArgumentNullException.ThrowIfNull(collectionIds);
        if (collectionIds.Count > 1000) throw new ArgumentException("Too many selected collections.", nameof(collectionIds));
        var scope = new HashSet<string>(StringComparer.Ordinal);
        foreach (string id in collectionIds) { ValidateId(id, nameof(collectionIds)); scope.Add(id); }
        return scope;
    }
    private static void ValidateId(string id, string parameter)
    {
        if (id is null || id.Length != 32 || id.Any(c => !char.IsAsciiHexDigitLower(c))) throw new ArgumentException("Knowledge IDs must be 32 lowercase hexadecimal characters.", parameter);
    }
    private static void ValidateFileName(string name)
    {
        if (string.IsNullOrWhiteSpace(name) || name.Length > 512 || name is "." or ".." || name.Any(c => char.IsControl(c) || c is '/' or '\\'))
            throw new ArgumentException("A file name must contain 1–512 characters and no directory path or control characters.", nameof(name));
    }
    private static string DocumentName(KnowledgeDocument document) => document.RelativePath ?? document.FileName;

    private static void ValidateRelativePath(string? path, string fileName)
    {
        if (path is null) return;
        if (path.Length is 0 or > 2048)
            throw new ArgumentException("Source paths must contain 1–2048 characters.", nameof(path));
        string[] parts = path.Split('/');
        if (path.Any(c => char.IsControl(c) || c == '\\')
            || path.Length >= 2 && char.IsAsciiLetter(path[0]) && path[1] == ':'
            || parts.Any(part => string.IsNullOrWhiteSpace(part) || part is "." or ".." || part.Length > 512)
            || !string.Equals(parts[^1], fileName, StringComparison.Ordinal))
            throw new ArgumentException("Source paths must be relative, contain no traversal, and end with the uploaded file name.", nameof(path));
    }
    private void ValidateSearchBounds(int limit, int chars)
    {
        if (limit <= 0 || limit > _options.MaxSearchResults) throw new ArgumentOutOfRangeException(nameof(limit));
        if (chars <= 0 || chars > _options.MaxContextCharacters) throw new ArgumentOutOfRangeException(nameof(chars));
    }
    private static void ValidateOptions(KnowledgeOptions o)
    {
        if (o.MaxFileBytes <= 0 || o.MaxStoredBytes <= 0 || o.MaxDocuments <= 0 || o.MaxCollections <= 0 || o.MaxExtractedCharacters <= 0
            || o.MaxSegments <= 0 || o.MaxChunksPerDocument <= 0 || o.MaxTotalChunks <= 0 || o.ChunkCharacters is < 16 or > 32000
            || o.ChunkOverlapCharacters < 0 || o.ChunkOverlapCharacters >= o.ChunkCharacters / 2 || o.MaxSearchResults is <= 0 or > 1000
            || o.MaxContextCharacters is <= 0 or > 1000000 || o.MaxDocumentContextCharacters is <= 0 or > 1000000
            || o.MaxQueryCharacters is <= 0 or > 100000 || o.EmbeddingBatchSize is <= 0 or > 256
            || o.MaxPdfOcrPages is < 0 or > 4096 || o.MaxPdfPagePixels is < 65536 or > 64_000_000
            || o.MaxPdfEmbeddedImagePixels is < 1 or > 256_000_000 || o.PdfOcrPageTimeoutSeconds is < 1 or > 600)
            throw new ArgumentOutOfRangeException(nameof(o), "Knowledge limits must be positive and chunk overlap must be less than half the chunk size.");
    }
    private static void ValidateVectors(float[][] vectors, int count, int dimensions)
    {
        if (vectors is null || vectors.Length != count || vectors.Any(v => v is null || v.Length != dimensions || v.Any(n => !float.IsFinite(n))))
            throw new InvalidDataException("Embedding provider returned an invalid count, dimension, or non-finite value.");
    }
    private static void ValidateSourceMetadata(KnowledgeLocation? location, string? modality)
    {
        if (modality?.Length > 64) throw new InvalidDataException("Source modality metadata exceeds 64 characters.");
        if (location is null) return;
        if (location.Sheet?.Length > 512 || location.Section?.Length > 1000 || location.Label?.Length > 1000)
            throw new InvalidDataException("Source location metadata exceeds its size limit.");
        if (location.Page < 1 || location.Row < 1 || location.EndRow < location.Row || location.StartLine < 1 || location.EndLine < location.StartLine
            || location.StartMilliseconds < 0 || location.EndMilliseconds < 0 || location.EndMilliseconds < location.StartMilliseconds
            || location.StartSeconds is double start && (!double.IsFinite(start) || start < 0)
            || location.EndSeconds is double end && (!double.IsFinite(end) || end < 0)
            || location.EndSeconds < location.StartSeconds)
            throw new InvalidDataException("Source location has an invalid page, row, line, or timestamp.");
    }
    private static float[] Normalize(float[] vector)
    {
        double norm = Math.Sqrt(vector.Sum(v => (double)v * v));
        if (norm <= 0 || !double.IsFinite(norm)) throw new InvalidDataException("Embedding provider returned a zero or invalid vector.");
        return vector.Select(v => (float)(v / norm)).ToArray();
    }
    private static double Dot(float[] first, float[] second)
    {
        float value = 0;
        int i = 0;
        for (; i <= first.Length - Vector<float>.Count; i += Vector<float>.Count)
            value += Vector.Dot(new Vector<float>(first, i), new Vector<float>(second, i));
        for (; i < first.Length; i++) value += first[i] * second[i];
        return value;
    }
    private void VerifyEmbeddingIdentity()
    {
        if (_embeddings?.ModelId != _embeddingModelId || _embeddings?.Dimensions != _embeddingDimensions)
            throw new InvalidOperationException("The embedding provider identity changed. Reopen the store with a stable provider and reindex affected documents.");
    }
    private KnowledgeCollection RequireCollection(string id) => _collections.TryGetValue(id, out var result) ? result : throw new KeyNotFoundException("Knowledge collection does not exist.");
    private StoredDocument RequireDocument(string collection, string document)
    {
        RequireCollection(collection);
        return _documents.TryGetValue(document, out var result) && result.Document.CollectionId == collection ? result : throw new KeyNotFoundException("Knowledge document does not exist in this collection.");
    }
    private static KnowledgeDocument PublicDocument(KnowledgeDocument document) => document with { Warnings = document.Warnings.ToArray() };
    private string CollectionPath(string id) => Path.Combine(_root, "collections", id + ".json");
    private string DocumentPath(string id) => Path.Combine(_root, "documents", id + ".json");
    private string ObjectPath(string hash) => Path.Combine(_root, "objects", hash);
    private void DeleteUnusedObject(string hash)
    {
        if (!_objectReferences.ContainsKey(hash)) TryDelete(ObjectPath(hash));
    }
    private void CollectUnreferencedObjects()
    {
        var live = _documents.Values.Select(d => d.Document.ContentHash).ToHashSet(StringComparer.Ordinal);
        foreach (string path in Directory.EnumerateFiles(Path.Combine(_root, "objects")))
            if (!live.Contains(Path.GetFileName(path))) TryDelete(path);
    }
    private static void RejectLink(string path)
    {
        if ((File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0) throw new IOException("Knowledge storage paths must not be symbolic links.");
    }
    private static void TryDelete(string path) { try { File.Delete(path); } catch (IOException) { } catch (UnauthorizedAccessException) { } }
    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);
    public void Dispose()
    {
        _mutations.Wait();
        try
        {
            lock (_sync)
            {
                if (_disposed) return;
                _disposed = true;
                while (_activeSearches > 0) Monitor.Wait(_sync);
                try { (_embeddings as IDisposable)?.Dispose(); }
                finally { _writerLock.Dispose(); }
            }
        }
        finally { _mutations.Release(); }
    }

    private sealed class StoredDocument
    {
        public int Schema { get; set; }
        public KnowledgeDocument Document { get; set; } = null!;
        public StoredChunk[] Chunks { get; set; } = [];
    }
    private sealed class StoredChunk
    {
        public string Text { get; set; } = string.Empty;
        public KnowledgeLocation? Location { get; set; }
        public string? Modality { get; set; }
        public float[]? Vector { get; set; }
    }
    private sealed record IndexedChunk(string Key, KnowledgeDocument Document, StoredChunk Chunk, int Index, int Length, string[] Terms);
    private sealed record DuplicateFingerprint(HashSet<int> Shingles, string Numbers);
}
