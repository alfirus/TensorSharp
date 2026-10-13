using System.Text.Json.Serialization;

namespace TensorSharp.Knowledge;

public sealed class KnowledgeOptions
{
    public long MaxFileBytes { get; init; } = 64L * 1024 * 1024;
    public long MaxStoredBytes { get; init; } = 2L * 1024 * 1024 * 1024;
    public int MaxDocuments { get; init; } = 10000;
    public int MaxCollections { get; init; } = 100;
    public int MaxExtractedCharacters { get; init; } = 2_000_000;
    public int MaxSegments { get; init; } = 10000;
    public int MaxChunksPerDocument { get; init; } = 10000;
    public int MaxTotalChunks { get; init; } = 100000;
    public int ChunkCharacters { get; init; } = 1200;
    public int ChunkOverlapCharacters { get; init; } = 120;
    public int MaxSearchResults { get; init; } = 50;
    public int MaxContextCharacters { get; init; } = 32000;
    /// <summary>Separate ceiling for all-or-nothing small-document context expansion.
    /// Hosts must also check serialized bytes and the active model's token budget.</summary>
    public int MaxDocumentContextCharacters { get; init; } = 64000;
    public int MaxQueryCharacters { get; init; } = 4000;
    public int EmbeddingBatchSize { get; init; } = 32;
    /// <summary>Maximum complete PDF pages submitted to OCR in one import or reindex.</summary>
    public int MaxPdfOcrPages { get; init; } = 128;
    /// <summary>Maximum pixels in the complete rendered page, including rotation.</summary>
    public int MaxPdfPagePixels { get; init; } = 16_777_216;
    /// <summary>Maximum sum of embedded-image pixels per page before native rendering;
    /// applies to individual images too, and counts repeated image placements again.</summary>
    public long MaxPdfEmbeddedImagePixels { get; init; } = 64_000_000;
    /// <summary>Cancellation deadline per page; native drawing observes cancellation between stages.</summary>
    public int PdfOcrPageTimeoutSeconds { get; init; } = 60;
}

[JsonConverter(typeof(JsonStringEnumConverter))]
public enum KnowledgeDocumentStatus { Ready, MetadataOnly, Partial }

public sealed record KnowledgeLocation
{
    public int? Page { get; init; }
    public string? Sheet { get; init; }
    public int? Row { get; init; }
    public int? EndRow { get; init; }
    public int? StartLine { get; init; }
    public int? EndLine { get; init; }
    public string? Section { get; init; }
    public long? StartMilliseconds { get; init; }
    public long? EndMilliseconds { get; init; }
    public double? StartSeconds { get; init; }
    public double? EndSeconds { get; init; }
    public string? Label { get; init; }
}

public sealed record KnowledgeSegment(string Text, KnowledgeLocation? Location = null, string? Modality = null);
public sealed record KnowledgeExtractionContext(string FilePath, string FileName, string? ContentType);
public sealed record KnowledgeExtraction
{
    public string? ExtractorId { get; init; }
    public IReadOnlyList<KnowledgeSegment> Segments { get; init; } = Array.Empty<KnowledgeSegment>();
    public KnowledgeDocumentStatus Status { get; init; } = KnowledgeDocumentStatus.Ready;
    public IReadOnlyList<string> Warnings { get; init; } = Array.Empty<string>();
}

public interface IKnowledgeExtractor
{
    Task<KnowledgeExtraction> ExtractAsync(KnowledgeExtractionContext context, CancellationToken cancellationToken = default);
}

/// <summary>A host-provided OCR reader for a complete rendered PDF page, never merely its largest image.
/// The file is a store-owned, immutable original; page numbers are one-based. Implementations must
/// honor cancellation and all supplied resource bounds, and return an empty result when unavailable.</summary>
public interface IKnowledgePdfPageExtractor
{
    /// <summary>Optional one-time dependency preparation. The core calls this only when a page
    /// needs OCR, before starting its per-page deadline. Failure prevents an incomplete import.
    /// Implementations without external setup need not override the default.</summary>
    Task PrepareAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
    Task<KnowledgePdfPageExtraction> ExtractPageAsync(KnowledgePdfPageContext context, CancellationToken cancellationToken = default);
}

public sealed record KnowledgePdfPageContext(string FilePath, string FileName, int PageNumber, int MaxCharacters, int MaxPixels);

public sealed record KnowledgePdfPageExtraction
{
    public string? Text { get; init; }
    public string? ExtractorId { get; init; }
    public IReadOnlyList<string> Warnings { get; init; } = Array.Empty<string>();
}

public interface IKnowledgeEmbeddingProvider
{
    string ModelId { get; }
    int Dimensions { get; }
    Task<float[][]> EmbedAsync(IReadOnlyList<string> texts, bool isQuery, CancellationToken cancellationToken = default);
}

public sealed record KnowledgeCollection(string Id, string Name, string? Description, DateTimeOffset CreatedAt);

public sealed record KnowledgeDocument
{
    public required string Id { get; init; }
    public required string CollectionId { get; init; }
    public required string FileName { get; init; }
    /// <summary>Original path within an uploaded folder, using forward slashes. Never a local filesystem path.</summary>
    public string? RelativePath { get; init; }
    public string Kind => Path.GetExtension(FileName).ToLowerInvariant() switch
    {
        ".csv" or ".tsv" or ".json" or ".jsonl" or ".ndjson" or ".xlsx" or ".xls" or ".ods"
            or ".parquet" or ".arrow" or ".feather" => "dataset",
        ".png" or ".jpg" or ".jpeg" or ".gif" or ".webp" or ".bmp" or ".tif" or ".tiff"
            or ".heic" or ".heif" or ".avif" or ".svg" or ".mp3" or ".wav" or ".m4a" or ".aac"
            or ".flac" or ".ogg" or ".opus" or ".mp4" or ".mov" or ".webm" or ".mkv" or ".avi" => "media",
        _ when ContentType?.StartsWith("image/", StringComparison.OrdinalIgnoreCase) == true
            || ContentType?.StartsWith("audio/", StringComparison.OrdinalIgnoreCase) == true
            || ContentType?.StartsWith("video/", StringComparison.OrdinalIgnoreCase) == true => "media",
        _ => "document",
    };
    public string? ContentType { get; init; }
    public required string ContentHash { get; init; }
    public int Version { get; init; }
    public long SizeBytes { get; init; }
    public int ChunkCount { get; init; }
    public KnowledgeDocumentStatus Status { get; init; }
    public IReadOnlyList<string> Warnings { get; init; } = Array.Empty<string>();
    public string? EmbeddingModelId { get; init; }
    public string? ExtractorId { get; init; }
    public DateTimeOffset CreatedAt { get; init; }
    public DateTimeOffset UpdatedAt { get; init; }
}

public sealed record KnowledgeSearchOptions
{
    /// <summary>Required scope. An empty list searches nothing; callers explicitly select collections.</summary>
    public IReadOnlyList<string> CollectionIds { get; init; } = Array.Empty<string>();
    public int Limit { get; init; } = 8;
    public int MaxCharacters { get; init; } = 12000;
}

/// <summary>A bounded view of indexed text; extraction status describes whether the original was fully indexed.</summary>
public sealed record KnowledgeDocumentPreview(KnowledgeDocument Document, string Text, bool Truncated,
    int ChunkIndex, KnowledgeLocation? Location, int? PreviousChunkIndex, int? NextChunkIndex);

public sealed record KnowledgeSearchResult
{
    public required string Reference { get; init; }
    public required string CollectionId { get; init; }
    public required string DocumentId { get; init; }
    public required string FileName { get; init; }
    public string? RelativePath { get; init; }
    public int Version { get; init; }
    public int ChunkIndex { get; init; }
    /// <summary>Total chunks in this source version, for bounded follow-up navigation.</summary>
    public int DocumentChunkCount { get; init; }
    public required string Text { get; init; }
    public KnowledgeLocation? Location { get; init; }
    public string? Modality { get; init; }
    public double Score { get; init; }
    public KnowledgeDocumentStatus Status { get; init; }
    public bool Truncated { get; init; }
}

public sealed class KnowledgeLimitException(string message) : InvalidOperationException(message);
public sealed class KnowledgeStaleReferenceException(string message) : InvalidOperationException(message);
