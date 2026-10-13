// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Knowledge;

namespace TensorSharp.AgentHost.Knowledge;

/// <summary>A host supplies an isolated, tool-free model pass. Implementations must honor cancellation
/// and await generation cleanup before returning or throwing; they must not invoke an agent loop.</summary>
public interface IKnowledgeReranker
{
    Task<string> RankAsync(string candidateJson, CancellationToken cancellationToken);
    void ObserveOutcome(string outcome, int candidates, long elapsedMilliseconds) { }
}

public sealed record KnowledgeRankedResults(IReadOnlyList<KnowledgeSearchResult> Results, string Order = "retrieval");

/// <summary>One best-effort relevance pass per request, shared by parent/child tool contexts.
/// It can reorder only original candidates; it never creates evidence or changes retrieval scores.</summary>
public sealed class KnowledgeReranking
{
    public const int MaxCandidates = 8;
    public const int MaxInputBytes = 12000;
    public static readonly TimeSpan Timeout = TimeSpan.FromSeconds(10);
    private readonly IKnowledgeReranker _provider;
    private readonly TimeSpan _timeout;
    private int _attempted;
    private string _outcome = "not_attempted";
    private long _elapsedMilliseconds;

    public KnowledgeReranking(IKnowledgeReranker provider) : this(provider, Timeout) { }
    internal KnowledgeReranking(IKnowledgeReranker provider, TimeSpan timeout)
    { _provider = provider ?? throw new ArgumentNullException(nameof(provider)); _timeout = timeout; }

    public bool CanAttempt => Volatile.Read(ref _attempted) == 0;
    /// <summary>Non-sensitive diagnostics. No query, excerpt or model output is recorded.</summary>
    public string Outcome => Volatile.Read(ref _outcome);
    public long ElapsedMilliseconds => Interlocked.Read(ref _elapsedMilliseconds);

    public async Task<KnowledgeRankedResults> RankAsync(string query, IReadOnlyList<KnowledgeSearchResult> candidates,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (candidates.Count < 3 || Interlocked.CompareExchange(ref _attempted, 1, 0) != 0)
            return new(candidates);
        var selected = candidates.Take(MaxCandidates).ToArray();
        var elapsed = Stopwatch.StartNew();
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        deadline.CancelAfter(_timeout);
        try
        {
            Volatile.Write(ref _outcome, "running");
            string input = BuildInput(query, selected);
            // Await the provider itself, not WhenAny/WaitAsync: fallback must not leave
            // a cancelled inference request consuming the same model behind the answer.
            string output = await _provider.RankAsync(input, deadline.Token).ConfigureAwait(false);
            cancellationToken.ThrowIfCancellationRequested();
            deadline.Token.ThrowIfCancellationRequested();
            if (!TryParseOrder(output, selected.Length, out int[] order))
            { Volatile.Write(ref _outcome, "invalid_output"); return new(candidates); }
            var ranked = order.Select(id => selected[id]).Concat(candidates.Skip(selected.Length)).ToArray();
            Volatile.Write(ref _outcome, "applied");
            return new(ranked, "llm_relevance");
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        { Volatile.Write(ref _outcome, "cancelled"); throw; }
        catch (OperationCanceledException) when (deadline.IsCancellationRequested)
        { Volatile.Write(ref _outcome, "timeout"); return new(candidates); }
        catch (Exception error) when (error is not OutOfMemoryException and not StackOverflowException)
        {
            if (cancellationToken.IsCancellationRequested)
            { Volatile.Write(ref _outcome, "cancelled"); cancellationToken.ThrowIfCancellationRequested(); }
            Volatile.Write(ref _outcome, "unavailable"); return new(candidates);
        }
        finally
        {
            Interlocked.Exchange(ref _elapsedMilliseconds, elapsed.ElapsedMilliseconds);
            try { _provider.ObserveOutcome(Outcome, selected.Length, ElapsedMilliseconds); }
            catch (Exception error) when (error is not OutOfMemoryException and not StackOverflowException) { }
        }
    }

    internal static string BuildInput(string query, IReadOnlyList<KnowledgeSearchResult> candidates)
    {
        int excerptCharacters = 800;
        string Serialize() => JsonSerializer.Serialize(new
        {
            query,
            candidates = candidates.Select((candidate, id) => new
            { id, fileName = Prefix(candidate.FileName, 160), modality = Prefix(candidate.Modality ?? "text", 32),
                excerpt = candidate.Modality is "image" or "video"
                    ? MediaExcerpt(candidate.Text, excerptCharacters) : Prefix(candidate.Text, excerptCharacters) }),
        }, JsonOptions);
        string input = Serialize();
        while (Encoding.UTF8.GetByteCount(input) > MaxInputBytes && excerptCharacters > 0)
        { excerptCharacters /= 2; input = Serialize(); }
        // File names and query are also untrusted and can be escape-heavy.
        if (Encoding.UTF8.GetByteCount(input) > MaxInputBytes)
            throw new InvalidOperationException("Reranking input exceeds its bounded context.");
        return input;
    }

    internal static bool TryParseOrder(string? output, int count, out int[] order)
    {
        order = Array.Empty<int>();
        if (string.IsNullOrWhiteSpace(output) || output.Length > 1024) return false;
        try
        {
            using var document = JsonDocument.Parse(output, new JsonDocumentOptions { MaxDepth = 3 });
            var root = document.RootElement;
            if (root.ValueKind != JsonValueKind.Object || root.EnumerateObject().Count() != 1
                || !root.TryGetProperty("scores", out var ids) || ids.ValueKind != JsonValueKind.Array
                || ids.GetArrayLength() != count) return false;
            var seen = new HashSet<int>();
            var parsed = new (int Id, int Score)[count];
            int index = 0;
            foreach (var pair in ids.EnumerateArray())
            {
                if (pair.ValueKind != JsonValueKind.Array || pair.GetArrayLength() != 2
                    || pair[0].ValueKind != JsonValueKind.Number || pair[1].ValueKind != JsonValueKind.Number
                    || !pair[0].TryGetInt32(out int id) || id < 0 || id >= count || !seen.Add(id)
                    || !pair[1].TryGetInt32(out int score) || score is < 0 or > 3) return false;
                parsed[index++] = (id, score);
            }
            // Ordinal usefulness is not a probability. Preserve the deterministic
            // baseline for ties, independent of the order of pairs in model output.
            order = parsed.OrderByDescending(pair => pair.Score).ThenBy(pair => pair.Id).Select(pair => pair.Id).ToArray();
            return true;
        }
        catch (JsonException) { return false; }
    }

    private static string Prefix(string text, int limit)
    {
        int length = Math.Min(text.Length, limit);
        if (length > 0 && char.IsHighSurrogate(text[length - 1])) length--;
        return text[..length];
    }

    // Image/frame captions may put visible text after a long scene description.
    // Preserve both ends so bounded reranking still sees those names and labels.
    private static string MediaExcerpt(string text, int limit)
    {
        const string omission = "\n…\n";
        if (text.Length <= limit || limit <= omission.Length) return Prefix(text, limit);
        string lead = Prefix(text, (limit - omission.Length) / 2);
        int start = text.Length - (limit - omission.Length - lead.Length);
        if (char.IsLowSurrogate(text[start])) start++;
        return lead + omission + text[start..];
    }
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };
}
