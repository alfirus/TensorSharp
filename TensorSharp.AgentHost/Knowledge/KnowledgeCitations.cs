// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp.Knowledge;

namespace TensorSharp.AgentHost.Knowledge;

/// <summary>Bind answer citations to evidence retrieved within this request.</summary>
public static class KnowledgeCitations
{
    /// <summary>
    /// Retrieval is a candidate set, not proof that an answer used every hit. Return
    /// only documents explicitly referenced by the final answer. Call after URL repair.
    /// A document URL identifies the document, not every neighboring chunk fetched.
    /// </summary>
    public static IReadOnlyList<KnowledgeSearchResult> UsedSources(string? answer, KnowledgeToolScope? scope)
    {
        if (scope == null || string.IsNullOrEmpty(answer)) return Array.Empty<KnowledgeSearchResult>();
        var result = new List<KnowledgeSearchResult>();
        var issued = scope.IssuedCitations;
        var markerSources = KnowledgeTools.CitedMarkers(answer).Where(issued.ContainsKey).Select(marker => issued[marker])
            .Select(source => (source.CollectionId, source.DocumentId, source.Version)).ToHashSet();
        foreach (var document in scope.Sources.GroupBy(source => (source.CollectionId, source.DocumentId, source.Version)))
        {
            var source = document.First();
            string url = KnowledgeTools.SourceUrl(source.CollectionId, source.DocumentId, source.Version);
            var citedChunk = document.FirstOrDefault(hit => hit.Reference != url && ContainsReference(answer, hit.Reference));
            bool citedMarker = markerSources.Contains((source.CollectionId, source.DocumentId, source.Version));
            if (citedChunk == null && !ContainsReference(answer, url) && !citedMarker) continue;
            // Do not endorse deleted, reindexed, or inaccessible originals as current
            // provenance. Only matched documents are opened; no library scan is needed.
            if (!IsCurrentSource(scope, source)) continue;
            result.Add(citedChunk ?? source with
            {
                Reference = url, ChunkIndex = -1, Location = null,
            });
        }
        return result;
    }

    internal static bool IsCurrentSource(KnowledgeToolScope scope, KnowledgeSearchResult source)
    {
        if (!scope.CollectionIds.Contains(source.CollectionId, StringComparer.Ordinal)) return false;
        try { using var original = scope.Store.OpenOriginal(source.CollectionId, source.DocumentId, source.Version); return true; }
        catch (Exception ex) when (ex is KeyNotFoundException or KnowledgeStaleReferenceException or IOException or UnauthorizedAccessException)
        { return false; }
    }

    private static bool ContainsReference(string text, string reference)
    {
        if (string.IsNullOrEmpty(reference)) return false;
        int offset = 0;
        while (offset <= text.Length - reference.Length)
        {
            int at = text.IndexOf(reference, offset, StringComparison.Ordinal);
            if (at < 0) return false;
            int end = at + reference.Length;
            bool starts = at == 0 || char.IsWhiteSpace(text[at - 1]) || "([<\"'`（【“‘「『，。！？；：".Contains(text[at - 1]);
            bool ends = end == text.Length || char.IsWhiteSpace(text[end]) || ")]>\"'`,;!，。！？；：）】”’」』".Contains(text[end])
                || (text[end] == '.' && (end + 1 == text.Length || char.IsWhiteSpace(text[end + 1])));
            if (starts && ends) return true;
            offset = end;
        }
        return false;
    }
}
