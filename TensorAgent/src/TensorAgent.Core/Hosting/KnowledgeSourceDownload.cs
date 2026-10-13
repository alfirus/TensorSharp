// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Globalization;

namespace TensorAgent.Core.Hosting;

/// <summary>Accepts only versioned original-source routes belonging to this app.</summary>
internal static class KnowledgeSourceDownload
{
    internal sealed record Source(string CollectionId, string DocumentId, int Version);

    internal static bool TryParse(string? urlOrPath, string pageUrl, out Source? source)
    {
        source = null;
        if (string.IsNullOrEmpty(urlOrPath) || urlOrPath.Contains('%')
            || urlOrPath.Contains("/../", StringComparison.Ordinal) || urlOrPath.Contains("/./", StringComparison.Ordinal)
            || !Uri.TryCreate(pageUrl, UriKind.Absolute, out Uri? page))
            return false;
        string pathAndQuery = urlOrPath;
        if (!urlOrPath.StartsWith("/", StringComparison.Ordinal))
        {
            if (!Uri.TryCreate(urlOrPath, UriKind.Absolute, out Uri? absolute)
                || absolute.UserInfo.Length != 0 || absolute.Fragment.Length != 0
                || Uri.Compare(absolute, page, UriComponents.SchemeAndServer, UriFormat.SafeUnescaped,
                    StringComparison.OrdinalIgnoreCase) != 0)
                return false;
            pathAndQuery = absolute.PathAndQuery;
        }
        // Do not normalize dot segments, encoded separators or protocol-relative URLs.
        string[] routeAndQuery = pathAndQuery.Split('?');
        if (routeAndQuery.Length != 2 || !routeAndQuery[1].StartsWith("version=", StringComparison.Ordinal)
            || !int.TryParse(routeAndQuery[1][8..], NumberStyles.None, CultureInfo.InvariantCulture, out int version)
            || version < 1)
            return false;
        string[] parts = routeAndQuery[0].Split('/');
        if (parts.Length != 8 || parts[0].Length != 0 || parts[1] != "api" || parts[2] != "knowledge"
            || parts[3] != "collections" || parts[5] != "documents" || parts[7] != "source"
            || !ValidId(parts[4]) || !ValidId(parts[6]))
            return false;
        source = new Source(parts[4], parts[6], version);
        return true;
    }

    private static bool ValidId(string value) => value.Length is > 0 and <= 128
        && value.All(c => c is >= 'a' and <= 'z' or >= 'A' and <= 'Z' or >= '0' and <= '9' or '_' or '-');
}
