// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorAgent.Core.Hosting;

namespace TensorAgent.Tests;

public sealed class KnowledgeSourceDownloadTests
{
    private const string PageUrl = "http://127.0.0.1:43210/?token=launch";
    private const string Route = "/api/knowledge/collections/c_1/documents/d-2/source?version=7";

    [Theory]
    [InlineData(Route)]
    [InlineData("http://127.0.0.1:43210" + Route)]
    public void AppSourceRoutesRetainTheirVersionAndIdentities(string url)
    {
        Assert.True(KnowledgeSourceDownload.TryParse(url, PageUrl, out var source));
        Assert.Equal(new KnowledgeSourceDownload.Source("c_1", "d-2", 7), source);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("/api/knowledge/collections/c/documents/d/source")]
    [InlineData("/api/knowledge/collections/c/documents/d/source?version=0")]
    [InlineData("/api/knowledge/collections/c/documents/d/source?version=-1")]
    [InlineData("/api/knowledge/collections/c/documents/d/source?version=999999999999")]
    [InlineData("/api/knowledge/collections/c/documents/d/source?version=1&other=2")]
    [InlineData("/api/knowledge/collections/c/documents/d/source?version=1#original")]
    [InlineData("/api/knowledge/collections/c/documents/../source?version=1")]
    [InlineData("/api/knowledge/collections/c/documents/%2e%2e/source?version=1")]
    [InlineData("/api/knowledge/collections/c%2Fd/documents/d/source?version=1")]
    [InlineData("/api/knowledge/collections/c/documents/d%5Cfile/source?version=1")]
    [InlineData("/api/knowledge/collections/c/documents/d:stream/source?version=1")]
    [InlineData("/api/knowledge/collections/c/documents/d/source/extra?version=1")]
    [InlineData("/api/knowledge/collections/c/documents//source?version=1")]
    [InlineData("/api/code/artifacts/run/file.txt?version=1")]
    [InlineData("//127.0.0.1:43210" + Route)]
    [InlineData("http://127.0.0.1:43211" + Route)]
    [InlineData("https://127.0.0.1:43210" + Route)]
    [InlineData("http://example.com" + Route)]
    [InlineData("http://user@127.0.0.1:43210" + Route)]
    [InlineData("http://127.0.0.1:43210/api/knowledge/collections/c/documents/../documents/d/source?version=1")]
    [InlineData("data:text/html,test")]
    public void InvalidOriginsTraversalAndUnversionedDownloadsAreRejected(string? url)
    {
        Assert.False(KnowledgeSourceDownload.TryParse(url, PageUrl, out var source));
        Assert.Null(source);
    }
}
