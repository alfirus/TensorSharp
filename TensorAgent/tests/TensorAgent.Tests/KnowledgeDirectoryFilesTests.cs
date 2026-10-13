// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorAgent.Core.Hosting;

namespace TensorAgent.Tests;

public sealed class KnowledgeDirectoryFilesTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-directory-" + Guid.NewGuid().ToString("N"));
    private readonly string _folder;

    public KnowledgeDirectoryFilesTests()
    {
        _folder = Path.Combine(_root, "Dataset");
        Directory.CreateDirectory(Path.Combine(_folder, "sub", "deep"));
        File.WriteAllText(Path.Combine(_folder, "data.csv"), "x,y\n1,2");
        File.WriteAllText(Path.Combine(_folder, "sub", "data.csv"), "x,y\n3,4");
        File.WriteAllText(Path.Combine(_folder, "sub", "deep", ".hidden.txt"), "A hidden source");
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, true); } catch (IOException) { }
    }

    [Fact]
    public void AFolderIncludesNestedAndHiddenFilesWithDistinctOriginalPaths()
    {
        var files = KnowledgeDirectoryFiles.Read(_folder);
        Assert.Equal(new[] { "Dataset/data.csv", "Dataset/sub/data.csv", "Dataset/sub/deep/.hidden.txt" },
            files.Select(file => file.RelativePath));
        Assert.All(files, file => { Assert.Null(file.Error); Assert.True(File.Exists(file.FullPath)); });
    }

    [Fact]
    public void AnUnreadableFolderIsAnExplicitFailure()
    {
        var files = KnowledgeDirectoryFiles.Read(Path.Combine(_root, "missing"));
        var failure = Assert.Single(files);
        Assert.Equal("missing", failure.RelativePath);
        Assert.NotEmpty(failure.Error!);
    }

    [SkippableFact]
    public void LinksAreReportedWithoutFollowingFilesOutsideTheFolderOrRecursiveCycles()
    {
        string outside = Path.Combine(_root, "outside.txt");
        File.WriteAllText(outside, "Outside selected folder");
        try
        {
            File.CreateSymbolicLink(Path.Combine(_folder, "link.txt"), outside);
            Directory.CreateSymbolicLink(Path.Combine(_folder, "cycle"), _folder);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or PlatformNotSupportedException)
        {
            Skip.If(true, "Creating symbolic links is unavailable: " + ex.Message);
        }
        var files = KnowledgeDirectoryFiles.Read(_folder);
        Assert.Equal(5, files.Count);
        Assert.All(files.Where(file => file.RelativePath is "Dataset/link.txt" or "Dataset/cycle"),
            file => Assert.Equal("Symbolic links are not imported.", file.Error));
        Assert.DoesNotContain(files, file => file.RelativePath.EndsWith("outside.txt", StringComparison.Ordinal));
    }
}
