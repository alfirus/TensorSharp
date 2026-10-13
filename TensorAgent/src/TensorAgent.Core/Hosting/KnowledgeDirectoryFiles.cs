// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
namespace TensorAgent.Core.Hosting;

/// <summary>A folder snapshot retaining every file's original relative path and explicit scan failures.</summary>
internal static class KnowledgeDirectoryFiles
{
    internal sealed record Entry(string FullPath, string RelativePath, string? Error = null);

    internal static IReadOnlyList<Entry> Read(string directory)
    {
        string root = Path.GetFullPath(directory);
        string rootName = new DirectoryInfo(root).Name;
        if (string.IsNullOrEmpty(rootName)) rootName = "folder";
        var entries = new List<Entry>();
        var remaining = new Stack<string>();
        remaining.Push(root);
        while (remaining.TryPop(out string? current))
        {
            try
            {
                foreach (string item in Directory.EnumerateFileSystemEntries(current).Order(StringComparer.Ordinal))
                {
                    string relative = rootName + "/" + Path.GetRelativePath(root, item).Replace(Path.DirectorySeparatorChar, '/');
                    try
                    {
                        FileAttributes attributes = File.GetAttributes(item);
                        if ((attributes & FileAttributes.ReparsePoint) != 0)
                            entries.Add(new Entry(item, relative, "Symbolic links are not imported."));
                        else if ((attributes & FileAttributes.Directory) != 0)
                            remaining.Push(item);
                        else
                            entries.Add(new Entry(item, relative));
                    }
                    catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                    {
                        entries.Add(new Entry(item, relative, ex.Message));
                    }
                }
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                string relative = current == root ? rootName : rootName + "/" + Path.GetRelativePath(root, current).Replace(Path.DirectorySeparatorChar, '/');
                entries.Add(new Entry(current, relative, ex.Message));
            }
        }
        return entries.OrderBy(entry => entry.RelativePath, StringComparer.Ordinal).ToArray();
    }
}
