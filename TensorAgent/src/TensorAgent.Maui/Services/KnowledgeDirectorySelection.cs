// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorAgent.Core.Hosting;

namespace TensorAgent.Maui.Services;

/// <summary>Retains native folder access until enumeration and import have finished.</summary>
internal abstract class KnowledgeDirectorySelection(string fullPath) : IDisposable
{
    public string FullPath { get; } = fullPath;
    public virtual Task<IReadOnlyList<KnowledgeDirectoryFiles.Entry>> ReadFilesAsync() =>
        Task.Run(() => KnowledgeDirectoryFiles.Read(FullPath));
    public virtual Task<Stream> OpenReadAsync(string fullPath, long maxBytes) => Task.FromResult<Stream>(File.OpenRead(fullPath));
    public abstract void Dispose();
}
