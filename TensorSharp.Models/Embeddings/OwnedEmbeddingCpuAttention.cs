// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Threading;

namespace TensorSharp.Models.Embeddings;

/// <summary>Optional stateless TensorSharp-owned CPU attention math. Each
/// encoder holds its own library handle; buffers/jobs remain managed and no
/// GGML backend, model cache or native worker pool is initialized.</summary>
internal sealed unsafe class OwnedEmbeddingCpuAttention : IDisposable
{
    private nint _library;
    private readonly delegate* unmanaged[Cdecl]<float*, float*, float*, int, int, int, int, int, int, float, int> _scores;
    private readonly delegate* unmanaged[Cdecl]<float*, float*, float*, int, int, int, int, int, int, int> _values;
    internal string LibraryPath { get; }
    internal string LibrarySha256 { get; }
    private OwnedEmbeddingCpuAttention(string path)
    {
        LibraryPath = Path.GetFullPath(path);
        _library = NativeLibrary.Load(LibraryPath);
        try
        {
            if (((delegate* unmanaged[Cdecl]<int>)NativeLibrary.GetExport(_library, "tensorcpu_abi_version"))() != 1)
                throw new InvalidDataException("TensorSharp CPU math ABI must be version 1.");
            if (((delegate* unmanaged[Cdecl]<int>)NativeLibrary.GetExport(_library, "tensorcpu_attention_avx512_supported"))() == 0)
                throw new PlatformNotSupportedException("Owned attention requires AVX512F/FMA and OS register support.");
            _scores = (delegate* unmanaged[Cdecl]<float*, float*, float*, int, int, int, int, int, int, float, int>)
                NativeLibrary.GetExport(_library, "tensorcpu_attention_scores8");
            _values = (delegate* unmanaged[Cdecl]<float*, float*, float*, int, int, int, int, int, int, int>)
                NativeLibrary.GetExport(_library, "tensorcpu_attention_values8");
            using var stream = File.OpenRead(LibraryPath);
            LibrarySha256 = Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
        }
        catch { NativeLibrary.Free(_library); throw; }
    }

    internal static OwnedEmbeddingCpuAttention? TryCreate(out string? fallbackReason)
    {
        fallbackReason = "not-requested";
        if (Environment.GetEnvironmentVariable("TENSORSHARP_EMBEDDING_NATIVE_ATTENTION") != "1") return null;
        if (!ManagedEmbeddingAttentionAvx512.IsSupported)
        {
            fallbackReason = "managed CPU feature/disable guard rejected AVX512";
            return null;
        }
        string? explicitPath = Environment.GetEnvironmentVariable("TENSORSHARP_CPU_NATIVE_LIBRARY");
        string name = OperatingSystem.IsWindows() ? "TensorSharpCpu.dll" : OperatingSystem.IsMacOS() ? "libTensorSharpCpu.dylib" : "libTensorSharpCpu.so";
        string path = string.IsNullOrEmpty(explicitPath) ? Path.Combine(AppContext.BaseDirectory, name) : explicitPath;
        try
        {
            var math = new OwnedEmbeddingCpuAttention(path);
            fallbackReason = null;
            return math;
        }
        catch (Exception error) when (error is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException or InvalidDataException or
                                     IOException or UnauthorizedAccessException or PlatformNotSupportedException)
        {
            fallbackReason = error.GetType().Name + ": " + error.Message;
            return null;
        }
    }

    internal static bool SupportsShape(int dimensions, int length)
        => dimensions > 0 && dimensions <= 1024 && dimensions % 16 == 0 && length > 0 && length <= 16384;

    // Callers prove the array ranges and row offsets while pinned. Any rejected
    // ABI call is a real integration failure rather than a silent passing path.
    internal void Scores(float* queries, float* keys, float* output, int queryStride, int keyStride,
        int outputStride, int dimensions, int length, int rows, float scale)
    {
        ObjectDisposedException.ThrowIf(_library == 0, this);
        int result = _scores(queries, keys, output, queryStride, keyStride, outputStride, dimensions, length, rows, scale);
        if (result != 0) throw new InvalidOperationException("Owned CPU attention score tile failed: " + result);
    }
    internal void Values(float* probabilities, float* values, float* output, int probabilityStride, int valueStride,
        int outputStride, int dimensions, int length, int rows)
    {
        ObjectDisposedException.ThrowIf(_library == 0, this);
        int result = _values(probabilities, values, output, probabilityStride, valueStride, outputStride, dimensions, length, rows);
        if (result != 0) throw new InvalidOperationException("Owned CPU attention value tile failed: " + result);
    }
    public void Dispose()
    {
        nint handle = Interlocked.Exchange(ref _library, 0);
        if (handle != 0) NativeLibrary.Free(handle);
    }
}
