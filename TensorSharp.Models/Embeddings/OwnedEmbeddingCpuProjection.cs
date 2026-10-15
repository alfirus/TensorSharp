// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#nullable enable
using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Threading;

namespace TensorSharp.Models.Embeddings;

/// <summary>Encoder-owned library lease for stateless Q8/Q16 VNNI math.
/// Existing managed quantization, buffers and CpuWorkerPool remain the caller's
/// responsibility. No GGML backend or process-global model state is created.</summary>
internal sealed unsafe class OwnedEmbeddingCpuProjection : IDisposable
{
    private nint _library;
    private readonly delegate* unmanaged[Cdecl]<short*, byte*, int, int> _pack;
    private readonly delegate* unmanaged[Cdecl]<sbyte*, int*, float*, byte*, float*, float*, int, int, int, int, int, int> _group;
    internal string LibraryPath { get; }
    internal string LibrarySha256 { get; }
    internal long SuccessfulCalls { get; private set; }
    internal long SuccessfulPackRows { get; private set; }
    internal long SuccessfulGroups { get; private set; }
    internal long SuccessfulRows { get; private set; }
    internal long Float32FallbackRows { get; private set; }

    private OwnedEmbeddingCpuProjection(string path)
    {
        LibraryPath = Path.GetFullPath(path);
        _library = NativeLibrary.Load(LibraryPath);
        try
        {
            if (((delegate* unmanaged[Cdecl]<int>)NativeLibrary.GetExport(_library, "tensorcpu_abi_version"))() != 1)
                throw new InvalidDataException("TensorSharp CPU math ABI must be version 1.");
            if (((delegate* unmanaged[Cdecl]<int>)NativeLibrary.GetExport(_library, "tensorcpu_q16_vnni_supported"))() != 1)
                throw new PlatformNotSupportedException("Owned Q16 projection requires AVX512VNNI/BW/F, AVX2/FMA and OS register support.");
            _pack = (delegate* unmanaged[Cdecl]<short*, byte*, int, int>)NativeLibrary.GetExport(_library, "tensorcpu_q16_vnni_pack");
            _group = (delegate* unmanaged[Cdecl]<sbyte*, int*, float*, byte*, float*, float*, int, int, int, int, int, int>)
                NativeLibrary.GetExport(_library, "tensorcpu_q16_vnni_packed_group");
            using var stream = File.OpenRead(LibraryPath);
            LibrarySha256 = Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
        }
        catch { NativeLibrary.Free(_library); throw; }
    }

    internal static OwnedEmbeddingCpuProjection? TryCreate(out string? fallbackReason)
        => TryCreate(Environment.GetEnvironmentVariable("TENSORSHARP_EMBEDDING_NATIVE_Q16") == "1",
                     Environment.GetEnvironmentVariable("TENSORSHARP_EMBEDDING_Q16_AVX512") == "1",
                     Environment.GetEnvironmentVariable("TENSORSHARP_CPU_NATIVE_LIBRARY"), out fallbackReason);

    // Non-text owners select their own explicit opt-in and activation-precision
    // prerequisite. Never mutate process environment to initialize an owner.
    internal static OwnedEmbeddingCpuProjection? TryCreate(bool requested, bool q16ActivationEnabled,
        string? libraryPath, out string? fallbackReason)
    {
        fallbackReason = "not-requested";
        if (!requested) return null;
        if (!q16ActivationEnabled)
        {
            fallbackReason = "Q16 activation experiment not enabled";
            return null;
        }
        if (!ManagedEmbeddingQ8Q16Avx512Matrix.IsSupported || Environment.GetEnvironmentVariable("TS_CPU_DISABLE_AVX512") == "1")
        {
            fallbackReason = "managed CPU feature/disable guard rejected AVX512";
            return null;
        }
        string name = OperatingSystem.IsWindows() ? "TensorSharpCpu.dll" : OperatingSystem.IsMacOS() ? "libTensorSharpCpu.dylib" : "libTensorSharpCpu.so";
        string path = string.IsNullOrEmpty(libraryPath) ? Path.Combine(AppContext.BaseDirectory, name) : libraryPath;
        try
        {
            var math = new OwnedEmbeddingCpuProjection(path);
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

    internal void Pack(short* input, byte* output, int count)
    {
        ObjectDisposedException.ThrowIf(_library == 0, this);
        int result = _pack(input, output, count);
        if (result != 0) throw new InvalidOperationException("Owned Q16 activation packing failed: " + result);
    }

    // Caller proves pinned buffer ranges. A rejected ABI call is a real failure,
    // never silently reclassified as successful native execution.
    internal void Group(sbyte* weights, int* sums, float* weightScales, byte* inputs, float* inputScales,
        float* output, int blocks, int inputStride, int outputStride, int rows, int columns)
    {
        ObjectDisposedException.ThrowIf(_library == 0, this);
        int result = _group(weights, sums, weightScales, inputs, inputScales, output,
            blocks, inputStride, outputStride, rows, columns);
        if (result != 0) throw new InvalidOperationException("Owned Q16 projection group failed: " + result);
    }

    // The embedding owner serializes Multiply calls; record only after both
    // synchronous worker stages and any subnormal F32 correction succeed.
    internal void RecordMultiply(int rows, int groups, int fallbackRows)
    {
        ++SuccessfulCalls;
        SuccessfulPackRows += rows;
        SuccessfulGroups += groups;
        SuccessfulRows += rows;
        Float32FallbackRows += fallbackRows;
    }

    public void Dispose()
    {
        nint handle = Interlocked.Exchange(ref _library, 0);
        if (handle != 0) NativeLibrary.Free(handle);
    }
}
