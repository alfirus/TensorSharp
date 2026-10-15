// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML;

public static partial class GgmlEmbeddingNative
{
    private const string DllName = "GgmlOps";
    static GgmlEmbeddingNative() => GgmlNative.EnsureImportResolverRegistered();

    public static string LastError(string fallback) => GgmlNative.LastNativeError(fallback);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    public static partial IntPtr TSGgml_EmbeddingLoad(string path, string backend, int device, int threads);

    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    public static partial void TSGgml_EmbeddingFree(IntPtr handle);

    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static unsafe partial int TSGgml_EmbeddingEncode(IntPtr handle, int* tokens, int* lengths,
        int batch, float* output, int capacity);

    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static unsafe partial int TSGgml_EmbeddingTokenRows(IntPtr handle, int* tokens, int count,
        float* output, int capacity);

    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static unsafe partial int TSGgml_EmbeddingEncodeInputs(IntPtr handle, float* inputs, int inputCapacity,
        int* lengths, int batch, float* output, int outputCapacity);

    public static unsafe void TokenRows(IntPtr handle, int[] tokens, float[] output)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        ArgumentNullException.ThrowIfNull(output);
        fixed (int* t = tokens)
        fixed (float* o = output)
            if (TSGgml_EmbeddingTokenRows(handle, t, tokens.Length, o, output.Length) != 0)
                throw new InvalidOperationException(LastError("Cannot read embedding token rows."));
    }

    public static unsafe void EncodeInputs(IntPtr handle, float[] inputs, int[] lengths, float[] output)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(lengths);
        ArgumentNullException.ThrowIfNull(output);
        fixed (float* x = inputs)
        fixed (int* l = lengths)
        fixed (float* o = output)
            if (TSGgml_EmbeddingEncodeInputs(handle, x, inputs.Length, l, lengths.Length, o, output.Length) != 0)
                throw new InvalidOperationException(LastError("Embedding input inference failed."));
    }

    public static unsafe void Encode(IntPtr handle, int[] tokens, int[] lengths, float[] output)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        ArgumentNullException.ThrowIfNull(lengths);
        ArgumentNullException.ThrowIfNull(output);
        long count = 0;
        foreach (int length in lengths)
        {
            if (length <= 0) throw new ArgumentException("Every sequence must have at least one token.", nameof(lengths));
            count += length;
        }
        if (count != tokens.Length) throw new ArgumentException("Lengths do not match the token buffer.", nameof(lengths));
        fixed (int* t = tokens)
        fixed (int* l = lengths)
        fixed (float* o = output)
            if (TSGgml_EmbeddingEncode(handle, t, l, lengths.Length, o, output.Length) != 0)
                throw new InvalidOperationException(GgmlNative.LastNativeError("Embedding inference failed."));
    }
}
