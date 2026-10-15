// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML;

/// <summary>Immutable F32 projection descriptors for a Gemma audio block.</summary>
public sealed record GgmlGemma4AudioLinear(Tensor Weight, Tensor Bias = null,
    bool HasClamp = false, float InputMinimum = float.MinValue, float InputMaximum = float.MaxValue,
    float OutputMinimum = float.MinValue, float OutputMaximum = float.MaxValue,
    IntPtr CompressedWeight = default, int CompressedWeightType = 0, long CompressedWeightBytes = 0);

/// <summary>One immutable layer in a conformer stack. Norm and linear ordering
/// and cache ownership follow <see cref="GgmlGemma4AudioOps.TryConformerBlock"/>.
/// RequireFloat32Precision selects scoped F32 learned projections on qualified
/// Metal/CUDA graphs and the optional qualified Linux CPU projection route.
/// Other ordinary ggml operations retain backend arithmetic.</summary>
public sealed record GgmlGemma4AudioBlock(Tensor Relative, Tensor[] Norms, GgmlGemma4AudioLinear[] Linears,
    int MaximumPast, int MaximumFuture, int Kernel, float Epsilon, float Cap,
    float GradientClip = 1e10f, float ResidualWeight = .5f, bool RequireFloat32Precision = false);

/// <summary>One immutable SSCP convolution, with kernel order
/// [output channel,input channel,kernel time,kernel frequency].</summary>
public sealed record GgmlGemma4AudioConvolution(Tensor Weight, int InputChannels, int OutputChannels,
    int KernelFrequency, int KernelTime, Tensor NormWeight = null, Tensor NormBias = null);

/// <summary>Bounded F32 audio operations built from unchanged upstream ggml
/// primitives in TensorSharp-owned native graphs.</summary>
public static class GgmlGemma4AudioOps
{
    public static bool TryAttention(Tensor result, Tensor query, Tensor key, Tensor value, Tensor relative,
        int sequence, int heads, int head, int maximumPast, int maximumFuture, float cap)
        => GgmlBasicOps.TryGemma4AudioAttention(result, query, key, value, relative,
            sequence, heads, head, maximumPast, maximumFuture, cap);

    public static bool TryDepthwise(Tensor result, Tensor input, Tensor weight, int sequence, int channels, int kernel)
        => GgmlBasicOps.TryGemma4AudioDepthwise(result, input, weight, sequence, channels, kernel);

    /// <summary>One graph submission for a complete F32 conformer block.
    /// Norms: FF1 pre/post, attention pre/post, convolution pre/post, FF2 pre/post,
    /// block, per-head dimension scale, causal depthwise weight.
    /// Linears: FF1 up/down, Q/K/V/output, convolution PW1/PW2, FF2 up/down.
    /// A missing norm is identity, matching the audio encoder's optional weights.
    /// Weight descriptors are immutable and cached; call InvalidateTensorHostBuffer
    /// on each weight before mutation or disposal. Optional Q8_0 buffers retain
    /// their original blocks; invalidate their host pointer before freeing it.
    /// requireFloat32Precision selects backend-qualified F32 learned projections;
    /// it does not change process-wide modes or promise exact ordinary ggml arithmetic.</summary>
    public static bool TryConformerBlock(Tensor result, Tensor input, Tensor relative,
        Tensor[] norms, GgmlGemma4AudioLinear[] linears, int sequence, int heads, int head,
        int maximumPast, int maximumFuture, int kernel, float epsilon, float cap,
        float gradientClip = 1e10f, float residualWeight = .5f, bool requireFloat32Precision = false)
        => GgmlBasicOps.TryGemma4AudioConformerBlock(result, input, relative, norms, linears,
            sequence, heads, head, maximumPast, maximumFuture, kernel, epsilon, cap, gradientClip, residualWeight, requireFloat32Precision);

    /// <summary>Submit one graph for all layers, keeping intermediate hidden
    /// states on the device. Immutable weights must be invalidated before
    /// mutation/disposal as described by <see cref="TryConformerBlock"/>.</summary>
    public static bool TryConformerStack(Tensor result, Tensor input, GgmlGemma4AudioBlock[] blocks,
        int sequence, int heads, int head)
        => GgmlBasicOps.TryGemma4AudioConformerStack(result, input, blocks, sequence, heads, head);

    /// <summary>Two stride-two, padding-one convolutions with channel LayerNorm
    /// and ReLU, followed by the learned projection. Input rows are
    /// [time,frequency*channels+channel]. Weights are immutable and require
    /// cache invalidation before mutation/disposal, as for conformer blocks.
    /// requireFloat32Precision selects scoped F32 convolution/projection operands
    /// on qualified Metal/CUDA graphs.</summary>
    public static bool TrySscp(Tensor result, Tensor input, GgmlGemma4AudioConvolution[] convolutions,
        GgmlGemma4AudioLinear projection, int frequency, int time, int channels, float epsilon, bool requireFloat32Precision = false)
        => GgmlBasicOps.TryGemma4AudioSscp(result, input, convolutions, projection, frequency, time, channels, epsilon, requireFloat32Precision);
}

[StructLayout(LayoutKind.Sequential)]
internal struct GgmlGemma4AudioLinearView
{
    public GgmlTensorView2D Weight, Bias;
    public int HasClamp;
    public float InputMinimum, InputMaximum, OutputMinimum, OutputMaximum;
    public IntPtr CompressedWeight;
    public int CompressedWeightType;
    public long CompressedWeightBytes;
}

[StructLayout(LayoutKind.Sequential)]
internal struct GgmlGemma4AudioBlockArgs
{
    public int StructBytes;
    public GgmlTensorView2D Result, Input, Relative;
    public IntPtr Weights, Linears;
    public int NumWeights, NumLinears, LinearStructBytes;
    public int Sequence, Heads, Head, MaximumPast, MaximumFuture, Kernel;
    public float Epsilon, Cap, GradientClip, ResidualWeight;
    public int RequireFloat32Precision, PrecisionPolicyVersion;
}

[StructLayout(LayoutKind.Sequential)]
internal struct GgmlGemma4AudioConvolutionView
{
    public GgmlTensorView2D Weight, NormWeight, NormBias;
    public int InputChannels, OutputChannels, KernelFrequency, KernelTime;
}

[StructLayout(LayoutKind.Sequential)]
internal struct GgmlGemma4AudioSscpArgs
{
    public int StructBytes;
    public GgmlTensorView2D Result, Input;
    public IntPtr Convolutions, Projection;
    public int ConvolutionStructBytes, LinearStructBytes;
    public int Frequency, Time, Channels;
    public float Epsilon;
    public int RequireFloat32Precision;
}

public partial class GgmlBasicOps
{
    internal static unsafe bool TryGemma4AudioSscp(Tensor result, Tensor input,
        GgmlGemma4AudioConvolution[] convolutions, GgmlGemma4AudioLinear projection,
        int frequency, int time, int channels, float epsilon, bool requireFloat32Precision)
    {
        if (frequency <= 0 || time <= 0 || channels <= 0 || (long)frequency * channels > int.MaxValue ||
            !float.IsFinite(epsilon) || epsilon <= 0 || convolutions?.Length != 2 || projection?.Weight?.DimensionCount != 2 ||
            !AudioMatrix(input, time, frequency * channels) || !TryCreateStandardView(input, out GgmlTensorView2D inputView)) return false;
        var temporaryViews = new System.Collections.Generic.List<Tensor>();
        try
        {
            bool View(Tensor tensor, int rows, int columns, bool optional, out GgmlTensorView2D view)
            {
                view = default;
                if (tensor == null) return optional;
                if ((rows == 1 && tensor.DimensionCount == 1 && tensor.Sizes[0] == columns) ||
                    (tensor.DimensionCount == 4 && tensor.ElementCount() == (long)rows * columns && tensor.IsContiguous()))
                {
                    tensor = tensor.View(rows, columns);
                    temporaryViews.Add(tensor);
                }
                return AudioMatrix(tensor, rows, columns) && TryCreateStandardView(tensor, out view);
            }
            int f = frequency, t = time, c = channels;
            var views = new GgmlGemma4AudioConvolutionView[2];
            for (int i = 0; i < views.Length; ++i)
            {
                var convolution = convolutions[i];
                if (convolution == null || convolution.InputChannels != c || convolution.OutputChannels <= 0 ||
                    convolution.KernelFrequency <= 0 || convolution.KernelTime <= 0 ||
                    (long)c * convolution.KernelFrequency * convolution.KernelTime > int.MaxValue ||
                    f + 2L < convolution.KernelFrequency || t + 2L < convolution.KernelTime) return false;
                int columns = c * convolution.KernelFrequency * convolution.KernelTime;
                if (!View(convolution.Weight, convolution.OutputChannels, columns, false, out views[i].Weight) ||
                    !View(convolution.NormWeight, 1, convolution.OutputChannels, true, out views[i].NormWeight) ||
                    !View(convolution.NormBias, 1, convolution.OutputChannels, true, out views[i].NormBias)) return false;
                views[i].InputChannels = c; views[i].OutputChannels = convolution.OutputChannels;
                views[i].KernelFrequency = convolution.KernelFrequency; views[i].KernelTime = convolution.KernelTime;
                f = checked((int)((f + 2L - convolution.KernelFrequency) / 2 + 1));
                t = checked((int)((t + 2L - convolution.KernelTime) / 2 + 1));
                c = convolution.OutputChannels;
                if (f <= 0 || t <= 0 || (long)f * t > int.MaxValue || (long)f * c > int.MaxValue) return false;
            }
            long outputColumns = projection.Weight.Sizes[0];
            if (outputColumns is <= 0 or > int.MaxValue || !AudioMatrix(result, t, (int)outputColumns) ||
                !TryCreateStandardView(result, out GgmlTensorView2D resultView)) return false;
            var linear = new GgmlGemma4AudioLinearView
            {
                HasClamp = projection.HasClamp ? 1 : 0,
                InputMinimum = projection.InputMinimum, InputMaximum = projection.InputMaximum,
                OutputMinimum = projection.OutputMinimum, OutputMaximum = projection.OutputMaximum,
                CompressedWeight = projection.CompressedWeight, CompressedWeightType = projection.CompressedWeightType,
                CompressedWeightBytes = projection.CompressedWeightBytes,
            };
            if (!View(projection.Weight, (int)outputColumns, f * c, false, out linear.Weight) ||
                !View(projection.Bias, 1, (int)outputColumns, true, out linear.Bias)) return false;
            fixed (GgmlGemma4AudioConvolutionView* pointer = views)
            {
                var args = new GgmlGemma4AudioSscpArgs
                {
                    StructBytes = sizeof(GgmlGemma4AudioSscpArgs), Result = resultView, Input = inputView,
                    Convolutions = (IntPtr)pointer, Projection = (IntPtr)(&linear),
                    ConvolutionStructBytes = sizeof(GgmlGemma4AudioConvolutionView), LinearStructBytes = sizeof(GgmlGemma4AudioLinearView),
                    Frequency = frequency, Time = time, Channels = channels, Epsilon = epsilon, RequireFloat32Precision = requireFloat32Precision ? 1 : 0,
                };
                GgmlNative.Gemma4AudioSscp(in args);
            }
            return true;
        }
        finally { foreach (var tensor in temporaryViews) tensor.Dispose(); }
    }

    internal static unsafe bool TryGemma4AudioConformerBlock(Tensor result, Tensor input, Tensor relative,
        Tensor[] norms, GgmlGemma4AudioLinear[] linears, int sequence, int heads, int head,
        int maximumPast, int maximumFuture, int kernel, float epsilon, float cap, float gradientClip, float residualWeight, bool requireFloat32Precision)
        => TryGemma4AudioConformerStack(result, input,
            [new(relative, norms, linears, maximumPast, maximumFuture, kernel, epsilon, cap, gradientClip, residualWeight, requireFloat32Precision)],
            sequence, heads, head);

    internal static unsafe bool TryGemma4AudioConformerStack(Tensor result, Tensor input, GgmlGemma4AudioBlock[] blocks,
        int sequence, int heads, int head)
    {
        if (sequence <= 0 || heads <= 0 || head <= 0 || heads > int.MaxValue / head ||
            blocks == null || blocks.Length is < 1 or > 128) return false;
        int hidden = heads * head;
        if (!AudioMatrix(result, sequence, hidden) || !AudioMatrix(input, sequence, hidden) ||
            !TryCreateStandardView(result, out GgmlTensorView2D outputView) ||
            !TryCreateStandardView(input, out GgmlTensorView2D inputView)) return false;
        var temporaryViews = new System.Collections.Generic.List<Tensor>();
        var pinned = new System.Collections.Generic.List<GCHandle>();
        try
        {
            bool View(Tensor tensor, int rows, int columns, bool optional, out GgmlTensorView2D desc)
            {
                desc = default;
                if (tensor == null) return optional;
                if (rows == 1 && tensor.DimensionCount == 1 && tensor.Sizes[0] == columns)
                {
                    tensor = tensor.View(1, columns);
                    temporaryViews.Add(tensor);
                }
                return AudioMatrix(tensor, rows, columns) && TryCreateStandardView(tensor, out desc);
            }
            var arguments = new GgmlGemma4AudioBlockArgs[blocks.Length];
            for (int layer = 0; layer < blocks.Length; ++layer)
            {
            var block = blocks[layer];
            if (block == null) return false;
            var norms = block.Norms; var linears = block.Linears;
            int maximumPast = block.MaximumPast, maximumFuture = block.MaximumFuture, kernel = block.Kernel;
            float epsilon = block.Epsilon, cap = block.Cap, gradientClip = block.GradientClip, residualWeight = block.ResidualWeight;
            if (maximumPast < 0 || maximumFuture < 0 || (long)maximumPast + maximumFuture + 1 > int.MaxValue || kernel <= 0 ||
                !float.IsFinite(epsilon) || epsilon <= 0 || !float.IsFinite(cap) || cap <= 0 ||
                !float.IsFinite(gradientClip) || gradientClip <= 0 || !float.IsFinite(residualWeight) ||
                norms == null || norms.Length != 11 || linears == null || linears.Length != 10 ||
                linears[0]?.Weight?.DimensionCount != 2 || linears[8]?.Weight?.DimensionCount != 2 ||
                !AudioMatrix(block.Relative, maximumPast + maximumFuture + 1, hidden) ||
                !TryCreateStandardView(block.Relative, out GgmlTensorView2D relativeView)) return false;
            var normViews = new GgmlTensorView2D[11];
            for (int i = 0; i < norms.Length; ++i)
                if (!View(norms[i], i == 10 ? hidden : 1, i == 10 ? kernel : i == 9 ? head : hidden,
                    true, out normViews[i])) return false;
            int ff1 = checked((int)linears[0].Weight.Sizes[0]), ff2 = checked((int)linears[8].Weight.Sizes[0]);
            if (ff1 <= 0 || ff2 <= 0) return false;
            int[] ins = [hidden, ff1, hidden, hidden, hidden, hidden, hidden, hidden, hidden, ff2];
            int[] outs = [ff1, hidden, hidden, hidden, hidden, hidden, checked(hidden * 2), hidden, ff2, hidden];
            var linearViews = new GgmlGemma4AudioLinearView[10];
            for (int i = 0; i < linears.Length; ++i)
            {
                var linear = linears[i];
                if (linear == null || !View(linear.Weight, outs[i], ins[i], false, out linearViews[i].Weight) ||
                    !View(linear.Bias, 1, outs[i], true, out linearViews[i].Bias)) return false;
                linearViews[i].HasClamp = linear.HasClamp ? 1 : 0;
                linearViews[i].InputMinimum = linear.InputMinimum;
                linearViews[i].InputMaximum = linear.InputMaximum;
                linearViews[i].OutputMinimum = linear.OutputMinimum;
                linearViews[i].OutputMaximum = linear.OutputMaximum;
                linearViews[i].CompressedWeight = linear.CompressedWeight;
                linearViews[i].CompressedWeightType = linear.CompressedWeightType;
                linearViews[i].CompressedWeightBytes = linear.CompressedWeightBytes;
            }
            pinned.Add(GCHandle.Alloc(normViews, GCHandleType.Pinned));
            IntPtr weightPointer = pinned[^1].AddrOfPinnedObject();
            pinned.Add(GCHandle.Alloc(linearViews, GCHandleType.Pinned));
                arguments[layer] = new GgmlGemma4AudioBlockArgs
                {
                    StructBytes = sizeof(GgmlGemma4AudioBlockArgs), Result = outputView, Input = inputView, Relative = relativeView,
                    Weights = weightPointer, Linears = pinned[^1].AddrOfPinnedObject(), NumWeights = normViews.Length, NumLinears = linearViews.Length,
                    LinearStructBytes = sizeof(GgmlGemma4AudioLinearView),
                    Sequence = sequence, Heads = heads, Head = head, MaximumPast = maximumPast, MaximumFuture = maximumFuture,
                    Kernel = kernel, Epsilon = epsilon, Cap = cap, GradientClip = gradientClip, ResidualWeight = residualWeight,
                    RequireFloat32Precision = block.RequireFloat32Precision ? 1 : 0, PrecisionPolicyVersion = 1,
                };
            }
            fixed (GgmlGemma4AudioBlockArgs* pointer = arguments)
                GgmlNative.Gemma4AudioConformerStack((IntPtr)pointer, arguments.Length);
            return true;
        }
        finally
        {
            foreach (var pin in pinned) pin.Free();
            foreach (var tensor in temporaryViews) tensor.Dispose();
        }
    }

    internal static bool TryGemma4AudioAttention(Tensor result, Tensor query, Tensor key, Tensor value, Tensor relative,
        int sequence, int heads, int head, int maximumPast, int maximumFuture, float cap)
    {
        if (sequence <= 0 || heads <= 0 || head <= 0 || maximumPast < 0 || maximumFuture < 0 ||
            !float.IsFinite(cap) || cap <= 0) return false;
        int hidden = checked(heads * head), window = checked(maximumPast + maximumFuture + 1);
        if (!AudioMatrix(result, sequence, hidden) || !AudioMatrix(query, sequence, hidden) ||
            !AudioMatrix(key, sequence, hidden) || !AudioMatrix(value, sequence, hidden) ||
            !AudioMatrix(relative, window, hidden)) return false;
        if (!TryCreateStandardView(result, out GgmlTensorView2D outputView) ||
            !TryCreateStandardView(query, out GgmlTensorView2D queryView) ||
            !TryCreateStandardView(key, out GgmlTensorView2D keyView) ||
            !TryCreateStandardView(value, out GgmlTensorView2D valueView) ||
            !TryCreateStandardView(relative, out GgmlTensorView2D relativeView)) return false;
        GgmlNative.Gemma4AudioAttention(outputView, queryView, keyView, valueView, relativeView,
            sequence, heads, head, maximumPast, maximumFuture, cap);
        return true;
    }

    internal static bool TryGemma4AudioDepthwise(Tensor result, Tensor input, Tensor weight, int sequence, int channels, int kernel)
    {
        if (sequence <= 0 || channels <= 0 || kernel <= 0 ||
            !AudioMatrix(result, sequence, channels) || !AudioMatrix(input, sequence, channels) ||
            !AudioMatrix(weight, channels, kernel)) return false;
        if (!TryCreateStandardView(result, out GgmlTensorView2D outputView) ||
            !TryCreateStandardView(input, out GgmlTensorView2D inputView) ||
            !TryCreateStandardView(weight, out GgmlTensorView2D weightView)) return false;
        GgmlNative.Gemma4AudioDepthwise(outputView, inputView, weightView, sequence, channels, kernel);
        return true;
    }

    private static bool AudioMatrix(Tensor tensor, int rows, int columns) =>
        tensor != null && tensor.Storage is GgmlStorage && tensor.ElementType == DType.Float32 &&
        tensor.DimensionCount == 2 && tensor.Sizes[0] == rows && tensor.Sizes[1] == columns;
}

internal static partial class GgmlNative
{
    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static partial int TSGgml_Gemma4AudioSscpF32(in GgmlGemma4AudioSscpArgs args);

    internal static void Gemma4AudioSscp(in GgmlGemma4AudioSscpArgs args)
        => CheckResult(TSGgml_Gemma4AudioSscpF32(in args), "Gemma4 audio F32 SSCP");

    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static partial int TSGgml_Gemma4AudioConformerStackF32(IntPtr blocks, int count);

    internal static void Gemma4AudioConformerStack(IntPtr blocks, int count)
        => CheckResult(TSGgml_Gemma4AudioConformerStackF32(blocks, count), "Gemma4 audio F32 conformer stack");

    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static partial int TSGgml_Gemma4AudioConformerBlockF32(in GgmlGemma4AudioBlockArgs args);

    internal static void Gemma4AudioConformerBlock(in GgmlGemma4AudioBlockArgs args)
        => CheckResult(TSGgml_Gemma4AudioConformerBlockF32(in args), "Gemma4 audio F32 conformer block");

    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static partial int TSGgml_Gemma4AudioAttentionF32(GgmlTensorView2D result,
        GgmlTensorView2D query, GgmlTensorView2D key, GgmlTensorView2D value, GgmlTensorView2D relative,
        int sequence, int heads, int head, int maximumPast, int maximumFuture, float cap);

    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static partial int TSGgml_Gemma4AudioDepthwiseF32(GgmlTensorView2D result,
        GgmlTensorView2D input, GgmlTensorView2D weight, int sequence, int channels, int kernel);

    internal static void Gemma4AudioAttention(GgmlTensorView2D result, GgmlTensorView2D query,
        GgmlTensorView2D key, GgmlTensorView2D value, GgmlTensorView2D relative,
        int sequence, int heads, int head, int maximumPast, int maximumFuture, float cap)
        => CheckResult(TSGgml_Gemma4AudioAttentionF32(result, query, key, value, relative,
            sequence, heads, head, maximumPast, maximumFuture, cap), "Gemma4 audio relative attention");

    internal static void Gemma4AudioDepthwise(GgmlTensorView2D result, GgmlTensorView2D input,
        GgmlTensorView2D weight, int sequence, int channels, int kernel)
        => CheckResult(TSGgml_Gemma4AudioDepthwiseF32(result, input, weight, sequence, channels, kernel),
            "Gemma4 audio causal depthwise convolution");
}
