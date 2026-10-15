// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;

namespace TensorSharp.MLX;

/// <summary>Device-resident zero-padded spatial columns for LTX convolutional VAEs.</summary>
public static class MlxLtx25VaeOps
{
    /// <summary>Join independently evaluated [1,H,W,C] frame buffers with one
    /// native concatenate, avoiding repeated updates of a whole video parent.</summary>
    public static bool TryConcatenateFrames(Tensor output, IReadOnlyList<Tensor> frames)
    {
        if (output?.Storage is not MlxStorage target || frames == null || frames.Count == 0 ||
            output.ElementType != DType.Float32 || output.DimensionCount != 4 || !output.IsContiguous() ||
            output.StorageOffset != 0 || output.ElementCount() != output.Storage.ElementCount || output.Sizes[0] != frames.Count)
            return false;
        foreach (var frame in frames)
            if (frame?.Storage is not MlxStorage source || !ReferenceEquals(source.AllocatorImpl, target.AllocatorImpl) ||
                frame.ElementType != DType.Float32 || frame.DimensionCount != 4 || !frame.IsContiguous() ||
                frame.Sizes[0] != 1 || frame.Sizes[1] != output.Sizes[1] || frame.Sizes[2] != output.Sizes[2] || frame.Sizes[3] != output.Sizes[3])
                return false;
        return MlxWorker.Shared.Invoke(() =>
        {
            var arrays = new MlxNative.MlxArray[frames.Count];
            MlxNative.MlxArray result = default;
            try
            {
                for (int i = 0; i < frames.Count; ++i) arrays[i] = ((MlxStorage)frames[i].Storage).CreateArrayView(frames[i]);
                result = MlxNative.Ltx25ConcatenateFrames(arrays);
                target.ReplaceDeviceArray(result); result = default;
                return true;
            }
            finally
            {
                foreach (var array in arrays) MlxNative.FreeArray(array);
                MlxNative.FreeArray(result);
            }
        });
    }

    /// <summary>Space-to-depth convolution output plus grouped-average skip,
    /// including repeated-first-frame temporal padding, entirely on the device.</summary>
    public static bool TryDownsampleResidual(Tensor output, Tensor input, Tensor convolution,
        int temporalStride, int spatialStride)
    {
        if (output?.Storage is not MlxStorage target || input?.Storage is not MlxStorage source ||
            convolution?.Storage is not MlxStorage conv || output.ElementType != DType.Float32 ||
            input.ElementType != DType.Float32 || convolution.ElementType != DType.Float32 ||
            output.DimensionCount != 4 || input.DimensionCount != 4 || convolution.DimensionCount != 4 ||
            !output.IsContiguous() || !input.IsContiguous() || !convolution.IsContiguous() ||
            !ReferenceEquals(target.AllocatorImpl, source.AllocatorImpl) || !ReferenceEquals(target.AllocatorImpl, conv.AllocatorImpl) ||
            temporalStride is not (1 or 2) || spatialStride is not (1 or 2) || output.ElementCount() > int.MaxValue)
            return false;
        int t = checked((int)input.Sizes[0]), h = checked((int)input.Sizes[1]), w = checked((int)input.Sizes[2]);
        int channels = checked((int)input.Sizes[3]), convChannels = checked((int)convolution.Sizes[3]);
        int factor = temporalStride * spatialStride * spatialStride, outputs = checked((int)output.Sizes[3]);
        if (t <= 0 || h <= 0 || w <= 0 || channels <= 0 || convChannels <= 0 || outputs != convChannels * factor ||
            channels * factor % outputs != 0 || (t + temporalStride - 1) % temporalStride != 0 ||
            h % spatialStride != 0 || w % spatialStride != 0 || output.Sizes[0] != (t + temporalStride - 1) / temporalStride ||
            output.Sizes[1] != h / spatialStride || output.Sizes[2] != w / spatialStride ||
            convolution.Sizes[0] != t || convolution.Sizes[1] != h || convolution.Sizes[2] != w) return false;
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray x = default, c = default, result = default;
            try
            {
                x = source.CreateArrayView(input); c = conv.CreateArrayView(convolution);
                result = MlxNative.Ltx25VaeDownsampleResidual(x, c, t, h, w, channels, convChannels, temporalStride, spatialStride);
                if (output.StorageOffset == 0 && output.ElementCount() == output.Storage.ElementCount)
                { target.ReplaceDeviceArray(result); result = default; }
                else target.UpdateDeviceSlice(output, result);
                return true;
            }
            finally { MlxNative.FreeArray(x); MlxNative.FreeArray(c); MlxNative.FreeArray(result); }
        });
    }

    /// <summary>Evaluate pending device work without copying image data to the
    /// host. Frame boundaries keep im2col graphs and streamed weights bounded.</summary>
    public static void Materialize(Tensor tensor)
    {
        if (tensor?.Storage is not MlxStorage storage)
            throw new ArgumentException("An MLX tensor is required.", nameof(tensor));
        MlxWorker.Shared.Invoke(() =>
        {
            var view = storage.CreateArrayView(tensor);
            try { MlxNative.Eval(view); }
            finally { MlxNative.FreeArray(view); }
        });
    }

    public static bool TryIm2Col(Tensor output, Tensor frame, int channels, int height, int width,
        int kernel, int startRow, int rows)
    {
        if (output?.Storage is not MlxStorage target || frame?.Storage is not MlxStorage source ||
            output.ElementType != DType.Float32 || frame.ElementType != DType.Float32 ||
            !output.IsContiguous() || !frame.IsContiguous() ||
            !ReferenceEquals(target.AllocatorImpl, source.AllocatorImpl) ||
            channels <= 0 || height <= 0 || width <= 0 || kernel is not (1 or 3) ||
            startRow < 0 || rows <= 0 || (long)startRow + rows > height ||
            frame.DimensionCount != 3 || frame.Sizes[0] != height || frame.Sizes[1] != width || frame.Sizes[2] != channels ||
            output.DimensionCount != 2 || output.Sizes[0] != (long)rows * width || output.Sizes[1] != (long)channels * kernel * kernel ||
            output.ElementCount() > int.MaxValue) return false;
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray input = default, result = default;
            try
            {
                input = source.CreateArrayView(frame);
                result = MlxNative.Ltx25VaeColumns(input, channels, height, width, kernel, startRow, rows);
                if (output.StorageOffset == 0 && output.ElementCount() == output.Storage.ElementCount)
                { target.ReplaceDeviceArray(result); result = default; }
                else target.UpdateDeviceSlice(output, result);
                return true;
            }
            finally { MlxNative.FreeArray(input); MlxNative.FreeArray(result); }
        });
    }
}

internal static partial class MlxNative
{
    internal static MlxArray Ltx25ConcatenateFrames(MlxArray[] frames)
    {
        var inputs = CreateVectorArray(frames);
        try
        {
            Check(mlx_concatenate_axis(out MlxArray output, inputs, 0, DefaultStream()), "concatenating LTX VAE frames");
            return output;
        }
        finally { if (inputs.IsValid) _ = mlx_vector_array_free(inputs); }
    }

    private static MlxFastMetalKernel ltx25VaeDownsampleKernel;
    private const string Ltx25VaeDownsampleSource = """
        const uint i = thread_position_in_grid.x;
        if (i >= Count) return;
        const int factor = TemporalStride * SpatialStride * SpatialStride;
        const int channel = i % OutputChannels;
        const int pixel = i / OutputChannels;
        const int ox = pixel % OutputWidth;
        const int oy = (pixel / OutputWidth) % OutputHeight;
        const int ot = pixel / (OutputHeight * OutputWidth);
        const int tap = channel % factor;
        const int ct = max(0, ot * TemporalStride + tap / (SpatialStride * SpatialStride) - (TemporalStride - 1));
        const int cy = oy * SpatialStride + (tap / SpatialStride) % SpatialStride;
        const int cx = ox * SpatialStride + tap % SpatialStride;
        float value = conv[((ct * Height + cy) * Width + cx) * ConvChannels + channel / factor];
        float residual = 0.0f;
        for (int g = 0; g < Groups; ++g) {
            const int grouped = channel * Groups + g;
            const int skipTap = grouped % factor;
            const int st = max(0, ot * TemporalStride + skipTap / (SpatialStride * SpatialStride) - (TemporalStride - 1));
            const int sy = oy * SpatialStride + (skipTap / SpatialStride) % SpatialStride;
            const int sx = ox * SpatialStride + skipTap % SpatialStride;
            residual += x[((st * Height + sy) * Width + sx) * Channels + grouped / factor];
        }
        y[i] = value + residual / Groups;
        """;

    internal static MlxArray Ltx25VaeDownsampleResidual(MlxArray input, MlxArray convolution, int frames,
        int height, int width, int channels, int convChannels, int temporalStride, int spatialStride)
    {
        int factor = temporalStride * spatialStride * spatialStride, outputs = checked(convChannels * factor);
        int ot = (frames + temporalStride - 1) / temporalStride, oh = height / spatialStride, ow = width / spatialStride;
        int count = checked(ot * oh * ow * outputs);
        lock (fastKernelSync)
            if (!ltx25VaeDownsampleKernel.IsValid)
                ltx25VaeDownsampleKernel = CreateFastMetalKernel("tensorsharp_ltx25_vae_downsample_residual_f32",
                    ["x", "conv"], ["y"], Ltx25VaeDownsampleSource, string.Empty);
        var config = mlx_fast_metal_kernel_config_new();
        MlxVectorArray inputs = default, result = default;
        try
        {
            AddTemplateInt(config, "Count", count); AddTemplateInt(config, "Channels", channels);
            AddTemplateInt(config, "Height", height); AddTemplateInt(config, "Width", width);
            AddTemplateInt(config, "ConvChannels", convChannels); AddTemplateInt(config, "OutputChannels", outputs);
            AddTemplateInt(config, "TemporalStride", temporalStride); AddTemplateInt(config, "SpatialStride", spatialStride);
            AddTemplateInt(config, "OutputHeight", oh); AddTemplateInt(config, "OutputWidth", ow);
            AddTemplateInt(config, "Groups", channels * factor / outputs);
            int[] shape = [ot, oh, ow, outputs];
            Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring LTX VAE downsample");
            Check(mlx_fast_metal_kernel_config_set_grid(config, count, 1, 1), "configuring LTX VAE downsample grid");
            Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring LTX VAE downsample threads");
            inputs = CreateVectorArray(input, convolution); result = mlx_vector_array_new();
            Check(mlx_fast_metal_kernel_apply(ref result, ltx25VaeDownsampleKernel, inputs, config, DefaultStream()), "running LTX VAE downsample");
            Check(mlx_vector_array_get(out MlxArray output, result, 0), "reading LTX VAE downsample");
            return output;
        }
        finally
        {
            if (inputs.IsValid) _ = mlx_vector_array_free(inputs);
            if (result.IsValid) _ = mlx_vector_array_free(result);
            if (config.IsValid) _ = mlx_fast_metal_kernel_config_free(config);
        }
    }

    private static MlxFastMetalKernel ltx25VaeColumnsKernel;
    private const string Ltx25VaeColumnsSource = """
        const uint i = thread_position_in_grid.x;
        if (i >= Count) return;
        const int taps = Channels * Kernel * Kernel;
        const int row = i / taps, tap = i % taps;
        const int channel = tap / (Kernel * Kernel);
        const int ky = (tap / Kernel) % Kernel, kx = tap % Kernel;
        const int sy = StartRow + row / Width - Kernel / 2 + ky;
        const int sx = row % Width - Kernel / 2 + kx;
        y[i] = sy >= 0 && sy < Height && sx >= 0 && sx < Width
            ? x[(sy * Width + sx) * Channels + channel] : 0.0f;
        """;

    internal static MlxArray Ltx25VaeColumns(MlxArray input, int channels, int height, int width,
        int kernel, int startRow, int rows)
    {
        int count = checked(rows * width * channels * kernel * kernel);
        lock (fastKernelSync)
            if (!ltx25VaeColumnsKernel.IsValid)
                ltx25VaeColumnsKernel = CreateFastMetalKernel("tensorsharp_ltx25_vae_columns_f32",
                    ["x"], ["y"], Ltx25VaeColumnsSource, string.Empty);
        var config = mlx_fast_metal_kernel_config_new();
        MlxVectorArray inputs = default, outputs = default;
        try
        {
            AddTemplateInt(config, "Count", count); AddTemplateInt(config, "Channels", channels);
            AddTemplateInt(config, "Height", height); AddTemplateInt(config, "Width", width);
            AddTemplateInt(config, "Kernel", kernel); AddTemplateInt(config, "StartRow", startRow);
            int[] shape = [rows * width, channels * kernel * kernel];
            Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length,
                ToMlxDtype(DType.Float32)), "configuring LTX VAE columns");
            Check(mlx_fast_metal_kernel_config_set_grid(config, count, 1, 1), "configuring LTX VAE grid");
            Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring LTX VAE threads");
            inputs = CreateVectorArray(input); outputs = mlx_vector_array_new();
            Check(mlx_fast_metal_kernel_apply(ref outputs, ltx25VaeColumnsKernel, inputs, config, DefaultStream()),
                "running LTX VAE columns");
            Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading LTX VAE columns");
            return result;
        }
        finally
        {
            if (inputs.IsValid) _ = mlx_vector_array_free(inputs);
            if (outputs.IsValid) _ = mlx_vector_array_free(outputs);
            if (config.IsValid) _ = mlx_fast_metal_kernel_config_free(config);
        }
    }
}
