// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Direct;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Ltx25;

/// <summary>The trained causal LTX video VAE encoder. Computes deterministic,
/// per-channel normalized latent means with Float32 activations on CPU, CUDA or
/// MLX. Checkpoint weights are streamed one convolution at a time.</summary>
public sealed class Ltx25VideoVaeEncoder : IDisposable
{
    private readonly SafetensorsFile _file;
    private readonly DirectContext _context;
    private readonly Ltx25VideoVaeOps _ops;
    private readonly string _prefix;
    private readonly int _patchSize, _baseChannels;
    private readonly Ltx25VideoVaeBlock[] _blocks;
    private readonly int _outputChannels;
    private readonly object _gate = new();
    private bool _disposed;
    public int LatentChannels { get; }
    public int TemporalScale { get; }
    public int SpatialScale { get; }

    public Ltx25VideoVaeEncoder(string path, IAllocator allocator)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(allocator);
        if (allocator is not (CpuAllocator or CudaAllocator or MlxAllocator))
            throw new NotSupportedException("LTX video VAE requires a CPU, CUDA or MLX direct allocator.");
        _context = new DirectContext(allocator);
        _file = new SafetensorsFile(path);
        try
        {
            if (!_file.Metadata.TryGetValue("config", out string metadata))
                throw new InvalidDataException("LTX video VAE encoding requires embedded architecture metadata.");
            using var doc = JsonDocument.Parse(metadata);
            var vae = doc.RootElement.GetProperty("vae");
            var config = vae.TryGetProperty("encoder", out var nested) ? nested : vae;
            string Kind(string key, string fallback) => config.TryGetProperty(key, out var value) ? value.GetString() : fallback;
            int Number(string key, int fallback) => config.TryGetProperty(key, out var value) ? value.GetInt32() : fallback;
            if ((vae.TryGetProperty("_class_name", out var className) &&
                 className.GetString() is not ("CausalVideoAutoencoder" or "CausalDiffusionVAE")) ||
                Number("dims", 3) != 3 || Number("in_channels", 3) != 3 ||
                Kind("norm_layer", "pixel_norm") != "pixel_norm" ||
                Kind("spatial_padding_mode", "zeros") != "zeros" ||
                (vae.TryGetProperty("use_quant_conv", out var quant) && quant.GetBoolean()))
                throw new NotSupportedException("LTX video encoding requires a causal convolutional RGB encoder with pixel normalization, zero spatial padding and no quant-conv.");
            LatentChannels = nested.ValueKind == JsonValueKind.Object
                ? Number("out_channels", 128) : Number("latent_channels", 128);
            _patchSize = Number("patch_size", 4);
            // The upstream VideoEncoder starts at latent_channels; the legacy
            // encoder_base_channels metadata field does not alter its graph.
            _baseChannels = LatentChannels;
            string logVariance = Kind("latent_log_var", "uniform");
            _outputChannels = logVariance switch
            {
                "uniform" or "constant" => checked(LatentChannels + 1),
                "per_channel" => checked(LatentChannels * 2),
                _ => throw new NotSupportedException("LTX video encoding requires uniform, constant or per-channel latent variance metadata."),
            };
            if (LatentChannels <= 0 || _patchSize <= 0 || _patchSize > 16)
                throw new InvalidDataException("Invalid LTX VAE encoder channel or patch dimensions.");
            var blocks = new List<Ltx25VideoVaeBlock>();
            var entries = config.TryGetProperty("blocks", out var configuredBlocks) ? configuredBlocks : config.GetProperty("encoder_blocks");
            int temporal = 1, spatial = _patchSize;
            foreach (var entry in entries.EnumerateArray())
            {
                string kind = entry[0].GetString();
                var parameters = entry[1];
                int layers = parameters.ValueKind == JsonValueKind.Number ? parameters.GetInt32()
                    : parameters.TryGetProperty("num_layers", out var n) ? n.GetInt32() : 0;
                int multiplier = parameters.ValueKind == JsonValueKind.Object && parameters.TryGetProperty("multiplier", out var m)
                    ? m.GetInt32() : kind == "res_x" ? 1 : 2;
                if (kind is not ("res_x" or "compress_time_res" or "compress_space_res" or "compress_all_res") ||
                    multiplier <= 0 || (kind == "res_x" && (layers <= 0 || multiplier != 1)) ||
                    (parameters.ValueKind == JsonValueKind.Object &&
                     parameters.TryGetProperty("inject_noise", out var noise) && noise.GetBoolean()))
                    throw new NotSupportedException("Unsupported LTX convolutional video VAE encoder block.");
                blocks.Add(new(kind, layers, multiplier));
                if (kind is "compress_time_res" or "compress_all_res") temporal = checked(temporal * 2);
                if (kind is "compress_space_res" or "compress_all_res") spatial = checked(spatial * 2);
            }
            if (blocks.Count == 0) throw new InvalidDataException("LTX VAE encoder blocks are empty.");
            _blocks = blocks.ToArray(); TemporalScale = temporal; SpatialScale = spatial;
            _prefix = _file.HasTensor("encoder.conv_in.conv.weight") ? "encoder." : "";
            ValidateWeights();
            _ops = new Ltx25VideoVaeOps(_context, _file, _prefix);
        }
        catch { _file.Dispose(); _context.Dispose(); throw; }
    }

    /// <summary>Encode contiguous [H,W,3] RGB floats in [0,1] to channel-major
    /// [LatentChannels,1,H/SpatialScale,W/SpatialScale] normalized latent means.</summary>
    public float[] EncodeImage(float[] frameMajorRgb, int height, int width,
        CancellationToken cancellationToken = default, Action<string, float[], int[]>? trace = null)
        => Encode(frameMajorRgb, 1, height, width, cancellationToken, trace);

    /// <summary>Encode a video whose every RGB frame equals the supplied [H,W,3]
    /// image, without allocating or projecting repeated pixels. The validated
    /// causal, repeat-padded convolution/pixel-normalization encoder preserves
    /// temporally constant features through every block. Consequently the
    /// single-frame graph computes every latent frame of this static video.
    /// Float32 tap folding has the same small rounding difference as EncodeImage.
    /// Encode remains the full temporal path for arbitrary videos and oracles.</summary>
    public float[] EncodeStaticVideo(float[] rgbFrame, int frames, int height, int width,
        CancellationToken cancellationToken = default, Action<string, float[], int[]>? trace = null)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (frames <= 0 || (frames - 1) % TemporalScale != 0)
                throw new ArgumentOutOfRangeException(nameof(frames), "LTX static video requires T = 1 + temporal scale*k.");
            cancellationToken.ThrowIfCancellationRequested();
            float[] single = EncodeImage(rgbFrame, height, width, cancellationToken,
                trace == null ? null : (name, values, shape) => trace("static." + name, values, shape));
            int temporal = 1 + (frames - 1) / TemporalScale;
            int spatial = checked(height / SpatialScale * (width / SpatialScale));
            var result = new float[checked(single.Length * temporal)];
            for (int channel = 0; channel < LatentChannels; ++channel)
            {
                cancellationToken.ThrowIfCancellationRequested();
                for (int time = 0; time < temporal; ++time)
                    Array.Copy(single, channel * spatial, result, (channel * temporal + time) * spatial, spatial);
            }
            trace?.Invoke("encoded", (float[])result.Clone(), [LatentChannels, temporal, height / SpatialScale, width / SpatialScale]);
            return result;
        }
    }

    /// <summary>Encode contiguous [T,H,W,3] RGB floats in [0,1] to channel-major
    /// normalized latent means. T must be 1 + TemporalScale*k; both spatial
    /// dimensions must divide by SpatialScale. The caller's pixels are preserved.
    /// Optional independent stage snapshots add device readbacks.</summary>
    public float[] Encode(float[] frameMajorRgb, int frames, int height, int width,
        CancellationToken cancellationToken = default, Action<string, float[], int[]>? trace = null)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ArgumentNullException.ThrowIfNull(frameMajorRgb);
            if (frames <= 0 || height <= 0 || width <= 0 || (frames - 1) % TemporalScale != 0 ||
                height % SpatialScale != 0 || width % SpatialScale != 0)
                throw new ArgumentOutOfRangeException(nameof(frames), "LTX VAE pixels require positive H/W divisible by the spatial scale and T = 1 + temporal scale*k.");
            if (frameMajorRgb.Length != checked(frames * height * width * 3))
                throw new ArgumentException("RGB pixels must be contiguous frame-major [T,H,W,3].", nameof(frameMajorRgb));
            foreach (float value in frameMajorRgb)
                if (!float.IsFinite(value) || value < 0 || value > 1)
                    throw new ArgumentException("LTX VAE RGB pixels must be finite and in [0,1].", nameof(frameMajorRgb));
            cancellationToken.ThrowIfCancellationRequested();
            int h = height / _patchSize, w = width / _patchSize, channels = checked(3 * _patchSize * _patchSize);
            var patched = new float[frameMajorRgb.Length];
            // Upstream patchify packs (channel, pixel-x, pixel-y), whereas
            // residual space-to-depth packs (channel, time, pixel-y, pixel-x).
            for (int t = 0; t < frames; ++t)
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x)
                        for (int c = 0; c < 3; ++c)
                            for (int px = 0; px < _patchSize; ++px)
                                for (int py = 0; py < _patchSize; ++py)
                                    patched[(((t * h + y) * w + x) * channels) + (c * _patchSize + px) * _patchSize + py] =
                                        frameMajorRgb[((t * height + y * _patchSize + py) * width + x * _patchSize + px) * 3 + c] * 2f - 1f;
            Tensor current = _context.FromFloats(patched, frames, h, w, channels);
            try
            {
                Trace("patchified", current, trace);
                Replace(ref current, _ops.Convolve(current, "conv_in.conv", cancellationToken, causal: true));
                Trace("conv_in", current, trace);
                for (int index = 0; index < _blocks.Length; ++index)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var block = _blocks[index]; string path = "down_blocks." + index;
                    if (block.Kind == "res_x")
                    {
                        for (int layer = 0; layer < block.Layers; ++layer)
                        {
                            string p = path + ".res_blocks." + layer;
                            using var activated1 = _ops.NormSilu(current);
                            using var conv1 = _ops.Convolve(activated1, p + ".conv1.conv", cancellationToken, causal: true);
                            using var activated2 = _ops.NormSilu(conv1);
                            using var conv2 = _ops.Convolve(activated2, p + ".conv2.conv", cancellationToken, causal: true);
                            Ops.Add(current, current, conv2); Trace(p, current, trace);
                        }
                    }
                    else
                    {
                        int st = block.Kind is "compress_time_res" or "compress_all_res" ? 2 : 1;
                        int ss = block.Kind is "compress_space_res" or "compress_all_res" ? 2 : 1;
                        int outputChannels = checked((int)current.Sizes[3] * block.Multiplier);
                        // Duplicating the first frame commutes with the causal
                        // repeat-padded convolution. Convolve before duplication
                        // to avoid twice the projection work for image references.
                        using var conv = _ops.Convolve(current, path + ".conv.conv", cancellationToken, causal: true);
                        if (_context.IsMlx)
                        {
                            var down = _context.NewF32((current.Sizes[0] + st - 1) / st, current.Sizes[1] / ss, current.Sizes[2] / ss, outputChannels);
                            try
                            {
                                if (!MlxLtx25VaeOps.TryDownsampleResidual(down, current, conv, st, ss))
                                    throw new NotSupportedException("MLX LTX video VAE residual downsample is unavailable.");
                                Replace(ref current, down); down = null;
                            }
                            finally { down?.Dispose(); }
                        }
                        else
                        {
                            using var down = SpaceToDepth(conv, st, ss);
                            using var residual = SpaceToDepth(current, st, ss);
                            int groups = checked((int)residual.Sizes[3] / outputChannels);
                            using var grouped = residual.View(residual.ElementCount() / residual.Sizes[3], outputChannels, groups);
                            using var mean = GroupMean(grouped, outputChannels, groups);
                            using var shaped = mean.View(down.Sizes);
                            Ops.Add(down, down, shaped); Replace(ref current, down.View(down.Sizes));
                        }
                    }
                    Trace(path, current, trace);
                }
                using (var activated = _ops.NormSilu(current))
                    Replace(ref current, _ops.Convolve(activated, "conv_out.conv", cancellationToken, causal: true));
                Trace("conv_out", current, trace);
                using (var means = current.Narrow(3, 0, LatentChannels)) Replace(ref current, _ops.Contiguous(means));
                using (var rows = current.View(current.ElementCount() / LatentChannels, LatentChannels))
                {
                    float[] offset = _file.ReadFloat32("per_channel_statistics.mean-of-means");
                    float[] scale = _file.ReadFloat32("per_channel_statistics.std-of-means");
                    for (int c = 0; c < LatentChannels; ++c) { offset[c] = -offset[c]; scale[c] = 1f / scale[c]; }
                    using var bias = _context.FromFloats(offset, LatentChannels);
                    using var gain = _context.FromFloats(scale, LatentChannels);
                    DirectOps.AddBiasRows(_context, rows, bias); DirectOps.MulColsRows(_context, rows, gain);
                }
                Trace("encoded", current, trace);
                float[] result = _ops.Planar(current);
                foreach (float value in result)
                    if (!float.IsFinite(value)) throw new InvalidOperationException("LTX video VAE encoder produced a non-finite latent.");
                return result;
            }
            finally { current.Dispose(); }
        }
    }

    private Tensor SpaceToDepth(Tensor input, int temporalStride, int spatialStride)
    {
        Tensor padded = null;
        try
        {
            if (temporalStride == 2)
            {
                padded = _context.NewF32(input.Sizes[0] + 1, input.Sizes[1], input.Sizes[2], input.Sizes[3]);
                using var first = input.Narrow(0, 0, 1);
                using var targetFirst = padded.Narrow(0, 0, 1); Ops.Copy(targetFirst, first);
                using var rest = padded.Narrow(0, 1, input.Sizes[0]); Ops.Copy(rest, input);
                input = padded;
            }
            long t = input.Sizes[0], h = input.Sizes[1], w = input.Sizes[2], c = input.Sizes[3];
            long ot = t / temporalStride, oh = h / spatialStride, ow = w / spatialStride;
            // Factor the seven-axis permutation into native four-axis copies.
            // CUDA's generic higher-rank copy would otherwise read back to CPU.
            using var heightGroups = input.View(t * oh, spatialStride, w, c);
            using var heightPermuted = heightGroups.Permute(0, 2, 1, 3);
            using var heightCopy = _ops.Contiguous(heightPermuted);
            using var widthGroups = heightCopy.View(t * oh, ow, spatialStride, spatialStride * c);
            using var widthPermuted = widthGroups.Permute(0, 1, 3, 2);
            using var widthCopy = _ops.Contiguous(widthPermuted);
            using var channelGroups = widthCopy.View(t * oh * ow, spatialStride, c, spatialStride);
            using var channelPermuted = channelGroups.Permute(0, 2, 1, 3);
            using var channelCopy = _ops.Contiguous(channelPermuted);
            using var timeGroups = channelCopy.View(ot, temporalStride, oh * ow, c * spatialStride * spatialStride);
            using var timePermuted = timeGroups.Permute(0, 2, 3, 1);
            using var timeCopy = _ops.Contiguous(timePermuted);
            using var finalGroups = timeCopy.View(ot * oh * ow, c, spatialStride * spatialStride, temporalStride);
            using var finalPermuted = finalGroups.Permute(0, 1, 3, 2);
            using var finalCopy = _ops.Contiguous(finalPermuted);
            return finalCopy.View(ot, oh, ow, c * temporalStride * spatialStride * spatialStride);
        }
        finally { padded?.Dispose(); }
    }

    private Tensor GroupMean(Tensor grouped, int channels, int groups)
    {
        if (!_context.IsCuda) return Ops.Mean(null, grouped, 2);
        var output = _context.NewF32(grouped.Sizes[0], channels, 1);
        try
        {
            // Existing adaptive channel-average kernel, with one head, is
            // exactly this contiguous grouped mean. Generic CUDA Reduce
            // intentionally falls back to CPU, so do not dispatch through it.
            if (!CudaMiniMaxH3Ops.TryTransform(output, grouped, null, null, 4,
                    1, checked(channels * groups), channels))
                throw new NotSupportedException("CUDA LTX video VAE grouped residual mean is unavailable.");
            return output;
        }
        catch { output.Dispose(); throw; }
    }

    private void ValidateWeights()
    {
        void Shape(string name, params long[] expected)
        {
            if (!_file.TensorShape(name).SequenceEqual(expected))
                throw new InvalidDataException($"LTX video encoder tensor '{name}' must have shape [{string.Join(',', expected)}].");
            if (_file.GetInfo(name).Dtype is not (SafetensorDtype.BF16 or SafetensorDtype.F16 or SafetensorDtype.F32))
                throw new NotSupportedException($"LTX video encoder tensor '{name}' must use BF16, F16 or Float32 storage.");
        }
        void Conv(string name, int input, int output)
        { Shape(_prefix + name + ".weight", output, input, 3, 3, 3); Shape(_prefix + name + ".bias", output); }
        Shape("per_channel_statistics.mean-of-means", LatentChannels);
        Shape("per_channel_statistics.std-of-means", LatentChannels);
        float[] standardDeviations = _file.ReadFloat32("per_channel_statistics.std-of-means");
        if (standardDeviations.Any(value => !float.IsFinite(value) || value <= 0) ||
            _file.ReadFloat32("per_channel_statistics.mean-of-means").Any(value => !float.IsFinite(value)))
            throw new InvalidDataException("LTX video encoder normalization statistics must have finite means and positive finite standard deviations.");
        int channels = _baseChannels;
        Conv("conv_in.conv", checked(3 * _patchSize * _patchSize), channels);
        for (int index = 0; index < _blocks.Length; ++index)
        {
            var block = _blocks[index]; string path = "down_blocks." + index;
            if (block.Kind == "res_x")
                for (int layer = 0; layer < block.Layers; ++layer)
                { Conv(path + ".res_blocks." + layer + ".conv1.conv", channels, channels); Conv(path + ".res_blocks." + layer + ".conv2.conv", channels, channels); }
            else
            {
                int factor = block.Kind == "compress_all_res" ? 8 : block.Kind == "compress_space_res" ? 4 : 2;
                int outputChannels = checked(channels * block.Multiplier);
                if (outputChannels % factor != 0 || channels * factor % outputChannels != 0)
                    throw new InvalidDataException("LTX video encoder downsample channels are inconsistent with its stride and residual groups.");
                Conv(path + ".conv.conv", channels, outputChannels / factor); channels = outputChannels;
            }
        }
        Conv("conv_out.conv", channels, _outputChannels);
    }
    private void Trace(string name, Tensor tensor, Action<string, float[], int[]>? trace)
    {
        if (trace != null) trace(name, _ops.Planar(tensor), [checked((int)tensor.Sizes[3]), checked((int)tensor.Sizes[0]),
            checked((int)tensor.Sizes[1]), checked((int)tensor.Sizes[2])]);
    }
    private static void Replace(ref Tensor current, Tensor replacement)
    { var previous = current; current = replacement; previous.Dispose(); }

    /// <summary>Release cached device weights. Convolution weights are already
    /// released after each operation; this also honors future context caches.</summary>
    public void ReleaseDeviceResidency()
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            _context.ReleaseDeviceResidency();
        }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            _disposed = true; _context.Dispose(); _file.Dispose();
        }
    }
}
