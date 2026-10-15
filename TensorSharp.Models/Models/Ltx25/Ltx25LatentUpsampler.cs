// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Direct;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Ltx25;

/// <summary>The publisher's x2 spatial latent upsampler (<c>LatentUpsampler</c>
/// with <c>dims=3</c>, <c>spatial_upsample</c>, no rational resampler), which the
/// two-stage DistilledPipeline runs between its half- and full-resolution passes.
/// Conv3d layers zero-pad every axis, GroupNorm(32) normalizes each group over the
/// whole clip, and the x2 step is a per-frame Conv2d followed by PixelShuffle(2).
/// BF16 weights widen exactly; activations stay Float32 on the caller's allocator.</summary>
public sealed class Ltx25LatentUpsampler : IDisposable
{
    private const int Groups = 32;
    private const float GroupNormEpsilon = 1e-5f;
    private readonly SafetensorsFile _file;
    private readonly DirectContext _context;
    private readonly Ltx25VideoVaeOps _ops;
    private readonly object _gate = new();
    private bool _disposed;
    public int LatentChannels { get; }
    public int MidChannels { get; }
    public int BlocksPerStage { get; }

    public Ltx25LatentUpsampler(string path, IAllocator allocator)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(allocator);
        if (allocator is not (CpuAllocator or CudaAllocator or MlxAllocator))
            throw new NotSupportedException("The LTX latent upsampler requires a CPU, CUDA or MLX direct allocator.");
        _context = new DirectContext(allocator);
        _file = new SafetensorsFile(path);
        try
        {
            if (!_file.Metadata.TryGetValue("config", out string json))
                throw new InvalidDataException("The LTX latent upsampler requires embedded architecture metadata.");
            using var doc = JsonDocument.Parse(json);
            var config = doc.RootElement;
            string Text(string key) => config.TryGetProperty(key, out var value) ? value.ToString() : "";
            int Number(string key) => config.TryGetProperty(key, out var value) ? value.GetInt32() : -1;
            bool Flag(string key) => config.TryGetProperty(key, out var value) && value.GetBoolean();
            if (Text("_class_name") != "LatentUpsampler" || Number("dims") != 3 || !Flag("spatial_upsample") ||
                Flag("temporal_upsample") || Flag("rational_resampler") ||
                !config.TryGetProperty("spatial_scale", out var scale) || scale.GetDouble() != 2.0)
                throw new NotSupportedException("LTX two-stage generation requires the x2 spatial latent upsampler (3D, no rational resampler).");
            LatentChannels = Number("in_channels");
            MidChannels = Number("mid_channels");
            BlocksPerStage = Number("num_blocks_per_stage");
            if (LatentChannels <= 0 || MidChannels <= 0 || MidChannels % Groups != 0 || BlocksPerStage <= 0)
                throw new InvalidDataException("Invalid LTX latent upsampler channel or block configuration.");
            ValidateWeights();
            _ops = new Ltx25VideoVaeOps(_context, _file, "");
        }
        catch { _file.Dispose(); _context.Dispose(); throw; }
    }

    private void ValidateWeights()
    {
        void Shape(string name, params long[] expected)
        {
            if (!_file.TensorShape(name).SequenceEqual(expected))
                throw new InvalidDataException($"LTX latent upsampler tensor '{name}' must have shape [{string.Join(',', expected)}].");
            if (_file.GetInfo(name).Dtype is not (SafetensorDtype.BF16 or SafetensorDtype.F16 or SafetensorDtype.F32))
                throw new NotSupportedException($"LTX latent upsampler tensor '{name}' must use BF16, F16 or Float32 storage.");
        }
        void Conv(string name, int input, int output) { Shape(name + ".weight", output, input, 3, 3, 3); Shape(name + ".bias", output); }
        void Norm(string name) { Shape(name + ".weight", MidChannels); Shape(name + ".bias", MidChannels); }
        Conv("initial_conv", LatentChannels, MidChannels);
        Norm("initial_norm");
        foreach (string stage in new[] { "res_blocks", "post_upsample_res_blocks" })
            for (int i = 0; i < BlocksPerStage; ++i)
            {
                string block = $"{stage}.{i}";
                Conv(block + ".conv1", MidChannels, MidChannels); Norm(block + ".norm1");
                Conv(block + ".conv2", MidChannels, MidChannels); Norm(block + ".norm2");
            }
        Shape("upsampler.0.weight", 4L * MidChannels, MidChannels, 3, 3);
        Shape("upsampler.0.bias", 4L * MidChannels);
        Conv("final_conv", MidChannels, LatentChannels);
    }

    /// <summary>Upsample channel-major normalized latents [C,T,H,W] to
    /// [C,T,2H,2W], applying <c>upsample_video</c>'s un-normalize/normalize
    /// with the video VAE's per-channel statistics.</summary>
    public float[] Upsample(float[] channelMajorLatents, int frames, int height, int width,
        float[] channelMean, float[] channelStd, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(channelMajorLatents);
        ArgumentNullException.ThrowIfNull(channelMean);
        ArgumentNullException.ThrowIfNull(channelStd);
        if (frames <= 0 || height <= 0 || width <= 0) throw new ArgumentOutOfRangeException(nameof(frames));
        if (channelMajorLatents.Length != checked(LatentChannels * frames * height * width) ||
            channelMean.Length != LatentChannels || channelStd.Length != LatentChannels)
            throw new ArgumentException("Latent upsampling requires channel-major [C,T,H,W] latents and one statistic per channel.");
        foreach (float value in channelMajorLatents)
            if (!float.IsFinite(value)) throw new ArgumentException("LTX latents must be finite.", nameof(channelMajorLatents));
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            Tensor current;
            using (var planar = _context.FromFloats(channelMajorLatents, LatentChannels, frames, height, width))
            using (var permuted = planar.Permute(1, 2, 3, 0)) current = _ops.Contiguous(permuted);
            try
            {
                using (var rows = current.View((long)frames * height * width, LatentChannels))
                using (var std = _context.FromFloats(channelStd, LatentChannels))
                using (var mean = _context.FromFloats(channelMean, LatentChannels))
                {
                    DirectOps.MulColsRows(_context, rows, std);
                    DirectOps.AddBiasRows(_context, rows, mean);
                }
                Replace(ref current, Convolve(current, "initial_conv", cancellationToken));
                Replace(ref current, GroupNorm(current, "initial_norm", silu: true));
                for (int i = 0; i < BlocksPerStage; ++i)
                    Replace(ref current, ResBlock(current, $"res_blocks.{i}", cancellationToken));
                using (var expanded = _ops.Convolve(current, "upsampler.0", cancellationToken))
                    Replace(ref current, PixelShuffle2x(expanded));
                for (int i = 0; i < BlocksPerStage; ++i)
                    Replace(ref current, ResBlock(current, $"post_upsample_res_blocks.{i}", cancellationToken));
                Replace(ref current, Convolve(current, "final_conv", cancellationToken));
                float[] inverseStd = Array.ConvertAll(channelStd, value => 1f / value);
                float[] shift = new float[LatentChannels];
                for (int c = 0; c < shift.Length; ++c) shift[c] = -channelMean[c] * inverseStd[c];
                using (var rows = current.View(current.ElementCount() / LatentChannels, LatentChannels))
                using (var scale = _context.FromFloats(inverseStd, LatentChannels))
                using (var bias = _context.FromFloats(shift, LatentChannels))
                {
                    DirectOps.MulColsRows(_context, rows, scale);
                    DirectOps.AddBiasRows(_context, rows, bias);
                }
                float[] result = _ops.Planar(current);
                foreach (float value in result)
                    if (!float.IsFinite(value)) throw new InvalidOperationException("LTX latent upsampler produced a non-finite value.");
                return result;
            }
            finally { current?.Dispose(); }
        }
    }

    private Tensor Convolve(Tensor input, string name, CancellationToken cancellationToken)
        => _ops.Convolve(input, name, cancellationToken, zeroTemporalPadding: true);

    /// <summary>ResBlock: conv1, norm1, SiLU, conv2, norm2, then SiLU of the residual sum.</summary>
    private Tensor ResBlock(Tensor input, string name, CancellationToken cancellationToken)
    {
        using var conv1 = Convolve(input, name + ".conv1", cancellationToken);
        using var activated = GroupNorm(conv1, name + ".norm1", silu: true);
        using var conv2 = Convolve(activated, name + ".conv2", cancellationToken);
        using var normalized = GroupNorm(conv2, name + ".norm2", silu: false);
        Ops.Add(normalized, normalized, input);
        long channels = input.Sizes[^1];
        using var rows = normalized.View(normalized.ElementCount() / channels, channels);
        using var activatedRows = DirectOps.Silu(_context, rows);
        return activatedRows.View(input.Sizes);
    }

    /// <summary>torch GroupNorm(32) over a channels-last clip: each group's mean
    /// and biased variance span its channels and every (T,H,W) position. The
    /// statistics accumulate in Float64 on the host; the per-channel affine and
    /// optional SiLU run on the device.</summary>
    private Tensor GroupNorm(Tensor input, string name, bool silu)
    {
        int channels = checked((int)input.Sizes[^1]);
        long positions = input.ElementCount() / channels;
        float[] values = input.GetElementsAsFloat(checked((int)input.ElementCount()));
        int perGroup = channels / Groups;
        var sums = new double[channels];
        var squares = new double[channels];
        int workers = Math.Max(1, Math.Min(Environment.ProcessorCount, 16));
        var partialSums = new double[workers][];
        var partialSquares = new double[workers][];
        Parallel.For(0, workers, worker =>
        {
            var sum = partialSums[worker] = new double[channels];
            long begin = positions * worker / workers, end = positions * (worker + 1) / workers;
            for (long row = begin; row < end; ++row)
            {
                long offset = row * channels;
                for (int c = 0; c < channels; ++c) sum[c] += values[offset + c];
            }
        });
        var groupMean = new double[Groups];
        for (int g = 0; g < Groups; ++g)
        {
            double total = 0;
            for (int worker = 0; worker < workers; ++worker)
                for (int c = g * perGroup; c < (g + 1) * perGroup; ++c) total += partialSums[worker][c];
            groupMean[g] = total / (positions * (double)perGroup);
        }
        Parallel.For(0, workers, worker =>
        {
            var square = partialSquares[worker] = new double[channels];
            long begin = positions * worker / workers, end = positions * (worker + 1) / workers;
            for (long row = begin; row < end; ++row)
            {
                long offset = row * channels;
                for (int c = 0; c < channels; ++c)
                {
                    double centered = values[offset + c] - groupMean[c / perGroup];
                    square[c] += centered * centered;
                }
            }
        });
        float[] gamma = _file.ReadFloat32(name + ".weight"), beta = _file.ReadFloat32(name + ".bias");
        var scale = new float[channels];
        var shift = new float[channels];
        for (int g = 0; g < Groups; ++g)
        {
            double variance = 0;
            for (int worker = 0; worker < workers; ++worker)
                for (int c = g * perGroup; c < (g + 1) * perGroup; ++c) variance += partialSquares[worker][c];
            double inverse = 1.0 / Math.Sqrt(variance / (positions * (double)perGroup) + GroupNormEpsilon);
            for (int c = g * perGroup; c < (g + 1) * perGroup; ++c)
            {
                scale[c] = (float)(gamma[c] * inverse);
                shift[c] = (float)(beta[c] - groupMean[g] * gamma[c] * inverse);
            }
        }
        var output = _ops.Contiguous(input);
        try
        {
            using (var rows = output.View(positions, channels))
            using (var scaleTensor = _context.FromFloats(scale, channels))
            using (var shiftTensor = _context.FromFloats(shift, channels))
            {
                DirectOps.MulColsRows(_context, rows, scaleTensor);
                DirectOps.AddBiasRows(_context, rows, shiftTensor);
            }
            if (!silu) return output;
            using var flat = output.View(positions, channels);
            using var activated = DirectOps.Silu(_context, flat);
            return activated.View(input.Sizes);
        }
        finally { if (silu) output.Dispose(); }
    }

    /// <summary>PixelShuffle(2) on channels-last [T,H,W,4C]: channel c*4+2*i+j
    /// moves to (2y+i, 2x+j).</summary>
    private Tensor PixelShuffle2x(Tensor input)
    {
        long t = input.Sizes[0], h = input.Sizes[1], w = input.Sizes[2], channels = input.Sizes[3] / 4;
        if (_context.IsCuda)
        {
            // The native strided copy supports four axes; factor the six-axis shuffle.
            using var grouped = input.View(t * h * w, channels, 4);
            using var channelsLast = grouped.Permute(0, 2, 1);
            using var reordered = _ops.Contiguous(channelsLast);
            using var heightGroups = reordered.View(t * h, w, 2, 2 * channels);
            using var heightFirst = heightGroups.Permute(0, 2, 1, 3);
            using var spatial = _ops.Contiguous(heightFirst);
            return spatial.View(t, h * 2, w * 2, channels);
        }
        using var split = input.View(t, h, w, channels, 2, 2);
        using var shuffled = split.Permute(0, 1, 4, 2, 5, 3);
        using var contiguous = _ops.Contiguous(shuffled);
        return contiguous.View(t, h * 2, w * 2, channels);
    }

    private static void Replace(ref Tensor current, Tensor replacement)
    { var previous = current; current = replacement; previous.Dispose(); }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            _disposed = true; _context.Dispose(); _file.Dispose();
        }
    }
}
