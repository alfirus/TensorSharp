// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Collections.Generic;
using System.IO;
using TensorSharp.Memory.Planning;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

internal sealed record GlmUnifiedVisionMemoryGeometry(int Patch, int Merge, int Hidden, int Intermediate,
    int Heads, int Projection, long HostWeightBytes, long DeviceWeightBytes, long LoadingTemporaryBytes)
{
    /// <summary>Device weight bytes one budgeted encode holds at once: the
    /// streamed encoder uploads one block at a time, and the projector's
    /// per-op linears upload one weight per op. 0 = unknown, in which case
    /// the whole tower is forecast.</summary>
    internal long StreamedWeightBytes { get; init; }
    internal long ImageWeightBytes => StreamedWeightBytes > 0 ? StreamedWeightBytes : DeviceWeightBytes;

    internal static GlmUnifiedVisionMemoryGeometry Read(string path)
    {
        using var file = GgufFile.OpenForMemoryPlanning(path);
        if (file.GetString("clip.projector_type") is not ("glm5v" or "glm5next" or "glm5-next"))
            throw new NotSupportedException("GLM unified vision requires its GLM5V projector.");
        int Int(string name, int fallback)
        {
            uint n = file.GetUint32("clip.vision." + name, (uint)fallback);
            if (n == 0 || n > int.MaxValue) throw new NotSupportedException("Invalid GLM vision metadata: " + name);
            return (int)n;
        }
        long weights = 0, temporary = 0, extra = 0, singleMaximum = 0;
        var blocks = new Dictionary<int, long>();
        foreach (var tensor in file.Tensors.Values)
        {
            long bytes = checked(tensor.NumElements * 4);
            weights = checked(weights + bytes + (64L << 10));
            string[] parts = tensor.Name.Split('.');
            if (parts.Length > 2 && parts[0] == "v" && parts[1] == "blk" && int.TryParse(parts[2], out int block))
                blocks[block] = checked(blocks.GetValueOrDefault(block) + bytes + (64L << 10));
            else
                singleMaximum = Math.Max(singleMaximum, checked(bytes + (64L << 10)));
            temporary = Math.Max(temporary, checked(bytes + file.GetTensorFileRegion(tensor.Name).ByteLength));
            if (tensor.Name is "v.patch_embd.weight" or "mm.patch_merger.weight") extra = checked(extra + 3 * bytes);
        }
        // Host originals and contiguous transposes coexist. Unified execution
        // requires the whole encoder graph, so each encoder weight has one
        // device representation; folded patch and merger copies remain extra.
        long host = checked(2 * weights + extra);
        long device = checked(weights + extra);
        int hidden = Int("embedding_length", 1024), heads = Int("attention.head_count", 16);
        if (hidden % heads != 0) throw new NotSupportedException("Invalid GLM vision head width.");
        long largestBlock = 0;
        foreach (long bytes in blocks.Values) largestBlock = Math.Max(largestBlock, bytes);
        return new(Int("patch_size", 14), Int("spatial_merge_size", 2), hidden,
            Int("feed_forward_length", 4096), heads, Int("projection_dim", 4096), host, device, temporary)
            { StreamedWeightBytes = checked(largestBlock + singleMaximum) };
    }

    /// <summary>Device peak of the smallest image GLM admits (16 merged tokens).</summary>
    internal long MinimumImageDevicePeak()
    {
        int side = Patch * Merge * 4;
        return checked(16L * Projection * 4 + ExecutionPeak(side, side).Device + ImageWeightBytes);
    }

    /// <summary>Device bytes a session plan must withhold from its expert
    /// cache for vision: only what its idle execution reservation cannot
    /// already hold for one minimum image. Each image is admitted, with its
    /// streamed weights, against the live budget when it arrives.</summary>
    internal static long PlacementBytes(GlmUnifiedVisionMemoryGeometry? vision, long idleExecutionBytes)
        => vision == null ? 0 : Math.Max(0, vision.MinimumImageDevicePeak() - Math.Max(0, idleExecutionBytes));

    internal InferenceMemoryBytes ExecutionPeak(int height, int width)
    {
        if (height <= 0 || width <= 0 || height % (Patch * Merge) != 0 || width % (Patch * Merge) != 0)
            throw new ArgumentOutOfRangeException(nameof(height));
        checked
        {
            long patches = (long)(height / Patch) * (width / Patch), merged = patches / Merge / Merge;
            if (patches * Hidden > int.MaxValue || merged * Projection > int.MaxValue)
                throw new NotSupportedException("GLM image exceeds the managed tensor range.");
            long payload = (64L << 20) + 24L * height * width
                + 4L * patches * (16L * Hidden + 8L * Intermediate)
                + 4L * Heads * patches * patches // exact score fallback is supported and must fit
                + 8L * merged * Projection;
            return new(payload, payload);
        }
    }
}

internal sealed record GlmVisionPreparationPlan(IReadOnlyList<QwenVisionImagePlan> Images,
    int ExpandedPromptTokens, long HostPeakBytes, long DevicePeakBytes)
{
    internal static GlmVisionPreparationPlan Create(IReadOnlyList<string> paths, int promptTokens,
        int placeholders, int context, GlmUnifiedVisionMemoryGeometry geometry, long? pendingHostWeightBytes = null)
    {
        if (pendingHostWeightBytes is < 0) throw new ArgumentOutOfRangeException(nameof(pendingHostWeightBytes));
        if (paths.Count != placeholders || promptTokens <= 0 || context <= 0)
            throw new ArgumentException("Every GLM image requires exactly one prompt placeholder.");
        var processor = new GlmNextImageProcessor(geometry.Patch, geometry.Merge);
        var images = new List<QwenVisionImagePlan>();
        long expanded = promptTokens, retained = 0, preparation = 0, hostWork = 0, deviceWork = 0;
        foreach (string name in paths)
        {
            string path = Path.GetFullPath(name);
            var info = new FileInfo(path);
            long bytes = info.Length, ticks = info.LastWriteTimeUtc.Ticks;
            if (bytes <= 0 || bytes > int.MaxValue) throw new NotSupportedException("Unsupported GLM image file size.");
            var (width, height) = QwenVisionPreparationPlan.ReadBoundedDimensions(path);
            if ((long)width * height > int.MaxValue / 4) throw new NotSupportedException("GLM decoded image exceeds the managed range.");
            var (w, h) = processor.SmartResize(width, height);
            int tokens = checked(h / geometry.Patch / geometry.Merge * (w / geometry.Patch / geometry.Merge));
            expanded = checked(expanded + tokens - 1);
            if (expanded > context) throw new InvalidOperationException("GLM image-expanded prompt exceeds its admitted context.");
            long embedding = checked((long)tokens * geometry.Projection * 4);
            long pixels = checked(bytes + 32L * height * width + 12L * h * w);
            var execution = geometry.ExecutionPeak(h, w);
            images.Add(new(path, bytes, ticks, height, width, h, w, tokens, embedding, pixels, execution));
            retained = checked(retained + embedding);
            preparation = checked(preparation + pixels);
            hostWork = Math.Max(hostWork, execution.Host);
            deviceWork = Math.Max(deviceWork, execution.Device);
        }
        return new(images, checked((int)expanded), checked(preparation + retained + hostWork + (pendingHostWeightBytes ?? geometry.HostWeightBytes)),
            checked(retained + deviceWork + geometry.ImageWeightBytes));
    }
    internal void ValidateFilesUnchanged()
    {
        foreach (var image in Images)
        {
            var info = new FileInfo(image.Path);
            if (info.Length != image.FileBytes || info.LastWriteTimeUtc.Ticks != image.LastWriteUtcTicks)
                throw new IOException("GLM admitted image changed before preparation.");
        }
    }
}

internal sealed class GlmVisionConstructionCleanupException : Exception
{
    internal GlmNextVisionEncoder UnreleasedEncoder { get; }
    internal GlmVisionConstructionCleanupException(GlmNextVisionEncoder encoder, Exception creation, Exception cleanup)
        : base("GLM vision creation and cleanup both failed; its model retains ownership for retry.", new AggregateException(creation, cleanup))
        => UnreleasedEncoder = encoder;
}
