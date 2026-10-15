// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using TensorSharp.Models.Direct;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Ltx25;

/// <summary>The LTX 2.5 Ingredients adapter uses publisher-scaled B @ A
/// updates, without a separate alpha/rank multiplier. Keep those updates in
/// F32 rather than merging and requantizing the base checkpoint.</summary>
internal sealed class Ltx25LoraAdapter : IDisposable
{
    private readonly SafetensorsFile _file;
    private readonly IAllocator _allocator;
    private readonly Dictionary<string, Pair> _pairs = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Projection> _cache = new(StringComparer.Ordinal);
    private bool _disposed;
    public int ProjectionCount => _pairs.Count;
    internal int CachedProjectionCount => _cache.Count;

    private sealed record Pair(string A, string B, int Input, int Output, int Rank);

    private sealed class Projection : IDisposable
    {
        public readonly DirectContext Context;
        public DirectLinear A, B;
        public Projection(IAllocator allocator) => Context = new(allocator);
        public void Dispose() => Context.Dispose();
    }

    public Ltx25LoraAdapter(string path, GgufFile checkpoint, Ltx25Config config, IAllocator allocator)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(allocator);
        _allocator = allocator;
        _file = new SafetensorsFile(path);
        try
        {
            if (!_file.Metadata.TryGetValue("model_version", out string version) || version != "2.5" ||
                !_file.Metadata.TryGetValue("reference_downscale_factor", out string downscale) || downscale != "1")
                throw new InvalidDataException("The Ingredients adapter must declare model_version=2.5 and reference_downscale_factor=1.");
            foreach (var tensor in _file.Tensors.Values)
            {
                string name = tensor.Name;
                const string prefix = "diffusion_model.";
                if (name.StartsWith(prefix, StringComparison.Ordinal)) name = name[prefix.Length..];
                const string aSuffix = ".lora_A.weight", bSuffix = ".lora_B.weight";
                bool isA = name.EndsWith(aSuffix, StringComparison.Ordinal);
                if (!isA && !name.EndsWith(bSuffix, StringComparison.Ordinal))
                    throw new InvalidDataException("Unsupported Ingredients tensor: " + tensor.Name);
                string target = name[..^(isA ? aSuffix.Length : bSuffix.Length)];
                if (!ValidTarget(target, config.Layers))
                    throw new InvalidDataException("Unsupported Ingredients projection: " + target);
                if (tensor.Dtype is not (SafetensorDtype.BF16 or SafetensorDtype.F16 or SafetensorDtype.F32) ||
                    tensor.Shape.Length != 2 || tensor.Shape.Any(size => size <= 0 || size > int.MaxValue))
                    throw new InvalidDataException("Ingredients factors must be nonempty BF16/F16/F32 matrices: " + tensor.Name);
                if (!isA) continue;
                string bName = tensor.Name[..^aSuffix.Length] + bSuffix;
                if (!_file.Tensors.TryGetValue(bName, out var b))
                    throw new InvalidDataException("Missing Ingredients B factor for " + target);
                if (!checkpoint.Tensors.TryGetValue(config.Prefix + target + ".weight", out var weight) ||
                    weight.Shape.Length != 2 || b.Shape.Length != 2 ||
                    tensor.Shape[0] != b.Shape[1] || tensor.Shape[1] != (long)weight.Shape[0] || b.Shape[0] != (long)weight.Shape[1])
                    throw new InvalidDataException("Ingredients factors do not match the base projection: " + target);
                if (!_pairs.TryAdd(target, new(tensor.Name, bName, checked((int)tensor.Shape[1]),
                    checked((int)b.Shape[0]), checked((int)tensor.Shape[0]))))
                    throw new InvalidDataException("Duplicate Ingredients projection: " + target);
            }
            if (_pairs.Count == 0 || _file.Tensors.Count != checked(_pairs.Count * 2))
                throw new InvalidDataException("Every Ingredients projection requires exactly one A/B factor pair.");
        }
        catch { _file.Dispose(); throw; }
    }

    private static bool ValidTarget(string target, int layers)
    {
        const string prefix = "transformer_blocks.";
        if (!target.StartsWith(prefix, StringComparison.Ordinal)) return false;
        int dot = target.IndexOf('.', prefix.Length);
        if (dot < 0 || !int.TryParse(target.AsSpan(prefix.Length, dot - prefix.Length), out int block) ||
            block < 0 || block >= layers || target[prefix.Length..dot] != block.ToString(System.Globalization.CultureInfo.InvariantCulture)) return false;
        return target[(dot + 1)..] is "attn1.to_q" or "attn1.to_k" or "attn1.to_v" or "attn1.to_out.0" or
            "attn2.to_q" or "attn2.to_k" or "attn2.to_v" or "attn2.to_out.0" or "ff.net.0.proj" or "ff.net.2";
    }

    private Projection Load(string name, Pair pair, CancellationToken cancellationToken)
    {
        if (_cache.TryGetValue(name, out var cached)) return cached;
        var projection = new Projection(_allocator);
        try
        {
            float[] a = _file.ReadFloat32(pair.A);
            cancellationToken.ThrowIfCancellationRequested();
            if (a.Any(value => !float.IsFinite(value))) throw new InvalidDataException("Ingredients A factor contains nonfinite values: " + name);
            projection.A = DirectLinear.FromFloats(projection.Context, a, pair.Input, pair.Rank, null, exactFloat32: true);
            float[] b = _file.ReadFloat32(pair.B);
            cancellationToken.ThrowIfCancellationRequested();
            if (b.Any(value => !float.IsFinite(value))) throw new InvalidDataException("Ingredients B factor contains nonfinite values: " + name);
            projection.B = DirectLinear.FromFloats(projection.Context, b, pair.Rank, pair.Output, null, exactFloat32: true);
            cancellationToken.ThrowIfCancellationRequested();
            _cache.Add(name, projection);
            return projection;
        }
        catch { projection.Dispose(); throw; }
    }

    /// <summary>Add the optional update to the caller-owned base output.
    /// The enclosing denoiser serializes calls and adapter cache access.</summary>
    public void Add(string name, Tensor input, Tensor output, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        cancellationToken.ThrowIfCancellationRequested();
        if (!_pairs.TryGetValue(name, out var pair)) return;
        if (input.DimensionCount != 2 || output.DimensionCount != 2 || input.Sizes[0] != output.Sizes[0] ||
            input.Sizes[1] != pair.Input || output.Sizes[1] != pair.Output)
            throw new ArgumentException("Ingredients projection dimensions differ: " + name);
        var projection = Load(name, pair, cancellationToken);
        using var lowRank = projection.A.Forward(input);
        cancellationToken.ThrowIfCancellationRequested();
        using var update = projection.B.Forward(lowRank);
        cancellationToken.ThrowIfCancellationRequested();
        Ops.Add(output, output, update);
    }

    public void ReleaseDeviceResidency()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        foreach (var projection in _cache.Values) projection.Context.ReleaseDeviceResidency();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var projection in _cache.Values) projection.Dispose();
        _cache.Clear();
        _file.Dispose();
    }
}
