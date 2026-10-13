// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Linq;
using TensorSharp.Memory;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

internal readonly record struct GlmUnifiedExpertProjection(GgmlTensorType Type,
    long BytesPerExpert, long TensorTailBytes, bool SupportsSelectedTier);

internal sealed record GlmUnifiedExpertLayout(GlmUnifiedExpertProjection Gate,
    GlmUnifiedExpertProjection Up, GlmUnifiedExpertProjection Down)
{
    internal (GgmlTensorType, GgmlTensorType, GgmlTensorType) Types => (Gate.Type, Up.Type, Down.Type);
    internal long BytesPerExpert => checked(Gate.BytesPerExpert + Up.BytesPerExpert + Down.BytesPerExpert);
    internal long TensorTailBytes => checked(Gate.TensorTailBytes + Up.TensorTailBytes + Down.TensorTailBytes);
    internal bool SupportsSelectedTier => Gate.SupportsSelectedTier && Up.SupportsSelectedTier && Down.SupportsSelectedTier;
}

/// <summary>A pre-load CUDA payload forecast, never an allocator measurement or
/// a physical hard-limit certificate. The loaded model must still pass the
/// native exact minimum check before enabling the selected-expert adapter.</summary>
internal sealed record GlmSelectedExpertMemoryPlan(int PrefillRows, int CpuExpertLayers,
    long ExecutionReserveBytes, long CacheCapacityBytes, long CacheMinimumForecastBytes)
{
    internal static long ForecastMinimum(GlmUnifiedMemoryGeometry geometry, int decodeRows, int prefillRows)
    {
        ArgumentNullException.ThrowIfNull(geometry);
        if (decodeRows is < 1 or > 3 || prefillRows is < 1 or > 128)
            throw new ArgumentOutOfRangeException(nameof(prefillRows));
        if (geometry.ExpertCount <= 0 || geometry.UsedExperts is < 1 or > 64
            || geometry.UsedExperts > geometry.ExpertCount || geometry.Hidden <= 0 || geometry.ExpertFf <= 0
            || geometry.ExpertLayerLayouts.Length != geometry.Layers)
            throw new NotSupportedException("Incomplete GLM selected-expert metadata geometry.");
        var layouts = geometry.ExpertLayerLayouts.OfType<GlmUnifiedExpertLayout>().ToArray();
        if (layouts.Length == 0 || layouts.Any(l => !l.SupportsSelectedTier
            || l.Gate.BytesPerExpert <= 0 || l.Up.BytesPerExpert <= 0 || l.Down.BytesPerExpert <= 0
            || l.Gate.TensorTailBytes < 0 || l.Up.TensorTailBytes < 0 || l.Down.TensorTailBytes < 0))
            throw new NotSupportedException("GLM selected-expert planning requires complete quantized gate/up/down layouts.");
        checked
        {
            // Native quotas are equal across type-triple groups. Every group
            // must therefore fit the largest triple, not the sum of unlike
            // per-group minima. Considering all trunk groups is conservative
            // when an operator cap leaves some expert layers CUDA resident.
            long groups = layouts.Select(l => l.Types).Distinct().LongCount();
            long rows = prefillRows > 8 ? Math.Max(decodeRows, prefillRows) : decodeRows;
            long union = Math.Min(geometry.ExpertCount, rows * geometry.UsedExperts);
            long weights = union * layouts.Max(l => l.BytesPerExpert);
            long tails = layouts.Max(l => l.TensorTailBytes);
            long retainedRows = (long)decodeRows * (decodeRows + 1) / 2
                + (prefillRows > 8 ? prefillRows : 0);
            // All narrow variants 1..N and one wide variant share the quantized
            // weight leaves. Sum every allocating F32/I32 tensor even when the
            // allocator aliases or reuses it: input, ids/routes, gate/up, both
            // optional clamps, SiLU/product, down/CONT/weighted, K-1 ADDs, output.
            long elementsPerRow = geometry.Hidden + 2L * geometry.UsedExperts
                + 6L * geometry.ExpertFf * geometry.UsedExperts
                + 4L * geometry.Hidden * geometry.UsedExperts;
            long scratch = 4L * retainedRows * elementsPerRow;
            // Unchanged ggml CUDA uses 128-byte alignment, quantized 512-row
            // tail padding (retained separately above), and this native graph
            // has at most 256 tensor descriptors per variant. Include a full
            // alignment unit for each descriptor, the three shared weights,
            // and the native per-group 1 MiB workspace allowance. No flash
            // attention tensor or F32 weight materialization occurs here.
            long variants = decodeRows + (prefillRows > 8 ? 1L : 0L);
            long alignment = 128L * (3 + 256 * variants);
            return groups * (weights + tails + scratch + alignment + (1L << 20));
        }
    }

    internal static GlmSelectedExpertMemoryPlan Select(GlmUnifiedMemoryGeometry geometry,
        UnifiedMemoryModelOptions options, int context, long deviceCeiling, long visionBytes)
    {
        ArgumentNullException.ThrowIfNull(geometry);
        ArgumentNullException.ThrowIfNull(options);
        if (context <= 0 || context > geometry.DeclaredContextTokens || options.PrefillChunkTokens is < 1 or > 512
            || options.MaximumRunningRequests is < 1 or > 8 || deviceCeiling < 0 || visionBytes < 0
            || options.MaximumExpertCacheBytes < 0)
            throw new ArgumentOutOfRangeException(nameof(options));
        if (options.EnableGlmSelectedExperts && options.MaximumRunningRequests > 3)
            throw new NotSupportedException("GLM selected-expert planning supports at most three decode rows.");
        int cap = Math.Min(128, Math.Min(context, options.PrefillChunkTokens));
        for (int rows = cap; rows >= 1; rows--)
        {
            // Mandatory live slots include the reusable primary. Idle terminal
            // prefix slots are optional owners, reclaimed by actual admission
            // when needed; do not reserve imaginary future cache allocations.
            long reserve = geometry.ExecutionPlacementReserve(context, rows, options.MaximumRunningRequests);
            int cpu;
            try { cpu = geometry.SelectCpuExpertLayers(deviceCeiling, reserve, visionBytes); }
            // Slot, dense and vision ownership stay fixed while smaller chunks
            // reduce scratch. This applies equally to mapped CPU experts: their
            // placement can fit a smaller chunk without any selected CUDA tier.
            catch (MemoryPressureException) when (rows > 1) { continue; }
            if (!options.EnableGlmSelectedExperts || cpu == 0)
                return new(rows, cpu, reserve, 0, 0);
            long capacity = GlmUnifiedMemoryGeometry.SelectExpertCacheCapacity(deviceCeiling,
                geometry.DenseDeviceBytes, reserve, visionBytes, options.MaximumExpertCacheBytes);
            long minimum = ForecastMinimum(geometry, options.MaximumRunningRequests, rows);
            if (capacity < minimum) continue;
            long totalReserve = checked(reserve + capacity);
            cpu = geometry.SelectCpuExpertLayers(deviceCeiling, totalReserve, visionBytes);
            return new(rows, cpu, totalReserve, capacity, minimum);
        }
        throw new MemoryPressureException("GLM selected-expert CUDA capacity cannot protect the complete narrow decode union and graph forecast after live slots, dense/vision weights and operator ceilings.");
    }
}
