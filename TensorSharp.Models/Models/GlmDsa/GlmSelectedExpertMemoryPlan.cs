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
    // Matches the native fixed routing-validation scratch. Larger top-k models
    // need a smaller row cap; do not resize or omit this independently owned
    // scratch simply to permit wider chunks.
    private const int MaximumRouteElements = 8192;
    private const int LegacyMaximumRouteElements = 4096;

    internal static GlmSelectedExpertMemoryPlan SelectForSession(GlmUnifiedMemoryGeometry geometry,
        UnifiedMemoryModelOptions options, int context, long deviceCeiling, long visionBytes)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (!options.EnableAdaptiveGlmPrefill) return Select(geometry, options, context, deviceCeiling, visionBytes);
        ArgumentNullException.ThrowIfNull(geometry);
        if (!options.EnableGlmSelectedExperts || options.MaximumRunningRequests is < 1 or > 3
            || options.PrefillChunkTokens is < 1 or > 512 || options.MaximumExpertCacheBytes < 0
            || context <= 0 || context > geometry.DeclaredContextTokens || deviceCeiling < 0 || visionBytes < 0)
            throw new ArgumentOutOfRangeException(nameof(options));
        int requestedRows = options.ResolveGlmPrefillChunkTokens();
        long reserve = geometry.AdaptiveExecutionPlacementReserve(context, options.MaximumRunningRequests);
        int cpu = geometry.SelectCpuExpertLayers(deviceCeiling, reserve, visionBytes);
        int maximum = Math.Min(context, requestedRows);
        if (cpu == 0) return new(maximum, 0, reserve, 0, 0);
        if (geometry.UsedExperts is < 1 or > 64) throw new NotSupportedException("Incomplete GLM selected-expert route geometry.");
        maximum = Math.Min(maximum, MaximumRouteElements / geometry.UsedExperts);
        long capacity = GlmUnifiedMemoryGeometry.SelectExpertCacheCapacity(deviceCeiling,
            geometry.DenseDeviceBytes, reserve, visionBytes, options.MaximumExpertCacheBytes);
        for (int rows = maximum; rows >= 1; rows--)
        {
            // Native installs this complete immutable phase floor once. Actual
            // requests may use smaller widths without replacing that plan.
            long minimum = ForecastMinimum(geometry, options.EnableGlmFlashMtp ? 3 : options.MaximumRunningRequests, rows);
            if (capacity < minimum) continue;
            long totalReserve = checked(reserve + capacity);
            cpu = geometry.SelectCpuExpertLayers(deviceCeiling, totalReserve, visionBytes);
            return new(rows, cpu, totalReserve, capacity, minimum);
        }
        throw new MemoryPressureException("GLM adaptive prefill cannot preserve complete slots, joint decode and its immutable selected-expert phase floor.");
    }

    internal static long ForecastMinimum(GlmUnifiedMemoryGeometry geometry, int decodeRows, int prefillRows,
        int maximumRouteElements = MaximumRouteElements)
    {
        ArgumentNullException.ThrowIfNull(geometry);
        if (decodeRows is < 1 or > 3 || prefillRows is < 1 or > 1024
            || maximumRouteElements is < 1 or > MaximumRouteElements)
            throw new ArgumentOutOfRangeException(nameof(prefillRows));
        if (geometry.ExpertCount <= 0 || geometry.UsedExperts is < 1 or > 64
            || geometry.UsedExperts > geometry.ExpertCount || geometry.Hidden <= 0 || geometry.ExpertFf <= 0
            || geometry.ExpertLayerLayouts.Length != geometry.Layers)
            throw new NotSupportedException("Incomplete GLM selected-expert metadata geometry.");
        if (prefillRows > maximumRouteElements / geometry.UsedExperts)
            throw new ArgumentOutOfRangeException(nameof(prefillRows), "GLM prefill exceeds its fixed selected-route scratch.");
        var members = geometry.ExpertLayerLayouts.OfType<GlmUnifiedExpertLayout>()
            .Select(l => (Layout: l, DecodeRows: decodeRows, PrefillRows: prefillRows))
            .Concat(geometry.Mtp == null ? [] : new[] { (Layout: geometry.Mtp.ExpertLayout,
                DecodeRows: geometry.Mtp.ShapeSpecificPlan ? 1 : decodeRows,
                PrefillRows: geometry.Mtp.ShapeSpecificPlan ? 1 : prefillRows) }).ToArray();
        if (members.Length == 0 || members.Any(m => !m.Layout.SupportsSelectedTier
            || m.Layout.Gate.BytesPerExpert <= 0 || m.Layout.Up.BytesPerExpert <= 0 || m.Layout.Down.BytesPerExpert <= 0
            || m.Layout.Gate.TensorTailBytes < 0 || m.Layout.Up.TensorTailBytes < 0 || m.Layout.Down.TensorTailBytes < 0))
            throw new NotSupportedException("GLM selected-expert planning requires complete quantized gate/up/down layouts.");
        checked
        {
            // Adapter v7 serializes helper phases under one mutex. Successful
            // return may leave a device-output copy queued; the GLM DSA caller
            // fences before downstream work. Owner destructors independently
            // synchronize before free/refund, including inter-phase reclaim.
            // A wide union replaces inactive payload only after its physical
            // release. Forecast one wide
            // group plus the other groups' narrow floors, not all-wide at once.
            // This is still not physical credit; native measures every layout
            // and shared callbacks independently admit each real owner.
            var groups = members.GroupBy(m => m.Layout.Types).ToArray();
            long totalNarrow = 0, largestExpansion = 0;
            foreach (var group in groups)
            {
                // Measure the final merged shape. A shared trunk/draft key retains
                // all trunk variants; independent byte maxima of incomparable
                // member shapes would not cover their combined union and scratch.
                int groupDecode = group.Max(m => m.DecodeRows), groupPrefill = group.Max(m => m.PrefillRows);
                long rows = groupPrefill > 8 ? Math.Max(groupDecode, groupPrefill) : groupDecode;
                long union = Math.Min(geometry.ExpertCount, rows * geometry.UsedExperts);
                long narrowUnion = Math.Min(geometry.ExpertCount, (long)groupDecode * geometry.UsedExperts);
                // Trunk groups retain compact prefill widths through eight. A
                // draft-only group retains only its admitted scalar variant.
                // Short prefill can use smaller microgroups without expanding
                // the protected union; retained graph scratch still sums widths.
                long narrowRows = Math.Max(groupDecode, Math.Min(groupPrefill, 8));
                long narrowRetainedRows = narrowRows * (narrowRows + 1) / 2;
                // Sum allocating F32/I32 tensors even when the allocator aliases
                // them: inputs, routes, gate/up, clamps, product, down and output.
                long elementsPerRow = geometry.Hidden + 2L * geometry.UsedExperts
                    + 6L * geometry.ExpertFf * geometry.UsedExperts
                    + 4L * geometry.Hidden * geometry.UsedExperts;
                long narrowScratch = 4L * narrowRetainedRows * elementsPerRow;
                long wideScratch = narrowScratch + 4L * (groupPrefill > 8 ? groupPrefill : 0) * elementsPerRow;
                // Unchanged CUDA uses 128-byte alignment and quantized tail
                // padding. Include each of at most 256 descriptors per variant,
                // three weight leaves and the native 1 MiB workspace allowance.
                long narrowAlignment = 128L * (3 + 256 * narrowRows);
                long wideAlignment = narrowAlignment + (groupPrefill > 8 ? 128L * 256 : 0);
                long expertBytes = group.Max(m => m.Layout.BytesPerExpert);
                long tail = group.Max(m => m.Layout.TensorTailBytes);
                long narrow = narrowUnion * expertBytes + tail + narrowScratch + narrowAlignment + (1L << 20);
                long wide = union * expertBytes + tail + wideScratch + wideAlignment + (1L << 20);
                totalNarrow += narrow;
                largestExpansion = Math.Max(largestExpansion, wide - narrow);
            }
            return totalNarrow + largestExpansion;
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
        // Larger admitted chunks amortize reading each layer's routed expert
        // union. Keep the operator cap and decrease it only when the complete
        // slots, graph generations and expert phase floor cannot coexist.
        int cap = Math.Min(context, options.PrefillChunkTokens);
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
            if (geometry.UsedExperts <= 0 || geometry.UsedExperts > 64)
                throw new NotSupportedException("Incomplete GLM selected-expert route geometry.");
            if (rows > LegacyMaximumRouteElements / geometry.UsedExperts) continue;
            long capacity = GlmUnifiedMemoryGeometry.SelectExpertCacheCapacity(deviceCeiling,
                geometry.DenseDeviceBytes, reserve, visionBytes, options.MaximumExpertCacheBytes);
            long minimum = ForecastMinimum(geometry, options.MaximumRunningRequests, rows, LegacyMaximumRouteElements);
            if (capacity < minimum) continue;
            long totalReserve = checked(reserve + capacity);
            cpu = geometry.SelectCpuExpertLayers(deviceCeiling, totalReserve, visionBytes);
            return new(rows, cpu, totalReserve, capacity, minimum);
        }
        throw new MemoryPressureException("GLM selected-expert CUDA capacity cannot protect the complete narrow decode union and graph forecast after live slots, dense/vision weights and operator ceilings.");
    }
}
