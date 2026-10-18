// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using TensorSharp.GGML;

namespace TensorSharp.Models;

public partial class MuseSparkModel
{
    private readonly bool _releaseGpuSourcePages =
        Environment.GetEnvironmentVariable("TS_MUSE_SPARK_RELEASE_GPU_SOURCE_PAGES") == "1";
    private readonly HashSet<string> _advisedGpuSourceWeights = new(StringComparer.Ordinal);
    private readonly HashSet<int> _advisedGpuExpertLayers = new();
    private static readonly string[] ExpertSourceSuffixes =
        ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"];
    internal long AdvisedGpuSourceBytes { get; private set; }
    internal int AdvisedGpuSourceWeightCount => _advisedGpuSourceWeights.Count;

    internal static bool CanReleaseGpuSourcePages(bool enabled, BackendType backend, bool offloaded)
        => enabled && backend == BackendType.GgmlCuda && !offloaded;

    private void ReleasePreloadedGpuSourcePages()
    {
        if (!CanReleaseGpuSourcePages(_releaseGpuSourcePages, _backend, false) || !OperatingSystem.IsLinux()) return;
        int previousRank = GgmlBasicOps.GetActiveRank();
        try
        {
            foreach (var (name, weight) in _quantWeights)
            {
                // Embedding rows may still be gathered on the CPU. Fused/copied
                // projections with no original GGUF tensor are also excluded.
                if (name == "token_embd.weight" || !weight.IsFileBacked || weight.DevicePreloadTooLarge ||
                    !ShouldPreloadCudaQuantWeightToDevice(name)) continue;
                GgmlBasicOps.SetActiveRank(PreloadRankForWeight(name));
                TryReleaseGpuSourcePages(name, weight.Data, weight.RawBytes, weight.CacheKey);
            }
        }
        finally { GgmlBasicOps.SetActiveRank(previousRank); }
        LogGpuSourcePageAdvice("preloaded weights");
    }

    private void ReleaseExpertGpuSourcePages(int layer)
    {
        // CPU experts include the optional streamed-to-GPU prefill route. They
        // must remain host-resident for the next decode or streamed prefill.
        if (!CanReleaseGpuSourcePages(_releaseGpuSourcePages, _backend, MoeCpuOffloadConfig.IsLayerOnCpu(layer)) ||
            !OperatingSystem.IsLinux() || _advisedGpuExpertLayers.Contains(layer)) return;
        int before = _advisedGpuSourceWeights.Count;
        bool complete = true;
        foreach (string suffix in ExpertSourceSuffixes)
        {
            string name = _prefixes[layer] + suffix;
            if (_stackedExpertWeights.TryGetValue(name, out var weight) && weight.IsExternalView)
                TryReleaseGpuSourcePages(name, weight.Data, weight.TotalRawBytes, weight.Data);
            complete &= _advisedGpuSourceWeights.Contains(name);
        }
        if (complete) _advisedGpuExpertLayers.Add(layer);
        if (_advisedGpuSourceWeights.Count != before) LogGpuSourcePageAdvice($"layer {layer} experts");
    }

    private void TryReleaseGpuSourcePages(string name, IntPtr data, long bytes, IntPtr cacheKey)
    {
        if (_advisedGpuSourceWeights.Contains(name) || !_gguf.Tensors.TryGetValue(name, out var info) ||
            !GgmlBasicOps.HasResidentDeviceCopy(cacheKey, bytes)) return;
        // The GGUF owner verifies pointer identity and full tensor length before
        // advising anything; an external view alone is not proof of a file map.
        if (!_gguf.TryReleaseMappedTensorPages(info, data, bytes, out long advisedBytes)) return;
        _advisedGpuSourceWeights.Add(name);
        AdvisedGpuSourceBytes += advisedBytes;
    }

    private void LogGpuSourcePageAdvice(string stage)
        => Console.WriteLine($"Muse Spark GPU source-page advice ({stage}): " +
            $"{AdvisedGpuSourceWeightCount} weights, {AdvisedGpuSourceBytes} bytes " +
            $"({AdvisedGpuSourceBytes / 1073741824.0:F2} GiB) cumulatively advised; mappings retained.");
}
