// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models;

internal static class MuseSparkArchitecture
{
    public static ModelArchitectureDescriptor Descriptor { get; } = new()
    {
        Id = "muse-spark",
        DisplayName = "Muse Spark",
        Aliases = new[] { "muse-spark", "muse_spark" },
        Factory = c => new MuseSparkModel(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup, c.LayerSplitDegree),
        MultiGpu = MultiGpuMode.LayerSplit,
        MultiGpuLimitation = "muse-spark implements whole-layer GPU placement; tensor-parallel sharding is not implemented.",
        SupportsDistributedTensorParallel = false,
        ProjectorFileHints = new[] { "*mmproj*Muse*Spark*.gguf", "*mmproj*.gguf" },
    };
}
