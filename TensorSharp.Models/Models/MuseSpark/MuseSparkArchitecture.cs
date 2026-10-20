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
        MultiGpu = MultiGpuMode.TensorParallel,
        SupportsLayerSplit = true,
        MultiGpuLimitation = "muse-spark uses attention tensor parallelism and whole-expert parallelism in one process; whole-layer placement remains available separately.",
        SupportsDistributedTensorParallel = false,
        ProjectorFileHints = new[] { "*mmproj*Muse*Spark*.gguf", "*mmproj*.gguf" },
    };
}
