using Xunit.Sdk;

namespace InferenceWeb.Tests;

[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
[AttributeUsage(AttributeTargets.Method)]
internal sealed class EmbeddingGemma2MlxFactAttribute : FactAttribute, ITraitAttribute
{
    public string RequiresValue => "Mlx,Models";
    public EmbeddingGemma2MlxFactAttribute()
        => Skip = TestGates.ModelSkip("TENSORSHARP_EMBEDDINGGEMMA2_MODEL", null) ?? TestGates.MlxSkip;
}

[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
[AttributeUsage(AttributeTargets.Method)]
internal sealed class EmbeddingGemma2VisionCudaFactAttribute : FactAttribute, ITraitAttribute
{
    public string RequiresValue => "GgmlCuda,Models";
    public EmbeddingGemma2VisionCudaFactAttribute()
        => Skip = TestGates.GgmlPinSkip(BackendType.GgmlCuda)
            ?? TestGates.ModelSkip("TENSORSHARP_EMBEDDINGGEMMA2_MODEL", "embeddinggemma")
            ?? TestGates.ModelSkip("TENSORSHARP_EMBEDDINGGEMMA2_MMPROJ", "mmproj");
}

[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
[AttributeUsage(AttributeTargets.Method)]
internal sealed class EmbeddingGemma2AudioCpuBlasFactAttribute : FactAttribute, ITraitAttribute
{
    public string RequiresValue => "GgmlCpu,Models";
    public EmbeddingGemma2AudioCpuBlasFactAttribute()
        => Skip = !OperatingSystem.IsLinux() ? "Optional audio BLAS qualification requires Linux." :
            TestGates.GgmlPinSkip(BackendType.GgmlCpu)
            ?? TestGates.ModelSkip("TENSORSHARP_EMBEDDINGGEMMA2_MMPROJ", "mmproj")
            ?? (Environment.GetEnvironmentVariable("TSG_EMBEDDING_LINUX_BLAS") != "1"
                ? "Optional audio BLAS qualification requires TSG_EMBEDDING_LINUX_BLAS=1." : null);
}
