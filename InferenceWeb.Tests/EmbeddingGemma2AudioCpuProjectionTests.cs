using System.Reflection;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class EmbeddingGemma2AudioCpuProjectionTests(ITestOutputHelper output)
{
    [EmbeddingGemma2AudioCpuBlasFact]
    public void QualifiedCpuAudioProjectionMatchesDenseGraphsAcrossEncoderLifetimes()
    {
        string path = Environment.GetEnvironmentVariable("TENSORSHARP_EMBEDDINGGEMMA2_MMPROJ")!;
        if (Directory.Exists(path)) path = TestGates.FindGguf(path, "mmproj")!;
        var allocator = new GgmlAllocator(new GgmlContext([0], GgmlBackendType.Cpu), 0);
        const int frames = 99;
        float[] mel = Enumerable.Range(0, frames * 128).Select(i => MathF.Log(.001f + (i % 19) * .005f)).ToArray();
        var requirement = typeof(Gemma4AudioEncoder).GetField("_requireGgmlFloat32Precision", BindingFlags.Instance | BindingFlags.NonPublic)!;
        string[] flags = ["TS_GEMMA4_AUDIO_SSCP_GRAPH", "TS_GEMMA4_AUDIO_BLOCK_GRAPH"];
        string?[] previous = flags.Select(Environment.GetEnvironmentVariable).ToArray();
        float[]? baseline = null;
        try
        {
            // Make both constructors submit the same complete native graphs;
            // only the explicit per-instance CPU projection requirement differs.
            foreach (string flag in flags) Environment.SetEnvironmentVariable(flag, "1");
            for (int lifetime = 0; lifetime < 3; ++lifetime)
            {
                bool precise = lifetime != 0;
                using var encoder = precise ? new Gemma4AudioEncoder(path, allocator, true) : new Gemma4AudioEncoder(path, allocator);
                Assert.Equal(precise, (bool)requirement.GetValue(encoder)!);
                Assert.False(encoder.IsEncoderFree);
                for (int repeat = 0; repeat < (precise ? 2 : 1); ++repeat)
                {
                    using Tensor result = encoder.Encode(mel, frames);
                    Assert.Equal(25, result.Sizes[0]);
                    float[] values = result.GetElementsAsFloat(checked((int)result.ElementCount()));
                    Assert.All(values, value => Assert.True(float.IsFinite(value)));
                    if (baseline is null) { baseline = values; continue; }
                    Assert.Equal(baseline.Length, values.Length);
                    double maximum = values.Zip(baseline, (a, b) => Math.Abs((double)a - b)).Max();
                    double dot = values.Zip(baseline, (a, b) => (double)a * b).Sum();
                    double norm = Math.Sqrt(values.Sum(a => (double)a * a) * baseline.Sum(a => (double)a * a));
                    Assert.True(norm > 0);
                    output.WriteLine($"lifetime {lifetime}, repeat {repeat}: dense/optional projection cosine {dot / norm:R}, maximum {maximum:R}");
                    Assert.InRange(dot / norm, .9999, 1.00000001);
                    Assert.InRange(maximum, 0, .002);
                }
            }
        }
        finally
        {
            for (int i = 0; i < flags.Length; ++i) Environment.SetEnvironmentVariable(flags[i], previous[i]);
            GgmlBasicOps.ReleaseReuseComputeBuffers();
            GgmlBasicOps.ClearHostBufferCache();
        }
    }
}
