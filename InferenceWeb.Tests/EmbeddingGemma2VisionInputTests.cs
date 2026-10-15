using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class EmbeddingGemma2VisionInputTests
{
    [Fact]
    public void CudaStackInteropHasTheVersionedNativeLayout()
    {
        var assembly = typeof(GgmlBasicOps).Assembly;
        Assert.Equal(528, System.Runtime.InteropServices.Marshal.SizeOf(assembly.GetType("TensorSharp.GGML.GgmlGemma4VisionStackBlock", true)!));
        Assert.Equal(128, System.Runtime.InteropServices.Marshal.SizeOf(assembly.GetType("TensorSharp.GGML.GgmlGemma4VisionStackArgs", true)!));
    }

    [Fact]
    public void CudaStackRefusesCpuStorageWithoutChangingCallerActivations()
    {
        var allocator = new TensorSharp.Cpu.CpuAllocator(BlasEnum.DotNet);
        using var input = new Tensor(allocator, DType.Float32, 1, 768);
        using var result = new Tensor(allocator, DType.Float32, 1, 768);
        float[] source = Enumerable.Repeat(.25f, 768).ToArray();
        float[] sentinel = Enumerable.Repeat(9876.5f, 768).ToArray();
        input.SetElementsAsFloat(source); result.SetElementsAsFloat(sentinel);
        Assert.False(GgmlBasicOps.TryGemma4VisionStack(result, input, new GgmlGemma4VisionBlock[16], [0], [0], 1e-6f, 100f));
        Assert.Equal(source, input.GetElementsAsFloat(768));
        Assert.Equal(sentinel, result.GetElementsAsFloat(768));
    }

    [EmbeddingGemma2VisionCudaFact]
    public void CudaEmbeddingVisionRetainsWeightsAcrossEncodesAndReleasesThemOnDispose()
    {
        string path = Environment.GetEnvironmentVariable("TENSORSHARP_EMBEDDINGGEMMA2_MMPROJ")!;
        if (Directory.Exists(path)) path = TestGates.FindGguf(path, "mmproj")!;
        var allocator = new GgmlAllocator(new GgmlContext([0], GgmlBackendType.Cuda), 0);
        const int width = 96, height = 96;
        const long blockMatrixBytes = 576L * 1024 * 1024;
        static bool UsedStack(Gemma4VisionEncoder encoder) => (bool)typeof(Gemma4VisionEncoder)
            .GetField("_lastGgmlVisionStackUsed", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.GetValue(encoder)!;
        float[] pixels = Enumerable.Range(0, width * height * 3).Select(i => (i % 251) / 250f).ToArray();
        // The suite serializes native tests. Start with an empty cache and
        // force a one-byte ceiling to exercise streaming refusal.
        // These test-only policy changes are restored even when a check fails.
        GgmlBasicOps.ClearHostBufferCache();
        GgmlBasicOps.SetDeviceCopyBudget(1);
        float[]? streamed = null;
        try
        {
            using var encoder = new Gemma4VisionEncoder(path, allocator,
                inputPixelsInZeroToOneRange: true, requireValidatedFloat32Precision: true);
            using Tensor output = encoder.Encode(pixels, width, height);
            Assert.False(UsedStack(encoder));
            streamed = output.GetElementsAsFloat((int)output.ElementCount());
            Assert.All(streamed, value => Assert.True(float.IsFinite(value)));
            Assert.Equal(0, GgmlBasicOps.DeviceCopyCacheResidentBytes());
        }
        finally { GgmlBasicOps.SetDeviceCopyBudget(0); }
        // Admit part of the tower, then refuse before compute. Filled cache
        // entries remain valid for the original one-layer streaming fallback.
        GgmlBasicOps.SetDeviceCopyBudget(38L * 1024 * 1024);
        try
        {
            using var encoder = new Gemma4VisionEncoder(path, allocator,
                inputPixelsInZeroToOneRange: true, requireValidatedFloat32Precision: true);
            using Tensor output = encoder.Encode(pixels, width, height);
            Assert.False(UsedStack(encoder));
            float[] partial = output.GetElementsAsFloat((int)output.ElementCount());
            Assert.All(partial, value => Assert.True(float.IsFinite(value)));
            Assert.InRange(GgmlBasicOps.DeviceCopyCacheResidentBytes(), 1, 38L * 1024 * 1024);
            Assert.InRange(partial.Zip(streamed!, (a, b) => Math.Abs(a - b)).Max(), 0, 2e-5f);
        }
        finally { GgmlBasicOps.SetDeviceCopyBudget(0); }
        long baseline = GgmlBasicOps.DeviceCopyCacheResidentBytes();
        float[]? firstLifetime = null;
        try
        {
            for (int lifetime = 0; lifetime < 2; ++lifetime)
            {
                try
                {
                    using var encoder = new Gemma4VisionEncoder(path, allocator,
                        inputPixelsInZeroToOneRange: true, requireValidatedFloat32Precision: true);
                    float[] Encode()
                    {
                        using Tensor output = encoder.Encode(pixels, width, height);
                        return output.GetElementsAsFloat((int)output.ElementCount());
                    }
                    float[] first = Encode();
                    Assert.True(UsedStack(encoder));
                    long residentAfterFirst = GgmlBasicOps.DeviceCopyCacheResidentBytes();
                    Assert.InRange(residentAfterFirst - baseline, blockMatrixBytes, blockMatrixBytes + 16L * 1024 * 1024);
                    Assert.Equal(streamed!.Length, first.Length);
                    Assert.All(first, value => Assert.True(float.IsFinite(value)));
                    Assert.InRange(first.Zip(streamed!, (a, b) => Math.Abs(a - b)).Max(), 0, 2e-5f);
                    for (int repeat = 0; repeat < 2; ++repeat)
                    {
                        float[] actual = Encode();
                        Assert.Equal(residentAfterFirst, GgmlBasicOps.DeviceCopyCacheResidentBytes());
                        Assert.Equal(first.Length, actual.Length);
                        Assert.All(actual, value => Assert.True(float.IsFinite(value)));
                        Assert.InRange(actual.Zip(first, (a, b) => Math.Abs(a - b)).Max(), 0, 2e-5f);
                    }
                    if (firstLifetime is not null)
                        Assert.InRange(first.Zip(firstLifetime, (a, b) => Math.Abs(a - b)).Max(), 0, 2e-5f);
                    firstLifetime = first;
                }
                finally
                {
                    // Dispose must invalidate every retained weight before its host
                    // allocation can be reused by the next tower in this process.
                    Assert.Equal(baseline, GgmlBasicOps.DeviceCopyCacheResidentBytes());
                }
            }
        }
        finally
        {
            GgmlBasicOps.SetDeviceCopyBudget(0);
            GgmlBasicOps.ClearHostBufferCache();
            GgmlBasicOps.ReleaseReuseComputeBuffers();
        }
    }

    [ModelFact("TENSORSHARP_EMBEDDINGGEMMA2_MMPROJ", GgmlBackend = BackendType.GgmlMetal)]
    public void RawImagePixelsMatchExplicitSiglipNormalizationAndLegacyConstructorContract()
    {
        string path = Environment.GetEnvironmentVariable("TENSORSHARP_EMBEDDINGGEMMA2_MMPROJ")!;
        var allocator = new GgmlAllocator(new GgmlContext([0], GgmlBackendType.Metal), 0);
        const int width = 96, height = 96;
        float[] raw = Enumerable.Range(0, width * height * 3).Select(i => (i % 251) / 250f).ToArray();
        float[] normalized = raw.Select(value => 2 * value - 1).ToArray();
        try
        {
            using var rawEncoder = new Gemma4VisionEncoder(path, allocator, inputPixelsInZeroToOneRange: true);
            Assert.False(rawEncoder.IsUnified);
            using var rawOutput = rawEncoder.Encode(raw, width, height);
            float[] actual = rawOutput.GetElementsAsFloat((int)rawOutput.ElementCount());
            using var legacyEncoder = new Gemma4VisionEncoder(path, allocator);
            using var expectedOutput = legacyEncoder.Encode(normalized, width, height);
            float[] expected = expectedOutput.GetElementsAsFloat((int)expectedOutput.ElementCount());
            Assert.Equal(expectedOutput.Sizes.ToArray(), rawOutput.Sizes.ToArray());
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; ++i)
            {
                Assert.True(float.IsFinite(actual[i]));
                Assert.InRange(Math.Abs(actual[i] - expected[i]), 0, 2e-5f);
            }
            // The existing two-argument contract consumes already normalized
            // pixels. A missing recenter operation must produce a detectable
            // difference on this nonuniform input rather than pass by accident.
            using var unnormalizedOutput = legacyEncoder.Encode(raw, width, height);
            float[] unnormalized = unnormalizedOutput.GetElementsAsFloat(expected.Length);
            Assert.True(unnormalized.Zip(expected, (a, b) => Math.Abs(a - b)).Max() > .01f);
        }
        finally
        {
            GgmlBasicOps.ReleaseReuseComputeBuffers();
            GgmlBasicOps.ClearHostBufferCache();
        }
    }
}
