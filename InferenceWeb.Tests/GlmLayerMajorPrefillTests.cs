// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

/// <summary>Layer-major execution must preserve the full logits and subsequent
/// recurrent/MLA state while reusing position-independent KDA graphs.</summary>
public sealed class GlmLayerMajorPrefillTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-glm-layer-major-" + Guid.NewGuid().ToString("N"));

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void FailGraphAllocs(int n);

    public void Dispose()
    {
        if (Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
    }

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData(3, 4)]
    [InlineData(65, 4)]
    [InlineData(65, 1024)]
    public void MixedAttention_LayerMajorMatchesChunkOrderAndContinuation(int chunk, int topK)
    {
        using var env = ConfigureEnvironment();
        string path = Fixture(mixed: true, topK);
        IntPtr handle = Load(path, chunk);
        try
        {
            int[] prompt = Tokens(277);
            var expected = new float[GgmlGlmNative.VocabSize(handle)];
            Assert.True(GgmlGlmNative.Forward(handle, prompt, expected));
            var expectedAfter = new float[expected.Length];
            Assert.True(GgmlGlmNative.Forward(handle, [93], expectedAfter));

            Assert.True(GgmlGlmNative.ResetChecked(handle));
            var actual = new float[expected.Length];
            Assert.Equal(1, GgmlGlmNative.ForwardLayerMajor(handle, prompt, actual, chunk));
            Assert.Equal(prompt.Length, GgmlGlmNative.NPast(handle));
            Assert.Equal(expected, actual);
            Assert.True(GgmlGlmNative.Forward(handle, [93], actual));
            Assert.Equal(expectedAfter, actual);
        }
        finally { GgmlGlmNative.Free(handle); }
    }

    [GlmNativeHooksFact]
    public void RecurrentGraph_ReplaysAcrossKvPaddingAndSparseBoundariesWithoutAllocation()
    {
        using var env = ConfigureEnvironment();
        // Each pair of 65-row chunks needs only the no-head and head graph.
        // The second forward crosses both the 256-key and top-k=128 boundaries.
        IntPtr handle = Load(Fixture(mixed: false, topK: 128), 65);
        IntPtr library = IntPtr.Zero;
        FailGraphAllocs fail = null;
        try
        {
            library = NativeLibrary.Load(TestGates.MappedNativeGgmlOpsPath());
            fail = Marshal.GetDelegateForFunctionPointer<FailGraphAllocs>(
                NativeLibrary.GetExport(library, "TSGgml_GlmTestFailGraphAllocs"));
            Assert.True(GgmlGlmNative.ConfigureGraphCache(handle, 2, trim: 1));
            int[] prompt = Tokens(130);
            var expected = new float[GgmlGlmNative.VocabSize(handle)];
            Assert.True(GgmlGlmNative.Forward(handle, prompt, expected));
            Assert.True(GgmlGlmNative.Forward(handle, prompt, expected));
            var expectedAfter = new float[expected.Length];
            Assert.True(GgmlGlmNative.Forward(handle, [93], expectedAfter));

            Assert.True(GgmlGlmNative.ResetChecked(handle));
            Assert.True(GgmlGlmNative.ConfigureGraphCache(handle, 2, trim: 1));
            var actual = new float[expected.Length];
            Assert.Equal(1, GgmlGlmNative.ForwardLayerMajor(handle, prompt, actual, 65));
            fail(1);
            try
            {
                Assert.Equal(1, GgmlGlmNative.ForwardLayerMajor(handle, prompt, actual, 65));
                Assert.Equal(260, GgmlGlmNative.NPast(handle));
                Assert.Equal(expected, actual);
            }
            finally { fail(0); }
            Assert.True(GgmlGlmNative.Forward(handle, [93], actual));
            Assert.Equal(expectedAfter, actual);
        }
        finally
        {
            fail?.Invoke(0);
            GgmlGlmNative.Free(handle);
            if (library != IntPtr.Zero) NativeLibrary.Free(library);
        }
    }

    private string Fixture(bool mixed, int topK)
    {
        Directory.CreateDirectory(_directory);
        return GlmDsaSyntheticModelBuilder.WriteGlm5NextTpFixture(
            Path.Combine(_directory, "fixture.gguf"), numHeads: 4, quantizeAttentionOutput: true,
            numLayers: mixed ? 2 : 1, mixedAttention: mixed, routedExperts: true,
            quantizeExperts: true, contextLength: 512, indexerTopK: topK);
    }

    private static IntPtr Load(string path, int chunk)
    {
        IntPtr handle = GgmlGlmNative.LoadModel(path, 1, 512, chunk, 2,
            backendName: "CPU", ctxIsHardLimit: true, loadMtp: false);
        Assert.NotEqual(IntPtr.Zero, handle);
        return handle;
    }

    private static int[] Tokens(int count)
        => Enumerable.Range(0, count).Select(i => 3 + (i * 17 + i / 7) % 120).ToArray();

    private static NativeEnvScope ConfigureEnvironment()
    {
        var env = new NativeEnvScope();
        foreach (string name in new[] { "TS_GLM_TRACE", "TS_GLM_TRACE_DUMP_DIR", "TS_GLM_NODES_PER_LAYER",
            "TS_GLM_MOE_MMAP", "TS_GLM_TP_SHARD", "TS_GLM_TP_OVERSUBSCRIBE" }) env.Set(name, null);
        return env;
    }
}
