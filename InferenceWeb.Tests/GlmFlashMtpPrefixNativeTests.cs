// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Memory;

namespace InferenceWeb.Tests;

public sealed partial class GlmFlashMtpNativeTests
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int PrefixVersion();
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int PrefixEnabled(IntPtr model);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int PrefixCommit(IntPtr model, int expectedBase, int acceptedDrafts, int verifyDrafts);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void PrefixFault(int stage, int kind);

    [GlmNativeHooksCudaFact]
    public void N2PrefixCommitPreservesScalarOracleAndPoisonsPartialCopies()
    {
        IntPtr library = NativeLibrary.Load(TestGates.MappedNativeGgmlOpsPath());
        try
        {
            T Export<T>(string name) where T : Delegate =>
                Marshal.GetDelegateForFunctionPointer<T>(NativeLibrary.GetExport(library, name));
            var version = Export<PrefixVersion>("TSGgml_GlmFlashMtpPrefixCommitVersion");
            var enabled = Export<PrefixEnabled>("TSGgml_GlmFlashMtpPrefixCommitEnabled");
            var commit = Export<PrefixCommit>("TSGgml_GlmCommitVerifiedPrefix");
            var fault = Export<PrefixFault>("TSGgml_GlmTestKdaSnapshotFault");
            Assert.Equal(1, version()); Assert.Equal(0, enabled(IntPtr.Zero));
            Assert.Equal(0, commit(IntPtr.Zero, 0, 0, 1));
            string path = Fixture();
            int[] prompt = [65, 66, 67, 68, 69];
            int[] continuation = [70, 71, 72, 73];
            var expectedLogits = new List<float[]>();
            var expectedHidden = new List<float[]>();
            float[] expectedWindowLogits = null, expectedWindowHidden = null;
            long baseSnapshot = 0;
            using (Environment())
            {
                Run(path, true, (model, _) =>
                {
                    Assert.Equal(0, enabled(model));
                    int hidden = GgmlGlmNative.HiddenSize(model), vocab = GgmlGlmNative.VocabSize(model);
                    var promptHidden = new float[prompt.Length * hidden]; var logits = new float[vocab];
                    Assert.True(GgmlGlmNative.SpecForward(model, prompt, promptHidden, logits, false));
                    Assert.True(GgmlGlmNative.KdaStateCapture(model));
                    baseSnapshot = SnapshotSize(model);
                    for (int i = 0; i < continuation.Length; ++i)
                    {
                        var h = new float[hidden]; var l = new float[vocab];
                        Assert.True(GgmlGlmNative.SpecForward(model, [continuation[i]], h, l, false));
                        Assert.All(h, v => Assert.True(float.IsFinite(v)));
                        Assert.All(l, v => Assert.True(float.IsFinite(v)));
                        expectedHidden.Add(h); expectedLogits.Add(l);
                    }
                    Assert.Equal(prompt.Length, GgmlGlmNative.KdaStateRestore(model));
                    expectedWindowHidden = new float[2 * hidden]; expectedWindowLogits = new float[2 * vocab];
                    Assert.True(GgmlGlmNative.SpecForward(model, continuation[..2], expectedWindowHidden, expectedWindowLogits, true));
                    Assert.Equal(expectedHidden[0], expectedWindowHidden[..hidden]);
                    Assert.Equal(expectedLogits[0], expectedWindowLogits[..vocab]);
                    Assert.Equal(expectedHidden[1], expectedWindowHidden[hidden..]);
                    Assert.Equal(expectedLogits[1], expectedWindowLogits[vocab..]);
                    Assert.Equal(0, commit(model, prompt.Length, 0, 1));
                });
            }
            using (Environment(prefixCommit: true))
            {
                Run(path, true, (model, budget) =>
                {
                    Assert.Equal(1, enabled(model));
                    int hidden = GgmlGlmNative.HiddenSize(model), vocab = GgmlGlmNative.VocabSize(model);
                    void Prefill()
                    {
                        Assert.True(GgmlGlmNative.ResetChecked(model));
                        Assert.True(GgmlGlmNative.SpecForward(model, prompt, new float[prompt.Length * hidden], new float[vocab], false));
                    }
                    void Verify()
                    {
                        var h = new float[2 * hidden]; var l = new float[2 * vocab];
                        Assert.True(GgmlGlmNative.SpecForward(model, continuation[..2], h, l, true));
                        Assert.All(h, v => Assert.True(float.IsFinite(v)));
                        Assert.All(l, v => Assert.True(float.IsFinite(v)));
                        Assert.Equal(expectedWindowHidden, h); Assert.Equal(expectedWindowLogits, l);
                    }
                    Prefill();
                    long before = budget.AllocationUsage.Where(v => v.Kind == 7).Sum(v => v.CommittedBytes);
                    fault(3, 1); Assert.False(GgmlGlmNative.KdaStateCapture(model));
                    Assert.Equal(prompt.Length, GgmlGlmNative.NPast(model));
                    Assert.Equal(2 * baseSnapshot, SnapshotSize(model));
                    Assert.Equal(SnapshotSize(model), budget.AllocationUsage.Where(v => v.Kind == 7).Sum(v => v.CommittedBytes) - before);
                    Assert.Equal(0, commit(model, prompt.Length, 0, 1));
                    Assert.True(GgmlGlmNative.KdaStateCapture(model));
                    Verify();
                    long owned = budget.AllocationUsage.Sum(v => v.CommittedBytes + v.PendingBytes);
                    foreach (var args in new[] { (prompt.Length - 1, 0, 1), (prompt.Length, 1, 1), (prompt.Length, 0, 2) })
                    {
                        Assert.Equal(0, commit(model, args.Item1, args.Item2, args.Item3));
                        Assert.Equal(prompt.Length + 2, GgmlGlmNative.NPast(model));
                        Assert.Equal(owned, budget.AllocationUsage.Sum(v => v.CommittedBytes + v.PendingBytes));
                    }
                    var seams = GgmlGlmNative.SelectedExpertExecutionStats(model);
                    Assert.Equal(1, commit(model, prompt.Length, 0, 1));
                    Assert.Equal(prompt.Length + 1, GgmlGlmNative.NPast(model));
                    Assert.Equal(seams, GgmlGlmNative.SelectedExpertExecutionStats(model)); // No trunk replay.
                    Assert.Equal(owned, budget.AllocationUsage.Sum(v => v.CommittedBytes + v.PendingBytes));
                    Assert.Equal(0, commit(model, prompt.Length, 0, 1)); // Consumed authority.
                    Assert.True(GgmlGlmNative.Rewind(model, prompt.Length + 1));
                    for (int i = 1; i < continuation.Length; ++i)
                    {
                        var h = new float[hidden]; var l = new float[vocab];
                        Assert.True(GgmlGlmNative.SpecForward(model, [continuation[i]], h, l, false));
                        Assert.Equal(expectedHidden[i], h); Assert.Equal(expectedLogits[i], l);
                    }

                    // Full acceptance leaves slot 0 of the K2 recurrence live.
                    // Skip prefix commit and prove the next scalar row sees the
                    // complete two-row state, including the convolution tail.
                    Prefill(); Assert.True(GgmlGlmNative.KdaStateCapture(model)); Verify();
                    var fullAcceptHidden = new float[hidden]; var fullAcceptLogits = new float[vocab];
                    Assert.True(GgmlGlmNative.SpecForward(model, [continuation[2]], fullAcceptHidden, fullAcceptLogits, false));
                    Assert.Equal(expectedHidden[2], fullAcceptHidden);
                    Assert.Equal(expectedLogits[2], fullAcceptLogits);
                    Assert.Equal(prompt.Length + 3, GgmlGlmNative.NPast(model));
                    Assert.Equal(0, commit(model, prompt.Length, 0, 1));

                    // Prefix snapshots are independent kind-7 arenas. Free the
                    // K2 graph's compute buffers before consuming its published
                    // prefix, then rebuild scalar compute and continue exactly.
                    Prefill(); Assert.True(GgmlGlmNative.KdaStateCapture(model)); Verify();
                    long snapshotBytesBeforeTrim = SnapshotSize(model);
                    long speculativeBeforeTrim = budget.AllocationUsage.Where(v => v.Kind == 7).Sum(v => v.CommittedBytes);
                    long graphBytesBeforeTrim = budget.AllocationUsage.Where(v => v.Kind is 2 or 5).Sum(v => v.CommittedBytes);
                    var seamsBeforeTrim = GgmlGlmNative.SelectedExpertExecutionStats(model);
                    Assert.True(GgmlGlmNative.ConfigureGraphCache(model, 8, trim: 1));
                    Assert.True(budget.AllocationUsage.Where(v => v.Kind is 2 or 5).Sum(v => v.CommittedBytes) < graphBytesBeforeTrim);
                    Assert.Equal(snapshotBytesBeforeTrim, SnapshotSize(model));
                    Assert.Equal(speculativeBeforeTrim, budget.AllocationUsage.Where(v => v.Kind == 7).Sum(v => v.CommittedBytes));
                    Assert.Equal(1, commit(model, prompt.Length, 0, 1));
                    Assert.Equal(prompt.Length + 1, GgmlGlmNative.NPast(model));
                    Assert.Equal(seamsBeforeTrim, GgmlGlmNative.SelectedExpertExecutionStats(model));
                    var afterTrimHidden = new float[hidden]; var afterTrimLogits = new float[vocab];
                    Assert.True(GgmlGlmNative.SpecForward(model, [continuation[1]], afterTrimHidden, afterTrimLogits, false));
                    Assert.Equal(expectedHidden[1], afterTrimHidden);
                    Assert.Equal(expectedLogits[1], afterTrimLogits);
                    Assert.Equal(prompt.Length + 2, GgmlGlmNative.NPast(model));

                    // Reuse the same captured graph and stable arena after reset.
                    Prefill(); Assert.True(GgmlGlmNative.KdaStateCapture(model)); Verify();
                    int other = GgmlGlmNative.SlotAlloc(model);
                    Assert.True(other > 0);
                    Assert.True(GgmlGlmNative.SetActiveSlot(model, other));
                    Assert.Equal(0, commit(model, prompt.Length, 0, 1));
                    Assert.True(GgmlGlmNative.SetActiveSlot(model, 0));
                    Assert.Equal(0, commit(model, prompt.Length, 0, 1));
                    Assert.True(GgmlGlmNative.SlotFree(model, other));
                    Assert.Equal(prompt.Length, GgmlGlmNative.KdaStateRestore(model));
                    Assert.Equal(0, commit(model, prompt.Length, 0, 1));
                    // Fresh same-position work cannot revive a prior window.
                    Verify(); Assert.Equal(1, commit(model, prompt.Length, 0, 1));

                    // A failed new capture invalidates the old prefix without touching live state.
                    Prefill(); Assert.True(GgmlGlmNative.KdaStateCapture(model)); Verify();
                    fault(1, 1); Assert.False(GgmlGlmNative.KdaStateCapture(model));
                    Assert.Equal(0, commit(model, prompt.Length, 0, 1));
                    Assert.Equal(prompt.Length + 2, GgmlGlmNative.NPast(model));

                    foreach (int kind in new[] { 1, 2 })
                    {
                        Prefill(); Assert.True(GgmlGlmNative.KdaStateCapture(model)); Verify();
                        fault(4, kind); Assert.Equal(-1, commit(model, prompt.Length, 0, 1));
                        Assert.Equal(prompt.Length + 2, GgmlGlmNative.NPast(model)); // Head never published early.
                        Assert.Equal(-1, GgmlGlmNative.KdaStateRestore(model));
                        Assert.Equal(-1, commit(model, prompt.Length, 0, 1));
                        var sentinel = Enumerable.Repeat(-123f, vocab).ToArray();
                        Assert.False(GgmlGlmNative.SpecForward(model, [70], new float[hidden], sentinel, false));
                        Assert.All(sentinel, v => Assert.Equal(-123f, v));
                        Prefill();
                        var h = new float[hidden]; var l = new float[vocab];
                        Assert.True(GgmlGlmNative.SpecForward(model, [70], h, l, false));
                        Assert.Equal(expectedHidden[0], h); Assert.Equal(expectedLogits[0], l);
                    }
                    Assert.Equal(2 * baseSnapshot, SnapshotSize(model));
                });
            }
        }
        finally { NativeLibrary.Free(library); }
    }

    [GlmNativeHooksCudaFact]
    public void N2PrefixCommitAndFullAcceptancePreserveExactStateAcross256RowPadding()
    {
        IntPtr library = NativeLibrary.Load(TestGates.MappedNativeGgmlOpsPath());
        try
        {
            var commit = Marshal.GetDelegateForFunctionPointer<PrefixCommit>(
                NativeLibrary.GetExport(library, "TSGgml_GlmCommitVerifiedPrefix"));
            var enabled = Marshal.GetDelegateForFunctionPointer<PrefixEnabled>(
                NativeLibrary.GetExport(library, "TSGgml_GlmFlashMtpPrefixCommitEnabled"));
            Directory.CreateDirectory(_directory);
            string path = GlmDsaSyntheticModelBuilder.WriteGlm5NextTpFixture(
                Path.Combine(_directory, "flash-mtp-prefix-boundary.gguf"),
                numHeads: 8, quantizeAttentionOutput: true, numLayers: 2, mixedAttention: true,
                routedExperts: true, quantizeExperts: true, contextLength: 512, indexerTopK: 512, includeMtp: true);
            int[] bases = [254, 255, 256];
            var oracles = new Dictionary<int, (int[] Tokens, float[][] Hidden, float[][] Logits)>();
            static int[] Prompt(int count) => Enumerable.Range(0, count).Select(i => 65 + i % 12).ToArray();
            static void ExactFinite(float[] expected, float[] actual)
            {
                Assert.All(expected, value => Assert.True(float.IsFinite(value)));
                Assert.All(actual, value => Assert.True(float.IsFinite(value)));
                Assert.Equal(expected, actual);
            }

            using (Environment())
            {
                RunPrefixBoundary(path, model =>
                {
                    Assert.Equal(0, enabled(model));
                    int hidden = GgmlGlmNative.HiddenSize(model), vocab = GgmlGlmNative.VocabSize(model);
                    foreach (int position in bases)
                    {
                        Assert.True(GgmlGlmNative.ResetChecked(model));
                        var logits = new float[vocab];
                        Assert.True(GgmlGlmNative.SpecForward(model, Prompt(position), new float[position * hidden], logits, false));
                        Assert.Equal(position, GgmlGlmNative.NPast(model));
                        Assert.True(GgmlGlmNative.KdaStateCapture(model));
                        int[] tokens = new int[3]; float[][] expectedHidden = new float[3][], expectedLogits = new float[3][];
                        for (int row = 0; row < 3; ++row)
                        {
                            tokens[row] = Argmax(logits);
                            expectedHidden[row] = new float[hidden]; expectedLogits[row] = new float[vocab];
                            Assert.True(GgmlGlmNative.SpecForward(model, [tokens[row]], expectedHidden[row], expectedLogits[row], false));
                            Assert.All(expectedHidden[row], value => Assert.True(float.IsFinite(value)));
                            Assert.All(expectedLogits[row], value => Assert.True(float.IsFinite(value)));
                            logits = expectedLogits[row];
                        }
                        oracles.Add(position, (tokens, expectedHidden, expectedLogits));
                        // Qualify ordinary K1 N2 arithmetic against the same
                        // scalar oracle at each side of the padding transition.
                        Assert.Equal(position, GgmlGlmNative.KdaStateRestore(model));
                        var h = new float[2 * hidden]; var l = new float[2 * vocab];
                        Assert.True(GgmlGlmNative.SpecForward(model, tokens[..2], h, l, true));
                        for (int row = 0; row < 2; ++row)
                        {
                            ExactFinite(expectedHidden[row], h[(row * hidden)..((row + 1) * hidden)]);
                            ExactFinite(expectedLogits[row], l[(row * vocab)..((row + 1) * vocab)]);
                        }
                    }
                });
            }

            using (Environment(prefixCommit: true))
            {
                RunPrefixBoundary(path, model =>
                {
                    Assert.Equal(1, enabled(model));
                    int hidden = GgmlGlmNative.HiddenSize(model), vocab = GgmlGlmNative.VocabSize(model);
                    foreach (int position in bases)
                    foreach (bool reject in new[] { true, false })
                    {
                        var oracle = oracles[position];
                        Assert.True(GgmlGlmNative.ResetChecked(model));
                        Assert.True(GgmlGlmNative.SpecForward(model, Prompt(position), new float[position * hidden], new float[vocab], false));
                        Assert.True(GgmlGlmNative.KdaStateCapture(model));
                        int second = reject ? (oracle.Tokens[1] + 1) % vocab : oracle.Tokens[1];
                        var h = new float[2 * hidden]; var l = new float[2 * vocab];
                        Assert.True(GgmlGlmNative.SpecForward(model, [oracle.Tokens[0], second], h, l, true));
                        Assert.All(h, value => Assert.True(float.IsFinite(value)));
                        Assert.All(l, value => Assert.True(float.IsFinite(value)));
                        Assert.Equal(position + 2, GgmlGlmNative.NPast(model));
                        ExactFinite(oracle.Hidden[0], h[..hidden]); ExactFinite(oracle.Logits[0], l[..vocab]);
                        if (reject)
                        {
                            Assert.NotEqual(second, Argmax(l[..vocab])); // Deterministic m0/k1 rejection.
                            var seams = GgmlGlmNative.SelectedExpertExecutionStats(model);
                            Assert.Equal(1, commit(model, position, 0, 1));
                            Assert.Equal(seams, GgmlGlmNative.SelectedExpertExecutionStats(model));
                            Assert.Equal(position + 1, GgmlGlmNative.NPast(model));
                            Assert.True(GgmlGlmNative.Rewind(model, position + 1));
                        }
                        else
                        {
                            Assert.Equal(second, Argmax(l[..vocab])); // Both verification rows remain accepted.
                            ExactFinite(oracle.Hidden[1], h[hidden..]); ExactFinite(oracle.Logits[1], l[vocab..]);
                        }
                        // At base255 the rejected row occupies physical KV row256.
                        // Its corrected replacement must be read with extent512;
                        // K2 full acceptance must also retain the correct final state.
                        for (int row = reject ? 1 : 2; row < 3; ++row)
                        {
                            var nextHidden = new float[hidden]; var nextLogits = new float[vocab];
                            Assert.True(GgmlGlmNative.SpecForward(model, [oracle.Tokens[row]], nextHidden, nextLogits, false));
                            ExactFinite(oracle.Hidden[row], nextHidden); ExactFinite(oracle.Logits[row], nextLogits);
                            Assert.Equal(position + row + 1, GgmlGlmNative.NPast(model));
                        }
                        Assert.Equal(0, commit(model, position, 0, 1));
                    }
                });
            }
        }
        finally { NativeLibrary.Free(library); }
    }

    private static void RunPrefixBoundary(string path, Action<IntPtr> action)
    {
        GgmlBasicOps.ClearHostBufferCache();
        GgmlBasicOps.ReleaseReuseComputeBuffers();
        var budget = new MemoryBudget([new("ram", 2L << 30), new("gpu", 2L << 30)]);
        using var native = new GgmlCacheBudgetScope(budget, [["gpu"], ["ram"]], true, ["ram"]);
        using var speculation = budget.Reserve([new("gpu", 256L << 20)]);
        native.SetSharedSpeculativeEnvelope(speculation);
        using var expertOwner = budget.Reserve([new("gpu", 256L << 20)]);
        native.SetSharedExpertEnvelope(expertOwner);
        using var experts = new GgmlHostMoeExpertCacheScope(256L << 20, 64, 1, 3);
        IntPtr model = GgmlGlmNative.LoadModel(path, 1, 512, 64, 2, nCpuMoe: int.MaxValue,
            backendName: "CUDA", tp: 1, ctxIsHardLimit: true, loadMtp: true);
        Assert.NotEqual(IntPtr.Zero, model);
        try
        {
            Assert.True(GgmlGlmNative.HasDraftHead(model));
            Assert.Equal(1, GgmlGlmNative.SelectedExpertGroupCount(model));
            long capacity = GgmlGlmNative.SelectedExpertMinimumBudget(model, 3);
            Assert.InRange(capacity, 1, experts.MaximumBudgetBytes);
            experts.UpdateBudget(capacity);
            Assert.True(GgmlGlmNative.ConfigureSelectedExpertSeams(model, true));
            action(model);
        }
        finally
        {
            GgmlGlmNative.Free(model);
            experts.Dispose();
            native.SetSharedExpertEnvelope(null);
            native.SetSharedSpeculativeEnvelope(null);
        }
        Assert.Null(native.CallbackError);
        Assert.Equal(0, native.ActiveAllocations);
    }
}
