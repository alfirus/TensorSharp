// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Text;
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Memory;

namespace InferenceWeb.Tests;

// Run alone with TS_TEST_GLM_CUDA=1 and TS_GLM_FLASH_MTP=1. These assertions
// require actual CUDA draft/target execution; unsupported devices are skipped
// by the gate, never counted as native model coverage.
public sealed partial class GlmFlashMtpNativeTests : IDisposable
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int ReferenceForward(IntPtr model, [In] int[] tokens, int count,
        [In] float[] hidden, int position, [Out] float[] logits, [Out] float[] outputHidden);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate long SnapshotBytes(IntPtr model);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void PriorBackendError();
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int BroadExpertPlan(IntPtr model);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int ExpertPlan(IntPtr model, [Out] long[] values, int length);

    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-flash-mtp-" + Guid.NewGuid().ToString("N"));

    [GlmNativeCudaFact]
    public void DraftWithoutHyperConnectionsPreservesTargetGreedyAndRejectedWindowRollback()
    {
        using var env = Environment();
        string path = Fixture();
        string bytes = Encoding.UTF8.GetString(File.ReadAllBytes(path));
        Assert.Contains("blk.2.nextn.eh_proj.weight", bytes);
        Assert.DoesNotContain("blk.2.hc_", bytes);
        int[] prompt = [65, 66, 67, 68, 69];
        var tokens = new List<int>();
        var expected = new List<float[]>();
        float[] promptLogits = null;
        Run(path, false, (model, _) =>
        {
            var logits = new float[GgmlGlmNative.VocabSize(model)];
            Assert.True(GgmlGlmNative.Forward(model, prompt, logits), GgmlBasicOps.LastNativeError());
            promptLogits = (float[])logits.Clone();
            for (int i = 0; i < 9; ++i)
            {
                int token = Argmax(logits);
                tokens.Add(token);
                Assert.True(GgmlGlmNative.Forward(model, [token], logits));
                expected.Add((float[])logits.Clone());
            }
        });
        Run(path, true, (model, budget) =>
        {
            int hidden = GgmlGlmNative.HiddenSize(model), vocab = GgmlGlmNative.VocabSize(model);
            var allHidden = new float[prompt.Length * hidden];
            var logits = new float[vocab];
            Assert.True(GgmlGlmNative.SpecForward(model, prompt, allHidden, logits, false), GgmlBasicOps.LastNativeError());
            Assert.Equal(promptLogits, logits);
            var paired = new float[allHidden.Length];
            Array.Copy(allHidden, 0, paired, hidden, (prompt.Length - 1) * hidden);
            Assert.True(GgmlGlmNative.DraftCatchUp(model, prompt, paired, 0));
            Assert.Equal(prompt.Length, GgmlGlmNative.NPast(model));
            Assert.Equal(0, SnapshotSize(model)); // Draft graph charges are not snapshot bytes.
            var previous = allHidden[^hidden..];
            for (int i = 0; i < 8;)
            {
                int position = prompt.Length + i;
                var before = GgmlGlmNative.SelectedExpertExecutionStats(model);
                var draftLogits = new float[vocab];
                var draftHidden = new float[hidden];
                Assert.True(GgmlGlmNative.DraftStep(model, tokens[i], previous, position, draftLogits, draftHidden));
                Assert.All(draftLogits, value => Assert.True(float.IsFinite(value)));
                Assert.Contains(draftHidden, value => value != 0);
                Assert.True(GgmlGlmNative.SelectedExpertExecutionStats(model).Seams > before.Seams);
                Assert.Equal(position, GgmlGlmNative.NPast(model));

                // Alternate zero and one accepted draft tokens before rejection.
                // Every accepted verification row must retain scalar target
                // logits; restore/replay must retain the same KDA state.
                int acceptedCount = i % 3 == 0 && i + 2 < 8 ? 2 : 1;
                int rejected = (tokens[i + acceptedCount] + 1) % vocab;
                int[] verification = [.. tokens.Skip(i).Take(acceptedCount), rejected];
                long snapshotBefore = SnapshotSize(model);
                long committedBefore = budget.AllocationUsage.Where(usage => usage.Kind == 7).Sum(usage => usage.CommittedBytes);
                Assert.True(GgmlGlmNative.KdaStateCapture(model));
                long snapshotAfter = SnapshotSize(model);
                long committedAfter = budget.AllocationUsage.Where(usage => usage.Kind == 7).Sum(usage => usage.CommittedBytes);
                Assert.True(snapshotAfter > 0);
                Assert.Equal(snapshotAfter - snapshotBefore, committedAfter - committedBefore);
                var verifyHidden = new float[verification.Length * hidden];
                var verifyLogits = new float[verification.Length * vocab];
                Assert.True(GgmlGlmNative.SpecForward(model, verification, verifyHidden, verifyLogits, true));
                for (int row = 0; row < acceptedCount; ++row)
                    Assert.Equal(expected[i + row], verifyLogits[(row * vocab)..((row + 1) * vocab)]);
                Assert.NotEqual(rejected, Argmax(verifyLogits[((acceptedCount - 1) * vocab)..(acceptedCount * vocab)]));
                Assert.Equal(position, GgmlGlmNative.KdaStateRestore(model));
                for (int row = 0; row < acceptedCount; ++row)
                {
                    var acceptedHidden = new float[hidden];
                    Assert.True(GgmlGlmNative.SpecForward(model, [tokens[i + row]], acceptedHidden, logits, false));
                    Assert.Equal(expected[i + row], logits);
                    Assert.Equal(tokens[i + row + 1], Argmax(logits));
                    Assert.Equal(position + row + 1, GgmlGlmNative.NPast(model));
                    Assert.True(GgmlGlmNative.DraftCatchUp(model, [tokens[i + row]], previous, position + row));
                    previous = acceptedHidden;
                }
                i += acceptedCount;
            }
            Assert.Contains(budget.AllocationUsage, usage => usage.Kind == 7 && usage.CommittedBytes > 0);
            var sentinel = Enumerable.Repeat(-123f, vocab).ToArray();
            Assert.False(GgmlGlmNative.DraftStep(model, 65, previous, 256, sentinel, new float[hidden]));
            Assert.All(sentinel, value => Assert.Equal(-123f, value));
        });
    }

    [GlmNativeHooksCudaFact]
    public void PublicDraftCoordinatesMatchReferenceInputsForWholeAndChunkedCatchUp()
    {
        using var env = Environment();
        string path = Fixture();
        int[] prompt = [65, 66, 67, 68, 69];
        IntPtr library = NativeLibrary.Load(TestGates.MappedNativeGgmlOpsPath());
        try
        {
            var reference = Marshal.GetDelegateForFunctionPointer<ReferenceForward>(
                NativeLibrary.GetExport(library, "TSGgml_GlmTestMtpReferenceForward"));
            var priorError = Marshal.GetDelegateForFunctionPointer<PriorBackendError>(
                NativeLibrary.GetExport(library, "TSGgml_GlmTestPriorBackendError"));
            foreach (int chunk in new[] { 1, prompt.Length - 1, prompt.Length })
            {
                float[] referenceLogits = null, referenceHidden = null;
                for (int mode = 0; mode < 2; ++mode)
                {
                    bool raw = mode == 0;
                    Run(path, true, (model, _) =>
                    {
                        int width = GgmlGlmNative.HiddenSize(model), vocab = GgmlGlmNative.VocabSize(model);
                        var targetHidden = new float[prompt.Length * width];
                        var targetLogits = new float[vocab];
                        Assert.True(GgmlGlmNative.SpecForward(model, prompt, targetHidden, targetLogits, false),
                            GgmlBasicOps.LastNativeError());
                        var paired = new float[targetHidden.Length];
                        Array.Copy(targetHidden, 0, paired, width, targetHidden.Length - width);
                        // A historical backend error advances the process epoch
                        // without poisoning this slot. KV-only graphs have no
                        // expert seams and must establish a fresh error fence.
                        priorError();
                        var beforeCatchUp = GgmlGlmNative.SelectedExpertExecutionStats(model);
                        for (int start = 0; start < prompt.Length; start += chunk)
                        {
                            int count = Math.Min(chunk, prompt.Length - start);
                            if (raw)
                            {
                                // Independent reference convention: token[p+1]
                                // pairs with target hidden[p] at draft position p.
                                int firstToken = Math.Max(1, start);
                                int end = start + count;
                                if (firstToken == end) continue;
                                Assert.Equal(1, reference(model, prompt[firstToken..end], end - firstToken,
                                    targetHidden[((firstToken - 1) * width)..((end - 1) * width)],
                                    firstToken - 1, null, new float[(end - firstToken) * width]));
                            }
                            else Assert.True(GgmlGlmNative.DraftCatchUp(model, prompt[start..(start + count)],
                                paired[(start * width)..((start + count) * width)], start));
                        }
                        var afterCatchUp = GgmlGlmNative.SelectedExpertExecutionStats(model);
                        // The oracle requests hidden output to execute the full
                        // draft block. Public catch-up must produce identical KV
                        // state for the next step without any expert execution.
                        if (raw) Assert.True(afterCatchUp.Seams > beforeCatchUp.Seams);
                        else Assert.Equal(beforeCatchUp.Seams, afterCatchUp.Seams);
                        int next = Argmax(targetLogits);
                        var logits = new float[vocab];
                        var hidden = new float[width];
                        if (raw) Assert.Equal(1, reference(model, [next], 1, targetHidden[^width..],
                            prompt.Length - 1, logits, hidden));
                        else Assert.True(GgmlGlmNative.DraftStep(model, next, targetHidden[^width..],
                            prompt.Length, logits, hidden));
                        Assert.Equal(prompt.Length, GgmlGlmNative.NPast(model));
                        Assert.All(logits, value => Assert.True(float.IsFinite(value)));
                        Assert.All(hidden, value => Assert.True(float.IsFinite(value)));
                        if (raw) { referenceLogits = logits; referenceHidden = hidden; }
                        else { Assert.Equal(referenceLogits, logits); Assert.Equal(referenceHidden, hidden); }
                    });
                }
            }
        }
        finally { NativeLibrary.Free(library); }
    }

    [GlmNativeHooksCudaFact]
    public void ShapeSpecificGroupsPreserveFullOracleAndRejectBeforeDraftKvWrites()
    {
        using var env = Environment(shapePlan: true);
        Assert.True(GgmlGlmNative.SupportsFlashMtpShapePlan());
        IntPtr library = NativeLibrary.Load(TestGates.MappedNativeGgmlOpsPath());
        try
        {
            var reference = Marshal.GetDelegateForFunctionPointer<ReferenceForward>(
                NativeLibrary.GetExport(library, "TSGgml_GlmTestMtpReferenceForward"));
            var readPlan = Marshal.GetDelegateForFunctionPointer<ExpertPlan>(
                NativeLibrary.GetExport(library, "TSGgml_GlmTestMtpExpertPlan"));
            var broadPlan = Marshal.GetDelegateForFunctionPointer<BroadExpertPlan>(
                NativeLibrary.GetExport(library, "TSGgml_GlmTestMtpBroadExpertPlan"));
            foreach (bool distinct in new[] { false, true })
            {
                string path = Fixture(distinct);
                float[] expectedLogits = null, expectedHidden = null;
                long broadMinimum = 0;
                // Independent fresh models: full-block oracle, public KV-only,
                // then public KV-only with rejected over-wide calls interposed.
                for (int mode = 0; mode < 3; ++mode)
                {
                    bool raw = mode == 0;
                    Run(path, true, (model, budget) =>
                    {
                        Assert.Equal(!raw, GgmlGlmNative.FlashMtpShapePlanEnabled(model));
                        Assert.Equal(0, broadPlan(model)); // Immutable after installation.
                        long[] plan = new long[5];
                        Assert.Equal(1, readPlan(model, plan, plan.Length));
                        Assert.Equal(distinct && !raw ? 1 : 3, plan[0]);
                        Assert.Equal(distinct && !raw ? 1 : 8, plan[1]);
                        if (raw) broadMinimum = plan[4];
                        else if (distinct) { Assert.True(plan[4] < broadMinimum); Assert.Equal(plan[2], plan[3]); }
                        else Assert.Equal(broadMinimum, plan[4]);
                        int[] prompt = [65, 66, 67, 68, 69];
                        int hidden = GgmlGlmNative.HiddenSize(model), vocab = GgmlGlmNative.VocabSize(model);
                        float[] targetHidden = new float[prompt.Length * hidden], targetLogits = new float[vocab];
                        Assert.True(GgmlGlmNative.SpecForward(model, prompt, targetHidden, targetLogits, false));
                        var before = GgmlGlmNative.SelectedExpertExecutionStats(model);
                        // Shared-group opt-in must also accept the trunk-sized
                        // full oracle; only a distinct draft group is scalar.
                        if (raw || (!distinct && mode == 2))
                            Assert.Equal(1, reference(model, prompt[1..], prompt.Length - 1,
                                targetHidden[..^hidden], 0, null, new float[(prompt.Length - 1) * hidden]));
                        else
                        {
                            float[] paired = new float[targetHidden.Length];
                            Array.Copy(targetHidden, 0, paired, hidden, targetHidden.Length - hidden);
                            Assert.True(GgmlGlmNative.DraftCatchUp(model, prompt, paired, 0));
                            Assert.Equal(before.Seams, GgmlGlmNative.SelectedExpertExecutionStats(model).Seams);
                        }
                        if (distinct && mode == 2)
                        {
                            var beforeRefusal = GgmlGlmNative.SelectedExpertExecutionStats(model);
                            var beforeCache = GgmlBasicOps.GetHostMoeExpertCacheUsage();
                            long owned = budget.AllocationUsage.Sum(v => v.CommittedBytes + v.PendingBytes);
                            foreach (int rows in new[] { 2, 4 }) foreach (bool logitsOnly in new[] { false, true })
                            {
                                var logits = Enumerable.Repeat(-123f, vocab).ToArray();
                                var outputHidden = Enumerable.Repeat(-123f, rows * hidden).ToArray();
                                // Overwrite old draft positions if erroneously executed:
                                // subsequent output equality detects hidden KV damage.
                                Assert.Equal(0, reference(model, prompt[..rows], rows,
                                    new float[rows * hidden], 0, logitsOnly ? logits : null,
                                    logitsOnly ? null : outputHidden));
                                Assert.All(logits, v => Assert.Equal(-123f, v));
                                Assert.All(outputHidden, v => Assert.Equal(-123f, v));
                            }
                            Assert.Equal(beforeRefusal, GgmlGlmNative.SelectedExpertExecutionStats(model));
                            Assert.Equal(beforeCache, GgmlBasicOps.GetHostMoeExpertCacheUsage());
                            Assert.Equal(owned, budget.AllocationUsage.Sum(v => v.CommittedBytes + v.PendingBytes));
                        }
                        var nextLogits = new float[vocab]; var nextHidden = new float[hidden];
                        Assert.True(GgmlGlmNative.DraftStep(model, Argmax(targetLogits), targetHidden[^hidden..],
                            prompt.Length, nextLogits, nextHidden));
                        Assert.All(nextLogits, value => Assert.True(float.IsFinite(value)));
                        Assert.All(nextHidden, value => Assert.True(float.IsFinite(value)));
                        if (raw) { expectedLogits = nextLogits; expectedHidden = nextHidden; }
                        else { Assert.Equal(expectedLogits, nextLogits); Assert.Equal(expectedHidden, nextHidden); }
                        if (distinct && !raw)
                        {
                            var entry = Assert.Single(GgmlBasicOps.GetHostMoeExpertCacheEntries().Where(e => e.Layer == 2));
                            Assert.Equal(1, entry.PlannedDecodeRows);
                            Assert.True(entry.Capacity >= 2);
                        }
                    }, groups: distinct ? 2 : 1, broad: raw ? broadPlan : null, checkMinimum: true);
                }
            }
        }
        finally { NativeLibrary.Free(library); }
    }

    private void Run(string path, bool mtp, Action<IntPtr, GgmlCacheBudgetScope> action,
        int groups = 1, BroadExpertPlan broad = null, bool checkMinimum = false)
    {
        GgmlBasicOps.ClearHostBufferCache();
        GgmlBasicOps.ReleaseReuseComputeBuffers();
        var budget = new MemoryBudget([new("ram", 2L << 30), new("gpu", 2L << 30)]);
        using var native = new GgmlCacheBudgetScope(budget, [["gpu"], ["ram"]], true, ["ram"]);
        using var speculation = budget.Reserve([new("gpu", 256L << 20)]);
        native.SetSharedSpeculativeEnvelope(speculation);
        // Planning uses the active scope's exact prefill/decode limits. Back
        // its initial ceiling before loading, then shrink to the measured
        // minimum before dispatch so the test exercises that minimum budget.
        using var expertOwner = budget.Reserve([new("gpu", 256L << 20)]);
        native.SetSharedExpertEnvelope(expertOwner);
        using var experts = new GgmlHostMoeExpertCacheScope(256L << 20, 8, groups, 3);
        IntPtr model = GgmlGlmNative.LoadModel(path, 1, 256, 8, 2, nCpuMoe: int.MaxValue,
            backendName: "CUDA", tp: 1, ctxIsHardLimit: true, loadMtp: mtp);
        Assert.NotEqual(IntPtr.Zero, model);
        try
        {
            Assert.Equal(mtp, GgmlGlmNative.HasDraftHead(model));
            if (broad != null) Assert.Equal(1, broad(model));
            // Groups identify projection triples. The optional distinct fixture
            // changes only the draft down projection, keeping trunk arithmetic.
            Assert.Equal(groups, GgmlGlmNative.SelectedExpertGroupCount(model));
            long capacity = GgmlGlmNative.SelectedExpertMinimumBudget(model, 3);
            Assert.InRange(capacity, 1, experts.MaximumBudgetBytes);
            if (checkMinimum)
            {
                experts.UpdateBudget(capacity - 1);
                long owned = native.AllocationUsage.Sum(v => v.CommittedBytes + v.PendingBytes);
                Assert.False(GgmlGlmNative.ConfigureSelectedExpertSeams(model, true));
                Assert.Equal(owned, native.AllocationUsage.Sum(v => v.CommittedBytes + v.PendingBytes));
                Assert.Empty(GgmlBasicOps.GetHostMoeExpertCacheEntries());
            }
            experts.UpdateBudget(capacity);
            Assert.Equal(capacity, experts.BudgetBytes);
            Assert.True(GgmlGlmNative.ConfigureSelectedExpertSeams(model, true));
            action(model, native);
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

    private string Fixture(bool distinct = false)
    {
        Directory.CreateDirectory(_directory);
        return GlmDsaSyntheticModelBuilder.WriteGlm5NextTpFixture(Path.Combine(_directory, distinct ? "flash-mtp-distinct.gguf" : "flash-mtp.gguf"),
            numHeads: 8, quantizeAttentionOutput: true, numLayers: 2, mixedAttention: true,
            routedExperts: true, quantizeExperts: true, contextLength: 256, indexerTopK: 256, includeMtp: true,
            distinctMtpExperts: distinct);
    }
    private static long SnapshotSize(IntPtr model)
    {
        IntPtr library = NativeLibrary.Load(TestGates.MappedNativeGgmlOpsPath());
        try
        {
            return Marshal.GetDelegateForFunctionPointer<SnapshotBytes>(
                NativeLibrary.GetExport(library, "TSGgml_GlmKdaSnapshotBytes"))(model);
        }
        finally { NativeLibrary.Free(library); }
    }
    private static NativeEnvScope Environment(bool shapePlan = false, bool prefixCommit = false)
    {
        var env = new NativeEnvScope();
        env.Set("TS_GLM_FLASH_MTP", "1");
        env.Set("TS_GLM_FLASH_MTP_SHAPE_PLAN", shapePlan ? "1" : null);
        env.Set("TS_GLM_FLASH_MTP_PREFIX_COMMIT", prefixCommit ? "1" : null);
        foreach (string name in new[] { "TS_GLM_TRACE", "TS_GLM_TRACE_DUMP_DIR", "TS_GLM_NODES_PER_LAYER",
            "TS_GLM_MOE_MMAP", "TS_GLM_TP_SHARD", "TS_GLM_TP_OVERSUBSCRIBE" }) env.Set(name, null);
        return env;
    }
    private static int Argmax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; ++i) if (values[i] > values[best]) best = i;
        return best;
    }
    public void Dispose()
    {
        if (Directory.Exists(_directory)) Directory.Delete(_directory, true);
    }
}
