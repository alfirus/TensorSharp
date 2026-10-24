// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class GgmlHostMoeVmmWeightsScopeTests
{
    private sealed class Native : IHostMoeExpertCacheScopeNative
    {
        internal readonly List<string> Calls = [];
        internal GgmlHostMoeVmmWeightsUsage Usage = new(1, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0);
        internal bool FailConfigure, FailTrim, FailClose, FailEnd, RetainOnClose;
        internal int Version = 1;
        internal long HostPlan = 100;
        public ulong Begin(long bytes, int rows, int layers, int decode) { Calls.Add("begin"); return 7; }
        public long Update(ulong token, long bytes) { Calls.Add("update"); return 0; }
        public long Trim(ulong token, long bytes)
        { Calls.Add("trim"); if (FailTrim) throw new InvalidOperationException("trim"); return 0; }
        public long End(ulong token)
        { Calls.Add("end"); if (FailEnd) throw new InvalidOperationException("end"); return 0; }
        public GgmlHostMoeExpertCacheUsage ReadUsage() => new(0, 1000, 0, 0, 0);
        public int VmmWeightsVersion() { Calls.Add("version"); return Version; }
        public long PlanVmmHostBytes() { Calls.Add("plan"); return HostPlan; }
        public void ConfigureVmmWeights(ulong token)
        {
            Calls.Add("configure"); Usage = Usage with { Configured = 1, HostTicketOwners = 2 };
            if (FailConfigure) throw new InvalidOperationException("configure quarantine");
        }
        public void CloseVmmWeights(ulong token)
        {
            Calls.Add("close"); if (FailClose) throw new InvalidOperationException("close quarantine");
            if (RetainOnClose) { Usage = Usage with { Closed = 1, QuarantinedBytes = 17 }; return; }
            Usage = Usage with { Closed = 1, HostTicketOwners = 0, LiveTables = 0,
                OwnedDeviceBytes = 0, PendingTicketBytes = 0, QuarantinedBytes = 0 };
        }
        public GgmlHostMoeVmmWeightsUsage ReadVmmUsage() { Calls.Add("stats"); return Usage; }
    }
    private static GgmlHostMoeExpertCacheScope Scope(Native native) => new(1000, 512, 3, native);

    [Theory]
    [InlineData(null, false)] [InlineData("0", false)] [InlineData("1", true)]
    public void ModeIsExactAndDefaultOff(string? value, bool expected)
        => Assert.Equal(expected, GgmlHostMoeExpertCacheScope.ResolveVmmWeightsMode(value));
    [Theory]
    [InlineData("")] [InlineData("true")] [InlineData("01")] [InlineData(" 1")] [InlineData("2")]
    public void InvalidModeCannotSilentlyChooseLegacy(string value)
        => Assert.Throws<ArgumentException>(() => GgmlHostMoeExpertCacheScope.ResolveVmmWeightsMode(value));

    [Fact]
    public void OffScopeNeverTouchesTheAdditiveAbi()
    {
        var native = new Native(); var scope = Scope(native);
        scope.Trim(0); scope.PrepareForModelDispose(); scope.Dispose();
        Assert.Equal(new[] { "begin", "trim", "end" }, native.Calls);
    }
    [Fact]
    public void VersionAndHostPlanAreRequiredBeforeNativeMutation()
    {
        var native = new Native { Version = 2 };
        Assert.Throws<NotSupportedException>(() => GgmlHostMoeExpertCacheScope.PlanVmmWeightsHostBytes(native));
        Assert.Equal(new[] { "version" }, native.Calls);
        native.Version = 1; native.HostPlan = 0;
        Assert.Throws<InvalidOperationException>(() => GgmlHostMoeExpertCacheScope.PlanVmmWeightsHostBytes(native));
        Assert.DoesNotContain("configure", native.Calls);
    }
    [Fact]
    public void FinalCloseIsCheckedBeforeEndAndExecutionDoesNotSampleStats()
    {
        var native = new Native(); var scope = Scope(native);
        scope.ConfigureVmmWeights(100); native.Calls.Clear();
        scope.ValidateVmmWeightsForExecution(); Assert.Empty(native.Calls);
        scope.PrepareForModelDispose(); scope.PrepareForModelDispose();
        Assert.Throws<InvalidOperationException>(scope.ValidateVmmWeightsForExecution);
        Assert.Throws<InvalidOperationException>(() => scope.UpdateBudget(1));
        scope.Dispose();
        Assert.Equal(new[] { "trim", "close", "stats", "end" }, native.Calls);
    }
    [Fact]
    public void ConfigureQuarantineRetainsTheScopeForFinalCleanupRetry()
    {
        var native = new Native { FailConfigure = true, FailClose = true }; var scope = Scope(native);
        Assert.Throws<InvalidOperationException>(() => scope.ConfigureVmmWeights(100));
        Assert.Throws<InvalidOperationException>(scope.ValidateVmmWeightsForExecution);
        Assert.Throws<InvalidOperationException>(scope.Dispose);
        Assert.DoesNotContain("end", native.Calls);
        native.FailClose = false; scope.Dispose();
        Assert.Equal(2, native.Calls.Count(c => c == "close")); Assert.Equal("end", native.Calls[^1]);
    }
    [Fact]
    public void TrimFailureStopsBeforeArenaAndEndAndKeepsRetryableMode()
    {
        var native = new Native(); var scope = Scope(native); scope.ConfigureVmmWeights(100);
        native.Calls.Clear(); native.FailTrim = true;
        Assert.Throws<InvalidOperationException>(scope.PrepareForModelDispose);
        Assert.Equal(new[] { "trim" }, native.Calls); scope.ValidateVmmWeightsForExecution();
        native.FailTrim = false; scope.Dispose();
        Assert.Equal(new[] { "trim", "trim", "close", "stats", "end" }, native.Calls);
    }
    [Fact]
    public void FailedEndAfterCertifiedFinalCloseRetriesOnlyEnd()
    {
        var native = new Native(); var scope = Scope(native); scope.ConfigureVmmWeights(100);
        native.FailEnd = true;
        Assert.Throws<InvalidOperationException>(scope.Dispose);
        native.FailEnd = false; native.Calls.Clear(); scope.Dispose();
        Assert.Equal(new[] { "end" }, native.Calls); // The certified final close is not repeated.
    }
    [Fact]
    public void ClaimedCloseWithOutstandingNativeTicketsDoesNotLendOwnership()
    {
        var native = new Native(); var scope = Scope(native); scope.ConfigureVmmWeights(100);
        native.RetainOnClose = true;
        Assert.Throws<InvalidOperationException>(scope.PrepareForModelDispose);
        Assert.DoesNotContain("end", native.Calls);
        native.RetainOnClose = false; scope.Dispose();
        Assert.Equal(2, native.Calls.Count(c => c == "close")); Assert.Equal("end", native.Calls[^1]);
    }
    [Theory]
    [InlineData(0, 0, 0)] [InlineData(1, -1, 0)] [InlineData(1, 0, 3)]
    public void MalformedNativeOwnershipSnapshotFailsClosed(long version, long bytes, long hostOwners)
    {
        var native = new Native(); var scope = Scope(native);
        native.Usage = native.Usage with { Version = version, OwnedDeviceBytes = bytes, HostTicketOwners = hostOwners };
        Assert.Throws<InvalidOperationException>(() => scope.ReadVmmUsage());
        scope.Dispose();
    }

    private static UnifiedMemoryModelOptions Options => new() { EnableGlmSelectedExperts = true,
        EnableAdaptiveGlmPrefill = true, MaximumRunningRequests = 1, RetainedConversationCount = 0,
        PrefixCheckpointCount = 0, PrefillChunkTokens = 512, AdaptiveGlmPrefillChunkTokens = 512 };
    private static string? Environment(string name) => name == "TS_HOST_MOE_DECODE_REBALANCE" ? "1" : null;
    [Fact]
    public void ProfileRequiresActualSinglePrimaryOptionsAndConflictsAreExplicit()
    {
        GlmUnifiedMemoryModelSession.ValidateVmmWeightsOptions(Options, BackendType.GgmlCuda, true, 0, false, Environment);
        foreach (var invalid in new[] { Options with { MaximumRunningRequests = 2 }, Options with { RetainedConversationCount = 1 },
            Options with { PrefixCheckpointCount = 1 }, Options with { AdaptiveGlmPrefillChunkTokens = 256 },
            Options with { MaximumHostExpertCacheBytes = 1 }, Options with { EnableGlmFlashMtp = true } })
            Assert.Throws<NotSupportedException>(() => GlmUnifiedMemoryModelSession.ValidateVmmWeightsOptions(invalid,
                BackendType.GgmlCuda, true, 0, false, Environment));
        foreach (string conflict in new[] { "TS_HOST_MOE_FILE_READ_WORKERS", "TS_HOST_MOE_RAM_CACHE_MB",
            "TS_HOST_MOE_DECODE_REBALANCE_RETAIN_PEER", "TS_HOST_MOE_PREFILL_DOWN_OVERLAP" })
            Assert.Throws<NotSupportedException>(() => GlmUnifiedMemoryModelSession.ValidateVmmWeightsOptions(Options,
                BackendType.GgmlCuda, true, 0, false, name => name == conflict ? "1" : Environment(name)));
    }
    [Fact]
    public void ActualModelDisposalGuardThrowsBeforeTheExecutorCanBeFreed()
    {
        var model = (GlmDsaModel)RuntimeHelpers.GetUninitializedObject(typeof(GlmDsaModel));
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
        typeof(GlmDsaModel).GetField("_nativeSync", flags)!.SetValue(model, new object());
        var executor = new NativeGlmExecutor((IntPtr)123); // Never used/called; guard must run first.
        typeof(GlmDsaModel).GetField("_exec", flags)!.SetValue(model, executor);
        int calls = 0; model.ConfigureUnifiedBeforeDispose(() => { calls++; throw new InvalidOperationException("quarantine"); });
        Assert.Throws<InvalidOperationException>(model.Dispose);
        Assert.Throws<InvalidOperationException>(model.PrepareUnifiedNativeDispose);
        Assert.Equal(2, calls); Assert.Same(executor, typeof(GlmDsaModel).GetField("_exec", flags)!.GetValue(model));
        Assert.Equal((IntPtr)123, typeof(NativeGlmExecutor).GetField("_handle", flags)!.GetValue(executor));
    }
    [Fact]
    public void CheckedExecutorRefusalKeepsActualHandleAndDoesNotRetryAfterSuccess()
    {
        bool allow = false; int calls = 0;
        var executor = new NativeGlmExecutor((IntPtr)123, true, (handle, required) =>
        { Assert.Equal((IntPtr)123, handle); Assert.True(required); calls++; return allow; });
        Assert.Throws<InvalidOperationException>(executor.Dispose);
        allow = true; executor.Dispose(); executor.Dispose(); Assert.Equal(2, calls);
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
        Assert.Equal(IntPtr.Zero, typeof(NativeGlmExecutor).GetField("_handle", flags)!.GetValue(executor));
    }
    [Fact]
    public void OldLibraryFallbackIsOnlyForOffAndNeverAfterARealRefusal()
    {
        int legacy = 0;
        Assert.True(GgmlGlmNative.CompleteCheckedFree(false, () => throw new EntryPointNotFoundException(), () => legacy++));
        Assert.Equal(1, legacy);
        Assert.Throws<NotSupportedException>(() => GgmlGlmNative.CompleteCheckedFree(true,
            () => throw new EntryPointNotFoundException(), () => legacy++));
        Assert.False(GgmlGlmNative.CompleteCheckedFree(false, () => 0, () => legacy++));
        Assert.False(GgmlGlmNative.CompleteCheckedFree(true, () => 0, () => legacy++));
        Assert.Equal(1, legacy);
    }
}
