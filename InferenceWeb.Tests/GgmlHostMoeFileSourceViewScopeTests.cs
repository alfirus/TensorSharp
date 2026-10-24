// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Reflection;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class GgmlHostMoeFileSourceViewScopeTests
{
    private static readonly GgmlHostMoeFileSourceViewPlan Plan = new(1, 200, 64, 136, 126, 1000, 4096, 65536);
    private sealed class Native : IHostMoeExpertCacheScopeNative
    {
        internal readonly List<string> Calls = [];
        internal GgmlHostMoeFileSourceViewPlan HostPlan = Plan;
        internal GgmlHostMoeFileSourceViewUsage Usage = new(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 200);
        internal int Version = 1, Enabled = 1;
        internal bool FailConfigure, FailConfigureBeforeOwner, FailClose, FailEnd, FailAcknowledgment, BadCloseStats;
        public ulong Begin(long bytes, int rows, int layers, int decode) { Calls.Add("begin"); return 17; }
        public long Update(ulong token, long bytes) { Calls.Add("update"); return 0; }
        public long Trim(ulong token, long bytes) { Calls.Add("trim"); return 0; }
        public long End(ulong token) { Calls.Add("end"); if (FailEnd) throw new InvalidOperationException("end"); return 0; }
        public GgmlHostMoeExpertCacheUsage ReadUsage() => new(0, 1000, 0, 0, 0);
        public int FileSourceViewVersion() { Calls.Add("version"); return Version; }
        public int FileSourceViewEnabled() { Calls.Add("enabled"); return Enabled; }
        public GgmlHostMoeFileSourceViewPlan PlanFileSourceViews(int sources, long sourceBytes)
        { Calls.Add("plan"); Assert.Equal(126, sources); Assert.Equal(1000, sourceBytes); return HostPlan; }
        public void ConfigureFileSourceViews(ulong token, int sources, long sourceBytes, long promise)
        {
            Calls.Add("configure"); Assert.Equal(17UL, token); Assert.Equal(126, sources);
            Assert.Equal(1000, sourceBytes); Assert.Equal(200, promise);
            if (FailConfigureBeforeOwner) throw new InvalidOperationException("configure refused before any new owner");
            Usage = new(1, 1, 0, 0, 126, 126, 0, 1000, 64, 64, 0, 0, 0, 126, 0, 200);
            if (FailConfigure) throw new InvalidOperationException("partial configure owns quarantine");
        }
        public void CloseFileSourceViews(ulong token)
        {
            Calls.Add("close"); Assert.Equal(17UL, token);
            if (FailConfigureBeforeOwner) return; // Current wrapper token validated; old helper is already zero-owner/closed.
            if (FailClose) throw new InvalidOperationException("physical cleanup retained");
            Usage = new(1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 126, 126, 200);
            if (FailAcknowledgment)
            {
                Usage = Usage with { Closing = 1, Closed = 0, HeldControlTicketBytes = 64, QuarantinedRecords = 1 };
                throw new InvalidOperationException("physical cleanup complete; exact native ticket ACK retained");
            }
            if (BadCloseStats) Usage = Usage with { HeldControlTicketBytes = 64 };
        }
        public GgmlHostMoeFileSourceViewUsage ReadFileSourceViewUsage(ulong token)
        { Calls.Add("stats"); return Usage; }
    }
    private static GgmlHostMoeExpertCacheScope Scope(Native native) => new(1000, 512, 3, native);

    [Theory]
    [InlineData(null, false)] [InlineData("0", false)] [InlineData("1", true)]
    public void ModeIsExplicitAndOffByDefault(string? text, bool expected)
        => Assert.Equal(expected, GgmlHostMoeExpertCacheScope.ResolveFileSourceViewMode(text));
    [Theory]
    [InlineData("")] [InlineData("01")] [InlineData("true")] [InlineData(" 1")] [InlineData("2")]
    public void InvalidModeNeverSilentlyUsesBufferedService(string text)
        => Assert.Throws<ArgumentException>(() => GgmlHostMoeExpertCacheScope.ResolveFileSourceViewMode(text));

    [Fact]
    public void OffScopeMakesNoAdditiveAbiCalls()
    {
        var native = new Native(); var scope = Scope(native);
        scope.Trim(0); scope.PrepareForModelDispose(); scope.Dispose();
        Assert.Equal(new[] { "begin", "trim", "end" }, native.Calls);
    }
    [Fact]
    public void VersionEnabledAndExactHostPromiseAreRequiredBeforeMutation()
    {
        var native = new Native { Version = 0 };
        Assert.Throws<NotSupportedException>(() => GgmlHostMoeExpertCacheScope.PlanFileSourceViews(native, 126, 1000));
        Assert.Equal(new[] { "version" }, native.Calls);
        native.Version = 1; native.Enabled = 0;
        Assert.Throws<NotSupportedException>(() => GgmlHostMoeExpertCacheScope.PlanFileSourceViews(native, 126, 1000));
        Assert.DoesNotContain("plan", native.Calls);
        native.Enabled = 1;
        Assert.Equal(Plan, GgmlHostMoeExpertCacheScope.PlanFileSourceViews(native, 126, 1000));
        var scope = Scope(native);
        Assert.Throws<NotSupportedException>(() => scope.ConfigureFileSourceViews(Plan with { TotalHostPromise = 201 }));
        Assert.DoesNotContain("configure", native.Calls);
        scope.Dispose();
    }
    [Fact]
    public void DirectNativeModelSupportHandshakeRejectsOldOrInactiveServiceWithoutAllocating()
    {
        var old = new Native { Version = 0 };
        Assert.Throws<NotSupportedException>(() => GgmlHostMoeExpertCacheScope.RequireFileSourceViewSupport(old));
        Assert.Equal(new[] { "version" }, old.Calls);
        foreach (int enabled in new[] { -1, 0 })
        {
            var inactive = new Native { Enabled = enabled };
            Assert.Throws<NotSupportedException>(() => GgmlHostMoeExpertCacheScope.RequireFileSourceViewSupport(inactive));
            Assert.Equal(new[] { "version", "enabled" }, inactive.Calls);
        }
        var supported = new Native(); GgmlHostMoeExpertCacheScope.RequireFileSourceViewSupport(supported);
        Assert.Equal(new[] { "version", "enabled" }, supported.Calls);
    }
    [Fact]
    public void CheckedMappedTrimSupportRequiresBothVersionAndSideEffectFreeCompanion()
    {
        int probes = 0;
        Assert.Throws<NotSupportedException>(() => GgmlGlmNative.ValidateCheckedMappedResidencySupport(
            () => 0, () => { probes++; return 0; }));
        Assert.Equal(0, probes);
        Assert.Throws<NotSupportedException>(() => GgmlGlmNative.ValidateCheckedMappedResidencySupport(
            () => throw new EntryPointNotFoundException(), () => { probes++; return 0; }));
        Assert.Throws<NotSupportedException>(() => GgmlGlmNative.ValidateCheckedMappedResidencySupport(
            () => 1, () => throw new EntryPointNotFoundException()));
        foreach (int unsupported in new[] { -1, 1, 2 })
            Assert.Throws<NotSupportedException>(() => GgmlGlmNative.ValidateCheckedMappedResidencySupport(
                () => 1, () => unsupported));
        GgmlGlmNative.ValidateCheckedMappedResidencySupport(() => 1, () => { probes++; return 0; });
        Assert.Equal(1, probes);
    }

    [Theory]
    [InlineData(0, false)] [InlineData(1, true)]
    public void CheckedMappedTrimDistinguishesAdvisoryFromComplete(int status, bool expected)
        => Assert.Equal(expected, GgmlGlmNative.CompleteCheckedMappedResidencyTrim(() => status));

    [Fact]
    public void CheckedMappedTrimFatalOrMissingStatusNeverFallsBackToAnAdvisoryBoolean()
    {
        foreach (int status in new[] { -1, -2, 2, int.MinValue, int.MaxValue })
            Assert.Throws<InvalidOperationException>(() => GgmlGlmNative.CompleteCheckedMappedResidencyTrim(() => status));
        Assert.Throws<NotSupportedException>(() => GgmlGlmNative.CompleteCheckedMappedResidencyTrim(
            () => throw new EntryPointNotFoundException()));
    }

    [Fact]
    public void FatalCheckedMappedTrimPreventsResampleRetryAndExecutionButPreservesCheckedCleanup()
    {
        const string host = "unified/ram", device = "unified/cuda0";
        using var environment = new NativeEnvScope();
        environment.Set(GgmlHostMoeExpertCacheScope.FileSourceViewEnvironmentVariable, "0");
        var budget = new MemoryBudget([new(host, 1000), new(device, 1000)]);
        using var root = budget.Reserve([new(host, 200), new(device, 200)]);
        using var state = root.TryTake([new(host, 100), new(device, 100)])!;
        state.Commit();
        var before = budget.Snapshot().ToArray();
        var physical = GlmUnifiedPhysicalObservation.Create(new(2000, 50, 1600, 800), before, 1000, 1000, 0, 0);
        MemoryCharge[] growth = [new(host, 25), new(device, 25)];
        int trims = 0, samples = 0, releases = 0;
        bool cleanupReady = false;
        var executor = new NativeGlmExecutor((IntPtr)123, false, true, (handle, required) =>
        {
            Assert.Equal((IntPtr)123, handle); Assert.True(required); trims++;
            return GgmlGlmNative.CompleteCheckedMappedResidencyTrim(() => -1);
        }, (handle, required) =>
        { Assert.Equal((IntPtr)123, handle); Assert.True(required); releases++; return cleanupReady; });
        Assert.Throws<InvalidOperationException>(() => physical.TryRefreshMappedHostPressure(growth,
            executor.TrimMappedResidency, () => { samples++; return physical; }, out _, out _));
        Assert.Equal(0, samples);
        Assert.Equal(before, budget.Snapshot().ToArray());
        Assert.Throws<InvalidOperationException>(() => executor.TrimMappedResidency());
        Assert.Throws<InvalidOperationException>(() => executor.Forward([1], new float[1]));
        Assert.Throws<InvalidOperationException>(() => executor.ForwardLayerMajor([1], new float[1], 512));
        Assert.Throws<InvalidOperationException>(() => executor.ResetChecked());
        Assert.Equal(1, trims);
        Assert.Throws<InvalidOperationException>(executor.Dispose);
        Assert.Equal((IntPtr)123, typeof(NativeGlmExecutor).GetField("_handle", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(executor));
        cleanupReady = true; executor.Dispose(); executor.Dispose();
        Assert.Equal(2, releases);
        Assert.Equal(IntPtr.Zero, typeof(NativeGlmExecutor).GetField("_handle", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(executor));
    }

    [Fact]
    public void LegacyMappedTrimForOffOrVmmDoesNotFollowLaterSourceViewEnvironmentChanges()
    {
        using var environment = new NativeEnvScope();
        environment.Set(GgmlHostMoeExpertCacheScope.FileSourceViewEnvironmentVariable, "1");
        int legacyCalls = 0, checkedCalls = 0;
        var executor = new NativeGlmExecutor((IntPtr)123, true, false, (_, required) =>
        {
            if (required) { checkedCalls++; throw new InvalidOperationException("Unexpected checked trim"); }
            legacyCalls++; return false;
        }, (_, required) => { Assert.True(required); return true; });
        Assert.False(executor.TrimMappedResidency());
        Assert.Equal(1, legacyCalls); Assert.Equal(0, checkedCalls);
        executor.Dispose();
    }
    [Fact]
    public void MalformedGeometryAndOverflowForecastFailBeforeConfigure()
    {
        foreach (var bad in new[] { Plan with { Sources = 125 }, Plan with { SourceBytes = 0 },
            Plan with { OwnedControlBytes = 0 }, Plan with { PteForecastBytes = -1 },
            Plan with { PageBytes = 0 }, Plan with { MapGranularity = 5000 },
            Plan with { OwnedControlBytes = long.MaxValue, PteForecastBytes = 1 } })
        {
            var native = new Native { HostPlan = bad };
            Assert.Throws<InvalidOperationException>(() => GgmlHostMoeExpertCacheScope.PlanFileSourceViews(native, 126, 1000));
            Assert.DoesNotContain("configure", native.Calls);
        }
    }
    [Fact]
    public void ConfiguredExecutionHasNoStatsSamplingAndClosePrecedesEnd()
    {
        var native = new Native(); var scope = Scope(native); scope.ConfigureFileSourceViews(Plan);
        native.Calls.Clear(); scope.ValidateFileSourceViewsForExecution(); Assert.Empty(native.Calls);
        scope.PrepareForModelDispose(); scope.PrepareForModelDispose();
        Assert.Throws<InvalidOperationException>(scope.ValidateFileSourceViewsForExecution);
        Assert.Throws<InvalidOperationException>(() => scope.UpdateBudget(1));
        scope.Dispose(); Assert.Equal(new[] { "close", "stats", "end" }, native.Calls);
    }
    [Fact]
    public void PartialConfigureCannotLoseTheOwnerAndCloseFailureStopsEnd()
    {
        var native = new Native { FailConfigure = true, FailClose = true }; var scope = Scope(native);
        Assert.Throws<InvalidOperationException>(() => scope.ConfigureFileSourceViews(Plan));
        Assert.Throws<InvalidOperationException>(scope.ValidateFileSourceViewsForExecution);
        Assert.Throws<InvalidOperationException>(scope.Dispose);
        Assert.DoesNotContain("end", native.Calls);
        native.FailClose = false; scope.Dispose();
        Assert.Equal(2, native.Calls.Count(c => c == "close")); Assert.Equal("end", native.Calls[^1]);
    }
    [Fact]
    public void PreownerConfigureFailureAcceptsAlreadyClosedZeroOwnerCleanupAtCurrentScopeToken()
    {
        var native = new Native { FailConfigureBeforeOwner = true,
            Usage = new(1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 126, 126, 300) };
        var scope = Scope(native);
        Assert.Throws<InvalidOperationException>(() => scope.ConfigureFileSourceViews(Plan));
        Assert.Throws<InvalidOperationException>(scope.ValidateFileSourceViewsForExecution);
        native.Calls.Clear(); scope.Dispose(); scope.Dispose();
        Assert.Equal(new[] { "close", "stats", "end" }, native.Calls);
        // This proves managed retry accepts the validated wrapper's zero-owner
        // receipt. Real native cross-token dispatch is a separate ABI fixture.
    }
    [Fact]
    public void PhysicalCleanupWithoutTicketAcknowledgmentRetainsTheScopeForRetry()
    {
        var native = new Native(); var scope = Scope(native); scope.ConfigureFileSourceViews(Plan);
        native.FailAcknowledgment = true;
        Assert.Throws<InvalidOperationException>(scope.PrepareForModelDispose);
        var retained = scope.ReadFileSourceViewUsage();
        Assert.Equal(0, retained.LiveViews); Assert.Equal(0, retained.OwnedControlBytes);
        Assert.Equal(0, retained.OwnedRecords); Assert.Equal(1, retained.QuarantinedRecords);
        Assert.Equal(64, retained.HeldControlTicketBytes); Assert.DoesNotContain("end", native.Calls);
        Assert.Throws<InvalidOperationException>(scope.ValidateFileSourceViewsForExecution);
        Assert.Throws<InvalidOperationException>(() => scope.UpdateBudget(1));
        native.FailAcknowledgment = false; scope.Dispose();
        Assert.Equal(2, native.Calls.Count(c => c == "close")); Assert.Equal("end", native.Calls[^1]);
        Assert.Equal(1, native.Calls.Count(c => c == "begin"));
        Assert.Equal(1, native.Calls.Count(c => c == "configure")); // Both closes check exact token17.
    }
    [Fact]
    public void ShellQuarantineRequiresClosingAnActuallyRetainedOwner()
    {
        foreach (var invalid in new[] {
            new GgmlHostMoeFileSourceViewUsage(1, 1, 0, 0, 0, 0, 0, 0, 0, 64, 0, 1, 0, 126, 126, 200),
            new GgmlHostMoeFileSourceViewUsage(1, 1, 1, 1, 0, 0, 0, 0, 0, 64, 0, 1, 0, 126, 126, 200),
            new GgmlHostMoeFileSourceViewUsage(1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 126, 126, 200),
            new GgmlHostMoeFileSourceViewUsage(1, 1, 1, 0, 0, 0, 0, 0, 0, 64, 0, 2, 0, 126, 126, 200) })
        {
            var native = new Native { Usage = invalid }; var scope = Scope(native);
            Assert.Throws<InvalidOperationException>(() => scope.ReadFileSourceViewUsage());
            scope.Dispose();
        }
        var slab = new Native { Usage = new(1, 1, 1, 0, 0, 0, 0, 0, 64, 64, 0, 1, 0, 126, 126, 200) };
        var slabScope = Scope(slab); Assert.Equal(1, slabScope.ReadFileSourceViewUsage().QuarantinedRecords);
        slabScope.Dispose(); // Snapshot read only; no configured resources are fabricated by this fake.
    }
    [Fact]
    public void CloseReportedWithOutstandingOwnerNeverDetachesAndAlreadyClosedRetryIsAccepted()
    {
        var native = new Native(); var scope = Scope(native); scope.ConfigureFileSourceViews(Plan);
        native.BadCloseStats = true;
        Assert.Throws<InvalidOperationException>(scope.PrepareForModelDispose);
        Assert.DoesNotContain("end", native.Calls);
        native.BadCloseStats = false; scope.PrepareForModelDispose(); scope.Dispose();
        Assert.Equal(2, native.Calls.Count(c => c == "close")); Assert.Equal("end", native.Calls[^1]);
    }
    [Fact]
    public void EndFailureAfterCertifiedCloseRetriesOnlyEnd()
    {
        var native = new Native(); var scope = Scope(native); scope.ConfigureFileSourceViews(Plan);
        native.FailEnd = true; Assert.Throws<InvalidOperationException>(scope.Dispose);
        native.Calls.Clear(); native.FailEnd = false; scope.Dispose(); Assert.Equal(new[] { "end" }, native.Calls);
    }
    private static UnifiedMemoryModelOptions Options => new() { EnableGlmSelectedExperts = true,
        EnableAdaptiveGlmPrefill = true, MaximumRunningRequests = 1, RetainedConversationCount = 0,
        PrefixCheckpointCount = 0, PrefillChunkTokens = 512, AdaptiveGlmPrefillChunkTokens = 512 };
    private static string? Environment(string name) => name switch
    { "TS_HOST_MOE_DECODE_REBALANCE" => "1", "TS_HOST_MOE_EXPERT_CACHE_LFU" => "0", _ => null };
    [Fact]
    public void SourceViewOnlyProfileRejectsEveryConflictingOwnerBeforeStartup()
    {
        GlmUnifiedMemoryModelSession.ValidateFileSourceViewOptions(Options, BackendType.GgmlCuda, true, 0, false, Environment);
        Assert.Throws<NotSupportedException>(() => GlmUnifiedMemoryModelSession.ValidateFileSourceViewOptions(Options,
            BackendType.GgmlCuda, true, 0, false, name => name == "TS_HOST_MOE_EXPERT_CACHE_LFU" ? null : Environment(name)));
        foreach (var options in new[] { Options with { MaximumRunningRequests = 2 }, Options with { RetainedConversationCount = 1 },
            Options with { PrefixCheckpointCount = 1 }, Options with { AdaptiveGlmPrefillChunkTokens = 256 },
            Options with { MaximumHostExpertCacheBytes = 1 }, Options with { EnableGlmFlashMtp = true } })
            Assert.Throws<NotSupportedException>(() => GlmUnifiedMemoryModelSession.ValidateFileSourceViewOptions(options,
                BackendType.GgmlCuda, true, 0, false, Environment));
        foreach (string flag in new[] { "TS_HOST_MOE_VMM_WEIGHTS", "TS_HOST_MOE_FILE_READ_SORT", "TS_HOST_MOE_FILE_RANDOM_ACCESS",
            "TS_HOST_MOE_FILE_READ_WORKERS", "TS_HOST_MOE_RAM_CACHE_MB", "TS_HOST_MOE_PREFILL_DOWN_OVERLAP",
            "TS_HOST_MOE_EXPERT_CACHE_LFU" })
            Assert.Throws<NotSupportedException>(() => GlmUnifiedMemoryModelSession.ValidateFileSourceViewOptions(Options,
                BackendType.GgmlCuda, true, 0, false, name => name == flag ? "1" : Environment(name)));
    }
    [Fact]
    public void SourceViewOnlyRequiredFreeIsLatchedAndRefusalPreservesTheActualExecutorOwner()
    {
        using var env = new NativeEnvScope(); env.Set("TS_HOST_MOE_VMM_WEIGHTS", "0");
        env.Set(GgmlHostMoeExpertCacheScope.FileSourceViewEnvironmentVariable, "1");
        bool requiredAtLoad = GgmlHostMoeExpertCacheScope.ResolveFileSourceViewMode(
            System.Environment.GetEnvironmentVariable(GgmlHostMoeExpertCacheScope.FileSourceViewEnvironmentVariable));
        int calls = 0; bool allow = false;
        var executor = new NativeGlmExecutor((IntPtr)123, requiredAtLoad, (handle, required) =>
        { Assert.Equal((IntPtr)123, handle); Assert.True(required); calls++; return allow; });
        env.Set(GgmlHostMoeExpertCacheScope.FileSourceViewEnvironmentVariable, "0");
        Assert.Throws<InvalidOperationException>(executor.Dispose);
        Assert.Equal((IntPtr)123, typeof(NativeGlmExecutor).GetField("_handle", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(executor));
        allow = true; executor.Dispose(); executor.Dispose(); Assert.Equal(2, calls);
    }
}
