// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

internal sealed class MappedRamCacheNative : IHostMoeRamCacheNative
{
    internal int Version = 1, Enabled = 1, VersionCalls, EnabledCalls, MappedStatsCalls;
    internal long Capacity;
    internal readonly List<long> Configurations = [];
    internal Action<long>? OnConfigure;
    internal GgmlHostMoeMappedRamCacheUsage Usage;
    public void Configure(long bytes) { Configurations.Add(bytes); OnConfigure?.Invoke(bytes); Capacity = bytes; }
    public GgmlHostMoeRamCacheUsage ReadUsage() => new(Capacity, 11, 12, 13, 14, 15, 16, 17, 18);
    public int MappedVersion() { VersionCalls++; return Version; }
    public int MappedEnabled() { EnabledCalls++; return Enabled; }
    public GgmlHostMoeMappedRamCacheUsage ReadMappedUsage()
    { MappedStatsCalls++; return Usage with { CapacityChargeBytes = Capacity }; }
    internal static Dictionary<string, string?> Environment(bool exclusive = false)
        => exclusive ? new()
        {
            ["TS_HOST_MOE_RAM_CACHE_MAPPED"] = "1",
            ["TS_HOST_MOE_RAM_CACHE_MAPPED_EXCLUSIVE_QUOTA"] = "1",
            ["TS_HOST_MOE_RAM_CACHE_MAPPED_QUOTA_MB"] = "512"
        } : new() { ["TS_HOST_MOE_RAM_CACHE_MAPPED"] = "1" };
    internal static Func<string, string?> Reader(Dictionary<string, string?> values)
        => name => values.TryGetValue(name, out string? value) ? value : null;
}

public sealed class GgmlHostMoeMappedRamCacheScopeTests
{
    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("0")]
    [InlineData("01")]
    [InlineData("true")]
    public void OrdinaryModeDoesNotProbeNewNativeAbi(string? flag)
    {
        var environment = new Dictionary<string, string?> { ["TS_HOST_MOE_RAM_CACHE_MAPPED"] = flag };
        var native = new MappedRamCacheNative { Version = 0, Enabled = 0 };
        using var scope = new GgmlHostMoeRamCacheScope(native, MappedRamCacheNative.Reader(environment));
        Assert.False(scope.IsMapped);
        scope.Configure(64);
        Assert.Equal(11, scope.ReadUsage().ResidentBytes);
        Assert.Throws<InvalidOperationException>(() => scope.ReadMappedUsage());
        scope.Dispose();
        Assert.Equal(0, native.VersionCalls);
        Assert.Equal(0, native.EnabledCalls);
        Assert.Equal(0, native.MappedStatsCalls);
    }

    [Theory]
    [InlineData(0, 1, true)]
    [InlineData(2, 1, true)]
    [InlineData(1, 0, false)]
    public void MappedCapabilityAndLatchFailBeforeNativeConfiguration(int version, int enabled, bool unsupported)
    {
        var native = new MappedRamCacheNative { Version = version, Enabled = enabled };
        var environment = MappedRamCacheNative.Environment();
        var failure = Record.Exception(() => new GgmlHostMoeRamCacheScope(native, MappedRamCacheNative.Reader(environment)));
        if (unsupported) Assert.IsType<NotSupportedException>(failure);
        else Assert.IsType<InvalidOperationException>(failure);
        Assert.Empty(native.Configurations);
        using var retry = new GgmlHostMoeRamCacheScope(new MappedRamCacheNative(), MappedRamCacheNative.Reader(environment));
    }

    [Theory]
    [InlineData("TS_HOST_MOE_RAM_CACHE_PINNED")]
    [InlineData("TS_HOST_MOE_RAM_CACHE_DIRECT_UPLOAD")]
    [InlineData("TS_HOST_MOE_RAM_CACHE_VICTIM")]
    [InlineData("TS_HOST_MOE_RAM_CACHE_PREFETCH_ONLY")]
    [InlineData("TS_HOST_MOE_RAM_CACHE_FREQUENCY")]
    [InlineData("TS_HOST_MOE_LAYER_PREFETCH")]
    [InlineData("TS_HOST_MOE_EXPERT_CACHE_DIRECT_READS")]
    [InlineData("TS_HOST_MOE_EXPERT_CACHE_PREFETCH")]
    [InlineData("TS_HOST_MOE_DECODE_PREFETCH")]
    public void ConflictingModeRefusesBeforeAnyNativeCall(string conflict)
    {
        var environment = MappedRamCacheNative.Environment(); environment[conflict] = "1";
        var native = new MappedRamCacheNative();
        var failure = Assert.Throws<ArgumentException>(() => new GgmlHostMoeRamCacheScope(native,
            MappedRamCacheNative.Reader(environment)));
        Assert.Contains(conflict, failure.Message);
        Assert.Equal(0, native.VersionCalls);
        Assert.Empty(native.Configurations);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("0")]
    [InlineData("-1")]
    [InlineData(" 512")]
    [InlineData("1.5")]
    [InlineData("8796093022208")]
    public void QuotaExpansionRequiresPositiveRepresentableCeiling(string? ceiling)
    {
        var environment = MappedRamCacheNative.Environment(true);
        environment["TS_HOST_MOE_RAM_CACHE_MAPPED_QUOTA_MB"] = ceiling;
        var native = new MappedRamCacheNative();
        Assert.Throws<ArgumentException>(() => new GgmlHostMoeRamCacheScope(native, MappedRamCacheNative.Reader(environment)));
        Assert.Empty(native.Configurations);
    }

    [Fact]
    public void ExistingQuotaAndExplicitExclusiveQuotaHaveSeparateContracts()
    {
        var environment = MappedRamCacheNative.Environment();
        environment["TS_HOST_MOE_RAM_CACHE_MAPPED_QUOTA_MB"] = "512";
        Assert.Throws<ArgumentException>(() => new GgmlHostMoeRamCacheScope(new MappedRamCacheNative(),
            MappedRamCacheNative.Reader(environment)));
        environment.Remove("TS_HOST_MOE_RAM_CACHE_MAPPED_QUOTA_MB");
        using (var existing = new GgmlHostMoeRamCacheScope(new MappedRamCacheNative(), MappedRamCacheNative.Reader(environment)))
        {
            Assert.True(existing.IsMapped);
            Assert.False(existing.UsesExclusiveMappedQuota);
            Assert.Equal(0, existing.MappedQuotaCeilingBytes);
        }
        environment = MappedRamCacheNative.Environment(true);
        using var exclusive = new GgmlHostMoeRamCacheScope(new MappedRamCacheNative(), MappedRamCacheNative.Reader(environment));
        Assert.True(exclusive.UsesExclusiveMappedQuota);
        Assert.Equal(512L << 20, exclusive.MappedQuotaCeilingBytes);
    }

    [Theory]
    [InlineData("TS_HOST_MOE_RAM_CACHE_MAPPED", "0")]
    [InlineData("TS_HOST_MOE_RAM_CACHE_MAPPED_EXCLUSIVE_QUOTA", "0")]
    [InlineData("TS_HOST_MOE_RAM_CACHE_MAPPED_QUOTA_MB", "1024")]
    [InlineData("TS_HOST_MOE_RAM_CACHE_PINNED", "1")]
    public void PositiveResizeRejectsEnvironmentDriftWhileDisposalCanClean(string name, string value)
    {
        var environment = MappedRamCacheNative.Environment(true);
        var native = new MappedRamCacheNative();
        using var scope = new GgmlHostMoeRamCacheScope(native, MappedRamCacheNative.Reader(environment));
        scope.Configure(64);
        int calls = native.Configurations.Count;
        environment[name] = value;
        Assert.NotNull(Record.Exception(() => scope.Configure(128)));
        Assert.Equal(calls, native.Configurations.Count);
        Assert.Equal(64, scope.CapacityBytes);
        scope.Dispose();
        Assert.Equal(0, native.Capacity);
        Assert.Throws<ObjectDisposedException>(() => scope.ReadMappedUsage());
    }

    [Fact]
    public void PositiveResizeRechecksNativeLatchAndLeavesOrdinaryUsageAbiUnchanged()
    {
        var environment = MappedRamCacheNative.Environment();
        var native = new MappedRamCacheNative();
        using var scope = new GgmlHostMoeRamCacheScope(native, MappedRamCacheNative.Reader(environment));
        scope.Configure(128);
        var legacy = scope.ReadUsage();
        Assert.Equal(new GgmlHostMoeRamCacheUsage(128, 11, 12, 13, 14, 15, 16, 17, 18), legacy);
        native.Usage = new(0, 72, 96, 80, 16, 2, 1, 8, 3, 4, 5, 6, 7, 8, 0, false);
        var mapped = scope.ReadMappedUsage();
        Assert.Equal(96, mapped.OwnedChargeBytes);
        Assert.Equal(80, mapped.LockedPageBytes);
        Assert.Equal(16, mapped.MetadataChargeBytes);
        Assert.Equal(72, mapped.LogicalResidentBytes);
        native.Enabled = 0;
        int calls = native.Configurations.Count;
        Assert.Throws<InvalidOperationException>(() => scope.Configure(128));
        Assert.Equal(calls, native.Configurations.Count);
        native.Usage = default; // Cumulative counters do not prevent complete cleanup.
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void DisposalRetainsExclusiveClaimUntilEveryNativeOwnerIsClean(int retained)
    {
        var environment = MappedRamCacheNative.Environment();
        var native = new MappedRamCacheNative();
        using var scope = new GgmlHostMoeRamCacheScope(native, MappedRamCacheNative.Reader(environment));
        scope.Configure(128);
        native.Usage = retained switch
        {
            0 => default(GgmlHostMoeMappedRamCacheUsage) with { ActiveLeases = 1 },
            1 => default(GgmlHostMoeMappedRamCacheUsage) with { PendingAdmissions = 1 },
            2 => default(GgmlHostMoeMappedRamCacheUsage) with { OwnedChargeBytes = 64, LockedPageBytes = 64, QuarantineChargeBytes = 64 },
            _ => default(GgmlHostMoeMappedRamCacheUsage) with { QuotaRestorePending = true }
        };
        Assert.Throws<InvalidOperationException>(scope.Dispose);
        Assert.Equal(128, scope.CapacityBytes);
        var second = new MappedRamCacheNative();
        Assert.Throws<InvalidOperationException>(() => new GgmlHostMoeRamCacheScope(second, MappedRamCacheNative.Reader(environment)));
        Assert.Empty(second.Configurations);
        native.Usage = default(GgmlHostMoeMappedRamCacheUsage) with { Hits = 5, Misses = 6, HitCopyBytes = 7 };
        scope.Dispose();
        using var retry = new GgmlHostMoeRamCacheScope(second, MappedRamCacheNative.Reader(environment));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void InvalidMappedAccountingCannotAuthorizeHostRelease(bool logicalExceedsOwned)
    {
        var environment = MappedRamCacheNative.Environment();
        var native = new MappedRamCacheNative();
        using var scope = new GgmlHostMoeRamCacheScope(native, MappedRamCacheNative.Reader(environment));
        native.Usage = logicalExceedsOwned
            ? default(GgmlHostMoeMappedRamCacheUsage) with { LogicalResidentBytes = 65, OwnedChargeBytes = 64, LockedPageBytes = 60, MetadataChargeBytes = 4 }
            : default(GgmlHostMoeMappedRamCacheUsage) with { OwnedChargeBytes = 64, LockedPageBytes = 64, MetadataChargeBytes = 1 };
        Assert.Throws<InvalidOperationException>(() => scope.Configure(128));
        Assert.Throws<InvalidOperationException>(() => scope.EnsureCanReleaseHostCredit(128));
        native.Usage = default;
    }
}
