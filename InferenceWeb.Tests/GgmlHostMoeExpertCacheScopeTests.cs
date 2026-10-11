// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

public sealed class GgmlHostMoeExpertCacheScopeTests
{
    private sealed class Native : IHostMoeExpertCacheScopeNative
    {
        public int Begins, Updates, Trims, Ends;
        public bool FailBegin, FailUpdate, FailTrim, FailEnd, Active;
        public long Budget;
        public ulong Begin(long bytes, int maxRows, int layers)
        {
            Begins++;
            if (FailBegin) throw new InvalidOperationException("begin");
            Assert.False(Active);
            Active = true; Budget = bytes;
            return 17;
        }
        public long Update(ulong token, long bytes)
        {
            Assert.Equal(17UL, token); Assert.True(Active); Updates++;
            if (FailUpdate) throw new InvalidOperationException("update");
            Budget = bytes; return 11;
        }
        public long Trim(ulong token, long bytes)
        {
            Assert.Equal(17UL, token); Assert.True(Active); Trims++;
            if (FailTrim) throw new InvalidOperationException("trim");
            return 13;
        }
        public long End(ulong token)
        {
            Assert.Equal(17UL, token); Assert.True(Active); Ends++;
            if (FailEnd) throw new InvalidOperationException("end");
            Active = false; return 19;
        }
        public GgmlHostMoeExpertCacheUsage ReadUsage()
        { Assert.True(Active); return new(14, Budget, 20, 30, 40); }
    }

    [Theory]
    [InlineData(-1, 64, 48)]
    [InlineData(100, 0, 48)]
    [InlineData(100, 129, 48)]
    [InlineData(100, 64, 0)]
    public void InvalidSettingsDoNotEnterNative(long bytes, int rows, int layers)
    {
        var native = new Native();
        Assert.Throws<ArgumentOutOfRangeException>(() => new GgmlHostMoeExpertCacheScope(bytes, rows, layers, native));
        Assert.Equal(0, native.Begins);
    }

    [Fact]
    public void FailedBeginHasNoOwnedCleanup()
    {
        var native = new Native { FailBegin = true };
        Assert.Throws<InvalidOperationException>(() => new GgmlHostMoeExpertCacheScope(100, 64, 48, native));
        Assert.False(native.Active); Assert.Equal(0, native.Ends);
    }

    [Fact]
    public void UpdateAndTrimHaveSeparateCeilingSemantics()
    {
        var native = new Native();
        using var scope = new GgmlHostMoeExpertCacheScope(100, 128, 48, native);
        Assert.Equal(100, scope.MaximumBudgetBytes); Assert.Equal(128, scope.MaxPrefillRows); Assert.Equal(48, scope.LayerCount);
        Assert.Equal(11, scope.UpdateBudget(0)); Assert.Equal(0, scope.BudgetBytes);
        Assert.Equal(11, scope.UpdateBudget(100)); Assert.Equal(100, scope.BudgetBytes);
        Assert.Equal(13, scope.Trim(0)); Assert.Equal(100, scope.BudgetBytes);
        Assert.Equal(14, scope.CachedBytes);
        Assert.Equal(new GgmlHostMoeExpertCacheUsage(14, 100, 20, 30, 40), scope.ReadUsage());
        Assert.Throws<ArgumentOutOfRangeException>(() => scope.UpdateBudget(101));
        Assert.Throws<ArgumentOutOfRangeException>(() => scope.UpdateBudget(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => scope.Trim(-1));
        Assert.Equal(2, native.Updates); Assert.Equal(1, native.Trims);
    }

    [Fact]
    public void MaintenanceFailuresKeepOwnerAndLastSuccessfulBudgetForRetry()
    {
        var native = new Native { FailUpdate = true, FailTrim = true, FailEnd = true };
        var scope = new GgmlHostMoeExpertCacheScope(100, 64, 12, native);
        Assert.Throws<InvalidOperationException>(() => scope.UpdateBudget(20));
        Assert.Throws<InvalidOperationException>(() => scope.Trim(0));
        Assert.Throws<InvalidOperationException>(scope.Dispose);
        Assert.True(native.Active); Assert.Equal(100, scope.BudgetBytes);
        native.FailUpdate = native.FailTrim = native.FailEnd = false;
        Assert.Equal(11, scope.UpdateBudget(20)); Assert.Equal(20, scope.BudgetBytes);
        scope.Dispose(); scope.Dispose();
        Assert.False(native.Active); Assert.Equal(2, native.Ends);
        Assert.Throws<ObjectDisposedException>(() => scope.Trim(0));
        Assert.Throws<ObjectDisposedException>(() => scope.UpdateBudget(10));
        Assert.Throws<ObjectDisposedException>(() => scope.ReadUsage());
    }

    [Fact]
    public void CancellationUnwindDisposesConfiguredOwnerOnce()
    {
        var native = new Native();
        Assert.Throws<OperationCanceledException>((Action)(() =>
        {
            using var scope = new GgmlHostMoeExpertCacheScope(0, 1, 1, native);
            throw new OperationCanceledException();
        }));
        Assert.False(native.Active); Assert.Equal(1, native.Ends);
    }
}
