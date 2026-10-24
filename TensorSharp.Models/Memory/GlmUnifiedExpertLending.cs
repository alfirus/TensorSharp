// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Memory;

namespace TensorSharp.Models;

/// <summary>Optional expert capacity borrows only observed unused ledger credit.
/// The maximum is a bound, never a reservation. Admission repays the loan before
/// spending that credit; native release and a fresh physical sample are required.</summary>
internal static class GlmUnifiedExpertLending
{
    private const string Host = UnifiedMemoryModelSession.HostPool;
    private const string Device = UnifiedMemoryModelSession.DevicePool;
    private const long Margin = 128L << 20;

    internal static long MaximumBudget(long startup, long deviceCeiling, long denseBytes,
        long operatorMaximum, bool enabled)
    {
        if (startup < 0 || deviceCeiling < 0 || denseBytes < 0 || operatorMaximum < startup)
            throw new ArgumentOutOfRangeException(nameof(startup));
        return enabled ? Math.Max(startup, GlmUnifiedMemoryGeometry.SelectExpertCacheCapacity(
            deviceCeiling, denseBytes, 0, 0, operatorMaximum)) : startup;
    }

    internal static long RecoveryTarget(GlmUnifiedPhysicalObservation physical, long current, long maximum,
        long minimumGrowthBytes = 512L << 20)
    {
        if (current < 0 || maximum < current || minimumGrowthBytes <= 0) throw new ArgumentOutOfRangeException(nameof(current));
        var device = physical.Ledger.Single(p => p.Pool == Device);
        long limit = Math.Min(device.Capacity, physical.Limits.Single(c => c.Pool == Device).Bytes);
        long increase = (long)Math.Max(0, (decimal)limit - device.Committed - device.Reserved - Margin);
        long target = (long)Math.Min(maximum, (decimal)current + increase);
        target = Math.Max(current, target / (1L << 20) * (1L << 20));
        return target - current >= minimumGrowthBytes ? target : current;
    }

    // A Device-only envelope still checks every supplied physical pool. After
    // its actual refusal, an unchanged SourceView Host promise may be the only
    // obstacle. A checked hint never refunds that owner: retry from new physics.
    internal static bool TryRefreshSourceViewRecovery(bool sourceView, bool recoverySucceeded,
        GlmUnifiedPhysicalObservation physical, long current, long refusedTarget, long maximum,
        long minimumGrowthBytes, Func<bool> trimHint, Func<GlmUnifiedPhysicalObservation> resample,
        out GlmUnifiedPhysicalObservation refreshed, out long target, out bool? hinted)
    {
        refreshed = physical; target = refusedTarget; hinted = null;
        if (!sourceView || recoverySucceeded || refusedTarget <= current) return false;
        MemoryCharge[] growth = [new(Host, 0), new(Device, checked(refusedTarget - current))];
        if (!physical.TryRefreshMappedHostPressure(growth, trimHint, resample, out refreshed, out hinted))
            return false;
        // Device availability can also change during maintenance. Keep the
        // original margin/threshold and do not reuse the earlier larger target.
        target = RecoveryTarget(refreshed, current, maximum, minimumGrowthBytes);
        return true;
    }

    internal static long RepaymentTarget(GlmUnifiedPhysicalObservation physical,
        IReadOnlyList<MemoryCharge> growth, long current, long minimum)
    {
        if (minimum < 0 || current < minimum) throw new ArgumentOutOfRangeException(nameof(minimum));
        // A device cache cannot repair a host shortfall. Do not evict useful
        // payload when this transaction cannot fit even after a complete loan
        // repayment. The caller may reclaim other idle owners and retry.
        if (physical.Shortfall(Host, growth) > 0) return current;
        decimal deficit = physical.Shortfall(Device, growth);
        if (deficit <= 0 || deficit > current - minimum) return current;
        return (long)Math.Max(minimum, current - deficit - Margin);
    }
}
