// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Memory;

namespace TensorSharp.Models;

/// <summary>Optional RAM payload spends only observed, admitted host credit.
/// Execution can reclaim it; native eviction precedes any ledger refund.</summary>
internal static class GlmUnifiedHostExpertCapacity
{
    private const string Host = UnifiedMemoryModelSession.HostPool;
    private const long Margin = 256L << 20;
    internal static long RecoveryTarget(GlmUnifiedPhysicalObservation physical, long current, long maximum)
    {
        if (current < 0 || maximum < current) throw new ArgumentOutOfRangeException(nameof(current));
        var host = physical.Ledger.Single(p => p.Pool == Host);
        long limit = Math.Min(host.Capacity, physical.Limits.Single(p => p.Pool == Host).Bytes);
        decimal slack = Math.Max(0, (decimal)limit - host.Committed - host.Reserved - Margin);
        long target = (long)Math.Min(maximum, current + slack);
        target = Math.Max(current, target / (1L << 20) * (1L << 20));
        return target - current >= (64L << 20) ? target : current;
    }
    internal static long RepaymentTarget(GlmUnifiedPhysicalObservation physical,
        IReadOnlyList<MemoryCharge> growth, long current)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(current);
        decimal deficit = physical.Shortfall(Host, growth);
        return deficit <= 0 ? current : (long)Math.Max(0, current - deficit - Margin);
    }
    internal static bool Resize(GgmlHostMoeRamCacheScope cache, BudgetReservation envelope,
        long target, IReadOnlyList<MemoryCharge> limits)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(target);
        var owned = envelope.OwnedCharges;
        if (owned.Count != 1 || owned[0].Pool != Host)
            throw new ArgumentException("RAM expert cache needs its single host-pool owner.", nameof(envelope));
        long previous = cache.CapacityBytes;
        if (target == previous && !cache.IsMapped) return true;
        MemoryCharge[] Charges(long bytes) => [new(Host, bytes)];
        if (target <= previous)
        {
            // A prior mapped cleanup may have retained its larger envelope.
            // Even an unchanged target must retry native retirement before any
            // released Host credit becomes available to another owner.
            cache.Configure(target);
            cache.EnsureCanReleaseHostCredit(target);
            if (target > owned[0].Bytes) return envelope.TryResizeTotal(Charges(target), limits);
            if (envelope.LiveChildCharges.Single().Bytes > target || !envelope.TryResizeTotal(Charges(target)))
                throw new InvalidOperationException("RAM expert eviction did not release its complete native ownership.");
            return true;
        }
        if (!envelope.TryResizeTotal(Charges(target), limits)) return false;
        try { cache.Configure(target); }
        catch (Exception error)
        {
            if (cache.IsMapped)
            {
                try
                {
                    // A mapped failure may retain a lock/quarantine or may
                    // have applied its lazy budget before a stats failure.
                    // Restore native policy and prove ownership first.
                    cache.Configure(previous);
                    cache.EnsureCanReleaseHostCredit(previous);
                }
                catch (Exception cleanup)
                {
                    throw new InvalidOperationException("Mapped RAM expert capacity cleanup failed; its Host reservation remains owned for retry.",
                        new AggregateException(error, cleanup));
                }
            }
            if (!envelope.TryResizeTotal(Charges(previous)))
                throw new InvalidOperationException("RAM expert capacity could not restore its previous reservation.", error);
            throw;
        }
        return true;
    }
}
