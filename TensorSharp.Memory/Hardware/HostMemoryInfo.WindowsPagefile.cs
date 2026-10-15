// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.Versioning;
using Microsoft.Win32;

namespace TensorSharp.Memory;

public static partial class HostMemoryInfo
{
    /// <summary>Free space a pagefile volume keeps when estimating automatic
    /// pagefile growth: the larger of this floor and 5% of the volume.</summary>
    internal const long PagefileVolumeReserveFloorBytes = 16L << 30;

    /// <summary>Commit the Windows memory manager can still add by growing a
    /// system-managed pagefile. GlobalMemoryStatusEx reports the CURRENT commit
    /// limit; a system-managed pagefile expands on demand toward
    /// max(3 x RAM, 4 GiB), capped at 1/8 of its volume, so on such hosts that
    /// limit is not the allowance a new commitment can obtain. On WDDM this
    /// matters: device allocations are charged to the process commit, and with
    /// ordinary desktop applications open the current limit leaves almost
    /// nothing for host allocations although RAM is plentiful.
    ///
    /// Only the common layout is credited: exactly one existing pagefile whose
    /// configuration is system-managed ("?:\pagefile.sys" or "X:\pagefile.sys"
    /// with zero sizes), or an explicit maximum above its current size. Growth
    /// keeps a free-space reserve on the volume. Anything else returns 0 and
    /// keeps the current-limit behavior.</summary>
    [SupportedOSPlatform("windows")]
    private static long WindowsPagefileGrowthBytes(long physicalTotal, long commitLimit)
    {
        try
        {
            using var key = Registry.LocalMachine.OpenSubKey(@"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management");
            if (key?.GetValue("PagingFiles") is not string[] configured
                || key.GetValue("ExistingPageFiles") is not string[] existing || existing.Length != 1)
                return 0;
            // "\??\C:\pagefile.sys" -> "C:\"
            string path = existing[0].StartsWith(@"\??\", StringComparison.Ordinal) ? existing[0][4..] : existing[0];
            string? root = System.IO.Path.GetPathRoot(path);
            if (string.IsNullOrEmpty(root)) return 0;
            var drive = new DriveInfo(root);
            if (!drive.IsReady) return 0;
            long? maximum = ConfiguredPagefileMaximum(configured, root, physicalTotal, drive.TotalSize);
            if (maximum is not > 0) return 0;
            return PagefileGrowth(maximum.Value, Math.Max(0, commitLimit - physicalTotal),
                drive.AvailableFreeSpace, drive.TotalSize);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or System.Security.SecurityException or ArgumentException)
        {
            return 0;
        }
    }

    /// <summary>The largest size the configured pagefile on <paramref name="root"/>
    /// may reach, or null when the configuration is not recognized.</summary>
    internal static long? ConfiguredPagefileMaximum(IReadOnlyList<string> configured, string root,
        long physicalTotal, long volumeTotal)
    {
        long systemManaged = Math.Min(Math.Max(3 * physicalTotal, 4L << 30), volumeTotal / 8);
        foreach (string entry in configured)
        {
            string[] parts = entry.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 0) continue;
            // "?:\pagefile.sys": automatic management on every volume.
            if (parts[0].StartsWith("?:", StringComparison.Ordinal)) return systemManaged;
            if (!parts[0].StartsWith(root, StringComparison.OrdinalIgnoreCase)) continue;
            if (parts.Length == 1) return systemManaged;
            if (parts.Length == 3 && long.TryParse(parts[1], out long initialMb) && long.TryParse(parts[2], out long maximumMb))
                return initialMb == 0 && maximumMb == 0 ? systemManaged
                    : maximumMb > 0 ? checked(maximumMb << 20) : null;
            return null;
        }
        return null;
    }

    internal static long PagefileGrowth(long maximumBytes, long currentBytes, long volumeFree, long volumeTotal)
    {
        if (maximumBytes <= 0 || currentBytes < 0 || volumeFree < 0 || volumeTotal <= 0) return 0;
        long reserve = Math.Max(PagefileVolumeReserveFloorBytes, volumeTotal / 20);
        return Math.Max(0, Math.Min(maximumBytes - currentBytes, volumeFree - reserve));
    }
}
