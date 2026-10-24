// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML;

internal static partial class GgmlNative
{
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeDecodeStagingVersion")]
    internal static partial int HostMoeDecodeStagingVersion();
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeDecodeStagingEnabled")]
    internal static partial int HostMoeDecodeStagingEnabled();
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoePrefillRingStagingVersion")]
    internal static partial int HostMoePrefillRingStagingVersion();
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoePrefillRingStagingEnabled")]
    internal static partial int HostMoePrefillRingStagingEnabled();
}

public partial class GgmlBasicOps
{
    /// <summary>Whether this native library opted into the bounded 72 MiB
    /// scalar GLM file arena. The capability preserves the 32 MiB path when
    /// the additional host reservation is refused before allocation.</summary>
    public static bool SupportsEnabledHostMoeDecodeStaging()
    {
        try { return GgmlNative.HostMoeDecodeStagingVersion() == 1 && GgmlNative.HostMoeDecodeStagingEnabled() == 1; }
        catch (Exception error) when (error is EntryPointNotFoundException or DllNotFoundException) { return false; }
    }

    /// <summary>Whether the native library independently enabled the bounded
    /// two36MiB wide-prefill ring. This is not evidence that a call used it.</summary>
    public static bool SupportsEnabledHostMoePrefillRingStaging()
    {
        try { return GgmlNative.HostMoePrefillRingStagingVersion() == 1 && GgmlNative.HostMoePrefillRingStagingEnabled() == 1; }
        catch (Exception error) when (error is EntryPointNotFoundException or DllNotFoundException) { return false; }
    }
}
