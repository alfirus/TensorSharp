// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models;

// Spark retains every linear KV row even on sliding-attention layers, so a
// prefix can rewind to any token. F16 uses host slabs; F32 also exposes the
// model-paged rows produced by its packed forward path.
public partial class MuseSparkModel : IPageOnlyPrefixCacheModel, IPrefixCacheModelDiagnostics
{
    public PrefixCacheCapabilities GetPrefixCacheCapabilities()
        => PageFamilyCapabilities(FamilyClass.S, TruncationKind.Any);

    public long QuerySpareBytes(ResourceClass cls) => QueryPrefixCacheSpareBytes(cls);
    public IReadOnlyCollection<string> RetainedPayloadKeys => Array.Empty<string>();
    public int PrivateHolderCount => 0;
    public int PrimaryCacheLength => _cacheSeqLen;
}
