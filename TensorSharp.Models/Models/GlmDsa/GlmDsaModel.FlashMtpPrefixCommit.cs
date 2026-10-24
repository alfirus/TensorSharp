// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;

namespace TensorSharp.Models;

public partial class GlmDsaModel
{
    private bool _unifiedMtpPrefixCommitEnabled;
    private int _g5nPrefixCommittedPosition = -1;

    internal void ConfigureUnifiedMtpPrefixCommit(bool enabled)
    {
        lock (_nativeSync)
        {
            if (enabled && (MemoryPolicy?.EnableGlmFlashMtp != true || !IsGlm5NextArch
                || _exec is not NativeGlmExecutor native || !native.FlashMtpPrefixCommitEnabled))
                throw new NotSupportedException("The native Flash MTP model did not adopt prefix-state commit.");
            _unifiedMtpPrefixCommitEnabled = enabled;
            _g5nPrefixCommittedPosition = -1;
        }
    }

    // Runtime passes the number of accepted DRAFTS and the draft window size.
    // The initial supported case rejects its sole draft but keeps batch row0.
    // Native commits that row's exact recurrent state and position atomically
    // with respect to success; a partial-copy failure poisons the native slot.
    public void SpecOnVerifyAccepted(int acceptedRows, int verifyRows)
    {
        _g5nPrefixCommittedPosition = -1;
        if (!_unifiedMtpPrefixCommitEnabled || acceptedRows != 0 || verifyRows != 1) return;
        lock (_nativeSync)
        {
            if (!_g5nSnapValid || _g5nSnapPos < 0 || _exec is not NativeGlmExecutor native)
                throw new InvalidOperationException("Flash prefix commit requires this verification window's recurrent snapshot.");
            int basePosition = _g5nSnapPos;
            try
            {
                int result = native.FlashMtpCommitVerifiedPrefix(basePosition, acceptedRows, verifyRows);
                int expected = checked(basePosition + (result == 1 ? 1 : 2));
                if (result is < 0 or > 1 || native.NPast != expected)
                    throw new InvalidOperationException("Flash prefix-state commit failed; reset the conversation before reuse.");
                if (result == 0) return; // No state copy occurred: original rollback remains authoritative.
                _cacheSeqLen = expected;
                _g5nPrefixCommittedPosition = expected;
                // TryCommitVerifiedPrefix now invokes the existing exact-position
                // rewind no-op; no recurrent rollback or accepted-prefix forward.
            }
            catch { Glm5NextInvalidateSnapshot(); throw; }
        }
    }
}
