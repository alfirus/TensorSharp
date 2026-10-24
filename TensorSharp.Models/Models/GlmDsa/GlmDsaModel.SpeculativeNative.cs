// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// NextN/MTP on the native whole-model executor.
//
// GLM-5.2 is 226 GiB of weights, so on the GGML backends the trunk, the draft
// block and both KV caches live inside ggml_ops_glm_dsa.cpp and the managed
// side is a marshalling layer. The three entry points below mirror the per-op
// implementations in GlmDsaModel.Mtp.cs exactly; what differs is that the
// native side keeps everything device-resident, so a verify batch is ONE graph
// over the whole speculative window and drafting never round-trips the hidden
// state through host memory except at the interface boundary the shared
// draft/verify core requires.
using System;
using TensorSharp.GGML;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models
{
    public partial class GlmDsaModel
    {
        /// <summary>
        /// Whether the operator asked for speculation before this model loaded.
        /// The draft block is a whole extra 256-expert decoder layer (~3 GiB of
        /// GLM-5.2 at IQ2_XXS, and it competes with the KV cache for the same
        /// VRAM the loader sizes the context against), so it is only paged in
        /// when it is going to be used. <c>--spec</c> sets TS_SPEC before the
        /// startup model loads, and a deployment may export it directly; either
        /// way the loader pages the block in exactly when the scheduler will
        /// speculate.
        ///
        /// The enable variable is parsed by <see cref="SpeculationOptions"/>
        /// itself, not re-derived here: the scheduler decides from the same
        /// resolution whether to speculate, and a private "anything but 0 is on"
        /// rule made <c>TS_SPEC=false</c> page the ~3 GiB block in for a
        /// scheduler that then never drafted with it.
        /// </summary>
        internal static bool NativeMtpRequested()
        {
            var options = SpeculationOptions.FromEnvironment();
            // N-gram drafting uses the trunk's verify/rollback path and no
            // learned weights. Loading NextN would consume VRAM for a block
            // that the selected algorithm cannot call (or print a misleading
            // standard-decode fallback notice on glm5next).
            return options.Enabled && !string.Equals(options.SpeculatorName,
                SpeculatorRegistry.NGram, StringComparison.OrdinalIgnoreCase);
        }

        private void SpecForwardNative(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            lock (_nativeSync)
            {
                int hidden = Config.HiddenSize;
                int vocab = Config.VocabSize;
                long needH = (long)tokens.Length * hidden;
                long needL = (allLogitsRows ? (long)tokens.Length : 1L) * vocab;

                // The native side writes exactly n*hidden / n*vocab floats, so a
                // caller buffer that is merely large enough is fine but a short one
                // would corrupt the heap. Stage through a scratch array in that case
                // (the plain-decode path deliberately hands a 1-row logits buffer).
                float[] hBuf = hAllOut != null && hAllOut.LongLength >= needH ? hAllOut : EnsureSpecH(needH);
                float[] lBuf = logitsOut != null && logitsOut.LongLength >= needL ? logitsOut : EnsureSpecLogits(needL);

                int chunk = LayerMajorPrefillChunkTokens;
                bool layerMajor = MemoryPolicy?.EnableGlmFlashMtp == true && !allLogitsRows
                    && chunk > 0 && tokens.Length > chunk && _exec is NativeGlmExecutor;
                PrepareUnifiedGraph(checked(_exec.NPast + tokens.Length),
                    allLogitsRows ? tokens.Length : Math.Min(tokens.Length, layerMajor ? chunk : _nativeUbatch), false,
                    allLogitsRows ? tokens.Length : 1, submittedTokens: tokens.Length, layerMajor: layerMajor);
                BeforeGraphCall();
                bool ok;
                try { ok = layerMajor
                    ? ((NativeGlmExecutor)_exec).SpecForwardLayerMajor(tokens, hBuf, lBuf, chunk)
                    : _exec.SpecForward(tokens, hBuf, lBuf, allLogitsRows); }
                finally { AfterGraphCall(); }
                if (!ok)
                    throw new InvalidOperationException("glm-dsa native speculative forward failed (see stderr).");

                if (!ReferenceEquals(hBuf, hAllOut) && hAllOut != null)
                    Array.Copy(hBuf, 0, hAllOut, 0, Math.Min(hAllOut.LongLength, needH));
                if (!ReferenceEquals(lBuf, logitsOut) && logitsOut != null)
                    Array.Copy(lBuf, 0, logitsOut, 0, Math.Min(logitsOut.LongLength, needL));

                _cacheSeqLen = _exec.NPast;
                // Match the ordinary forward's diagnostic boundary. The caller
                // may reuse a larger verify buffer, but this path emits one row.
                if (!allLogitsRows) DumpLogitsIfRequested(lBuf, vocab);
            }
        }

        private void MtpDraftStepNative(int token, float[] hPrev, int pos, float[] logitsOut, float[] hOut)
        {
            lock (_nativeSync)
            {
                float[] lBuf = logitsOut ?? EnsureSpecLogits(Config.VocabSize);
                float[] hBuf = hOut ?? EnsureSpecH(Config.HiddenSize);
                PrepareUnifiedDraftGraph(checked(pos + 1), 1);
                BeforeGraphCall();
                bool ok;
                try { ok = _exec.DraftStep(token, hPrev, pos, lBuf, hBuf); }
                finally { AfterGraphCall(); }
                if (!ok)
                    throw new InvalidOperationException("glm-dsa native MTP draft step failed (see stderr).");
            }
        }

        private void MtpCatchUpNative(int[] tokens, float[] hRows, int startPos)
        {
            lock (_nativeSync)
            {
                int chunk = MemoryPolicy?.EnableGlmFlashMtp == true && LayerMajorPrefillChunkTokens > 0
                    ? Math.Min(_nativeUbatch, LayerMajorPrefillChunkTokens) : _nativeUbatch;
                if (MemoryPolicy?.EnableGlmFlashMtp == true && tokens.Length > chunk)
                {
                    // Every native chunk may introduce another retained draft
                    // graph. Re-admit between chunks using its actual live
                    // children, preserving the native chunk arithmetic/order.
                    for (int offset = 0; offset < tokens.Length; offset += chunk)
                    {
                        int count = Math.Min(chunk, tokens.Length - offset);
                        MtpCatchUpNative(tokens.AsSpan(offset, count).ToArray(),
                            hRows.AsSpan(offset * Config.HiddenSize, count * Config.HiddenSize).ToArray(), startPos + offset);
                    }
                    return;
                }
                PrepareUnifiedDraftGraph(checked(startPos + tokens.Length), Math.Min(tokens.Length, _nativeUbatch), GlmUnifiedMtpGraphPurpose.CatchUp);
                BeforeGraphCall();
                bool ok;
                try { ok = _exec.DraftCatchUp(tokens, hRows, startPos); }
                finally { AfterGraphCall(); }
                if (!ok)
                    throw new InvalidOperationException("glm-dsa native MTP catch-up failed (see stderr).");
            }
        }

        private void RewindNative(int length)
        {
            lock (_nativeSync)
            {
                if (_exec.Rewind(length))
                    _cacheSeqLen = length;
            }
        }

        // Staging buffers for the short-buffer cases above. Grown, never shrunk;
        // a speculative window is at most MtpMaxDraftTokens+1 rows.
        private float[] _specHScratch;
        private float[] _specLogitsScratch;
        private Action<int, int, GlmUnifiedMtpGraphPurpose> _unifiedDraftAdmission;
        internal void ConfigureUnifiedDraftAdmission(Action<int, int, GlmUnifiedMtpGraphPurpose> admission)
            => _unifiedDraftAdmission = admission ?? throw new ArgumentNullException(nameof(admission));
        internal long UnifiedSnapshotBytes()
        {
            lock (_nativeSync)
            {
                if (MemoryPolicy?.EnableGlmFlashMtp != true || _exec is not NativeGlmExecutor native)
                    throw new InvalidOperationException("Flash MTP snapshot ownership requires its native executor.");
                long bytes = native.KdaSnapshotBytes;
                if (bytes < 0) throw new InvalidOperationException("Flash MTP native snapshot ownership is unavailable.");
                return bytes;
            }
        }
        private void PrepareUnifiedDraftGraph(int context, int rows, GlmUnifiedMtpGraphPurpose purpose = GlmUnifiedMtpGraphPurpose.DraftStep)
        {
            if (MemoryPolicy?.EnableGlmFlashMtp != true) return;
            if (_unifiedDraftAdmission == null) throw new InvalidOperationException("Flash MTP draft admission is not configured.");
            _unifiedDraftAdmission(context, rows, purpose);
        }

        private float[] EnsureSpecH(long need)
        {
            if (_specHScratch == null || _specHScratch.LongLength < need)
                _specHScratch = new float[need];
            return _specHScratch;
        }

        private float[] EnsureSpecLogits(long need)
        {
            if (_specLogitsScratch == null || _specLogitsScratch.LongLength < need)
                _specLogitsScratch = new float[need];
            return _specLogitsScratch;
        }
    }
}
