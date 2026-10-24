// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    /// <summary>
    /// A GLM whole-model executor: the native ggml executor (<c>--backend ggml_cuda</c>,
    /// <c>ggml_vulkan</c>, ...) or the direct-CUDA engine (GLM-5.3-Flash on <c>--backend cuda</c>).
    /// Both keep every weight and cache themselves and share one contract, so the model's slot,
    /// prefix-cache and batched-decode logic runs unchanged on either. Forward, reset, rewind and
    /// the head act on the ACTIVE slot.
    /// </summary>
    internal interface IGlmExecutor : IDisposable
    {
        /// <summary>"native" or "cuda", for the KV-state fingerprint and diagnostics.</summary>
        string Kind { get; }
        int VocabSize { get; }
        int ContextSize { get; }
        /// <summary>Head of the active slot.</summary>
        int NPast { get; }
        /// <summary>The NextN/MTP draft block loaded and usable.</summary>
        bool HasDraftHead { get; }

        /// <summary>Evaluate <paramref name="tokens"/> at the active slot's head; the last token's logits.</summary>
        bool Forward(int[] tokens, float[] logitsOut);
        /// <summary>Empty the active slot; true when it is usable at position 0.</summary>
        bool ResetChecked();
        /// <summary>Move the active slot's head back; false when refused (glm5next reaches only its
        /// head or 0).</summary>
        bool Rewind(int nPast);

        /// <summary>A new, empty slot, not made active; -1 when the devices are full.</summary>
        int SlotAlloc();
        /// <summary>Complete exact-head slot copying, independently qualified
        /// from donation. Unsupported executors keep their existing behavior.</summary>
        bool SupportsSlotCopy => false;
        /// <summary>No allocation or active selection change. A failed partial
        /// copy requires checked reset of the destination before reuse.</summary>
        bool SlotCopyInto(int sourceSlot, int destinationSlot, int expectedHead) => false;
        /// <summary>Actual native buffer payload; zero when unsupported.</summary>
        long SlotBytes(int slot) => 0;
        bool SetActiveSlot(int slot);
        /// <summary>Free a slot's caches; the active slot cannot be freed.</summary>
        bool SlotFree(int slot);
        /// <summary>Replace the retained slots, oldest first, that a step which does not fit may free.</summary>
        bool SetReclaimableSlots(int[] slots);
        /// <summary>Retained slots freed since the last call, into <paramref name="buffer"/>; how many.</summary>
        int TakeReclaimedSlots(int[] buffer);
        /// <summary>One token for each of several slots at their heads; false when declined.</summary>
        bool ForwardBatchedDecode(int[] slots, int[] tokens, int[] positions, float[] logits);

        /// <summary>Projected vision rows overriding the embeddings of the next Forward's tokens from
        /// <paramref name="index"/> on.</summary>
        bool QueueVisionRows(float[] rows, int nRows, int index);
        void ClearVisionRows();

        // ---- NextN/MTP speculative decoding (executors without a draft block decline) ----
        bool SpecForward(int[] tokens, float[] hOut, float[] logitsOut, bool allLogitsRows);
        bool DraftStep(int token, float[] hPrev, int pos, float[] logitsOut, float[] hOut);
        bool DraftCatchUp(int[] tokens, float[] hRows, int startPos);
        bool KdaStateCapture();
        /// <summary>Restore the snapshot into the active slot; its position, or -1.</summary>
        int KdaStateRestore();
    }

    /// <summary>The native ggml executor behind <see cref="IGlmExecutor"/>.</summary>
    internal sealed class NativeGlmExecutor : IGlmExecutor
    {
        private IntPtr _handle;
        private bool _requireCheckedFree;
        private readonly bool _requireCheckedMappedTrim;
        private bool _mappedTrimFailed;
        private readonly Func<IntPtr, bool, bool> _releaseForTests;
        private readonly Func<IntPtr, bool, bool> _mappedTrimForTests;

        public NativeGlmExecutor(IntPtr handle, bool requireCheckedFree = false, bool requireCheckedMappedTrim = false)
        { _handle = handle; _requireCheckedFree = requireCheckedFree || requireCheckedMappedTrim; _requireCheckedMappedTrim = requireCheckedMappedTrim; }
        internal NativeGlmExecutor(IntPtr handle, bool requireCheckedFree, Func<IntPtr, bool, bool> release)
            : this(handle, requireCheckedFree) => _releaseForTests = release;
        internal NativeGlmExecutor(IntPtr handle, bool requireCheckedFree, bool requireCheckedMappedTrim,
            Func<IntPtr, bool, bool> trim, Func<IntPtr, bool, bool> release)
            : this(handle, requireCheckedFree, requireCheckedMappedTrim)
        { _mappedTrimForTests = trim; _releaseForTests = release; }
        internal void RequireCheckedDisposal() => _requireCheckedFree = true;
        private void ThrowIfMappedTrimFailed()
        {
            if (_mappedTrimFailed)
                throw new InvalidOperationException("Native GLM source-view trimming failed; execution cannot resume before model cleanup.");
        }

        public string Kind => "native";
        public int VocabSize => GgmlGlmNative.VocabSize(_handle);
        public int ContextSize => GgmlGlmNative.CtxSize(_handle);
        public int NPast => GgmlGlmNative.NPast(_handle);
        internal long[] HostTokenEmbeddingInfo => GgmlGlmNative.HostTokenEmbeddingInfo(_handle);
        internal long[] DeviceBatonInfo => GgmlGlmNative.LayerMajorDeviceBatonInfo(_handle);
        internal void CloseDeviceBaton() => GgmlGlmNative.CloseLayerMajorDeviceBaton(_handle);
        internal long[] CausalRouteNeighborInfo => GgmlGlmNative.CausalRouteNeighborInfo(_handle);
        internal void ConfigureCausalRouteNeighbor(GgmlHostMoeExpertCacheScope scope, long bytes)
            => scope.ConfigureCausalRouteNeighbor(_handle,bytes);
        internal void CloseCausalRouteNeighbor() => GgmlGlmNative.CloseCausalRouteNeighbor(_handle);
        internal bool ConfigureGraphCache(int capacity, int trim) => GgmlGlmNative.ConfigureGraphCache(_handle, capacity, trim);
        internal bool TrimMappedResidency()
        {
            ThrowIfMappedTrimFailed();
            try
            {
                return _mappedTrimForTests != null ? _mappedTrimForTests(_handle, _requireCheckedMappedTrim)
                    : _requireCheckedMappedTrim ? GgmlGlmNative.TrimMappedResidencyChecked(_handle)
                    : GgmlGlmNative.TrimMappedResidency(_handle);
            }
            catch
            {
                if (_requireCheckedMappedTrim) _mappedTrimFailed = true;
                throw;
            }
        }
        internal GgmlGlmBatchExecution LastBatchedExecution => GgmlGlmNative.TryGetLastBatchedExecution(_handle);
        internal bool ConfigureSharedGraphOwner(bool enabled) => GgmlGlmNative.ConfigureSharedGraphOwner(_handle, enabled);
        internal bool ConfigureLayerMajorGraphCap(int profileVersion, long capBytes)
            => GgmlGlmNative.ConfigureLayerMajorGraphCap(_handle, profileVersion, capBytes);
        internal int SelectedExpertGroups => GgmlGlmNative.SelectedExpertGroupCount(_handle);
        internal bool FlashMtpShapePlanEnabled => GgmlGlmNative.FlashMtpShapePlanEnabled(_handle);
        internal bool FlashMtpPrefixCommitEnabled => GgmlGlmNative.FlashMtpPrefixCommitEnabled(_handle);
        internal int FlashMtpCommitVerifiedPrefix(int basePosition, int acceptedDrafts, int verifyDrafts)
            => GgmlGlmNative.FlashMtpCommitVerifiedPrefix(_handle, basePosition, acceptedDrafts, verifyDrafts);
        internal long PlanSelectedExpertMinimum(int rows) => GgmlGlmNative.SelectedExpertMinimumBudget(_handle, rows);
        internal bool ConfigureSelectedExperts(bool enabled) => GgmlGlmNative.ConfigureSelectedExpertSeams(_handle, enabled);
        public bool HasDraftHead => GgmlGlmNative.HasDraftHead(_handle);

        public bool Forward(int[] tokens, float[] logitsOut)
        { ThrowIfMappedTrimFailed(); return GgmlGlmNative.Forward(_handle, tokens, logitsOut); }
        internal int ForwardLayerMajor(int[] tokens, float[] logitsOut, int chunkRows)
        { ThrowIfMappedTrimFailed(); return GgmlGlmNative.ForwardLayerMajor(_handle, tokens, logitsOut, chunkRows); }
        internal bool SpecForwardLayerMajor(int[] tokens, float[] hOut, float[] logitsOut, int chunkRows)
            => GgmlGlmNative.SpecForwardLayerMajor(_handle, tokens, hOut, logitsOut, chunkRows);
        public bool ResetChecked()
        { ThrowIfMappedTrimFailed(); return GgmlGlmNative.ResetChecked(_handle); }
        public bool Rewind(int nPast) => GgmlGlmNative.Rewind(_handle, nPast);
        public int SlotAlloc() => GgmlGlmNative.SlotAlloc(_handle);
        public bool SupportsSlotCopy => GgmlGlmNative.SupportsSlotCopy();
        public bool SlotCopyInto(int sourceSlot, int destinationSlot, int expectedHead)
            => GgmlGlmNative.SlotCopyInto(_handle, sourceSlot, destinationSlot, expectedHead);
        public long SlotBytes(int slot) => GgmlGlmNative.SlotBytes(_handle, slot);
        public bool SetActiveSlot(int slot) => GgmlGlmNative.SetActiveSlot(_handle, slot);
        public bool SlotFree(int slot) => GgmlGlmNative.SlotFree(_handle, slot);
        public bool SetReclaimableSlots(int[] slots) => GgmlGlmNative.SetReclaimableSlots(_handle, slots);
        public int TakeReclaimedSlots(int[] buffer) => GgmlGlmNative.TakeReclaimedSlots(_handle, buffer);
        public bool ForwardBatchedDecode(int[] slots, int[] tokens, int[] positions, float[] logits)
        { ThrowIfMappedTrimFailed(); return GgmlGlmNative.ForwardBatchedDecode(_handle, slots, tokens, positions, logits); }
        public bool QueueVisionRows(float[] rows, int nRows, int index) => GgmlGlmNative.QueueVisionRows(_handle, rows, nRows, index);
        public void ClearVisionRows() => GgmlGlmNative.ClearVisionRows(_handle);
        public bool SpecForward(int[] tokens, float[] hOut, float[] logitsOut, bool allLogitsRows)
            => GgmlGlmNative.SpecForward(_handle, tokens, hOut, logitsOut, allLogitsRows);
        public bool DraftStep(int token, float[] hPrev, int pos, float[] logitsOut, float[] hOut)
            => GgmlGlmNative.DraftStep(_handle, token, hPrev, pos, logitsOut, hOut);
        public bool DraftCatchUp(int[] tokens, float[] hRows, int startPos) => GgmlGlmNative.DraftCatchUp(_handle, tokens, hRows, startPos);
        public bool KdaStateCapture() => GgmlGlmNative.KdaStateCapture(_handle);
        public int KdaStateRestore() => GgmlGlmNative.KdaStateRestore(_handle);
        internal long KdaSnapshotBytes => GgmlGlmNative.KdaSnapshotBytes(_handle);

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                bool required = _requireCheckedFree
                    || Environment.GetEnvironmentVariable(GgmlHostMoeExpertCacheScope.VmmWeightsEnvironmentVariable) == "1"
                    || Environment.GetEnvironmentVariable(GgmlHostMoeExpertCacheScope.FileSourceViewEnvironmentVariable) == "1";
                bool released = _releaseForTests == null ? GgmlGlmNative.FreeChecked(_handle, required)
                    : _releaseForTests(_handle, required);
                if (!released) throw new InvalidOperationException("Native GLM still owns expert-cache resources; checked disposal must be retried.");
                _handle = IntPtr.Zero;
            }
        }
    }
}
