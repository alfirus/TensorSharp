// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// P/Invoke surface for the native GLM (glm-dsa) whole-model executor
// (TensorSharp.GGML.Native/ggml_ops_glm_dsa.cpp). The native side loads the
// (split) GGUF itself, places the layers across every visible GPU, owns the
// MLA and lightning-indexer KV caches, and runs prefill / decode ubatches
// through ggml_backend_sched.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML
{
    /// <summary>Completed logical native GLM graph and its exact slot-head
    /// updates. BackendSplits counts scheduler spans, not device launches.</summary>
    public sealed record GgmlGlmBatchExecution(ulong ExecutionId, int Rows, int Nodes,
        int BackendSplits, bool GraphReused, IReadOnlyList<int> Slots,
        IReadOnlyList<int> PositionsBefore, IReadOnlyList<int> PositionsAfter);

    public static partial class GgmlGlmNative
    {
        private const string DllName = "GgmlOps";
        private const CallingConvention Conv = CallingConvention.Cdecl;

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmUnifiedBudgetVersion();

        public static void RequireUnifiedBudgetSupport()
        {
            try
            {
                if (TSGgml_GlmUnifiedBudgetVersion() != 2) throw new NotSupportedException("Unified GLM requires native budget interface version 2 with vendor-safe scheduler allocation types.");
                // Require the companion dispatch-failure fence, not just byte accounting.
                _ = TSGgml_GlmLastBatchedFailureNeedsReset(IntPtr.Zero);
                _ = TSGgml_GlmConfigureSharedGraphOwner(IntPtr.Zero, 1);
                _ = TSGgml_GlmTrimMappedResidency(IntPtr.Zero);
            }
            catch (EntryPointNotFoundException error) { throw new NotSupportedException("Unified GLM requires the matching native library with complete GLM buffer ownership.", error); }
        }


        /// <summary>No routed-expert CPU offload (the default).</summary>
        public const int CpuMoeNone = 0;
        /// <summary>Offload only as many leading layers as the VRAM actually needs.</summary>
        public const int CpuMoeAuto = -1;

        static GgmlGlmNative()
        {
            GgmlNative.EnsureImportResolverRegistered();
        }

        // Paths cross as UTF-8: ggml_fopen decodes them as UTF-8 on Windows,
        // while CharSet.Ansi would marshal the active code page.
        [DllImport(DllName, CallingConvention = Conv)]
        private static extern IntPtr TSGgml_GlmLoadModel([MarshalAs(UnmanagedType.LPUTF8Str)] string ggufPath,
            int nGpu, int nCtx, int nUbatch, int nThreads, int nCpuMoe,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string backendName, int tp, int ctxIsHardLimit, int loadMtp);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmHostTokenEmbeddingVersion();
        [DllImport(DllName, CallingConvention = Conv)]
        private static extern IntPtr TSGgml_GlmLoadModelWithHostTokenEmbedding(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string ggufPath,
            int nGpu, int nCtx, int nUbatch, int nThreads, int nCpuMoe,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string backendName, int tp, int ctxIsHardLimit, int loadMtp, int enabled);
        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmHostTokenEmbeddingInfo(IntPtr handle, [Out] long[] values, int length);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmLayerMajorDeviceBatonVersion();
        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmLayerMajorDeviceBatonEnabled();
        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmLayerMajorDeviceBatonClose(IntPtr handle);
        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmLayerMajorDeviceBatonInfo(IntPtr handle, [Out] long[] values, int length);
        public static void RequireLayerMajorDeviceBatonSupport()
            => CompleteLayerMajorDeviceBatonHandshake(true,TSGgml_GlmLayerMajorDeviceBatonVersion,
                TSGgml_GlmLayerMajorDeviceBatonEnabled,() => TSGgml_GlmLayerMajorDeviceBatonInfo(IntPtr.Zero,new long[12],12),
                () => TSGgml_GlmLayerMajorDeviceBatonClose(IntPtr.Zero));
        internal static void CompleteLayerMajorDeviceBatonHandshake(bool requested,Func<int> version,
            Func<int> enabled,Func<int> nullInfo,Func<int> nullClose)
        {
            if (!requested) return;
            try {
                if (version() != 1 || enabled() != 1 || nullInfo() != 0 || nullClose() != 1)
                    throw new NotSupportedException("Layer-major device baton requires native version1 and checked ownership.");
            } catch (EntryPointNotFoundException error) {
                throw new NotSupportedException("The native library lacks the checked layer-major Device bank.",error);
            }
        }
        public static void CloseLayerMajorDeviceBaton(IntPtr handle)
            => CompleteLayerMajorDeviceBatonClose(() => TSGgml_GlmLayerMajorDeviceBatonClose(handle));
        internal static void CompleteLayerMajorDeviceBatonClose(Func<int> close)
        {
            if (close() != 1)
                throw new InvalidOperationException("Layer-major Device bank cleanup retains its physical owner or checked ticket.");
        }
        public static long[] LayerMajorDeviceBatonInfo(IntPtr handle)
        {
            var info = new long[12];
            if (TSGgml_GlmLayerMajorDeviceBatonInfo(handle,info,info.Length) != 1)
                throw new InvalidOperationException("Cannot read layer-major Device bank ownership.");
            return info;
        }

        public static int HostTokenEmbeddingVersion()
        {
            try { return TSGgml_GlmHostTokenEmbeddingVersion(); }
            catch (Exception error) when (error is EntryPointNotFoundException or DllNotFoundException)
            { throw new NotSupportedException("Host token embedding requires the matching native library.", error); }
        }

        /// <summary>Actual-handle profile data, never inferred from the process
        /// flag. A missing/off certificate fails closed for the opt-in loader.</summary>
        public static long[] HostTokenEmbeddingInfo(IntPtr handle)
        {
            var values = new long[12];
            try
            {
                if (TSGgml_GlmHostTokenEmbeddingInfo(handle, values, values.Length) != 1)
                    throw new NotSupportedException("The native GLM handle did not activate host token embedding.");
                return values;
            }
            catch (Exception error) when (error is EntryPointNotFoundException or DllNotFoundException)
            { throw new NotSupportedException("Host token embedding requires the matching native library.", error); }
        }

        /// <summary>Explicit immutable raw-row mode. The native loader also
        /// requires exact environment opt-in and accounted allocation owners;
        /// the legacy loader never silently adopts this mode.</summary>
        public static IntPtr LoadModelWithHostTokenEmbedding(string ggufPath, int nGpu, int nCtx, int nUbatch,
            int nThreads, int nCpuMoe, string backendName, int tp, bool ctxIsHardLimit)
        {
            if (HostTokenEmbeddingVersion() != 1)
                throw new NotSupportedException("Host token embedding requires native profile 1.");
            try
            {
                return TSGgml_GlmLoadModelWithHostTokenEmbedding(ggufPath, nGpu, nCtx, nUbatch, nThreads,
                    nCpuMoe, backendName ?? string.Empty, tp, ctxIsHardLimit ? 1 : 0, 0, 1);
            }
            catch (Exception error) when (error is EntryPointNotFoundException or DllNotFoundException)
            { throw new NotSupportedException("Host token embedding requires the matching native loader.", error); }
        }

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmVocabSize(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmCtxSize(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmNPast(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmConfigureGraphCache(IntPtr handle, int capacity, int trim);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmConfigureSharedGraphOwner(IntPtr handle, int enabled);

        /// <summary>Configure a stable session owner for all single-rank graph
        /// payloads. Ownership changes require a quiescent, empty graph cache.</summary>
        public static bool ConfigureSharedGraphOwner(IntPtr handle, bool enabled)
            => TSGgml_GlmConfigureSharedGraphOwner(handle, enabled ? 1 : 0) != 0;

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmLayerMajorGraphCapVersion();
        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmConfigureLayerMajorGraphCap(IntPtr handle, int profileVersion, long capBytes);

        public static int LayerMajorGraphCapVersion()
        {
            try { return TSGgml_GlmLayerMajorGraphCapVersion(); }
            catch (Exception error) when (error is EntryPointNotFoundException or DllNotFoundException)
            { throw new NotSupportedException("Strict layer-major graph caps require the matching native library.", error); }
        }

        /// <summary>Latch an immutable payload cap on this actual handle before
        /// its first graph. A version probe alone does not certify the handle.</summary>
        public static bool ConfigureLayerMajorGraphCap(IntPtr handle, int profileVersion, long capBytes)
        {
            try { return TSGgml_GlmConfigureLayerMajorGraphCap(handle, profileVersion, capBytes) == 1; }
            catch (Exception error) when (error is EntryPointNotFoundException or DllNotFoundException)
            { throw new NotSupportedException("Strict layer-major graph caps require the matching native library.", error); }
        }

        /// <summary>Quiescent graph-cache policy. trim=1 releases all graphs;
        /// trim=2 releases only prefill graphs. State slots and weights stay live.</summary>
        public static bool ConfigureGraphCache(IntPtr handle, int capacity, int trim = 0)
            => TSGgml_GlmConfigureGraphCache(handle, capacity, trim) != 0;

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmTrimMappedResidency(IntPtr handle);
        public static bool TrimMappedResidency(IntPtr handle) => TSGgml_GlmTrimMappedResidency(handle) != 0;
        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmTrimMappedResidencyVersion();
        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmTrimMappedResidencyChecked(IntPtr handle);
        public static void RequireCheckedMappedResidencySupport()
            => ValidateCheckedMappedResidencySupport(TSGgml_GlmTrimMappedResidencyVersion,
                () => TSGgml_GlmTrimMappedResidencyChecked(IntPtr.Zero));
        internal static void ValidateCheckedMappedResidencySupport(Func<int> version, Func<int> nullProbe)
        {
            try
            {
                // The NULL probe is expressly side-effect-free and verifies
                // the companion export before any real model is loaded.
                if (version() != 1 || nullProbe() != 0)
                    throw new NotSupportedException("Source views require checked mapped-residency interface version 1.");
            }
            catch (EntryPointNotFoundException error)
            { throw new NotSupportedException("Source views require checked mapped-residency trimming before model load.", error); }
        }
        /// <summary>Quiescent source-view working-set hint. False is advisory;
        /// invalid owner/state throws and never becomes a successful boolean hint.</summary>
        public static bool TrimMappedResidencyChecked(IntPtr handle)
            => CompleteCheckedMappedResidencyTrim(() => TSGgml_GlmTrimMappedResidencyChecked(handle));
        internal static bool CompleteCheckedMappedResidencyTrim(Func<int> trim)
        {
            int result;
            try { result = trim(); }
            catch (EntryPointNotFoundException error)
            { throw new NotSupportedException("Source views cannot fall back to legacy mapped-residency trimming.", error); }
            return result switch
            {
                0 => false,
                1 => true,
                _ => throw new InvalidOperationException("Native GLM source-view pressure trim failed; its owner must be retained for checked cleanup.")
            };
        }
        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmSelectedExpertAdapterVersion();
        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmFlashMtpAdapterVersion();
        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmFlashMtpShapePlanVersion();
        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmFlashMtpShapePlanEnabled(IntPtr handle);
        public static bool SupportsFlashMtpShapePlan()
        {
            try { return TSGgml_GlmFlashMtpShapePlanVersion() == 1 && TSGgml_GlmFlashMtpShapePlanEnabled(IntPtr.Zero) == 0; }
            catch (EntryPointNotFoundException) { return false; }
        }
        public static bool FlashMtpShapePlanEnabled(IntPtr handle) => TSGgml_GlmFlashMtpShapePlanEnabled(handle) == 1;
        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmFlashMtpPrefixCommitVersion();
        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmFlashMtpPrefixCommitEnabled(IntPtr handle);
        [DllImport(DllName, CallingConvention = Conv, EntryPoint = "TSGgml_GlmCommitVerifiedPrefix")]
        private static extern int TSGgml_GlmFlashMtpCommitVerifiedPrefix(IntPtr handle, int basePosition, int acceptedDrafts, int verifyDrafts);
        public static bool SupportsFlashMtpPrefixCommit()
        {
            try { return TSGgml_GlmFlashMtpPrefixCommitVersion() == 1
                && TSGgml_GlmFlashMtpPrefixCommitEnabled(IntPtr.Zero) == 0
                && TSGgml_GlmFlashMtpCommitVerifiedPrefix(IntPtr.Zero, 0, 0, 1) == 0; }
            catch (EntryPointNotFoundException) { return false; }
        }
        public static bool FlashMtpPrefixCommitEnabled(IntPtr handle) => TSGgml_GlmFlashMtpPrefixCommitEnabled(handle) == 1;
        // Zero is a refusal before any state copy. A negative result is a
        // failed commit, never permission to replay from a partly changed state.
        public static int FlashMtpCommitVerifiedPrefix(IntPtr handle, int basePosition, int acceptedDrafts, int verifyDrafts)
            => TSGgml_GlmFlashMtpCommitVerifiedPrefix(handle, basePosition, acceptedDrafts, verifyDrafts);
        public static void RequireFlashMtpSupport()
        {
            try
            {
                if (TSGgml_GlmFlashMtpAdapterVersion() != 1 || TSGgml_GlmSelectedExpertAdapterVersion() != 8
                    || TSGgml_GlmKdaSnapshotBytes(IntPtr.Zero) != -1)
                    throw new NotSupportedException("Flash MTP requires its exact draft, verification and shared-budget native adapter.");
            }
            catch (EntryPointNotFoundException error)
            { throw new NotSupportedException("Flash MTP requires the matching updated native library.", error); }
        }
        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmSelectedExpertGroupCount(IntPtr handle);
        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmSelectedExpertMinimumBudget(IntPtr handle, int decodeRows, out long bytes);
        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmConfigureSelectedExpertSeams(IntPtr handle, int enabled);
        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmSelectedExpertExecutionStats(IntPtr handle, out ulong graphs, out ulong calls);
        public static int SelectedExpertGroupCount(IntPtr handle) => TSGgml_GlmSelectedExpertGroupCount(handle);
        public static void RequireSelectedExpertPlanningSupport()
        {
            try
            {
                ValidateSelectedExpertAdapterVersion(TSGgml_GlmSelectedExpertAdapterVersion());
                _ = TSGgml_GlmSelectedExpertMinimumBudget(IntPtr.Zero, 1, out _);
            }
            catch (EntryPointNotFoundException error)
            {
                throw new NotSupportedException("GLM selected-expert execution requires its matching updated native library.", error);
            }
        }
        internal static void ValidateSelectedExpertAdapterVersion(int version)
        {
            if (version is not (7 or 8))
                throw new NotSupportedException("GLM selected-expert execution requires split scratch ownership, exact serialized phase borrowing and admitted prefill widths through 1024 with 8192 route elements (adapter version 7).");
        }
        /// <summary>Quiescent size-only native plan. Includes complete quantized
        /// tables for every possible routed union and all planned graph variants;
        /// allocates no device payload and changes no resident expert table.</summary>
        public static long SelectedExpertMinimumBudget(IntPtr handle, int decodeRows)
        {
            if (decodeRows is < 1 or > 3) throw new ArgumentOutOfRangeException(nameof(decodeRows));
            RequireSelectedExpertPlanningSupport();
            if (TSGgml_GlmSelectedExpertMinimumBudget(handle, decodeRows, out long bytes) != 1 || bytes <= 0)
                throw new NotSupportedException("GLM exact compact expert geometry could not be planned.");
            return bytes;
        }
        /// <summary>Experimental explicit model-backend expert adapter. Requires
        /// an active matching expert-cache scope and empty graphs/slot heads.</summary>
        public static bool ConfigureSelectedExpertSeams(IntPtr handle, bool enabled)
        {
            RequireSelectedExpertPlanningSupport();
            return TSGgml_GlmConfigureSelectedExpertSeams(handle, enabled ? 1 : 0) != 0;
        }
        public static (ulong Graphs, ulong Seams) SelectedExpertExecutionStats(IntPtr handle)
        {
            if (TSGgml_GlmSelectedExpertExecutionStats(handle, out ulong graphs, out ulong calls) != 1)
                throw new InvalidOperationException("GLM selected-expert evidence is unavailable.");
            return (graphs, calls);
        }


        [DllImport(DllName, CallingConvention = Conv)]
        private static extern unsafe int TSGgml_GlmForward(IntPtr handle, int* tokens, int nTokens, float* logitsOut);
        [DllImport(DllName, CallingConvention = Conv)]
        private static extern unsafe int TSGgml_GlmForwardLayerMajor(IntPtr handle, int* tokens, int nTokens, float* logitsOut, int chunkRows);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern void TSGgml_GlmReset(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmRewind(IntPtr handle, int nPast);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmForwardBatchedDecode(IntPtr handle, int n, int[] slotIds,
            int[] tokens, int[] positions, float[] logitsOut);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmLastBatchedFailureNeedsReset(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmLastBatchedExecution(IntPtr handle, out ulong id,
            out int rows, out int nodes, out int backendSplits, out int reused,
            [Out] int[] slots, [Out] int[] before, [Out] int[] after, int capacity);

        /// <summary>Query while the native executor is quiescent or locked.
        /// Legacy ordinary deployments without this diagnostic return null.</summary>
        public static GgmlGlmBatchExecution TryGetLastBatchedExecution(IntPtr handle)
        {
            const int capacity = 16;
            int[] slots = new int[capacity], before = new int[capacity], after = new int[capacity];
            try
            {
                if (TSGgml_GlmLastBatchedExecution(handle, out var id, out var rows, out var nodes,
                    out var splits, out var reused, slots, before, after, capacity) != 1) return null;
                if (id == 0 || rows is < 2 or > capacity || nodes <= 0 || splits <= 0)
                    throw new InvalidOperationException("Native GLM returned invalid successful batch evidence.");
                Array.Resize(ref slots, rows); Array.Resize(ref before, rows); Array.Resize(ref after, rows);
                return new(id, rows, nodes, splits, reused != 0,
                    Array.AsReadOnly(slots), Array.AsReadOnly(before), Array.AsReadOnly(after));
            }
            catch (EntryPointNotFoundException) { return null; }
        }

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmSlotAlloc(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmSlotCopyVersion();

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmSlotCopyInto(IntPtr handle, int sourceSlot, int destinationSlot, int expectedHead);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern long TSGgml_GlmSlotBytes(IntPtr handle, int slotId);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmSetActiveSlot(IntPtr handle, int slotId);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmSlotFree(IntPtr handle, int slotId);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmSetReclaimableSlots(IntPtr handle, int[] slotIds, int n);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmTakeReclaimedSlots(IntPtr handle, int[] slotIds, int cap);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern void TSGgml_GlmFree(IntPtr handle);
        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmFreeChecked(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern unsafe int TSGgml_GlmQueueVisionRows(IntPtr handle, float* rows, int nRows, int index);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern void TSGgml_GlmClearVisionRows(IntPtr handle);

        // ---- NextN/MTP speculative decoding -------------------------------

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmHasMtp(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmHiddenSize(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern unsafe int TSGgml_GlmSpecForward(IntPtr handle, int* tokens, int nTokens,
            float* hOut, float* logitsOut, int allLogitsRows);
        [DllImport(DllName, CallingConvention = Conv)]
        private static extern unsafe int TSGgml_GlmSpecForwardLayerMajor(IntPtr handle, int* tokens, int nTokens,
            float* hOut, float* logitsOut, int chunkRows);
        public static unsafe bool SpecForwardLayerMajor(IntPtr handle, int[] tokens, float[] hOut, float[] logitsOut, int chunkRows)
        {
            ArgumentNullException.ThrowIfNull(tokens);
            ArgumentNullException.ThrowIfNull(hOut);
            ArgumentNullException.ThrowIfNull(logitsOut);
            if (tokens.Length == 0 || chunkRows <= 0 || hOut.LongLength < (long)tokens.Length * HiddenSize(handle)
                || logitsOut.Length < VocabSize(handle)) throw new ArgumentException("Flash MTP layer-major buffers are too short.");
            fixed (int* t = tokens) fixed (float* h = hOut) fixed (float* l = logitsOut)
                return TSGgml_GlmSpecForwardLayerMajor(handle, t, tokens.Length, h, l, chunkRows) == 1;
        }

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern unsafe int TSGgml_GlmMtpDraftStep(IntPtr handle, int token, float* hPrev, int pos,
            float* logitsOut, float* hOut);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern unsafe int TSGgml_GlmMtpCatchUp(IntPtr handle, int* tokens, int nTokens,
            float* hRows, int startPos);

        // ---- glm5next KDA recurrent-state snapshot (speculative rollback) ----

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmKdaStateCapture(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmKdaStateRestore(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern long TSGgml_GlmKdaSnapshotBytes(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_GlmResetChecked(IntPtr handle);

        /// <param name="nGpu">GPUs to spread the layers over; 0 = every visible device.</param>
        /// <param name="nCpuMoe">Leading layers whose routed experts stay in system RAM;
        /// <see cref="CpuMoeAuto"/> offloads the fewest that make the model fit.</param>
        /// <param name="backendName">ggml backend registry name ("CUDA" / "Vulkan" / "Metal");
        /// null or empty accepts any GPU backend.</param>
        /// <param name="tp">Tensor-parallel ranks. 1 (the default) layer-splits the
        /// model across the visible GPUs; &gt;1 runs EVERY layer on every rank with
        /// the attention heads and the routed experts sharded, so all the GPUs work
        /// on each token instead of one at a time.</param>
        /// <param name="ctxIsHardLimit">True when <paramref name="nCtx"/> came from
        /// the caller (MAX_CONTEXT) and must be honoured or refused; false when it
        /// is only what the GGUF advertises, which the loader may cap to whatever
        /// the devices can actually hold beside the weights.</param>
        /// <param name="loadMtp">Load the trailing NextN/MTP draft block. It is a
        /// whole extra decoder layer (~3 GiB of GLM-5.2 at IQ2_XXS), so it stays
        /// unloaded unless speculation was actually requested.</param>
        public static IntPtr LoadModel(string ggufPath, int nGpu, int nCtx, int nUbatch, int nThreads,
            int nCpuMoe = CpuMoeNone, string backendName = null, int tp = 1, bool ctxIsHardLimit = false,
            bool loadMtp = false)
            => TSGgml_GlmLoadModel(ggufPath, nGpu, nCtx, nUbatch, nThreads, nCpuMoe, backendName ?? string.Empty, tp,
                                   ctxIsHardLimit ? 1 : 0, loadMtp ? 1 : 0);

        public static int VocabSize(IntPtr handle) => TSGgml_GlmVocabSize(handle);
        public static int CtxSize(IntPtr handle) => TSGgml_GlmCtxSize(handle);
        public static int NPast(IntPtr handle) => TSGgml_GlmNPast(handle);

        /// <summary>Evaluate <paramref name="tokens"/> at the active slot's current
        /// position. <paramref name="logitsOut"/>, when non-null, receives the last
        /// token's logits.</summary>
        public static unsafe bool Forward(IntPtr handle, int[] tokens, float[] logitsOut)
        {
            fixed (int* t = tokens)
            fixed (float* l = logitsOut)
            {
                return TSGgml_GlmForward(handle, t, tokens.Length, l) != 0;
            }
        }

        /// <summary>Layer-major prefill of <paramref name="tokens"/> in chunks of at
        /// most <paramref name="chunkRows"/>: each trunk layer runs over every chunk
        /// before the next layer, so its routed experts are staged once. 1 on success,
        /// 0 on failure (the slot then needs a reset), -1 when this configuration
        /// requires the ordinary <see cref="Forward"/>.</summary>
        public static unsafe int ForwardLayerMajor(IntPtr handle, int[] tokens, float[] logitsOut, int chunkRows)
        {
            fixed (int* t = tokens)
            fixed (float* l = logitsOut)
            {
                return TSGgml_GlmForwardLayerMajor(handle, t, tokens.Length, l, chunkRows);
            }
        }

        public static void Reset(IntPtr handle) => TSGgml_GlmReset(handle);

        /// <summary>Queue projected vision-embedding rows (glm5next) to override the
        /// token embeddings of image-placeholder positions in the NEXT Forward call.
        /// <paramref name="index"/> is the first placeholder's position within that
        /// call's token array. The queue is consumed by the forward.</summary>
        public static unsafe bool QueueVisionRows(IntPtr handle, float[] rows, int nRows, int index)
        {
            fixed (float* r = rows)
            {
                return TSGgml_GlmQueueVisionRows(handle, r, nRows, index) != 0;
            }
        }

        /// <summary>Drop queued vision rows (a cancelled or re-planned prompt).</summary>
        public static void ClearVisionRows(IntPtr handle) => TSGgml_GlmClearVisionRows(handle);

        /// <summary>Drop the KV of the tokens after <paramref name="nPast"/>.</summary>
        public static bool Rewind(IntPtr handle, int nPast) => TSGgml_GlmRewind(handle, nPast) != 0;

        /// <summary>One decode step for several sequences at once — one token per
        /// sequence, sharing a single read of the weights. Returns false when the
        /// native side declines the batch (tensor parallelism, a position that
        /// disagrees with its slot, a repeated slot), in which case the caller
        /// stays on the per-sequence path. Failures after dispatch throw because
        /// every participating slot must be reset before reuse.</summary>
        public static bool ForwardBatchedDecode(IntPtr handle, int[] slotIds, int[] tokens,
                                                int[] positions, float[] logitsOut)
        {
            if (TSGgml_GlmForwardBatchedDecode(handle, slotIds.Length, slotIds, tokens, positions, logitsOut) == 1) return true;
            try
            {
                if (TSGgml_GlmLastBatchedFailureNeedsReset(handle) != 0)
                    throw new InvalidOperationException("Native GLM batched compute failed after dispatch; all participating slots require reset. Serial retry is unsafe.");
            }
            catch (EntryPointNotFoundException) { } // compatibility for ordinary legacy executors
            return false;
        }

        /// <summary>Allocate a sequence slot (own KV caches, shared weights);
        /// -1 on failure.</summary>
        public static int SlotAlloc(IntPtr handle) => TSGgml_GlmSlotAlloc(handle);

        /// <summary>Whether this DLL exposes the complete, allocation-free exact
        /// slot-copy contract. Model/backend qualification remains the caller's
        /// responsibility; older ordinary deployments retain donate-only reuse.</summary>
        public static bool SupportsSlotCopy()
        {
            try { return TSGgml_GlmSlotCopyVersion() == 1; }
            catch (EntryPointNotFoundException) { return false; }
        }

        /// <summary>Copy complete MLA/indexer/KDA (and loaded draft) state into
        /// an already admitted independent slot, at exactly the source's head.
        /// Does not select a slot or allocate payloads. A partial failure keeps
        /// the destination unusable until checked reset; source stays unchanged.</summary>
        public static bool SlotCopyInto(IntPtr handle, int sourceSlot, int destinationSlot, int expectedHead)
            => TSGgml_GlmSlotCopyInto(handle, sourceSlot, destinationSlot, expectedHead) == 1;

        /// <summary>Actual allocated native buffer payload for this slot, or
        /// zero for an unavailable/unknown slot. Geometry is only a forecast.</summary>
        public static long SlotBytes(IntPtr handle, int slotId)
        {
            try { return TSGgml_GlmSlotBytes(handle, slotId); }
            catch (EntryPointNotFoundException) { return 0; }
        }

        /// <summary>Select the slot Forward / Reset / NPast act on.</summary>
        public static bool SetActiveSlot(IntPtr handle, int slotId) => TSGgml_GlmSetActiveSlot(handle, slotId) != 0;

        /// <summary>Free a slot's caches and the graphs built against them.</summary>
        public static bool SlotFree(IntPtr handle, int slotId) => TSGgml_GlmSlotFree(handle, slotId) != 0;

        /// <summary>Replace the retained slots, oldest first, that a graph which does not fit may free.</summary>
        public static bool SetReclaimableSlots(IntPtr handle, int[] slotIds)
            => TSGgml_GlmSetReclaimableSlots(handle, slotIds, slotIds.Length) != 0;

        /// <summary>Retained slots freed for a graph since the last call, into <paramref name="buffer"/>; how many.</summary>
        public static int TakeReclaimedSlots(IntPtr handle, int[] buffer)
            => TSGgml_GlmTakeReclaimedSlots(handle, buffer, buffer.Length);

        public static void Free(IntPtr handle) => TSGgml_GlmFree(handle);
        public static void RequireCheckedFreeSupport()
        {
            try
            {
                if (TSGgml_GlmFreeChecked(IntPtr.Zero) != 1)
                    throw new NotSupportedException("Native GLM checked disposal is unavailable.");
            }
            catch (EntryPointNotFoundException error)
            { throw new NotSupportedException("Checked expert resources require native checked model disposal.", error); }
        }
        internal static bool CompleteCheckedFree(bool required, Func<int> checkedFree, Action legacyFree)
        {
            try { return checkedFree() == 1; }
            catch (EntryPointNotFoundException error)
            {
                if (required) throw new NotSupportedException("Checked expert resources require native checked model disposal.", error);
                legacyFree(); return true;
            }
        }
        /// <summary>Refusal preserves the handle's actual native owner. Legacy
        /// fallback is only for an older library when checked disposal is not
        /// required; a present export returning failure never falls back.</summary>
        public static bool FreeChecked(IntPtr handle, bool required)
            => CompleteCheckedFree(required, () => TSGgml_GlmFreeChecked(handle), () => TSGgml_GlmFree(handle));

        // ---- NextN/MTP speculative decoding -------------------------------

        /// <summary>True when the trailing NextN/MTP block was loaded and is usable.</summary>
        public static bool HasDraftHead(IntPtr handle) => TSGgml_GlmHasMtp(handle) != 0;

        /// <summary>Hidden size (n_embd) — the width of one h_nextn row.</summary>
        public static int HiddenSize(IntPtr handle) => TSGgml_GlmHiddenSize(handle);

        /// <summary>Trunk forward that also captures the post-output_norm hidden
        /// state of every row into <paramref name="hOut"/>. With
        /// <paramref name="allLogitsRows"/> the LM head runs on every row
        /// (n*vocab floats out); otherwise only the last row's logits are written.</summary>
        public static unsafe bool SpecForward(IntPtr handle, int[] tokens, float[] hOut,
                                              float[] logitsOut, bool allLogitsRows)
        {
            fixed (int* t = tokens)
            fixed (float* h = hOut)
            fixed (float* l = logitsOut)
            {
                return TSGgml_GlmSpecForward(handle, t, tokens.Length, h, l, allLogitsRows ? 1 : 0) != 0;
            }
        }

        /// <summary>One NextN draft step at <paramref name="pos"/>.</summary>
        public static unsafe bool DraftStep(IntPtr handle, int token, float[] hPrev, int pos,
                                               float[] logitsOut, float[] hOut)
        {
            fixed (float* hp = hPrev)
            fixed (float* l = logitsOut)
            fixed (float* h = hOut)
            {
                return TSGgml_GlmMtpDraftStep(handle, token, hp, pos, l, h) != 0;
            }
        }

        /// <summary>Replay verified tokens through the NextN block so its KV cache
        /// tracks the trunk. Row k of <paramref name="hRows"/> is the trunk hidden
        /// state of the token preceding <c>tokens[k]</c>.</summary>
        public static unsafe bool DraftCatchUp(IntPtr handle, int[] tokens, float[] hRows, int startPos)
        {
            fixed (int* t = tokens)
            fixed (float* h = hRows)
            {
                return TSGgml_GlmMtpCatchUp(handle, t, tokens.Length, h, startPos) != 0;
            }
        }

        // ---- glm5next KDA recurrent-state snapshot (speculative rollback) ----

        /// <summary>Copy the active slot's KDA recurrent state (every rank and
        /// layer, device-to-device) into the executor's snapshot arena and record
        /// the slot's position. Taken right before a speculative verify batch.</summary>
        public static bool KdaStateCapture(IntPtr handle) => TSGgml_GlmKdaStateCapture(handle) != 0;

        /// <summary>Copy the snapshot back into the active slot and rewind it to the
        /// captured position. Returns that position, or -1 when no usable snapshot
        /// exists for the active slot (never taken, taken of another slot, or
        /// invalidated by a reset / free).</summary>
        public static int KdaStateRestore(IntPtr handle) => TSGgml_GlmKdaStateRestore(handle);

        /// <summary>Actual live snapshot arenas, independent of snapshot validity.
        /// Does not include graph workspace. A negative result means unavailable.</summary>
        public static long KdaSnapshotBytes(IntPtr handle) => TSGgml_GlmKdaSnapshotBytes(handle);

        public static bool ResetChecked(IntPtr handle) => TSGgml_GlmResetChecked(handle) != 0;
    }
}
