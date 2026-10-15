// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Diagnostics;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    /// <summary>
    /// Layer-major prefill for host-placed experts.
    ///
    /// When routed experts stream from file for every prefill chunk, a chunk of
    /// a few hundred tokens touches nearly every expert of every layer, so the
    /// ordinary chunk-major order re-reads the whole expert set (tens of GB on
    /// Qwen3.8-Flash-Next) once per chunk. This path instead runs ONE layer over
    /// every chunk of the forward before moving to the next layer, with the
    /// native seam keeping that layer's complete expert stack resident on the
    /// device (TSGgml_HostMoeSetLayerResidency): each layer's experts cross the
    /// disk and the bus once per forward.
    ///
    /// The arithmetic is the token-span kernel's own, one layer per span call.
    /// Chunks keep the admitted VRAM width, so device peaks are unchanged; the
    /// whole forward's residual stays in host memory between layers. Causality
    /// holds because a layer processes its chunks in order: KV rows, GDN/PLE
    /// recurrent state and QSA history advance exactly as in chunk-major order.
    /// </summary>
    public partial class Qwen4ExpModel
    {
        // Graph slot for layer-major spans within the active holder's range.
        // The ordinary token span uses the first slot; this keeps its graph warm.
        private const int LayerMajorSpanSlot = 15;

        private int _layerMajorChunkTokens;

        /// <summary>Admitted per-chunk width for layer-major prefill, or 0 to keep
        /// chunk-major execution. Set by the unified-memory session before a
        /// request's prefill forwards; only forwards longer than one chunk use it.</summary>
        internal int LayerMajorPrefillChunkTokens
        {
            get => _layerMajorChunkTokens;
            set => _layerMajorChunkTokens = value >= 0 ? value : throw new ArgumentOutOfRangeException(nameof(value));
        }

        /// <summary>Minimum chunk width for layer-major prefill. Below it the
        /// routed union of a chunk is a small fraction of the layer, and staging
        /// the complete stack would read more than the ordinary compact path.</summary>
        internal const int LayerMajorMinimumChunkTokens = 64;

        private bool LayerMajorPrefillEligible(int seqLen)
        {
            int chunk = _layerMajorChunkTokens;
            if (chunk < LayerMajorMinimumChunkTokens || seqLen <= chunk) return false;
            if (!IsGgmlBackend || _tokenGraphUnsupported || _specForwardActive || IsTensorParallel
                || LayerSplitDegree > 1 || _pendingMRoPEPositions != null || _visionEmbeddingsList.Count > 0
                || _diagnosticSpanLayers > 0 || _gdnVerify || _fusedGateUpExperts
                || _fusedFfnUnsupported || _fusedGdnUnsupported || _fusedAttnUnsupported
                || !string.IsNullOrEmpty(_driverDump))
                return false;
            bool hostExperts = false;
            if (_expertOnHost != null)
                foreach (bool host in _expertOnHost) hostExperts |= host;
            if (!hostExperts) return false;
            // Growing the KV cache invalidates the attention/QSA descriptors, so
            // grow first (the chunk-major path does the same before its spans).
            EnsureCacheCapacity(_cacheSeqLen + seqLen);
            // Lazily built descriptors; failing any of them keeps chunk-major order.
            if (!EnsureFfnArgs() || !EnsureGdnArgs() || !EnsureAttnArgs() || !EnsureQsaArgs() || !EnsureHeadArgs())
                return false;
            return EnsurePleArgs() || Array.IndexOf(_isPle, true) < 0;
        }

        /// <summary>A forward wider than the admitted chunk width that cannot run
        /// layer-major still runs at that width: text forwards are split into
        /// chunk-major sub-forwards. Media forwards are never granted that width.</summary>
        private float[] ForwardAdmittedChunks(int[] tokens)
        {
            if (_pendingMRoPEPositions != null || _visionEmbeddingsList.Count > 0)
                throw new InvalidOperationException(
                    $"qwen4exp: a {tokens.Length}-token media forward exceeds the admitted {_layerMajorChunkTokens}-token prefill width.");
            float[] logits = null;
            for (int offset = 0; offset < tokens.Length; offset += _layerMajorChunkTokens)
                logits = ForwardCoreInner(tokens.AsSpan(offset, Math.Min(_layerMajorChunkTokens, tokens.Length - offset)).ToArray());
            return logits;
        }

        private unsafe float[] ForwardLayerMajor(int[] tokens)
        {
            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            int chunk = _layerMajorChunkTokens;
            int chunks = (seqLen + chunk - 1) / chunk;
            long tBegin = Stopwatch.GetTimestamp();

            if (_layerKinds == null)
            {
                _layerKinds = GC.AllocateArray<byte>(Config.NumLayers, pinned: true);
                for (int l = 0; l < Config.NumLayers; l++)
                    _layerKinds[l] = _isRecurrent[l] ? (byte)1 : (byte)0;
            }

            // Capacity was grown, and the descriptors rebuilt, by the eligibility check.
            if (_attnArgs == null || _ffnArgs == null || _gdnArgs == null || _headArgs == null)
                throw new InvalidOperationException("qwen4exp layer-major prefill: span descriptors are unavailable.");
            // The complete history is written once; each chunk passes its own
            // prefix length, which is all the kernel reads.
            PrepareQsaHistory(startPos, seqLen);

            Tensor res;
            using (Tensor embd = Embedding(tokens))
                res = BroadcastToStreams(embd, seqLen);
            try
            {
                if (_pleArgs != null)
                {
                    // The n-gram hash advances its history once, in token order.
                    int[] pleRows = ComputePleRows(tokens, startPos);
                    long pleNeed = (long)seqLen * Config.HiddenSize;
                    if (_pleEmbBuf == null || _pleEmbBuf.Length < pleNeed)
                        _pleEmbBuf = GC.AllocateArray<float>(checked((int)pleNeed), pinned: true);
                    fixed (float* eb = _pleEmbBuf)
                        GatherPleRowsRaw(eb, pleRows, seqLen);
                }

                // One causal mask per chunk, at the padded width the kernel reads.
                // Masks of a long continuation can be large: chunks form groups
                // whose masks fit the budget, and each group runs layer-major.
                int firstAttn = Array.IndexOf(_layerKinds, (byte)0);
                DType kvType = firstAttn >= 0 ? _kCache[firstAttn].ElementType : DType.Float16;
                _spanLogitsValid = false;
                float* resBase = GetFloatPtr(res);
                GgmlBasicOps.SetHostMoeLayerResidency(true);
                try
                {
                    for (int c0 = 0; c0 < chunks;)
                    {
                        var masks = new System.Collections.Generic.List<ushort[]>();
                        long maskBytes = 0;
                        int c1 = c0;
                        do
                        {
                            int pos = startPos + c1 * chunk, rows = Math.Min(chunk, seqLen - c1 * chunk);
                            if (masks.Count > 0 && maskBytes + 2L * LayerMajorMaskWidth(pos + rows, kvType) * rows
                                > Qwen4ExpUnifiedGeometry.LayerMajorMaskBudgetBytes)
                                break;
                            int padded = BuildAttnMask(pos + rows, rows, pos, kvType);
                            var mask = GC.AllocateArray<ushort>(checked(padded * rows), pinned: true);
                            Array.Copy(_attnMask, mask, mask.Length);
                            masks.Add(mask);
                            maskBytes += 2L * mask.Length;
                            c1++;
                        } while (c1 < chunks);
                        RunLayerMajorGroup(resBase, startPos, seqLen, chunk, c0, c1, masks, c1 == chunks);
                        c0 = c1;
                    }
                }
                finally
                {
                    GgmlBasicOps.SetHostMoeLayerResidency(false);
                }
                InvalidateTensorDeviceCache(res);
                _kvCacheHostStale = true;
                _deviceStateAuthoritative = true;
                if (!_spanLogitsValid)
                    throw new InvalidOperationException("qwen4exp layer-major prefill produced no logits.");
            }
            finally
            {
                res.Dispose();
            }

            if (LayerMajorLog)
                Console.Error.WriteLine($"[q4e-layer-major] T={seqLen} chunks={chunks}x{chunk} pos={startPos} " +
                    $"{(Stopwatch.GetTimestamp() - tBegin) * 1000.0 / Stopwatch.Frequency:F0} ms");
            // The forward-width PLE rows are not a decode-time buffer.
            if (_pleEmbBuf != null && _pleEmbBuf.Length > (long)chunk * Config.HiddenSize) _pleEmbBuf = null;
            _logitsBuffer = _spanLogits;
            _cacheSeqLen += seqLen;
            _forwardCount++;
            _forwardSw.Stop();
            ReportQ4eProfile(seqLen);
            return _logitsBuffer;
        }

        private int LayerMajorMaskWidth(int totalLen, DType kvType)
        {
            if (!UseFlashAttn(kvType, Config.HeadDim)) return totalLen;
            int padded = (totalLen + KvStride - 1) / KvStride * KvStride;
            return Math.Min(padded, _kvCacheCapacity);
        }

        /// <summary>Run chunks [c0, c1) through every layer, one layer at a time.
        /// The head runs only for the forward's final chunk.</summary>
        private unsafe void RunLayerMajorGroup(float* resBase, int startPos, int seqLen, int chunk,
            int c0, int c1, System.Collections.Generic.List<ushort[]> masks, bool finalGroup)
        {
            int slot = _seqSlotBase + LayerMajorSpanSlot;
            fixed (Qwen4ExpFfnArgs* fp = _ffnArgs)
            fixed (Qwen4ExpGdnArgs* gp = _gdnArgs)
            fixed (Qwen4ExpAttnArgs* ap = _attnArgs)
            fixed (Qwen4ExpQsaArgs* qp = _qsaArgs)
            fixed (int* qpositions = _qsaPositions)
            fixed (byte* kp = _layerKinds)
            fixed (Qwen4ExpHeadArgs* hp = _headArgs)
            fixed (float* lp = _spanLogits)
            fixed (Qwen4ExpPleArgs* pp = _pleArgs)
            fixed (float* eb = _pleEmbBuf)
            {
                for (int il = 0; il < Config.NumLayers; il++)
                {
                    long tLayer = Stopwatch.GetTimestamp();
                    bool pleHere = _pleArgs != null && _pleLayerIndex == il;
                    HintNextHostExpertLayer(il);
                    for (int c = c0; c < c1; c++)
                    {
                        int row = c * chunk, rows = Math.Min(chunk, seqLen - row), pos = startPos + row;
                        // Only the forward's final row needs logits.
                        bool head = finalGroup && il == Config.NumLayers - 1 && c == c1 - 1;
                        fixed (ushort* mp = masks[c - c0])
                        {
                            bool ok = ExecuteQwen4ExpSpan(
                                (IntPtr)fp, (IntPtr)gp, (IntPtr)ap, (IntPtr)kp,
                                il, il + 1,
                                (IntPtr)(resBase + (long)row * _hcDim), (IntPtr)mp,
                                Config.HiddenSize, _hc, _hcLowRank, rows,
                                _headKDim, _headVDim, _numKHeads, _numVHeads, _convKernel,
                                Config.HeadDim, Config.NumHeads, Config.NumKVHeads,
                                _kvCacheCapacity, pos + rows, pos,
                                _ropeDimCount, Config.RopeBase, 1.0f / Config.RopeScale, _attnScale,
                                _numExperts, _numExpertsUsed, _expertFf, _sharedFf,
                                Config.Eps, cacheSlot: slot,
                                head: head ? (IntPtr)hp : IntPtr.Zero,
                                logitsOut: head ? (IntPtr)lp : IntPtr.Zero,
                                ple: pleHere ? (IntPtr)pp : IntPtr.Zero,
                                pleLayer: pleHere ? _pleLayerIndex : -1,
                                pleEmb: pleHere ? (IntPtr)(eb + (long)row * Config.HiddenSize) : IntPtr.Zero,
                                ropePosition: pos - _mropeCacheGap,
                                device: DeviceForLayer(il),
                                qsa: (IntPtr)qp, qsaPositions: (IntPtr)qpositions, qsaPositionCount: pos + rows);
                            if (!ok)
                            {
                                // Earlier layers/chunks already advanced KV and
                                // recurrent state; no fallback can replay them.
                                _tokenGraphUnsupported = true;
                                throw new InvalidOperationException(
                                    $"qwen4exp layer-major prefill failed at layer {il}, chunk {c}: " + GgmlBasicOps.LastNativeError());
                            }
                            if (head) _spanLogitsValid = true;
                        }
                    }
                    PhaseLog(seqLen, $"layer-major[{il}] x{c1 - c0}", tLayer);
                }
            }
        }

        /// <summary>Tell the native seam which host-expert layer runs after
        /// <paramref name="il"/>, so its experts are read while this layer computes.
        /// The final layer of a group hints nothing: the next group restarts at 0.</summary>
        private void HintNextHostExpertLayer(int il)
        {
            for (int next = il + 1; next < Config.NumLayers; next++)
            {
                if (!IsExpertLayerOnHost(next)) continue;
                ref var a = ref _ffnArgs[next];
                GgmlBasicOps.SetHostMoeLayerResidencyHint(a.GateExps, a.UpExps, a.DownExps,
                    a.GateExpsBytes, a.UpExpsBytes, a.DownExpsBytes, _numExperts);
                return;
            }
            GgmlBasicOps.SetHostMoeLayerResidencyHint(IntPtr.Zero, IntPtr.Zero, IntPtr.Zero, 0, 0, 0, 0);
        }

        private static readonly bool LayerMajorLog =
            string.Equals(Environment.GetEnvironmentVariable("TS_Q4E_LAYER_MAJOR_LOG"), "1", StringComparison.Ordinal);
    }
}
