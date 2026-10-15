// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.MLX;
using TensorSharp.Cuda;

namespace TensorSharp.Models
{
    public partial class Gemma4VisionEncoder : IDisposable
    {
        /// <summary>
        /// Tower geometry, resolved from either an mmproj GGUF's metadata or (for a raw HF shard,
        /// which has no metadata block) from the tensor shapes themselves.
        /// </summary>
        private sealed class TowerSpec
        {
            public int HiddenSize;
            public int IntermediateSize;
            public int NumHeads;
            public int BlockCount;
            public float Eps;
            public int ProjectionDim;
            public int PatchSize;
            public int NMerge;
            public float RopeTheta;
            public string ProjectorType = "gemma4v";
            public float[] ImageMean = Array.Empty<float>();
            public float[] ImageStd = Array.Empty<float>();
        }

        private readonly Dictionary<string, Tensor> _weights = new();
        private readonly Dictionary<string, Tensor> _transposedWeights = new();
        private sealed class CompressedMatrix
        {
            private GCHandle _pinned;
            public readonly int Input, OriginalInput, Output, Bytes;
            public IntPtr Pointer => _pinned.AddrOfPinnedObject();
            public CompressedMatrix(byte[] bytes, int input, int output)
            {
                OriginalInput = input; Input = input; Output = output;
                if (input % 64 == 0)
                {
                    // A zero Q8_0 block preserves every dot product. The 32
                    // extra channels select unchanged MLX's F32 SIMD QMM
                    // instead of its M5 TF32 NAX path (K must be %64 there).
                    int rowBytes = checked(input / 32 * 34);
                    if (bytes.Length != checked(rowBytes * output)) throw new ArgumentException("Invalid Q8_0 vision row size.");
                    var padded = new byte[checked((rowBytes + 34) * output)];
                    for (int row = 0; row < output; ++row)
                        Buffer.BlockCopy(bytes, row * rowBytes, padded, row * (rowBytes + 34), rowBytes);
                    bytes = padded; Input = checked(input + 32);
                }
                Bytes = bytes.Length; _pinned = GCHandle.Alloc(bytes, GCHandleType.Pinned);
            }
            public void Free() { if (_pinned.IsAllocated) _pinned.Free(); }
        }
        private readonly Dictionary<string, CompressedMatrix> _compressedMatrices = new();
        private readonly IAllocator _allocator;
        private readonly bool _useNativeAttention;
        private readonly bool _useMlxDeviceOps;
        private readonly bool _useCudaDeviceOps;
        private readonly bool _usePreciseMetalVision;
        private readonly bool _requirePreciseGgmlCudaVision;
        private readonly bool _recenterPixels;
        // Optional ModelBase reference for cooperative GpuComputeLock yielding.
        // When set, Encode() releases the lock between encoder blocks so the
        // engine worker can run one inference step per block — keeping
        // concurrent decode requests responsive during long image encodes.
        private ModelBase _hostModel;

        private readonly int _hiddenSize;
        private readonly int _intermediateSize;
        private readonly int _numHeads;
        private readonly int _blockCount;
        private readonly float _eps;
        private readonly int _projectionDim;
        private readonly int _patchSize;
        private readonly int _nMerge;
        private readonly float _ropeTheta;

        // "gemma4uv" (Gemma 4 unified vision embedder, e.g. Gemma-4-12B) uses a
        // completely different, block-less vision path: a single large-patch
        // conv (patch_size * n_merge) followed by three pytorch LayerNorms,
        // learned 2D positional embeddings, an unweighted RMSNorm and a linear
        // projection. The classic "gemma4v" path (SigLIP transformer with
        // v.blk.N.* blocks, e.g. Gemma-4-E4B) keeps the original encode flow.
        private readonly bool _isUnified;
        private readonly string _projectorType;
        // image_mean / image_std read from the mmproj. The unified embedder is
        // trained on raw [0,1] pixels (mean=0, std=1); the SigLIP path keeps the
        // SigLIP [-1,1] preprocessing.
        private readonly float[] _imageMean;
        private readonly float[] _imageStd;

        // pytorch nn.LayerNorm default epsilon used by the unified embedder.
        private const float UnifiedLayerNormEps = 1e-5f;

        private struct ClampParams
        {
            public float InMin, InMax, OutMin, OutMax;
            public bool HasClamp;
        }

        private readonly Dictionary<string, ClampParams> _clampParams = new();
        private readonly Dictionary<long, Rope2DCache> _ropeCache = new();
        private Tensor _onesForNorm;

        /// <summary>
        /// Per-geometry 2D rope tables. The fused native block takes only PosX/PosY and
        /// re-derives the angles on-device (ggml_rope_ext with the same rope_theta); the
        /// precomputed Cos/Sin arrays are what the managed per-op fallback
        /// (<see cref="Apply2DRoPE"/>) walks. The two must stay in lockstep: both rotate
        /// head dims [0, hd/2) by the patch X index and [hd/2, hd) by its Y index, pairing
        /// element j with element j + hd/4 at angle pos * ropeTheta^(-2j / (hd/2)).
        /// </summary>
        private sealed class Rope2DCache
        {
            public required int[] PosX { get; init; }
            public required int[] PosY { get; init; }
            public required float[] CosX { get; init; }
            public required float[] SinX { get; init; }
            public required float[] CosY { get; init; }
            public required float[] SinY { get; init; }
            public Tensor DeviceCosine;
            public Tensor DeviceSine;
        }

        public int ProjectionDim => _projectionDim;

        /// <summary>True when this mmproj is a Gemma 4 "unified" vision embedder
        /// (projector_type "gemma4uv"), which uses the block-less encode path.</summary>
        public bool IsUnified => _isUnified;

        /// <summary>Per-channel image mean read from the mmproj (or null).</summary>
        public float[] ImageMean => _imageMean;

        /// <summary>Per-channel image std read from the mmproj (or null).</summary>
        public float[] ImageStd => _imageStd;

        /// <summary>Attach the model that owns this encoder so the per-block
        /// loop in <see cref="Encode"/> can yield the GPU compute lock between
        /// blocks. Set once after construction.</summary>
        public void SetHostModel(ModelBase model) => _hostModel = model;

        /// <param name="projectorPath">
        /// An mmproj GGUF, or a HuggingFace <c>.safetensors</c> shard holding a
        /// <c>model.encoder.vision_tower.*</c> tower. The safetensors form exists because some
        /// published GGUF conversions drop the vision tower entirely (see
        /// Gemma4VisionEncoder.Safetensors.cs).
        /// </param>
        public Gemma4VisionEncoder(string projectorPath, IAllocator allocator)
            : this(projectorPath, allocator, inputPixelsInZeroToOneRange: false) { }

        /// <param name="inputPixelsInZeroToOneRange">For the classic tower,
        /// recenter raw [0,1] pixels to [-1,1] inside patch embedding. Existing
        /// callers that already supply SigLIP-normalized pixels leave this false.
        /// The unified tower always consumes raw pixels.</param>
        /// <remarks>The verified 768-wide, 16-block EmbeddingGemma 2 Q8 GGUF tower
        /// defaults to MLX padded F32 device math. Its legacy generic M5 route
        /// produced raw-row max error 0.01887 against the independent full-tower
        /// oracle (bound 0.01). CUDA uses its qualified F32 device path and Metal
        /// uses F32 learned projections. Each backend has a diagnostic opt-out.
        /// This precision choice is not a claim
        /// of performance parity with llama.cpp.</remarks>
        public Gemma4VisionEncoder(string projectorPath, IAllocator allocator, bool inputPixelsInZeroToOneRange)
            : this(projectorPath, allocator, inputPixelsInZeroToOneRange, requireValidatedFloat32Precision: false) { }

        /// <summary>The embedding constructor requires scoped precise CUDA
        /// blocks for the independently qualified Q8 tower. Existing chat
        /// constructors retain their original backend arithmetic.</summary>
        public Gemma4VisionEncoder(string projectorPath, IAllocator allocator, bool inputPixelsInZeroToOneRange,
            bool requireValidatedFloat32Precision)
        {
            _allocator = allocator;
            _recenterPixels = inputPixelsInZeroToOneRange;

            GgufFile gguf = null;
            SafetensorsFile safetensors = null;
            TowerSpec spec;
            if (IsSafetensorsProjector(projectorPath))
            {
                safetensors = new SafetensorsFile(projectorPath);
                spec = ReadSafetensorsSpec(safetensors, projectorPath);
            }
            else
            {
                gguf = new GgufFile(projectorPath);
                spec = ReadGgufSpec(gguf);
            }

            _hiddenSize = spec.HiddenSize;
            _intermediateSize = spec.IntermediateSize;
            _numHeads = spec.NumHeads;
            // The validated EmbeddingGemma 2 Q8 tower defaults to padded F32
            // projection/attention math. Generic M5 NAX F32 dispatch narrows
            // products to TF32 and failed the full raw-row quality gate.
            // DEVICE=0 retains that legacy route for diagnostics; DEVICE=1
            // keeps the existing explicit experiment on other geometries.
            string deviceSetting = Environment.GetEnvironmentVariable("TS_GEMMA4_VISION_DEVICE");
            _useMlxDeviceOps = allocator is MlxAllocator && (deviceSetting == "1" ||
                deviceSetting != "0" && IsValidatedQ8VisionTower(spec, gguf));
            string cudaSetting = Environment.GetEnvironmentVariable("TS_GEMMA4_VISION_CUDA_DEVICE");
            _useCudaDeviceOps = allocator is CudaAllocator && (cudaSetting == "1" ||
                cudaSetting != "0" && IsValidatedQ8VisionTower(spec, gguf));
            string metalSetting = Environment.GetEnvironmentVariable("TS_GEMMA4_VISION_METAL_PRECISE");
            _usePreciseMetalVision = allocator is GgmlAllocator metalAllocator &&
                metalAllocator.Context.BackendType == GgmlBackendType.Metal &&
                (metalSetting == "1" || metalSetting != "0" && IsValidatedQ8VisionTower(spec, gguf));
            _requirePreciseGgmlCudaVision = requireValidatedFloat32Precision && allocator is GgmlAllocator cudaAllocator &&
                cudaAllocator.Context.BackendType == GgmlBackendType.Cuda;
            if (_requirePreciseGgmlCudaVision && !IsValidatedQ8VisionTower(spec, gguf))
            {
                gguf?.Dispose(); safetensors?.Dispose();
                throw new NotSupportedException("Validated GGML CUDA vision precision requires the qualified EmbeddingGemma 2 Q8 tower.");
            }
            // The EmbeddingGemma 2 tower's 64-channel heads use MLX's F32 tiled
            // attention. Other geometries retain their existing attention path.
            _useNativeAttention = allocator is GgmlAllocator ||
                ((_useMlxDeviceOps || _useCudaDeviceOps) && _hiddenSize / _numHeads == 64);
            _blockCount = spec.BlockCount;
            _eps = spec.Eps;
            _projectionDim = spec.ProjectionDim;
            _patchSize = spec.PatchSize;
            _nMerge = spec.NMerge;
            _ropeTheta = spec.RopeTheta;
            _projectorType = spec.ProjectorType;
            _isUnified = string.Equals(_projectorType, "gemma4uv", StringComparison.Ordinal);
            _imageMean = spec.ImageMean;
            _imageStd = spec.ImageStd;

            Console.WriteLine($"Vision encoder: type={_projectorType}, hidden={_hiddenSize}, " +
                $"intermediate={_intermediateSize}, heads={_numHeads}, blocks={_blockCount}, " +
                $"projDim={_projectionDim}, patch={_patchSize}, nMerge={_nMerge}" +
                (_isUnified ? $", unifiedPatch={_patchSize * _nMerge}" : string.Empty) +
                (safetensors != null ? ", source=safetensors" : string.Empty));

            try
            {
                InitializeCpuVisionProjection(spec, gguf);
                if (safetensors != null) LoadWeightsFromSafetensors(safetensors);
                else LoadWeights(gguf);
            }
            catch
            {
                // Failed construction must release pinned compressed weights
                // and any device cache entries already acquired by this tower.
                Dispose();
                throw;
            }
            finally { safetensors?.Dispose(); gguf?.Dispose(); }
        }

        private static bool IsValidatedQ8VisionTower(TowerSpec spec, GgufFile gguf)
        {
            if (gguf == null || spec.ProjectorType != "gemma4v" || spec.HiddenSize != 768 ||
                spec.IntermediateSize != 3072 || spec.NumHeads != 12 || spec.BlockCount != 16 ||
                spec.ProjectionDim != 512 || spec.PatchSize != 16 || spec.NMerge != 3) return false;
            bool Matrix(string name, int input, int output)
                => gguf.Tensors.TryGetValue(name, out var tensor) && tensor.Type == GgmlTensorType.Q8_0 &&
                    tensor.Shape.Length == 2 && tensor.Shape[0] == (ulong)input && tensor.Shape[1] == (ulong)output;
            if (!Matrix("mm.input_projection.weight", 768, 512)) return false;
            if (!gguf.Tensors.TryGetValue("v.patch_embd.weight", out var patch) || patch.Type != GgmlTensorType.F32 ||
                patch.Shape.Length != 4 || patch.Shape[0] != 16 || patch.Shape[1] != 16 || patch.Shape[2] != 3 || patch.Shape[3] != 768) return false;
            for (int block = 0; block < 16; ++block)
            {
                string p = $"v.blk.{block}.";
                foreach (string name in new[] { "attn_q", "attn_k", "attn_v", "attn_out" })
                    if (!Matrix(p + name + ".weight", 768, 768)) return false;
                if (!Matrix(p + "ffn_gate.weight", 768, 3072) || !Matrix(p + "ffn_up.weight", 768, 3072) ||
                    !Matrix(p + "ffn_down.weight", 3072, 768)) return false;
            }
            return true;
        }

        private static TowerSpec ReadGgufSpec(GgufFile gguf)
        {
            int nMerge = (int)gguf.GetUint32("clip.vision.projector_scale_factor", 0);
            if (nMerge == 0)
            {
                // llama.cpp's canonical spelling (clip-impl.h KEY_PROJ_SCALE_FACTOR). Accept both so
                // an mmproj produced by either toolchain resolves the pooling factor explicitly
                // instead of silently landing on the default.
                nMerge = (int)gguf.GetUint32("clip.vision.projector.scale_factor", 0);
            }
            if (nMerge == 0) nMerge = 3;

            string projectorType = gguf.GetString("clip.vision.projector_type", "gemma4v") ?? "gemma4v";
            return new TowerSpec
            {
                HiddenSize = (int)gguf.GetUint32("clip.vision.embedding_length", 768),
                IntermediateSize = (int)gguf.GetUint32("clip.vision.feed_forward_length", 3072),
                NumHeads = (int)gguf.GetUint32("clip.vision.attention.head_count", 12),
                BlockCount = (int)gguf.GetUint32("clip.vision.block_count", 16),
                Eps = gguf.GetFloat32("clip.vision.attention.layer_norm_epsilon", 1e-6f),
                ProjectionDim = (int)gguf.GetUint32("clip.vision.projection_dim", 2560),
                PatchSize = (int)gguf.GetUint32("clip.vision.patch_size", 16),
                NMerge = nMerge,
                RopeTheta = 100f,
                ProjectorType = projectorType,
                ImageMean = gguf.GetFloatArray("clip.vision.image_mean"),
                ImageStd = gguf.GetFloatArray("clip.vision.image_std"),
            };
        }

        private void LoadWeights(GgufFile gguf)
        {
            Console.Write("Loading vision encoder weights...");
            int count = 0;
            foreach (var kv in gguf.Tensors)
            {
                var info = kv.Value;
                if (!info.Name.StartsWith("v.") && !info.Name.StartsWith("mm.input_projection"))
                    continue;

                byte[] raw = gguf.ReadTensorData(info);
                long numElements = info.NumElements;
                float[] f32 = new float[numElements];

                if (info.Type == GgmlTensorType.F32)
                    Buffer.BlockCopy(raw, 0, f32, 0, raw.Length);
                else if (_allocator is GgmlAllocator)
                    NativeDequant.DequantizeToFloat32((int)info.Type, raw, 0, f32, 0, numElements);
                else
                    ManagedQuantizedOps.DequantizeToFloat32((int)info.Type, raw, 0, f32, 0, numElements);

                long[] ggufShape = new long[info.Shape.Length];
                for (int i = 0; i < info.Shape.Length; i++)
                    ggufShape[i] = (long)info.Shape[i];

                long[] tsShape = new long[ggufShape.Length];
                for (int i = 0; i < ggufShape.Length; i++)
                    tsShape[i] = ggufShape[ggufShape.Length - 1 - i];

                var tensor = new Tensor(_allocator, DType.Float32, tsShape);
                tensor.SetElementsAsFloat(f32);
                _weights[info.Name] = tensor;
                RetainCpuVisionProjection(info.Name, raw, info.Type, tsShape);
                if (_useMlxDeviceOps && info.Type == GgmlTensorType.Q8_0 &&
                    (info.Shape.Length == 2 || info.Name == "v.patch_embd.weight" && info.Shape.Length == 4))
                {
                    int output = checked((int)info.Shape[^1]);
                    int input = checked((int)(numElements / output));
                    if (input % 32 == 0) _compressedMatrices.Add(info.Name, new CompressedMatrix(raw, input, output));
                }
                count++;

                if (info.Name.Contains("input_min") || info.Name.Contains("input_max") ||
                    info.Name.Contains("output_min") || info.Name.Contains("output_max"))
                {
                    string linearKey = info.Name.Substring(0, info.Name.LastIndexOf('.'));
                    if (!_clampParams.ContainsKey(linearKey))
                        _clampParams[linearKey] = new ClampParams
                        {
                            InMin = float.MinValue, InMax = float.MaxValue,
                            OutMin = float.MinValue, OutMax = float.MaxValue,
                            HasClamp = false
                        };

                    var cp = _clampParams[linearKey];
                    cp.HasClamp = true;
                    if (info.Name.EndsWith("input_min")) cp.InMin = f32[0];
                    else if (info.Name.EndsWith("input_max")) cp.InMax = f32[0];
                    else if (info.Name.EndsWith("output_min")) cp.OutMin = f32[0];
                    else if (info.Name.EndsWith("output_max")) cp.OutMax = f32[0];
                    _clampParams[linearKey] = cp;
                }
            }
            Console.WriteLine($" done ({count} tensors, {_clampParams.Count} clampable linears)");
            ModelBase.LogCudaVram(_allocator, "after vision encoder weight load");
        }

        public unsafe Tensor Encode(float[] pixelValues, int imgWidth, int imgHeight)
        {
            // Host-model callers yield their GPU lock and retain the prior owner contract.
            if (_requirePreciseGgmlCudaVision && _hostModel == null)
            {
                lock (_ggmlVisionStackSync)
                {
                    ObjectDisposedException.ThrowIf(_ggmlVisionStackDisposed, this);
                    return EncodeCore(pixelValues, imgWidth, imgHeight);
                }
            }
            if (CpuVisionProjectionRequested)
            {
                lock (_cpuVisionProjectionSync)
                {
                    ObjectDisposedException.ThrowIf(_cpuVisionDisposed, this);
                    return EncodeCore(pixelValues, imgWidth, imgHeight);
                }
            }
            return EncodeCore(pixelValues, imgWidth, imgHeight);
        }

        private unsafe Tensor EncodeCore(float[] pixelValues, int imgWidth, int imgHeight)
        {
            if (_allocator is TensorSharp.MLX.MlxAllocator && _hostModel == null &&
                !TensorSharp.MLX.MlxWorker.Shared.IsOnWorkerThread)
                return TensorSharp.MLX.MlxWorker.Shared.Invoke(() => Encode(pixelValues, imgWidth, imgHeight));
            if (_isUnified)
                return EncodeUnified(pixelValues, imgWidth, imgHeight);

            int patchesX = imgWidth / _patchSize;
            int patchesY = imgHeight / _patchSize;
            int numPatches = patchesX * patchesY;
            int headDim = _hiddenSize / _numHeads;
            Rope2DCache ropeCache = GetOrCreateRopeCache(patchesX, patchesY, headDim);


            // TS_VISION_STAGE=1: per-stage wall clock.
            var stageSw = VisionStageEnabled ? System.Diagnostics.Stopwatch.StartNew() : null;
            double tPatch = 0, tPos = 0, tBlocks = 0, tPool = 0;

            var hidden = PatchEmbed(pixelValues, imgWidth, imgHeight, patchesX, patchesY);
            if (stageSw != null) { tPatch = stageSw.Elapsed.TotalMilliseconds; stageSw.Restart(); }
            VisionTrace("patch_embed", hidden);
            AddPositionEmbedding2D(hidden, ropeCache, numPatches);
            if (stageSw != null) { tPos = stageSw.Elapsed.TotalMilliseconds; stageSw.Restart(); }
            VisionTrace("pos_embed", hidden);

            bool stackDone = TryGgmlVisionStack(ref hidden, numPatches, ropeCache);
            if (!stackDone) for (int i = 0; i < _blockCount; i++)
            {
                if (!VisionStageEnabled)
                    Console.Write($"\r  Vision encoder block {i + 1}/{_blockCount}...");
                hidden = EncoderBlock(hidden, i, numPatches, headDim, ropeCache);
                VisionTrace($"block{i}", hidden);
                // Yield the GPU compute lock between encoder blocks so the
                // engine worker can run an inference step. Without this,
                // a single image encode (16–32 blocks, 100ms–2s+ total)
                // freezes every other in-flight chat/decode request for
                // its entire duration; with this, each yield admits one
                // engine step (≈50–200ms), interleaving encoder progress
                // with inference progress. No-op when the encoder is
                // running outside a GpuComputeLock scope.
                _hostModel?.YieldGpuComputeLock();
            }
            if (stageSw != null) { tBlocks = stageSw.Elapsed.TotalMilliseconds; stageSw.Restart(); }
            if (!VisionStageEnabled) Console.WriteLine(" done");

            var projected = PoolAndProject(hidden, patchesX, patchesY, numPatches);
            if (stageSw != null)
            {
                tPool = stageSw.Elapsed.TotalMilliseconds;
                Console.Error.WriteLine(
                    $"[vision-stage] patches={numPatches} patch_embed={tPatch:F1}ms pos_embed={tPos:F1}ms " +
                    $"blocks({_blockCount})={tBlocks:F1}ms pool+project={tPool:F1}ms " +
                    $"total={tPatch + tPos + tBlocks + tPool:F1}ms");
            }
            VisionTrace("projected", projected);
            hidden.Dispose();


            return projected;
        }

        // TS_VISION_STAGE=1: per-stage wall clock for the encode (patch embed / position
        // embed / the 16 fused blocks / pool+project). Inert unless set.
        private static readonly bool VisionStageEnabled =
            Environment.GetEnvironmentVariable("TS_VISION_STAGE") is string sv && sv.Length > 0 && sv != "0";

        // TS_VISION_TRACE=1: mean / rms / min / max of a stage's activations.
        // The encoder is backend-agnostic C# over Ops, so when one backend reads
        // an image correctly and another hallucinates, the divergence is in a
        // single op and this is what finds which one.
        private static readonly bool VisionTraceEnabled =
            Environment.GetEnvironmentVariable("TS_VISION_TRACE") is string v && v.Length > 0 && v != "0";

        private static unsafe void VisionTrace(string stage, Tensor t)
        {
            if (!VisionTraceEnabled || t == null) return;
            long n = 1;
            for (int i = 0; i < t.Sizes.Length; i++) n *= t.Sizes[i];
            float* p = GetFloatPtr(t);
            double sum = 0, sumSq = 0;
            float lo = float.MaxValue, hi = float.MinValue;
            int nan = 0;
            for (long i = 0; i < n; i++)
            {
                float x = p[i];
                if (float.IsNaN(x) || float.IsInfinity(x)) { nan++; continue; }
                sum += x; sumSq += (double)x * x;
                if (x < lo) lo = x;
                if (x > hi) hi = x;
            }
            Console.WriteLine($"[vision-trace] {stage,-12} n={n} mean={sum / n:F6} rms={Math.Sqrt(sumSq / n):F6} " +
                              $"min={lo:F4} max={hi:F4} nonfinite={nan}");
        }

        /// <summary>
        /// Encode path for the Gemma 4 "unified" vision embedder (projector_type
        /// "gemma4uv", used by Gemma-4-12B). Mirrors llama.cpp's
        /// clip_graph_gemma4uv::build():
        ///   im2col(patch = patch_size * n_merge) -> LayerNorm(patch_norm_1)
        ///   -> patch_embd matmul + bias -> LayerNorm(patch_norm_2)
        ///   -> + 2D learned position embeddings -> LayerNorm(patch_norm_3 / pos_norm)
        ///   -> unweighted RMSNorm -> linear projection to the text embedding dim.
        /// There are no transformer blocks (clip.vision.block_count == 0).
        /// </summary>
        private unsafe Tensor EncodeUnified(float[] pixelValues, int imgWidth, int imgHeight)
        {
            // The unified variant folds the n_merge "token merging" directly into
            // a larger conv patch, so the effective patch is patch_size * n_merge
            // (e.g. 16 * 3 = 48) and there is no separate pooling step.
            int P = _patchSize * _nMerge;
            const int C = 3;
            int patchDim = P * P * C;          // e.g. 48 * 48 * 3 = 6912
            int patchesX = imgWidth / P;
            int patchesY = imgHeight / P;
            int numPatches = patchesX * patchesY;

            // 1. im2col: each patch becomes a flat row ordered [c][ky][kx] to match
            //    the converted v.patch_embd.weight layout (same ordering as the
            //    SigLIP PatchEmbed above).
            var cols = new Tensor(_allocator, DType.Float32, numPatches, patchDim);
            float* colsPtr = GetFloatPtr(cols);
            fixed (float* pixSrc = pixelValues)
            {
                long colsPtrL = (long)colsPtr, pixSrcL = (long)pixSrc;
                // Each (c, ky) source span of P pixels is contiguous in kx, so the
                // im2col packing is a parallel set of MemoryCopy row writes.
                Parallel.For(0, patchesY, py =>
                {
                    float* dst = (float*)colsPtrL;
                    float* pix = (float*)pixSrcL;
                    for (int px = 0; px < patchesX; px++)
                    {
                        int patchIdx = py * patchesX + px;
                        float* row = dst + (long)patchIdx * patchDim;
                        for (int c = 0; c < C; c++)
                        {
                            long imgChannelOffset = (long)c * imgHeight * imgWidth;
                            long outChannelOffset = (long)c * P * P;
                            for (int ky = 0; ky < P; ky++)
                            {
                                long srcOffset = imgChannelOffset + (long)(py * P + ky) * imgWidth + px * P;
                                Buffer.MemoryCopy(pix + srcOffset, row + outChannelOffset + (long)ky * P,
                                    P * sizeof(float), P * sizeof(float));
                            }
                        }
                    }
                });
            }

            // 2. patch_norm_1: pytorch LayerNorm over the patch vector.
            Ops.LayerNorm(cols, cols, _weights["v.patch_norm.1.weight"],
                _weights["v.patch_norm.1.bias"], UnifiedLayerNormEps);

            // 3. patch embedding (linear) + bias -> [numPatches, hiddenSize].
            var hidden = new Tensor(_allocator, DType.Float32, numPatches, _hiddenSize);
            Ops.Addmm(hidden, 0, hidden, 1f, cols, GetOrCreateTransposedWeight("v.patch_embd.weight"));
            cols.Dispose();
            AddBiasInPlace(hidden, _weights["v.patch_embd.bias"], numPatches, _hiddenSize);

            // 4. patch_norm_2: LayerNorm over the embedding dim.
            Ops.LayerNorm(hidden, hidden, _weights["v.patch_norm.2.weight"],
                _weights["v.patch_norm.2.bias"], UnifiedLayerNormEps);

            // 5. add learned 2D (x, y) position embeddings.
            AddUnifiedPositionEmbedding(hidden, patchesX, numPatches);

            // 6. patch_norm_3 (pos_norm): LayerNorm over the embedding dim.
            Ops.LayerNorm(hidden, hidden, _weights["v.patch_norm.3.weight"],
                _weights["v.patch_norm.3.bias"], UnifiedLayerNormEps);

            // 7. embedding_pre_projection_norm: unweighted RMSNorm (eps = hparams.eps).
            ApplyUnweightedRMSNorm(hidden, numPatches, _hiddenSize);

            // 8. project to the text embedding dimension.
            var projected = LinearProjection(hidden, "mm.input_projection.weight");
            hidden.Dispose();

            return projected;
        }

        private unsafe void AddBiasInPlace(Tensor t, Tensor bias, int rows, int cols)
        {
            float* p = GetFloatPtr(t);
            float* b = GetFloatPtr(bias);
            int vLen = Vector<float>.Count;
            long pL = (long)p, bL = (long)b;
            Parallel.For(0, rows, r =>
            {
                float* row = (float*)pL + (long)r * cols;
                float* bias2 = (float*)bL;
                int d = 0;
                for (; d <= cols - vLen; d += vLen)
                    TensorComputePrimitives.StoreVector(row + d,
                        TensorComputePrimitives.LoadVector(row + d)
                        + TensorComputePrimitives.LoadVector(bias2 + d));
                for (; d < cols; d++)
                    row[d] += bias2[d];
            });
        }

        private unsafe void AddUnifiedPositionEmbedding(Tensor hidden, int patchesX, int numPatches)
        {
            // v.position_embd.weight is stored as two stacked lookup tables
            // (x then y), each [maxPos, hiddenSize]; pos_x = patch % cols,
            // pos_y = patch / cols.
            var posEmbd = _weights["v.position_embd.weight"];
            int maxPos = (int)posEmbd.Sizes[1];
            float* posPtr = GetFloatPtr(posEmbd);
            float* xTable = posPtr;
            float* yTable = posPtr + (long)maxPos * _hiddenSize;
            float* dstPtr = GetFloatPtr(hidden);
            int hiddenSize = _hiddenSize;
            int vLen = Vector<float>.Count;
            long dstPtrL = (long)dstPtr, xTableL = (long)xTable, yTableL = (long)yTable;

            Parallel.For(0, numPatches, p =>
            {
                int x = p % patchesX;
                int y = p / patchesX;
                float* dstRow = (float*)dstPtrL + (long)p * hiddenSize;
                float* xRow = (float*)xTableL + (long)x * hiddenSize;
                float* yRow = (float*)yTableL + (long)y * hiddenSize;
                int d = 0;
                for (; d <= hiddenSize - vLen; d += vLen)
                {
                    var acc = TensorComputePrimitives.LoadVector(dstRow + d)
                            + TensorComputePrimitives.LoadVector(xRow + d)
                            + TensorComputePrimitives.LoadVector(yRow + d);
                    TensorComputePrimitives.StoreVector(dstRow + d, acc);
                }
                for (; d < hiddenSize; d++)
                    dstRow[d] += xRow[d] + yRow[d];
            });
        }

        // Patch embedding (Conv2D, stride = kernel = patch_size) reformulated as
        // im2col + GEMM, matching llama.cpp's ggml_conv_2d and the already-optimized
        // Qwen35 path. The per-patch im2col packing is parallelised with SIMD row
        // copies; the heavy compute becomes a single matmul that the backend
        // (CPU BLAS or GPU) handles efficiently, replacing the original
        // single-threaded scalar quintuple loop.
        private unsafe Tensor PatchEmbed(float[] pixelValues, int imgW, int imgH, int patchesX, int patchesY)
        {
            int numPatches = patchesX * patchesY;
            int C = 3, P = _patchSize;
            int patchStride = C * P * P;

            // v.patch_embd.weight post-load is [hiddenSize, C, P, P] contiguous in
            // [f][c][ky][kx] order — exactly the im2col row ordering below — so it
            // views directly as a [hiddenSize, patchStride] matmul matrix.

            var im2col = new Tensor(_allocator, DType.Float32, numPatches, patchStride);
            float* im2colPtr = GetFloatPtr(im2col);

            fixed (float* pixSrc = pixelValues)
            {
                long pixSrcL = (long)pixSrc;
                Parallel.For(0, patchesY, py =>
                {
                    float* pix = (float*)pixSrcL;
                    for (int px = 0; px < patchesX; px++)
                    {
                        int patchIdx = py * patchesX + px;
                        float* outRow = im2colPtr + (long)patchIdx * patchStride;
                        int yBase = py * P;
                        int xBase = px * P;
                        for (int c = 0; c < C; c++)
                        {
                            long imgChannelOffset = (long)c * imgH * imgW;
                            long outChannelOffset = (long)c * P * P;
                            for (int ky = 0; ky < P; ky++)
                            {
                                long srcOffset = imgChannelOffset + (long)(yBase + ky) * imgW + xBase;
                                long dstOffset = outChannelOffset + (long)ky * P;
                                if (_recenterPixels)
                                    for (int kx = 0; kx < P; ++kx)
                                        outRow[dstOffset + kx] = 2f * pix[srcOffset + kx] - 1f;
                                else
                                    Buffer.MemoryCopy(pix + srcOffset, outRow + dstOffset,
                                        P * sizeof(float), P * sizeof(float));
                            }
                        }
                    }
                });
            }

            var result = new Tensor(_allocator, DType.Float32, numPatches, _hiddenSize);
            if (!TryCompressedProjection(result, im2col, "v.patch_embd.weight"))
                Ops.Addmm(result, 0, result, 1.0f, im2col, GetOrCreatePatchEmbedTransposed(patchStride));
            im2col.Dispose();
            return result;
        }

        // Cache the transposed [patchStride, hiddenSize] patch-embed weight used by
        // the im2col GEMM. v.patch_embd.weight is already contiguous as
        // [hiddenSize, patchStride], so we transpose once.
        private Tensor GetOrCreatePatchEmbedTransposed(int patchStride)
        {
            const string key = "v.patch_embd.weight.2d.T";
            if (_transposedWeights.TryGetValue(key, out var cached))
                return cached;

            var convWeight = _weights["v.patch_embd.weight"];
            int outDim = (int)convWeight.Sizes[0];
            using var weight2D = convWeight.View(outDim, patchStride);
            using var weightViewT = weight2D.Transpose();
            var result = Ops.NewContiguous(weightViewT);
            _transposedWeights[key] = result;
            return result;
        }

        private unsafe void AddPositionEmbedding2D(Tensor hidden, Rope2DCache ropeCache, int numPatches)
        {
            var posEmbd = _weights["v.position_embd.weight"];
            if (_useMlxDeviceOps && Environment.GetEnvironmentVariable("TS_GEMMA4_VISION_SPATIAL_DEVICE") == "1")
            {
                int patchesX = ropeCache.PosX.Length > 0 ? ropeCache.PosX[^1] + 1 : 0;
                int patchesY = ropeCache.PosY.Length > 0 ? ropeCache.PosY[^1] + 1 : 0;
                if (!MlxGemma4VisionOps.TryAddPosition2D(hidden, posEmbd, patchesX, patchesY))
                    throw new NotSupportedException("MLX vision position lookup requires matching contiguous F32 patch/tables.");
                return;
            }
            int maxPos = (int)posEmbd.Sizes[1];
            float* posPtr = GetFloatPtr(posEmbd);
            float* xTable = posPtr;
            float* yTable = posPtr + maxPos * _hiddenSize;
            float* dstPtr = GetFloatPtr(hidden);
            int hiddenSize = _hiddenSize;
            int vLen = Vector<float>.Count;

            long dstPtrL = (long)dstPtr, xTableL = (long)xTable, yTableL = (long)yTable;
            var posX = ropeCache.PosX;
            var posY = ropeCache.PosY;
            Parallel.For(0, numPatches, p =>
            {
                float* dstRow = (float*)dstPtrL + (long)p * hiddenSize;
                float* xRow = (float*)xTableL + (long)posX[p] * hiddenSize;
                float* yRow = (float*)yTableL + (long)posY[p] * hiddenSize;
                int d = 0;
                for (; d <= hiddenSize - vLen; d += vLen)
                {
                    var acc = TensorComputePrimitives.LoadVector(dstRow + d)
                            + TensorComputePrimitives.LoadVector(xRow + d)
                            + TensorComputePrimitives.LoadVector(yRow + d);
                    TensorComputePrimitives.StoreVector(dstRow + d, acc);
                }
                for (; d < hiddenSize; d++)
                    dstRow[d] += xRow[d] + yRow[d];
            });
        }

        private Tensor EncoderBlock(Tensor hidden, int blockIdx, int numPatches, int headDim,
            Rope2DCache ropeCache)
        {
            string prefix = $"v.blk.{blockIdx}";

            // Fused fast path: run the whole block (attention + gated MLP) as one
            // on-device GGML graph, keeping every intermediate on the GPU instead
            // of round-tripping each of the ~30 sub-ops through host memory. Falls
            // back to the per-op path on any failure (e.g. flash-attn unsupported
            // for this head_dim/backend). No-op unless the encoder runs on a GGML
            // allocator (_useNativeAttention).
            if (_allocator is GgmlAllocator
                && TryFusedEncoderBlock(hidden, prefix, numPatches, headDim, ropeCache))
                return hidden;

            using var attnNormed = RMSNormOp(hidden, $"{prefix}.ln1.weight");
            using var attnOut = VisionSelfAttention(attnNormed, prefix, numPatches, headDim,
                ropeCache);
            using var postAttnNormed = RMSNormOp(attnOut, $"{prefix}.attn_post_norm.weight");

            Ops.Add(postAttnNormed, postAttnNormed, hidden);
            hidden.Dispose();

            using var ffnNormed = RMSNormOp(postAttnNormed, $"{prefix}.ln2.weight");
            using var mlpOut = VisionMLP(ffnNormed, prefix);
            using var postFfnNormed = RMSNormOp(mlpOut, $"{prefix}.ffn_post_norm.weight");

            var result = new Tensor(_allocator, DType.Float32, postAttnNormed.Sizes);
            Ops.Add(result, postAttnNormed, postFfnNormed);

            string scaleKey = $"v.blk.{blockIdx}.out_scale.weight";
            if (_weights.TryGetValue(scaleKey, out var scaleTensor))
                Ops.Mul(result, result, scaleTensor);

            return result;
        }

        private static readonly string[] _clampLinearSuffixes =
            { "attn_q", "attn_k", "attn_v", "attn_out", "ffn_gate", "ffn_up", "ffn_down" };

        /// <summary>
        /// Run the entire SigLIP encoder block through the fused native GGML kernel
        /// (one graph dispatch). Modifies <paramref name="hidden"/> in place and
        /// returns true on success; returns false (leaving hidden untouched) when a
        /// required weight is missing, an out_scale is present (unsupported by the
        /// kernel), or the native call throws — the caller then uses the per-op path.
        /// </summary>
        private bool TryFusedEncoderBlock(Tensor hidden, string prefix, int numPatches,
            int headDim, Rope2DCache ropeCache)
        {
            // The fused kernel doesn't implement the optional per-channel out_scale.
            if (_weights.ContainsKey($"{prefix}.out_scale.weight"))
            {
                if (_requirePreciseGgmlCudaVision)
                    throw new NotSupportedException("Required precise CUDA vision blocks do not support an optional output scale.");
                return false;
            }

            if (!_weights.TryGetValue($"{prefix}.ln1.weight", out var ln1)
                || !_weights.TryGetValue($"{prefix}.attn_q.weight", out var qW)
                || !_weights.TryGetValue($"{prefix}.attn_k.weight", out var kW)
                || !_weights.TryGetValue($"{prefix}.attn_v.weight", out var vW)
                || !_weights.TryGetValue($"{prefix}.attn_q_norm.weight", out var qNorm)
                || !_weights.TryGetValue($"{prefix}.attn_k_norm.weight", out var kNorm)
                || !_weights.TryGetValue($"{prefix}.attn_post_norm.weight", out var apn)
                || !_weights.TryGetValue($"{prefix}.attn_out.weight", out var outW)
                || !_weights.TryGetValue($"{prefix}.ln2.weight", out var ln2)
                || !_weights.TryGetValue($"{prefix}.ffn_gate.weight", out var gateW)
                || !_weights.TryGetValue($"{prefix}.ffn_up.weight", out var upW)
                || !_weights.TryGetValue($"{prefix}.ffn_down.weight", out var downW)
                || !_weights.TryGetValue($"{prefix}.ffn_post_norm.weight", out var fpn))
            {
                if (_requirePreciseGgmlCudaVision)
                    throw new NotSupportedException("Required precise CUDA vision block is missing a learned projection or normalization weight.");
                return false;
            }

            float[] clamps = BuildClampArray(prefix);
            try
            {
                GgmlBasicOps.FusedGemma4VisionBlock(hidden, _eps, ln1,
                    qW, kW, vW, qNorm, kNorm, apn, outW,
                    ropeCache.PosX, ropeCache.PosY, _ropeTheta,
                    ln2, gateW, upW, downW, fpn,
                    clamps, numPatches, _numHeads, headDim, _usePreciseMetalVision, _requirePreciseGgmlCudaVision);
                return true;
            }
            catch
            {
                // A required precision path must not silently fall back to the
                // generic Metal multiplication that failed the raw quality gate.
                if (_usePreciseMetalVision || _requirePreciseGgmlCudaVision) throw;
                return false;
            }
        }

        // Pack the per-linear QAT activation clamps into the 28-float layout the
        // fused kernel expects: {q,k,v,out,gate,up,down} x {inMin,inMax,outMin,outMax}.
        // Absent clamps use ±float.MaxValue (the kernel treats |bound|>=3e38 as "none").
        private float[] BuildClampArray(string prefix)
        {
            var c = new float[28];
            for (int i = 0; i < _clampLinearSuffixes.Length; i++)
            {
                int b = i * 4;
                if (_clampParams.TryGetValue($"{prefix}.{_clampLinearSuffixes[i]}", out var cp) && cp.HasClamp)
                {
                    c[b] = cp.InMin; c[b + 1] = cp.InMax; c[b + 2] = cp.OutMin; c[b + 3] = cp.OutMax;
                }
                else
                {
                    c[b] = float.MinValue; c[b + 1] = float.MaxValue;
                    c[b + 2] = float.MinValue; c[b + 3] = float.MaxValue;
                }
            }
            return c;
        }

        private unsafe Tensor VisionSelfAttention(Tensor input, string prefix, int numPatches, int headDim,
            Rope2DCache ropeCache)
        {
            var q = ClippableLinear(input, $"{prefix}.attn_q");
            var k = ClippableLinear(input, $"{prefix}.attn_k");
            var v = ClippableLinear(input, $"{prefix}.attn_v");

            ApplyPerHeadRMSNorm(q, _weights[$"{prefix}.attn_q_norm.weight"], numPatches, headDim);
            ApplyPerHeadRMSNorm(k, _weights[$"{prefix}.attn_k_norm.weight"], numPatches, headDim);
            ApplyUnweightedRMSNorm(v, _numHeads * numPatches, headDim);

            Apply2DRoPE(q, ropeCache, numPatches, headDim);
            Apply2DRoPE(k, ropeCache, numPatches, headDim);

            if (_useNativeAttention)
            {
                using var q4 = q.View(1, numPatches, _numHeads, headDim);
                using var k4 = k.View(1, numPatches, _numHeads, headDim);
                using var v4 = v.View(1, numPatches, _numHeads, headDim);
                using var attn4 = _useMlxDeviceOps || _useCudaDeviceOps
                    ? new Tensor(_allocator, DType.Float32, 1, numPatches, _numHeads, headDim)
                    : Ops.ScaledDotProductAttention(null, q4, k4, v4, null, 1f);
                if (_useMlxDeviceOps && !MlxGemma4VisionOps.TrySelfAttention(attn4, q4, k4, v4))
                    throw new NotSupportedException("MLX Gemma4 vision attention requires matching F32 64-channel heads.");
                if (_useCudaDeviceOps && !CudaGemma4VisionOps.TrySelfAttention(attn4, q4, k4, v4))
                    throw new NotSupportedException("CUDA Gemma4 vision attention requires matching F32 64-channel heads.");
                using var flat = attn4.View(numPatches, _hiddenSize);
                q.Dispose();
                k.Dispose();
                v.Dispose();
                return ClippableLinear(flat, $"{prefix}.attn_out");
            }

            using var qR = q.View(numPatches, _numHeads, headDim);
            using var kR = k.View(numPatches, _numHeads, headDim);
            using var vR = v.View(numPatches, _numHeads, headDim);
            using var qT0 = qR.Transpose(0, 1);
            using var kT0 = kR.Transpose(0, 1);
            using var vT0 = vR.Transpose(0, 1);
            using var qHeads = Ops.NewContiguous(qT0);
            using var kHeads = Ops.NewContiguous(kT0);
            using var vHeads = Ops.NewContiguous(vT0);
            q.Dispose();
            k.Dispose();
            v.Dispose();

            using var kT = kHeads.Transpose(1, 2);
            var scores = new Tensor(_allocator, DType.Float32, _numHeads, numPatches, numPatches);
            Ops.AddmmBatch(scores, 0, scores, 1f, qHeads, kT);
            Ops.Softmax(scores, scores);

            var attnOutput = new Tensor(_allocator, DType.Float32, _numHeads, numPatches, headDim);
            Ops.AddmmBatch(attnOutput, 0, attnOutput, 1f, scores, vHeads);
            scores.Dispose();

            using var transposed = attnOutput.Transpose(0, 1);
            using var contiguous = Ops.NewContiguous(transposed);
            using var flatContig = contiguous.View(numPatches, _hiddenSize);
            attnOutput.Dispose();

            return ClippableLinear(flatContig, $"{prefix}.attn_out");
        }

        // 2D RoPE: split the head dim into an X-rotated first half and a
        // Y-rotated second half. Parallelised across patches with SIMD over the
        // (quarter-dim) rotation pairs.
        private unsafe void Apply2DRoPE(Tensor data, Rope2DCache ropeCache, int numPatches, int headDim)
        {
            if (_useMlxDeviceOps || _useCudaDeviceOps)
            {
                EnsureDeviceRopeTables(ropeCache, numPatches, headDim);
                bool success = _useMlxDeviceOps
                    ? MlxGemma4VisionOps.TryRope2D(data, ropeCache.DeviceCosine, ropeCache.DeviceSine, numPatches, _numHeads, headDim)
                    : CudaGemma4VisionOps.TryRope2D(data, ropeCache.DeviceCosine, ropeCache.DeviceSine, numPatches, _numHeads, headDim);
                if (!success) throw new NotSupportedException("Gemma4 vision device RoPE requires contiguous F32 head rows.");
                return;
            }
            float* ptr = GetFloatPtr(data);
            int halfDim = headDim / 2;
            int quarterDim = halfDim / 2;
            int numHeads = _numHeads;
            int vLen = Vector<float>.Count;

            long ptrL = (long)ptr;
            fixed (float* cosX = ropeCache.CosX, sinX = ropeCache.SinX,
                          cosY = ropeCache.CosY, sinY = ropeCache.SinY)
            {
                long cosXL = (long)cosX, sinXL = (long)sinX, cosYL = (long)cosY, sinYL = (long)sinY;
                Parallel.For(0, numPatches, p =>
                {
                    float* dataPtr = (float*)ptrL;
                    int ropeBase = p * quarterDim;
                    float* cX = (float*)cosXL + ropeBase, sX = (float*)sinXL + ropeBase;
                    float* cY = (float*)cosYL + ropeBase, sY = (float*)sinYL + ropeBase;

                    for (int h = 0; h < numHeads; h++)
                    {
                        float* head = dataPtr + ((long)p * numHeads + h) * headDim;
                        float* headY = head + halfDim;
                        int j = 0;
                        for (; j <= quarterDim - vLen; j += vLen)
                        {
                            var x0 = TensorComputePrimitives.LoadVector(head + j);
                            var x1 = TensorComputePrimitives.LoadVector(head + j + quarterDim);
                            var cv = TensorComputePrimitives.LoadVector(cX + j);
                            var sv = TensorComputePrimitives.LoadVector(sX + j);
                            TensorComputePrimitives.StoreVector(head + j, x0 * cv - x1 * sv);
                            TensorComputePrimitives.StoreVector(head + j + quarterDim, x0 * sv + x1 * cv);

                            var y0 = TensorComputePrimitives.LoadVector(headY + j);
                            var y1 = TensorComputePrimitives.LoadVector(headY + j + quarterDim);
                            var cvy = TensorComputePrimitives.LoadVector(cY + j);
                            var svy = TensorComputePrimitives.LoadVector(sY + j);
                            TensorComputePrimitives.StoreVector(headY + j, y0 * cvy - y1 * svy);
                            TensorComputePrimitives.StoreVector(headY + j + quarterDim, y0 * svy + y1 * cvy);
                        }
                        for (; j < quarterDim; j++)
                        {
                            float x0 = head[j], x1 = head[j + quarterDim];
                            head[j] = x0 * cX[j] - x1 * sX[j];
                            head[j + quarterDim] = x0 * sX[j] + x1 * cX[j];
                            float y0 = headY[j], y1 = headY[j + quarterDim];
                            headY[j] = y0 * cY[j] - y1 * sY[j];
                            headY[j + quarterDim] = y0 * sY[j] + y1 * cY[j];
                        }
                    }
                });
            }
        }

        private void ApplyPerHeadRMSNorm(Tensor data, Tensor normWeight, int numPatches, int headDim)
        {
            int total = _numHeads * numPatches;
            using var reshaped = data.View(total, headDim);
            Ops.RMSNorm(reshaped, reshaped, normWeight, null, _eps);
        }

        private void ApplyUnweightedRMSNorm(Tensor data, int numVectors, int dim)
        {
            if (_onesForNorm == null || (int)_onesForNorm.Sizes[0] != dim)
            {
                _onesForNorm?.Dispose();
                _onesForNorm = new Tensor(_allocator, DType.Float32, dim);
                Ops.Fill(_onesForNorm, 1f);
            }
            using var reshaped = data.View(numVectors, dim);
            Ops.RMSNorm(reshaped, reshaped, _onesForNorm, null, _eps);
        }

        private unsafe Tensor VisionMLP(Tensor input, string prefix)
        {
            var gate = ClippableLinear(input, $"{prefix}.ffn_gate");
            var up = ClippableLinear(input, $"{prefix}.ffn_up");

            // gelu_pytorch_tanh(gate) * up. Gemma 4's vision config declares
            // hidden_activation = "gelu_pytorch_tanh"; Ops.GELUMul is the tanh-approximate GELU and
            // is a fused kernel on the GGML backends. (llama.cpp reaches QuickGELU here only because
            // its mmproj files omit clip.use_gelu and clip.cpp then defaults to FFN_GELU_QUICK.)
            Ops.GELUMul(gate, gate, up);
            up.Dispose();

            var down = ClippableLinear(gate, $"{prefix}.ffn_down");
            gate.Dispose();
            return down;
        }

        private unsafe Tensor ClippableLinear(Tensor input, string prefix)
        {
            string weightName = $"{prefix}.weight";
            var weight = _weights[weightName];
            int seqLen = (int)input.Sizes[0];
            int outDim = (int)weight.Sizes[0];

            Tensor contiguousInput = input.IsContiguous() ? null : Ops.NewContiguous(input);
            Tensor src = contiguousInput ?? input;

            bool hasClamp = _clampParams.TryGetValue(prefix, out var cp) && cp.HasClamp;

            if (hasClamp)
                Clamp(src, cp.InMin, cp.InMax);

            var result = new Tensor(_allocator, DType.Float32, seqLen, outDim);
            if (!TryCompressedProjection(result, src, weightName))
                Ops.Addmm(result, 0, result, 1f, src, GetOrCreateTransposedWeight(weightName));

            contiguousInput?.Dispose();

            if (hasClamp)
                Clamp(result, cp.OutMin, cp.OutMax);

            return result;
        }

        private unsafe void Clamp(Tensor t, float min, float max)
        {
            if (_useCudaDeviceOps)
            {
                if (!CudaMiniMaxH3Ops.TryClamp(t, t, min, max))
                    throw new NotSupportedException("CUDA Gemma4 vision clamp requires contiguous F32 tensors.");
                return;
            }
            if (_useMlxDeviceOps)
            {
                if (!MlxMiniMaxH3Ops.TryClamp(t, t, min, max))
                    throw new NotSupportedException("MLX Gemma4 vision clamp requires contiguous F32 tensors.");
                return;
            }
            float* ptr = GetFloatPtr(t);
            int count = (int)t.ElementCount();
            for (int i = 0; i < count; i++)
            {
                if (ptr[i] < min) ptr[i] = min;
                else if (ptr[i] > max) ptr[i] = max;
            }
        }

        private unsafe Tensor PoolAndProject(Tensor visionOutput, int patchesX, int patchesY, int numPatches)
        {
            int mergedX = patchesX / _nMerge;
            int mergedY = patchesY / _nMerge;
            int mergedPatches = mergedX * mergedY;

            var pooled = new Tensor(_allocator, DType.Float32, mergedPatches, _hiddenSize);
            if (_useMlxDeviceOps && _nMerge == 3 && Environment.GetEnvironmentVariable("TS_GEMMA4_VISION_SPATIAL_DEVICE") == "1")
            {
                if (!MlxGemma4VisionOps.TryPool3x3(pooled, visionOutput, patchesX, patchesY))
                {
                    pooled.Dispose();
                    throw new NotSupportedException("MLX vision pooling requires complete contiguous F32 3x3 boxes.");
                }
            }
            else
            {
            float* srcPtr = GetFloatPtr(visionOutput);
            float* dstPtr = GetFloatPtr(pooled);
            int hiddenSize = _hiddenSize;
            int nMerge = _nMerge;
            int vLen = Vector<float>.Count;

            long srcPtrL = (long)srcPtr, dstPtrL = (long)dstPtr;
            Parallel.For(0, mergedY, py =>
            {
                float* src = (float*)srcPtrL;
                float* dst = (float*)dstPtrL;
                for (int px = 0; px < mergedX; px++)
                {
                    int outIdx = py * mergedX + px;
                    float* outRow = dst + (long)outIdx * hiddenSize;
                    for (int d = 0; d < hiddenSize; d++)
                        outRow[d] = 0;

                    int count = 0;
                    for (int ky = 0; ky < nMerge; ky++)
                    {
                        for (int kx = 0; kx < nMerge; kx++)
                        {
                            int srcY = py * nMerge + ky;
                            int srcX = px * nMerge + kx;
                            if (srcY < patchesY && srcX < patchesX)
                            {
                                float* srcRow = src + (long)(srcY * patchesX + srcX) * hiddenSize;
                                int d = 0;
                                for (; d <= hiddenSize - vLen; d += vLen)
                                    TensorComputePrimitives.StoreVector(outRow + d,
                                        TensorComputePrimitives.LoadVector(outRow + d)
                                        + TensorComputePrimitives.LoadVector(srcRow + d));
                                for (; d < hiddenSize; d++)
                                    outRow[d] += srcRow[d];
                                count++;
                            }
                        }
                    }

                    float invCount = 1f / count;
                    var invVec = new Vector<float>(invCount);
                    int e = 0;
                    for (; e <= hiddenSize - vLen; e += vLen)
                        TensorComputePrimitives.StoreVector(outRow + e,
                            TensorComputePrimitives.LoadVector(outRow + e) * invVec);
                    for (; e < hiddenSize; e++)
                        outRow[e] *= invCount;
                }
            });
            }

            VisionTrace("pool.mean", pooled);

            // Scale by sqrt(hiddenSize)
            float scale = MathF.Sqrt(_hiddenSize);
            Ops.Mul(pooled, pooled, scale);
            VisionTrace("pool.scale", pooled);

            // Vision standardization before projection (matches Ollama)
            if (_weights.TryGetValue("v.std_bias", out var stdBias) &&
                _weights.TryGetValue("v.std_scale", out var stdScale))
            {
                Ops.Sub(pooled, pooled, stdBias);
                Ops.Mul(pooled, pooled, stdScale);
                VisionTrace("pool.std", pooled);
            }

            // Gemma4MultimodalEmbedder: embedding_pre_projection_norm (unweighted RMSNorm over the
            // *vision* hidden size) runs BEFORE embedding_projection, matching
            // Gemma4MultimodalEmbedder.forward() upstream and the gemma4uv path above (step 7/8).
            ApplyUnweightedRMSNorm(pooled, mergedPatches, _hiddenSize);
            VisionTrace("pool.prenorm", pooled);

            var projected = LinearProjection(pooled, "mm.input_projection.weight");
            VisionTrace("pool.proj", projected);
            pooled.Dispose();

            return projected;
        }

        private Tensor LinearProjection(Tensor input, string weightName)
        {
            var weight = _weights[weightName];
            int seqLen = (int)input.Sizes[0];
            int outDim = (int)weight.Sizes[0];

            var result = new Tensor(_allocator, DType.Float32, seqLen, outDim);
            if (!TryCompressedProjection(result, input, weightName))
                Ops.Addmm(result, 0, result, 1f, input, GetOrCreateTransposedWeight(weightName));
            return result;
        }

        private bool TryCompressedProjection(Tensor result, Tensor input, string weightName)
        {
            if (TryCpuVisionProjection(result, input, weightName)) return true;
            if (_usePreciseMetalVision)
            {
                var source = _weights[weightName];
                using var flattened = source.DimensionCount == 4
                    ? source.View(source.Sizes[0], input.Sizes[1]) : null;
                if (!GgmlBasicOps.TryGemma4ProjectionFloat32(result, input, flattened ?? source))
                    throw new NotSupportedException("Metal precise vision projection requires compatible F32 operands: " + weightName);
                return true;
            }
            if (_useCudaDeviceOps)
            {
                Tensor weight = weightName == "v.patch_embd.weight" && _weights[weightName].DimensionCount == 4
                    ? GetOrCreatePatchEmbedTransposed(checked((int)input.Sizes[1])) : GetOrCreateTransposedWeight(weightName);
                if (!CudaGemma4VisionOps.TryProjectionFloat32(result, input, weight))
                    throw new InvalidOperationException("CUDA pedantic F32 vision projection failed: " + weightName);
                return true;
            }
            if (!_compressedMatrices.TryGetValue(weightName, out var compressed)) return false;
            using Tensor padded = compressed.Input == compressed.OriginalInput ? null :
                new Tensor(_allocator, DType.Float32, input.Sizes[0], compressed.Input);
            if (padded != null && !MlxGemma4VisionOps.TryPadProjectionInput(padded, input, compressed.Input - compressed.OriginalInput))
                throw new InvalidOperationException("MLX vision projection zero padding failed: " + weightName);
            if (!MlxQuantizedOps.TryAddmmQuantizedToFloat32(result, padded ?? input, compressed.Pointer, compressed.Pointer,
                (int)GgmlTensorType.Q8_0, compressed.Input, compressed.Output, compressed.Bytes, preserveFloat32: true))
                throw new InvalidOperationException("MLX F32 vision compressed projection failed: " + weightName);
            return true;
        }

        private Tensor RMSNormOp(Tensor input, string weightName)
        {
            var alpha = _weights[weightName];
            return Ops.RMSNorm(null, input, alpha, null, _eps);
        }

        private Tensor CreateIntTensor(int[] data, params long[] sizes)
        {
            var tensor = new Tensor(_allocator, DType.Int32, sizes);
            tensor.SetElementsAsInt(data);
            return tensor;
        }

        private static unsafe float* GetFloatPtr(Tensor t) =>
            TensorComputePrimitives.GetFloatPointer(t);

        private Tensor GetOrCreateTransposedWeight(string weightName)
        {
            if (_transposedWeights.TryGetValue(weightName, out var transposed))
                return transposed;

            using var weightViewT = _weights[weightName].Transpose();
            transposed = Ops.NewContiguous(weightViewT);
            _transposedWeights[weightName] = transposed;
            return transposed;
        }

        private Rope2DCache GetOrCreateRopeCache(int patchesX, int patchesY, int headDim)
        {
            long key = ((long)patchesX << 32) | (uint)patchesY;
            if (_ropeCache.TryGetValue(key, out var cache))
                return cache;

            int numPatches = patchesX * patchesY;
            int halfDim = headDim / 2;
            int quarterDim = halfDim / 2;
            int[] posX = new int[numPatches];
            int[] posY = new int[numPatches];
            float[] cosX = new float[numPatches * quarterDim];
            float[] sinX = new float[numPatches * quarterDim];
            float[] cosY = new float[numPatches * quarterDim];
            float[] sinY = new float[numPatches * quarterDim];
            float[] invFreq = new float[quarterDim];

            for (int j = 0; j < quarterDim; j++)
                invFreq[j] = (float)(1.0 / Math.Pow(_ropeTheta, 2.0 * j / halfDim));

            for (int p = 0; p < numPatches; p++)
            {
                int x = p % patchesX;
                int y = p / patchesX;
                posX[p] = x;
                posY[p] = y;

                int baseIdx = p * quarterDim;
                for (int j = 0; j < quarterDim; j++)
                {
                    float angleX = x * invFreq[j];
                    float angleY = y * invFreq[j];
                    cosX[baseIdx + j] = MathF.Cos(angleX);
                    sinX[baseIdx + j] = MathF.Sin(angleX);
                    cosY[baseIdx + j] = MathF.Cos(angleY);
                    sinY[baseIdx + j] = MathF.Sin(angleY);
                }
            }

            cache = new Rope2DCache
            {
                PosX = posX,
                PosY = posY,
                CosX = cosX,
                SinX = sinX,
                CosY = cosY,
                SinY = sinY,
            };
            _ropeCache[key] = cache;
            return cache;
        }

        private void EnsureDeviceRopeTables(Rope2DCache cache, int patches, int headDim)
        {
            if (cache.DeviceCosine != null) return;
            int half = headDim / 2, quarter = headDim / 4;
            var cosine = new float[patches, half];
            var sine = new float[patches, half];
            for (int patch = 0; patch < patches; ++patch)
            for (int channel = 0; channel < quarter; ++channel)
            {
                int index = patch * quarter + channel;
                cosine[patch, channel] = cache.CosX[index];
                sine[patch, channel] = cache.SinX[index];
                cosine[patch, quarter + channel] = cache.CosY[index];
                sine[patch, quarter + channel] = cache.SinY[index];
            }
            Tensor deviceCosine = Tensor.FromArray(_allocator, cosine);
            try { cache.DeviceSine = Tensor.FromArray(_allocator, sine); cache.DeviceCosine = deviceCosine; }
            catch { deviceCosine.Dispose(); throw; }
        }

        public void Dispose()
        {
            // Host-model callers yield their GPU lock and retain the prior owner contract.
            if (_requirePreciseGgmlCudaVision && _hostModel == null)
            {
                lock (_ggmlVisionStackSync)
                {
                    if (_ggmlVisionStackDisposed) return;
                    _ggmlVisionStackDisposed = true;
                    DisposeCore();
                }
                return;
            }
            if (!CpuVisionProjectionRequested) { DisposeCore(); return; }
            lock (_cpuVisionProjectionSync)
            {
                _cpuVisionDisposed = true;
                try { DisposeCore(); }
                finally { ReleaseCpuVisionProjection(); }
            }
        }

        private void DisposeCore()
        {
            // The fused block binds each weight ZERO-COPY on unified-memory Metal: GGML
            // keeps an MTLBuffer that wraps these exact host pages, cached by host pointer.
            // Freeing the pages without dropping that wrapper leaves a live GPU mapping of
            // memory the allocator is about to hand out again, and the NEXT encoder in the
            // process then computes from whatever the GPU still sees there. That is silent
            // and total: it was measured at 131% relative L2 (worst-token cosine -0.09)
            // against the numpy oracle, on the second tower built in one process, while the
            // first tower stayed correct. Invalidate before freeing, the same contract
            // ModelBase.ReleaseGgmlDeviceResidency keeps for the language model's weights.
            ReleaseDeviceBindings();
            foreach (var matrix in _compressedMatrices.Values)
            {
                try { MlxQuantizedOps.ReleaseQuantizedWeight((MlxAllocator)_allocator, matrix.Pointer); }
                finally { matrix.Free(); }
            }
            _compressedMatrices.Clear();
            _onesForNorm?.Dispose();
            foreach (var w in _transposedWeights.Values)
                w.Dispose();
            _transposedWeights.Clear();
            foreach (var w in _weights.Values)
                w.Dispose();
            _weights.Clear();
            foreach (var cache in _ropeCache.Values)
            {
                cache.DeviceCosine?.Dispose();
                cache.DeviceSine?.Dispose();
            }
            _ropeCache.Clear();
        }

        /// <summary>
        /// Drop any GGML device binding (zero-copy host-pointer wrapper or device copy)
        /// held against this tower's weight buffers. Idempotent, and a no-op on a
        /// non-GGML allocator.
        /// </summary>
        private void ReleaseDeviceBindings()
        {
            if (_allocator is not GgmlAllocator)
                return;
            foreach (var w in _weights.Values)
            {
                if (w != null)
                    GgmlBasicOps.InvalidateTensorHostBuffer(w);
            }
            foreach (var w in _transposedWeights.Values)
            {
                if (w != null)
                    GgmlBasicOps.InvalidateTensorHostBuffer(w);
            }
        }
    }
}

