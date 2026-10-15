// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Embeddings;

/// <summary>F32 EmbeddingGemma 2 execution on the direct MLX and CUDA tensor backends.</summary>
internal sealed class TensorEmbeddingGemma2Encoder : IDisposable
{
    private sealed class CompressedWeight
    {
        public readonly GCHandle Pinned;
        public readonly GgmlTensorType Type;
        public readonly int Input, Output, Bytes;
        public Tensor DensePrefill;
        public Tensor DenseHalfPrefill;
        public IntPtr Pointer => Pinned.AddrOfPinnedObject();
        public CompressedWeight(byte[] data, GgmlTensorType type, int input, int output)
        { Pinned = GCHandle.Alloc(data, GCHandleType.Pinned); Type = type; Input = input; Output = output; Bytes = data.Length; }
    }
    private sealed record Layer(Tensor Qkv, Tensor Attention, Tensor GateUp, Tensor Down,
        Tensor Gate, Tensor Projection, Tensor AttentionNorm, Tensor QNorm, Tensor KNorm,
        Tensor AttentionPostNorm, Tensor FfnNorm, Tensor FfnPostNorm, Tensor PleNorm,
        int Head, int KvHeads, bool Sliding, bool HalfAttentionSafe, bool HalfQkvSafe, bool HalfGateUpSafe, float Scale);

    private readonly IAllocator _allocator;
    private readonly CudaEmbeddingPhaseProfiler _cudaProfiler;
    private readonly CudaEmbeddingAttentionPlan _cudaAttentionPlan;
    private readonly CudaEmbeddingProjectionPlan _cudaProjectionPlan;
    private readonly Dictionary<Tensor, CudaEmbeddingProjectionPlan.PreparedWeight> _cudaNormalizedProjections = new();
    private readonly List<Tensor> _weights = new();
    private readonly Dictionary<Tensor, CompressedWeight> _compressed = new();
    private readonly Dictionary<string, (Tensor Cos, Tensor Sin)> _ropeCache = new();
    private readonly Dictionary<string, Tensor> _maskCache = new();
    private readonly MlxEmbeddingAttentionPlan _mlxAttentionPlan;
    private readonly MlxRaggedEmbeddingAttentionPlan _mlxRaggedAttentionPlan;
    private readonly Dictionary<int, float[]> _tokenCache = new();
    private readonly int _dim, _out, _heads, _pleDim, _ff, _vocab, _window;
    private readonly float _epsilon, _localBase, _globalBase;
    private readonly GgmlTensorType _tokenType;
    private readonly int _tokenRowBytes;
    private readonly Tensor _pleProjection, _pleNorm, _outputNorm, _outputProjection;
    private readonly Layer[] _layers;
    private byte[] _tokenData;
    private bool _disposed;

    public IAllocator Allocator => _allocator;

    public TensorEmbeddingGemma2Encoder(GgufFile file, IAllocator allocator)
    {
        ArgumentNullException.ThrowIfNull(file);
        _allocator = allocator ?? throw new ArgumentNullException(nameof(allocator));
        if (allocator is not (MlxAllocator or CudaAllocator))
            throw new NotSupportedException("Direct EmbeddingGemma 2 tensor execution requires MLX or CUDA.");
        bool cudaTf32Attention = Environment.GetEnvironmentVariable("TS_EMBEDDING_CUDA_MATERIALIZED_ATTENTION_TF32") == "1";
        if (cudaTf32Attention && (allocator is not CudaAllocator ||
                Environment.GetEnvironmentVariable("TS_EMBEDDING_CUDA_MATERIALIZED_ATTENTION") != "1"))
            throw new NotSupportedException("The CUDA TF32 attention diagnostic requires CUDA and TS_EMBEDDING_CUDA_MATERIALIZED_ATTENTION=1.");
        if (allocator is MlxAllocator) { _mlxAttentionPlan = new(); _mlxRaggedAttentionPlan = new(); }
        const string p = "gemma-embedding2.";
        if (file.GetString("general.architecture") != "gemma-embedding2")
            throw new InvalidDataException("Expected an EmbeddingGemma 2 model.");
        _dim = Positive(file, p + "embedding_length");
        _out = Positive(file, p + "embedding_length_out");
        _heads = Positive(file, p + "attention.head_count");
        _pleDim = Positive(file, p + "embedding_length_per_layer_input");
        _ff = Positive(file, p + "feed_forward_length");
        int layers = Positive(file, p + "block_count");
        if (_dim > 16384 || _out > 16384 || layers > 128 || _heads > 128 || _ff > 65536 || _pleDim > 16384)
            throw new InvalidDataException("EmbeddingGemma 2 tensor dimensions exceed supported limits.");
        int slidingWindow = Positive(file, p + "attention.sliding_window");
        // llama.cpp's symmetric mask uses floor(full width / 2), including odd widths.
        _window = slidingWindow / 2;
        EmbeddingGemma2TextMetadata.Validate(file,
            Positive(file, p + "attention.key_length"), Positive(file, p + "attention.key_length_swa"));
        _epsilon = file.GetFloat32(p + "attention.layer_norm_rms_epsilon", 1e-6f);
        _globalBase = file.GetFloat32(p + "rope.freq_base", 1000000);
        _localBase = file.GetFloat32(p + "rope.freq_base_swa", 10000);
        if (!float.IsFinite(_epsilon) || _epsilon <= 0 || !float.IsFinite(_globalBase) || _globalBase <= 0 ||
            !float.IsFinite(_localBase) || _localBase <= 0)
            throw new InvalidDataException("EmbeddingGemma 2 norm or rotary parameters are invalid.");
        try
        {
            if (allocator is CudaAllocator cuda)
            {
                _cudaProfiler = CudaEmbeddingPhaseProfiler.TryCreate(cuda);
                if (Environment.GetEnvironmentVariable("TS_EMBEDDING_CUDA_MATERIALIZED_ATTENTION") == "1")
                    _cudaAttentionPlan = new(cuda, computePolicy: cudaTf32Attention
                        ? CudaEmbeddingAttentionComputePolicy.FastTf32WithF32Storage
                        : CudaEmbeddingAttentionComputePolicy.PedanticF32);
                if (Environment.GetEnvironmentVariable("TS_EMBEDDING_CUDA_NORMALIZED_HALF") == "1")
                    _cudaProjectionPlan = new(cuda);
            }
            var token = Require(file, "token_embd.weight", _dim);
            _vocab = checked((int)token.Shape[1]);
            _tokenType = token.Type;
            _tokenRowBytes = checked((int)ManagedQuantizedOps.RowSize((int)_tokenType, _dim));
            _tokenData = file.ReadTensorData(token);
            _pleProjection = Projection(file, "per_layer_model_proj.weight", _dim, checked(layers * _pleDim));
            _pleNorm = Vector(file, "per_layer_proj_norm.weight", _pleDim);
            _outputNorm = Vector(file, "output_norm.weight", _dim);
            _outputProjection = Projection(file, "output.weight", _dim, _out);
            var pattern = file.GetBoolArray(p + "attention.sliding_window_pattern");
            var numericPattern = file.GetInt32Array(p + "attention.sliding_window_pattern");
            int period = pattern != null || numericPattern != null ? 6
                : checked((int)file.GetUint32(p + "attention.sliding_window_pattern", 6));
            var kvCounts = file.GetInt32Array(p + "attention.head_count_kv");
            int localHead = Positive(file, p + "attention.key_length_swa");
            int globalHead = Positive(file, p + "attention.key_length");
            _layers = new Layer[layers];
            for (int i = 0; i < layers; ++i)
            {
                bool local = pattern != null ? pattern.Length == layers ? pattern[i]
                    : throw new InvalidDataException("Sliding window pattern must match the layer count.")
                    : numericPattern != null ? numericPattern.Length == layers && numericPattern[i] is 0 or 1
                    ? numericPattern[i] != 0 : throw new InvalidDataException("Sliding window pattern must match the layer count.")
                    : period == 0 || i % period < period - 1;
                int head = local ? localHead : globalHead;
                string b = $"blk.{i}.";
                var kInfo = Require(file, b + "attn_k.weight", _dim);
                int kWidth = checked((int)kInfo.Shape[1]);
                int kv = kWidth / head;
                if ((head & 1) != 0 || head <= 0 || kWidth % head != 0 || kv <= 0 || _heads % kv != 0 ||
                    (kvCounts != null && (kvCounts.Length != layers || kvCounts[i] != kv)))
                    throw new InvalidDataException("EmbeddingGemma 2 grouped attention dimensions are invalid.");
                int qWidth = checked(_heads * head);
                var qkv = JoinedProjection(file, _dim, [b + "attn_q.weight", b + "attn_k.weight", b + "attn_v.weight"], [qWidth, kWidth, kWidth]);
                var gateUp = JoinedProjection(file, _dim, [b + "ffn_gate.weight", b + "ffn_up.weight"], [_ff, _ff]);
                PrepareCudaNormalizedProjection(file, qkv, b + "attn_norm.weight",
                    [b + "attn_q.weight", b + "attn_k.weight", b + "attn_v.weight"]);
                PrepareCudaNormalizedProjection(file, gateUp, b + "ffn_norm.weight",
                    [b + "ffn_gate.weight", b + "ffn_up.weight"]);
                float scale = Read(file, b + "layer_output_scale.weight", 1)[0];
                if (!float.IsFinite(scale)) throw new InvalidDataException("EmbeddingGemma 2 layer scale must be finite.");
                _layers[i] = new(qkv, Projection(file, b + "attn_output.weight", qWidth, _dim), gateUp,
                    Projection(file, b + "ffn_down.weight", _ff, _dim),
                    Projection(file, b + "inp_gate.weight", _dim, _pleDim),
                    Projection(file, b + "proj.weight", _pleDim, _dim),
                    Vector(file, b + "attn_norm.weight", _dim), Vector(file, b + "attn_q_norm.weight", head),
                    Vector(file, b + "attn_k_norm.weight", head), Vector(file, b + "post_attention_norm.weight", _dim),
                    Vector(file, b + "ffn_norm.weight", _dim), Vector(file, b + "post_ffw_norm.weight", _dim),
                    Vector(file, b + "post_norm.weight", _dim), head, kv, local,
                    FitsHalfAttention(file, b + "attn_q_norm.weight", b + "attn_k_norm.weight", head),
                    allocator is MlxAllocator && FitsHalfProjection(file, b + "attn_norm.weight", _dim,
                        [b + "attn_q.weight", b + "attn_k.weight", b + "attn_v.weight"]),
                    allocator is MlxAllocator && FitsHalfProjection(file, b + "ffn_norm.weight", _dim,
                        [b + "ffn_gate.weight", b + "ffn_up.weight"]), scale);
            }
        }
        catch { Dispose(); throw; }
    }

    public void TokenRows(int[] tokens, float[] output)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (output.Length != checked(tokens.Length * _dim)) throw new ArgumentException("Token row output dimensions are invalid.");
        float scale = MathF.Sqrt(_dim);
        for (int i = 0; i < tokens.Length; ++i)
        {
            int token = tokens[i];
            if ((uint)token >= (uint)_vocab) throw new ArgumentException("Token is outside the embedding vocabulary.");
            if (!_tokenCache.TryGetValue(token, out var row))
            {
                row = new float[_dim];
                ManagedQuantizedOps.DequantizeToFloat32((int)_tokenType, _tokenData, checked(token * _tokenRowBytes), row, 0, _dim);
                for (int d = 0; d < _dim; ++d) row[d] *= scale;
                if (_tokenCache.Count >= 4096) _tokenCache.Clear();
                _tokenCache[token] = row;
            }
            row.CopyTo(output, i * _dim);
        }
    }

    public void Encode(int[] flat, int[] lengths, float[] output, CancellationToken cancellationToken)
    {
        var inputs = new float[checked(flat.Length * _dim)];
        TokenRows(flat, inputs);
        EncodeInputs(inputs, lengths, output, cancellationToken);
    }

    public void EncodeInputs(float[] inputs, int[] lengths, float[] output, CancellationToken cancellationToken)
    {
        // MLX operators already dispatch to this worker. Keeping the complete
        // graph construction there removes a queue round trip per operation.
        if (_allocator is MlxAllocator && !MlxWorker.Shared.IsOnWorkerThread)
        {
            MlxWorker.Shared.Invoke(() => { EncodeInputs(inputs, lengths, output, cancellationToken); return true; });
            return;
        }
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (output.Length != checked(lengths.Length * _out)) throw new ArgumentException("Embedding output dimensions are invalid.");
        int total = 0, maximumLength = 0;
        bool equal = true;
        foreach (int length in lengths)
        {
            if (length < 1 || length > 8192) throw new ArgumentException("Embedding input dimensions are invalid.");
            total = checked(total + length);
            maximumLength = Math.Max(maximumLength, length);
            equal &= length == lengths[0];
        }
        if (inputs.Length != checked(total * _dim)) throw new ArgumentException("Embedding input lengths do not match the supplied rows.");
        if (lengths.Length == 0) return;
        // Share every projection and pointwise operation across a batch. Equal
        // lengths use SDPA's batch dimension; short mixed lengths use a block mask.
        bool packedCuda = _allocator is CudaAllocator cuda && total <= 4096 && CudaFusedOps.HasEmbeddingAttention(cuda);
        bool packedMlx = _allocator is MlxAllocator && total <= 4096;
        foreach (var layer in _layers) packedMlx &= layer.Head is 256 or 512;
        // Narrow variant windows amplify the numerical differences between
        // packed and independent device matrix/attention kernels. Preserve the
        // published radius-512 batching path and use independent graphs when
        // a smaller local window actually masks positions in a sequence.
        bool independentNarrowWindows = _window < 512 && lengths.Length > 1 && maximumLength - 1 > _window;
        if (!independentNarrowWindows && (equal || total <= 1024 || packedCuda || packedMlx))
        {
            EncodeBatch(inputs, lengths, output, 0, cancellationToken);
            return;
        }
        int offset = 0;
        for (int i = 0; i < lengths.Length; ++i)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int tokens = lengths[i];
            if (tokens < 1 || tokens > 8192 || (long)offset + (long)tokens * _dim > inputs.Length)
                throw new ArgumentException("Embedding input dimensions are invalid.");
            var sequence = new float[checked(tokens * _dim)];
            Array.Copy(inputs, offset, sequence, 0, sequence.Length);
            EncodeBatch(sequence, [tokens], output, i * _out, cancellationToken);
            offset += sequence.Length;
        }
        if (offset != inputs.Length) throw new ArgumentException("Embedding input lengths do not match the supplied rows.");
    }

    private void EncodeBatch(float[] inputs, int[] lengths, float[] output, int outputOffset, CancellationToken cancellationToken)
    {
        try { EncodeBatchCore(inputs, lengths, output, outputOffset, cancellationToken); }
        catch (Exception failure)
        {
            List<Exception> cleanupFailures = null;
            try { _cudaProfiler?.AbortStep("embedding-step-failed-or-canceled"); }
            catch (Exception cleanup) { (cleanupFailures ??= new()).Add(cleanup); }
            try { _cudaAttentionPlan?.AbortPendingWork(); }
            catch (Exception cleanup) { (cleanupFailures ??= new()).Add(cleanup); }
            try { _cudaProjectionPlan?.AbortPendingWork(); }
            catch (Exception cleanup) { (cleanupFailures ??= new()).Add(cleanup); }
            if (cleanupFailures != null)
            {
                cleanupFailures.Insert(0, failure);
                throw new AggregateException("Embedding failed and CUDA cleanup also failed.", cleanupFailures);
            }
            throw;
        }
    }

    private void EncodeBatchCore(float[] inputs, int[] lengths, float[] output, int outputOffset, CancellationToken cancellationToken)
    {
        int tokens = inputs.Length / _dim, batch = lengths.Length;
        bool equal = true;
        foreach (int length in lengths) equal &= length == lengths[0];
        int attentionBatch = equal ? batch : 1, attentionTokens = equal ? lengths[0] : tokens;
        bool packedMlx = !equal && tokens > 1024 && _allocator is MlxAllocator;
        string geometry = string.Join(",", lengths);
        int[] positions = new int[tokens];
        int tokenOffset = 0, maximumLength = 0;
        foreach (int length in lengths)
        {
            for (int t = 0; t < length; ++t) positions[tokenOffset + t] = t;
            tokenOffset += length;
            maximumLength = Math.Max(maximumLength, length);
        }
        // A single bounded block mask is faster than many tiny global
        // attention dispatches when every packed sequence is short. Longer
        // sequences retain the segmented path and its bounded score memory.
        bool segmentedMlxAttention = packedMlx && !(tokens <= 1536 && maximumLength <= 512);
        using var hidden = new Tensor(_allocator, DType.Float32, tokens, _dim);
        hidden.SetElementsAsFloat(inputs);
        _cudaProfiler?.BeginStep(lengths, _dim, _heads, _layers.Length);
        using var pleFlat = Linear(hidden, _pleProjection, packedMlx);
        Ops.Mul(pleFlat, pleFlat, 1 / MathF.Sqrt(_dim));
        using var pleNormRows = pleFlat.View(checked(tokens * _layers.Length), _pleDim);
        Ops.RMSNorm(pleNormRows, pleNormRows, _pleNorm, null, _epsilon);
        using var ple = pleFlat.View(tokens, _layers.Length, _pleDim);
        _cudaProfiler?.End(CudaEmbeddingPhase.InitialPle);
        // CUDA flash attention consumes sequence bounds directly, so it neither
        // repeats the grouped K/V heads nor materializes quadratic score masks.
        using var sequenceBounds = !equal && _allocator is CudaAllocator ? BuildSequenceBounds(lengths, tokens) : null;
        Tensor fullMask = null, localMask = null;
        var ropeTables = new Dictionary<(int Head, float Base), (Tensor Cos, Tensor Sin)>();
        try
        {
            for (int i = 0; i < _layers.Length; ++i)
            {
                cancellationToken.ThrowIfCancellationRequested();
                Layer layer = _layers[i];
                using var normed = _cudaProjectionPlan == null
                    ? Ops.RMSNorm(null, hidden, layer.AttentionNorm, null, _epsilon)
                    : new Tensor(_allocator, DType.Float32, tokens, _dim);
                using var qkv = _cudaProjectionPlan == null
                    ? Linear(normed, layer.Qkv, packedMlx, layer.HalfQkvSafe)
                    : CudaNormalizedProjection(hidden, normed, layer.Qkv, cancellationToken);
                _cudaProfiler?.End(CudaEmbeddingPhase.NormQkv);
                int qWidth = _heads * layer.Head, kvWidth = layer.KvHeads * layer.Head;
                using var qView = qkv.Narrow(1, 0, qWidth);
                using var kView = qkv.Narrow(1, qWidth, kvWidth);
                using var vView = qkv.Narrow(1, qWidth + kvWidth, kvWidth);
                using var q = Ops.AsContiguous(qView);
                using var k = Ops.AsContiguous(kView);
                using var v = Ops.AsContiguous(vView);
                using (var qr = q.View(tokens * _heads, layer.Head)) Ops.RMSNorm(qr, qr, layer.QNorm, null, _epsilon);
                using (var kr = k.View(tokens * layer.KvHeads, layer.Head)) Ops.RMSNorm(kr, kr, layer.KNorm, null, _epsilon);
                using (var vr = v.View(tokens * layer.KvHeads, layer.Head)) Ops.RMSNorm(vr, vr, null, null, _epsilon);
                float basis = layer.Sliding ? _localBase : _globalBase;
                if (!ropeTables.TryGetValue((layer.Head, basis), out var table))
                    ropeTables[(layer.Head, basis)] = table = BuildRope(positions, layer.Head, basis, geometry);
                ApplyRope(q, table, _heads, tokens, layer.Head);
                ApplyRope(k, table, layer.KvHeads, tokens, layer.Head);
                _cudaProfiler?.End(CudaEmbeddingPhase.SplitNormRope);
                using var q4 = q.View(attentionBatch, attentionTokens, _heads, layer.Head);
                using var k4 = k.View(attentionBatch, attentionTokens, layer.KvHeads, layer.Head);
                using var v4 = v.View(attentionBatch, attentionTokens, layer.KvHeads, layer.Head);
                using var attention = new Tensor(_allocator, DType.Float32, attentionBatch, attentionTokens, _heads, layer.Head);
                // The measured short-input path keeps one streaming dispatch.
                // Bounded window matrices and global F32 products remain an
                // opt-in experiment; long local windows keep streaming.
                bool flash = false;
                // The streaming APIs reserve radius zero for global attention.
                // A full GGUF window of one instead means self-only attention
                // and needs the explicit local mask below.
                bool selfOnlyLocal = layer.Sliding && _window == 0;
                bool windowCoversSequences = !layer.Sliding || maximumLength - 1 <= _window;
                if (_cudaAttentionPlan != null && maximumLength >= 256 && !selfOnlyLocal)
                {
                    if (!layer.Sliding)
                    {
                        flash = CudaEmbeddingAttentionOps.TryGlobalAttention(attention, q4, k4, v4,
                            _cudaAttentionPlan, lengths, 1, cancellationToken);
                        if (!flash) throw new NotSupportedException("Requested CUDA materialized global attention rejected this geometry.");
                    }
                    else if (maximumLength <= CudaEmbeddingAttentionPlan.MaximumWindowSequence)
                        // A batch with more mask geometries than the bounded
                        // cache can hold safely falls back before dispatch.
                        flash = CudaEmbeddingAttentionOps.TryWindowAttention(attention, q4, k4, v4,
                            _cudaAttentionPlan, lengths, _window, 1, cancellationToken);
                }
                if (!flash && _allocator is CudaAllocator && !selfOnlyLocal)
                    flash = CudaFusedOps.TryEmbeddingAttention(attention, q4, k4, v4,
                        attentionBatch, _heads, layer.KvHeads, layer.Head, attentionTokens, layer.Sliding ? _window : 0, 1, sequenceBounds);
                if (!flash && _allocator is MlxAllocator && !selfOnlyLocal && (segmentedMlxAttention || equal && (attentionTokens >= 4096 || layer.Sliding && attentionTokens >= 1536)))
                    flash = MlxEmbeddingOps.TryAttention(attention, q4, k4, v4, layer.Sliding ? _window : 0, 1, _mlxAttentionPlan,
                        useFloat16Operands: layer.HalfAttentionSafe, sequenceLengths: segmentedMlxAttention ? lengths : null);
                if (!flash && !equal && _allocator is MlxAllocator && attentionTokens <= 1536 && maximumLength <= 512 && windowCoversSequences)
                    // Single short requests use F32 attention. Keep the same
                    // operands in packed batches: fitting F16 is an overflow
                    // proof, not a proof of batch/single numerical consistency.
                    flash = MlxRaggedEmbeddingOps.TryAttention(attention, q4, k4, v4, lengths, 1, _mlxRaggedAttentionPlan,
                        useFloat16Operands: false);
                if (!flash)
                {
                    if (!equal && fullMask == null) fullMask = BuildMask(lengths, equal, local: false);
                    if (layer.Sliding && localMask == null && maximumLength - 1 > _window)
                        localMask = BuildMask(lengths, equal, local: true);
                    if (!equal && _allocator is MlxAllocator && attentionTokens <= 1536 && maximumLength <= 512 && layer.HalfAttentionSafe)
                        flash = MlxEmbeddingOps.TryMaskedAttention(attention, q4, k4, v4,
                            layer.Sliding ? localMask ?? fullMask : fullMask, 1, useFloat16Operands: false);
                }
                if (!flash)
                {
                    using var expandedK = _allocator is CudaAllocator && _heads != layer.KvHeads
                        ? Ops.RepeatInterleave(null, k4, _heads / layer.KvHeads, 2) : k4.CopyRef();
                    using var expandedV = _allocator is CudaAllocator && _heads != layer.KvHeads
                        ? Ops.RepeatInterleave(null, v4, _heads / layer.KvHeads, 2) : v4.CopyRef();
                    Ops.ScaledDotProductAttention(attention, q4, expandedK, expandedV,
                        layer.Sliding ? localMask ?? fullMask : fullMask, 1);
                }
                _cudaProfiler?.End(layer.Sliding ? CudaEmbeddingPhase.LocalAttention : CudaEmbeddingPhase.GlobalAttention);
                using var attentionRows = attention.View(tokens, qWidth);
                using var attentionProjected = Linear(attentionRows, layer.Attention, packedMlx);
                AddNormResidual(hidden, attentionProjected, layer.AttentionPostNorm);
                _cudaProfiler?.End(CudaEmbeddingPhase.AttentionOutputResidual);
                using var ffnNormed = _cudaProjectionPlan == null
                    ? Ops.RMSNorm(null, hidden, layer.FfnNorm, null, _epsilon)
                    : new Tensor(_allocator, DType.Float32, tokens, _dim);
                using var gateUp = _cudaProjectionPlan == null
                    ? Linear(ffnNormed, layer.GateUp, packedMlx, layer.HalfGateUpSafe)
                    : CudaNormalizedProjection(hidden, ffnNormed, layer.GateUp, cancellationToken);
                using var gated = new Tensor(_allocator, DType.Float32, tokens, _ff);
                bool fused = _allocator is MlxAllocator ? MlxFusedOps.TryGeluMulSplit(gated, gateUp, _ff)
                    : CudaFusedOps.TryGELUMulSplit(gated, gateUp, _ff);
                if (!fused) throw new NotSupportedException("The selected direct backend does not provide the required F32 GELU kernel.");
                using var ffn = Linear(gated, layer.Down, packedMlx);
                AddNormResidual(hidden, ffn, layer.FfnPostNorm);
                _cudaProfiler?.End(CudaEmbeddingPhase.FeedForward);
                using var pleGate = Linear(hidden, layer.Gate, packedMlx);
                Ops.GELU(pleGate, pleGate);
                using var pleView = ple.Select(1, i);
                using var pleRows = Ops.AsContiguous(pleView);
                Ops.Mul(pleGate, pleGate, pleRows);
                using var pleProjected = Linear(pleGate, layer.Projection, packedMlx);
                AddNormResidual(hidden, pleProjected, layer.PleNorm);
                Ops.Mul(hidden, hidden, layer.Scale);
                _cudaProfiler?.End(CudaEmbeddingPhase.PleResidual);
            }
            Ops.RMSNorm(hidden, hidden, _outputNorm, null, _epsilon);
            // A reduction through the direct backend's GEMM avoids the CUDA
            // generic Mean operation, which currently downloads to a CPU fallback.
            using var averaging = new Tensor(_allocator, DType.Float32, batch, tokens);
            var factors = new float[checked(batch * tokens)];
            tokenOffset = 0;
            for (int s = 0; s < batch; ++s)
            {
                Array.Fill(factors, 1f / lengths[s], s * tokens + tokenOffset, lengths[s]);
                tokenOffset += lengths[s];
            }
            averaging.SetElementsAsFloat(factors);
            using var mean = new Tensor(_allocator, DType.Float32, batch, _dim);
            if (!TryCudaLinear(mean, averaging, hidden))
                Ops.Addmm(mean, 0, mean, 1, averaging, hidden);
            using var projected = Linear(mean, _outputProjection);
            _cudaProfiler?.End(CudaEmbeddingPhase.FinalPoolHead);
            float[] values = projected.GetElementsAsFloat(checked(batch * _out));
            _cudaAttentionPlan?.NotifyReadbackCompleted();
            _cudaProjectionPlan?.NotifyReadbackCompleted();
            _cudaProfiler?.FinishStep();
            for (int s = 0; s < batch; ++s)
            {
                double norm = 0;
                for (int d = 0; d < _out; ++d)
                {
                    float value = values[s * _out + d];
                    if (!float.IsFinite(value)) throw new InvalidOperationException("Direct EmbeddingGemma 2 produced non-finite values.");
                    norm += (double)value * value;
                }
                if (norm <= 0) throw new InvalidOperationException("Direct EmbeddingGemma 2 produced a zero embedding.");
                float inverse = (float)(1 / Math.Sqrt(norm));
                for (int d = 0; d < _out; ++d) output[outputOffset + s * _out + d] = values[s * _out + d] * inverse;
            }
        }
        finally
        {
            foreach (var table in ropeTables.Values) { table.Cos.Dispose(); table.Sin.Dispose(); }
            fullMask?.Dispose(); localMask?.Dispose();
        }
    }

    private Tensor BuildSequenceBounds(int[] lengths, int tokens)
    {
        var values = new int[checked(tokens * 2)];
        int offset = 0;
        foreach (int length in lengths)
        {
            for (int t = offset; t < offset + length; ++t) { values[t * 2] = offset; values[t * 2 + 1] = offset + length; }
            offset += length;
        }
        var result = new Tensor(_allocator, DType.Int32, 1, tokens, 2);
        try { result.SetElementsAsInt(values); return result; }
        catch { result.Dispose(); throw; }
    }

    private Tensor BuildMask(int[] lengths, bool equal, bool local)
    {
        string key = $"{equal}:{local}:{string.Join(",", lengths)}";
        if (_maskCache.TryGetValue(key, out var cached)) return cached.CopyRef();
        int total = 0;
        foreach (int length in lengths) total += length;
        int tokens = equal ? lengths[0] : total, batch = equal ? lengths.Length : 1;
        int[] sequence = new int[tokens], positions = new int[tokens];
        if (!equal)
        {
            int offset = 0;
            for (int s = 0; s < lengths.Length; ++s)
            {
                for (int t = 0; t < lengths[s]; ++t) { sequence[offset + t] = s; positions[offset + t] = t; }
                offset += lengths[s];
            }
        }
        else for (int t = 0; t < tokens; ++t) positions[t] = t;
        int maskHeads = _allocator is CudaAllocator ? _heads : 1;
        var values = new float[checked(batch * maskHeads * tokens * tokens)];
        for (int b = 0; b < batch; ++b)
            for (int h = 0; h < maskHeads; ++h)
            for (int q = 0; q < tokens; ++q)
                for (int k = 0; k < tokens; ++k)
                    if (sequence[q] != sequence[k] || (local && Math.Abs(positions[q] - positions[k]) > _window))
                        values[((b * maskHeads + h) * tokens + q) * tokens + k] = float.NegativeInfinity;
        var result = new Tensor(_allocator, DType.Float32, batch, maskHeads, tokens, tokens);
        result.SetElementsAsFloat(values);
        // Keep common short-batch geometries, but avoid retaining huge masks
        // for uncommon near-context-limit requests.
        if (values.Length <= 4 * 1024 * 1024)
        {
            if (_maskCache.Count >= 8) { foreach (var tensor in _maskCache.Values) tensor.Dispose(); _maskCache.Clear(); }
            _maskCache.Add(key, result.CopyRef());
        }
        return result;
    }

    private (Tensor Cos, Tensor Sin) BuildRope(int[] positions, int head, float basis, string geometry)
    {
        string key = $"{head}:{basis:R}:{geometry}";
        if (_ropeCache.TryGetValue(key, out var cached)) return (cached.Cos.CopyRef(), cached.Sin.CopyRef());
        int tokens = positions.Length;
        int half = head / 2;
        var frequencies = new double[half];
        for (int d = 0; d < half; ++d) frequencies[d] = Math.Pow(basis, -2.0 * d / head);
        var cos = new float[checked(tokens * half)];
        var sin = new float[cos.Length];
        for (int t = 0; t < tokens; ++t)
            for (int d = 0; d < half; ++d)
            {
                double angle = positions[t] * frequencies[d];
                cos[t * half + d] = (float)Math.Cos(angle);
                sin[t * half + d] = (float)Math.Sin(angle);
            }
        var c = new Tensor(_allocator, DType.Float32, checked(tokens * half));
        var s = new Tensor(_allocator, DType.Float32, checked(tokens * half));
        c.SetElementsAsFloat(cos); s.SetElementsAsFloat(sin);
        if (_ropeCache.Count >= 8)
        { foreach (var table in _ropeCache.Values) { table.Cos.Dispose(); table.Sin.Dispose(); } _ropeCache.Clear(); }
        _ropeCache.Add(key, (c.CopyRef(), s.CopyRef()));
        return (c, s);
    }

    private void ApplyRope(Tensor values, (Tensor Cos, Tensor Sin) table, int heads, int tokens, int head)
    {
        bool success = _allocator is MlxAllocator
            ? MlxFusedOps.TryNeoXRoPEFlatInPlace(values, table.Cos, table.Sin, heads, tokens, head, head / 2)
            : CudaFusedOps.TryNeoxRopeFlat(values, table.Cos, table.Sin, heads, tokens, head, head / 2);
        if (!success) throw new NotSupportedException("The selected direct backend does not provide the required NeoX rotary kernel.");
    }

    private Tensor Linear(Tensor input, Tensor weight, bool packedPrefill = false, bool boundedNormalizedOperands = false)
    {
        var result = new Tensor(_allocator, DType.Float32, input.Sizes[0], weight.Sizes[1]);
        try
        {
            if (_compressed.TryGetValue(weight, out var compressed))
            {
                if (input.Sizes[0] >= 256 && (input.Sizes[0] <= 768 || packedPrefill))
                {
                    // Dense F32 GEMM has higher prefill throughput on Apple
                    // GPUs. Materialize its exact GGUF values once; short inputs
                    // keep the compressed bandwidth advantage and F32 operands.
                    if (compressed.DensePrefill == null)
                    {
                        var rows = new float[checked(compressed.Input * compressed.Output)];
                        ManagedQuantizedOps.DequantizeToFloat32((int)compressed.Type, (byte[])compressed.Pinned.Target, 0, rows, 0, rows.Length);
                        var transposed = new float[rows.Length];
                        for (int row = 0; row < compressed.Output; ++row)
                            for (int column = 0; column < compressed.Input; ++column)
                                transposed[column * compressed.Output + row] = rows[row * compressed.Input + column];
                        var dense = new Tensor(_allocator, DType.Float32, compressed.Input, compressed.Output);
                        _weights.Add(dense); dense.SetElementsAsFloat(transposed); compressed.DensePrefill = dense;
                    }
                    if (boundedNormalizedOperands && input.Sizes[0] <= 768)
                    {
                        // Only post-RMS QKV and gate/up projections enter this
                        // path. Constructor bounds cover input components and
                        // every complete or partial dot, including F16 weight
                        // rounding. Raw hidden/PLE and down projections stay F32.
                        if (compressed.DenseHalfPrefill == null)
                        {
                            var half = new Tensor(_allocator, DType.Float16, compressed.Input, compressed.Output);
                            _weights.Add(half); Ops.Copy(half, compressed.DensePrefill); compressed.DenseHalfPrefill = half;
                        }
                        if (!MlxEmbeddingOps.TryNormalizedHalfProjection(result, input, compressed.DenseHalfPrefill))
                            throw new NotSupportedException("MLX could not execute a bounded normalized projection.");
                    }
                    else Ops.Addmm(result, 0, result, 1, input, compressed.DensePrefill);
                }
                else if (!MlxQuantizedOps.TryAddmmQuantizedToFloat32(result, input, compressed.Pointer, compressed.Pointer,
                    (int)compressed.Type, compressed.Input, compressed.Output, compressed.Bytes, preserveFloat32: true))
                    throw new NotSupportedException("MLX could not execute the selected compressed F32 projection.");
            }
            else if (!TryCudaLinear(result, input, weight))
                Ops.Addmm(result, 0, result, 1, input, weight);
            return result;
        }
        catch { result.Dispose(); throw; }
    }

    private bool TryCudaLinear(Tensor result, Tensor input, Tensor weight)
    {
        if (_allocator is not CudaAllocator) return false;
        // TF32 mantissa rounding can change short-input embeddings with the
        // projection's batch shape, including the published window. Use the
        // same pedantic F32 arithmetic for every projection and mean-pooling
        // GEMM, scoped to this call without changing global cuBLAS policy.
        return CudaEmbeddingOps.TryLinearFloat32(result, input, weight);
    }

    private void AddNormResidual(Tensor residual, Tensor input, Tensor weight)
    {
        bool fused = _allocator is MlxAllocator
            ? MlxFusedOps.TryRmsNormAddInPlace(residual, input, weight, _epsilon)
            : CudaFusedOps.TryRmsNormResidualAdd(residual, input, weight, _epsilon);
        if (fused) return;
        Ops.RMSNorm(input, input, weight, null, _epsilon);
        Ops.Add(residual, residual, input);
    }

    private Tensor Projection(GgufFile file, string name, int input, int output)
        => JoinedProjection(file, input, [name], [output]);

    private void PrepareCudaNormalizedProjection(GgufFile file, Tensor weight, string norm, string[] projections)
    {
        if (_cudaProjectionPlan == null) return;
        int elements = 0;
        foreach (string name in projections)
            elements = checked(elements + (int)Require(file, name, _dim).NumElements);
        var rowMajor = new float[elements];
        int offset = 0;
        foreach (string name in projections)
        {
            var values = Read(file, name, checked((int)Require(file, name, _dim).NumElements));
            values.CopyTo(rowMajor, offset); offset += values.Length;
        }
        if (!_cudaProjectionPlan.TryPrepareWeight(Read(file, norm, _dim), rowMajor, _dim, out var prepared))
            throw new NotSupportedException($"The requested CUDA normalized Half projection rejected '{norm}' or its weights.");
        _cudaNormalizedProjections.Add(weight, prepared);
    }

    private Tensor CudaNormalizedProjection(Tensor rawInput, Tensor normalizedOutput, Tensor weight,
        CancellationToken cancellationToken)
    {
        if (!_cudaNormalizedProjections.TryGetValue(weight, out var prepared))
            throw new InvalidOperationException("The requested CUDA normalized projection has no immutable descriptor.");
        var result = new Tensor(_allocator, DType.Float32, rawInput.Sizes[0], weight.Sizes[1]);
        try
        {
            // The helper owns F32 RMSNorm and the subsequent Half conversion.
            // An arbitrary tensor cannot bypass normalization or its range proof.
            if (!_cudaProjectionPlan.TryNormalizeAndProject(result, normalizedOutput, rawInput,
                    prepared, _epsilon, cancellationToken))
                throw new NotSupportedException("The requested CUDA normalized Half projection rejected the execution geometry.");
            return result;
        }
        catch { result.Dispose(); throw; }
    }

    private Tensor JoinedProjection(GgufFile file, int input, string[] names, int[] outputs)
    {
        int total = 0;
        foreach (int output in outputs) total = checked(total + output);
        var infos = new GgufTensorInfo[names.Length];
        bool sameType = true;
        for (int i = 0; i < names.Length; ++i)
        {
            infos[i] = Require(file, names[i], input, outputs[i]);
            sameType &= infos[i].Type == infos[0].Type;
        }
        if (_allocator is MlxAllocator mlx && sameType && infos[0].Type == GgmlTensorType.Q8_0)
        {
            // Concatenating output rows preserves exact GGUF Q8 blocks for
            // fused QKV and gate/up projections. The metadata tensor remains
            // lazy; no dense F32 weight allocation is made on this path.
            int rowBytes = checked((int)ManagedQuantizedOps.RowSize((int)infos[0].Type, input));
            var raw = new byte[checked(rowBytes * total)];
            int byteOffset = 0;
            foreach (var info in infos)
            {
                byte[] data = file.ReadTensorData(info);
                if (data.Length != checked(rowBytes * (int)info.Shape[1]))
                    throw new InvalidDataException("Compressed projection byte count does not match its rows.");
                data.CopyTo(raw, byteOffset); byteOffset += data.Length;
            }
            var compressedTensor = new Tensor(_allocator, DType.Float32, input, total);
            _weights.Add(compressedTensor);
            var compressed = new CompressedWeight(raw, infos[0].Type, input, total);
            _compressed.Add(compressedTensor, compressed);
            MlxQuantizedOps.PreloadQuantizedWeight(mlx, compressed.Pointer, compressed.Pointer,
                (int)compressed.Type, input, total, raw.Length);
            return compressedTensor;
        }
        var values = new float[checked(input * total)];
        int column = 0;
        for (int i = 0; i < names.Length; ++i)
        {
            Require(file, names[i], input, outputs[i]);
            var rowMajor = Read(file, names[i], checked(input * outputs[i]));
            for (int row = 0; row < outputs[i]; ++row)
                for (int d = 0; d < input; ++d)
                    values[d * total + column + row] = rowMajor[row * input + d];
            column += outputs[i];
        }
        var result = new Tensor(_allocator, DType.Float32, input, total);
        _weights.Add(result);
        result.SetElementsAsFloat(values);
        return result;
    }

    private Tensor Vector(GgufFile file, string name, int count)
    {
        var values = Read(file, name, count);
        var result = new Tensor(_allocator, DType.Float32, count);
        _weights.Add(result);
        result.SetElementsAsFloat(values);
        return result;
    }

    private static float[] Read(GgufFile file, string name, int count)
    {
        if (!file.Tensors.TryGetValue(name, out var info) || info.NumElements != count)
            throw new InvalidDataException($"EmbeddingGemma 2 tensor '{name}' has invalid dimensions.");
        var values = new float[count];
        ManagedQuantizedOps.DequantizeToFloat32((int)info.Type, file.ReadTensorData(info), 0, values, 0, count);
        return values;
    }

    private static bool FitsHalfAttention(GgufFile file, string qName, string kName, int head)
        => CanUseHalfAttention(Read(file, qName, head), Read(file, kName, head), head);

    internal static bool CanUseHalfAttention(ReadOnlySpan<float> qWeights, ReadOnlySpan<float> kWeights, int head)
    {
        if (head <= 0 || qWeights.Length != head || kWeights.Length != head) return false;
        // Weighted RMS-normalized head norms are at most sqrt(head)*max(|w|).
        // Cauchy-Schwarz therefore bounds every full or partial QK dot by
        // head*max(|q_w|)*max(|k_w|). The unfused MLX path stores that score
        // in F16; bound it too, with a fourfold margin from overflow.
        double qMax = 0, kMax = 0;
        foreach (float value in qWeights)
        { if (!float.IsFinite(value)) return false; qMax = Math.Max(qMax, Math.Abs(value)); }
        foreach (float value in kWeights)
        { if (!float.IsFinite(value)) return false; kMax = Math.Max(kMax, Math.Abs(value)); }
        return qMax * Math.Sqrt(head) <= 16384 && kMax * Math.Sqrt(head) <= 16384 && head * qMax * kMax <= 16384;
    }

    private static bool FitsHalfProjection(GgufFile file, string norm, int input, string[] projections)
    {
        var gains = Read(file, norm, input);
        foreach (string name in projections)
        {
            var info = Require(file, name, input);
            if (!CanUseHalfProjection(gains, Read(file, name, checked((int)info.NumElements)), input)) return false;
        }
        return true;
    }

    internal static bool CanUseHalfProjection(ReadOnlySpan<float> gains, ReadOnlySpan<float> rowMajorWeights, int input)
    {
        if (input <= 0 || gains.Length != input || rowMajorWeights.Length == 0 || rowMajorWeights.Length % input != 0) return false;
        double maxGain = 0;
        foreach (float gain in gains)
        { if (!float.IsFinite(gain)) return false; maxGain = Math.Max(maxGain, Math.Abs(gain)); }
        // RMS-normalized rows have L2 norm <=sqrt(input). Cauchy-Schwarz
        // bounds each full or partial projection dot. A factor of two covers
        // input narrowing and norm/reduction rounding, with a further fourfold
        // margin below F16 overflow. Include absolute F16 subnormal rounding.
        if (2 * Math.Sqrt(input) * maxGain > 16384) return false;
        for (int row = 0; row < rowMajorWeights.Length; row += input)
        {
            double squared = 0, absoluteHalfWeights = 0;
            for (int column = 0; column < input; ++column)
            {
                float value = rowMajorWeights[row + column];
                if (!float.IsFinite(value) || Math.Abs(value) > 16384) return false;
                double half = (double)(System.Half)value;
                double weighted = gains[column] * half;
                squared += weighted * weighted;
                absoluteHalfWeights += Math.Abs(half);
            }
            if (2 * Math.Sqrt(input * squared) + Math.ScaleB(absoluteHalfWeights, -24) > 16384) return false;
        }
        return true;
    }

    private static GgufTensorInfo Require(GgufFile file, string name, int input, int output = 0)
    {
        if (!file.Tensors.TryGetValue(name, out var info) || info.Shape.Length != 2 || info.Shape[0] != (ulong)input ||
            (output != 0 && info.Shape[1] != (ulong)output))
            throw new InvalidDataException($"EmbeddingGemma 2 projection '{name}' has invalid dimensions.");
        return info;
    }

    private static int Positive(GgufFile file, string key)
    {
        int value = checked((int)file.GetUint32(key));
        if (value <= 0) throw new InvalidDataException($"EmbeddingGemma 2 is missing '{key}'.");
        return value;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        try { _cudaProfiler?.Dispose(); }
        finally
        {
            try
            {
                if (_cudaAttentionPlan != null)
                    Console.Error.WriteLine("EMBEDDING_CUDA_ATTENTION " + System.Text.Json.JsonSerializer.Serialize(new
                    {
                        compute_type = _cudaAttentionPlan.SelectedComputeType,
                        compute_policy = _cudaAttentionPlan.ComputePolicy.ToString(),
                        head_tile = _cudaAttentionPlan.HeadTile,
                        score_budget_bytes = _cudaAttentionPlan.ScoreBudgetBytes,
                        score_capacity_bytes = _cudaAttentionPlan.ScoreCapacityBytes,
                        calls = _cudaAttentionPlan.SuccessfulCalls,
                        segments = _cudaAttentionPlan.SuccessfulSegments,
                        head_tiles = _cudaAttentionPlan.SuccessfulHeadTiles,
                        query_rows = _cudaAttentionPlan.SuccessfulQueryRows,
                        window_calls = _cudaAttentionPlan.SuccessfulWindowCalls,
                        mask_entries = _cudaAttentionPlan.MaskEntries,
                        mask_capacity_bytes = _cudaAttentionPlan.MaskCapacityBytes,
                        mask_cache_hits = _cudaAttentionPlan.MaskCacheHits,
                        mask_uploads = _cudaAttentionPlan.MaskUploads,
                        mask_evictions = _cudaAttentionPlan.MaskEvictions,
                    }));
            }
            finally
            {
                try { _cudaAttentionPlan?.Dispose(); }
                finally
                {
                    try
                    {
                        if (_cudaProjectionPlan != null)
                            Console.Error.WriteLine("EMBEDDING_CUDA_NORMALIZED_PROJECTIONS " + System.Text.Json.JsonSerializer.Serialize(new
                            {
                                compute_type = _cudaProjectionPlan.ComputeType,
                                prepared_weights = _cudaProjectionPlan.PreparedWeightCount,
                                weight_capacity_bytes = _cudaProjectionPlan.WeightCapacityBytes,
                                workspace_capacity_bytes = _cudaProjectionPlan.WorkspaceCapacityBytes,
                                calls = _cudaProjectionPlan.SuccessfulCalls,
                                rows = _cudaProjectionPlan.SuccessfulRows,
                                weight_uses = _cudaProjectionPlan.WeightUses,
                                workspace_reuses = _cudaProjectionPlan.WorkspaceReuses,
                                proof_rejections = _cudaProjectionPlan.ProofRejections,
                                budget_rejections = _cudaProjectionPlan.BudgetRejections,
                            }));
                    }
                    finally
                    {
                        try { _cudaProjectionPlan?.Dispose(); }
                        finally { _cudaNormalizedProjections.Clear(); DisposeWeights(); }
                    }
                }
            }
        }
    }

    private void DisposeWeights()
    {
        _mlxAttentionPlan?.Dispose();
        _mlxRaggedAttentionPlan?.Dispose();
        foreach (var table in _ropeCache.Values) { table.Cos.Dispose(); table.Sin.Dispose(); }
        _ropeCache.Clear();
        foreach (var mask in _maskCache.Values) mask.Dispose();
        _maskCache.Clear();
        foreach (var compressed in _compressed.Values)
        {
            try { MlxQuantizedOps.ReleaseQuantizedWeight((MlxAllocator)_allocator, compressed.Pointer); }
            finally { compressed.Pinned.Free(); }
        }
        _compressed.Clear();
        foreach (var tensor in _weights) tensor.Dispose();
        _weights.Clear();
        _tokenCache.Clear();
        _tokenData = null;
    }
}
