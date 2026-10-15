// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading;
using System.Text.Json;
using TensorSharp.Runtime;
using Matrix = TensorSharp.Models.Embeddings.ManagedEmbeddingEncoder.Matrix;

namespace TensorSharp.Models.Embeddings;

/// <summary>Managed EmbeddingGemma 2 graph with F32 hidden states. Linear weights
/// expand to F32 unless the experimental packed Q8 kernel is explicitly enabled.
/// The vocabulary stays compressed. Explicit CPU attention and Q16 projection experiments
/// can call stateless owned native math; the portable fallback remains managed and
/// no GGML inference library is initialized.</summary>
internal sealed class ManagedEmbeddingGemma2Encoder : IDisposable
{
    private sealed record Layer(int Head, int Kv, bool Local, Matrix Qkv, Matrix O,
        Matrix GateUp, Matrix Down, Matrix InputGate, Matrix Projection,
        float[] AttentionNorm, float[] QNorm, float[] KNorm, float[] PostAttentionNorm,
        float[] FfnNorm, float[] PostFfnNorm, float[] PostNorm, float Scale);
    private readonly int _dim, _output, _pleDim, _heads, _ff, _window;
    private readonly float _eps, _localBase, _globalBase;
    private Matrix _tokens, _ple, _projection;
    private float[] _pleNorm, _outputNorm;
    private Layer[] _layers;
    private readonly CpuWorkerPool _pool;
    private readonly OwnedEmbeddingCpuProjection? _nativeProjection;
    private readonly string? _nativeProjectionFallback;
    private readonly bool _nativeProjectionRequested = Environment.GetEnvironmentVariable("TENSORSHARP_EMBEDDING_NATIVE_Q16") == "1";
    private readonly OwnedEmbeddingCpuAttention? _nativeAttention;
    private readonly string? _nativeAttentionFallback;
    private readonly bool _nativeAttentionRequested = Environment.GetEnvironmentVariable("TENSORSHARP_EMBEDDING_NATIVE_ATTENTION") == "1";
    private long _nativeAttentionTiles, _nativeAttentionRows, _managedAttentionTiles;
    private readonly bool _fusedGateGelu = Environment.GetEnvironmentVariable("TENSORSHARP_EMBEDDING_FUSED_GATE_GELU") == "1";
    private long _fusedGateGeluCalls, _fusedGateGeluRows;
    private float[] _x = [], _normalized = [], _next = [], _qkv = [], _q = [], _k = [], _v = [], _attn = [], _gateUp = [], _gate = [], _up = [], _pleInputs = [], _pleGate = [], _localK = [], _localV = [];
    private CancellationToken _cancellation;
    private readonly bool _profileEnabled = Environment.GetEnvironmentVariable("TENSORSHARP_EMBEDDING_CPU_PROFILE") == "1";
    private long _profileTicks, _profileCalls, _profileRows;
    private enum Phase
    {
        BufferLayout, PleScaling, PleNormalization, RmsNormalization, Activation,
        ResidualGating, FusedGateGelu, RopeTables, Rope, AttentionBuffers, AttentionClear, AttentionPacking, AttentionCompute, Pooling,
        FinalNormalization, OutputNormalization, Count,
    }
    private readonly long[] _phaseTicks = new long[(int)Phase.Count], _phaseCalls = new long[(int)Phase.Count], _phaseRows = new long[(int)Phase.Count];
    private readonly long[] _pendingTicks = new long[(int)Phase.Count], _pendingCalls = new long[(int)Phase.Count], _pendingRows = new long[(int)Phase.Count];
    private long _inputTicks, _inputCalls, _inputRows, _tokenRowTicks, _tokenRowCalls, _tokenRowRows;
    private readonly Dictionary<(int Head, float Base), (int Length, float[] Sin, float[] Cos)> _ropeTables = new();

    // Callers serialize encoder execution. Phase records are committed only
    // after successful Compute; scopes allocate no objects or emit any output.
    private readonly struct PhaseTimer : IDisposable
    {
        private readonly ManagedEmbeddingGemma2Encoder _owner;
        private readonly Phase _phase;
        private readonly int _rows;
        private readonly long _start;
        internal PhaseTimer(ManagedEmbeddingGemma2Encoder owner, Phase phase, int rows)
        {
            _owner = owner._profileEnabled ? owner : null;
            _phase = phase; _rows = rows;
            _start = _owner != null ? Stopwatch.GetTimestamp() : 0;
        }
        public void Dispose()
        {
            if (_owner == null) return;
            int index = (int)_phase;
            _owner._pendingTicks[index] += Stopwatch.GetTimestamp() - _start;
            ++_owner._pendingCalls[index]; _owner._pendingRows[index] += _rows;
        }
    }
    private PhaseTimer Profile(Phase phase, int rows) => new(this, phase, rows);

    public ManagedEmbeddingGemma2Encoder(GgufFile file, int threads)
    {
        const string p = "gemma-embedding2.";
        _dim = Positive(file, p + "embedding_length");
        _output = Positive(file, p + "embedding_length_out");
        _pleDim = Positive(file, p + "embedding_length_per_layer_input");
        _heads = Positive(file, p + "attention.head_count");
        _ff = Positive(file, p + "feed_forward_length");
        _window = Positive(file, p + "attention.sliding_window") / 2;
        int layers = Positive(file, p + "block_count");
        int globalHead = Positive(file, p + "attention.key_length"), localHead = Positive(file, p + "attention.key_length_swa");
        EmbeddingGemma2TextMetadata.Validate(file, globalHead, localHead);
        if (_dim > 16384 || _output > 16384 || _pleDim > 16384 || _ff > 65536 || layers > 128 ||
            _heads > 128 || globalHead > 1024 || localHead > 1024 || globalHead % ManagedEmbeddingMath.ValueWidth != 0 || localHead % ManagedEmbeddingMath.ValueWidth != 0)
            throw new InvalidDataException("Invalid EmbeddingGemma 2 encoder geometry, attention or pooling metadata.");
        _eps = file.GetFloat32(p + "attention.layer_norm_rms_epsilon", 1e-6f);
        _localBase = file.GetFloat32(p + "rope.freq_base_swa", 10000);
        _globalBase = file.GetFloat32(p + "rope.freq_base", 1000000);
        if (!float.IsFinite(_eps) || _eps <= 0 || !float.IsFinite(_localBase) || _localBase <= 0 || !float.IsFinite(_globalBase) || _globalBase <= 0)
            throw new InvalidDataException("Invalid EmbeddingGemma 2 norm epsilon or rotary base.");
        var local = file.GetBoolArray(p + "attention.sliding_window_pattern")
            ?? throw new InvalidDataException("EmbeddingGemma 2 requires a sliding-window layer pattern.");
        var kv = file.GetInt32Array(p + "attention.head_count_kv")
            ?? throw new InvalidDataException("EmbeddingGemma 2 requires per-layer KV head counts.");
        if (local.Length != layers || kv.Length != layers) throw new InvalidDataException("Attention metadata length must match the layer count.");
        _nativeProjection = OwnedEmbeddingCpuProjection.TryCreate(out _nativeProjectionFallback);
        try
        {
            _tokens = ManagedEmbeddingEncoder.ReadMatrix(file, "token_embd.weight", _dim);
            _ple = ReadLinear(file, "per_layer_model_proj", _dim, checked(_pleDim * layers));
            _projection = ReadLinear(file, "output", _dim, _output);
            _pleNorm = ReadVector(file, "per_layer_proj_norm", _pleDim);
            _outputNorm = ReadVector(file, "output_norm", _dim);
            _layers = new Layer[layers];
            for (int i = 0; i < layers; ++i)
            {
                string b = $"blk.{i}.";
                int head = local[i] ? localHead : globalHead;
                if (kv[i] <= 0 || _heads % kv[i] != 0) throw new InvalidDataException("Invalid EmbeddingGemma 2 GQA head count.");
                _layers[i] = new(head, kv[i], local[i],
                    JoinedLinear(file, _dim, [b + "attn_q", b + "attn_k", b + "attn_v"], [head * _heads, head * kv[i], head * kv[i]]),
                    ReadLinear(file, b + "attn_output", head * _heads, _dim),
                    JoinedLinear(file, _dim, [b + "ffn_gate", b + "ffn_up"], [_ff, _ff]), ReadLinear(file, b + "ffn_down", _ff, _dim),
                    ReadLinear(file, b + "inp_gate", _dim, _pleDim), ReadLinear(file, b + "proj", _pleDim, _dim),
                    ReadVector(file, b + "attn_norm", _dim), ReadVector(file, b + "attn_q_norm", head), ReadVector(file, b + "attn_k_norm", head),
                    ReadVector(file, b + "post_attention_norm", _dim), ReadVector(file, b + "ffn_norm", _dim),
                    ReadVector(file, b + "post_ffw_norm", _dim), ReadVector(file, b + "post_norm", _dim),
                    ReadVector(file, b + "layer_output_scale", 1)[0]);
            }
            _nativeAttention = OwnedEmbeddingCpuAttention.TryCreate(out _nativeAttentionFallback);
            _pool = new CpuWorkerPool(threads == 0 ? 4 : threads);
        }
        catch
        {
            try { _nativeAttention?.Dispose(); }
            finally { _nativeProjection?.Dispose(); }
            throw;
        }
    }

    private static int Positive(GgufFile file, string key)
    {
        int value = checked((int)file.GetUint32(key));
        return value > 0 ? value : throw new InvalidDataException("Invalid or missing metadata: " + key);
    }
    private Matrix ReadLinear(GgufFile file, string name, int input, int output)
    {
        var matrix = ManagedEmbeddingEncoder.ReadMatrix(file, name + ".weight", input, output);
        matrix.ProfileName = name;
        // These projections consume unnormalized hidden/media rows, where
        // activation quantization is particularly sensitive. Retain F32
        // arithmetic just as the output projection does.
        return PrepareLinear(matrix, requireFloat: name is "output" or "per_layer_model_proj" || name.EndsWith("inp_gate", StringComparison.Ordinal));
    }
    private Matrix JoinedLinear(GgufFile file, int input, string[] names, int[] outputs)
    {
        var matrices = new Matrix[names.Length];
        int total = 0;
        bool sameType = true;
        for (int i = 0; i < names.Length; ++i)
        {
            matrices[i] = ManagedEmbeddingEncoder.ReadMatrix(file, names[i] + ".weight", input, outputs[i]);
            total = checked(total + outputs[i]);
            sameType &= matrices[i].Type == matrices[0].Type;
        }
        var type = sameType ? matrices[0].Type : GgmlTensorType.F32;
        int rowBytes = checked((int)ManagedQuantizedOps.RowSize((int)type, input));
        var data = new byte[checked(rowBytes * total)];
        int offset = 0;
        foreach (var matrix in matrices)
        {
            if (sameType) Buffer.BlockCopy(matrix.Data, 0, data, offset, matrix.Data.Length);
            else
            {
                var values = new float[checked(input * matrix.Output)];
                ManagedQuantizedOps.DequantizeToFloat32((int)matrix.Type, matrix.Data, 0, values, 0, values.Length);
                Buffer.BlockCopy(values, 0, data, offset, values.Length * sizeof(float));
            }
            offset += checked(rowBytes * matrix.Output);
        }
        return PrepareLinear(new Matrix(data, type, input, total)
        {
            ProfileName = string.Join("+", names),
            ProfileSourceFormat = sameType ? type.ToString() : "mixed:" + string.Join(",", Array.ConvertAll(matrices, m => m.Type.ToString())),
        });
    }
    private Matrix PrepareLinear(Matrix matrix, bool requireFloat = false)
    {
        int input = matrix.Input, output = matrix.Output;
        if (!requireFloat && matrix.Type == GgmlTensorType.Q8_0 && ManagedEmbeddingQ8Q16Avx512Matrix.IsSupported &&
            Environment.GetEnvironmentVariable("TENSORSHARP_EMBEDDING_Q16_AVX512") == "1")
        {
            matrix.PrepareQ16Projection(_nativeProjection);
            return matrix;
        }
        if (!requireFloat && matrix.Type == GgmlTensorType.Q8_0 && ManagedEmbeddingQ8AvxMatrix.IsSupported &&
            Environment.GetEnvironmentVariable("TENSORSHARP_EMBEDDING_Q8_AVX") == "1")
        {
            matrix.PrepareAvxProjection();
            return matrix;
        }
        if (matrix.Type == GgmlTensorType.F32)
            return new Matrix(matrix.Data, matrix.Type, input, output)
            { UsePackedFloat = true, ProfileName = matrix.ProfileName, ProfileSourceFormat = matrix.ProfileSourceFormat };
        var values = new float[checked(input * output)];
        ManagedQuantizedOps.DequantizeToFloat32((int)matrix.Type, matrix.Data, 0, values, 0, values.Length);
        var data = new byte[checked(values.Length * sizeof(float))];
        Buffer.BlockCopy(values, 0, data, 0, data.Length);
        return new Matrix(data, GgmlTensorType.F32, input, output)
        { UsePackedFloat = true, ProfileName = matrix.ProfileName, ProfileSourceFormat = matrix.ProfileSourceFormat };
    }
    private static float[] ReadVector(GgufFile file, string name, int size)
    {
        var matrix = ManagedEmbeddingEncoder.ReadMatrix(file, name + ".weight", size, 1);
        var values = new float[size]; matrix.Row(0, values, 0);
        foreach (float value in values) if (!float.IsFinite(value)) throw new InvalidDataException("Non-finite parameter: " + name);
        return values;
    }
    private static void Ensure(ref float[] data, int count) { if (data.Length < count) data = new float[count]; }

    public void TokenRows(int[] tokens, float[] output)
    {
        long start = _profileEnabled ? Stopwatch.GetTimestamp() : 0;
        for (int i = 0; i < tokens.Length; ++i) _tokens.Row(tokens[i], output, i * _dim);
        TensorPrimitives.Multiply(output, MathF.Sqrt(_dim), output);
        if (_profileEnabled)
        {
            _tokenRowTicks += Stopwatch.GetTimestamp() - start;
            ++_tokenRowCalls; _tokenRowRows += tokens.Length;
        }
    }
    public void Encode(int[] tokens, int[] lengths, float[] output, CancellationToken cancellationToken)
    {
        long start = _profileEnabled ? Stopwatch.GetTimestamp() : 0;
        Ensure(ref _x, checked(tokens.Length * _dim));
        for (int i = 0; i < tokens.Length; ++i) _tokens.Row(tokens[i], _x, i * _dim);
        TensorPrimitives.Multiply(_x.AsSpan(0, tokens.Length * _dim), MathF.Sqrt(_dim), _x.AsSpan(0, tokens.Length * _dim));
        long elapsed = _profileEnabled ? Stopwatch.GetTimestamp() - start : 0;
        Compute(tokens.Length, lengths, output, cancellationToken);
        if (_profileEnabled) { _inputTicks += elapsed; ++_inputCalls; _inputRows += tokens.Length; }
    }
    public void EncodeInputs(float[] inputs, int[] lengths, float[] output, CancellationToken cancellationToken)
    {
        long start = _profileEnabled ? Stopwatch.GetTimestamp() : 0;
        Ensure(ref _x, inputs.Length); inputs.CopyTo(_x, 0);
        long elapsed = _profileEnabled ? Stopwatch.GetTimestamp() - start : 0;
        Compute(inputs.Length / _dim, lengths, output, cancellationToken);
        if (_profileEnabled) { _inputTicks += elapsed; ++_inputCalls; _inputRows += inputs.Length / _dim; }
    }
    private void Compute(int rows, int[] lengths, float[] output, CancellationToken cancellationToken)
    {
        long start = _profileEnabled ? Stopwatch.GetTimestamp() : 0;
        if (_profileEnabled) { Array.Clear(_pendingTicks); Array.Clear(_pendingCalls); Array.Clear(_pendingRows); }
        ComputeCore(rows, lengths, output, cancellationToken);
        if (_profileEnabled)
        {
            _profileTicks += Stopwatch.GetTimestamp() - start;
            ++_profileCalls; _profileRows += rows;
            for (int i = 0; i < (int)Phase.Count; ++i)
            {
                _phaseTicks[i] += _pendingTicks[i]; _phaseCalls[i] += _pendingCalls[i]; _phaseRows[i] += _pendingRows[i];
            }
        }
    }
    private void ComputeCore(int rows, int[] lengths, float[] output, CancellationToken cancellationToken)
    {
        _cancellation = cancellationToken;
        using (Profile(Phase.BufferLayout, rows))
        {
            Ensure(ref _normalized, rows * _dim); Ensure(ref _next, rows * _dim);
            Ensure(ref _pleInputs, checked(rows * _pleDim * _layers.Length));
        }
        _ple.Multiply(_x, rows, _pleInputs, _pool, cancellationToken);
        var ple = _pleInputs.AsSpan(0, rows * _pleDim * _layers.Length);
        using (Profile(Phase.PleScaling, rows)) TensorPrimitives.Multiply(ple, 1 / MathF.Sqrt(_dim), ple);
        Norm(_pleInputs, rows * _layers.Length, _pleDim, _pleNorm, Phase.PleNormalization);
        for (int layerIndex = 0; layerIndex < _layers.Length; ++layerIndex)
        {
            _cancellation.ThrowIfCancellationRequested();
            var layer = _layers[layerIndex];
            int qDim = _heads * layer.Head, kvDim = layer.Kv * layer.Head;
            using (Profile(Phase.BufferLayout, rows))
            {
                Ensure(ref _q, rows * qDim); Ensure(ref _k, rows * kvDim); Ensure(ref _v, rows * kvDim); Ensure(ref _attn, rows * qDim);
                Array.Copy(_x, _normalized, rows * _dim);
            }
            Norm(_normalized, rows, _dim, layer.AttentionNorm);
            int qkvDim = checked(qDim + 2 * kvDim);
            using (Profile(Phase.BufferLayout, rows)) Ensure(ref _qkv, checked(rows * qkvDim));
            layer.Qkv.Multiply(_normalized, rows, _qkv, _pool, cancellationToken);
            using (Profile(Phase.BufferLayout, rows))
            for (int row = 0; row < rows; ++row)
            {
                Array.Copy(_qkv, row * qkvDim, _q, row * qDim, qDim);
                Array.Copy(_qkv, row * qkvDim + qDim, _k, row * kvDim, kvDim);
                Array.Copy(_qkv, row * qkvDim + qDim + kvDim, _v, row * kvDim, kvDim);
            }
            Norm(_q, rows * _heads, layer.Head, layer.QNorm); Norm(_k, rows * layer.Kv, layer.Head, layer.KNorm); Norm(_v, rows * layer.Kv, layer.Head, null);
            Rope(_q, _heads, layer.Head, lengths, layer.Local ? _localBase : _globalBase);
            Rope(_k, layer.Kv, layer.Head, lengths, layer.Local ? _localBase : _globalBase);
            Attention(lengths, layer);
            layer.O.Multiply(_attn, rows, _next, _pool, cancellationToken);
            Norm(_next, rows, _dim, layer.PostAttentionNorm); Add(_x, _next, rows * _dim);
            using (Profile(Phase.BufferLayout, rows))
            {
                Array.Copy(_x, _normalized, rows * _dim);
                Ensure(ref _gate, rows * _ff);
                if (!_fusedGateGelu) Ensure(ref _up, rows * _ff);
                Ensure(ref _gateUp, checked(rows * _ff * 2));
            }
            Norm(_normalized, rows, _dim, layer.FfnNorm);
            layer.GateUp.Multiply(_normalized, rows, _gateUp, _pool, cancellationToken);
            if (_fusedGateGelu)
            {
                using (Profile(Phase.FusedGateGelu, rows))
                    ManagedEmbeddingGatedGelu.Apply(_gateUp, _gate, rows, _ff, _pool, cancellationToken);
                ++_fusedGateGeluCalls; _fusedGateGeluRows += rows;
            }
            else
            {
                using (Profile(Phase.BufferLayout, rows))
                for (int row = 0; row < rows; ++row)
                {
                    Array.Copy(_gateUp, row * _ff * 2, _gate, row * _ff, _ff);
                    Array.Copy(_gateUp, row * _ff * 2 + _ff, _up, row * _ff, _ff);
                }
                Gelu(_gate, rows, _ff);
                using (Profile(Phase.ResidualGating, rows))
                    TensorPrimitives.Multiply(_gate.AsSpan(0, rows * _ff), _up.AsSpan(0, rows * _ff), _gate.AsSpan(0, rows * _ff));
            }
            layer.Down.Multiply(_gate, rows, _next, _pool, cancellationToken);
            Norm(_next, rows, _dim, layer.PostFfnNorm); Add(_x, _next, rows * _dim);
            using (Profile(Phase.BufferLayout, rows)) Ensure(ref _pleGate, rows * _pleDim);
            layer.InputGate.Multiply(_x, rows, _pleGate, _pool, cancellationToken); Gelu(_pleGate, rows, _pleDim);
            using (Profile(Phase.ResidualGating, rows))
            _pool.For(rows, row => TensorPrimitives.Multiply(_pleGate.AsSpan(row * _pleDim, _pleDim),
                _pleInputs.AsSpan((row * _layers.Length + layerIndex) * _pleDim, _pleDim), _pleGate.AsSpan(row * _pleDim, _pleDim)), cancellationToken);
            layer.Projection.Multiply(_pleGate, rows, _next, _pool, cancellationToken);
            Norm(_next, rows, _dim, layer.PostNorm); Add(_x, _next, rows * _dim);
            using (Profile(Phase.ResidualGating, rows)) TensorPrimitives.Multiply(_x.AsSpan(0, rows * _dim), layer.Scale, _x.AsSpan(0, rows * _dim));
        }
        Norm(_x, rows, _dim, _outputNorm, Phase.FinalNormalization);
        float[] pooled;
        using (Profile(Phase.Pooling, rows))
        {
            pooled = new float[checked(lengths.Length * _dim)];
            int offset = 0;
            for (int b = 0; b < lengths.Length; ++b)
            {
                var mean = pooled.AsSpan(b * _dim, _dim);
                for (int row = 0; row < lengths[b]; ++row) TensorPrimitives.Add(mean, _x.AsSpan((offset + row) * _dim, _dim), mean);
                TensorPrimitives.Multiply(mean, 1f / lengths[b], mean); offset += lengths[b];
            }
        }
        _projection.Multiply(pooled, lengths.Length, output, _pool, cancellationToken);
        using (Profile(Phase.OutputNormalization, lengths.Length))
        for (int b = 0; b < lengths.Length; ++b)
        {
            var vector = output.AsSpan(b * _output, _output);
            double squared = 0;
            foreach (float value in vector) { if (!float.IsFinite(value)) throw new InvalidDataException("Non-finite embedding output."); squared += (double)value * value; }
            if (!(squared > 0)) throw new InvalidDataException("Zero embedding output.");
            TensorPrimitives.Multiply(vector, (float)(1 / Math.Sqrt(squared)), vector);
        }
    }
    private void Add(float[] target, float[] other, int count)
    {
        using var profile = Profile(Phase.ResidualGating, count / _dim);
        TensorPrimitives.Add(target.AsSpan(0, count), other.AsSpan(0, count), target.AsSpan(0, count));
    }
    private void Norm(float[] data, int rows, int width, float[] weight, Phase phase = Phase.RmsNormalization)
    {
        using var profile = Profile(phase, rows);
        _pool.For(rows, row =>
        {
            var values = data.AsSpan(row * width, width);
            float scale = 1 / MathF.Sqrt(TensorPrimitives.Dot(values, values) / width + _eps);
            TensorPrimitives.Multiply(values, scale, values);
            if (weight != null) TensorPrimitives.Multiply(values, weight, values);
        }, _cancellation);
    }
    private void Gelu(float[] data, int rows, int width)
    {
        using var profile = Profile(Phase.Activation, rows);
        int jobs = Math.Min(rows, Math.Max(1, _pool.ThreadCount * 4));
        _pool.For(jobs, job =>
        {
            var scratch = ArrayPool<float>.Shared.Rent(width);
            try
            {
                var temporary = scratch.AsSpan(0, width);
                for (int row = rows * job / jobs; row < rows * (job + 1) / jobs; ++row)
                {
                    var x = data.AsSpan(row * width, width);
                    TensorPrimitives.Multiply(x, x, temporary); TensorPrimitives.Multiply(temporary, x, temporary);
                    TensorPrimitives.MultiplyAdd(temporary, .044715f, x, temporary); TensorPrimitives.Multiply(temporary, .7978845608028654f, temporary);
                    TensorPrimitives.Tanh(temporary, temporary); TensorPrimitives.Add(temporary, 1f, temporary);
                    TensorPrimitives.Multiply(x, temporary, x); TensorPrimitives.Multiply(x, .5f, x);
                }
            }
            finally { ArrayPool<float>.Shared.Return(scratch); }
        }, _cancellation);
    }
    private void Rope(float[] data, int heads, int head, int[] lengths, float basis)
    {
        int offset = 0, half = head / 2;
        int maxLength = 0;
        foreach (int length in lengths) maxLength = Math.Max(maxLength, length);
        if (!_ropeTables.TryGetValue((head, basis), out var table) || table.Length < maxLength)
        {
            using var tableProfile = Profile(Phase.RopeTables, maxLength);
            var sin = new float[checked(maxLength * half)];
            var cos = new float[sin.Length];
            for (int j = 0; j < half; ++j)
            {
                float frequency = MathF.Pow(basis, -2f * j / head);
                for (int position = 0; position < maxLength; ++position)
                    (sin[position * half + j], cos[position * half + j]) = MathF.SinCos(position * frequency);
            }
            table = (maxLength, sin, cos); _ropeTables[(head, basis)] = table;
        }
        using var profile = Profile(Phase.Rope, _profileEnabled ? lengths.Sum() : 0);
        foreach (int length in lengths)
        {
            int start = offset;
            _pool.For(length, position =>
            {
                for (int h = 0; h < heads; ++h)
                {
                    int row = ((start + position) * heads + h) * head;
                    for (int j = 0; j < half; ++j)
                    {
                        float sin = table.Sin[position * half + j], cos = table.Cos[position * half + j];
                        float a = data[row + j], b = data[row + j + half];
                        data[row + j] = a * cos - b * sin; data[row + j + half] = b * cos + a * sin;
                    }
                }
            }, _cancellation);
            offset += length;
        }
    }
    private unsafe void Attention(int[] lengths, Layer layer)
    {
        int sequenceStart = 0;
        foreach (int length in lengths)
        {
            int stride = ManagedEmbeddingMath.PaddedKeyStride(length), head = layer.Head;
            bool avx512 = ManagedEmbeddingAttentionAvx512.IsSupported;
            bool native = _nativeAttention != null && avx512 && OwnedEmbeddingCpuAttention.SupportsShape(head, length);
            int queryTile = native ? 8 : 4;
            int width = avx512 ? 16 : ManagedEmbeddingMath.ValueWidth, qDim = _heads * head, kvDim = layer.Kv * head;
            int start = sequenceStart;
            using (Profile(Phase.AttentionBuffers, length))
            {
                Ensure(ref _localK, checked(layer.Kv * head * stride)); Ensure(ref _localV, checked(layer.Kv * head * stride));
            }
            using (Profile(Phase.AttentionClear, length))
            {
                Array.Clear(_localK, 0, layer.Kv * head * stride); Array.Clear(_localV, 0, layer.Kv * head * stride);
            }
            using (Profile(Phase.AttentionPacking, length))
            {
                for (int h = 0; h < layer.Kv; ++h)
                for (int token = 0; token < length; ++token)
                for (int d = 0; d < head; ++d)
                {
                    _localK[(h * head + d) * stride + token] = _k[(start + token) * kvDim + h * head + d];
                    _localV[h * head * stride + (d / width * width) * stride + token * width + d % width] = _v[(start + token) * kvDim + h * head + d];
                }
            }
            using (Profile(Phase.AttentionCompute, length))
            fixed (float* q = _q, keys = _localK, values = _localV, result = _attn)
            {
                nint qAddress = (nint)q, keyAddress = (nint)keys, valueAddress = (nint)values, outputAddress = (nint)result;
                int groups = (length + queryTile - 1) / queryTile;
                _pool.For(_heads * groups, work =>
                {
                    int h = work / groups, first = work % groups * queryTile, active = Math.Min(queryTile, length - first), kv = h / (_heads / layer.Kv);
                    int begin = layer.Local ? Math.Max(0, first - _window) : 0;
                    int end = layer.Local ? Math.Min(length, first + active - 1 + _window + 1) : length;
                    int count = end - begin;
                    var scratch = ArrayPool<float>.Shared.Rent(checked(count * queryTile));
                    try
                    {
                        fixed (float* probabilities = scratch)
                        {
                            float* firstQuery = (float*)qAddress + (start + first) * qDim + h * head;
                            if (native)
                                _nativeAttention.Scores(firstQuery, (float*)keyAddress + kv * head * stride + begin,
                                    probabilities, qDim, stride, count, head, count, active, 1f);
                            else if (avx512)
                                ManagedEmbeddingAttentionAvx512.Scores(firstQuery, firstQuery + Math.Min(1, active - 1) * qDim,
                                    firstQuery + Math.Min(2, active - 1) * qDim, firstQuery + Math.Min(3, active - 1) * qDim,
                                    (float*)keyAddress + kv * head * stride + begin, stride, head, count, 1, probabilities);
                            else
                                ManagedEmbeddingWideMath.Scores(firstQuery, firstQuery + Math.Min(1, active - 1) * qDim,
                                    firstQuery + Math.Min(2, active - 1) * qDim, firstQuery + Math.Min(3, active - 1) * qDim,
                                    (float*)keyAddress + kv * head * stride + begin, stride, head, count, 1, probabilities);
                            // The native entry stores real rows only. Its tail
                            // tiles require no duplicate softmax work. Existing
                            // managed four-lane initialization stays unchanged.
                            for (int lane = 0; lane < (native ? active : 4); ++lane)
                            {
                                var row = new Span<float>(probabilities + lane * count, count);
                                if (layer.Local)
                                    for (int k = 0; k < count; ++k) if (Math.Abs(begin + k - first - Math.Min(lane, active - 1)) > _window) row[k] = float.NegativeInfinity;
                                float max = TensorPrimitives.Max(row);
                                float sum = ManagedEmbeddingWideMath.ExpSum(probabilities + lane * count, count, max);
                                TensorPrimitives.Multiply(row, 1 / sum, row);
                            }
                            if (native)
                                _nativeAttention.Values(probabilities, (float*)valueAddress + kv * head * stride + begin * width,
                                    (float*)outputAddress + (start + first) * qDim + h * head, count, stride, qDim, head, count, active);
                            else if (avx512)
                                ManagedEmbeddingAttentionAvx512.Values(probabilities, count, (float*)valueAddress + kv * head * stride + begin * width,
                                    stride, head, (float*)outputAddress + (start + first) * qDim + h * head, qDim, active);
                            else
                                ManagedEmbeddingWideMath.Values(probabilities, count, (float*)valueAddress + kv * head * stride + begin * width,
                                    stride, head, (float*)outputAddress + (start + first) * qDim + h * head, qDim, active);
                        }
                    }
                    finally { ArrayPool<float>.Shared.Return(scratch); }
                }, _cancellation);
                if (native)
                {
                    _nativeAttentionTiles += (long)_heads * groups;
                    _nativeAttentionRows += (long)_heads * length;
                }
                else _managedAttentionTiles += (long)_heads * groups;
            }
            sequenceStart += length;
        }
    }
    public void Dispose()
    {
        // A failed running-job guard means workers may still hold function
        // pointers. Keep their library leases until the pool actually drains.
        _pool.Dispose();
        try
        {
            var matrices = new List<Matrix> { _ple, _projection };
            foreach (var layer in _layers)
                matrices.AddRange([layer.Qkv, layer.O, layer.GateUp, layer.Down, layer.InputGate, layer.Projection]);
            if ((_nativeAttentionRequested || _nativeProjectionRequested || _fusedGateGelu) && _layers.Length != 0)
                Console.Error.WriteLine("EMBEDDING_CPU_MATH " + JsonSerializer.Serialize(new
                {
                    backend = "CPU", graph = "managed", native_math = _nativeAttentionTiles > 0 || _nativeProjection?.SuccessfulCalls > 0,
                    attention_math = _nativeAttentionTiles > 0 ? (_managedAttentionTiles > 0 ? "owned-native-avx512-f32-8query+managed-fallback" : "owned-native-avx512-f32-8query") : "managed-fallback",
                    library = _nativeAttention?.LibraryPath, library_sha256 = _nativeAttention?.LibrarySha256,
                    fallback_reason = _nativeAttentionFallback, successful_attention_tiles = _nativeAttentionTiles,
                    managed_attention_tiles = _managedAttentionTiles,
                    successful_attention_rows = _nativeAttentionRows,
                    projection_math = _nativeProjection?.SuccessfulCalls > 0 ? "owned-native-q8-q16-vnni" : "managed",
                    projection_library = _nativeProjection?.LibraryPath, projection_library_sha256 = _nativeProjection?.LibrarySha256,
                    projection_fallback_reason = _nativeProjectionFallback,
                    successful_projection_calls = _nativeProjection?.SuccessfulCalls ?? 0,
                    successful_projection_pack_rows = _nativeProjection?.SuccessfulPackRows ?? 0,
                    successful_projection_groups = _nativeProjection?.SuccessfulGroups ?? 0,
                    successful_projection_rows = _nativeProjection?.SuccessfulRows ?? 0,
                    managed_q16_projection_calls = matrices.Sum(matrix => matrix.ManagedQ16Calls),
                    q16_float32_fallback_rows = matrices.Sum(matrix => matrix.Q16Float32FallbackRows),
                    fused_gate_gelu_requested = _fusedGateGelu,
                    successful_fused_gate_gelu_calls = _fusedGateGeluCalls,
                    successful_fused_gate_gelu_rows = _fusedGateGeluRows,
                    intentional_f32_projections = new[] { "per_layer_model_proj", "output", "*.inp_gate" },
                }));
            if (_profileEnabled && _layers.Length != 0)
            {
                Console.Error.WriteLine("EMBEDDING_CPU_PROFILE " + JsonSerializer.Serialize(new
                {
                    backend = "CPU", graph = "managed", native_math = _nativeAttentionTiles > 0 || _nativeProjection?.SuccessfulCalls > 0,
                    threads = _pool.ThreadCount,
                    geometry = new { hidden = _dim, output = _output, ple = _pleDim, heads = _heads, feed_forward = _ff, layers = _layers.Length, half_window = _window },
                    attention_math = _nativeAttentionTiles > 0 ? (_managedAttentionTiles > 0 ? "owned-native-avx512-f32-8query+managed-fallback" : "owned-native-avx512-f32-8query") : ManagedEmbeddingAttentionAvx512.IsSupported ? "managed-avx512" : "managed-wide",
                    rope_math = "managed-scalar", normalization_math = "TensorPrimitives", activation_math = "TensorPrimitives-Tanh-GELU",
                    ffn_gate_math = _fusedGateGeluCalls > 0 ? "managed-joined-gelu-up-fused" : "managed-split-gelu-up",
                    successful_compute_calls = _profileCalls, input_rows = _profileRows,
                    compute_elapsed_ms = _profileTicks * 1000.0 / Stopwatch.Frequency,
                    projection_elapsed_ms = matrices.Sum(matrix => matrix.ProfileElapsedMs),
                    projections = matrices.Select(matrix => matrix.ProfileSnapshot()).ToArray(),
                    phases = Enumerable.Range(0, (int)Phase.Count).Select(index => new
                    {
                        name = ((Phase)index).ToString(), parent = "compute", accounting = "exclusive-leaf",
                        successful_calls = _phaseCalls[index], input_rows = _phaseRows[index],
                        elapsed_ms = _phaseTicks[index] * 1000.0 / Stopwatch.Frequency,
                    }).ToArray(),
                    input_preparation = new { parent = "encode", accounting = "exclusive-outside-compute", successful_calls = _inputCalls, input_rows = _inputRows, elapsed_ms = _inputTicks * 1000.0 / Stopwatch.Frequency },
                    token_row_export = new { parent = "TokenRows", accounting = "separate-outside-encode", successful_calls = _tokenRowCalls, input_rows = _tokenRowRows, elapsed_ms = _tokenRowTicks * 1000.0 / Stopwatch.Frequency },
                    limitations = "Opt-in diagnostic timings include managed scheduling and warmup. Compute excludes model loading, tokenization and initial input preparation. Phases are disjoint successful-compute leaf wall times; derived parent totals overlap their children and must not be added. Projection records count successful Multiply calls. No formal performance claim.",
                }));
            }
        }
        finally
        {
            try { _nativeAttention?.Dispose(); }
            finally { _nativeProjection?.Dispose(); }
            _tokens = _ple = _projection = null;
            _pleNorm = _outputNorm = [];
            _layers = [];
            _ropeTables.Clear();
            _x = _normalized = _next = _qkv = _q = _k = _v = _attn = _gateUp = _gate = _up = _pleInputs = _pleGate = _localK = _localV = [];
        }
    }
}
