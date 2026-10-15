// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#nullable enable
using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading;

namespace TensorSharp.Models.Embeddings;

/// <summary>Q8_0 weights with signed 16-bit activation tiles. F32 block scales
/// retain activation range; integer products are exact, including -128 weights.
/// The extra activation precision reduces Q8 activation error without expanding
/// weight storage to F32. This experimental route requires end-to-end gates.</summary>
internal sealed class ManagedEmbeddingQ8Q16Avx512Matrix
{
    public static bool IsSupported => Avx512F.IsSupported && Avx512BW.IsSupported && Avx2.IsSupported &&
        Environment.GetEnvironmentVariable("TS_CPU_DISABLE_AVX512") != "1";
    private readonly short[]? _weights;
    private readonly sbyte[]? _nativeWeights;
    private readonly int[]? _weightSums;
    private readonly OwnedEmbeddingCpuProjection? _native;
    private readonly float[] _scales;
    private readonly int _input, _output, _blocks, _columnTiles;
    internal bool UsesNative => _native != null;
    internal long ManagedCalls { get; private set; }
    internal long Float32FallbackRows { get; private set; }

    public ManagedEmbeddingQ8Q16Avx512Matrix(byte[] source, int input, int output, OwnedEmbeddingCpuProjection? native = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (input <= 0 || input % 32 != 0 || output <= 0 || source.Length != checked(input / 32 * 34 * output))
            throw new ArgumentException("Invalid Q8_0 matrix shape or byte count.");
        _input = input; _output = output; _blocks = input / 32; _columnTiles = (output + 15) / 16;
        _native = native;
        if (native == null) _weights = new short[checked(_columnTiles * _blocks * 512)];
        else
        {
            // Preserve int8 weight bandwidth. The encoder owns the native
            // library lease; this matrix never disposes or caches it globally.
            _nativeWeights = new sbyte[checked(_columnTiles * _blocks * 512)];
            _weightSums = new int[checked(_columnTiles * _blocks * 16)];
        }
        _scales = new float[checked(_columnTiles * _blocks * 16)];
        for (int tile = 0; tile < _columnTiles; ++tile)
        for (int block = 0; block < _blocks; ++block)
        for (int column = 0; column < 16 && tile * 16 + column < output; ++column)
        {
            int original = ((tile * 16 + column) * _blocks + block) * 34;
            ushort bits = (ushort)(source[original] | source[original + 1] << 8);
            float scale = (float)BitConverter.UInt16BitsToHalf(bits);
            if (!float.IsFinite(scale)) throw new ArgumentException("Q8_0 weight scales must be finite.");
            int index = (tile * _blocks + block) * 16 + column;
            _scales[index] = scale;
            for (int part = 0; part < 16; ++part)
            for (int value = 0; value < 2; ++value)
            {
                byte quant = source[original + 2 + part * 2 + value];
                int dim = part * 2 + value;
                if (_nativeWeights != null)
                {
                    _nativeWeights[(tile * _blocks + block) * 512 + dim / 4 * 64 + column * 4 + dim % 4] = unchecked((sbyte)quant);
                    _weightSums![index] += unchecked((sbyte)quant);
                }
                else _weights![(tile * _blocks + block) * 512 + part * 32 + column * 2 + value] = unchecked((sbyte)quant);
            }
        }
    }

    public unsafe void Multiply(float[] input, int rows, float[] output, CpuWorkerPool pool,
        CancellationToken cancellationToken = default)
    {
        if (!IsSupported) throw new PlatformNotSupportedException("AVX512F/BW are required for packed Q8 matrices.");
        ArgumentNullException.ThrowIfNull(input); ArgumentNullException.ThrowIfNull(output); ArgumentNullException.ThrowIfNull(pool);
        if (rows <= 0 || input.Length < checked(rows * _input) || output.Length < checked(rows * _output))
            throw new ArgumentException("Input or output storage does not match the Q8 matrix shape.");
        if (ReferenceEquals(input, output))
            throw new ArgumentException("Q16 projection requires separate input/output arrays to preserve F32 fallback rows.");
        cancellationToken.ThrowIfCancellationRequested();
        if (_native != null) { MultiplyNative(input, rows, output, pool, cancellationToken); return; }
        short[] activations = ArrayPool<short>.Shared.Rent(checked(rows * _input));
        float[]? scales = null;
        bool[]? fallbackRows = null;
        try
        {
            scales = ArrayPool<float>.Shared.Rent(checked(rows * _blocks));
            fallbackRows = ArrayPool<bool>.Shared.Rent(rows);
            fixed (short* ap = activations)
            fixed (float* asp = scales)
            fixed (float* xp = input)
            fixed (short* wp = _weights)
            fixed (float* wsp = _scales)
            fixed (float* yp = output)
            {
                nint activationAddress = (nint)ap, scaleAddress = (nint)asp, inputAddress = (nint)xp;
                nint weightAddress = (nint)wp, weightScaleAddress = (nint)wsp, outputAddress = (nint)yp;
                pool.For(rows, row => fallbackRows[row] = Quantize((float*)inputAddress + (long)row * _input,
                    (short*)activationAddress + (long)row * _input, (float*)scaleAddress + (long)row * _blocks,
                    _blocks), cancellationToken);
                const int columnsPerTask = 4, rowsPerTask = 16;
                int columnGroups = (_columnTiles + columnsPerTask - 1) / columnsPerTask;
                int rowGroups = (rows + rowsPerTask - 1) / rowsPerTask;
                pool.For(checked(columnGroups * rowGroups), work =>
                {
                    int firstTile = work % columnGroups * columnsPerTask;
                    int endTile = Math.Min(firstTile + columnsPerTask, _columnTiles);
                    int firstRow = work / columnGroups * rowsPerTask;
                    int endRow = Math.Min(firstRow + rowsPerTask, rows);
                    for (int tile = firstTile; tile < endTile; ++tile)
                    for (int row = firstRow; row < endRow; row += 4)
                        Tile((short*)weightAddress + (long)tile * _blocks * 512,
                            (float*)weightScaleAddress + (long)tile * _blocks * 16,
                            (short*)activationAddress + (long)row * _input,
                            (float*)scaleAddress + (long)row * _blocks,
                            (float*)outputAddress + (long)row * _output + tile * 16,
                            _blocks, _input, _output, Math.Min(4, endRow - row), Math.Min(16, _output - tile * 16));
                }, cancellationToken);
                // A subnormal block scale cannot represent Q16 precision.
                // Recompute affected rows with original F32 activations and
                // exactly dequantized weights, rather than flushing tiny values.
                if (fallbackRows.AsSpan(0, rows).Contains(true)) pool.For(rows, row =>
                {
                    if (!fallbackRows[row]) return;
                    for (int column = 0; column < _output; ++column)
                    {
                        int tile = column / 16, lane = column % 16;
                        float sum = 0;
                        for (int block = 0; block < _blocks; ++block)
                        {
                            float scale = ((float*)weightScaleAddress)[(tile * _blocks + block) * 16 + lane];
                            for (int dim = 0; dim < 32; ++dim)
                            {
                                short quant = ((short*)weightAddress)[(tile * _blocks + block) * 512 + dim / 2 * 32 + lane * 2 + dim % 2];
                                float value = ((float*)inputAddress)[(long)row * _input + block * 32 + dim];
                                sum = MathF.FusedMultiplyAdd(value, quant * scale, sum);
                            }
                        }
                        ((float*)outputAddress)[(long)row * _output + column] = sum;
                    }
                }, cancellationToken);
                ++ManagedCalls;
                for (int row = 0; row < rows; ++row) if (fallbackRows[row]) ++Float32FallbackRows;
            }
        }
        finally
        {
            ArrayPool<short>.Shared.Return(activations);
            if (scales != null) ArrayPool<float>.Shared.Return(scales);
            if (fallbackRows != null) ArrayPool<bool>.Shared.Return(fallbackRows);
        }
    }

    // A bounded span adapter for contiguous CPU tensor storage. The owner keeps
    // that storage and its library lease alive until the synchronous pool jobs
    // drain, including cancellation/failure and F32 subnormal row correction.
    internal unsafe void MultiplyNative(ReadOnlySpan<float> input, int rows, Span<float> output, CpuWorkerPool pool,
        CancellationToken cancellationToken = default)
    {
        if (!IsSupported || _native == null) throw new PlatformNotSupportedException("A qualified owned Q16 native lease is required.");
        ArgumentNullException.ThrowIfNull(pool);
        if (rows <= 0 || input.Length < checked(rows * _input) || output.Length < checked(rows * _output))
            throw new ArgumentException("Input or output span does not match the Q8 matrix shape.");
        if (input.Overlaps(output)) throw new ArgumentException("Q16 input and output spans must not overlap.");
        cancellationToken.ThrowIfCancellationRequested();
        fixed (float* xp = input)
        fixed (float* yp = output)
            MultiplyNative(xp, rows, yp, pool, cancellationToken);
    }

    private unsafe void MultiplyNative(float[] input, int rows, float[] output, CpuWorkerPool pool, CancellationToken cancellationToken)
    {
        fixed (float* xp = input)
        fixed (float* yp = output)
            MultiplyNative(xp, rows, yp, pool, cancellationToken);
    }

    private unsafe void MultiplyNative(float* input, int rows, float* output, CpuWorkerPool pool, CancellationToken cancellationToken)
    {
        short[] activations = ArrayPool<short>.Shared.Rent(checked(rows * _input));
        byte[]? packed = null;
        float[]? scales = null;
        bool[]? fallbackRows = null;
        try
        {
            packed = ArrayPool<byte>.Shared.Rent(checked(rows * _input * 2));
            scales = ArrayPool<float>.Shared.Rent(checked(rows * _blocks));
            fallbackRows = ArrayPool<bool>.Shared.Rent(rows);
            fixed (short* ap = activations)
            fixed (byte* pap = packed)
            fixed (sbyte* wp = _nativeWeights)
            fixed (int* sp = _weightSums)
            fixed (float* asp = scales, wsp = _scales)
            {
                nint aa = (nint)ap, pa = (nint)pap, wa = (nint)wp, sa = (nint)sp;
                nint xa = (nint)input, asa = (nint)asp, wsa = (nint)wsp, ya = (nint)output;
                pool.For(rows, row =>
                {
                    fallbackRows[row] = Quantize((float*)xa + (long)row * _input,
                        (short*)aa + (long)row * _input, (float*)asa + (long)row * _blocks, _blocks);
                    _native!.Pack((short*)aa + (long)row * _input, (byte*)pa + (long)row * _input * 2, _input);
                }, cancellationToken);
                const int columnsPerTask = 4, rowsPerTask = 16;
                int columnGroups = (_columnTiles + columnsPerTask - 1) / columnsPerTask;
                int rowGroups = (rows + rowsPerTask - 1) / rowsPerTask;
                int groups = checked(columnGroups * rowGroups);
                pool.For(groups, work =>
                {
                    int firstTile = work % columnGroups * columnsPerTask;
                    int endTile = Math.Min(firstTile + columnsPerTask, _columnTiles);
                    int firstRow = work / columnGroups * rowsPerTask;
                    int activeRows = Math.Min(rowsPerTask, rows - firstRow);
                    _native!.Group((sbyte*)wa + (long)firstTile * _blocks * 512,
                        (int*)sa + (long)firstTile * _blocks * 16, (float*)wsa + (long)firstTile * _blocks * 16,
                        (byte*)pa + (long)firstRow * _input * 2, (float*)asa + (long)firstRow * _blocks,
                        (float*)ya + (long)firstRow * _output + firstTile * 16,
                        _blocks, _input * 2, _output, activeRows, Math.Min((endTile - firstTile) * 16, _output - firstTile * 16));
                }, cancellationToken);
                // Native dots share the managed quantization exactly. A tiny
                // block scale still requires recomputing its entire original
                // activation row after all bounded group calls have drained.
                if (fallbackRows.AsSpan(0, rows).Contains(true)) pool.For(rows, row =>
                {
                    if (!fallbackRows[row]) return;
                    for (int col = 0; col < _output; ++col)
                    {
                        int tile = col / 16, lane = col % 16;
                        float sum = 0;
                        for (int block = 0; block < _blocks; ++block)
                        for (int dim = 0; dim < 32; ++dim)
                        {
                            int quant = ((sbyte*)wa)[(tile * _blocks + block) * 512 + dim / 4 * 64 + lane * 4 + dim % 4];
                            float scale = ((float*)wsa)[(tile * _blocks + block) * 16 + lane];
                            sum = MathF.FusedMultiplyAdd(((float*)xa)[(long)row * _input + block * 32 + dim], quant * scale, sum);
                        }
                        ((float*)ya)[(long)row * _output + col] = sum;
                    }
                }, cancellationToken);
                int fallbackCount = 0;
                for (int row = 0; row < rows; ++row) if (fallbackRows[row]) ++fallbackCount;
                Float32FallbackRows += fallbackCount;
                _native!.RecordMultiply(rows, groups, fallbackCount);
            }
        }
        finally
        {
            ArrayPool<short>.Shared.Return(activations);
            if (packed != null) ArrayPool<byte>.Shared.Return(packed);
            if (scales != null) ArrayPool<float>.Shared.Return(scales);
            if (fallbackRows != null) ArrayPool<bool>.Shared.Return(fallbackRows);
        }
    }

    internal static unsafe bool Quantize(float* input, short* output, float* scales, int blocks)
    {
        bool fallback = false;
        var absoluteMask = Vector256.Create(0x7fffffff).AsSingle();
        for (int block = 0; block < blocks; ++block)
        {
            float* x = input + block * 32;
            var x0 = Vector256.Load(x); var x1 = Vector256.Load(x + 8);
            var x2 = Vector256.Load(x + 16); var x3 = Vector256.Load(x + 24);
            var invalid = Avx.Or(Avx.Compare(x0, x0, FloatComparisonMode.UnorderedNonSignaling),
                Avx.Compare(x1, x1, FloatComparisonMode.UnorderedNonSignaling));
            invalid = Avx.Or(invalid, Avx.Or(Avx.Compare(x2, x2, FloatComparisonMode.UnorderedNonSignaling),
                Avx.Compare(x3, x3, FloatComparisonMode.UnorderedNonSignaling)));
            var maximum = Avx.Max(Avx.And(x0, absoluteMask), Avx.And(x1, absoluteMask));
            maximum = Avx.Max(maximum, Avx.Max(Avx.And(x2, absoluteMask), Avx.And(x3, absoluteMask)));
            maximum = Avx.Max(maximum, Avx.Permute2x128(maximum, maximum, 1));
            maximum = Avx.Max(maximum, Avx.Shuffle(maximum, maximum, 0x4e));
            maximum = Avx.Max(maximum, Avx.Shuffle(maximum, maximum, 0xb1));
            float scale = maximum.ToScalar() / 32767.0f;
            if (Avx.MoveMask(invalid) != 0 || !float.IsFinite(scale))
                throw new ArgumentException("Q16 activations must be finite.");
            if (maximum.ToScalar() > 0f && scale < 1.17549435e-38f)
            {
                fallback = true;
                scales[block] = 0f;
                Vector256<short>.Zero.Store(output + block * 32);
                Vector256<short>.Zero.Store(output + block * 32 + 16);
                continue;
            }
            scales[block] = scale;
            float inverseScale = scale == 0 ? 0 : 1.0f / scale;
            var inverse = Vector256.Create(inverseScale);
            // A tiny finite scale can have an infinite reciprocal. Division
            // preserves finite normalized quants in that uncommon case.
            var divisor = Vector256.Create(scale);
            bool multiply = float.IsFinite(inverseScale);
            var q0 = Avx.ConvertToVector256Int32(multiply ? Avx.Multiply(x0, inverse) : Avx.Divide(x0, divisor));
            var q1 = Avx.ConvertToVector256Int32(multiply ? Avx.Multiply(x1, inverse) : Avx.Divide(x1, divisor));
            var q2 = Avx.ConvertToVector256Int32(multiply ? Avx.Multiply(x2, inverse) : Avx.Divide(x2, divisor));
            var q3 = Avx.ConvertToVector256Int32(multiply ? Avx.Multiply(x3, inverse) : Avx.Divide(x3, divisor));
            var first = Avx2.Permute4x64(Avx2.PackSignedSaturate(q0, q1).AsInt64(), 0xd8).AsInt16();
            var last = Avx2.Permute4x64(Avx2.PackSignedSaturate(q2, q3).AsInt64(), 0xd8).AsInt16();
            first.Store(output + block * 32);
            last.Store(output + block * 32 + 16);
        }
        return fallback;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector512<int> DotPart(Vector512<short> weights, short* activation)
    {
        var pair = Vector512.Create(Unsafe.ReadUnaligned<int>(activation)).AsInt16();
        // Each block has 32 products. 32*32768*128 <= 134,217,728,
        // well inside signed int32; no saturation or sign correction is needed.
        return Avx512BW.MultiplyAddAdjacent(weights, pair);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Tile(short* weights, float* weightScales, short* input, float* inputScales,
        float* output, int blocks, int inputStride, int outputStride, int rows, int columns)
    {
        var sum0 = Vector512<float>.Zero; var sum1 = Vector512<float>.Zero;
        var sum2 = Vector512<float>.Zero; var sum3 = Vector512<float>.Zero;
        for (int block = 0; block < blocks; ++block)
        {
            var d0 = Vector512<int>.Zero; var d1 = Vector512<int>.Zero;
            var d2 = Vector512<int>.Zero; var d3 = Vector512<int>.Zero;
            for (int part = 0; part < 16; ++part)
            {
                var w = Vector512.Load(weights + block * 512 + part * 32);
                short* a = input + block * 32 + part * 2;
                d0 = Avx512F.Add(d0, DotPart(w, a));
                if (rows > 1) d1 = Avx512F.Add(d1, DotPart(w, a + inputStride));
                if (rows > 2) d2 = Avx512F.Add(d2, DotPart(w, a + inputStride * 2));
                if (rows > 3) d3 = Avx512F.Add(d3, DotPart(w, a + inputStride * 3));
            }
            var scale = Vector512.Load(weightScales + block * 16);
            sum0 = Accumulate(sum0, d0, scale, inputScales[block]);
            if (rows > 1) sum1 = Accumulate(sum1, d1, scale, inputScales[blocks + block]);
            if (rows > 2) sum2 = Accumulate(sum2, d2, scale, inputScales[blocks * 2 + block]);
            if (rows > 3) sum3 = Accumulate(sum3, d3, scale, inputScales[blocks * 3 + block]);
        }
        Store(sum0, output, columns);
        if (rows > 1) Store(sum1, output + outputStride, columns);
        if (rows > 2) Store(sum2, output + outputStride * 2, columns);
        if (rows > 3) Store(sum3, output + outputStride * 3, columns);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> Accumulate(Vector512<float> sum, Vector512<int> dot, Vector512<float> weightScale, float inputScale)
        // Apply the normal-sized weight scale first. Multiplying two tiny
        // scales before the integer dot loses subnormal activation precision.
        => Avx512F.FusedMultiplyAdd(Avx512F.Multiply(Avx512F.ConvertToVector512Single(dot), weightScale),
            Vector512.Create(inputScale), sum);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Store(Vector512<float> values, float* output, int columns)
    {
        if (columns == 16) Avx512F.Store(output, values);
        else for (int column = 0; column < columns; ++column) output[column] = values.GetElement(column);
    }
}
