// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading;

namespace TensorSharp.Models.Embeddings;

/// <summary>
/// Managed AVX2 Q8_0 matrix tiles. Eight SIMD lanes compute eight output
/// columns, and four rows share each weight load. Packing retains every GGUF
/// scale and signed quant exactly, including -128 weights.
/// </summary>
internal sealed class ManagedEmbeddingQ8AvxMatrix
{
    public static bool IsSupported => Avx2.IsSupported;
    private readonly byte[] _weights;
    private readonly float[] _scales;
    private readonly int _input, _output, _blocks, _columnTiles;
    private readonly bool _hasNegative128;
    private readonly bool _preserveFloat32ActivationScales;

    public ManagedEmbeddingQ8AvxMatrix(byte[] source, int input, int output,
        bool preserveFloat32ActivationScales = false)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (input <= 0 || input % 32 != 0 || output <= 0 || source.Length != checked(input / 32 * 34 * output))
            throw new ArgumentException("Invalid Q8_0 matrix shape or byte count.");
        _input = input; _output = output; _blocks = input / 32; _columnTiles = (output + 7) / 8;
        _preserveFloat32ActivationScales = preserveFloat32ActivationScales;
        _weights = new byte[checked(_columnTiles * _blocks * 256)];
        _scales = new float[checked(_columnTiles * _blocks * 8)];
        bool negative128 = false;
        for (int tile = 0; tile < _columnTiles; ++tile)
        for (int block = 0; block < _blocks; ++block)
        for (int column = 0; column < 8 && tile * 8 + column < output; ++column)
        {
            int original = ((tile * 8 + column) * _blocks + block) * 34;
            ushort bits = (ushort)(source[original] | source[original + 1] << 8);
            float scale = (float)BitConverter.UInt16BitsToHalf(bits);
            if (!float.IsFinite(scale)) throw new ArgumentException("Q8_0 weight scales must be finite.");
            _scales[(tile * _blocks + block) * 8 + column] = scale;
            for (int part = 0; part < 8; ++part)
            for (int value = 0; value < 4; ++value)
            {
                byte quant = source[original + 2 + part * 4 + value];
                _weights[(tile * _blocks + block) * 256 + part * 32 + column * 4 + value] = quant;
                negative128 |= quant == 128;
            }
        }
        _hasNegative128 = negative128;
    }

    public unsafe void Multiply(float[] input, int rows, float[] output, CpuWorkerPool pool,
        CancellationToken cancellationToken = default)
    {
        if (!IsSupported) throw new PlatformNotSupportedException("AVX2 is required for packed Q8 matrices.");
        ArgumentNullException.ThrowIfNull(input); ArgumentNullException.ThrowIfNull(output); ArgumentNullException.ThrowIfNull(pool);
        if (rows <= 0 || input.Length < checked(rows * _input) || output.Length < checked(rows * _output))
            throw new ArgumentException("Input or output storage does not match the Q8 matrix shape.");
        cancellationToken.ThrowIfCancellationRequested();
        byte[] activations = ArrayPool<byte>.Shared.Rent(checked(rows * _input));
        float[] scales = null;
        try
        {
            scales = ArrayPool<float>.Shared.Rent(checked(rows * _blocks));
            fixed (byte* ap = activations)
            fixed (float* asp = scales)
            fixed (float* xp = input)
            fixed (byte* wp = _weights)
            fixed (float* wsp = _scales)
            fixed (float* yp = output)
            {
                nint activationAddress = (nint)ap, scaleAddress = (nint)asp, inputAddress = (nint)xp;
                nint weightAddress = (nint)wp, weightScaleAddress = (nint)wsp, outputAddress = (nint)yp;
                pool.For(rows, row => Quantize((float*)inputAddress + (long)row * _input,
                    (sbyte*)activationAddress + (long)row * _input, (float*)scaleAddress + (long)row * _blocks,
                    _blocks, _preserveFloat32ActivationScales), cancellationToken);
                const int columnsPerTask = 8, rowsPerTask = 16;
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
                        Tile((sbyte*)weightAddress + (long)tile * _blocks * 256,
                            (float*)weightScaleAddress + (long)tile * _blocks * 8,
                            (sbyte*)activationAddress + (long)row * _input,
                            (float*)scaleAddress + (long)row * _blocks,
                            (float*)outputAddress + (long)row * _output + tile * 8,
                            _blocks, _input, _output, Math.Min(4, endRow - row),
                            Math.Min(8, _output - tile * 8), _hasNegative128);
                }, cancellationToken);
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(activations);
            if (scales != null) ArrayPool<float>.Shared.Return(scales);
        }
    }

    internal static unsafe void Quantize(float* input, sbyte* output, float* scales, int blocks, bool preserveFloat32Scales)
    {
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
            float scale = maximum.ToScalar() / 127.0f;
            if (Avx.MoveMask(invalid) != 0 || !float.IsFinite(scale))
                throw new ArgumentException("Q8 activations must be finite.");
            // Quants always use the original F32 scale. The default scale
            // storage matches Q8_0; F32 storage avoids its range/rounding loss.
            float storedScale = preserveFloat32Scales ? scale : (float)(System.Half)scale;
            if (!float.IsFinite(storedScale))
                throw new ArgumentException("Q8 activation scale exceeds the finite F16 storage range.");
            scales[block] = storedScale;
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
            Avx2.Permute4x64(Avx2.PackSignedSaturate(first, last).AsInt64(), 0xd8).AsSByte().Store(output + block * 32);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> DotPart(Vector256<sbyte> weights, Vector256<sbyte> activations, bool negative128)
    {
        var absolute = Avx2.Abs(activations);
        var signed = Avx2.Sign(weights, activations);
        // |activation|<=127 guarantees that maddubs cannot saturate a pair.
        var pairs = Avx2.MultiplyAddAdjacent(absolute, signed);
        var dot = Avx2.MultiplyAddAdjacent(pairs, Vector256.Create((short)1));
        if (negative128)
        {
            // psignb(-128, negative) remains -128 rather than +128. Restore
            // exactly 256*|activation| in the affected integer dot lanes.
            var correctionMask = Avx2.And(Avx2.CompareEqual(weights, Vector256.Create((sbyte)-128)),
                Avx2.CompareGreaterThan(Vector256<sbyte>.Zero, activations)).AsByte();
            var corrections = Avx2.MultiplyAddAdjacent(Avx2.And(absolute, correctionMask), Vector256.Create((sbyte)1));
            dot = Avx2.Add(dot, Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(corrections, Vector256.Create((short)1)), 8));
        }
        return dot;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Tile(sbyte* weights, float* weightScales, sbyte* input, float* inputScales,
        float* output, int blocks, int inputStride, int outputStride, int rows, int columns, bool negative128)
    {
        var sum0 = Vector256<float>.Zero; var sum1 = Vector256<float>.Zero;
        var sum2 = Vector256<float>.Zero; var sum3 = Vector256<float>.Zero;
        for (int block = 0; block < blocks; ++block)
        {
            var d0 = Vector256<int>.Zero; var d1 = Vector256<int>.Zero;
            var d2 = Vector256<int>.Zero; var d3 = Vector256<int>.Zero;
            for (int part = 0; part < 8; ++part)
            {
                var w = Vector256.Load(weights + block * 256 + part * 32);
                sbyte* a = input + block * 32 + part * 4;
                d0 = Avx2.Add(d0, DotPart(w, Vector256.Create(Unsafe.ReadUnaligned<int>(a)).AsSByte(), negative128));
                if (rows > 1) d1 = Avx2.Add(d1, DotPart(w, Vector256.Create(Unsafe.ReadUnaligned<int>(a + inputStride)).AsSByte(), negative128));
                if (rows > 2) d2 = Avx2.Add(d2, DotPart(w, Vector256.Create(Unsafe.ReadUnaligned<int>(a + inputStride * 2)).AsSByte(), negative128));
                if (rows > 3) d3 = Avx2.Add(d3, DotPart(w, Vector256.Create(Unsafe.ReadUnaligned<int>(a + inputStride * 3)).AsSByte(), negative128));
            }
            var scale = Vector256.Load(weightScales + block * 8);
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
    private static Vector256<float> Accumulate(Vector256<float> sum, Vector256<int> dot, Vector256<float> weightScale, float inputScale)
    {
        var scale = Avx.Multiply(weightScale, Vector256.Create(inputScale));
        var values = Avx.ConvertToVector256Single(dot);
        return Fma.IsSupported ? Fma.MultiplyAdd(values, scale, sum) : Avx.Add(sum, Avx.Multiply(values, scale));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Store(Vector256<float> values, float* output, int columns)
    {
        if (columns == 8) values.Store(output);
        else for (int column = 0; column < columns; ++column) output[column] = values.GetElement(column);
    }
}
