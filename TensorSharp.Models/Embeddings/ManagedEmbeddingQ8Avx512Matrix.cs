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

/// <summary>Managed AVX512BW Q8_0 tiles with sixteen output columns and four
/// rows. Signed activations are split into two bounded unsigned pieces so
/// maddubs remains exact for every weight, including -128.</summary>
internal sealed class ManagedEmbeddingQ8Avx512Matrix
{
    public static bool IsSupported => Avx512F.IsSupported && Avx512BW.IsSupported && Avx2.IsSupported;
    private readonly byte[] _weights;
    private readonly float[] _scales;
    private readonly int[] _corrections;
    private readonly int _input, _output, _blocks, _columnTiles;
    private readonly bool _preserveFloat32ActivationScales;

    public ManagedEmbeddingQ8Avx512Matrix(byte[] source, int input, int output,
        bool preserveFloat32ActivationScales = true)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (input <= 0 || input % 32 != 0 || output <= 0 || source.Length != checked(input / 32 * 34 * output))
            throw new ArgumentException("Invalid Q8_0 matrix shape or byte count.");
        _input = input; _output = output; _blocks = input / 32; _columnTiles = (output + 15) / 16;
        _preserveFloat32ActivationScales = preserveFloat32ActivationScales;
        _weights = new byte[checked(_columnTiles * _blocks * 512)];
        _scales = new float[checked(_columnTiles * _blocks * 16)];
        _corrections = new int[_scales.Length];
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
            int sum = 0;
            for (int part = 0; part < 8; ++part)
            for (int value = 0; value < 4; ++value)
            {
                byte quant = source[original + 2 + part * 4 + value];
                _weights[(tile * _blocks + block) * 512 + part * 64 + column * 4 + value] = quant;
                sum += unchecked((sbyte)quant);
            }
            _corrections[index] = -128 * sum;
        }
    }

    public unsafe void Multiply(float[] input, int rows, float[] output, CpuWorkerPool pool,
        CancellationToken cancellationToken = default)
    {
        if (!IsSupported) throw new PlatformNotSupportedException("AVX512F/BW are required for packed Q8 matrices.");
        ArgumentNullException.ThrowIfNull(input); ArgumentNullException.ThrowIfNull(output); ArgumentNullException.ThrowIfNull(pool);
        if (rows <= 0 || input.Length < checked(rows * _input) || output.Length < checked(rows * _output))
            throw new ArgumentException("Input or output storage does not match the Q8 matrix shape.");
        cancellationToken.ThrowIfCancellationRequested();
        byte[] activations = ArrayPool<byte>.Shared.Rent(checked(rows * _input * 2));
        float[]? scales = null;
        try
        {
            scales = ArrayPool<float>.Shared.Rent(checked(rows * _blocks));
            fixed (byte* ap = activations)
            fixed (float* asp = scales)
            fixed (float* xp = input)
            fixed (byte* wp = _weights)
            fixed (float* wsp = _scales)
            fixed (int* cp = _corrections)
            fixed (float* yp = output)
            {
                nint activationAddress = (nint)ap, scaleAddress = (nint)asp, inputAddress = (nint)xp;
                nint weightAddress = (nint)wp, weightScaleAddress = (nint)wsp, correctionAddress = (nint)cp, outputAddress = (nint)yp;
                pool.For(rows, row => Quantize((float*)inputAddress + (long)row * _input,
                    (byte*)activationAddress + (long)row * _input * 2, (float*)scaleAddress + (long)row * _blocks,
                    _blocks, _preserveFloat32ActivationScales), cancellationToken);
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
                        Tile((sbyte*)weightAddress + (long)tile * _blocks * 512,
                            (float*)weightScaleAddress + (long)tile * _blocks * 16,
                            (int*)correctionAddress + (long)tile * _blocks * 16,
                            (byte*)activationAddress + (long)row * _input * 2,
                            (float*)scaleAddress + (long)row * _blocks,
                            (float*)outputAddress + (long)row * _output + tile * 16,
                            _blocks, _input * 2, _output, Math.Min(4, endRow - row), Math.Min(16, _output - tile * 16));
                }, cancellationToken);
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(activations);
            if (scales != null) ArrayPool<float>.Shared.Return(scales);
        }
    }

    private static unsafe void Quantize(float* input, byte* output, float* scales, int blocks, bool preserveFloat32Scales)
    {
        ManagedEmbeddingQ8AvxMatrix.Quantize(input, (sbyte*)output, scales, blocks, preserveFloat32Scales);
        var lowMask = Vector256.Create((byte)0x7f); var highMask = Vector256.Create((byte)0x80);
        // Work backwards because the low/high representation expands each
        // 32-byte quant block to 64 bytes in the same rented row allocation.
        for (int block = blocks - 1; block >= 0; --block)
        {
            var quant = Vector256.Load(output + block * 32);
            Avx2.And(quant, lowMask).Store(output + block * 64);
            Avx2.And(Avx2.Xor(quant, highMask), highMask).Store(output + block * 64 + 32);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector512<int> DotPart(Vector512<sbyte> weights, byte* activation)
    {
        var low = Vector512.Create(Unsafe.ReadUnaligned<int>(activation)).AsByte();
        var high = Vector512.Create(Unsafe.ReadUnaligned<int>(activation + 32)).AsByte();
        // low<=127 and high is 0/128. Even a pair of -128 weights has
        // a representable signed-16 sum, so neither maddubs can saturate.
        var a = Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(low, weights), Vector512.Create((short)1));
        var b = Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(high, weights), Vector512.Create((short)1));
        return Avx512F.Add(a, b);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Tile(sbyte* weights, float* weightScales, int* corrections, byte* input, float* inputScales,
        float* output, int blocks, int inputStride, int outputStride, int rows, int columns)
    {
        var sum0 = Vector512<float>.Zero; var sum1 = Vector512<float>.Zero;
        var sum2 = Vector512<float>.Zero; var sum3 = Vector512<float>.Zero;
        for (int block = 0; block < blocks; ++block)
        {
            var d0 = Vector512<int>.Zero; var d1 = Vector512<int>.Zero;
            var d2 = Vector512<int>.Zero; var d3 = Vector512<int>.Zero;
            for (int part = 0; part < 8; ++part)
            {
                var w = Vector512.Load(weights + block * 512 + part * 64);
                byte* a = input + block * 64 + part * 4;
                d0 = Avx512F.Add(d0, DotPart(w, a));
                if (rows > 1) d1 = Avx512F.Add(d1, DotPart(w, a + inputStride));
                if (rows > 2) d2 = Avx512F.Add(d2, DotPart(w, a + inputStride * 2));
                if (rows > 3) d3 = Avx512F.Add(d3, DotPart(w, a + inputStride * 3));
            }
            var correction = Vector512.Load(corrections + block * 16);
            var scale = Vector512.Load(weightScales + block * 16);
            sum0 = Accumulate(sum0, Avx512F.Add(d0, correction), scale, inputScales[block]);
            if (rows > 1) sum1 = Accumulate(sum1, Avx512F.Add(d1, correction), scale, inputScales[blocks + block]);
            if (rows > 2) sum2 = Accumulate(sum2, Avx512F.Add(d2, correction), scale, inputScales[blocks * 2 + block]);
            if (rows > 3) sum3 = Accumulate(sum3, Avx512F.Add(d3, correction), scale, inputScales[blocks * 3 + block]);
        }
        Store(sum0, output, columns);
        if (rows > 1) Store(sum1, output + outputStride, columns);
        if (rows > 2) Store(sum2, output + outputStride * 2, columns);
        if (rows > 3) Store(sum3, output + outputStride * 3, columns);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> Accumulate(Vector512<float> sum, Vector512<int> dot, Vector512<float> weightScale, float inputScale)
        => Avx512F.FusedMultiplyAdd(Avx512F.ConvertToVector512Single(dot),
            Avx512F.Multiply(weightScale, Vector512.Create(inputScale)), sum);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Store(Vector512<float> values, float* output, int columns)
    {
        if (columns == 16) Avx512F.Store(output, values);
        else for (int column = 0; column < columns; ++column) output[column] = values.GetElement(column);
    }
}
