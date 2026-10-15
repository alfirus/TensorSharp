using System.Numerics;
using TensorSharp;
using TensorSharp.MLX;

namespace InferenceWeb.Tests;

public sealed class Gemma4VisionMlxSpatialTests
{
    [MlxTheory]
    [InlineData(1, 1, 1, 1f)]
    [InlineData(6, 9, 17, 1f)]
    [InlineData(48, 48, 768, 80000f)]
    public void PositionLookupPreservesLegacyF32AssociationAndReadOnlyTableOffsets(int patchesX, int patchesY, int channels, float magnitude)
    {
        using var allocator = new MlxAllocator();
        int rows = patchesX * patchesY, maxPositions = Math.Max(patchesX, patchesY) + 2;
        var data = new float[rows, channels]; var table = new float[3, maxPositions, channels];
        for (int row = 0; row < rows; row++)for (int channel = 0; channel < channels; channel++)
            data[row, channel] = channel % 5 == 0 ? 16777216f : ((row * 7 + channel * 11) % 29 - 14) * magnitude / 17f;
        for (int position = 0; position < maxPositions; position++)for (int channel = 0; channel < channels; channel++)
        {
            table[0, position, channel] = 37f;
            table[1, position, channel] = channel % 5 == 0 ? 1f : ((position * 3 + channel * 5) % 19 - 9) * magnitude / 23f;
            table[2, position, channel] = channel % 5 == 0 ? -16777216f : ((position * 7 + channel * 3) % 23 - 11) * magnitude / 29f;
        }
        using var hidden = Tensor.FromArray(allocator, data); using var tables = Tensor.FromArray(allocator, table);
        using var positions = tables.Narrow(0, 1, 2);
        Assert.True(MlxGemma4VisionOps.TryAddPosition2D(hidden, positions, patchesX, patchesY));
        float[] actual = hidden.GetElementsAsFloat(rows * channels);
        int vectorChannels = channels / Vector<float>.Count * Vector<float>.Count;
        for (int row = 0; row < rows; row++)for (int channel = 0; channel < channels; channel++)
        {
            float a = data[row, channel], x = table[1, row % patchesX, channel], y = table[2, row / patchesX, channel];
            float expected = channel < vectorChannels
                ? (float)((double)(float)((double)a + x) + y)
                : (float)((double)a + (float)((double)x + y));
            Assert.Equal(BitConverter.SingleToInt32Bits(expected), BitConverter.SingleToInt32Bits(actual[row * channels + channel]));
            double exact = (double)a + x + y;
            double bound = (Math.Abs((double)a) + Math.Abs((double)x) + Math.Abs((double)y)) * 4 * Math.ScaleB(1, -24);
            Assert.InRange(Math.Abs(actual[row * channels + channel] - exact), 0, Math.Max(bound, 1e-30));
        }
        Assert.Equal(table.Cast<float>().ToArray(), tables.GetElementsAsFloat(table.Length));
        Assert.False(MlxGemma4VisionOps.TryAddPosition2D(hidden, positions, patchesX + 1, patchesY));
        using var backing = new Tensor(allocator, DType.Float32, rows + 1, channels);
        using var offsetTarget = backing.Narrow(0, 1, rows);
        Assert.False(MlxGemma4VisionOps.TryAddPosition2D(offsetTarget, positions, patchesX, patchesY));
        if (channels > 1)
        {
            using var strided = hidden.Narrow(1, 0, channels - 1);
            Assert.False(MlxGemma4VisionOps.TryAddPosition2D(strided, positions, patchesX, patchesY));
        }
    }

    [MlxTheory]
    [InlineData(3, 3, 1, 1f)]
    [InlineData(6, 9, 17, 1f)]
    [InlineData(7, 5, 19, 80000f)]
    [InlineData(48, 48, 768, 80000f)]
    public void ThreeByThreePoolMatchesOrderedF32AndDoubleBoundAndCropsIncompleteBoxes(int patchesX, int patchesY, int channels, float magnitude)
    {
        using var allocator = new MlxAllocator();
        int rows = patchesX * patchesY, mergedX = patchesX / 3, mergedY = patchesY / 3;
        var data = new float[rows + 2, channels];
        for (int row = 0; row < rows + 2; row++)for (int channel = 0; channel < channels; channel++)
            data[row, channel] = ((row * 7 + channel * 11) % 29 - 14) * magnitude / 17f;
        // A cancellation sequence reveals a reordered/pairwise reduction. The
        // independent double bound is separate from the original F32 contract.
        for (int channel = 0; channel < channels; channel += 5)
        { data[1, channel] = 16777216f; data[2, channel] = 1f; data[3, channel] = -16777216f; }
        using var backing = Tensor.FromArray(allocator, data); using var input = backing.Narrow(0, 1, rows);
        using var result = new Tensor(allocator, DType.Float32, mergedX * mergedY, channels);
        Assert.True(MlxGemma4VisionOps.TryPool3x3(result, input, patchesX, patchesY));
        float[] actual = result.GetElementsAsFloat(mergedX * mergedY * channels);
        for (int py = 0; py < mergedY; py++)for (int px = 0; px < mergedX; px++)for (int channel = 0; channel < channels; channel++)
        {
            float ordered = 0; double exact = 0, absolute = 0;
            for (int ky = 0; ky < 3; ky++)for (int kx = 0; kx < 3; kx++)
            {
                float value = data[1 + (py * 3 + ky) * patchesX + px * 3 + kx, channel];
                ordered = (float)((double)ordered + value); exact += value; absolute += Math.Abs((double)value);
            }
            ordered *= 1f / 9f;
            float valueActual = actual[(py * mergedX + px) * channels + channel];
            Assert.Equal(BitConverter.SingleToInt32Bits(ordered), BitConverter.SingleToInt32Bits(valueActual));
            double bound = absolute * 12 * Math.ScaleB(1, -24) / 9;
            Assert.InRange(Math.Abs(valueActual - exact / 9), 0, Math.Max(bound, 1e-30));
        }
        Assert.Equal(data.Cast<float>().ToArray(), backing.GetElementsAsFloat(data.Length));
        using var wrong = new Tensor(allocator, DType.Float32, mergedX * mergedY + 1, channels);
        Assert.False(MlxGemma4VisionOps.TryPool3x3(wrong, input, patchesX, patchesY));
        using var offsetTarget = wrong.Narrow(0, 1, mergedX * mergedY);
        Assert.False(MlxGemma4VisionOps.TryPool3x3(offsetTarget, input, patchesX, patchesY));
        Assert.False(MlxGemma4VisionOps.TryPool3x3(input, input, patchesX, patchesY));
    }

    [MlxFact]
    public void SpatialOpsRejectAliasingShortGridsAndWrongAllocatorWithoutMutation()
    {
        using var allocator = new MlxAllocator(); using var peer = new MlxAllocator();
        using var hidden = Tensor.FromArray(allocator, new float[16, 17]);
        using var alias = hidden.View(2, 8, 17);
        Assert.False(MlxGemma4VisionOps.TryAddPosition2D(hidden, alias, 4, 4));
        using var positions = Tensor.FromArray(peer, new float[2, 8, 17]);
        Assert.False(MlxGemma4VisionOps.TryAddPosition2D(hidden, positions, 4, 4));
        using var result = new Tensor(allocator, DType.Float32, 1, 17);
        Assert.False(MlxGemma4VisionOps.TryPool3x3(result, hidden, 2, 8));
        using var peerResult = new Tensor(peer, DType.Float32, 1, 17);
        Assert.False(MlxGemma4VisionOps.TryPool3x3(peerResult, hidden, 4, 4));
        Assert.Equal(new float[16 * 17], hidden.GetElementsAsFloat(16 * 17));
    }
}
