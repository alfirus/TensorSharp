using TensorSharp;
using TensorSharp.GGML;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class Gemma4PreciseMetalProjectionTests
{
    [GgmlFact(BackendType.GgmlMetal)]
    public void ScopedProjectionMatchesDoubleWithOffsetsTailsAndLargeFloat32Range()
    {
        var allocator = new GgmlAllocator(new GgmlContext([0], GgmlBackendType.Metal), 0);
        foreach (var (rows, inner, columns, largeInput, largeWeight) in new[]
        {
            (1, 9, 11, false, false), (17, 64, 35, false, false), (129, 64, 41, false, false),
            (17, 9, 11, true, false), (17, 9, 11, false, true),
        })
        {
            var x = Enumerable.Range(0, rows * inner).Select(i => (i * 7 % 29 - 14) / (largeWeight ? 23000f : 23f)).ToArray();
            var w = Enumerable.Range(0, columns * inner).Select(i => (i * 11 % 37 - 18) / (largeInput ? 27000f : 27f)).ToArray();
            if (largeInput) x[3] = 80000.125f;
            if (largeWeight) w[5] = 80000.125f;
            using var backing = new Tensor(allocator, DType.Float32, rows + 2, columns);
            backing.SetElementsAsFloat(Enumerable.Repeat(12345.5f, (rows + 2) * columns).ToArray());
            using var result = backing.Narrow(0, 1, rows);
            using var inputBacking = new Tensor(allocator, DType.Float32, rows + 2, inner);
            using var input = inputBacking.Narrow(0, 1, rows);
            input.SetElementsAsFloat(x);
            using var weight = new Tensor(allocator, DType.Float32, columns, inner);
            weight.SetElementsAsFloat(w);
            Assert.True(GgmlBasicOps.TryGemma4ProjectionFloat32(result, input, weight));
            var actual = result.GetElementsAsFloat(rows * columns);
            double error = 0;
            for (int row = 0; row < rows; ++row)
                for (int column = 0; column < columns; ++column)
                {
                    double expected = 0;
                    for (int d = 0; d < inner; ++d) expected += (double)x[row * inner + d] * w[column * inner + d];
                    Assert.True(float.IsFinite(actual[row * columns + column]));
                    error = Math.Max(error, Math.Abs(expected - actual[row * columns + column]));
                }
            Assert.True(error <= 3e-5, $"{rows}x{inner}x{columns}, input80K={largeInput}, weight80K={largeWeight}: {error:R}");
            var all = backing.GetElementsAsFloat((rows + 2) * columns);
            Assert.All(all.Take(columns).Concat(all.Skip((rows + 1) * columns)), value => Assert.Equal(12345.5f, value));
            GgmlBasicOps.InvalidateTensorHostBuffer(weight);
        }
        using var square = new Tensor(allocator, DType.Float32, 9, 9);
        using var other = new Tensor(allocator, DType.Float32, 9, 9);
        Assert.False(GgmlBasicOps.TryGemma4ProjectionFloat32(square, square, other));
        Assert.False(GgmlBasicOps.TryGemma4ProjectionFloat32(square, other, square));
        using var half = new Tensor(allocator, DType.Float16, 9, 9);
        Assert.False(GgmlBasicOps.TryGemma4ProjectionFloat32(square, other, half));
        GgmlBasicOps.ReleaseReuseComputeBuffers();
    }
}
