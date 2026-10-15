// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

/// <summary>The budgeted GLM vision path runs one graph per encoder block with
/// weights uploaded for that graph only. It must compute exactly what the
/// cached whole-encoder graph computes: same builders, same F32 weights.</summary>
public sealed class GlmVisionStreamedEncoderTests
{
    [GgmlFact(BackendType.GgmlCuda)]
    public void StreamedBlockGraphsMatchTheWholeEncoderGraph()
    {
        const int blocks = 2, hidden = 64, heads = 4, headDim = hidden / heads, halfDim = headDim / 2;
        const int intermediate = 96, patches = 16;
        var allocator = new GgmlAllocator(new GgmlContext([0], GgmlBackendType.Cuda), 0);
        var owned = new List<Tensor>();
        Tensor Make(float seed, params long[] sizes)
        {
            var tensor = new Tensor(allocator, DType.Float32, sizes);
            owned.Add(tensor);
            long count = tensor.ElementCount();
            var values = new float[count];
            for (long i = 0; i < count; i++) values[i] = 0.05f * MathF.Sin(seed + i * 0.37f);
            tensor.SetElementsAsFloat(values);
            return tensor;
        }
        Tensor Norm(float seed, long size)
        {
            var tensor = Make(seed, size);
            tensor.SetElementsAsFloat(tensor.GetElementsAsFloat((int)size).Select(v => 1f + v).ToArray());
            return tensor;
        }
        Tensor[] PerBlock(Func<int, Tensor> make) => Enumerable.Range(0, blocks).Select(make).ToArray();
        var ln1 = PerBlock(b => Norm(1 + b, hidden));
        var qkv = PerBlock(b => Make(2 + b, 3 * hidden, hidden));
        var qkvBias = PerBlock(b => Make(3 + b, 3 * hidden));
        var qNorm = PerBlock(b => Norm(4 + b, headDim));
        var kNorm = PerBlock(b => Norm(5 + b, headDim));
        var output = PerBlock(b => Make(6 + b, hidden, hidden));
        var outputBias = PerBlock(b => Make(7 + b, hidden));
        var ln2 = PerBlock(b => Norm(8 + b, hidden));
        var gate = PerBlock(b => Make(9 + b, intermediate, hidden));
        var gateBias = PerBlock(b => Make(10 + b, intermediate));
        var up = PerBlock(b => Make(11 + b, intermediate, hidden));
        var upBias = PerBlock(b => Make(12 + b, intermediate));
        var down = PerBlock(b => Make(13 + b, hidden, intermediate));
        var downBias = PerBlock(b => Make(14 + b, hidden));
        var cos = new float[patches * halfDim];
        var sin = new float[patches * halfDim];
        for (int i = 0; i < cos.Length; i++) { cos[i] = MathF.Cos(i * 0.21f); sin[i] = MathF.Sin(i * 0.21f); }
        try
        {
            float[] Encode(bool streamed)
            {
                using var state = new Tensor(allocator, DType.Float32, patches, hidden);
                var input = new float[patches * hidden];
                for (int i = 0; i < input.Length; i++) input[i] = MathF.Cos(i * 0.13f);
                state.SetElementsAsFloat(input);
                Assert.True(GgmlBasicOps.GlmVisionEncoder(state, 1e-5f, 1f / MathF.Sqrt(headDim), 10f,
                    patches, heads, headDim, halfDim, cos, sin,
                    ln1, qkv, qkvBias, qNorm, kNorm, output, outputBias, ln2,
                    gate, gateBias, up, upBias, down, downBias, streamed));
                return state.GetElementsAsFloat(patches * hidden);
            }
            float[] whole = Encode(false), streamedResult = Encode(true), again = Encode(true);
            Assert.All(whole, v => Assert.True(float.IsFinite(v)));
            Assert.NotEqual(new float[patches * hidden], whole);
            double numerator = 0, denominator = 0;
            for (int i = 0; i < whole.Length; i++)
            {
                numerator += Math.Pow(whole[i] - streamedResult[i], 2);
                denominator += Math.Pow(whole[i], 2);
            }
            Assert.True(Math.Sqrt(numerator / denominator) <= 1e-5, "streamed encoder diverged from the whole graph");
            Assert.Equal(streamedResult, again);
        }
        finally
        {
            foreach (var tensor in owned) tensor.Dispose();
        }
    }
}
