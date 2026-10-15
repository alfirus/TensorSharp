using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Direct;
using TensorSharp.Models.MiniMaxH3;

namespace InferenceWeb.Tests;

public sealed class MiniMaxH3TensorOpsTests
{
    [MlxFact]
    public void MlxPartialRotaryPreservesTailAndMatchesScalar() => Mlx(Rotary);
    [CudaFact]
    public void CudaPartialRotaryPreservesTailAndMatchesScalar() => Cuda(Rotary);
    [MlxFact]
    public void MlxCausalGqaMatchesDoubleReferenceWithLargeValues() => Mlx(Attention);
    [MlxFact]
    public void MlxChunkedCausalGqaPreservesAbsoluteMaskPositions() => Mlx((context, operations) => Attention(context, operations, 3));
    [CudaFact]
    public void CudaCausalGqaMatchesDoubleReferenceWithLargeValues() => Cuda(Attention);
    [MlxFact]
    public void MlxVaeInterleavedProjectionPreservesEveryElement() => Mlx(Interleaved);
    [CudaFact]
    public void CudaVaeInterleavedProjectionPreservesEveryElement() => Cuda(Interleaved);
    [MlxFact]
    public void MlxVisionGeluErfMatchesScalarAndRetainsFloat32Range() => Mlx(Gelu);
    [CudaFact]
    public void CudaVisionGeluErfMatchesScalarAndRetainsFloat32Range() => Cuda(Gelu);

    private static void Mlx(Action<DirectContext, MiniMaxH3TensorOps> test)
        => MlxWorker.Shared.Invoke(() =>
        {
            using var allocator = new MlxAllocator();
            using var context = new DirectContext(allocator);
            using var operations = new MiniMaxH3TensorOps(context);
            test(context, operations);
        });

    private static void Cuda(Action<DirectContext, MiniMaxH3TensorOps> test)
    {
        using var allocator = new CudaAllocator();
        using var context = new DirectContext(allocator);
        using var operations = new MiniMaxH3TensorOps(context);
        test(context, operations);
    }

    private static void Rotary(DirectContext context, MiniMaxH3TensorOps operations)
    {
        const int rows = 7, heads = 3, head = 32, rotary = 24;
        float[] input = Enumerable.Range(0, rows * heads * head).Select(i => MathF.Sin(i * .19f) * 4).ToArray();
        var cosine = new float[rows * rotary];
        var sine = new float[cosine.Length];
        for (int row = 0; row < rows; ++row)
        for (int channel = 0; channel < rotary / 2; ++channel)
        {
            float angle = (row + 1) * (channel + 1) * .071f;
            cosine[row * rotary + channel] = cosine[row * rotary + channel + rotary / 2] = MathF.Cos(angle);
            sine[row * rotary + channel] = sine[row * rotary + channel + rotary / 2] = MathF.Sin(angle);
        }
        // H3's text refiner uses a prefix of the full packed table. Reusing the
        // same table for different sequence lengths must rebuild its compact view.
        foreach (int sequence in new[] { 3, rows, 3, rows })
        {
            float[] current = input.Take(sequence * heads * head).ToArray();
            using var tensor = context.FromFloats(current, sequence, heads * head);
            operations.RopeHalf(tensor, cosine, sine, sequence, heads, head, rotary);
            float[] actual = tensor.GetElementsAsFloat(current.Length);
            for (int row = 0; row < sequence; ++row)
            for (int h = 0; h < heads; ++h)
            for (int channel = 0; channel < head; ++channel)
            {
                int index = (row * heads + h) * head + channel;
                double expected = input[index];
                if (channel < rotary / 2)
                    expected = (double)input[index] * cosine[row * rotary + channel] -
                        (double)input[index + rotary / 2] * sine[row * rotary + channel];
                else if (channel < rotary)
                    expected = (double)input[index] * cosine[row * rotary + channel] +
                        (double)input[index - rotary / 2] * sine[row * rotary + channel];
                if (channel >= rotary) Assert.Equal(input[index], actual[index]);
                else Assert.InRange(Math.Abs(actual[index] - expected), 0, 1e-6);
            }
        }
    }

    private static void Attention(DirectContext context, MiniMaxH3TensorOps operations) => Attention(context, operations, 0);

    private static void Attention(DirectContext context, MiniMaxH3TensorOps operations, int maximumQueryRows)
    {
        const int rows = 17, heads = 4, kvHeads = 2, head = 32;
        float[] q = Enumerable.Range(0, rows * heads * head).Select(i => MathF.Sin(i * .13f)).ToArray();
        float[] k = Enumerable.Range(0, rows * kvHeads * head).Select(i => MathF.Cos(i * .17f)).ToArray();
        float[] v = Enumerable.Range(0, k.Length).Select(i => MathF.Sin(i * .31f) * 80000).ToArray();
        using var qt = context.FromFloats(q, rows, heads * head);
        using var kt = context.FromFloats(k, rows, kvHeads * head);
        using var vt = context.FromFloats(v, rows, kvHeads * head);
        using var result = operations.CausalGqaAttention(qt, kt, vt, rows, heads, kvHeads, head, maximumQueryRows);
        float[] actual = result.GetElementsAsFloat(rows * heads * head);
        double squaredError = 0, squaredReference = 0;
        for (int row = 0; row < rows; ++row)
        for (int h = 0; h < heads; ++h)
        {
            int kv = h / (heads / kvHeads);
            var scores = new double[row + 1];
            for (int key = 0; key <= row; ++key)
                for (int c = 0; c < head; ++c)
                    scores[key] += (double)q[(row * heads + h) * head + c] * k[(key * kvHeads + kv) * head + c] / Math.Sqrt(head);
            double maximum = scores.Max();
            double sum = 0;
            for (int key = 0; key <= row; ++key) sum += scores[key] = Math.Exp(scores[key] - maximum);
            for (int c = 0; c < head; ++c)
            {
                double expected = 0;
                for (int key = 0; key <= row; ++key) expected += scores[key] / sum * v[(key * kvHeads + kv) * head + c];
                float value = actual[(row * heads + h) * head + c];
                Assert.True(float.IsFinite(value));
                squaredError += (value - expected) * (value - expected);
                squaredReference += expected * expected;
            }
        }
        Assert.InRange(Math.Sqrt(squaredError / squaredReference), 0, 2e-5);
    }

    private static void Interleaved(DirectContext context, MiniMaxH3TensorOps operations)
    {
        const int rows = 3, heads = 2, head = 32;
        float[] input = Enumerable.Range(0, rows * heads * 3 * head).Select(i => (float)(i - 77)).ToArray();
        using var tensor = context.FromFloats(input, rows, heads * 3 * head);
        for (int component = 0; component < 3; ++component)
        {
            using var split = MiniMaxH3TensorOps.SplitInterleaved(tensor, rows, heads, head, component);
            float[] actual = split.GetElementsAsFloat(rows * heads * head);
            for (int row = 0; row < rows; ++row)
            for (int h = 0; h < heads; ++h)
            for (int c = 0; c < head; ++c)
                Assert.Equal(input[((row * heads + h) * 3 + component) * head + c], actual[(row * heads + h) * head + c]);
        }
    }

    private static void Gelu(DirectContext context, MiniMaxH3TensorOps operations)
    {
        float[] input = [-80000, -9, -3, -1, -.1f, -0f, 0, .1f, 1, 3, 9, 80000];
        using var tensor = context.FromFloats(input, 1, input.Length);
        operations.GeluErf(tensor);
        float[] actual = tensor.GetElementsAsFloat(input.Length);
        for (int i = 0; i < input.Length; ++i)
        {
            double x = input[i] / Math.Sqrt(2), absolute = Math.Abs(x), t = 1 / (1 + .3275911 * absolute);
            double polynomial = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - .284496736) * t + .254829592) * t;
            double erf = Math.Sign(x) * (1 - polynomial * Math.Exp(-absolute * absolute));
            double expected = .5 * input[i] * (1 + erf);
            Assert.True(float.IsFinite(actual[i]));
            Assert.InRange(Math.Abs(actual[i] - expected), 0, Math.Max(1, Math.Abs(expected)) * 1e-6);
        }
    }
}
