using TensorSharp;
using TensorSharp.GGML;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>Independent double equations for the typed Metal block. The two
/// large-range cases catch narrowing of either activations or learned weights.
/// The 37-row case exercises a partial SIMD tile with unrounded inputs.</summary>
public sealed class Gemma4VisionPreciseMetalTests
{
    [GgmlFact(BackendType.GgmlCpu)]
    public void CpuBlockBorrowsReadOnlyWeightsOnlyForTheCurrentCall()
    {
        const int hidden = 64, intermediate = 128, rows = 7;
        const float epsilon = 1e-6f;
        var allocator = new GgmlAllocator(new GgmlContext([0], GgmlBackendType.Cpu), 0);
        // Rebuilding the tower and changing the contents of the same live weight
        // tensor catches retained copies/mappings. The caller owns every weight;
        // CPU borrowing must never write them or require cache invalidation.
        for (int lifetime = 0; lifetime < 2; ++lifetime)
        {
            var owned = new List<Tensor>();
            Tensor Make(float[] values, params long[] shape)
            {
                var tensor = new Tensor(allocator, DType.Float32, shape);
                owned.Add(tensor); tensor.SetElementsAsFloat(values); return tensor;
            }
            try
            {
                var initial = Enumerable.Range(0, rows * hidden).Select(i => (i * 17 % 71 - 35) / 23f).ToArray();
                float[] Matrix(int inputs, int outputs, int seed) => Enumerable.Range(0, inputs * outputs)
                    .Select(i => (i * 13 + seed * 17 + lifetime * 7) % 61 / 271f - .1123f).ToArray();
                var qw = Matrix(hidden, hidden, 1); var kw = Matrix(hidden, hidden, 2);
                var vw = Matrix(hidden, hidden, 3); var ow = Matrix(hidden, hidden, 4);
                var gw = Matrix(hidden, intermediate, 5); var uw = Matrix(hidden, intermediate, 6);
                var dw = Matrix(intermediate, hidden, 7);
                var one = Enumerable.Repeat(1f, hidden).ToArray();
                var px = Enumerable.Range(0, rows).Select(i => i % 3).ToArray();
                var py = Enumerable.Range(0, rows).Select(i => i / 3).ToArray();
                var clamps = Enumerable.Range(0, 28).Select(i => i % 2 == 0 ? float.MinValue : float.MaxValue).ToArray();
                var input = Make(initial, rows, hidden); var norm = Make(one, hidden);
                var q = Make(qw, hidden, hidden); var k = Make(kw, hidden, hidden);
                var v = Make(vw, hidden, hidden); var output = Make(ow, hidden, hidden);
                var gate = Make(gw, intermediate, hidden); var up = Make(uw, intermediate, hidden);
                var down = Make(dw, hidden, intermediate);
                foreach (float adjustment in new[] { 0f, .0237f })
                {
                    if (adjustment != 0)
                        for (int i = 0; i < qw.Length; ++i) qw[i] += adjustment * (i % 3 - 1);
                    q.SetElementsAsFloat(qw);
                    input.SetElementsAsFloat(initial);
                    GgmlBasicOps.FusedGemma4VisionBlock(input, epsilon, norm, q, k, v,
                        norm, norm, norm, output, px, py, 10000f, norm, gate, up, down,
                        norm, clamps, rows, 1, hidden);
                    var expected = Reference(initial, rows, hidden, intermediate, epsilon,
                        qw, kw, vw, ow, gw, uw, dw, px, py);
                    var actual = input.GetElementsAsFloat(initial.Length);
                    double error = actual.Zip(expected, (a, b) => Math.Abs(a - b)).Max();
                    Assert.True(error < .0001, $"lifetime={lifetime}, adjustment={adjustment}: error={error:R}");
                    foreach (var (tensor, original) in new[] { (q, qw), (k, kw), (v, vw),
                        (output, ow), (gate, gw), (up, uw), (down, dw), (norm, one) })
                        Assert.Equal(original, tensor.GetElementsAsFloat(original.Length));
                }
            }
            finally
            {
                foreach (var tensor in owned) tensor.Dispose();
                GgmlBasicOps.ReleaseReuseComputeBuffers();
            }
        }
    }

    [GgmlFact(BackendType.GgmlMetal)]
    public void TypedMetalBlockPreservesUnroundedRowsAndBothLargeFloat32Ranges()
        => VerifyTypedBlock(GgmlBackendType.Metal, requireMetalPrecision: true);

    [GgmlFact(BackendType.GgmlCuda)]
    public void CudaBlockPreservesUnroundedRowsAndBothLargeFloat32Ranges()
        => VerifyTypedBlock(GgmlBackendType.Cuda, requireMetalPrecision: false);

    private static void VerifyTypedBlock(GgmlBackendType backend, bool requireMetalPrecision)
    {
        const int hidden = 64, intermediate = 128;
        const float epsilon = 1e-6f;
        var allocator = new GgmlAllocator(new GgmlContext([0], backend), 0);
        foreach (var (rows, largeInput, largeWeight) in new[] { (37, false, false), (3, true, false), (7, false, true) })
        {
            var owned = new List<Tensor>();
            Tensor Make(float[] values, params long[] shape)
            {
                var tensor = new Tensor(allocator, DType.Float32, shape);
                owned.Add(tensor); tensor.SetElementsAsFloat(values); return tensor;
            }
            try
            {
                var initial = Enumerable.Range(0, rows * hidden).Select(i => (i * 17 % 71 - 35) / 23f).ToArray();
                if (largeInput) initial[7] = 80000.125f;
                float[] Matrix(int inputs, int outputs, int seed) => Enumerable.Range(0, inputs * outputs)
                    .Select(i => (i * 13 + seed * 17) % 61 / 271f - .1123f).ToArray();
                var qw = Matrix(hidden, hidden, 1); var kw = Matrix(hidden, hidden, 2);
                var vw = Matrix(hidden, hidden, 3); var ow = Matrix(hidden, hidden, 4);
                var gw = Matrix(hidden, intermediate, 5); var uw = Matrix(hidden, intermediate, 6);
                var dw = Matrix(intermediate, hidden, 7);
                if (largeWeight) { qw[9] = 80000.125f; gw[23] = -80000.125f; }
                var one = Enumerable.Repeat(1f, hidden).ToArray();
                var px = Enumerable.Range(0, rows).Select(i => i % 7).ToArray();
                var py = Enumerable.Range(0, rows).Select(i => i / 7).ToArray();
                var clamps = Enumerable.Range(0, 28).Select(i => i % 2 == 0 ? float.MinValue : float.MaxValue).ToArray();
                var input = Make(initial, rows, hidden);
                var norm = Make(one, hidden);
                GgmlBasicOps.FusedGemma4VisionBlock(input, epsilon, norm,
                    Make(qw, hidden, hidden), Make(kw, hidden, hidden), Make(vw, hidden, hidden),
                    norm, norm, norm, Make(ow, hidden, hidden), px, py, 10000f,
                    norm, Make(gw, intermediate, hidden), Make(uw, intermediate, hidden),
                    Make(dw, hidden, intermediate), norm, clamps, rows, 1, hidden,
                    requireFloat32MetalPrecision: requireMetalPrecision,
                    requireFloat32CudaPrecision: backend == GgmlBackendType.Cuda);
                var expected = Reference(initial, rows, hidden, intermediate, epsilon, qw, kw, vw, ow, gw, uw, dw, px, py);
                var actual = input.GetElementsAsFloat(initial.Length);
                Assert.All(actual, value => Assert.True(float.IsFinite(value)));
                double error = actual.Zip(expected, (a, b) => Math.Abs(a - b)).Max();
                Assert.True(error <= (largeInput ? .009 : .001), $"rows={rows}, input80K={largeInput}, weight80K={largeWeight}: error={error:R}");
            }
            finally
            {
                foreach (var tensor in owned)
                {
                    GgmlBasicOps.InvalidateTensorHostBuffer(tensor);
                    tensor.Dispose();
                }
                GgmlBasicOps.ReleaseReuseComputeBuffers();
            }
        }
    }

    private static double[] Reference(float[] input, int rows, int hidden, int intermediate, double epsilon,
        float[] qw, float[] kw, float[] vw, float[] ow, float[] gw, float[] uw, float[] dw, int[] px, int[] py)
    {
        double[] Norm(double[] x, int width)
        {
            var y = new double[x.Length];
            for (int row = 0; row < x.Length / width; ++row)
            {
                double square = 0;
                for (int d = 0; d < width; ++d) square += x[row * width + d] * x[row * width + d];
                double factor = 1 / Math.Sqrt(square / width + epsilon);
                for (int d = 0; d < width; ++d) y[row * width + d] = x[row * width + d] * factor;
            }
            return y;
        }
        double[] Linear(double[] x, float[] weight, int inner, int output)
        {
            var y = new double[rows * output];
            for (int row = 0; row < rows; ++row)
                for (int column = 0; column < output; ++column)
                    for (int d = 0; d < inner; ++d) y[row * output + column] += x[row * inner + d] * weight[column * inner + d];
            return y;
        }
        void Rope(double[] x)
        {
            for (int row = 0; row < rows; ++row)
                for (int axis = 0; axis < 2; ++axis)
                    for (int d = 0; d < hidden / 4; ++d)
                    {
                        double angle = (axis == 0 ? px[row] : py[row]) * Math.Pow(10000, -2.0 * d / (hidden / 2));
                        int a = row * hidden + axis * (hidden / 2) + d, b = a + hidden / 4;
                        double left = x[a], right = x[b];
                        x[a] = left * Math.Cos(angle) - right * Math.Sin(angle);
                        x[b] = left * Math.Sin(angle) + right * Math.Cos(angle);
                    }
        }
        var original = input.Select(value => (double)value).ToArray();
        var normalized = Norm(original, hidden);
        var q = Norm(Linear(normalized, qw, hidden, hidden), hidden);
        var k = Norm(Linear(normalized, kw, hidden, hidden), hidden);
        var v = Norm(Linear(normalized, vw, hidden, hidden), hidden);
        Rope(q); Rope(k);
        var attention = new double[rows * hidden];
        for (int row = 0; row < rows; ++row)
        {
            var scores = new double[rows];
            for (int key = 0; key < rows; ++key)
                for (int d = 0; d < hidden; ++d) scores[key] += q[row * hidden + d] * k[key * hidden + d];
            double maximum = scores.Max(), total = 0;
            for (int key = 0; key < rows; ++key) { scores[key] = Math.Exp(scores[key] - maximum); total += scores[key]; }
            for (int key = 0; key < rows; ++key)
                for (int d = 0; d < hidden; ++d) attention[row * hidden + d] += scores[key] / total * v[key * hidden + d];
        }
        var postAttention = Norm(Linear(attention, ow, hidden, hidden), hidden);
        var residual = original.Zip(postAttention, (a, b) => a + b).ToArray();
        var ffnInput = Norm(residual, hidden);
        var gate = Linear(ffnInput, gw, hidden, intermediate); var up = Linear(ffnInput, uw, hidden, intermediate);
        for (int i = 0; i < gate.Length; ++i)
            gate[i] = .5 * gate[i] * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (gate[i] + .044715 * gate[i] * gate[i] * gate[i]))) * up[i];
        var postFfn = Norm(Linear(gate, dw, intermediate, hidden), hidden);
        return residual.Zip(postFfn, (a, b) => a + b).ToArray();
    }
}
