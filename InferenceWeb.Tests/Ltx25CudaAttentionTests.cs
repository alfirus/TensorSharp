using TensorSharp;
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

public sealed class Ltx25CudaAttentionTests
{
    [CudaTheory]
    [InlineData(7, 7, 3, 5, 1, 1f, 1f)]
    [InlineData(17, 17, 4, 64, 2, 1f, 1f)]
    [InlineData(33, 33, 3, 128, 3, 80000f, 1f)]
    [InlineData(129, 129, 4, 64, 4, 1f, 1f)]
    [InlineData(65, 65, 32, 128, 32, 1f, 1f)]
    [InlineData(3, 3, 32, 64, 7, 1f, 1f)]
    [InlineData(17, 17, 5, 64, 2, 1f, -1f)]
    [InlineData(7, 7, 3, 128, 2, 80000f, 0f)]
    [InlineData(17, 33, 5, 128, 2, 1f, 1f)]
    [InlineData(65, 7, 32, 64, 32, 80000f, 1f)]
    public void BatchedStridedHeadsAndWorkspaceTilesMatchDoubleSoftmaxOracle(
        int tokens, int keys, int heads, int dim, int tile, float valueMagnitude, float scaleFactor)
        => CheckAttentionOracle(tokens, keys, heads, dim, tile, valueMagnitude, scaleFactor);

    [CudaTheory]
    [InlineData(17, 33, 5, 64, 5)]
    [InlineData(8201, 7, 3, 5, 23)]
    [InlineData(3, 10752, 3, 5, 2)]
    [InlineData(19, 31, 5, 7, 1)]
    public void QueryTilesAndLargeAsymmetricSequencesMatchFullKeyDoubleOracle(int tokens, int keys, int heads, int dim, int queryRows)
        => CheckAttentionOracle(tokens, keys, heads, dim, 1, 1f, 1f, queryRows);

    private static void CheckAttentionOracle(int tokens, int keys, int heads, int dim, int tile,
        float valueMagnitude, float scaleFactor, int queryRows = 0)
    {
        using var allocator = new CudaAllocator();
        using var session = new CudaLtx25AttentionSession(allocator, (long)tile * (queryRows > 0 ? queryRows : tokens) * keys * sizeof(float));
        int width = heads * dim;
        var qv = new float[tokens + 2, width];
        var kv = new float[keys + 2, width];
        var vv = new float[keys + 2, width];
        var initial = new float[tokens + 2, width];
        for (int row = 0; row < tokens + 2; row++)
        for (int c = 0; c < width; c++)
        {
            qv[row, c] = MathF.Sin(row * .71f + c * .19f) * .7f;
            initial[row, c] = 37f;
        }
        for (int row = 0; row < keys + 2; row++)
        for (int c = 0; c < width; c++)
        {
            kv[row, c] = MathF.Cos(row * .23f - c * .37f) * .8f;
            vv[row, c] = (MathF.Sin(row * .41f + c * .53f) + .2f) * valueMagnitude;
        }
        using var qb = Tensor.FromArray(allocator, qv);
        using var kb = Tensor.FromArray(allocator, kv);
        using var vb = Tensor.FromArray(allocator, vv);
        using var yb = Tensor.FromArray(allocator, initial);
        using var q = qb.Narrow(0, 1, tokens);
        using var k = kb.Narrow(0, 1, keys);
        using var v = vb.Narrow(0, 1, keys);
        using var y = yb.Narrow(0, 1, tokens);
        float scale = scaleFactor / MathF.Sqrt(dim);
        Assert.True(session.TryAttention(y, q, k, v, heads, dim, scale));
        Assert.Equal(1, session.ExecutedAttentions);
        Assert.InRange(session.ScoreCapacityBytes, 1, session.ScoreBudgetBytes);
        if (queryRows > 0) Assert.InRange(session.LastQueryTileRows, 1, queryRows);
        float[] actual = yb.GetElementsAsFloat(initial.Length);
        var scores = new double[keys];
        for (int h = 0; h < heads; h++)
        for (int row = 0; row < tokens; row++)
        {
            for (int key = 0; key < keys; key++)
            {
                double dot = 0;
                for (int c = 0; c < dim; c++) dot += (double)qv[row + 1, h * dim + c] * kv[key + 1, h * dim + c];
                scores[key] = dot * scale;
            }
            double max = scores.Max(), sum = 0;
            for (int key = 0; key < keys; key++) sum += scores[key] = Math.Exp(scores[key] - max);
            for (int c = 0; c < dim; c++)
            {
                double expected = 0, absoluteProducts = 0;
                for (int key = 0; key < keys; key++)
                {
                    double product = scores[key] / sum * vv[key + 1, h * dim + c];
                    expected += product; absoluteProducts += Math.Abs(product);
                }
                float value = actual[(row + 1) * width + h * dim + c];
                Assert.True(float.IsFinite(value));
                double tolerance = 2e-6 + 4e-6 * absoluteProducts;
                Assert.True(Math.Abs(value - expected) <= tolerance,
                    $"head={h}, row={row}, column={c}: expected {expected:R}, actual {value:R}, tolerance {tolerance:R}");
            }
        }
        for (int c = 0; c < width; c++)
        {
            Assert.Equal(37f, actual[c]);
            Assert.Equal(37f, actual[(tokens + 1) * width + c]);
        }
        Assert.True(session.TryAttention(y, q, k, v, heads, dim, scale));
        Assert.Equal(actual, yb.GetElementsAsFloat(initial.Length));
        Assert.Equal(2, session.ExecutedAttentions);
        if (queryRows > 0)
        {
            using var wholeHead = new CudaLtx25AttentionSession(allocator, (long)Math.Min(heads, 2) * tokens * keys * sizeof(float));
            Assert.True(wholeHead.TryAttention(y, q, k, v, heads, dim, scale));
            Assert.Equal(tokens, wholeHead.LastQueryTileRows);
            float[] untiled = yb.GetElementsAsFloat(initial.Length);
            for (int i = 0; i < actual.Length; ++i) Assert.InRange(Math.Abs(actual[i] - untiled[i]), 0, 2e-5);
        }
        session.Dispose();
        Assert.Equal(0, session.ScoreCapacityBytes);
        Assert.Throws<ObjectDisposedException>(() => session.TryAttention(y, q, k, v, heads, dim, scale));
    }

    [CudaFact]
    public void Ingredients10752TokenAttentionHasBoundedWorkspaceAndMatchesDoubleSamplesAcrossQueryAndHeadTails()
    {
        const int tokens = 10752, heads = 5, dim = 128, width = heads * dim;
        using var allocator = new CudaAllocator();
        using var session = new CudaLtx25AttentionSession(allocator);
        using var smaller = new CudaLtx25AttentionSession(allocator, 32L << 20);
        var qv = new float[tokens + 2, width]; var kv = new float[tokens + 2, width];
        var vv = new float[tokens + 2, width]; var initial = new float[tokens + 2, width];
        for (int row = 0; row < tokens + 2; ++row)
            for (int channel = 0; channel < width; ++channel)
            {
                qv[row, channel] = MathF.Sin(row * .071f + channel * .19f) * .7f;
                kv[row, channel] = MathF.Cos(row * .023f - channel * .37f) * .8f;
                vv[row, channel] = MathF.Sin(row * .041f + channel * .53f) * .8f + (row > tokens / 2 ? 3f : -2f);
                initial[row, channel] = 37f;
            }
        using var qb = Tensor.FromArray(allocator, qv); using var kb = Tensor.FromArray(allocator, kv);
        using var vb = Tensor.FromArray(allocator, vv); using var yb = Tensor.FromArray(allocator, initial);
        using var q = qb.Narrow(0, 1, tokens); using var k = kb.Narrow(0, 1, tokens);
        using var v = vb.Narrow(0, 1, tokens); using var y = yb.Narrow(0, 1, tokens);
        float scale = 1 / MathF.Sqrt(dim);
        Assert.True(session.TryAttention(y, q, k, v, heads, dim, scale));
        Assert.InRange(session.LastQueryTileRows, 1, tokens - 1);
        Assert.InRange(session.LastHeadTileCount, 2, heads - 1);
        Assert.InRange(session.ScoreCapacityBytes, 1, 256L << 20);
        float[] actual = yb.GetElementsAsFloat(initial.Length);
        Assert.All(actual, value => Assert.True(float.IsFinite(value)));
        var rows = new HashSet<int> { 0, 1, tokens - 2, tokens - 1 };
        for (int boundary = session.LastQueryTileRows; boundary < tokens; boundary += session.LastQueryTileRows)
            foreach (int row in new[] { boundary - 1, boundary }) rows.Add(row);
        var scores = new double[tokens];
        double maximumError = 0;
        foreach (int row in rows.Order())
            for (int head = 0; head < heads; ++head)
            {
                for (int key = 0; key < tokens; ++key)
                {
                    double dot = 0;
                    for (int channel = 0; channel < dim; ++channel)
                        dot += (double)qv[row + 1, head * dim + channel] * kv[key + 1, head * dim + channel];
                    scores[key] = dot * scale;
                }
                double maximum = scores.Max(), sum = 0;
                for (int key = 0; key < tokens; ++key) sum += scores[key] = Math.Exp(scores[key] - maximum);
                for (int channel = 0; channel < dim; ++channel)
                {
                    double expected = 0, products = 0;
                    for (int key = 0; key < tokens; ++key)
                    {
                        double product = scores[key] / sum * vv[key + 1, head * dim + channel];
                        expected += product; products += Math.Abs(product);
                    }
                    double error = Math.Abs(actual[(row + 1) * width + head * dim + channel] - expected);
                    maximumError = Math.Max(maximumError, error);
                    Assert.True(error <= 2e-6 + 4e-6 * products,
                        $"10752-token oracle row={row}, head={head}, channel={channel}: error={error:G9}, expected={expected:G9}");
                }
            }
        for (int channel = 0; channel < width; ++channel)
        {
            Assert.Equal(37f, actual[channel]); Assert.Equal(37f, actual[(tokens + 1) * width + channel]);
        }
        // An independently sized workspace changes every query boundary. Compare
        // all output elements, not just the double-oracle sample rows.
        Assert.True(smaller.TryAttention(y, q, k, v, heads, dim, scale));
        Assert.True(smaller.LastQueryTileRows < session.LastQueryTileRows);
        Assert.InRange(smaller.ScoreCapacityBytes, 1, 32L << 20);
        float[] retiled = yb.GetElementsAsFloat(initial.Length);
        for (int i = 0; i < actual.Length; ++i) Assert.InRange(Math.Abs(actual[i] - retiled[i]), 0, 2e-5);
        session.ReleaseWorkspace(); Assert.Equal(0, session.ScoreCapacityBytes);
        Assert.True(session.TryAttention(y, q, k, v, heads, dim, scale));
        Assert.Equal(actual, yb.GetElementsAsFloat(initial.Length));
        Assert.Equal(2, session.ExecutedAttentions);
        Console.WriteLine($"[ltx25-attention-oracle] tokens={tokens} heads={heads} dim={dim} queryTile={session.LastQueryTileRows} " +
            $"headTile={session.LastHeadTileCount} scoreBytes={session.ScoreCapacityBytes} sampledRows={rows.Count} maximumError={maximumError:G9}");
    }

    [CudaFact]
    public void InsufficientBudgetInvalidShapesAndAliasedOutputRefuseBeforeDispatch()
    {
        using var allocator = new CudaAllocator();
        using var session = new CudaLtx25AttentionSession(allocator, 4);
        using var sufficient = new CudaLtx25AttentionSession(allocator);
        using var q = Tensor.FromArray(allocator, new float[3, 8]);
        using var y = Tensor.FromArray(allocator, new float[,] { { 37, 37, 37, 37, 37, 37, 37, 37 },
            { 37, 37, 37, 37, 37, 37, 37, 37 }, { 37, 37, 37, 37, 37, 37, 37, 37 } });
        using var shortKeys = q.Narrow(0, 0, 2);
        Assert.False(session.TryAttention(y, q, q, q, 2, 4, .5f));
        Assert.False(sufficient.TryAttention(q, q, q, q, 2, 4, .5f));
        Assert.False(sufficient.TryAttention(y, q, shortKeys, q, 2, 4, .5f));
        Assert.False(sufficient.TryAttention(y, q, q, q, 2, 3, .5f));
        Assert.False(sufficient.TryAttention(y, q, q, q, 2, 4, float.NaN));
        Assert.All(y.GetElementsAsFloat(24), x => Assert.Equal(37, x));
        Assert.Equal(0, sufficient.ExecutedAttentions);
        Assert.Equal(0, sufficient.ScoreCapacityBytes);
        Assert.Equal(0, session.ScoreCapacityBytes);
    }

    [CudaFact]
    public void WorkspaceGrowsReusesAndRebuildsAfterResidencyRelease()
    {
        using var allocator = new CudaAllocator();
        using var session = new CudaLtx25AttentionSession(allocator, 4096);
        long previous = 0;
        foreach (int tokens in new[] { 3, 7, 3 })
        {
            using var q = Tensor.FromArray(allocator, new float[tokens, 8]);
            using var v = new Tensor(allocator, DType.Float32, tokens, 8);
            using var y = new Tensor(allocator, DType.Float32, tokens, 8);
            Ops.Fill(v, 80000f);
            Assert.True(session.TryAttention(y, q, q, v, 2, 4, 0));
            Assert.All(y.GetElementsAsFloat(tokens * 8), x => Assert.InRange(x, 79999.98f, 80000.02f));
            Assert.True(session.ScoreCapacityBytes >= previous);
            previous = session.ScoreCapacityBytes;
        }
        session.ReleaseWorkspace();
        Assert.Equal(0, session.ScoreCapacityBytes);
        using var input = Tensor.FromArray(allocator, new float[3, 8]);
        using var output = new Tensor(allocator, DType.Float32, 3, 8);
        Assert.True(session.TryAttention(output, input, input, input, 2, 4, 1));
        Assert.All(output.GetElementsAsFloat(24), x => Assert.Equal(0, x));
        Assert.True(session.ScoreCapacityBytes > 0);
        Assert.Equal(4, session.ExecutedAttentions);
    }
}
