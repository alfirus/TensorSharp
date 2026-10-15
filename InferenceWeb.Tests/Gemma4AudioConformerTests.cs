using System.Text.Json;
using System.Buffers.Binary;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class Gemma4AudioConformerTests(ITestOutputHelper output)
{
    [Fact]
    public void ManagedAudioPrecisionDescriptorsMatchTheVersionedNativeAbi()
    {
        Assert.Equal(8, IntPtr.Size); // TensorSharp's supported native desktop targets are 64-bit.
        var assembly = typeof(GgmlGemma4AudioOps).Assembly;
        Type Descriptor(string name) => assembly.GetType("TensorSharp.GGML." + name, throwOnError: true)!;
        var block = Descriptor("GgmlGemma4AudioBlockArgs");
        var sscp = Descriptor("GgmlGemma4AudioSscpArgs");
        Assert.Equal(184, Marshal.SizeOf(block));
        Assert.Equal(172, Marshal.OffsetOf(block, "RequireFloat32Precision").ToInt32());
        Assert.Equal(176, Marshal.OffsetOf(block, "PrecisionPolicyVersion").ToInt32());
        Assert.Equal(120, Marshal.SizeOf(sscp));
        Assert.Equal(112, Marshal.OffsetOf(sscp, "RequireFloat32Precision").ToInt32());
        Assert.Equal(112, Marshal.SizeOf(Descriptor("GgmlGemma4AudioLinearView")));
        Assert.Equal(112, Marshal.SizeOf(Descriptor("GgmlGemma4AudioConvolutionView")));
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void NativeCpuCompleteConformerMatchesIndependentDoubleFixture() => Verify(GgmlBackendType.Cpu);

    [GgmlFact(BackendType.GgmlMetal)]
    public void NativeMetalCompleteConformerMatchesIndependentDoubleFixture() => Verify(GgmlBackendType.Metal);

    [GgmlFact(BackendType.GgmlCuda)]
    public void NativeCudaCompleteConformerMatchesIndependentDoubleFixture() => Verify(GgmlBackendType.Cuda);

    [GgmlFact(BackendType.GgmlCpu)]
    public void NativeCpuLongCompleteConformerMatchesIndependentDoubleFixture() => Verify(GgmlBackendType.Cpu, true);

    [GgmlFact(BackendType.GgmlCpu)]
    public void NativeCpuPreciseCompleteConformerMatchesIndependentDoubleFixture() => Verify(GgmlBackendType.Cpu, requirePrecision: true);

    [GgmlFact(BackendType.GgmlCpu)]
    public void NativeCpuPreciseLongCompleteConformerMatchesIndependentDoubleFixture() => Verify(GgmlBackendType.Cpu, true, true);

    [GgmlFact(BackendType.GgmlCuda)]
    public void NativeCudaLongCompleteConformerMatchesIndependentDoubleFixture() => Verify(GgmlBackendType.Cuda, true);

    [Gemma4AudioPreciseMetalFact]
    public void NativeMetalPreciseLongCompleteConformerMatchesIndependentDoubleFixture()
        => Verify(GgmlBackendType.Metal, true);

    private void Verify(GgmlBackendType backend, bool preciseLong = false, bool requirePrecision = false)
    {
        using var fixture = JsonDocument.Parse(File.ReadAllText(FixturePath));
        var root = fixture.RootElement;
        int hidden = root.GetProperty("hidden").GetInt32(), heads = root.GetProperty("heads").GetInt32();
        int head = root.GetProperty("head").GetInt32(), kernel = root.GetProperty("kernel").GetInt32();
        var allocator = new GgmlAllocator(new GgmlContext([0], backend), 0);
        var owned = new List<Tensor>();
        Tensor TensorFrom(JsonElement element, params long[] shape)
        {
            var tensor = new Tensor(allocator, DType.Float32, shape);
            owned.Add(tensor);
            tensor.SetElementsAsFloat(element.EnumerateArray().Select(value => value.GetSingle()).ToArray());
            return tensor;
        }
        try
        {
            var normalized = root.GetProperty("norms").EnumerateArray().Select(value => TensorFrom(value, hidden)).ToArray();
            var perDimension = TensorFrom(root.GetProperty("per_dim_scale"), head);
            var depthwise = TensorFrom(root.GetProperty("depthwise"), hidden, kernel);
            var linears = root.GetProperty("linears").EnumerateArray().Select(item => new GgmlGemma4AudioLinear(
                TensorFrom(item.GetProperty("weight"), item.GetProperty("output").GetInt32(), item.GetProperty("input").GetInt32()),
                TensorFrom(item.GetProperty("bias"), item.GetProperty("output").GetInt32()), item.GetProperty("has_clamp").GetBoolean(),
                item.GetProperty("input_min").GetSingle(), item.GetProperty("input_max").GetSingle(),
                item.GetProperty("output_min").GetSingle(), item.GetProperty("output_max").GetSingle())).ToArray();
            foreach (var item in root.GetProperty("cases").EnumerateArray())
            {
                if (item.GetProperty("precise_long").GetBoolean() != preciseLong) continue;
                int sequence = item.GetProperty("sequence").GetInt32(), past = item.GetProperty("past").GetInt32(), future = item.GetProperty("future").GetInt32();
                var input = TensorFrom(item.GetProperty("input"), sequence, hidden);
                var relative = TensorFrom(item.GetProperty("relative"), past + future + 1, hidden);
                using var result = new Tensor(allocator, DType.Float32, sequence, hidden);
                Tensor[] norms = new Tensor[11];
                if (!item.GetProperty("absent_norms").GetBoolean()) Array.Copy(normalized, norms, normalized.Length);
                norms[9] = perDimension; norms[10] = depthwise;
                float epsilon = item.GetProperty("epsilon").GetSingle(), cap = item.GetProperty("cap").GetSingle();
                Assert.True(GgmlGemma4AudioOps.TryConformerBlock(result, input, relative, norms, linears,
                    sequence, heads, head, past, future, kernel, epsilon, cap,
                    item.GetProperty("grad_clip").GetSingle(), item.GetProperty("residual_weight").GetSingle(), requirePrecision));
                var actual = result.GetElementsAsFloat(sequence * hidden);
                var expected = item.GetProperty("expected").EnumerateArray().Select(value => value.GetDouble()).ToArray();
                double error = actual.Zip(expected, (a, b) => Math.Abs(a - b)).Max();
                output.WriteLine($"{item.GetProperty("name").GetString()}: maximum absolute error {error:R}");
                Assert.Equal(DType.Float32, result.ElementType);
                Assert.All(actual, value => Assert.True(float.IsFinite(value)));
                Assert.True(error <= 3e-5, $"{item.GetProperty("name").GetString()}: absolute error {error:R} exceeds independent 3e-5 gate.");
                var descriptor = new GgmlGemma4AudioBlock(relative, norms, linears, past, future, kernel, epsilon, cap,
                    item.GetProperty("grad_clip").GetSingle(), item.GetProperty("residual_weight").GetSingle(), requirePrecision);
                var second = descriptor with { Relative = TensorFrom(item.GetProperty("relative_two"), past + future + 1, hidden) };
                var third = descriptor with { Relative = TensorFrom(item.GetProperty("relative_three"), past + future + 1, hidden) };
                foreach (int count in new[] { 2, 3 })
                {
                    Assert.True(GgmlGemma4AudioOps.TryConformerStack(result, input,
                        count == 2 ? [descriptor, second] : [descriptor, second, third], sequence, heads, head));
                    var stackActual = result.GetElementsAsFloat(sequence * hidden);
                    var stackExpected = item.GetProperty(count == 2 ? "expected_two" : "expected_three")
                        .EnumerateArray().Select(value => value.GetDouble()).ToArray();
                    double stackError = stackActual.Zip(stackExpected, (a, b) => Math.Abs(a - b)).Max();
                    output.WriteLine($"{item.GetProperty("name").GetString()} stack{count}: {stackError:R}");
                    Assert.All(stackActual, value => Assert.True(float.IsFinite(value)));
                    Assert.True(stackError <= 3e-5, $"Independent stack{count} absolute error {stackError:R} exceeds unchanged 3e-5 gate.");
                }
                Assert.False(GgmlGemma4AudioOps.TryConformerStack(result, input, [], sequence, heads, head));
                Assert.False(GgmlGemma4AudioOps.TryConformerBlock(result, input, relative, norms, linears,
                    0, heads, head, past, future, kernel, epsilon, cap));
                Assert.False(GgmlGemma4AudioOps.TryConformerBlock(result, input, relative, norms[..10], linears,
                    sequence, heads, head, past, future, kernel, epsilon, cap));
                if (preciseLong && backend == GgmlBackendType.Metal && sequence == 128 &&
                    Environment.GetEnvironmentVariable("TS_GEMMA4_AUDIO_PRECISE_ROWS") == "1")
                {
                    using var oversizedInput = new Tensor(allocator, DType.Float32, 16000, hidden);
                    using var oversizedResult = new Tensor(allocator, DType.Float32, 16000, hidden);
                    var boundaryError = Assert.Throws<InvalidOperationException>(() => GgmlGemma4AudioOps.TryConformerStack(
                        oversizedResult, oversizedInput, [descriptor, descriptor, descriptor], 16000, heads, head));
                    Assert.Contains("bounded graph capacity", boundaryError.Message, StringComparison.Ordinal);
                }
            }
            if (!preciseLong) VerifyQuantizedIntegerFixture(allocator, owned);
        }
        finally
        {
            foreach (var tensor in owned) { GgmlBasicOps.InvalidateTensorHostBuffer(tensor); tensor.Dispose(); }
            GgmlBasicOps.ReleaseReuseComputeBuffers(); GgmlBasicOps.ClearHostBufferCache();
        }
    }

    private void VerifyQuantizedIntegerFixture(GgmlAllocator allocator, List<Tensor> owned)
    {
        const int hidden = 32, sequence = 17, kernel = 5;
        var shapes = new[] { (hidden, 64), (64, hidden), (hidden, hidden), (hidden, hidden),
            (hidden, hidden), (hidden, hidden), (hidden, 2 * hidden), (hidden, hidden), (hidden, 64), (64, hidden) };
        var dense = new GgmlGemma4AudioLinear[10];
        var compressed = new GgmlGemma4AudioLinear[10];
        var pins = new List<GCHandle>();
        Tensor Make(float[] values, params long[] shape)
        {
            var tensor = new Tensor(allocator, DType.Float32, shape);
            owned.Add(tensor); tensor.SetElementsAsFloat(values); return tensor;
        }
        try
        {
            for (int index = 0; index < dense.Length; ++index)
            {
                var (inputs, outputs) = shapes[index];
                var quant = new sbyte[inputs * outputs];
                // Nonzero integer FFN-up and sigmoid-gate operands exercise the
                // packed row layout, while zero down/data branches make their
                // residual exactly zero. Attention is an identity V/output and
                // a one-key window, so the independent final answer is 2*x.
                if (index is 0 or 8)
                    for (int i = 0; i < quant.Length; ++i) quant[i] = (sbyte)((i * 7 + index) % 11 - 5);
                if (index is 4 or 5 or 7)
                    for (int i = 0; i < hidden; ++i) quant[i * inputs + i] = 32;
                if (index == 6)
                    for (int i = 0; i < hidden; ++i) quant[(hidden + i) * inputs + i] = 32;
                var values = quant.Select(value => value / 32f).ToArray();
                var weight = Make(values, outputs, inputs);
                dense[index] = new(weight);
                var bytes = new byte[outputs * (inputs / 32) * 34];
                for (int block = 0; block < quant.Length / 32; ++block)
                {
                    BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(block * 34),
                        BitConverter.HalfToInt16Bits((System.Half)(1 / 32f)));
                    for (int d = 0; d < 32; ++d) bytes[block * 34 + 2 + d] = unchecked((byte)quant[block * 32 + d]);
                }
                pins.Add(GCHandle.Alloc(bytes, GCHandleType.Pinned));
                compressed[index] = dense[index] with { CompressedWeight = pins[^1].AddrOfPinnedObject(),
                    CompressedWeightType = 8, CompressedWeightBytes = bytes.Length };
            }
            float[] x = Enumerable.Range(0, sequence * hidden).Select(index => (index % 2 == 0 ? 1f : -1f) / 128).ToArray();
            var input = Make(x, sequence, hidden); var relative = Make(new float[hidden], 1, hidden);
            Tensor[] norms = new Tensor[11];
            norms[10] = Make(Enumerable.Range(0, hidden * kernel).Select(index => (index % 7 - 3) / 16f).ToArray(), hidden, kernel);
            using var denseResult = new Tensor(allocator, DType.Float32, sequence, hidden);
            using var quantResult = new Tensor(allocator, DType.Float32, sequence, hidden);
            foreach (var (result, weights) in new[] { (denseResult, dense), (quantResult, compressed) })
            {
                Assert.True(GgmlGemma4AudioOps.TryConformerBlock(result, input, relative, norms, weights,
                    sequence, 2, 16, 0, 0, kernel, 1e-6f, 2, 1024));
                var values = result.GetElementsAsFloat(x.Length);
                double error = values.Zip(x, (a, b) => Math.Abs((double)a - 2 * b)).Max();
                output.WriteLine($"{(ReferenceEquals(result, quantResult) ? "Q8" : "dense")} integer fixture: {error:R}");
                Assert.All(values, value => Assert.True(float.IsFinite(value)));
                Assert.True(error <= 3e-5, $"Independent integer fixture error {error:R} exceeds unchanged 3e-5 gate.");
            }
            var a = denseResult.GetElementsAsFloat(x.Length); var b = quantResult.GetElementsAsFloat(x.Length);
            Assert.InRange(a.Zip(b, (left, right) => Math.Abs(left - right)).Max(), 0, 3e-5);
            foreach (int count in new[] { 2, 3 })
            {
                var descriptor = new GgmlGemma4AudioBlock(relative, norms, compressed, 0, 0, kernel, 1e-6f, 2, 1024);
                Assert.True(GgmlGemma4AudioOps.TryConformerStack(quantResult, input,
                    Enumerable.Repeat(descriptor, count).ToArray(), sequence, 2, 16));
                var values = quantResult.GetElementsAsFloat(x.Length);
                double error = values.Zip(x, (value, initial) => Math.Abs((double)value - (1 << count) * initial)).Max();
                output.WriteLine($"Q8 integer stack{count}: {error:R}");
                Assert.All(values, value => Assert.True(float.IsFinite(value)));
                Assert.True(error <= 3e-5, $"Independent Q8 stack{count} error {error:R} exceeds unchanged 3e-5 gate.");
            }
        }
        finally
        {
            foreach (var pin in pins)
            {
                try { GgmlBasicOps.InvalidateHostBuffer(pin.AddrOfPinnedObject()); }
                finally { pin.Free(); }
            }
        }
    }

    private static string FixturePath
    {
        get
        {
            for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory != null; directory = directory.Parent)
                if (File.Exists(Path.Combine(directory.FullName, "TensorSharp.slnx")))
                    return Path.Combine(directory.FullName, "InferenceWeb.Tests", "Fixtures", "EmbeddingGemma2", "audio-conformer.json");
            throw new DirectoryNotFoundException("TensorSharp source tree is required for the independent audio conformer fixture.");
        }
    }
}

/// <summary>Native getenv must receive this diagnostic before process startup;
/// managed environment mutation does not configure the native runtime.</summary>
[Xunit.Sdk.TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
public sealed class Gemma4AudioPreciseMetalFactAttribute : FactAttribute, Xunit.Sdk.ITraitAttribute
{
    public string RequiresValue => BackendType.GgmlMetal.ToString();
    public Gemma4AudioPreciseMetalFactAttribute()
        => Skip = TestGates.GgmlPinSkip(BackendType.GgmlMetal) ??
            (Environment.GetEnvironmentVariable("TS_GEMMA4_AUDIO_PRECISE_ROWS") == "1" ||
             Environment.GetEnvironmentVariable("TS_GEMMA4_AUDIO_METAL_F32_MM") == "1" ||
             Environment.GetEnvironmentVariable("TS_GEMMA4_AUDIO_METAL_MPS_MM") == "1" ? null :
                "Requires TS_GEMMA4_AUDIO_PRECISE_ROWS=1, TS_GEMMA4_AUDIO_METAL_F32_MM=1, or TS_GEMMA4_AUDIO_METAL_MPS_MM=1 at process startup for native F32 precision.");
}
