using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text.Json;
using TensorSharp;
using TensorSharp.GGML;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class Gemma4AudioSscpGraphTests(ITestOutputHelper output)
{
    [GgmlFact(BackendType.GgmlCpu)]
    public void NativeCpuWholeSscpMatchesIndependentDoubleFixture() => Verify(GgmlBackendType.Cpu);

    [GgmlFact(BackendType.GgmlCpu)]
    public void NativeCpuPreciseWholeSscpMatchesIndependentDoubleFixture() => Verify(GgmlBackendType.Cpu, true);

    [GgmlFact(BackendType.GgmlMetal)]
    public void NativeMetalWholeSscpMatchesIndependentDoubleFixture() => Verify(GgmlBackendType.Metal);

    [GgmlFact(BackendType.GgmlCuda)]
    public void NativeCudaWholeSscpMatchesIndependentDoubleFixture() => Verify(GgmlBackendType.Cuda);

    private void Verify(GgmlBackendType backend, bool requirePrecision = false)
    {
        using var fixture = JsonDocument.Parse(File.ReadAllText(FixturePath));
        var allocator = new GgmlAllocator(new GgmlContext([0], backend), 0);
        var tensors = new List<Tensor>();
        var pins = new List<GCHandle>();
        Tensor Make(float[] values, params long[] shape)
        {
            var tensor = new Tensor(allocator, DType.Float32, shape);
            tensors.Add(tensor); tensor.SetElementsAsFloat(values); return tensor;
        }
        float[] Values(JsonElement element) => element.EnumerateArray().Select(value => value.GetSingle()).ToArray();
        Tensor Optional(JsonElement element, int columns) => element.ValueKind == JsonValueKind.Null ? null! : Make(Values(element), columns);
        try
        {
            foreach (var item in fixture.RootElement.GetProperty("cases").EnumerateArray())
            {
                string name = item.GetProperty("name").GetString()!;
                int frequency = item.GetProperty("width").GetInt32(), time = item.GetProperty("height").GetInt32();
                int channels = item.GetProperty("channels").GetInt32();
                float[] planar = Values(item.GetProperty("input")), rows = new float[planar.Length];
                for (int t = 0; t < time; ++t)
                for (int f = 0; f < frequency; ++f)
                for (int c = 0; c < channels; ++c)
                    rows[t * frequency * channels + f * channels + c] = planar[(c * time + t) * frequency + f];
                var input = Make(rows, time, frequency * channels);
                var convolutions = item.GetProperty("convolutions").EnumerateArray().Select(conv =>
                {
                    int inputs = conv.GetProperty("input_channels").GetInt32(), outputs = conv.GetProperty("output_channels").GetInt32();
                    int kf = conv.GetProperty("kernel_frequency").GetInt32(), kt = conv.GetProperty("kernel_time").GetInt32();
                    return new GgmlGemma4AudioConvolution(Make(Values(conv.GetProperty("weight")), outputs, inputs, kt, kf),
                        inputs, outputs, kf, kt, Optional(conv.GetProperty("norm_weight"), outputs), Optional(conv.GetProperty("norm_bias"), outputs));
                }).ToArray();
                var projection = item.GetProperty("projection");
                int columns = projection.GetProperty("input").GetInt32(), outputs = projection.GetProperty("output").GetInt32();
                float[] matrix = Values(projection.GetProperty("weight"));
                var linear = new GgmlGemma4AudioLinear(Make(matrix, outputs, columns), Make(Values(projection.GetProperty("bias")), outputs));
                int outputRows = item.GetProperty("output_rows").GetInt32();
                var result = Make(new float[outputRows * outputs], outputRows, outputs);
                double[] expected = item.GetProperty("expected").EnumerateArray().Select(value => value.GetDouble()).ToArray();
                void Check(GgmlGemma4AudioLinear descriptor, string variant)
                {
                    Assert.True(GgmlGemma4AudioOps.TrySscp(result, input, convolutions, descriptor,
                        frequency, time, channels, item.GetProperty("epsilon").GetSingle(), requirePrecision));
                    float[] actual = result.GetElementsAsFloat(expected.Length);
                    double error = actual.Zip(expected, (a, b) => Math.Abs(a - b)).Max();
                    output.WriteLine($"{name}/{variant}: maximum absolute error {error:R}");
                    Assert.Equal(DType.Float32, result.ElementType);
                    Assert.All(actual, value => Assert.True(float.IsFinite(value)));
                    Assert.True(error <= 3e-5, $"{name}/{variant}: independent scalar absolute error {error:R} exceeds 3e-5.");
                }
                Check(linear, "dense");
                if (projection.GetProperty("q8_scale").ValueKind != JsonValueKind.Null)
                {
                    float scale = projection.GetProperty("q8_scale").GetSingle();
                    byte[] quantized = new byte[columns / 32 * outputs * 34];
                    for (int row = 0; row < outputs; ++row)
                    for (int block = 0; block < columns / 32; ++block)
                    {
                        int offset = (row * (columns / 32) + block) * 34;
                        BinaryPrimitives.WriteUInt16LittleEndian(quantized.AsSpan(offset, 2), BitConverter.HalfToUInt16Bits((System.Half)scale));
                        for (int i = 0; i < 32; ++i)
                        {
                            float value = matrix[row * columns + block * 32 + i];
                            int q = checked((int)MathF.Round(value / scale));
                            Assert.Equal(value, q * scale); Assert.InRange(q, -127, 127);
                            quantized[offset + 2 + i] = unchecked((byte)(sbyte)q);
                        }
                    }
                    pins.Add(GCHandle.Alloc(quantized, GCHandleType.Pinned));
                    Check(linear with { CompressedWeight = pins[^1].AddrOfPinnedObject(), CompressedWeightType = 8,
                        CompressedWeightBytes = quantized.Length }, "exact-q8");
                }
                Assert.False(GgmlGemma4AudioOps.TrySscp(result, input, convolutions[..1], linear, frequency, time, channels,
                    item.GetProperty("epsilon").GetSingle()));
                Assert.False(GgmlGemma4AudioOps.TrySscp(result, input, convolutions, linear, frequency, time, channels, float.NaN));
            }
        }
        finally
        {
            foreach (var pin in pins)
            {
                try { GgmlBasicOps.InvalidateHostBuffer(pin.AddrOfPinnedObject()); }
                finally { pin.Free(); }
            }
            foreach (var tensor in tensors) { GgmlBasicOps.InvalidateTensorHostBuffer(tensor); tensor.Dispose(); }
            GgmlBasicOps.ReleaseReuseComputeBuffers(); GgmlBasicOps.ClearHostBufferCache();
        }
    }

    private static string FixturePath
    {
        get
        {
            for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory != null; directory = directory.Parent)
                if (File.Exists(Path.Combine(directory.FullName, "TensorSharp.slnx")))
                    return Path.Combine(directory.FullName, "InferenceWeb.Tests", "Fixtures", "EmbeddingGemma2", "audio-sscp.json");
            throw new DirectoryNotFoundException("TensorSharp source tree is required for the independent audio SSCP fixture.");
        }
    }
}
