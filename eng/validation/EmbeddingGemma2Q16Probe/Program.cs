using System.Diagnostics;
using System.Text.Json;
using TensorSharp.Models;
using TensorSharp.Models.Embeddings;

if (!ManagedEmbeddingQ8Q16Avx512Matrix.IsSupported)
{
    Console.Error.WriteLine("AVX512F/BW are unavailable; checks were not run.");
    return 2;
}
var cases = new List<object>();
foreach (var (inputs, outputs, rows, threads, amplitude) in new[] {
    (32, 1, 1, 1, 8f), (64, 7, 2, 1, 8f), (96, 17, 3, 3, 8f),
    (512, 31, 7, 4, 8f), (1024, 67, 17, 4, 8f), (512, 512, 32, 8, 8f),
    (512, 4096, 128, 8, 8f), (64, 9, 7, 3, 80_000_000f), (64, 9, 7, 3, 1e-37f), (64, 9, 7, 3, 1e-42f) })
{
    int blocks = inputs / 32;
    var weights = new byte[checked(outputs * blocks * 34)];
    for (int column = 0; column < outputs; ++column)
    for (int block = 0; block < blocks; ++block)
    {
        int offset = (column * blocks + block) * 34;
        ushort bits = BitConverter.HalfToUInt16Bits((Half)(.005f * (1 + (column + block) % 9)));
        weights[offset] = (byte)bits; weights[offset + 1] = (byte)(bits >> 8);
        for (int d = 0; d < 32; ++d)
            weights[offset + 2 + d] = unchecked((byte)(sbyte)(-128 + (column * 41 + block * 17 + d * 19) % 256));
    }
    var x = new float[checked(inputs * rows)];
    for (int row = 0; row < rows; ++row)
    for (int d = 0; d < inputs; ++d)
        x[row * inputs + d] = row == 0 ? 0 : row % 3 == 1 ? ((d % 32) - 16) * (amplitude / 16)
            : MathF.Sin((row * inputs + d) * .017f) * (amplitude / 8) * (1 + row % 5);
    var matrix = new ManagedEmbeddingQ8Q16Avx512Matrix(weights, inputs, outputs);
    using var pool = new CpuWorkerPool(threads);
    var output = new float[checked(rows * outputs)];
    matrix.Multiply(x, rows, output, pool);
    double maxQuantizedError = 0, maxFloatError = 0, maxBoundFraction = 0;
    for (int row = 0; row < rows; ++row)
    for (int column = 0; column < outputs; ++column)
    {
        float expected = 0, expectedF32 = 0;
        bool f32Fallback = false;
        double exactFloat = 0, quantizationBound = 0, absoluteProducts = 0;
        for (int block = 0; block < blocks; ++block)
        {
            int io = row * inputs + block * 32, wo = (column * blocks + block) * 34;
            float max = 0;
            for (int d = 0; d < 32; ++d) max = Math.Max(max, Math.Abs(x[io + d]));
            float scale = max / 32767f;
            f32Fallback |= max > 0f && scale < 1.17549435e-38f;
            float inverse = scale == 0 ? 0 : 1f / scale;
            float weightScale = (float)BitConverter.UInt16BitsToHalf((ushort)(weights[wo] | weights[wo + 1] << 8));
            int dot = 0;
            for (int d = 0; d < 32; ++d)
            {
                float value = float.IsFinite(inverse) ? x[io + d] * inverse : x[io + d] / scale;
                int quant = (int)MathF.Round(value, MidpointRounding.ToEven);
                int weight = unchecked((sbyte)weights[wo + 2 + d]);
                dot += quant * weight;
                double dequantizedWeight = (float)(weightScale * weight);
                expectedF32 = MathF.FusedMultiplyAdd(x[io + d], (float)dequantizedWeight, expectedF32);
                exactFloat += x[io + d] * dequantizedWeight;
                absoluteProducts += Math.Abs(x[io + d] * dequantizedWeight);
                quantizationBound += .501 * scale * Math.Abs(dequantizedWeight);
            }
            expected = MathF.FusedMultiplyAdd(dot * weightScale, scale, expected);
        }
        if (f32Fallback) expected = expectedF32;
        if (BitConverter.SingleToInt32Bits(expected) != BitConverter.SingleToInt32Bits(output[row * outputs + column]))
            throw new InvalidDataException($"Scalar integer oracle mismatch at [{row},{column}]: {expected:R} vs {output[row * outputs + column]:R}.");
        double error = Math.Abs(output[row * outputs + column] - exactFloat);
        // Include half an F32 subnormal ULP per product/addition at tiny ranges.
        double bound = quantizationBound + 4e-6 * absoluteProducts + inputs * .7e-45 + 1e-44;
        if (error > bound) throw new InvalidDataException($"Independent F32 product error {error} exceeds {bound}.");
        maxQuantizedError = Math.Max(maxQuantizedError, Math.Abs(output[row * outputs + column] - expected));
        maxFloatError = Math.Max(maxFloatError, error);
        maxBoundFraction = Math.Max(maxBoundFraction, error / bound);
    }
    var single = new float[outputs];
    matrix.Multiply(x.AsSpan((rows - 1) * inputs, inputs).ToArray(), 1, single, pool);
    if (!single.SequenceEqual(output.AsSpan((rows - 1) * outputs, outputs).ToArray()))
        throw new InvalidDataException("Single and batched paths differ.");
    var samples = new List<double>();
    for (int repeat = 0; repeat < 7; ++repeat)
    {
        var timer = Stopwatch.StartNew(); matrix.Multiply(x, rows, output, pool); samples.Add(timer.Elapsed.TotalMilliseconds);
    }
    cases.Add(new { inputs, outputs, rows, threads, amplitude, max_quantized_error = maxQuantizedError,
        max_f32_error = maxFloatError, max_error_bound_fraction = maxBoundFraction, scalar_reference_bit_exact = true,
        subnormal_f32_fallback_case = amplitude < 1e-30f,
        samples_ms = samples, passed = true });
    Console.WriteLine($"{inputs}x{outputs}, rows{rows}: bit-exact integer oracle, F32 error {maxFloatError:R}, median {samples.Order().ElementAt(3):F4}ms");
}
if (args.Length == 1)
{
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(args[0]))!);
    File.WriteAllText(args[0], JsonSerializer.Serialize(new { completed = true, cases,
        limitations = "Standalone arithmetic and timing only; model quality and performance are separate gates." },
        new JsonSerializerOptions { WriteIndented = true }) + "\n");
}
return 0;
