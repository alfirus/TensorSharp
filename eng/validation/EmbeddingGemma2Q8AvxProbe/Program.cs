using System.Diagnostics;
using System.Text.Json;
using TensorSharp.Models;
using TensorSharp.Models.Embeddings;

if (!ManagedEmbeddingQ8AvxMatrix.IsSupported)
{
    Console.Error.WriteLine("AVX2 is unavailable; kernel checks were not run.");
    return 2;
}

var report = new List<object>();
foreach (var (inputs, outputs, rows, threads, minimumWeight, floatScale, amplitude) in new[] {
    (32,1,1,1,-127,false,8f), (64,7,2,1,-127,false,8f), (96,8,3,3,-127,false,8f), (512,9,4,4,-127,false,8f),
    (512,31,7,4,-127,false,8f), (1024,67,17,4,-127,false,8f), (64,8,4,3,-128,false,8f), (512,17,17,4,-128,false,8f),
    (512,512,32,8,-127,false,8f), (512,4096,128,8,-127,false,8f),
    (64,9,7,4,-128,true,8_400_001f), (512,17,17,4,-128,true,80_000_000f),
    (64,9,7,4,-128,true,1e-37f), (512,4096,128,8,-127,true,8f) })
{
    int blocks = inputs / 32;
    var weights = new byte[checked(outputs * blocks * 34)];
    for (int column = 0; column < outputs; ++column)
    for (int block = 0; block < blocks; ++block)
    {
        int offset = (column * blocks + block) * 34;
        float scale = .005f * (1 + (column + block) % 9);
        ushort bits = BitConverter.HalfToUInt16Bits((Half)scale);
        weights[offset] = (byte)bits; weights[offset + 1] = (byte)(bits >> 8);
        for (int d = 0; d < 32; ++d)
            weights[offset + 2 + d] = unchecked((byte)(sbyte)(minimumWeight + (column * 41 + block * 17 + d * 19) % (128 - minimumWeight)));
    }
    var x = new float[checked(inputs * rows)];
    for (int row = 0; row < rows; ++row)
    for (int d = 0; d < inputs; ++d)
        x[row * inputs + d] = row == 0 ? 0 : row % 3 == 1 ? ((d % 32) - 16) * (amplitude / 16) :
            MathF.Sin((row * inputs + d) * .017f) * (amplitude / 8) * (1 + row % 5);
    var matrix = new ManagedEmbeddingQ8AvxMatrix(weights, inputs, outputs, preserveFloat32ActivationScales: floatScale);
    var matrix512 = ManagedEmbeddingQ8Avx512Matrix.IsSupported
        ? new ManagedEmbeddingQ8Avx512Matrix(weights, inputs, outputs, preserveFloat32ActivationScales: floatScale) : null;
    using var pool = new CpuWorkerPool(threads);
    var output = new float[checked(rows * outputs)];
    matrix.Multiply(x, rows, output, pool);
    var output512 = new float[output.Length];
    if (matrix512 != null)
    {
        matrix512.Multiply(x, rows, output512, pool);
        if (!output512.SequenceEqual(output)) throw new InvalidDataException("AVX512 and AVX2 arithmetic differ.");
    }
    double maximum = 0;
    for (int row = 0; row < rows; ++row)
    for (int column = 0; column < outputs; ++column)
    {
        float expected = 0;
        for (int block = 0; block < blocks; ++block)
        {
            int inputOffset = row * inputs + block * 32, weightOffset = (column * blocks + block) * 34;
            float max = 0;
            for (int d = 0; d < 32; ++d) max = Math.Max(max, Math.Abs(x[inputOffset + d]));
            float scale = max / 127f;
            float stored = floatScale ? scale : (float)(Half)scale;
            float weightScale = (float)BitConverter.UInt16BitsToHalf((ushort)(weights[weightOffset] | weights[weightOffset + 1] << 8));
            float inverse = scale == 0 ? 0 : 1f / scale;
            int dot = 0;
            for (int d = 0; d < 32; ++d)
            {
                float quant = float.IsFinite(inverse) ? x[inputOffset + d] * inverse : x[inputOffset + d] / scale;
                dot += (int)MathF.Round(quant, MidpointRounding.ToEven) * unchecked((sbyte)weights[weightOffset + 2 + d]);
            }
            expected = MathF.FusedMultiplyAdd(dot, weightScale * stored, expected);
        }
        double error = Math.Abs(expected - output[row * outputs + column]);
        maximum = Math.Max(maximum, error);
        if (System.Runtime.Intrinsics.X86.Fma.IsSupported && BitConverter.SingleToInt32Bits(expected) != BitConverter.SingleToInt32Bits(output[row * outputs + column]))
            throw new InvalidDataException($"FMA scalar oracle differs at [{row},{column}]: expected {expected:R}, actual {output[row * outputs + column]:R}");
        if (!(error <= 2e-5 * Math.Max(1, Math.Abs(expected))))
            throw new InvalidDataException($"Mismatch [{row},{column}]: expected {expected:R}, actual {output[row * outputs + column]:R}");
    }
    // Exact batching invariant for the same activations, including odd tails.
    var single = new float[outputs];
    matrix.Multiply(x.AsSpan((rows - 1) * inputs, inputs).ToArray(), 1, single, pool);
    if (!single.SequenceEqual(output.AsSpan((rows - 1) * outputs, outputs).ToArray()))
        throw new InvalidDataException("Batch and single integer dot paths differ.");
    if (matrix512 != null)
    {
        matrix512.Multiply(x.AsSpan((rows - 1) * inputs, inputs).ToArray(), 1, single, pool);
        if (!single.SequenceEqual(output.AsSpan((rows - 1) * outputs, outputs).ToArray()))
            throw new InvalidDataException("AVX512 batch and single integer dot paths differ.");
    }
    var samples = new List<double>();
    var samples512 = new List<double>();
    for (int repeat = 0; repeat < 5; ++repeat)
    {
        var watch = Stopwatch.StartNew(); matrix.Multiply(x, rows, output, pool); samples.Add(watch.Elapsed.TotalMilliseconds);
        if (matrix512 != null)
        {
            watch.Restart(); matrix512.Multiply(x, rows, output512, pool); samples512.Add(watch.Elapsed.TotalMilliseconds);
        }
    }
    report.Add(new { inputs, outputs, rows, threads, minimum_weight = minimumWeight, maximum_error = maximum,
        preserve_float32_activation_scales = floatScale, amplitude,
        samples_ms = samples, avx512_ran = matrix512 != null, avx512_samples_ms = samples512,
        avx512_scalar_bit_exact = matrix512 != null, passed = true });
    Console.WriteLine($"{inputs}x{outputs}, rows{rows}, threads{threads}, minW{minimumWeight}: max error {maximum:R}, AVX2 {samples.Order().ElementAt(2):F4}ms"+
        (matrix512 == null ? " AVX512 unavailable" : $" AVX512 {samples512.Order().ElementAt(2):F4}ms"));
}
if (args.Length == 1)
{
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(args[0]))!);
    File.WriteAllText(args[0], JsonSerializer.Serialize(new { completed = true, processor_count = Environment.ProcessorCount,
        cases = report, limitations = "Standalone arithmetic and scheduler checks only; no model or end-to-end performance claim." },
        new JsonSerializerOptions { WriteIndented = true }) + "\n");
}
return 0;
