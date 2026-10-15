using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Models;
using TensorSharp.Models.Embeddings;

if (args.Length != 2) throw new ArgumentException("Pass native library path and JSON output path.");
if (!ManagedEmbeddingQ8Q16Avx512Matrix.IsSupported) { Console.Error.WriteLine("AVX512 unavailable; no VNNI checks executed."); return 2; }
using var native = new NativePrototype(Path.GetFullPath(args[0]));
if (!native.Supported) { Console.Error.WriteLine("Native VNNI unavailable; no VNNI checks executed."); return 2; }
var edgeCoverage = native.CheckEdges();
List<double> Measure(Action action)
{
    for (int warm = 0; warm < 3; ++warm) action();
    var samples = new List<double>();
    for (int repeat = 0; repeat < 15; ++repeat)
    {
        long start = Stopwatch.GetTimestamp(); int calls = 0;
        do { action(); ++calls; } while (Stopwatch.GetElapsedTime(start).TotalMilliseconds < 50);
        samples.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds / calls);
    }
    return samples;
}
var results = new List<object>();
foreach (var (inputs, outputs, rows, threads, amplitude) in new[] {
    (32, 1, 1, 1, 8f), (64, 7, 2, 1, 8f), (96, 17, 3, 3, 8f),
    (512, 31, 7, 4, 8f), (1024, 67, 17, 4, 8f), (512, 512, 32, 8, 8f),
    (512, 4096, 128, 8, 8f), (2048, 512, 128, 8, 8f), (1024, 512, 128, 8, 8f),
    (512, 2048, 23, 8, 8f), (512, 12288, 256, 8, 8f),
    (64, 9, 7, 3, 80_000_000f), (64, 9, 7, 3, 1e-37f), (64, 9, 7, 3, 1e-42f) })
{
    int blocks = inputs / 32, tiles = (outputs + 15) / 16;
    var source = new byte[checked(outputs * blocks * 34)];
    var packed = new sbyte[checked(tiles * blocks * 512)];
    var scales = new float[checked(tiles * blocks * 16)];
    var sums = new int[scales.Length];
    for (int col = 0; col < outputs; ++col)
    for (int block = 0; block < blocks; ++block)
    {
        int original = (col * blocks + block) * 34;
        ushort bits = BitConverter.HalfToUInt16Bits((Half)(.005f * (1 + (col + block) % 9)));
        source[original] = (byte)bits; source[original + 1] = (byte)(bits >> 8);
        int tileBlock = col / 16 * blocks + block, lane = col % 16;
        scales[tileBlock * 16 + lane] = (float)BitConverter.UInt16BitsToHalf(bits);
        for (int dim = 0; dim < 32; ++dim)
        {
            sbyte weight = (sbyte)(-128 + (col * 41 + block * 17 + dim * 19) % 256);
            source[original + dim + 2] = unchecked((byte)weight);
            packed[tileBlock * 512 + dim / 4 * 64 + lane * 4 + dim % 4] = weight;
            sums[tileBlock * 16 + lane] += weight;
        }
    }
    var x = new float[checked(rows * inputs)];
    for (int row = 0; row < rows; ++row)
    for (int dim = 0; dim < inputs; ++dim)
        x[row * inputs + dim] = row == 0 ? 0 : row % 3 == 1 ? (dim % 32 - 16) * (amplitude / 16)
            : MathF.Sin((row * inputs + dim) * .017f) * (amplitude / 8) * (1 + row % 5);
    var managed = new ManagedEmbeddingQ8Q16Avx512Matrix(source, inputs, outputs);
    using var pool = new CpuWorkerPool(threads);
    var reference = new float[checked(rows * outputs)];
    managed.Multiply(x, rows, reference, pool);
    var activations = new short[x.Length]; var activationScales = new float[rows * blocks]; var fallback = new bool[rows];
    var packedActivations = new byte[checked(x.Length * 2)];
    var output = new float[reference.Length];
    void MultiplyNative(int layout) => native.Multiply(x, inputs, outputs, rows, packed, sums, scales,
        activations, packedActivations, activationScales, fallback, output, pool, layout);
    MultiplyNative(0);
    // Evaluate the independent scalar oracle once per shape, before timing.
    for (int row = 0; row < rows; ++row)
    for (int col = 0; col < outputs; ++col)
    {
        float expected = 0;
        for (int block = 0; block < blocks; ++block)
        {
            int dot = 0, offset = (col * blocks + block) * 34;
            float ws = (float)BitConverter.UInt16BitsToHalf((ushort)(source[offset] | source[offset + 1] << 8));
            for (int dim = 0; dim < 32; ++dim)
            {
                int weight = unchecked((sbyte)source[offset + dim + 2]);
                if (fallback[row]) expected = MathF.FusedMultiplyAdd(x[row * inputs + block * 32 + dim], weight * ws, expected);
                else dot += activations[row * inputs + block * 32 + dim] * weight;
            }
            if (!fallback[row]) expected = MathF.FusedMultiplyAdd(dot * ws, activationScales[row * blocks + block], expected);
        }
        if (BitConverter.SingleToInt32Bits(expected) != BitConverter.SingleToInt32Bits(reference[row * outputs + col]))
            throw new InvalidDataException("Independent scalar integer/F32 fallback oracle mismatch.");
    }
    var variants = new List<object>();
    foreach (int layout in new[] { 0, 1, 2 })
    {
        MultiplyNative(layout);
        if (!reference.SequenceEqual(output))
        {
            int bad = Enumerable.Range(0, output.Length).First(i => BitConverter.SingleToInt32Bits(reference[i]) != BitConverter.SingleToInt32Bits(output[i]));
            throw new InvalidDataException($"VNNI layout{layout} mismatch [{bad}]: {output[bad]:R} versus managed {reference[bad]:R}.");
        }
        var samples = Measure(() => MultiplyNative(layout));
        variants.Add(new { layout, tile_rows = 8, samples_ms = samples, median_ms = samples.Order().ElementAt(7), scalar_bit_exact = true, managed_bit_exact = true });
    }
    var baseline = Measure(() => managed.Multiply(x, rows, reference, pool));
    results.Add(new { inputs, outputs, rows, threads, amplitude, managed_samples_ms = baseline,
        managed_median_ms = baseline.Order().ElementAt(7), native_variants = variants });
    Console.WriteLine($"{inputs}x{outputs} rows{rows}: scalar and managed bit exact; managed {baseline.Order().ElementAt(7):F4}ms");
}
string Hash(string path) => Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(args[1]))!);
File.WriteAllText(args[1], JsonSerializer.Serialize(new { completed = true, native_library = args[0],
    native_library_sha256 = Hash(args[0]), probe_sha256 = Hash(typeof(NativePrototype).Assembly.Location), edge_coverage = edgeCoverage, cases = results,
    timing_method = "3 warmup calls then 15 batches of at least 50ms each; samples are batch-mean milliseconds per full matrix call.",
    limitations = "Stateless native microprototype; timings include the same managed AVX2 quantization and CpuWorkerPool scheduling, plus P/Invoke and native tiled math. No production engine integration or full-model performance claim." }, new JsonSerializerOptions { WriteIndented = true }) + "\n");
return 0;

unsafe sealed class NativePrototype : IDisposable
{
    private readonly nint _library;
    private readonly delegate* unmanaged[Cdecl]<sbyte*, int*, float*, short*, float*, float*, int, int, int, int, int, int> _tile;
    private readonly delegate* unmanaged[Cdecl]<short*, byte*, int, int> _pack;
    private readonly delegate* unmanaged[Cdecl]<sbyte*, int*, float*, byte*, float*, float*, int, int, int, int, int, int> _packedTile, _packedGroup;
    public readonly bool Supported;
    public NativePrototype(string path)
    {
        _library = NativeLibrary.Load(path);
        try
        {
            var abi = (delegate* unmanaged[Cdecl]<int>)NativeLibrary.GetExport(_library, "tensorcpu_abi_version");
            if (abi() != 1) throw new InvalidDataException("Unsupported TensorSharp CPU library ABI.");
            var supported = (delegate* unmanaged[Cdecl]<int>)NativeLibrary.GetExport(_library, "tensorcpu_q16_vnni_supported");
            _tile = (delegate* unmanaged[Cdecl]<sbyte*, int*, float*, short*, float*, float*, int, int, int, int, int, int>)NativeLibrary.GetExport(_library, "tensorcpu_q16_vnni_tile");
            _pack = (delegate* unmanaged[Cdecl]<short*, byte*, int, int>)NativeLibrary.GetExport(_library, "tensorcpu_q16_vnni_pack");
            _packedTile = (delegate* unmanaged[Cdecl]<sbyte*, int*, float*, byte*, float*, float*, int, int, int, int, int, int>)NativeLibrary.GetExport(_library, "tensorcpu_q16_vnni_packed_tile");
            _packedGroup = (delegate* unmanaged[Cdecl]<sbyte*, int*, float*, byte*, float*, float*, int, int, int, int, int, int>)NativeLibrary.GetExport(_library, "tensorcpu_q16_vnni_packed_group");
            Supported = supported() == 1;
        }
        catch { NativeLibrary.Free(_library); throw; }
    }
    public object CheckEdges()
    {
        int[] values = [-32768, -32767, -256, -1, 0, 1, 255, 256, 32767];
        var a = new short[12 * 32]; var packedA = new byte[a.Length * 2]; var w = new sbyte[512]; var sums = new int[16];
        var ws = Enumerable.Repeat(.125f, 16).ToArray(); var scale = Enumerable.Repeat(.25f, 12).ToArray();
        var y = Enumerable.Repeat(1234567f, 12 * 19).ToArray();
        for (int row = 0; row < 12; ++row)
        for (int dim = 0; dim < 32; ++dim) a[row * 32 + dim] = (short)values[(row + dim) % values.Length];
        for (int col = 0; col < 16; ++col)
        for (int dim = 0; dim < 32; ++dim)
        {
            sbyte weight = (sbyte)(-128 + (col * 41 + dim * 19) % 256);
            w[dim / 4 * 64 + col * 4 + dim % 4] = weight; sums[col] += weight;
        }
        fixed (short* ap = a)
        fixed (byte* pap = packedA)
        fixed (sbyte* wp = w)
        fixed (int* sp = sums)
        fixed (float* wsp = ws, asp = scale, yp = y)
        {
            if (_tile(null, sp, wsp, ap, asp, yp, 1, 32, 19, 1, 16) != -1 ||
                _tile(wp, sp, wsp, ap, asp, yp, 0, 32, 19, 1, 16) != -1 ||
                _tile(wp, sp, wsp, ap, asp, yp, 1, 31, 19, 1, 16) != -1 ||
                _tile(wp, sp, wsp, ap, asp, yp, 1, 32, 19, 13, 16) != -1 ||
                _tile(wp, sp, wsp, ap, asp, yp, 1, 32, 19, 1, 17) != -1)
                throw new InvalidDataException("Native argument bounds were not enforced.");
            if (_pack(ap, pap, a.Length) != 0 || _pack(ap, pap, 15) != -1 ||
                _packedTile(wp, sp, wsp, pap, asp, yp, 1, 64, 19, 13, 16) != -1 ||
                _packedGroup(wp, sp, wsp, pap, asp, yp, 1, 64, 19, 25, 16) != -1 ||
                _packedGroup(wp, sp, wsp, pap, asp, yp, 1, 64, 80, 1, 65) != -1)
                throw new InvalidDataException("Packed API bounds were not enforced.");
            for (int i = 0; i < a.Length; ++i)
                if (packedA[i / 4 * 8 + i % 4] != (a[i] & 255) || packedA[i / 4 * 8 + i % 4 + 4] != (a[i] >> 8) + 128)
                    throw new InvalidDataException("Independent low/high byte representation mismatch.");
            for (int layout = 0; layout < 3; ++layout)
            for (int rows = 1; rows <= 12; ++rows)
            {
                Array.Fill(y, 1234567f);
                int status = layout == 0 ? _tile(wp, sp, wsp, ap, asp, yp, 1, 32, 19, rows, 16)
                    : layout == 1 ? _packedTile(wp, sp, wsp, pap, asp, yp, 1, 64, 19, rows, 16)
                    : _packedGroup(wp, sp, wsp, pap, asp, yp, 1, 64, 19, rows, 16);
                if (status != 0) throw new InvalidDataException("Edge tile rejected.");
                for (int row = 0; row < 12; ++row)
                for (int col = 0; col < 19; ++col)
                {
                    float expected = 1234567f;
                    if (row < rows && col < 16)
                    {
                        int dot = 0;
                        for (int dim = 0; dim < 32; ++dim) dot += a[row * 32 + dim] * w[dim / 4 * 64 + col * 4 + dim % 4];
                        expected = MathF.FusedMultiplyAdd(dot * .125f, .25f, 0);
                    }
                    if (BitConverter.SingleToInt32Bits(expected) != BitConverter.SingleToInt32Bits(y[row * 19 + col]))
                        throw new InvalidDataException("Signed-int16 edge or output canary mismatch.");
                }
            }
        }
        return new { signed_int16_min_tested = true, all_row_variants_1_to_12 = true, output_stride_canaries = true,
            independent_low_high_encoding = true, invalid_arguments_rejected = 9 };
    }
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Multiply(float[] input, int inputs, int outputs, int rows, sbyte[] weights, int[] weightSums, float[] weightScales,
        short[] activations, byte[] packedActivations, float[] activationScales, bool[] fallback, float[] output, CpuWorkerPool pool, int layout)
    {
        int blocks = inputs / 32, tiles = (outputs + 15) / 16, tileRows = 8;
        fixed (float* xp = input, asp = activationScales, wsp = weightScales, yp = output)
        fixed (short* ap = activations)
        fixed (byte* pap = packedActivations)
        fixed (sbyte* wp = weights)
        fixed (int* sum = weightSums)
        {
            nint xa = (nint)xp, asa = (nint)asp, wsa = (nint)wsp, ya = (nint)yp, aa = (nint)ap, pa = (nint)pap, wa = (nint)wp, sa = (nint)sum;
            pool.For(rows, row =>
            {
                fallback[row] = ManagedEmbeddingQ8Q16Avx512Matrix.Quantize((float*)xa + (long)row * inputs,
                    (short*)aa + (long)row * inputs, (float*)asa + (long)row * blocks, blocks);
                if (layout != 0 && _pack((short*)aa + (long)row * inputs, (byte*)pa + (long)row * inputs * 2, inputs) != 0)
                    throw new InvalidDataException("Native packing failed.");
            });
            int columnGroups = (tiles + 3) / 4, rowStep = tileRows * 2, rowGroups = (rows + rowStep - 1) / rowStep;
            pool.For(columnGroups * rowGroups, work =>
            {
                int firstTile = work % columnGroups * 4, endTile = Math.Min(firstTile + 4, tiles);
                int firstRow = work / columnGroups * rowStep, endRow = Math.Min(firstRow + rowStep, rows);
                if (layout == 2)
                {
                    int status = _packedGroup((sbyte*)wa + (long)firstTile * blocks * 512, (int*)sa + (long)firstTile * blocks * 16,
                        (float*)wsa + (long)firstTile * blocks * 16, (byte*)pa + (long)firstRow * inputs * 2,
                        (float*)asa + (long)firstRow * blocks, (float*)ya + (long)firstRow * outputs + firstTile * 16,
                        blocks, inputs * 2, outputs, endRow - firstRow, Math.Min((endTile - firstTile) * 16, outputs - firstTile * 16));
                    if (status != 0) throw new InvalidDataException("Native group status " + status);
                    return;
                }
                for (int tile = firstTile; tile < endTile; ++tile)
                for (int row = firstRow; row < endRow; row += tileRows)
                {
                    int status = layout == 0 ? _tile((sbyte*)wa + (long)tile * blocks * 512, (int*)sa + (long)tile * blocks * 16,
                        (float*)wsa + (long)tile * blocks * 16, (short*)aa + (long)row * inputs,
                        (float*)asa + (long)row * blocks, (float*)ya + (long)row * outputs + tile * 16,
                        blocks, inputs, outputs, Math.Min(tileRows, endRow - row), Math.Min(16, outputs - tile * 16))
                        : _packedTile((sbyte*)wa + (long)tile * blocks * 512, (int*)sa + (long)tile * blocks * 16,
                        (float*)wsa + (long)tile * blocks * 16, (byte*)pa + (long)row * inputs * 2,
                        (float*)asa + (long)row * blocks, (float*)ya + (long)row * outputs + tile * 16,
                        blocks, inputs * 2, outputs, Math.Min(tileRows, endRow - row), Math.Min(16, outputs - tile * 16));
                    if (status != 0) throw new InvalidDataException("Native bounded tile status " + status);
                }
            });
            if (fallback.AsSpan().Contains(true)) pool.For(rows, row =>
            {
                if (!fallback[row]) return;
                for (int col = 0; col < outputs; ++col)
                {
                    float value = 0; int tile = col / 16, lane = col % 16;
                    for (int block = 0; block < blocks; ++block)
                    for (int dim = 0; dim < 32; ++dim)
                    {
                        int weight = weights[(tile * blocks + block) * 512 + dim / 4 * 64 + lane * 4 + dim % 4];
                        float scale = weightScales[(tile * blocks + block) * 16 + lane];
                        value = MathF.FusedMultiplyAdd(input[row * inputs + block * 32 + dim], weight * scale, value);
                    }
                    output[row * outputs + col] = value;
                }
            });
        }
    }
    public void Dispose() => NativeLibrary.Free(_library);
}
