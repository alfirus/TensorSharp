using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Models;
using TensorSharp.Models.Embeddings;

if (args.Length != 2) throw new ArgumentException("Pass native library path and JSON output path.");
if (!ManagedEmbeddingAttentionAvx512.IsSupported) { Console.Error.WriteLine("AVX512 unavailable; no attention math executed."); return 2; }
using var native = new NativeAttention(Path.GetFullPath(args[0]));
if (!native.Supported) { Console.Error.WriteLine("Native AVX512 unavailable; no attention math executed."); return 2; }
var reports = new List<object>();
foreach (var (dimensions, length, queries, threads, amplitude) in new[] {
    (16, 1, 1, 1, 1f), (32, 15, 2, 1, 1f), (64, 16, 3, 1, 1f),
    (64, 17, 4, 1, 1f), (128, 31, 5, 1, 1f), (128, 32, 6, 1, 1f),
    (128, 33, 7, 1, 1f), (256, 511, 8, 8, 1f), (256, 512, 8, 8, 1f),
    (512, 1025, 8, 8, 1f), (512, 8192, 8, 8, 1f),
    (32, 17, 8, 1, 80_000_000f), (32, 17, 8, 1, 1e-37f) })
{
    // Match the model's padded-key formula and four heads per token on the
    // full eight-query shapes. Small tails additionally exercise unaligned
    // query/output strides, with unwritten stride lanes checked as canaries.
    int vector = System.Numerics.Vector<float>.Count;
    int stride = (length + 2 * vector - 2) / vector * vector;
    int queryStride = dimensions * 4 + (queries < 8 ? 3 : 0), outStride = dimensions * 4 + (queries < 8 ? 5 : 0);
    float scoreScale = queries < 8 ? .3f : 1f;
    int jobs = threads == 1 ? 1 : threads * 4;
    const int prefix = 17;
    var random = new Random(117 + dimensions + length + queries);
    float[] q = Enumerable.Range(0, 8 * queryStride).Select(_ => ((float)random.NextDouble() * 2 - 1) * amplitude).ToArray();
    float[] k = Enumerable.Range(0, dimensions * stride).Select(_ => (float)random.NextDouble() * 2 - 1).ToArray();
    float[] v = Enumerable.Range(0, dimensions * stride).Select(_ => (float)random.NextDouble() * 2 - 1).ToArray();
    float[] p = Enumerable.Range(0, 8 * length).Select(_ => ((float)random.NextDouble() * 2 - 1) / length).ToArray();
    float[] scores = Enumerable.Repeat(9876.5f, prefix + jobs * 8 * length + 19).ToArray();
    float[] values = Enumerable.Repeat(9876.5f, prefix + jobs * 8 * outStride + 19).ToArray();
    using var pool = new CpuWorkerPool(threads);
    unsafe void Run(bool owned)
    {
        fixed (float* qp = q, kp = k, vp = v, pp = p, sp = scores, yp = values)
        {
            nint qa = (nint)qp, ka = (nint)kp, va = (nint)vp, pa = (nint)pp, sa = (nint)(sp + prefix), ya = (nint)(yp + prefix);
            pool.For(jobs, job =>
            {
                float* outputScores = (float*)sa + job * 8 * length;
                float* outputValues = (float*)ya + job * 8 * outStride;
                if (owned)
                {
                    native.Scores((float*)qa, (float*)ka, outputScores, queryStride, stride, length, dimensions, length, queries, scoreScale);
                    native.Values((float*)pa, (float*)va, outputValues, length, stride, outStride, dimensions, length, queries);
                }
                else
                for (int first = 0; first < queries; first += 4)
                {
                    int active = Math.Min(4, queries - first);
                    float* a = (float*)qa + first * queryStride;
                    ManagedEmbeddingAttentionAvx512.Scores(a, a + Math.Min(1, active - 1) * queryStride,
                        a + Math.Min(2, active - 1) * queryStride, a + Math.Min(3, active - 1) * queryStride,
                        (float*)ka, stride, dimensions, length, scoreScale, outputScores + first * length);
                    ManagedEmbeddingAttentionAvx512.Values((float*)pa + first * length, length, (float*)va,
                        stride, dimensions, outputValues + first * outStride, outStride, active);
                }
            });
        }
    }
    // The scalar oracle reads original arrays, rather than either implementation's output.
    float[] expectedScores = new float[queries * length], expectedValues = new float[queries * dimensions];
    for (int row = 0; row < queries; ++row)
    {
        for (int key = 0; key < length; ++key)
        {
            float sum = 0; double exact = 0, absolute = 0;
            for (int d = 0; d < dimensions; ++d)
            {
                float a = q[row * queryStride + d], b = k[d * stride + key];
                sum = MathF.FusedMultiplyAdd(a, b, sum); exact += (double)a * b; absolute += Math.Abs((double)a * b);
            }
            expectedScores[row * length + key] = sum * scoreScale;
            if (Math.Abs((double)sum - exact) > dimensions * 1.1920928955078125e-7 * absolute + 1e-42)
                throw new InvalidDataException("Scalar score double-product error bound failed.");
        }
        for (int d = 0; d < dimensions; ++d)
        {
            float sum = 0; double exact = 0, absolute = 0;
            for (int key = 0; key < length; ++key)
            {
                float a = v[d / 16 * 16 * stride + key * 16 + d % 16], b = p[row * length + key];
                sum = MathF.FusedMultiplyAdd(a, b, sum); exact += (double)a * b; absolute += Math.Abs((double)a * b);
            }
            expectedValues[row * dimensions + d] = sum;
            if (Math.Abs((double)sum - exact) > length * 1.1920928955078125e-7 * absolute + 1e-42)
                throw new InvalidDataException("Scalar value double-product error bound failed.");
        }
    }
    void Check(bool owned)
    {
        Array.Fill(scores, 9876.5f); Array.Fill(values, 9876.5f); Run(owned);
        for (int job = 0; job < jobs; ++job)
        for (int row = 0; row < queries; ++row)
        {
            for (int key = 0; key < length; ++key)
                if (BitConverter.SingleToInt32Bits(scores[prefix + (job * 8 + row) * length + key]) != BitConverter.SingleToInt32Bits(expectedScores[row * length + key]))
                    throw new InvalidDataException($"Score bit oracle failed: owned={owned}, dim={dimensions}, length={length}, row={row}, key={key}.");
            for (int d = 0; d < dimensions; ++d)
                if (BitConverter.SingleToInt32Bits(values[prefix + (job * 8 + row) * outStride + d]) != BitConverter.SingleToInt32Bits(expectedValues[row * dimensions + d]))
                    throw new InvalidDataException($"Value bit oracle failed: owned={owned}, dim={dimensions}, length={length}, row={row}, d={d}.");
        }
        if (scores[^19..].Any(x => x != 9876.5f) || values[^19..].Any(x => x != 9876.5f)) throw new InvalidDataException("Suffix canary changed.");
        if (scores.AsSpan(0, prefix).ContainsAnyExcept(9876.5f) || values.AsSpan(0, prefix).ContainsAnyExcept(9876.5f)) throw new InvalidDataException("Leading canary changed.");
        for (int job = 0; job < jobs; ++job)
        for (int row = 0; row < 8; ++row)
            if (values.AsSpan(prefix + (job * 8 + row) * outStride + (row < queries ? dimensions : 0), row < queries ? outStride - dimensions : outStride).ContainsAnyExcept(9876.5f))
                throw new InvalidDataException("Value stride/inactive-row canary changed.");
        if (owned)
            for (int job = 0; job < jobs; ++job)
                if (scores.AsSpan(prefix + (job * 8 + queries) * length, (8 - queries) * length).ContainsAnyExcept(9876.5f))
                    throw new InvalidDataException("Native score inactive-row canary changed.");
    }
    Check(false); Check(true);
    double MeasureSample(bool owned)
    {
        long begin = Stopwatch.GetTimestamp(); int calls = 0;
        do { Run(owned); ++calls; } while (Stopwatch.GetElapsedTime(begin).TotalMilliseconds < 40);
        return Stopwatch.GetElapsedTime(begin).TotalMilliseconds / calls;
    }
    for (int warm = 0; warm < 3; ++warm) { Run(false); Run(true); }
    var managed = new List<double>(); var owned = new List<double>();
    for (int repeat = 0; repeat < 9; ++repeat)
        if (repeat % 2 == 0) { managed.Add(MeasureSample(false)); owned.Add(MeasureSample(true)); }
        else { owned.Add(MeasureSample(true)); managed.Add(MeasureSample(false)); }
    reports.Add(new { dimensions, length, queries, threads, jobs, query_stride = queryStride, key_value_stride = stride, output_stride = outStride, score_scale = scoreScale, amplitude, scalar_bit_exact = true,
        double_product_error_bound = true, stride_and_tail_canaries = true, managed_samples_ms = managed,
        native_samples_ms = owned, managed_median_ms = managed.Order().ElementAt(4), native_median_ms = owned.Order().ElementAt(4) });
    Console.WriteLine($"dim={dimensions} keys={length} queries={queries} threads={threads}: {managed.Order().ElementAt(4):F4}/{owned.Order().ElementAt(4):F4} ms");
}
Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(args[1]))!);
File.WriteAllText(args[1], JsonSerializer.Serialize(new
{
    completed = true, native_library = native.Path, native_library_sha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(native.Path))).ToLowerInvariant(),
    probe_sha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(typeof(NativeAttention).Assembly.Location))).ToLowerInvariant(),
    cases = reports, rejected_argument_checks = native.CheckArguments(),
    timing_method = "Three initial warmups per implementation, nine timed pairs alternating AB/BA order, each batch at least 40 ms; sample is mean per full worker-pool call including native transitions. Current four-query calls versus owned eight-query calls perform the same active output work; managed score tails additionally compute duplicate inactive rows.",
    limitations = "Isolated score/value math microgate. Jobs reuse the same input arrays, giving unusually favorable shared cache locality relative to distinct per-head/query blocks. No softmax/masking, RoPE, packing, model quality or HTTP performance is validated by this report. Only actual AVX512 CPU execution counts; unavailable architectures are not passes. Raw-pointer ABI callers must prove buffer extents; argument shape checks cannot establish allocation bounds.",
}, new JsonSerializerOptions { WriteIndented = true }) + "\n");
return 0;

sealed unsafe class NativeAttention : IDisposable
{
    public string Path { get; }
    public bool Supported { get; }
    private readonly nint _library;
    private readonly delegate* unmanaged[Cdecl]<float*,float*,float*,int,int,int,int,int,int,float,int> _scores;
    private readonly delegate* unmanaged[Cdecl]<float*,float*,float*,int,int,int,int,int,int,int> _values;
    public NativeAttention(string path)
    {
        Path = path; _library = NativeLibrary.Load(path);
        try
        {
            if (((delegate* unmanaged[Cdecl]<int>)NativeLibrary.GetExport(_library,"tensorcpu_abi_version"))() != 1) throw new InvalidDataException("CPU ABI version mismatch.");
            Supported = ((delegate* unmanaged[Cdecl]<int>)NativeLibrary.GetExport(_library,"tensorcpu_attention_avx512_supported"))() != 0;
            _scores = (delegate* unmanaged[Cdecl]<float*,float*,float*,int,int,int,int,int,int,float,int>)NativeLibrary.GetExport(_library,"tensorcpu_attention_scores8");
            _values = (delegate* unmanaged[Cdecl]<float*,float*,float*,int,int,int,int,int,int,int>)NativeLibrary.GetExport(_library,"tensorcpu_attention_values8");
        }
        catch { NativeLibrary.Free(_library); throw; }
    }
    public void Scores(float* q,float* k,float* y,int qs,int ks,int ys,int d,int len,int rows,float scale)
    { if (_scores(q,k,y,qs,ks,ys,d,len,rows,scale) != 0) throw new InvalidDataException("Native score call failed."); }
    public void Values(float* p,float* v,float* y,int ps,int vs,int ys,int d,int len,int rows)
    { if (_values(p,v,y,ps,vs,ys,d,len,rows) != 0) throw new InvalidDataException("Native value call failed."); }
    public int CheckArguments()
    {
        float* data = stackalloc float[32]; int checkedCount = 0;
        foreach (int rows in new[] {0,9})
        {
            if (_scores(data,data,data,16,1,1,16,1,rows,1) != -1 || _values(data,data,data,1,1,16,16,1,rows) != -1) throw new InvalidDataException("Invalid row bound accepted.");
            checkedCount += 2;
        }
        if (_scores(null,data,data,16,1,1,16,1,1,1) != -1 || _scores(data,data,data,15,1,1,16,1,1,1) != -1 ||
            _scores(data,data,data,16,1,1,16,1,1,float.NaN) != -1 || _values(data,data,data,1,1,16,15,1,1) != -1)
            throw new InvalidDataException("Invalid attention argument accepted.");
        checkedCount += 4;
        foreach (int len in new[] { 0, 16385 })
        {
            int stride = Math.Max(1, len);
            if (_scores(data,data,data,16,stride,stride,16,len,1,1) != -1 || _values(data,data,data,stride,stride,16,16,len,1) != -1)
                throw new InvalidDataException("Invalid key-length bound accepted.");
            checkedCount += 2;
        }
        foreach (int dimension in new[] { 0, 1025 })
        {
            int stride = Math.Max(16, dimension);
            if (_scores(data,data,data,stride,1,1,dimension,1,1,1) != -1 || _values(data,data,data,1,1,stride,dimension,1,1) != -1)
                throw new InvalidDataException("Invalid dimension bound accepted.");
            checkedCount += 2;
        }
        if (_scores(data,data,data,16,0,1,16,1,1,1) != -1 || _scores(data,data,data,16,1,0,16,1,1,1) != -1 ||
            _values(data,data,data,0,1,16,16,1,1) != -1 || _values(data,data,data,1,0,16,16,1,1) != -1 ||
            _values(data,data,data,1,1,15,16,1,1) != -1 || _scores(data,data,data,16,1,1,16,1,1,float.PositiveInfinity) != -1)
            throw new InvalidDataException("Invalid stride or scale accepted.");
        return checkedCount + 6;
    }
    public void Dispose() => NativeLibrary.Free(_library);
}
