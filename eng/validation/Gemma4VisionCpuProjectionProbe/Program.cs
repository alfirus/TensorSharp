using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Cpu;
using TensorSharp.Models;
using TensorSharp.Models.Embeddings;

if (args.Length is < 2 or > 3 || args.Length == 3 && args[2] != "--policy-only")
    throw new ArgumentException("Pass owned library, output JSON and optional --policy-only.");
string library = Path.GetFullPath(args[0]), outputPath = Path.GetFullPath(args[1]);
Environment.SetEnvironmentVariable("TENSORSHARP_EMBEDDING_NATIVE_Q16", "0");
Environment.SetEnvironmentVariable("TENSORSHARP_EMBEDDING_Q16_AVX512", "0");
if (OwnedEmbeddingCpuProjection.TryCreate(out var textReason) != null || textReason != "not-requested") throw new InvalidDataException("Text factory opt-out changed.");
if (OwnedEmbeddingCpuProjection.TryCreate(false, true, library, out var disabled) != null || disabled != "not-requested") throw new InvalidDataException("Disabled owner loaded native math.");
if (OwnedEmbeddingCpuProjection.TryCreate(true, false, library, out var prerequisite) != null || prerequisite != "Q16 activation experiment not enabled") throw new InvalidDataException("Per-owner activation prerequisite bypassed.");
Environment.SetEnvironmentVariable("TS_CPU_DISABLE_AVX512", "1");
if (OwnedEmbeddingCpuProjection.TryCreate(true, true, library, out var guard) != null || string.IsNullOrEmpty(guard)) throw new InvalidDataException("Disable guard bypassed.");
Environment.SetEnvironmentVariable("TS_CPU_DISABLE_AVX512", null);
var cases = new List<object>();
bool inFlightCancellation = false;
void Write(bool completed, bool native, OwnedEmbeddingCpuProjection math, string reason)
{
    Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
    string Hash(string p) => Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(p))).ToLowerInvariant();
    File.WriteAllText(outputPath, JsonSerializer.Serialize(new { completed,
        native_execution_requested = native, native_math_executed = (math?.SuccessfulCalls ?? 0) > 0,
        policy_executed = 4, policy_passed = 4, policy_only = args.Length == 3, unavailable_reason = reason,
        cases, successful_calls = math?.SuccessfulCalls ?? 0, successful_rows = math?.SuccessfulRows ?? 0,
        in_flight_span_cancellation_drained_and_pool_reused = inFlightCancellation,
        successful_groups = math?.SuccessfulGroups ?? 0, f32_subnormal_correction_rows = math?.Float32FallbackRows ?? 0,
        library_sha256 = math?.LibrarySha256, probe_sha256 = Hash(typeof(Program).Assembly.Location),
        tensor_primitives_sha256 = Hash(typeof(System.Numerics.Tensors.TensorPrimitives).Assembly.Location),
        runtime = Environment.Version.ToString(), cpu_sgemm_kernel = CpuSgemm.ActiveKernelName,
        text_environment_unchanged = Environment.GetEnvironmentVariable("TENSORSHARP_EMBEDDING_NATIVE_Q16") == "0" && Environment.GetEnvironmentVariable("TENSORSHARP_EMBEDDING_Q16_AVX512") == "0",
        limitations = "Synthetic weighted-RMS/QAT and range fixtures, not actual learned norm/clamp weights. Whole spans compare bitwise with the existing native array bridge; independent scalar/double oracle samples up to five rows and fifteen output columns per shape, including endpoints. Timings use linked production F32 CpuSgemm with the same pool, immutable warm weights/activations, three warmups, five alternating AB/BA batches; each timing group reuses one projection's warm weights/arrays unlike the full 16-block tower, so cache locality is favorable; no Tensor allocation or full model/HTTP claim. Policy-only or unavailable hardware is never a native arithmetic pass." }, new JsonSerializerOptions { WriteIndented = true }) + "\n");
}
if (args.Length == 3) { Write(true, false, null, "policy-only"); return 0; }
using var math = OwnedEmbeddingCpuProjection.TryCreate(true, true, library, out var reason);
if (math == null) { Write(false, false, null, reason); return 2; }
var shapes = (from dims in new[] { (768,768), (768,3072), (3072,768), (768,512) }
              from rows in new[] { 36,108,2304 }
              select (dims.Item1,dims.Item2,rows,dims.Item1 == 3072 ? "gated-gelu-qat" : dims.Item2 == 512 ? "unweighted-rms" : "weighted-rms-qat",1f)).ToList();
shapes.AddRange(new[] { (768,769,17,"large-finite",80_000_000f), (3072,769,17,"tiny",1e-37f), (768,769,17,"subnormal",1e-42f) });
try
{
foreach (var (inputs, outputs, rows, mode, amplitude) in shapes)
{
    int blocks = inputs / 32, activeInput = checked(rows * inputs), activeOutput = checked(rows * outputs);
    var raw = new byte[checked(outputs * blocks * 34)];
    var transposedWeight = new float[checked(inputs * outputs)];
    for (int col = 0; col < outputs; ++col)
    for (int block = 0; block < blocks; ++block)
    {
        int p = (col * blocks + block) * 34;
        ushort bits = BitConverter.HalfToUInt16Bits((Half)(.001f * (1 + (col + block) % 11)));
        raw[p] = (byte)bits; raw[p+1] = (byte)(bits >> 8);
        float ws = (float)BitConverter.UInt16BitsToHalf(bits);
        for (int d = 0; d < 32; ++d)
        {
            sbyte quant = (sbyte)(-128 + (col * 41 + block * 17 + d * 19) % 256);
            raw[p+2+d] = unchecked((byte)quant); transposedWeight[(block*32+d)*outputs+col] = quant * ws;
        }
    }
    const int leading = 23, trailing = 19; const float canary = 98765f;
    var storage = Enumerable.Repeat(canary, activeInput+leading+trailing).ToArray();
    for (int row = 0; row < rows; ++row)
    {
        double square = 0;
        for (int d = 0; d < inputs; ++d)
        {
            float x = mode.Contains("rms") ? MathF.Sin((row*inputs+d)*.017f) + .3f*MathF.Cos(d*.037f)
                : mode == "gated-gelu-qat" ? MathF.Sin((row*inputs+d)*.017f)*4 : (d%32-16)*(amplitude/16);
            if (mode == "gated-gelu-qat")
                x = Math.Clamp(.5f*x*(1+MathF.Tanh(.7978845608028654f*(x+.044715f*x*x*x)))*MathF.Cos(d*.037f)*2,-16f,16f);
            storage[leading+row*inputs+d] = x; square += (double)x*x;
        }
        if (mode.Contains("rms"))
        {
            float scale = 1/MathF.Sqrt((float)(square/inputs)+1e-6f);
            for (int d = 0; d < inputs; ++d)
                storage[leading+row*inputs+d] = Math.Clamp(storage[leading+row*inputs+d]*scale*(mode == "unweighted-rms" ? 1 : .5f+(d%11)*.125f),-2.75f,2.75f);
        }
    }
    var original = (float[])storage.Clone(); var flatInput = storage.AsSpan(leading,activeInput).ToArray();
    var candidate = Enumerable.Repeat(canary,activeOutput+leading+trailing).ToArray();
    var arrayBridge = new float[activeOutput]; var f32 = new float[activeOutput];
    using var pool = new CpuWorkerPool(8);
    CpuParallel.SetRunner((count,body) => pool.For(count,body),() => pool.ThreadCount);
    var matrix = new ManagedEmbeddingQ8Q16Avx512Matrix(raw,inputs,outputs,math);
    void Native() => matrix.MultiplyNative(storage.AsSpan(leading,activeInput),rows,candidate.AsSpan(leading,activeOutput),pool);
    unsafe void Float32()
    {
        fixed (float* xp = storage, wp = transposedWeight, yp = f32)
            CpuSgemm.Gemm(rows,outputs,inputs,1,xp+leading,inputs,1,wp,outputs,1,0,yp,outputs);
    }
    matrix.Multiply(flatInput,rows,arrayBridge,pool); Native(); Float32();
    for (int i = 0; i < activeOutput; ++i)
        if (BitConverter.SingleToInt32Bits(arrayBridge[i]) != BitConverter.SingleToInt32Bits(candidate[leading+i])) throw new InvalidDataException("Span/array native output differs.");
    bool InputBitsUnchanged()
    {
        for (int i = 0; i < storage.Length; ++i)
            if (BitConverter.SingleToInt32Bits(storage[i]) != BitConverter.SingleToInt32Bits(original[i])) return false;
        return true;
    }
    if (!InputBitsUnchanged() || candidate.AsSpan(0,leading).ContainsAnyExcept(canary) || candidate.AsSpan(leading+activeOutput).ContainsAnyExcept(canary)) throw new InvalidDataException("Input/leading/trailing canary changed.");
    int oracleSamples = 0; double maxErrorFraction = 0;
    foreach (int row in new[] { 0,1,rows/2,rows-2,rows-1 }.Distinct())
    foreach (int col in Enumerable.Range(0,Math.Min(outputs,12)).Concat(new[] { outputs/2,outputs-2,outputs-1 }).Distinct())
    {
        float expected = 0; double full = 0, originalMagnitude = 0, quantizedMagnitude = 0, inputError = 0; bool fallback = false;
        var scales = new float[blocks]; var q = new short[inputs];
        for (int b = 0; b < blocks; ++b)
        {
            float maximum = 0; for (int d = 0; d < 32; ++d) maximum = Math.Max(maximum,Math.Abs(flatInput[row*inputs+b*32+d]));
            float scale = maximum/32767f;
            if (maximum > 0 && scale < 1.17549435e-38f) { fallback = true; continue; }
            scales[b] = scale; float inv = scale == 0 ? 0 : 1/scale;
            for (int d = 0; d < 32; ++d)
                q[b*32+d] = (short)Math.Clamp(MathF.Round(float.IsFinite(inv) ? flatInput[row*inputs+b*32+d]*inv : flatInput[row*inputs+b*32+d]/scale),short.MinValue,short.MaxValue);
        }
        for (int b = 0; b < blocks; ++b)
        {
            int offset = (col*blocks+b)*34, dot = 0;
            float ws = (float)BitConverter.UInt16BitsToHalf((ushort)(raw[offset]|raw[offset+1]<<8));
            for (int d = 0; d < 32; ++d)
            {
                float x = flatInput[row*inputs+b*32+d], weight = unchecked((sbyte)raw[offset+2+d])*ws;
                full += (double)x*weight; originalMagnitude += Math.Abs((double)x*weight);
                if (fallback) expected = MathF.FusedMultiplyAdd(x,weight,expected);
                else { dot += q[b*32+d]*unchecked((sbyte)raw[offset+2+d]); inputError += Math.Abs((double)x-q[b*32+d]*(double)scales[b])*Math.Abs(weight); }
            }
            if (!fallback) { expected = MathF.FusedMultiplyAdd(dot*ws,scales[b],expected); quantizedMagnitude += Math.Abs(dot*(double)ws*scales[b]); }
        }
        float actual = candidate[leading+row*outputs+col];
        if (BitConverter.SingleToInt32Bits(expected) != BitConverter.SingleToInt32Bits(actual)) throw new InvalidDataException("Independent scalar bit oracle failed.");
        double unit = Math.Pow(2,-24), n = fallback ? inputs+2 : blocks+3;
        double bound = inputError + n*unit/(1-n*unit)*(fallback ? originalMagnitude : quantizedMagnitude) +
            (fallback ? inputs : blocks*3)*Math.Pow(2,-150);
        double error = Math.Abs(full-actual);
        if (!double.IsFinite(error) || error > bound) throw new InvalidDataException("Independent double product/quantization bound failed.");
        double baselineSteps = inputs+2, baselineBound = baselineSteps*unit/(1-baselineSteps*unit)*originalMagnitude + baselineSteps*Math.Pow(2,-150);
        if (!float.IsFinite(f32[row*outputs+col]) || Math.Abs(full-f32[row*outputs+col]) > baselineBound)
            throw new InvalidDataException("Linked production F32 SGEMM independent double bound failed.");
        maxErrorFraction = Math.Max(maxErrorFraction,error/bound); ++oracleSamples;
    }
    long before = math.SuccessfulCalls; int rejected = 0;
    void Reject(Action body)
    {
        try { body(); } catch (ArgumentException) { ++rejected; return; }
        throw new InvalidDataException("Invalid span storage accepted.");
    }
    var overlap = new float[leading+Math.Max(activeInput,activeInput/2+activeOutput)+trailing];
    Reject(() => matrix.MultiplyNative(overlap.AsSpan(leading,activeInput),rows,overlap.AsSpan(leading+activeInput/2,activeOutput),pool));
    Reject(() => matrix.MultiplyNative(storage.AsSpan(leading,activeInput-1),rows,candidate.AsSpan(leading,activeOutput),pool));
    Reject(() => matrix.MultiplyNative(storage.AsSpan(leading,activeInput),rows,candidate.AsSpan(leading,activeOutput-1),pool));
    using (var cancellation = new CancellationTokenSource())
    {
        cancellation.Cancel(); bool canceled = false;
        try { matrix.MultiplyNative(storage.AsSpan(leading,activeInput),rows,candidate.AsSpan(leading,activeOutput),pool,cancellation.Token); } catch (OperationCanceledException) { canceled = true; }
        if (!canceled || before != math.SuccessfulCalls) throw new InvalidDataException("Canceled/invalid work counted as native success.");
    }
    foreach (float invalid in new[] { float.NaN,float.PositiveInfinity,float.NegativeInfinity })
    {
        float saved = storage[leading]; storage[leading] = invalid; bool finiteRejected = false;
        try { Native(); }
        catch (ArgumentException) { finiteRejected = true; }
        catch (AggregateException error) when (error.Flatten().InnerExceptions.All(inner => inner is ArgumentException)) { finiteRejected = true; }
        finally { storage[leading] = saved; }
        if (!finiteRejected || math.SuccessfulCalls != before) throw new InvalidDataException("Nonfinite span input counted as native success.");
    }
    for (int warm = 0; warm < 3; ++warm) { Float32(); Native(); }
    List<double> originalSamples = [], nativeSamples = [];
    double Sample(Action action)
    {
        double ms = 0; int count = 0;
        do { long start = Stopwatch.GetTimestamp(); action(); ms += Stopwatch.GetElapsedTime(start).TotalMilliseconds; ++count; } while (ms < 20);
        return ms/count;
    }
    for (int repeat = 0; repeat < 5; ++repeat)
        if (repeat%2 == 0) { originalSamples.Add(Sample(Float32)); nativeSamples.Add(Sample(Native)); }
        else { nativeSamples.Add(Sample(Native)); originalSamples.Add(Sample(Float32)); }
    if (!InputBitsUnchanged() || candidate.AsSpan(0,leading).ContainsAnyExcept(canary) || candidate.AsSpan(leading+activeOutput).ContainsAnyExcept(canary)) throw new InvalidDataException("Timing changed input/canaries.");
    cases.Add(new { inputs,outputs,rows,threads = 8,mode,amplitude,passed = true, whole_span_array_bits_equal = true,
        independent_scalar_bits_sampled = oracleSamples, independent_double_bound_max_fraction = maxErrorFraction,
        input_and_leading_trailing_canaries = true, invalid_span_guards_rejected = rejected, pre_cancellation_not_counted = true,
        nonfinite_span_inputs_rejected = 3,
        f32_correction_rows = matrix.Float32FallbackRows, baseline_samples_ms = originalSamples, native_samples_ms = nativeSamples,
        baseline_median_ms = originalSamples.Order().ElementAt(2),native_median_ms = nativeSamples.Order().ElementAt(2) });
    CpuParallel.SetRunner(null,null);
}
// Observe an actual native output write before requesting cancellation. The
// span/pinned scratch and owner lease must outlive all jobs even on this path.
using (var pool = new CpuWorkerPool(4))
using (var cancellation = new CancellationTokenSource())
using (var armed = new ManualResetEventSlim())
{
    const int rows = 4096, inputs = 768, outputs = 3072;
    var raw = Enumerable.Repeat((byte)1,outputs*(inputs/32)*34).ToArray();
    ushort scaleBits = BitConverter.HalfToUInt16Bits((Half).01f);
    for (int p = 0; p < raw.Length; p += 34) { raw[p] = (byte)scaleBits; raw[p+1] = (byte)(scaleBits>>8); }
    var matrix = new ManagedEmbeddingQ8Q16Avx512Matrix(raw,inputs,outputs,math);
    var input = Enumerable.Repeat(1f,rows*inputs).ToArray();
    var output = Enumerable.Repeat(98765f,rows*outputs).ToArray();
    long before = math.SuccessfulCalls;
    var observer = Task.Run(() =>
    {
        armed.Set();
        bool started = SpinWait.SpinUntil(() => Volatile.Read(ref output[0]) != 98765f,TimeSpan.FromSeconds(5));
        if (started) cancellation.Cancel();
        return started;
    });
    if (!armed.Wait(TimeSpan.FromSeconds(5))) throw new InvalidDataException("Span cancellation observer not armed.");
    bool rejected = false;
    try { matrix.MultiplyNative(input,rows,output,pool,cancellation.Token); }
    catch (OperationCanceledException) { rejected = true; }
    if (!await observer || !rejected || math.SuccessfulCalls != before) throw new InvalidDataException("Actual in-flight span cancellation not exercised or counted as success.");
    matrix.MultiplyNative(input.AsSpan(0,inputs),1,output.AsSpan(0,outputs),pool);
    inFlightCancellation = true;
}
Write(true,true,math,null);
}
catch (Exception error)
{
    Write(false,true,math,error.GetType().Name+": "+error.Message);
    throw;
}
finally { CpuParallel.SetRunner(null,null); }
Console.WriteLine($"Executed {cases.Count} vision-shape projection cases with actual native calls, sampled independent oracle, span/array bits and canaries.");
return 0;
