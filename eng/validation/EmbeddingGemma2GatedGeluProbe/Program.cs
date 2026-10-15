using System.Buffers;
using System.Diagnostics;
using System.Numerics.Tensors;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Models;
using TensorSharp.Models.Embeddings;

if (args.Length is < 1 or > 2 || args.Length == 2 && args[1] != "--numerical-only")
    throw new ArgumentException("Pass output JSON and optional --numerical-only.");
bool timing = args.Length == 1;
var cases = new List<object>();
var shapes = new[] { (1,1,1), (3,7,1), (8,15,3), (9,17,4), (15,31,4), (17,257,4),
    (32,511,8), (32,512,8), (113,2048,8), (311,2048,8), (912,2048,8), (1025,2048,8) };
foreach (var (rows, width, threads) in shapes)
{
    using var pool = new CpuWorkerPool(threads);
    var source = new float[checked(rows * width * 2 + 19)];
    Array.Fill(source, 98765f);
    float[] edge = [-80_000_000f, -20f, -8f, -3f, -1f, -1e-37f, -1e-42f, -0f, 0f, 1e-42f, 1e-37f, .5f, 3f, 8f, 20f, 80_000_000f];
    for (int row = 0; row < rows; ++row)
    for (int dim = 0; dim < width; ++dim)
    {
        source[row * width * 2 + dim] = dim % 17 == 0 ? edge[(row + dim) % edge.Length] : MathF.Sin((row * width + dim) * .017f) * 3;
        source[row * width * 2 + width + dim] = MathF.Cos((row * width + dim) * .037f) * 2;
    }
    var joined = (float[])source.Clone();
    var legacy = Enumerable.Repeat(98765f, rows * width + 19).ToArray();
    var up = new float[rows * width];
    var fused = (float[])legacy.Clone();
    Legacy(source, legacy, up, rows, width, pool);
    ManagedEmbeddingGatedGelu.Apply(joined, fused, rows, width, pool);
    double maxError = 0, maxBoundFraction = 0;
    for (int row = 0; row < rows; ++row)
    for (int dim = 0; dim < width; ++dim)
    {
        int i = row * width + dim, gate = row * width * 2 + dim, u = gate + width;
        if (BitConverter.SingleToInt32Bits(fused[i]) != BitConverter.SingleToInt32Bits(legacy[i]))
            throw new InvalidDataException($"Legacy bit identity failed at [{row},{dim}].");
        if (BitConverter.SingleToInt32Bits(joined[u]) != BitConverter.SingleToInt32Bits(source[u]))
            throw new InvalidDataException("Read-only up half changed.");
        double x = source[gate], expected = .5 * x * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (x + .044715 * x * x * x))) * source[u];
        double error = Math.Abs(expected - fused[i]), bound = 1e-5 * (1 + Math.Abs(expected));
        if (!double.IsFinite(error) || error > bound) throw new InvalidDataException("Independent double GELU bound failed.");
        maxError = Math.Max(maxError, error); maxBoundFraction = Math.Max(maxBoundFraction, error / bound);
    }
    if (fused.AsSpan(rows * width).ContainsAnyExcept(98765f) || joined.AsSpan(rows * width * 2).ContainsAnyExcept(98765f))
        throw new InvalidDataException("Stale-capacity suffix canary changed.");
    int guardChecks = 0;
    void Reject(Action action)
    {
        try { action(); }
        catch (ArgumentException) { ++guardChecks; return; }
        catch (OverflowException) { ++guardChecks; return; }
        throw new InvalidDataException("Invalid shape/alias was accepted.");
    }
    Reject(() => ManagedEmbeddingGatedGelu.Apply(joined, joined, rows, width, pool));
    Reject(() => ManagedEmbeddingGatedGelu.Apply(joined, fused, 0, width, pool));
    Reject(() => ManagedEmbeddingGatedGelu.Apply(joined, fused, -1, width, pool));
    Reject(() => ManagedEmbeddingGatedGelu.Apply(joined, fused, rows, 0, pool));
    Reject(() => ManagedEmbeddingGatedGelu.Apply(joined, fused, int.MaxValue, int.MaxValue, pool));
    Reject(() => ManagedEmbeddingGatedGelu.Apply(joined, new float[rows * width - 1], rows, width, pool));
    Reject(() => ManagedEmbeddingGatedGelu.Apply(new float[rows * width * 2 - 1], fused, rows, width, pool));
    using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
    bool cancellation = false;
    try { ManagedEmbeddingGatedGelu.Apply(joined, fused, rows, width, pool, cancelled.Token); }
    catch (OperationCanceledException) { cancellation = true; }
    if (!cancellation) throw new InvalidDataException("Cancelled work was accepted.");
    Array.Copy(source, joined, source.Length);
    ManagedEmbeddingGatedGelu.Apply(joined, fused, rows, width, pool); // Pool still usable after rejected calls.

    List<double> legacySamples = [], fusedSamples = [];
    if (timing)
    {
        for (int warm = 0; warm < 3; ++warm)
        {
            Array.Copy(source, joined, source.Length); Legacy(joined, legacy, up, rows, width, pool);
            Array.Copy(source, joined, source.Length); ManagedEmbeddingGatedGelu.Apply(joined, fused, rows, width, pool);
        }
        double Sample(bool candidate)
        {
            double elapsed = 0; int count = 0;
            do
            {
                // Joined projection buffers are freshly overwritten in the real
                // model. Reset both variants outside their measured arithmetic.
                Array.Copy(source, joined, source.Length);
                long start = Stopwatch.GetTimestamp();
                if (candidate) ManagedEmbeddingGatedGelu.Apply(joined, fused, rows, width, pool);
                else Legacy(joined, legacy, up, rows, width, pool);
                elapsed += Stopwatch.GetElapsedTime(start).TotalMilliseconds; ++count;
            } while (elapsed < 20);
            return elapsed / count;
        }
        for (int repeat = 0; repeat < 5; ++repeat)
        {
            if (repeat % 2 == 0) { legacySamples.Add(Sample(false)); fusedSamples.Add(Sample(true)); }
            else { fusedSamples.Add(Sample(true)); legacySamples.Add(Sample(false)); }
        }
    }
    cases.Add(new { rows, width, threads, legacy_bit_exact = true, independent_double_bound_passed = true,
        max_absolute_double_error = maxError, max_error_bound_fraction = maxBoundFraction,
        up_unchanged = true, suffix_canaries_passed = true, invalid_guards_rejected = guardChecks,
        pre_cancel_rejected_and_pool_reusable = true, legacy_samples_ms = legacySamples, fused_samples_ms = fusedSamples,
        legacy_median_ms = timing ? legacySamples.Order().ElementAt(2) : (double?)null,
        fused_median_ms = timing ? fusedSamples.Order().ElementAt(2) : (double?)null });
}

// Worker lifetime is caller-owned. An attempted dispose inside a live job must
// fail, after which the pool must remain usable and drain before real disposal.
using (var lifetime = new CpuWorkerPool(4))
{
    int rejected = 0, work = 0;
    lifetime.For(8, _ => { try { lifetime.Dispose(); } catch (InvalidOperationException) { Interlocked.Increment(ref rejected); } });
    if (rejected != 8) throw new InvalidDataException("Active-job worker disposal was not rejected.");
    lifetime.For(19, _ => Interlocked.Increment(ref work));
    if (work != 19) throw new InvalidDataException("Worker pool did not remain usable.");
    lifetime.Dispose(); lifetime.Dispose();
    bool disposed = false;
    try { ManagedEmbeddingGatedGelu.Apply(new float[2], new float[1], 1, 1, lifetime); } catch (ObjectDisposedException) { disposed = true; }
    if (!disposed) throw new InvalidDataException("Disposed pool accepted fused work.");
}
// Arm cancellation only after an actual output row becomes visible. This
// proves work started, then checks that scratch/jobs drain before propagation.
using (var pool = new CpuWorkerPool(4))
using (var cancel = new CancellationTokenSource())
using (var armed = new ManualResetEventSlim())
{
    const int rows = 4096, width = 2048;
    var joined = Enumerable.Repeat(1f, rows * width * 2).ToArray();
    var output = Enumerable.Repeat(98765f, rows * width).ToArray();
    var trigger = Task.Run(() =>
    {
        armed.Set();
        bool started = SpinWait.SpinUntil(() => Volatile.Read(ref output[0]) != 98765f, TimeSpan.FromSeconds(5));
        if (started) cancel.Cancel();
        return started;
    });
    if (!armed.Wait(TimeSpan.FromSeconds(5))) throw new InvalidDataException("Cancellation observer did not start.");
    bool rejected = false;
    try { ManagedEmbeddingGatedGelu.Apply(joined, output, rows, width, pool, cancel.Token); }
    catch (OperationCanceledException) { rejected = true; }
    if (!await trigger || !rejected) throw new InvalidDataException("In-flight cancellation was not actually exercised.");
    ManagedEmbeddingGatedGelu.Apply(new[] { 1f, 1f }, new float[1], 1, 1, pool);
}
string Hash(string path) => Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(args[0]))!);
File.WriteAllText(args[0], JsonSerializer.Serialize(new { completed = true, cases, timing_executed = timing,
    runtime = Environment.Version.ToString(), architecture = System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString(),
    probe_sha256 = Hash(typeof(ManagedEmbeddingGatedGelu).Assembly.Location),
    tensor_primitives_sha256 = Hash(typeof(TensorPrimitives).Assembly.Location), worker_lifetime_passed = true,
    in_flight_cancellation_executed_and_pool_reused = true,
    limitations = "Arithmetic/layout microprototype only. Independent double bound is conservative; legacy bit identity preserves the existing approximation. Timing includes the same worker scheduling/scratch pool; identical input reset occurs outside measured work, with warm cache and alternating AB/BA order. No model or HTTP performance claim." }, new JsonSerializerOptions { WriteIndented = true }) + "\n");
Console.WriteLine($"Executed {cases.Count} shapes: legacy bits/double/up/tails/canaries/guards/pool lifetime passed. Timing executed={timing}.");
return 0;

static void Legacy(float[] joined, float[] gate, float[] up, int rows, int width, CpuWorkerPool pool)
{
    for (int row = 0; row < rows; ++row)
    {
        Array.Copy(joined, row * width * 2, gate, row * width, width);
        Array.Copy(joined, row * width * 2 + width, up, row * width, width);
    }
    int jobs = Math.Min(rows, Math.Max(1, pool.ThreadCount * 4));
    pool.For(jobs, job =>
    {
        var scratch = ArrayPool<float>.Shared.Rent(width);
        try
        {
            var temp = scratch.AsSpan(0, width);
            for (int row = rows * job / jobs; row < rows * (job + 1) / jobs; ++row)
            {
                var x = gate.AsSpan(row * width, width);
                TensorPrimitives.Multiply(x,x,temp); TensorPrimitives.Multiply(temp,x,temp);
                TensorPrimitives.MultiplyAdd(temp,.044715f,x,temp); TensorPrimitives.Multiply(temp,.7978845608028654f,temp);
                TensorPrimitives.Tanh(temp,temp); TensorPrimitives.Add(temp,1f,temp);
                TensorPrimitives.Multiply(x,temp,x); TensorPrimitives.Multiply(x,.5f,x);
            }
        }
        finally { ArrayPool<float>.Shared.Return(scratch); }
    });
    TensorPrimitives.Multiply(gate.AsSpan(0, rows * width),up.AsSpan(0,rows * width),gate.AsSpan(0,rows * width));
}
