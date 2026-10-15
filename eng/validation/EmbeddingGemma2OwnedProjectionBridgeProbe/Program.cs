using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Models;
using TensorSharp.Models.Embeddings;

if (args.Length is < 2 or > 3 || args.Length == 3 && args[2] != "--policy-only")
    throw new ArgumentException("Pass owned library path, output JSON and optional --policy-only.");
string library = Path.GetFullPath(args[0]), output = Path.GetFullPath(args[1]);
Environment.SetEnvironmentVariable("TENSORSHARP_CPU_NATIVE_LIBRARY", library);
Environment.SetEnvironmentVariable("TENSORSHARP_EMBEDDING_NATIVE_Q16", "0");
Environment.SetEnvironmentVariable("TENSORSHARP_EMBEDDING_Q16_AVX512", "1");
if (OwnedEmbeddingCpuProjection.TryCreate(out string? disabled) != null || disabled != "not-requested")
    throw new InvalidDataException("Disabled option initialized native math.");
Environment.SetEnvironmentVariable("TENSORSHARP_EMBEDDING_NATIVE_Q16", "1");
Environment.SetEnvironmentVariable("TENSORSHARP_EMBEDDING_Q16_AVX512", "0");
if (OwnedEmbeddingCpuProjection.TryCreate(out string? prerequisite) != null || prerequisite != "Q16 activation experiment not enabled")
    throw new InvalidDataException("Native projection bypassed the Q16 activation prerequisite.");
Environment.SetEnvironmentVariable("TENSORSHARP_EMBEDDING_Q16_AVX512", "1");
Environment.SetEnvironmentVariable("TS_CPU_DISABLE_AVX512", "1");
if (OwnedEmbeddingCpuProjection.TryCreate(out string? guarded) != null || string.IsNullOrEmpty(guarded) || ManagedEmbeddingQ8Q16Avx512Matrix.IsSupported)
    throw new InvalidDataException("Runtime disable guard did not preserve fallback.");
Environment.SetEnvironmentVariable("TS_CPU_DISABLE_AVX512", null);
var cases = new List<object>();
void Report(bool native, bool lifetime, string? reason, OwnedEmbeddingCpuProjection? math)
{
    Directory.CreateDirectory(Path.GetDirectoryName(output)!);
    string Hash(string path) => Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
    File.WriteAllText(output, JsonSerializer.Serialize(new
    {
        completed = native || args.Length == 3, fallback_policy_checks = 3, fallback_policy_passed = true,
        native_math_executed = native, native_calls = math?.SuccessfulCalls ?? 0,
        successful_pack_rows = math?.SuccessfulPackRows ?? 0, successful_groups = math?.SuccessfulGroups ?? 0,
        successful_rows = math?.SuccessfulRows ?? 0, float32_subnormal_fallback_rows = math?.Float32FallbackRows ?? 0,
        missing_library_fallback_passed = native, rejected_native_abi_calls = native ? 1 : 0,
        cases, peer_release_and_idempotent_disposal_passed = lifetime,
        library, library_sha256 = math?.LibrarySha256, probe_sha256 = Hash(typeof(OwnedEmbeddingCpuProjection).Assembly.Location),
        unavailable_reason = reason,
        limitations = "Production Q16 matrix bridge, independent scalar integer/F32 fallback bit oracle and unchanged managed comparison. Policy-only/unavailable math is not a native pass. No model/HTTP/performance claim.",
    }, new JsonSerializerOptions { WriteIndented = true }) + "\n");
}
if (args.Length == 3) { Report(false, false, "policy-only", null); return 0; }
var first = OwnedEmbeddingCpuProjection.TryCreate(out string? firstFailure);
if (first == null) { Report(false, false, firstFailure, null); return 2; }
using (first)
using (var peer = OwnedEmbeddingCpuProjection.TryCreate(out string? peerFailure) ?? throw new InvalidDataException(peerFailure))
{
    Environment.SetEnvironmentVariable("TENSORSHARP_CPU_NATIVE_LIBRARY", Path.Combine(library, "missing-library.so"));
    if (OwnedEmbeddingCpuProjection.TryCreate(out string? missing) != null || string.IsNullOrEmpty(missing))
        throw new InvalidDataException("Missing library did not preserve the managed fallback.");
    Environment.SetEnvironmentVariable("TENSORSHARP_CPU_NATIVE_LIBRARY", library);
    first.Dispose(); first.Dispose(); // OS reference count must preserve peer.
    unsafe
    {
        bool rejected = false;
        try { first.Pack(null, null, 32); }
        catch (ObjectDisposedException) { rejected = true; }
        if (!rejected) throw new InvalidDataException("Disposed projection bridge call was accepted.");
        rejected = false;
        try { peer.Pack(null, null, 32); }
        catch (InvalidOperationException) { rejected = true; }
        if (!rejected || peer.SuccessfulCalls != 0) throw new InvalidDataException("Rejected native ABI call counted as successful.");
    }
    foreach (var (inputs, outputs, rows, threads, amplitude) in new[] {
        (32, 1, 1, 1, 8f), (64, 7, 2, 1, 8f), (96, 17, 3, 3, 8f),
        (512, 31, 7, 4, 8f), (1024, 67, 17, 4, 8f), (512, 512, 32, 8, 8f),
        (512, 4096, 33, 8, 8f), (2048, 512, 23, 8, 8f), (1024, 512, 16, 8, 8f),
        (512, 2048, 23, 8, 8f), (512, 12288, 17, 8, 8f),
        (64, 9, 7, 3, 80_000_000f), (64, 9, 7, 3, 1e-37f), (64, 9, 7, 3, 1e-42f) })
    {
        int blocks = inputs / 32;
        var source = new byte[checked(outputs * blocks * 34)];
        for (int col = 0; col < outputs; ++col)
        for (int block = 0; block < blocks; ++block)
        {
            int original = (col * blocks + block) * 34;
            ushort bits = BitConverter.HalfToUInt16Bits((Half)(.005f * (1 + (col + block) % 9)));
            source[original] = (byte)bits; source[original + 1] = (byte)(bits >> 8);
            for (int dim = 0; dim < 32; ++dim)
                source[original + dim + 2] = unchecked((byte)(-128 + (col * 41 + block * 17 + dim * 19) % 256));
        }
        var x = new float[checked(rows * inputs)];
        for (int row = 0; row < rows; ++row)
        for (int dim = 0; dim < inputs; ++dim)
            x[row * inputs + dim] = row == 0 ? 0 : row % 3 == 1 ? (dim % 32 - 16) * (amplitude / 16)
                : MathF.Sin((row * inputs + dim) * .017f) * (amplitude / 8) * (1 + row % 5);
        var managed = new ManagedEmbeddingQ8Q16Avx512Matrix(source, inputs, outputs);
        var candidate = new ManagedEmbeddingQ8Q16Avx512Matrix(source, inputs, outputs, peer);
        using var pool = new CpuWorkerPool(threads);
        var reference = new float[checked(rows * outputs)];
        var actual = Enumerable.Repeat(98765f, reference.Length + 19).ToArray();
        managed.Multiply(x, rows, reference, pool);
        candidate.Multiply(x, rows, actual, pool);
        var quants = new short[x.Length]; var scales = new float[rows * blocks]; var fallback = new bool[rows];
        unsafe
        {
            fixed (float* xp = x, sp = scales)
            fixed (short* qp = quants)
                for (int row = 0; row < rows; ++row)
                    fallback[row] = ManagedEmbeddingQ8Q16Avx512Matrix.Quantize(xp + row * inputs, qp + row * inputs, sp + row * blocks, blocks);
        }
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
                    else dot += quants[row * inputs + block * 32 + dim] * weight;
                }
                if (!fallback[row]) expected = MathF.FusedMultiplyAdd(dot * ws, scales[row * blocks + block], expected);
            }
            int i = row * outputs + col, bits = BitConverter.SingleToInt32Bits(expected);
            if (BitConverter.SingleToInt32Bits(reference[i]) != bits || BitConverter.SingleToInt32Bits(actual[i]) != bits)
                throw new InvalidDataException($"Production native matrix/scalar oracle mismatch [{row},{col}].");
        }
        if (actual.AsSpan(reference.Length).ContainsAnyExcept(98765f)) throw new InvalidDataException("Projection suffix canary changed.");
        long before = peer.SuccessfulCalls;
        bool aliasRejected = false, cancellationRejected = false;
        var alias = new float[Math.Max(x.Length, reference.Length)];
        try { candidate.Multiply(alias, rows, alias, pool); } catch (ArgumentException) { aliasRejected = true; }
        using var cancel = new CancellationTokenSource(); cancel.Cancel();
        try { candidate.Multiply(x, rows, actual, pool, cancel.Token); } catch (OperationCanceledException) { cancellationRejected = true; }
        if (!aliasRejected || !cancellationRejected || peer.SuccessfulCalls != before)
            throw new InvalidDataException("Aliased/cancelled call counted as native success.");
        foreach (float invalid in new[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity })
        {
            float saved = x[0]; x[0] = invalid;
            bool finiteRejected = false;
            try { candidate.Multiply(x, rows, actual, pool); }
            catch (ArgumentException) { finiteRejected = true; }
            catch (AggregateException error) when (error.Flatten().InnerExceptions.All(inner => inner is ArgumentException)) { finiteRejected = true; }
            finally { x[0] = saved; }
            if (!finiteRejected || peer.SuccessfulCalls != before)
                throw new InvalidDataException("Nonfinite activation counted as successful native work.");
        }
        cases.Add(new { inputs, outputs, rows, threads, amplitude, scalar_bit_exact = true, managed_bit_exact = true,
            suffix_canaries_passed = true, alias_rejected = true, cancellation_rejected = true,
            nonfinite_inputs_rejected = 3,
            float32_fallback_rows = fallback.Count(value => value) });
    }
    if (peer.SuccessfulCalls != cases.Count || peer.SuccessfulGroups <= 0 || peer.Float32FallbackRows == 0)
        throw new InvalidDataException("Actual native calls or subnormal fallback coverage missing.");
    Report(true, true, null, peer);
}
return 0;
