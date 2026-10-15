using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Models.Embeddings;

if (args.Length is < 2 or > 3 || args.Length == 3 && args[2] != "--policy-only")
    throw new ArgumentException("Pass owned library path, output JSON and optional --policy-only.");
string library = Path.GetFullPath(args[0]), output = Path.GetFullPath(args[1]);
Environment.SetEnvironmentVariable("TENSORSHARP_CPU_NATIVE_LIBRARY", library);
Environment.SetEnvironmentVariable("TENSORSHARP_EMBEDDING_NATIVE_ATTENTION", "0");
if (OwnedEmbeddingCpuAttention.TryCreate(out string? disabled) != null || disabled != "not-requested")
    throw new InvalidDataException("Disabled option initialized native math.");
Environment.SetEnvironmentVariable("TENSORSHARP_EMBEDDING_NATIVE_ATTENTION", "1");
Environment.SetEnvironmentVariable("TS_CPU_DISABLE_AVX512", "1");
if (OwnedEmbeddingCpuAttention.TryCreate(out string? guarded) != null || string.IsNullOrEmpty(guarded))
    throw new InvalidDataException("Runtime disable guard did not preserve the managed fallback.");
Environment.SetEnvironmentVariable("TS_CPU_DISABLE_AVX512", null);
void Report(bool native, bool lifetime, int nativeCalls, string? reason, string? hash)
{
    Directory.CreateDirectory(Path.GetDirectoryName(output)!);
    File.WriteAllText(output, JsonSerializer.Serialize(new
    {
        fallback_policy_checks = 2, fallback_policy_passed = true, native_math_executed = native,
        native_calls = nativeCalls, peer_release_and_idempotent_disposal_passed = lifetime,
        library, library_sha256 = hash, unavailable_reason = reason,
        limitations = "Isolated owned loader/call ABI and peer handle lifetime. No model or performance claim; policy-only/unavailable math is not a native pass.",
    }, new JsonSerializerOptions { WriteIndented = true }) + "\n");
}
if (args.Length == 3) { Report(false, false, 0, "policy-only", null); return 0; }
var first = OwnedEmbeddingCpuAttention.TryCreate(out string? firstFailure);
if (first == null) { Report(false, false, 0, firstFailure, null); return 2; }
using (first)
using (var peer = OwnedEmbeddingCpuAttention.TryCreate(out string? peerFailure) ?? throw new InvalidDataException(peerFailure))
{
    first.Dispose(); first.Dispose(); // Must release exactly one OS library lease.
    const int dim = 32, keys = 17, rows = 3, qs = 39, ks = 32, ss = 23, ys = 37, prefix = 11;
    var q = new float[rows * qs]; var k = new float[dim * ks]; var v = new float[dim * ks];
    var p = new float[rows * ss];
    for (int r = 0; r < rows; ++r)
    for (int d = 0; d < dim; ++d) q[r * qs + d] = ((r * 3 + d) % 11 - 5) * .125f;
    for (int d = 0; d < dim; ++d)
    for (int key = 0; key < keys; ++key)
    {
        k[d * ks + key] = ((d + key * 7) % 17 - 8) * .0625f;
        v[(d / 16 * 16) * ks + key * 16 + d % 16] = ((d * 3 + key) % 13 - 6) * .125f;
    }
    for (int r = 0; r < rows; ++r)
    for (int key = 0; key < keys; ++key) p[r * ss + key] = (r + 1) * .0625f / (key + 1);
    var scores = Enumerable.Repeat(9876.5f, prefix + rows * ss + 19).ToArray();
    var values = Enumerable.Repeat(9876.5f, prefix + rows * ys + 19).ToArray();
    unsafe
    {
        fixed (float* qp = q, kp = k, vp = v, pp = p, sp = scores, yp = values)
        {
            peer.Scores(qp, kp, sp + prefix, qs, ks, ss, dim, keys, rows, .3f);
            peer.Values(pp, vp, yp + prefix, ss, ks, ys, dim, keys, rows);
            bool rejected = false;
            try { first.Scores(qp, kp, sp + prefix, qs, ks, ss, dim, keys, rows, .3f); }
            catch (ObjectDisposedException) { rejected = true; }
            if (!rejected) throw new InvalidDataException("Disposed bridge call was accepted.");
        }
    }
    for (int r = 0; r < rows; ++r)
    {
        for (int key = 0; key < keys; ++key)
        {
            float sum = 0;
            for (int d = 0; d < dim; ++d) sum = MathF.FusedMultiplyAdd(k[d * ks + key], q[r * qs + d], sum);
            if (BitConverter.SingleToInt32Bits(scores[prefix + r * ss + key]) != BitConverter.SingleToInt32Bits(sum * .3f))
                throw new InvalidDataException("Bridge score ABI differs from scalar FMA oracle.");
        }
        for (int d = 0; d < dim; ++d)
        {
            float sum = 0;
            for (int key = 0; key < keys; ++key)
                sum = MathF.FusedMultiplyAdd(v[(d / 16 * 16) * ks + key * 16 + d % 16], p[r * ss + key], sum);
            if (BitConverter.SingleToInt32Bits(values[prefix + r * ys + d]) != BitConverter.SingleToInt32Bits(sum))
                throw new InvalidDataException("Bridge value ABI differs from scalar FMA oracle.");
        }
        if (scores.AsSpan(prefix + r * ss + keys, ss - keys).ContainsAnyExcept(9876.5f) ||
            values.AsSpan(prefix + r * ys + dim, ys - dim).ContainsAnyExcept(9876.5f))
            throw new InvalidDataException("Bridge stride canary changed.");
    }
    if (scores.AsSpan(0, prefix).ContainsAnyExcept(9876.5f) || scores.AsSpan(scores.Length - 19).ContainsAnyExcept(9876.5f) ||
        values.AsSpan(0, prefix).ContainsAnyExcept(9876.5f) || values.AsSpan(values.Length - 19).ContainsAnyExcept(9876.5f))
        throw new InvalidDataException("Bridge leading/suffix canary changed.");
    using var stream = File.OpenRead(library);
    Report(true, true, 2, null, Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant());
}
return 0;
