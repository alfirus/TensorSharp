using System.Buffers.Binary;
using System.Security.Cryptography;

internal sealed record BatchOptions(string Model, string Output, int Workers, string Policy, bool IncludeLong, bool Diagnostic, string Cases, string RouteSet,
    string ModelSha, string ModelsSha, string RuntimeSha, string ServerSha, string NativeSha, string? LongFixture, string? LongFixtureSha)
{
    internal static BatchOptions Parse(string[] args)
    {
        string[] required = ["--model", "--output", "--workers", "--policy", "--expected-model-sha256", "--expected-models-sha256", "--expected-runtime-sha256", "--expected-server-sha256", "--expected-native-sha256"];
        string[] optional = ["--include-long", "--diagnostic", "--cases", "--route-set", "--long-token-fixture", "--expected-long-fixture-sha256"];
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 0; i < args.Length; i += 2)
            if (i + 1 == args.Length || !required.Contains(args[i]) && !optional.Contains(args[i]) || !values.TryAdd(args[i], args[i + 1]))
                throw new ArgumentException("Pass documented unique option/value pairs.");
        if (required.Any(key => !values.ContainsKey(key))) throw new ArgumentException("Missing required batch-quality option.");
        if (!int.TryParse(values["--workers"], out int workers) || workers is not (7 or 8)) throw new ArgumentException("Explicit --workers 7 or 8 is required.");
        if (values["--policy"] is not ("default" or "optional")) throw new ArgumentException("Policy must be default or optional.");
        bool Flag(string name) => values.GetValueOrDefault(name, "false") switch { "true" => true, "false" => false, _ => throw new ArgumentException("Flags require true or false.") };
        bool includeLong = Flag("--include-long");
        bool diagnostic = Flag("--diagnostic");
        string cases = values.GetValueOrDefault("--cases", "all"), routeSet = values.GetValueOrDefault("--route-set", "all");
        BatchFixtures.ValidateSelection(cases, includeLong);
        if (routeSet is not ("all" or "native-token-only") || routeSet == "native-token-only" && (!diagnostic || cases == "all"))
            throw new ArgumentException("Native-token-only requires diagnostic true and an explicit bounded --cases selection.");
        string? fixture = values.GetValueOrDefault("--long-token-fixture"), fixtureSha = values.GetValueOrDefault("--expected-long-fixture-sha256");
        if (includeLong != (fixture != null && fixtureSha != null) || !includeLong && (fixture != null || fixtureSha != null))
            throw new ArgumentException("Long mode requires one hash-bound original8192-token fixture; omit both fixture options otherwise.");
        foreach (string key in required.Where(key => key.Contains("sha256"))) ValidateSha(values[key]);
        if (fixtureSha != null) ValidateSha(fixtureSha);
        return new(values["--model"], values["--output"], workers, values["--policy"], includeLong, diagnostic, cases, routeSet,
            values["--expected-model-sha256"], values["--expected-models-sha256"], values["--expected-runtime-sha256"], values["--expected-server-sha256"], values["--expected-native-sha256"], fixture, fixtureSha);
    }
    private static void ValidateSha(string value)
    {
        if (value.Length != 64 || value.Any(c => c is not (>= '0' and <= '9') and not (>= 'a' and <= 'f'))) throw new ArgumentException("Expected lowercase SHA256.");
    }
}

internal sealed record BatchCase(string Name, int[][] Tokens, bool Long = false)
{
    internal int[] Lengths => Tokens.Select(row => row.Length).ToArray();
    internal int[] Padded => Lengths.Select(length => (length + 15) / 16 * 16).ToArray();
    internal bool SameBucket => Padded.Distinct().Count() == 1;
    internal bool MixedLengths => Lengths.Distinct().Count() > 1;
    internal int Unique => Tokens.Select(BatchMath.TokenHash).Distinct().Count();
    internal int LearnedForwards(bool nativeOnly) => nativeOnly ? 1 : Long ? Unique + 3 : 2 * Unique + 4 + PublicMicrobatches;
    internal long LearnedTokenRows(bool nativeOnly) => nativeOnly ? Lengths.Sum() :
        Tokens.GroupBy(BatchMath.TokenHash).Sum(group => (long)group.First().Length) * (Long ? 1 : 2) + Lengths.Sum() * (Long ? 3L : 5L);
    internal int PublicMicrobatches
    {
        get
        {
            var order = Lengths.Order().ToArray(); int calls = 0;
            for (int start = 0; start < order.Length;)
            {
                int end = start + 1, sum = order[start];
                while (end < order.Length && end - start < 64 && sum + order[end] <= 4096) sum += order[end++];
                start = end; ++calls;
            }
            return calls;
        }
    }
    internal object Plan(bool nativeOnly = false) => new { name = Name, lengths = Lengths, padded_lengths = Padded, same_bucket = SameBucket, mixed_lengths = MixedLengths,
        native_microbatch_tokens = Padded.Sum(), unique_sequences = Unique, public_batch_executed = !nativeOnly && !Long,
        expected_public_batch_native_forwards = nativeOnly || Long ? 0 : PublicMicrobatches,
        expected_learned_forwards = LearnedForwards(nativeOnly),
        expected_token_row_graphs = nativeOnly ? 0 : Unique,
        route = nativeOnly ? "diagnostic whole native token batch only; singleton quality comparisons omitted" :
            Long ? "public singletons; whole native token batch/reverse and whole native external-input batch" : "public token/external singletons; public sorted token batch; whole native token/external batches and reversals" };
}

internal static class BatchFixtures
{
    internal static readonly string[] Names = ["bucket16", "bucket32", "bucket512", "bucket528-window-cutoff", "equal32-control", "multiple-buckets-control", "observed-batch32-short-geometry", "bucket8192-boundary"];
    internal static void ValidateSelection(string selection, bool includeLong)
    {
        if (selection == "all") return;
        string[] names = selection.Split(',');
        if (names.Any(name => !Names.Contains(name)) || names.Distinct().Count() != names.Length || !includeLong && names.Contains("bucket8192-boundary"))
            throw new ArgumentException("Select distinct documented case names, with long mode enabled for bucket8192-boundary.");
    }
    internal static BatchCase[] Select(BatchCase[] cases, string selection) => selection == "all" ? cases :
        selection.Split(',').Select(name => cases.Single(c => c.Name == name)).ToArray();
    internal static BatchCase[] Create(int[] seed, int[]? longTokens = null)
    {
        if (seed.Length < 4) throw new ArgumentException("Seed must contain BOS/interior/EOS.");
        int[] Row(int length, int variant)
        {
            var row = new int[length]; row[0] = seed[0]; row[^1] = seed[^1];
            for (int i = 1; i < length - 1; ++i) row[i] = seed[1 + (i - 1 + variant) % (seed.Length - 2)];
            return row;
        }
        BatchCase Case(string name, int[] lengths, int firstVariant)
        {
            var rows = lengths.Select((length, i) => Row(length, firstVariant + i)).ToArray(); rows[^1] = (int[])rows[0].Clone(); return new(name, rows);
        }
        var result = new List<BatchCase>
        {
            Case("bucket16", [9,15,16,9], 0), Case("bucket32", [17,31,32,17], 3),
            Case("bucket512", [497,511,512,497], 6), Case("bucket528-window-cutoff", [513,527,528,513], 9),
            Case("equal32-control", [23,23,23,23], 12), Case("multiple-buckets-control", [15,17,33,15], 15)
        };
        var observed = Enumerable.Range(0, 32).Select(i => Row(i < 10 ? 15 : 16, i < 10 ? 18 : 19)).ToArray();
        result.Add(new("observed-batch32-short-geometry", observed));
        if (longTokens != null)
        {
            if (longTokens.Length != 8192) throw new ArgumentException("Require original8192-token fixture.");
            var shorter = longTokens[..8191]; shorter[^1] = longTokens[^1];
            result.Add(new("bucket8192-boundary", [shorter, (int[])longTokens.Clone()], true));
        }
        foreach (var c in result)
            if (c.Tokens.Any(row => row.Length is < 1 or > 8192) || c.Padded.Sum() > 65536) throw new ArgumentException("Fixture exceeds native token limits.");
        return result.ToArray();
    }
}

internal sealed record VectorGate(bool Passed, double Cosine, double MaximumError);
internal static class BatchMath
{
    internal static string TokenHash(int[] values)
    {
        byte[] bytes = new byte[checked(values.Length * 4)];
        for (int i = 0; i < values.Length; ++i) BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(i * 4, 4), values[i]);
        return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }
    internal static string FloatHash(float[] values)
    {
        byte[] bytes = new byte[checked(values.Length * 4)];
        for (int i = 0; i < values.Length; ++i) BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4, 4), values[i]);
        return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }
    internal static void Normalized(float[] values, int width)
    {
        if (values.Length != width || values.Any(value => !float.IsFinite(value))) throw new InvalidDataException("Invalid complete finite vector.");
        double norm = Math.Sqrt(values.Sum(value => (double)value * value));
        if (Math.Abs(norm - 1) > 1e-5) throw new InvalidDataException("Vector is not unit-normalized.");
    }
    internal static VectorGate Compare(float[] expected, float[] actual)
    {
        Normalized(expected, expected.Length); Normalized(actual, expected.Length);
        double a = expected.Sum(x => (double)x * x), b = actual.Sum(x => (double)x * x);
        double cosine = expected.Zip(actual, (x,y) => (double)x * y).Sum() / Math.Sqrt(a * b);
        double error = expected.Zip(actual, (x,y) => Math.Abs((double)x - y)).Max();
        return new(cosine >= .99999 && error <= .0005, cosine, error);
    }
    internal static float[] Prefix(float[] values, int width)
    {
        if (width is not (128 or 256 or 512) || values.Length != 768 || values.Any(x => !float.IsFinite(x))) throw new ArgumentException("Invalid MRL request.");
        double norm = Math.Sqrt(values.Take(width).Sum(x => (double)x * x));
        if (!(norm > 0)) throw new InvalidDataException("Zero MRL prefix.");
        return values.Take(width).Select(x => (float)(x / norm)).ToArray();
    }
}
