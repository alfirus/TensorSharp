using System.Text.Json;

internal sealed record CanonicalOptions(BatchOptions Batch, string BenchmarkReport, string BenchmarkReportSha)
{
    internal static CanonicalOptions Parse(string[] args)
    {
        var core = new List<string>();
        string? report = null, sha = null;
        var keys = new HashSet<string>(StringComparer.Ordinal);
        string[] inherited = ["--model", "--output", "--workers", "--policy", "--expected-model-sha256",
            "--expected-models-sha256", "--expected-runtime-sha256", "--expected-server-sha256", "--expected-native-sha256"];
        for (int i = 0; i < args.Length; i += 2)
        {
            if (i + 1 >= args.Length || !keys.Add(args[i])) throw new ArgumentException("Pass unique option/value pairs.");
            switch (args[i])
            {
                case "--benchmark-report": report = args[i + 1]; break;
                case "--expected-benchmark-report-sha256": sha = args[i + 1]; break;
                default:
                    if (!inherited.Contains(args[i])) throw new ArgumentException("This bounded canonical scope has no case/long/diagnostic selector.");
                    core.Add(args[i]); core.Add(args[i + 1]); break;
            }
        }
        if (string.IsNullOrWhiteSpace(report) || !CanonicalFixture.IsSha(sha)) throw new ArgumentException("Require an exact hash-bound original benchmark report.");
        return new(BatchOptions.Parse(core.ToArray()), report, sha!);
    }
}

internal sealed record CanonicalInput(string[] Texts, bool OriginalStrictPass, string SourceNativeSha);

internal static class CanonicalFixture
{
    internal const string DistinctName = "canonical-batch32-short-distinct";
    internal const string DuplicateName = "canonical-batch32-short-duplicate-control";
    internal static bool IsSha(string? value) => value is { Length: 64 } && value.All(c => c is >= '0' and <= '9' or >= 'a' and <= 'f');
    private static void Require(bool value, string message) { if (!value) throw new InvalidDataException(message); }

    internal static CanonicalInput Parse(byte[] bytes, BatchOptions options)
    {
        using var doc = JsonDocument.Parse(bytes);
        UniqueKeys(doc.RootElement);
        var root = doc.RootElement;
        Require(root.GetProperty("model_sha256").GetString() == options.ModelSha, "Original benchmark model identity differs.");
        Require(root.GetProperty("binary_artifacts_unchanged").ValueKind == JsonValueKind.True, "Original benchmark runtime guard was not stable.");
        var before = Artifacts(root.GetProperty("binary_artifacts"));
        var after = Artifacts(root.GetProperty("binary_artifacts_after"));
        Require(before.Count == after.Count && before.All(pair => after.TryGetValue(pair.Key, out var value) && value == pair.Value), "Original benchmark before/after identities differ.");
        string Identity(string basename)
        {
            var values = before.Where(pair => Path.GetFileName(pair.Key) == basename).ToArray();
            Require(values.Length == 1, "Original benchmark identity must be unique: " + basename);
            return values[0].Value.Sha;
        }
        Require(Identity("TensorSharp.Models.dll") == options.ModelsSha && Identity("TensorSharp.Runtime.dll") == options.RuntimeSha &&
            Identity("TensorSharp.Server.dll") == options.ServerSha, "Original benchmark tokenizer/runtime identity differs.");
        string sourceNative = Identity("libGgmlOps.so"); // Informational baseline; candidate is independently bound by the runner.
        var values = root.GetProperty("benchmark_inputs").GetProperty("batch32_short");
        Require(values.ValueKind == JsonValueKind.Array, "Require the actual saved batch32_short strings.");
        string[] texts = values.EnumerateArray().Select(value => value.ValueKind == JsonValueKind.String ? value.GetString()! : throw new InvalidDataException("Benchmark inputs must be strings.")).ToArray();
        Require(texts.Length == 32 && texts.All(text => !string.IsNullOrWhiteSpace(text)) && texts.Distinct(StringComparer.Ordinal).Count() == 32,
            "Require exactly32 distinct nonempty original benchmark strings.");
        var strict = root.GetProperty("strict_pass");
        Require(strict.ValueKind is JsonValueKind.True or JsonValueKind.False, "Original benchmark outcome is unavailable.");
        return new(texts, strict.GetBoolean(), sourceNative);
    }

    private sealed record Artifact(string Sha, long Bytes);
    private static Dictionary<string, Artifact> Artifacts(JsonElement element)
    {
        var result = new Dictionary<string, Artifact>(StringComparer.Ordinal);
        foreach (var row in element.GetProperty("tensorsharp").EnumerateArray())
        {
            string? path = row.GetProperty("path").GetString(), sha = row.GetProperty("sha256").GetString();
            long bytes = row.GetProperty("bytes").GetInt64();
            Require(!string.IsNullOrWhiteSpace(path) && IsSha(sha) && bytes > 0, "Invalid saved benchmark artifact.");
            Require(result.TryAdd(path!, new(sha!, bytes)), "Duplicate saved benchmark artifact path.");
        }
        Require(result.Count > 0, "Original benchmark artifact binding is empty.");
        return result;
    }

    private static void UniqueKeys(JsonElement element)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            var names = new HashSet<string>(StringComparer.Ordinal);
            foreach (var field in element.EnumerateObject())
            {
                Require(names.Add(field.Name), "Duplicate benchmark JSON key: " + field.Name);
                UniqueKeys(field.Value);
            }
        }
        else if (element.ValueKind == JsonValueKind.Array)
            foreach (var item in element.EnumerateArray()) UniqueKeys(item);
    }

    internal static BatchCase[] Create(int[][] tokens)
    {
        Require(tokens.Length == 32 && tokens.Select((row, i) => row.Length == (i < 10 ? 15 : 16)).All(value => value),
            "Actual frozen tokenizer must observe10x15 then22x16 token rows.");
        Require(tokens.Select(BatchMath.TokenHash).Distinct(StringComparer.Ordinal).Count() == 32,
            "Actual frozen tokenizer must observe32 distinct token sequences.");
        var distinct = new BatchCase(DistinctName, tokens.Select(row => (int[])row.Clone()).ToArray());
        var duplicate = new BatchCase(DuplicateName, [(int[])tokens[0].Clone(), (int[])tokens[10].Clone(), (int[])tokens[0].Clone()]);
        BatchCase[] cases = [distinct, duplicate];
        Require(cases.All(c => c.SameBucket && c.MixedLengths && c.Padded.Sum() <= 65536) &&
            distinct.Unique == 32 && duplicate.Unique == 2 && cases.Sum(c => c.LearnedForwards(false)) == 78 && cases.Sum(c => c.Unique) == 34,
            "Canonical scope/bounded forward contract differs.");
        return cases;
    }
}
