using System.Text.Json;
using System.Text.Json.Nodes;

int controls = 0;
void Check(bool value, string message) { if (!value) throw new InvalidOperationException(message); ++controls; }
void Reject(Action action, string message)
{
    bool rejected = false;
    try { action(); } catch (Exception error) when (error is ArgumentException or InvalidDataException or JsonException or KeyNotFoundException) { rejected = true; }
    Check(rejected, message);
}
string sha = new('a', 64), native = new('b', 64);
string[] core = ["--model", "/model/embeddinggemma-2-Q8_0.gguf", "--output", "/new/report.json", "--workers", "7", "--policy", "default",
    "--expected-model-sha256", sha, "--expected-models-sha256", sha, "--expected-runtime-sha256", sha, "--expected-server-sha256", sha, "--expected-native-sha256", native];
string[] valid = [.. core, "--benchmark-report", "/original/results.json", "--expected-benchmark-report-sha256", sha];
var options = CanonicalOptions.Parse(valid);
Check(options.Batch.Workers == 7 && options.Batch.Cases == "all" && !options.Batch.Diagnostic && !options.Batch.IncludeLong && options.Batch.RouteSet == "all", "Bounded CLI changed.");
Check(options.BenchmarkReportSha == sha && options.BenchmarkReport == "/original/results.json", "CLI report binding changed.");
foreach (string workers in new[] { "0", "4", "6", "9" })
{
    string[] changed = (string[])valid.Clone(); changed[5] = workers;
    Reject(() => CanonicalOptions.Parse(changed), "Unqualified workers accepted.");
}
foreach (string flag in new[] { "--cases", "--include-long", "--diagnostic", "--route-set", "--long-token-fixture", "--unknown" })
    Reject(() => CanonicalOptions.Parse([.. valid, flag, "all"]), "Unbounded selector accepted.");
Reject(() => CanonicalOptions.Parse(core), "Unbound benchmark accepted.");
Reject(() => CanonicalOptions.Parse([.. core, "--benchmark-report", "/report"]), "Missing report SHA accepted.");
Reject(() => CanonicalOptions.Parse([.. valid, "--workers", "8"]), "Duplicate option accepted.");
Reject(() => CanonicalOptions.Parse([.. valid, "--policy"]), "Unpaired option accepted.");
foreach (string invalid in new[] { "", "a", new string('A', 64), new string('x', 64) })
{
    var changed = (string[])valid.Clone(); changed[^1] = invalid;
    Reject(() => CanonicalOptions.Parse(changed), "Malformed report hash accepted.");
}
JsonObject Source()
{
    JsonArray Artifacts() => new(new[] { "TensorSharp.Models.dll", "TensorSharp.Runtime.dll", "TensorSharp.Server.dll", "libGgmlOps.so" }
        .Select(name => (JsonNode)new JsonObject { ["path"] = "/frozen/" + name, ["sha256"] = name.EndsWith(".so") ? native : sha, ["bytes"] = 1 }).ToArray());
    return new() { ["model_sha256"] = sha, ["strict_pass"] = false, ["binary_artifacts_unchanged"] = true,
        ["binary_artifacts"] = new JsonObject { ["tensorsharp"] = Artifacts() },
        ["binary_artifacts_after"] = new JsonObject { ["tensorsharp"] = Artifacts() },
        ["benchmark_inputs"] = new JsonObject { ["batch32_short"] = new JsonArray(Enumerable.Range(0,32).Select(i => (JsonNode?)JsonValue.Create("actual text " + i)).ToArray()) } };
}
CanonicalInput Parse(JsonObject node) => CanonicalFixture.Parse(JsonSerializer.SerializeToUtf8Bytes(node), options.Batch);
var source = Source(); var input = Parse(source);
Check(input.Texts.Length == 32 && input.Texts.Distinct().Count() == 32 && !input.OriginalStrictPass && input.SourceNativeSha == native, "Original negative outcome/input binding changed.");
foreach (string key in new[] { "model_sha256", "strict_pass", "binary_artifacts_unchanged", "benchmark_inputs", "binary_artifacts_after" })
{
    var changed = Source(); changed.Remove(key);
    Reject(() => Parse(changed), "Missing report field accepted.");
}
{
    var changed = Source(); changed["model_sha256"] = native;
    Reject(() => Parse(changed), "Other model accepted.");
    changed = Source(); changed["binary_artifacts_unchanged"] = false;
    Reject(() => Parse(changed), "Unstable source runtime accepted.");
    changed = Source(); changed["strict_pass"] = "false";
    Reject(() => Parse(changed), "Unavailable original outcome accepted.");
    changed = Source(); changed["binary_artifacts_after"]!["tensorsharp"]![0]!["sha256"] = native;
    Reject(() => Parse(changed), "Before/after artifact mismatch accepted.");
    changed = Source(); changed["binary_artifacts"]!["tensorsharp"]![0]!["bytes"] = 0;
    Reject(() => Parse(changed), "Malformed source artifact accepted.");
}
foreach (int index in new[] { 0, 1, 2 })
{
    var changed = Source();
    foreach (string phase in new[] { "binary_artifacts", "binary_artifacts_after" }) changed[phase]!["tensorsharp"]![index]!["sha256"] = native;
    Reject(() => Parse(changed), "Other frozen tokenizer/runtime identity accepted.");
}
{
    var changed = Source(); var array = changed["binary_artifacts"]!["tensorsharp"]!.AsArray(); array.Add(array[0]!.DeepClone());
    Reject(() => Parse(changed), "Duplicate source artifact path accepted.");
    changed = Source(); changed["benchmark_inputs"]!["batch32_short"]![31] = "actual text 0";
    Reject(() => Parse(changed), "Repeated canonical text accepted.");
    changed = Source(); changed["benchmark_inputs"]!["batch32_short"]![0] = " ";
    Reject(() => Parse(changed), "Empty canonical text accepted.");
    changed = Source(); changed["benchmark_inputs"]!["batch32_short"]!.AsArray().RemoveAt(31);
    Reject(() => Parse(changed), "Incomplete canonical batch accepted.");
    changed = Source(); changed["benchmark_inputs"]!["batch32_short"]![0] = 123;
    Reject(() => Parse(changed), "Non-string canonical text accepted.");
}
{
    string raw = Source().ToJsonString(); raw = raw.Insert(1, "\"strict_pass\":true,");
    Reject(() => CanonicalFixture.Parse(System.Text.Encoding.UTF8.GetBytes(raw), options.Batch), "Duplicate JSON key accepted.");
}
int[][] Rows() => Enumerable.Range(0,32).Select(i => Enumerable.Range(0, i < 10 ? 15 : 16).Select(x => x == 0 ? 1 : x == (i < 10 ? 14 : 15) ? 2 : 17 + i).ToArray()).ToArray();
var tokens = Rows(); var before = tokens.Select(BatchMath.TokenHash).ToArray(); var cases = CanonicalFixture.Create(tokens);
Check(cases.Length == 2 && cases[0].Name == CanonicalFixture.DistinctName && cases[1].Name == CanonicalFixture.DuplicateName, "Case scope changed.");
Check(cases.Sum(c => c.LearnedForwards(false)) == 78 && cases.Sum(c => c.Unique) == 34 && cases.Sum(c => c.LearnedTokenRows(false)) == 3806, "Forward/token-row budget changed.");
Check(cases[0].Unique == 32 && cases[0].Padded.Sum() == 512 && cases[1].Unique == 2 && cases[1].Padded.Sum() == 48, "Padded/unique geometry changed.");
Check(cases.All(c => c.SameBucket && c.MixedLengths && c.PublicMicrobatches == 1), "Actual canonical bucket topology changed.");
Check(cases[1].Tokens[0].SequenceEqual(tokens[0]) && cases[1].Tokens[1].SequenceEqual(tokens[10]) && cases[1].Tokens[2].SequenceEqual(tokens[0]), "Duplicate source rows changed.");
Check(before.SequenceEqual(tokens.Select(BatchMath.TokenHash)), "Fixture construction mutated source.");
cases[0].Tokens[0][1] += 1;
Check(before.SequenceEqual(tokens.Select(BatchMath.TokenHash)) && cases[1].Tokens[0].SequenceEqual(tokens[0]), "Cloned case rows alias original/duplicate sources.");
Reject(() => CanonicalFixture.Create(Rows()[..31]), "Missing token sequence accepted.");
{
    var changed = Rows(); changed[31] = (int[])changed[10].Clone();
    Reject(() => CanonicalFixture.Create(changed), "Repeated distinct sequence accepted.");
    changed = Rows(); changed[0] = changed[0][..14];
    Reject(() => CanonicalFixture.Create(changed), "Unexpected actual tokenizer geometry accepted.");
    changed = Rows(); (changed[0], changed[10]) = (changed[10], changed[0]);
    Reject(() => CanonicalFixture.Create(changed), "Wrong actual tokenizer order accepted.");
}
Console.WriteLine($"CANONICAL_BATCH_PURE_CONTROLS passed={controls} failed=0 model_executed=0");
