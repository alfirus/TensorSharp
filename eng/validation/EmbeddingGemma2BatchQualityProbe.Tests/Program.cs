using System.Text.Json;

int passed = 0;
void Check(bool value, string name) { if (!value) throw new Exception("Batch quality control failed: " + name); ++passed; }
void Reject(Action action, string name)
{
    bool rejected = false;
    try { action(); } catch (Exception error) when (error is ArgumentException or InvalidDataException) { rejected = true; }
    Check(rejected, name);
}
string sha = new('a', 64);
string[] valid = ["--model", "model.gguf", "--output", "out.json", "--workers", "7", "--policy", "default",
    "--expected-model-sha256", sha, "--expected-models-sha256", sha, "--expected-runtime-sha256", sha, "--expected-server-sha256", sha, "--expected-native-sha256", sha];
string[] Change(string key, string value)
{
    var result = (string[])valid.Clone(); result[Array.IndexOf(result, key) + 1] = value; return result;
}
var options = BatchOptions.Parse(valid);
Check(options.Workers == 7 && options.Policy == "default" && options.Cases == "all" && options.RouteSet == "all" && !options.Diagnostic && !options.IncludeLong, "default CLI is explicit seven-worker core quality");
Check(BatchOptions.Parse(Change("--workers", "8")).Workers == 8, "eight-worker CLI accepted");
Check(BatchOptions.Parse(Change("--policy", "optional")).Policy == "optional", "optional policy explicit");
foreach (string worker in new[] { "4", "0", "-1", "9", "auto" }) Reject(() => BatchOptions.Parse(Change("--workers", worker)), "unqualified worker rejected: " + worker);
Reject(() => BatchOptions.Parse(valid[..^2]), "missing expected identity rejected");
Reject(() => BatchOptions.Parse([..valid, "--workers", "8"]), "duplicate CLI rejected");
Reject(() => BatchOptions.Parse([..valid, "--unknown", "x"]), "unknown CLI rejected");
Reject(() => BatchOptions.Parse([..valid, "--diagnostic"]), "missing option value rejected");
Reject(() => BatchOptions.Parse(Change("--policy", "fast")), "unknown compute policy rejected");
foreach (string invalid in new[] { "A" + sha[1..], sha[..63], new string('z', 64) }) Reject(() => BatchOptions.Parse(Change("--expected-native-sha256", invalid)), "bad identity rejected");
Reject(() => BatchOptions.Parse([..valid, "--include-long", "yes"]), "nonboolean long rejected");
Reject(() => BatchOptions.Parse([..valid, "--include-long", "true"]), "long without bound fixture rejected");
Reject(() => BatchOptions.Parse([..valid, "--long-token-fixture", "fixture.json"]), "long fixture without mode rejected");
Reject(() => BatchOptions.Parse([..valid, "--include-long", "true", "--long-token-fixture", "fixture.json"]), "long without expected fixture SHA rejected");
Check(BatchOptions.Parse([..valid, "--include-long", "true", "--long-token-fixture", "fixture.json", "--expected-long-fixture-sha256", sha]).IncludeLong, "hash-bound long accepted");
Reject(() => BatchOptions.Parse([..valid, "--cases", "bucket8192-boundary"]), "long case cannot be silently selected without bound fixture");
foreach (string selection in new[] { "", "bucket16,bucket16", "unknown", "bucket16, bucket32", "all,bucket16" })
    Reject(() => BatchOptions.Parse([..valid, "--cases", selection]), "invalid case selector rejected: " + selection);
Reject(() => BatchOptions.Parse([..valid, "--route-set", "native-token-only"]), "reduced scope requires diagnostic and bounded cases");
Reject(() => BatchOptions.Parse([..valid, "--diagnostic", "true", "--route-set", "native-token-only"]), "reduced scope rejects all-case default");
var diagnostic = BatchOptions.Parse([..valid, "--diagnostic", "true", "--cases", "observed-batch32-short-geometry", "--route-set", "native-token-only"]);
Check(diagnostic.Diagnostic && diagnostic.RouteSet == "native-token-only", "single native-batch diagnostic explicit");

int[] seed = [2,11,12,13,14,15,16,17,18,19,20,1];
var cases = BatchFixtures.Create(seed);
Check(cases.Length == 7 && cases.All(c => !c.Long), "seven default bounded cases");
int[][] expectedLengths = [[9,15,16,9], [17,31,32,17], [497,511,512,497], [513,527,528,513], [23,23,23,23], [15,17,33,15]];
for (int i = 0; i < expectedLengths.Length; ++i)
{
    Check(cases[i].Lengths.SequenceEqual(expectedLengths[i]), "target exact raw lengths: " + cases[i].Name);
    Check(cases[i].Tokens[0].SequenceEqual(cases[i].Tokens[^1]), "true duplicate row: " + cases[i].Name);
}
Check(cases.All(c => c.Tokens.All(row => row[0] == seed[0] && row[^1] == seed[^1] && row.Skip(1).SkipLast(1).All(seed.Contains))), "all synthetic rows preserve BOS/EOS and observed seed vocabulary");
Check(cases[0].Padded.All(n => n == 16) && cases[1].Padded.All(n => n == 32) && cases[2].Padded.All(n => n == 512) && cases[3].Padded.All(n => n == 528), "same-bucket targets and sliding cutoff geometry");
Check(!cases[4].MixedLengths && cases[4].SameBucket && !cases[5].SameBucket && cases[5].MixedLengths, "equal and multi-bucket controls unaffected by new same-bucket branch");
Check(cases[6].Lengths.SequenceEqual(Enumerable.Repeat(15,10).Concat(Enumerable.Repeat(16,22))) && cases[6].Padded.Sum() == 512 && cases[6].Unique == 2, "observed batch32 geometry and duplicates");
Check(cases.All(c => c.PublicMicrobatches == 1 && c.Padded.Sum() <= 65536), "default cases public microbatch once and native cap bounded");
Check(BatchFixtures.Select(cases, "bucket512,bucket16").Select(c => c.Name).SequenceEqual(new[] { "bucket512", "bucket16" }), "selector order deterministic");
Check(cases[6].LearnedForwards(true) == 1 && cases[6].LearnedTokenRows(true) == 502, "diagnostic exactly one 502-token native learned forward");
int[] original = Enumerable.Range(0,8192).ToArray(); original[0] = 2; original[^1] = 1;
var longCase = BatchFixtures.Create(seed, original)[^1];
Check(longCase.Long && longCase.Lengths.SequenceEqual(new[] { 8191,8192 }) && longCase.Padded.Sum() == 16384, "long native pair explicit bounded padded length");
Check(longCase.Tokens[1].SequenceEqual(original) && longCase.Tokens[0][..^1].SequenceEqual(original[..8190]) && longCase.Tokens[0][^1] == 1, "original long IDs unchanged and shorter drops penultimate retaining EOS");
Check(longCase.PublicMicrobatches == 2 && longCase.LearnedForwards(false) == 5 && longCase.LearnedTokenRows(false) == 65532, "long public split and five-forward resource budget accurately recorded");
Reject(() => BatchFixtures.Create([2,1]), "insufficient observed seed rejected");
Reject(() => BatchFixtures.Create(seed, [2,1]), "wrong long fixture length rejected");
Check(cases.Sum(c => c.LearnedForwards(false)) == 75 && cases.Sum(c => c.Unique) == 20, "full selected core budget based on actual unique token content");
using (var plan = JsonDocument.Parse(JsonSerializer.Serialize(longCase.Plan())))
    Check(!plan.RootElement.GetProperty("public_batch_executed").GetBoolean() && plan.RootElement.GetProperty("expected_public_batch_native_forwards").GetInt32() == 0, "long plan cannot imply public whole-batch execution");
using (var plan = JsonDocument.Parse(JsonSerializer.Serialize(cases[6].Plan(true))))
    Check(!plan.RootElement.GetProperty("public_batch_executed").GetBoolean() && plan.RootElement.GetProperty("expected_learned_forwards").GetInt32() == 1 && plan.RootElement.GetProperty("expected_token_row_graphs").GetInt32() == 0, "diagnostic manifest cannot imply omitted full quality forwards");

float[] unit = new float[768]; unit[0] = .6f; unit[1] = .8f;
Check(BatchMath.Compare(unit, (float[])unit.Clone()).Passed, "identical complete vectors pass");
var negative = unit.Select(x => -x).ToArray(); Check(!BatchMath.Compare(unit, negative).Passed, "opposite vectors fail unchanged cosine");
var preciseFailure = new float[768]; preciseFailure[0] = (float)Math.Sqrt(1 - .0006 * .0006); preciseFailure[1] = .0006f;
var axis = new float[768]; axis[0] = 1;
var maxGate = BatchMath.Compare(axis, preciseFailure);
Check(!maxGate.Passed && maxGate.Cosine > .99999 && maxGate.MaximumError > .0005, "strict maximum-error gate retained independently of cosine");
Reject(() => BatchMath.Compare(unit, unit[..767]), "short output rejected");
var nan = (float[])unit.Clone(); nan[4] = float.NaN; Reject(() => BatchMath.Compare(unit, nan), "nonfinite output rejected before JSON");
Reject(() => BatchMath.Normalized(new float[768], 768), "zero vector rejected");
foreach (int width in new[] { 128,256,512 })
{
    var prefix = BatchMath.Prefix(unit, width); BatchMath.Normalized(prefix, width);
    Check(prefix.Length == width && Math.Abs(prefix[0] - .6) < 1e-6 && Math.Abs(prefix[1] - .8) < 1e-6, "independent prefix normalization: " + width);
}
Reject(() => BatchMath.Prefix(unit, 768), "unsupported MRL width rejected");
var noPrefix = new float[768]; noPrefix[767] = 1; Reject(() => BatchMath.Prefix(noPrefix, 128), "zero MRL prefix rejected");
Reject(() => BatchMath.Prefix(nan, 128), "nonfinite MRL source rejected");
Check(BatchMath.TokenHash([1,2]) != BatchMath.TokenHash([2,1]) && BatchMath.FloatHash([0f]) != BatchMath.FloatHash([-0f]), "raw byte hashes retain order and signed zero");
Console.WriteLine($"BATCH_QUALITY_CONTROLS passed={passed} failed=0 model_executed=0");
