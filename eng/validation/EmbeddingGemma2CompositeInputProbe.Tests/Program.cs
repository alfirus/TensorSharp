using System.Text.Json;
using System.Text.Json.Nodes;

JsonObject Valid()
{
    var flags = new JsonObject(); var sources = new JsonObject();
    foreach (string key in DefaultHttpSource.RequiredOptimizationFlags) { flags[key] = null; sources[key] = null; }
    return new JsonObject
    {
        ["completed"] = true, ["passed"] = true, ["server_exit_code"] = 0, ["require_embedding_default"] = true,
        ["audio_log_policy"] = "HuggingFaceAddFloor", ["binary_artifacts"] = new JsonArray(new JsonObject { ["path"] = "fixture.dll", ["sha256"] = "fixture" }),
        ["cases"] = new JsonArray(new JsonObject { ["name"] = "case", ["parts"] = new JsonArray(new JsonObject { ["kind"] = "text", ["value"] = "original" }) }),
        ["environment_audit"] = new JsonObject
        {
            ["require_embedding_default"] = true, ["resolved_server_environment"] = true,
            ["scope"] = "direct host child environment", ["audio_log_policy"] = "HuggingFaceAddFloor",
            ["optimization_flags"] = flags, ["optimization_flag_sources"] = sources,
        },
    };
}
JsonElement Element(JsonNode value) => JsonSerializer.SerializeToElement(value);
int passed = 0;
void Check(string name, Action action, bool reject)
{
    bool rejected = false;
    try { action(); } catch (Exception error) when (error is InvalidDataException or KeyNotFoundException or InvalidOperationException) { rejected = true; }
    if (rejected != reject) throw new Exception("Incorrect default-HTTP validation: " + name);
    ++passed;
}
void Mutation(string name, Action<JsonObject> change)
{
    var value = Valid(); change(value); Check(name, () => DefaultHttpSource.Validate(Element(value)), true);
}
Check("valid canonical report", () => DefaultHttpSource.Validate(Element(Valid())), false);
foreach (string flag in new[] { "completed", "passed", "require_embedding_default" }) Mutation(flag, x => x[flag] = false);
Mutation("server failure", x => x["server_exit_code"] = 1);
Mutation("controlled audio", x => x["audio_log_policy"] = "LlamaClampFloor");
Mutation("unresolved child", x => x["environment_audit"]!["resolved_server_environment"] = false);
Mutation("unguarded audit", x => x["environment_audit"]!["require_embedding_default"] = false);
Mutation("wrapper audit", x => x["environment_audit"]!["scope"] = "immediate launcher only");
Mutation("audit policy mismatch", x => x["environment_audit"]!["audio_log_policy"] = "LlamaClampFloor");
Mutation("present CUPTI=0", x => x["environment_audit"]!["optimization_flags"]!["TSG_CUPTI_ACTIVITY_OUTPUT"] = "0");
Mutation("present empty override", x => x["environment_audit"]!["optimization_flags"]!["TSG_EMBEDDING_CPU_ATTENTION"] = "");
Mutation("missing CUPTI", x => x["environment_audit"]!["optimization_flags"]!.AsObject().Remove("TSG_CUPTI_ACTIVITY_OUTPUT"));
Mutation("unknown selector", x => x["environment_audit"]!["optimization_flags"]!["NEW_SELECTOR"] = null);
Mutation("selector source present", x => x["environment_audit"]!["optimization_flag_sources"]!["TSG_CUPTI_ACTIVITY_OUTPUT"] = "inherited");
Mutation("selector source omitted", x => x["environment_audit"]!["optimization_flag_sources"]!.AsObject().Remove("TSG_CUPTI_ACTIVITY_OUTPUT"));
Mutation("empty cases", x => x["cases"] = new JsonArray());
Mutation("duplicate case name", x => x["cases"]!.AsArray().Add(x["cases"]![0]!.DeepClone()));
Mutation("unknown part", x => x["cases"]![0]!["parts"]![0]!["kind"] = "unknown");
Mutation("empty parts", x => x["cases"]![0]!["parts"] = new JsonArray());
Mutation("empty artifacts", x => x["binary_artifacts"] = new JsonArray());
Mutation("changed artifacts", x => x["binary_artifacts_unchanged"] = false);
string duplicate = Valid().ToJsonString().Replace("\"TSG_CUPTI_ACTIVITY_OUTPUT\":null", "\"TSG_CUPTI_ACTIVITY_OUTPUT\":null,\"TSG_CUPTI_ACTIVITY_OUTPUT\":null", StringComparison.Ordinal);
Check("duplicate selector properties", () => { using var doc = JsonDocument.Parse(duplicate); DefaultHttpSource.Validate(doc.RootElement); }, true);
var canonical = Valid(); var manifest = new JsonObject { ["cases"] = canonical["cases"]!.DeepClone() };
Check("exact manifest sequence", () => DefaultHttpSource.MatchManifest(Element(canonical), Element(manifest)), false);
manifest["cases"]![0]!["parts"]![0]!["value"] = "changed";
Check("manifest part mismatch", () => DefaultHttpSource.MatchManifest(Element(canonical), Element(manifest)), true);
manifest["cases"]![0]!["name"] = "other";
Check("manifest case mismatch", () => DefaultHttpSource.MatchManifest(Element(canonical), Element(manifest)), true);
Console.WriteLine($"DEFAULT_HTTP_SOURCE_TESTS passed={passed} failed=0 model_executed=0");
