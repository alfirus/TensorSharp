using System.Buffers.Binary;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Models.Embeddings;
using TensorSharp.Runtime;

// This observes actual tokenizer lengths and proves layout eligibility only.
// It never constructs an encoder or executes an embedding graph.
if (args.Length != 3) throw new ArgumentException("Pass <model.gguf> <case-to-text-array.json> <new-report.json>.");
string model = Path.GetFullPath(args[0]), manifest = Path.GetFullPath(args[1]), output = Path.GetFullPath(args[2]);
if (File.Exists(output)) throw new IOException("Preserve existing reports; output already exists.");
string Hash(string path) { using var file = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(file)).ToLowerInvariant(); }
var guarded = new[] { model, manifest, typeof(EmbeddingModel).Assembly.Location, typeof(GgufFile).Assembly.Location };
var before = guarded.Distinct().ToDictionary(path => path, Hash);
using var gguf = new GgufFile(model);
if (gguf.GetString("general.architecture") != "gemma-embedding2") throw new InvalidDataException("Require Gemma embedding2.");
var tokenizerType = typeof(EmbeddingModel).Assembly.GetType("TensorSharp.Models.Embeddings.EmbeddingTokenizer", throwOnError: true)!;
var create = tokenizerType.GetMethod("Create", BindingFlags.Static | BindingFlags.Public)!;
var tokenizer = (ITokenizer)create.Invoke(null, [gguf])!;
long Field(string name) => Convert.ToInt64(gguf.Metadata["gemma-embedding2." + name]);
int context = checked((int)Math.Min(8192, Field("context_length")));
int Pad(int length) => Math.Min(context, checked((length + 15) / 16 * 16));
using var inputs = JsonDocument.Parse(File.ReadAllBytes(manifest));
if (inputs.RootElement.ValueKind != JsonValueKind.Object) throw new InvalidDataException("Require case-name to text-array object.");
var results = new List<object>();
foreach (var scenario in inputs.RootElement.EnumerateObject())
{
    string[] texts = scenario.Value.EnumerateArray().Select(item => item.GetString()
        ?? throw new InvalidDataException("Require string inputs.")).ToArray();
    if (texts.Length < 1 || texts.Length > 1024) throw new InvalidDataException("Invalid batch count.");
    var rows = texts.Select(text => tokenizer.Encode(text)).ToArray();
    int[] lengths = rows.Select(ids => ids.Count).ToArray();
    if (lengths.Any(length => length < 1 || length > context) || lengths.Sum() > 65536)
        throw new InvalidDataException("Lengths outside the encoder contract.");
    int[] padded = lengths.Select(Pad).ToArray();
    if (padded.Sum() > 65536) throw new InvalidDataException("Padded microbatch exceeds the native token cap.");
    bool sameLength = lengths.All(length => length == lengths[0]);
    bool sameBucket = padded.All(length => length == padded[0]);
    bool topologyCandidate = texts.Length > 1 && !sameLength && sameBucket && Field("pooling_type") == 1;
    int[] oldStarts = new int[texts.Length], candidateStarts = new int[texts.Length];
    for (int i = 1; i < texts.Length; ++i) oldStarts[i] = checked(oldStarts[i - 1] + padded[i - 1]);
    for (int i = 0; i < texts.Length; ++i) candidateStarts[i] = checked(i * padded[0]);
    if (sameBucket && !oldStarts.SequenceEqual(candidateStarts)) throw new InvalidOperationException("Flat row order changed.");
    string[] tokenHashes = rows.Select(ids => {
        var bytes = new byte[checked(ids.Count * 4)];
        for (int i = 0; i < ids.Count; ++i) BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(i * 4, 4), ids[i]);
        return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }).ToArray();
    results.Add(new { name = scenario.Name, batch = texts.Length, actual_token_lengths = lengths,
        padded_lengths = padded, token_id_sha256_le = tokenHashes, original_equal_actual_lengths = sameLength,
        equal_existing_padding = sameBucket, theoretical_bucket_topology_candidate = topologyCandidate,
        original_flat_starts = oldStarts, candidate_flat_starts = sameBucket ? candidateStarts : null,
        identical_flat_starts = sameBucket ? (bool?)oldStarts.SequenceEqual(candidateStarts) : null,
        original_attention_calls_per_layer = sameLength ? 1 : texts.Length,
        candidate_attention_calls_per_layer = topologyCandidate ? 1 : sameLength ? 1 : texts.Length,
        original_attention_concat_nodes_per_layer = sameLength ? 0 : texts.Length - 1,
        candidate_attention_concat_nodes_per_layer = topologyCandidate ? 0 : sameLength ? 0 : texts.Length - 1 });
}
if (results.Count == 0) throw new InvalidDataException("Require at least one case.");
var after = guarded.Distinct().ToDictionary(path => path, Hash);
bool unchanged = before.Count == after.Count && before.All(pair => after[pair.Key] == pair.Value);
if (!unchanged) throw new InvalidOperationException("A guarded input changed during observation.");
using var destination = new FileStream(output, FileMode.CreateNew, FileAccess.Write, FileShare.None);
JsonSerializer.Serialize(destination, new { completed = true, created_utc = DateTimeOffset.UtcNow,
    scope = "Actual frozen production tokenizer lengths and source-derived batch geometry; zero embedding/native forwards.",
    model, architecture = gguf.GetString("general.architecture"), effective_context = context, pooling_type = Field("pooling_type"),
    before_hashes = before, after_hashes = after, guarded_inputs_unchanged = unchanged, cases = results,
    native_admission_observed = false, model_quality_qualified = false, latency_qualified = false,
    limitations = "Topology eligibility still requires production CPU geometry/F32/worker predicates. Operation counts are source-derived, not an executed graph profile. Mean pooling F32 ordering changes and requires full-model quality gates." },
    new JsonSerializerOptions { WriteIndented = true });
Console.WriteLine($"Observed {results.Count} tokenizer/layout cases; zero embedding forwards.");
