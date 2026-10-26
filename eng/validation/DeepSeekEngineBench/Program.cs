// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

var options = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i += 2)
{
    if (!args[i].StartsWith("--", StringComparison.Ordinal) || i + 1 >= args.Length)
        throw new ArgumentException("Expected --name value pairs.");
    options.Add(args[i][2..], args[i + 1]);
}
int Number(string name, int fallback) => options.TryGetValue(name, out string? value) ? int.Parse(value) : fallback;
string modelPath = Path.GetFullPath(options["model"]);
var jsonOptions = new JsonSerializerOptions { WriteIndented = true };
if (options.TryGetValue("export-cases", out string? export))
{
    using var gguf = new GgufFile(modelPath);
    ITokenizer tokenizer = ModelBase.CreateTokenizerFromGguf(gguf);
    var prompts = new List<(string Name, string Prompt, int Max)>
    {
        ("arithmetic", "Reply with only the integer result: 17 multiplied by 23.", 32),
        ("geography", "What is the capital of France? Reply with only the city name.", 32),
        ("sorting", "Sort these integers in ascending order: 9, 2, 11, -3. Reply with only the list.", 48),
        ("json", "Return exactly a JSON object with keys answer and unit, with integer answer 42 and string unit meters. No explanation.", 64),
        ("code", "Write a Python function is_prime(n) that correctly handles integers below 2. Include only code.", 128),
        ("retrieval", "In the following records, Ada has 7 books, Ben has 12 books, and Cleo has 4 books. Who has the most books? Reply with only the person's name.", 32),
        ("performance-short", "Explain how a hash table works, including collisions, resizing, and time complexity. Write a detailed answer with examples.", 128),
    };
    string paragraph = "A distributed inference system stores neural network layers on different machines. " +
        "Each machine computes its assigned layers and sends intermediate activations to the next stage. " +
        "The weights, key and value caches, network latency, and device memory bandwidth all affect performance. " +
        "A fair benchmark records identical token inputs, actual output counts, warmups, errors, and each device's work. ";
    prompts.Add(("performance-prefill", string.Concat(Enumerable.Repeat(paragraph, 12)) +
        "\nExplain the main performance bottlenecks and describe a controlled benchmark in detail.", 128));
    var cases = prompts.Select(p => new
    {
        name = p.Name, prompt = p.Prompt,
        rendered_prompt = ChatTemplate.RenderDeepSeek41([new ChatMessage { Role = "user", Content = p.Prompt }], enableThinking: false),
        ids = tokenizer.Encode(ChatTemplate.RenderDeepSeek41([new ChatMessage { Role = "user", Content = p.Prompt }],
            enableThinking: false), addSpecial: false), max_new = p.Max,
    }).ToArray();
    File.WriteAllText(export, JsonSerializer.Serialize(new { schema = 1, thinking = false, temperature = 0,
        tokenizer_model = gguf.GetString("tokenizer.ggml.model"), vocab_size = tokenizer.VocabSize,
        eos_ids = tokenizer.EosTokenIds, cases }, jsonOptions));
    Console.WriteLine($"Exported {cases.Length} identical-token cases to {export}");
    return;
}
string output = Path.GetFullPath(options["output"]);
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
string? captureDirectory = options.TryGetValue("capture-logits", out string? capture) ? Path.GetFullPath(capture) : null;
int greedyTransportOption = Number("greedy-transport", 0);
if (greedyTransportOption is not (0 or 1)) throw new ArgumentException("--greedy-transport must be 0 or 1.");
bool greedyTransport = greedyTransportOption == 1;
if (greedyTransport && captureDirectory != null)
    throw new ArgumentException("Compact greedy transport cannot capture full vocabulary rows; run a separate full-logit replay.");
if (greedyTransport) Environment.SetEnvironmentVariable("TS_DSV41_PIPELINE_GREEDY_TRANSPORT", "1");
if (captureDirectory != null) Directory.CreateDirectory(captureDirectory);
using JsonDocument casesDocument = JsonDocument.Parse(File.ReadAllText(options["cases"]));
var casesToRun = casesDocument.RootElement.GetProperty("cases").EnumerateArray().ToArray();
bool forcedContinuation = casesToRun.Any(item => item.TryGetProperty("forced_ids", out _));
int repeats = Number("repeats", 3), warmups = Number("warmups", 1), context = Number("context", 4096);
if (repeats < 1 || warmups < 0 || context < 1) throw new ArgumentOutOfRangeException(nameof(repeats));
Environment.SetEnvironmentVariable("MAX_CONTEXT", context.ToString());
Environment.SetEnvironmentVariable("TS_DSV41_YOCO_KEEP", "0");
Environment.SetEnvironmentVariable("TS_DSV41_ENGRAM_WARM", "0");
Environment.SetEnvironmentVariable("TS_DSV41_ENGRAM_DEVICE", "0");
string cpuMoe = options.GetValueOrDefault("moe-cpu", "none");
if (cpuMoe != "none") MoeCpuOffloadConfig.SetLayers(int.Parse(cpuMoe));
var results = new List<object>();
var report = new Dictionary<string, object?>
{
    ["schema"] = 1, ["engine"] = options.GetValueOrDefault("engine", "TensorSharp"),
    ["source_revision"] = options.GetValueOrDefault("source-revision", "unrecorded"),
    ["model"] = modelPath, ["started_utc"] = DateTimeOffset.UtcNow,
    ["case_file_sha256"] = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(options["cases"]))),
    ["context"] = context, ["temperature"] = 0, ["exact_prefill"] = true,
    ["prefix_reuse"] = false, ["dspark"] = false, ["results"] = results, ["status"] = "running",
    ["pipeline_peer"] = Environment.GetEnvironmentVariable("TS_DSV41_PIPELINE_PEER"),
    ["options"] = options, ["quality_qualified"] = false, ["performance_qualified"] = false,
    ["measurement_scope"] = greedyTransport
        ? "Greedy raw-token engine execution, with host argmax after full-logit prefill and exact tail argmax for later single-token forwards; cache-position checks included, reset recorded separately. Completed generation does not establish semantic quality."
        : "Greedy raw-token engine execution, including host argmax and cache-position checks; reset time is recorded separately. Completed generation does not establish semantic quality.",
    ["logit_capture_directory"] = captureDirectory,
    ["capture_excluded_from_performance"] = captureDirectory != null,
    ["teacher_forced_replay"] = forcedContinuation,
    ["performance_excluded"] = captureDirectory != null || forcedContinuation,
    ["greedy_transport"] = greedyTransport,
    ["greedy_transport_contract"] = greedyTransport
        ? "Prefill returns full F32 logits; each later single-token forward returns the tail's first argmax of a complete finite F32 vocabulary. No approximate or synthesized logits."
        : null,
};
void Save()
{
    string temporary = output + ".tmp";
    File.WriteAllText(temporary, JsonSerializer.Serialize(report, jsonOptions));
    File.Move(temporary, output, overwrite: true);
}
Save();
try
{
    var load = Stopwatch.StartNew();
    int tp = Number("tp", 1);
    using ModelBase model = ModelBase.Create(modelPath, BackendType.GgmlCuda,
        tpDegree: tp, layerSplitDegree: Number("gpus", tp > 1 ? 1 : 2));
    report["load_seconds"] = load.Elapsed.TotalSeconds;
    if (model is not DeepSeek4Model deepSeek)
        throw new InvalidOperationException("This benchmark requires the DeepSeek whole-model executor.");
    report["kv_fingerprint"] = model.KVStateFingerprint;
    int CacheHead() => deepSeek.CacheSeqLen;
    for (int repeat = -warmups; repeat < repeats; ++repeat)
    {
        foreach (JsonElement item in casesToRun)
        {
            int[] ids = item.GetProperty("ids").EnumerateArray().Select(t => t.GetInt32()).ToArray();
            int maxNew = item.GetProperty("max_new").GetInt32();
            int[]? forced = item.TryGetProperty("forced_ids", out JsonElement forcedIds)
                ? forcedIds.EnumerateArray().Select(t => t.GetInt32()).ToArray() : null;
            if (forced != null && (forced.Length != maxNew || forced.Any(t => t < 0 || t >= model.Config.VocabSize)))
                throw new ArgumentException("Teacher-forced IDs must contain exactly max_new in-vocabulary tokens.");
            if (ids.Length == 0 || ids.Any(token => token < 0 || token >= model.Config.VocabSize)
                || maxNew < 1 || (long)ids.Length + maxNew > context)
                throw new ArgumentException("Invalid case token/context bounds.");
            var generated = new List<int>();
            var tokenTimes = new List<double>();
            var argmaxIds = new List<int>();
            var captures = new List<object>();
            var result = new Dictionary<string, object?>
            {
                ["name"] = item.GetProperty("name").GetString(), ["repeat"] = repeat, ["warmup"] = repeat < 0,
                ["prompt_tokens"] = ids.Length, ["max_new"] = maxNew, ["generated_ids"] = generated,
                ["token_times_seconds"] = tokenTimes, ["status"] = "running",
                ["teacher_forced"] = forced != null, ["argmax_ids"] = argmaxIds, ["logit_captures"] = captures,
            };
            results.Add(result); Save();
            Stopwatch? timer = null;
            int compactPredictions = 0;
            try
            {
                result["cache_before_reset"] = CacheHead();
                var reset = Stopwatch.StartNew();
                model.ResetKVCache();
                result["reset_seconds"] = reset.Elapsed.TotalSeconds;
                result["cache_after_reset"] = CacheHead();
                if (CacheHead() != 0) throw new InvalidDataException("Model reset left a nonzero cache head.");
                result["head_expert_cache_before"] = ExpertCacheCounters.Read();
                timer = Stopwatch.StartNew();
                float[] logits = model.ForwardRefill(ids);
                result["prefill_seconds"] = timer.Elapsed.TotalSeconds;
                result["cache_after_prefill"] = CacheHead();
                if (CacheHead() != ids.Length) throw new InvalidDataException("Prefill advanced to the wrong cache head.");
                string finish = "length";
                int? compactNext = null;
                for (int i = 0; i < maxNew; ++i)
                {
                    int next;
                    if (compactNext.HasValue)
                    {
                        next = compactNext.Value;
                        compactNext = null;
                    }
                    else
                    {
                        if (logits.Length != model.Config.VocabSize || logits.Any(v => !float.IsFinite(v)))
                            throw new InvalidDataException("Model returned invalid or nonfinite logits.");
                        next = 0;
                        for (int t = 1; t < logits.Length; ++t) if (logits[t] > logits[next]) next = t;
                    }
                    argmaxIds.Add(next);
                    if (captureDirectory != null)
                    {
                        string rowPath = Path.Combine(captureDirectory, $"case-{results.Count - 1:D4}-row-{i:D4}.f32");
                        if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("Logit captures require little endian.");
                        byte[] raw = MemoryMarshal.AsBytes(logits.AsSpan()).ToArray();
                        File.WriteAllBytes(rowPath, raw);
                        captures.Add(new { path = rowPath, position = CacheHead(), vocab = logits.Length,
                            sha256 = Convert.ToHexStringLower(SHA256.HashData(raw)), argmax = next,
                            forced_id = forced == null ? (int?)null : forced[i] });
                    }
                    if (forced != null) next = forced[i];
                    generated.Add(next); tokenTimes.Add(timer.Elapsed.TotalSeconds);
                    if (model.Tokenizer.IsEos(next)) { finish = "stop"; break; }
                    if (i + 1 < maxNew)
                    {
                        if (greedyTransport)
                        {
                            if (model is not IGreedyForwardModel compact
                                || !compact.TryForwardGreedyToken(next, out int predicted))
                                throw new InvalidOperationException("The requested compact greedy forward was declined; no full-logit timing fallback is allowed in this campaign.");
                            if ((uint)predicted >= (uint)model.Config.VocabSize)
                                throw new InvalidDataException("Compact greedy reply is outside the model vocabulary.");
                            compactNext = predicted;
                            ++compactPredictions;
                        }
                        else logits = model.ForwardRefill([next]);
                        if (CacheHead() != ids.Length + generated.Count)
                            throw new InvalidDataException("Decode advanced to the wrong cache head.");
                    }
                }
                double elapsed = timer.Elapsed.TotalSeconds;
                double span = tokenTimes[^1] - tokenTimes[0];
                result["finish_reason"] = finish;
                result["ttft_seconds"] = tokenTimes[0];
                result["elapsed_seconds"] = elapsed;
                result["head_expert_cache_after"] = ExpertCacheCounters.Read();
                result["decode_tokens_per_second"] = generated.Count > 1 && span > 0
                    ? (double?)(generated.Count - 1) / span : null;
                result["cache_after_generation"] = CacheHead();
                if (CacheHead() != ids.Length + generated.Count - 1)
                    throw new InvalidDataException("Generation ended at the wrong cache head.");
                result["status"] = "completed";
            }
            catch (Exception error)
            {
                result["status"] = "failed";
                result["error"] = error.ToString();
                if (timer != null) result["elapsed_seconds"] = timer.Elapsed.TotalSeconds;
                throw;
            }
            finally
            {
                result["generated_tokens"] = generated.Count;
                result["compact_greedy_predictions"] = compactPredictions;
                try { result["output"] = model.Tokenizer.Decode(generated); }
                catch (Exception error)
                {
                    result["status"] = "failed";
                    result["output_decode_error"] = error.ToString();
                    throw;
                }
                finally { Save(); }
            }
            Console.WriteLine(JsonSerializer.Serialize(result));
        }
    }
    report["status"] = "completed";
}
catch (Exception error)
{
    report["status"] = "failed"; report["error"] = error.ToString();
    Environment.ExitCode = 1;
    Console.Error.WriteLine(error);
}
finally { report["finished_utc"] = DateTimeOffset.UtcNow; Save(); }

internal static class ExpertCacheCounters
{
    [DllImport("GgmlOps", CallingConvention = CallingConvention.Cdecl)]
    private static extern int TSGgml_HostMoeExpertCacheStats(out long reserved, out long budget,
        out long hits, out long misses, out long calls);

    [DllImport("GgmlOps", CallingConvention = CallingConvention.Cdecl)]
    private static extern int TSGgml_HostMoeExpertHostCacheStats(out long reserved, out long budget,
        out long hits, out long sourceReads, out long evictions);

    private static object? ReadHost()
    {
        try
        {
            return TSGgml_HostMoeExpertHostCacheStats(out long reserved, out long budget,
                out long hits, out long sourceReads, out long evictions) != 0
                ? new { reserved_bytes = reserved, budget_bytes = budget, hits, source_reads = sourceReads, evictions } : null;
        }
        catch (Exception error) when (error is DllNotFoundException or EntryPointNotFoundException)
        {
            return null;
        }
    }

    internal static object? Read()
    {
        try
        {
            return TSGgml_HostMoeExpertCacheStats(out long reserved, out long budget,
                out long hits, out long misses, out long calls) != 0
                ? new { reserved_bytes = reserved, budget_bytes = budget, hits, misses, calls, host = ReadHost() } : null;
        }
        catch (Exception error) when (error is DllNotFoundException or EntryPointNotFoundException)
        {
            return null;
        }
    }
}
