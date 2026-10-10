// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using InferenceWeb.Tests;
using TensorSharp.GGML;
using TensorSharp.Memory;

if (args.Length != 2 || args[0] is not ("CPU" or "CUDA"))
    throw new ArgumentException("Usage: GlmUploadBudgetProbe CPU|CUDA <new-output-directory>");
string backend = args[0], directory = Path.GetFullPath(args[1]);
if (Directory.Exists(directory)) throw new IOException("Use a fresh evidence directory.");
Directory.CreateDirectory(directory);
// Native CRT environment may already be initialized before Main on Windows.
// Require process-start values rather than relying on managed setenv updates.
foreach (var (name, value) in new[] { ("TS_GLM_LOAD_THREADS", "2"), ("TS_GLM_LOAD_CHUNK_MB", "1"),
    ("TS_GLM_MOE_MMAP", "0"), ("TS_CPU_MOE_THREADS", "2") })
    if (Environment.GetEnvironmentVariable(name) != value)
        throw new ArgumentException($"Set {name}={value} before launching the probe.");
string model = GlmDsaSyntheticModelBuilder.Write(Path.Combine(directory, "tiny-glm.gguf"));
int checks = 0;
void Require(bool value, string message) { ++checks; if (!value) throw new InvalidOperationException(message); }
IntPtr Load() => GgmlGlmNative.LoadModel(model, backend == "CUDA" ? 1 : 0, 256, 16, 2,
    nCpuMoe: backend == "CUDA" ? 0 : 5, backendName: backend, tp: 1, ctxIsHardLimit: true);
float[] Evaluate(IntPtr handle) {
    Require(handle != IntPtr.Zero, "Successful load returned null.");
    int[] prompt = Enumerable.Range(0, 37).Select(i => 3 + i * 11 % 90).ToArray();
    var all = new List<float>();
    foreach (int[] tokens in new[] { prompt, new[] {17}, new[] {31}, new[] {43} }) {
        float[] logits = new float[GgmlGlmNative.VocabSize(handle)];
        Require(GgmlGlmNative.Forward(handle, tokens, logits), "Forward failed.");
        Require(logits.All(float.IsFinite), "Nonfinite logits.");
        all.AddRange(logits);
    }
    return all.ToArray();
}
float[] reference;
IntPtr first = Load();
try { reference = Evaluate(first); }
finally { if (first != IntPtr.Zero) GgmlGlmNative.Free(first); }
var budget = new MemoryBudget([new("host", 1), new("device", 1L << 30)]);
long refusals;
float[] actual;
using (var scope = new GgmlCacheBudgetScope(budget, [["device"]], false, ["host"])) {
    IntPtr refused = Load();
    try { Require(refused == IntPtr.Zero, "One-byte host quota failed to refuse native upload."); }
    finally { if (refused != IntPtr.Zero) GgmlGlmNative.Free(refused); }
    refusals = scope.AllocationRefusals.Count;
    Require(refusals > 0 && scope.CallbackError == null, "Expected ordinary pressure without callback error.");
    Require(scope.ActiveAllocations == 0 && budget.Snapshot().All(x => x.Reserved + x.Committed == 0),
        "Failed load leaked staging ownership.");
    Require(budget.TrySetCapacity("host", 8L << 20), "Cannot increase quota after failed load.");
    IntPtr retry = Load();
    try { actual = Evaluate(retry); }
    finally { if (retry != IntPtr.Zero) GgmlGlmNative.Free(retry); }
    Require(scope.ActiveAllocations == 0 && budget.Snapshot().All(x => x.Reserved + x.Committed == 0),
        "Successful retry leaked staging ownership.");
    Require(scope.CallbackError == null, "Callback error after retry.");
}
Require(reference.Select(BitConverter.SingleToInt32Bits).SequenceEqual(actual.Select(BitConverter.SingleToInt32Bits)),
    "Uploader budgeting changed full logits.");
Require(budget.HighWatermarks().Single(x => x.Pool == "host").Committed > 0, "No host staging was committed.");
string Hash(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
byte[] output = new byte[actual.Length * sizeof(float)];
Buffer.BlockCopy(actual, 0, output, 0, output.Length);
File.WriteAllBytes(Path.Combine(directory, "logits.f32"), output);
var native = Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
    .Where(m => m.ModuleName.Contains("GgmlOps", StringComparison.OrdinalIgnoreCase))
    .Select(m => new { path = m.FileName, sha256 = Hash(File.ReadAllBytes(m.FileName)) }).ToArray();
Require(native.Length == 1, "Expected one loaded native module.");
File.WriteAllText(Path.Combine(directory, "report.json"), JsonSerializer.Serialize(new {
    passed = true, backend, checks, refusals, native, fixture_sha256 = Hash(File.ReadAllBytes(model)),
    full_logits_bit_exact = true, logit_elements = actual.Length, logits_sha256 = Hash(output),
    peaks = budget.HighWatermarks(), final_budget = budget.Snapshot(),
    scope = "Real native synthetic GLM loader and 37-token prefill + 3 decode steps. Two uploader workers, host scratch refusal and same-process retry. No real-model semantics, full physical cap or TP qualification."
}, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"PASS {checks} checks; {actual.Length} full-vocabulary logits bit-exact; native {native[0].sha256}");
