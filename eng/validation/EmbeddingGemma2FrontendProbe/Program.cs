using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Models;

var options = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i += 2)
    if (i + 1 >= args.Length || !args[i].StartsWith("--", StringComparison.Ordinal) || !options.TryAdd(args[i][2..], args[i + 1]))
        throw new ArgumentException("Pass --report <json> [--warmup 5] [--repeats 30] [--samples 1600,16000,80000,480000].");
int warmup = int.Parse(options.GetValueOrDefault("warmup", "5"));
int repeats = int.Parse(options.GetValueOrDefault("repeats", "30"));
int[] lengths = options.GetValueOrDefault("samples", "1600,16000,80000,480000").Split(',').Select(int.Parse).ToArray();
if (warmup < 0 || warmup > 10000 || repeats < 1 || repeats > 10000 || lengths.Any(value => value is < 161 or > 16000000))
    throw new ArgumentOutOfRangeException(nameof(args));
string report = Path.GetFullPath(options.GetValueOrDefault("report") ?? throw new ArgumentException("Missing --report."));
var cases = new List<object>();
foreach (int length in lengths)
{
    float[] samples = Enumerable.Range(0, length).Select(i =>
        .2f * MathF.Sin(2 * MathF.PI * 440 * i / 16000) + .1f * MathF.Cos(2 * MathF.PI * 1237 * i / 16000)).ToArray();
    foreach (Gemma4AudioLogPolicy policy in Enum.GetValues<Gemma4AudioLogPolicy>())
    {
        for (int i = 0; i < warmup; ++i) Gemma4AudioPreprocessor.ComputeMelSpectrogram(samples, policy);
        double[] timings = new double[repeats];
        float[] mel = [];
        int frames = 0;
        for (int i = 0; i < repeats; ++i)
        {
            long started = Stopwatch.GetTimestamp();
            (mel, frames) = Gemma4AudioPreprocessor.ComputeMelSpectrogram(samples, policy);
            timings[i] = Stopwatch.GetElapsedTime(started).TotalMilliseconds;
        }
        if (mel.Any(value => !float.IsFinite(value))) throw new InvalidDataException("Frontend returned non-finite mel values.");
        var sorted = timings.Order().ToArray();
        cases.Add(new { samples = length, frames, policy = policy.ToString(),
            mel_sha256 = Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(mel.AsSpan()))),
            median_ms = (sorted[(repeats - 1) / 2] + sorted[repeats / 2]) / 2, samples_ms = timings });
    }
}
Directory.CreateDirectory(Path.GetDirectoryName(report)!);
File.WriteAllText(report, JsonSerializer.Serialize(new
{
    completed = true, created_utc = DateTimeOffset.UtcNow, warmup, repeats,
    os = RuntimeInformation.OSDescription, architecture = RuntimeInformation.ProcessArchitecture.ToString(),
    processor_count = Environment.ProcessorCount, model_assembly_sha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(Gemma4AudioPreprocessor).Assembly.Location))),
    cases, limitations = "Synthetic 16 kHz waveform frontend only; excludes file decoding, audio tower, pooling, HTTP, model loading and device kernels. No end-to-end or llama.cpp parity claim.",
}, new JsonSerializerOptions { WriteIndented = true }) + "\n");
Console.WriteLine(report);
