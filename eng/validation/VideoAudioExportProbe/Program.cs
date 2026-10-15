using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Models.Media;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.Video;
using TensorSharp.Models.WanVideo;

if (args.Length != 2) throw new ArgumentException("Expected source generation report and new export report path.");
string source = Path.GetFullPath(args[0]), output = Path.GetFullPath(args[1]);
if (File.Exists(output) || File.Exists(output + ".mp4")) throw new IOException("Export evidence already exists.");
using var report = JsonDocument.Parse(File.ReadAllText(source));
var settings = report.RootElement.GetProperty("settings");
int width = settings.GetProperty("Width").GetInt32(), height = settings.GetProperty("Height").GetInt32(),
    count = settings.GetProperty("Frames").GetInt32(), fps = settings.GetProperty("Fps").GetInt32();
string stem = source + ".0";
float[] raw = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(stem + ".rgb.f32")).ToArray();
int frameSize = checked(width * height * 3);
if (raw.Length != checked(frameSize * count)) throw new InvalidDataException("Unexpected RGB geometry.");
var frames = Enumerable.Range(0, count).Select(i => new RgbImage(width, height, raw.AsSpan(i * frameSize, frameSize).ToArray())).ToArray();
var audio = new GeneratedVideoAudio { SampleRate = 48000,
    Channels = Enumerable.Range(0, 2).Select(i => MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(stem + $".audio-{i}.f32")).ToArray()).ToArray() };
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
var timer = Stopwatch.StartNew();
string codec = VideoIO.SaveMp4(output + ".mp4", frames, fps, audio, out bool muxed);
if (!muxed) throw new InvalidOperationException("Generated audio was not muxed.");
var probe = MediaCodecs.Video.Probe(output + ".mp4");
if (probe.Width != width || probe.Height != height || probe.FrameCount != count || Math.Abs(probe.Fps - fps) > .01)
    throw new InvalidDataException("Exported video geometry changed.");
var decode = new ProcessStartInfo(Environment.GetEnvironmentVariable("TS_FFMPEG") ?? "ffmpeg")
    { UseShellExecute = false, RedirectStandardError = true };
foreach (string value in new[] { "-i", output + ".mp4", "-map", "0:a:0", "-c:a", "pcm_f32le", "-f", "f32le", output + ".decoded-audio.f32" })
    decode.ArgumentList.Add(value);
using var process = Process.Start(decode)!;
string decoderLog = process.StandardError.ReadToEnd(); process.WaitForExit();
File.WriteAllText(output + ".decode.log", decoderLog);
float[] decoded = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(output + ".decoded-audio.f32")).ToArray();
if (process.ExitCode != 0 || !decoderLog.Contains("48000 Hz, stereo", StringComparison.Ordinal) ||
    decoded.Length < audio.SampleCount * 2 || decoded.Length > (audio.SampleCount + 2048) * 2 ||
    decoded.Any(value => !float.IsFinite(value)))
    throw new InvalidDataException("Exported stereo audio did not decode.");
string Hash(string path) { using var file = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(file)); }
File.WriteAllText(output, JsonSerializer.Serialize(new { completed = true,
    scope = "Codec-only retry of actual saved generated RGB/audio; no model inference rerun or new generation timing.",
    source_report = source, source_report_sha256 = Hash(source), codec, muxed,
    export_and_decode_ms = timer.Elapsed.TotalMilliseconds, video = probe,
    audio = new { sample_rate = 48000, channel_count = 2, decoded_samples = decoded.Length / 2, original_samples = audio.SampleCount },
    media_sha256 = Hash(output + ".mp4"), media_path = output + ".mp4" }, new JsonSerializerOptions { WriteIndented = true }) + "\n");
