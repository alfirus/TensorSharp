using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp.Models;
using TensorSharp.Models.Embeddings;
using TensorSharp.Models.Media;
using TensorSharp.Runtime;

var options = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i += 2)
{
    if (i + 1 >= args.Length || args[i] is not ("--model" or "--pair" or "--http" or "--output") || !options.TryAdd(args[i], args[i + 1]))
        throw new ArgumentException("Pass --model MODEL.gguf and exactly one of --pair/--http REPORT.json, plus --output EMPTY_DIRECTORY.");
}
bool httpSource = options.ContainsKey("--http");
if (options.Count != 3 || !options.ContainsKey("--model") || !options.ContainsKey("--output") || options.ContainsKey("--pair") == httpSource)
    throw new ArgumentException("Require --model, exactly one of --pair/--http, and --output.");
string model = Path.GetFullPath(options["--model"]), pairPath = Path.GetFullPath(options[httpSource ? "--http" : "--pair"]);
string output = Path.GetFullPath(options["--output"]);
if (Directory.Exists(output) && Directory.EnumerateFileSystemEntries(output).Any())
    throw new IOException("The proof output directory must be empty; retained evidence is never replaced.");
string Hash(string path) { using var f = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(f)).ToLowerInvariant(); }
string pairHash = Hash(pairPath);
using var pairDocument = JsonDocument.Parse(File.ReadAllText(pairPath));
var pair = pairDocument.RootElement;
if (httpSource) DefaultHttpSource.Validate(pair);
if (!pair.GetProperty("completed").GetBoolean()) throw new InvalidDataException("The paired report is incomplete.");
string modelHash = Hash(model);
if (pair.GetProperty("model").GetProperty("sha256").GetString() != modelHash)
    throw new InvalidDataException("The supplied model differs from the paired inference model.");
Assembly modelsAssembly = typeof(EmbeddingGemma2MultimodalEncoder).Assembly;
Assembly runtimeAssembly = typeof(ITokenizer).Assembly;
string modelsHash = Hash(modelsAssembly.Location), runtimeHash = Hash(runtimeAssembly.Location);
var artifacts = (httpSource ? pair.GetProperty("binary_artifacts") : pair.GetProperty("binary_artifacts").GetProperty("tensorsharp")).EnumerateArray().ToArray();
void RequirePairedAssembly(string filename, string hash)
{
    var matching = artifacts.Where(x => Path.GetFileName(x.GetProperty("path").GetString()) == filename).ToArray();
    if (matching.Length != 1 || matching[0].GetProperty("sha256").GetString() != hash)
        throw new InvalidDataException("The loaded " + filename + " is not the exact paired inference assembly.");
}
RequirePairedAssembly("TensorSharp.Models.dll", modelsHash);
RequirePairedAssembly("TensorSharp.Runtime.dll", runtimeHash);
string snapshot = typeof(Program).Assembly.GetCustomAttributes<AssemblyMetadataAttribute>()
    .Single(x => x.Key == "TensorSharpCompositeProbeSnapshot").Value!;
string rid = (OperatingSystem.IsMacOS() ? "osx" : OperatingSystem.IsLinux() ? "linux" :
    OperatingSystem.IsWindows() ? "win" : throw new PlatformNotSupportedException()) + "-" +
    RuntimeInformation.ProcessArchitecture.ToString().ToLowerInvariant();
string codecName = OperatingSystem.IsMacOS() ? "libOpenCvSharpExtern.dylib" :
    OperatingSystem.IsWindows() ? "OpenCvSharpExtern.dll" : "libOpenCvSharpExtern.so";
string codecPath = Path.Combine(snapshot, "runtimes", rid, "native", codecName);
if (!File.Exists(codecPath)) throw new FileNotFoundException("Frozen snapshot is missing its actual-platform video codec.", codecPath);
string codecHash = Hash(codecPath);
// Direct frozen assembly references have no NuGet runtimeTargets in this
// probe's deps.json. Resolve the exact frozen host codec explicitly, with its
// normal loader-relative dependencies, rather than falling back to a system
// OpenCV library. This changes no decoder or preprocessing code.
NativeLibrary.SetDllImportResolver(typeof(OpenCvSharp.VideoCapture).Assembly,
    (name, assembly, searchPath) => name == "OpenCvSharpExtern" ? NativeLibrary.Load(codecPath) : IntPtr.Zero);
if (pair.TryGetProperty("binary_artifacts_unchanged", out var unchanged) && !unchanged.GetBoolean())
    throw new InvalidDataException("The paired inference binary identity changed during its run.");

var embeddingTokenizer = modelsAssembly.GetType("TensorSharp.Models.Embeddings.EmbeddingTokenizer", throwOnError: true)!;
var createTokenizer = embeddingTokenizer.GetMethod("Create", BindingFlags.Static | BindingFlags.Public)!;
using var gguf = new GgufFile(model);
var tokenizer = (ITokenizer)createTokenizer.Invoke(null, [gguf])!;
int Lookup(string text)
{
    int id = tokenizer.LookupToken(text);
    if (id < 0) throw new InvalidDataException("Missing media token " + text);
    return id;
}
if (tokenizer.BosTokenId < 0 || tokenizer.EosTokenIds.Length != 1)
    throw new InvalidDataException("Unexpected embedding tokenizer special token configuration.");
var tokenIds = new { bos = tokenizer.BosTokenId, eos = tokenizer.EosTokenIds[0],
    image_start = Lookup("<|image>"), image_end = Lookup("<image|>"),
    audio_start = Lookup("<|audio>"), audio_end = Lookup("<audio|>") };
var defaultOptions = httpSource ? new EmbeddingGemma2MultimodalOptions() : null;
if (defaultOptions != null && (defaultOptions.VideoPrefix != "" || defaultOptions.IncludeVideoTimestamps ||
    defaultOptions.ImageSoftTokens != 280 || defaultOptions.VideoSoftTokens != 140 || defaultOptions.VideoFramesPerSecond != 1 ||
    defaultOptions.MaxVideoFrames != 32 || defaultOptions.AudioLogPolicy.ToString() != "HuggingFaceAddFloor"))
    throw new InvalidDataException("Observed frozen multimodal options differ from the expected canonical HF defaults.");
bool controlledPrefix = !httpSource && pair.GetProperty("controlled_video_prefix").GetBoolean();
string videoPrefix = defaultOptions?.VideoPrefix ?? (controlledPrefix ? "Video:" : "");
var distinctTexts = new HashSet<string>(StringComparer.Ordinal);
var textRuns = new List<object>();
var videos = new Dictionary<string, string>(StringComparer.Ordinal);
var images = new Dictionary<string, string>(StringComparer.Ordinal);
var audios = new Dictionary<string, string>(StringComparer.Ordinal);
string manifestPath = pair.GetProperty("manifest").GetProperty("path").GetString()!;
string? manifestHash = null;
if (httpSource)
{
    manifestHash = Hash(manifestPath);
    if (manifestHash != pair.GetProperty("manifest").GetProperty("sha256").GetString())
        throw new InvalidDataException("The HTTP manifest changed since inference.");
    using var manifestDocument = JsonDocument.Parse(File.ReadAllText(manifestPath));
    DefaultHttpSource.MatchManifest(pair, manifestDocument.RootElement);
}
foreach (var input in pair.GetProperty(httpSource ? "cases" : "inputs").EnumerateArray())
{
    var run = new StringBuilder();
    var runs = new List<object>();
    void Flush()
    {
        if (run.Length == 0) return;
        string text = run.ToString(); distinctTexts.Add(text);
        runs.Add(new { text, ids = tokenizer.Encode(text, addSpecial: false).ToArray() }); run.Clear();
    }
    foreach (var part in input.GetProperty("parts").EnumerateArray())
    {
        string kind = part.GetProperty("kind").GetString()!, value = part.GetProperty("value").GetString()!;
        if (kind == "text") { distinctTexts.Add(value); run.Append(value); continue; }
        // media-pair.py inserts this text part into the candidate HTTP payload.
        // Adjacent text parts are merged by the production composite encoder.
        if (kind == "video" && controlledPrefix) run.Append(videoPrefix);
        Flush();
        if (kind is not ("video" or "image" or "audio")) throw new InvalidDataException("Unexpected paired media kind.");
        string path = Path.GetFullPath(Path.IsPathRooted(value) ? value : Path.Combine(Path.GetDirectoryName(manifestPath)!, value));
        string hash = Hash(path);
        var identity = pair.GetProperty("media").EnumerateArray().Where(x => x.GetProperty("path").GetString() == path).ToArray();
        if (identity.Length != 1 || identity[0].GetProperty("sha256").GetString() != hash)
            throw new InvalidDataException("Media bytes differ from the paired HTTP input.");
        (kind == "video" ? videos : kind == "image" ? images : audios).TryAdd(path, hash);
    }
    Flush(); textRuns.Add(new { name = input.GetProperty("name").GetString(), runs });
}
distinctTexts.Add(videoPrefix);
var texts = distinctTexts.OrderBy(x => x, StringComparer.Ordinal)
    .Select(text => new { text, ids = tokenizer.Encode(text, addSpecial: false).ToArray() }).ToArray();
var encoder = typeof(EmbeddingGemma2MultimodalEncoder);
var sampleFrames = encoder.GetMethod("SampleVideoFrames", BindingFlags.NonPublic | BindingFlags.Static)!;
var frameToRgba = encoder.GetMethod("FrameToRgba", BindingFlags.NonPublic | BindingFlags.Static)!;
var utils = modelsAssembly.GetType("TensorSharp.Models.ImageProcessorUtils", throwOnError: true)!;
var resize = utils.GetMethod("ResizeRgbaToChannelFirstBicubic", BindingFlags.NonPublic | BindingFlags.Static)!;
var decodeImage = utils.GetMethod("DecodeImageToRGBA", BindingFlags.NonPublic | BindingFlags.Static)!;
var videoSize = typeof(Gemma4ImageProcessor).GetMethod("CalcAspectRatioPreservingSize", BindingFlags.NonPublic | BindingFlags.Static)!;
Directory.CreateDirectory(output);
var imageProcessor = new Gemma4ImageProcessor(minTokens: 280, maxTokens: 280,
    imageMean: [0, 0, 0], imageStd: [1, 1, 1], referenceSizing: true);
var imageProofs = new List<object>(); int imageNumber = 0;
foreach (var (path, hash) in images)
{
    object?[] arguments = [File.ReadAllBytes(path), 0, 0];
    var rgba = (byte[])decodeImage.Invoke(null, arguments)!;
    int width = (int)arguments[1]!, height = (int)arguments[2]!;
    var (chw, processedWidth, processedHeight) = imageProcessor.ProcessImage(path);
    if (rgba.Length != checked(width * height * 4) || chw.Length != checked(processedWidth * processedHeight * 3) || chw.Any(x => !float.IsFinite(x)))
        throw new InvalidDataException("Invalid production image preprocessing output.");
    string stem = $"image-{imageNumber:000}", rgbaName = stem + ".rgba8", pixelsName = stem + ".f32";
    File.WriteAllBytes(Path.Combine(output, rgbaName), rgba);
    using (var stream = File.Create(Path.Combine(output, pixelsName)))
    using (var writer = new BinaryWriter(stream)) foreach (float value in chw) writer.Write(value);
    if (Hash(path) != hash) throw new InvalidDataException("Image changed while decoding.");
    imageProofs.Add(new { path, sha256 = hash, source_width = width, source_height = height,
        processed_width = processedWidth, processed_height = processedHeight,
        soft_tokens = imageProcessor.ComputeOutputTokens(processedWidth, processedHeight),
        rgba = rgbaName, rgba_sha256 = Hash(Path.Combine(output, rgbaName)), pixels = pixelsName, pixels_sha256 = Hash(Path.Combine(output, pixelsName)),
        pixels_shape = new[] { 3, processedHeight, processedWidth }, pixels_dtype = "little-endian float32" });
    ++imageNumber;
}
var videoProofs = new List<object>(); int videoNumber = 0;
foreach (var (path, hash) in videos)
{
    var info = MediaCodecs.Video.Probe(path);
    var indices = (int[])sampleFrames.Invoke(null, [info, 1.0, 32])!;
    var frames = new List<object>(); int received = 0;
    MediaCodecs.Video.ReadFrames(path, indices, (index, pixels, width, height, stride, layout) =>
    {
        if (received >= indices.Length || index != indices[received]) throw new InvalidDataException("Decoder returned frames out of order.");
        object?[] sizeArguments = [width, height, 16, 3, 140, 0, 0];
        videoSize.Invoke(null, sizeArguments);
        int processedWidth = (int)sizeArguments[5]!, processedHeight = (int)sizeArguments[6]!;
        var rgba = (byte[])frameToRgba.Invoke(null, [pixels, width, height, stride, layout])!;
        var chw = (float[])resize.Invoke(null, [rgba, width, height, processedWidth, processedHeight,
            new float[] { 0, 0, 0 }, new float[] { 1, 1, 1 }])!;
        if (rgba.Length != checked(width * height * 4) || chw.Length != checked(processedWidth * processedHeight * 3) || chw.Any(x => !float.IsFinite(x)))
            throw new InvalidDataException("Invalid production frame preprocessing output.");
        string stem = $"video-{videoNumber:000}-frame-{received:000}";
        string rgbaName = stem + ".rgba8", pixelsName = stem + ".f32";
        File.WriteAllBytes(Path.Combine(output, rgbaName), rgba);
        using (var stream = File.Create(Path.Combine(output, pixelsName)))
        using (var writer = new BinaryWriter(stream)) foreach (float value in chw) writer.Write(value);
        frames.Add(new { index, source_width = width, source_height = height, decoded_stride = stride, decoded_layout = layout.ToString(),
            processed_width = processedWidth, processed_height = processedHeight, soft_tokens = checked(processedWidth / 48 * (processedHeight / 48)),
            rgba = rgbaName, rgba_sha256 = Hash(Path.Combine(output, rgbaName)), pixels = pixelsName, pixels_sha256 = Hash(Path.Combine(output, pixelsName)),
            pixels_shape = new[] { 3, processedHeight, processedWidth }, pixels_dtype = "little-endian float32" });
        ++received;
    });
    if (received != indices.Length) throw new InvalidDataException("Decoder did not return every selected frame.");
    if (Hash(path) != hash) throw new InvalidDataException("Video changed while decoding.");
    videoProofs.Add(new { path, sha256 = hash, fps = info.Fps, frame_count = info.FrameCount,
        source_width = info.Width, source_height = info.Height, sampled_indices = indices, frames });
    ++videoNumber;
}
var codecAssemblies = AppDomain.CurrentDomain.GetAssemblies().Where(a => !a.IsDynamic &&
    (a.GetName().Name?.Contains("OpenCv", StringComparison.OrdinalIgnoreCase) == true ||
     a.GetName().Name?.Contains("Magick", StringComparison.OrdinalIgnoreCase) == true))
    .Select(a => new { name = a.FullName, path = a.Location, sha256 = Hash(a.Location) }).ToArray();
var nativeCodecs = NativeCodecModules.Paths().Distinct(StringComparer.Ordinal).Where(File.Exists)
    .Where(p => new[] { "opencv", "avcodec", "avformat", "avutil", "swscale", "ffmpeg" }
        .Any(n => Path.GetFileName(p).Contains(n, StringComparison.OrdinalIgnoreCase)))
    .OrderBy(p => p, StringComparer.Ordinal).Select(p => new { path = p, sha256 = Hash(p), bytes = new FileInfo(p).Length }).ToArray();
if (Hash(pairPath) != pairHash || Hash(model) != modelHash || (httpSource && Hash(manifestPath) != manifestHash))
    throw new InvalidDataException("Inference report, model or HTTP manifest changed during proof execution.");
var report = new
{
    completed = true, model, model_sha256 = modelHash, models_assembly = modelsAssembly.Location, models_assembly_sha256 = modelsHash,
    runtime_assembly = runtimeAssembly.Location, runtime_assembly_sha256 = runtimeHash,
    tokenizer_type = tokenizer.GetType().FullName, token_ids = tokenIds, texts, case_text_runs = textRuns,
    controlled_video_prefix = controlledPrefix, video_prefix = videoPrefix,
    video_prefix_source = "Explicit candidate HTTP text insertion from the saved controlled pair",
    include_video_timestamps = false, video_soft_tokens = 140, video_frames_per_second = 1.0, max_video_frames = 32, videos = videoProofs,
    image_soft_tokens = 280, image_reference_sizing = true, images = imageProofs,
    audios = audios.Select(x => new { path = x.Key, sha256 = x.Value, bytes = new FileInfo(x.Key).Length }).ToArray(),
    paired_report_path = pairPath, paired_report_sha256 = pairHash,
    video_decoder = MediaCodecs.Video.GetType().FullName, codec_assemblies = codecAssemblies, loaded_native_codecs = nativeCodecs,
    frozen_snapshot = snapshot, video_codec_path = codecPath, video_codec_sha256 = codecHash,
    probe_assembly = typeof(Program).Assembly.Location, probe_assembly_sha256 = Hash(typeof(Program).Assembly.Location),
    dotnet_runtime = Environment.Version.ToString(), os = RuntimeInformation.OSDescription, architecture = RuntimeInformation.ProcessArchitecture.ToString(),
    created_utc = DateTimeOffset.UtcNow,
    execution_scope = "Production frozen tokenizer, image/video decoder, sampling, RGBA conversion and bicubic CHW preprocessing only. Audio files identified by SHA256. No learned model or backend inference.",
    limitations = "Observed production input identity, not an independent tokenizer/decoder/math oracle. Paired Models/Runtime identities are checked; native execution validates only this process OS and architecture. Frame artifact paths are relative to this proof directory."
};
string serialized;
if (!httpSource) serialized = JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true });
else
{
    var httpReport = JsonSerializer.SerializeToNode(report)!.AsObject();
    httpReport.Remove("paired_report_path"); httpReport.Remove("paired_report_sha256");
    httpReport["source_report_kind"] = "default_hf_http";
    httpReport["source_report_path"] = pairPath; httpReport["source_report_sha256"] = pairHash;
    httpReport["manifest_path"] = manifestPath; httpReport["manifest_sha256"] = manifestHash;
    httpReport["audio_log_policy"] = defaultOptions!.AudioLogPolicy.ToString();
    httpReport["video_prefix_source"] = "Observed frozen EmbeddingGemma2MultimodalOptions defaults; no controlled HTTP insertion";
    httpReport["default_options_observed"] = JsonSerializer.SerializeToNode(new
    {
        video_prefix = defaultOptions.VideoPrefix, include_video_timestamps = defaultOptions.IncludeVideoTimestamps,
        image_soft_tokens = defaultOptions.ImageSoftTokens, video_soft_tokens = defaultOptions.VideoSoftTokens,
        video_frames_per_second = defaultOptions.VideoFramesPerSecond, max_video_frames = defaultOptions.MaxVideoFrames,
        audio_log_policy = defaultOptions.AudioLogPolicy.ToString(), type = defaultOptions.GetType().FullName,
    });
    httpReport["limitations"] = "Observed default-HF production input identity, not an independent tokenizer/decoder/math oracle. HTTP Models/Runtime and model/media/report/manifest identities are checked; native codec observation qualifies only this process OS and architecture. Frame paths are relative to this proof directory.";
    serialized = httpReport.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
}
File.WriteAllText(Path.Combine(output, "composite-input-proof.json"), serialized + "\n");
Console.WriteLine($"Completed {videoProofs.Count} videos and {texts.Length} distinct text fragments from exact {(httpSource ? "default-HF HTTP" : "paired")} assemblies.");

static class NativeCodecModules
{
    [DllImport("/usr/lib/libSystem.B.dylib", EntryPoint = "_dyld_image_count")]
    private static extern uint ImageCount();
    [DllImport("/usr/lib/libSystem.B.dylib", EntryPoint = "_dyld_get_image_name")]
    private static extern IntPtr ImageName(uint index);
    public static IEnumerable<string> Paths()
    {
        if (OperatingSystem.IsMacOS())
        {
            for (uint i = 0, n = ImageCount(); i < n; ++i)
                if (Marshal.PtrToStringUTF8(ImageName(i)) is string name) yield return name;
        }
        else
        {
            using var process = Process.GetCurrentProcess();
            foreach (ProcessModule module in process.Modules) yield return module.FileName;
        }
    }
}
