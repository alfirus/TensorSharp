using System.Diagnostics;
using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.MLX;
using TensorSharp.Models;

var options = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i += 2)
    if (!args[i].StartsWith("--") || i + 1 >= args.Length || !options.TryAdd(args[i][2..], args[i + 1]))
        throw new ArgumentException("Pass --projector --pixels --width --height --route --output --report as value pairs.");
string Required(string name) => options.GetValueOrDefault(name) ?? throw new ArgumentException("Missing --" + name);
string projector = Path.GetFullPath(Required("projector")), pixelsPath = Path.GetFullPath(Required("pixels"));
string outputPath = Path.GetFullPath(Required("output")), reportPath = Path.GetFullPath(Required("report"));
string route = Required("route");
if (route is not ("legacy" or "device")) throw new ArgumentException("Route must be legacy or device.");
Environment.SetEnvironmentVariable("TS_GEMMA4_VISION_SPATIAL_DEVICE", route == "device" ? "1" : "0");
if (Environment.GetEnvironmentVariable("TS_VISION_TRACE") is not (null or "" or "0"))
    throw new ArgumentException("Disable activation trace/readbacks for this synchronized stage diagnostic.");
int width = int.Parse(Required("width")), height = int.Parse(Required("height"));
int warmup = int.Parse(options.GetValueOrDefault("warmup", "1")), repeats = int.Parse(options.GetValueOrDefault("repeats", "3"));
if (width <= 0 || height <= 0 || width % 16 != 0 || height % 16 != 0 || warmup < 0 || repeats < 1 || repeats > 100 ||
    new FileInfo(pixelsPath).Length != checked(3L * width * height * sizeof(float))) throw new ArgumentException("Invalid pixels/geometry/repeats.");
byte[] pixelBytes = File.ReadAllBytes(pixelsPath); float[] pixels = MemoryMarshal.Cast<byte, float>(pixelBytes).ToArray();
if (pixels.Any(value => !float.IsFinite(value) || value < 0 || value > 1)) throw new ArgumentException("Pixels must be F32 [0,1].");
const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
var type = typeof(Gemma4VisionEncoder);
MethodInfo Method(string name) => type.GetMethod(name, flags) ?? throw new MissingMethodException(type.FullName, name);
int Field(Gemma4VisionEncoder encoder, string name) => (int)(type.GetField(name, flags)?.GetValue(encoder) ?? throw new MissingFieldException(name));
object? Invoke(MethodInfo method, Gemma4VisionEncoder encoder, params object[] arguments)
{
    try { return method.Invoke(encoder, arguments); }
    catch (TargetInvocationException error) when (error.InnerException != null)
    { ExceptionDispatchInfo.Capture(error.InnerException).Throw(); throw; }
}
string Hash(string path) { using var stream = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant(); }
var records = new List<object>();
int rows = 0, dimensions = 0;
double modelLoadMs;
using (var allocator = new MlxAllocator())
{
    var load = Stopwatch.StartNew();
    using var encoder = new Gemma4VisionEncoder(projector, allocator, inputPixelsInZeroToOneRange: true);
    modelLoadMs = load.Elapsed.TotalMilliseconds;
    int patch = Field(encoder, "_patchSize"), hiddenSize = Field(encoder, "_hiddenSize"), heads = Field(encoder, "_numHeads");
    int blocks = Field(encoder, "_blockCount"), merge = Field(encoder, "_nMerge");
    if (patch != 16 || hiddenSize != 768 || heads != 12 || blocks != 16 || merge != 3)
        throw new InvalidDataException("This diagnostic requires the verified EmbeddingGemma2 classic vision tower.");
    if (type.GetField("_useMlxDeviceOps", flags)?.GetValue(encoder) is not true)
        throw new InvalidDataException("This diagnostic requires the validated F32 MLX vision device route.");
    int patchesX = width / patch, patchesY = height / patch, patches = patchesX * patchesY, head = hiddenSize / heads;
    MethodInfo patchMethod = Method("PatchEmbed"), positionMethod = Method("AddPositionEmbedding2D"), blockMethod = Method("EncoderBlock"), poolMethod = Method("PoolAndProject");
    MlxWorker.Shared.Invoke(() =>
    {
        object cache = Invoke(Method("GetOrCreateRopeCache"), encoder, patchesX, patchesY, head)!;
        for (int repetition = 0; repetition < warmup + repeats; repetition++)
        {
            var phases = new List<object>(); Tensor? hidden = null, output = null;
            var overall = Stopwatch.StartNew();
            Tensor Stage(string name, Func<Tensor> issue)
            {
                var clock = Stopwatch.StartNew();
                Tensor result = issue(); double issueMs = clock.Elapsed.TotalMilliseconds;
                MlxGemma4VisionOps.EvaluateForProfile(result);
                phases.Add(new { name, issue_ms = issueMs, completed_wall_ms = clock.Elapsed.TotalMilliseconds,
                    timing = "synchronized-stage-wall-includes-lazy-evaluation-and-completion" });
                return result;
            }
            try
            {
                hidden = Stage("patch", () => (Tensor)Invoke(patchMethod, encoder, pixels, width, height, patchesX, patchesY)!);
                hidden = Stage("position", () => { Invoke(positionMethod, encoder, hidden!, cache, patches); return hidden!; });
                hidden = Stage("blocks", () =>
                {
                    for (int block = 0; block < blocks; block++)
                        hidden = (Tensor)Invoke(blockMethod, encoder, hidden!, block, patches, head, cache)!;
                    return hidden!;
                });
                output = Stage("pool_project", () => (Tensor)Invoke(poolMethod, encoder, hidden!, patchesX, patchesY, patches)!);
                double completeMs = overall.Elapsed.TotalMilliseconds;
                rows = (int)output.Sizes[0]; dimensions = (int)output.Sizes[1];
                float[] values = output.GetElementsAsFloat(checked(rows * dimensions));
                if (values.Any(value => !float.IsFinite(value))) throw new InvalidDataException("Nonfinite vision output.");
                if (repetition == warmup + repeats - 1)
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
                    File.WriteAllBytes(outputPath, MemoryMarshal.AsBytes(values.AsSpan()).ToArray());
                }
                records.Add(new { repetition, is_warmup = repetition < warmup, completed = true, completed_wall_ms = completeMs, phases });
            }
            finally { output?.Dispose(); hidden?.Dispose(); }
        }
    });
}
var binaries = Directory.EnumerateFiles(AppContext.BaseDirectory, "*", SearchOption.AllDirectories).Where(path =>
    (Path.GetFileName(path).StartsWith("TensorSharp", StringComparison.Ordinal) && Path.GetExtension(path) == ".dll") ||
    Path.GetExtension(path) is ".dylib" or ".so" or ".metallib" or ".ptx")
    .Select(path => new { path, sha256 = Hash(path) }).ToArray();
Directory.CreateDirectory(Path.GetDirectoryName(reportPath)!);
File.WriteAllText(reportPath, JsonSerializer.Serialize(new
{
    completed = true, backend = "MLX", route, spatial_device = Environment.GetEnvironmentVariable("TS_GEMMA4_VISION_SPATIAL_DEVICE"),
    projector, projector_sha256 = Hash(projector), pixels = pixelsPath, pixels_sha256 = Hash(pixelsPath), width, height, rows, dimensions,
    output = outputPath, output_sha256 = Hash(outputPath), warmup, repeats, model_load_ms = modelLoadMs,
    binary_artifacts = binaries, records, created_utc = DateTimeOffset.UtcNow, platform = RuntimeInformation.OSDescription,
    process_architecture = RuntimeInformation.ProcessArchitecture.ToString(),
    limitations = "Private production stages are invoked through reflection on the same MLX worker. Each stage explicitly evaluates and waits; this perturbs lazy scheduling and is a diagnostic, not formal HTTP latency or pure GPU kernel timing. " +
        "Output host copy is outside stage timing. Model load is separate; cold process does not mean cold file cache. No resize, text pooling or protocol is measured. Original F32 qmm/SDPA math remains in use.",
}, new JsonSerializerOptions { WriteIndented = true }) + "\n");
Console.WriteLine(JsonSerializer.Serialize(new { completed = true, route, rows, dimensions, report = reportPath }));
