using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Models;

if (args.Length < 2) throw new ArgumentException("Pass output directory then one or more image paths.");
string output = Path.GetFullPath(args[0]); Directory.CreateDirectory(output);
string Hash(string path) { using var f = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(f)).ToLowerInvariant(); }
var assembly = typeof(Gemma4ImageProcessor).Assembly;
var utility = assembly.GetType("TensorSharp.Models.ImageProcessorUtils", throwOnError: true)!;
var decode = utility.GetMethod("DecodeImageToRGBA", BindingFlags.Static | BindingFlags.NonPublic)!;
var processor = new Gemma4ImageProcessor(minTokens: 280, maxTokens: 280,
    imageMean: [0, 0, 0], imageStd: [1, 1, 1], referenceSizing: true);
var cases = new List<object>();
foreach (string arg in args.Skip(1))
{
    string path = Path.GetFullPath(arg), name = Path.GetFileNameWithoutExtension(path);
    object?[] arguments = [File.ReadAllBytes(path), 0, 0];
    var rgba = (byte[])decode.Invoke(null, arguments)!;
    int width = (int)arguments[1]!, height = (int)arguments[2]!;
    var (pixels, processedWidth, processedHeight) = processor.ProcessImage(path);
    if (rgba.Length != checked(width * height * 4) || pixels.Length != checked(processedWidth * processedHeight * 3) || pixels.Any(x => !float.IsFinite(x)))
        throw new InvalidDataException("Invalid actual image processor output.");
    string rgbaPath = Path.Combine(output, name + ".rgba8"), pixelPath = Path.Combine(output, name + ".f32");
    File.WriteAllBytes(rgbaPath, rgba);
    using (var stream = File.Create(pixelPath))
    using (var writer = new BinaryWriter(stream)) foreach (float value in pixels) writer.Write(value);
    cases.Add(new { image = path, image_sha256 = Hash(path), source_width = width, source_height = height,
        processed_width = processedWidth, processed_height = processedHeight, soft_tokens = processor.ComputeOutputTokens(processedWidth, processedHeight),
        rgba = rgbaPath, rgba_sha256 = Hash(rgbaPath), pixels = pixelPath, pixels_sha256 = Hash(pixelPath) });
}
File.WriteAllText(Path.Combine(output, "production-preprocessing.json"), JsonSerializer.Serialize(new
{
    completed = true, cases, policy = "Gemma4ImageProcessor referenceSizing=true, patch16/merge3, 280 soft tokens, mean0/std1",
    models_assembly = assembly.Location, models_assembly_sha256 = Hash(assembly.Location),
    probe_sha256 = Hash(typeof(Program).Assembly.Location), runtime = Environment.Version.ToString(),
    os = RuntimeInformation.OSDescription, process_architecture = RuntimeInformation.ProcessArchitecture.ToString(),
    execution_scope = "Only published PNG decode/size/bicubic/normalization helpers; no model or inference backend is executed. A copied foreign-platform managed assembly does not validate that platform's native runtime.",
    limitations = "Actual processor execution from a frozen inference snapshot only; no model/backend execution or geometry inference from token counts.",
}, new JsonSerializerOptions { WriteIndented = true }) + "\n");
