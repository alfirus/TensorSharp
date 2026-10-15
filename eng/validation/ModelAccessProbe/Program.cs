using System.Net.Http;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Runtime;

if (args.Length != 2) throw new ArgumentException("Usage: ModelAccessProbe <config.json> <new-report.json>");
string presetPath = Path.GetFullPath(args[0]), reportPath = Path.GetFullPath(args[1]);
if (File.Exists(reportPath)) throw new IOException("Refusing to overwrite an existing access report.");
using JsonDocument preset = JsonDocument.Parse(File.ReadAllText(presetPath), new JsonDocumentOptions
{ CommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true });
using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(45) };
var checks = new List<object>();
bool allAccessible = true;
foreach (JsonProperty entry in preset.RootElement.EnumerateObject())
{
    if (entry.Value.ValueKind != JsonValueKind.Object || !entry.Value.TryGetProperty("urls", out JsonElement urls)) continue;
    string url = urls[0].GetString()!.Replace("${hf}", "https://huggingface.co", StringComparison.Ordinal);
    if (!Uri.TryCreate(url, UriKind.Absolute, out Uri? uri) || !HuggingFaceAuth.IsHuggingFace(uri) || !string.IsNullOrEmpty(uri.Query))
        throw new ArgumentException("This probe accepts only public pinned huggingface.co URLs without signed query strings.");
    using var request = new HttpRequestMessage(HttpMethod.Head, uri);
    HuggingFaceAuth.Apply(request);
    bool authenticated = request.Headers.Authorization != null;
    try
    {
        using var response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
        allAccessible &= response.IsSuccessStatusCode;
        // Do not serialize redirect locations, response request URLs, credentials,
        // headers or exception messages: CDN locations contain temporary signatures.
        checks.Add(new { option = entry.Name, public_url = url, authenticated,
            status_code = (int)response.StatusCode, accessible = response.IsSuccessStatusCode,
            content_length = response.Content.Headers.ContentLength });
    }
    catch (Exception error) when (error is HttpRequestException or TaskCanceledException)
    {
        allAccessible = false;
        checks.Add(new { option = entry.Name, public_url = url, authenticated, accessible = false,
            error_type = error.GetType().Name });
    }
}
Directory.CreateDirectory(Path.GetDirectoryName(reportPath)!);
File.WriteAllText(reportPath, JsonSerializer.Serialize(new
{
    created_utc = DateTimeOffset.UtcNow, config_sha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(presetPath))),
    scope = "Authenticated HEAD access only; no model payloads downloaded and no inference performed.",
    all_accessible = allAccessible, checks,
}, new JsonSerializerOptions { WriteIndented = true }) + "\n");
Console.WriteLine($"Asset access checks completed: {checks.Count}, all accessible: {allAccessible}");
Environment.ExitCode = allAccessible ? 0 : 2;
