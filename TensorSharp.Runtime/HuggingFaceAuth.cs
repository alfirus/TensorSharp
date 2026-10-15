using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;

namespace TensorSharp.Runtime;

/// <summary>Uses the current host's Hugging Face login for gated model downloads.
/// Credentials are attached only to HTTPS requests to huggingface.co. The HTTP
/// redirect handler removes Authorization before following a CDN redirect.</summary>
public static class HuggingFaceAuth
{
    public static void Apply(HttpRequestMessage request)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (!IsHuggingFace(request.RequestUri) || request.Headers.Authorization != null) return;
        Apply(request, ResolveToken(Environment.GetEnvironmentVariable, ReadTokenFile));
    }

    public static bool IsHuggingFace(Uri? uri) => uri is { IsAbsoluteUri: true } &&
        uri.Scheme == Uri.UriSchemeHttps && uri.IsDefaultPort &&
        string.Equals(uri.IdnHost, "huggingface.co", StringComparison.OrdinalIgnoreCase);

    internal static void Apply(HttpRequestMessage request, string? token)
    {
        if (IsHuggingFace(request.RequestUri) && request.Headers.Authorization == null && Normalize(token) is { } valid)
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", valid);
    }

    internal static string? ResolveToken(Func<string, string?> environment, Func<string, string?> readFile)
    {
        foreach (string variable in new[] { "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN" })
            if (Normalize(environment(variable)) is { } token) return token;
        string? path = environment("HF_TOKEN_PATH");
        if (string.IsNullOrWhiteSpace(path))
        {
            string? home = environment("HF_HOME");
            if (string.IsNullOrWhiteSpace(home))
            {
                string? cache = environment("XDG_CACHE_HOME");
                if (string.IsNullOrWhiteSpace(cache))
                    cache = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache");
                home = Path.Combine(cache, "huggingface");
            }
            path = Path.Combine(home, "token");
        }
        return Normalize(readFile(path));
    }

    private static string? Normalize(string? token)
    {
        if (string.IsNullOrWhiteSpace(token)) return null;
        token = token.Trim();
        foreach (char character in token)
            if (char.IsWhiteSpace(character) || char.IsControl(character)) return null;
        return token;
    }

    private static string? ReadTokenFile(string path)
    {
        try { return File.Exists(path) ? File.ReadAllText(path) : null; }
        catch (IOException) { return null; }
        catch (UnauthorizedAccessException) { return null; }
    }
}
