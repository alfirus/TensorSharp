using System.Net.Http;
using System.Net.Http.Headers;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class HuggingFaceAuthTests
{
    [Theory]
    [InlineData("https://huggingface.co/owner/repo/resolve/commit/model.bin", true)]
    [InlineData("https://HUGGINGFACE.CO/owner/repo/model.bin", true)]
    [InlineData("http://huggingface.co/model.bin", false)]
    [InlineData("https://huggingface.co:8443/model.bin", false)]
    [InlineData("https://huggingface.co.example/model.bin", false)]
    [InlineData("https://cdn-lfs.huggingface.co/model.bin", false)]
    [InlineData("https://example.org/model.bin", false)]
    public void CredentialIsRestrictedToTheOfficialHttpsOrigin(string url, bool authorized)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        HuggingFaceAuth.Apply(request, "hf_test_only");
        Assert.Equal(authorized, request.Headers.Authorization != null);
        if (authorized) Assert.Equal("Bearer hf_test_only", request.Headers.Authorization!.ToString());
    }

    [Fact]
    public void ExplicitAuthorizationIsPreservedAndMalformedCredentialsAreNotAttached()
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, "https://huggingface.co/test");
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", "explicit_test");
        HuggingFaceAuth.Apply(request, "hf_test_only");
        Assert.Equal("explicit_test", request.Headers.Authorization.Parameter);
        request.Headers.Authorization = null;
        HuggingFaceAuth.Apply(request, "hf_test\r\nInjected: value");
        Assert.Null(request.Headers.Authorization);
    }

    [Fact]
    public void EnvironmentCredentialTakesPrecedenceWithoutReadingTheCache()
    {
        string ReadFile(string _) => throw new Xunit.Sdk.XunitException("Cache must not be read when HF_TOKEN is set.");
        Assert.Equal("hf_environment", HuggingFaceAuth.ResolveToken(
            key => key == "HF_TOKEN" ? " hf_environment\n" : null!, ReadFile));
    }

    [Theory]
    [InlineData("HF_TOKEN_PATH", "/test/login-token", "/test/login-token")]
    [InlineData("HF_HOME", "/test/hf", "/test/hf/token")]
    [InlineData("XDG_CACHE_HOME", "/test/cache", "/test/cache/huggingface/token")]
    public void CachedLoginUsesTheConfiguredHuggingFaceLocation(string variable, string value, string expected)
    {
        string actual = null!;
        Assert.Equal("hf_cached", HuggingFaceAuth.ResolveToken(key => key == variable ? value : null!, path =>
        { actual = path; return "hf_cached\n"; }));
        Assert.Equal(expected, actual.Replace('\\', '/'));
    }
}
