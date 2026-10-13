using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using UglyToad.PdfPig;

namespace InferenceWeb.Tests;

public sealed class KnowledgePdfFixtureTests
{
    [Fact]
    public void PdfPigConfirmsScanPagesHaveNoTextLayerAndRetainAllRasterBlocks()
    {
        Assembly assembly = typeof(KnowledgePdfFixtureTests).Assembly;
        const string prefix = "InferenceWeb.Tests.Fixtures.Knowledge.";
        using Stream manifestStream = assembly.GetManifestResourceStream(prefix + "manifest.json")!;
        using JsonDocument manifest = JsonDocument.Parse(manifestStream);
        foreach (JsonElement fixture in manifest.RootElement.GetProperty("files").EnumerateArray())
        {
            string name = fixture.GetProperty("fileName").GetString()!;
            using Stream input = assembly.GetManifestResourceStream(prefix + name)!;
            using var bytes = new MemoryStream();
            input.CopyTo(bytes);
            byte[] data = bytes.ToArray();
            Assert.Equal(fixture.GetProperty("sha256").GetString(), Convert.ToHexString(SHA256.HashData(data)).ToLowerInvariant());
            using PdfDocument document = PdfDocument.Open(data);
            JsonElement[] pages = fixture.GetProperty("pages").EnumerateArray().ToArray();
            Assert.Equal(pages.Length, document.NumberOfPages);
            foreach (JsonElement expected in pages)
            {
                var page = document.GetPage(expected.GetProperty("page").GetInt32());
                if (expected.GetProperty("selectableText").GetBoolean())
                    Assert.Contains(expected.TryGetProperty("selectableExpected", out var selectable)
                        ? selectable.GetString() : expected.GetProperty("expectedCode").GetString(), page.Text);
                else
                    Assert.True(string.IsNullOrWhiteSpace(page.Text), $"{name} page {page.Number} must require OCR; found text: {page.Text}");
                Assert.Equal(expected.GetProperty("imageCount").GetInt32(), page.GetImages().Count());
            }
        }
    }
}
