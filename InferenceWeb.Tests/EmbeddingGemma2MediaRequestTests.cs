using TensorSharp.Server.ProtocolAdapters;

namespace InferenceWeb.Tests;

public sealed class EmbeddingGemma2MediaRequestTests
{
    [Theory]
    [InlineData("https://example.com/image.png", "image", "base64 data URIs")]
    [InlineData("data:image/png,abc", "image", "base64 data URIs")]
    [InlineData("data:base64,AA==", "image", "base64 data URIs")]
    [InlineData("data:;base64,AA==", "image", "Unsupported image media type")]
    [InlineData("data:text/plain;base64,aGVsbG8=", "image", "Unsupported image media type")]
    [InlineData("data:image/png;base64,aGVsbG8=", "audio", "Unsupported audio media type")]
    [InlineData("data:image/png;base64,", "image", "must not be empty")]
    public void InvalidMediaDataUrisAreRejectedBeforeCreatingFiles(string uri, string modality, string message)
    {
        string directory = Path.Combine(Path.GetTempPath(), "embeddinggemma2-media-uri-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            var error = Assert.ThrowsAny<Exception>(() => EmbeddingAdapter.SaveDataUri(uri, directory, modality));
            Assert.Contains(message, error.Message);
            Assert.Empty(Directory.EnumerateFiles(directory));
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public void MalformedBase64IsRejectedWithoutCreatingFiles()
    {
        string directory = Path.Combine(Path.GetTempPath(), "embeddinggemma2-media-uri-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            Assert.Throws<FormatException>(() => EmbeddingAdapter.SaveDataUri("data:image/png;base64,!!!", directory, "image"));
            Assert.Empty(Directory.EnumerateFiles(directory));
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Theory]
    [InlineData("data:image/png;BASE64,AQID", "image", ".png")]
    [InlineData("data:audio/wav;base64,AQID", "audio", ".wav")]
    [InlineData("data:video/mp4;base64,AQID", "video", ".mp4")]
    public void AcceptedMediaDataUrisKeepTheirBytesAndStayInsideTheRequestDirectory(string uri, string modality, string extension)
    {
        string directory = Path.Combine(Path.GetTempPath(), "embeddinggemma2-media-uri-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            string path = EmbeddingAdapter.SaveDataUri(uri, directory, modality);
            Assert.Equal(directory, Path.GetDirectoryName(path));
            Assert.Equal(extension, Path.GetExtension(path));
            Assert.Equal(new byte[] { 1, 2, 3 }, File.ReadAllBytes(path));
            Assert.Single(Directory.EnumerateFiles(directory));
        }
        finally { Directory.Delete(directory, recursive: true); }
    }
}
