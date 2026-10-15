using System.Text.Json;
using TensorSharp.Models;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class EmbeddingGemma2AudioResamplerTests(ITestOutputHelper output)
{
    [Fact]
    public void BandlimitedResamplingMatchesExecutedIndependentTorchaudioAcrossRatesAndBoundaries()
    {
        using var fixture = ReadFixture();
        float tolerance = fixture.RootElement.GetProperty("absolute_tolerance").GetSingle();
        foreach (var row in fixture.RootElement.GetProperty("cases").EnumerateArray())
        {
            float[] samples = row.GetProperty("samples").EnumerateArray().Select(value => value.GetSingle()).ToArray();
            float[] expected = row.GetProperty("expected").EnumerateArray().Select(value => value.GetSingle()).ToArray();
            int rate = row.GetProperty("source_rate").GetInt32();
            float[] actual = Gemma4AudioResampler.Resample(samples, rate, 16000);
            Assert.Equal(expected.Length, actual.Length);
            Assert.Equal((int)Math.Ceiling(samples.Length * 16000.0 / rate), actual.Length);
            float error = actual.Zip(expected, (a, b) => Math.Abs(a - b)).Max();
            output.WriteLine($"{row.GetProperty("name").GetString()}: maximum PCM error {error:R}");
            Assert.All(actual, value => Assert.True(float.IsFinite(value)));
            Assert.InRange(error, 0, tolerance);
        }
    }

    [Fact]
    public void DownsamplingSuppressesAliasingWhileRetainingTheSpeechPassband()
    {
        float[] Tone(int frequency) => Enumerable.Range(0, 4800).Select(i => (float)(.25 * Math.Sin(2 * Math.PI * frequency * i / 48000))).ToArray();
        double Rms(float[] samples) => Math.Sqrt(samples.Skip(400).Take(800).Average(value => (double)value * value));
        double alias = Rms(Gemma4AudioResampler.Resample(Tone(11000), 48000, 16000));
        double passband = Rms(Gemma4AudioResampler.Resample(Tone(1000), 48000, 16000));
        output.WriteLine($"11kHz alias RMS {alias:R}; 1kHz passband RMS {passband:R}");
        Assert.InRange(alias, 0, 2e-5);
        Assert.InRange(Math.Abs(passband - .25 / Math.Sqrt(2)), 0, 2e-5);
    }

    [Fact]
    public void Original16KhzAndEmptyInputsKeepTheirExactArrays()
    {
        float[] samples = [.25f, -.125f, float.Epsilon];
        Assert.Same(samples, Gemma4AudioResampler.Resample(samples, 16000, 16000));
        Assert.Same(Array.Empty<float>(), Gemma4AudioResampler.Resample(Array.Empty<float>(), 48000, 16000));
        Assert.Throws<ArgumentOutOfRangeException>(() => Gemma4AudioResampler.Resample(samples, 0, 16000));
        Assert.Throws<InvalidDataException>(() => Gemma4AudioResampler.Resample([float.NaN], 48000, 16000));
    }

    private static JsonDocument ReadFixture()
    {
        for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory != null; directory = directory.Parent)
            if (File.Exists(Path.Combine(directory.FullName, "TensorSharp.slnx")))
                return JsonDocument.Parse(File.ReadAllText(Path.Combine(directory.FullName, "InferenceWeb.Tests", "Fixtures", "EmbeddingGemma2", "audio-resampler.json")));
        throw new DirectoryNotFoundException("TensorSharp source tree is required for the independent resampling fixture.");
    }
}
