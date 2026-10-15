using System.Text.Json;
using System.Buffers.Binary;
using TensorSharp.Models;
using TensorSharp.Models.Media;
using TensorSharp.Models.Video;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class EmbeddingGemma2AudioFrontendTests(ITestOutputHelper output)
{
    [Fact]
    public void AudioModelUsesPinnedHfExclusiveLeftWindowAtDistanceElevenAndTwelve()
    {
        using var fixture = JsonDocument.Parse(File.ReadAllText(Path.Combine(Path.GetDirectoryName(FixturePath)!, "audio-attention-mask.json")));
        var root = fixture.RootElement;
        foreach (var item in root.GetProperty("cases").EnumerateArray())
            Assert.Equal(item.GetProperty("expected").GetBoolean(), Gemma4AudioEncoder.IsWithinAudioAttentionHorizon(
                item.GetProperty("query").GetInt32(), item.GetProperty("key").GetInt32(),
                root.GetProperty("left_window_argument").GetInt32(), root.GetProperty("right_window_argument").GetInt32()));
        foreach (int query in new[] { 12, 24, 25, 96, 128 })
        {
            Assert.True(Gemma4AudioEncoder.IsWithinAudioAttentionHorizon(query, query, 12, 0));
            Assert.True(Gemma4AudioEncoder.IsWithinAudioAttentionHorizon(query, query - 11, 12, 0));
            Assert.False(Gemma4AudioEncoder.IsWithinAudioAttentionHorizon(query, query - 12, 12, 0));
            Assert.False(Gemma4AudioEncoder.IsWithinAudioAttentionHorizon(query, query + 1, 12, 0));
        }
        Assert.False(Gemma4AudioEncoder.IsWithinAudioAttentionHorizon(0, int.MinValue, 12, 0));
    }

    [Fact]
    public void WavDecoderValidatesChunkBoundsAndPcmFramesBeforeReadingSamples()
    {
        byte[] valid = Wave(1, 16, [0xE8, 0x03, 0x30, 0xF8]);
        Assert.Equal(new[] { 1000 / 32768f, -2000 / 32768f }, Gemma4AudioPreprocessor.DecodeWAV(valid));
        var mutations = new List<byte[]> { valid[..10], valid[..^1], Wave(1, 16, [1, 2, 3]) };
        void Mutate(Action<byte[]> action) { byte[] copy = (byte[])valid.Clone(); action(copy); mutations.Add(copy); }
        Mutate(bytes => BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(16), uint.MaxValue));
        Mutate(bytes => BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(40), uint.MaxValue));
        Mutate(bytes => BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(16), 10));
        Mutate(bytes => BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(22), 0));
        Mutate(bytes => BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(24), 0));
        Mutate(bytes => BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(32), 3));
        Mutate(bytes => BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(34), 12));
        foreach (byte[] invalid in mutations)
            Assert.Throws<InvalidDataException>(() => Gemma4AudioPreprocessor.DecodeWAV(invalid));
    }

    [Fact]
    public void WavDecoderRejectsUnsupportedCompressionAndNonFiniteFloatSamples()
    {
        Assert.Throws<NotSupportedException>(() => Gemma4AudioPreprocessor.DecodeWAV(Wave(2, 16, [1, 2, 3, 4])));
        foreach (float sample in new[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity })
        {
            byte[] data = new byte[4];
            BinaryPrimitives.WriteSingleLittleEndian(data, sample);
            Assert.Throws<InvalidDataException>(() => Gemma4AudioPreprocessor.DecodeWAV(Wave(3, 32, data)));
        }
    }

    [Fact]
    public void ShortResampledWaveformProducesOneFiniteBandlimitedSample()
    {
        byte[] wave = Wave(1, 16, [0xE8, 0x03, 0x30, 0xF8]);
        BinaryPrimitives.WriteUInt32LittleEndian(wave.AsSpan(24), 48000);
        BinaryPrimitives.WriteUInt32LittleEndian(wave.AsSpan(28), 96000);
        float sample = Assert.Single(Gemma4AudioPreprocessor.DecodeWAV(wave));
        Assert.True(float.IsFinite(sample));
        Assert.InRange(Math.Abs(sample - Gemma4AudioResampler.Resample([1000 / 32768f, -2000 / 32768f], 48000, 16000)[0]), 0, 1e-7f);
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void SpectrogramRejectsNonFiniteSamplesBeforeFrameOrFilterProcessing(float value)
    {
        // Even a short input or a sample on the Hann window's zero endpoint
        // must fail instead of hiding corrupt PCM in a skipped/zero-weight bin.
        Assert.Throws<InvalidDataException>(() => Gemma4AudioPreprocessor.ComputeMelSpectrogram([value]));
        var samples = new float[1024];
        samples[0] = value;
        Assert.Throws<InvalidDataException>(() => Gemma4AudioPreprocessor.ComputeMelSpectrogram(samples));
    }

    [Fact]
    public void PlatformAudioDecoderRejectsInvalidMetadataAndChannelBuffers()
    {
        DecodedAudio[] invalid = [
            new() { SampleRate = 16000, Channels = [] },
            new() { SampleRate = 16000, Channels = [null!] },
            new() { SampleRate = 16000, Channels = [[]] },
            new() { SampleRate = 0, Channels = [[0, 0]] },
            new() { SampleRate = 16000, Channels = [[0, 0], [0]] },
            new() { SampleRate = 16000, Channels = [[0, 0], null!] },
            // A downsample that keeps only the first value must still reject
            // a non-finite value in the discarded input portion.
            new() { SampleRate = 24000, Channels = [[0, float.NaN]] },
        ];
        IAudioDecoder original = MediaCodecs.Audio;
        try
        {
            foreach (var audio in invalid)
            {
                MediaCodecs.Audio = new AudioDecoder(audio);
                Assert.Throws<InvalidDataException>(() => Gemma4AudioPreprocessor.DecodeAudioFile("fixture.flac"));
            }
        }
        finally { MediaCodecs.Audio = original; }
    }

    [Fact]
    public void PlatformAudioDownmixDoesNotOverflowWhenAveragingFiniteChannels()
    {
        IAudioDecoder original = MediaCodecs.Audio;
        try
        {
            MediaCodecs.Audio = new AudioDecoder(new DecodedAudio
            {
                SampleRate = 16000, Channels = [[float.MaxValue, .5f], [float.MaxValue, -.25f]],
            });
            Assert.Equal(new[] { float.MaxValue, .125f }, Gemma4AudioPreprocessor.DecodeAudioFile("fixture.flac"));
        }
        finally { MediaCodecs.Audio = original; }
    }

    [Fact]
    public void SpectrogramMatchesIndependentOfficialAlgorithmAcrossFrameBoundariesAndWaveforms()
        => VerifySpectrogram(FixturePath, Gemma4AudioLogPolicy.HuggingFaceAddFloor);

    [Fact]
    public void ExplicitLlamaClampPolicyMatchesIndependentFrontendWithoutChangingHfDefault()
    {
        VerifySpectrogram(Path.Combine(Path.GetDirectoryName(FixturePath)!, "audio-frontend-llama-clamp.json"), Gemma4AudioLogPolicy.LlamaClampFloor);
        float[] samples = Enumerable.Range(0, 1024).Select(i => .2f * MathF.Sin(2 * MathF.PI * 440 * i / 16000)).ToArray();
        var original = Gemma4AudioPreprocessor.ComputeMelSpectrogram(samples);
        var explicitDefault = Gemma4AudioPreprocessor.ComputeMelSpectrogram(samples, Gemma4AudioLogPolicy.HuggingFaceAddFloor);
        var compatibility = Gemma4AudioPreprocessor.ComputeMelSpectrogram(samples, Gemma4AudioLogPolicy.LlamaClampFloor);
        Assert.Equal(original.numFrames, explicitDefault.numFrames);
        Assert.Equal(original.melData, explicitDefault.melData);
        Assert.True(original.melData.Zip(compatibility.melData, (a, b) => Math.Abs(a - b)).Max() > .001f);
        Assert.Throws<ArgumentOutOfRangeException>(() => Gemma4AudioPreprocessor.ComputeMelSpectrogram(samples, (Gemma4AudioLogPolicy)17));
    }

    private void VerifySpectrogram(string path, Gemma4AudioLogPolicy policy)
    {
        using var fixture = JsonDocument.Parse(File.ReadAllText(path));
        float tolerance = fixture.RootElement.GetProperty("absolute_tolerance").GetSingle();
        foreach (var item in fixture.RootElement.GetProperty("cases").EnumerateArray())
        {
            string name = item.GetProperty("name").GetString()!;
            float[] samples = item.GetProperty("samples").EnumerateArray().Select(value => value.GetSingle()).ToArray();
            float[] expected = item.GetProperty("expected_mel").EnumerateArray().Select(value => value.GetSingle()).ToArray();
            var (actual, frames) = Gemma4AudioPreprocessor.ComputeMelSpectrogram(samples, policy);
            Assert.Equal(item.GetProperty("expected_frames").GetInt32(), frames);
            if (frames == 0) Assert.Null(actual);
            else
            {
                Assert.Equal(frames * 128, actual.Length);
                Assert.Equal(expected.Length, actual.Length);
                float maximumError = actual.Zip(expected, (a, b) => Math.Abs(a - b)).Max();
                output.WriteLine($"{name}: {frames} frames, maximum mel error {maximumError:R}");
                Assert.All(actual, value => Assert.True(float.IsFinite(value)));
                Assert.True(maximumError <= tolerance, $"{name}: maximum error {maximumError:R}, tolerance {tolerance:R}");
            }
            Assert.Equal(item.GetProperty("expected_tokens_padded").GetInt32(), Gemma4AudioPreprocessor.ComputeAudioTokenCount(samples));
            Assert.Equal(item.GetProperty("expected_tokens_valid").GetInt32(), Gemma4AudioPreprocessor.ComputeAudioTokenCount(samples, padToMultiple: false));
        }
    }

    private static string FixturePath
    {
        get
        {
            for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory != null; directory = directory.Parent)
                if (File.Exists(Path.Combine(directory.FullName, "TensorSharp.slnx")))
                    return Path.Combine(directory.FullName, "InferenceWeb.Tests", "Fixtures", "EmbeddingGemma2", "audio-frontend.json");
            throw new DirectoryNotFoundException("TensorSharp source tree is required for the independent audio frontend fixture.");
        }
    }

    private static byte[] Wave(ushort format, ushort bits, byte[] data)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        writer.Write("RIFF"u8); writer.Write(checked((uint)(36 + data.Length + (data.Length & 1)))); writer.Write("WAVE"u8);
        writer.Write("fmt "u8); writer.Write(16U); writer.Write(format); writer.Write((ushort)1);
        writer.Write(16000U); writer.Write(checked((uint)(16000 * bits / 8))); writer.Write((ushort)(bits / 8)); writer.Write(bits);
        writer.Write("data"u8); writer.Write((uint)data.Length); writer.Write(data);
        if ((data.Length & 1) != 0) writer.Write((byte)0);
        return stream.ToArray();
    }

    private sealed class AudioDecoder(DecodedAudio audio) : IAudioDecoder
    {
        public DecodedAudio Decode(string path) => audio;
    }
}
