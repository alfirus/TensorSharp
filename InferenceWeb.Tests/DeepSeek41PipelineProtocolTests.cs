// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Globalization;
using System.Net;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// Pure managed transport/identity gates. These tests do not load a model or
/// claim native stage equivalence, real GPU execution, or model quality.
/// </summary>
public sealed class DeepSeek41PipelineProtocolTests
{
    [Theory]
    [InlineData("127.0.0.1:8500", "127.0.0.1", 8500)]
    [InlineData("[::1]:8501", "::1", 8501)]
    [InlineData("192.0.2.1:65535", "192.0.2.1", 65535)]
    public void NumericEndpointPreservesAddressAndPort(string value, string address, int port)
    {
        IPEndPoint endpoint = DeepSeek41PipelineExecutor.ParseEndpoint(value);
        Assert.Equal(IPAddress.Parse(address), endpoint.Address);
        Assert.Equal(port, endpoint.Port);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("localhost:8500")]
    [InlineData("127.0.0.1")]
    [InlineData("127.0.0.1:0")]
    [InlineData("127.0.0.1:65536")]
    [InlineData("127.0.0.1:-1")]
    public void InvalidEndpointsAreRejectedBeforeConnecting(string? value)
        => Assert.Throws<ArgumentException>(() => DeepSeek41PipelineExecutor.ParseEndpoint(value));

    [Fact]
    public void WorkerRefusesPublicBindBeforeLoadingNativeModel()
        => Assert.Throws<ArgumentException>(() => DeepSeek41PipelineWorker.Run(
            "missing-model.gguf", "0.0.0.0:8500", "missing-manifest.json"));

    [Fact]
    public void PipelineCacheNamespaceDiffersFromAnUnsplitNativeExecutor()
    {
        // No constructors run and no native handle exists: exercise only the
        // production fingerprint's executor selector, not a model load.
        var model = (DeepSeek4Model)RuntimeHelpers.GetUninitializedObject(typeof(DeepSeek4Model));
        typeof(ModelBase).GetProperty(nameof(ModelBase.Config)).SetValue(model,
            new ModelConfig { Architecture = "deepseek41", NumLayers = 40, HiddenSize = 5120 });
        string native = model.KVStateFingerprint;
        typeof(DeepSeek4Model).GetField("_pipelineExec", BindingFlags.Instance | BindingFlags.NonPublic)
            .SetValue(model, RuntimeHelpers.GetUninitializedObject(typeof(DeepSeek41PipelineExecutor)));
        string pipeline = model.KVStateFingerprint;
        Assert.Contains("|exec=native|", native);
        Assert.Contains($"|exec=pipeline-v{DeepSeek41PipelineExecutor.Version}-cut{DeepSeek41PipelineExecutor.Cut}|", pipeline);
        Assert.NotEqual(native, pipeline);
    }

    [Fact]
    public void ManifestIdentityUsesEveryModelShardButExcludesCompanionsAndRecordOrder()
    {
        using var fixture = new CheckpointFixture();
        string first = fixture.Identity();
        Assert.Matches("^[a-f0-9]{64}$", first);
        var files = fixture.Document["files"].AsArray();
        files.Add(new JsonObject { ["path"] = "dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf",
            ["verified"] = false, ["sha256"] = "unverified companion intentionally excluded" });
        files.Add(new JsonObject { ["path"] = "mmproj-BF16.gguf", ["verified"] = false });
        fixture.Save();
        Assert.Equal(first, fixture.Identity());
        JsonNode[] reversed = files.Reverse().Select(node => node.DeepClone()).ToArray();
        fixture.Document["files"] = new JsonArray(reversed);
        fixture.Save();
        Assert.Equal(first, fixture.Identity());
        JsonNode second = fixture.ModelRecords().Single(node => node["path"].GetValue<string>().EndsWith("-00002-of-00002.gguf", StringComparison.Ordinal));
        byte[] replacement = Encoding.UTF8.GetBytes("synthetic-shard-2");
        File.WriteAllBytes(fixture.Shards[1], replacement);
        File.SetLastWriteTimeUtc(fixture.Shards[1], DateTime.UnixEpoch.AddSeconds(
            fixture.Document["finished_unix"].GetValue<double>() - 1));
        string replacementHash = Convert.ToHexStringLower(SHA256.HashData(replacement));
        second["sha256"] = replacementHash;
        second["actual_sha256"] = replacementHash;
        fixture.Save();
        Assert.NotEqual(first, fixture.Identity());
    }

    [Theory]
    [InlineData("complete")]
    [InlineData("quant")]
    [InlineData("repo")]
    [InlineData("revision")]
    [InlineData("finished_unix")]
    [InlineData("files")]
    public void IncompleteManifestIsRejected(string field)
    {
        using var fixture = new CheckpointFixture();
        fixture.Document.Remove(field);
        fixture.Save();
        Assert.Throws<InvalidDataException>(() => fixture.Identity());
    }

    [Fact]
    public void NullOrWrongTypedManifestPathsAreRejected()
    {
        using var fixture = new CheckpointFixture();
        fixture.ModelRecords().First()["path"] = null;
        fixture.Save();
        Assert.Throws<InvalidDataException>(() => fixture.Identity());
        fixture.Document["files"].AsArray()[0]["path"] = 123;
        fixture.Save();
        Assert.Throws<InvalidDataException>(() => fixture.Identity());
    }

    [Fact]
    public void UnfinishedOrUnverifiedOrWrongHashRecordsAreRejected()
    {
        using var fixture = new CheckpointFixture();
        fixture.Document["complete"] = false;
        fixture.Save();
        Assert.Throws<InvalidDataException>(() => fixture.Identity());
        fixture.Document["complete"] = true;
        fixture.ModelRecords().First()["verified"] = false;
        fixture.Save();
        Assert.Throws<InvalidDataException>(() => fixture.Identity());
        fixture.ModelRecords().First()["verified"] = true;
        fixture.ModelRecords().First()["actual_sha256"] = new string('c', 64);
        fixture.Save();
        Assert.Throws<InvalidDataException>(() => fixture.Identity());
        fixture.ModelRecords().First()["sha256"] = new string('z', 64);
        fixture.ModelRecords().First()["actual_sha256"] = new string('z', 64);
        fixture.Save();
        Assert.Throws<InvalidDataException>(() => fixture.Identity());
    }

    [Fact]
    public void MissingAndDuplicateShardRecordsCannotMasqueradeAsCompleteCheckpoint()
    {
        using var fixture = new CheckpointFixture();
        var records = fixture.Document["files"].AsArray();
        JsonNode last = records.Last().DeepClone();
        records.RemoveAt(records.Count - 1);
        fixture.Save();
        Assert.Throws<InvalidDataException>(() => fixture.Identity());
        records.Add(records.First().DeepClone());
        fixture.Save();
        Assert.Throws<InvalidDataException>(() => fixture.Identity());
        records.RemoveAt(records.Count - 1);
        records.Add(last);
        fixture.Save();
        Assert.NotEmpty(fixture.Identity());
    }

    [Fact]
    public void ManifestCannotRedirectShardToNestedPathOrDifferentModel()
    {
        using var fixture = new CheckpointFixture();
        JsonNode first = fixture.ModelRecords().First();
        string original = first["path"].GetValue<string>();
        first["path"] = CheckpointFixture.Quant + "/nested/" + Path.GetFileName(original);
        fixture.Save();
        Assert.Throws<InvalidDataException>(() => fixture.Identity());
        first["path"] = original.Replace("DeepSeek", "Different", StringComparison.Ordinal);
        fixture.Save();
        Assert.Throws<InvalidDataException>(() => fixture.Identity());
    }

    [Fact]
    public void ChangedSizeOrMissingShardIsRejected()
    {
        using var fixture = new CheckpointFixture();
        string last = fixture.Shards.Last();
        File.AppendAllText(last, "changed");
        Assert.Throws<InvalidDataException>(() => fixture.Identity());
        File.Delete(last);
        Assert.Throws<InvalidDataException>(() => fixture.Identity());
    }

    [Fact]
    public void SameSizeModificationAfterVerificationIsRejectedWithinSameSecond()
    {
        using var fixture = new CheckpointFixture();
        // Explicit fractional times exercise the previous ceiling-to-seconds
        // gap without depending on timer precision or a wall-clock race.
        fixture.Document["finished_unix"] = 1_800_000_000.125;
        foreach (string shard in fixture.Shards)
            File.SetLastWriteTimeUtc(shard, DateTime.UnixEpoch.AddSeconds(1_799_999_999));
        fixture.Save();
        Assert.NotEmpty(fixture.Identity());
        File.SetLastWriteTimeUtc(fixture.Shards.Last(), DateTime.UnixEpoch.AddSeconds(1_800_000_000.5));
        Assert.Throws<InvalidDataException>(() => fixture.Identity());
    }

    [Theory]
    [InlineData("")]
    [InlineData("not-a-revision")]
    [InlineData("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")]
    public void RevisionMustBePinnedToACommit(string revision)
    {
        using var fixture = new CheckpointFixture();
        fixture.Document["revision"] = revision;
        fixture.Save();
        Assert.Throws<InvalidDataException>(() => fixture.Identity());
    }

    [Fact]
    public void IdentityIsCultureIndependent()
    {
        using var fixture = new CheckpointFixture();
        string expected = fixture.Identity();
        CultureInfo previous = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo("ar-SA");
            Assert.Equal(expected, fixture.Identity());
        }
        finally { CultureInfo.CurrentCulture = previous; }
    }

    [Fact]
    public void HandshakeMatchesBinaryWriterFramingAndLeavesDimensionsUnread()
    {
        string hash = new('a', 64);
        using var stream = Handshake(hash);
        using var reader = new BinaryReader(stream, Encoding.UTF8, true);
        Assert.Equal(hash, DeepSeek41PipelineExecutor.ReadHandshakeIdentity(reader));
        Assert.Equal(5120, reader.ReadInt32());
    }

    [Theory]
    [InlineData(0, DeepSeek41PipelineExecutor.Version)]
    [InlineData(DeepSeek41PipelineExecutor.Magic, 0)]
    [InlineData(DeepSeek41PipelineExecutor.Magic, DeepSeek41PipelineExecutor.Version - 1)]
    [InlineData(DeepSeek41PipelineExecutor.Magic, DeepSeek41PipelineExecutor.Version + 1)]
    public void WrongMagicOrVersionIsRejected(int magic, int version)
    {
        using var stream = Handshake(new string('a', 64), magic, version);
        using var reader = new BinaryReader(stream);
        Assert.Throws<InvalidDataException>(() => DeepSeek41PipelineExecutor.ReadHandshakeIdentity(reader));
    }

    [Theory]
    [InlineData("")]
    [InlineData("short")]
    [InlineData("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaA")]
    [InlineData("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaz")]
    public void InvalidIdentityLengthOrEncodingIsRejected(string identity)
    {
        using var stream = Handshake(identity);
        using var reader = new BinaryReader(stream);
        Assert.Throws<InvalidDataException>(() => DeepSeek41PipelineExecutor.ReadHandshakeIdentity(reader));
    }

    [Fact]
    public void TruncatedHandshakeFailsWithoutReadingBeyondStream()
    {
        using var complete = Handshake(new string('a', 64));
        using var truncated = new MemoryStream(complete.ToArray().AsSpan(0, 30).ToArray());
        using var reader = new BinaryReader(truncated);
        Assert.Throws<EndOfStreamException>(() => DeepSeek41PipelineExecutor.ReadHandshakeIdentity(reader));
    }

    [Fact]
    public void ArrayTransportRoundTripsExactFloatBitsAndLittleEndianIntegersAcrossShortReads()
    {
        using var stream = new MemoryStream();
        int[] integers = { 0x01020304, -1, int.MinValue };
        float[] floats = { 0, -0.0f, 1.25f, float.NegativeInfinity, BitConverter.Int32BitsToSingle(unchecked((int)0x7fc00001)) };
        DeepSeek41PipelineExecutor.WriteArray(stream, integers);
        DeepSeek41PipelineExecutor.WriteArray(stream, floats);
        Assert.Equal(new byte[] { 4, 3, 2, 1 }, stream.ToArray().AsSpan(0, 4).ToArray());
        stream.Position = 0;
        using var fragmented = new FragmentedReadStream(stream, 3);
        int[] actualIntegers = new int[integers.Length];
        float[] actualFloats = new float[floats.Length];
        DeepSeek41PipelineExecutor.ReadArray(fragmented, actualIntegers.AsSpan());
        DeepSeek41PipelineExecutor.ReadArray(fragmented, actualFloats.AsSpan());
        Assert.Equal(integers, actualIntegers);
        Assert.Equal(floats.Select(BitConverter.SingleToInt32Bits), actualFloats.Select(BitConverter.SingleToInt32Bits));
    }

    [Fact]
    public void ShortArrayReadThrowsInsteadOfLeavingAFalseSuccessfulFrame()
    {
        using var stream = new MemoryStream(new byte[7]);
        float[] values = new float[2];
        Assert.Throws<EndOfStreamException>(() => DeepSeek41PipelineExecutor.ReadArray(stream, values.AsSpan()));
    }

    private static MemoryStream Handshake(string identity, int magic = DeepSeek41PipelineExecutor.Magic,
        int version = DeepSeek41PipelineExecutor.Version)
    {
        var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(magic); writer.Write(version); writer.Write(identity); writer.Write(5120);
        }
        stream.Position = 0;
        return stream;
    }

    private sealed class FragmentedReadStream(Stream inner, int maximum) : Stream
    {
        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override void Flush() => throw new NotSupportedException();
        public override int Read(byte[] buffer, int offset, int count) => inner.Read(buffer, offset, Math.Min(count, maximum));
        public override int Read(Span<byte> buffer) => inner.Read(buffer[..Math.Min(buffer.Length, maximum)]);
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    private sealed class CheckpointFixture : IDisposable
    {
        public const string Quant = "UD-IQ2_XXS";
        private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-pipeline-" + Guid.NewGuid().ToString("N"));
        public string Manifest { get; }
        public string[] Shards { get; }
        public JsonObject Document { get; }

        public CheckpointFixture()
        {
            Directory.CreateDirectory(_directory);
            Manifest = Path.Combine(_directory, "manifest.json");
            Shards = Enumerable.Range(1, 2).Select(index => Path.Combine(_directory,
                $"DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-{index:D5}-of-00002.gguf")).ToArray();
            var records = new JsonArray();
            for (int i = 0; i < Shards.Length; i++)
            {
                byte[] bytes = Encoding.UTF8.GetBytes("synthetic-shard-" + i);
                File.WriteAllBytes(Shards[i], bytes);
                string hash = Convert.ToHexStringLower(SHA256.HashData(bytes));
                records.Add(new JsonObject { ["path"] = Quant + "/" + Path.GetFileName(Shards[i]),
                    ["size"] = bytes.Length, ["sha256"] = hash, ["actual_sha256"] = hash, ["verified"] = true });
            }
            Document = new JsonObject { ["repo"] = "unsloth/DeepSeek-V4.1-Flash-1017-GGUF",
                ["revision"] = "60ad3651641b5c97679c807e064ab1d7a3ca1455", ["quant"] = Quant,
                ["complete"] = true, ["finished_unix"] = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0 + 1,
                ["files"] = records };
            Save();
        }

        public IEnumerable<JsonNode> ModelRecords() => Document["files"].AsArray().Where(node =>
            node["path"].GetValue<string>().StartsWith(Quant + "/", StringComparison.Ordinal));
        public void Save() => File.WriteAllText(Manifest, Document.ToJsonString());
        public string Identity() => DeepSeek41PipelineExecutor.CheckpointIdentity(Shards[0], Manifest);
        public void Dispose() => Directory.Delete(_directory, recursive: true);
    }
}
