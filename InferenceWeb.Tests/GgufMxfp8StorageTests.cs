// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// Published MXFP8 headers are readable without teaching unchanged ggml a new
/// type. Layout: one E8M0 scale byte and 32 E4M3 values per 32-value block.
/// This fixture verifies storage metadata/bounds only, not numerical decoding.
/// </summary>
public sealed class GgufMxfp8StorageTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-mxfp8-storage-" + Guid.NewGuid().ToString("N"));

    public GgufMxfp8StorageTests() => Directory.CreateDirectory(_directory);
    public void Dispose() => Directory.Delete(_directory, recursive: true);

    [Theory]
    [InlineData(43)]
    [InlineData(67)]
    public void PublishedAndLegacyIdsHaveIdenticalBlockLayoutAndExactRawReads(int type)
    {
        string path = Path.Combine(_directory, "table.gguf");
        byte[] payload = Enumerable.Range(0, 132).Select(i => (byte)(i * 17 + 3)).ToArray();
        SyntheticGguf.Write(path, [],
        [
            new SyntheticGguf.Tensor { Name = "per_layer_token_embd.weight", Dims = [64, 2],
                Type = (SyntheticGguf.GgmlType)type, RawBlocks = payload },
            new SyntheticGguf.Tensor { Name = "neighbor.weight", Dims = [1], Data = [17.5f] },
        ]);
        using var file = new GgufFile(path);
        var table = file.Tensors["per_layer_token_embd.weight"];
        Assert.Equal((GgmlTensorType)type, table.Type);
        Assert.Equal(32, GgufFile.GetBlockSize(table.Type));
        Assert.Equal(33, GgufFile.GetTypeSize(table.Type));
        Assert.Equal(payload.Length, file.GetTensorByteCount(table));
        Assert.Equal(payload, file.ReadTensorData(table));
        Assert.Equal(BitConverter.GetBytes(17.5f), file.ReadTensorData(file.Tensors["neighbor.weight"]));
        Assert.Equal(file.DataOffset + (long)file.Tensors["neighbor.weight"].Offset + 4,
            file.GetRequiredLength(out string? lastTensor));
        Assert.Equal("neighbor.weight", lastTensor);
        file.ThrowIfTruncated();
    }

    [Theory]
    [InlineData(43)]
    [InlineData(67)]
    public void TruncatedTableIsIncludedInRequiredLengthAndShortReadChecks(int type)
    {
        string path = Path.Combine(_directory, "truncated.gguf");
        SyntheticGguf.Write(path, [], [new SyntheticGguf.Tensor { Name = "table", Dims = [32, 2],
            Type = (SyntheticGguf.GgmlType)type, RawBlocks = new byte[66] }]);
        long required;
        using (var complete = new GgufFile(path))
        {
            required = complete.GetRequiredLength(out string? last);
            Assert.Equal(complete.DataOffset + 66, required);
            Assert.Equal("table", last);
        }
        using (var stream = new FileStream(path, FileMode.Open, FileAccess.Write)) stream.SetLength(required - 1);
        using var file = new GgufFile(path);
        Assert.Throws<IOException>(() => file.ThrowIfTruncated());
        Assert.Throws<EndOfStreamException>(() => file.ReadTensorData(file.Tensors["table"]));
    }

    [Theory]
    [InlineData(43)]
    [InlineData(67)]
    public void LargeTablesUse64BitByteCountsWithoutAllocatingPayload(int type)
    {
        using var file = EmptyFile();
        var table = new GgufTensorInfo { Name = "large-table", Type = (GgmlTensorType)type,
            Shape = [2048, 100_000_000] };
        Assert.Equal(211_200_000_000L, file.GetTensorByteCount(table));
        table.Shape = [64, 3, 2];
        Assert.Equal(396L, file.GetTensorByteCount(table));
    }

    [Theory]
    [InlineData(43)]
    [InlineData(67)]
    public void PartialRowsZeroDimensionsAndOverflowCannotSilentlyUnderestimateStorage(int type)
    {
        using var file = EmptyFile();
        ulong[][] invalid = [[], [0], [31], [33, 32], [32, 0], [ulong.MaxValue], [32, (ulong)long.MaxValue]];
        foreach (ulong[] shape in invalid)
        {
            var table = new GgufTensorInfo { Name = "bad-table", Type = (GgmlTensorType)type, Shape = shape };
            Assert.Throws<InvalidDataException>(() => file.GetTensorByteCount(table));
        }
    }

    [Theory]
    [InlineData(43)]
    [InlineData(67)]
    public void StorageRecognitionDoesNotAdvertiseManagedOrNativeGenericDequantization(int type)
    {
        Assert.False(ManagedQuantizedOps.SupportsDequantization((GgmlTensorType)type));
        Assert.False(ManagedQuantizedOps.SupportsCpuQuantizedStorage((GgmlTensorType)type));
        Assert.Throws<NotSupportedException>(() => ManagedQuantizedOps.RowSize(type, 32));
        Assert.Contains("Engram loader", Assert.Throws<NotSupportedException>(() => NativeDequant.RowSize(type, 32)).Message);
        Assert.Throws<NotSupportedException>(() => NativeDequant.DequantizeToFloat32(type, new byte[33], 0, new float[32], 0, 32));
        Assert.Throws<NotSupportedException>(() => NativeDequant.DequantizeToFloat32(type, IntPtr.Zero, new float[32], 0, 32));
        Assert.Throws<NotSupportedException>(() => NativeDequant.DequantizeToFloat32Native(type, IntPtr.Zero, IntPtr.Zero, 32));
    }

    private GgufFile EmptyFile()
    {
        string path = Path.Combine(_directory, Guid.NewGuid().ToString("N") + ".gguf");
        SyntheticGguf.Write(path, [], []);
        return new GgufFile(path);
    }
}
