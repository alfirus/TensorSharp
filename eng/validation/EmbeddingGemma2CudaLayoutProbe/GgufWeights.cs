using System.Text;

/// <summary>Read-only bounded GGUF tensor reader for this standalone probe.
/// It reads only requested F32/BF16/Q8_0 matrices, without importing model code.</summary>
internal sealed class GgufWeights : IDisposable
{
    private sealed record Info(long[] Shape, int Type, long Offset);
    private readonly FileStream _file;
    private readonly BinaryReader _reader;
    private readonly Dictionary<string, Info> _tensors = new();
    private readonly long _dataStart;

    internal GgufWeights(string path)
    {
        _file = File.OpenRead(path); _reader = new(_file, Encoding.UTF8, leaveOpen: true);
        try
        {
            if (_reader.ReadUInt32() != 0x46554747 || _reader.ReadUInt32() != 3)
                throw new InvalidDataException("Expected GGUF version3.");
            ulong tensors = _reader.ReadUInt64(), fields = _reader.ReadUInt64();
            if (tensors > 100_000 || fields > 100_000) throw new InvalidDataException("Unbounded GGUF header.");
            uint alignment = 32;
            for (ulong i = 0; i < fields; i++)
            {
                string name = Text(); uint type = _reader.ReadUInt32();
                if (name == "general.alignment" && type == 4) alignment = _reader.ReadUInt32();
                else Skip(type);
            }
            if (alignment is 0 or > 4096 || (alignment & (alignment - 1)) != 0)
                throw new InvalidDataException("Invalid GGUF alignment.");
            for (ulong i = 0; i < tensors; i++)
            {
                string name = Text(); uint dimensions = _reader.ReadUInt32();
                if (dimensions is 0 or > 4) throw new InvalidDataException("Invalid tensor dimensions.");
                var shape = new long[dimensions];
                for (int d = 0; d < shape.Length; d++) shape[d] = checked((long)_reader.ReadUInt64());
                int type = checked((int)_reader.ReadUInt32()); long offset = checked((long)_reader.ReadUInt64());
                _tensors.Add(name, new(shape, type, offset));
            }
            _dataStart = checked((_file.Position + alignment - 1) / alignment * alignment);
        }
        catch { Dispose(); throw; }
    }

    private string Text()
    {
        ulong count = _reader.ReadUInt64();
        if (count > 1 << 20) throw new InvalidDataException("Unbounded header string.");
        byte[] text = _reader.ReadBytes((int)count);
        if (text.Length != (int)count) throw new EndOfStreamException();
        return Encoding.UTF8.GetString(text);
    }
    private void Advance(long count)
    {
        long end = checked(_file.Position + count);
        if (end > _file.Length) throw new EndOfStreamException();
        _file.Position = end;
    }
    private void Skip(uint type)
    {
        if (type == 8) { Advance(checked((long)_reader.ReadUInt64())); return; }
        if (type == 9)
        {
            uint element = _reader.ReadUInt32(); ulong count = _reader.ReadUInt64();
            if (element == 9 || count > 10_000_000) throw new InvalidDataException("Unbounded metadata array.");
            if (element == 8) for (ulong i = 0; i < count; i++) Skip(element);
            else Advance(checked((long)count * Size(element)));
            return;
        }
        Advance(Size(type));
    }
    private static int Size(uint type) => type switch
    { 0 or 1 or 7 => 1, 2 or 3 => 2, 4 or 5 or 6 => 4, 10 or 11 or 12 => 8, _ => throw new InvalidDataException("Unknown metadata type.") };

    internal float[] Joined(string[] names, int inner, int columns)
    {
        var output = new float[checked(inner * columns)]; int offset = 0;
        foreach (string name in names)
        {
            if (!_tensors.TryGetValue(name, out var info) || info.Shape.Length != 2 || info.Shape[0] != inner ||
                info.Shape[1] <= 0 || info.Shape[1] > columns)
                throw new InvalidDataException($"Unexpected matrix {name}.");
            int rows = checked((int)info.Shape[1]), count = checked(inner * rows);
            if (offset + count > output.Length) throw new InvalidDataException("Joined matrix exceeds dimensions.");
            _file.Position = checked(_dataStart + info.Offset);
            if (info.Type == 8)
            {
                if (inner % 32 != 0) throw new InvalidDataException("Q8_0 rows must align to32 values.");
                for (int i = 0; i < count; i += 32)
                {
                    float scale = (float)BitConverter.UInt16BitsToHalf(_reader.ReadUInt16());
                    for (int j = 0; j < 32; j++) output[offset + i + j] = scale * _reader.ReadSByte();
                }
            }
            else if (info.Type == 30)
                for (int i = 0; i < count; i++) output[offset + i] = BitConverter.Int32BitsToSingle(_reader.ReadUInt16() << 16);
            else if (info.Type == 0)
                for (int i = 0; i < count; i++) output[offset + i] = _reader.ReadSingle();
            else throw new NotSupportedException($"Probe reader does not decode GGUF type{info.Type}.");
            offset += count;
        }
        if (offset != output.Length) throw new InvalidDataException("Incomplete joined matrix.");
        if (output.Any(value => !float.IsFinite(value))) throw new InvalidDataException("Nonfinite matrix weight.");
        return output;
    }
    public void Dispose() { _reader.Dispose(); _file.Dispose(); }
}
