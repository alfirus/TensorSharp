using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Ltx25;

/// <summary>The embedded Gemma4 Unified BPE and LTX's BOS/truncation policy.</summary>
public sealed class Ltx25PromptTokenizer
{
    private readonly BpeTokenizer _tokenizer;
    public int MaximumLength { get; }
    public Ltx25PromptTokenizer(ReadOnlySpan<byte> tokenizerJson, int maximumLength = 1024)
    {
        if (maximumLength is < 1 or > 1024) throw new ArgumentOutOfRangeException(nameof(maximumLength));
        MaximumLength = maximumLength;
        using var doc = JsonDocument.Parse(tokenizerJson.ToArray());
        var root = doc.RootElement; var model = root.GetProperty("model");
        if (model.GetProperty("type").GetString() != "BPE" ||
            root.GetProperty("normalizer").GetProperty("content").GetString() != "▁")
            throw new NotSupportedException("LTX requires the authentic Gemma4 Unified space-escaped BPE tokenizer.");
        var entries = model.GetProperty("vocab").EnumerateObject().ToArray();
        int length = entries.Max(p => p.Value.GetInt32()) + 1;
        var vocab = new string[length]; var types = Enumerable.Repeat(1, length).ToArray();
        foreach (var entry in entries) vocab[entry.Value.GetInt32()] = entry.Name;
        if (vocab.Any(v => v == null)) throw new InvalidDataException("Tokenizer vocabulary is not contiguous.");
        foreach (var entry in root.GetProperty("added_tokens").EnumerateArray())
        {
            int id = entry.GetProperty("id").GetInt32();
            if ((uint)id >= vocab.Length || vocab[id] != entry.GetProperty("content").GetString())
                throw new InvalidDataException("Embedded tokenizer added-token identity differs from its vocabulary.");
            if (entry.GetProperty("special").GetBoolean()) types[id] = 3;
        }
        var merges = model.GetProperty("merges").EnumerateArray().Select(m =>
            m.ValueKind == JsonValueKind.Array ? m[0].GetString() + " " + m[1].GetString() : m.GetString()).ToArray();
        if (vocab.Length <= 3 || vocab[0] != "<pad>" || vocab[1] != "<eos>" || vocab[2] != "<bos>")
            throw new InvalidDataException("Embedded Gemma special-token IDs differ from the LTX contract.");
        _tokenizer = new(vocab, types, merges, 2, [1], false, false, "gemma4");
    }

    /// <summary>Valid token IDs only; padding is represented by connector registers later.</summary>
    public int[] Encode(string prompt)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        var ids = _tokenizer.Encode(prompt.Trim(), false);
        if (ids.Count == 0 || ids[0] != 2) ids.Insert(0, 2);
        if (ids.Count > MaximumLength) ids.RemoveRange(MaximumLength, ids.Count - MaximumLength);
        return ids.ToArray();
    }
}
