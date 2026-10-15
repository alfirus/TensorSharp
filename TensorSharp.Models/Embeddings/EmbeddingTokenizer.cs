// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Embeddings;

/// <summary>
/// GGUF encoder tokenizers: BERT WordPiece and SentencePiece unigram (XLM-R).
/// Unigram uses the model's compiled normalizer and global Viterbi scores;
/// greedy BPE merging is not interchangeable with this algorithm.
/// </summary>
internal sealed class EmbeddingTokenizer : ITokenizer, ISpecialTokenVocabulary
{
    private readonly bool _wordPiece, _addBos, _addEos, _lowercase, _stripAccents;
    private readonly bool _addPrefix, _removeSpaces;
    private readonly int _unknown;
    private readonly int[] _types;
    private readonly float[] _scores;
    private readonly float _unknownScore;
    private readonly Dictionary<string, int> _lookup;
    private readonly ByteTrie _pieces = new(), _specials = new(), _userDefined = new();
    private readonly uint[] _charsmap;
    private readonly byte[] _replacements;

    public string[] Vocab { get; }
    public int BosTokenId { get; }
    public int[] EosTokenIds { get; }
    public int VocabSize => Vocab.Length;
    public IReadOnlyCollection<int> SpecialTokenIds { get; }

    public static ITokenizer Create(GgufFile file)
    {
        ArgumentNullException.ThrowIfNull(file);
        if (file.GetString("tokenizer.ggml.model", "") != "gemma4") return new EmbeddingTokenizer(file);
        var vocabulary = file.GetStringArray("tokenizer.ggml.tokens")
            ?? throw new InvalidDataException("Embedding GGUF has no tokenizer vocabulary.");
        var types = file.GetInt32Array("tokenizer.ggml.token_type");
        var merges = file.GetStringArray("tokenizer.ggml.merges")
            ?? throw new InvalidDataException("Gemma embedding tokenizer requires BPE merge ranks.");
        int bos = ReadId(file, "bos_token_id", -1), eos = ReadId(file, "eos_token_id", -1);
        if (vocabulary.Length == 0 || (types != null && types.Length != vocabulary.Length) ||
            (uint)bos >= (uint)vocabulary.Length || (uint)eos >= (uint)vocabulary.Length)
            throw new InvalidDataException("Gemma embedding tokenizer has invalid vocabulary or special token IDs.");
        return new BpeTokenizer(vocabulary, types, merges, bos, [eos],
            file.GetBool("tokenizer.ggml.add_bos_token", true),
            file.GetBool("tokenizer.ggml.add_eos_token", true), "gemma4");
    }

    private EmbeddingTokenizer(GgufFile file)
    {
        ArgumentNullException.ThrowIfNull(file);
        string kind = file.GetString("tokenizer.ggml.model", "");
        _wordPiece = kind == "bert";
        if (!_wordPiece && kind != "t5")
            throw new NotSupportedException($"Embedding tokenizer '{kind}' is not supported; expected bert or t5.");
        Vocab = file.GetStringArray("tokenizer.ggml.tokens")
            ?? throw new InvalidDataException("Embedding GGUF has no tokenizer vocabulary.");
        _types = file.GetInt32Array("tokenizer.ggml.token_type") ?? Enumerable.Repeat(1, Vocab.Length).ToArray();
        _scores = file.GetFloatArray("tokenizer.ggml.scores") ?? new float[Vocab.Length];
        if (_types.Length != Vocab.Length || _scores.Length != Vocab.Length)
            throw new InvalidDataException("Embedding tokenizer vocabulary, token types, and scores must have equal lengths.");
        BosTokenId = ReadId(file, "bos_token_id", _wordPiece ? 101 : -1);
        // The historical GGUF key intentionally spells separator as 'seperator'.
        int eos = _wordPiece
            ? ReadId(file, "seperator_token_id", ReadId(file, "eos_token_id", 102))
            : ReadId(file, "eos_token_id", 1);
        EosTokenIds = eos >= 0 ? new[] { eos } : Array.Empty<int>();
        _unknown = ReadId(file, "unknown_token_id", _wordPiece ? 100 : 2);
        _addBos = _wordPiece || file.GetBool("tokenizer.ggml.add_bos_token", false);
        _addEos = _wordPiece || file.GetBool("tokenizer.ggml.add_eos_token", true);
        _addPrefix = file.GetBool("tokenizer.ggml.add_space_prefix", false);
        _removeSpaces = file.GetBool("tokenizer.ggml.remove_extra_whitespaces", false);
        _lowercase = file.GetBool("tokenizer.ggml.normalizer.lowercase", true);
        _stripAccents = file.GetBool("tokenizer.ggml.normalizer.strip_accents", _lowercase);
        if (_unknown < 0 || _unknown >= Vocab.Length ||
            (_addBos && (BosTokenId < 0 || BosTokenId >= Vocab.Length)) ||
            (_addEos && (eos < 0 || eos >= Vocab.Length)))
            throw new InvalidDataException("Embedding tokenizer special token IDs are outside its vocabulary.");

        _lookup = new Dictionary<string, int>(Vocab.Length, StringComparer.Ordinal);
        var specials = new List<int>();
        float minScore = float.MaxValue;
        for (int i = 0; i < Vocab.Length; i++)
        {
            _lookup[Vocab[i]] = i;
            if (_types[i] == 1) minScore = Math.Min(minScore, _scores[i]);
            if (_wordPiece || _types[i] is 1 or 4 or 5) _pieces.Add(Vocab[i], i);
            if (_types[i] is 2 or 3 or 4)
            {
                _specials.Add(Vocab[i], i);
                specials.Add(i);
            }
            if (_types[i] == 4) _userDefined.Add(Vocab[i], i);
        }
        _unknownScore = (minScore == float.MaxValue ? 0 : minScore) - 10;
        SpecialTokenIds = specials.ToArray();
        if (file.Metadata.TryGetValue("tokenizer.ggml.precompiled_charsmap", out object raw))
        {
            byte[] bytes = raw switch
            {
                byte[] unsigned => unsigned,
                sbyte[] signed => Array.ConvertAll(signed, b => unchecked((byte)b)),
                _ => throw new InvalidDataException("SentencePiece charsmap must be a byte array.")
            };
            if (bytes.Length < 4) throw new InvalidDataException("SentencePiece charsmap header is truncated.");
            uint size = BinaryPrimitives.ReadUInt32LittleEndian(bytes);
            if (size == 0 || size % 4 != 0 || size >= bytes.Length - 4)
                throw new InvalidDataException("SentencePiece charsmap trie has an invalid size.");
            _charsmap = new uint[size / 4];
            for (int i = 0; i < _charsmap.Length; i++)
                _charsmap[i] = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(4 + i * 4));
            _replacements = bytes.AsSpan(checked((int)size + 4)).ToArray();
        }
    }

    private static int ReadId(GgufFile file, string key, int fallback) =>
        unchecked((int)file.GetUint32("tokenizer.ggml." + key, unchecked((uint)fallback)));

    public List<int> Encode(string text, bool addSpecial = true)
    {
        ArgumentNullException.ThrowIfNull(text);
        var output = new List<int>();
        if (addSpecial && _addBos) output.Add(BosTokenId);
        byte[] input = Encoding.UTF8.GetBytes(text);
        int start = 0;
        for (int offset = 0; offset < input.Length;)
        {
            var special = _specials.Longest(input, offset);
            if (special.Id >= 0)
            {
                EncodeText(input.AsSpan(start, offset - start), output);
                output.Add(special.Id);
                offset += special.Length;
                start = offset;
            }
            else offset += Utf8Length(input[offset]);
        }
        EncodeText(input.AsSpan(start), output);
        if (addSpecial && _addEos) output.Add(EosTokenIds[0]);
        return output;
    }

    private void EncodeText(ReadOnlySpan<byte> input, List<int> output)
    {
        if (input.IsEmpty) return;
        if (_wordPiece) EncodeWordPiece(Encoding.UTF8.GetString(input), output);
        else EncodeUnigram(NormalizeUnigram(input), output);
    }

    private void EncodeWordPiece(string text, List<int> output)
    {
        if (_stripAccents) text = text.Normalize(NormalizationForm.FormD);
        var word = new StringBuilder();
        void Flush()
        {
            if (word.Length == 0) return;
            byte[] bytes = Encoding.UTF8.GetBytes("▁" + word);
            int initial = output.Count;
            for (int i = 0; i < bytes.Length;)
            {
                var piece = _pieces.Longest(bytes, i);
                if (piece.Id < 0)
                {
                    output.RemoveRange(initial, output.Count - initial);
                    output.Add(_unknown);
                    break;
                }
                output.Add(piece.Id);
                i += piece.Length;
            }
            word.Clear();
        }
        foreach (Rune original in text.EnumerateRunes())
        {
            UnicodeCategory category = Rune.GetUnicodeCategory(original);
            if (Rune.IsWhiteSpace(original)) { Flush(); continue; }
            if (original.Value is 0 or 0xfffd || category is UnicodeCategory.Control or UnicodeCategory.Format)
                continue;
            if (_stripAccents && category == UnicodeCategory.NonSpacingMark) continue;
            Rune rune = _lowercase ? Rune.ToLowerInvariant(original) : original;
            int c = rune.Value;
            bool punctuation = category is UnicodeCategory.ConnectorPunctuation or UnicodeCategory.DashPunctuation
                or UnicodeCategory.OpenPunctuation or UnicodeCategory.ClosePunctuation or UnicodeCategory.InitialQuotePunctuation
                or UnicodeCategory.FinalQuotePunctuation or UnicodeCategory.OtherPunctuation;
            bool asciiSymbol = c is >= 33 and <= 47 or >= 58 and <= 64 or >= 91 and <= 96 or >= 123 and <= 126;
            bool chinese = c is >= 0x4e00 and <= 0x9fff or >= 0x3400 and <= 0x4dbf
                or >= 0x20000 and <= 0x2a6df or >= 0x2a700 and <= 0x2b73f or >= 0x2b740 and <= 0x2b81f
                or >= 0x2b920 and <= 0x2ceaf or >= 0xf900 and <= 0xfaff or >= 0x2f800 and <= 0x2fa1f;
            if (punctuation || asciiSymbol || chinese)
            {
                Flush(); word.Append(rune.ToString()); Flush();
            }
            else word.Append(rune.ToString());
        }
        Flush();
    }

    private byte[] NormalizeUnigram(ReadOnlySpan<byte> input)
    {
        var output = new List<byte>(input.Length + 16);
        bool prefixed = false, nonWhitespace = false;
        for (int i = 0; i < input.Length;)
        {
            ReadOnlySpan<byte> replacement;
            int consumed = _userDefined.Longest(input, i).Length;
            if (consumed > 0) replacement = input.Slice(i, consumed);
            else
            {
                var match = MatchCharsmap(input, i);
                consumed = match.Length;
                if (consumed > 0)
                {
                    int end = Array.IndexOf(_replacements, (byte)0, match.Replacement);
                    if (end < 0) throw new InvalidDataException("SentencePiece charsmap replacement is unterminated.");
                    replacement = _replacements.AsSpan(match.Replacement, end - match.Replacement);
                }
                else
                {
                    consumed = Utf8Length(input[i]);
                    replacement = input.Slice(i, consumed);
                }
            }
            foreach (byte c in replacement)
            {
                if (c != 32)
                {
                    if (!nonWhitespace)
                    {
                        nonWhitespace = true;
                        if ((_addPrefix && !prefixed) || _removeSpaces)
                        {
                            AddSpace(output); prefixed = true;
                        }
                    }
                    output.Add(c);
                }
                else
                {
                    nonWhitespace = false;
                    if (!_removeSpaces) AddSpace(output);
                }
            }
            i += consumed;
        }
        return output.ToArray();
    }

    private static void AddSpace(List<byte> output) { output.Add(0xe2); output.Add(0x96); output.Add(0x81); }
    private static int Utf8Length(byte first) => first < 0x80 ? 1 : first < 0xe0 ? 2 : first < 0xf0 ? 3 : 4;

    private uint CharsmapNode(uint index) => index < _charsmap.Length
        ? _charsmap[index] : throw new InvalidDataException("SentencePiece charsmap index is outside the trie.");
    private static uint NodeBase(uint value) => (value >> 10) << (int)((value & (1U << 9)) >> 6);

    private (int Length, int Replacement) MatchCharsmap(ReadOnlySpan<byte> input, int start)
    {
        if (_charsmap == null) return (0, 0);
        uint index = NodeBase(CharsmapNode(0));
        int length = 0, replacement = 0;
        for (int i = start; i < input.Length && input[i] != 0; i++)
        {
            index ^= input[i];
            uint node = CharsmapNode(index);
            if ((node & 0x800000ffU) != input[i]) break;
            index ^= NodeBase(node);
            if ((node & 0x100) != 0)
            {
                length = i - start + 1;
                uint value = CharsmapNode(index) & 0x7fffffff;
                if (value >= _replacements.Length)
                    throw new InvalidDataException("SentencePiece charsmap replacement offset is out of bounds.");
                replacement = (int)value;
            }
        }
        return (length, replacement);
    }

    private void EncodeUnigram(byte[] input, List<int> output)
    {
        if (input.Length == 0) return;
        var best = new double[input.Length + 1];
        Array.Fill(best, double.NegativeInfinity);
        best[0] = 0;
        var previous = new int[best.Length];
        var tokens = new int[best.Length];
        for (int start = 0; start < input.Length;)
        {
            int runeLength = Utf8Length(input[start]);
            int node = 0;
            bool singleRune = false;
            for (int end = start; end < input.Length; end++)
            {
                node = _pieces.Next(node, input[end]);
                if (node < 0) break;
                int id = _pieces.Token(node);
                if (id < 0) continue;
                if (end + 1 - start == runeLength) singleRune = true;
                double score = best[start] + (_types[id] == 4 ? 0.0 : _scores[id]);
                if (score > best[end + 1])
                {
                    best[end + 1] = score; previous[end + 1] = start; tokens[end + 1] = id;
                }
            }
            if (!singleRune && best[start] + _unknownScore > best[start + runeLength])
            {
                best[start + runeLength] = best[start] + _unknownScore;
                previous[start + runeLength] = start;
                tokens[start + runeLength] = _unknown;
            }
            start += runeLength;
        }
        int initial = output.Count;
        bool lastUnknown = false;
        for (int end = input.Length; end > 0; end = previous[end])
        {
            int id = tokens[end];
            if (!(lastUnknown && id == _unknown)) output.Add(id);
            lastUnknown = id == _unknown;
        }
        output.Reverse(initial, output.Count - initial);
    }

    public int LookupToken(string tokenStr) => _lookup.TryGetValue(tokenStr, out int id) ? id : -1;
    public bool IsEos(int tokenId) => Array.IndexOf(EosTokenIds, tokenId) >= 0;
    public void AppendTokenBytes(int tokenId, List<byte> buffer)
    {
        if ((uint)tokenId >= Vocab.Length) throw new ArgumentOutOfRangeException(nameof(tokenId));
        if (_types[tokenId] == 3) return;
        buffer.AddRange(Encoding.UTF8.GetBytes(Vocab[tokenId].Replace('▁', ' ')));
    }
    public string Decode(List<int> ids)
    {
        var buffer = new List<byte>();
        foreach (int id in ids) AppendTokenBytes(id, buffer);
        string text = Encoding.UTF8.GetString(buffer.ToArray());
        return text.StartsWith(' ') ? text.Substring(1) : text;
    }

    // One dictionary for all trie edges avoids a separate dictionary allocation
    // for each node in XLM-R's 250,002-piece multilingual vocabulary.
    private sealed class ByteTrie
    {
        private readonly Dictionary<long, int> _edges = new();
        private readonly List<int> _tokens = new() { -1 };
        public int Next(int node, byte value) => _edges.TryGetValue(((long)node << 8) | value, out int next) ? next : -1;
        public int Token(int node) => _tokens[node];
        public void Add(string text, int id)
        {
            if (text.Length == 0) return;
            int node = 0;
            foreach (byte value in Encoding.UTF8.GetBytes(text))
            {
                long key = ((long)node << 8) | value;
                if (!_edges.TryGetValue(key, out int next))
                {
                    next = _tokens.Count; _tokens.Add(-1); _edges.Add(key, next);
                }
                node = next;
            }
            _tokens[node] = id;
        }
        public (int Id, int Length) Longest(ReadOnlySpan<byte> bytes, int start)
        {
            int node = 0, id = -1, length = 0;
            for (int i = start; i < bytes.Length; i++)
            {
                node = Next(node, bytes[i]);
                if (node < 0) break;
                if (_tokens[node] >= 0) { id = _tokens[node]; length = i - start + 1; }
            }
            return (id, length);
        }
    }
}
