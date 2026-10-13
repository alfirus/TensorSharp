using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Knowledge;
using TensorSharp.Models.Embeddings;

namespace TensorSharp.Server.Knowledge;

/// <summary>A separately owned sentence encoder; knowledge indexing never replaces the chat model.
/// Identity includes weights and prefixes so incompatible vector spaces cannot be mixed.</summary>
public sealed class GgufKnowledgeEmbeddingProvider : IKnowledgeEmbeddingProvider, IDisposable
{
    private readonly IEmbeddingModel _model;
    private readonly string _queryPrefix;
    private readonly string _documentPrefix;
    public string ModelId { get; }
    public int Dimensions => _model.Dimensions;

    public GgufKnowledgeEmbeddingProvider(string path, string backend = "CPU", string queryPrefix = "", string documentPrefix = "")
    {
        using var stream = File.OpenRead(path);
        string hash = Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
        _model = EmbeddingModel.Load(path, new EmbeddingModelOptions { Backend = backend });
        _queryPrefix = queryPrefix;
        _documentPrefix = documentPrefix;
        ModelId = $"gguf:sha256:{hash}:mean-parts-v1:{Convert.ToHexString(SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(queryPrefix + "\0" + documentPrefix)))}";
    }

    public async Task<float[][]> EmbedAsync(IReadOnlyList<string> texts, bool isQuery, CancellationToken cancellationToken = default)
    {
        var inputs = new List<int[]>();
        var owners = new List<int>();
        string prefix = isQuery ? _queryPrefix : _documentPrefix;
        void Add(string text, int owner)
        {
            cancellationToken.ThrowIfCancellationRequested();
            try { inputs.Add(_model.Tokenize(prefix + text)); owners.Add(owner); }
            catch (ArgumentException) when (text.Length > 1)
            {
                // A character chunk may exceed a model's token window, especially for CJK/code.
                // Embed all pieces, never silently truncate the tail of the evidence.
                int mid = text.Length / 2;
                if (char.IsHighSurrogate(text[mid - 1]) && mid < text.Length) mid++;
                if (mid >= text.Length) throw;
                Add(text[..mid], owner); Add(text[mid..], owner);
            }
        }
        for (int i = 0; i < texts.Count; i++) Add(texts[i], i);
        var output = Enumerable.Range(0, texts.Count).Select(_ => new float[Dimensions]).ToArray();
        for (int start = 0; start < inputs.Count; start += 32)
        {
            int count = Math.Min(32, inputs.Count - start);
            var batch = inputs.GetRange(start, count);
            var result = await _model.EmbedTokensAsync(batch, cancellationToken).ConfigureAwait(false);
            for (int i = 0; i < count; i++)
                for (int dimension = 0; dimension < Dimensions; dimension++)
                    output[owners[start + i]][dimension] += result.Embeddings[i][dimension] * batch[i].Length;
        }
        foreach (var vector in output)
        {
            double norm = Math.Sqrt(vector.Sum(x => (double)x * x));
            if (!(norm > 0) || !double.IsFinite(norm)) throw new InvalidDataException("Embedding model returned an invalid vector.");
            for (int i = 0; i < vector.Length; i++) vector[i] = (float)(vector[i] / norm);
        }
        return output;
    }

    public void Dispose() => _model.Dispose();
}
