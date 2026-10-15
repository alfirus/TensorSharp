// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Embeddings;

/// <summary>A sentence encoder with normalized embeddings and no autoregressive KV cache.</summary>
public interface IEmbeddingModel : IDisposable
{
    string ModelName { get; }
    string Architecture { get; }
    int Dimensions { get; }
    int MaxTokens { get; }
    int VocabularySize { get; }
    int[] Tokenize(string text, bool truncate = false);
    Task<EmbeddingBatchResult> EmbedTokensAsync(IReadOnlyList<int[]> inputs, CancellationToken cancellationToken = default);
}

public sealed record EmbeddingBatchResult(float[][] Embeddings, int PromptTokens);

public sealed class EmbeddingModelOptions
{
    public string Backend { get; init; } = "CPU";
    public int Device { get; init; }
    public int Threads { get; init; }
    /// <summary>Zero uses the supported model window (8192 for EmbeddingGemma 2).
    /// A positive value may reduce it.</summary>
    public int MaxTokens { get; init; }
    public string ModelName { get; init; }
}

/// <summary>GGUF sentence encoders on managed CPU, direct CUDA/MLX, and native ggml backends.</summary>
public sealed class EmbeddingModel : IEmbeddingModel
{
    private readonly ITokenizer _tokenizer;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly ManagedEmbeddingEncoder _managed;
    private readonly ManagedEmbeddingGemma2Encoder _managedGemma;
    private readonly TensorEmbeddingGemma2Encoder _tensor;
    private readonly IAllocator _inputAllocator;
    private IntPtr _handle;
    private bool _disposed;

    public string ModelName { get; }
    public string Architecture { get; }
    public int Dimensions { get; }
    /// <summary>Width of token and media inputs before the final output projection.</summary>
    public int HiddenSize { get; }
    public int MaxTokens { get; }
    public int VocabularySize => _tokenizer.VocabSize;
    /// <summary>CPU is managed; CUDA and MLX use their own tensor executors;
    /// GGML_CPU, GGML_METAL and GGML_CUDA use native ggml.</summary>
    public string Backend { get; }
    public bool IsManaged => _managed != null || _managedGemma != null;
    internal IAllocator InputAllocator => _inputAllocator;

    private EmbeddingModel(string path, EmbeddingModelOptions options)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (options.Device < 0 || options.Threads < 0 || options.Threads > 512 || options.MaxTokens < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "Device, thread count, and context limit must be nonnegative; threads must not exceed 512.");
        Backend = (options.Backend ?? "CPU").ToUpperInvariant() switch
        {
            "CPU" => "CPU",
            "GGML_CPU" => "GGML_CPU",
            "METAL" or "GGML_METAL" => "GGML_METAL",
            "GGML_CUDA" => "GGML_CUDA",
            "CUDA" => "CUDA",
            "MLX" => "MLX",
            _ => throw new NotSupportedException($"Unsupported embedding backend '{options.Backend}'."),
        };
        if (Backend == "CPU" && options.Device != 0)
            throw new ArgumentOutOfRangeException(nameof(options.Device), "The managed CPU embedding backend supports device 0 only.");
        using var file = new GgufFile(path);
        Architecture = file.GetString("general.architecture") ?? string.Empty;
        bool isNomicBert = Architecture == "nomic-bert";
        bool isGemma2 = Architecture == "gemma-embedding2";
        if (Architecture != "bert" && !isNomicBert && !isGemma2)
            throw new NotSupportedException($"Embedding architecture '{Architecture}' is not supported. Expected bert, nomic-bert, or gemma-embedding2.");
        if (!isGemma2 && Backend is "CUDA" or "MLX")
            throw new NotSupportedException($"The {Backend} tensor embedding executor currently requires EmbeddingGemma 2; select a GGML backend for {Architecture}.");
        string prefix = Architecture + ".";
        uint pooling = file.GetUint32(prefix + "pooling_type", 0);
        if (pooling is < 1 or > 3 || (isGemma2 && pooling != 1))
            throw new NotSupportedException($"Embedding GGUF must specify mean, CLS, or last-token pooling in {prefix}pooling_type.");
        int hiddenSize = checked((int)file.GetUint32(prefix + "embedding_length"));
        HiddenSize = hiddenSize;
        Dimensions = isGemma2 ? checked((int)file.GetUint32(prefix + "embedding_length_out")) : hiddenSize;
        int modelContext = checked((int)file.GetUint32(prefix + "context_length"));
        if (hiddenSize <= 0 || Dimensions <= 0 || modelContext <= 0) throw new InvalidDataException("Embedding GGUF is missing valid dimensions or context length.");
        // This GGUF records the backbone's rotary range (262144), whereas the
        // published embedding model supports an 8192-token shared input window.
        if (isGemma2) modelContext = Math.Min(modelContext, 8192);
        if (options.MaxTokens > modelContext)
            throw new ArgumentOutOfRangeException(nameof(options.MaxTokens), $"The context limit must not exceed the model's {modelContext} tokens.");
        MaxTokens = options.MaxTokens == 0 ? modelContext : options.MaxTokens;
        ModelName = string.IsNullOrWhiteSpace(options.ModelName) ? Path.GetFileNameWithoutExtension(path) : options.ModelName;
        _tokenizer = EmbeddingTokenizer.Create(file);
        if (!file.Tensors.TryGetValue("token_embd.weight", out var tokenEmbedding) ||
            tokenEmbedding.Shape.Length != 2 || tokenEmbedding.Shape[0] != (ulong)hiddenSize ||
            tokenEmbedding.Shape[1] != (ulong)_tokenizer.VocabSize)
            throw new InvalidDataException("Embedding tensor dimensions do not match the model dimension and tokenizer vocabulary.");
        if (isGemma2 && (!file.Tensors.TryGetValue("output.weight", out var projection) ||
            projection.Shape.Length != 2 || projection.Shape[0] != (ulong)hiddenSize || projection.Shape[1] != (ulong)Dimensions))
            throw new InvalidDataException("EmbeddingGemma 2 output projection does not match the hidden and output dimensions.");
        if (Backend == "CPU" && isGemma2)
            _managedGemma = new ManagedEmbeddingGemma2Encoder(file, options.Threads);
        else if (Backend == "CPU")
            _managed = new ManagedEmbeddingEncoder(file, options.Threads);
        else if (Backend is "CUDA" or "MLX")
        {
            IAllocator allocator;
            if (Backend == "MLX")
            {
                TensorSharp.MLX.MlxBackend.Register();
                allocator = new TensorSharp.MLX.MlxAllocator(options.Device);
            }
            else allocator = new TensorSharp.Cuda.CudaAllocator(options.Device);
            _inputAllocator = allocator;
            try { _tensor = new TensorEmbeddingGemma2Encoder(file, allocator); }
            catch { (allocator as IDisposable)?.Dispose(); throw; }
        }
        else
            _handle = LoadNative(path, Backend, options.Device, options.Threads);
    }

    // Keep native initialization behind a distinct call so the pure C# path never
    // probes, loads, or changes process-wide state for any native backend.
    private static IntPtr LoadNative(string path, string backend, int device, int threads)
    {
        var handle = GgmlEmbeddingNative.TSGgml_EmbeddingLoad(Path.GetFullPath(path), backend, device, threads);
        if (handle == IntPtr.Zero) throw new InvalidOperationException(GgmlEmbeddingNative.LastError("Cannot load embedding model."));
        return handle;
    }

    public static EmbeddingModel Load(string path, EmbeddingModelOptions options = null) => new(path, options ?? new());

    public int[] Tokenize(string text, bool truncate = false)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(text);
        var tokens = _tokenizer.Encode(text, addSpecial: true);
        if (tokens.Count > MaxTokens)
        {
            if (!truncate) throw new ArgumentException($"Input has {tokens.Count} tokens and exceeds the model context limit of {MaxTokens}.", nameof(text));
            int last = tokens[^1];
            tokens.RemoveRange(MaxTokens, tokens.Count - MaxTokens);
            // Preserve the terminal separator/EOS supplied by the tokenizer.
            if (_tokenizer.IsEos(last)) tokens[^1] = last;
        }
        if (tokens.Count == 0) throw new ArgumentException("Input produces no tokens.", nameof(text));
        return tokens.ToArray();
    }

    public async Task<EmbeddingBatchResult> EmbedTokensAsync(IReadOnlyList<int[]> inputs, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(inputs);
        cancellationToken.ThrowIfCancellationRequested();
        if (inputs.Count == 0) return new(Array.Empty<float[]>(), 0);
        var owned = new int[inputs.Count][];
        int totalTokens = 0;
        for (int i = 0; i < inputs.Count; ++i)
        {
            var row = inputs[i] ?? throw new ArgumentException("Input sequences must not be null.", nameof(inputs));
            if (row.Length == 0 || row.Length > MaxTokens)
                throw new ArgumentException($"Every input must have between 1 and {MaxTokens} tokens.", nameof(inputs));
            owned[i] = (int[])row.Clone();
            foreach (int token in owned[i])
                if ((uint)token >= (uint)VocabularySize) throw new ArgumentException($"Token {token} is outside the model vocabulary.", nameof(inputs));
            totalTokens = checked(totalTokens + row.Length);
        }
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            // Run computation off the caller's synchronization context. The gate
            // remains held until an in-flight device submission completes, even on cancellation.
            return await Task.Run(() =>
            {
                var results = new float[owned.Length][];
                var order = Enumerable.Range(0, owned.Length).OrderBy(i => owned[i].Length).ToArray();
                for (int start = 0; start < order.Length;)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    int end = start + 1, batchTokens = owned[order[start]].Length;
                    // Compact batches preserve each complete sequence in one graph.
                    while (end < order.Length && end - start < 64 && batchTokens + owned[order[end]].Length <= 4096)
                        batchTokens += owned[order[end++]].Length;
                    int count = end - start;
                    var lengths = new int[count];
                    int flatCount = 0;
                    for (int i = 0; i < count; ++i) { lengths[i] = owned[order[start + i]].Length; flatCount += lengths[i]; }
                    var flat = new int[flatCount];
                    int offset = 0;
                    for (int i = 0; i < count; ++i) { owned[order[start + i]].CopyTo(flat, offset); offset += lengths[i]; }
                    var output = new float[checked(count * Dimensions)];
                    if (_managedGemma != null) _managedGemma.Encode(flat, lengths, output, cancellationToken);
                    else if (_tensor != null) _tensor.Encode(flat, lengths, output, cancellationToken);
                    else if (_managed != null) _managed.Encode(flat, lengths, output, cancellationToken);
                    else GgmlEmbeddingNative.Encode(_handle, flat, lengths, output);
                    for (int i = 0; i < count; ++i)
                    {
                        var row = new float[Dimensions];
                        Array.Copy(output, i * Dimensions, row, 0, Dimensions);
                        results[order[start + i]] = row;
                    }
                    start = end;
                }
                cancellationToken.ThrowIfCancellationRequested();
                return new EmbeddingBatchResult(results, totalTokens);
            }, CancellationToken.None).ConfigureAwait(false);
        }
        finally { _gate.Release(); }
    }

    public Task<EmbeddingBatchResult> EmbedAsync(IReadOnlyList<string> inputs, bool truncate = false, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        return EmbedTokensAsync(inputs.Select(text => Tokenize(text, truncate)).ToArray(), cancellationToken);
    }

    /// <summary>Returns EmbeddingGemma 2 token rows with the model's sqrt(hidden) scale.
    /// Media encoder rows already use the input scale and must not be scaled again.</summary>
    public float[] GetTokenEmbeddings(int[] tokens)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        RequireGemmaInputs();
        if (tokens.Length == 0) return Array.Empty<float>();
        if (tokens.Length > MaxTokens) throw new ArgumentException("Token rows exceed the context limit.", nameof(tokens));
        var owned = (int[])tokens.Clone();
        foreach (int token in owned)
            if ((uint)token >= (uint)VocabularySize) throw new ArgumentException("Token is outside the model vocabulary.", nameof(tokens));
        _gate.Wait();
        try
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            var output = new float[checked(owned.Length * HiddenSize)];
            if (_managedGemma != null) _managedGemma.TokenRows(owned, output);
            else if (_tensor != null) _tensor.TokenRows(owned, output);
            else GgmlEmbeddingNative.TokenRows(_handle, owned, output);
            return output;
        }
        finally { _gate.Release(); }
    }

    /// <summary>Encodes one complete EmbeddingGemma 2 sequence of row-major F32
    /// token and media inputs. Text rows are sqrt(hidden)-scaled; media rows are raw.</summary>
    public async Task<EmbeddingBatchResult> EmbedInputsAsync(float[] inputs, int tokens, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        RequireGemmaInputs();
        if (tokens < 1 || tokens > MaxTokens || inputs.Length != checked(tokens * HiddenSize))
            throw new ArgumentException($"Input must contain 1 to {MaxTokens} rows of {HiddenSize} values.", nameof(inputs));
        var owned = (float[])inputs.Clone();
        if (owned.Any(value => !float.IsFinite(value))) throw new ArgumentException("Input rows must be finite.", nameof(inputs));
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                var output = new float[Dimensions];
                if (_managedGemma != null) _managedGemma.EncodeInputs(owned, [tokens], output, cancellationToken);
                else if (_tensor != null) _tensor.EncodeInputs(owned, [tokens], output, cancellationToken);
                else GgmlEmbeddingNative.EncodeInputs(_handle, owned, [tokens], output);
                cancellationToken.ThrowIfCancellationRequested();
                return new EmbeddingBatchResult([output], tokens);
            }, CancellationToken.None).ConfigureAwait(false);
        }
        finally { _gate.Release(); }
    }

    private void RequireGemmaInputs()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (Architecture != "gemma-embedding2")
            throw new NotSupportedException("Input embeddings require an EmbeddingGemma 2 model.");
    }

    public void Dispose()
    {
        _gate.Wait();
        try
        {
            if (_disposed) return;
            _disposed = true;
            _managed?.Dispose();
            _managedGemma?.Dispose();
            _tensor?.Dispose();
            (_inputAllocator as IDisposable)?.Dispose();
            if (_handle != IntPtr.Zero) { GgmlEmbeddingNative.TSGgml_EmbeddingFree(_handle); _handle = IntPtr.Zero; }
        }
        finally { _gate.Release(); }
    }
}
