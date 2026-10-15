// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text.Json;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

public enum CudaEmbeddingPhase
{
    InitialPle, NormQkv, SplitNormRope, LocalAttention, GlobalAttention,
    AttentionOutputResidual, FeedForward, PleResidual, FinalPoolHead
}

public sealed record CudaEmbeddingPhaseProfile(long Step, int Tokens, int Batches, int Hidden, int Heads, int Layers,
    int[] SequenceLengths, double WallMs, double HostIssueMs, double GpuSpanTotalMs,
    Dictionary<string, double> PerStageMs, Dictionary<string, long> StageCalls,
    Dictionary<string, double> HostIssueStageMs, string Scope, string Limitations);

/// <summary>Opt-in diagnostic events on the encoder's existing CUDA stream.
/// Call End(FinalPoolHead) before the normal output readback, and FinishStep
/// after that readback has completed. Normal collection never synchronizes.
/// It preserves backend arithmetic and is disabled unless explicitly requested.</summary>
public sealed class CudaEmbeddingPhaseProfiler : IDisposable
{
    private static readonly string[] Names =
    ["initial_ple", "norm_qkv", "split_norm_rope", "local_attention", "global_attention",
        "attention_output_residual", "feed_forward", "ple_residual", "final_pool_head"];
    private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower };
    private readonly CudaAllocator _allocator;
    private readonly CudaStageTimer _timer;
    private readonly long[] _calls = new long[Names.Length];
    private readonly double[] _hostMs = new double[Names.Length];
    private bool _active, _finalEnded, _disposed;
    private int[] _lengths;
    private int _tokens, _hidden, _heads, _layers;
    private long _step, _start, _lastMark;

    private CudaEmbeddingPhaseProfiler(CudaAllocator allocator)
    {
        _allocator = allocator ?? throw new ArgumentNullException(nameof(allocator));
        _timer = new CudaStageTimer("embedding-cuda", 3, Names);
    }

    /// <summary>Return null on the normal path. The caller can use null-safe
    /// hooks without creating events, contexts or additional diagnostics.</summary>
    public static CudaEmbeddingPhaseProfiler TryCreate(CudaAllocator allocator)
        => Environment.GetEnvironmentVariable("TS_EMBEDDING_CUDA_PHASES") == "1" ? new(allocator) : null;

    /// <summary>Start after tokenization/input preparation and before the
    /// first PLE projection. A serialized encoder must finish/abort each step.</summary>
    public void BeginStep(int[] sequenceLengths, int hidden, int heads, int layers)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_active) throw new InvalidOperationException("A CUDA embedding profile step is already active.");
        if (sequenceLengths == null || sequenceLengths.Length == 0 || hidden <= 0 || heads <= 0 || layers <= 0)
            throw new ArgumentException("Invalid CUDA embedding profile geometry.");
        int tokens = 0;
        foreach (int length in sequenceLengths)
        {
            if (length <= 0 || length > 8192) throw new ArgumentException("Invalid CUDA embedding sequence length.");
            tokens = checked(tokens + length);
        }
        _lengths = (int[])sequenceLengths.Clone(); _tokens = tokens; _hidden = hidden; _heads = heads; _layers = layers;
        Array.Clear(_calls); Array.Clear(_hostMs); _finalEnded = false;
        // Remain abortable if creating/recording the first event fails.
        _start = _lastMark = Stopwatch.GetTimestamp(); _step++; _active = true;
        _timer.SpanStart(_allocator.Context, _allocator.Stream.Handle);
    }

    /// <summary>Close the GPU work queued since the previous mark, and charge
    /// its host issue interval separately. This records an event without waiting.</summary>
    public void End(CudaEmbeddingPhase phase)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_active || _finalEnded || !Enum.IsDefined(phase)) throw new InvalidOperationException("Invalid CUDA embedding phase mark.");
        _timer.End(_allocator.Context, _allocator.Stream.Handle, (int)phase);
        long now = Stopwatch.GetTimestamp();
        _hostMs[(int)phase] += (now - _lastMark) * 1000.0 / Stopwatch.Frequency;
        _lastMark = now; _calls[(int)phase]++;
        _finalEnded = phase == CudaEmbeddingPhase.FinalPoolHead;
    }

    /// <summary>Collect after the encoder's existing completed output readback.
    /// No stream/event synchronize is issued; CUDA_ERROR_NOT_READY fails rather
    /// than changing the normal execution schedule. One structured row is logged.</summary>
    public CudaEmbeddingPhaseProfile FinishStep()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_active || !_finalEnded) throw new InvalidOperationException("Finish requires a final pool/head phase and completed readback.");
        double wall = Stopwatch.GetElapsedTime(_start).TotalMilliseconds;
        double[] times = WithCurrentContext(() => _timer.CollectAndReset(synchronize: false));
        var gpu = new Dictionary<string, double>(); var calls = new Dictionary<string, long>(); var host = new Dictionary<string, double>();
        double sum = 0;
        for (int i = 0; i < Names.Length; i++)
        { gpu.Add(Names[i], times[i]); calls.Add(Names[i], _calls[i]); host.Add(Names[i], _hostMs[i]); sum += times[i]; }
        var report = new CudaEmbeddingPhaseProfile(_step, _tokens, _lengths.Length, _hidden, _heads, _layers,
            (int[])_lengths.Clone(), wall, (_lastMark - _start) * 1000.0 / Stopwatch.Frequency, sum, gpu, calls, host,
            "Direct CUDA embedding compute phases, excluding model load/tokenization/input preparation before BeginStep.",
            "GPU event intervals include stream idle time from host issue/allocation; host issue and wall are separate and must not be added to GPU spans. " +
            "Profiling events perturb scheduling and are not a formal throughput measurement. Collection follows the caller's existing readback and adds no normal-path synchronize. " +
            "Backend math/precision dispatch is unchanged. Allocator public hit/miss counters do not account for its >=2MiB large-pool branch.");
        _active = false;
        Console.Error.WriteLine("EMBEDDING_CUDA_PHASES " + JsonSerializer.Serialize(report, JsonOptions));
        return report;
    }

    /// <summary>Exceptional/canceled-step cleanup. This is the only profiler
    /// path that drains the stream: pending event handles return to their pool
    /// before disposal. It never frees model/device-weight cache entries.</summary>
    public void AbortStep(string reason = "aborted")
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_active) return;
        _allocator.Synchronize();
        WithCurrentContext(() => _timer.CollectAndReset(synchronize: false));
        _active = false;
        Console.Error.WriteLine("EMBEDDING_CUDA_PHASES_ABORT " + JsonSerializer.Serialize(new { step = _step, reason,
            tokens = _tokens, batches = _lengths.Length, successful_phase_marks = _calls }, JsonOptions));
    }

    private T WithCurrentContext<T>(Func<T> operation)
    {
        CudaDriverApi.cuCtxGetCurrent(out var previous).ThrowOnError();
        _allocator.Context.MakeCurrent();
        try { return operation(); }
        finally { CudaDriverApi.cuCtxSetCurrent(previous).ThrowOnError(); }
    }

    /// <summary>The encoder owns the profiler and must dispose it before its
    /// allocator. Already-finished steps have only idle, reusable event handles.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        try { if (_active) AbortStep("dispose-before-finish"); }
        finally
        {
            try { WithCurrentContext(() => { _timer.Dispose(); return 0; }); }
            finally { _disposed = true; _active = false; }
        }
    }
}
