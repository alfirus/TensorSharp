// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#nullable enable
using System;
using System.Collections.Generic;
using System.Text.Json;
using TensorSharp.Cpu;
using TensorSharp.Models.Embeddings;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

public partial class Gemma4VisionEncoder
{
    // Independent, default-OFF vision experiment. Text projection options never
    // implicitly quantize vision activations. Dense F32 fallback stays retained.
    private readonly bool _cpuVisionQ16Requested = Environment.GetEnvironmentVariable("TS_GEMMA4_VISION_CPU_Q16") == "1";
    private readonly object _cpuVisionProjectionSync = new();
    private readonly Dictionary<string, ManagedEmbeddingQ8Q16Avx512Matrix> _cpuVisionMatrices = new();
    private readonly HashSet<string> _cpuVisionQ8Keys = new(StringComparer.Ordinal);
    private OwnedEmbeddingCpuProjection? _cpuVisionProjection;
    private string? _cpuVisionProjectionFallback = "not-requested", _cpuVisionLastProjectionFallback;
    private bool _cpuVisionSupportedGeometry, _cpuVisionDisposed, _cpuVisionMathReleased;
    private int _cpuVisionUsedPoolThreads;
    private long _cpuVisionFallbackCalls, _cpuVisionFallbackRows, _cpuVisionF32IntentCalls;
    private bool CpuVisionProjectionRequested => _cpuVisionQ16Requested && _allocator is CpuAllocator;

    private void InitializeCpuVisionProjection(TowerSpec spec, GgufFile gguf)
    {
        if (!CpuVisionProjectionRequested) return;
        _cpuVisionSupportedGeometry = IsValidatedQ8VisionTower(spec, gguf);
        if (!_cpuVisionSupportedGeometry) { _cpuVisionProjectionFallback = "unsupported classic Q8 vision geometry"; return; }
        if (!CpuParallelBinding.Enabled) { _cpuVisionProjectionFallback = "persistent CPU pool disabled"; return; }
        _cpuVisionProjection = OwnedEmbeddingCpuProjection.TryCreate(requested: true, q16ActivationEnabled: true,
            libraryPath: Environment.GetEnvironmentVariable("TENSORSHARP_CPU_NATIVE_LIBRARY"), fallbackReason: out _cpuVisionProjectionFallback);
    }

    private static bool IsCpuVisionLearnedProjection(string name)
        => name == "mm.input_projection.weight" || name.StartsWith("v.blk.", StringComparison.Ordinal) &&
           (name.EndsWith(".attn_q.weight", StringComparison.Ordinal) || name.EndsWith(".attn_k.weight", StringComparison.Ordinal) ||
            name.EndsWith(".attn_v.weight", StringComparison.Ordinal) || name.EndsWith(".attn_out.weight", StringComparison.Ordinal) ||
            name.EndsWith(".ffn_gate.weight", StringComparison.Ordinal) || name.EndsWith(".ffn_up.weight", StringComparison.Ordinal) ||
            name.EndsWith(".ffn_down.weight", StringComparison.Ordinal));

    private void RetainCpuVisionProjection(string name, byte[] raw, GgmlTensorType type, long[] shape)
    {
        if (!CpuVisionProjectionRequested || type != GgmlTensorType.Q8_0 || shape.Length != 2 || !IsCpuVisionLearnedProjection(name)) return;
        _cpuVisionQ8Keys.Add(name);
        if (_cpuVisionProjection != null)
            _cpuVisionMatrices.Add(name, new ManagedEmbeddingQ8Q16Avx512Matrix(raw, checked((int)shape[1]), checked((int)shape[0]), _cpuVisionProjection));
    }

    private unsafe bool TryCpuVisionProjection(Tensor result, Tensor input, string name)
    {
        if (!CpuVisionProjectionRequested) return false;
        if (!_cpuVisionQ8Keys.Contains(name)) { ++_cpuVisionF32IntentCalls; return false; }
        bool available = _cpuVisionMatrices.TryGetValue(name, out var matrix);
        long rows = input.DimensionCount == 2 ? input.Sizes[0] : 0;
        if (!available || !ManagedEmbeddingQ8Q16Avx512Matrix.IsSupported || rows <= 0 || rows > 16384 || input.DimensionCount != 2 || result.DimensionCount != 2 ||
            input.ElementType != DType.Float32 || result.ElementType != DType.Float32 ||
            input.Storage is not CpuStorage || result.Storage is not CpuStorage || !input.IsContiguous() || !result.IsContiguous() ||
            result.Sizes[0] != rows || input.Sizes[1] != _weights[name].Sizes[1] || result.Sizes[1] != _weights[name].Sizes[0])
        {
            ++_cpuVisionFallbackCalls; _cpuVisionFallbackRows += Math.Max(0, rows);
            _cpuVisionLastProjectionFallback = available ? "unsupported shape/storage/runtime feature" : _cpuVisionProjectionFallback;
            return false;
        }
        int count = checked((int)rows);
        var source = new ReadOnlySpan<float>(GetFloatPtr(input), checked(count * (int)input.Sizes[1]));
        var destination = new Span<float>(GetFloatPtr(result), checked(count * (int)result.Sizes[1]));
        // Borrow the existing Core CPU pool; this tower never disposes it or
        // creates a second set of spinning workers. The outer CPU encode lock
        // keeps storage/library leases alive through all synchronous jobs.
        var pool = CpuWorkerPool.Shared;
        matrix!.MultiplyNative(source, count, destination, pool);
        _cpuVisionUsedPoolThreads = pool.ThreadCount;
        return true;
    }

    private void ReleaseCpuVisionProjection()
    {
        if (_cpuVisionMathReleased) return;
        _cpuVisionMathReleased = true;
        try
        {
            Console.Error.WriteLine("VISION_CPU_MATH " + JsonSerializer.Serialize(new
            {
                backend = "CPU", graph = "managed-tensor", q16_requested = _cpuVisionQ16Requested,
                supported_q8_geometry = _cpuVisionSupportedGeometry,
                borrowed_pool_threads = _cpuVisionUsedPoolThreads,
                packed_matrices = _cpuVisionMatrices.Count, eligible_q8_matrices = _cpuVisionQ8Keys.Count,
                native_math = (_cpuVisionProjection?.SuccessfulCalls ?? 0) > 0,
                math = "owned-native-Q8-Q16-VNNI-with-dense-F32-fallback",
                library_path = _cpuVisionProjection?.LibraryPath, library_sha256 = _cpuVisionProjection?.LibrarySha256,
                fallback_reason = _cpuVisionProjectionFallback,
                last_projection_fallback = _cpuVisionLastProjectionFallback,
                successful_projection_calls = _cpuVisionProjection?.SuccessfulCalls ?? 0,
                successful_projection_rows = _cpuVisionProjection?.SuccessfulRows ?? 0,
                successful_projection_groups = _cpuVisionProjection?.SuccessfulGroups ?? 0,
                successful_projection_pack_rows = _cpuVisionProjection?.SuccessfulPackRows ?? 0,
                q16_float32_correction_rows = _cpuVisionProjection?.Float32FallbackRows ?? 0,
                eligible_fallback_calls = _cpuVisionFallbackCalls, eligible_fallback_rows = _cpuVisionFallbackRows,
                deliberate_f32_projection_calls = _cpuVisionF32IntentCalls,
                limitations = "Opt-in Q16 activation experiment. Patch F32 and other backend routes are retained; native counters alone are not model quality/performance proof. Encode retains its existing synchronous cancellation contract.",
            }));
        }
        finally
        {
            _cpuVisionMatrices.Clear(); _cpuVisionQ8Keys.Clear();
            _cpuVisionProjection?.Dispose(); _cpuVisionProjection = null;
        }
    }
}
