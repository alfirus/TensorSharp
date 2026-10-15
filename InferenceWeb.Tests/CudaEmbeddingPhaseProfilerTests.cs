using TensorSharp;
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

public sealed class CudaEmbeddingPhaseProfilerTests
{
    [Fact]
    public void DisabledProfilerDoesNotRequireAnAllocator()
    {
        string? previous = Environment.GetEnvironmentVariable("TS_EMBEDDING_CUDA_PHASES");
        try
        {
            Environment.SetEnvironmentVariable("TS_EMBEDDING_CUDA_PHASES", "0");
            Assert.Null(CudaEmbeddingPhaseProfiler.TryCreate(null!));
        }
        finally { Environment.SetEnvironmentVariable("TS_EMBEDDING_CUDA_PHASES", previous); }
    }

    [CudaFact]
    public void CompletedReadbackCollectsStructuredPhasesAndCanStartANewStep()
    {
        string? previous = Environment.GetEnvironmentVariable("TS_EMBEDDING_CUDA_PHASES");
        try
        {
            Environment.SetEnvironmentVariable("TS_EMBEDDING_CUDA_PHASES", "1");
            using var allocator = new CudaAllocator();
            using var input = Tensor.FromArray(allocator, new float[] { 1, -2, 4, -8 });
            using var profiler = CudaEmbeddingPhaseProfiler.TryCreate(allocator)!;
            int[] lengths = [13, 29];
            for (int step = 1; step <= 3; step++)
            {
                profiler.BeginStep(lengths, 512, 4, 24);
                Ops.Mul(input, input, 2f);
                profiler.End(CudaEmbeddingPhase.NormQkv);
                Ops.Mul(input, input, .5f);
                profiler.End(CudaEmbeddingPhase.NormQkv);
                profiler.End(CudaEmbeddingPhase.FinalPoolHead);
                Assert.Equal(new float[] { 1, -2, 4, -8 }, input.GetElementsAsFloat(4));
                CudaEmbeddingPhaseProfile report = profiler.FinishStep();
                Assert.Equal(step, report.Step);
                Assert.Equal(42, report.Tokens); Assert.Equal(2, report.Batches);
                Assert.Equal(lengths, report.SequenceLengths);
                Assert.NotSame(lengths, report.SequenceLengths);
                Assert.Equal(512, report.Hidden); Assert.Equal(4, report.Heads); Assert.Equal(24, report.Layers);
                Assert.Equal(2, report.StageCalls["norm_qkv"]);
                Assert.Equal(1, report.StageCalls["final_pool_head"]);
                Assert.Equal(3, report.StageCalls.Values.Sum());
                Assert.Equal(9, report.PerStageMs.Count);
                Assert.All(report.PerStageMs.Values, value => Assert.True(double.IsFinite(value) && value >= 0));
                Assert.All(report.HostIssueStageMs.Values, value => Assert.True(double.IsFinite(value) && value >= 0));
                Assert.Equal(report.PerStageMs.Values.Sum(), report.GpuSpanTotalMs);
                Assert.InRange(report.HostIssueMs, 0, report.WallMs);
                Assert.Throws<InvalidOperationException>(() => profiler.FinishStep());
            }
        }
        finally { Environment.SetEnvironmentVariable("TS_EMBEDDING_CUDA_PHASES", previous); }
    }

    [CudaFact]
    public void AbortDrainsPendingMarksAndKeepsReuseAndPeerTensorsValid()
    {
        string? previous = Environment.GetEnvironmentVariable("TS_EMBEDDING_CUDA_PHASES");
        try
        {
            Environment.SetEnvironmentVariable("TS_EMBEDDING_CUDA_PHASES", "1");
            using var allocator = new CudaAllocator(); using var peer = new CudaAllocator();
            using var input = Tensor.FromArray(allocator, new float[] { 1, -2, 4, -8 });
            using var peerInput = Tensor.FromArray(peer, new float[] { 3, -5 });
            using var profiler = CudaEmbeddingPhaseProfiler.TryCreate(allocator)!;
            Assert.Throws<ArgumentException>(() => profiler.BeginStep([0], 512, 4, 24));
            for (int iteration = 0; iteration < 3; iteration++)
            {
                profiler.BeginStep([29], 512, 4, 24);
                Ops.Mul(input, input, 2f);
                if (iteration != 0) profiler.End(CudaEmbeddingPhase.FeedForward);
                Assert.Throws<InvalidOperationException>(() => profiler.BeginStep([13], 512, 4, 24));
                profiler.AbortStep("test-pending-marks");
                profiler.AbortStep("already-aborted");
                Assert.Throws<InvalidOperationException>(() => profiler.End(CudaEmbeddingPhase.FinalPoolHead));
            }
            profiler.BeginStep([13], 512, 4, 24);
            Ops.Mul(input, input, .125f);
            profiler.End(CudaEmbeddingPhase.FinalPoolHead);
            Assert.Equal(new float[] { 1, -2, 4, -8 }, input.GetElementsAsFloat(4));
            CudaEmbeddingPhaseProfile report = profiler.FinishStep();
            Assert.Equal(1, report.StageCalls.Values.Sum());
            Assert.Equal(0, report.StageCalls["feed_forward"]);
            Assert.Equal(0, report.PerStageMs["feed_forward"]);
            Ops.Mul(peerInput, peerInput, 2f);
            Assert.Equal(new float[] { 6, -10 }, peerInput.GetElementsAsFloat(2));
        }
        finally { Environment.SetEnvironmentVariable("TS_EMBEDDING_CUDA_PHASES", previous); }
    }

    [CudaFact]
    public void EarlyDisposeDrainsPendingWorkAndIsIdempotentBeforeAllocatorDisposal()
    {
        string? previous = Environment.GetEnvironmentVariable("TS_EMBEDDING_CUDA_PHASES");
        try
        {
            Environment.SetEnvironmentVariable("TS_EMBEDDING_CUDA_PHASES", "1");
            using var allocator = new CudaAllocator();
            using var input = Tensor.FromArray(allocator, new float[] { 3, -5 });
            var profiler = CudaEmbeddingPhaseProfiler.TryCreate(allocator)!;
            profiler.BeginStep([13], 512, 4, 24);
            Ops.Mul(input, input, 2f);
            profiler.End(CudaEmbeddingPhase.NormQkv);
            profiler.Dispose(); profiler.Dispose();
            Assert.Throws<ObjectDisposedException>(() => profiler.BeginStep([13], 512, 4, 24));
            Assert.Throws<ObjectDisposedException>(() => profiler.End(CudaEmbeddingPhase.NormQkv));
            Assert.Throws<ObjectDisposedException>(() => profiler.FinishStep());
            Assert.Equal(new float[] { 6, -10 }, input.GetElementsAsFloat(2));
        }
        finally { Environment.SetEnvironmentVariable("TS_EMBEDDING_CUDA_PHASES", previous); }
    }
}
