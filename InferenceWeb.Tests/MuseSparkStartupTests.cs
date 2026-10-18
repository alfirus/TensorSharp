// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using TensorSharp;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

[Collection("Muse Spark model")]
public sealed class MuseSparkStartupTests
{
    [Theory]
    [InlineData(KvCacheDtype.F32, true)]
    [InlineData(KvCacheDtype.F16, false)]
    public async Task MetadataOnlyFirstShardLoadsThroughFactoryAndDeclaresActualRuntimeModes(
        KvCacheDtype dtype, bool paged)
    {
        string directory = Path.Combine(Path.GetTempPath(), "ts-spark-startup-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        string[] names = ["MAX_CONTEXT", "TS_KV_INITIAL_TOKENS", "TS_MUSE_SPARK_PREFILL_CHUNK",
            "TENSORSHARP_TP_DEGREE", "TENSORSHARP_LAYER_SPLIT_DEGREE"];
        string?[] saved = names.Select(Environment.GetEnvironmentVariable).ToArray();
        KvCacheDtype savedDtype = KvCacheDtypeConfig.Current;
        bool savedExplicit = KvCacheDtypeConfig.IsExplicitlySet;
        try
        {
            string[] values = ["16", "2", "4", "1", "1"];
            for (int i = 0; i < names.Length; i++) Environment.SetEnvironmentVariable(names[i], values[i]);
            KvCacheDtypeConfig.Set(dtype);
            var fixture = new MuseSparkSyntheticModelBuilder();
            string first = Path.Combine(directory, "spark-00001-of-00002.gguf");
            fixture.Write(Path.Combine(directory, "spark-00002-of-00002.gguf"));
            var metadata = new MuseSparkSyntheticModelBuilder();
            metadata.Weights.Clear();
            metadata.Write(first);

            // Reproduce the published layout: no tensors and no alignment pad in
            // shard 1; every tensor belongs to a sibling file.
            long tableEnd;
            using (var probe = GgufFile.OpenWithoutSiblingShards(first))
            {
                Assert.Empty(probe.Tensors);
                tableEnd = probe.GetRequiredLength(out _);
            }
            byte[] header = File.ReadAllBytes(first);
            ulong metadataCount = BitConverter.ToUInt64(header, 16);
            using (var file = new FileStream(first, FileMode.Open, FileAccess.Write))
            using (var writer = new BinaryWriter(file, Encoding.UTF8))
            {
                file.Position = 16; writer.Write(metadataCount + 1);
                file.Position = tableEnd;
                byte[] key = Encoding.UTF8.GetBytes("split.count");
                writer.Write((ulong)key.Length); writer.Write(key); writer.Write(4u); writer.Write(2u);
                file.SetLength(file.Position);
            }
            using (var merged = new GgufFile(first))
            {
                merged.ThrowIfTruncated();
                Assert.Equal(fixture.Weights.Count, merged.Tensors.Count);
                Assert.Equal(2, merged.FilePaths.Count);
            }
            using var loaded = ModelBase.Create(first, BackendType.Cpu);
            var model = Assert.IsType<MuseSparkModel>(loaded);
            Assert.Equal(16, model.MaxContextLength);
            Assert.Equal(64, model.Config.DeclaredContextLength);
            ExecutionCapabilities capabilities = ExecutionCapabilities.FromModel(model);
            Assert.Equal(paged, capabilities.BatchedForwardAvailable);
            Assert.True(capabilities.SupportsKvStateSnapshot);
            Assert.True(capabilities.SupportsSpeculativeTrunk);
            Assert.False(capabilities.HasDraftHead);
            using var drafter = SpeculatorRegistry.Create(model,
                new SpeculationOptions { Enabled = true, SpeculatorName = SpeculatorRegistry.NGram }, out string? refusal);
            Assert.Null(refusal);
            Assert.IsType<NGramSpeculator>(drafter);

            int[] tokens = [7, 21, 42, 11, 9];
            float[] actual = model.Forward(tokens), expected = fixture.Reference(tokens);
            Assert.All(actual, x => Assert.True(float.IsFinite(x)));
            Assert.True(actual.Zip(expected, (a, b) => MathF.Abs(a - b)).Max() < 3e-3f);
            Assert.Throws<InvalidOperationException>(() => model.Forward(new int[12]));
            model.ResetKVCache();
            using var engine = new InferenceEngine(model, new SchedulerConfig
            {
                BlockSize = 2, NumBlocks = 32, EnablePrefixCaching = true,
                MaxNumBatchedTokens = 8, SoloPrefillChunkSize = 8,
                MaxPrefillChunkSize = 4, StopRepetition = false,
            });
            Assert.True(engine.PrefixCacheActive);
            async Task<SequenceState> Run(string id, int[] prompt)
            {
                var sequence = new SequenceState(id, prompt, 2, 2, SamplingConfig.Greedy, cacheScope: "startup");
                await engine.SubmitRequest(sequence).Completion.WaitAsync(TimeSpan.FromSeconds(20));
                Assert.True(sequence.OutputTokens.Count > 0);
                return sequence;
            }
            int[] sharedPrompt = [7, 21, 42, 11, 9, 86, 102, 68, 52];
            var cold = await Run("cold", sharedPrompt);
            await Run("displace", [132, 41, 24, 62, 13, 45, 16, 91, 8]);
            var warm = await Run("warm", sharedPrompt);
            Assert.Equal(0, cold.PrefixCacheReusedTokens);
            Assert.Equal(8, warm.PrefixCacheReusedTokens);
            Assert.Equal(cold.OutputTokens, warm.OutputTokens);
        }
        finally
        {
            for (int i = 0; i < names.Length; i++) Environment.SetEnvironmentVariable(names[i], saved[i]);
            KvCacheDtypeConfig.RestoreForTests(savedDtype, savedExplicit);
            Directory.Delete(directory, true);
        }
    }
}
