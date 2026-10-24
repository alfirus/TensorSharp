// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

[Collection(EngineEnvironmentCollection.Name)]
public sealed class ModelLogitsDumpTests
{
    [Fact]
    public void RequestedPositionDumpsExactlyOneVocabularyRowFromAnOversizedBuffer()
    {
        WithModel((model, path) =>
        {
            using var environment = new EnvScope();
            environment.Set("TS_DUMP_LOGITS", path);
            environment.Set("TS_DUMP_LOGITS_AT", "33");
            float[] values = [1.25f, -2.5f, 3.75f, 999, 998, 997];
            Assert.Same(values, model.Dump(values, 32, 3));
            Assert.False(File.Exists(path));
            Assert.Same(values, model.Dump(values, 33, 3, warmup: true));
            Assert.False(File.Exists(path));
            Assert.Same(values, model.Dump(values, 33, 3));
            byte[] expected = values.Take(3).SelectMany(BitConverter.GetBytes).ToArray();
            Assert.Equal(expected, File.ReadAllBytes(path));
            model.Dump([0, 0, 0], 33, 3);
            Assert.Equal(expected, File.ReadAllBytes(path));
            Assert.Equal(new float[] { 1.25f, -2.5f, 3.75f, 999, 998, 997 }, values);
        });
    }

    [Fact]
    public void OrdinaryDumpStillWritesTheFullArrayAndDisabledDumpDoesNoFileWork()
    {
        WithModel((model, path) =>
        {
            using var environment = new EnvScope();
            environment.Set("TS_DUMP_LOGITS", null);
            environment.Set("TS_DUMP_LOGITS_AT", null);
            float[] values = [1, 2, 3, 4];
            Assert.Same(values, model.Dump(values, 1));
            Assert.False(File.Exists(path));
            environment.Set("TS_DUMP_LOGITS", path);
            Assert.Same(values, model.Dump(values, 1));
            Assert.Equal(values.SelectMany(BitConverter.GetBytes).ToArray(), File.ReadAllBytes(path));
        });
    }

    private static void WithModel(Action<DumpModel, string> test)
    {
        string prefix = Path.Combine(Path.GetTempPath(), "ts-logits-dump-" + Guid.NewGuid().ToString("N"));
        string modelPath = prefix + ".gguf", dumpPath = prefix + ".bin";
        try
        {
            using (var writer = new BinaryWriter(File.Create(modelPath)))
            {
                writer.Write(0x46554747u); writer.Write(3u);
                writer.Write(0UL); writer.Write(0UL); writer.Write(new byte[8]);
            }
            using var model = new DumpModel(modelPath);
            test(model, dumpPath);
        }
        finally { File.Delete(modelPath); File.Delete(dumpPath); }
    }

    private sealed class DumpModel(string path) : ModelBase(path, BackendType.Cpu)
    {
        public float[] Dump(float[] logits, int position, int? count = null, bool warmup = false)
        {
            _cacheSeqLen = position; _inWarmup = warmup;
            return DumpLogitsIfRequested(logits, count);
        }
        protected override float[] ForwardCore(int[] tokens) => [];
        protected override void ResetKVCacheCore() { }
    }
}
