// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models;

var options = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i += 2)
{
    if (!args[i].StartsWith("--", StringComparison.Ordinal) || i + 1 >= args.Length)
        throw new ArgumentException("Use --model PATH --manifest PATH [--listen 127.0.0.1:6202] " +
            "[--gpus 2] [--context 4096] [--ubatch 256] [--threads 16].");
    options.Add(args[i][2..], args[i + 1]);
}
int Number(string name, int fallback) => options.TryGetValue(name, out string? value) ? int.Parse(value) : fallback;
DeepSeek41PipelineWorker.Run(options["model"], options.GetValueOrDefault("listen", "127.0.0.1:6202"),
    options["manifest"], Number("gpus", 2), Number("context", 4096), Number("ubatch", 256), Number("threads", 16));
