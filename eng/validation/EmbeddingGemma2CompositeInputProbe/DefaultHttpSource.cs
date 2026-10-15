using System.Text.Json;

internal static class DefaultHttpSource
{
    // Exact nonsecret selector set from embeddinggemma2_http_environment.py.
    // Refuse missing/new selectors rather than silently accepting an old audit.
    internal static readonly string[] RequiredOptimizationFlags =
    [
        "NVIDIA_TF32_OVERRIDE", "TS_CPU_DISABLE_AVX512", "TSG_EMBEDDING_F32_LINEAR",
        "TSG_EMBEDDING_LINUX_BLAS", "TSG_EMBEDDING_BLAS_LIBRARY", "TSG_EMBEDDING_CPU_ATTENTION",
        "TSG_EMBEDDING_CPU_PROFILE", "TSG_EMBEDDING_CPU_ATTENTION_PROFILE", "TSG_EMBEDDING_CPU_ATTENTION_SIMD",
        "TSG_CUPTI_ACTIVITY_OUTPUT", "TENSORSHARP_EMBEDDING_CPU_PROFILE", "TENSORSHARP_EMBEDDING_FUSED_GATE_GELU",
        "TENSORSHARP_EMBEDDING_NATIVE_ATTENTION", "TENSORSHARP_EMBEDDING_NATIVE_Q16", "TENSORSHARP_EMBEDDING_Q16_AVX512",
        "TENSORSHARP_EMBEDDING_Q8_AVX", "TS_EMBEDDING_CUDA_MATERIALIZED_ATTENTION",
        "TS_EMBEDDING_CUDA_MATERIALIZED_ATTENTION_TF32", "TS_EMBEDDING_CUDA_NORMALIZED_HALF", "TS_EMBEDDING_CUDA_PHASES",
        "TS_EMBEDDING_AUDIO_LOG_POLICY", "TS_GEMMA4_AUDIO_SSCP_GRAPH", "TS_GEMMA4_AUDIO_BLOCK_GRAPH",
        "TS_GEMMA4_AUDIO_PRECISE_ROWS", "TS_GEMMA4_AUDIO_METAL_F32_MM", "TS_GEMMA4_AUDIO_METAL_MPS_MM",
        "TS_GEMMA4_AUDIO_METAL_FUSED_OPS", "TS_GEMMA4_AUDIO_METAL_SIMD_MM", "TS_GEMMA4_AUDIO_METAL_PROFILE",
        "TS_GEMMA4_AUDIO_CUDA_PRECISE_SSCP", "TS_GEMMA4_AUDIO_CUDA_PRECISE_LINEAR", "TS_GEMMA4_AUDIO_GGML_CUDA_PEDANTIC",
        "TS_GEMMA4_VISION_DEVICE", "TS_GEMMA4_VISION_CUDA_DEVICE", "TS_GEMMA4_VISION_CPU_Q16",
        "TS_GEMMA4_VISION_METAL_PRECISE", "TS_GEMMA4_VISION_METAL_MPS_MM", "TS_GEMMA4_VISION_METAL_SPLIT_MM",
        "TS_GEMMA4_VISION_METAL_MIXED_MPS", "TS_GEMMA4_VISION_SPATIAL_DEVICE", "TS_GEMMA4_VISION_SPLIT_PROJECTION",
        "TS_GEMMA4_VISION_SIMD_PROJECTION", "TS_GEMMA4_VISION_SPLIT_ATTENTION", "TS_VISION_STAGE", "TS_VISION_TRACE", "TS_VISION_PROF",
    ];

    private static void Require(bool value, string message)
    {
        if (!value) throw new InvalidDataException(message);
    }

    private static void True(JsonElement value, string name) =>
        Require(value.GetProperty(name).ValueKind == JsonValueKind.True, "Default HTTP proof requires " + name + " true.");

    private static void NullSelectors(JsonElement audit, string name)
    {
        var values = audit.GetProperty(name);
        Require(values.ValueKind == JsonValueKind.Object, "Missing exact selector audit.");
        var properties = values.EnumerateObject().ToArray();
        var expected = RequiredOptimizationFlags.ToHashSet(StringComparer.Ordinal);
        Require(properties.Length == expected.Count && properties.Select(x => x.Name).ToHashSet(StringComparer.Ordinal).SetEquals(expected),
            "Default HTTP proof requires the complete current selector audit, including CUPTI.");
        Require(properties.All(x => x.Value.ValueKind == JsonValueKind.Null), "Default HTTP proof requires every optimization selector/source absent.");
    }

    internal static void Validate(JsonElement report)
    {
        True(report, "completed"); True(report, "passed"); True(report, "require_embedding_default");
        Require(report.GetProperty("server_exit_code").GetInt32() == 0, "HTTP server did not exit successfully.");
        Require(report.GetProperty("audio_log_policy").GetString() == "HuggingFaceAddFloor", "HTTP report is not the canonical HF audio policy.");
        var audit = report.GetProperty("environment_audit");
        True(audit, "require_embedding_default"); True(audit, "resolved_server_environment");
        Require(audit.GetProperty("scope").GetString() == "direct host child environment", "HTTP child environment is unresolved.");
        Require(audit.GetProperty("audio_log_policy").GetString() == "HuggingFaceAddFloor", "HTTP environment audio policy disagrees.");
        NullSelectors(audit, "optimization_flags"); NullSelectors(audit, "optimization_flag_sources");
        Require(report.GetProperty("binary_artifacts").ValueKind == JsonValueKind.Array && report.GetProperty("binary_artifacts").GetArrayLength() > 0,
            "HTTP proof requires recorded inference artifacts.");
        var cases = report.GetProperty("cases");
        Require(cases.ValueKind == JsonValueKind.Array && cases.GetArrayLength() > 0, "HTTP report has no cases.");
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var item in cases.EnumerateArray())
        {
            Require(item.GetProperty("name").GetString() is string name && name.Length > 0 && names.Add(name), "HTTP case names must be unique.");
            var parts = item.GetProperty("parts");
            Require(parts.ValueKind == JsonValueKind.Array && parts.GetArrayLength() > 0, "HTTP case has no parts.");
            foreach (var part in parts.EnumerateArray())
                Require(part.GetProperty("kind").GetString() is "text" or "image" or "audio" or "video" &&
                    part.GetProperty("value").ValueKind == JsonValueKind.String, "Unexpected HTTP media part.");
        }
        if (report.TryGetProperty("binary_artifacts_unchanged", out var unchanged))
            Require(unchanged.ValueKind == JsonValueKind.True, "HTTP inference artifacts changed during execution.");
    }

    internal static void MatchManifest(JsonElement report, JsonElement manifest)
    {
        var actual = report.GetProperty("cases").EnumerateArray().ToArray();
        var original = manifest.GetProperty("cases").EnumerateArray().ToArray();
        Require(actual.Length == original.Length, "HTTP cases differ from the bound manifest.");
        for (int i = 0; i < actual.Length; ++i)
        {
            Require(actual[i].GetProperty("name").GetString() == original[i].GetProperty("name").GetString(), "HTTP case order/name differs from manifest.");
            var left = actual[i].GetProperty("parts").EnumerateArray().ToArray();
            var right = original[i].GetProperty("parts").EnumerateArray().ToArray();
            Require(left.Length == right.Length, "HTTP media sequence differs from manifest.");
            for (int j = 0; j < left.Length; ++j)
                Require(left[j].GetProperty("kind").GetString() == right[j].GetProperty("kind").GetString() &&
                    left[j].GetProperty("value").GetString() == right[j].GetProperty("value").GetString(), "HTTP part differs from manifest.");
        }
    }
}
