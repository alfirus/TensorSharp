// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Globalization;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Server;

/// <summary>A lifecycle-owned session, with a CPU-only test seam for release failures.</summary>
internal sealed class UnifiedMemorySessionHandle
{
    private readonly Func<SchedulerConfig, SchedulerConfig> _configure;
    private readonly Func<string> _featureReport;
    private readonly Action _dispose;

    internal UnifiedMemorySessionHandle(ModelBase model,
        Func<SchedulerConfig, SchedulerConfig> configure, Func<string> featureReport, Action dispose)
    {
        Model = model;
        _configure = configure ?? throw new ArgumentNullException(nameof(configure));
        _featureReport = featureReport ?? throw new ArgumentNullException(nameof(featureReport));
        _dispose = dispose ?? throw new ArgumentNullException(nameof(dispose));
    }

    internal static UnifiedMemorySessionHandle Wrap(IUnifiedMemoryModelSession session)
        => new(session.Model, session.ConfigureScheduler, () => session.FeatureReport, session.Dispose);

    internal static UnifiedMemorySessionHandle CreateOwned(string path, string projector, BackendType backend)
    {
        // Resolve explicit host tuning before any model/native allocation. The
        // same prefill cap must govern load forecasts and the final scheduler;
        // applying only SchedulerConfig would retain the wrong memory plan.
        var options = ResolveHostOptions();
        var session = UnifiedMemoryModelSession.Create(path, projector, backend, options);
        try { return Wrap(session); }
        catch (Exception creation)
        {
            // Creating this managed adapter happens after the model owns real
            // buffers. Even failure here must not orphan the original session.
            try { session.Dispose(); }
            catch (Exception cleanup)
            { throw new UnifiedMemoryModelLoadException(session, new AggregateException(creation, cleanup)); }
            throw;
        }
    }

    internal static UnifiedMemoryModelOptions ResolveHostOptions()
    {
        const string chunkVariable = "TS_SCHED_PREFILL_CHUNK";
        const string graphCapVariable = "TS_GLM_LAYER_MAJOR_GRAPH_CAP_MB";
        const string hostEmbeddingVariable = "TS_GLM_HOST_TOKEN_EMBEDDING";
        // This explicit experimental profile is single-primary and cannot
        // retain conversation/prefix slots. Resolve before any model/native
        // allocation; ordinary Host defaults remain independent of the flag.
        string graphCap = Environment.GetEnvironmentVariable(graphCapVariable);
        bool strictGraphCap = graphCap switch
        {
            null or "0" => false,
            "512" or "768" => true,
            _ => throw new ArgumentException(graphCapVariable + " must be unset, 0, 512 or 768.")
        };
        bool hostEmbedding = Environment.GetEnvironmentVariable(hostEmbeddingVariable) switch
        {
            null or "0" => false,
            "1" => true,
            _ => throw new ArgumentException(hostEmbeddingVariable + " must be unset, 0 or 1.")
        };
        bool singlePrimary = strictGraphCap || hostEmbedding;
        string profileVariable = strictGraphCap ? graphCapVariable : hostEmbeddingVariable;
        // This path is already an explicit --unified-memory opt-in. GLM's
        // matching native planner admits its serialized selected-expert tier;
        // Qwen ignores this architecture-specific option. Public API defaults
        // stay independent of this Host policy.
        var scheduler = SchedulerConfig.FromEnvironment();
        if (singlePrimary && scheduler.EnablePrefixCaching
            && Environment.GetEnvironmentVariable("TS_SCHED_PREFIX_CACHE") != null)
            throw new ArgumentException(profileVariable + " requires prefix caching disabled; remove its explicit enable setting or use --no-prefix-cache.");
        // Capacity growth lets Qwen's per-layer expert tables regrow, with
        // their hot slots, after a request boundary trims them for a prefill.
        var options = new UnifiedMemoryModelOptions { EnableGlmSelectedExperts = true,
            EnableAdaptiveGlmPrefill = true, AdaptiveGlmPrefillChunkTokens = 1024,
            EnableExpertCacheCapacityGrowth = true,
            PrefixCheckpointCount = scheduler.EnablePrefixCaching
                ? (scheduler.ExecutionOptionsOverride ?? ExecutionOptions.FromEnvironment()).PrefixCheckpointBudget : 0 };
        if (singlePrimary)
            options = options with { MaximumRunningRequests = 1, RetainedConversationCount = 0,
                PrefixCheckpointCount = 0, AdaptiveGlmPrefillChunkTokens = 512 };
        string chunk = Environment.GetEnvironmentVariable(chunkVariable);
        if (chunk != null)
        {
            if (!int.TryParse(chunk, NumberStyles.None, CultureInfo.InvariantCulture, out int rows)
                || rows is < 1 or > 512)
                throw new ArgumentException("--unified-memory requires TS_SCHED_PREFILL_CHUNK (--prefill-chunk-size) to be an integer from 1 through 512.");
            options = options with { PrefillChunkTokens = rows, AdaptiveGlmPrefillChunkTokens = rows };
        }
        string layerMajor = Environment.GetEnvironmentVariable("TS_UNIFIED_LAYER_MAJOR_TOKENS");
        if (layerMajor != null)
        {
            // 0 restores chunk-major prefill at the admitted width.
            if (!int.TryParse(layerMajor, NumberStyles.None, CultureInfo.InvariantCulture, out int tokens)
                || tokens > 65536)
                throw new ArgumentException("TS_UNIFIED_LAYER_MAJOR_TOKENS must be an integer from 0 through 65536.");
            options = options with { LayerMajorPrefillTokens = tokens };
        }
        string growth = Environment.GetEnvironmentVariable("TS_UNIFIED_EXPERT_CACHE_GROWTH");
        if (growth != null)
        {
            // 0 keeps each trimmed table at its smaller size until a later trim.
            if (growth is not ("0" or "1"))
                throw new ArgumentException("TS_UNIFIED_EXPERT_CACHE_GROWTH must be 0 or 1.");
            options = options with { EnableExpertCacheCapacityGrowth = growth == "1" };
        }
        string hostExperts = Environment.GetEnvironmentVariable("TS_HOST_MOE_RAM_CACHE_MB");
        if (hostExperts != null)
        {
            if (!long.TryParse(hostExperts, NumberStyles.None, CultureInfo.InvariantCulture, out long mib)
                || mib > long.MaxValue / (1L << 20))
                throw new ArgumentException("TS_HOST_MOE_RAM_CACHE_MB must be a nonnegative integer number of MiB within Int64 bytes.");
            options = options with { MaximumHostExpertCacheBytes = mib * (1L << 20) };
        }
        string flashMtp = Environment.GetEnvironmentVariable("TS_GLM_FLASH_MTP");
        if (flashMtp != null)
        {
            if (flashMtp is not ("0" or "1"))
                throw new ArgumentException("TS_GLM_FLASH_MTP must be 0 or 1.");
            if (singlePrimary && flashMtp == "1")
                throw new ArgumentException(profileVariable + " cannot combine with TS_GLM_FLASH_MTP=1.");
            if (flashMtp == "1")
                options = options with { EnableGlmFlashMtp = true, MaximumRunningRequests = 1,
                    PrefixCheckpointCount = 0, RetainedConversationCount = 0 };
        }
        string running = Environment.GetEnvironmentVariable("TS_UNIFIED_MAX_RUNNING_REQUESTS");
        if (running != null)
        {
            if (!int.TryParse(running, NumberStyles.None, CultureInfo.InvariantCulture, out int count)
                || count is < 1 or > 3 || ((options.EnableGlmFlashMtp || singlePrimary) && count != 1))
                throw new ArgumentException("TS_UNIFIED_MAX_RUNNING_REQUESTS must be 1 through 3 (exactly 1 for experimental Flash MTP, strict layer-major graph caps or host token embedding).");
            options = options with { MaximumRunningRequests = count };
        }
        if (strictGraphCap && (options.PrefillChunkTokens != 512 || options.AdaptiveGlmPrefillChunkTokens != 512
            || options.LayerMajorPrefillTokens <= 512))
            throw new ArgumentException(graphCapVariable + " requires prefill chunk 512 and TS_UNIFIED_LAYER_MAJOR_TOKENS greater than 512.");
        foreach (string variable in new[] { "TS_PER_SEQ_FUSED", "TS_BATCHED_FUSED_DECODE" })
        {
            string value = Environment.GetEnvironmentVariable(variable);
            // Match ExecutionOptions' recognized false values. Qualified
            // unified execution needs both paths; refuse an explicit opt-out
            // instead of silently overriding it in ConfigureScheduler.
            if (!options.EnableGlmFlashMtp && (value == "0" || string.Equals(value, "false", StringComparison.OrdinalIgnoreCase)))
                throw new NotSupportedException($"--unified-memory requires {variable} to remain enabled; remove its explicit disable setting or use the ordinary model-loading mode.");
        }
        return options;
    }

    internal ModelBase Model { get; }
    internal SchedulerConfig ConfigureScheduler(SchedulerConfig config) => _configure(config);
    internal string FeatureReport => _featureReport();
    internal void Dispose() => _dispose();
}
