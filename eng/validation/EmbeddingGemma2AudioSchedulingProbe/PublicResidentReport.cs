internal static class PublicResidentReport
{
    // This mode observes public results, not the public method's internal
    // preparation, native callers or timing breakdown. Reject accidental data
    // reuse from the resident path before serializing unavailable observations.
    internal static Dictionary<string, object?> Observations(bool publicApi,
        IReadOnlyCollection<double> prepare, IReadOnlyCollection<double> text,
        IReadOnlyCollection<double> clone, IReadOnlyCollection<object> callers,
        IReadOnlyCollection<CallerIdentity> identities, bool? inputBits)
    {
        if (publicApi && (prepare.Count != 0 || text.Count != 0 || clone.Count != 0 || callers.Count != 0 || identities.Count != 0 || inputBits != null))
            throw new InvalidDataException("Public API internals were not observed and must remain unavailable.");
        if (!publicApi && (prepare.Count == 0 || text.Count == 0 || clone.Count == 0 || callers.Count == 0 || identities.Count == 0 || inputBits == null))
            throw new InvalidDataException("Resident observations must be recorded completely.");
        return new Dictionary<string, object?>
        {
            ["observation_scope"] = publicApi ? "public API result and caller wall time only" : "resident internal synchronous callback",
            ["prepare_samples_ms"] = publicApi ? null : prepare,
            ["text_samples_ms"] = publicApi ? null : text,
            ["clone_samples_ms"] = publicApi ? null : clone,
            ["caller_observations"] = publicApi ? null : callers,
            ["distinct_managed_callers"] = publicApi ? null : identities.Select(x => x.ManagedThreadId).Distinct().Count(),
            ["distinct_native_callers"] = publicApi ? null : identities.Select(x => x.NativeThreadId).Distinct().Count(),
            ["prepared_bit_identity"] = inputBits,
            ["unavailable_observations"] = publicApi ? new[] { "prepared input bytes", "native caller identities", "preparation/clone/text timing breakdown" } : Array.Empty<string>()
        };
    }
}
