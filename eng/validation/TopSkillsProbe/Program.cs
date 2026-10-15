using System.Diagnostics;
using System.Text.Json;
using TensorSharp.AgentHost.Skills;

if (args.Length != 2)
    throw new ArgumentException("Usage: TopSkillsProbe <skill-root> <ignored-output.json>");
string skillsRoot = Path.GetFullPath(args[0]);
string output = Path.GetFullPath(args[1]);
using JsonDocument sources = JsonDocument.Parse(File.ReadAllText(Path.Combine(skillsRoot, "top-skills-provenance.json")));
HashSet<string> imported = sources.RootElement.GetProperty("installed").EnumerateArray()
    .Select(e => e.GetProperty("name").GetString()!).ToHashSet(StringComparer.Ordinal);
var full = new SkillRegistry(new SkillRegistryOptions { Roots = [skillsRoot] });
if (full.Errors.Count != 0) throw new InvalidOperationException(string.Join("; ", full.Errors));
if (full.Skills.Count == 0) throw new InvalidOperationException("No skills loaded; an empty catalog cannot validate integration.");
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
string baselineRoot = Path.Combine(Path.GetDirectoryName(output)!, ".catalog-baseline-" + Guid.NewGuid().ToString("N"));
Directory.CreateDirectory(baselineRoot);
foreach (string directory in Directory.EnumerateDirectories(skillsRoot)
    .Where(p => !imported.Contains(Path.GetFileName(p))))
{
    foreach (string file in Directory.EnumerateFiles(directory, "*", SearchOption.AllDirectories))
    {
        string relative = Path.GetRelativePath(skillsRoot, file);
        string copy = Path.Combine(baselineRoot, relative);
        Directory.CreateDirectory(Path.GetDirectoryName(copy)!);
        File.Copy(file, copy);
    }
}

object Measure(string label, IReadOnlyList<string> roots)
{
    const int scanRepeats = 20;
    const int planRepeats = 5000;
    var registry = new SkillRegistry(new SkillRegistryOptions { Roots = roots });
    var scan = new List<double>();
    for (int i = 0; i < scanRepeats; i++)
    {
        var clock = Stopwatch.StartNew();
        var refreshed = registry.Refresh();
        scan.Add(clock.Elapsed.TotalMilliseconds);
        if (refreshed.Errors.Count != 0) throw new InvalidOperationException("Registry refresh failed.");
    }
    var contexts = new List<object>();
    foreach (int context in new[] { 8192, 32768, 131072 })
    {
        var options = new SkillPromptOptions { ContextTokens = context, ToolsAvailable = true };
        SkillPlan plan = SkillPrompt.Plan([], registry.Skills, options);
        _ = SkillPrompt.Plan([], registry.Skills, options); // JIT warmup, excluded.
        long bytesBefore = GC.GetAllocatedBytesForCurrentThread();
        var clock = Stopwatch.StartNew();
        for (int i = 0; i < planRepeats; i++)
            _ = SkillPrompt.Plan([], registry.Skills, options);
        double microseconds = clock.Elapsed.TotalMicroseconds / planRepeats;
        long allocated = (GC.GetAllocatedBytesForCurrentThread() - bytesBefore) / planRepeats;
        contexts.Add(new { context, catalog = plan.Catalog.Count, omitted = plan.OmittedFromCatalog,
            prompt_bytes = System.Text.Encoding.UTF8.GetByteCount(plan.Instructions),
            approximate_prompt_tokens = plan.ApproximateTokens, plan_us_mean = microseconds,
            allocated_bytes_per_plan = allocated });
        if (plan.OmittedFromCatalog != 0) throw new InvalidOperationException($"{label}: {plan.OmittedFromCatalog} skills omitted at {context} context.");
    }
    scan.Sort();
    return new { label, skills = registry.Skills.Count, files = registry.Skills.Sum(s => s.Files.Count),
        stored_bytes = registry.Skills.Sum(s => s.TotalBytes), refresh_ms_median = scan[scan.Count / 2],
        refresh_ms_p95 = scan[(int)Math.Ceiling(scan.Count * .95) - 1],
        scan_repeats = scanRepeats, plan_repeats = planRepeats, contexts };
}

try
{
    var report = new
    {
        measured_at_utc = DateTimeOffset.UtcNow,
        runtime = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription,
        os = System.Runtime.InteropServices.RuntimeInformation.OSDescription,
        processor_count = Environment.ProcessorCount,
        excluded_from_baseline = imported.Order().ToArray(),
        limitations = new[] { "Warm filesystem registry refresh, not a cold boot.", "Catalog tokens use the host's 4 UTF-8 bytes/token approximation, not model tokenization.", "No inference or mobile-device speed is measured by this probe.", "One-root baseline copy excludes the active imports listed in this report; copying is outside the timed region." },
        measurements = new[] { Measure("preexisting", [baselineRoot]), Measure("with_top_skills", [skillsRoot]) },
    };
    File.WriteAllText(output, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine(output);
}
finally { Directory.Delete(baselineRoot, recursive: true); }
