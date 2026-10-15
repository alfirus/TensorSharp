// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.

using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;

string repository = Directory.GetCurrentDirectory();
string findSkills = Path.Combine(repository, "TensorAgent", "skills", "find-skills");
if (!File.Exists(Path.Combine(findSkills, "SKILL.md")))
    throw new ArgumentException("Run from the TensorSharp repository root so the bundled find-skills can be located.");
string output = Path.GetFullPath(args.Length > 0 ? args[0] : Path.Combine(repository,
    "artifacts", "find-skills", "live-" + DateTimeOffset.UtcNow.ToString("yyyyMMdd-HHmmss")));
Directory.CreateDirectory(output);
string installDirectory = Path.Combine(output, "installed");
if (Directory.Exists(installDirectory) && Directory.EnumerateFileSystemEntries(installDirectory).Any())
    throw new ArgumentException("Use a fresh output directory: this probe deliberately tests first-time installation.");

var json = new JsonSerializerOptions { WriteIndented = true };
var calls = new List<object>();
var packages = new List<object>();
var report = new Dictionary<string, object?>
{
    ["startedUtc"] = DateTimeOffset.UtcNow,
    ["platform"] = RuntimeInformation.OSDescription,
    ["architecture"] = RuntimeInformation.ProcessArchitecture.ToString(),
    ["dotnet"] = Environment.Version.ToString(),
    ["outputDirectory"] = output,
    ["calls"] = calls,
    ["packages"] = packages,
    ["limitations"] = new[]
    {
        "Live upstream availability, content, and latency can change. Exact revisions and installed file hashes are recorded.",
        "Two actual upstream packages and one artifact-producing Python helper are covered; arbitrary marketplace skills are not universally compatible.",
        "Warm search timings are in-process cache measurements, not network service throughput or concurrent-load benchmarks.",
        "This probe directly invokes the production tools and script runtime. Real model behavior is measured separately by model_probe.py.",
        "No CUDA, Linux, Windows, iOS, Android, or TensorAgent native GUI execution is claimed by this run.",
    },
    ["passed"] = false,
};

try
{
    var registry = new SkillRegistry(new SkillRegistryOptions
    {
        Roots = new[] { findSkills },
        InstallDirectory = installDirectory,
    });
    Require(registry.TryGet("find-skills", out _), "The bundled find-skills did not load.");
    var session = SkillMarketplaceSession.Create(registry, registry.Skills, allowNetwork: true)
        ?? throw new InvalidOperationException("Native marketplace session was not enabled.");
    var manager = new SessionWorkspaceManager(Path.Combine(output, "workspaces"));
    SessionWorkspace workspace = manager.GetOrCreate("find-skills-live-probe");
    var runner = new SkillScriptRunner(new SkillScriptRunnerOptions
    {
        Workspace = workspace,
        Sandbox = SkillSandboxMode.Required,
        AllowNetwork = false,
        Timeout = TimeSpan.FromSeconds(30),
        MaxOutputBytes = 64 * 1024,
    });
    report["scriptSandbox"] = SkillSandboxFactory.DescribeHost();
    Require(runner.CanRun, runner.UnavailableReason ?? "The required script sandbox is unavailable.");
    var context = new SkillToolContext(registry.Skills)
    {
        Marketplace = session,
        ScriptRunner = runner,
        Workspace = workspace,
    };
    var declared = SkillTools.Merge(null, allowScripts: true, out _, session);
    Require(new[] { "skills_search", "skills_inspect", "skills_install", "skills_read", "skills_run" }
        .All(name => declared.Any(tool => tool.Name == name)), "A required tool was not declared.");

    SkillToolResult Execute(string label, string name, params (string Key, object Value)[] arguments)
    {
        var call = new ToolCall { Name = name, Arguments = arguments.ToDictionary(pair => pair.Key, pair => (object?)pair.Value) };
        var watch = Stopwatch.StartNew();
        SkillToolResult result = SkillTools.Execute(call, context);
        watch.Stop();
        calls.Add(new { label, tool = name, arguments = call.Arguments, milliseconds = watch.Elapsed.TotalMilliseconds,
            result.Ok, result.Content, result.SkillId, result.ResourcePath });
        Console.WriteLine($"{label}: {(result.Ok ? "ok" : "FAILED")} {watch.Elapsed.TotalMilliseconds:F2} ms");
        Require(result.Ok, $"{label}: {result.Content}");
        return result;
    }

    Skill Install(string query, string owner, string package)
    {
        var search = Execute(package + " search", "skills_search", ("query", query), ("owner", owner));
        Require(search.Content.Contains(package, StringComparison.Ordinal), $"Search did not return {package}.");
        var inspection = Execute(package + " inspect", "skills_inspect", ("package", package));
        using var inspectedJson = JsonDocument.Parse(inspection.Content);
        string revision = Property(inspectedJson.RootElement, "Revision").GetString()!;
        Require(Regex.IsMatch(revision, "^[a-fA-F0-9]{40}$"), "Inspection did not return an immutable Git commit.");
        var install = Execute(package + " install", "skills_install", ("package", package), ("revision", revision));
        string id = install.SkillId ?? throw new InvalidOperationException("Install did not return a skill ID.");
        Require(registry.TryGet(id, out Skill installed), "Installed skill is missing from the registry.");
        Require(context.TryResolve(id, out _, out _), "Installed skill is not reachable in the same request.");
        var manifest = Execute(package + " manifest", "skills_read", ("skill", id), ("path", "SKILL.md"));
        Require(manifest.Content.Contains(installed.Manifest.Name, StringComparison.OrdinalIgnoreCase),
            "Installed instructions were not returned.");
        var provenance = installed.Files.Select(file => new
        {
            path = file.Path,
            bytes = file.Bytes,
            sha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(Path.Combine(installed.RootDirectory, file.Path)))),
        }).ToArray();
        packages.Add(new { package, revision, id, fileCount = installed.Files.Count, totalBytes = installed.TotalBytes, files = provenance });
        return installed;
    }

    Skill instructions = Install("web-design-guidelines", "vercel-labs", "vercel-labs/agent-skills@web-design-guidelines");
    Require(instructions.Files.All(file => file.Kind != SkillFileKind.Script), "The instruction-only probe package now ships scripts; review this fixture.");
    Skill scripts = Install("skill-creator", "anthropics", "anthropics/skills@skill-creator");
    string helper = "scripts/aggregate_benchmark.py";
    Require(scripts.Files.Any(file => file.Path == helper), "The inspected package does not contain the expected helper.");
    Execute("read actual upstream helper", "skills_read", ("skill", scripts.Id), ("path", helper));
    string reference = scripts.Files.First(file => file.Kind == SkillFileKind.Reference && file.IsText).Path;
    Execute("read actual upstream reference", "skills_read", ("skill", scripts.Id), ("path", reference));

    // Six deterministic inputs exercise arithmetic, file traversal, JSON/Markdown
    // output, and the real session workspace; this is more than a --help smoke.
    foreach (string configuration in new[] { "with_skill", "without_skill" })
    {
        for (int index = 0; index < 3; index++)
        {
            string dir = Path.Combine(workspace.WorkDirectory, "benchmark-input", "eval-1", configuration, $"run-{index + 1}");
            Directory.CreateDirectory(dir);
            double passRate = index == 1 ? 0.5 : configuration == "with_skill" ? 1.0 : 0.0;
            File.WriteAllText(Path.Combine(dir, "grading.json"), JsonSerializer.Serialize(new
            {
                summary = new { pass_rate = passRate, passed = (int)(passRate * 2), failed = 2 - (int)(passRate * 2), total = 2 },
                timing = new { total_duration_seconds = configuration == "with_skill" ? 2.0 : 4.0 },
                execution_metrics = new { output_chars = configuration == "with_skill" ? 200 : 400, total_tool_calls = 1, errors_encountered = 0 },
                expectations = new[] { new { text = "deterministic arithmetic fixture", passed = passRate == 1, evidence = "probe input" } },
            }, json));
        }
    }
    Execute("run actual upstream aggregation", "skills_run", ("skill", scripts.Id), ("path", helper),
        ("args", "benchmark-input --skill-name find-skills-live-probe --output benchmark.json"));
    string benchmarkPath = Path.Combine(workspace.WorkDirectory, "benchmark.json");
    Require(File.Exists(benchmarkPath), "The actual upstream script produced no JSON artifact.");
    Require(File.Exists(Path.ChangeExtension(benchmarkPath, ".md")), "The actual upstream script produced no Markdown artifact.");
    using (var benchmark = JsonDocument.Parse(File.ReadAllText(benchmarkPath)))
    {
        JsonElement root = benchmark.RootElement;
        Require(root.GetProperty("runs").GetArrayLength() == 6, "The actual upstream helper did not aggregate all six runs.");
        JsonElement summary = root.GetProperty("run_summary");
        Require(summary.GetProperty("with_skill").GetProperty("pass_rate").GetProperty("mean").GetDouble() == 0.8333,
            "Incorrect primary mean from the upstream script.");
        Require(summary.GetProperty("without_skill").GetProperty("pass_rate").GetProperty("mean").GetDouble() == 0.1667,
            "Incorrect baseline mean from the upstream script.");
        Require(summary.GetProperty("delta").GetProperty("pass_rate").GetString() == "+0.67", "Incorrect actual upstream delta.");
        report["scriptArtifact"] = root.Clone();
    }
    File.Copy(benchmarkPath, Path.Combine(output, "upstream-benchmark.json"), overwrite: true);
    File.Copy(Path.ChangeExtension(benchmarkPath, ".md"), Path.Combine(output, "upstream-benchmark.md"), overwrite: true);
    var list = Execute("same-request catalog", "skills_list");
    Require(list.Content.Contains(instructions.Id, StringComparison.Ordinal) && list.Content.Contains(scripts.Id, StringComparison.Ordinal),
        "The same-request catalog did not expand after installation.");

    var timings = new List<double>();
    for (int index = 0; index < 20; index++)
    {
        var watch = Stopwatch.StartNew();
        var result = SkillTools.Execute(new ToolCall
        {
            Name = "skills_search",
            Arguments = new() { ["query"] = "skill-creator", ["owner"] = "anthropics" },
        }, context);
        watch.Stop();
        Require(result.Ok && result.Content.Contains("anthropics/skills@skill-creator", StringComparison.Ordinal), "A warm search returned incorrect results.");
        timings.Add(watch.Elapsed.TotalMilliseconds);
    }
    double[] ordered = timings.Order().ToArray();
    report["warmSearch"] = new { samples = timings, count = timings.Count, medianMilliseconds = (ordered[9] + ordered[10]) / 2,
        p95Milliseconds = ordered[18], minimumMilliseconds = ordered[0], maximumMilliseconds = ordered[^1] };
    report["passed"] = true;
}
catch (Exception exception)
{
    report["error"] = exception.ToString();
    Console.Error.WriteLine(exception);
    Environment.ExitCode = 1;
}
finally
{
    report["finishedUtc"] = DateTimeOffset.UtcNow;
    File.WriteAllText(Path.Combine(output, "report.json"), JsonSerializer.Serialize(report, json));
    Console.WriteLine($"Report: {Path.Combine(output, "report.json")}");
}

static void Require(bool condition, string message)
{
    if (!condition) throw new InvalidOperationException(message);
}

static JsonElement Property(JsonElement element, string name) => element.EnumerateObject()
    .First(property => property.Name.Equals(name, StringComparison.OrdinalIgnoreCase)).Value;
