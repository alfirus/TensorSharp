using System.Text.Json;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Shell;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;

namespace TensorAgent.Tests;

[Collection(ProcessEnvironmentCollection.Name)]
public sealed class TopSkillsHostTests
{
    [Theory]
    [InlineData(AgentExecutionMode.Auto)]
    [InlineData(AgentExecutionMode.InProcess)]
    public async Task AppHostExposesTheImportedCatalogAndReadsItsWorkflows(AgentExecutionMode execution)
    {
        DirectoryInfo? repository = new(AppContext.BaseDirectory);
        while (repository != null && !Directory.Exists(Path.Combine(repository.FullName, "TensorAgent", "skills")))
            repository = repository.Parent;
        Assert.NotNull(repository);
        string root = Path.Combine(Path.GetTempPath(), "ta-top-skills-" + Guid.NewGuid().ToString("N"));
        try
        {
            var paths = new AgentPaths(Path.Combine(root, "data"), Path.Combine(root, "cache"))
            {
                BundledSkillsDirectory = Path.Combine(repository!.FullName, "TensorAgent", "skills"),
                DeviceClass = DeviceClass.Desktop,
                ExecutionMode = execution,
            };
            using var host = new AgentAppHost(paths);
            string[] imported = ["find-skills", "grill-me", "grill-with-docs", "tdd", "handoff"];
            foreach (string name in imported)
            {
                Assert.True(host.Skills.TryGet(name, out Skill workflow));
                SkillToolResult workflowRead = SkillTools.Execute(new ToolCall { Name = SkillTools.ReadToolName,
                    Arguments = new Dictionary<string, object?> { ["skill"] = name, ["path"] = "SKILL.md" } },
                    new SkillToolContext([workflow]));
                Assert.True(workflowRead.Ok, workflowRead.Content);
                Assert.Contains("Host capabilities", workflowRead.Content, StringComparison.Ordinal);
            }
            Skill selected = Assert.Single(host.Skills.Skills, skill => skill.Id == "grill-me");
            SkillToolResult read = SkillTools.Execute(new ToolCall { Name = SkillTools.ReadToolName,
                Arguments = new Dictionary<string, object?> { ["skill"] = selected.Id, ["path"] = "SKILL.md" } },
                new SkillToolContext([selected]));
            Assert.True(read.Ok, read.Content);
            Assert.Contains("frontier", read.Content, StringComparison.Ordinal);
            host.Server.Start();
            using var client = new HttpClient { BaseAddress = new Uri(host.Server.BaseUrl) };
            client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={host.Server.Token}");
            using JsonDocument catalog = JsonDocument.Parse(await client.GetStringAsync("api/skills"));
            foreach (string name in imported)
                Assert.Contains(catalog.RootElement.GetProperty("skills").EnumerateArray(),
                    item => item.GetProperty("id").GetString() == name);
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }
}
