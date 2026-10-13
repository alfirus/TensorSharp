// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using TensorSharp.AgentHost.Agents;
using TensorSharp.AgentHost.Knowledge;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Knowledge;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class KnowledgeAgentScopeTests
{
    [Fact]
    public async Task ReadOnlyDescendantsRetainExactlyTheParentsKnowledgeCapability()
    {
        string root = Path.Combine(Path.GetTempPath(), "knowledge-child-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var store = new KnowledgeStore(root, new BuiltInKnowledgeExtractor());
            var selected = await store.CreateCollectionAsync("selected");
            var hidden = await store.CreateCollectionAsync("not selected");
            await store.ImportAsync(selected.Id, "notes.txt", new MemoryStream(Encoding.UTF8.GetBytes("cobalt orchard rendezvous")));
            await store.ImportAsync(hidden.Id, "secret.txt", new MemoryStream(Encoding.UTF8.GetBytes("cobalt hidden source")));
            var parent = new SkillToolContext(Array.Empty<Skill>())
            { Knowledge = new KnowledgeToolScope(store, new[] { selected.Id }) };
            using var child = AgentWorkspace.Create(parent, "knowledge-child-" + Guid.NewGuid().ToString("N"));
            using var grandchild = AgentWorkspace.Create(child.Context, "knowledge-grandchild-" + Guid.NewGuid().ToString("N"));
            Assert.Same(parent.Knowledge, child.Context.Knowledge);
            Assert.Same(parent.Knowledge, grandchild.Context.Knowledge);
            Assert.True(AgentWorkspace.AllowsTool(parent, false, KnowledgeTools.Search));
            Assert.True(grandchild.AllowsTool(KnowledgeTools.Read));
            Assert.True(MultiAgentTools.IsReadOnlyTool(KnowledgeTools.List));
            Assert.False(AgentWorkspace.AllowsTool(new SkillToolContext(Array.Empty<Skill>()), false, KnowledgeTools.Read));
            var result = SkillTools.Execute(new ToolCall { Name = KnowledgeTools.Search, Arguments = new() { ["query"] = "cobalt" } }, grandchild.Context);
            Assert.True(result.Ok, result.Content);
            Assert.Contains("orchard", result.Content);
            Assert.DoesNotContain("hidden source", result.Content);

            var selectedDocument = store.ListDocuments(selected.Id).Single();
            var openCall = new ToolCall
            {
                Name = KnowledgeTools.Open,
                Arguments = new() { ["collection_id"] = selected.Id, ["document_id"] = selectedDocument.Id, ["version"] = selectedDocument.Version },
            };
            Assert.True(grandchild.AllowsTool(KnowledgeTools.Open));
            var opened = KnowledgeTools.Execute(openCall, grandchild.Context.Knowledge, grandchild.Workspace);
            Assert.True(opened.Ok, opened.Content);
            using var json = System.Text.Json.JsonDocument.Parse(opened.Content);
            string relative = json.RootElement.GetProperty("path").GetString()!;
            Assert.False(Path.IsPathRooted(relative));
            Assert.DoesNotContain(root, opened.Content);
            Assert.DoesNotContain(grandchild.Workspace.WorkDirectory, opened.Content);
            Assert.Equal("cobalt orchard rendezvous", File.ReadAllText(Path.Combine(grandchild.Workspace.WorkDirectory, relative)));
            Assert.False(File.Exists(Path.Combine(child.Workspace.WorkDirectory, relative)));
            var foreignDocument = store.ListDocuments(hidden.Id).Single();
            openCall.Arguments["collection_id"] = hidden.Id;
            openCall.Arguments["document_id"] = foreignDocument.Id;
            Assert.False(KnowledgeTools.Execute(openCall, grandchild.Context.Knowledge, grandchild.Workspace).Ok);
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }
}
