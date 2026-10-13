// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using TensorSharp.AgentHost.Knowledge;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat;
using TensorSharp.Knowledge;
using TensorSharp.Server.ResponseSerializers;
using TensorSharp.Server.Skills;

namespace TensorSharp.Server.Knowledge;

/// <summary>Per-request knowledge capability and compatible protocol extensions. Never reused across requests.</summary>
internal sealed class KnowledgeProtocolRequest(KnowledgeToolScope? scope)
{
    public KnowledgeToolScope? Scope { get; } = scope;
    public bool Enabled => Scope != null;
    public int CitationRepairs { get; private set; }
    public SkillRequestPlan? Plan { get; set; }
    private readonly Dictionary<int, (string Output, bool Truncated)> _toolOutput = new();
    private string? _outputTool;
    private readonly StringBuilder _pendingOutput = new();
    private int _outputBytes;
    private bool _outputTruncated;
    private string _answer = "";

    /// <summary>Record bounded stdout only from executed host tools, never model claims.</summary>
    public void Observe(ChatStreamUpdate update)
    {
        if (!Enabled || Plan == null || update.ToolProgressPhase == null) return;
        if (update.ToolProgressPhase == "running")
        {
            if (_outputTool != update.ToolProgressName) ResetOutput(update.ToolProgressName);
            if (!_outputTruncated && update.ToolProgressPiece is { Length: > 0 } piece)
            {
                foreach (var rune in piece.EnumerateRunes())
                {
                    if (_outputBytes + rune.Utf8SequenceLength > 512) { _outputTruncated = true; break; }
                    _pendingOutput.Append(rune.ToString());
                    _outputBytes += rune.Utf8SequenceLength;
                }
            }
        }
        else if (update.ToolProgressPhase == "finished")
        {
            lock (Plan.Invocations)
            {
                int index = Plan.Invocations.Count - 1;
                if (index is >= 0 and < 16 && Plan.Invocations[index].Tool == update.ToolProgressName
                    && _outputTool == update.ToolProgressName)
                    _toolOutput[index] = (_pendingOutput.ToString(), _outputTruncated);
            }
            ResetOutput(null);
        }
    }

    private void ResetOutput(string? tool)
    {
        _outputTool = tool;
        _pendingOutput.Clear();
        _outputBytes = 0;
        _outputTruncated = false;
    }

    public static async Task<KnowledgeProtocolRequest?> ParseAsync(HttpContext context, JsonElement body,
        KnowledgeStore? store, int contextTokens, bool ollama = false, IKnowledgeReranker? reranker = null)
    {
        try { return new(KnowledgeRequestContext.Parse(body, store, contextTokens, reranker)); }
        catch (WebUiRequestRejectedException ex)
        {
            await ErrorAsync(context, ex.StatusCode, ex.Message, ollama).ConfigureAwait(false);
            return null;
        }
    }

    public async Task<List<ChatMessage>?> PrepareAsync(HttpContext context, List<ChatMessage> messages,
        bool toolsAvailable, bool ollama = false, Func<List<ChatMessage>, bool>? completeDocumentFits = null)
    {
        try { return await KnowledgeRequestContext.PrepareAsync(messages, Scope, toolsAvailable, context.RequestAborted,
            completeDocumentFits).ConfigureAwait(false); }
        catch (Exception ex) when (ex is ArgumentException or KeyNotFoundException or KnowledgeStaleReferenceException)
        {
            await ErrorAsync(context, 400, "The selected knowledge sources changed or could not be retrieved. Select the collections again.", ollama).ConfigureAwait(false);
            return null;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            await ErrorAsync(context, 503, "The knowledge library is temporarily unavailable.", ollama).ConfigureAwait(false);
            return null;
        }
    }

    public string Normalize(string text, bool structured = false)
    {
        if (structured) { _answer = text; return text; }
        var normalized = KnowledgeTools.NormalizeCitationUrls(text, Scope);
        CitationRepairs += normalized.RepairCount;
        _answer = normalized.Text;
        return normalized.Text;
    }

    public object WithSources(object payload)
    {
        if (Scope == null) return payload;
        var result = JsonSerializer.SerializeToNode(payload, JsonOptions.IgnoreNulls)!.AsObject();
        result["knowledge_sources"] = JsonSerializer.SerializeToNode(KnowledgeCitations.UsedSources(_answer, Scope).Select(KnowledgeTools.DescribeSource));
        result["knowledge_citation_repairs"] = CitationRepairs;
        SkillToolInvocation[] steps;
        bool stepsTruncated = false;
        if (Plan == null) steps = Array.Empty<SkillToolInvocation>();
        else lock (Plan.Invocations)
        {
            stepsTruncated = Plan.Invocations.Count > 16;
            steps = Plan.Invocations.Take(16).ToArray();
        }
        result["knowledge_tool_steps"] = JsonSerializer.SerializeToNode(steps.Select((step, index) =>
        {
            _toolOutput.TryGetValue(index, out var preview);
            return new { tool = step.Tool, round = step.Round, ok = step.Ok, resultBytes = step.ResultBytes,
                output = preview.Output ?? "", outputTruncated = preview.Truncated };
        }));
        result["knowledge_tool_steps_truncated"] = stepsTruncated;
        return result;
    }

    private static Task ErrorAsync(HttpContext context, int status, string message, bool ollama)
    {
        context.Response.StatusCode = status;
        return context.Response.WriteAsJsonAsync(ollama ? (object)new { error = message }
            : new { error = new { message, type = status < 500 ? "invalid_request_error" : "server_error" } });
    }
}
