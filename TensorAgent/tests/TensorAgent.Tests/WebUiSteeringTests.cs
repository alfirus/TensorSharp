// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System.Text.Json;

namespace TensorAgent.Tests;

public sealed partial class WebUiPageTests
{
    [WebJavaScriptTheory]
    [InlineData("token", "Partial answer")]
    [InlineData("thinking", "Useful reasoning")]
    public void EnterSteersWhileTheResponseIsActiveAndKeepsItsPartialHistory(string kind, string partial)
    {
        JsonElement result = Run($$"""
            var sends = 0;
            R['/api/chat'] = function (call) {
              sends++;
              if (sends === 1) return { __status: 200, headers: { 'X-TensorAgent-Turn': 'old' },
                body: { __sse: [{ {{kind}}: '{{partial}}' }], __then: 'hang' } };
              return { __status: 200, headers: { 'X-TensorAgent-Turn': 'new' }, body: { __sse: [
                { steeringMessages: [{ role: 'user', content: 'Explain' },
                  { role: 'assistant', content: '{{(kind == "token" ? partial : "")}}', thinking: '{{(kind == "thinking" ? partial : "")}}' },
                  { role: 'user', content: 'Focus on examples' }] },
                { token: 'An example' }, { done: true }
              ] } };
            };
            """, """
            var input = __page.byId['text'];
            input.value = 'Explain';
            __page.byId['send'].dispatch('click');
            return settle(10).then(function () {
              var active = window.TensorAgent.isGenerating();
              input.value = 'Focus on examples';
              input.dispatch('input');
              var title = __page.byId['send'].title;
              input.dispatch('keydown', { key: 'Enter', shiftKey: false, repeat: false, isComposing: false,
                preventDefault: function () {} });
              return settle(15).then(function () {
                return { active: active, title: title, sent: __page.requests('/api/chat').map(function (call) { return call.body; }),
                  history: window.TensorAgent.history(), transcript: __page.transcript(),
                  stopped: __page.calls.filter(function (call) { return /\/stop$/.test(call.path); }).length,
                  generating: window.TensorAgent.isGenerating() };
              });
            });
            """);
        Assert.True(result.GetProperty("active").GetBoolean());
        Assert.Equal("Steer response", result.GetProperty("title").GetString());
        Assert.Equal(2, result.GetProperty("sent").GetArrayLength());
        JsonElement steering = result.GetProperty("sent")[1];
        Assert.True(steering.GetProperty("steer").GetBoolean());
        Assert.Equal(partial, steering.GetProperty("messages")[1].GetProperty(kind == "token" ? "content" : "thinking").GetString());
        Assert.Equal(new[] { "Explain", kind == "token" ? partial : "", "Focus on examples", "An example" },
            result.GetProperty("history").EnumerateArray().Select(message => message.GetProperty("content").GetString()));
        Assert.Equal(0, result.GetProperty("stopped").GetInt32());
        Assert.False(result.GetProperty("generating").GetBoolean());
    }

    [WebJavaScriptTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void ARecoveryLookupFromTheInterruptedResponseCannotReplaceTheSteeredResponse(bool running)
    {
        JsonElement result = Run($$"""
            var sends = 0, answerLookup;
            R['/api/chat'] = function () {
              sends++;
              return { __status: 200, headers: { 'X-TensorAgent-Turn': sends === 1 ? 'old' : 'new' },
                body: { __sse: [{ token: sends === 1 ? 'Old partial' : 'New answer' }],
                  __then: sends === 1 ? 'reject' : 'hang' } };
            };
            R['/api/agent/turns'] = function () { return new Promise(function (answer) { answerLookup = answer; }); };
            R['/api/agent/turns/old'] = { __sse: [{ token: 'Stale replay' }, { done: true }] };
            """, $$"""
            __page.byId['text'].value = 'Original';
            __page.byId['send'].dispatch('click');
            return settle(10).then(function () {
              window.TensorAgent.resumeTurn();
              return settle(10).then(function () {
                __page.byId['text'].value = 'New direction';
                __page.byId['send'].dispatch('click');
                return settle(10).then(function () {
                  answerLookup({ turn: {{(running ? "{ id: 'old', running: true }" : "null")}} });
                  return settle(10).then(function () {
                    var result = { sent: __page.requests('/api/chat').length, generating: window.TensorAgent.isGenerating(),
                      oldReaders: __page.requests('/api/agent/turns/old').length,
                      sessions: __page.requests('/api/sessions').length, transcript: __page.transcript() };
                    window.TensorAgent.stop();
                    return result;
                  });
                });
              });
            });
            """);
        Assert.Equal(2, result.GetProperty("sent").GetInt32());
        Assert.True(result.GetProperty("generating").GetBoolean());
        Assert.Equal(0, result.GetProperty("oldReaders").GetInt32());
        Assert.Equal(1, result.GetProperty("sessions").GetInt32());
        Assert.Contains("New answer", result.GetProperty("transcript").GetRawText(), StringComparison.Ordinal);
        Assert.DoesNotContain("Stale replay", result.GetProperty("transcript").GetRawText(), StringComparison.Ordinal);
    }

    [WebJavaScriptFact]
    public void ANewPromptAfterStopRetainsTheVisiblePartialAnswerAndWaitsForTheStoppedTurn()
    {
        JsonElement result = Run("""
            var sends = 0;
            R['/api/chat'] = function () {
              sends++;
              return { __status: 200, headers: { 'X-TensorAgent-Turn': 'turn-' + sends },
                body: { __sse: [{ token: sends === 1 ? 'Partial' : 'Continuation' }], __then: 'hang' } };
            };
            R['/api/agent/turns/turn-1/stop'] = { stopped: true };
            """, """
            __page.byId['text'].value = 'Original';
            __page.byId['send'].dispatch('click');
            return settle(10).then(function () {
              __page.byId['send'].dispatch('click');
              __page.byId['text'].value = 'New direction';
              __page.byId['send'].dispatch('click');
              return settle(10).then(function () {
                var sent = __page.requests('/api/chat').map(function (call) { return call.body; });
                window.TensorAgent.stop();
                return { sent: sent };
              });
            });
            """);
        JsonElement second = result.GetProperty("sent")[1];
        Assert.True(second.GetProperty("steer").GetBoolean());
        Assert.Equal("Partial", second.GetProperty("messages")[1].GetProperty("content").GetString());
    }

    [WebJavaScriptFact]
    public void SteeringBeforeSharedDraftHeadersKeepsAttachmentsAndAcknowledgesTheDraftOnceAccepted()
    {
        JsonElement result = Run("""
            var shareTaken = false, sends = 0;
            R['/api/agent/share/claim'] = function () {
              return shareTaken ? { share: null } : { share: { id: 'shared-1', text: 'Explain the shared file', newChat: false,
                attachments: [{ ok: true, file: 'shared.txt', fileName: 'shared.txt', mediaType: 'text', url: '/uploads/shared.txt' }] } };
            };
            R['/api/chat'] = function (call) {
              sends++;
              shareTaken = true;
              if (sends === 1) return new Promise(function () {});
              return { __status: 200, headers: { 'X-TensorAgent-Turn': 'new' },
                body: { __sse: [{ steeringMessages: call.body.messages }, { token: 'Shared file summary' }, { done: true }] } };
            };
            """, """
            window.TensorAgent.__fromHost('takeShare', 'e30=');
            return wait(120).then(function () {
              __page.byId['send'].dispatch('click');
              return settle(10).then(function () {
                __page.byId['text'].value = 'Use a short summary';
                __page.byId['send'].dispatch('click');
                return settle(10).then(function () {
                  return { sent: __page.requests('/api/chat').map(function (call) { return call.body; }),
                    shares: __page.byId['chips'].querySelectorAll('.shared').length,
                    history: window.TensorAgent.history() };
                });
              });
            });
            """);
        Assert.Equal(2, result.GetProperty("sent").GetArrayLength());
        JsonElement second = result.GetProperty("sent")[1];
        Assert.True(second.GetProperty("steer").GetBoolean());
        Assert.Equal("shared-1", second.GetProperty("shareIds")[0].GetString());
        Assert.Equal("shared.txt", second.GetProperty("messages")[0].GetProperty("attachments")[0].GetProperty("file").GetString());
        Assert.Equal("shared.txt", result.GetProperty("history")[0].GetProperty("attachments")[0].GetProperty("file").GetString());
        Assert.Equal(0, result.GetProperty("shares").GetInt32());
    }

    [WebJavaScriptFact]
    public void ASteerDuringPrefillIgnoresRetiredResponseHeadersAndFrames()
    {
        JsonElement result = Run("""
            var sends = 0, answerOld;
            R['/api/chat'] = function (call) {
              sends++;
              if (sends === 1) return new Promise(function (answer) { answerOld = answer; });
              return { __status: 200, headers: { 'X-TensorAgent-Turn': 'new' },
                body: { __sse: [{ token: 'New response' }], __then: 'hang' } };
            };
            R['/api/agent/turns/new/stop'] = { stopped: true };
            """, """
            __page.byId['text'].value = 'Original';
            __page.byId['send'].dispatch('click');
            return settle(10).then(function () {
              __page.byId['text'].value = 'New direction';
              __page.byId['send'].dispatch('click');
              return settle(10).then(function () {
                answerOld({ __status: 200, headers: { 'X-TensorAgent-Turn': 'old' },
                  body: { __sse: [{ token: 'Stale response' }, { done: true }] } });
                return settle(10).then(function () {
                  var generating = window.TensorAgent.isGenerating();
                  var transcript = __page.transcript();
                  __page.byId['send'].dispatch('click');
                  return settle(10).then(function () {
                    return { sent: __page.requests('/api/chat').length, generating: generating,
                      transcript: transcript, newStops: __page.requests('/api/agent/turns/new/stop').length,
                      oldStops: __page.requests('/api/agent/turns/old/stop').length };
                  });
                });
              });
            });
            """);
        Assert.Equal(2, result.GetProperty("sent").GetInt32());
        Assert.True(result.GetProperty("generating").GetBoolean());
        Assert.DoesNotContain("Stale response", result.GetProperty("transcript").GetRawText(), StringComparison.Ordinal);
        Assert.Contains("New response", result.GetProperty("transcript").GetRawText(), StringComparison.Ordinal);
        Assert.Equal(1, result.GetProperty("newStops").GetInt32());
        Assert.Equal(0, result.GetProperty("oldStops").GetInt32());
    }
}
