using System.Text.Json;

namespace TensorAgent.Tests;

public sealed partial class WebUiPageTests
{
    private const string KnowledgeChats = """
        var historyOne = { id: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', title: '<img src=x onerror=attack()>', messageCount: 4, updatedAt: '2026-10-10T12:00:00Z' };
        var historyTwo = { id: 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', title: 'Earlier design discussion', messageCount: 2 };
        R['/api/knowledge/chats?offset=0&limit=100'] = { chats: [historyOne], total: 101, nextOffset: 100 };
        R['/api/knowledge/chats?offset=100&limit=100'] = { chats: [historyTwo], total: 101, nextOffset: null };
        function selectHistoryChat(id, checked) {
          var row = __page.byId['knowledge-chat-list'].querySelectorAll('.knowledge-chat-row')
            .filter(function (item) { return item.getAttribute('data-conversation') === id; })[0];
          var choice = row.querySelector('.knowledge-chat-toggle'); choice.checked = checked; choice.dispatch('change');
        }
        function openHistoryPicker() { __page.byId['knowledge-add-chats'].dispatch('click'); return settle(20); }
        """;

    [WebJavaScriptFact]
    public void KnowledgeChatPickerPreservesSelectionAcrossPagesAndImportsOnlySelectedHistory()
    {
        JsonElement result = Run(KnowledgeLibrary + KnowledgeChats + """
            var imports = 0;
            R['/api/knowledge/collections/alpha/chats'] = function (call) {
              imports++; libraryCollections[0].documentCount = 2;
              libraryDocuments = [{ id: 'chat', collectionId: 'alpha', fileName: 'conversation.md',
                relativePath: 'Chats/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/conversation.md', version: 1, chunkCount: 2, status: 'Ready' }];
              return { documents: libraryDocuments, importedChats: [{ id: historyOne.id }, { id: historyTwo.id }], errors: [] };
            };
            """, """
            return openKnowledgeLibrary().then(openHistoryPicker).then(function () {
              var row = __page.byId['knowledge-chat-list'].querySelector('.knowledge-chat-row');
              var safeTitle = { text: row.querySelector('strong').textContent, html: row.querySelector('strong').innerHTML };
              selectHistoryChat(historyOne.id, true);
              __page.byId['knowledge-chat-pagination'].querySelector('.knowledge-chat-next').dispatch('click');
              return settle(20).then(function () {
                selectHistoryChat(historyTwo.id, true);
                __page.byId['knowledge-chat-pagination'].querySelector('.knowledge-chat-previous').dispatch('click');
                return settle(20);
              }).then(function () {
                var checkedAgain = __page.byId['knowledge-chat-list'].querySelector('.knowledge-chat-toggle').checked;
                __page.byId['knowledge-chat-import'].dispatch('click'); __page.byId['knowledge-chat-import'].dispatch('click');
                return settle(35).then(function () {
                  var call = __page.requests('/api/knowledge/collections/alpha/chats')[0];
                  return { title: safeTitle, checkedAgain: checkedAgain, imports: imports, method: call.method,
                    selected: call.body.conversationIds, count: __page.byId['knowledge-chat-count'].textContent,
                    status: __page.byId['knowledge-status'].textContent, documents: __page.byId['knowledge-documents'].textContent,
                    disabled: __page.byId['knowledge-chat-import'].disabled, selectedCollections: __page.byId['knowledgechips'].textContent };
                });
              });
            });
            """);
        Assert.Equal("<img src=x onerror=attack()>", result.GetProperty("title").GetProperty("text").GetString());
        Assert.Equal("", result.GetProperty("title").GetProperty("html").GetString());
        Assert.True(result.GetProperty("checkedAgain").GetBoolean());
        Assert.Equal(1, result.GetProperty("imports").GetInt32());
        Assert.Equal("POST", result.GetProperty("method").GetString());
        Assert.Equal(new[] { new string('a', 32), new string('b', 32) }, Strings(result, "selected"));
        Assert.Equal("0 selected", result.GetProperty("count").GetString());
        Assert.Contains("Indexed chats: 2", result.GetProperty("status").GetString());
        Assert.Contains("Chats/", result.GetProperty("documents").GetString());
        Assert.True(result.GetProperty("disabled").GetBoolean());
        Assert.Equal("", result.GetProperty("selectedCollections").GetString());
    }

    [WebJavaScriptFact]
    public void KnowledgeChatImportKeepsIncompleteChatsSelectedAndShowsMissingAttachmentErrors()
    {
        JsonElement result = Run(KnowledgeLibrary + KnowledgeChats + """
            R['/api/knowledge/chats?offset=0&limit=100'] = { chats: [historyOne, historyTwo] };
            R['/api/knowledge/collections/alpha/chats'] = { importedChats: [{ id: historyOne.id }], errors: [
              { conversationId: historyOne.id, title: historyOne.title, fileName: 'photo.png', error: 'Attachment is unavailable.' },
              { conversationId: historyTwo.id, title: historyTwo.title, error: 'Chat no longer exists.' }] };
            """, """
            return openKnowledgeLibrary().then(openHistoryPicker).then(function () {
              selectHistoryChat(historyOne.id, true); selectHistoryChat(historyTwo.id, true);
              __page.byId['knowledge-chat-import'].dispatch('click'); return settle(25);
            }).then(function () {
              return { choices: __page.byId['knowledge-chat-list'].querySelectorAll('.knowledge-chat-toggle').map(function (box) { return box.checked; }),
                error: __page.byId['knowledge-error'].textContent, count: __page.byId['knowledge-chat-count'].textContent,
                status: __page.byId['knowledge-status'].textContent, retry: !__page.byId['knowledge-chat-import'].disabled };
            });
            """);
        Assert.Equal(new[] { true, true }, result.GetProperty("choices").EnumerateArray().Select(item => item.GetBoolean()));
        Assert.Contains("photo.png: Attachment is unavailable.", result.GetProperty("error").GetString());
        Assert.Contains("Chat no longer exists.", result.GetProperty("error").GetString());
        Assert.Equal("2 selected", result.GetProperty("count").GetString());
        Assert.Contains("Indexed chats: 1 · issues to review: 2", result.GetProperty("status").GetString());
        Assert.True(result.GetProperty("retry").GetBoolean());
    }

    [WebJavaScriptFact]
    public void KnowledgeChatSearchFiltersOnServerAndIgnoresLateResponsesAfterCollectionSwitch()
    {
        JsonElement result = Run(KnowledgeLibrary + KnowledgeChats + Held + """
            R['/api/knowledge/chats?offset=0&limit=100&q=design%20%26%20art'] = { chats: [historyTwo] };
            """, """
            return openKnowledgeLibrary().then(openHistoryPicker).then(function () {
              selectHistoryChat(historyOne.id, true);
              __page.byId['knowledge-chat-query'].value = 'design & art';
              __page.byId['knowledge-chat-query'].dispatch('input'); return wait(230);
            }).then(function () {
              var found = __page.byId['knowledge-chat-list'].textContent;
              selectHistoryChat(historyTwo.id, true);
              hold('/api/knowledge/chats?offset=0&limit=100&q=slow');
              __page.byId['knowledge-chat-query'].value = 'slow'; __page.byId['knowledge-chat-query'].dispatch('input');
              return wait(230).then(function () {
                knowledgeCollection('beta').querySelector('.knowledge-browse').dispatch('click');
                heldFor('/api/knowledge/chats?offset=0&limit=100&q=slow')[0].answer({ chats: [historyOne] });
                return settle(20);
              }).then(function () {
                var hidden = __page.byId['knowledge-chat-picker'].hidden;
                return openHistoryPicker().then(function () {
                  return { found: found, hidden: hidden, count: __page.byId['knowledge-chat-count'].textContent,
                    active: __page.byId['knowledge-active-name'].textContent,
                    imported: __page.calls.filter(function (call) { return call.method === 'POST' && call.url.endsWith('/chats'); }).length };
                });
              });
            });
            """);
        Assert.Contains("Earlier design discussion", result.GetProperty("found").GetString());
        Assert.DoesNotContain("<img src=x", result.GetProperty("found").GetString());
        Assert.True(result.GetProperty("hidden").GetBoolean());
        Assert.Equal("0 selected", result.GetProperty("count").GetString());
        Assert.Equal("Private notes", result.GetProperty("active").GetString());
        Assert.Equal(0, result.GetProperty("imported").GetInt32());
    }

    [WebJavaScriptFact]
    public void KnowledgeChatImportFailureAfterSwitchDoesNotReplaceAnotherCollectionsError()
    {
        JsonElement result = Run(KnowledgeLibrary + KnowledgeChats + """
            var failImport;
            R['/api/knowledge/collections/alpha/chats'] = function () {
              return new Promise(function (_, reject) { failImport = reject; });
            };
            R['/api/knowledge/search'] = { __status: 400, body: { error: 'Keep this collection error.' } };
            """, """
            return openKnowledgeLibrary().then(openHistoryPicker).then(function () {
              selectHistoryChat(historyOne.id, true); __page.byId['knowledge-chat-import'].dispatch('click');
              return settle(15);
            }).then(function () {
              knowledgeCollection('beta').querySelector('.knowledge-browse').dispatch('click');
              return settle(15);
            }).then(runKnowledgeSearch).then(function () {
              failImport(new Error('Earlier chat import failed.')); return settle(25);
            }).then(function () {
              return { active: __page.byId['knowledge-active-name'].textContent,
                error: __page.byId['knowledge-error'].textContent, pickerHidden: __page.byId['knowledge-chat-picker'].hidden,
                enabled: !__page.byId['knowledge-add-chats'].disabled, status: __page.byId['knowledge-status'].textContent };
            });
            """);
        Assert.Equal("Private notes", result.GetProperty("active").GetString());
        Assert.Contains("Keep this collection error.", result.GetProperty("error").GetString());
        Assert.DoesNotContain("Earlier chat import failed.", result.GetProperty("error").GetString());
        Assert.True(result.GetProperty("pickerHidden").GetBoolean());
        Assert.True(result.GetProperty("enabled").GetBoolean());
        Assert.Equal("", result.GetProperty("status").GetString());
    }

    [WebJavaScriptFact]
    public void KnowledgeChatPickerBoundsSelectionsAndRetainsThemAfterTransportFailure()
    {
        JsonElement result = Run(KnowledgeLibrary + KnowledgeChats + """
            var manyChats = Array.from({ length: 33 }, function (_, i) {
              return { id: (i + 1).toString(16).padStart(32, '0'), title: 'Chat ' + i, messageCount: 2 };
            });
            R['/api/knowledge/chats?offset=0&limit=100'] = { chats: manyChats };
            R['/api/knowledge/collections/alpha/chats'] = { __reject: 'Storage unavailable.' };
            """, """
            return openKnowledgeLibrary().then(openHistoryPicker).then(function () {
              manyChats.forEach(function (chat) { selectHistoryChat(chat.id, true); });
              var limitError = __page.byId['knowledge-error'].textContent;
              __page.byId['knowledge-chat-import'].dispatch('click'); return settle(25).then(function () {
                return { limitError: limitError, selected: __page.requests('/api/knowledge/collections/alpha/chats')[0].body.conversationIds.length,
                  count: __page.byId['knowledge-chat-count'].textContent, error: __page.byId['knowledge-error'].textContent,
                  lastChecked: __page.byId['knowledge-chat-list'].querySelectorAll('.knowledge-chat-toggle')[32].checked,
                  enabled: !__page.byId['knowledge-chat-import'].disabled };
              });
            });
            """);
        Assert.Contains("at most 32", result.GetProperty("limitError").GetString());
        Assert.Equal(32, result.GetProperty("selected").GetInt32());
        Assert.Equal("32 selected", result.GetProperty("count").GetString());
        Assert.Contains("Storage unavailable.", result.GetProperty("error").GetString());
        Assert.False(result.GetProperty("lastChecked").GetBoolean());
        Assert.True(result.GetProperty("enabled").GetBoolean());
    }
}
