using System.Text.Json;

namespace TensorAgent.Tests;

public sealed partial class WebUiPageTests
{
    private const string KnowledgeLibrary = """
        var libraryCollections = [{ id: 'alpha', name: 'Research' }, { id: 'beta', name: 'Private notes' }];
        var libraryDocuments = [{ id: 'doc1', collectionId: 'alpha', fileName: 'notes.txt', version: 1,
          chunkCount: 2, status: 'Ready', warnings: [] }];
        var libraryHit = { reference: 'knowledge:alpha/doc1/v1/0', collectionId: 'alpha', documentId: 'doc1',
          fileName: '<img src=x onerror=attack()>', version: 1, chunkIndex: 0, text: '<script>attack()</script>\nEvidence.',
          location: { page: 3, startLine: 8 }, sourceUrl: 'javascript:attack()', score: 0.5 };
        R['/api/knowledge/collections'] = function (call) {
          if (call.method === 'POST') {
            var collection = { id: 'created', name: call.body.name };
            libraryCollections.push(collection);
            return { collection: collection };
          }
          return { collections: libraryCollections };
        };
        R['/api/knowledge/collections/alpha/documents'] = function () { return { documents: libraryDocuments }; };
        R['/api/knowledge/collections/beta/documents'] = { documents: [{ id: 'private', collectionId: 'beta',
          fileName: 'private.txt', version: 1, chunkCount: 1, status: 'Ready', warnings: [] }] };
        R['/api/knowledge/collections/created/documents'] = { documents: [] };
        R['/api/knowledge/collections/alpha/documents/doc1/reindex'] = function () {
          libraryDocuments[0].version++;
          return { document: libraryDocuments[0] };
        };
        R['/api/knowledge/collections/alpha/documents/doc1'] = function () { libraryDocuments = []; return { deleted: true }; };
        R['/api/knowledge/collections/alpha'] = function () {
          libraryCollections = libraryCollections.filter(function (c) { return c.id !== 'alpha'; });
          return { deleted: true };
        };
        R['/api/knowledge/search'] = { results: [libraryHit] };
        R['/api/knowledge/read'] = { results: [Object.assign({}, libraryHit, { text: 'Expanded evidence from page three.' })] };
        R['/api/knowledge/ocr/status'] = { managed: false, status: null };
        var knowledgeNav = document.createElement('button');
        knowledgeNav.className = 'opt'; knowledgeNav.setAttribute('data-sheet', 'knowledge-sheet');
        document.getElementById('nav-sheet').appendChild(knowledgeNav);
        function knowledgeCollection(id) {
          return document.getElementById('knowledge-collections').querySelectorAll('.knowledge-collection')
            .filter(function (row) { return row.getAttribute('data-collection') === id; })[0];
        }
        function openKnowledgeLibrary() { knowledgeNav.dispatch('click'); return settle(25); }
        function chooseKnowledge(id, checked) {
          var checkbox = knowledgeCollection(id).querySelector('.knowledge-toggle');
          checkbox.checked = checked; checkbox.dispatch('change');
        }
        function runKnowledgeSearch() {
          document.getElementById('knowledge-query').value = 'evidence';
          document.getElementById('knowledge-search').dispatch('click');
          return settle(20);
        }
        """;

    [WebJavaScriptFact]
    public void KnowledgeSelectionIsExplicitAndSavedConversationScopeNeverLeaksIntoAnotherChat()
    {
        JsonElement result = Run(KnowledgeLibrary + """
            R['/api/sessions?conversation=saved'] = { sessionId: 's2', conversationId: 'saved',
              messages: [{ role: 'user', content: 'Earlier' }], knowledgeCollections: ['beta'] };
            """, """
            return openKnowledgeLibrary().then(function () {
              chooseKnowledge('alpha', true);
              __page.byId['text'].value = 'First'; __page.byId['send'].dispatch('click');
              window.TensorAgent.openConversation('saved');
              return settle(25);
            }).then(function () {
              var restored = { alpha: knowledgeCollection('alpha').querySelector('.knowledge-toggle').checked,
                beta: knowledgeCollection('beta').querySelector('.knowledge-toggle').checked };
              window.TensorAgent.openConversation(null);
              return settle(25).then(function () {
                __page.byId['text'].value = 'Second'; __page.byId['send'].dispatch('click');
                return { restored: restored, requests: __page.requests('/api/chat').map(function (c) { return c.body; }),
                  chips: __page.byId['knowledgechips'].children.length };
              });
            });
            """);
        JsonElement[] requests = result.GetProperty("requests").EnumerateArray().ToArray();
        Assert.Equal(2, requests.Length);
        Assert.Equal(new[] { "alpha" }, Strings(requests[0], "knowledgeCollections"));
        Assert.Empty(Strings(requests[1], "knowledgeCollections"));
        Assert.False(result.GetProperty("restored").GetProperty("alpha").GetBoolean());
        Assert.True(result.GetProperty("restored").GetProperty("beta").GetBoolean());
        Assert.Equal(0, result.GetProperty("chips").GetInt32());
    }

    [WebJavaScriptFact]
    public void KnowledgeSearchAndReadCarryBoundsAndRenderSourceTextWithoutExecutingMarkupOrUrls()
    {
        JsonElement result = Run(KnowledgeLibrary, """
            return openKnowledgeLibrary().then(runKnowledgeSearch).then(function () {
              var row = __page.byId['knowledge-results'].querySelector('.knowledge-result');
              var displayed = { title: row.querySelector('strong').textContent, text: row.querySelector('pre').textContent,
                html: row.querySelector('pre').innerHTML, source: row.querySelector('.knowledge-source').href,
                location: row.querySelector('.hint-block').textContent };
              row.querySelector('.knowledge-expand').dispatch('click');
              return settle(20).then(function () {
                return { displayed: displayed, search: __page.requests('/api/knowledge/search')[0].body,
                  read: __page.requests('/api/knowledge/read')[0].body, expanded: __page.byId['knowledge-read'].textContent };
              });
            });
            """);
        JsonElement shown = result.GetProperty("displayed");
        Assert.Equal("<img src=x onerror=attack()>", shown.GetProperty("title").GetString());
        Assert.Contains("<script>attack()</script>", shown.GetProperty("text").GetString());
        Assert.Equal("", shown.GetProperty("html").GetString());
        Assert.Equal("/api/knowledge/collections/alpha/documents/doc1/source?version=1", shown.GetProperty("source").GetString());
        Assert.Contains("Page 3", shown.GetProperty("location").GetString());
        Assert.Equal(new[] { "alpha" }, Strings(result.GetProperty("search"), "collectionIds"));
        Assert.Equal(8, result.GetProperty("search").GetProperty("limit").GetInt32());
        Assert.Equal(12000, result.GetProperty("read").GetProperty("maxCharacters").GetInt32());
        Assert.Equal(1, result.GetProperty("read").GetProperty("adjacentChunks").GetInt32());
        Assert.Contains("Expanded evidence", result.GetProperty("expanded").GetString());
    }

    [WebJavaScriptFact]
    public void KnowledgeCollectionSwitchDiscardsSlowDocumentSearchAndReadResponses()
    {
        JsonElement result = Run(KnowledgeLibrary + Held, """
            return openKnowledgeLibrary().then(runKnowledgeSearch).then(function () {
              hold('/api/knowledge/read'); hold('/api/knowledge/search');
              hold('/api/knowledge/collections/alpha/documents');
              __page.byId['knowledge-results'].querySelector('.knowledge-expand').dispatch('click');
              __page.byId['knowledge-search'].dispatch('click');
              knowledgeCollection('alpha').querySelector('.knowledge-browse').dispatch('click');
              knowledgeCollection('beta').querySelector('.knowledge-browse').dispatch('click');
              return settle(15);
            }).then(function () {
              heldFor('/api/knowledge/read')[0].answer({ results: [libraryHit] });
              heldFor('/api/knowledge/search')[0].answer({ results: [libraryHit] });
              heldFor('/api/knowledge/collections/alpha/documents')[0].answer({ documents: libraryDocuments });
              return settle(20);
            }).then(function () {
              return { active: __page.byId['knowledge-active-name'].textContent,
                documents: __page.byId['knowledge-documents'].textContent,
                results: __page.byId['knowledge-results'].textContent, read: __page.byId['knowledge-read'].textContent };
            });
            """);
        Assert.Equal("Private notes", result.GetProperty("active").GetString());
        Assert.Contains("private.txt", result.GetProperty("documents").GetString());
        Assert.DoesNotContain("notes.txt", result.GetProperty("documents").GetString());
        Assert.Equal("", result.GetProperty("results").GetString());
        Assert.Equal("", result.GetProperty("read").GetString());
    }

    [WebJavaScriptFact]
    public void KnowledgeUploadPreservesMultipartNamesAndDisplaysPartialFailuresAndExtractionLimits()
    {
        JsonElement result = Run(KnowledgeLibrary + """
            R['/api/knowledge/collections/alpha/documents'] = function (call) {
              if (call.method === 'POST') {
                libraryDocuments = [{ id: 'image', collectionId: 'alpha', fileName: 'scan.png', version: 1,
                  chunkCount: 0, status: 'MetadataOnly', warnings: ['OCR model is unavailable.'] }];
                return { documents: libraryDocuments, errors: [{ fileName: 'bad.pdf', error: 'Malformed document.' }] };
              }
              return { documents: libraryDocuments };
            };
            """, """
            return openKnowledgeLibrary().then(function () {
              var input = __page.byId['knowledge-file-input'];
              input.files = [{ name: 'scan.png' }, { name: 'bad.pdf' }];
              input.dispatch('change', { target: input });
              return settle(25);
            }).then(function () {
              var call = __page.requests('/api/knowledge/collections/alpha/documents').filter(function (c) { return c.method === 'POST'; })[0];
              return { names: call.parts.filter(function (p) { return p.name === 'files'; }).map(function (p) { return p.fileName; }),
                fields: call.parts.filter(function (p) { return p.name === 'files'; }).map(function (p) { return p.name; }),
                error: __page.byId['knowledge-error'].textContent, documents: __page.byId['knowledge-documents'].textContent,
                enabled: !__page.byId['knowledge-upload'].disabled, composerUploads: __page.requests('/api/upload').length };
            });
            """);
        Assert.Equal(new[] { "scan.png", "bad.pdf" }, Strings(result, "names"));
        Assert.Equal(new[] { "files", "files" }, Strings(result, "fields"));
        Assert.Contains("bad.pdf: Malformed document.", result.GetProperty("error").GetString());
        Assert.Contains("Metadata only", result.GetProperty("documents").GetString());
        Assert.Contains("OCR model is unavailable.", result.GetProperty("documents").GetString());
        Assert.True(result.GetProperty("enabled").GetBoolean());
        Assert.Equal(0, result.GetProperty("composerUploads").GetInt32());
    }

    [WebJavaScriptFact]
    public void KnowledgeReindexInvalidatesDisplayedEvidenceAndCollectionDeletionRemovesChatSelection()
    {
        JsonElement result = Run(KnowledgeLibrary, """
            return openKnowledgeLibrary().then(function () { chooseKnowledge('alpha', true); return runKnowledgeSearch(); })
            .then(function () {
              __page.byId['knowledge-documents'].querySelector('.knowledge-reindex').dispatch('click');
              return settle(25);
            }).then(function () {
              var reindexed = { source: __page.byId['knowledge-documents'].querySelector('.knowledge-source').href,
                results: __page.byId['knowledge-results'].textContent };
              var remove = __page.byId['knowledge-collection-actions'].querySelector('.knowledge-delete');
              remove.dispatch('click');
              var before = __page.requests('/api/knowledge/collections/alpha').length;
              remove.dispatch('click');
              return settle(25).then(function () {
                __page.byId['text'].value = 'Continue'; __page.byId['send'].dispatch('click');
                return { reindexed: reindexed, before: before, selected: __page.requests('/api/chat')[0].body.knowledgeCollections,
                  remaining: !!knowledgeCollection('alpha'), active: __page.byId['knowledge-active-name'].textContent };
              });
            });
            """);
        Assert.EndsWith("version=2", result.GetProperty("reindexed").GetProperty("source").GetString());
        Assert.Equal("", result.GetProperty("reindexed").GetProperty("results").GetString());
        Assert.Equal(0, result.GetProperty("before").GetInt32());
        Assert.Empty(Strings(result, "selected"));
        Assert.False(result.GetProperty("remaining").GetBoolean());
        Assert.Equal("Private notes", result.GetProperty("active").GetString());
    }

    [WebJavaScriptFact]
    public void KnowledgeCreateDoesNotDuplicatePendingMutationAndFailedDeleteKeepsDocumentVisible()
    {
        JsonElement result = Run(KnowledgeLibrary + Held, """
            return openKnowledgeLibrary().then(function () {
              hold('/api/knowledge/collections');
              document.getElementById('knowledge-name').value = 'New library';
              __page.byId['knowledge-create'].dispatch('click'); __page.byId['knowledge-create'].dispatch('click');
              return settle(15);
            }).then(function () {
              var pending = heldFor('/api/knowledge/collections').length;
              heldFor('/api/knowledge/collections')[0].answer({ __status: 400, body: { error: 'Collection limit reached.' } });
              return settle(20).then(function () {
                R['/api/knowledge/collections/alpha/documents/doc1'] = { __status: 500, body: { error: 'Storage unavailable.' } };
                var remove = __page.byId['knowledge-documents'].querySelector('.knowledge-delete');
                remove.dispatch('click'); remove.dispatch('click');
                return settle(20).then(function () {
                  return { pending: pending, error: __page.byId['knowledge-error'].textContent,
                    documents: __page.byId['knowledge-documents'].textContent, enabled: !__page.byId['knowledge-create'].disabled };
                });
              });
            }).catch(function (e) { return { failure: String(e), stack: e.stack, documents: __page.byId['knowledge-documents'].textContent,
              calls: __page.requests('/api/knowledge/collections'), error: __page.byId['knowledge-error'].textContent }; });
            """);
        Assert.True(result.TryGetProperty("pending", out JsonElement pending), result.ToString());
        Assert.Equal(1, pending.GetInt32());
        Assert.Contains("Storage unavailable.", result.GetProperty("error").GetString());
        Assert.Contains("notes.txt", result.GetProperty("documents").GetString());
        Assert.True(result.GetProperty("enabled").GetBoolean());
    }

    [WebJavaScriptFact]
    public void KnowledgeStreamSourcesUseHostIdentitiesAndSurviveIntoTheNextRequest()
    {
        JsonElement result = Run(KnowledgeLibrary + """
            R['/api/chat'] = { __sse: [
              { knowledgeSources: [libraryHit] }, { token: '来源：/api/knowledge/collections/alpha/documents/doc1/source?version=1。' },
              { knowledgeSources: [libraryHit, libraryHit, Object.assign({}, libraryHit,
                { reference: 'bad', documentId: '../outside' })] }, { done: true }] };
            """, """
            __page.byId['text'].value = 'Read my sources'; __page.byId['send'].dispatch('click');
            return settle(35).then(function () {
              var links = __page.byId['chat'].querySelectorAll('.knowledge-citations .knowledge-source');
              var shown = links.map(function (a) { return { href: a.href, label: a.textContent, html: a.innerHTML }; });
              __page.byId['text'].value = 'Continue'; __page.byId['send'].dispatch('click');
              return { shown: shown, messages: __page.requests('/api/chat')[1].body.messages };
            });
            """);
        JsonElement link = Assert.Single(result.GetProperty("shown").EnumerateArray());
        Assert.Equal("/api/knowledge/collections/alpha/documents/doc1/source?version=1", link.GetProperty("href").GetString());
        Assert.Contains("<img src=x onerror=attack()>", link.GetProperty("label").GetString());
        Assert.Equal("", link.GetProperty("html").GetString());
        JsonElement assistant = result.GetProperty("messages").EnumerateArray().First(m => m.GetProperty("role").GetString() == "assistant");
        JsonElement source = Assert.Single(assistant.GetProperty("knowledgeSources").EnumerateArray());
        Assert.Equal("doc1", source.GetProperty("documentId").GetString());
        Assert.False(source.TryGetProperty("sourceUrl", out _));
    }

    [WebJavaScriptFact]
    public void KnowledgeSavedSourcesRenderWithVersionAndOriginalReference()
    {
        JsonElement result = Run(KnowledgeLibrary + """
            R['/api/sessions?conversation=saved'] = { sessionId: 's2', conversationId: 'saved',
              knowledgeCollections: ['alpha'], messages: [{ role: 'assistant', content: 'Saved answer [source](/api/knowledge/collections/alpha/documents/doc1/source?version=4)',
                knowledgeSources: [Object.assign({}, libraryHit, { version: 4, chunkIndex: -1 })] }] };
            """, """
            window.TensorAgent.openConversation('saved');
            return settle(25).then(function () {
              var link = __page.byId['chat'].querySelector('.knowledge-source');
              __page.byId['text'].value = 'Continue'; __page.byId['send'].dispatch('click');
              return { href: link.href, title: link.title, label: link.textContent,
                saved: __page.requests('/api/chat')[0].body.messages[0].knowledgeSources[0] };
            });
            """);
        Assert.EndsWith("version=4", result.GetProperty("href").GetString());
        Assert.Equal("/api/knowledge/collections/alpha/documents/doc1/source?version=4", result.GetProperty("title").GetString());
        Assert.DoesNotContain("-1", result.GetProperty("label").GetString());
        Assert.Equal(4, result.GetProperty("saved").GetProperty("version").GetInt32());
    }

    [WebJavaScriptFact]
    public void KnowledgeSourcesFollowTheFinalAnswerAndEmptySnapshotClearsSavedSources()
    {
        JsonElement result = Run(KnowledgeLibrary + """
            R['/api/chat'] = { __sse: [
              { token: 'Initially [cited](/api/knowledge/collections/alpha/documents/doc1/source?version=1).' },
              { knowledgeSources: [libraryHit] }, { replace: 'The final answer uses no source.' },
              { knowledgeSources: [] }, { done: true }] };
            """, """
            __page.byId['text'].value = 'Read my sources'; __page.byId['send'].dispatch('click');
            return settle(35).then(function () {
              var shown = __page.byId['chat'].querySelectorAll('.knowledge-citations .knowledge-source').length;
              __page.byId['text'].value = 'Continue'; __page.byId['send'].dispatch('click');
              return { shown: shown, messages: __page.requests('/api/chat')[1].body.messages };
            });
            """);
        Assert.Equal(0, result.GetProperty("shown").GetInt32());
        var assistant = result.GetProperty("messages").EnumerateArray().First(m => m.GetProperty("role").GetString() == "assistant");
        Assert.False(assistant.TryGetProperty("knowledgeSources", out _));
    }

    [WebJavaScriptFact]
    public void LegacySavedSourcesKeepOnlyExactCitedReferences()
    {
        JsonElement result = Run(KnowledgeLibrary + """
            R['/api/sessions?conversation=saved'] = { sessionId: 's2', conversationId: 'saved',
              messages: [{ role: 'assistant', content: 'Evidence [knowledge:alpha/doc1/v1/0].', knowledgeSources: [libraryHit,
                Object.assign({}, libraryHit, { reference: 'unused', documentId: 'doc2' })] },
                { role: 'assistant', content: 'Uncited answer', knowledgeSources: [libraryHit] }] };
            """, """
            window.TensorAgent.openConversation('saved');
            return settle(25).then(function () {
              var links = __page.byId['chat'].querySelectorAll('.knowledge-citations .knowledge-source');
              __page.byId['text'].value = 'Continue'; __page.byId['send'].dispatch('click');
              return { shown: links.length, messages: __page.requests('/api/chat')[0].body.messages };
            });
            """);
        Assert.Equal(1, result.GetProperty("shown").GetInt32());
        var messages = result.GetProperty("messages");
        Assert.Single(messages[0].GetProperty("knowledgeSources").EnumerateArray());
        Assert.False(messages[1].TryGetProperty("knowledgeSources", out _));
    }

    [WebJavaScriptFact]
    public void SavedChineseCitationsAcceptPunctuationButRejectExternalOriginsAndPathPrefixes()
    {
        JsonElement result = Run(KnowledgeLibrary + """
            var sourceUrl = '/api/knowledge/collections/alpha/documents/doc1/source?version=1';
            R['/api/sessions?conversation=saved'] = { sessionId: 's2', conversationId: 'saved',
              messages: [
                { role: 'assistant', content: '来源：' + sourceUrl + '。', knowledgeSources: [libraryHit] },
                { role: 'assistant', content: '评语：' + libraryHit.reference + '】', knowledgeSources: [libraryHit] },
                { role: 'assistant', content: '来源：https://untrusted.example' + sourceUrl + '。', knowledgeSources: [libraryHit] },
                { role: 'assistant', content: '来源：//untrusted.example' + sourceUrl + '。', knowledgeSources: [libraryHit] },
                { role: 'assistant', content: '来源：/prefix' + sourceUrl + '。', knowledgeSources: [libraryHit] }
              ] };
            """, """
            window.TensorAgent.openConversation('saved');
            return settle(25).then(function () {
              var links = __page.byId['chat'].querySelectorAll('.knowledge-citations .knowledge-source');
              __page.byId['text'].value = 'Continue'; __page.byId['send'].dispatch('click');
              return { shown: links.map(function (a) { return a.href; }),
                messages: __page.requests('/api/chat')[0].body.messages };
            });
            """);
        Assert.Equal(2, result.GetProperty("shown").GetArrayLength());
        Assert.All(result.GetProperty("shown").EnumerateArray(), href =>
            Assert.Equal("/api/knowledge/collections/alpha/documents/doc1/source?version=1", href.GetString()));
        var messages = result.GetProperty("messages");
        Assert.Single(messages[0].GetProperty("knowledgeSources").EnumerateArray());
        Assert.Single(messages[1].GetProperty("knowledgeSources").EnumerateArray());
        for (int i = 2; i < 5; i++) Assert.False(messages[i].TryGetProperty("knowledgeSources", out _));
    }

    [WebJavaScriptFact]
    public void KnowledgeRefreshRemovesSelectionsDeletedOutsideThisPage()
    {
        JsonElement result = Run(KnowledgeLibrary, """
            return openKnowledgeLibrary().then(function () {
              chooseKnowledge('alpha', true);
              libraryCollections = libraryCollections.filter(function (c) { return c.id !== 'alpha'; });
              __page.byId['knowledge-refresh'].dispatch('click');
              return settle(25);
            }).then(function () {
              __page.byId['text'].value = 'Continue'; __page.byId['send'].dispatch('click');
              return { scope: __page.requests('/api/chat')[0].body.knowledgeCollections,
                chips: __page.byId['knowledgechips'].children.length,
                active: __page.byId['knowledge-active-name'].textContent };
            });
            """);
        Assert.Empty(Strings(result, "scope"));
        Assert.Equal(0, result.GetProperty("chips").GetInt32());
        Assert.Equal("Private notes", result.GetProperty("active").GetString());
    }

    [WebJavaScriptFact]
    public void KnowledgeDocumentsLoadBoundedPagesAndIgnoreLatePageAfterSwitch()
    {
        JsonElement result = Run(KnowledgeLibrary + Held + """
            libraryDocuments = Array.from({ length: 100 }, function (_, i) {
              return { id: 'doc' + i, collectionId: 'alpha', fileName: 'notes-' + i + '.txt',
                version: 1, chunkCount: 1, status: 'Ready', warnings: [] };
            });
            R['/api/knowledge/collections/alpha/documents'] = { documents: libraryDocuments, total: 201, nextOffset: 100 };
            R['/api/knowledge/collections/alpha/documents?offset=100&limit=100'] = {
              documents: [Object.assign({}, libraryDocuments[0], { id: 'doc100', fileName: 'notes-100.txt' })],
              total: 201, nextOffset: 200 };
            """, """
            return openKnowledgeLibrary().then(function () {
              var first = __page.byId['knowledge-documents'].querySelectorAll('.knowledge-document').length;
              __page.byId['knowledge-pagination'].querySelector('.knowledge-more').dispatch('click');
              return settle(20).then(function () {
                var second = __page.byId['knowledge-documents'].querySelectorAll('.knowledge-document').length;
                hold('/api/knowledge/collections/alpha/documents?offset=200&limit=100');
                var more = __page.byId['knowledge-pagination'].querySelector('.knowledge-more');
                more.dispatch('click'); more.dispatch('click');
                return settle(15).then(function () {
                  knowledgeCollection('beta').querySelector('.knowledge-browse').dispatch('click');
                  heldFor('/api/knowledge/collections/alpha/documents?offset=200&limit=100')[0]
                    .answer({ documents: libraryDocuments, total: 201, nextOffset: null });
                  return settle(20).then(function () {
                    return { first: first, second: second, pending: heldFor('/api/knowledge/collections/alpha/documents?offset=200&limit=100').length,
                      active: __page.byId['knowledge-active-name'].textContent, docs: __page.byId['knowledge-documents'].textContent,
                      more: !!__page.byId['knowledge-pagination'].querySelector('.knowledge-more') };
                  });
                });
              });
            });
            """);
        Assert.Equal(100, result.GetProperty("first").GetInt32());
        Assert.Equal(1, result.GetProperty("second").GetInt32());
        Assert.Equal(1, result.GetProperty("pending").GetInt32());
        Assert.Equal("Private notes", result.GetProperty("active").GetString());
        Assert.Contains("private.txt", result.GetProperty("docs").GetString());
        Assert.DoesNotContain("notes-", result.GetProperty("docs").GetString());
        Assert.False(result.GetProperty("more").GetBoolean());
    }
}
