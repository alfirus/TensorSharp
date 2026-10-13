using System.Text.Json;

namespace TensorAgent.Tests;

public sealed partial class WebUiPageTests
{
    [WebJavaScriptFact]
    public void KnowledgeFolderUploadIncludesNestedPathsAndBatchesWithoutDroppingFiles()
    {
        JsonElement result = Run(KnowledgeLibrary + """
            R['/api/knowledge/collections/alpha/documents'] = function (call) {
              if (call.method !== 'POST') return { documents: libraryDocuments };
              return { documents: call.parts.filter(function (p) { return p.name === 'files'; }).map(function (part, index) {
                return { id: 'added' + index, fileName: part.fileName, status: 'Ready' };
              }), errors: [] };
            };
            """, """
            return openKnowledgeLibrary().then(function () {
              var input = __page.byId['knowledge-folder-input'];
              input.files = Array.from({ length: 17 }, function (_, index) {
                return { name: 'notes.txt', size: 1024, webkitRelativePath: 'Research/section' + index + '/notes.txt' };
              });
              input.dispatch('change', { target: input });
              return settle(50);
            }).then(function () {
              var calls = __page.requests('/api/knowledge/collections/alpha/documents').filter(function (c) { return c.method === 'POST'; });
              return { batches: calls.map(function (c) { return c.parts.filter(function (p) { return p.name === 'files'; }).length; }),
                paths: calls.map(function (c) { return c.parts.filter(function (p) { return p.name.indexOf('relativePath') === 0; }).map(function (p) { return p.value; }); }),
                ordinals: calls.map(function (c) { return c.parts.filter(function (p) { return p.name.indexOf('relativePath') === 0; }).map(function (p) { return p.name; }); }),
                status: __page.byId['knowledge-status'].textContent, enabled: !__page.byId['knowledge-upload-folder'].disabled,
                error: __page.byId['knowledge-error'].textContent };
            });
            """);
        Assert.Equal(new[] { 16, 1 }, result.GetProperty("batches").EnumerateArray().Select(p => p.GetInt32()));
        Assert.Equal("Research/section0/notes.txt", result.GetProperty("paths")[0][0].GetString());
        Assert.Equal("Research/section16/notes.txt", result.GetProperty("paths")[1][0].GetString());
        Assert.Equal("relativePath0", result.GetProperty("ordinals")[1][0].GetString());
        Assert.Contains("17 files added", result.GetProperty("status").GetString());
        Assert.True(result.GetProperty("enabled").GetBoolean());
        Assert.Equal("", result.GetProperty("error").GetString());
    }

    [WebJavaScriptFact]
    public void KnowledgeFolderUploadHonorsByteLimitsAndReportsOversizedFiles()
    {
        JsonElement result = Run(KnowledgeLibrary + """
            R['/api/knowledge/collections/alpha/documents'] = function (call) {
              if (call.method !== 'POST') return { documents: libraryDocuments };
              return { documents: [{ id: 'added', status: 'Ready' }], errors: [] };
            };
            """, """
            return openKnowledgeLibrary().then(function () {
              var input = __page.byId['knowledge-folder-input'];
              input.files = [{ name: 'large.csv', size: 64 * 1024 * 1024, webkitRelativePath: 'Data/a/large.csv' },
                { name: 'large.csv', size: 64 * 1024 * 1024, webkitRelativePath: 'Data/b/large.csv' },
                { name: 'too-large.csv', size: 64 * 1024 * 1024 + 1, webkitRelativePath: 'Data/c/too-large.csv' }];
              input.dispatch('change', { target: input });
              return settle(45);
            }).then(function () {
              var calls = __page.requests('/api/knowledge/collections/alpha/documents').filter(function (c) { return c.method === 'POST'; });
              return { batches: calls.map(function (c) { return c.parts.filter(function (p) { return p.name === 'files'; }).length; }),
                error: __page.byId['knowledge-error'].textContent, status: __page.byId['knowledge-status'].textContent };
            });
            """);
        Assert.Equal(new[] { 1, 1 }, result.GetProperty("batches").EnumerateArray().Select(p => p.GetInt32()));
        Assert.Contains("Data/c/too-large.csv", result.GetProperty("error").GetString());
        Assert.Contains("64 MB", result.GetProperty("error").GetString());
        Assert.Contains("2 files added · 1 need attention", result.GetProperty("status").GetString());
    }

    [WebJavaScriptFact]
    public void KnowledgeFolderUploadContinuesPastFileFailuresAndStopsOnAmbiguousTransportFailure()
    {
        JsonElement result = Run(KnowledgeLibrary + """
            var attempts = 0;
            R['/api/knowledge/collections/alpha/documents'] = function (call) {
              if (call.method !== 'POST') return { documents: libraryDocuments };
              if (++attempts === 1) return { documents: [{ id: 'added' }], errors: [{ fileIndex: 2, fileName: 'notes.txt', error: 'Unsupported content.' }] };
              return { __reject: 'Load failed' };
            };
            """, """
            return openKnowledgeLibrary().then(function () {
              var input = __page.byId['knowledge-folder-input'];
              input.files = Array.from({ length: 33 }, function (_, index) {
                return { name: 'notes.txt', size: 1, webkitRelativePath: 'Folder/' + index + '/notes.txt' };
              });
              input.dispatch('change', { target: input });
              return settle(45);
            }).then(function () {
              return { attempts: attempts, error: __page.byId['knowledge-error'].textContent,
                enabled: !__page.byId['knowledge-upload'].disabled,
                remaining: __page.byId['knowledge-documents'].textContent };
            });
            """);
        Assert.Equal(2, result.GetProperty("attempts").GetInt32());
        Assert.Contains("Folder/2/notes.txt: Unsupported content.", result.GetProperty("error").GetString());
        Assert.Contains("Upload stopped after 16 of 33 files", result.GetProperty("error").GetString());
        Assert.Contains("Refresh", result.GetProperty("error").GetString());
        Assert.True(result.GetProperty("enabled").GetBoolean());
        Assert.Contains("notes.txt", result.GetProperty("remaining").GetString());
    }

    [WebJavaScriptFact]
    public void KnowledgeLibraryFiltersAllFilesOnServerAndDisplaysCollectionSummaryAndPaths()
    {
        JsonElement result = Run(KnowledgeLibrary + """
            libraryCollections[0].documentCount = 504;
            R['/api/knowledge/collections/alpha/documents'] = { documents: libraryDocuments, total: 504, collectionTotal: 504,
              nextOffset: 100, summary: { documents: 500, media: 2, datasets: 2, ready: 498, partial: 3, metadataOnly: 3, sizeBytes: 1024 } };
            R['/api/knowledge/collections/alpha/documents?q=Data%2F2026&type=dataset&sort=size'] = {
              documents: [{ id: 'data', collectionId: 'alpha', fileName: 'data.csv', relativePath: 'Data/2026/data.csv', kind: 'dataset',
                version: 1, sizeBytes: 120, chunkCount: 3, status: 'Ready' }], total: 1, collectionTotal: 504,
              summary: { documents: 500, media: 2, datasets: 2, ready: 498, partial: 3, metadataOnly: 3, sizeBytes: 1024 } };
            """, """
            return openKnowledgeLibrary().then(function () {
              var before = { summary: __page.byId['knowledge-summary'].textContent, collection: knowledgeCollection('alpha').textContent };
              __page.byId['knowledge-file-query'].value = 'Data/2026';
              __page.byId['knowledge-type'].value = 'dataset';
              __page.byId['knowledge-sort'].value = 'size';
              __page.byId['knowledge-sort'].dispatch('change');
              return settle(25).then(function () {
                return { before: before, count: __page.byId['knowledge-page-count'].textContent,
                  filesCount: __page.byId['knowledge-files-count'].textContent, documents: __page.byId['knowledge-documents'].textContent,
                  request: __page.calls.filter(function (c) { return c.url.indexOf('?q=') >= 0; })[0].url,
                  next: !!__page.byId['knowledge-pagination'].querySelector('.knowledge-more') };
              });
            });
            """);
        Assert.Contains("Documents · 500", result.GetProperty("before").GetProperty("summary").GetString());
        Assert.Contains("3 metadata only", result.GetProperty("before").GetProperty("summary").GetString());
        Assert.Contains("504 files", result.GetProperty("before").GetProperty("collection").GetString());
        Assert.Equal("Showing 1–1 of 1 files", result.GetProperty("count").GetString());
        Assert.Equal("504 files", result.GetProperty("filesCount").GetString());
        Assert.Contains("Data/2026/data.csv", result.GetProperty("documents").GetString());
        Assert.Contains("Dataset", result.GetProperty("documents").GetString());
        Assert.EndsWith("?q=Data%2F2026&type=dataset&sort=size", result.GetProperty("request").GetString());
        Assert.False(result.GetProperty("next").GetBoolean());
    }

    [WebJavaScriptFact]
    public void KnowledgeRenameUpdatesTitleAndChatSelectionWithoutCreatingAnotherCollection()
    {
        JsonElement result = Run(KnowledgeLibrary + """
            var renames = 0;
            R['/api/knowledge/collections/alpha/rename'] = function (call) {
              renames++; libraryCollections[0].name = call.body.name;
              return { collection: libraryCollections[0] };
            };
            """, """
            return openKnowledgeLibrary().then(function () {
              chooseKnowledge('alpha', true);
              __page.byId['knowledge-collection-actions'].querySelector('.knowledge-rename').dispatch('click');
              var input = __page.byId['knowledge-collection-actions'].querySelector('.knowledge-rename-input');
              input.value = '  Dataset <script>safe</script>  ';
              var save = __page.byId['knowledge-collection-actions'].querySelector('.knowledge-rename-save');
              save.dispatch('click'); save.dispatch('click');
              return settle(25);
            }).then(function () {
              __page.byId['text'].value = 'Read the collection'; __page.byId['send'].dispatch('click');
              return { renames: renames, title: __page.byId['knowledge-active-name'].textContent,
                html: __page.byId['knowledge-active-name'].innerHTML, chips: __page.byId['knowledgechips'].textContent,
                scope: __page.requests('/api/chat')[0].body.knowledgeCollections,
                created: __page.requests('/api/knowledge/collections').filter(function (c) { return c.method === 'POST'; }).length };
            });
            """);
        Assert.Equal(1, result.GetProperty("renames").GetInt32());
        Assert.Equal("Dataset <script>safe</script>", result.GetProperty("title").GetString());
        Assert.Equal("", result.GetProperty("html").GetString());
        Assert.Contains("Dataset <script>safe</script>", result.GetProperty("chips").GetString());
        Assert.Equal(new[] { "alpha" }, Strings(result, "scope"));
        Assert.Equal(0, result.GetProperty("created").GetInt32());
    }

    [WebJavaScriptFact]
    public void KnowledgeNativeDirectoryProgressIgnoresOtherRequestsAndReleasesBusyOnCompletion()
    {
        JsonElement result = Run(KnowledgeLibrary, """
            return openKnowledgeLibrary().then(function () {
              window.TensorAgent.nativeReady({ dictation: false, knowledgeDirectory: true });
              __page.byId['knowledge-upload-folder'].dispatch('click');
              return settle(15);
            }).then(function () {
              var call = __page.requests('/api/agent/events').filter(function (c) { return c.body.type === 'pick-knowledge-directory'; })[0];
              var out = { collection: call.body.collectionId, busy: __page.byId['knowledge-upload'].disabled };
              out.stale = window.TensorAgent.knowledgeDirectoryProgress({ collectionId: 'alpha', requestId: 'older', done: true });
              window.TensorAgent.knowledgeDirectoryProgress({ collectionId: 'alpha', requestId: call.body.requestId, completed: 2, total: 3 });
              out.progress = __page.byId['knowledge-status'].textContent;
              window.TensorAgent.knowledgeDirectoryProgress({ collectionId: 'alpha', requestId: call.body.requestId,
                completed: 3, total: 3, imported: 2, failed: 1, done: true, errors: [{ fileName: 'Folder/bad.bin', error: 'No extractor.' }] });
              return settle(25).then(function () {
                out.enabled = !__page.byId['knowledge-upload'].disabled; out.status = __page.byId['knowledge-status'].textContent;
                out.error = __page.byId['knowledge-error'].textContent; return out;
              });
            });
            """);
        Assert.Equal("alpha", result.GetProperty("collection").GetString());
        Assert.True(result.GetProperty("busy").GetBoolean());
        Assert.False(result.GetProperty("stale").GetBoolean());
        Assert.Contains("2 of 3 files", result.GetProperty("progress").GetString());
        Assert.True(result.GetProperty("enabled").GetBoolean());
        Assert.Contains("2 files added · 1 need attention", result.GetProperty("status").GetString());
        Assert.Contains("Folder/bad.bin: No extractor.", result.GetProperty("error").GetString());
    }
    [WebJavaScriptFact]
    public void KnowledgeRenameKeepsTheLatestNameWhenAnOlderCatalogRefreshArrivesLate()
    {
        JsonElement result = Run(KnowledgeLibrary + Held + """
            R['/api/knowledge/collections/alpha/rename'] = function (call) {
              libraryCollections[0] = Object.assign({}, libraryCollections[0], { name: call.body.name });
              return { collection: libraryCollections[0] };
            };
            """, """
            return openKnowledgeLibrary().then(function () {
              chooseKnowledge('alpha', true);
              hold('/api/knowledge/collections'); __page.byId['knowledge-refresh'].dispatch('click');
              return settle(15);
            }).then(function () {
              __page.byId['knowledge-collection-actions'].querySelector('.knowledge-rename').dispatch('click');
              __page.byId['knowledge-collection-actions'].querySelector('.knowledge-rename-input').value = 'Updated name';
              __page.byId['knowledge-collection-actions'].querySelector('.knowledge-rename-save').dispatch('click');
              return settle(20);
            }).then(function () {
              heldFor('/api/knowledge/collections')[0].answer({ collections: [{ id: 'alpha', name: 'Research' }, { id: 'beta', name: 'Private notes' }] });
              return settle(20);
            }).then(function () {
              return { title: __page.byId['knowledge-active-name'].textContent, row: knowledgeCollection('alpha').textContent,
                chips: __page.byId['knowledgechips'].textContent, documents: __page.byId['knowledge-documents'].textContent };
            });
            """);
        Assert.Equal("Updated name", result.GetProperty("title").GetString());
        Assert.Contains("Updated name", result.GetProperty("row").GetString());
        Assert.Contains("Updated name", result.GetProperty("chips").GetString());
        Assert.Contains("notes.txt", result.GetProperty("documents").GetString());
    }

    [WebJavaScriptFact]
    public void KnowledgeUploadFinishingAfterCollectionSwitchRefreshesCountsWithoutChangingContentOrErrors()
    {
        JsonElement result = Run(KnowledgeLibrary + """
            var finishUpload;
            R['/api/knowledge/collections/alpha/documents'] = function (call) {
              if (call.method === 'POST') return new Promise(function (answer) { finishUpload = answer; });
              return { documents: libraryDocuments };
            };
            R['/api/knowledge/search'] = { __status: 400, body: { error: 'Keep this collection error.' } };
            """, """
            return openKnowledgeLibrary().then(function () {
              var input = __page.byId['knowledge-folder-input'];
              input.files = [{ name: 'notes.txt', size: 10, webkitRelativePath: 'Folder/notes.txt' }];
              input.dispatch('change', { target: input });
              return settle(15);
            }).then(function () {
              knowledgeCollection('beta').querySelector('.knowledge-browse').dispatch('click');
              return settle(15);
            }).then(runKnowledgeSearch).then(function () {
              libraryCollections[0].documentCount = 11;
              finishUpload({ documents: [], errors: [{ fileName: 'notes.txt', fileIndex: 0, error: 'Upload extraction error.' }] });
              return settle(30);
            }).then(function () {
              return { title: __page.byId['knowledge-active-name'].textContent, alpha: knowledgeCollection('alpha').textContent,
                documents: __page.byId['knowledge-documents'].textContent, error: __page.byId['knowledge-error'].textContent,
                enabled: !__page.byId['knowledge-upload'].disabled };
            });
            """);
        Assert.Equal("Private notes", result.GetProperty("title").GetString());
        Assert.Contains("11 files", result.GetProperty("alpha").GetString());
        Assert.Contains("private.txt", result.GetProperty("documents").GetString());
        Assert.DoesNotContain("Folder/notes.txt", result.GetProperty("documents").GetString());
        Assert.Contains("Keep this collection error.", result.GetProperty("error").GetString());
        Assert.DoesNotContain("Upload extraction error.", result.GetProperty("error").GetString());
        Assert.True(result.GetProperty("enabled").GetBoolean());
    }

    [WebJavaScriptFact]
    public void KnowledgeOrdinaryFileUploadSendsExactNamesAlongsideMultipartFilenameEscaping()
    {
        JsonElement result = Run(KnowledgeLibrary + """
            R['/api/knowledge/collections/alpha/documents'] = function (call) {
              if (call.method !== 'POST') return { documents: libraryDocuments };
              return { documents: [], errors: [] };
            };
            """, """
            return openKnowledgeLibrary().then(function () {
              var input = __page.byId['knowledge-file-input'];
              input.files = [{ name: 'report; "quoted".txt' }, { name: 'literal%22.txt' }, { name: 'данные.csv' }];
              input.dispatch('change', { target: input });
              return settle(30);
            }).then(function () {
              var call = __page.requests('/api/knowledge/collections/alpha/documents').filter(function (c) { return c.method === 'POST'; })[0];
              return { filenames: call.parts.filter(function (p) { return p.name === 'files'; }).map(function (p) { return p.fileName; }),
                metadata: call.parts.filter(function (p) { return p.name.indexOf('relativePath') === 0; }).map(function (p) { return { field: p.name, path: p.value }; }) };
            });
            """);
        Assert.Equal(new[] { "report; \"quoted\".txt", "literal%22.txt", "данные.csv" }, Strings(result, "filenames"));
        JsonElement[] metadata = result.GetProperty("metadata").EnumerateArray().ToArray();
        Assert.Equal(3, metadata.Length);
        Assert.Equal("relativePath0", metadata[0].GetProperty("field").GetString());
        Assert.Equal("report; \"quoted\".txt", metadata[0].GetProperty("path").GetString());
        Assert.Equal("literal%22.txt", metadata[1].GetProperty("path").GetString());
        Assert.Equal("данные.csv", metadata[2].GetProperty("path").GetString());
    }

}
