using System.Text.Json;

namespace TensorAgent.Tests;

public sealed partial class WebUiPageTests
{
    [WebJavaScriptTheory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void KnowledgePdfPreviewUsesBrowserCapabilityAndShowsIndexedTextBeforeOriginal(bool pdfViewerEnabled, bool native)
    {
        JsonElement result = Run(KnowledgeLibrary + "window.navigator.pdfViewerEnabled = " + (pdfViewerEnabled ? "true;" : "false;") + """
            R['/api/knowledge/collections/alpha/documents/doc1/preview?version=1'] = {
              document: { fileName: 'report.pdf' }, text: 'Readable PDF content.', inlineContentType: 'application/pdf', location: { page: 7 } };
            """, """
            if (NATIVE) window.TensorAgent.nativeReady({});
            var link = document.createElement('a');
            link.href = '/api/knowledge/collections/alpha/documents/doc1/source?version=1';
            document.dispatch('click', { target: link, preventDefault: function () {} });
            return settle(15).then(function () {
              var viewer = document.body.querySelector('.knowledge-source-viewer'), frame = viewer.querySelector('iframe');
              var content = viewer.querySelector('.knowledge-source-content');
              return { text: viewer.textContent, frame: !!frame, url: frame ? frame.src : null,
                textBeforeOriginal: !frame || Array.from(content.children).indexOf(viewer.querySelector('pre')) < Array.from(content.children).indexOf(frame) };
            });
            """.Replace("NATIVE", native ? "true" : "false"));
        Assert.Contains("Readable PDF content.", result.GetProperty("text").GetString());
        Assert.Equal(pdfViewerEnabled && !native, result.GetProperty("frame").GetBoolean());
        Assert.True(result.GetProperty("textBeforeOriginal").GetBoolean());
        if (pdfViewerEnabled && !native)
            Assert.EndsWith("version=1&inline=1#page=7", result.GetProperty("url").GetString());
    }

    [WebJavaScriptFact]
    public void KnowledgeCitationOpensAtCitedChunkAndCanBrowseFurtherExcerpts()
    {
        JsonElement result = Run(KnowledgeLibrary + """
            R['/api/sessions?conversation=saved'] = { sessionId: 's2', conversationId: 'saved',
              messages: [{ role: 'assistant', content: 'Evidence [knowledge:alpha/doc1/v1/20].',
                knowledgeSources: [Object.assign({}, libraryHit, { reference: 'knowledge:alpha/doc1/v1/20', chunkIndex: 20 })] }] };
            R['/api/knowledge/collections/alpha/documents/doc1/preview?version=1&chunkIndex=20'] = {
              text: 'Evidence at the cited chunk.', nextChunkIndex: 30, previousChunkIndex: 0, truncated: true };
            R['/api/knowledge/collections/alpha/documents/doc1/preview?version=1&chunkIndex=30'] = {
              text: 'Later evidence.', previousChunkIndex: 20, nextChunkIndex: null, truncated: false };
            """, """
            window.TensorAgent.openConversation('saved');
            return settle(25).then(function () {
              __page.byId['chat'].querySelector('.knowledge-source').dispatch('click');
              return settle(15).then(function () {
                var first = document.body.querySelector('.knowledge-source-viewer').textContent;
                document.body.querySelector('.knowledge-source-next').dispatch('click');
                return settle(15).then(function () {
                  return { first: first, next: document.body.querySelector('.knowledge-source-viewer').textContent,
                    previous: !!document.body.querySelector('.knowledge-source-previous'),
                    requests: __page.calls.filter(function (call) { return call.url === '/api/knowledge/collections/alpha/documents/doc1/preview?version=1&chunkIndex=30'; }).length };
                });
              });
            });
            """);
        Assert.Contains("Evidence at the cited chunk.", result.GetProperty("first").GetString());
        Assert.Contains("Later evidence.", result.GetProperty("next").GetString());
        Assert.DoesNotContain("Evidence at the cited chunk.", result.GetProperty("next").GetString());
        Assert.True(result.GetProperty("previous").GetBoolean());
        Assert.Equal(1, result.GetProperty("requests").GetInt32());
    }

    [WebJavaScriptFact]
    public void KnowledgeCitationOpensIndexedTextInsideThePageAndKeepsOriginalDownload()
    {
        JsonElement result = Run(KnowledgeLibrary + """
            R['/api/chat'] = { __sse: [{ token: 'Evidence [source](/api/knowledge/collections/alpha/documents/doc1/source?version=1).' },
              { knowledgeSources: [Object.assign({}, libraryHit, { relativePath: 'Research/notes.txt' })] }, { done: true }] };
            R['/api/knowledge/collections/alpha/documents/doc1/preview?version=1'] = {
              document: { fileName: 'notes.txt', relativePath: 'Research/notes.txt', warnings: ['Some content was not extracted.'] },
              text: '<script>attack()</script>\nEvidence.', truncated: true, inlineContentType: null };
            """, """
            __page.byId['text'].value = 'Read'; __page.byId['send'].dispatch('click');
            return settle(35).then(function () {
              var source = __page.byId['chat'].querySelector('.knowledge-source'), prevented = false;
              source.dispatch('click', { preventDefault: function () { prevented = true; } });
              return settle(20).then(function () {
                var viewer = document.body.querySelector('.knowledge-source-viewer');
                var preview = viewer.querySelector('pre'), download = viewer.querySelector('a');
                return { prevented: prevented, sourceDownload: source.getAttribute('download'),
                  dialog: viewer.querySelector('section').getAttribute('role'), title: viewer.querySelector('strong').textContent,
                  text: preview.textContent, html: preview.innerHTML, contents: viewer.textContent,
                  download: download.href, filename: download.getAttribute('download'),
                  previewRequests: __page.calls.filter(function (call) { return call.url === '/api/knowledge/collections/alpha/documents/doc1/preview?version=1'; }).length };
              });
            });
            """);
        Assert.True(result.GetProperty("prevented").GetBoolean());
        Assert.Equal(JsonValueKind.Null, result.GetProperty("sourceDownload").ValueKind);
        Assert.Equal("dialog", result.GetProperty("dialog").GetString());
        Assert.Equal("Research/notes.txt", result.GetProperty("title").GetString());
        Assert.Contains("<script>attack()</script>", result.GetProperty("text").GetString());
        Assert.Equal("", result.GetProperty("html").GetString());
        Assert.Contains("Some content was not extracted.", result.GetProperty("contents").GetString());
        Assert.Contains("Showing an excerpt", result.GetProperty("contents").GetString());
        Assert.Equal("/api/knowledge/collections/alpha/documents/doc1/source?version=1", result.GetProperty("download").GetString());
        Assert.Equal(1, result.GetProperty("previewRequests").GetInt32());
    }

    [WebJavaScriptFact]
    public void KnowledgeMarkdownSourceLinksUsePreviewOnlyForThisOriginAndValidIdentities()
    {
        JsonElement result = Run(KnowledgeLibrary + """
            R['/api/knowledge/collections/alpha/documents/doc1/preview?version=4'] = {
              document: { fileName: 'notes.txt' }, text: 'Saved source version four.', inlineContentType: null };
            """, """
            var prevented = [];
            function clickSource(href) {
              var link = document.createElement('a'); link.href = href; link.textContent = 'Saved source';
              document.dispatch('click', { target: link, preventDefault: function () { prevented.push(href); } });
            }
            clickSource('https://untrusted.example/api/knowledge/collections/alpha/documents/doc1/source?version=4');
            clickSource('/api/knowledge/collections/alpha/documents/../source?version=4');
            clickSource('/api/knowledge/collections/alpha/documents/doc1/source?version=0');
            clickSource('/api/knowledge/collections/alpha/documents/doc1/source?version=4');
            return settle(20).then(function () {
              return { prevented: prevented, text: document.body.querySelector('.knowledge-source-viewer').textContent,
                requests: __page.calls.filter(function (call) { return call.url === '/api/knowledge/collections/alpha/documents/doc1/preview?version=4'; }).length };
            });
            """);
        Assert.Single(result.GetProperty("prevented").EnumerateArray());
        Assert.Contains("Saved source version four.", result.GetProperty("text").GetString());
        Assert.Equal(1, result.GetProperty("requests").GetInt32());
    }

    [WebJavaScriptFact]
    public void KnowledgeSourcePreviewRejectsActiveMediaAndEmbedsAllowedMediaUsingOwnedUrl()
    {
        JsonElement result = Run(KnowledgeLibrary + """
            R['/api/knowledge/collections/alpha/documents/doc1/preview?version=1'] = {
              document: { fileName: 'unsafe.svg' }, text: '', inlineContentType: 'image/svg+xml', sourceUrl: 'javascript:attack()' };
            R['/api/knowledge/collections/alpha/documents/doc2/preview?version=2'] = {
              document: { fileName: 'scan.png' }, text: '', inlineContentType: 'image/png', sourceUrl: 'https://untrusted.example/scan.png' };
            """, """
            function open(id, version) {
              var link = document.createElement('a');
              link.href = '/api/knowledge/collections/alpha/documents/' + id + '/source?version=' + version;
              document.dispatch('click', { target: link, preventDefault: function () {} });
              return settle(15);
            }
            return open('doc1', 1).then(function () {
              var unsafe = { image: !!document.body.querySelector('img'), text: document.body.querySelector('.knowledge-source-viewer').textContent };
              return open('doc2', 2).then(function () {
                var image = document.body.querySelector('.knowledge-source-viewer').querySelector('img');
                return { unsafe: unsafe, src: image.src, alt: image.alt };
              });
            });
            """);
        Assert.False(result.GetProperty("unsafe").GetProperty("image").GetBoolean());
        Assert.Contains("No readable text was indexed", result.GetProperty("unsafe").GetProperty("text").GetString());
        Assert.Equal("/api/knowledge/collections/alpha/documents/doc2/source?version=2&inline=1", result.GetProperty("src").GetString());
        Assert.Equal("scan.png", result.GetProperty("alt").GetString());
    }

    [WebJavaScriptFact]
    public void KnowledgeSourcePreviewIgnoresResponsesAfterSwitchOrClose()
    {
        JsonElement result = Run(KnowledgeLibrary + Held, """
            function open(id) {
              var link = document.createElement('a');
              link.href = '/api/knowledge/collections/alpha/documents/' + id + '/source?version=1';
              document.dispatch('click', { target: link, preventDefault: function () {} });
            }
            hold('/api/knowledge/collections/alpha/documents/doc1/preview?version=1');
            hold('/api/knowledge/collections/alpha/documents/doc2/preview?version=1');
            open('doc1'); open('doc2');
            return settle(15).then(function () {
              heldFor('/api/knowledge/collections/alpha/documents/doc1/preview?version=1')[0].answer({ text: 'Old evidence.' });
              return settle(15).then(function () {
                var staleVisible = document.body.querySelector('.knowledge-source-viewer').textContent.indexOf('Old evidence.') >= 0;
                document.dispatch('keydown', { key: 'Escape', preventDefault: function () {} });
                heldFor('/api/knowledge/collections/alpha/documents/doc2/preview?version=1')[0].answer({ text: 'Closed evidence.' });
                return settle(15).then(function () { return { staleVisible: staleVisible, open: !!document.body.querySelector('.knowledge-source-viewer'), overflow: document.body.style.overflow }; });
              });
            });
            """);
        Assert.False(result.GetProperty("staleVisible").GetBoolean());
        Assert.False(result.GetProperty("open").GetBoolean());
        Assert.Equal("", result.GetProperty("overflow").GetString());
    }

    [WebJavaScriptFact]
    public void KnowledgeSourceStalePreviewExplainsFailureAndNativeDownloadUsesHostBridge()
    {
        JsonElement result = Run(KnowledgeLibrary + """
            R['/api/knowledge/collections/alpha/documents/doc1/preview?version=1'] = {
              __status: 409, body: { error: 'This source has been reindexed.' } };
            """, """
            window.TensorAgent.nativeReady({});
            var link = document.createElement('a');
            link.href = '/api/knowledge/collections/alpha/documents/doc1/source?version=1';
            document.dispatch('click', { target: link, preventDefault: function () {} });
            return settle(15).then(function () {
              var viewer = document.body.querySelector('.knowledge-source-viewer'), prevented = false;
              viewer.querySelector('a').dispatch('click', { preventDefault: function () { prevented = true; } });
              return settle(15).then(function () {
                return { text: viewer.textContent, prevented: prevented,
                  events: __page.requests('/api/agent/events').map(function (c) { return c.body; }) };
              });
            });
            """);
        Assert.Contains("This source has been reindexed.", result.GetProperty("text").GetString());
        Assert.True(result.GetProperty("prevented").GetBoolean());
        JsonElement sent = Assert.Single(result.GetProperty("events").EnumerateArray(), e => e.GetProperty("type").GetString() == "open-knowledge-source");
        Assert.Equal("/api/knowledge/collections/alpha/documents/doc1/source?version=1", sent.GetProperty("url").GetString());
    }
}
