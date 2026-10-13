using System.Text.Json;

namespace TensorAgent.Tests;

public sealed partial class WebUiPageTests
{
    [WebJavaScriptFact]
    public void KnowledgeImageResultsAndExpandedExcerptsShareAVersionedThumbnailPreview()
    {
        JsonElement result = Run(KnowledgeLibrary + """
            libraryHit.fileName = 'scan.png'; libraryHit.modality = 'image';
            R['/api/knowledge/read'] = { results: [Object.assign({}, libraryHit, { text: 'Expanded visible text.' })] };
            R['/api/knowledge/collections/alpha/documents/doc1/preview?version=1'] = {
              document: { fileName: 'scan.png' }, text: 'Visible text.', inlineContentType: 'image/png',
              sourceUrl: 'https://untrusted.example/scan.png' };
            """, """
            return openKnowledgeLibrary().then(runKnowledgeSearch).then(function () {
              var row = __page.byId['knowledge-results'].querySelector('.knowledge-result');
              var image = row.querySelector('img'), thumbnailLink = row.querySelector('.knowledge-thumbnail-link');
              var shown = { src: image.src, lazy: image.loading, alt: image.alt, link: thumbnailLink.href,
                label: thumbnailLink.getAttribute('aria-label') };
              row.querySelector('.knowledge-expand').dispatch('click');
              return settle(20).then(function () {
                var expanded = __page.byId['knowledge-read'].querySelector('img');
                image.dispatch('error');
                return { shown: shown, expanded: expanded.src, failedImageRemoved: !row.querySelector('img'),
                  fallback: !!row.querySelector('.knowledge-thumbnail-placeholder'),
                  previews: __page.calls.filter(function (call) { return call.url.indexOf('/preview?') >= 0; }).length };
              });
            });
            """);
        JsonElement shown = result.GetProperty("shown");
        Assert.Equal("/api/knowledge/collections/alpha/documents/doc1/source?version=1&inline=1", shown.GetProperty("src").GetString());
        Assert.Equal("lazy", shown.GetProperty("lazy").GetString());
        Assert.Equal("", shown.GetProperty("alt").GetString());
        Assert.Contains("scan.png", shown.GetProperty("label").GetString());
        Assert.Equal("/api/knowledge/collections/alpha/documents/doc1/source?version=1", shown.GetProperty("link").GetString());
        Assert.Equal(shown.GetProperty("src").GetString(), result.GetProperty("expanded").GetString());
        Assert.True(result.GetProperty("failedImageRemoved").GetBoolean());
        Assert.True(result.GetProperty("fallback").GetBoolean());
        Assert.Equal(1, result.GetProperty("previews").GetInt32());
    }

    [WebJavaScriptFact]
    public void KnowledgeSavedCitationThumbnailsKeepOnlyCitedHostIdentitiesAndOpenOriginalPreview()
    {
        JsonElement result = Run(KnowledgeLibrary + """
            libraryHit.fileName = 'scan.png'; libraryHit.modality = 'image';
            R['/api/sessions?conversation=saved'] = { sessionId: 's2', conversationId: 'saved',
              messages: [{ role: 'assistant', content: 'Evidence [knowledge:alpha/doc1/v1/0].', knowledgeSources: [libraryHit,
                Object.assign({}, libraryHit, { documentId: 'doc2', reference: 'unused' }),
                Object.assign({}, libraryHit, { documentId: '../outside' })] }] };
            R['/api/knowledge/collections/alpha/documents/doc1/preview?version=1'] = {
              document: { fileName: 'scan.png' }, text: 'Visible text.', inlineContentType: 'image/png' };
            """, """
            window.TensorAgent.openConversation('saved');
            return settle(25).then(function () {
              var box = __page.byId['chat'].querySelector('.knowledge-citations'), link = box.querySelector('.knowledge-source');
              var shown = { src: link.querySelector('img').src, label: link.querySelector('.knowledge-source-label').textContent,
                links: box.querySelectorAll('a').length, previews: __page.calls.filter(function (call) { return call.url.indexOf('/preview?') >= 0; }).map(function (call) { return call.url; }) };
              link.dispatch('click');
              return settle(20).then(function () {
                return { shown: shown, original: document.body.querySelector('.knowledge-source-viewer').querySelector('img').src };
              });
            });
            """);
        JsonElement shown = result.GetProperty("shown");
        Assert.Equal(1, shown.GetProperty("links").GetInt32());
        Assert.Contains("scan.png", shown.GetProperty("label").GetString());
        Assert.Equal("/api/knowledge/collections/alpha/documents/doc1/source?version=1&inline=1", shown.GetProperty("src").GetString());
        Assert.Equal(shown.GetProperty("src").GetString(), result.GetProperty("original").GetString());
        Assert.Equal("/api/knowledge/collections/alpha/documents/doc1/preview?version=1",
            Assert.Single(shown.GetProperty("previews").EnumerateArray()).GetString());
    }

    [WebJavaScriptFact]
    public void KnowledgeVideoThumbnailsSeekToRetrievedFrameAndOtherMediaHaveSafeFallbacks()
    {
        JsonElement result = Run(KnowledgeLibrary + """
            R['/api/knowledge/search'] = { results: [
              Object.assign({}, libraryHit, { fileName: 'movie.mp4', modality: 'video', location: { startMilliseconds: 4250 } }),
              Object.assign({}, libraryHit, { documentId: 'audio', fileName: 'recording.wav', modality: 'audio' }),
              Object.assign({}, libraryHit, { documentId: 'svg', fileName: 'unsafe.svg', modality: 'image' }),
              Object.assign({}, libraryHit, { documentId: '../outside', fileName: 'outside.png', modality: 'image' })] };
            R['/api/knowledge/collections/alpha/documents/doc1/preview?version=1'] = { inlineContentType: 'video/mp4', sourceUrl: 'javascript:attack()' };
            R['/api/knowledge/collections/alpha/documents/audio/preview?version=1'] = { inlineContentType: 'audio/wav' };
            R['/api/knowledge/collections/alpha/documents/svg/preview?version=1'] = { inlineContentType: 'image/svg+xml' };
            """, """
            return openKnowledgeLibrary().then(runKnowledgeSearch).then(function () {
              var box = __page.byId['knowledge-results'], video = box.querySelector('video');
              video.duration = 10; video.dispatch('loadedmetadata');
              return { src: video.src, muted: video.muted, preload: video.preload, seek: video.currentTime,
                playsInline: video.getAttribute('playsinline'), images: box.querySelectorAll('img').length,
                thumbnails: box.querySelectorAll('.knowledge-thumbnail-link').length,
                fallbacks: box.querySelectorAll('.knowledge-thumbnail-placeholder').length,
                previews: __page.calls.filter(function (call) { return call.url.indexOf('/preview?') >= 0; }).length };
            });
            """);
        Assert.Equal("/api/knowledge/collections/alpha/documents/doc1/source?version=1&inline=1", result.GetProperty("src").GetString());
        Assert.True(result.GetProperty("muted").GetBoolean());
        Assert.Equal("metadata", result.GetProperty("preload").GetString());
        Assert.Equal("", result.GetProperty("playsInline").GetString());
        Assert.Equal(4.25, result.GetProperty("seek").GetDouble());
        Assert.Equal(0, result.GetProperty("images").GetInt32());
        Assert.Equal(3, result.GetProperty("thumbnails").GetInt32());
        Assert.Equal(2, result.GetProperty("fallbacks").GetInt32());
        Assert.Equal(3, result.GetProperty("previews").GetInt32());
    }

    [WebJavaScriptFact]
    public void KnowledgeThumbnailFailuresDoNotHideSourcesOrReuseMetadataFromAnotherVersion()
    {
        JsonElement result = Run(KnowledgeLibrary + """
            libraryHit.fileName = 'scan.png'; libraryHit.modality = 'image';
            R['/api/knowledge/collections/alpha/documents/doc1/preview?version=1'] = {
              __status: 409, body: { error: 'This source has been reindexed.' } };
            R['/api/knowledge/collections/alpha/documents/doc1/preview?version=2'] = { inlineContentType: 'image/png' };
            """, """
            return openKnowledgeLibrary().then(runKnowledgeSearch).then(function () {
              var first = __page.byId['knowledge-results'];
              var stale = { fallback: !!first.querySelector('.knowledge-thumbnail-placeholder'),
                link: first.querySelector('.knowledge-source').href, error: __page.byId['knowledge-error'].textContent };
              libraryHit.version = 2;
              return runKnowledgeSearch().then(function () {
                var image = __page.byId['knowledge-results'].querySelector('img');
                return { stale: stale, updated: image.src,
                  previews: __page.calls.filter(function (call) { return call.url.indexOf('/preview?') >= 0; }).map(function (call) { return call.url; }) };
              });
            });
            """);
        Assert.True(result.GetProperty("stale").GetProperty("fallback").GetBoolean());
        Assert.EndsWith("version=1", result.GetProperty("stale").GetProperty("link").GetString());
        Assert.Equal("", result.GetProperty("stale").GetProperty("error").GetString());
        Assert.Equal("/api/knowledge/collections/alpha/documents/doc1/source?version=2&inline=1", result.GetProperty("updated").GetString());
        Assert.Equal(2, result.GetProperty("previews").GetArrayLength());
    }

    [WebJavaScriptFact]
    public void KnowledgeThumbnailRetriesFailedMetadataOnNextSearchAndCachesRecoveredPreview()
    {
        JsonElement result = Run(KnowledgeLibrary + """
            libraryHit.fileName = 'scan.png'; libraryHit.modality = 'metadata';
            var previews = 0;
            R['/api/knowledge/collections/alpha/documents/doc1/preview?version=1'] = function () {
              previews++;
              return previews === 1 ? { __reject: 'Server briefly unavailable.' } : { inlineContentType: 'image/png' };
            };
            """, """
            return openKnowledgeLibrary().then(runKnowledgeSearch).then(function () {
              var fallback = !!__page.byId['knowledge-results'].querySelector('.knowledge-thumbnail-placeholder');
              return runKnowledgeSearch().then(function () {
                var recovered = __page.byId['knowledge-results'].querySelector('img').src;
                return runKnowledgeSearch().then(function () {
                  return { fallback: fallback, recovered: recovered, cached: __page.byId['knowledge-results'].querySelector('img').src,
                    previews: previews, error: __page.byId['knowledge-error'].textContent };
                });
              });
            });
            """);
        Assert.True(result.GetProperty("fallback").GetBoolean());
        Assert.Equal("/api/knowledge/collections/alpha/documents/doc1/source?version=1&inline=1", result.GetProperty("recovered").GetString());
        Assert.Equal(result.GetProperty("recovered").GetString(), result.GetProperty("cached").GetString());
        Assert.Equal(2, result.GetProperty("previews").GetInt32());
        Assert.Equal("", result.GetProperty("error").GetString());
    }
}
