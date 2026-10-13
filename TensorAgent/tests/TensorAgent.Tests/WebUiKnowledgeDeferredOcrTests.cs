using System.Text.Json;

namespace TensorAgent.Tests;

public sealed partial class WebUiPageTests
{
    private const string ManagedPdfSetup = """
        var pdfSetup = { managed: true, status: { state: 'not_started', downloadedBytes: 0, totalBytes: 100 } };
        R['/api/knowledge/ocr/status'] = function () { return pdfSetup; };
        R['/api/knowledge/ocr/prepare'] = function () { throw new Error('Setup must be deferred to OCR-eligible pages.'); };
        function uploadKnowledgeFile(name) {
          var input = __page.byId['knowledge-file-input'];
          input.files = [{ name: name }]; input.dispatch('change', { target: input });
          return settle(25);
        }
        function knowledgeUploads() {
          return __page.requests('/api/knowledge/collections/alpha/documents').filter(function (c) { return c.method === 'POST'; });
        }
        """;

    [WebJavaScriptFact]
    public void KnowledgePdfUploadPollsSetupProgressDuringTheActualUpload()
    {
        JsonElement result = Run(KnowledgeLibrary + ManagedPdfSetup + """
            var uploadFinish;
            R['/api/knowledge/collections/alpha/documents'] = function (call) {
              if (call.method === 'POST') return new Promise(function (answer) { uploadFinish = answer; });
              return { documents: libraryDocuments };
            };
            """, """
            return openKnowledgeLibrary().then(function () {
              pdfSetup.status = { state: 'downloading', downloadedBytes: 25, totalBytes: 100 };
              return uploadKnowledgeFile('paper.PDF');
            }).then(function () {
              var out = { inFlight: knowledgeUploads().length, progress: __page.byId['knowledge-status'].textContent,
                busy: __page.byId['knowledge-upload'].disabled };
              pdfSetup.status = { state: 'downloading', downloadedBytes: 65, totalBytes: 100 };
              return wait(450).then(function () {
                out.laterProgress = __page.byId['knowledge-status'].textContent;
                pdfSetup.status = { state: 'ready', downloadedBytes: 100, totalBytes: 100 };
                uploadFinish({ documents: libraryDocuments, errors: [] });
                return settle(25);
              }).then(function () {
                out.after = knowledgeUploads().length;
                out.prepares = __page.requests('/api/knowledge/ocr/prepare').length;
                out.fileName = knowledgeUploads()[0].parts[0].fileName;
                out.enabled = !__page.byId['knowledge-upload'].disabled;
                out.status = __page.byId['knowledge-status'].textContent;
                return out;
              });
            });
            """);
        Assert.Equal(1, result.GetProperty("inFlight").GetInt32());
        Assert.Contains("25%", result.GetProperty("progress").GetString());
        Assert.Contains("65%", result.GetProperty("laterProgress").GetString());
        Assert.True(result.GetProperty("busy").GetBoolean());
        Assert.Equal(1, result.GetProperty("after").GetInt32());
        Assert.Equal(0, result.GetProperty("prepares").GetInt32());
        Assert.Equal("paper.PDF", result.GetProperty("fileName").GetString());
        Assert.True(result.GetProperty("enabled").GetBoolean());
        Assert.Contains("1 files added", result.GetProperty("status").GetString());
    }

    [WebJavaScriptFact]
    public void KnowledgePdfSetupRetryRepeatsOnlyConfirmedFailedInputsEvenWithDuplicateNames()
    {
        JsonElement result = Run(KnowledgeLibrary + ManagedPdfSetup + """
            var attempts = 0;
            R['/api/knowledge/collections/alpha/documents'] = function (call) {
              if (call.method !== 'POST') return { documents: libraryDocuments };
              if (++attempts === 1) return { documents: [libraryDocuments[0]], errors: [
                { fileName: 'scan.pdf', fileIndex: 1, code: 'pdf_ocr_setup_failed', error: 'Download failed.' },
                { fileName: 'bad.pdf', fileIndex: 2, error: 'Malformed PDF.' },
                { fileName: 'missing.pdf', fileIndex: 99, code: 'pdf_ocr_setup_failed', error: 'Invalid input ordinal.' }
              ] };
              return { documents: [], errors: [] };
            };
            """, """
            return openKnowledgeLibrary().then(function () {
              var input = __page.byId['knowledge-file-input'];
              input.files = [{ name: 'scan.pdf', marker: 'committed' }, { name: 'scan.pdf', marker: 'failed' }, { name: 'bad.pdf', marker: 'malformed' }];
              input.dispatch('change', { target: input });
              return settle(25);
            }).then(function () {
              var out = { error: __page.byId['knowledge-error'].textContent,
                retryVisible: __page.byId['knowledge-ocr-retry'].style.display !== 'none' };
              __page.byId['knowledge-ocr-retry'].dispatch('click');
              return settle(30).then(function () {
                out.attempts = attempts; out.firstCount = knowledgeUploads()[0].parts.filter(function (p) { return p.name === 'files'; }).length;
                out.retryCount = knowledgeUploads()[1].parts.filter(function (p) { return p.name === 'files'; }).length;
                out.retried = knowledgeUploads()[1].parts[0].value.marker;
                out.retryHidden = __page.byId['knowledge-ocr-retry'].style.display === 'none';
                return out;
              });
            });
            """);
        Assert.Contains("Retry to continue", result.GetProperty("error").GetString());
        Assert.True(result.GetProperty("retryVisible").GetBoolean());
        Assert.Equal(2, result.GetProperty("attempts").GetInt32());
        Assert.Equal(3, result.GetProperty("firstCount").GetInt32());
        Assert.Equal(1, result.GetProperty("retryCount").GetInt32());
        Assert.Equal("failed", result.GetProperty("retried").GetString());
        Assert.True(result.GetProperty("retryHidden").GetBoolean());
    }

    [WebJavaScriptFact]
    public void KnowledgePdfReindexFailureCanRetryOnlyWhenTheServerConfirmsSetupFailure()
    {
        JsonElement result = Run(KnowledgeLibrary + ManagedPdfSetup + """
            libraryDocuments[0].fileName = 'scan.pdf';
            var attempts = 0;
            R['/api/knowledge/collections/alpha/documents/doc1/reindex'] = function () {
              if (++attempts === 1) return { __status: 503, body: { code: 'pdf_ocr_setup_failed', error: 'Setup failed.' } };
              libraryDocuments[0].version++; return { document: libraryDocuments[0] };
            };
            """, """
            return openKnowledgeLibrary().then(function () {
              __page.byId['knowledge-documents'].querySelector('.knowledge-reindex').dispatch('click');
              return settle(20);
            }).then(function () {
              var out = { versionBeforeRetry: libraryDocuments[0].version,
                retryVisible: __page.byId['knowledge-ocr-retry'].style.display !== 'none' };
              __page.byId['knowledge-ocr-retry'].dispatch('click');
              return settle(20).then(function () {
                out.version = libraryDocuments[0].version; out.attempts = attempts;
                out.prepares = __page.requests('/api/knowledge/ocr/prepare').length;
                return out;
              });
            });
            """);
        Assert.Equal(1, result.GetProperty("versionBeforeRetry").GetInt32());
        Assert.True(result.GetProperty("retryVisible").GetBoolean());
        Assert.Equal(2, result.GetProperty("version").GetInt32());
        Assert.Equal(2, result.GetProperty("attempts").GetInt32());
        Assert.Equal(0, result.GetProperty("prepares").GetInt32());
    }

    [WebJavaScriptFact]
    public void KnowledgeCollectionChangeDropsFailedPdfReindexRetry()
    {
        JsonElement result = Run(KnowledgeLibrary + ManagedPdfSetup + """
            libraryDocuments[0].fileName = 'scan.pdf';
            R['/api/knowledge/collections/alpha/documents/doc1/reindex'] = { __status: 503,
              body: { code: 'pdf_ocr_setup_failed', error: 'Setup failed.' } };
            """, """
            return openKnowledgeLibrary().then(function () {
              __page.byId['knowledge-documents'].querySelector('.knowledge-reindex').dispatch('click');
              return settle(20);
            }).then(function () {
              var out = { retryVisible: __page.byId['knowledge-ocr-retry'].style.display !== 'none' };
              knowledgeCollection('beta').querySelector('.knowledge-browse').dispatch('click');
              return settle(20).then(function () {
                out.retryHidden = __page.byId['knowledge-ocr-retry'].style.display === 'none';
                __page.byId['knowledge-ocr-retry'].dispatch('click'); return settle(20);
              }).then(function () {
                out.reindexes = __page.requests('/api/knowledge/collections/alpha/documents/doc1/reindex').length; return out;
              });
            });
            """);
        Assert.True(result.GetProperty("retryVisible").GetBoolean());
        Assert.True(result.GetProperty("retryHidden").GetBoolean());
        Assert.Equal(1, result.GetProperty("reindexes").GetInt32());
    }

    [WebJavaScriptFact]
    public void KnowledgeTextPdfUploadAndReindexNeverPrepareOrWaitForStatus()
    {
        JsonElement result = Run(KnowledgeLibrary + ManagedPdfSetup + Held + """
            libraryDocuments[0].fileName = 'text-only.pdf';
            """, """
            return openKnowledgeLibrary().then(function () {
              hold('/api/knowledge/ocr/status');
              return uploadKnowledgeFile('text-only.pdf');
            }).then(function () {
              var out = { uploads: knowledgeUploads().length, enabled: !__page.byId['knowledge-upload'].disabled };
              __page.byId['knowledge-documents'].querySelector('.knowledge-reindex').dispatch('click');
              return settle(20).then(function () {
                out.reindexes = __page.requests('/api/knowledge/collections/alpha/documents/doc1/reindex').length;
                out.prepares = __page.requests('/api/knowledge/ocr/prepare').length;
                heldFor('/api/knowledge/ocr/status').forEach(function (h) { h.answer({ managed: true,
                  status: { state: 'downloading', downloadedBytes: 1, totalBytes: 100 } }); });
                return settle(20);
              }).then(function () { out.lateStatus = __page.byId['knowledge-status'].textContent; return out; });
            });
            """);
        Assert.Equal(1, result.GetProperty("uploads").GetInt32());
        Assert.True(result.GetProperty("enabled").GetBoolean());
        Assert.Equal(1, result.GetProperty("reindexes").GetInt32());
        Assert.Equal(0, result.GetProperty("prepares").GetInt32());
        Assert.Equal("", result.GetProperty("lateStatus").GetString());
    }

    [WebJavaScriptFact]
    public void KnowledgeTextUploadSkipsPdfStatusAndAmbiguousFailureCannotRetryEvenWhenSetupFailed()
    {
        JsonElement result = Run(KnowledgeLibrary + ManagedPdfSetup + """
            pdfSetup.status = { state: 'failed', retryable: true };
            R['/api/knowledge/collections/alpha/documents'] = function (call) {
              if (call.method === 'POST') return { __status: 503, body: { error: 'Connection lost after sending file.' } };
              return { documents: libraryDocuments };
            };
            """, """
            return openKnowledgeLibrary().then(function () { return uploadKnowledgeFile('notes.txt'); }).then(function () {
              var out = { statusReads: __page.requests('/api/knowledge/ocr/status').length,
                textRetryHidden: __page.byId['knowledge-ocr-retry'].style.display === 'none' };
              return uploadKnowledgeFile('scan.pdf').then(function () {
                out.pdfRetryHidden = __page.byId['knowledge-ocr-retry'].style.display === 'none';
                __page.byId['knowledge-ocr-retry'].dispatch('click'); return settle(20);
              }).then(function () { out.uploads = knowledgeUploads().length; return out; });
            });
            """);
        Assert.Equal(0, result.GetProperty("statusReads").GetInt32());
        Assert.True(result.GetProperty("textRetryHidden").GetBoolean());
        Assert.True(result.GetProperty("pdfRetryHidden").GetBoolean());
        Assert.Equal(2, result.GetProperty("uploads").GetInt32());
    }
}
