using System;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Knowledge;

namespace TensorSharp.Server.Knowledge;

/// <summary>Prepares dependencies only when the core encounters an eligible OCR page.
/// Import and reindex through the SDK and every host API use the same path.</summary>
internal sealed class ManagedPdfKnowledgePageExtractor(
    Func<CancellationToken, Task<PdfOcrOptions>> prepare, PdfOcrOptions requested) : IKnowledgePdfPageExtractor
{
    private readonly object _gate = new();
    private PdfOcrOptions? _options;
    private PdfKnowledgePageExtractor? _pdf;

    public async Task PrepareAsync(CancellationToken cancellationToken = default)
    {
        var options = await prepare(cancellationToken).ConfigureAwait(false);
        lock (_gate)
        {
            if (_pdf == null || _options != options)
            {
                _options = options;
                _pdf = new PdfKnowledgePageExtractor(MergeOptions(requested, options));
            }
        }
    }

    internal static PdfOcrOptions MergeOptions(PdfOcrOptions requested, PdfOcrOptions installed)
        => requested with { Provider = "tesseract", PdfToPpmExecutable = installed.PdfToPpmExecutable,
            TesseractExecutable = installed.TesseractExecutable, TessdataDirectory = installed.TessdataDirectory,
            ProcessEnvironment = installed.ProcessEnvironment, Languages = requested.Languages ?? installed.Languages };

    public async Task<KnowledgePdfPageExtraction> ExtractPageAsync(KnowledgePdfPageContext context,
        CancellationToken cancellationToken = default)
    {
        PdfKnowledgePageExtractor? pdf;
        lock (_gate) pdf = _pdf;
        if (pdf == null)
        {
            await PrepareAsync(cancellationToken).ConfigureAwait(false);
            lock (_gate) pdf = _pdf;
        }
        return await pdf!.ExtractPageAsync(context, cancellationToken).ConfigureAwait(false);
    }
}
