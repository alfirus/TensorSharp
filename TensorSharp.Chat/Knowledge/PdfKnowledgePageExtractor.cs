using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Knowledge;
using TensorSharp.Models.Media;

namespace TensorSharp.Server.Knowledge;

/// <summary>Local OCR of complete PDF pages: Apple Vision on macOS/iOS/Mac Catalyst,
/// Poppler and Tesseract on Windows/Linux. No chat model or network service is required.</summary>
public sealed class PdfKnowledgePageExtractor : IKnowledgePdfPageExtractor
{
    private readonly string _provider;
    private readonly PortablePdfPageExtractor _portable;

    public PdfKnowledgePageExtractor(PdfOcrOptions options = null)
    {
        options ??= PdfOcrOptions.FromEnvironment();
        _provider = options.Provider?.Trim().ToLowerInvariant() ?? "auto";
        if (_provider is not ("auto" or "apple" or "tesseract" or "none"))
            throw new ArgumentException("PDF OCR provider must be auto, apple, tesseract or none.", nameof(options));
        _portable = new PortablePdfPageExtractor(options);
    }

    public async Task<KnowledgePdfPageExtraction> ExtractPageAsync(KnowledgePdfPageContext context,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (context.MaxCharacters <= 0)
            throw new KnowledgeLimitException("PDF OCR has no remaining extraction character budget; split the document.");
        if (_provider == "none")
            return new KnowledgePdfPageExtraction { Warnings = new[] { "PDF OCR is disabled by configuration. Enable a provider and reindex scanned pages." } };
        if (_provider == "tesseract" || (_provider == "auto" && !ApplePdfOcr.IsSupported && PortablePdfPageExtractor.IsSupported))
            return await _portable.ExtractPageAsync(context, cancellationToken).ConfigureAwait(false);
        if (!ApplePdfOcr.IsSupported)
            return new KnowledgePdfPageExtraction
            {
                Warnings = new[] { "This platform has no configured PDF OCR provider. Supply an IKnowledgePdfPageExtractor and reindex scanned pages." }
            };
        string text;
        try
        {
            text = await Task.Run(() => ApplePdfOcr.RecognizePage(context.FilePath, context.PageNumber,
                context.MaxPixels, cancellationToken, context.MaxCharacters), cancellationToken).ConfigureAwait(false);
        }
        catch (PdfOcrLimitException error)
        {
            throw new KnowledgeLimitException(error.Message);
        }
        cancellationToken.ThrowIfCancellationRequested();
        if (text?.Length > context.MaxCharacters)
            throw new KnowledgeLimitException("PDF OCR text exceeds the remaining extraction limit; split the document.");
        return new KnowledgePdfPageExtraction
        {
            Text = text,
            ExtractorId = "apple-vision-accurate-v1:" + RuntimeInformation.OSDescription
        };
    }
}
