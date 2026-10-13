using System;
using System.Collections.Generic;

namespace TensorSharp.Server.Knowledge;

/// <summary>Local PDF OCR configuration. TensorAgent can supply an app-managed installation.</summary>
public sealed record PdfOcrOptions
{
    /// <summary>auto prefers Apple Vision, otherwise local Tesseract on desktop. apple, tesseract and none select explicitly.</summary>
    public string Provider { get; init; } = "auto";
    public string PdfToPpmExecutable { get; init; }
    public string TesseractExecutable { get; init; }
    public string TessdataDirectory { get; init; }
    /// <summary>Tesseract language names separated by +. Null selects installed English and Chinese packs.</summary>
    public string Languages { get; init; }
    /// <summary>Per-child-process environment for an isolated native runtime. Does not change the host environment.</summary>
    public IReadOnlyDictionary<string, string> ProcessEnvironment { get; init; }
    /// <summary>Portable render resolution before applying the page pixel budget.</summary>
    public int Dpi { get; init; } = 300;
    /// <summary>Safety ceiling for direct provider calls. The knowledge extractor's per-page
    /// deadline normally expires first; this ceiling never extends a caller's deadline.</summary>
    public int TimeoutSeconds { get; init; } = 600;
    /// <summary>Maximum source PDF size for direct provider calls, before PdfPig or Poppler opens it.</summary>
    public long MaxInputBytes { get; init; } = 64L * 1024 * 1024;
    /// <summary>Maximum sum of embedded image pixels on one page, counting repeated placements.
    /// This preflight bounds normal decoding work, not all internal allocations of a native renderer.</summary>
    public long MaxEmbeddedImagePixels { get; init; } = 64_000_000;

    public static PdfOcrOptions FromEnvironment() => new()
    {
        Provider = Environment.GetEnvironmentVariable("TS_KNOWLEDGE_PDF_OCR") ?? "auto",
        PdfToPpmExecutable = Environment.GetEnvironmentVariable("TS_KNOWLEDGE_PDFTOPPM_PATH"),
        TesseractExecutable = Environment.GetEnvironmentVariable("TS_KNOWLEDGE_TESSERACT_PATH"),
        TessdataDirectory = Environment.GetEnvironmentVariable("TS_KNOWLEDGE_TESSDATA_DIRECTORY"),
        Languages = Environment.GetEnvironmentVariable("TS_KNOWLEDGE_OCR_LANGUAGES"),
    };
}
