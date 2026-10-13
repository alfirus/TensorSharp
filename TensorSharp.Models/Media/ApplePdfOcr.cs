// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace TensorSharp.Models.Media;

/// <summary>Local whole-page PDF rasterization and text recognition using Apple's
/// system CoreGraphics and Vision frameworks. No model, subprocess or network is used.
/// This uses the C/Objective-C ABI so the same implementation runs in a net10.0 Mac
/// host and in the iOS/Mac Catalyst app; it does not reference desktop-only packages.</summary>
public static class ApplePdfOcr
{
    public static bool IsSupported => OperatingSystem.IsMacOSVersionAtLeast(13)
        || OperatingSystem.IsIOSVersionAtLeast(16) || OperatingSystem.IsMacCatalystVersionAtLeast(16);

    /// <summary>Recognize a one-based page, honoring its crop box and rotation. Rendering
    /// is limited to 200 DPI and maxPixels. Cancellation can interrupt Vision recognition;
    /// synchronous CoreGraphics drawing is checked before and after, not forcibly aborted.</summary>
    public static string RecognizePage(string path, int pageNumber, int maxPixels, CancellationToken cancellationToken = default, int maxCharacters = 2_000_000)
    {
        if (maxCharacters <= 0) throw new ArgumentOutOfRangeException(nameof(maxCharacters));
        return WithPage(path, pageNumber, maxPixels, cancellationToken, (image, _, _) => Recognize(image, cancellationToken, maxCharacters));
    }

    /// <summary>Render the complete visible page for a configured alternative extractor.</summary>
    public static byte[] RenderPagePng(string path, int pageNumber, int maxPixels, CancellationToken cancellationToken = default)
        => WithPage(path, pageNumber, maxPixels, cancellationToken, (image, width, height) =>
        {
            IntPtr provider = CGImageGetDataProvider(image);
            IntPtr data = CGDataProviderCopyData(provider);
            if (data == IntPtr.Zero) throw new InvalidDataException("The PDF page raster could not be read.");
            try
            {
                int length = checked(width * height * 4);
                if ((long)CFDataGetLength(data) != length) throw new InvalidDataException("Unexpected PDF raster stride.");
                var pixels = new byte[length];
                Marshal.Copy(CFDataGetBytePtr(data), pixels, 0, length);
                return ManagedMediaProvider.Instance.EncodePng(pixels, width, height, 4);
            }
            finally { CFRelease(data); }
        });

    private static T WithPage<T>(string path, int pageNumber, int maxPixels, CancellationToken ct, Func<IntPtr, int, int, T> consume)
    {
        if (!IsSupported) throw new PlatformNotSupportedException("Local Apple PDF OCR requires macOS 13 or iOS/Mac Catalyst 16 or later.");
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (pageNumber < 1) throw new ArgumentOutOfRangeException(nameof(pageNumber));
        if (maxPixels is < 65536 or > 67_108_864) throw new ArgumentOutOfRangeException(nameof(maxPixels));
        ct.ThrowIfCancellationRequested();
        // Keep framework handles for process lifetime: Objective-C classes outlive any one request.
        _ = Frameworks.Value;
        IntPtr pool = Send(ObjcClass("NSAutoreleasePool"), Sel("new"));
        IntPtr url = IntPtr.Zero, document = IntPtr.Zero, color = IntPtr.Zero, context = IntPtr.Zero, image = IntPtr.Zero;
        try
        {
            byte[] name = Encoding.UTF8.GetBytes(Path.GetFullPath(path));
            url = CFURLCreateFromFileSystemRepresentation(IntPtr.Zero, name, name.Length, false);
            if (url == IntPtr.Zero) throw new InvalidDataException("The PDF source path could not be opened.");
            document = CGPDFDocumentCreateWithURL(url);
            if (document == IntPtr.Zero || !CGPDFDocumentIsUnlocked(document))
                throw new InvalidDataException("The PDF is corrupt or password protected.");
            if ((nuint)pageNumber > CGPDFDocumentGetNumberOfPages(document)) throw new ArgumentOutOfRangeException(nameof(pageNumber));
            IntPtr page = CGPDFDocumentGetPage(document, (nuint)pageNumber);
            Rect box = CGPDFPageGetBoxRect(page, 1); // crop box
            double widthPoints = box.Width, heightPoints = box.Height;
            if (CGPDFPageGetRotationAngle(page) % 180 != 0) (widthPoints, heightPoints) = (heightPoints, widthPoints);
            if (!double.IsFinite(box.X) || !double.IsFinite(box.Y) || !double.IsFinite(widthPoints) || !double.IsFinite(heightPoints)
                || widthPoints <= 0 || heightPoints <= 0 || widthPoints > 1_000_000 || heightPoints > 1_000_000)
                throw new InvalidDataException("The PDF page has invalid dimensions.");
            double scale = Math.Min(200.0 / 72.0, Math.Sqrt(maxPixels / (widthPoints * heightPoints)));
            int width = Math.Max(1, (int)Math.Floor(widthPoints * scale));
            int height = Math.Max(1, (int)Math.Floor(heightPoints * scale));
            if ((long)width * height > maxPixels || width > 32768 || height > 32768)
                throw new InvalidDataException("The PDF page dimensions exceed the raster limit.");
            color = CGColorSpaceCreateDeviceRGB();
            context = CGBitmapContextCreate(IntPtr.Zero, (nuint)width, (nuint)height, 8, (nuint)checked(width * 4), color, 0x4001);
            if (context == IntPtr.Zero) throw new InvalidDataException("The PDF page raster could not be allocated.");
            var target = new Rect { Width = width, Height = height };
            CGContextSetRGBFillColor(context, 1, 1, 1, 1);
            CGContextFillRect(context, target);
            // CGPDFPageGetDrawingTransform fits down but never scales a PDF above 100%.
            // Apply the raster DPI scale explicitly; let CoreGraphics handle crop/rotation
            // in page-point coordinates, otherwise a 200-DPI canvas contains 72-DPI text.
            CGContextScaleCTM(context, scale, scale);
            var pageTarget = new Rect { Width = width / scale, Height = height / scale };
            CGContextConcatCTM(context, CGPDFPageGetDrawingTransform(page, 1, pageTarget, 0, true));
            ct.ThrowIfCancellationRequested();
            CGContextDrawPDFPage(context, page);
            ct.ThrowIfCancellationRequested();
            image = CGBitmapContextCreateImage(context);
            if (image == IntPtr.Zero) throw new InvalidDataException("The PDF page did not render.");
            return consume(image, width, height);
        }
        finally
        {
            if (image != IntPtr.Zero) CGImageRelease(image);
            if (context != IntPtr.Zero) CGContextRelease(context);
            if (color != IntPtr.Zero) CGColorSpaceRelease(color);
            if (document != IntPtr.Zero) CGPDFDocumentRelease(document);
            if (url != IntPtr.Zero) CFRelease(url);
            if (pool != IntPtr.Zero) SendVoid(pool, Sel("drain"));
        }
    }

    private static string Recognize(IntPtr image, CancellationToken ct, int maxCharacters)
    {
        IntPtr request = Send(ObjcClass("VNRecognizeTextRequest"), Sel("new"));
        IntPtr handler = IntPtr.Zero;
        if (request == IntPtr.Zero) throw new NotSupportedException("Apple Vision text recognition is unavailable.");
        try
        {
            // Explicit revision avoids the English-only legacy default for a plain .NET host
            // which has no Apple SDK-linked application bundle. Revision 3 adds CJK recognition.
            SendInteger(request, Sel("setRevision:"), 3);
            SendInteger(request, Sel("setRecognitionLevel:"), 0); // accurate
            SendBool(request, Sel("setUsesLanguageCorrection:"), false); // preserve codes/numbers
            SendBool(request, Sel("setAutomaticallyDetectsLanguage:"), true);
            SelectLanguages(request);
            IntPtr options = Send(ObjcClass("NSDictionary"), Sel("dictionary"));
            handler = SendTwo(Send(ObjcClass("VNImageRequestHandler"), Sel("alloc")), Sel("initWithCGImage:options:"), image, options);
            if (handler == IntPtr.Zero) throw new NotSupportedException("Apple Vision image processing is unavailable.");
            IntPtr requests = SendOne(ObjcClass("NSArray"), Sel("arrayWithObject:"), request);
            IntPtr error = IntPtr.Zero;
            using (ct.Register(() => SendVoid(request, Sel("cancel"))))
            {
                ct.ThrowIfCancellationRequested();
                bool ok = SendPerform(handler, Sel("performRequests:error:"), requests, ref error);
                ct.ThrowIfCancellationRequested();
                if (!ok) throw new InvalidDataException("Apple Vision could not recognize this PDF page: " + ErrorText(error));
            }
            IntPtr observations = Send(request, Sel("results"));
            nuint count = observations == IntPtr.Zero ? 0 : SendCount(observations, Sel("count"));
            if (count > 100000) throw new InvalidDataException("PDF OCR returned too many text observations.");
            var lines = new List<(double Y, double X, string Text)>();
            int characters = 0;
            for (nuint i = 0; i < count; i++)
            {
                ct.ThrowIfCancellationRequested();
                IntPtr observation = SendIndex(observations, Sel("objectAtIndex:"), i);
                IntPtr candidates = SendIndex(observation, Sel("topCandidates:"), 1);
                if (candidates == IntPtr.Zero || SendCount(candidates, Sel("count")) == 0) continue;
                IntPtr candidate = SendIndex(candidates, Sel("objectAtIndex:"), 0);
                string text = StringValue(Send(candidate, Sel("string")));
                if (string.IsNullOrWhiteSpace(text)) continue;
                if (text.Length > maxCharacters - characters - (lines.Count > 0 ? 1 : 0))
                    throw new PdfOcrLimitException("PDF OCR text exceeds the remaining extraction limit; split the document.");
                characters += text.Length + (lines.Count > 0 ? 1 : 0);
                Rect bounds;
                if (RuntimeInformation.ProcessArchitecture == System.Runtime.InteropServices.Architecture.X64) SendRectX64(out bounds, observation, Sel("boundingBox"));
                else bounds = SendRect(observation, Sel("boundingBox"));
                lines.Add((bounds.Y + bounds.Height, bounds.X, text));
            }
            // Top-to-bottom, left-to-right is a textual reading order, not reconstruction
            // of complex tables/columns. Keep OCR distinct from the selectable text layer.
            return string.Join("\n", lines.OrderByDescending(line => Math.Round(line.Y, 3)).ThenBy(line => line.X).Select(line => line.Text));
        }
        finally
        {
            if (handler != IntPtr.Zero) SendVoid(handler, Sel("release"));
            SendVoid(request, Sel("release"));
        }
    }

    private static void SelectLanguages(IntPtr request)
    {
        IntPtr error = IntPtr.Zero;
        IntPtr supported = SendError(request, Sel("supportedRecognitionLanguagesAndReturnError:"), ref error);
        if (supported == IntPtr.Zero) return;
        var names = new HashSet<string>(StringComparer.Ordinal);
        nuint count = SendCount(supported, Sel("count"));
        for (nuint i = 0; i < Math.Min(count, 256); i++) names.Add(StringValue(SendIndex(supported, Sel("objectAtIndex:"), i)));
        IntPtr selected = Send(ObjcClass("NSMutableArray"), Sel("array"));
        // Vision requires Chinese first, paired only with English.
        // https://developer.apple.com/documentation/vision/recognizing-text-in-images
        foreach (string language in new[] { "zh-Hans", "zh-Hant", "en-US" })
            if (names.Contains(language)) SendOne(selected, Sel("addObject:"), SendString(ObjcClass("NSString"), Sel("stringWithUTF8String:"), language));
        if (SendCount(selected, Sel("count")) > 0) SendOne(request, Sel("setRecognitionLanguages:"), selected);
    }

    private static string StringValue(IntPtr value) => value == IntPtr.Zero ? "" : Marshal.PtrToStringUTF8(Send(value, Sel("UTF8String"))) ?? "";
    private static string ErrorText(IntPtr error) => error == IntPtr.Zero ? "unknown recognition error" : StringValue(Send(error, Sel("localizedDescription")));
    private static IntPtr ObjcClass(string name) => objc_getClass(name);
    private static IntPtr Sel(string name) => sel_registerName(name);
    private static readonly Lazy<IntPtr[]> Frameworks = new(() => new[]
    {
        NativeLibrary.Load("/System/Library/Frameworks/Foundation.framework/Foundation"),
        NativeLibrary.Load("/System/Library/Frameworks/Vision.framework/Vision"),
    });
    private const string ObjC = "/usr/lib/libobjc.A.dylib";
    private const string CG = "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics";
    private const string CF = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation";
    [StructLayout(LayoutKind.Sequential)] private struct Rect { public double X, Y, Width, Height; }
    [StructLayout(LayoutKind.Sequential)] private struct Transform { public double A, B, C, D, Tx, Ty; }
    [DllImport(ObjC)] private static extern IntPtr objc_getClass(string name);
    [DllImport(ObjC)] private static extern IntPtr sel_registerName(string name);
    [DllImport(ObjC, EntryPoint = "objc_msgSend")] private static extern IntPtr Send(IntPtr target, IntPtr selector);
    [DllImport(ObjC, EntryPoint = "objc_msgSend")] private static extern void SendVoid(IntPtr target, IntPtr selector);
    [DllImport(ObjC, EntryPoint = "objc_msgSend")] private static extern IntPtr SendOne(IntPtr target, IntPtr selector, IntPtr value);
    [DllImport(ObjC, EntryPoint = "objc_msgSend")] private static extern IntPtr SendTwo(IntPtr target, IntPtr selector, IntPtr first, IntPtr second);
    [DllImport(ObjC, EntryPoint = "objc_msgSend")] private static extern void SendInteger(IntPtr target, IntPtr selector, nint value);
    [DllImport(ObjC, EntryPoint = "objc_msgSend")] private static extern void SendBool(IntPtr target, IntPtr selector, [MarshalAs(UnmanagedType.I1)] bool value);
    [DllImport(ObjC, EntryPoint = "objc_msgSend")] private static extern IntPtr SendIndex(IntPtr target, IntPtr selector, nuint index);
    [DllImport(ObjC, EntryPoint = "objc_msgSend")] private static extern nuint SendCount(IntPtr target, IntPtr selector);
    [DllImport(ObjC, EntryPoint = "objc_msgSend")] private static extern IntPtr SendError(IntPtr target, IntPtr selector, ref IntPtr error);
    [DllImport(ObjC, EntryPoint = "objc_msgSend")] [return: MarshalAs(UnmanagedType.I1)]
    private static extern bool SendPerform(IntPtr target, IntPtr selector, IntPtr requests, ref IntPtr error);
    [DllImport(ObjC, EntryPoint = "objc_msgSend")] private static extern IntPtr SendString(IntPtr target, IntPtr selector, [MarshalAs(UnmanagedType.LPUTF8Str)] string value);
    [DllImport(ObjC, EntryPoint = "objc_msgSend")] private static extern Rect SendRect(IntPtr target, IntPtr selector);
    [DllImport(ObjC, EntryPoint = "objc_msgSend_stret")] private static extern void SendRectX64(out Rect result, IntPtr target, IntPtr selector);
    [DllImport(CF)] private static extern IntPtr CFURLCreateFromFileSystemRepresentation(IntPtr allocator, byte[] buffer, nint length, [MarshalAs(UnmanagedType.I1)] bool directory);
    [DllImport(CF)] private static extern void CFRelease(IntPtr value);
    [DllImport(CF)] private static extern nint CFDataGetLength(IntPtr data);
    [DllImport(CF)] private static extern IntPtr CFDataGetBytePtr(IntPtr data);
    [DllImport(CG)] private static extern IntPtr CGPDFDocumentCreateWithURL(IntPtr url);
    [DllImport(CG)] private static extern void CGPDFDocumentRelease(IntPtr document);
    [DllImport(CG)] [return: MarshalAs(UnmanagedType.I1)] private static extern bool CGPDFDocumentIsUnlocked(IntPtr document);
    [DllImport(CG)] private static extern nuint CGPDFDocumentGetNumberOfPages(IntPtr document);
    [DllImport(CG)] private static extern IntPtr CGPDFDocumentGetPage(IntPtr document, nuint pageNumber);
    [DllImport(CG)] private static extern Rect CGPDFPageGetBoxRect(IntPtr page, int box);
    [DllImport(CG)] private static extern int CGPDFPageGetRotationAngle(IntPtr page);
    [DllImport(CG)] private static extern Transform CGPDFPageGetDrawingTransform(IntPtr page, int box, Rect rect, int rotate, [MarshalAs(UnmanagedType.I1)] bool preserveAspect);
    [DllImport(CG)] private static extern IntPtr CGColorSpaceCreateDeviceRGB();
    [DllImport(CG)] private static extern void CGColorSpaceRelease(IntPtr space);
    [DllImport(CG)] private static extern IntPtr CGBitmapContextCreate(IntPtr data, nuint width, nuint height, nuint bits, nuint bytesPerRow, IntPtr color, uint info);
    [DllImport(CG)] private static extern void CGContextSetRGBFillColor(IntPtr context, double r, double g, double b, double a);
    [DllImport(CG)] private static extern void CGContextFillRect(IntPtr context, Rect rect);
    [DllImport(CG)] private static extern void CGContextScaleCTM(IntPtr context, double sx, double sy);
    [DllImport(CG)] private static extern void CGContextConcatCTM(IntPtr context, Transform transform);
    [DllImport(CG)] private static extern void CGContextDrawPDFPage(IntPtr context, IntPtr page);
    [DllImport(CG)] private static extern IntPtr CGBitmapContextCreateImage(IntPtr context);
    [DllImport(CG)] private static extern void CGContextRelease(IntPtr context);
    [DllImport(CG)] private static extern void CGImageRelease(IntPtr image);
    [DllImport(CG)] private static extern IntPtr CGImageGetDataProvider(IntPtr image);
    [DllImport(CG)] private static extern IntPtr CGDataProviderCopyData(IntPtr provider);
}

/// <summary>PDF OCR exceeded a resource budget; callers must not treat this as a blank page.</summary>
public sealed class PdfOcrLimitException(string message) : IOException(message);
