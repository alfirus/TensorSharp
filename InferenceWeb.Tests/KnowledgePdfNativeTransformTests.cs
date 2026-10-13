using TensorSharp.Models.Media;

namespace InferenceWeb.Tests;

/// <summary>Actual CoreGraphics rendering of offset crop boxes and all page rotations.</summary>
public sealed class KnowledgePdfNativeTransformTests : IDisposable
{
    private readonly string _path = Path.Combine(Path.GetTempPath(), "ts-pdf-transform-" + Guid.NewGuid().ToString("N") + ".pdf");

    public KnowledgePdfNativeTransformTests()
    {
        using var input = typeof(KnowledgePdfNativeTransformTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.Fixtures.Knowledge.crop-rotation.pdf")!;
        using var output = File.Create(_path);
        input.CopyTo(output);
    }

    public void Dispose() => File.Delete(_path);

    [ApplePdfOcrFact]
    public void RasterFillsTheOffsetCropBoxAt200DpiForAllPageRotations()
    {
        for (int page = 1; page <= 4; page++)
        {
            byte[] png = ApplePdfOcr.RenderPagePng(_path, page, 16_777_216);
            byte[] rgba = ManagedMediaProvider.Instance.DecodeRgba(png, out int width, out int height);
            Assert.Equal(page % 2 == 1 ? (1666, 1111) : (1111, 1666), (width, height));
            AssertCorners(rgba, width, height, page);
        }
    }

    [ApplePdfOcrFact]
    public void PixelBudgetDownscalesTheCompleteCropInsteadOfClippingIt()
    {
        for (int page = 1; page <= 4; page++)
        {
            byte[] png = ApplePdfOcr.RenderPagePng(_path, page, 65_536);
            byte[] rgba = ManagedMediaProvider.Instance.DecodeRgba(png, out int width, out int height);
            Assert.Equal(page % 2 == 1 ? (313, 209) : (209, 313), (width, height));
            Assert.InRange((long)width * height, 64_000, 65_536);
            AssertCorners(rgba, width, height, page);
        }
    }

    [ApplePdfOcrFact]
    public void VisionReadsUprightTextOnRotatedCropsAndExcludesOutsideText()
    {
        foreach (int page in new[] { 2, 4 })
        {
            string text = ApplePdfOcr.RecognizePage(_path, page, 16_777_216);
            Assert.Contains($"CROP {(page - 1) * 90}", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("OUTSIDE", text, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("9463", text);
        }
    }

    private static void AssertCorners(byte[] pixels, int width, int height, int page)
    {
        // Colors are defined in PDF coordinates before /Rotate: TL blue, TR yellow,
        // BL red, BR green. /Rotate rotates clockwise in the displayed page.
        string[] expected = page switch
        {
            1 => ["blue", "yellow", "red", "green"],
            2 => ["red", "blue", "green", "yellow"],
            3 => ["green", "red", "yellow", "blue"],
            4 => ["yellow", "green", "blue", "red"],
            _ => throw new ArgumentOutOfRangeException(nameof(page))
        };
        (int x, int y)[] corners = [(width / 20, height / 20), (width * 19 / 20, height / 20),
            (width / 20, height * 19 / 20), (width * 19 / 20, height * 19 / 20)];
        for (int i = 0; i < corners.Length; i++)
        {
            var (x, y) = corners[i];
            int offset = (y * width + x) * 4;
            byte[] color = expected[i] switch
            {
                "red" => [255, 0, 0, 255], "green" => [0, 255, 0, 255],
                "blue" => [0, 0, 255, 255], "yellow" => [255, 255, 0, 255],
                _ => throw new InvalidOperationException()
            };
            Assert.True(color.AsSpan().SequenceEqual(pixels.AsSpan(offset, 4)),
                $"Page {page}, corner {i}: expected {expected[i]} ({string.Join(',', color)}), " +
                $"actual {string.Join(',', pixels.AsSpan(offset, 4).ToArray())}. Crop, rotation or DPI scale is wrong.");
        }
    }
}
