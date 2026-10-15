// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;

namespace TensorSharp.Models.Ltx25;

/// <summary>The Ingredients adapter was trained with separate reference-sheet
/// and generated-video descriptions. Preserve explicit descriptions and put
/// ordinary user prompts in that format without inventing image contents.</summary>
public static class Ltx25ReferencePrompt
{
    public static string Format(string prompt, int images)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        if (images is < 1 or > Ltx25ReferenceConditioning.MaxImages)
            throw new ArgumentOutOfRangeException(nameof(images));
        bool canonical = prompt.Contains("Reference sheet:", StringComparison.OrdinalIgnoreCase)
            && prompt.Contains("Generated video:", StringComparison.OrdinalIgnoreCase);
        bool headings = prompt.Contains("Reference Sheet Description", StringComparison.OrdinalIgnoreCase)
            && prompt.Contains("Target Description", StringComparison.OrdinalIgnoreCase);
        if (canonical || headings) return prompt;
        int columns = (int)Math.Ceiling(Math.Sqrt(images)), rows = (images + columns - 1) / columns;
        string inventory = images == 1 ? "The sheet contains one input image."
            : $"The sheet contains {images} input image panels in a {columns}-column, {rows}-row grid, ordered left to right and top to bottom.";
        return $"Reference sheet: {inventory} Panel positions identify the reference sources. Use their visible subjects, objects and appearances to compose the video requested below; the user's requested composition and action determine the generated scene.\n\nGenerated video:\n{prompt}";
    }
}
