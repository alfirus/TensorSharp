using TensorSharp.Models.Ltx25;

namespace InferenceWeb.Tests;

public sealed class Ltx25ReferencePromptTests
{
    [Theory]
    [InlineData(1, "one input image")]
    [InlineData(3, "3 input image panels in a 2-column, 2-row grid")]
    public void PlainPromptsUseTheTrainedFormatWithoutInventingOrTranslatingContents(int images, string inventory)
    {
        const string prompt = "请让图中的女人来个飞吻";
        string formatted = Ltx25ReferencePrompt.Format(prompt, images);
        Assert.StartsWith("Reference sheet:", formatted);
        Assert.Contains(inventory, formatted);
        Assert.EndsWith("Generated video:\n" + prompt, formatted);
        Assert.Equal(formatted, Ltx25ReferencePrompt.Format(formatted, images));
    }

    [Theory]
    [InlineData("Reference sheet: A blue toy.\nGenerated video: The toy spins.")]
    [InlineData("### Reference Sheet Description\nA blue toy.\n### Target Description\nThe toy spins.")]
    public void ExplicitReferenceAndTargetDescriptionsArePreserved(string prompt)
        => Assert.Equal(prompt, Ltx25ReferencePrompt.Format(prompt, 1));

    [Fact]
    public void ReferencePanelInventoryDoesNotOverrideARequestedSplitScreen()
    {
        const string prompt = "Show the two characters on a split screen, then bring them together.";
        string formatted = Ltx25ReferencePrompt.Format(prompt, 2);
        Assert.EndsWith(prompt, formatted);
        Assert.Contains("the user's requested composition", formatted);
        Assert.DoesNotContain("one continuous", formatted);
    }
}
