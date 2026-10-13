using System.Text.Json;

namespace TensorAgent.Tests;

public sealed partial class WebUiPageTests
{
    [WebJavaScriptFact]
    public void CompatibleLorasAreOneTapFromTheComposerAndSearchLocally()
    {
        JsonElement result = Run(ImageModel + TwoStyles, """
            var button = __page.byId['composer-loras'];
            var visible = button.style.display !== 'none';
            button.dispatch('click');
            return settle(20).then(function () {
              var reads = __page.requests('/api/agent/loras').length;
              __page.byId['lora-search'].value = 'film';
              __page.byId['lora-search'].dispatch('input');
              var rows = __page.byId['lora-list'].querySelectorAll('.lorarow');
              var result = { visible: visible, rows: rows.length,
                name: rows[0].getAttribute('data-lora'),
                context: __page.byId['lora-context'].textContent,
                label: rows[0].querySelector('INPUT').getAttribute('aria-label'),
                readCount: __page.requests('/api/agent/loras').length - reads };
              __page.byId['lora-close'].dispatch('click');
              result.closed = !__page.byId['lora-sheet'].classList.contains('on');
              return result;
            });
            """);
        Assert.True(result.GetProperty("visible").GetBoolean());
        Assert.Equal(1, result.GetProperty("rows").GetInt32());
        Assert.Equal("film", result.GetProperty("name").GetString());
        Assert.Equal("Compatible with Qwen-Image 2.1", result.GetProperty("context").GetString());
        Assert.Equal("Use Film Stills", result.GetProperty("label").GetString());
        Assert.Equal(0, result.GetProperty("readCount").GetInt32());
        Assert.True(result.GetProperty("closed").GetBoolean());
    }

    [WebJavaScriptTheory]
    [InlineData("{}", "{ supportsLoras: true, loras: [{ id: 'film' }] }")]
    [InlineData("{ loaded: 'gemma.gguf', architecture: 'gemma3' }", "{ supportsLoras: true, loras: [{ id: 'film' }] }")]
    [InlineData("{ loaded: 'qwen.gguf', architecture: 'qwen_image' }", "{ supportsLoras: false, loras: [{ id: 'film' }] }")]
    [InlineData("{ loaded: 'qwen.gguf', architecture: 'qwen_image' }", "{ supportsLoras: true, loras: [] }")]
    [InlineData("{ loaded: 'qwen.gguf', architecture: 'qwen_image' }", "{ supportsLoras: true, loadedModelFile: 'different.gguf', loras: [{ id: 'film' }] }")]
    public void UnsupportedModelsHaveNoLoraEntry(string model, string loras)
    {
        JsonElement result = Run($$"""
            R['/api/models'] = {{model}};
            R['/api/agent/loras'] = {{loras}};
            """, """
            __page.byId['model'].dispatch('click');
            return settle(20).then(function () {
              __page.byId['composer-loras'].dispatch('click');
              return {
                composer: __page.byId['composer-loras'].style.display,
                details: __page.byId['open-loras'].style.display,
                sheetOpen: __page.byId['lora-sheet'].classList.contains('on')
              };
            });
            """);
        Assert.Equal("none", result.GetProperty("composer").GetString());
        Assert.Equal("none", result.GetProperty("details").GetString());
        Assert.False(result.GetProperty("sheetOpen").GetBoolean());
    }

    [WebJavaScriptFact]
    public void SwitchingModelsHidesLorasAndIgnoresThePreviousModelsLateResponse()
    {
        JsonElement result = Run(ImageModel + TwoStyles + Held, """
            hold('/api/agent/loras');
            __page.byId['composer-loras'].dispatch('click');
            return settle(20).then(function () {
              R['/api/models'] = { loaded: 'gemma.gguf', architecture: 'gemma3' };
              document.dispatch('visibilitychange');
              return wait(400);
            }).then(function () {
              heldFor('/api/agent/loras').forEach(function (h) { h.answer(loraRows([])); });
              return settle(20);
            }).then(function () {
              return { composer: __page.byId['composer-loras'].style.display,
                details: __page.byId['open-loras'].style.display,
                open: __page.byId['lora-sheet'].classList.contains('on') };
            });
            """);
        Assert.Equal("none", result.GetProperty("composer").GetString());
        Assert.Equal("none", result.GetProperty("details").GetString());
        Assert.False(result.GetProperty("open").GetBoolean());
    }

    [WebJavaScriptFact]
    public void SwitchingBetweenImageModelsWhileSavingRefreshesTheNewModelsLoras()
    {
        JsonElement result = Run(ImageModel + TwoStyles + Held, """
            __page.byId['composer-loras'].dispatch('click');
            return settle(20).then(function () {
              hold('/api/agent/loras/choice');
              var box = loraRow('film').querySelector('INPUT');
              box.checked = true;
              box.dispatch('change');
              return settle(20);
            }).then(function () {
              R['/api/models'] = { loaded: 'turbo.gguf', architecture: 'qwen_image' };
              R['/api/agent/loras'] = Object.assign(loraRows([]), { loadedModelFile: 'turbo.gguf' });
              document.dispatch('visibilitychange');
              return wait(400);
            }).then(function () {
              heldFor('/api/agent/loras/choice')[0].answer(loraRows([{ id: 'film', strength: 0.7 }]));
              return settle(20);
            }).then(function () {
              return { composer: __page.byId['composer-loras'].style.display,
                hint: __page.byId['composer-loras-hint'].textContent };
            });
            """);
        Assert.NotEqual("none", result.GetProperty("composer").GetString());
        Assert.Equal("Choose a style or edit", result.GetProperty("hint").GetString());
    }

    [WebJavaScriptFact]
    public void ClosingLorasBeforeADownloadResponseDoesNotRestartPolling()
    {
        JsonElement result = Run(ImageModel + TwoStyles + Held, """
            hold('/api/agent/loras');
            __page.byId['composer-loras'].dispatch('click');
            return settle(20).then(function () {
              __page.byId['lora-close'].dispatch('click');
              var d = loraRows([]);
              d.loras[0].download = { running: true, progress: { fraction: 0.25 } };
              heldFor('/api/agent/loras').forEach(function (h) { h.answer(d); });
              return settle(20);
            }).then(function () {
              var reads = __page.requests('/api/agent/loras').length;
              return wait(1200).then(function () {
                return { addedReads: __page.requests('/api/agent/loras').length - reads,
                  open: __page.byId['lora-sheet'].classList.contains('on') };
              });
            });
            """);
        Assert.Equal(0, result.GetProperty("addedReads").GetInt32());
        Assert.False(result.GetProperty("open").GetBoolean());
    }
}
