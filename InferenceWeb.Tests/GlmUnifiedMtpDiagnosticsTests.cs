// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Globalization;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class GlmUnifiedMtpDiagnosticsTests
{
    private static GlmUnifiedMtpMemorySample Sample => new("request-a", 2048, 512, 1,
        new(100, 1000), new(20, 300), new(200, 800), new(40, 400), 150, 6000);

    [Theory]
    [InlineData(null, false)]
    [InlineData("0", false)]
    [InlineData("true", false)]
    [InlineData("1", true)]
    public void DiagnosticsRequireExactOptIn(string? value, bool enabled)
        => Assert.Equal(enabled, GlmUnifiedMtpDiagnostics.IsEnabled(value));

    [Fact]
    public void AlternatingUnchangedPhasesDoNotLogEachDecodeStep()
    {
        var diagnostics = new GlmUnifiedMtpDiagnostics();
        Assert.NotNull(diagnostics.Record("target-admitted", Sample, 0));
        Assert.NotNull(diagnostics.Record("draft-admitted", Sample, 512));
        Assert.Null(diagnostics.Record("target-admitted", Sample, 513));
        Assert.Null(diagnostics.Record("draft-admitted", Sample, 515));
        // Live allocations, replacement credit, shape and request changes are
        // material even when the phase alternation itself is unchanged.
        Assert.NotNull(diagnostics.Record("draft-admitted", Sample with { DraftLive = new(40, 450) }, 516));
        Assert.NotNull(diagnostics.Record("target-admitted", Sample with { TargetOwned = new(100, 1100) }, 516));
        Assert.NotNull(diagnostics.Record("target-admitted", Sample with { Rows = 3 }, 516));
        Assert.NotNull(diagnostics.Record("target-admitted", Sample with { RequestId = "request-b" }, 0));
    }

    [Fact]
    public void ExplicitPrefillBoundaryAlwaysReportsAndDistinguishesCreditFromLiveChildren()
    {
        var diagnostics = new GlmUnifiedMtpDiagnostics();
        var original = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo("fr-FR");
            string line = diagnostics.Record("prefill-complete", Sample, 1872, force: true)!;
            Assert.Contains("request=request-a context=2048 rows=512 output_rows=1 target_position=1872", line);
            Assert.Contains("kind5_owned_host=100 kind5_owned_device=1000 kind5_live_host=20 kind5_live_device=300", line);
            Assert.Contains("kind7_owned_host=200 kind7_owned_device=800 kind7_live_host=40 kind7_live_device=400", line);
            Assert.Contains("snapshot_forecast_device=150 snapshot_live_device=0 snapshot_missing_device=150 expert_budget_device=6000 unit=bytes", line);
            Assert.Equal(line, diagnostics.Record("prefill-complete", Sample, 1872, force: true));
            Assert.Contains("snapshot_forecast_device=150 snapshot_live_device=140 snapshot_missing_device=10",
                diagnostics.Record("prefill-complete", Sample with { SnapshotLiveBytes = 140 }, 1872)!);
        }
        finally { CultureInfo.CurrentCulture = original; }
    }
}
