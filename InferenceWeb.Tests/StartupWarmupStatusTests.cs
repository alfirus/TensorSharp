using System;
using TensorSharp.Server.Hosting;

namespace InferenceWeb.Tests;

public class StartupWarmupStatusTests
{
    [Fact]
    public void AResultBelongsOnlyToItsOriginalModel()
    {
        var model = new object();
        var store = new StartupWarmupStatusStore();
        var status = BothModes();
        store.Set(model, status);
        Assert.Same(status, store.Get(model));
        Assert.Null(store.Get(null));
        Assert.Null(store.Get(new object()));
    }

    [Fact]
    public void CompletionRequiresBothModesAndDoesNotClaimPrefixRetention()
    {
        var stages = new[] { new StartupWarmupStatus.Stage(false, true, 100),
            new StartupWarmupStatus.Stage(true, true, 200) };
        var status = new StartupWarmupStatus(stages, false);
        stages[0] = new(false, false, 400);
        Assert.True(status.Completed);
        Assert.False(status.PrefixRetentionRequested);
        Assert.Equal(300, status.ElapsedMilliseconds);
        Assert.False(new StartupWarmupStatus(stages, true).Completed);
        Assert.False(new StartupWarmupStatus(new[] { stages[1] }, true).Completed);
        Assert.False(new StartupWarmupStatus(new[] { stages[1], stages[1] }, true).Completed);
    }

    private static StartupWarmupStatus BothModes() => new(new[] {
        new StartupWarmupStatus.Stage(false, true, 1),
        new StartupWarmupStatus.Stage(true, true, 1) }, true);
}
