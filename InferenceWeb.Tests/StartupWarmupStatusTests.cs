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

    [Fact]
    public void JointStatusRequiresFreshNativeWidthProofAndLeavesBothModeCompletionIndependent()
    {
        var serial = new[] { new StartupWarmupStatus.Stage(false, true, 1), new StartupWarmupStatus.Stage(true, true, 1) };
        var joint = new[] { new StartupWarmupStatus.JointStage(2, true, true, true, 7, 100, 1, "accepted"),
            new StartupWarmupStatus.JointStage(3, true, true, true, 8, 150, 2, "accepted") };
        var status = new StartupWarmupStatus(serial, true, joint, true);
        Assert.True(status.Completed); Assert.True(status.JointCompleted); Assert.True(status.JointWarmupRequested);
        Assert.False(status.JointCleanupPending); Assert.Equal(3, status.JointElapsedMilliseconds);
        joint[1] = joint[1] with { CleanupCompleted = false, CleanupPending = true, Detail = "cleanup pending" };
        Assert.True(status.JointCompleted); // Defensive snapshot.
        var failed = new StartupWarmupStatus(serial, true, joint, true);
        Assert.True(failed.Completed); Assert.False(failed.JointCompleted); Assert.True(failed.JointCleanupPending);
        Assert.False(new StartupWarmupStatus(serial, true, new[] { joint[0], joint[0] }, true).JointCompleted);
        Assert.False(new StartupWarmupStatus(serial, true,
            new[] { joint[0], joint[1] with { CleanupCompleted = true, NativeAcceptedExecutionId = 0 } }, true).JointCompleted);
        Assert.False(new StartupWarmupStatus(serial, true,
            new[] { joint[0], joint[1] with { CleanupCompleted = true, NativeAcceptedExecutionId = 9 } }, true).JointCompleted);
        var skipped = new StartupWarmupStatus(serial, true, jointWarmupSkipReason: "joint decode disabled");
        Assert.True(skipped.Completed); Assert.False(skipped.JointWarmupRequested);
        Assert.Equal("joint decode disabled", skipped.JointWarmupSkipReason);
        Assert.False(BothModes().JointWarmupRequested); Assert.False(BothModes().JointCompleted);
    }

    private static StartupWarmupStatus BothModes() => new(new[] {
        new StartupWarmupStatus.Stage(false, true, 1),
        new StartupWarmupStatus.Stage(true, true, 1) }, true);
}
