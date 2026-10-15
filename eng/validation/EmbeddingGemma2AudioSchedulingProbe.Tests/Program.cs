int passed = 0;
void Check(bool condition, string name)
{
    if (!condition) throw new Exception("Persistent worker control failed: " + name);
    ++passed;
}
void Completes(Task task) => task.WaitAsync(TimeSpan.FromSeconds(10)).GetAwaiter().GetResult();
using (var first = new PersistentWorker("control-first"))
using (var second = new PersistentWorker("control-second"))
{
    int id = first.Invoke(() => Environment.CurrentManagedThreadId);
    Check(id == first.ManagedThreadId && id != Environment.CurrentManagedThreadId, "owned thread identity");
    Check(first.Invoke(() => Environment.CurrentManagedThreadId) == id, "persistent identity");
    Check(second.Invoke(() => Environment.CurrentManagedThreadId) != id, "distinct workers");
    Check(first.ActiveCallbacks == 0 && second.ActiveCallbacks == 0, "callbacks idle before completion publication");
    bool propagated = false;
    try { first.Invoke<int>(() => throw new InvalidDataException("sentinel")); }
    catch (InvalidDataException error) when (error.Message == "sentinel") { propagated = true; }
    Check(propagated && first.ActiveCallbacks == 0, "callback exception propagated");
    Check(first.Invoke(() => 42) == 42, "healthy after exception");
    bool selfRejected = false;
    try { first.Invoke(() => { first.Dispose(); return true; }); }
    catch (InvalidOperationException) { selfRejected = true; }
    Check(selfRejected && first.Invoke(() => 43) == 43, "self-join rejected without poisoning worker");
    Check(first.Invoke(() => first.ActiveCallbacks) == 1, "one active callback");
}
var draining = new PersistentWorker("control-draining");
using var entered = new ManualResetEventSlim();
using var release = new ManualResetEventSlim();
var task1 = draining.Submit(() =>
{
    entered.Set();
    if (!release.Wait(TimeSpan.FromSeconds(10))) throw new TimeoutException("Control release timed out.");
    return 1;
});
if (!entered.Wait(TimeSpan.FromSeconds(10))) throw new TimeoutException("Control entry timed out.");
var task2 = draining.Submit(() => 2);
var disposing = Task.Run(draining.Dispose);
release.Set();
Completes(disposing); Completes(task1); Completes(task2);
Check(task1.Result == 1 && task2.Result == 2, "dispose drains previously submitted work");
Check(draining.Joined && draining.ActiveCallbacks == 0, "joined with no active callback");
draining.Dispose();
Check(draining.Joined, "idempotent dispose");
bool disposedRejected = false;
try { _ = draining.Submit(() => 3); } catch (ObjectDisposedException) { disposedRejected = true; }
Check(disposedRejected, "submission after disposal rejected");
Check(SchedulingModes.Parse(null, null) == SchedulingMode.SameSplit && SchedulingModes.Parse("same-split", null) == SchedulingMode.SameSplit, "omitted and explicit legacy modes unchanged");
Check(SchedulingModes.Parse("threadpool-resident", null) == SchedulingMode.ThreadPoolResident && SchedulingModes.Parse("threadpool-resident", "image.png") == SchedulingMode.ThreadPoolResident, "new mode and optional image accepted");
Check(SchedulingModes.Parse("public-resident", null) == SchedulingMode.PublicResident && SchedulingModes.Parse("public-resident", "image.png") == SchedulingMode.PublicResident, "public comparison and separate common prewarm accepted");
Check(SchedulingModes.Phases(SchedulingMode.SameSplit).SequenceEqual(new[] { ("same-a", false), ("split-a", true), ("split-b", true), ("same-b", false) }), "legacy same-split ABBA preserved");
Check(SchedulingModes.Phases(SchedulingMode.ThreadPoolResident).SequenceEqual(new[] { ("whole-taskrun-a", true), ("resident-a", false), ("resident-b", false), ("whole-taskrun-b", true) }), "legacy whole-callback ABBA preserved");
Check(SchedulingModes.Phases(SchedulingMode.PublicResident).SequenceEqual(new[] { ("public-a", true), ("resident-a", false), ("resident-b", false), ("public-b", true) }), "public comparison ABBA labels and routes");
foreach (var invalid in new[] { ("unknown", (string?)null), ("same-split", "image.png"), ((string?)null, "image.png"), ("threadpool-resident", " "), ("public-resident", " ") })
{
    bool rejected = false;
    try { _ = SchedulingModes.Parse(invalid.Item1, invalid.Item2); } catch (ArgumentException) { rejected = true; }
    Check(rejected, "invalid mode/prewarm combination rejected");
}
using (var owner = new PersistentWorker("control-whole-callback-owner"))
{
    int calls = 0;
    Func<(CallerIdentity Identity, int Number)> callback = () => (CallerIdentity.Capture(), Interlocked.Increment(ref calls));
    var resident = SchedulingDispatch.Run(false, owner, callback);
    var pooled = SchedulingDispatch.Run(true, owner, callback);
    var residentAgain = SchedulingDispatch.Run(false, owner, callback);
    Check(resident.Identity.ManagedThreadId == owner.ManagedThreadId && !resident.Identity.IsThreadPoolThread && residentAgain.Identity == resident.Identity, "resident whole callback retains caller identity");
    Check(pooled.Identity.IsThreadPoolThread && pooled.Identity.ManagedThreadId != owner.ManagedThreadId && calls == 3 && resident.Number == 1 && pooled.Number == 2 && residentAgain.Number == 3, "same whole callback executes exactly once through either route");
    foreach (bool route in new[] { false, true })
    {
        bool propagated = false;
        try { SchedulingDispatch.Run<int>(route, owner, () => throw new InvalidDataException("whole-sentinel")); }
        catch (InvalidDataException error) when (error.Message == "whole-sentinel") { propagated = true; }
        Check(propagated && owner.ActiveCallbacks == 0, "whole callback failure propagates after completion");
    }
    Check(SchedulingDispatch.Run(true, owner, () => 7) == 7 && SchedulingDispatch.Run(false, owner, () => 8) == 8, "both routes healthy after failure");
}
var unobserved = PublicResidentReport.Observations(true, [], [], [], [], [], null);
using (var serialized = System.Text.Json.JsonDocument.Parse(System.Text.Json.JsonSerializer.Serialize(unobserved)))
{
    foreach (string key in new[] { "prepare_samples_ms", "text_samples_ms", "clone_samples_ms", "caller_observations", "distinct_managed_callers", "distinct_native_callers", "prepared_bit_identity" })
        Check(serialized.RootElement.GetProperty(key).ValueKind == System.Text.Json.JsonValueKind.Null, "unobserved public field serialized null: " + key);
    Check(serialized.RootElement.GetProperty("unavailable_observations").GetArrayLength() == 3, "public limitations explicit");
}
var observed = PublicResidentReport.Observations(false, [1d], [2d], [.1d], [new { identity = new CallerIdentity(4, 5, false) }], [new CallerIdentity(4, 5, false)], true);
Check((int)observed["distinct_native_callers"]! == 1 && observed["prepared_bit_identity"] is true, "resident actual observations retained");
foreach (int fault in Enumerable.Range(0, 7))
{
    bool rejected = false;
    try { _ = PublicResidentReport.Observations(true, fault == 0 ? [1d] : [], fault == 1 ? [1d] : [], fault == 2 ? [1d] : [],
        fault == 3 ? [new object()] : [], fault == 4 ? [new CallerIdentity(1, 2, false)] : [], fault == 5 ? false : fault == 6 ? true : null); }
    catch (InvalidDataException) { rejected = true; }
    Check(rejected, "fabricated public observation rejected");
}
bool missingResidentRejected = false;
try { _ = PublicResidentReport.Observations(false, [], [], [], [], [], null); }
catch (InvalidDataException) { missingResidentRejected = true; }
Check(missingResidentRejected, "missing resident observations rejected");
Console.WriteLine($"AUDIO_SCHEDULING_WORKER_CONTROLS passed={passed} failed=0 model_executed=0");
