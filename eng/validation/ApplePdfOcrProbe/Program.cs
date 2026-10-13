using Foundation;
using UIKit;

namespace ApplePdfOcrProbe;

public static class Program
{
    public static void Main(string[] args) => UIApplication.Main(args, null, typeof(AppDelegate));
}

[Register("PdfProbeAppDelegate")]
public sealed class AppDelegate : UIApplicationDelegate
{
    public override bool FinishedLaunching(UIApplication application, NSDictionary? launchOptions) => true;
}

[Register("PdfProbeSceneDelegate")]
public sealed class SceneDelegate : UIWindowSceneDelegate
{
    public override UIWindow? Window { get; set; }
    private static int _started;

    public override void WillConnect(UIScene scene, UISceneSession session, UISceneConnectionOptions connectionOptions)
    {
        var controller = new UIViewController();
        controller.View!.BackgroundColor = UIColor.White;
        var label = new UILabel(new CoreGraphics.CGRect(20, 100, 340, 150))
        {
            Text = "TensorSharp PDF OCR validation is running. Results are saved in this probe's Documents folder.",
            TextColor = UIColor.Black, Lines = 0
        };
        controller.View.AddSubview(label);
        Window = new UIWindow((UIWindowScene)scene) { RootViewController = controller };
        Window.MakeKeyAndVisible();
        if (Interlocked.Exchange(ref _started, 1) != 0) return;
        _ = Task.Run(async () =>
        {
            try
            {
                bool passed = await Probe.RunAsync();
                BeginInvokeOnMainThread(() => label.Text = passed ? "PDF OCR validation passed." : "PDF OCR validation failed. See probe-result.json.");
            }
            catch (Exception error)
            {
                Console.WriteLine("PDF_PROBE_FATAL " + error);
                File.WriteAllText(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "probe-fatal.txt"), error.ToString());
                BeginInvokeOnMainThread(() => label.Text = "Probe failed before completing its report. See probe-fatal.txt.");
            }
        });
    }
}
