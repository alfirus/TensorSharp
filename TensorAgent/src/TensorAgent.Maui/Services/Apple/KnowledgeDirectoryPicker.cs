// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using Foundation;
using TensorAgent.Core.Hosting;
using UIKit;
using UniformTypeIdentifiers;

namespace TensorAgent.Maui.Services;

/// <summary>Folder selection for iOS and Mac Catalyst, including security-scoped file-provider access.</summary>
internal static class KnowledgeDirectoryPicker
{
    public static async Task<KnowledgeDirectorySelection?> PickAsync()
    {
        UIViewController host = TopViewController()
            ?? throw new InvalidOperationException("The folder picker could not find an active window.");
        var completion = new TaskCompletionSource<NSUrl?>(TaskCreationOptions.RunContinuationsAsynchronously);
        using var picker = new UIDocumentPickerViewController([UTTypes.Folder], false)
        {
            AllowsMultipleSelection = false,
        };
        using var callbacks = new PickerDelegate(completion);
        picker.Delegate = callbacks;
        await host.PresentViewControllerAsync(picker, true);
        NSUrl? selected = await completion.Task;
        if (selected is null) return null;
        return new AppleSelection(selected);
    }

    private static UIViewController? TopViewController()
    {
        UIViewController? controller = UIApplication.SharedApplication.ConnectedScenes
            .OfType<UIWindowScene>().SelectMany(scene => scene.Windows)
            .FirstOrDefault(window => window.IsKeyWindow)?.RootViewController;
        while (controller?.PresentedViewController is { } presented) controller = presented;
        return controller;
    }

    private sealed class PickerDelegate(TaskCompletionSource<NSUrl?> completion) : UIDocumentPickerDelegate
    {
        public override void DidPickDocument(UIDocumentPickerViewController controller, NSUrl[] urls) =>
            completion.TrySetResult(urls.FirstOrDefault());
        public override void WasCancelled(UIDocumentPickerViewController controller) => completion.TrySetResult(null);
    }

    private sealed class AppleSelection : KnowledgeDirectorySelection
    {
        private readonly NSUrl _url;
        private readonly bool _scoped;
        public AppleSelection(NSUrl url) : base(url.Path ?? throw new IOException("The selected folder has no local path."))
        {
            if (string.IsNullOrWhiteSpace(FullPath)) throw new IOException("The selected folder has no local path.");
            _url = url;
            _scoped = url.StartAccessingSecurityScopedResource();
        }

        public override Task<IReadOnlyList<KnowledgeDirectoryFiles.Entry>> ReadFilesAsync() => Task.Run(() =>
        {
            IReadOnlyList<KnowledgeDirectoryFiles.Entry>? files = null;
            Coordinate(_url, url => files = KnowledgeDirectoryFiles.Read(url.Path!));
            return files ?? throw new IOException("The selected folder could not be read.");
        });

        public override Task<Stream> OpenReadAsync(string fullPath, long maxBytes) => Task.Run<Stream>(() =>
        {
            // A coordinated read materializes cloud/provider files before releasing
            // the read assertion. Import the private copy after that assertion ends.
            string temp = Path.Combine(Path.GetTempPath(), "knowledge-import-" + Guid.NewGuid().ToString("N"));
            try
            {
                using NSUrl file = NSUrl.FromFilename(fullPath);
                Coordinate(file, url =>
                {
                    using FileStream source = File.OpenRead(url.Path!);
                    if (source.Length > maxBytes)
                        throw new TensorSharp.Knowledge.KnowledgeLimitException("File exceeds the configured upload limit.");
                    using FileStream output = File.Create(temp);
                    byte[] buffer = new byte[81920];
                    long copied = 0;
                    int read;
                    while ((read = source.Read(buffer)) != 0)
                    {
                        copied += read;
                        if (copied > maxBytes)
                            throw new TensorSharp.Knowledge.KnowledgeLimitException("File exceeds the configured upload limit.");
                        output.Write(buffer, 0, read);
                    }
                });
                return new FileStream(temp, FileMode.Open, FileAccess.Read, FileShare.Read,
                    81920, FileOptions.Asynchronous | FileOptions.DeleteOnClose);
            }
            catch { try { File.Delete(temp); } catch (IOException) { } throw; }
        });

        private static void Coordinate(NSUrl url, Action<NSUrl> read)
        {
            using var coordinator = new NSFileCoordinator();
            Exception? failure = null;
            coordinator.CoordinateRead(url, NSFileCoordinatorReadingOptions.WithoutChanges, out NSError? error,
                coordinated => { try { read(coordinated); } catch (Exception ex) { failure = ex; } });
            if (failure is not null) throw failure;
            if (error is not null) throw new IOException(error.LocalizedDescription);
        }

        public override void Dispose()
        {
            if (_scoped) _url.StopAccessingSecurityScopedResource();
            _url.Dispose();
        }
    }
}
