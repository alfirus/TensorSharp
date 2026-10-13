// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using Windows.Storage;
using Windows.Storage.Pickers;

namespace TensorAgent.Maui.Services;

/// <summary>Owned-window folder selection for the unpackaged Windows desktop app.</summary>
internal static class KnowledgeDirectoryPicker
{
    public static async Task<KnowledgeDirectorySelection?> PickAsync()
    {
        var picker = new FolderPicker { SuggestedStartLocation = PickerLocationId.DocumentsLibrary };
        picker.FileTypeFilter.Add("*");
        var window = (Microsoft.UI.Xaml.Window)Application.Current!.Windows[0].Handler!.PlatformView!;
        WinRT.Interop.InitializeWithWindow.Initialize(picker, WinRT.Interop.WindowNative.GetWindowHandle(window));
        StorageFolder? selected = await picker.PickSingleFolderAsync();
        if (selected is not null && string.IsNullOrWhiteSpace(selected.Path))
            throw new IOException("The selected folder has no local path.");
        return selected is null ? null : new WindowsSelection(selected.Path);
    }

    private sealed class WindowsSelection(string fullPath) : KnowledgeDirectorySelection(fullPath)
    {
        public override void Dispose() { }
    }
}
