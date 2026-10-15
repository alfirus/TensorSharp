// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Diagnostics;
using System.Text.Json;

namespace InferenceWeb.Tests;

public sealed class CudaPtxBuildSettingsTests
{
    [Theory]
    [InlineData(true, null, "compute_120", "compute_86", "compute_61")]
    [InlineData(true, "auto", "compute_120", "compute_86", "compute_61")]
    [InlineData(true, "native", "compute_120", "compute_86", "compute_61")]
    [InlineData(true, "compute_61", "compute_120", "compute_86", "compute_61")]
    [InlineData(true, "sm_61", "compute_120", "compute_86", "compute_61")]
    [InlineData(false, "auto", "compute_120", "compute_61", "compute_120")]
    [InlineData(false, "native", "compute_86", "compute_61", "compute_86")]
    [InlineData(false, "auto", "", "compute_86", "compute_86")]
    [InlineData(false, "compute_120", "compute_86", "compute_61", "compute_120")]
    public async Task ActualMsbuildResolutionKeepsRefreshPortableAndOrdinaryBuildsGpuSpecific(
        bool refresh, string? requested, string detected, string fallback, string expected)
    {
        var properties = new Dictionary<string, string>
        {
            ["TensorSharpUpdateCommittedPtx"] = refresh.ToString(),
            ["_DetectedCudaArch"] = detected,
            ["_DetectCudaArchExitCode"] = "0",
            ["CudaArchFallback"] = fallback,
        };
        if (requested != null) properties["CudaArch"] = requested;
        var result = await RunTarget("ResolveCudaPtxArchitecture", properties);
        Assert.True(result.ExitCode == 0, result.Output + result.Error);
        using var json = JsonDocument.Parse(result.Output);
        var actual = json.RootElement.GetProperty("Properties");
        Assert.Equal(expected, actual.GetProperty("_ResolvedCudaArch").GetString());
        Assert.StartsWith(expected + "_", actual.GetProperty("_CudaBuildSettingsKey").GetString());
    }

    [Theory]
    [InlineData("compute_86")]
    [InlineData("compute_120")]
    [InlineData("sm_120")]
    public async Task RefreshRejectsExplicitNewerTargetsBeforeInvokingNvcc(string requested)
    {
        var result = await RunTarget("ResolveCudaPtxArchitecture", new()
        {
            ["TensorSharpUpdateCommittedPtx"] = "true", ["CudaArch"] = requested,
        });
        Assert.NotEqual(0, result.ExitCode);
        Assert.Contains("must target compute_61", result.Output + result.Error);
        Assert.Contains("omit TensorSharpUpdateCommittedPtx", result.Output + result.Error);
    }

    [Fact]
    public async Task CopyTargetRefusesAnInconsistentResolvedArchitectureBeforeWritingFiles()
    {
        var result = await RunTarget("UpdateCommittedCudaPtx", new()
        {
            ["TensorSharpUpdateCommittedPtx"] = "true", ["_NvccExitCode"] = "0",
            ["_ResolvedCudaArch"] = "compute_120",
        });
        Assert.NotEqual(0, result.ExitCode);
        Assert.Contains("Refusing to overwrite the portable committed CUDA PTX baseline", result.Output + result.Error);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task IntermediatePtxIsCopiedOnlyWhenNvccWasAvailable(bool nvccAvailable)
    {
        string directory = NewDirectory();
        string input = Path.Combine(directory, "intermediate"), output = Path.Combine(directory, "output");
        string kernels = Path.Combine(directory, "kernels");
        Directory.CreateDirectory(input);
        Directory.CreateDirectory(kernels);
        File.WriteAllText(Path.Combine(kernels, "probe.cu"), "// Isolated expected kernel for the copy target.\n");
        File.WriteAllText(Path.Combine(input, "probe.ptx"), ".target sm_120\n");
        try
        {
            var result = await RunTarget("CopyCudaPtxToOutput", new()
            {
                ["_NvccExitCode"] = nvccAvailable ? "0" : "1", ["PtxOutputDir"] = input, ["OutDir"] = output,
                ["CudaKernelDir"] = kernels,
            });
            Assert.True(result.ExitCode == 0, result.Output + result.Error);
            Assert.Equal(nvccAvailable, File.Exists(Path.Combine(output, "cuda_kernels", "probe.ptx")));
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public async Task ActualRefreshRecompilesStaleGpuSpecificPtxEvenWithAPortableStamp()
    {
        string directory = NewDirectory();
        string tools = Path.Combine(directory, "tools"), kernels = Path.Combine(directory, "kernels");
        string intermediate = Path.Combine(directory, "intermediate"), stamps = Path.Combine(directory, "stamps");
        string baseline = Path.Combine(directory, "baseline");
        foreach (string path in new[] { tools, kernels, intermediate, stamps, baseline }) Directory.CreateDirectory(path);
        string stub = Path.Combine(tools, OperatingSystem.IsWindows() ? "nvcc.cmd" : "nvcc");
        if (OperatingSystem.IsWindows())
            File.WriteAllText(stub, """
                @echo off
                if "%~1"=="--version" exit /b 0
                setlocal enabledelayedexpansion
                :next
                if "%~1"=="" goto done
                set "argument=%~1"
                if "%~1"=="-arch" goto architecture
                if "!argument:~0,6!"=="-arch=" set "arch=!argument:~6!"
                if "%~1"=="-o" goto output
                shift
                goto next
                :architecture
                shift
                set "arch=%~1"
                shift
                goto next
                :output
                shift
                set "out=%~1"
                shift
                goto next
                :done
                if not defined out exit /b 2
                if not defined arch exit /b 3
                > "%out%" echo .target !arch:compute_=sm_!
                exit /b 0
                """.Replace("\n", "\r\n", StringComparison.Ordinal) + "\r\n");
        else
        {
            File.WriteAllText(stub, """
                #!/bin/sh
                if [ "$1" = "--version" ]; then exit 0; fi
                while [ "$#" -gt 0 ]; do
                    case "$1" in
                        -arch=*) arch=${1#-arch=} ;;
                        -o) shift; output=$1 ;;
                    esac
                    shift
                done
                [ -n "$output" ] || exit 2
                printf '.target sm_%s\n' "${arch#compute_}" > "$output"
                """ + "\n");
            File.SetUnixFileMode(stub, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
        }
        File.WriteAllText(Path.Combine(kernels, "probe.cu"), "// The fake toolchain checks build policy, not CUDA arithmetic.\n");
        string stale = Path.Combine(intermediate, "probe.ptx");
        string stamp = Path.Combine(stamps, "tensorsharp_cuda_compute_61_O3_extradevicevectorization.stamp");
        File.WriteAllText(stale, ".target sm_120\n");
        File.WriteAllText(stamp, "arch=compute_61\n");
        File.SetLastWriteTimeUtc(stale, DateTime.UtcNow.AddMinutes(5));
        File.SetLastWriteTimeUtc(stamp, DateTime.UtcNow.AddMinutes(5));
        try
        {
            var result = await RunTarget("CompileCudaPtx", new()
            {
                ["TensorSharpUpdateCommittedPtx"] = "true", ["CudaArch"] = "auto", ["CudaArchFallback"] = "compute_120",
                ["CudaKernelDir"] = kernels, ["PtxOutputDir"] = intermediate, ["CudaPtxStampDir"] = stamps,
                ["CudaCommittedPtxDir"] = baseline, ["OutDir"] = Path.Combine(directory, "output"),
                ["_CompilerBinDir"] = "unused-stub-compiler",
            }, new() { ["PATH"] = tools + Path.PathSeparator + Environment.GetEnvironmentVariable("PATH") });
            Assert.True(result.ExitCode == 0, result.Output + result.Error);
            Assert.Equal(".target sm_61", File.ReadAllText(Path.Combine(baseline, "probe.ptx")).Trim());
            Assert.Equal(".target sm_61", File.ReadAllText(stale).Trim());
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public async Task NoNvccFallbackReplacesANewerGpuSpecificDestinationThroughSdkCopyTargets()
    {
        string directory = NewDirectory();
        string baseline = Path.Combine(directory, "baseline"), output = Path.Combine(directory, "output");
        string tools = Path.Combine(directory, "tools");
        foreach (string path in new[] { baseline, tools, Path.Combine(output, "cuda_kernels") }) Directory.CreateDirectory(path);
        string stub = Path.Combine(tools, OperatingSystem.IsWindows() ? "nvcc.cmd" : "nvcc");
        File.WriteAllText(stub, OperatingSystem.IsWindows() ? "@exit /b 1\r\n" : "#!/bin/sh\nexit 1\n");
        if (!OperatingSystem.IsWindows())
            File.SetUnixFileMode(stub, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
        File.WriteAllText(Path.Combine(baseline, "probe.ptx"), ".target sm_61\n");
        string destination = Path.Combine(output, "cuda_kernels", "probe.ptx");
        File.WriteAllText(destination, ".target sm_120\n");
        File.SetLastWriteTimeUtc(destination, DateTime.UtcNow.AddMinutes(5));
        try
        {
            var result = await RunTarget("_CopySourceItemsToOutputDirectory", new()
            {
                ["CudaCommittedPtxDir"] = baseline, ["OutDir"] = output + Path.DirectorySeparatorChar,
                ["_GetChildProjectCopyToOutputDirectoryItems"] = "false", ["BuildProjectReferences"] = "false",
            }, new() { ["PATH"] = tools + Path.PathSeparator + Environment.GetEnvironmentVariable("PATH") });
            Assert.True(result.ExitCode == 0, result.Output + result.Error);
            Assert.Equal(".target sm_61", File.ReadAllText(destination).Trim());
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    private static string NewDirectory()
    {
        string directory = Path.Combine(RepositoryRoot(), "artifacts", "tests", "CudaPtxBuildSettings", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static string RepositoryRoot()
    {
        for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory != null; directory = directory.Parent)
            if (File.Exists(Path.Combine(directory.FullName, "TensorSharp.slnx"))) return directory.FullName;
        throw new DirectoryNotFoundException("TensorSharp source tree is required for the CUDA build-settings probes.");
    }

    // Execute the real project targets without restoring or compiling managed
    // projects. Architecture probes need no toolchain; the compilation probe
    // uses a fake nvcc and isolated fixture paths, never repository PTX.
    private static async Task<(int ExitCode, string Output, string Error)> RunTarget(
        string target, Dictionary<string, string> properties, Dictionary<string, string>? environment = null)
    {
        string root = RepositoryRoot();
        string? host = Environment.GetEnvironmentVariable("DOTNET_HOST_PATH");
        var start = new ProcessStartInfo(string.IsNullOrEmpty(host) ? "dotnet" : host)
        {
            WorkingDirectory = root,
            UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardOutput = true, RedirectStandardError = true,
        };
        foreach (string name in new[] { "CudaArch", "CudaArchFallback", "TensorSharpUpdateCommittedPtx", "TENSORSHARP_CUDA_ARCH" })
            start.Environment.Remove(name);
        if (environment != null)
            foreach ((string name, string value) in environment) start.Environment[name] = value;
        foreach (string argument in new[] { "msbuild", Path.Combine(root, "TensorSharp.Backends.Cuda", "TensorSharp.Backends.Cuda.csproj"),
            "-nologo", "-verbosity:quiet", "-target:" + target, "-getProperty:_ResolvedCudaArch,_CudaBuildSettingsKey" })
            start.ArgumentList.Add(argument);
        foreach ((string name, string value) in properties) start.ArgumentList.Add("-property:" + name + "=" + value);
        using var process = Process.Start(start)!;
        var output = process.StandardOutput.ReadToEndAsync();
        var error = process.StandardError.ReadToEndAsync();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        try { await process.WaitForExitAsync(timeout.Token); }
        catch (OperationCanceledException)
        {
            process.Kill(entireProcessTree: true);
            Assert.Fail("The CUDA build-settings probe did not complete within 30 seconds.");
        }
        return (process.ExitCode, await output, await error);
    }
}
