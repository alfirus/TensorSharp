param(
    [string] $CudaRoot = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4",
    [string] $GgmlBuildRoot = "C:/Works/TensorSharp/TensorSharp.GGML.Native/build-windows-cuda13",
    [string] $OutputDirectory = "artifacts/perf/glm-memory-tier-20261015/cuda-vmm-expert-table-probe-v27"
)
$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$upstream = Join-Path $workspace "ExternalProjects/ggml"
$revision = (& git -C $upstream rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $revision -ne "3612c0af93f2cfc8f048295300a61187723b12ae") {
    throw "This standalone source contract requires unchanged ggml 3612c0af93f2cfc8f048295300a61187723b12ae."
}
$upstreamStatus = @(& git -C $upstream status --porcelain)
if ($LASTEXITCODE -ne 0 -or $upstreamStatus.Count -ne 0) { throw "The upstream ggml checkout has staged, unstaged or untracked modifications." }
$probeOutput = [System.IO.Path]::GetFullPath((Join-Path $workspace $OutputDirectory))
if (Test-Path -LiteralPath (Join-Path $probeOutput "cuda-vmm-expert-table-probe.exe")) {
    throw "Choose a fresh ignored output directory; an existing probe binary is frozen evidence."
}
$buildRoot = (Resolve-Path -LiteralPath $GgmlBuildRoot).Path
$ninja = Join-Path $buildRoot "build.ninja"
$ninjaText = Get-Content -Raw -LiteralPath $ninja
$reference = [regex]::Match($ninjaText, '(?ms)^build GgmlOpsHostTokenEmbeddingTest\.exe:.*?(?=^#|\z)').Value
if (-not $reference -or $reference -notmatch 'ggml-cuda\.lib' -or $reference -notmatch 'cublasLt\.lib' -or $reference -notmatch 'cuda\.lib') {
    throw "The frozen HostTokenEmbeddingTest static-link reference is unavailable; inspect this build's target before linking."
}
# Match the existing unchanged CUDA13 HostTokenEmbeddingTest link contract.
# This wrapper never configures, rebuilds, patches or writes to ggml/buildRoot.
$libraries = @(
    (Join-Path $buildRoot "ggml-cuda.lib"), (Join-Path $buildRoot "ggml-cpu.lib"), (Join-Path $buildRoot "ggml-base.lib"),
    (Join-Path $CudaRoot "lib/x64/cudart.lib"), (Join-Path $CudaRoot "lib/x64/cublas.lib"),
    (Join-Path $CudaRoot "lib/x64/cublasLt.lib"), (Join-Path $CudaRoot "lib/x64/cuda.lib"),
    (Join-Path $CudaRoot "lib/x64/cudadevrt.lib")
)
foreach ($library in $libraries) {
    if (-not (Test-Path -LiteralPath $library -PathType Leaf)) { throw "Required existing static/import library is absent: $library" }
}
. (Join-Path $workspace "eng/vs-locate.ps1")
$probeVs = Get-VisualStudioInstallation
if ($null -eq $probeVs) { throw "MSVC x64 toolset is unavailable." }
Import-VcVarsEnvironment $probeVs.VcVars64
New-Item -ItemType Directory -Path $probeOutput -Force | Out-Null
$source = Join-Path $PSScriptRoot "cuda-vmm-expert-table-probe.cpp"
$executable = Join-Path $probeOutput "cuda-vmm-expert-table-probe.exe"
$object = Join-Path $probeOutput "cuda-vmm-expert-table-probe.obj"
$arguments = @("/nologo", "/std:c++17", "/EHsc", "/W4", "/O2", "/Ob2", "/DNDEBUG", "/DWIN32", "/D_WINDOWS", "/DNOMINMAX", "/MD", "/bigobj", "/utf-8",
    "/I$(Join-Path $upstream 'include')", "/I$(Join-Path $upstream 'src')", "/I$(Join-Path $CudaRoot 'include')",
    "/Fo$object", "/Fe$executable", $source, "/link", "/machine:x64", "/INCREMENTAL:NO", "/subsystem:console") + $libraries + @(
    "kernel32.lib", "user32.lib", "gdi32.lib", "winspool.lib", "shell32.lib", "ole32.lib", "oleaut32.lib", "uuid.lib", "comdlg32.lib", "advapi32.lib")
$log = Join-Path $probeOutput "build.log"
& cl.exe @arguments 2>&1 | Tee-Object -FilePath $log
if ($LASTEXITCODE -ne 0) { throw "Standalone probe compile/link failed: $LASTEXITCODE" }
$receipt = [ordered]@{
    status = "built_only_not_executed"
    utc = [DateTime]::UtcNow.ToString("o")
    upstreamRevision = $revision
    upstreamPorcelainStatus = "clean (staged, unstaged and untracked)"
    ggmlBuildRoot = $buildRoot
    source = $source
    sourceSha256 = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
    buildReference = "Existing GgmlOpsHostTokenEmbeddingTest from build.ninja; no buildRoot writes or regeneration"
    buildNinjaSha256 = (Get-FileHash -LiteralPath $ninja -Algorithm SHA256).Hash
    referenceLinkBlock = $reference
    libraries = @($libraries | ForEach-Object { [ordered]@{ path = $_; sha256 = (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash } })
    compiler = (Get-Command cl.exe).Source
    arguments = $arguments
    executable = $executable
    executableSha256 = (Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash
    buildLog = $log
    gpuExecuted = $false
    execution = "After exclusive GPU handoff only: prepend CUDA13.4/bin/x64 to PATH; run executable --execute; 0=passed, 77=unavailable/not coverage, 1=failure, 2=usage/no CUDA initialization"
}
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $probeOutput "build-receipt.json") -Encoding utf8
Write-Output "Built $executable. The wrapper never executes the probe or initializes CUDA."
