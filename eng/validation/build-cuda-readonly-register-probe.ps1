param(
    [string] $CudaRoot = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4",
    [string] $OutputDirectory = "artifacts/perf/glm-memory-tier-20261015/cuda-readonly-register-probe"
)
$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
. (Join-Path $workspace "eng/vs-locate.ps1")
$probeVs = Get-VisualStudioInstallation
if ($null -eq $probeVs) { throw "MSVC x64 toolset is unavailable." }
Import-VcVarsEnvironment $probeVs.VcVars64
$probeOutput = [System.IO.Path]::GetFullPath((Join-Path $workspace $OutputDirectory))
New-Item -ItemType Directory -Path $probeOutput -Force | Out-Null
$probeSource = Join-Path $PSScriptRoot "cuda-readonly-register-probe.cpp"
$probeExe = Join-Path $probeOutput "cuda-readonly-register-probe.exe"
$probeObject = Join-Path $probeOutput "cuda-readonly-register-probe.obj"
$probeLibrary = Join-Path $CudaRoot "lib/x64/cudart.lib"
if (-not (Test-Path -LiteralPath $probeLibrary)) { throw "Missing CUDA runtime import library: $probeLibrary" }
$probeArguments = @("/nologo", "/std:c++17", "/EHsc", "/W4", "/O2", "/MD", "/utf-8",
    "/I$(Join-Path $CudaRoot 'include')", "/Fo$probeObject", "/Fe$probeExe", $probeSource,
    "/link", $probeLibrary)
$probeLog = Join-Path $probeOutput "build.log"
& cl.exe @probeArguments 2>&1 | Tee-Object -FilePath $probeLog
if ($LASTEXITCODE -ne 0) { throw "Probe compile/link failed: $LASTEXITCODE" }
$probeReceipt = [ordered]@{
    status = "built_only"
    builtUtc = [DateTime]::UtcNow.ToString("o")
    cudaRoot = $CudaRoot
    cudaRuntimeImportLibrary = $probeLibrary
    cudaRuntimeImportLibrarySha256 = (Get-FileHash -LiteralPath $probeLibrary -Algorithm SHA256).Hash
    cudaRuntimeHeaderSha256 = (Get-FileHash -LiteralPath (Join-Path $CudaRoot 'include/cuda_runtime_api.h') -Algorithm SHA256).Hash
    compiler = (Get-Command cl.exe).Source
    arguments = $probeArguments
    source = $probeSource
    sourceSha256 = (Get-FileHash -LiteralPath $probeSource -Algorithm SHA256).Hash
    executable = $probeExe
    executableSha256 = (Get-FileHash -LiteralPath $probeExe -Algorithm SHA256).Hash
    buildLog = $probeLog
    gpuExecuted = $false
}
$probeReceipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $probeOutput "build-receipt.json") -Encoding utf8
Write-Output "Built $probeExe; this wrapper does not execute the probe or initialize CUDA."
