<#
.SYNOPSIS
Validate frozen GLM unified-memory native binaries and synthetic model fixtures.
.DESCRIPTION
Run after freezing binaries and reserving a serial GPU slot. Imports the x64
MSVC runtime environment, prepends the selected CUDA runtime, verifies exact
case lists and DLL identity, and preserves generated logs in a fresh output
directory. Skipped or unavailable required cases fail validation. This script
does not configure, build, or benchmark a full production model.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$NativeTestDirectory,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-fA-F]{64}$')][string]$ExpectedNativeSha256,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [string]$ManagedBinDirectory,
    [ValidateCount(1,14)][ValidateSet('CPU6','MappedCUDA3','FocusedGPU11','FileRing4','FileReadWorkers10','RetainedPeer8','VmmWeights3','VmmGlm3','VmmCoarse1','VmmCache6','VmmPressure1','FileReadSort2','SourceView1','Synthetic13')]
    [string[]]$Batches = @('CPU6','MappedCUDA3','FocusedGPU11','Synthetic13'),
    [ValidateRange(0,16)][int]$FileReadWorkers = 0,
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
    [string]$CudaRuntime = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
if (@($Batches | Select-Object -Unique).Count -ne $Batches.Count) { throw 'Each validation batch may be selected only once.' }

$cases = [ordered]@{
    CPU6 = @('host-moe-cache-exact-file-ring','host-moe-immutable-cache','host-moe-ram-probation',
        'host-moe-mapped-ram','host-moe-cache-phase-quotas','expert-file-source-portable')
    MappedCUDA3 = @('host-moe-mapped-ram-cuda-q8','host-moe-mapped-ram-cuda-iq2','host-moe-mapped-ram-cuda-pageable')
    FocusedGPU11 = @('glm-selected-split-lifecycle-q8','glm-selected-split-lifecycle-iq2',
        'glm-selected-scalar-staging72-q8','glm-selected-scalar-staging72-pageable',
        'glm-selected-prefill-ring72-q8','glm-selected-prefill-ring72-combined',
        'glm-selected-prefill-coalesce16-q8','glm-selected-prefill-coalesce16-pageable',
        'host-moe-ram-cache-cuda-q8','host-moe-ram-cache-cuda-iq2','host-moe-ram-cache-cuda-pageable')
    FileRing4 = @('glm-selected-prefill-ring72-q8','glm-selected-prefill-ring72-combined',
        'glm-selected-prefill-coalesce16-q8','glm-selected-prefill-coalesce16-pageable')
    FileReadWorkers10 = @('host-moe-file-read-workers','deepseek41-engram-parallel-io',
        'bounded-expert-read-pipeline','expert-file-source-portable',
        'host-moe-file-batch-cpu','host-moe-file-batch-mixed-cpu',
        'host-moe-file-batch-cuda','host-moe-file-batch-mixed-cuda',
        'glm-selected-split-lifecycle-q8','glm-selected-split-lifecycle-iq2')
    RetainedPeer8 = @('host-moe-cache-phase-quotas','host-moe-cache-measured-group-quotas',
        'host-moe-cache-row-route-policy','host-moe-cache-scoped-configuration',
        'glm-selected-phase-decode-rebalance','glm-selected-phase-rebalance-retain-peer',
        'glm-selected-phase-rebalance-retain-peer-guard','glm-selected-split-lifecycle-iq2')
    VmmWeights3 = @('host-moe-vmm-weights-q8_0','host-moe-vmm-weights-iq1_m','host-moe-vmm-weights-iq2_xxs')
    VmmGlm3 = @('host-moe-vmm-weights-glm_g3','host-moe-vmm-weights-glm_g4','host-moe-vmm-weights-glm_g11')
    VmmCoarse1 = @('host-moe-vmm-weights-coarse_q8')
    VmmCache6 = @('host-moe-vmm-cache-slot-policy','shared-cache-budget-lifetime',
        'host-moe-cache-phase-quotas','host-moe-cache-scoped-configuration',
        'glm-selected-vmm-cache-three-groups','glm-selected-split-lifecycle-iq2')
    VmmPressure1 = @('glm-selected-vmm-cache-pressure')
    FileReadSort2 = @('host-moe-miss-read-order','glm-selected-file-read-sort-q8')
    SourceView1 = @('expert-file-source-view-lifecycle')
}
$syntheticCases = @(
    'InferenceWeb.Tests.GlmFlashMtpNativeTests.N2PrefixCommitPreservesScalarOracleAndPoisonsPartialCopies',
    'InferenceWeb.Tests.GlmFlashMtpNativeTests.DraftWithoutHyperConnectionsPreservesTargetGreedyAndRejectedWindowRollback',
    'InferenceWeb.Tests.GlmFlashMtpNativeTests.N2PrefixCommitAndFullAcceptancePreserveExactStateAcross256RowPadding',
    'InferenceWeb.Tests.GlmFlashMtpNativeTests.ShapeSpecificGroupsPreserveFullOracleAndRejectBeforeDraftKvWrites',
    'InferenceWeb.Tests.GlmFlashMtpNativeTests.PublicDraftCoordinatesMatchReferenceInputsForWholeAndChunkedCatchUp'
)
foreach ($method in @('CaptureAndPartialRestoreExceptionsAreContainedAndResetRecovers',
                      'CudaCaptureAndPartialRestoreExceptionsAreContainedAndResetRecovers')) {
    foreach ($stage in @(1,3)) {
        foreach ($kind in @(1,2)) {
            $syntheticCases += "InferenceWeb.Tests.Glm5NextNativeSnapshotBoundaryTests.$method(captureStage: $stage, kind: $kind)"
        }
    }
}
$filter = 'FullyQualifiedName~GlmFlashMtpNativeTests|FullyQualifiedName~Glm5NextNativeSnapshotBoundaryTests'

function Assert-SameNames([string[]]$Expected, [string[]]$Actual, [string]$Context) {
    if ($Expected.Count -ne $Actual.Count -or @($Actual | Select-Object -Unique).Count -ne $Actual.Count -or
        @(Compare-Object ($Expected | Sort-Object) ($Actual | Sort-Object)).Count) {
        throw "$Context names/count differ; expected $($Expected.Count), actual $($Actual.Count)."
    }
}
function Invoke-Logged([string]$Program, [string[]]$ArgumentList, [string]$Log) {
    $priorPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Program @ArgumentList 2>&1 | Tee-Object -FilePath $Log | Out-Host
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $priorPreference }
    return $code
}
function Assert-NativeHash {
    if ((Get-FileHash -LiteralPath $nativeDll -Algorithm SHA256).Hash -ine $ExpectedNativeSha256) {
        throw 'Frozen native DLL SHA256 differs from the explicitly supplied identity.'
    }
}
function Save-Receipt {
    $receipt | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $receiptPath -Encoding UTF8
}

$NativeTestDirectory = (Resolve-Path -LiteralPath $NativeTestDirectory).Path
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $OutputDirectory) { throw 'Use a fresh output directory; previous receipts will not be overwritten.' }
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$receiptPath = Join-Path $OutputDirectory 'validation-summary.json'
$nativeDll = Join-Path $NativeTestDirectory 'GgmlOps.dll'
$receipt = [ordered]@{
    status='running'; started_utc=[DateTime]::UtcNow.ToString('o'); native_directory=$NativeTestDirectory
    expected_native_sha256=$ExpectedNativeSha256.ToUpperInvariant(); batches=[ordered]@{}
    file_read_workers=$FileReadWorkers
    limitations=@('Synthetic fixtures and focused cache lifecycle regressions; no full GLM5.3Flash model or performance qualification.',
        'Any skipped, unavailable, disabled, missing or unexecuted required case prevents a passed batch.',
        'Windows file-source fixture does not execute POSIX truncate/CLOEXEC scenarios.',
        'No physical SSD sharing, unique file-cache page identity or speed claim follows from these tests.')
}
foreach ($batch in $Batches) { $receipt.batches[$batch] = [ordered]@{status='not_run'} }

# Clear only the relevant experiment variables, and restore them in finally.
# No environment dump is written. CTest's per-case ENVIRONMENT remains authoritative.
$controlledNames = @('PATH','GGML_CUDA_NO_PINNED','TS_TEST_GGML_BACKEND','TS_TEST_GLM_CUDA',
    'TS_TEST_GLM_NATIVE_HOOKS','TS_TEACHER_TOKEN_EXPORT','TS_GLM_FLASH_MTP',
    'TS_GLM_FLASH_MTP_SHAPE_PLAN','TS_GLM_FLASH_MTP_PREFIX_COMMIT','TS_GLM_TRACE','TS_GLM_TRACE_DUMP_DIR',
    'TS_GLM_MOE_MMAP','TS_GLM_TP_SHARD','TS_GLM_TP_OVERSUBSCRIBE','TS_GLM_NODES_PER_LAYER',
    'TS_GLM_HOST_TOKEN_EMBEDDING','TS_GLM_LAYER_MAJOR_GRAPH_CAP_MB','TS_GLM_DECODE_GRAPH_RESERVE',
    'TS_HOST_MOE_FILE_READ_WORKERS','TS_HOST_MOE_FILE_RANDOM_ACCESS','TS_HOST_MOE_FILE_SOURCE_VIEW',
    'TS_HOST_MOE_FILE_SOURCE_VIEW_PREFETCH','TS_GLM_SELECTED_EXPERT_TIMING','TS_DSV4_PERF')
$hostFlags = @([regex]::Matches((Get-Content -LiteralPath (Join-Path $RepoRoot 'TensorSharp.GGML.Native\ggml_ops_host_moe_cache.cpp') -Raw),
    '"(TS_HOST_MOE_[A-Z0-9_]+)"') | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique)
$controlledNames += $hostFlags
$oldEnvironment = @{}
foreach ($name in $controlledNames) { $oldEnvironment[$name] = [Environment]::GetEnvironmentVariable($name,'Process') }
try {
    Assert-NativeHash
    if (!(Test-Path -LiteralPath $CudaRuntime -PathType Container)) { throw "Missing runtime PATH directory: $CudaRuntime" }
    . (Join-Path $RepoRoot 'eng\vs-locate.ps1')
    $vsTaskInstall = Get-VisualStudioInstallation
    if (!$vsTaskInstall) { throw 'The required x64 MSVC installation is unavailable.' }
    Import-VcVarsEnvironment $vsTaskInstall.VcVars64
    foreach ($name in $controlledNames | Where-Object { $_ -ne 'PATH' }) {
        # PowerShell can coerce a null string argument to an empty value. The
        # strict native profile parsers distinguish empty from genuinely unset.
        Remove-Item -LiteralPath ('Env:' + $name) -ErrorAction SilentlyContinue
    }
    $env:PATH = "$CudaRuntime;$NativeTestDirectory;$env:PATH"
    $env:TS_TEST_GGML_BACKEND = 'cpu'
    $env:TS_TEST_GLM_CUDA = '1'
    $env:TS_TEST_GLM_NATIVE_HOOKS = '1'
    $env:TS_HOST_MOE_FILE_READ_WORKERS = "$FileReadWorkers"
    $receipt.toolchain = [ordered]@{vcvars=$vsTaskInstall.VcVars64; cuda_runtime_path=$CudaRuntime}
    $upstream = Join-Path $RepoRoot 'ExternalProjects\ggml'
    $revision = & git -C $upstream rev-parse HEAD
    if ($LASTEXITCODE) { throw 'Cannot identify upstream ggml revision.' }
    $dirty = @(& git -C $upstream status --porcelain)
    if ($LASTEXITCODE) { throw 'Cannot establish upstream ggml status.' }
    $receipt.upstream = [ordered]@{revision="$revision".Trim(); status=$dirty}
    if ($dirty.Count) { throw 'Upstream ggml is dirty; validation cannot qualify an unchanged dependency.' }
    $receipt.native_files = @(Get-ChildItem -LiteralPath $NativeTestDirectory -File |
        Where-Object { $_.Extension -in @('.exe','.dll') -or $_.Name -eq 'CTestTestfile.cmake' } |
        ForEach-Object { [ordered]@{name=$_.Name; sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash} })
    Save-Receipt

    foreach ($batch in @('CPU6','MappedCUDA3','FocusedGPU11','FileRing4','FileReadWorkers10','RetainedPeer8','VmmWeights3','VmmGlm3','VmmCoarse1','VmmCache6','VmmPressure1','FileReadSort2','SourceView1')) {
        if ($batch -notin $Batches) { continue }
        $expected = $cases[$batch]
        $regex = '^(' + (($expected | ForEach-Object { [regex]::Escape($_) }) -join '|') + ')$'
        $discoveryPath = Join-Path $OutputDirectory "$batch-discovery.json"
        $discoverCode = Invoke-Logged 'ctest' @('--test-dir',$NativeTestDirectory,'--show-only=json-v1','-R',$regex) $discoveryPath
        if ($discoverCode) { throw "$batch CTest discovery failed ($discoverCode)." }
        $discovered = Get-Content -LiteralPath $discoveryPath -Raw | ConvertFrom-Json
        Assert-SameNames $expected @($discovered.tests | ForEach-Object { $_.name }) "$batch discovery"
        foreach ($test in $discovered.tests) {
            $executable = [IO.Path]::GetFullPath([string]$test.command[0])
            if ([IO.Path]::GetDirectoryName($executable) -ine $NativeTestDirectory) {
                throw "$batch points outside the frozen native folder: $executable"
            }
            if (!(Test-Path -LiteralPath $executable -PathType Leaf)) { throw "Missing frozen executable: $executable" }
        }
        Assert-NativeHash
        $receipt.batches[$batch] = [ordered]@{status='running'; expected=$expected}
        Save-Receipt
        $xmlPath = Join-Path $OutputDirectory "$batch.xml"
        $exitCode = Invoke-Logged 'ctest' @('--test-dir',$NativeTestDirectory,'--parallel','1','--verbose',
            '--output-on-failure','--no-tests=error','-R',$regex,'--output-junit',$xmlPath) (Join-Path $OutputDirectory "$batch.log")
        $lastTest = Join-Path $NativeTestDirectory 'Testing\Temporary\LastTest.log'
        if (Test-Path -LiteralPath $lastTest) { Copy-Item -LiteralPath $lastTest -Destination (Join-Path $OutputDirectory "$batch-LastTest.log") }
        if (!(Test-Path -LiteralPath $xmlPath)) { throw "$batch produced no JUnit receipt; exit=$exitCode." }
        [xml]$xml = Get-Content -LiteralPath $xmlPath -Raw
        $executed = @($xml.SelectNodes('//testcase'))
        Assert-SameNames $expected @($executed | ForEach-Object { $_.GetAttribute('name') }) "$batch execution"
        $bad = @($executed | Where-Object { $_.GetAttribute('status') -ne 'run' -or $_.SelectSingleNode('failure|error|skipped') })
        $receipt.batches[$batch] = [ordered]@{
            status=$(if (!$exitCode -and !$bad.Count) {'passed'} else {'failed'}); exit_code=$exitCode
            executed=$executed.Count; failed_or_skipped=$bad.Count
            cases=@($executed | ForEach-Object { [ordered]@{name=$_.GetAttribute('name'); status=$_.GetAttribute('status')} })
        }
        Save-Receipt
        if ($exitCode -or $bad.Count) { throw "$batch failed or had unavailable/skipped cases; later batches remain not_run." }
        Assert-NativeHash
    }

    if ('Synthetic13' -in $Batches) {
        if (!$ManagedBinDirectory) { throw 'Synthetic13 requires an explicit frozen ManagedBinDirectory.' }
        $ManagedBinDirectory = (Resolve-Path -LiteralPath $ManagedBinDirectory).Path
        $privateBin = Join-Path $OutputDirectory 'synthetic-private'
        New-Item -ItemType Directory -Path $privateBin | Out-Null
        # Inputs are frozen and these cases never write their managed binaries.
        # Link immutable files where supported, preserving the directory tree;
        # cross-volume/unsupported hardlinks fall back to a private file copy.
        $managedPrefix = $ManagedBinDirectory.TrimEnd([char[]]'\/') + [IO.Path]::DirectorySeparatorChar
        $managedLinked = 0; $managedCopied = 0; $managedNativeExcluded = 0
        foreach ($item in Get-ChildItem -LiteralPath $ManagedBinDirectory -Force -Recurse) {
            $relative = $item.FullName.Substring($managedPrefix.Length)
            if ($relative -ieq 'GgmlOps.dll') { $managedNativeExcluded++; continue }
            $destination = Join-Path $privateBin $relative
            if ($item.PSIsContainer) {
                [IO.Directory]::CreateDirectory($destination) | Out-Null
                continue
            }
            [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($destination)) | Out-Null
            try {
                New-Item -ItemType HardLink -Path $destination -Target $item.FullName -ErrorAction Stop | Out-Null
                $managedLinked++
            } catch {
                # Never overwrite a link if a provider created it before
                # reporting failure; that would write into the frozen input.
                if (Test-Path -LiteralPath $destination) { throw }
                [IO.File]::Copy($item.FullName, $destination, $false)
                $managedCopied++
            }
        }
        # Native LAST and COPY ONLY. The root managed DLL is never linked, and
        # overwrite=false prevents even an unexpected existing link being used.
        [IO.File]::Copy($nativeDll, (Join-Path $privateBin 'GgmlOps.dll'), $false)
        $receipt.synthetic_staging = [ordered]@{strategy='immutable_hardlinks_with_copy_fallback';
            linked_files=$managedLinked; copied_files=$managedCopied; excluded_root_native=$managedNativeExcluded;
            frozen_native='private_copy_last_no_overwrite'}
        if ((Get-FileHash -LiteralPath (Join-Path $privateBin 'GgmlOps.dll') -Algorithm SHA256).Hash -ine $ExpectedNativeSha256) {
            throw 'Private synthetic native DLL identity mismatch.'
        }
        $receipt.synthetic_files = @('InferenceWeb.Tests.dll','TensorSharp.Models.dll','TensorSharp.Backends.GGML.dll','GgmlOps.dll') |
            ForEach-Object { [ordered]@{name=$_; sha256=(Get-FileHash -LiteralPath (Join-Path $privateBin $_) -Algorithm SHA256).Hash} }
        $results = Join-Path $OutputDirectory 'synthetic-results'
        $receipt.batches.Synthetic13 = [ordered]@{status='running'; expected=$syntheticCases; filter=$filter}
        Save-Receipt
        $syntheticArgs = @('vstest',(Join-Path $privateBin 'InferenceWeb.Tests.dll'),
            "/TestCaseFilter:$filter",'/Logger:trx;LogFileName=synthetic13.trx',"/ResultsDirectory:$results")
        $exitCode = Invoke-Logged 'dotnet' $syntheticArgs (Join-Path $OutputDirectory 'Synthetic13.log')
        $trxPath = Join-Path $results 'synthetic13.trx'
        if (!(Test-Path -LiteralPath $trxPath)) { throw "Synthetic13 produced no TRX receipt; exit=$exitCode." }
        [xml]$trx = Get-Content -LiteralPath $trxPath -Raw
        $ns = New-Object System.Xml.XmlNamespaceManager($trx.NameTable)
        $ns.AddNamespace('t','http://microsoft.com/schemas/VisualStudio/TeamTest/2010')
        $executed = @($trx.SelectNodes('//t:UnitTestResult',$ns))
        Assert-SameNames $syntheticCases @($executed | ForEach-Object { $_.GetAttribute('testName') }) 'Synthetic13 execution'
        $bad = @($executed | Where-Object { $_.GetAttribute('outcome') -ne 'Passed' })
        $receipt.batches.Synthetic13 = [ordered]@{
            status=$(if (!$exitCode -and !$bad.Count) {'passed'} else {'failed'}); exit_code=$exitCode
            executed=$executed.Count; failed_or_skipped=$bad.Count; cuda_cases=9; cpu_cases=4
            cases=@($executed | ForEach-Object { [ordered]@{name=$_.GetAttribute('testName'); outcome=$_.GetAttribute('outcome')} })
        }
        Save-Receipt
        if ($exitCode -or $bad.Count) { throw 'Synthetic13 failed or skipped/unavailable cases.' }
    }
    Assert-NativeHash
    foreach ($batch in $Batches) {
        if ($receipt.batches[$batch].status -ne 'passed') { throw "Requested batch did not execute successfully: $batch" }
    }
    $receipt.status = 'passed'
} catch {
    $receipt.status = 'failed'
    $receipt.error = $_.Exception.Message
    throw
} finally {
    $receipt.finished_utc = [DateTime]::UtcNow.ToString('o')
    Save-Receipt
    foreach ($name in $controlledNames) {
        if ($null -eq $oldEnvironment[$name]) {
            Remove-Item -LiteralPath ('Env:' + $name) -ErrorAction SilentlyContinue
        } else {
            [Environment]::SetEnvironmentVariable($name,$oldEnvironment[$name],'Process')
        }
    }
}
