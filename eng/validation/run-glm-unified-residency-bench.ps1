<#
.SYNOPSIS
Run one GLM residency control or candidate, or fresh balanced same-binary ABBA arms.
.DESCRIPTION
Requires the root task's exclusive GPU slot for the entire invocation. It does
not acquire that slot, probe other processes, build, or run unrelated tests.
Control runs one experiment-OFF arm against the frozen baseline; it is not a
balanced ABBA or speed qualification. Candidate enables the selected experiment
and compares both quick-plan API
responses to the frozen v23 A1 oracle. Abba runs off/on/on/off,
compares its fresh A1 to that
oracle and the remaining arms to fresh A1. Every arm requires the canonical
1872-token full-logit hash and actual request-scoped experiment activity.
Generated receipts go under ignored artifacts/.
#>
[CmdletBinding()]
param(
    [ValidateSet('Candidate','Control','Abba')][string]$Mode = 'Candidate',
    [ValidateSet('GraphCap','HostTokenEmbedding','PrefillDownOverlap','DecodeGraphReserve','FileReadWorkers','FileRandomAccess','FileReadSort','FileSourceView','FileSourceViewPrefetch','DecodeRetainPeer','VmmWeights')][string]$Experiment = 'HostTokenEmbedding',
    [ValidateRange(1,16)][int]$ReadWorkers = 8,
    [ValidateRange(0,16)][int]$FixedReadWorkers = 0,
    [ValidateSet(0,512,768)][int]$FixedGraphCapMiB = 0,
    [ValidateSet(0,1)][int]$FixedHostTokenEmbedding = 0,
    [ValidateSet(0,1)][int]$FixedDecodeRebalance = 0,
    [ValidateSet(0,1)][int]$FixedDecodeGraphReserve = 0,
    [switch]$SelectedExpertDiagnostics,
    [switch]$GpuClockDiagnostics,
    [switch]$RouteTrace,
    [switch]$RequireVmmPressure,
    [ValidateSet(512,768)][int]$CapMiB = 768,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-fA-F]{64}$')][string]$ExpectedNativeSha256,
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')][string]$OutputPrefix,
    [switch]$ExclusiveGpuSlot,
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
    [string]$EvidenceRoot,
    [string]$ServerDirectory,
    [string]$BaselineReport,
    [string]$Model = 'C:\Works\models\GLM-5.3-Flash-GGUF\GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf',
    [string]$Python = 'python',
    [int]$Port = 42190,
    [string]$CudaRuntime = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
if (!$ExclusiveGpuSlot) { throw 'Reserve the root task exclusive GPU slot, then pass -ExclusiveGpuSlot. No workload has been launched.' }
if ($SelectedExpertDiagnostics -and $Mode -ne 'Candidate') { throw 'Selected-expert diagnostics add host timing/logging and cannot qualify a performance ABBA.' }
if ($GpuClockDiagnostics -and ($Mode -ne 'Candidate' -or !$SelectedExpertDiagnostics)) { throw 'GpuClockDiagnostics requires Candidate mode and SelectedExpertDiagnostics; sampled runs are not performance controls.' }
if ($GpuClockDiagnostics -and $Experiment -in @('FileReadWorkers','FileRandomAccess')) { throw 'GPU clock diagnostics cannot qualify file-reader or random-access performance experiments.' }
if ($RouteTrace -and !$SelectedExpertDiagnostics) { throw 'RouteTrace is restricted to explicitly labelled SelectedExpertDiagnostics runs.' }
if ($Experiment -eq 'GraphCap' -and $FixedGraphCapMiB) { throw 'FixedGraphCapMiB applies when GraphCap is held constant.' }
if ($Experiment -eq 'HostTokenEmbedding' -and $FixedHostTokenEmbedding) { throw 'FixedHostTokenEmbedding applies when HostTokenEmbedding is held constant.' }
if ($Experiment -eq 'PrefillDownOverlap' -and $FixedHostTokenEmbedding -ne 1) { throw 'PrefillDownOverlap requires -FixedHostTokenEmbedding 1 for the guarded native profile.' }
if ($Experiment -eq 'DecodeGraphReserve' -and $FixedDecodeGraphReserve) { throw 'FixedDecodeGraphReserve applies when DecodeGraphReserve is held constant.' }
if ($Experiment -eq 'DecodeGraphReserve' -and $FixedHostTokenEmbedding -ne 1) { throw 'DecodeGraphReserve qualification requires -FixedHostTokenEmbedding 1 for the fixed geometry.' }
if ($Experiment -eq 'FileReadWorkers' -and ($FixedHostTokenEmbedding -ne 1 -or $FixedGraphCapMiB -or $FixedDecodeRebalance -or $FixedDecodeGraphReserve -or $SelectedExpertDiagnostics)) { throw 'FileReadWorkers requires fixed Host embedding 1 with graph cap, reserve, rebalance and selected diagnostics disabled.' }
if ($Experiment -eq 'FileRandomAccess' -and ($FixedHostTokenEmbedding -ne 1 -or $FixedGraphCapMiB -or $FixedDecodeRebalance -or $FixedDecodeGraphReserve -or $SelectedExpertDiagnostics)) { throw 'FileRandomAccess requires fixed Host embedding 1 with graph cap, reserve, rebalance and selected diagnostics disabled.' }
if ($Experiment -in @('FileReadSort','FileSourceView','FileSourceViewPrefetch') -and ($FixedHostTokenEmbedding -ne 1 -or $FixedDecodeRebalance -ne 1 -or $FixedGraphCapMiB -or $FixedDecodeGraphReserve -or $FixedReadWorkers -or $SelectedExpertDiagnostics -or $GpuClockDiagnostics -or $RouteTrace)) { throw 'FileReadSort/FileSourceView/FileSourceViewPrefetch requires fixed Host embedding 1 and decode rebalance 1, with other read/cache/graph diagnostics disabled.' }
if ($Experiment -eq 'DecodeRetainPeer' -and ($FixedHostTokenEmbedding -ne 1 -or $FixedDecodeRebalance -ne 1 -or $FixedGraphCapMiB -or $FixedDecodeGraphReserve -or $SelectedExpertDiagnostics -or $GpuClockDiagnostics)) { throw 'DecodeRetainPeer requires fixed Host embedding 1 and decode rebalance 1, with graph cap, reserve and selected/GPU diagnostics disabled.' }
if ($Experiment -eq 'VmmWeights' -and ($FixedHostTokenEmbedding -ne 1 -or $FixedDecodeRebalance -ne 1 -or $FixedGraphCapMiB -or $FixedDecodeGraphReserve -or $SelectedExpertDiagnostics -or $GpuClockDiagnostics -or $RouteTrace)) { throw 'VmmWeights requires fixed Host embedding 1 and decode rebalance 1, with graph cap, reserve and selected/GPU diagnostics disabled.' }
if ($RequireVmmPressure -and $Experiment -ne 'VmmWeights') { throw 'RequireVmmPressure applies only to VmmWeights ON arms.' }
if ($Experiment -ne 'FileRandomAccess' -and $FixedReadWorkers) { throw 'FixedReadWorkers is restricted to the FileRandomAccess isolation profile.' }
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
if (!$EvidenceRoot) { $EvidenceRoot = Join-Path $RepoRoot 'artifacts\perf\glm-memory-tier-20261015' }
$EvidenceRoot = (Resolve-Path -LiteralPath $EvidenceRoot).Path
if (!$ServerDirectory) {
    $defaultServer = if ($Experiment -in @('FileSourceView','FileSourceViewPrefetch')) { 'file-source-view-cuda13-bin' } elseif ($Experiment -eq 'FileReadSort') { 'file-read-sort-cuda13-bin' } elseif ($Experiment -eq 'VmmWeights') { 'vmm-weights-cuda13-bin' } elseif ($Experiment -eq 'DecodeRetainPeer') { 'retained-peer-cuda13-bin' } elseif ($Experiment -eq 'FileRandomAccess') { 'file-random-access-cuda13-bin' } elseif ($Experiment -eq 'FileReadWorkers') { 'file-read-workers-cuda13-bin' } elseif ($Experiment -in @('PrefillDownOverlap','DecodeGraphReserve')) { 'prefill-down-overlap-cuda13-bin' } elseif ($Experiment -eq 'HostTokenEmbedding') { 'host-token-embedding-cuda13-bin' } else { 'strict-graph-cap-cuda13-hostfix-bin' }
    $ServerDirectory = Join-Path $EvidenceRoot $defaultServer
}
$ServerDirectory = (Resolve-Path -LiteralPath $ServerDirectory).Path
if (!$BaselineReport) { $BaselineReport = Join-Path $EvidenceRoot 'plain-mapped-probation8-quick-v23-cuda13-retry1-20261022-a1\report.json' }
$BaselineReport = (Resolve-Path -LiteralPath $BaselineReport).Path
$harness = Join-Path $RepoRoot 'eng\validation\unified-perf-bench.py'
$harnessHash = 'C80AFB4058CBA4C5C0FE36674252971776DCA5CFA90CD82F4E8483C82441D24B'
$harnessHelper = Join-Path $RepoRoot 'eng\validation\glm_decode_graph_reserve.py'
$harnessHelperHash = 'B0F65700F1280B4121F8560CC55F99FFC82E3D7B29AB64DFBC7AC9D7CBAF405E'
$fileWorkersHelper = Join-Path $RepoRoot 'eng\validation\glm_file_read_workers.py'
$fileWorkersHelperHash = 'FB4A1BFDECB3D3DD5FBCAF2F50B7AA7305331410C247F5A7D1EE9D77E4062EA4'
$retainPeerHelper = Join-Path $RepoRoot 'eng\validation\glm_decode_retain_peer.py'
$retainPeerHelperHash = 'D6BBE3725CC4B4EB8CD438A67343E2360B1D649B7084275BFDFBB058613510E7'
$vmmWeightsHelper = Join-Path $RepoRoot 'eng\validation\glm_vmm_weights.py'
$vmmWeightsHelperHash = '0227A46FA583551A6866DB633A130BA32A84FA4716C9842392EEA827726B3BC3'
$fileSortHelper = Join-Path $RepoRoot 'eng\validation\glm_file_read_sort.py'
$fileSortHelperHash = '8A8CF75D42E73F45A4E5168E13D527A97C60210EC841B5BC7E5D1CEB42D4110F'
$fileViewHelper = Join-Path $RepoRoot 'eng\validation\glm_file_source_view.py'
$fileViewHelperHash = '55DAAFCA8F2A7C3B832F80CF3630D687583E8D133431B73BA99C7C8D68CBD98A'
$gpuClockHelper = Join-Path $RepoRoot 'eng\validation\glm_gpu_clock_telemetry.py'
$gpuClockHelperHash = 'B266D41E4677D4F88BCD4280E1D8E14E7CB80B7659D3A06604C99DC474AAF1A8'
$canonicalHash = '497C969504743B703F98C84998A210663AF9AF4BB973481A873EF2DA6576AD85'
$server = Join-Path $ServerDirectory 'TensorSharp.Server.Host.exe'
foreach ($path in @($Model,$server,$harness,$harnessHelper,$fileWorkersHelper,$retainPeerHelper,$vmmWeightsHelper,$fileSortHelper,$fileViewHelper,$BaselineReport)) {
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required file unavailable: $path" }
}
if (!(Test-Path -LiteralPath $CudaRuntime -PathType Container)) { throw "CUDA runtime directory unavailable: $CudaRuntime" }
if ((Get-FileHash -LiteralPath $harness -Algorithm SHA256).Hash -ine $harnessHash) { throw 'The frozen validation harness SHA256 changed.' }
if ((Get-FileHash -LiteralPath $harnessHelper -Algorithm SHA256).Hash -ine $harnessHelperHash) { throw 'The frozen decode-reserve helper SHA256 changed.' }
if ((Get-FileHash -LiteralPath $fileWorkersHelper -Algorithm SHA256).Hash -ine $fileWorkersHelperHash) { throw 'The frozen file-reader helper SHA256 changed.' }
if ((Get-FileHash -LiteralPath $retainPeerHelper -Algorithm SHA256).Hash -ine $retainPeerHelperHash) { throw 'The frozen retained-peer helper SHA256 changed.' }
if ((Get-FileHash -LiteralPath $vmmWeightsHelper -Algorithm SHA256).Hash -ine $vmmWeightsHelperHash) { throw 'The frozen VMM weights helper SHA256 changed.' }
if ((Get-FileHash -LiteralPath $fileSortHelper -Algorithm SHA256).Hash -ine $fileSortHelperHash) { throw 'The frozen file-sort helper SHA256 changed.' }
if ((Get-FileHash -LiteralPath $fileViewHelper -Algorithm SHA256).Hash -ine $fileViewHelperHash) { throw 'The frozen file-view helper SHA256 changed.' }
if ($GpuClockDiagnostics -and (!(Test-Path -LiteralPath $gpuClockHelper -PathType Leaf) -or (Get-FileHash -LiteralPath $gpuClockHelper -Algorithm SHA256).Hash -ine $gpuClockHelperHash)) { throw 'The frozen GPU clock monitor is unavailable or changed.' }
$binaryNames = @('GgmlOps.dll','TensorSharp.Server.Host.exe','TensorSharp.Server.Host.dll',
    'TensorSharp.Server.dll','TensorSharp.Chat.dll','TensorSharp.Runtime.dll','TensorSharp.Models.dll','TensorSharp.Backends.GGML.dll')
$binaryHashes = [ordered]@{}
foreach ($name in $binaryNames) { $binaryHashes[$name] = (Get-FileHash -LiteralPath (Join-Path $ServerDirectory $name) -Algorithm SHA256).Hash }
if ($binaryHashes['GgmlOps.dll'] -ine $ExpectedNativeSha256) { throw 'Private server native DLL differs from the explicitly frozen SHA256.' }

$armNames = if ($Mode -eq 'Abba') { @('a1','b1','b2','a2') } elseif ($Mode -eq 'Control') { @('control') } else { @('candidate') }
$receiptPath = Join-Path $EvidenceRoot ($OutputPrefix + '-launcher.json')
if (Test-Path -LiteralPath $receiptPath) { throw 'Output prefix already has a launcher receipt; choose a fresh prefix.' }
foreach ($arm in $armNames) {
    foreach ($suffix in @("-$arm", "-${arm}-console.log", "-${arm}-logits.f32")) {
        if (Test-Path -LiteralPath (Join-Path $EvidenceRoot ($OutputPrefix + $suffix))) {
            throw 'An arm output already exists; choose a fresh output prefix.'
        }
    }
}
$selectedTimingValue = if ($SelectedExpertDiagnostics) { '1' } else { '0' }
$dsv4PerfValue = if ($SelectedExpertDiagnostics) { '3' } else { '0' }
$routeTraceValue = if ($RouteTrace) { '1' } else { '0' }
$common = [ordered]@{
    TS_HOST_MOE_DECODE_STAGING_72MB='1'; TS_HOST_MOE_PREFILL_RING_72MB='1'
    TS_HOST_MOE_PREFILL_RING_COALESCE16='0'; TS_HOST_MOE_PHASE_COLD_GROWTH='0'
    TS_HOST_MOE_PREFILL_DOWN_OVERLAP='0'
    TS_GLM_GRAPH_MEMORY_DIAGNOSTICS='0'; TS_SCHED_PREFILL_CHUNK='512'; MAX_CONTEXT='2048'
    TS_GGML_STRICT_RECURRENT_PROJECTIONS='0'; TS_GLM_OP_OFFLOAD='0'; TS_UNIFIED_LAYER_MAJOR_TOKENS='8192'
    TS_UNIFIED_MAX_RUNNING_REQUESTS='1'; TS_HOST_MOE_EXPERT_CACHE_TIMING='1'
    TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS='1'; TS_GLM_FLASH_MTP_FORWARD_TIMING='1'
    TS_HOST_MOE_RAM_CACHE_MB='0'; TS_HOST_MOE_RAM_CACHE_MAPPED='0'
    TS_HOST_MOE_RAM_CACHE_MAPPED_EXCLUSIVE_QUOTA='0'; TS_HOST_MOE_RAM_CACHE_MAPPED_QUOTA_MB='0'
    TS_HOST_MOE_RAM_CACHE_VICTIM='0'; TS_HOST_MOE_RAM_CACHE_PINNED='0'
    TS_HOST_MOE_RAM_CACHE_DIRECT_UPLOAD='0'; TS_HOST_MOE_RAM_CACHE_FREQUENCY='0'
    TS_HOST_MOE_RAM_CACHE_PREFETCH_ONLY='0'; TS_HOST_MOE_LAYER_PREFETCH='0'
    TS_HOST_MOE_EXPERT_CACHE_PREFETCH='0'; TS_HOST_MOE_DECODE_PREFETCH='0'
    TS_HOST_MOE_EXPERT_CACHE_DIRECT_READS='0'; TS_HOST_MOE_EXPERT_CACHE_ORDERED_READS='0'
    TS_HOST_MOE_EXPERT_CACHE_RING_READS='2'; TS_HOST_MOE_DECODE_REBALANCE="$FixedDecodeRebalance"
    TS_HOST_MOE_DECODE_REBALANCE_RETAIN_PEER='0'
    TS_HOST_MOE_VMM_WEIGHTS='0'; TS_HOST_MOE_EXPERT_CACHE_LFU='0'
    TS_HOST_MOE_FILE_READ_WORKERS='0'; TS_HOST_MOE_FILE_RANDOM_ACCESS='0'; TS_HOST_MOE_FILE_READ_SORT='0'
    TS_HOST_MOE_FILE_SOURCE_VIEW_PREFETCH='0'
    TS_GGML_Q8_ORDERED_SMALL_BATCH='0'; TS_GGML_Q8_ORDERED_GATE_PREFILL='0'
    TS_GGML_Q8_GATE_PREFILL_TILE16='0'; TS_GLM_FLASH_MTP='0'
    TS_GLM_FLASH_MTP_SHAPE_PLAN='0'; TS_GLM_FLASH_MTP_PREFIX_COMMIT='0'; TS_GLM_DECODE_GRAPH_RESERVE="$FixedDecodeGraphReserve"
    TS_GLM_SELECTED_EXPERT_TIMING=$selectedTimingValue; TS_DSV4_PERF=$dsv4PerfValue; TS_HOST_MOE_ROUTE_TRACE=$routeTraceValue
}
$receipt = [ordered]@{status='running';started_utc=[DateTime]::UtcNow.ToString('o');mode=$Mode;experiment=$Experiment;selected_read_workers=$ReadWorkers;fixed_read_workers=$FixedReadWorkers;selected_cap_mib=$CapMiB;fixed_cap_mib=$FixedGraphCapMiB;fixed_decode_rebalance=$FixedDecodeRebalance;fixed_host_token_embedding=$FixedHostTokenEmbedding;fixed_decode_graph_reserve=$FixedDecodeGraphReserve
    selected_expert_diagnostics=[bool]$SelectedExpertDiagnostics;gpu_clock_diagnostics=[bool]$GpuClockDiagnostics;route_trace=[bool]$RouteTrace;require_vmm_pressure=[bool]$RequireVmmPressure;timing_eligible=(!$SelectedExpertDiagnostics -and !$RouteTrace -and !$GpuClockDiagnostics)
    exclusive_gpu_slot_attested=$true;server_directory=$ServerDirectory;native_sha256=$ExpectedNativeSha256.ToUpperInvariant()
    binaries=$binaryHashes;harness_sha256=$harnessHash;harness_helper_sha256=$harnessHelperHash;file_workers_helper_sha256=$fileWorkersHelperHash;retain_peer_helper_sha256=$retainPeerHelperHash;vmm_weights_helper_sha256=$vmmWeightsHelperHash;file_sort_helper_sha256=$fileSortHelperHash;file_view_helper_sha256=$fileViewHelperHash;gpu_clock_helper_sha256=$(if ($GpuClockDiagnostics) { $gpuClockHelperHash } else { $null });baseline_report=$BaselineReport
    baseline_report_sha256=(Get-FileHash -LiteralPath $BaselineReport -Algorithm SHA256).Hash
    model=$Model;canonical_prefill_tokens=1872;canonical_logit_values=154880;canonical_logit_sha256=$canonicalHash
    environment=$common;arms=[ordered]@{};scope='Quick PP1872+D32 and PP33+D128; exact API and one full-logit vector, not general semantic quality or total VRAM.'}
if ($Mode -eq 'Control') {
    $receipt.control_comparison = [ordered]@{design='single_disabled_arm_against_frozen_baseline';balanced_abba=$false;speed_qualified=$false}
}
foreach ($arm in $armNames) { $receipt.arms[$arm] = [ordered]@{status='not_run'} }
function Save-Receipt { $receipt | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $receiptPath -Encoding UTF8 }
$originalPath = $env:PATH
try {
    $env:PATH = "$CudaRuntime;$env:PATH"
    Save-Receipt
    foreach ($arm in $armNames) {
        foreach ($name in $binaryNames) {
            if ((Get-FileHash -LiteralPath (Join-Path $ServerDirectory $name) -Algorithm SHA256).Hash -ine $binaryHashes[$name]) {
                throw "Frozen binary changed before $arm : $name"
            }
        }
        if ((Get-FileHash -LiteralPath $harness -Algorithm SHA256).Hash -ine $harnessHash) { throw 'Harness changed between arms.' }
        if ((Get-FileHash -LiteralPath $harnessHelper -Algorithm SHA256).Hash -ine $harnessHelperHash) { throw 'Decode-reserve helper changed between arms.' }
        if ((Get-FileHash -LiteralPath $fileWorkersHelper -Algorithm SHA256).Hash -ine $fileWorkersHelperHash) { throw 'File-reader helper changed between arms.' }
        if ((Get-FileHash -LiteralPath $retainPeerHelper -Algorithm SHA256).Hash -ine $retainPeerHelperHash) { throw 'Retained-peer helper changed between arms.' }
        if ((Get-FileHash -LiteralPath $vmmWeightsHelper -Algorithm SHA256).Hash -ine $vmmWeightsHelperHash) { throw 'VMM weights helper changed between arms.' }
        if ((Get-FileHash -LiteralPath $fileSortHelper -Algorithm SHA256).Hash -ine $fileSortHelperHash) { throw 'File-sort helper changed between arms.' }
        if ((Get-FileHash -LiteralPath $fileViewHelper -Algorithm SHA256).Hash -ine $fileViewHelperHash) { throw 'File-view helper changed between arms.' }
        if ($GpuClockDiagnostics -and (Get-FileHash -LiteralPath $gpuClockHelper -Algorithm SHA256).Hash -ine $gpuClockHelperHash) { throw 'GPU clock monitor changed before diagnostic arm.' }
        $enabled = ($Mode -eq 'Candidate') -or ($Mode -eq 'Abba' -and $arm.StartsWith('b'))
        $cap = if ($Experiment -eq 'GraphCap') { if ($enabled) { "$CapMiB" } else { '0' } } else { "$FixedGraphCapMiB" }
        $hostEmbedding = if ($Experiment -eq 'HostTokenEmbedding') { if ($enabled) { '1' } else { '0' } } else { "$FixedHostTokenEmbedding" }
        $downOverlap = if ($Experiment -eq 'PrefillDownOverlap' -and $enabled) { '1' } else { '0' }
        $decodeGraphReserve = if ($Experiment -eq 'DecodeGraphReserve') { if ($enabled) { '1' } else { '0' } } else { "$FixedDecodeGraphReserve" }
        $fileReadWorkers = if ($Experiment -eq 'FileReadWorkers' -and $enabled) { "$ReadWorkers" } elseif ($Experiment -eq 'FileRandomAccess') { "$FixedReadWorkers" } else { '0' }
        $fileRandomAccess = if ($Experiment -eq 'FileRandomAccess' -and $enabled) { '1' } else { '0' }
        $fileReadSort = if ($Experiment -eq 'FileReadSort' -and $enabled) { '1' } else { '0' }
        $fileSourceView = if ($Experiment -in @('FileSourceView','FileSourceViewPrefetch') -and $enabled) { '1' } else { '0' }
        $fileSourcePrefetch = if ($Experiment -eq 'FileSourceViewPrefetch' -and $enabled) { '1' } else { '0' }
        $retainPeer = if ($Experiment -eq 'DecodeRetainPeer' -and $enabled) { '1' } else { '0' }
        $vmmWeights = if ($Experiment -eq 'VmmWeights' -and $enabled) { '1' } else { '0' }
        $tag = "$OutputPrefix-$arm"
        $out = Join-Path $EvidenceRoot $tag
        $logits = Join-Path $EvidenceRoot ($tag + '-logits.f32')
        $console = Join-Path $EvidenceRoot ($tag + '-console.log')
        $compare = if ($Mode -eq 'Abba' -and $arm -ne 'a1') {
            Join-Path $EvidenceRoot ($OutputPrefix + '-a1\report.json')
        } else { $BaselineReport }
        $arguments = @($harness,'--server',$server,'--model',$Model,'--plan','quick','--repeat','1',
            '--port',"$Port",'--unified-memory','--output',$out,'--compare-with',$compare)
        if ($GpuClockDiagnostics) { $arguments += '--gpu-telemetry' }
        foreach ($pair in $common.GetEnumerator()) { $arguments += @('--env',($pair.Key + '=' + $pair.Value)) }
        if ($Experiment -in @('FileReadWorkers','FileRandomAccess')) { $arguments += '--require-file-read-workers' }
        if ($Experiment -eq 'FileRandomAccess') { $arguments += '--require-file-random-access' }
        if ($Experiment -eq 'FileReadSort') { $arguments += '--require-file-read-sort' }
        if ($Experiment -eq 'FileSourceView') { $arguments += '--require-file-source-view' }
        if ($Experiment -eq 'FileSourceViewPrefetch') { $arguments += '--require-file-source-view-prefetch' }
        if ($RequireVmmPressure -and $vmmWeights -eq '1') { $arguments += '--require-vmm-pressure' }
        $arguments += @('--env',('TS_GLM_LAYER_MAJOR_GRAPH_CAP_MB=' + $cap),
            '--env',('TS_GLM_HOST_TOKEN_EMBEDDING=' + $hostEmbedding),
            '--env',('TS_HOST_MOE_PREFILL_DOWN_OVERLAP=' + $downOverlap),
            '--env',('TS_GLM_DECODE_GRAPH_RESERVE=' + $decodeGraphReserve),
            '--env',('TS_HOST_MOE_FILE_READ_WORKERS=' + $fileReadWorkers),
            '--env',('TS_HOST_MOE_FILE_RANDOM_ACCESS=' + $fileRandomAccess),
            '--env',('TS_HOST_MOE_FILE_READ_SORT=' + $fileReadSort),
            '--env',('TS_HOST_MOE_FILE_SOURCE_VIEW=' + $fileSourceView),
            '--env',('TS_HOST_MOE_FILE_SOURCE_VIEW_PREFETCH=' + $fileSourcePrefetch),
            '--env',('TS_HOST_MOE_DECODE_REBALANCE_RETAIN_PEER=' + $retainPeer),
            '--env',('TS_HOST_MOE_VMM_WEIGHTS=' + $vmmWeights),
            '--env',('TS_DUMP_LOGITS=' + $logits),'--env','TS_DUMP_LOGITS_AT=1872')
        $receipt.arms[$arm] = [ordered]@{status='running';cap_mib=[int]$cap;host_token_embedding=[int]$hostEmbedding;prefill_down_overlap=[int]$downOverlap;decode_graph_reserve=[int]$decodeGraphReserve;file_read_workers=[int]$fileReadWorkers;file_random_access=[int]$fileRandomAccess;file_read_sort=[int]$fileReadSort;file_source_view=[int]$fileSourceView;file_source_view_prefetch=[int]$fileSourcePrefetch;decode_retain_peer=[int]$retainPeer;vmm_weights=[int]$vmmWeights;report=(Join-Path $out 'report.json');compare_with=$compare}
        Save-Receipt
        & $Python @arguments *> $console
        $exitCode = $LASTEXITCODE
        $receipt.arms[$arm].exit_code = $exitCode
        Write-Output "$tag exit=$exitCode"
        Get-Content -LiteralPath $console -Tail 6
        if ($exitCode) { throw "Benchmark arm failed: $arm; remaining arms were not run." }
        $report = Get-Content -LiteralPath (Join-Path $out 'report.json') -Raw | ConvertFrom-Json
        if ($report.records.Count -ne 2 -or !$report.response_comparison.passed -or
            $report.response_comparison.compared_requests -ne 2) { throw "Exact two-request API qualification failed: $arm" }
        if ($GpuClockDiagnostics) {
            $gpu = $report.gpu_telemetry
            if ($report.timing_eligible -ne $false -or !$gpu.completed -or !$gpu.clock_power_observed -or
                !$gpu.crash_safe_job -or !$gpu.cleanup.completed -or $gpu.requests.Count -ne 2 -or
                @($gpu.requests | Where-Object { !$_.observations.valid -or !$_.observations.core_observed -or !$_.observations.sample_count }).Count) {
                throw "Bounded GPU clock observations did not qualify both actual request windows: $arm"
            }
            $receipt.arms[$arm].gpu_clock_observation = $gpu
        }
        $actualHash = (Get-FileHash -LiteralPath $logits -Algorithm SHA256).Hash
        if ((Get-Item -LiteralPath $logits).Length -ne 154880 * 4 -or $actualHash -ine $canonicalHash) {
            throw "Canonical full1872 logit vector differs: $arm"
        }
        if ($cap -ne '0') {
            $q = $report.strict_layer_major_graph_cap_qualification
            if (!$q.passed -or $q.requested_cap_bytes -ne [int64]$cap * 1048576 -or $q.completed_calls -lt 1 -or
                $q.failed_calls -or $q.malformed_records) { throw "No actual qualified strict cap$cap event: $arm" }
            $receipt.arms[$arm].strict_qualification = $q
        }
        if ($hostEmbedding -eq '1') {
            $q = $report.host_token_embedding_qualification
            if (!$q.passed -or $q.requests.Count -ne 2 -or @($q.requests | Where-Object { !$_.passed }).Count) {
                throw "Raw host embedding uploads did not cover both measured prompts: $arm"
            }
            $receipt.arms[$arm].host_embedding_qualification = $q
        }
        if ($downOverlap -eq '1') {
            $q = $report.prefill_down_overlap_qualification
            if (!$q.passed -or $q.requests.Count -ne 2 -or @($q.requests | Where-Object { !$_.passed }).Count) {
                throw "Completed Down overlap did not qualify both measured requests: $arm"
            }
            $receipt.arms[$arm].prefill_down_overlap_qualification = $q
        }
        if ($decodeGraphReserve -eq '1') {
            $q = $report.decode_graph_reserve_qualification
            if (!$q.passed -or $q.requests.Count -ne 2 -or @($q.requests | Where-Object { !$_.passed }).Count) {
                throw "Decode graph admission credit did not qualify both measured requests: $arm"
            }
            $receipt.arms[$arm].decode_graph_reserve_qualification = $q
        }
        if ($Experiment -in @('FileReadWorkers','FileRandomAccess')) {
            $q = $report.file_read_workers_qualification
            if (!$q.passed -or $q.requested_workers -ne [int]$fileReadWorkers -or $q.requests.Count -ne 2 -or @($q.requests | Where-Object { !$_.passed -or !$_.source_read_bytes }).Count) { throw "No qualified joined file-reader activity for both requests: $arm" }
            $receipt.arms[$arm].file_read_workers_qualification = $q
            if ($Experiment -eq 'FileRandomAccess' -and $q.requested_random_access_hint -ne [int]$fileRandomAccess) { throw "Applied random-access source hint did not qualify: $arm" }
        }
        if ($Experiment -eq 'FileReadSort') {
            $q = $report.file_read_sort_qualification
            if (!$q.passed -or $q.requested_sort -ne [int]$fileReadSort -or $q.requests.Count -ne 2 -or @($q.requests | Where-Object { !$_.passed -or !$_.phases.prefill.calls -or !$_.phases.decode.calls }).Count) { throw "File-sort request-scoped layer/phase coverage did not qualify: $arm" }
            if ($fileReadSort -eq '1' -and @($q.requests | Where-Object { !$_.phases.prefill.changed_activity_observed -or !$_.phases.decode.changed_activity_observed }).Count) { throw "Actual changed miss order did not qualify both phases of both requests: $arm" }
            $receipt.arms[$arm].file_read_sort_qualification = $q
        }
        if ($Experiment -in @('FileSourceView','FileSourceViewPrefetch')) {
            $q = $report.file_source_view_qualification
            if (!$q.passed -or $q.requested_view -ne [int]$fileSourceView -or $q.requests.Count -ne 2 -or @($q.requests | Where-Object { !$_.passed -or !$_.phases.prefill.calls -or !$_.phases.decode.calls }).Count) { throw "File-view joined source-byte and layer/phase coverage did not qualify: $arm" }
            if ($fileSourceView -eq '1' -and @($q.requests | Where-Object { !$_.phases.prefill.mapped_activity_observed -or !$_.phases.decode.mapped_activity_observed }).Count) { throw "Actual mapped source copies did not qualify both phases of both requests: $arm" }
            $receipt.arms[$arm].file_source_view_qualification = $q
            $prefetch = $report.file_source_view_prefetch_qualification
            if (!$prefetch.passed -or $prefetch.requested_view -ne [int]$fileSourceView -or $prefetch.requested_prefetch -ne [int]$fileSourcePrefetch -or $prefetch.requests.Count -ne 2 -or @($prefetch.requests | Where-Object { !$_.passed -or !$_.phases.prefill.calls -or !$_.phases.decode.calls }).Count) { throw "Joined demand-prefetch coverage did not qualify both measured requests: $arm" }
            if ($fileSourcePrefetch -eq '1' -and @($prefetch.requests | Where-Object { !$_.phases.prefill.prefetch_activity_observed -or !$_.phases.decode.prefetch_activity_observed }).Count) { throw "Actual exact-demand prefetch did not qualify both phases of both requests: $arm" }
            $receipt.arms[$arm].file_source_view_prefetch_qualification = $prefetch
        }
        if ($retainPeer -eq '1') {
            $q = $report.decode_retain_peer_qualification
            if (!$q.passed -or $q.requests.Count -ne 2 -or !$q.active_requests -or @($q.requests | Where-Object { !$_.valid -or $_.newly_granted_bytes -or ($_.passed -and (!$_.retained_hot_slots -or !$_.retained_payload_bytes)) }).Count) { throw "Published hot expert payload retention had no measured activity or invalid request evidence: $arm" }
            $receipt.arms[$arm].decode_retain_peer_qualification = $q
        }
        if ($vmmWeights -eq '1') {
            $q = $report.vmm_weights_qualification
            if (!$q.passed -or $q.requests.Count -ne 2 -or !$q.reshape_events -or @($q.requests | Where-Object { !$_.passed -or !$_.prefill_calls -or !$_.decode_calls -or $_.newly_granted_bytes }).Count) { throw "Actual VMM prefill/decode use and checked capacity redistribution did not qualify: $arm" }
            if ($RequireVmmPressure -and (!$q.pressure_required -or !$q.pressure_events -or @($q.requests | Where-Object { !$_.pressure_events }).Count)) { throw "Actual checked VMM pressure retirement did not qualify every measured request: $arm" }
            $receipt.arms[$arm].vmm_weights_qualification = $q
        }
        $receipt.arms[$arm].status = 'passed'
        $receipt.arms[$arm].canonical_logit_sha256 = $actualHash
        $receipt.arms[$arm].exact_api_requests = 2
        $receipt.arms[$arm].records = @($report.records | Select-Object name,prompt_tokens,prompt_s,eval_tokens,eval_s,wall_s)
        Save-Receipt
    }
    $receipt.status = 'passed'
} catch {
    $receipt.status = 'failed'; $receipt.error = $_.Exception.Message
    foreach ($arm in $armNames) { if ($receipt.arms[$arm].status -eq 'running') { $receipt.arms[$arm].status = 'failed' } }
    throw
} finally {
    $env:PATH = $originalPath
    $receipt.finished_utc = [DateTime]::UtcNow.ToString('o')
    Save-Receipt
}
