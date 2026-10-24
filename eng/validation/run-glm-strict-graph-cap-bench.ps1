<#
.SYNOPSIS
Run the strict graph-cap experiment through the shared residency benchmark.
#>
[CmdletBinding()]
param(
    [ValidateSet('Candidate','Abba')][string]$Mode = 'Candidate',
    [ValidateSet(512,768)][int]$CapMiB = 768,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-fA-F]{64}$')][string]$ExpectedNativeSha256,
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')][string]$OutputPrefix,
    [switch]$ExclusiveGpuSlot,
    [string]$RepoRoot,
    [string]$EvidenceRoot,
    [string]$ServerDirectory,
    [string]$BaselineReport,
    [string]$Model,
    [string]$Python,
    [int]$Port,
    [string]$CudaRuntime
)
& (Join-Path $PSScriptRoot 'run-glm-unified-residency-bench.ps1') -Experiment GraphCap @PSBoundParameters
