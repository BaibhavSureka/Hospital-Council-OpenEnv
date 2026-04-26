param(
    [string]$RepoId = "BAIBHAV1234/hospital-council-openenv",
    [string]$StageDir = ".hf_space_stage",
    [string]$CommitMessage = "Clean Space deploy"
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $scriptRoot
$stagePath = Join-Path $root $StageDir

if (-not (Test-Path -LiteralPath $stagePath)) {
    & (Join-Path $scriptRoot "prepare_hf_space.ps1") -StageDir $StageDir
}

hf upload $RepoId $stagePath . --repo-type space --commit-message $CommitMessage
