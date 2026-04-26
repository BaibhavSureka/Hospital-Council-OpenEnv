param(
    [string]$StageDir = ".hf_space_stage"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$stagePath = Join-Path $root $StageDir

if (Test-Path -LiteralPath $stagePath) {
    Remove-Item -LiteralPath $stagePath -Recurse -Force
}

New-Item -ItemType Directory -Path $stagePath | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stagePath "hospital_council_env") | Out-Null

Copy-Item -LiteralPath (Join-Path $root "README.md") -Destination $stagePath
Copy-Item -LiteralPath (Join-Path $root "Blog.md") -Destination $stagePath
Copy-Item -LiteralPath (Join-Path $root "Dockerfile") -Destination $stagePath
Copy-Item -LiteralPath (Join-Path $root "run_openenv_demo.py") -Destination $stagePath
Copy-Item -LiteralPath (Join-Path $root ".gitignore") -Destination $stagePath

$envSource = Join-Path $root "hospital_council_env"
$envDest = Join-Path $stagePath "hospital_council_env"

$envKeep = @(
    "__init__.py",
    "augmentation.py",
    "client.py",
    "Dockerfile",
    "models.py",
    "openenv.yaml",
    "pyproject.toml",
    "README.md",
    "rubrics.py",
    "simulator.py",
    "uv.lock"
)

foreach ($name in $envKeep) {
    Copy-Item -LiteralPath (Join-Path $envSource $name) -Destination $envDest
}

Copy-Item -LiteralPath (Join-Path $envSource "server") -Destination $envDest -Recurse
Copy-Item -LiteralPath (Join-Path $envSource "training") -Destination $envDest -Recurse
Copy-Item -LiteralPath (Join-Path $root "docs") -Destination $stagePath -Recurse

$junk = @(
    (Join-Path $stagePath ".venv"),
    (Join-Path $stagePath "physionet.org"),
    (Join-Path $stagePath "artifacts"),
    (Join-Path $stagePath "hospital_council_env\build"),
    (Join-Path $stagePath "hospital_council_env\openenv_hospital_council_env.egg-info"),
    (Join-Path $stagePath "hospital_council_env\__pycache__"),
    (Join-Path $stagePath "hospital_council_env\server\__pycache__"),
    (Join-Path $stagePath "hospital_council_env\training\__pycache__")
)

foreach ($path in $junk) {
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
}

$fileCount = (Get-ChildItem -LiteralPath $stagePath -Recurse -File).Count
$totalBytes = (Get-ChildItem -LiteralPath $stagePath -Recurse -File | Measure-Object -Property Length -Sum).Sum
$sizeMb = [math]::Round(($totalBytes / 1MB), 2)

Write-Host "Prepared Hugging Face staging folder:"
Write-Host "  Path: $stagePath"
Write-Host "  Files: $fileCount"
Write-Host "  Size: $sizeMb MB"
