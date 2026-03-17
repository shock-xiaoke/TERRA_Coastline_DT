param(
    [string]$CondaEnv = "terra",
    [string]$OutDir = "."
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda command not found. Open an Anaconda/Miniconda shell first."
}

if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
}

$historyPath = Join-Path $OutDir "environment.history.yml"
$lockPath = Join-Path $OutDir "requirements.lock.local.txt"

Write-Host "[1/2] Exporting Conda history to $historyPath"
conda env export --name $CondaEnv --from-history | Out-File -Encoding utf8 $historyPath

Write-Host "[2/2] Exporting pip lockfile to $lockPath"
conda run -n $CondaEnv python -m pip freeze | Out-File -Encoding utf8 $lockPath

Write-Host "Done."
Write-Host "Generated:"
Write-Host " - $historyPath"
Write-Host " - $lockPath"
