param(
    [string]$SourceDir   = "data/labelme_work",
    [string]$ArchivePath = "labelme_work.tar.gz"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $SourceDir)) {
    throw "Source directory not found: $SourceDir"
}

# ── Print manifest ────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Dataset contents:"
Write-Host ("-" * 52)

$totalJson = 0
Get-ChildItem -Directory $SourceDir | Sort-Object Name | ForEach-Object {
    $count = (Get-ChildItem -Recurse -Filter "*.json" $_.FullName).Count
    $totalJson += $count
    Write-Host ("  {0,-35} {1,4} JSON files" -f $_.Name, $count)
}

Write-Host ("-" * 52)
Write-Host ("  Total JSON files: {0}" -f $totalJson)
Write-Host ""

if ($totalJson -lt 300) {
    Write-Host "[WARN] Only $totalJson files found. Expected ~700+ (original + 2021-2026 corrections)."
    Write-Host "       Make sure data\labelme_work\ve_pred_2021_2026\ is present and reviewed."
    Write-Host ""
}

# ── Pack ──────────────────────────────────────────────────────────────────────
$sourceResolved = Resolve-Path $SourceDir
$sourceParent   = Split-Path $sourceResolved -Parent
$sourceLeaf     = Split-Path $sourceResolved -Leaf

Write-Host "Packing $sourceResolved  ->  $ArchivePath ..."
tar -czf $ArchivePath -C $sourceParent $sourceLeaf

$sizeMB = [math]::Round((Get-Item $ArchivePath).Length / 1MB, 1)
Write-Host "Done.  Archive size: ${sizeMB} MB"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Push code to GitHub:"
Write-Host "       cd TERRA_UGLA_serverrepo && git add -A && git commit -m 'Add v2 training scripts' && git push"
Write-Host ""
Write-Host "  2. Transfer archive to server:"
Write-Host "       scp $ArchivePath <user>@<server>:/path/to/TERRA_Coastline_DT/"
Write-Host ""
Write-Host "  3. On the server:"
Write-Host "       git pull"
Write-Host "       bash scripts/setup_training_data.sh labelme_work.tar.gz"
Write-Host "       bash scripts/retrain_ve_v2.sh"
