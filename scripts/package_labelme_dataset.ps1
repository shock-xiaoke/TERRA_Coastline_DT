param(
    [string]$SourceDir = "data/labelme_work",
    [string]$ArchivePath = "labelme_work.tar.gz"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $SourceDir)) {
    throw "Source directory not found: $SourceDir"
}

$sourceResolved = Resolve-Path $SourceDir
$sourceParent = Split-Path $sourceResolved -Parent
$sourceLeaf = Split-Path $sourceResolved -Leaf

Write-Host "Packing $sourceResolved -> $ArchivePath"
tar -czf $ArchivePath -C $sourceParent $sourceLeaf

Write-Host "Done. Transfer example:"
Write-Host "scp $ArchivePath <user>@<server>:/data/terra/"
