param(
    [string]$StartDate = "2016-01-01",
    [string]$EndDate = "2026-03-10",
    [int]$MaxCloudPct = 30,
    [int]$MaxImagesPerMonth = 1,
    [switch]$BuildChips,
    [int]$ChipSize = 512,
    [int]$Stride = 512,
    [switch]$OnlyTrainAois,
    [switch]$SkipExisting = $true,
    [switch]$DryRun,
    [string]$PythonExe = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-Python {
    if ($PythonExe) {
        if (Test-Path $PythonExe) {
            return @{
                Exe = $PythonExe
                BaseArgs = @()
            }
        }
        $customCmd = Get-Command $PythonExe -ErrorAction SilentlyContinue
        if ($customCmd) {
            return @{
                Exe = $customCmd.Source
                BaseArgs = @()
            }
        }
        throw "PythonExe not found: $PythonExe"
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        return @{
            Exe = $pythonCmd.Source
            BaseArgs = @()
        }
    }

    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd) {
        return @{
            Exe = $pyCmd.Source
            BaseArgs = @("-3")
        }
    }

    throw "Python launcher not found. Install python (or py) and rerun."
}

function Get-MonthWindows {
    param(
        [datetime]$GlobalStart,
        [datetime]$GlobalEnd
    )

    $cursor = Get-Date -Year $GlobalStart.Year -Month $GlobalStart.Month -Day 1
    $windows = @()

    while ($cursor -le $GlobalEnd) {
        $monthStart = $cursor
        if ($monthStart -lt $GlobalStart) {
            $monthStart = $GlobalStart
        }

        $monthEnd = $cursor.AddMonths(1).AddDays(-1)
        if ($monthEnd -gt $GlobalEnd) {
            $monthEnd = $GlobalEnd
        }

        $windows += [PSCustomObject]@{
            Start = $monthStart
            End = $monthEnd
        }
        $cursor = $cursor.AddMonths(1)
    }

    return $windows
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

[datetime]$globalStart = [datetime]::Parse($StartDate)
[datetime]$globalEnd = [datetime]::Parse($EndDate)
if ($globalEnd -lt $globalStart) {
    throw "EndDate must be on or after StartDate."
}
if ($MaxImagesPerMonth -lt 1) {
    throw "MaxImagesPerMonth must be >= 1."
}

$aois = @(
    # @{ Id = "aoi_01"; Role = "train_pool"; BBox = @(-2.888374, 56.352126, -2.826576, 56.36563) },
    @{ Id = "aoi_02"; Role = "train_pool"; BBox = @(-2.769928, 56.277483, -2.584534, 56.337476) }
    # @{ Id = "aoi_03"; Role = "train_pool"; BBox = @(-2.828465, 56.378749, -2.792244, 56.44143) },
    # @{ Id = "aoi_04"; Role = "train_pool"; BBox = @(-2.884769, 56.365298, -2.82546, 56.378132) },
    # @{ Id = "aoi_05"; Role = "train_pool"; BBox = @(-4.633656, 55.474962, -4.615459, 55.518719) },
    # @{ Id = "aoi_06"; Role = "train_pool"; BBox = @(-4.740257, 55.421123, -4.65271, 55.443524) },
    # @{ Id = "aoi_07"; Role = "train_pool"; BBox = @(-4.677858, 55.925596, -4.591599, 55.936462) },
    # @{ Id = "aoi_08"; Role = "train_pool"; BBox = @(-4.887714, 55.900101, -4.871235, 55.941942) },
    # @{ Id = "aoi_09"; Role = "train_pool"; BBox = @(-3.661022, 56.008555, -3.557167, 56.022948) },
    # @{ Id = "aoi_10_holdout"; Role = "holdout_candidate"; BBox = @(-2.674742, 56.282295, -2.595005, 56.323772) }
)

if ($OnlyTrainAois) {
    $aois = $aois | Where-Object { $_.Role -eq "train_pool" }
}

$python = Resolve-Python
$depCheckArgs = @()
$depCheckArgs += $python.BaseArgs
$depCheckArgs += @(
    "-c",
    "import sys; import numpy, requests, PIL, rasterio; print(sys.executable)"
)

if (-not $DryRun) {
    & $python.Exe @depCheckArgs
    if ($LASTEXITCODE -ne 0) {
        throw @"
Python dependency check failed (requires numpy, requests, pillow, rasterio).
Please install dependencies first, e.g.:
  pip install -r requirements_fixed.txt
Or specify a ready interpreter with:
  -PythonExe C:\path\to\python.exe
"@
    }
}

$monthlyWindows = Get-MonthWindows -GlobalStart $globalStart -GlobalEnd $globalEnd

$jobs = @()
foreach ($aoi in $aois) {
    foreach ($win in $monthlyWindows) {
        $runId = "{0}_{1}" -f $aoi.Id, $win.Start.ToString("yyyyMM")
        $manifestPath = Join-Path $projectRoot ("data\runs\{0}\manifest.json" -f $runId)

        $jobs += [PSCustomObject]@{
            AoiId = $aoi.Id
            Role = $aoi.Role
            BBox = $aoi.BBox
            Start = $win.Start
            End = $win.End
            RunId = $runId
            ManifestPath = $manifestPath
        }
    }
}

Write-Host ("ProjectRoot: {0}" -f $projectRoot)
Write-Host ("AOIs: {0}, MonthWindows: {1}, TotalJobs: {2}" -f $aois.Count, $monthlyWindows.Count, $jobs.Count)
Write-Host ("Date range: {0} -> {1}" -f $globalStart.ToString("yyyy-MM-dd"), $globalEnd.ToString("yyyy-MM-dd"))

$results = @()
$failures = @()
$warnings = @()

foreach ($job in $jobs) {
    if ($SkipExisting -and (Test-Path $job.ManifestPath)) {
        Write-Host ("[SKIP] {0} {1} (manifest exists)" -f $job.AoiId, $job.Start.ToString("yyyy-MM"))
        $results += [PSCustomObject]@{
            aoi_id = $job.AoiId
            role = $job.Role
            month = $job.Start.ToString("yyyy-MM")
            run_id = $job.RunId
            status = "skipped"
            note = "manifest_exists"
        }
        continue
    }

    $bbox = $job.BBox
    $cmdArgs = @()
    $cmdArgs += $python.BaseArgs
    $cmdArgs += @(
        "scripts/pull_sentinelhub_dataset.py",
        "--aoi-id", $job.AoiId,
        "--bbox", [string]$bbox[0], [string]$bbox[1], [string]$bbox[2], [string]$bbox[3],
        "--start-date", $job.Start.ToString("yyyy-MM-dd"),
        "--end-date", $job.End.ToString("yyyy-MM-dd"),
        "--max-cloud-pct", [string]$MaxCloudPct,
        "--max-images", [string]$MaxImagesPerMonth,
        "--run-id", $job.RunId
    )

    if ($BuildChips) {
        $cmdArgs += @("--build-chips", "--chip-size", [string]$ChipSize, "--stride", [string]$Stride)
    }

    Write-Host ("[RUN ] {0} {1} role={2}" -f $job.AoiId, $job.Start.ToString("yyyy-MM"), $job.Role)
    if ($DryRun) {
        Write-Host ("       {0} {1}" -f $python.Exe, ($cmdArgs -join " "))
        $results += [PSCustomObject]@{
            aoi_id = $job.AoiId
            role = $job.Role
            month = $job.Start.ToString("yyyy-MM")
            run_id = $job.RunId
            status = "dry_run"
            note = ""
        }
        continue
    }

    & $python.Exe @cmdArgs
    if ($LASTEXITCODE -ne 0) {
        $message = "python exit code $LASTEXITCODE"
        Write-Warning ("[FAIL] {0} {1}: {2}" -f $job.AoiId, $job.Start.ToString("yyyy-MM"), $message)
        $failures += [PSCustomObject]@{
            aoi_id = $job.AoiId
            role = $job.Role
            month = $job.Start.ToString("yyyy-MM")
            run_id = $job.RunId
            error = $message
        }
        $results += [PSCustomObject]@{
            aoi_id = $job.AoiId
            role = $job.Role
            month = $job.Start.ToString("yyyy-MM")
            run_id = $job.RunId
            status = "failed"
            note = $message
        }
        continue
    }

    $note = ""
    if (Test-Path $job.ManifestPath) {
        $manifest = Get-Content $job.ManifestPath | ConvertFrom-Json
        $sceneObjs = @()
        if ($null -ne $manifest -and $manifest.PSObject.Properties.Name -contains "scenes" -and $null -ne $manifest.scenes) {
            $sceneObjs = @($manifest.scenes)
        }
        $demoScenes = @($sceneObjs | Where-Object { $null -ne $_ -and $null -ne $_.scene -and $_.scene.collection -eq "demo" })
        if ($demoScenes.Count -gt 0) {
            $note = "demo_fallback"
            $warnings += [PSCustomObject]@{
                aoi_id = $job.AoiId
                role = $job.Role
                month = $job.Start.ToString("yyyy-MM")
                run_id = $job.RunId
                warning = "demo_fallback"
            }
            Write-Warning ("[WARN] {0} {1}: demo_fallback" -f $job.AoiId, $job.Start.ToString("yyyy-MM"))
        }
    }

    $results += [PSCustomObject]@{
        aoi_id = $job.AoiId
        role = $job.Role
        month = $job.Start.ToString("yyyy-MM")
        run_id = $job.RunId
        status = "ok"
        note = $note
    }
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$reportDir = Join-Path $projectRoot "data\runs\batch_reports"
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null

$resultsPath = Join-Path $reportDir ("batch_monthly_results_{0}.csv" -f $timestamp)
$results | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $resultsPath

if ($failures.Count -gt 0) {
    $failurePath = Join-Path $reportDir ("batch_monthly_failures_{0}.csv" -f $timestamp)
    $failures | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $failurePath
}
if ($warnings.Count -gt 0) {
    $warningPath = Join-Path $reportDir ("batch_monthly_warnings_{0}.csv" -f $timestamp)
    $warnings | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $warningPath
}

$okCount = @($results | Where-Object { $_.status -eq "ok" }).Count
$skipCount = @($results | Where-Object { $_.status -eq "skipped" }).Count
$failCount = @($results | Where-Object { $_.status -eq "failed" }).Count
$dryRunCount = @($results | Where-Object { $_.status -eq "dry_run" }).Count

Write-Host ""
Write-Host "Done."
Write-Host ("Results CSV: {0}" -f $resultsPath)
Write-Host ("ok={0}, skipped={1}, failed={2}, dry_run={3}, warnings={4}" -f $okCount, $skipCount, $failCount, $dryRunCount, $warnings.Count)

if ($failCount -gt 0) {
    throw "Batch finished with failures. Check batch_reports CSV files."
}
