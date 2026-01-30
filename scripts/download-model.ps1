# Download Model from Vast.ai with Resume Support
# Uses rsync via WSL for resumable, fast downloads
#
# USAGE:
#   .\scripts\download-model.ps1
#   .\scripts\download-model.ps1 -ModelName "Qwen-Qwen2.5-Coder-32B-Instruct-heretic"

param(
    [string]$ModelName = "",
    [string]$LocalDir = ".\models"
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     Download Model from Vast.ai (Resumable)                  ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Load .env file for Vast.ai API key
$envFile = Join-Path $PSScriptRoot "..\.env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^([^#=]+)=(.*)$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            if ($value -and $value -ne 'your_api_key_here') {
                [Environment]::SetEnvironmentVariable($name, $value, 'Process')
            }
        }
    }
}

# Check WSL is available
$wslCheck = wsl --status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: WSL is required for rsync downloads" -ForegroundColor Red
    Write-Host "Install with: wsl --install" -ForegroundColor Yellow
    exit 1
}

# Get instance info
Write-Host "Finding Vast.ai instance..." -ForegroundColor Gray
$instanceInfo = wsl -e bash -c "vastai show instances --raw 2>/dev/null" | ConvertFrom-Json

if (-not $instanceInfo -or $instanceInfo.Count -eq 0) {
    Write-Host "ERROR: No Vast.ai instances found" -ForegroundColor Red
    Write-Host "Create one with: .\runpod.ps1 vast-create-pod" -ForegroundColor Yellow
    exit 1
}

$instance = $instanceInfo | Where-Object { $_.actual_status -eq "running" } | Select-Object -First 1
if (-not $instance) {
    Write-Host "ERROR: No running Vast.ai instances" -ForegroundColor Red
    exit 1
}

$instanceId = $instance.id
Write-Host "Instance: $instanceId" -ForegroundColor Green

# Get SSH URL
$sshUrl = wsl -e bash -c "vastai ssh-url $instanceId 2>/dev/null"
if ($sshUrl -match '([^@]+)@([^:]+):(\d+)') {
    $sshUser = $matches[1]
    $sshHost = $matches[2]
    $sshPort = $matches[3]
} elseif ($sshUrl -match '-p\s+(\d+)\s+([^@]+)@(\S+)') {
    $sshPort = $matches[1]
    $sshUser = $matches[2]
    $sshHost = $matches[3]
} else {
    Write-Host "ERROR: Could not parse SSH URL: $sshUrl" -ForegroundColor Red
    exit 1
}

Write-Host "SSH: $sshUser@$sshHost:$sshPort" -ForegroundColor Gray
Write-Host ""

# List available models if none specified
if (-not $ModelName) {
    Write-Host "Scanning for models on remote..." -ForegroundColor Yellow
    $models = wsl -e bash -c "ssh -o StrictHostKeyChecking=no -p $sshPort $sshUser@$sshHost 'ls -1 /workspace/models/ 2>/dev/null'"

    if (-not $models -or $models -match "No such file") {
        Write-Host "ERROR: No models found in /workspace/models/" -ForegroundColor Red
        exit 1
    }

    $modelList = $models -split "`n" | Where-Object { $_.Trim() }

    if ($modelList.Count -eq 0) {
        Write-Host "ERROR: No models found" -ForegroundColor Red
        exit 1
    }

    Write-Host ""
    Write-Host "Available models:" -ForegroundColor Cyan
    $i = 1
    foreach ($m in $modelList) {
        Write-Host "  [$i] $m" -ForegroundColor White
        $i++
    }
    Write-Host ""
    $selection = Read-Host "Select model number"

    if ($selection -match '^\d+$' -and [int]$selection -le $modelList.Count -and [int]$selection -gt 0) {
        $ModelName = $modelList[[int]$selection - 1].Trim()
    } else {
        Write-Host "Invalid selection" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "Model: $ModelName" -ForegroundColor Yellow
Write-Host ""

# Get model size
Write-Host "Checking model size..." -ForegroundColor Gray
$sizeOutput = wsl -e bash -c "ssh -o StrictHostKeyChecking=no -p $sshPort $sshUser@$sshHost 'du -sh /workspace/models/$ModelName 2>/dev/null'"
$modelSize = ($sizeOutput -split '\s+')[0]
Write-Host "Size: $modelSize" -ForegroundColor Cyan
Write-Host ""

# Create local directory
if (-not (Test-Path $LocalDir)) {
    New-Item -ItemType Directory -Path $LocalDir -Force | Out-Null
}

# Convert Windows path to WSL path
$wslLocalDir = ($LocalDir -replace '^\.', (Get-Location).Path) -replace '^([A-Za-z]):', '/mnt/$1' -replace '\\', '/'
$wslLocalDir = $wslLocalDir.ToLower()

Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Yellow
Write-Host "  Starting Download (rsync with resume)" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Yellow
Write-Host ""
Write-Host "Source: /workspace/models/$ModelName" -ForegroundColor Gray
Write-Host "Dest:   $LocalDir\$ModelName" -ForegroundColor Gray
Write-Host ""
Write-Host "If interrupted, run this script again to RESUME." -ForegroundColor Green
Write-Host ""

$startTime = Get-Date

# Ensure WSL has SSH key with correct permissions
wsl -e bash -c "mkdir -p ~/.ssh && cp /mnt/c/Users/$env:USERNAME/.ssh/id_ed25519 ~/.ssh/ 2>/dev/null && chmod 600 ~/.ssh/id_ed25519"

# Use rsync for resumable download
# --partial: Keep partially transferred files
# --progress: Show progress
# --compress: Compress during transfer (helps on slow connections)
# -avz: Archive mode, verbose, compress
$rsyncCmd = "rsync -avz --partial --progress -e 'ssh -o StrictHostKeyChecking=no -p $sshPort' $sshUser@${sshHost}:/workspace/models/$ModelName $wslLocalDir/"

Write-Host "Running: rsync ..." -ForegroundColor Gray
Write-Host ""

wsl -e bash -c $rsyncCmd

$endTime = Get-Date
$duration = $endTime - $startTime

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host "  DOWNLOAD COMPLETE" -ForegroundColor Green
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host ""
    Write-Host "Duration: $([math]::Round($duration.TotalMinutes, 1)) minutes" -ForegroundColor Cyan
    Write-Host "Location: $LocalDir\$ModelName" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Convert to GGUF: .\scripts\convert-to-gguf.ps1 -ModelPath '$LocalDir\$ModelName'" -ForegroundColor White
    Write-Host "  2. Compare models:  .\scripts\compare-models.ps1" -ForegroundColor White
    Write-Host "  3. Stop instance:   .\runpod.ps1 vast-stop" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "Download incomplete or failed (exit code: $LASTEXITCODE)" -ForegroundColor Yellow
    Write-Host "Run this script again to RESUME the download." -ForegroundColor Cyan
}
