# Quick Start Script for Qwen2.5-Coder-32B Abliteration
#
# USAGE:
#   .\start-abliteration.ps1           # Full guided setup
#   .\start-abliteration.ps1 -Step 1   # Run specific step only
#
# STEPS:
#   1 = Create instance
#   2 = Setup bruno
#   3 = Start abliteration
#   4 = Monitor

param(
    [int]$Step = 0
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  HERETIC ABLITERATION - QUICK START" -ForegroundColor Cyan
Write-Host "  Model: Qwen/Qwen2.5-Coder-32B-Instruct" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

function Step1-CreateInstance {
    Write-Host "--- STEP 1: Creating Vast.ai Instance ---" -ForegroundColor Yellow
    Write-Host "  GPU: 4x RTX 4090 - 96GB total VRAM" -ForegroundColor White
    Write-Host "  Cost: about 1.50 USD/hr" -ForegroundColor White
    Write-Host ""

    $envFile = Join-Path $PSScriptRoot ".env"
    if (-not (Test-Path $envFile)) {
        Write-Host "ERROR: .env file not found!" -ForegroundColor Red
        exit 1
    }

    Write-Host "Creating instance..." -ForegroundColor Cyan
    & .\runpod.ps1 vast-create-pod RTX_4090 4

    Write-Host "Done!" -ForegroundColor Green
}

function Step2-SetupHeretic {
    Write-Host "--- STEP 2: Installing Heretic ---" -ForegroundColor Yellow
    Write-Host ""

    & .\runpod.ps1 vast-setup

    Write-Host "Done!" -ForegroundColor Green
}

function Step3-StartAbliteration {
    Write-Host "--- STEP 3: Starting Abliteration ---" -ForegroundColor Yellow
    Write-Host "  Model: Qwen/Qwen2.5-Coder-32B-Instruct" -ForegroundColor White
    Write-Host "  Trials: 100" -ForegroundColor White
    Write-Host "  Storage: SQLite - resumes if interrupted" -ForegroundColor White
    Write-Host "  Estimated time: 5-6 hours" -ForegroundColor White
    Write-Host ""

    Write-Host "Starting abliteration in background..." -ForegroundColor Cyan

    # Use single quotes to prevent PowerShell from interpreting special characters
    & .\runpod.ps1 vast-exec 'export HF_HOME=/workspace/.cache/huggingface; cd /workspace; nohup bruno --model Qwen/Qwen2.5-Coder-32B-Instruct --auto-select true --auto-select-path /workspace/models --storage sqlite:////workspace/bruno_study.db --study-name qwen32b-abliteration > /workspace/bruno.log 2>&1 &'

    Start-Sleep -Seconds 3

    Write-Host ""
    Write-Host "Abliteration started!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Monitor with: .\runpod.ps1 vast-watch" -ForegroundColor Yellow
}

function Step4-Monitor {
    Write-Host "--- STEP 4: Live Monitoring ---" -ForegroundColor Yellow
    & .\runpod.ps1 vast-watch
}

switch ($Step) {
    0 {
        Write-Host "This script will:" -ForegroundColor White
        Write-Host "  1. Create a Vast.ai instance with 4x RTX 4090" -ForegroundColor White
        Write-Host "  2. Install bruno" -ForegroundColor White
        Write-Host "  3. Start the abliteration - runs 5-6 hours" -ForegroundColor White
        Write-Host ""

        $confirm = Read-Host "Continue? (y/n)"
        if ($confirm -ne "y") {
            Write-Host "Cancelled." -ForegroundColor Yellow
            exit 0
        }

        Step1-CreateInstance
        Step2-SetupHeretic
        Step3-StartAbliteration

        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "  ABLITERATION RUNNING!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Commands:" -ForegroundColor Yellow
        Write-Host "  .\runpod.ps1 vast-watch    - Live dashboard" -ForegroundColor White
        Write-Host "  .\runpod.ps1 vast-progress - Quick status" -ForegroundColor White
        Write-Host "  .\runpod.ps1 vast-stop     - Stop instance" -ForegroundColor White
        Write-Host ""

        $monitor = Read-Host "Open dashboard now? (y/n)"
        if ($monitor -eq "y") {
            Step4-Monitor
        }
    }
    1 { Step1-CreateInstance }
    2 { Step2-SetupHeretic }
    3 { Step3-StartAbliteration }
    4 { Step4-Monitor }
    default {
        Write-Host "Invalid step. Use 0-4." -ForegroundColor Red
    }
}
