# Quick Start Script for Qwen2.5-Coder-32B Abliteration
# Run this tomorrow morning to get started quickly!
#
# USAGE:
#   .\start-tomorrow.ps1           # Full guided setup
#   .\start-tomorrow.ps1 -Step 1   # Run specific step only
#
# ESTIMATED TIMES:
#   Step 1: Create instance     ~2 min
#   Step 2: Setup heretic       ~3 min  
#   Step 3: Start abliteration  ~5 min (then runs for ~5-6 hours)
#   TOTAL: ~10 minutes to get running
#
# REQUIREMENTS:
#   - Vast.ai API key in .env file (VAST_API_KEY=xxx)
#   - SSH key at ~/.ssh/id_ed25519

param(
    [int]$Step = 0  # 0 = run all steps, 1-4 = run specific step
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     HERETIC ABLITERATION - QUICK START                       ║" -ForegroundColor Cyan
Write-Host "║     Model: Qwen/Qwen2.5-Coder-32B-Instruct                   ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Configuration
$MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
$GPU_TIER = "RTX_4090"
$NUM_GPUS = 4  # 32B model needs ~65GB VRAM, 4x RTX 4090 = 96GB

function Step1-CreateInstance {
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Yellow
    Write-Host "  STEP 1: Creating Vast.ai Instance (4x RTX 4090)" -ForegroundColor Yellow
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  GPU: 4x RTX 4090 (96GB total VRAM)" -ForegroundColor White
    Write-Host "  Cost: ~$1.40-1.60/hr" -ForegroundColor White
    Write-Host "  Model: $MODEL" -ForegroundColor White
    Write-Host ""
    
    # Check for API key
    $envFile = Join-Path $PSScriptRoot ".env"
    if (-not (Test-Path $envFile)) {
        Write-Host "ERROR: .env file not found!" -ForegroundColor Red
        Write-Host "Create .env with: VAST_API_KEY=your_key_here" -ForegroundColor Yellow
        exit 1
    }
    
    # Create instance with 4 GPUs
    Write-Host "Creating instance..." -ForegroundColor Cyan
    & .\runpod.ps1 vast-create-pod $GPU_TIER $NUM_GPUS
    
    Write-Host ""
    Write-Host "✓ Instance created!" -ForegroundColor Green
    Write-Host ""
}

function Step2-SetupHeretic {
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Yellow
    Write-Host "  STEP 2: Installing Heretic" -ForegroundColor Yellow
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Yellow
    Write-Host ""
    
    & .\runpod.ps1 vast-setup
    
    Write-Host ""
    Write-Host "✓ Heretic installed!" -ForegroundColor Green
    Write-Host ""
}

function Step3-StartAbliteration {
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Yellow
    Write-Host "  STEP 3: Starting Abliteration with Persistent Storage" -ForegroundColor Yellow
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Model: $MODEL" -ForegroundColor White
    Write-Host "  Trials: 100" -ForegroundColor White
    Write-Host "  Storage: SQLite (resumes if interrupted)" -ForegroundColor White
    Write-Host "  Estimated time: 5-6 hours" -ForegroundColor White
    Write-Host ""
    
    # Upload the optimized config
    Write-Host "Uploading optimized config..." -ForegroundColor Cyan
    & .\runpod.ps1 vast-exec 'cat > /workspace/config.toml << EOF
# Optimized config for Qwen2.5-Coder-32B on 4x RTX 4090
n_trials = 100
batch_size = 8
max_batch_size = 8
max_response_length = 150
prune_trials = true
n_startup_trials = 25

[good_prompts]
dataset = "mlabonne/harmless_alpaca"
split = "train[:500]"
column = "text"

[bad_prompts]
dataset = "mlabonne/harmful_behaviors"
split = "train[:300]"
column = "text"

[good_evaluation_prompts]
dataset = "mlabonne/harmless_alpaca"
split = "train[500:700]"
column = "text"

[bad_evaluation_prompts]
dataset = "mlabonne/harmful_behaviors"
split = "train[300:450]"
column = "text"
EOF'
    
    Write-Host "Starting abliteration in background..." -ForegroundColor Cyan
    
    # Start with persistent storage so it can resume
    & .\runpod.ps1 vast-exec "export HF_HOME=/workspace/.cache/huggingface && cd /workspace && nohup heretic --model $MODEL --auto-select true --auto-select-path /workspace/models --storage sqlite:////workspace/heretic_study.db --study-name qwen32b-abliteration > /workspace/heretic.log 2>&1 & echo 'Started PID:' `$!"
    
    Start-Sleep -Seconds 3
    
    Write-Host ""
    Write-Host "✓ Abliteration started!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Monitor progress with:" -ForegroundColor Yellow
    Write-Host "  .\runpod.ps1 vast-watch" -ForegroundColor White
    Write-Host ""
    Write-Host "Or view live logs:" -ForegroundColor Yellow
    Write-Host "  ssh -i ~/.ssh/id_ed25519 -p PORT root@HOST 'tail -f /workspace/heretic.log'" -ForegroundColor White
    Write-Host ""
}

function Step4-Monitor {
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Yellow
    Write-Host "  STEP 4: Live Monitoring Dashboard" -ForegroundColor Yellow
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Yellow
    Write-Host ""
    
    & .\runpod.ps1 vast-watch
}

# Main execution
switch ($Step) {
    0 {
        # Run all steps with confirmations
        Write-Host "This script will:" -ForegroundColor White
        Write-Host "  1. Create a Vast.ai instance with 4x RTX 4090 (~$1.50/hr)" -ForegroundColor White
        Write-Host "  2. Install heretic" -ForegroundColor White
        Write-Host "  3. Start the abliteration (runs ~5-6 hours)" -ForegroundColor White
        Write-Host "  4. Open live monitoring dashboard" -ForegroundColor White
        Write-Host ""
        
        $confirm = Read-Host "Continue? (y/n)"
        if ($confirm -ne "y" -and $confirm -ne "Y") {
            Write-Host "Cancelled." -ForegroundColor Yellow
            exit 0
        }
        
        Step1-CreateInstance
        Step2-SetupHeretic
        Step3-StartAbliteration
        
        Write-Host ""
        Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
        Write-Host "║     ABLITERATION RUNNING!                                    ║" -ForegroundColor Green
        Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
        Write-Host ""
        Write-Host "Estimated completion: ~5-6 hours" -ForegroundColor Cyan
        Write-Host "Estimated cost: ~$8-10" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Commands:" -ForegroundColor Yellow
        Write-Host "  .\runpod.ps1 vast-watch           # Live dashboard" -ForegroundColor White
        Write-Host "  .\runpod.ps1 vast-progress        # Quick status check" -ForegroundColor White
        Write-Host "  .\runpod.ps1 vast-download-model  # Download when complete" -ForegroundColor White
        Write-Host "  .\runpod.ps1 vast-stop            # Stop instance (save money)" -ForegroundColor White
        Write-Host ""
        Write-Host "AFTER ABLITERATION COMPLETES:" -ForegroundColor Yellow
        Write-Host "  .\scripts\test-on-cloud.ps1       # TEST ON CLOUD FIRST (no download needed!)" -ForegroundColor White
        Write-Host "  .\scripts\download-model.ps1      # Download model (resumable, 60GB)" -ForegroundColor White
        Write-Host "  .\runpod.ps1 vast-stop            # Stop instance when done" -ForegroundColor White
        Write-Host ""
        
        $monitor = Read-Host "Open live dashboard now? (y/n)"
        if ($monitor -eq "y" -or $monitor -eq "Y") {
            Step4-Monitor
        }
    }
    1 { Step1-CreateInstance }
    2 { Step2-SetupHeretic }
    3 { Step3-StartAbliteration }
    4 { Step4-Monitor }
    default {
        Write-Host "Invalid step. Use 0-4." -ForegroundColor Red
        Write-Host "  0 = Run all steps" -ForegroundColor White
        Write-Host "  1 = Create instance" -ForegroundColor White
        Write-Host "  2 = Setup heretic" -ForegroundColor White
        Write-Host "  3 = Start abliteration" -ForegroundColor White
        Write-Host "  4 = Monitor" -ForegroundColor White
    }
}
