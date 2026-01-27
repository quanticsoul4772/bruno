# Verbosity Direction Spike Experiment (Windows)
#
# This script runs the complete spike experiment to test if a
# "verbosity direction" can be extracted from LLMs.
#
# Usage: .\run_spike.ps1 -Model "meta-llama/Llama-3.1-8B-Instruct"

param(
    [string]$Model = "meta-llama/Llama-3.1-8B-Instruct"
)

$ErrorActionPreference = "Stop"

$ModelName = Split-Path $Model -Leaf
$OutputDir = ".\${ModelName}-concise"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Verbosity Direction Spike Experiment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Model: $Model"
Write-Host "Output: $OutputDir"
Write-Host ""

# Step 0: Create datasets
Write-Host "[Step 0] Creating local datasets..." -ForegroundColor Yellow
python experiments/verbosity/load_local_dataset.py
Write-Host ""

# Step 1: Measure baseline verbosity
Write-Host "[Step 1] Measuring baseline verbosity..." -ForegroundColor Yellow
python experiments/verbosity/eval_verbosity.py `
    --model $Model `
    --num-prompts 30 `
    --output "experiments/verbosity/baseline_${ModelName}.json"
Write-Host ""

# Step 2: Run heretic with verbosity config
Write-Host "[Step 2] Extracting verbosity direction..." -ForegroundColor Yellow
Write-Host "This will run heretic with the verbosity config."
Write-Host "When finished, save the model to: $OutputDir"
Write-Host ""
Write-Host "Press Enter to start heretic, or Ctrl+C to abort."
Read-Host

# Copy config
Copy-Item "experiments/verbosity/config.verbosity.toml" "config.toml" -Force

# Run heretic with auto-select
heretic --model $Model --auto-select --auto-select-path $OutputDir

Write-Host ""
Write-Host "[Step 3] Measuring modified model verbosity..." -ForegroundColor Yellow
python experiments/verbosity/eval_verbosity.py `
    --original $Model `
    --modified $OutputDir `
    --num-prompts 30 `
    --output "experiments/verbosity/comparison_${ModelName}.json"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Experiment Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Results saved to:"
Write-Host "  - experiments/verbosity/baseline_${ModelName}.json"
Write-Host "  - experiments/verbosity/comparison_${ModelName}.json"
Write-Host ""
Write-Host "Modified model saved to: $OutputDir"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Review the comparison metrics"
Write-Host "  2. Chat with the modified model to test capability"
Write-Host "  3. Try composing with refusal direction"
