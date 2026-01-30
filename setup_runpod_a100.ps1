# Setup RunPod A100 80GB for 32B Model Abliteration
# Run this script in PowerShell after pod creation

param(
    [string]$PodSSH = "nb97jc5nabyama-64410cb0@ssh.runpod.io"
)

Write-Host "Setting up RunPod A100 80GB for 32B model abliteration..." -ForegroundColor Cyan
Write-Host "Pod: $PodSSH`n" -ForegroundColor Gray

# Phase 1: Install dependencies
Write-Host "[1/5] Installing git and heretic..." -ForegroundColor Yellow
$commands = @"
apt-get update -qq && apt-get install -y -qq git &&
pip install -q git+https://github.com/quanticsoul4772/heretic.git &&
echo 'INSTALL_COMPLETE'
"@

$result = ssh -T $PodSSH $commands 2>&1
if ($result -match "INSTALL_COMPLETE") {
    Write-Host "  ✓ Git and heretic installed" -ForegroundColor Green
} else {
    Write-Host "  ✗ Installation failed" -ForegroundColor Red
    Write-Host $result
    exit 1
}

# Phase 2: Create directories and config
Write-Host "`n[2/5] Creating directories and config..." -ForegroundColor Yellow
$configCommands = @"
mkdir -p /workspace/models /workspace/.cache/huggingface &&
export HF_HOME=/workspace/.cache/huggingface &&
echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc &&
cat > /workspace/config.toml << 'CONFIGEOF'
model = \"Qwen/Qwen2.5-Coder-32B-Instruct\"

# Single A100 80GB - all optimizations enabled
device_map = \"auto\"
cache_weights = true
compile = true
iterative_rounds = 1
use_pca_extraction = false

# Aggressive batching
batch_size = 0
max_batch_size = 64
refusal_check_tokens = 20

# Optuna
n_trials = 200
n_startup_trials = 30
storage = \"sqlite:////workspace/heretic_32b.db\"
study_name = \"qwen32b_a100\"

# Auto-save
auto_select = true
auto_select_path = \"/workspace/models/Qwen2.5-Coder-32B-Instruct-heretic\"
CONFIGEOF
cat /workspace/config.toml &&
echo 'CONFIG_COMPLETE'
"@

$result = ssh -T $PodSSH $configCommands 2>&1
if ($result -match "CONFIG_COMPLETE") {
    Write-Host "  ✓ Config created" -ForegroundColor Green
    Write-Host ($result | Select-String -Pattern "model =")
} else {
    Write-Host "  ✗ Config creation failed" -ForegroundColor Red
    exit 1
}

# Phase 3: Verify GPU
Write-Host "`n[3/5] Verifying GPU..." -ForegroundColor Yellow
$gpuCheck = ssh -T $PodSSH "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader" 2>&1
Write-Host "  GPU: $gpuCheck" -ForegroundColor Cyan

# Phase 4: Test run with 1 trial
Write-Host "`n[4/5] Running test (1 trial to validate config)..." -ForegroundColor Yellow
Write-Host "  This will take ~15-20 minutes..." -ForegroundColor Gray

$testCommands = @"
cd /workspace &&
sed -i 's/n_trials = 200/n_trials = 1/' config.toml &&
timeout 1800 heretic 2>&1 | tee heretic_test.log &&
echo 'TEST_COMPLETE'
"@

$testResult = ssh -T $PodSSH $testCommands 2>&1

if ($testResult -match "TEST_COMPLETE") {
    Write-Host "  ✓ Test trial completed successfully!" -ForegroundColor Green

    # Check for errors
    $errors = $testResult | Select-String -Pattern "OutOfMemory|RuntimeError.*meta|Traceback"
    if ($errors) {
        Write-Host "  ⚠ Warning: Errors detected in test run" -ForegroundColor Yellow
        Write-Host $errors
    } else {
        Write-Host "  ✓ No errors detected" -ForegroundColor Green
    }
} else {
    Write-Host "  ✗ Test run failed or timed out" -ForegroundColor Red
    Write-Host "  Check: ssh -T $PodSSH 'tail -100 /workspace/heretic_test.log'"
    exit 1
}

# Phase 5: Start production run
Write-Host "`n[5/5] Starting production run (200 trials, ~12-15 hours)..." -ForegroundColor Yellow

$prodCommands = @"
cd /workspace &&
sed -i 's/n_trials = 1/n_trials = 200/' config.toml &&
nohup heretic > heretic.log 2>&1 & sleep 2 &&
ps aux | grep '[h]eretic' &&
echo 'PRODUCTION_STARTED'
"@

$prodResult = ssh -T $PodSSH $prodCommands 2>&1

if ($prodResult -match "PRODUCTION_STARTED") {
    Write-Host "  ✓ Production run started!" -ForegroundColor Green
    Write-Host "`n" + ("="*70) -ForegroundColor Cyan
    Write-Host "  ABLITERATION RUNNING" -ForegroundColor Green
    Write-Host ("="*70) -ForegroundColor Cyan
    Write-Host "`nMonitor progress with:" -ForegroundColor Yellow
    Write-Host "  ssh -T $PodSSH 'tail -f /workspace/heretic.log'" -ForegroundColor White
    Write-Host "`nOr check trial count:" -ForegroundColor Yellow
    Write-Host "  ssh -T $PodSSH 'grep \"Running trial\" /workspace/heretic.log | tail -5'" -ForegroundColor White
    Write-Host "`nExpected completion: ~12-15 hours from now" -ForegroundColor Gray
    Write-Host "Estimated cost: `$14-18 total`n" -ForegroundColor Gray
} else {
    Write-Host "  ✗ Failed to start production run" -ForegroundColor Red
    Write-Host $prodResult
}
