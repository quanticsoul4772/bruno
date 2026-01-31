#!/bin/bash
# RunPod A100 80GB Setup Script for 32B Model
# Upload this to pod and execute

set -e

echo "========================================"
echo " Heretic 32B Model Setup on A100 80GB"
echo "========================================"
echo

# Phase 1: Install dependencies
echo "[1/5] Installing git and heretic..."
apt-get update -qq
apt-get install -y -qq git
pip install -q git+https://github.com/quanticsoul4772/heretic.git
echo "  ✓ Heretic installed: $(pip show heretic-llm | grep Version)"

# Phase 2: Setup environment
echo
echo "[2/5] Configuring environment..."
mkdir -p /workspace/models /workspace/.cache/huggingface
export HF_HOME=/workspace/.cache/huggingface
echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc
echo "  ✓ Directories created"

# Phase 3: Create optimized config
echo
echo "[3/5] Creating config.toml..."
cat > /workspace/config.toml << 'EOF'
model = "Qwen/Qwen2.5-Coder-32B-Instruct"

# Single A100 80GB - all optimizations enabled
device_map = "auto"
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
storage = "sqlite:////workspace/heretic_32b.db"
study_name = "qwen32b_a100"

# Auto-save
auto_select = true
auto_select_path = "/workspace/models/Qwen2.5-Coder-32B-Instruct-heretic"
EOF

echo "  ✓ Config created:"
cat /workspace/config.toml | grep -E 'model|device_map|cache_weights|n_trials'

# Phase 4: Verify GPU
echo
echo "[4/5] Verifying GPU state..."
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
ps aux | grep python | grep -v grep || echo "  ✓ No Python processes running"

# Phase 5: Test run (1 trial)
echo
echo "[5/5] Running test (1 trial, ~15-20 min)..."
cd /workspace
sed -i 's/n_trials = 200/n_trials = 1/' config.toml

heretic > heretic_test.log 2>&1

# Check results
echo
echo "Test Results:"
grep -E "Trial 1|KL divergence|Refusals|Model saved" heretic_test.log | tail -5

# Check for errors
if grep -q "OutOfMemoryError\|RuntimeError.*meta" heretic_test.log; then
    echo
    echo "❌ ERRORS DETECTED - DO NOT START PRODUCTION RUN"
    echo "Check: tail -100 /workspace/heretic_test.log"
    exit 1
fi

# Verify model saved
if [ -d "/workspace/models/Qwen2.5-Coder-32B-Instruct-heretic" ]; then
    echo "  ✓ Model saved successfully"
    ls -lh /workspace/models/Qwen2.5-Coder-32B-Instruct-heretic/ | head -5
else
    echo "  ✗ Model NOT saved - check logs"
    exit 1
fi

# Restore config for production
sed -i 's/n_trials = 1/n_trials = 200/' /workspace/config.toml

echo
echo "========================================"
echo "  ✓ SETUP COMPLETE - READY FOR PRODUCTION"
echo "========================================"
echo
echo "Start production run (200 trials):"
echo "  cd /workspace && nohup heretic > heretic.log 2>&1 &"
echo
echo "Monitor:"
echo "  tail -f /workspace/heretic.log"
echo
echo "Expected: ~12-15 hours, ~\$14-18 total cost"
