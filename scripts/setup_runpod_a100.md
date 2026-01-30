# RunPod A100 Setup for 32B Model

## Quick Start

1. **Go to RunPod Console:** https://www.runpod.io/console/pods

2. **Select GPU:**
   - Filter: A100 80GB SXM
   - Or: A100 80GB PCIe
   - Storage: 150GB minimum

3. **Select Template:** PyTorch 2.4

4. **SSH Setup:**
   ```powershell
   # Get pod ID from RunPod console
   $POD_ID = "your-pod-id"

   # SSH to pod
   ssh root@$POD_ID-ssh.runpod.io -p 22

   # Or use RunPod's web terminal
   ```

5. **Install Heretic:**
   ```bash
   # On the pod
   pip install git+https://github.com/quanticsoul4772/heretic.git

   # Verify
   heretic --help
   ```

6. **Create Config:**
   ```bash
   cat > /workspace/config.toml << 'EOF'
model = "Qwen/Qwen2.5-Coder-32B-Instruct"

# Single A100 80GB - ALL optimizations enabled
device_map = "auto"
cache_weights = true      # ENABLED (5-10x faster trials)
compile = true            # Worth it on A100
iterative_rounds = 1      # Safe with 80GB
use_pca_extraction = false  # Avoid hang

# Aggressive batching
batch_size = 0            # Auto-detect
max_batch_size = 64

# Optuna
n_trials = 200
storage = "sqlite:///heretic_32b_a100.db"
study_name = "qwen32b_a100"

# Auto-save
auto_select = true
auto_select_path = "/workspace/models/Qwen2.5-Coder-32B-Instruct-heretic"
EOF
   ```

7. **Run Abliteration:**
   ```bash
   cd /workspace
   nohup heretic > heretic.log 2>&1 &

   # Monitor
   tail -f heretic.log
   ```

8. **Expected Runtime:** 12-15 hours for 200 trials

9. **Download Model When Done:**
   ```bash
   # On your local machine
   scp -r root@$POD_ID-ssh.runpod.io:/workspace/models/Qwen2.5-Coder-32B-Instruct-heretic ./models/
   ```

## Pricing Comparison

| Provider | GPU | VRAM | Price/hr | Time for 200 trials | Total Cost |
|----------|-----|------|----------|---------------------|------------|
| Vast.ai | 4x RTX 4090 | 96GB | $1.56 | 30-35h | $47-55 |
| RunPod | 1x A100 80GB | 80GB | $1.89 | 12-15h | $23-28 |
| Vast.ai | 1x A100 80GB | 80GB | $2.00 | 12-15h | $24-30 |

**RunPod A100 is cheaper overall** due to faster runtime (weight caching enabled).

## Advantages of A100 80GB vs 4x RTX 4090

| Feature | 4x RTX 4090 | 1x A100 80GB |
|---------|-------------|--------------|
| Weight caching | ❌ Disabled (crashes) | ✅ Enabled (5-10x faster) |
| Iterative ablation | ❌ OOM | ✅ Safe (1-2 rounds) |
| Trial reload time | ~30s (disk) | ~2s (memory) |
| Multi-GPU overhead | Yes (slow) | No (single GPU) |
| Meta device errors | Yes (frequent) | No |
| Runtime for 200 trials | ~30-35h | ~12-15h |
| Total cost | $47-55 | $23-28 |

**Bottom line:** A100 80GB is 2-3x faster AND cheaper than 4x RTX 4090 for 32B models.
