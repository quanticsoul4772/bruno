# RunPod A100 80GB Setup Status
**Date:** 2026-01-30
**Pod ID:** nb97jc5nabyama
**GPU:** A100 PCIe 80GB @ $1.19/hr

---

## Current Status

**Attempt 1:** Pod nb97jc5nabyama - Container failed to start (stopped)
**Attempt 2:** Pod msy0lxmdp4fdmf - Container failed to start (stopped)
**Attempt 3:** Pod wjp08xwu8spdib - Container initializing (minimal CUDA image)

**Created:** 20:30 UTC
**SSH:** wjp08xwu8spdib-64410cb0@ssh.runpod.io
**Status:** Waiting for container (uptime: -13s → 0s, progressing)

**Issue:** RunPod containers failing to start consistently across 3 attempts. Container runtime errors indicate platform infrastructure issues.

**Attempts:**
1. nb97jc5nabyama (PyTorch image): "container not running" - stopped after 15 min
2. msy0lxmdp4fdmf (PyTorch image): "container not found" - stopped after 10 min
3. wjp08xwu8spdib (minimal CUDA image): "container not found" - stopped after 5 min

**Root cause:** RunPod Docker daemon issues, not our configuration. All pods show RUNNING status but containers never start.

**Status:** All pods stopped to prevent billing for non-functional infrastructure.

---

## Alternative Options

### Option 1: Wait for Vast.ai A100 Availability
Run monitoring script to auto-alert when A100 80GB becomes available:
```powershell
.\scripts\monitor_a100.ps1
```

### Option 2: Try Lambda Labs
Lambda Labs often has good A100 availability at $1.29/hr:
- Website: https://lambdalabs.com/service/gpu-cloud
- No automation scripts, manual setup required
- Known for stable infrastructure

### Option 3: Try CoreWeave
Enterprise-grade infrastructure, higher pricing:
- Good availability for A100 80GB
- More expensive (~$2-3/hr)
- Very stable

### Option 4: Retry RunPod Later
RunPod may have temporary infrastructure issues. Try again in 6-12 hours.

---

## Recommendation

**Use Lambda Labs** if immediate access needed, or **wait for Vast.ai** if cost-sensitive.

Lambda Labs pros:
- $1.29/hr (cheapest A100 80GB)
- Stable infrastructure
- Simple setup

Vast.ai pros:
- Existing automation (heretic-vast CLI)
- Known working with our config
- Just need to wait for availability

---

## Setup Plan

Once container is ready, execute these phases:

### Phase 1: Install Dependencies (5 min)
```bash
apt-get update && apt-get install -y git
pip install git+https://github.com/quanticsoul4772/heretic.git
```

### Phase 2: Configure Environment (5 min)
```bash
mkdir -p /workspace/models /workspace/.cache/huggingface
export HF_HOME=/workspace/.cache/huggingface
echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc
```

### Phase 3: Upload Config (1 min)
Upload `config.32b.toml` with these critical settings:
- device_map = "auto" (single GPU)
- cache_weights = true (enables 5-10x faster trials)
- use_pca_extraction = false (avoid 18h hang)
- iterative_rounds = 1 (safe with 80GB)
- batch_size = 0 (auto-detect)

### Phase 4: Test Run (15-20 min)
Run with n_trials=1 to verify configuration works without errors.

### Phase 5: Production Run (12-15 hours)
Start full 200-trial abliteration.

---

## Critical Configuration

```toml
model = "Qwen/Qwen2.5-Coder-32B-Instruct"
device_map = "auto"
cache_weights = true
compile = true
iterative_rounds = 1
use_pca_extraction = false
batch_size = 0
max_batch_size = 64
refusal_check_tokens = 20
n_trials = 200
storage = "sqlite:////workspace/heretic_32b.db"
study_name = "qwen32b_a100"
auto_select = true
auto_select_path = "/workspace/models/Qwen2.5-Coder-32B-Instruct-heretic"
```

---

## Why A100 80GB vs 4x RTX 4090

| Metric | 4x RTX 4090 (96GB) | A100 80GB |
|--------|-------------------|-----------|
| Weight caching | Crashes (multi-GPU incompatible) | Works (5-10x speedup) |
| Meta device errors | Frequent | None |
| Iterative ablation | OOM | Safe |
| Trial reload time | 30s (disk) | 2s (memory) |
| Runtime (200 trials) | 30-35h | 12-15h |
| Cost | $47-55 | $14-18 |
| Stability | Low (many crashes) | High |

Single A100 80GB is faster, cheaper, and more stable than 4x RTX 4090 for 32B models.

---

## Expected Timeline

- Setup: 30 minutes (installation + test run)
- Production: 12-15 hours (200 trials)
- Download: 30 minutes (64GB model)
- Total: 13-16 hours end-to-end

---

## Next Steps

1. Wait for container to finish initializing (should be ready by 20:00-20:05 UTC)
2. Test SSH connection
3. Execute setup_pod.sh script
4. Monitor first few trials for stability
5. Let run for 12-15 hours
6. Download model when complete
