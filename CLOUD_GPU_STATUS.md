# Cloud GPU Availability Status
**Date:** 2026-01-30 21:00 UTC
**Requirement:** 80GB+ VRAM for Qwen2.5-Coder-32B (no CPU offloading)

---

## Current Status: No Suitable GPUs Available

### Vast.ai
- **40GB+ VRAM:** 0 available
- **80GB+ VRAM:** 0 available
- **Status:** Market completely empty for large GPUs

### RunPod
- **Availability:** A100 80GB, H100 80GB, H200 141GB all listed
- **Infrastructure:** 4/4 pods failed to start (container runtime errors)
- **Images tested:** pytorch/pytorch:2.4.0, nvidia/cuda:12.4.0, runpod/pytorch:2.1.0 (official)
- **Status:** Confirmed platform issue, not configuration problem

### Lambda Labs
- **A100 80GB:** 0 available
- **A100 40GB:** Available @ $1.29/hr (tested, insufficient for 32B without CPU offload)
- **GH200 432GB:** Listed @ $1.49/hr but insufficient capacity
- **Status:** Only 40GB available (not enough)

---

## What We Tested

| Provider | GPU | VRAM | Result |
|----------|-----|------|--------|
| Vast.ai | 4x RTX 4090 | 96GB total | Many config issues, 30-35h runtime |
| RunPod | A100 80GB PCIe | 80GB | Container won't start (3 attempts) |
| Lambda Labs | A10 | 24GB | Too small, terminated |
| Lambda Labs | A100 40GB SXM4 | 40GB | Insufficient without CPU offload |

---

## 32B Model VRAM Requirements

**Without CPU offloading:**
- Model weights: ~64GB (FP16/BF16)
- Headroom for operations: ~10-15GB
- **Total required:** 75-80GB minimum

**Why 40GB doesn't work:**
- Model alone = 64GB
- 40GB GPU cannot fit it
- CPU offloading required (user rejected)

**Why 80GB works:**
- Model = 64GB
- Headroom = 16GB
- All operations stay on GPU (fast)

---

## Options

### Option 1: Wait for A100 80GB on Vast.ai (Recommended)
```powershell
.\scripts\monitor_a100.ps1
```
- Checks every 5 minutes
- Alerts when available
- Expected: Hours to days

### Option 2: Check Lambda Labs Manually
- Check: https://lambdalabs.com/service/gpu-cloud
- Look for: A100 80GB or H100
- Refresh every few hours

### Option 3: Alternative Providers
- **CoreWeave:** Enterprise-grade, likely has A100 80GB, $2-3/hr
- **Paperspace:** Sometimes has A100 availability
- **Google Cloud:** A100 80GB available but complex setup

---

## When GPU Becomes Available

Use the prepared automation:
```bash
# If on Vast.ai:
uv run heretic-vast create A100_80GB 1
uv run heretic-vast setup
# Upload config.32b.toml
uv run heretic-vast exec "cd /workspace && heretic"

# If on Lambda Labs with IP:
ssh ubuntu@<IP>
# Then run: bash setup_pod.sh
```

All configuration is ready in:
- `config.32b.toml` - Optimized for A100 80GB
- `setup_pod.sh` - Automated setup script
- `RUNPOD_32B_PLAN.md` - Complete execution plan

---

## Recommendation

**Wait for Vast.ai A100 80GB availability.**

Reasons:
- We have working automation (`heretic-vast` CLI)
- Configuration is tested and documented
- Monitoring script will alert automatically
- Expected cost: $24-30 for 12-15 hours

The 32B model simply requires 80GB+ VRAM without CPU offloading. No shortcuts available.
