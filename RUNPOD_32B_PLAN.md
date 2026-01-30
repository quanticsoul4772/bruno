# RunPod A100 80GB Setup Plan for 32B Model
**Date:** 2026-01-30
**Model:** Qwen/Qwen2.5-Coder-32B-Instruct
**Goal:** Stable abliteration with all optimizations enabled

---

## Executive Summary

After 20+ hours of troubleshooting on Vast.ai 4x RTX 4090, we identified that **multi-GPU setup is fundamentally incompatible** with heretic's weight caching optimization. Switching to **RunPod A100 80GB** will:

- Enable weight caching: 5-10x faster trials (2s vs 30s reload)
- Avoid meta device errors: Single GPU = no sharding issues
- Enable iterative ablation: 80GB headroom allows quality improvements
- **Save time:** 12-15h vs 30-35h
- **Save money:** ~$21-26 vs $47-55

---

## Pre-Flight Issues Identified

### Critical Issues from Vast.ai Experience

1. **✅ SOLVED: Multi-GPU weight caching** - A100 80GB = single GPU, caching works
2. **✅ SOLVED: Meta device errors** - Single GPU = no sharding, no meta tensors
3. **✅ SOLVED: Iterative ablation OOM** - 80GB VRAM = plenty of headroom
4. **⚠️ UNSOLVED: PCA extraction hang** - Still need `use_pca_extraction=false`
5. **⚠️ UNSOLVED: Helpfulness dataset config** - Missing language code for c4 dataset
6. **⚠️ NEW RISK: RunPod-specific issues** - Need to identify before starting

### New Risks for RunPod

1. **Persistent storage configuration**
   - RunPod pods can be ephemeral
   - Need to ensure Optuna DB and models saved to persistent volume
   - **Risk:** Lose all progress if pod restarts without persistent storage

2. **Network volume mount**
   - RunPod uses network volumes that can disconnect
   - **Risk:** SQLite corruption if network volume disconnects mid-write
   - **Mitigation:** Use container disk for Optuna DB, network volume for models only

3. **Auto-termination on idle**
   - RunPod may terminate idle pods
   - **Risk:** GPU shows 0% util during residual extraction → looks idle
   - **Mitigation:** Disable auto-termination in pod settings

4. **HuggingFace token not set**
   - Logs showed "unauthenticated requests" warnings
   - **Risk:** Rate limiting, slower downloads
   - **Mitigation:** Set HF_TOKEN environment variable

5. **RunPodCTL vs SSH access**
   - Need to verify which access method works best
   - **Decision:** Use SSH for monitoring (more reliable than runpodctl exec)

---

## Optimized Configuration for A100 80GB

```toml
# config.a100.toml - Optimized for single A100 80GB
model = "Qwen/Qwen2.5-Coder-32B-Instruct"

# ============================================================================
# SINGLE GPU - ALL OPTIMIZATIONS ENABLED
# ============================================================================

device_map = "auto"              # Single GPU (no sharding)
cache_weights = true             # ENABLED (5-10x faster, works on single GPU)
compile = true                   # Worth it on A100 (~1.5-2x speedup)
iterative_rounds = 1             # Safe with 80GB (adds quality)

# ============================================================================
# CRITICAL FIXES FROM VAST.AI EXPERIENCE
# ============================================================================

# CRITICAL: Disable PCA to avoid 18+ hour hang
use_pca_extraction = false

# FIX: Helpfulness dataset needs language config
# Disable orthogonalization instead since dataset is broken
# helpfulness_prompts.dataset = "allenai/c4"
# helpfulness_prompts.config = "en"  # <-- This field doesn't exist in heretic
# orthogonalize_directions = false  # Alternative: just disable it

# ============================================================================
# PERFORMANCE TUNING
# ============================================================================

# Aggressive batching (A100 has plenty of VRAM)
batch_size = 0                   # Auto-detect optimal
max_batch_size = 64              # Allow large batches

# Faster refusal detection
refusal_check_tokens = 20        # Reduced from 30 (refusals appear early)

# ============================================================================
# OPTUNA OPTIMIZATION
# ============================================================================

n_trials = 200
n_startup_trials = 30

# CRITICAL: Save to container disk (not network volume)
storage = "sqlite:////workspace/heretic_32b.db"  # Note: /workspace is container disk

# Study name for resuming
study_name = "qwen32b_a100"

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Auto-select best trial and save
auto_select = true

# CRITICAL: Save model to network volume (persists after pod deletion)
# RunPod network volumes are at /runpod-volume or /workspace/volume
auto_select_path = "/workspace/models/Qwen2.5-Coder-32B-Instruct-heretic"

# Optional: Auto-upload to HuggingFace
# hf_upload = "your-username/qwen-32b-coder-heretic"
# hf_private = false

# ============================================================================
# EXPECTED PERFORMANCE
# ============================================================================

# With this configuration on A100 80GB:
# - Model loading: ~3-4 minutes (single GPU, fast NVMe)
# - Residual extraction: ~8-12 minutes (800 prompts @ auto-detected batch)
# - Per trial: ~4-5 minutes (weight cache = 2s reload, not 30s)
# - Total: ~12-15 hours for 200 trials
#
# vs 4x RTX 4090 (30-35 hours):
# - 2-3x faster overall
# - 50% cheaper ($21 vs $47)
# - Much more stable (no meta device errors)
```

---

## Step-by-Step Execution Plan

### Phase 1: Pod Creation and Verification (10 minutes)

**1.1 Create Pod on RunPod**
```powershell
# Option A: Use runpod.ps1 automation
.\runpod.ps1 create-pod

# Option B: Manual creation on RunPod web console
# Go to: https://www.runpod.io/console/pods
# Select: A100 80GB SXM (or PCIe)
# Template: PyTorch 2.4 (or pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel)
# Storage: 150GB container + 200GB network volume
# Expose: Port 22 (SSH)
```

**1.2 Get Pod Connection Info**
```powershell
# Get pod ID from RunPod console or:
.\runpod.ps1 list-pods

# Note the SSH command, format:
# ssh root@<pod-id>-ssh.runpod.io -p 22
```

**1.3 Verify Pod**
```powershell
# SSH to pod
ssh root@<pod-id>-ssh.runpod.io

# Check GPU
nvidia-smi
# Should show: A100-SXM4-80GB or A100-PCIE-80GB

# Check storage
df -h /workspace
# Should show 150GB container disk

# Exit
exit
```

**Expected time:** 5-10 minutes (pod creation usually fast on RunPod)

**Success criteria:**
- [ ] Pod running
- [ ] A100 80GB visible in nvidia-smi
- [ ] SSH access working
- [ ] Storage adequate (150GB+)

---

### Phase 2: Environment Setup (15 minutes)

**2.1 Install Heretic**
```bash
# SSH to pod
ssh root@<pod-id>-ssh.runpod.io

# Install git
apt-get update && apt-get install -y git

# Install heretic
pip install git+https://github.com/quanticsoul4772/heretic.git

# Verify installation
heretic --help | head -20
```

**2.2 Set Environment Variables**
```bash
# Set HuggingFace token (if needed for gated models)
export HF_TOKEN="your_hf_token_here"

# Add to bashrc for persistence
echo 'export HF_TOKEN="your_hf_token_here"' >> ~/.bashrc

# Set HF cache to workspace (persists)
export HF_HOME=/workspace/.cache/huggingface
echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc

# Create directories
mkdir -p /workspace/models /workspace/.cache/huggingface
```

**2.3 Upload Optimized Config**
```bash
# Still on pod, create config
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

# Faster refusal detection
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

# Verify config
cat /workspace/config.toml
```

**Success criteria:**
- [ ] Heretic installed and working
- [ ] HF_TOKEN set (verify with `echo $HF_TOKEN`)
- [ ] Config file created at /workspace/config.toml
- [ ] Directories created

---

### Phase 3: Pre-Run Validation (5 minutes)

**3.1 GPU Memory Check**
```bash
# Verify clean GPU state
nvidia-smi

# Should show:
# - 1x A100 80GB
# - ~1GB used (base OS)
# - No Python processes
```

**3.2 Storage Check**
```bash
# Verify space
df -h /workspace
# Need: 150GB free (32B model = ~64GB, cache = ~32GB, output = ~64GB)

# Check write permissions
touch /workspace/test.txt && rm /workspace/test.txt
```

**3.3 Network Check**
```bash
# Verify HuggingFace connectivity
curl -I https://huggingface.co
# Should return: HTTP/2 200

# Test model download (just metadata, fast)
python3 << 'EOF'
from transformers import AutoConfig
config = AutoConfig.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct")
print(f"Model has {config.num_hidden_layers} layers")
EOF
# Should print: Model has 64 layers
```

**Success criteria:**
- [ ] GPU shows 1x A100 80GB with minimal usage
- [ ] 150GB+ free space
- [ ] HuggingFace accessible
- [ ] Model metadata loads successfully

---

### Phase 4: Dry Run Test (20 minutes)

**4.1 Test with n_trials=1**
```bash
# Modify config for test run
sed -i 's/n_trials = 200/n_trials = 1/' /workspace/config.toml

# Run single trial (test all components)
cd /workspace
heretic 2>&1 | tee heretic_test.log

# This will test:
# - Model loading with device_map="auto"
# - Weight caching with cache_weights=true
# - Abliteration logic
# - Evaluation (KL + refusals)
```

**4.2 Monitor Test Run**
```bash
# In another terminal, watch GPU
watch -n 5 nvidia-smi

# Expected pattern:
# - Model loading: High GPU util (80-100%), VRAM fills to ~60GB
# - Residual extraction: Medium util (40-60%), VRAM ~65GB
# - Trial execution: Medium util, VRAM stable at ~60-65GB
# - Weight cache reload: Low util, VRAM stable (no reload from disk)
```

**4.3 Verify Test Results**
```bash
# Check for completion
grep -E "Trial 1|KL divergence|Refusals|Auto-selecting" heretic_test.log

# Should see:
# - Trial 1 completed
# - KL divergence: <number>
# - Refusals: <number>/100
# - Model saved to /workspace/models/...

# Check model was saved
ls -lh /workspace/models/Qwen2.5-Coder-32B-Instruct-heretic/
```

**Expected time:** 15-20 minutes for 1 trial

**Success criteria:**
- [ ] Trial 1 completes without errors
- [ ] GPU memory stable at ~60-65GB (no OOM)
- [ ] Weight caching works (reload fast, no disk I/O)
- [ ] Model saved to /workspace/models/
- [ ] NO meta device errors
- [ ] NO OOM errors

---

### Phase 5: Full Production Run (12-15 hours)

**5.1 Update Config for Full Run**
```bash
# Restore n_trials=200
sed -i 's/n_trials = 1/n_trials = 200/' /workspace/config.toml

# Verify
grep n_trials /workspace/config.toml
# Should show: n_trials = 200
```

**5.2 Start Production Run**
```bash
# Start in background with nohup
cd /workspace
nohup heretic > heretic.log 2>&1 &

# Get process ID
ps aux | grep '[h]eretic'
# Note the PID (e.g., 1234)

# Verify single process
ps aux | grep '[h]eretic' | wc -l
# Should show: 1 (not 2+)
```

**5.3 Monitor Progress**
```bash
# Follow logs
tail -f /workspace/heretic.log

# Or from local PowerShell:
ssh root@<pod-id>-ssh.runpod.io 'tail -f /workspace/heretic.log'
```

**5.4 Checkpoint Verification (Every 2 hours)**
```bash
# Check Optuna database exists
ls -lh /workspace/heretic_32b.db
# Size should grow over time

# Check trial progress
grep "Running trial" /workspace/heretic.log | tail -5

# Check GPU memory (should be stable)
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# Expected: ~60-65GB used consistently (no growth)
```

**Expected checkpoints:**
- 2h: ~25-30 trials complete
- 6h: ~75-90 trials complete
- 12h: ~150-180 trials complete
- 15h: 200 trials complete, model saved

---

### Phase 6: Completion and Download (30 minutes)

**6.1 Verify Completion**
```bash
# Check final log messages
tail -50 /workspace/heretic.log

# Should see:
# - "Optimization finished!"
# - "Auto-selecting trial <N>"
# - "Model saved to /workspace/models/..."

# Verify model exists
ls -lh /workspace/models/Qwen2.5-Coder-32B-Instruct-heretic/
# Should show: config.json, model.safetensors (or sharded), tokenizer files
```

**6.2 Download Model to Local**
```powershell
# From local PowerShell
scp -r root@<pod-id>-ssh.runpod.io:/workspace/models/Qwen2.5-Coder-32B-Instruct-heretic ./models/

# Or use runpod.ps1 automation:
.\runpod.ps1 download-model Qwen2.5-Coder-32B-Instruct-heretic
```

**6.3 Terminate Pod**
```powershell
# Stop billing
.\runpod.ps1 stop-pod

# Or via RunPod console:
# https://www.runpod.io/console/pods → Stop pod
```

---

## Critical Configuration Checklist

Before starting the full run, verify ALL of these:

```bash
# 1. Config file correctness
cat /workspace/config.toml | grep -E 'device_map|cache_weights|compile|iterative|use_pca|batch_size'

# MUST show:
# device_map = "auto"
# cache_weights = true
# compile = true
# iterative_rounds = 1
# use_pca_extraction = false
# batch_size = 0
# max_batch_size = 64

# 2. No existing processes
ps aux | grep python
nvidia-smi | grep python
# Both should be EMPTY

# 3. GPU clean state
nvidia-smi --query-gpu=memory.used --format=csv,noheader
# Should show: ~1000 MiB (base OS only)

# 4. HF_TOKEN set
echo $HF_TOKEN
# Should show your token (not empty)

# 5. Verify heretic version
pip show heretic-llm | grep Version
# Should show: 1.0.1 or later

# 6. Test config loads
heretic --help > /dev/null && echo "Config OK" || echo "Config ERROR"
```

---

## Monitoring Strategy

### What to Monitor Every 2 Hours

```bash
# 1. Trial progress
grep "Running trial" /workspace/heretic.log | tail -3

# 2. GPU memory (should be stable)
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

# 3. Process still running
ps aux | grep '[h]eretic' || echo "PROCESS DIED"

# 4. Optuna DB size (should grow)
ls -lh /workspace/heretic_32b.db

# 5. No errors in recent log
tail -50 /workspace/heretic.log | grep -i "error\|traceback\|failed" || echo "No errors"
```

### Red Flags to Watch For

**🚨 STOP IMMEDIATELY if you see:**

1. **Meta device errors:**
   ```
   RuntimeError: Tensor on device meta is not on the expected device cuda:0!
   ```
   **Action:** Config is wrong, stop and fix

2. **OOM errors:**
   ```
   OutOfMemoryError: CUDA out of memory
   ```
   **Action:** Reduce batch_size or disable iterative_rounds

3. **PCA hang:**
   ```
   * Extracting 3 refusal directions using contrastive PCA...
   (GPU util = 0% for >30 minutes)
   ```
   **Action:** use_pca_extraction should be false, restart

4. **Multiple processes:**
   ```bash
   ps aux | grep heretic | wc -l
   # Shows: 2 or more
   ```
   **Action:** Kill all and restart

5. **GPU memory growth:**
   ```
   Hour 0: 60GB
   Hour 2: 65GB
   Hour 4: 70GB  # ← Growing = memory leak
   ```
   **Action:** Restart before OOM

---

## Rollback/Recovery Procedures

### If trial fails mid-run:

```bash
# 1. Check Optuna DB exists
ls -lh /workspace/heretic_32b.db

# 2. Note how many trials completed
grep "Running trial" /workspace/heretic.log | tail -1
# Shows: Running trial <N> of 200

# 3. Simply restart heretic (will resume from trial N+1)
cd /workspace
nohup heretic > heretic.log 2>&1 &

# Optuna storage auto-resumes from last completed trial
```

### If pod restarts unexpectedly:

```bash
# 1. Check if Optuna DB survived
ls -lh /workspace/heretic_32b.db

# If DB exists:
cd /workspace
nohup heretic > heretic.log 2>&1 &
# Will resume from last completed trial

# If DB is gone (network volume unmounted):
# Start fresh, ~15 hours lost
```

**Prevention:** Use container disk for DB, not network volume

---

## Expected Timeline

| Phase | Duration | Cumulative | Activity |
|-------|----------|------------|----------|
| Pod creation | 5 min | 5 min | RunPod console |
| Setup | 15 min | 20 min | Install heretic, config |
| Validation | 5 min | 25 min | Pre-flight checks |
| Dry run (1 trial) | 20 min | 45 min | Test configuration |
| **Full run** | **12-15h** | **13-16h** | 200 trials |
| Download | 30 min | 13.5-16.5h | scp model to local |
| **Total** | **~14-17h** | - | End-to-end |

**Cost:**
- A100 80GB @ $1.74/hr × 14-17h = **$24-30 total**
- vs Vast.ai 4x RTX 4090: $47-55 + debugging time

---

## Success Metrics

After completion, verify:

```bash
# 1. All 200 trials completed
grep "Running trial 200" /workspace/heretic.log

# 2. Model saved
ls -lh /workspace/models/Qwen2.5-Coder-32B-Instruct-heretic/
# Should show: ~64GB model directory

# 3. Best trial selected
grep "Auto-selecting trial" /workspace/heretic.log
# Shows which trial was chosen

# 4. Quality metrics
grep -A 2 "Auto-selecting" /workspace/heretic.log
# Shows: Refusals: X/100, KL divergence: Y.YY

# 5. No crashes
grep -i "OutOfMemoryError\|RuntimeError.*meta\|Traceback" /workspace/heretic.log | wc -l
# Should be: 0
```

---

## Risk Assessment and Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| PCA extraction hang | High | 18+ hours lost | use_pca_extraction=false |
| OOM during trials | Low | Trial fails, auto-retries | iterative_rounds=1 (not 2) |
| Network volume disconnect | Low | DB corruption | Use container disk for DB |
| Pod auto-termination | Low | Progress lost | Disable auto-stop in settings |
| HF rate limiting | Medium | Slower download | Set HF_TOKEN |
| Multiple processes | Low | OOM | Check ps before starting |
| Weight cache failure | Very Low | Slow trials | Single GPU makes this stable |

**Overall risk:** LOW (much lower than 4x RTX 4090 setup)

---

## Differences from Vast.ai Experience

| Issue | Vast.ai (4x RTX 4090) | RunPod (A100 80GB) |
|-------|----------------------|-------------------|
| Weight caching | ❌ Crashes (multi-GPU) | ✅ Works (single GPU) |
| Meta device errors | ✅ Frequent | ❌ None expected |
| Iterative ablation | ❌ OOM | ✅ Safe (80GB) |
| Trial reload time | ~30s (disk) | ~2s (cache) |
| GPU util during extraction | 0% (stuck) | Should be 40-60% |
| Multiple processes risk | High (no lock file) | Same (need manual check) |

---

## Post-Run Actions

After successful completion:

1. **Download model:** `scp -r root@<pod>:/workspace/models/... ./models/`
2. **Save Optuna DB:** `scp root@<pod>:/workspace/heretic_32b.db ./`
3. **Save logs:** `scp root@<pod>:/workspace/heretic.log ./logs/qwen32b_a100.log`
4. **Document results:** Add to knowledge.md with actual timing data
5. **Terminate pod:** Stop billing via RunPod console
6. **Update LESSONS_LEARNED.md:** Add A100 success story
7. **Test model:** Run `python chat_app.py` and load the abliterated model

---

## Execution Checklist

### Before Starting
- [ ] Vast.ai pod terminated (DONE)
- [ ] RunPod account accessible
- [ ] A100 80GB available on RunPod
- [ ] HF_TOKEN ready
- [ ] Local storage has 70GB+ free (for model download)

### During Setup
- [ ] Pod created with A100 80GB
- [ ] SSH access working
- [ ] Heretic installed (version 1.0.1+)
- [ ] Config file created with correct settings
- [ ] HF_TOKEN environment variable set
- [ ] Pre-flight checks pass

### During Run
- [ ] Single heretic process only
- [ ] GPU memory stable at ~60-65GB
- [ ] No meta device errors
- [ ] No OOM errors
- [ ] Trials progressing (~12 per hour)

### After Completion
- [ ] 200 trials completed
- [ ] Model saved to /workspace/models/
- [ ] Model downloaded to local ./models/
- [ ] Pod terminated
- [ ] Results documented

---

## Ready to Execute?

Once you create the RunPod pod and give me the SSH connection details, I'll execute this plan step-by-step, validating each phase before proceeding to the next.

**What I need from you:**
1. RunPod pod ID (after you create it)
2. Or tell me to use .\runpod.ps1 to automate the creation

Should I proceed with creating the pod using runpod.ps1?
