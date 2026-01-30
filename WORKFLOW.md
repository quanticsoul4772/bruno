# Heretic Cloud GPU Workflow

This document covers cloud GPU deployment for heretic on Vast.ai and RunPod.

---

## ⛔ CRITICAL: Storage Requirements

**ALWAYS calculate disk requirements BEFORE creating an instance!**

### Storage Calculation Formula

```
Required Disk = (Source Model Size × 2) + 20GB buffer
```

### Quick Reference Table

| Model Size | Source Model | Output Model | Buffer | **Minimum** | **Recommended** |
|------------|--------------|--------------|--------|-------------|------------------|
| 7B-8B | ~15 GB | ~15 GB | 10 GB | 50 GB | **75 GB** |
| 14B | ~30 GB | ~30 GB | 10 GB | 75 GB | **100 GB** |
| **32B** | **~65 GB** | **~65 GB** | **10 GB** | **150 GB** | **200 GB** |
| 70B-72B | ~140 GB | ~140 GB | 15 GB | 300 GB | **400 GB** |

### Qwen2.5-Coder-32B Breakdown

| Item | Size |
|------|------|
| Source model (Qwen2.5-Coder-32B-Instruct) | ~65 GB |
| Abliterated model output | ~65 GB |
| Optuna SQLite database | ~1 GB |
| HuggingFace cache | ~5 GB |
| System/logs/misc | ~5 GB |
| **TOTAL** | **~141 GB** |

**DEFAULT 100GB IS NOT ENOUGH!** Use **200GB minimum** for 32B models.

### ⛔ What Happens When You Exceed Quota

1. Training completes successfully ✓
2. Model is saved to disk ✓
3. Disk usage exceeds quota (e.g., 116GB > 100GB limit)
4. Instance stops with "Disk quota exceeded" error
5. **YOU CANNOT:**
   - Start the instance
   - SSH into the instance
   - Run `vastai execute` commands (returns null)
   - Use `vastai copy` (requires running instance)
   - Delete files to free space
   - Increase the quota
6. **ALL DATA IS PERMANENTLY LOST**

---

## 🚀 Quick Start (Qwen2.5-Coder-32B)

**One command to rule them all:**
```powershell
.\start-abliteration.ps1
```

Or step-by-step:
```powershell
# 1. Create instance with 4x RTX 4090 (~$1.50/hr, ~2 min)
.\runpod.ps1 vast-create-pod RTX_4090 4

# 2. Install heretic (~3 min)
.\runpod.ps1 vast-setup

# 3. Start abliteration with persistent storage (~5 min setup, then runs 5-6 hrs)
.\runpod.ps1 vast-exec "export HF_HOME=/workspace/.cache/huggingface && cd /workspace && nohup heretic --model Qwen/Qwen2.5-Coder-32B-Instruct --auto-select true --auto-select-path /workspace/models --storage sqlite:////workspace/heretic_study.db --study-name qwen32b > /workspace/heretic.log 2>&1 &"

# 4. Monitor progress
.\runpod.ps1 vast-watch

# 5. When complete, download and stop
.\runpod.ps1 vast-download-model
.\runpod.ps1 vast-stop
```

**Key Features:**
- ✅ **Persistent storage** - Resumes if interrupted (uses SQLite via `--storage`)
- ✅ **4x RTX 4090** - 96GB VRAM for 32B model
- ✅ **~$8-10 total cost** for full 100-trial run
- ✅ **Performance optimizations** - In-memory weight caching, parallel evaluation, early stopping

---

## Working Commands (USE THESE!)

The `heretic-vast` CLI (Python) works. The PowerShell scripts have issues from bash.

### Primary Commands (heretic-vast CLI)

```bash
# List instances
uv run heretic-vast list

# Create instance (4x RTX 4090 for 32B model)
uv run heretic-vast create RTX_4090 4

# Setup heretic on instance
uv run heretic-vast setup

# Run abliteration with RESUME SUPPORT
uv run heretic-vast exec "export HF_HOME=/workspace/.cache/huggingface && cd /workspace && nohup heretic --model Qwen/Qwen2.5-Coder-32B-Instruct --auto-select true --auto-select-path /workspace/models --storage sqlite:////workspace/heretic_study.db --study-name qwen32b-abliteration > /workspace/heretic.log 2>&1 &"

# Monitor progress
uv run heretic-vast watch
uv run heretic-vast progress

# Check status
uv run heretic-vast status

# View logs
uv run heretic-vast exec "tail -100 /workspace/heretic.log"

# Stop instance (save money)
uv run heretic-vast stop

# Start stopped instance
uv run heretic-vast start

# Download model when complete
uv run heretic-vast download /workspace/models
```

### Direct SSH Commands (When heretic-vast Fails)

```bash
# Get instance info
vastai show instances --raw | grep -E '"id":|"ssh_host":|"ssh_port":|"actual_status"'

# Install heretic on running instance (replace PORT with actual port)
ssh -o StrictHostKeyChecking=no -p PORT root@ssh4.vast.ai 'pip install git+https://github.com/quanticsoul4772/abliteration-workflow.git'

# Start training with ALL required flags
ssh -o StrictHostKeyChecking=no -p PORT root@ssh4.vast.ai '
  export HF_TOKEN=YOUR_TOKEN
  export HF_HOME=/workspace/.cache/huggingface
  cd /workspace
  nohup heretic \
    --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --auto-select true \
    --auto-select-path /workspace/models \
    --storage sqlite:////workspace/heretic_study.db \
    --study-name qwen32b-abliteration \
    --cache-weights false \
    > /workspace/heretic.log 2>&1 &
  echo "Started PID: $!"
'

# Monitor logs
ssh -o StrictHostKeyChecking=no -p PORT root@ssh4.vast.ai 'tail -f /workspace/heretic.log'

# Check if heretic is running
ssh -o StrictHostKeyChecking=no -p PORT root@ssh4.vast.ai 'ps aux | grep heretic | grep -v grep'

# Check progress
ssh -o StrictHostKeyChecking=no -p PORT root@ssh4.vast.ai 'tail -50 /workspace/heretic.log'
```

---

## Vast.ai CLI Reference (heretic-vast)

Heretic includes a dedicated Python CLI for Vast.ai with a rich terminal dashboard.

### Installation

```bash
# Install heretic with Vast.ai support
pip install heretic-llm fabric rich
```

### Configuration

Set your Vast.ai API key:

```bash
# Option 1: Environment variable
export VAST_API_KEY="your-api-key"

# Option 2: .env file in project directory
echo 'VAST_API_KEY="your-api-key"' > .env
```

Get your API key from: https://cloud.vast.ai/account/

### GPU Tiers

| Tier | VRAM | Max Price | Best For |
|------|------|-----------|----------|
| RTX_4090 | 24GB | $0.50/hr | 7B-8B models |
| A6000 | 48GB | $0.80/hr | 14B-30B models |
| A100_40GB | 40GB | $1.00/hr | 14B-32B models |
| A100_80GB | 80GB | $2.00/hr | 32B-70B models |
| A100_SXM | 80GB | $2.50/hr | 70B+ (fastest) |
| H100 | 80GB | $4.00/hr | 70B+ (latest) |

### Command Reference

```bash
# Instance Management
heretic-vast create TIER [NUM_GPUS]  # Create instance
heretic-vast list                     # List your instances
heretic-vast start [ID]               # Start stopped instance
heretic-vast stop [ID]                # Stop (pause billing)
heretic-vast terminate ID             # Destroy permanently

# Setup & Execution
heretic-vast setup [ID]               # Install heretic
heretic-vast run MODEL [ID]           # Run abliteration
heretic-vast exec "command" [ID]      # Run any command
heretic-vast connect [ID]             # Interactive SSH

# Monitoring
heretic-vast status [ID]              # GPU status snapshot
heretic-vast progress [ID]            # Check abliteration progress
heretic-vast watch [ID]               # Live dashboard (Ctrl+C to exit)

# Models
heretic-vast models [ID]              # List saved models
heretic-vast download [MODEL] [ID]    # Download model locally

# Info
heretic-vast tiers                    # Show GPU tier info
heretic-vast gpus TIER                # Search available GPUs
```

### Live Dashboard

The `heretic-vast watch` command shows a live dashboard:

```
┌─────────────────────────────────────────────────────────────────┐
│                  VAST.AI ABLITERATION DASHBOARD                  │
├─────────────────────────────────────────────────────────────────┤
│ Instance: 12345678  │  Time: 14:32:15  │  Watch: 01:23:45       │
├─────────────────────┬───────────────────────────────────────────┤
│ Process             │ GPUs                                      │
│ Status: ● RUNNING   │ GPU   Util   VRAM           Temp   Power  │
│ Model: Qwen/Qwen2.5 │ 0     63%    69.6/80GB      45°C   250W   │
│ Runtime: 165:33     │ 1     64%    72.7/80GB      47°C   245W   │
│ CPU: 110%           │                                           │
├─────────────────────┴───────────────────────────────────────────┤
│ Output Models                                                    │
│ (No models saved yet)                                            │
├─────────────────────────────────────────────────────────────────┤
│ Disk: 45G/150G (30% used)  │  Refresh: 10s  │  Ctrl+C to exit   │
└─────────────────────────────────────────────────────────────────┘
```

### Model Size Guidelines

| Model Size | Recommended Setup | Estimated Time* |
|------------|-------------------|----------------|
| 7B-8B | 1x RTX_4090 | 20-30 min |
| 14B | 1x A6000 or A100_40GB | 30-45 min |
| 32B | 1x A100_80GB | 45-90 min |
| 70B-72B | 2x A100_80GB | 2-3 hours |

*Times reflect performance optimizations (in-memory caching, early stopping, parallel eval)

---

## Resume Support (CRITICAL!)

The `--storage sqlite:////workspace/heretic_study.db --study-name qwen32b-abliteration` flags enable resume:

- SQLite database persists on `/workspace`
- If training stops, restart instance and run same command
- Optuna automatically detects existing study and continues

### Resume After Interruption

```bash
# 1. Check if instance is still running
uv run heretic-vast list

# 2. If stopped, restart it
uv run heretic-vast start

# 3. Wait for instance to start (~30 seconds)

# 4. Re-run the SAME command - it will RESUME automatically
uv run heretic-vast exec "export HF_HOME=/workspace/.cache/huggingface && cd /workspace && nohup heretic --model Qwen/Qwen2.5-Coder-32B-Instruct --auto-select true --auto-select-path /workspace/models --storage sqlite:////workspace/heretic_study.db --study-name qwen32b-abliteration > /workspace/heretic.log 2>&1 &"

# 5. Verify it's running and resumed
uv run heretic-vast progress
```

---

## After Training Completes

```bash
# 1. Check for completed model
uv run heretic-vast models

# 2. Download model to local machine (~60GB, may take 30+ min)
uv run heretic-vast download /workspace/models

# Alternative: Use rsync for resumable download
rsync -avz --progress -e "ssh -p PORT" root@ssh4.vast.ai:/workspace/models/ ./qwen32b-abliterated/

# 3. STOP THE INSTANCE (stop billing!)
uv run heretic-vast stop
```

### Downloading Models from Vast.ai

**THREE OPTIONS - USE WHAT WORKS:**

1. **Direct rsync from WSL (MOST RELIABLE)**:
   ```bash
   # From WSL Ubuntu terminal:
   cd /mnt/c/Development/Projects/heretic
   mkdir -p ./models/downloaded-model
   rsync -avz --progress -e "ssh -o StrictHostKeyChecking=no -p PORT" root@ssh1.vast.ai:/workspace/models/ ./models/downloaded-model/
   ```

2. **PowerShell script**: `.\runpod.ps1 vast-download-model`

3. **Python CLI**: `heretic-vast download`

**SSH Key Issues from WSL:**
WSL has its own SSH keys separate from Windows. If you get "Permission denied (publickey)":
```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
cp /mnt/c/Users/YOUR_WINDOWS_USERNAME/.ssh/id_rsa ~/.ssh/
chmod 600 ~/.ssh/id_rsa
```

---

## Creating Instances with Correct Disk Size

### Using vastai CLI directly (RECOMMENDED)

```bash
# First, find an offer with sufficient disk
vastai search offers "num_gpus=4 gpu_name=RTX_4090 disk_space>=200 rentable=true" --raw

# Then create with specific disk size
vastai create instance OFFER_ID --disk 200 --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel --onstart-cmd "pip install git+https://github.com/quanticsoul4772/abliteration-workflow.git"
```

### Prevention Checklist

Before creating ANY cloud instance for training:

- [ ] Calculate required storage: `(model_size × 2) + 20GB`
- [ ] Verify offer has sufficient disk space
- [ ] Create instance with explicit `--disk` parameter
- [ ] After instance starts, verify disk size: `df -h /workspace`
- [ ] Monitor disk usage during training
- [ ] Download model IMMEDIATELY when training completes
- [ ] Never treat cloud instance storage as your only copy

### Monitoring Disk During Training

```bash
# Check disk usage
ssh -p PORT root@HOST 'df -h /workspace'

# Check what's using space
ssh -p PORT root@HOST 'du -sh /workspace/*'

# Watch disk usage in real-time
ssh -p PORT root@HOST 'watch -n 60 df -h /workspace'
```

---

## Troubleshooting

### Common Issues

**"No instance ID found"**
```bash
uv run heretic-vast list  # Check if instance exists
uv run heretic-vast create RTX_4090 4  # Create new if needed
```

**SSH Connection Issues**
- Wait 30-60 seconds after instance starts
- Run `uv run heretic-vast list` to get updated SSH info

**"heretic command not found"**
```bash
ssh -o StrictHostKeyChecking=no -p PORT root@ssh4.vast.ai 'pip install git+https://github.com/quanticsoul4772/abliteration-workflow.git'
```

**pip install breaks transformers/PyTorch**
```bash
# Solution: Reinstall correct transformers version
ssh -p PORT root@ssh4.vast.ai 'pip install transformers>=4.55.2'

# Prevention: Never use --force-reinstall, use regular pip install --upgrade
```

**Training Not Resuming**
```bash
# Check that the SQLite database exists
ssh -p PORT root@HOST 'ls -la /workspace/*.db'
```

**GPU Out of Memory (OOM) on 32B+ Models**
```bash
# Solution: Use --cache-weights false flag
heretic --model Qwen/Qwen2.5-Coder-32B-Instruct --cache-weights false
```

### SSH Authentication Failures

**Root Causes & Solutions:**

1. **SSH agent not running** - The key exists but isn't loaded:
   ```bash
   eval $(ssh-agent -s)
   ssh-add ~/.ssh/id_ed25519
   ```

2. **Key permissions wrong** - Must be 600:
   ```bash
   chmod 600 ~/.ssh/id_ed25519
   ```

3. **Key not attached to instance** - Even if registered on account:
   ```bash
   vastai attach ssh INSTANCE_ID "$(cat ~/.ssh/id_ed25519.pub)"
   ```

4. **Instance needs reboot** - After attaching SSH key:
   ```bash
   vastai reboot instance INSTANCE_ID
   # Wait 30-60 seconds for reboot
   ```

5. **Wrong instance/port** - After reboot, port may change:
   ```bash
   uv run heretic-vast list  # Get new SSH details
   ```

6. **Fabric vs direct SSH** - If `heretic-vast` fails but direct SSH works:
   ```bash
   ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -p PORT root@ssh1.vast.ai "command"
   ```

**Debugging SSH Issues:**
```bash
# Check if key is loaded
ssh-add -l

# Check keys registered on Vast.ai
vastai show ssh-keys

# Test with verbose output
ssh -vvv -i ~/.ssh/id_ed25519 -p PORT root@ssh1.vast.ai "echo test" 2>&1 | tail -30

# Compare key fingerprints
ssh-keygen -lf ~/.ssh/id_ed25519.pub  # Local key
vastai show ssh-keys                   # Remote key
```

**Full Recovery Workflow:**
```bash
# 1. Start SSH agent and add key
eval $(ssh-agent -s) && ssh-add ~/.ssh/id_ed25519

# 2. Verify key is on Vast.ai
vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)" 2>/dev/null || echo "Key exists"

# 3. Get instance ID
INSTANCE_ID=$(uv run heretic-vast list 2>/dev/null | grep -oP '\d{8}' | head -1)

# 4. Attach key to instance
vastai attach ssh $INSTANCE_ID "$(cat ~/.ssh/id_ed25519.pub)"

# 5. Reboot instance
vastai reboot instance $INSTANCE_ID

# 6. Wait and get new SSH details
sleep 45
uv run heretic-vast list

# 7. Test connection
ssh -o StrictHostKeyChecking=no -p NEW_PORT root@ssh1.vast.ai "echo SUCCESS"
```

---

## Pain Points & Resolutions

### ⛔ CATASTROPHIC: Disk Quota Data Loss (June 2025)

**Problem:** Vast.ai instance exceeded disk quota (100GB). Instance stopped and **ALL training data was lost** - 200 trials of Qwen2.5-Coder-32B abliteration (~$15 GPU cost, 6+ hours).

**Root Cause:** Default Vast.ai disk is 100GB. A 32B model requires ~141GB.

**Resolution:** Always use **200GB minimum** for 32B models.

### SSH Key Not Recognized on New Pods (RunPod)

**Problem:** SSH authentication fails with "Permission denied (publickey)".

**Resolution:** Add your SSH public key to RunPod account settings (Settings → SSH Keys) BEFORE creating pods.

### Heretic Crashes with EOFError in Non-Interactive Mode

**Problem:** When running heretic via `nohup`, it crashes from `prompt_toolkit` because there's no TTY.

**Resolution:** Use `--auto-select true` flag for non-interactive operation.

### PowerShell Scripts Fail from Bash

**Problem:** Running `.ps1` scripts from bash fails with syntax errors.

**Resolution:** Use direct SSH commands or `heretic-vast` CLI instead.

### pip install --force-reinstall Breaks Environment

**Problem:** After `--force-reinstall`, heretic fails with "Could not import module".

**Resolution:** Run `pip install transformers>=4.55.2`. Prevention: Never use `--force-reinstall`.

### Instance Stop vs Process Kill

**Problem:** Stopping a Vast.ai instance KILLS running processes immediately.

**Resolution:** Check `heretic-vast progress` for "Models Saved" before stopping.

---

## RunPod Workflow (Alternative)

### Initial Setup (One-Time)

```powershell
# 1. Install WSL (requires admin PowerShell, then reboot)
wsl --install

# 2. Install runpodctl CLI
.\runpod.ps1 install-runpodctl

# 3. Configure API key
# Set $RUNPOD_API_KEY in runpod.ps1

# 4. Add SSH key to RunPod (Settings → SSH Keys)

# 5. Verify all tools
.\runpod.ps1 check-tools
```

### Quick Start

```powershell
.\runpod.ps1 create-pod                      # Create pod
.\runpod.ps1 setup                           # Install heretic
.\runpod.ps1 run Qwen/Qwen3-4B-Instruct-2507 # Abliterate
.\runpod.ps1 stop-pod                        # Stop when done
```

### Command Reference

```powershell
# Setup & Tools
.\runpod.ps1 install-runpodctl  # Download runpodctl CLI
.\runpod.ps1 check-tools        # Verify WSL, runpodctl, SSH, API key

# Pod Management
.\runpod.ps1 create-pod     # Create and start pod
.\runpod.ps1 list-pods      # Show all pods
.\runpod.ps1 get-ssh        # Get SSH connection info
.\runpod.ps1 start-pod      # Restart stopped pod
.\runpod.ps1 stop-pod       # Stop (pause billing)
.\runpod.ps1 terminate-pod  # Delete permanently

# Core Commands
.\runpod.ps1 setup          # Install heretic on pod
.\runpod.ps1 test           # Run quick test (Qwen3-4B)
.\runpod.ps1 run <model>    # Process any model
.\runpod.ps1 status         # GPU status (nvidia-smi)
.\runpod.ps1 progress       # Check abliteration progress
.\runpod.ps1 exec 'cmd'     # Run any command

# Advanced
.\runpod.ps1 connect        # Interactive SSH
.\runpod.ps1 monitor        # Live GPU stats
```

### Cost Optimization

| Task | RTX 4090 Cost |
|------|---------------|
| Setup | $0.05 (~10 min) |
| Qwen3-4B | $0.09 (~15 min) |
| Llama-8B | $0.17 (~30 min) |
| Total session | ~$0.30 |

---

## Running Experiments

Heretic includes an experiments framework for testing new behavioral directions.

### Verbosity Spike Experiment

```bash
# 1. Create local datasets
python experiments/verbosity/load_local_dataset.py

# 2. Copy verbosity config
cp experiments/verbosity/config.verbosity.toml config.toml

# 3. Run heretic (use Qwen - not gated)
heretic --model Qwen/Qwen2.5-7B-Instruct --auto-select true

# 4. Evaluate verbosity change (use 4-bit quantization for 8GB VRAM)
python experiments/verbosity/test_comparison_4bit.py
```

See `experiments/verbosity/README.md` for detailed instructions.

---

## Configuration Details

### Training Config (config-qwen32b.toml)
- **Trials:** 100
- **Batch size:** 8
- **Max response length:** 150
- **Prune trials:** true
- **Startup trials:** 25
- **Training data:** 500 harmless + 300 harmful prompts
- **Eval data:** 200 harmless + 150 harmful prompts

### Hardware
- **GPU:** 4x RTX 4090 (96GB total VRAM)
- **Cost:** ~$1.40-1.60/hr
- **Total estimated cost:** ~$8-10 for full run
- **Runtime:** ~5-6 hours

### Critical Flags for Resume Support
```
--storage sqlite:////workspace/heretic_study.db
--study-name qwen32b-abliteration
```

---

## Git Workflow

### Push Changes
```bash
git add -A
git commit -m "Your message"
git push fork master  # Push to abliteration-workflow (PRIMARY)
```

### If Git Lock File Blocks Push
```powershell
powershell -Command "Get-Process git*, git-remote* -ErrorAction SilentlyContinue | Stop-Process -Force"
git push fork master
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `start-abliteration.ps1` | One-command quick start for Qwen32B training |
| `runpod.ps1` | All cloud GPU commands (Vast.ai and RunPod) |
| `config-qwen32b.toml` | Optimized config for Qwen2.5-Coder-32B |
| `.env` | API keys (VAST_API_KEY, HF_TOKEN) |
| `WORKFLOW.md` | This document |
| `CLAUDE.md` | AI assistant instructions |

---

## Session History (June 2025)

### Session 1: Lost to Disk Quota (100GB instance)
- **Instance ID:** 30720863
- **Status:** ❌ DATA LOST - Disk quota exceeded (116GB > 100GB)
- **Trials completed:** 200 (but model couldn't be downloaded)
- **Best result:** Trial 199 - 0 refusals, 0.51 KL divergence
- **Lesson:** 100GB is NOT enough for 32B models. Need 200GB minimum.

### Session 2: Terminated Early (200GB instance)
- **Instance ID:** 30733503
- **Status:** ✅ Terminated by user (end of day)
- **Trials completed:** ~13 of 200
- **Storage:** 200GB (correct size)
- **Cost:** ~$2.21/hr

---

## ✅ CHECKLIST: Starting Tomorrow's Training

**BEFORE creating instance:**
```
□ 1. Verify disk requirement: 200GB minimum for 32B models
□ 2. Search for 4x RTX 4090 with 200GB+ disk
□ 3. Create instance with explicit --disk 200 flag
```

**Create instance:**
```bash
# Search for offers
vastai search offers "num_gpus>=4 gpu_name=RTX_4090 disk_space>=200 rentable=true" --raw | head -50

# Create with 200GB disk (replace OFFER_ID)
vastai create instance OFFER_ID --disk 200 --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel --onstart-cmd "pip install huggingface_hub && pip install git+https://github.com/quanticsoul4772/abliteration-workflow.git"
```

**After instance starts:**
```bash
# Verify instance is running
uv run heretic-vast list

# Start training with resume support
ssh -o StrictHostKeyChecking=no -p PORT root@HOST '
  export HF_TOKEN=YOUR_TOKEN
  nohup heretic \
    --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --batch-size 4 \
    --max-batch-size 8 \
    --cache-weights false \
    --auto-select true \
    --auto-select-path /workspace/models \
    --storage sqlite:////workspace/heretic_study.db \
    --study-name qwen32b-abliteration \
    > /workspace/heretic.log 2>&1 &
  sleep 5
  ps aux | grep python | grep -v grep
  tail -20 /workspace/heretic.log
'
```

**When training completes (200 trials):**
```bash
# Download model IMMEDIATELY (before disk fills up)
scp -r -P PORT root@HOST:/workspace/models ./qwen32b-abliterated

# Or use rsync for resumable download
rsync -avz --progress -e "ssh -p PORT" root@HOST:/workspace/models/ ./qwen32b-abliterated/

# Then destroy instance
vastai destroy instance INSTANCE_ID
```

---

## Key Learnings (Don't Forget!)

1. **Always use `--storage` and `--study-name`** for resume support
2. **Push to `fork` remote** (abliteration-workflow), not heretic-fork
3. **SQLite database persists on /workspace** - survives instance restarts
4. **Kill zombie git processes** if push hangs
5. **4x RTX 4090 = 96GB VRAM** - sufficient for 32B models
6. **Vast.ai is 50% cheaper** than RunPod
7. **Monitor with `heretic-vast watch`** for live dashboard
8. **Stop instance when done** - billing continues until stopped!
9. **Use `--cache-weights false` for 32B+ models** - prevents OOM
10. **Set HF_TOKEN on server** for faster downloads
11. **PowerShell scripts fail from bash** - use direct SSH commands instead
12. **Never use `pip install --force-reinstall`** on server - breaks dependencies
13. **NEVER judge instance status by uptime** - only check if process is running
14. **ALWAYS use 200GB disk for 32B models** - 100GB default is NOT enough
