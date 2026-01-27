# Heretic Cloud GPU Workflow

This document covers cloud GPU deployment for heretic on RunPod and Vast.ai.

---

## Vast.ai CLI (heretic-vast)

Heretic includes a dedicated Python CLI for Vast.ai with a rich terminal dashboard.

### Installation

```bash
# Install heretic with Vast.ai support
pip install heretic-llm fabric rich

# Or if already installed
pip install fabric rich
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

### Quick Start

```bash
# See available GPU tiers
heretic-vast tiers

# Create instance (2x A100 80GB for 70B+ models)
heretic-vast create A100_80GB 2

# Install heretic on the instance
heretic-vast setup

# Run abliteration
heretic-vast run Qwen/Qwen2.5-72B-Instruct

# Monitor progress (in another terminal)
heretic-vast watch

# Stop when done (pause billing)
heretic-vast stop
```

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

| Model Size | Recommended Setup | Estimated Time |
|------------|-------------------|----------------|
| 7B-8B | 1x RTX_4090 | 30-45 min |
| 14B | 1x A6000 or A100_40GB | 45-60 min |
| 32B | 1x A100_80GB | 1-2 hours |
| 70B-72B | 2x A100_80GB | 3-4 hours |

### Troubleshooting

**"fabric not installed"**
```bash
pip install fabric
```

**"VAST_API_KEY not set"**
- Set environment variable or create `.env` file

**"No running instance found"**
- Run `heretic-vast list` to check instance status
- Instance may still be starting (wait 30-60 seconds)

**SSH connection fails**
- Ensure SSH key is added to Vast.ai account
- Check that instance is fully started

---

# RunPod Workflow (Alternative)

## Initial Setup (One-Time)

```powershell
# 1. Install WSL (requires admin PowerShell, then reboot)
wsl --install

# 2. Install runpodctl CLI
.\runpod.ps1 install-runpodctl

# 3. Configure API key (edit runpod.ps1 or get from runpod.io/console/user/settings)
# Set $RUNPOD_API_KEY in runpod.ps1

# 4. Add SSH key to RunPod (Settings → SSH Keys)
# Upload your ~/.ssh/id_ed25519.pub

# 5. Verify all tools
.\runpod.ps1 check-tools
```

## Quick Start (3 Commands)

```powershell
.\runpod.ps1 create-pod                      # Create pod
.\runpod.ps1 setup                           # Install heretic
.\runpod.ps1 run Qwen/Qwen3-4B-Instruct-2507 # Abliterate
.\runpod.ps1 stop-pod                        # Stop when done
```

## Monitor Progress

```powershell
# Check abliteration progress (process, GPU, workspace)
.\runpod.ps1 progress

# Check GPU status
.\runpod.ps1 status

# Execute any command on the pod
.\runpod.ps1 exec 'nvidia-smi'
.\runpod.ps1 exec 'ps aux | grep heretic'
```

## Complete Workflow

### Step 1: Create Pod

```powershell
cd C:\Development\Projects\heretic
.\runpod.ps1 create-pod
```

This:
- Creates an RTX 4090 pod with PyTorch 2.4.0 image
- Waits for pod to start
- Auto-configures SSH connection details
- Updates the SSH proxy address in the script

### Step 2: Setup Heretic

```powershell
.\runpod.ps1 setup
```

This runs automatically via WSL:
- Sets HuggingFace cache to volume
- Installs heretic-llm
- Clears any GPU-hogging processes
- Shows GPU status

### Step 3: Run Abliteration

```powershell
# Test with small model
.\runpod.ps1 test

# Or run specific model
.\runpod.ps1 run meta-llama/Llama-3.1-8B-Instruct
```

### Step 4: Save Results

When heretic shows Pareto-optimal trials:
1. Select a trial (low refusals + low KL divergence)
2. Choose "Save the model to a local folder"
   - Path: `/workspace/models/your-model-name`
3. Choose "Upload the model to Hugging Face" (optional)

### Step 5: Stop Pod

```powershell
.\runpod.ps1 stop-pod
```

This stops billing while preserving your volume data.

## Command Reference

### Setup & Tools

```powershell
.\runpod.ps1 install-runpodctl  # Download runpodctl CLI
.\runpod.ps1 check-tools        # Verify WSL, runpodctl, SSH, API key
```

### Pod Management

```powershell
.\runpod.ps1 create-pod     # Create and start pod
.\runpod.ps1 list-pods      # Show all pods (uses runpodctl)
.\runpod.ps1 get-ssh        # Get SSH connection info
.\runpod.ps1 start-pod      # Restart stopped pod
.\runpod.ps1 stop-pod       # Stop (pause billing)
.\runpod.ps1 terminate-pod  # Delete permanently
.\runpod.ps1 gpus           # Available GPU types
```

### Core Commands

```powershell
.\runpod.ps1 setup          # Install heretic on pod
.\runpod.ps1 test           # Run quick test (Qwen3-4B)
.\runpod.ps1 run <model>    # Process any model
.\runpod.ps1 status         # GPU status (nvidia-smi)
.\runpod.ps1 progress       # Check abliteration progress
.\runpod.ps1 exec 'cmd'     # Run any command
```

### Advanced

```powershell
.\runpod.ps1 connect        # Interactive SSH
.\runpod.ps1 monitor        # Live GPU stats
.\runpod.ps1 hf-login       # Configure HuggingFace token
.\runpod.ps1 hf-test        # Test HF authentication
```

## Technical Details

### Hybrid Approach

The script uses a hybrid approach leveraging the best of each tool:

| Tool | Purpose | Why |
|------|---------|-----|
| **runpodctl** | Pod management, SSH info | Fast, reliable CLI |
| **WSL + SSH heredoc** | Command execution | Required for PTY support |
| **GraphQL API** | Pod creation, advanced ops | Full control |

### Why WSL?

RunPod's SSH proxy requires a pseudo-terminal (PTY) that Windows SSH cannot provide in non-interactive mode. The `runpodctl exec` command only supports Python scripts, not general shell commands.

**Solution:** Use WSL with heredoc syntax:

```powershell
# This is what happens under the hood:
powershell -Command "wsl -e bash -c 'ssh -tt podid-userid@ssh.runpod.io <<EOF
commands here
exit
EOF'"
```

WSL provides a real Linux environment with proper PTY support.

## Model Processing Times

| Model | RTX 4090 | RTX 3090 |
|-------|----------|----------|
| Qwen/Qwen3-4B | 15 min | 30 min |
| Llama-3.1-8B | 30 min | 45 min |
| Mistral-7B | 25 min | 40 min |

## Cost Optimization

1. **Stop pods when idle** - `.\runpod.ps1 stop-pod`
2. **Use RTX 4090** - Best cost/performance ratio ($0.34/hr)
3. **Volume persists** - No need to re-download models
4. **Batch your work** - Do multiple models in one session

| Task | RTX 4090 Cost |
|------|---------------|
| Setup | $0.05 (~10 min) |
| Qwen3-4B | $0.09 (~15 min) |
| Llama-8B | $0.17 (~30 min) |
| Total session | ~$0.30 |

## Troubleshooting

### Check Tools First

```powershell
.\runpod.ps1 check-tools
```

This verifies:
- WSL installed and working
- runpodctl available
- SSH key exists
- API key configured

### Common Issues

**"container not found"**
- Check pod ID matches in SSH proxy address
- Pod may still be starting - wait 30-60 seconds
- Run `.\runpod.ps1 list-pods` to verify status

**"Connection refused"**
- Use SSH proxy (automatic) not direct TCP
- Wait for pod to fully start

**"Your SSH client doesn't support PTY"**
- Script should use WSL automatically
- Verify WSL is installed: `wsl --status`

**"Permission denied"**
- SSH key not added to RunPod
- Run `.\runpod.ps1 check-tools` to verify

**"CUDA out of memory"**
- Run `.\runpod.ps1 exec 'fuser -k /dev/nvidia*'` to clear GPU
- Or reduce batch size in config

## Resources

- Project: https://github.com/p-e-w/heretic
- Paper: https://arxiv.org/abs/2406.11717
- Models: https://huggingface.co/collections/p-e-w/the-bestiary
- Roadmap: [ROADMAP.md](ROADMAP.md)

---

## Running Experiments

Heretic includes an experiments framework for testing new behavioral directions.

### Verbosity Spike Experiment

Test whether a "verbosity direction" exists and can be extracted:

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

### Verbosity Spike Results (Completed)

The verbosity spike was run on Vast.ai (2x A100-80GB) with 50 Optuna trials.

**Key Findings:**
- ✅ Factual questions get concise responses (6-30 words)
- ⚠️ Open-ended questions remain verbose (195-203 words)
- The ablation reduces elaboration on simple facts but doesn't suppress natural elaboration on complex topics

**Model Location:** `./models/Qwen2.5-7B-Instruct-heretic` (15.2 GB)

**To test locally (requires RTX 4070 or better with 8GB+ VRAM):**
```bash
# Use system Python (not uv run) to ensure CUDA works
python experiments/verbosity/test_comparison_4bit.py
```
