# Heretic RunPod Workflow

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
