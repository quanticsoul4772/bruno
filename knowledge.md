# Project Knowledge

Heretic is a tool for automatic censorship removal (abliteration) from language models using Optuna-based hyperparameter optimization.

## Quickstart

```bash
# Install dependencies (requires uv)
uv sync --all-extras --dev

# Run the tool
uv run heretic <model-name>
# or
uv run heretic --model <model-name>
```

## Commands

```bash
# Format code
uv run ruff format .

# Lint and check imports
uv run ruff check --extend-select I .

# Build package
uv build
```

## Architecture

- `src/heretic/main.py` - Entry point and optimization loop
- `src/heretic/model.py` - Model loading, abliteration, residual extraction
- `src/heretic/evaluator.py` - KL divergence and refusal counting
- `src/heretic/config.py` - Pydantic settings (CLI > env > config.toml > defaults)
- `src/heretic/utils.py` - Helper functions
- `config.default.toml` - Reference configuration file

### Core Flow

1. Load model via `Model` class with dtype fallback chain
2. Load good/bad prompt datasets
3. Auto-detect optimal batch size if `batch_size=0`
4. Compute per-layer refusal directions from residual activations
5. Run Optuna multi-objective optimization (minimize KL divergence + refusals)
6. Present Pareto-optimal trials for user selection
7. Save/upload/chat with abliterated model

## Conventions

- **Formatting/linting**: Ruff (format + lint with import sorting)
- **Package manager**: uv
- **Settings cascade**: CLI args > env vars (HERETIC_ prefix) > config.toml > defaults
- **Python version**: >=3.10
- **License**: AGPL-3.0-or-later

## Key Dependencies

- `transformers` - Model loading
- `optuna` - Hyperparameter optimization
- `accelerate` - Multi-GPU/device support
- `pydantic-settings` - Configuration management
- `rich` - Terminal output formatting
- `questionary` - Interactive prompts

## Gotchas

- No test suite currently exists (CI only runs format, lint, build)
- GPU required for reasonable performance (CUDA, XPU, MLU, MPS supported)
- `batch_size=0` triggers auto-detection
- KL divergence values above 1 usually indicate model damage

---

# RunPod Deployment Guide

## The Complete Solution

We use a **hybrid approach** that leverages the best of each tool:

| Tool | Best For | Limitation |
|------|----------|------------|
| **runpodctl** | Pod management, SSH info | No general command execution |
| **WSL + SSH heredoc** | Command execution | Requires WSL |
| **GraphQL API** | Advanced operations | More verbose |

**Key insight:** RunPod's SSH proxy requires a pseudo-terminal (PTY), which Windows SSH cannot provide in non-interactive mode. The `runpodctl exec` command only supports Python scripts, not general shell commands. The solution is to use WSL with heredoc syntax for command execution.

## Quick Start (3 Commands)

```powershell
# 1. Create pod (auto-configures SSH)
.\runpod.ps1 create-pod

# 2. Setup heretic (automated via WSL)
.\runpod.ps1 setup

# 3. Run abliteration
.\runpod.ps1 run Qwen/Qwen3-4B-Instruct-2507
```

That's it! The script handles everything automatically.

## Prerequisites

1. **WSL installed** - Run `wsl --install` in admin PowerShell
2. **runpodctl installed** - Run `.\runpod.ps1 install-runpodctl`
3. **SSH key** - At `~/.ssh/id_ed25519`
4. **RunPod API key** - Get from https://runpod.io/console/user/settings

Verify all tools: `.\runpod.ps1 check-tools`

## How It Works

### Tool Responsibilities

**runpodctl** (fast, reliable for pod management):
```powershell
# List pods
.\runpodctl.exe get pod

# Get SSH connection info
.\runpodctl.exe ssh connect

# Start/stop pods
.\runpodctl.exe start pod <id>
.\runpodctl.exe stop pod <id>
```

**GraphQL API** (fallback, advanced operations):
- Pod creation with custom configuration
- Getting user ID for SSH proxy address
- Operations when runpodctl is unavailable

### The WSL + Heredoc Solution (for command execution)

```powershell
# This is what happens under the hood:
powershell -Command "wsl -e bash -c 'ssh -tt -o StrictHostKeyChecking=no podid-userid@ssh.runpod.io <<EOF
echo Hello World
python --version
exit
EOF'"
```

**Why this works:**
1. `wsl -e bash` - Runs bash in WSL, which has proper PTY support
2. `ssh -tt` - Forces TTY allocation
3. `<<EOF...EOF` - Heredoc pipes commands while maintaining the TTY
4. The SSH proxy sees a real PTY and accepts the connection

**Why other approaches fail:**
- Windows SSH without WSL: No PTY → "Your SSH client doesn't support PTY"
- Direct TCP port: Often unreachable even when pod is running
- winpty: Still fails with "stdin is not a tty"

### SSH Key in WSL

The script automatically copies your SSH key to WSL with correct permissions:
```bash
mkdir -p ~/.ssh
cp /mnt/c/Users/USERNAME/.ssh/id_ed25519 ~/.ssh/
chmod 600 ~/.ssh/id_ed25519
```

This is necessary because Windows NTFS permissions show as 0777 in WSL, which SSH rejects.

## Pod Configuration

| Setting | Recommended | Why |
|---------|-------------|-----|
| **Image** | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` | Compatible with transformers 4.55+ |
| **Container Disk** | 40GB minimum | 20GB fills with pip packages |
| **Volume** | 50GB+ | For model downloads |
| **GPU** | RTX 4090 (24GB) | Best cost/performance |

## Critical Environment Setup

**ALWAYS set HuggingFace cache to the volume:**
```bash
export HF_HOME=/workspace/.cache/huggingface
mkdir -p /workspace/.cache/huggingface
```

Without this, downloads go to `/root/.cache` on the container disk, which quickly fills up.

## GPU Memory Management

RunPod containers often have pre-loaded processes consuming GPU memory:
```bash
# Check what's using the GPU
nvidia-smi

# Kill all GPU processes (nuclear option)
fuser -k /dev/nvidia*
```

## PyTorch Compatibility

| Image | PyTorch | transformers | Heretic |
|-------|---------|--------------|----------|
| `pytorch:2.4.0-py3.11-cuda12.4.1` | 2.4.0 | 4.55+ ✅ | Works |
| `pytorch:2.1.0-py3.10-cuda12.1.1` | 2.1.0 | 4.55+ ❌ | Needs upgrade |

If using older image, upgrade PyTorch:
```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Troubleshooting

### "container not found" on SSH
- Wrong pod ID or user ID in SSH proxy address
- Pod container crashed (check RunPod console)
- Pod is still starting (wait 30-60 seconds)

### "Connection refused" on direct TCP
- RunPod direct TCP ports are often unreliable
- Use SSH proxy instead: `podid-userid@ssh.runpod.io`

### "Your SSH client doesn't support PTY"
- Use WSL: `wsl -e ssh ...`
- Or connect interactively from PowerShell

### "No space left on device"
```bash
# Set HF cache to volume
export HF_HOME=/workspace/.cache/huggingface
mkdir -p /workspace/.cache/huggingface

# Clear failed downloads
rm -rf /root/.cache/huggingface/hub/models--*
```

### "CUDA out of memory"
```bash
# Kill other GPU processes
fuser -k /dev/nvidia*

# Or reduce batch size in config
max_batch_size = 64  # instead of 128
```

### "Could not find Qwen3ForCausalLM"
```bash
# Upgrade transformers
pip install --upgrade transformers>=4.55.2
```

## Complete runpod.ps1 Command Reference

### Pod Management
```powershell
.\runpod.ps1 create-pod              # Create RTX 4090 pod
.\runpod.ps1 create-pod "NVIDIA RTX A6000"  # Specific GPU
.\runpod.ps1 list-pods               # Show all pods
.\runpod.ps1 stop-pod                # Stop (preserves volume)
.\runpod.ps1 start-pod               # Restart stopped pod
.\runpod.ps1 terminate-pod <id>      # Delete permanently
.\runpod.ps1 gpus                    # List available GPUs
```

### Heretic Operations
```powershell
.\runpod.ps1 setup                   # Install heretic via WSL
.\runpod.ps1 test                    # Run Qwen3-4B test
.\runpod.ps1 run <model>             # Abliterate any model
.\runpod.ps1 exec 'command'          # Run any command
.\runpod.ps1 status                  # GPU status
```

### Interactive
```powershell
.\runpod.ps1 connect                 # Interactive SSH
.\runpod.ps1 monitor                 # Live GPU monitoring
```

## Cost Estimates

| GPU | Price | 4B Model | 8B Model |
|-----|-------|----------|----------|
| RTX 4090 | $0.34/hr | ~15 min ($0.09) | ~30 min ($0.17) |
| RTX 3090 | $0.22/hr | ~30 min ($0.11) | ~45 min ($0.17) |
| RTX A6000 | $0.53/hr | ~12 min ($0.11) | ~20 min ($0.18) |

## Debugging Journey Summary

We went through extensive debugging to find the WSL solution:

1. **Direct TCP ports** - Often show "connection refused" even when pod is running
2. **SSH proxy** - Works for interactive sessions but fails with "PTY not supported" for automation
3. **winpty** - Failed with "stdin is not a tty"
4. **ssh -t -t** - Allocates TTY but hangs or times out
5. **WSL + heredoc** - ✅ Works! Proper PTY from Linux environment

The key insight: RunPod's SSH proxy specifically checks for PTY and Windows SSH (even with -t flag) doesn't properly allocate one in non-interactive mode. WSL provides a real Linux environment where PTY allocation works correctly.
