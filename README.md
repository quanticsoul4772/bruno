# Heretic on RunPod

## Directory Structure

```
C:\Development\Projects\heretic\
|-- README.md              - This file
|-- runpod.ps1             - Windows automation
|-- config.toml            - Experiment config
|-- config.default.toml    - Reference config
`-- src/                   - Source code
```

## RunPod Pod Configuration

**Template**: PyTorch 2.x
**GPU**: RTX 5090 (recommended) or RTX 4090
**Volume**: 50GB+
**Port**: TCP 22

### GPU Options

| GPU | VRAM | Cost/hr | 8B Model Time |
|-----|------|---------|---------------|
| RTX 5090 | 32GB | $0.50-0.89 | ~17 min |
| RTX 4090 | 24GB | $0.34 | ~30 min |
| RTX 3090 | 24GB | $0.22-0.43 | ~45 min |
| L40S | 48GB | $0.70-1.00 | ~23 min |

## Setup

Edit `runpod.ps1`:
```powershell
$RUNPOD_HOST = "abc123xyz.runpod.io"  # Your pod hostname
$RUNPOD_PORT = "12345"                 # Your SSH port
```

Initialize:
```powershell
cd C:\Development\Projects\heretic
.\runpod.ps1 setup
```

## Usage

### Test Installation
```powershell
.\runpod.ps1 test
```
Runs Qwen3-4B. Time: 10-15 min (RTX 5090), 20 min (RTX 4090), 30 min (RTX 3090)

### Process Model
```powershell
.\runpod.ps1 run meta-llama/Llama-3.1-8B-Instruct
```
Time: 17 min (RTX 5090), 30 min (RTX 4090), 45 min (RTX 3090)

### GPU Monitoring
```powershell
.\runpod.ps1 monitor
```

### SSH Connection
```powershell
.\runpod.ps1 connect
```

### File Operations
```powershell
.\runpod.ps1 upload <file>
.\runpod.ps1 download <file>
.\runpod.ps1 sync  # Upload config.toml
```

## Model Examples

**Small (4-8B)**
- `Qwen/Qwen3-4B-Instruct-2507` (10-15 min on RTX 5090)
- `meta-llama/Llama-3.1-8B-Instruct` (17 min on RTX 5090)
- `mistralai/Mistral-7B-Instruct-v0.3` (15 min on RTX 5090)

**Medium (13B)**
- `meta-llama/Llama-2-13b-chat-hf` (25 min on RTX 5090)

**Large (30B+)**
Requires L40S (48GB) or A100 (80GB)

## Configuration

Edit `config.toml`:
- `n_trials`: Optimization iterations (default 200)
- `batch_size`: 0 for auto-detect
- `max_batch_size`: 128 default, 256 for RTX 5090
- `dtypes`: Model precision fallback order

Sync changes:
```powershell
.\runpod.ps1 sync
```

## RTX 5090 Optimization

```toml
batch_size = 0          # Auto-detect
max_batch_size = 256    # Increase from default 128
dtypes = ["auto"]       # Uses bfloat16
```

## Troubleshooting

**OOM errors**: Use smaller model, reduce batch_size, or try float16
**Slow performance**: Check GPU utilization with `.\runpod.ps1 status`
**Connection issues**: Verify pod is running, check firewall

## Cost Comparison (8B Model)

```
RTX 5090: $0.89/hr x 0.28hr = $0.25 per run
RTX 4090: $0.34/hr x 0.50hr = $0.17 per run
RTX 3090: $0.43/hr x 0.75hr = $0.32 per run
```

## Post-Processing

After abliteration:
1. Save locally (RunPod volume)
2. Upload to HuggingFace (requires token)
3. Chat test (interactive)

Download:
```powershell
.\runpod.ps1 download output/
```

## References

- Project: https://github.com/p-e-w/heretic
- Paper: https://arxiv.org/abs/2406.11717
- Models: https://huggingface.co/collections/p-e-w/the-bestiary
- RTX 5090 Benchmarks: https://www.runpod.io/blog/rtx-5090-llm-benchmarks
