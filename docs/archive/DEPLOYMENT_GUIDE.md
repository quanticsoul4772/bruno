# Heretic Cloud GPU Deployment Guide

Quick reference for deploying heretic on Vast.ai or other cloud GPU providers.

## Prerequisites

- `uv` installed locally
- Vast.ai API key in `.env` file
- SSH key configured for Vast.ai

## Quick Start (Recommended Path)

### 1. Build Wheel Locally

```bash
# Ensure latest changes
git pull

# Build wheel
uv build

# Verify config.default.toml is included
unzip -l dist/heretic_llm-*.whl | grep toml
```

### 2. Create GPU Instance

```bash
# List available GPUs
uv run heretic-vast tiers

# Create instance (example: H200)
uv run heretic-vast create H200 1

# Wait ~30 seconds for startup
sleep 30

# Check status
uv run heretic-vast list
```

### 3. Deploy Heretic

```bash
# Get instance details
uv run heretic-vast list

# Upload wheel
scp -P <PORT> -o StrictHostKeyChecking=no \
    dist/heretic_llm-*.whl \
    root@<HOST>:/workspace/

# Install (SSH in or use heretic-vast exec)
uv run heretic-vast exec "pip install /workspace/heretic_llm-*.whl --force-reinstall"
```

### 4. Run Abliteration

**For 7B models:**
```bash
uv run heretic-vast exec "
export HF_HOME=/workspace/.cache/huggingface
cd /workspace
nohup heretic \\
  --model Qwen/Qwen2.5-7B-Instruct \\
  --auto-select true \\
  --auto-select-path /workspace/models \\
  --storage sqlite:///heretic_study.db \\
  --study-name qwen7b \\
  --n-trials 50 \\
  --compile \\
  > /workspace/heretic.log 2>&1 &
"
```

**For 32B models:**
```bash
uv run heretic-vast exec "
export HF_HOME=/workspace/.cache/huggingface
cd /workspace
nohup heretic \\
  --model Qwen/Qwen2.5-Coder-32B-Instruct \\
  --auto-select true \\
  --auto-select-path /workspace/models \\
  --storage sqlite:////workspace/heretic_study.db \\
  --study-name qwen32b \\
  --cache-weights false \\
  --n-trials 200 \\
  --unhelpfulness-prompts.config en \\
  > /workspace/heretic.log 2>&1 &
"
```

### 5. Monitor Progress

**Option 1: Live dashboard (recommended)**
```bash
uv run heretic-vast watch
```

**Option 2: Follow logs**
```bash
uv run heretic-vast exec "tail -f /workspace/heretic.log"
```

**Option 3: Check progress**
```bash
uv run heretic-vast progress
```

### 6. Download Results

```bash
# List available models
uv run heretic-vast models

# Download specific model
uv run heretic-vast download <MODEL_NAME>

# Or download all models
uv run heretic-vast exec "tar -czf /workspace/models.tar.gz /workspace/models/"
scp -P <PORT> root@<HOST>:/workspace/models.tar.gz ./
```

### 7. Cleanup

```bash
# Stop instance (KILLS running processes!)
uv run heretic-vast stop

# Or terminate permanently
uv run heretic-vast terminate
```

## Critical Flags for Different Model Sizes

| Model Size | Required Flags | Notes |
|------------|----------------|-------|
| 7B | None | Can use default settings |
| 13B | `--cache-weights false` | Weight caching uses too much memory |
| 32B+ | `--cache-weights false`<br>`--unhelpfulness-prompts.config en` | C4 config required for orthogonalization |
| 70B+ | `--cache-weights false`<br>`--unhelpfulness-prompts.config en`<br>`--batch-size 1` | May need smaller batch size |

## Common Issues & Solutions

### Issue: `ValueError: Config name is missing`

**Cause:** C4 dataset requires explicit config parameter.

**Solution:** Add `--unhelpfulness-prompts.config en` to command.

### Issue: GPU Out of Memory

**Solutions:**
1. Disable weight caching: `--cache-weights false`
2. Reduce batch size: `--batch-size 1` or `--max-batch-size 32`
3. Use larger GPU tier

### Issue: Changes not taking effect after wheel reinstall

**Cause:** Python cache not cleared.

**Solution:**
```bash
uv run heretic-vast exec "
find /usr/local/lib/python3.10/dist-packages/heretic -name '*.pyc' -delete
pkill -f 'heretic --model'
"
```

### Issue: Process crashed but no error in log

**Check:**
```bash
# Check if process is actually running
uv run heretic-vast exec "ps aux | grep 'heretic --model'"

# Check last 100 lines of log
uv run heretic-vast exec "tail -100 /workspace/heretic.log"

# Check for OOM in system logs
uv run heretic-vast exec "dmesg | tail -20"
```

## Performance Tips

### GPU PCA Optimization (v1.0.1+)

The GPU PCA optimization provides **15-20x speedup** for large models:
- 32B models: PCA completes in ~5 minutes (was 4-6 hours)
- Automatic - no configuration needed
- Keeps tensors on GPU throughout computation

### Other Optimizations

```bash
# Enable torch.compile (1.5-2x inference speedup)
--compile

# Resume interrupted runs
--storage sqlite:///study.db --study-name <NAME>

# Faster refusal detection (30 tokens instead of full generation)
--refusal-check-tokens 30

# Auto-select best trial
--auto-select true --auto-select-path /workspace/models
```

## Disk Space Requirements

### Without Orthogonalization (--orthogonalize-directions false)

| Model Size | Minimum Disk | Recommended |
|------------|--------------|-------------|
| 7B | 50GB | 100GB |
| 13B | 75GB | 150GB |
| 32B | 150GB | 200GB |
| 70B | 300GB | 400GB |

**Formula:** `(model_size × 2) + 20GB` for model + cache + results

### With Orthogonalization (Default - Uses C4 Dataset)

**v1.1.0+ with C4 Streaming:**

| Model Size | Minimum Disk | Recommended | Notes |
|------------|--------------|-------------|-------|
| 7B | 75GB | 100GB | Streaming eliminates C4 overhead |
| 13B | 100GB | 150GB | Streaming eliminates C4 overhead |
| 32B | **200GB** | **250GB** | ✅ C4 streams on-demand (~0GB) |
| 70B | **300GB** | **400GB** | ✅ C4 streams on-demand (~0GB) |

**C4 Streaming (v1.1.0+):**
- ✅ **C4 dataset now streams automatically** - no disk space overhead
- Downloads only the exact examples needed as they're requested
- Requires network connectivity during dataset loading
- No configuration changes needed - works automatically for any dataset with "c4" in name

**Example: 32B model with orthogonalization (v1.1.0+)**
```
Qwen2.5-Coder-32B model:     65.5GB
C4 dataset (streaming):      ~0GB (on-demand)
BART-MNLI (neural detect):   1.6GB
Other datasets:              0.1GB
Working space:               20GB
─────────────────────────────────────
TOTAL:                       ~87GB ✅
```

**Disk Space Savings:** C4 streaming eliminates 30-50GB download overhead, reducing total requirements from 150-230GB to ~87GB for 32B models.

---

### Legacy: v1.0.x Disk Requirements (Pre-Streaming)

**Only relevant if using v1.0.x or earlier:**

| Model Size | Minimum Disk | Recommended | Notes |
|------------|--------------|-------------|-------|
| 32B | **400GB** | **500GB** | v1.0.x downloads C4 shards |
| 70B | **500GB** | **600GB** | v1.0.x downloads C4 shards |

**Legacy C4 Issue (v1.0.x):**
- C4 dataset downloaded 65-150GB even for small splits
- Caused "No space left on device" errors on 200GB disks
- **Solution:** Upgrade to v1.1.0+ for streaming support

**Legacy Solutions (if stuck on v1.0.x):**
1. Recreate with 400GB+ disk
2. Reduce C4 split: `--unhelpfulness-prompts.split "train[:50]"`
3. Disable orthogonalization: `--orthogonalize-directions false`

## Cost Optimization

### Choose Right GPU Tier

| Model Size | Minimum GPU | Cost/Hour | Notes |
|------------|-------------|-----------|-------|
| 7B | RTX 3090 (24GB) | $0.20-0.40 | Good for testing |
| 13B | RTX 4090 (24GB) | $0.40-0.70 | Best value |
| 32B | A100 (80GB) | $1.50-2.50 | Or H100/H200 |
| 70B | H100 (80GB) | $2.00-4.00 | Requires 80GB+ |

### Reduce Runtime

1. **Start with fewer trials:** Test with `--n-trials 10` first
2. **Use resume:** Save progress with `--storage` to avoid restarts
3. **Monitor closely:** Stop as soon as results are acceptable
4. **Check progress:** `heretic-vast progress` shows trial count

### Example Cost Calculation

**32B model on H200 ($2.14/hr):**
- Model download: ~5 minutes
- Baseline validation: ~10 minutes
- PCA extraction: ~5 minutes (GPU optimized)
- C4 dataset download: ~20 minutes
- 200 trials @ ~3 min/trial: ~10 hours

**Total:** ~11 hours = **$23.54**

## Monitoring Checklist

Before leaving abliteration running:

- [ ] Process is actually running (`heretic-vast progress`)
- [ ] GPU utilization >0% (model loaded)
- [ ] No errors in last 50 log lines
- [ ] Trial 1 completes successfully
- [ ] Storage/study name configured for resume
- [ ] Estimated cost calculated based on n_trials

## Emergency Recovery

### Resume interrupted run

```bash
# Same command with same --storage and --study-name
heretic --model <MODEL> \\
  --storage sqlite:///heretic_study.db \\
  --study-name <SAME_NAME> \\
  # ... other flags ...
```

### Download partial results

```bash
# Even if not complete, download what exists
uv run heretic-vast exec "ls -la /workspace/models/"
uv run heretic-vast download <MODEL_NAME>
```

### Check Optuna database

```bash
uv run heretic-vast exec "sqlite3 /workspace/heretic_study.db 'SELECT COUNT(*) FROM trials;'"
```

## Best Practices

1. **Test locally first:** Use 7B model on local GPU to validate setup
2. **Start small:** Begin with 10-20 trials, scale up after validation
3. **Monitor actively:** Check progress every 30 minutes for first hour
4. **Save early:** Configure `--storage` and `--study-name` from the start
5. **Document commands:** Save exact command used for reproducibility
6. **Watch costs:** Set a budget and estimate runtime before starting
7. **Verify downloads:** Check model files are complete before terminating instance

## Troubleshooting Commands

```bash
# Check GPU status
uv run heretic-vast status

# Check disk space
uv run heretic-vast exec "df -h /workspace"

# Check memory usage
uv run heretic-vast exec "free -h"

# Check running processes
uv run heretic-vast exec "ps aux | grep python"

# Check Optuna trials
uv run heretic-vast exec "cd /workspace && python -c 'import optuna; study = optuna.load_study(study_name=\"qwen32b\", storage=\"sqlite:///heretic_study.db\"); print(f\"Trials: {len(study.trials)}\")'"

# Kill stuck process
uv run heretic-vast exec "pkill -f 'heretic --model'"

# Restart fresh
uv run heretic-vast exec "pkill -f heretic && rm /workspace/heretic.log && <START_COMMAND>"
```

## Support & Documentation

- Main docs: `CLAUDE.md`
- GPU optimization: `GPU_PCA_IMPLEMENTATION.md`
- Workflow reference: `WORKFLOW.md`
- Issue tracker: https://github.com/p-e-w/heretic/issues
