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

| Model Size | Minimum Disk | Recommended | Notes |
|------------|--------------|-------------|-------|
| 7B | 100GB | 150GB | C4 adds ~50GB |
| 13B | 150GB | 200GB | C4 adds ~50-100GB |
| 32B | **400GB** | **500GB** | C4 adds ~65-150GB |
| 70B | **500GB** | **600GB** | C4 adds ~100-200GB |

**C4 Dataset Disk Issue:**
- C4 (Colossal Clean Crawled Corpus) is ~800GB total for "en" variant
- Even small splits like `train[:200]` download 65-150GB due to shard-based downloading
- HuggingFace downloads entire shards (~320MB compressed each), not individual examples
- 200GB disk will run out of space during C4 download for 32B models

**Why C4 is so large:**
- 1024 shards, ~356,000 examples per shard
- Examples distributed across all shards for balanced sampling
- Small splits trigger downloading many shards to get representative sample

**Example: 32B model with orthogonalization**
```
Qwen2.5-Coder-32B model:     62GB
C4 dataset (train[:200]):    65-150GB (partial, 21% of shards)
BART-MNLI (neural detect):   1.6GB
Working space:               20GB
─────────────────────────────────────
TOTAL:                       150-230GB
```

**Solutions if you hit disk limit:**
1. **Recreate with 400GB+ disk** (recommended)
2. Reduce C4 split: `--unhelpfulness-prompts.split "train[:50]"`
3. Use smaller dataset instead of C4
4. Disable orthogonalization: `--orthogonalize-directions false`

**Important:** Default 100GB is insufficient for 32B+ models with orthogonalization.

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
