# Heretic Quick Reference Card

Essential commands and flags for daily use.

## Common Commands

```bash
# Build & deploy
uv build
scp -P <PORT> dist/heretic_llm-*.whl root@<HOST>:/workspace/
uv run heretic-vast exec "pip install /workspace/heretic_llm-*.whl --force-reinstall"

# Instance management
uv run heretic-vast create <GPU_TYPE> <COUNT>
uv run heretic-vast list
uv run heretic-vast start
uv run heretic-vast stop
uv run heretic-vast terminate

# Monitoring
uv run heretic-vast watch          # Live dashboard (best)
uv run heretic-vast progress        # Quick status
uv run heretic-vast exec "tail -f /workspace/heretic.log"  # Raw logs

# Download results
uv run heretic-vast models
uv run heretic-vast download <MODEL_NAME>
```

## Run Commands by Model Size

### 7B Models
```bash
heretic --model Qwen/Qwen2.5-7B-Instruct \
  --auto-select true \
  --n-trials 50 \
  --compile
```

### 32B Models
```bash
heretic --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --auto-select true \
  --cache-weights false \
  --unhelpfulness-prompts.config en \
  --storage sqlite:///study.db \
  --study-name qwen32b \
  --n-trials 200
```

### 70B Models
```bash
heretic --model Qwen/Qwen2.5-70B-Instruct \
  --auto-select true \
  --cache-weights false \
  --unhelpfulness-prompts.config en \
  --batch-size 1 \
  --max-batch-size 16 \
  --storage sqlite:///study.db \
  --study-name qwen70b \
  --n-trials 200
```

## Critical Flags

| Flag | When to Use |
|------|-------------|
| `--cache-weights false` | **Required for 13B+** |
| `--unhelpfulness-prompts.config en` | **Required for 32B+** (C4 dataset) |
| `--auto-select true` | Recommended (auto-saves best trial) |
| `--storage sqlite:///study.db` | Recommended (resume support) |
| `--compile` | Optional (1.5-2x inference speedup) |

## Model Size Guidelines

**v1.1.0+ with C4 Streaming:**

| Size | GPU | Disk | Cache Weights | C4 Config |
|------|-----|------|---------------|-----------|
| 7B | 24GB | 100GB | ✅ Yes | ❌ No |
| 13B | 24GB | 150GB | ❌ No | ❌ No |
| 32B | 80GB | **200GB** | ❌ No | ✅ Yes |
| 70B | 80GB | **300GB** | ❌ No | ✅ Yes |

**Note:** v1.1.0+ streams C4 on-demand (~0GB overhead). Network required during dataset loading.

**Legacy (v1.0.x):** C4 downloaded 65-150GB. Use 400GB+ disk or upgrade to v1.1.0+.

## Troubleshooting

```bash
# Process not running?
uv run heretic-vast exec "ps aux | grep 'heretic --model'"

# Check errors
uv run heretic-vast exec "tail -100 /workspace/heretic.log | grep -i error"

# GPU OOM?
uv run heretic-vast exec "dmesg | grep -i 'out of memory'"

# Clear cache after reinstall
uv run heretic-vast exec "find /usr/local/lib/python3.10/dist-packages/heretic -name '*.pyc' -delete"

# Restart fresh
uv run heretic-vast exec "pkill -f heretic && rm /workspace/heretic.log"
```

## Performance Stats (H200)

| Operation | Time | Notes |
|-----------|------|-------|
| Model download (32B) | ~5 min | One-time, cached |
| PCA extraction (32B) | **~5 min** | GPU optimized (was 4-6 hrs) |
| Single trial (32B) | ~3 min | Varies by model |
| 200 trials (32B) | ~10 hrs | Total cost: ~$22 |

## Cost Quick Math

```
Cost = GPU_rate × (setup_time + trial_time × n_trials)

Example (32B on H200 @ $2.14/hr):
= $2.14 × (0.5hr + 0.05hr × 200)
= $2.14 × 10.5hr
= $22.47
```

## Git Workflow

```bash
# Update docs
git add CLAUDE.md GPU_PCA_IMPLEMENTATION.md
git commit -m "Document <change>"

# Push to fork (abliteration-workflow)
git push fork master

# NEVER push to main heretic repo
# git push origin master  # ❌ WRONG
```

## Emergency Stops

```bash
# Stop instance (KILLS process)
uv run heretic-vast stop

# Kill process only (keeps instance)
uv run heretic-vast exec "pkill -f 'heretic --model'"

# Download before stopping
uv run heretic-vast models
uv run heretic-vast download <MODEL>
uv run heretic-vast stop
```
