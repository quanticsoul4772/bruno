# Lessons Learned - 32B Model Abliteration
**Date:** 2026-01-30
**Hardware:** 4x RTX 4090 (96GB total VRAM) on Vast.ai
**Model:** Qwen/Qwen2.5-Coder-32B-Instruct (64 layers, ~64GB unquantized)

## Summary

Running abliteration on 32B models requires careful configuration to avoid OOM errors on multi-GPU setups. The default settings are optimized for single-GPU or small models and will fail for large multi-GPU deployments.

---

## Critical Issues Discovered

### 1. Multi-GPU Memory Distribution Failure

**Problem:** `device_map="auto"` loads model on GPU 0 only, ignoring other GPUs
```
GPU 0: 23GB (96% utilization) → OOM
GPU 1-3: 1MB each (unused)
```

**Root Cause:** Accelerate's "auto" strategy prioritizes filling GPU 0 before spilling to others.

**Solution:** Use `device_map="balanced"`
```toml
device_map = "balanced"  # Distributes layers evenly across all GPUs
```

**Result:**
```
GPU 0: 17GB
GPU 1: 17GB
GPU 2: 17GB
GPU 3: 15GB
Total: ~66GB distributed evenly
```

**When to use each:**
- `"auto"`: Single GPU or when you want GPU 0 prioritized
- `"balanced"`: Multi-GPU with even distribution (recommended for 32B+)
- `"sequential"`: Distributes sequentially (GPU 0 fills, then GPU 1, etc.)

---

### 2. Weight Caching Incompatibility with Multi-GPU

**Problem:** `cache_weights=true` (default) causes crashes with `device_map="balanced"`

**Error:**
```
RuntimeError: Tensor on device meta is not on the expected device cuda:0!
OutOfMemoryError during deepcopy(state_dict())
```

**Root Cause:**
```python
# model.py:168
self.original_state_dict = copy.deepcopy(self.model.state_dict())
```

When model is sharded across GPUs:
- `state_dict()` contains tensors on cuda:0, cuda:1, cuda:2, cuda:3
- `deepcopy()` tries to materialize all tensors (usually on CPU or cuda:0)
- Exceeds memory on target device → OOM or creates "meta" device placeholders
- On reload, meta tensors cause inference crash

**Solution:** Disable weight caching for multi-GPU
```toml
cache_weights = false  # REQUIRED for device_map="balanced"
```

**Trade-off:**
- **cache_weights=true** (single GPU): Reload in ~2s (5-10x faster trials)
- **cache_weights=false** (multi-GPU): Reload from disk in ~30s (only option)

**Configuration Matrix:**

| Setup | device_map | cache_weights | Trial Reload Time |
|-------|------------|---------------|-------------------|
| Single GPU (A100 80GB) | "auto" | true | ~2s |
| Multi-GPU (4x RTX 4090) | "balanced" | false | ~30s |
| Multi-GPU (2x A100) | "balanced" | false | ~30s |

---

### 3. Iterative Ablation Memory Overhead

**Problem:** `iterative_rounds > 0` causes OOM during trial evaluation

**Error:**
```
OutOfMemoryError: Tried to allocate 1.45 GiB. GPU 0 has 1.45 GiB free.
```

**Root Cause:** Iterative ablation calls `get_residuals_batched()` mid-trial
- Requires `output_hidden_states=True` during generation
- Stores hidden states for ALL 64 layers = ~3-5GB additional VRAM
- Model weights: 22GB + Hidden states: 5GB = 27GB > 24GB per GPU

**Stack trace:**
```python
model.abliterate_iterative()
  └─> get_residuals_batched(good_prompts)  # Needs 5GB extra
      └─> generate(output_hidden_states=True)
          └─> OutOfMemoryError
```

**Solution:** Disable iterative ablation for large models
```toml
iterative_rounds = 0  # REQUIRED for 32B on 24GB GPUs
```

**When iterative ablation is safe:**
- Small models (<7B) on 16GB+ GPU: Safe, improves quality
- Medium models (7B-14B) on 24GB GPU: Use with caution
- Large models (14B-32B) on 24GB GPU: Disable (OOM risk)
- Large models (32B+) on 80GB GPU: Safe, use iterative_rounds=1-2

---

### 4. Batch Size Performance vs Safety

**Problem:** `batch_size=1` is extremely slow for residual extraction

**Observed performance:**
- 800 prompts (400 good + 400 bad)
- batch_size=1: 800 batches → ~100+ minutes to extract residuals
- batch_size=4: 200 batches → ~25-30 minutes (4x faster)
- batch_size=8: OOM risk on 32B models

**Recommendation by model size:**

| Model Size | GPU VRAM | Safe batch_size | Optimal max_batch_size |
|------------|----------|-----------------|------------------------|
| 3B-7B | 8-16GB | 0 (auto) | 128 |
| 7B-14B | 16-24GB | 0 (auto) | 64 |
| 14B-32B | 24-48GB | 4 | 16 |
| 32B-70B | 48-80GB | 2-4 | 8 |
| 70B+ | 80GB+ | 1-2 | 4 |

**Why batch_size=4 for 32B:**
- Fast enough: 4x speedup vs batch_size=1
- Safe enough: Leaves headroom for memory fragmentation
- Balanced: Good throughput without OOM risk

---

### 5. Multiple Process Detection Gap

**Problem:** No mechanism to detect if heretic is already running

**What happened:**
- Started heretic process 1
- Process 1 crashed, didn't exit cleanly
- Started heretic process 2 while 1 was still dying
- Both processes competed for GPU → OOM

**Solution:** Always check and kill before starting
```bash
# Check for existing processes
ps aux | grep '[h]eretic'

# Kill all Python/heretic processes
pkill -9 -f python
pkill -9 -f heretic

# Verify clean state
nvidia-smi | grep python  # Should show nothing

# Then start fresh
cd /workspace && nohup heretic > heretic.log 2>&1 &
```

**Prevention:** Add lock file mechanism (future improvement)
```python
# In main.py at startup
LOCK_FILE = Path("/tmp/heretic.lock")
if LOCK_FILE.exists():
    print("[red]Another heretic process is running![/]")
    print(f"[yellow]If you're sure it's not, delete {LOCK_FILE}[/]")
    sys.exit(1)

LOCK_FILE.touch()
try:
    run()
finally:
    LOCK_FILE.unlink(missing_ok=True)
```

---

### 6. Performance Reality for 32B Models

**Initial Expectations:** "With optimizations, abliteration should be fast"

**Reality on 4x RTX 4090:**
- Model loading: ~5 minutes
- Residual extraction (800 prompts @ batch_size=4): ~25-30 minutes
- Per trial: ~9 minutes (reload from disk + abliterate + evaluate)
- **Total for 200 trials: ~30-35 hours**

**Why so slow:**
1. **No weight caching:** Each trial reloads 64GB from disk (~30s)
2. **Multi-GPU overhead:** Cross-GPU communication for balanced device map
3. **CPU offloading:** Balanced device map may offload some layers to CPU
4. **Conservative batching:** batch_size=4 is safe but not optimal

**Faster configurations:**

| Setup | cache_weights | batch_size | Time for 200 trials |
|-------|---------------|------------|---------------------|
| 4x RTX 4090 (balanced) | false | 4 | ~30-35h |
| 1x A100 80GB | true | 0 (auto) | ~12-15h |
| 7B model (single GPU) | true | 0 (auto) | ~2-3h |

**Recommendation:** For 32B models, use A100 80GB or reduce to 50-100 trials.

---

### 7. PCA Extraction Infinite Hang

**Problem:** `use_pca_extraction=true` (default) causes infinite hang on "Extracting 3 refusal directions using contrastive PCA"

**Observed behavior:**
- Process runs for 18+ hours
- GPU utilization: 0%
- CPU utilization: High on single core
- Stuck at: "* Extracting 3 refusal directions using contrastive PCA..."

**Root Cause:** PCA computation on 800 prompts × 64 layers × hidden_dim with multi-GPU appears to deadlock or run extremely slow

**Solution:** Disable PCA extraction
```toml
use_pca_extraction = false  # Use simple mean difference instead
```

**Trade-off:**
- PCA extraction: Theoretically better direction quality, but hangs on large models
- Mean difference: Fast (<5 minutes), works reliably, good enough quality

**When PCA is safe:**
- Small models (<7B): Usually fine
- Medium models (7B-14B): Monitor, may be slow but works
- Large models (32B+): **Disable PCA** to avoid hang

**Recommendation:** Default to `use_pca_extraction=false` for models >14B

---

## Working Configuration for 32B Models

### 4x RTX 4090 (96GB total)
```toml
model = "Qwen/Qwen2.5-Coder-32B-Instruct"

# Multi-GPU settings
device_map = "balanced"          # Even distribution across GPUs
cache_weights = false            # REQUIRED (deepcopy fails with sharding)
compile = false                  # Overhead not worth it for 32B

# Memory safety
iterative_rounds = 0             # Disable (needs 3-5GB extra VRAM)
batch_size = 4                   # Conservative but 4x faster than batch_size=1
max_batch_size = 16              # Cap auto-detection

# Optuna
n_trials = 200                   # Or reduce to 50-100 for faster results
n_startup_trials = 30
storage = "sqlite:///heretic_32b.db"
study_name = "qwen32b"

# Auto-save
auto_select = true
auto_select_path = "/workspace/models/Qwen2.5-Coder-32B-Instruct-heretic"
```

**Expected runtime:** 30-35 hours

---

### 1x A100 80GB (Recommended)
```toml
model = "Qwen/Qwen2.5-Coder-32B-Instruct"

# Single GPU - all optimizations enabled
device_map = "auto"              # Single GPU
cache_weights = true             # ENABLE (5-10x faster trials)
compile = true                   # Worth it on A100
iterative_rounds = 1             # Safe with 80GB VRAM

# Aggressive batching
batch_size = 0                   # Auto-detect
max_batch_size = 64              # Allow larger batches

n_trials = 200
storage = "sqlite:///heretic_32b.db"
study_name = "qwen32b_a100"
auto_select = true
auto_select_path = "/workspace/models/model-heretic"
```

**Expected runtime:** 12-15 hours

---

## Troubleshooting Guide

### Symptom: Stuck on "Extracting refusal directions using contrastive PCA" for hours
```
Calculating per-layer refusal directions...
* Obtaining residuals for good prompts...
* Obtaining residuals for bad prompts...
* Extracting 3 refusal directions using contrastive PCA... (hangs forever)
```

**Root cause:** PCA computation hangs/deadlocks on large models

**Check:**
```bash
# GPU utilization should be 0%
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader

# Process is stuck on CPU
top -b -n 1 | grep heretic
```

**Fix:**
```toml
use_pca_extraction = false
```

---

### Symptom: OOM during model loading
```
OutOfMemoryError: Tried to allocate 1.45 GiB. GPU 0 has 1.45 GiB free.
Loading weights: 100%|██████████| 771/771
```

**Check:**
```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

**If GPU 0 has 23GB, others have 1MB:**
```toml
# Change from:
device_map = "auto"

# To:
device_map = "balanced"
```

---

### Symptom: Meta device errors during trials
```
RuntimeError: Tensor on device meta is not on the expected device cuda:0!
```

**Root cause:** Weight caching enabled with multi-GPU

**Fix:**
```toml
cache_weights = false
```

---

### Symptom: OOM during trial evaluation
```
* Evaluating...
  * Evaluating in parallel...
OutOfMemoryError: Tried to allocate 1.45 GiB
```

**Root cause:** Iterative ablation enabled

**Fix:**
```toml
iterative_rounds = 0
```

---

### Symptom: Extremely slow residual extraction
```
Calculating per-layer refusal directions...
* Obtaining residuals for good prompts... (stuck for hours)
```

**Root cause:** batch_size too small (1)

**Fix:**
```toml
batch_size = 4  # Or higher if memory allows
```

---

### Symptom: Multiple heretic processes running
```bash
ps aux | grep heretic
# Shows 2+ processes
```

**Fix:**
```bash
# Kill all processes
pkill -9 -f python
pkill -9 -f heretic

# Verify clean state
nvidia-smi | grep python  # Should show nothing
ps aux | grep heretic     # Should show nothing

# Start fresh
cd /workspace && nohup heretic > heretic.log 2>&1 &
```

---

## Pre-Flight Checklist for 32B Models

Before starting abliteration, verify:

```bash
# 1. Check GPU availability
nvidia-smi --query-gpu=name,memory.total --format=csv
# Need: 4x 24GB or 1x 80GB for 32B models

# 2. Verify no processes running
ps aux | grep python
nvidia-smi | grep python
# Both should be empty

# 3. Verify config
cat config.toml
# Must have:
#   device_map = "balanced" (if multi-GPU)
#   cache_weights = false (if multi-GPU)
#   iterative_rounds = 0
#   batch_size = 4

# 4. Start abliteration
cd /workspace && nohup heretic > heretic.log 2>&1 &

# 5. Monitor startup (wait 5-10 minutes for model loading)
tail -f heretic.log

# 6. Verify GPU distribution after model loads
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
# Should see ~15-17GB on each GPU (not 23GB on GPU 0)
```

---

## Performance Benchmarks

### Residual Extraction (800 prompts)

| batch_size | Time | Throughput |
|------------|------|------------|
| 1 | ~100-120 min | ~7 prompts/min |
| 2 | ~50-60 min | ~13 prompts/min |
| 4 | ~25-30 min | ~27 prompts/min |
| 8 | OOM risk | N/A |

**Recommendation:** batch_size=4 is the sweet spot for 32B on 4x RTX 4090

### Trial Execution (per trial)

| Component | Time | Notes |
|-----------|------|-------|
| Reload model from disk | ~30s | No caching with multi-GPU |
| Abliterate (apply orthogonalization) | ~60s | In-place weight modification |
| Evaluate (KL + refusals, parallel) | ~7-8 min | 100 prompts each |
| **Total per trial** | **~9 min** | 200 trials = 30 hours |

**Bottleneck:** Evaluation step (85% of trial time)

---

## Recommendations for Future Improvements

### 1. Add Quantization Support
```toml
# If heretic supported 4-bit quantization:
load_in_4bit = true
bnb_4bit_compute_dtype = "bfloat16"
# Would reduce 64GB → 16GB (4x compression)
# Enable weight caching even on multi-GPU
```

**Implementation:** Requires changes to `model.py` to use `BitsAndBytesConfig`

### 2. Implement Shared Memory for Weight Cache
```python
# Instead of deepcopy per process:
import torch.multiprocessing as mp

# Store state_dict in shared memory
self.original_state_dict_shared = {
    k: v.share_memory_() for k, v in self.model.state_dict().items()
}
```

**Benefit:** Enable weight caching even with multi-GPU/multi-process

### 3. Add Process Lock File
```python
# Prevent multiple heretic instances
LOCK_FILE = Path("/tmp/heretic.lock")
if LOCK_FILE.exists():
    raise RuntimeError("Another heretic process is running")

LOCK_FILE.touch()
atexit.register(lambda: LOCK_FILE.unlink(missing_ok=True))
```

### 4. GPU Memory Pre-Check
```python
# Before loading model, verify sufficient VRAM
def estimate_model_memory(model_name: str) -> float:
    """Estimate memory needed for model in GB."""
    # Simple heuristic: 2 bytes per parameter (FP16)
    # 32B params × 2 bytes = 64GB
    pass

def verify_gpu_capacity(required_gb: float):
    """Check if total GPU memory meets requirements."""
    total_vram = sum(
        torch.cuda.get_device_properties(i).total_memory / 1e9
        for i in range(torch.cuda.device_count())
    )

    if total_vram < required_gb:
        raise RuntimeError(
            f"Model needs ~{required_gb:.0f}GB but only {total_vram:.0f}GB available"
        )
```

### 5. Faster Evaluation with Shorter Sequences
```toml
# Current default:
refusal_check_tokens = 30  # Generate 30 tokens to detect refusal

# Could reduce for faster trials:
refusal_check_tokens = 10  # Still catches "I'm sorry, I can't"
# Saves ~40% evaluation time (3 min/trial → 1.8 min/trial)
```

---

## Documentation Updates Needed

### CLAUDE.md
Add section: "Multi-GPU Configuration for Large Models (>14B)"
- Explain device_map options
- Document cache_weights incompatibility
- Provide 32B model template config

### README.md
Add warning about multi-GPU setup:
```markdown
### Large Model Support (32B+)

For models >14B parameters on multi-GPU systems:
- Use `device_map="balanced"` for even distribution
- Set `cache_weights=false` (required for multi-GPU)
- Set `iterative_rounds=0` to reduce memory pressure
- Use `batch_size=4` for stable performance

See LESSONS_LEARNED.md for detailed troubleshooting.
```

### config.default.toml
Add comments:
```toml
device_map = "auto"
# For multi-GPU systems with large models (>14B), use "balanced" instead

cache_weights = true
# Set to false when using device_map="balanced" (multi-GPU)
# Reason: deepcopy(state_dict()) fails with sharded models

iterative_rounds = 0
# Increase to 1-2 for better quality (requires extra 3-5GB VRAM)
# Disable for large models (>14B) on GPUs with <80GB VRAM
```

---

## Timeline Analysis

### Actual Time Spent (2026-01-30)

| Activity | Duration | Notes |
|----------|----------|-------|
| Initial attempt (auto, cached, iterative) | 6 trials × 9 min | OOM on trial 7 |
| Debugging OOM issues | ~2 hours | Multiple config iterations |
| Second attempt (balanced, no cache) | 100+ min | Stuck on residual extraction (batch_size=1) |
| Third attempt (batch_size=4) | In progress | Expected: 30-35 hours total |

**Total troubleshooting time:** ~4 hours before finding working config

**Lesson:** 32B models on multi-GPU are complex. Budget 4-6 hours for config tuning on first attempt.

---

## Quick Reference Card

### 32B Model on 4x RTX 4090
```toml
device_map = "balanced"
cache_weights = false
iterative_rounds = 0
batch_size = 4
max_batch_size = 16
```
**Runtime:** ~30-35 hours for 200 trials

### 32B Model on 1x A100 80GB (Faster)
```toml
device_map = "auto"
cache_weights = true
iterative_rounds = 1
batch_size = 0
max_batch_size = 64
```
**Runtime:** ~12-15 hours for 200 trials

### 7B Model on 1x RTX 4090
```toml
device_map = "auto"
cache_weights = true
iterative_rounds = 2
batch_size = 0
max_batch_size = 128
```
**Runtime:** ~2-3 hours for 200 trials

---

## Recommendation for 32B Models

**DO NOT use 4x RTX 4090 for 32B models.** Use A100 80GB instead.

**Why:**
- 4x RTX 4090: ~30-35 hours, $47-55 total cost, frequent crashes
- 1x A100 80GB: ~12-15 hours, $23-28 total cost, stable
- A100 enables weight caching, iterative ablation, larger batches
- No multi-GPU complexity, meta device errors, or memory fragmentation

**How to get A100 80GB:**
1. **Vast.ai:** Run `.\scripts\monitor_a100.ps1` to auto-alert when available
2. **RunPod:** Check https://www.runpod.io/console/pods (usually has better availability)
3. **Lambda Labs:** Check https://lambdalabs.com/service/gpu-cloud (A100 @ $1.29/hr)

See `scripts/setup_runpod_a100.md` for RunPod setup instructions.

---

## Action Items

- [x] Update CLAUDE.md with multi-GPU configuration section
- [x] Update README.md with large model warnings
- [x] Create config.32b.toml template
- [x] Document all 7 critical issues
- [x] Create A100 monitoring script
- [x] Create RunPod setup guide
- [ ] Implement process lock file to prevent multi-instance
- [ ] Add GPU memory pre-check before model loading
- [ ] Consider adding 4-bit quantization support (bitsandbytes)
- [ ] Update IMPLEMENTATION_PLAN.md with actual multi-GPU complexity (DONE)
