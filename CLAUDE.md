# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## ⛔⛔⛔ CATASTROPHIC FAILURE PREVENTION ⛔⛔⛔

### RULE 1: NEVER RUN DESTRUCTIVE COMMANDS WITHOUT EXPLICIT CONFIRMATION
**Destructive commands:** `pkill`, `kill`, `rm -rf`, `git reset --hard`, `DROP TABLE`, stopping instances

**BEFORE running ANY destructive command:**
1. State EXACTLY what the command will do
2. State that it is IRREVERSIBLE
3. Ask "Should I proceed? Type 'yes' to confirm."
4. WAIT for explicit "yes" - NOT "complete", "finish", "do it"

### RULE 2: AMBIGUOUS REQUESTS → ASK FIRST
**Dangerous phrases that need clarification:**
- "Complete the training" → Does this mean STOP or LET IT FINISH?
- "Finish the job" → Does this mean TERMINATE or WAIT FOR COMPLETION?
- "End the process" → Does this mean KILL or WAIT UNTIL DONE?

**ALWAYS ASK:** "Do you mean (A) let it continue to completion, or (B) stop it now?"

### RULE 3: TRUST USER OBSERVATIONS OVER SYSTEM DATA
**When user says "I see X" and your data says "Y":**
- NEVER say "it's fine" or "everything is working"
- IMMEDIATELY investigate the discrepancy
- Say: "There's a discrepancy. Let me investigate what you're seeing."

### RULE 4: REPEATED USER CONCERNS = STOP AND INVESTIGATE
**If the user raises the same concern more than once:**
- STOP whatever you're doing
- Do NOT repeat "it's fine"
- Deeply investigate from THEIR perspective
- The user is probably RIGHT

### RULE 5: ALWAYS CHECK DISK SPACE BEFORE CREATING CLOUD INSTANCES
**BEFORE creating ANY Vast.ai instance:**
1. Calculate required storage: `(model_size × 2) + 20GB`
2. For 32B models: **MINIMUM 200GB disk** (C4 streaming enabled since v1.1.0)
3. For 70B models: **MINIMUM 300GB disk**
4. **DEFAULT 100GB IS NOT ENOUGH FOR 32B MODELS**

**Note:** v1.1.0+ uses streaming for C4 dataset, eliminating 30-50GB download overhead.

---

## ⚠️ MANDATORY PRE-ACTION CHECKLIST

**STOP. Before taking ANY action, complete this checklist:**

```
□ 1. Have I read CLAUDE.md COMPLETELY? (Not skimmed - READ)
□ 2. Have I read WORKFLOW.md? (Contains heretic-vast commands)
□ 3. What does the user ACTUALLY want? (Write it down)
□ 4. What do I ALREADY have? (Check conversation, local files, downloaded models)
□ 5. Do existing tools handle this? (heretic-vast CLI, runpod.ps1 script)
□ 6. Can local hardware do this? (RTX 4070, 8GB VRAM - use 4-bit quantization)
□ 7. Is cloud NECESSARY? (Default answer: NO)
□ 8. What's the SIMPLEST approach?
```

### Anti-Patterns to Avoid

| Trigger | WRONG Response | RIGHT Response |
|---------|----------------|----------------|
| "Test model" | Start Vast.ai | Check if results exist locally |
| "Need GPU" | Rent cloud GPU | Check local 4070 first |
| "Error occurred" | Try workaround | Fix the actual error |
| "Compare models" | Load both at once | Sequential with memory clearing |
| User instruction | Treat as suggestion | Treat as **HARD CONSTRAINT** |

---

## ⚠️ WORKING COMMANDS - USE THESE ⚠️

**The PowerShell scripts (runpod.ps1, start-abliteration.ps1) have issues. USE THE PYTHON CLI INSTEAD:**

```bash
# Use heretic-vast CLI (Python) - THIS WORKS
uv run heretic-vast list              # Check instances
uv run heretic-vast create RTX_4090 4 # Create 4x RTX 4090
uv run heretic-vast setup             # Install heretic
uv run heretic-vast watch             # Monitor progress
uv run heretic-vast stop              # Stop instance
```

**Full training command with RESUME SUPPORT:**
```bash
uv run heretic-vast exec "export HF_HOME=/workspace/.cache/huggingface && cd /workspace && nohup heretic --model Qwen/Qwen2.5-Coder-32B-Instruct --auto-select true --auto-select-path /workspace/models --storage sqlite:////workspace/heretic_study.db --study-name qwen32b-abliteration > /workspace/heretic.log 2>&1 &"
```

### Git: ALWAYS push to `fork` (abliteration-workflow)
```bash
git push fork master   # ✅ CORRECT
git push heretic-fork  # ❌ WRONG - causes split repos
```

---

## Project Overview

Heretic is a tool for **neural behavior modification** in language models using activation direction analysis and Optuna-based optimization. While best known for abliteration (refusal removal), the technique is general and can extract/modify any behavioral direction encoded in model weights.

**Vision:** A personal neural engineering workbench for understanding and reshaping LLM behavior at the weight level. See [ROADMAP.md](ROADMAP.md) for full vision.

### Phase 1-7 Advanced Improvements (NEW)

Heretic now includes 7 phases of advanced abliteration improvements:

| Phase | Feature | Default | Description |
|-------|---------|---------|-------------|
| 1 | Neural Refusal Detection | ✅ ON | Zero-shot NLI for detecting soft refusals |
| 2 | Supervised Probing + Ensemble | ✅ ON | Linear probes combined with PCA |
| 3 | Activation Calibration | ✅ ON | Scale weights based on activation strength |
| 4 | Concept Cones | OFF | Cluster harmful prompts by category |
| 5 | CAA (Contrastive Activation Addition) | OFF | Add compliance direction |
| 6 | Circuit-Level Ablation | OFF | Target specific attention heads (⚠️ No GQA) |
| 7 | Warm-Start Transfer | ✅ ON | Model family profiles for faster Optuna |

**Key files for advanced features:**
- `src/heretic/model.py` - All Phase 1-7 implementations + Sacred Direction Preservation
- `src/heretic/config.py` - 50+ configuration settings with Pydantic validators
- `src/heretic/constants.py` - Centralized magic numbers (Thresholds, LayerPos, Memory, etc.)
- `src/heretic/phases/` - Modular pipeline phases (dataset loading, direction extraction, optimization, saving)
- `src/heretic/transfer.py` - Model family profiles for warm-start
- `src/heretic/exceptions.py` - 22+ custom exception types including `SacredDirectionError`

## Build and Development Commands

```bash
# Install dependencies (requires uv)
uv sync --all-extras --dev

# Run the tool
uv run heretic <model-name>
# or
uv run heretic --model <model-name>

# Format code
uv run ruff format .

# Lint and check imports
uv run ruff check --extend-select I .

# Build package
uv build
```

## Architecture

### Core Flow (`src/heretic/main.py`)

The abliteration pipeline is organized into modular phases (see `src/heretic/phases/`):

1. **Dataset Loading** (`phases.dataset_loading`)
   - Load good/bad prompts via `load_datasets()`
   - Returns `DatasetBundle` with all prompt datasets

2. **Direction Extraction** (`phases.direction_extraction`)
   - Extract refusal directions via `extract_refusal_directions()`
   - Apply helpfulness orthogonalization via `apply_helpfulness_orthogonalization()`
   - Extract sacred capability directions via `extract_sacred_directions()`
   - Returns `DirectionExtractionResult` with all direction tensors

3. **Optimization** (`phases.optimization`)
   - Create/resume Optuna study via `create_study()`
   - Enqueue warm-start trials via `enqueue_warm_start_trials()`
   - Run optimization loop via `run_optimization()`

4. **Model Saving** (`phases.model_saving`)
   - Save locally via `save_model_local()`
   - Upload to HuggingFace via `upload_model_huggingface()`

### Key Components

**`Model` (`src/heretic/model.py`)**
- Wraps HuggingFace model loading with dtype fallback chain
- `get_layers()`: Handles both text-only and multimodal model architectures
- `get_layer_matrices()`: Extracts abliterable weight matrices (attn.o_proj, mlp.down_proj) supporting dense and MoE variants
- `abliterate()`: Applies orthogonalization with per-layer weight interpolation
- `get_residuals_batched()`: Extracts hidden states for refusal direction computation
- **Phase 1-7 Methods:**
  - `get_refusal_directions_pca()`: Multi-direction contrastive PCA extraction
  - `get_refusal_directions_supervised()`: Train linear probes per layer
  - `get_refusal_directions_ensemble()`: Combine probe + PCA directions
  - `compute_refusal_activation_stats()`: Activation statistics for calibration
  - `get_calibrated_parameters()`: Apply activation-based weight scaling
  - `get_refusal_directions_concept_cones()`: Category-specific directions via clustering
  - `abliterate_concept_cones()`: Weighted cone-specific ablation
  - `extract_compliance_direction()`: Compliance direction for CAA
  - `abliterate_with_caa()`: Combined removal + addition
  - `discover_refusal_circuits()`: Find refusal-mediating attention heads
  - `abliterate_circuits()`: Head-specific ablation (non-GQA only)
- **Performance optimizations:**
  - In-memory weight caching: Caches original `state_dict` for fast reset (~5-10x faster than reloading from disk)
  - Optional `torch.compile()` support for ~1.5-2x inference speedup
  - Memory leak fix: `device_projectors_cache` cleared after abliteration

**`Evaluator` (`src/heretic/evaluator.py`)**
- Computes KL divergence from first-token probability distributions
- Counts refusals using configurable marker strings (pre-compiled regex for 5-10x faster matching)
- Returns multi-objective score tuple `(kl_divergence, refusals)`
- **Performance optimizations:**
  - Early stopping: Generates only 30 tokens for refusal detection (configurable via `refusal_check_tokens`)
  - Parallel evaluation: KL divergence and refusal counting run concurrently via ThreadPoolExecutor

**`Settings` (`src/heretic/config.py`)**
- Pydantic-based configuration with layered sources: CLI args > env vars > config.toml
- Key settings: `n_trials`, `batch_size`, `dtypes`, `refusal_markers`
- **Phase 1-7 settings (50+ total):**
  - Neural detection: `use_neural_refusal_detection`, `neural_detection_for_optuna`
  - Supervised probing: `use_supervised_probing`, `ensemble_probe_pca`, `min_probe_accuracy`
  - Activation calibration: `use_activation_calibration`, `activation_target_percentile`
  - Concept cones: `use_concept_cones`, `n_concept_cones`, `min_cone_size`
  - CAA: `use_caa`, `caa_addition_strength`, `caa_max_overlap`
  - Circuit ablation: `use_circuit_ablation`, `n_refusal_circuits` (⚠️ No GQA support)
  - Warm-start: `use_warm_start_params`, `model_family`, `warm_start_n_trials`
  - Validation: `enable_validation`, `run_mmlu_validation`

**`transfer.py` (`src/heretic/transfer.py`)** (NEW)
- Model family profiles for warm-start parameter initialization
- Profiles for: llama, qwen, mistral, gemma, phi
- `get_warm_start_params()`: Generate Optuna warm-start trials
- `detect_model_family()`: Auto-detect model family from name

### Key Directories
- `src/heretic/` - Core abliteration logic
- `src/heretic/phases/` - Modular pipeline phases (NEW)
  - `dataset_loading.py` - Dataset loading and bundling
  - `direction_extraction.py` - Refusal/sacred direction extraction
  - `optimization.py` - Optuna study management
  - `model_saving.py` - Model persistence and upload
- `src/heretic/constants.py` - Centralized magic numbers and thresholds (NEW)
- `src/heretic/vast.py` - Vast.ai cloud CLI
- `experiments/` - Behavioral direction experiments
- `models/` - Local modified models (gitignored)
- `chat_history/` - Saved chat sessions (gitignored)

## Configuration

Settings cascade: CLI > environment (HERETIC_ prefix) > config.toml > defaults

Key parameters in `config.toml`:
- `model`: HuggingFace model ID or local path
- `n_trials`: Optimization iterations (default 200)
- `batch_size`: 0 for auto-detection
- `max_batch_size`: Upper limit for auto-detection
- `dtypes`: Fallback chain for model loading precision
- `compile`: Enable torch.compile() for faster inference (default: false)
- `refusal_check_tokens`: Tokens to generate for refusal detection (default: 30)
- `storage`: Optuna SQLite storage URL for resume support (default: sqlite:///heretic_study.db)
- `study_name`: Optuna study name for resuming experiments

**Phase 1-7 Advanced Settings (enabled by default):**
- `use_neural_refusal_detection`: Zero-shot NLI detection (default: true)
- `ensemble_probe_pca`: Combine probe + PCA directions (default: true)
- `use_activation_calibration`: Adaptive weight scaling (default: true)
- `use_warm_start_params`: Model family warm-start (default: true)
- `enable_validation`: Validation framework (default: true)

**Phase 1-7 Advanced Settings (disabled by default):**
- `use_concept_cones`: Category-specific directions
- `use_caa`: Contrastive Activation Addition
- `use_circuit_ablation`: Attention head targeting (⚠️ No GQA support)

**Sacred Direction Preservation (experimental, disabled by default):**
- `use_sacred_directions`: Orthogonalize refusal direction against capability-encoding directions to preserve model capabilities (MMLU, etc.)
- `n_sacred_directions`: Number of sacred directions to extract (default: 5)
- `sacred_prompts`: Dataset specification for capability prompts (default: cais/mmlu)
- `sacred_baseline_prompts`: Dataset specification for baseline text (default: wikitext)
- `sacred_overlap_threshold`: Warning threshold for refusal-sacred overlap (default: 0.3)

**How Sacred Direction Preservation Works:**
1. Extract "sacred" directions from capability prompts (e.g., MMLU questions) via contrastive PCA
2. Compute overlap between refusal direction and sacred directions
3. Orthogonalize refusal direction against ALL sacred directions
4. Abliterate with the orthogonalized direction (mathematically cannot damage sacred capabilities)

**Key Methods in `model.py`:**
- `extract_sacred_directions()`: Extract capability-encoding directions
- `orthogonalize_against_sacred()`: Remove sacred components from refusal direction
- `compute_sacred_overlap()`: Measure refusal-capability overlap before orthogonalization

## Performance Optimizations

| Optimization | Speedup | Flag/Config | Status |
|--------------|---------|-------------|--------|
| **GPU-accelerated PCA** | **15-20x faster** (4-6 hrs → 15-20 min) | Automatic (v1.0.1+) | ✅ **NEW** |
| In-memory weight caching | ~5-10x faster model reset | Automatic (disable for 32B+) | ✅ |
| torch.compile() | ~1.5-2x inference speedup | `--compile` | ✅ |
| Early stopping for refusals | ~40-60% faster evaluation | `--refusal-check-tokens 30` | ✅ |
| Parallel evaluation | ~20-30% faster per trial | Automatic | ✅ |
| Resume support | Can resume interrupted runs | `--storage sqlite:///study.db` | ✅ |

**GPU PCA Optimization (v1.0.1):**
- Replaces CPU-bound numpy eigendecomposition with GPU torch operations
- Keeps tensors on GPU throughout PCA computation (`torch.linalg.eigh`)
- **Impact:** 32B models (64 layers × 5120 dims) complete PCA in minutes instead of hours
- **H200 tested:** Qwen2.5-Coder-32B PCA completed in <5 minutes
- See `GPU_PCA_IMPLEMENTATION.md` for technical details

**Example with all optimizations:**
```bash
heretic --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --compile \
  --storage sqlite:///heretic_study.db \
  --study-name qwen32b \
  --auto-select true \
  --cache-weights false \
  --unhelpfulness-prompts.config en
```

## Heretic CLI Flags Reference

| Flag | Notes |
|------|-------|
| `--auto-select` | **REQUIRES a boolean value**: Use `--auto-select true`, NOT `--auto-select` |
| `--compile` | Enable torch.compile() for ~1.5-2x inference speedup |
| `--storage` | Optuna storage URL for resume support (e.g., `sqlite:///study.db`) |
| `--study-name` | Name for the Optuna study (default: `heretic_study`) |
| `--refusal-check-tokens` | Tokens for refusal detection (default: 30, lower = faster) |
| `--cache-weights` | Enable/disable in-memory weight caching (default: true). **SET TO FALSE FOR 32B+ MODELS** |
| `--unhelpfulness-prompts.config` | Dataset config for C4 (required: `"en"`). CLI overrides TOML! |
| `--orthogonalize-directions` | Enable helpfulness direction orthogonalization (default: true) |

**Examples:**
```bash
# 7B model with all optimizations
heretic --model Qwen/Qwen2.5-7B-Instruct \
  --auto-select true \
  --n-trials 20 \
  --compile

# 32B model (must disable weight caching + C4 config)
heretic --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --auto-select true \
  --cache-weights false \
  --storage sqlite:///study.db \
  --unhelpfulness-prompts.config en

# Full production run on H200
heretic --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --auto-select true \
  --auto-select-path /workspace/models \
  --storage sqlite:////workspace/heretic_study.db \
  --study-name qwen32b-abliteration \
  --cache-weights false \
  --n-trials 200 \
  --unhelpfulness-prompts.config en
```

---

## Conventions

- **Formatting/linting:** `ruff format .` and `ruff check --extend-select I .`
- **Type hints:** Use comprehensive type hints for all functions/methods
- **Error handling:** Use custom exception classes with descriptive messages
- **CSS:** Use Gradio CSS variables (e.g., `--body-text-color`) for theme compatibility

**Things to avoid:**
- No emojis or unicode symbols in output
- No `torch_dtype` (use `dtype`)
- No type casting to `any` - keep proper types
- No print statements - use `logging` module instead

---

## Chat Interface (`chat_app.py`)

**Architecture:**
- `ModelManager` class: Handles model loading, caching, and streaming text generation
- Custom exception hierarchy: `HereticError` base class with specific exceptions:
  - `ModelNotFoundError`, `ModelValidationError`, `ModelLoadError`
  - `CUDAOutOfMemoryError`, `TokenizationError`, `GenerationError`
- Structured logging via Python's `logging` module (logger: `heretic_chat`)

**Key Features:**
- Auto-discovers models in `models/` directory with validation
- Validates model files before loading (config.json, weights, tokenizer)
- Streaming token generation via `TextIteratorStreamer`
- Real-time GPU memory monitoring display
- Chat history persistence to `chat_history/` as JSON

**Patterns:**
- Model validation: Check for config.json, model weights, and tokenizer files
- GPU monitoring: Use `torch.cuda.memory_reserved()` for accurate usage
- Tokenization: Use `apply_chat_template(tokenize=True, return_tensors="pt")` for cross-model compatibility
- Message validation: Ensure all message content is string (not None) for Qwen compatibility

---

## Vast.ai CLI (`src/heretic/vast.py`)

Dedicated CLI for Vast.ai GPU cloud management:

```bash
# ⛔ CRITICAL: Check disk requirements first!
# 32B model needs 200GB disk, 70B needs 400GB
# Default 100GB is NOT enough for 32B models!
heretic-vast create A100_80GB 2   # Create 2x A100 instance
heretic-vast setup                 # Install heretic
heretic-vast run MODEL             # Run abliteration
heretic-vast watch                 # Live dashboard
heretic-vast stop                  # Stop billing
```

**Key Components:**
- `VastConfig`: Configuration from env vars / .env file
- `GPU_TIERS`: Preset configurations for different GPU types
- `get_connection()`: Fabric SSH connection management
- `watch_dashboard()`: Rich live terminal dashboard

**Dependencies:** `fabric`, `rich`, `click`

---

## Experiments Framework

Experiments for testing new behavioral directions live in `experiments/`:

```
experiments/
├── verbosity/          # Completed: padding direction extraction
├── verbosity_v2/       # Completed: isolated padding vs complexity
└── hedging/            # Ready: hedging marker extraction
```

### Running Experiments

```bash
# 1. Prepare dataset
python experiments/verbosity/load_local_dataset.py

# 2. Copy config
cp experiments/verbosity/config.verbosity.toml config.toml

# 3. Run heretic
heretic --model meta-llama/Llama-3.1-8B-Instruct

# 4. Evaluate
python experiments/verbosity/eval_verbosity.py --original MODEL --modified OUTPUT
```

### Verbosity Spike Results (Qwen 7B)

| Metric | Value |
|--------|-------|
| Model | Qwen/Qwen2.5-7B-Instruct |
| Trials | 50 Optuna optimization trials |
| Model Saved | `./models/Qwen2.5-7B-Instruct-heretic` (15.2 GB) |

**Test Results (RTX 4070, 4-bit quantization):**
- ✅ **Factual questions**: Concise responses (6-30 words)
- ⚠️ **Open-ended questions**: Still verbose (195-203 words)

**Conclusion:** The ablation removes padding on simple questions but preserves appropriate depth on complex questions.

---

## Critical Gotchas (Lessons Learned)

### Local Hardware
- **User has RTX 4070 with 8GB VRAM** - Use this for testing when possible
- **7B models need 4-bit quantization** to fit in 8GB (use BitsAndBytesConfig)
- **Cannot load two 7B models simultaneously** - Run comparisons sequentially with memory clearing

### PyTorch CUDA Issue with uv
`uv run` uses lockfile which may revert to CPU-only torch.
- Solution 1: Use system Python directly: `python script.py`
- Solution 2: Force reinstall: `uv pip install --reinstall torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121`

### Tokenizer Issues
**Heretic saves tokenizer_config.json with `extra_special_tokens` as a list** instead of dict, causing transformers to fail.

**Fix:**
```python
import json; p='models/MODEL/tokenizer_config.json'; c=json.load(open(p,encoding='utf-8')); c['extra_special_tokens']={}; json.dump(c,open(p,'w',encoding='utf-8'),indent=2)
```

### Dataset Loading
- **heretic uses `load_dataset()` from HuggingFace**, NOT `load_from_disk()`
- `DatasetSpecification` only supports: `dataset`, `split`, `column` - NO `data_files` field

### Gated Models
- **Llama models are gated** - require HuggingFace authentication
- Use `huggingface-cli login` or set `HF_TOKEN` environment variable
- **Qwen models are NOT gated** - use these for quick testing

### GPU Out of Memory (OOM) on 32B+ Models
**Problem:** The 32B model causes OOM when heretic tries to cache weights in memory.

**Solution:** Use `--cache-weights false` flag:
```bash
heretic --model Qwen/Qwen2.5-Coder-32B-Instruct --cache-weights false
```

### Circuit Ablation Not Working
**Problem:** Circuit-level ablation fails or returns 0 circuits ablated.

**Cause:** GQA (Grouped Query Attention) models are not compatible with circuit ablation.

**Affected models:** Llama 3.x, Qwen2.5, and other GQA models.

**Solution:** Use standard layer-level ablation instead:
```bash
# Disable circuit ablation for GQA models
heretic --model meta-llama/Llama-3.1-8B-Instruct --use-circuit-ablation false
```

### Ensemble Probe Failing
**Problem:** RuntimeError about class imbalance or low accuracy.

**Cause:** Supervised probing requires balanced data (≥10 samples per class) and reasonable accuracy.

**Solution:** Check your bad_prompts dataset triggers actual refusals, or use PCA-only mode:
```bash
heretic --model MODEL --ensemble-probe-pca false --use-pca-extraction true
```

### pip install --force-reinstall Breaks Environment
**Problem:** After running `pip install --force-reinstall`, heretic fails with "Could not import module 'Qwen2ForCausalLM'".

**Solution:** `pip install transformers>=4.55.2`

**Prevention:** Never use `--force-reinstall`. Use `--upgrade` instead.

### PowerShell Scripts Fail from Bash/Codebuff
**Problem:** Running `.ps1` scripts from bash often fails.

**Solution:** Use direct SSH commands or `heretic-vast` CLI instead.

### Instance Stop vs Process Kill
- **Stopping a Vast.ai instance KILLS running processes** - no graceful save
- Check `heretic-vast progress` for "Models Saved" before stopping

### C4 Dataset Configuration Error
**Problem:** `ValueError: Config name is missing` when loading C4 dataset.

**Cause:** C4 and other multi-variant datasets require an explicit config parameter (e.g., "en", "realnewslike").

**Solution:** Add config to CLI command or TOML file:
```bash
# Via CLI (recommended - overrides TOML)
heretic --model MODEL --unhelpfulness-prompts.config en

# Via TOML (only works if not overridden by CLI)
[unhelpfulness_prompts]
dataset = "allenai/c4"
config = "en"
split = "train[:200]"
column = "text"
```

**Important:** CLI arguments ALWAYS override TOML config. If you pass `--unhelpfulness-prompts.dataset` on CLI, you must also pass `--unhelpfulness-prompts.config` or it defaults to `None`.

### C4 Dataset Disk Space Requirements

**RESOLVED in v1.1.0+:** C4 dataset now uses streaming, eliminating the disk space overhead.

**Previous Problem (v1.0.x and earlier):** "No space left on device" while downloading C4 dataset, even with 200GB disk.

**Previous Cause:** C4 (Colossal Clean Crawled Corpus) download behavior:
- **Total size:** ~800GB for "en" variant
- **Structure:** 1024 shards (~320MB each compressed, 1-2GB uncompressed)
- **Download behavior:** HuggingFace downloaded entire shards, not individual examples
- **Split `train[:200]`:** Downloaded 200+ shards (~65GB+) to get 200 examples

**Current Solution (v1.1.0+):**
- C4 dataset automatically streams on-demand
- Only downloads the exact examples needed (~0GB disk overhead)
- No configuration changes required
- Requires network connectivity during dataset loading

**Implementation Details:**
- Automatic detection: Any dataset with "c4" in name uses streaming
- Maintains API compatibility: Returns same `list[str]` as before
- Error handling: Fails loudly with clear messages if network unavailable
- See `claudedocs/c4_streaming_implementation.md` for technical details

**Fallback for v1.0.x (if not upgrading):**
- `split = "train[:200]"` triggers downloading many shards to get representative sample
- Each shard contains ~356,000 examples, but library downloads whole shards

**Disk usage breakdown (32B model with C4):**
- Qwen2.5-Coder-32B model: 62GB
- C4 dataset (train[:200]): 65-150GB (partial download)
- Working space: 20GB
- **Total:** 150-230GB minimum

**Solutions:**

**Option 1: Use 400GB+ disk** (RECOMMENDED)
```bash
# Recreate instance with proper disk space
vastai search offers 'gpu_name=H200 disk_space>=400 num_gpus=1'
vastai create instance <ID> --image ubuntu:22.04 --disk 400
```

**Option 2: Reduce C4 split size**
```toml
[unhelpfulness_prompts]
split = "train[:50]"  # 75% reduction, still effective
```

**Option 3: Use smaller unhelpfulness dataset**
```toml
[unhelpfulness_prompts]
dataset = "wikitext"
config = "wikitext-2-raw-v1"
split = "train[:200]"
column = "text"
```

**Best practice:** For 32B+ models with orthogonalization, use **400GB minimum disk**.

### Package Data Files Not Included in Wheel
**Problem:** Config files or data files in project root aren't included in built wheel.

**Solution:**
1. Move data files to `src/<package>/` directory
2. Configure build backend in `pyproject.toml`:
```toml
[tool.uv.build-backend]
module-name = "heretic"
data-files = { "heretic" = ["config.default.toml"] }
```

**Verification:** `unzip -l dist/*.whl | grep toml`

### Python Cache After Wheel Reinstall
**Problem:** After reinstalling wheel with `pip install --force-reinstall`, changes don't take effect.

**Cause:** Python caches `.pyc` files and modules in `__pycache__/`.

**Solution:** Clear cache after reinstall:
```bash
find /path/to/package -name '*.pyc' -delete
find /path/to/package -name '__pycache__' -type d -exec rm -rf {} +
# Or restart Python process
pkill -f 'python.*heretic'
```

### Pydantic Settings Priority (CRITICAL)
**Priority order** (highest to lowest):
1. **CLI arguments** (e.g., `--model foo`)
2. Environment variables (e.g., `HERETIC_MODEL=foo`)
3. TOML config file (e.g., `model = "foo"`)

**Gotcha:** CLI default values (including `None`) override TOML config!

**Example:**
```toml
# config.toml
[unhelpfulness_prompts]
config = "en"
```

```bash
# This FAILS - CLI default (None) overrides TOML
heretic --model MODEL

# This WORKS - CLI explicitly sets config
heretic --model MODEL --unhelpfulness-prompts.config en

# This WORKS - Don't pass ANY unhelpfulness args via CLI
# (but only if workspace doesn't have config.toml)
cd /workspace/empty_dir && heretic --model MODEL
```

**Best Practice:** Either pass ALL settings via CLI, OR ensure no CLI defaults conflict with TOML.

---

## ⛔ DO NOT DO THESE THINGS ⛔

### NEVER judge instance status by uptime
- Vast.ai shared GPUs show ACCUMULATED uptime from all users
- **ONLY check if heretic process is running, NOT the uptime**

### NEVER create a new instance without explicit permission
- If user says "wait" - WAIT
- If instance is "loading" - WAIT
- If you're not sure - ASK

### NEVER take destructive actions without asking
- Don't stop instances
- Don't terminate instances
- Don't create new instances
- **ALWAYS ASK FIRST**

### When user says STOP - STOP IMMEDIATELY
- Don't spawn more agents
- Don't make more tool calls
- Don't try to "fix" anything
- Just STOP and LISTEN
