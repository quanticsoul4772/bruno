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
□ 2. Have I read WORKFLOW.md? (Contains bruno-vast commands)
□ 3. What does the user ACTUALLY want? (Write it down)
□ 4. What do I ALREADY have? (Check conversation, local files, downloaded models)
□ 5. Do existing tools handle this? (bruno-vast CLI, runpod.ps1 script)
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
# Use bruno-vast CLI (Python) - THIS WORKS
uv run bruno-vast list              # Check instances
uv run bruno-vast create RTX_4090 4 # Create 4x RTX 4090
uv run bruno-vast setup             # Install bruno
uv run bruno-vast watch             # Monitor progress
uv run bruno-vast stop              # Stop instance
```

**⚠️ FALLBACK: If bruno-vast create fails to find GPUs, use direct vastai CLI:**
```bash
# bruno-vast may report "No offers available" even when GPUs exist
# Use direct vastai CLI as fallback:
vastai search offers "gpu_name=H200 disk_space>=200 rentable=true" --order dph_total
vastai create instance <OFFER_ID> --disk 200 --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
```

**Full training command with RESUME SUPPORT:**
```bash
uv run bruno-vast exec "export HF_HOME=/workspace/.cache/huggingface && cd /workspace && nohup bruno --model Qwen/Qwen2.5-Coder-32B-Instruct --auto-select true --auto-select-path /workspace/models --storage sqlite:////workspace/bruno_study.db --study-name qwen32b-abliteration > /workspace/bruno.log 2>&1 &"
```

### Git: ALWAYS push to `fork` (abliteration-workflow)
```bash
git push fork master   # ✅ CORRECT
git push bruno-fork  # ❌ WRONG - causes split repos
```

---

## Project Overview

Bruno is a tool for **neural behavior modification** in language models using activation direction analysis and Optuna-based optimization. While best known for abliteration (refusal removal), the technique is general and can extract/modify any behavioral direction encoded in model weights.

**Vision:** A personal neural engineering workbench for understanding and reshaping LLM behavior at the weight level. See [ROADMAP.md](ROADMAP.md) for full vision.

## Dependencies & Security

**Critical Requirements:**
- **PyTorch >= 2.8.0** (currently 2.10.0+cu128)
  - **Security:** Patches CVE for torch.load RCE vulnerability (< 2.6.0)
  - **Security:** Patches CVE for CTC loss DoS vulnerability (<= 2.7.1)
  - CUDA 12.8 (backward compatible with CUDA 12.1+ hardware)
- **Python:** 3.10, 3.11, 3.12 (< 3.13 due to torch wheel availability)
- **transformers >= 4.48.2** (Moonlight requires 4.48.2 exactly)

**Installation:**
```bash
uv sync --all-extras --dev
```

### Phase 1-7 Advanced Improvements (NEW)

Bruno now includes 7 phases of advanced abliteration improvements:

| Phase | Feature | Default | Description |
|-------|---------|---------|-------------|
| 1 | Neural Refusal Detection | ✅ ON | Zero-shot NLI for detecting soft refusals |
| 2 | Supervised Probing + Ensemble | ✅ ON | Linear probes combined with PCA |
| 3 | Activation Calibration | ✅ ON | Scale weights based on activation strength |
| 4 | Concept Cones | ✅ ON | Cluster harmful prompts by category |
| 5 | CAA (Contrastive Activation Addition) | ✅ ON | Add compliance direction |
| 6 | Circuit-Level Ablation | OFF | Target specific attention heads (⚠️ No GQA) |
| 7 | Warm-Start Transfer | ✅ ON | Model family profiles for faster Optuna |
| 8 | MPOA (Norm-Preserving Ablation) | ✅ ON | Preserves weight matrix norms for better capability retention |

**Key files for advanced features:**
- `src/bruno/model.py` - All Phase 1-7 implementations + Sacred Direction Preservation
- `src/bruno/config.py` - 50+ configuration settings with Pydantic validators
- `src/bruno/constants.py` - Centralized magic numbers (Thresholds, LayerPos, Memory, etc.)
- `src/bruno/phases/` - Modular pipeline phases (dataset loading, direction extraction, optimization, saving)
- `src/bruno/transfer.py` - Model family profiles for warm-start
- `src/bruno/exceptions.py` - 22+ custom exception types including `SacredDirectionError`

## Build and Development Commands

```bash
# Install dependencies (requires uv)
uv sync --all-extras --dev

# Run the tool
uv run bruno <model-name>
# or
uv run bruno --model <model-name>

# Format code
uv run ruff format .

# Lint and check imports
uv run ruff check --extend-select I .

# Build package
uv build
```

## Architecture

### Core Flow (`src/bruno/main.py`)

The abliteration pipeline is organized into modular phases (see `src/bruno/phases/`):

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

**`Model` (`src/bruno/model.py`)**
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
  - **Layer-wise weight caching** (v1.2.0+): Selective caching of abliterable components only
    - Reduces cache memory by 55-75% vs full state_dict (62GB → 28GB for 32B models)
    - Enables caching for 32B+ models on 141GB GPUs (previously required `--cache-weights false`)
    - Fast reload: 10-15s vs 60-120s disk reload
    - Auto-validation with cache statistics logging
    - Comprehensive error handling prevents silent failures
  - Optional `torch.compile()` support for ~1.5-2x inference speedup
  - Memory leak fix: `device_projectors_cache` cleared after abliteration

**`Evaluator` (`src/bruno/evaluator.py`)**
- Computes KL divergence from first-token probability distributions
- Counts refusals using configurable marker strings (pre-compiled regex for 5-10x faster matching)
- Returns multi-objective score tuple `(kl_divergence, refusals)`
- **Performance optimizations:**
  - Early stopping: Generates only 30 tokens for refusal detection (configurable via `refusal_check_tokens`)
  - Parallel evaluation: KL divergence and refusal counting run concurrently via ThreadPoolExecutor

**`Settings` (`src/bruno/config.py`)**
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

**`transfer.py` (`src/bruno/transfer.py`)** (NEW)
- Model family profiles for warm-start parameter initialization
- Profiles for: llama, qwen, mistral, gemma, phi
- `get_warm_start_params()`: Generate Optuna warm-start trials
- `detect_model_family()`: Auto-detect model family from name

### Key Directories
- `src/bruno/` - Core abliteration logic
- `src/bruno/phases/` - Modular pipeline phases (NEW)
  - `dataset_loading.py` - Dataset loading and bundling
  - `direction_extraction.py` - Refusal/sacred direction extraction
  - `optimization.py` - Optuna study management
  - `model_saving.py` - Model persistence and upload
- `src/bruno/constants.py` - Centralized magic numbers and thresholds (NEW)
- `src/bruno/vast.py` - Vast.ai cloud CLI
- `experiments/` - Behavioral direction experiments
- `models/` - Local modified models (gitignored)
- `chat_history/` - Saved chat sessions (gitignored)

## Configuration

Settings cascade: CLI > environment (BRUNO_ prefix) > config.toml > defaults

Key parameters in `config.toml`:
- `model`: HuggingFace model ID or local path
- `n_trials`: Optimization iterations (default 200)
- `batch_size`: 0 for auto-detection
- `max_batch_size`: Upper limit for auto-detection
- `dtypes`: Fallback chain for model loading precision
- `compile`: Enable torch.compile() for faster inference (default: false)
- `refusal_check_tokens`: Tokens to generate for refusal detection (default: 30)
- `storage`: Optuna SQLite storage URL for resume support (default: sqlite:///bruno_study.db)
- `study_name`: Optuna study name for resuming experiments

**Phase 1-7 Advanced Settings (enabled by default):**
- `use_neural_refusal_detection`: Zero-shot NLI detection (default: true)
- `ensemble_probe_pca`: Combine probe + PCA directions (default: true)
- `use_activation_calibration`: Adaptive weight scaling (default: true)
- `use_concept_cones`: Category-specific directions (default: true)
- `use_caa`: Contrastive Activation Addition (default: true)
- `use_mpoa`: Norm-preserving ablation (default: true)
- `use_warm_start_params`: Model family warm-start (default: true)
- `enable_validation`: Validation framework (default: true)

**Phase 1-7 Advanced Settings (disabled by default):**
- `use_circuit_ablation`: Attention head targeting (⚠️ No GQA support - doesn't work with Llama 3.x, Qwen2.5, etc.)

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
| **GPU-accelerated PCA** | **15-20x faster** (4-6 hrs → 15-20 min) | Automatic (v1.0.1+) | ✅ |
| **Layer-wise weight caching** | **6-12x faster reload**, 55-75% less memory | Automatic (v1.2.0+) | ✅ **NEW** |
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

**Layer-Wise Weight Caching (v1.2.0):**
- **Problem Solved:** 32B models previously required `--cache-weights false` due to OOM (62GB model + 62GB cache = 124GB > 141GB GPU)
- **Solution:** Cache only abliterable components (attn.o_proj, mlp.down_proj) instead of full state_dict
- **Memory Impact:** 55-75% reduction (62GB → 28GB cache for 32B models)
- **Performance Impact:** 10-15s reload vs 60-120s disk reload (6-12x faster)
- **Training Impact:** Saves 3-4 hours per 200-trial run (30% faster overall)
- **Cost Impact:** ~$6-8 savings per run on H200 @ $2.14/hr
- **Auto-validation:** Cache statistics logged on creation, comprehensive error handling prevents silent failures
- See `claudedocs/layer_wise_cache_implementation.md` for technical details

**Cache Usage Examples:**

```bash
# 7B model - caching always recommended
bruno --model Qwen/Qwen2.5-7B-Instruct \
  --cache-weights true \
  --n-trials 200

# 32B model - NOW WORKS with caching enabled! (v1.2.0+)
bruno --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --cache-weights true \
  --n-trials 200 \
  --unhelpfulness-prompts.config en

# Cache statistics output (verify success):
# * Cache size: 28.4 GB (14,874,368,000 params, 43.2% of model)

# Disable caching (fallback for extreme memory constraints)
bruno --model MODEL --cache-weights false
```

**When to Disable Caching:**
- GPU has <100GB memory for 32B models (cache + model + working memory)
- OOM errors during cache creation (will see clear error message)
- Testing disk reload performance
- **No longer needed for 32B models on 141GB+ GPUs** (v1.2.0+)

**Example with all optimizations:**
```bash
bruno --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --compile \
  --storage sqlite:///bruno_study.db \
  --study-name qwen32b \
  --auto-select true \
  --cache-weights true \
  --unhelpfulness-prompts.config en
```

## Bruno CLI Flags Reference

| Flag | Notes |
|------|-------|
| `--auto-select` | **REQUIRES a boolean value**: Use `--auto-select true`, NOT `--auto-select` |
| `--compile` | Enable torch.compile() for ~1.5-2x inference speedup |
| `--storage` | Optuna storage URL for resume support (e.g., `sqlite:///bruno_study.db`) |
| `--study-name` | Name for the Optuna study (default: `bruno_study`) |
| `--refusal-check-tokens` | Tokens for refusal detection (default: 30, lower = faster) |
| `--cache-weights` | Enable/disable in-memory weight caching (default: true). **Now works for 32B+ models!** (v1.2.0+) |
| `--unhelpfulness-prompts.config` | Dataset config for C4 (required: `"en"`). CLI overrides TOML! |
| `--orthogonalize-directions` | Enable helpfulness direction orthogonalization (default: true) |

**Examples:**
```bash
# 7B model with all optimizations
bruno --model Qwen/Qwen2.5-7B-Instruct \
  --auto-select true \
  --n-trials 20 \
  --compile

# 32B model - caching now enabled! (v1.2.0+)
bruno --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --auto-select true \
  --cache-weights true \
  --storage sqlite:///study.db \
  --unhelpfulness-prompts.config en

# Full production run on H200 (with caching for 3-4 hour speedup!)
bruno --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --auto-select true \
  --auto-select-path /workspace/models \
  --storage sqlite:////workspace/bruno_study.db \
  --study-name qwen32b-abliteration \
  --cache-weights true \
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
- Custom exception hierarchy: `BrunoError` base class with specific exceptions:
  - `ModelNotFoundError`, `ModelValidationError`, `ModelLoadError`
  - `CUDAOutOfMemoryError`, `TokenizationError`, `GenerationError`
- Structured logging via Python's `logging` module (logger: `bruno_chat`)

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

## Vast.ai CLI (`src/bruno/vast.py`)

Dedicated CLI for Vast.ai GPU cloud management:

```bash
# ⛔ CRITICAL: Check disk requirements first!
# 32B model needs 200GB disk, 70B needs 400GB
# Default 100GB is NOT enough for 32B models!
bruno-vast create A100_80GB 2   # Create 2x A100 instance
bruno-vast setup                 # Install bruno
bruno-vast run MODEL             # Run abliteration
bruno-vast watch                 # Live dashboard
bruno-vast stop                  # Stop billing
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

# 3. Run bruno
bruno --model meta-llama/Llama-3.1-8B-Instruct

# 4. Evaluate
python experiments/verbosity/eval_verbosity.py --original MODEL --modified OUTPUT
```

### Verbosity Spike Results (Qwen 7B)

| Metric | Value |
|--------|-------|
| Model | Qwen/Qwen2.5-7B-Instruct |
| Trials | 50 Optuna optimization trials |
| Model Saved | `./models/Qwen2.5-7B-Instruct-bruno` (15.2 GB) |

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
**Bruno saves tokenizer_config.json with `extra_special_tokens` as a list** instead of dict, causing transformers to fail.

**Fix:**
```python
import json; p='models/MODEL/tokenizer_config.json'; c=json.load(open(p,encoding='utf-8')); c['extra_special_tokens']={}; json.dump(c,open(p,'w',encoding='utf-8'),indent=2)
```

### Dataset Loading
- **bruno uses `load_dataset()` from HuggingFace**, NOT `load_from_disk()`
- `DatasetSpecification` only supports: `dataset`, `split`, `column` - NO `data_files` field

### Gated Models
- **Llama models are gated** - require HuggingFace authentication
- Use `huggingface-cli login` or set `HF_TOKEN` environment variable
- **Qwen models are NOT gated** - use these for quick testing

### GPU Out of Memory (OOM) on 32B+ Models

**RESOLVED in v1.2.0+** - Layer-wise caching enables weight caching for 32B models!

**Previous Problem (v1.1.x and earlier):** 32B models caused OOM when caching weights (62GB model + 62GB cache = 124GB > 141GB GPU).

**Current Solution (v1.2.0+):** Selective layer-wise caching reduces cache memory by 55-75%:
```bash
# This NOW WORKS on H200! (previously caused OOM)
bruno --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --cache-weights true \
  --n-trials 200

# Cache output confirms success:
# * Cache size: 28.4 GB (14,874,368,000 params, 43.2% of model)
```

**⚠️ CRITICAL: H100 80GB vs H200 141GB for 32B Models**

| GPU | VRAM | Model (64GB) + Cache (28GB) | Result |
|-----|------|----------------------------|--------|
| **H100 80GB** | 80GB | 92GB needed | ❌ **OOM - Does NOT fit** |
| **H200 141GB** | 141GB | 92GB needed | ✅ **Works with caching** |

**Why H100 80GB fails with cache-weights:**
- 32B model weights: ~64GB
- Layer-wise cache: ~28GB
- Total needed: ~92GB > 80GB available
- Error: `ModelLoadError: Insufficient memory to create weight cache`

**Memory breakdown (32B model on 141GB H200):**
- Model weights: 62-64GB
- Layer-wise cache: ~28GB (vs 62GB full cache)
- HuggingFace cache: 5-10GB
- Working memory: 10GB
- **Total: ~105-110GB** ✅ Fits comfortably on 141GB GPU

**GPU Selection for 32B Models:**
```bash
# H200 141GB - RECOMMENDED for 32B with caching
vastai search offers "gpu_name=H200 disk_space>=200 rentable=true" --order dph_total

# H100 80GB - Only works WITHOUT caching (slower)
bruno --model Qwen/Qwen2.5-Coder-32B-Instruct --cache-weights false
```

**Benefits of layer-wise caching (H200 only for 32B):**
- 6-12x faster reload (10-15s vs 60-120s disk reload)
- Saves 3-4 hours per 200-trial run
- ~$6-8 cost savings on cloud GPUs

### Circuit Ablation Not Working
**Problem:** Circuit-level ablation fails or returns 0 circuits ablated.

**Cause:** GQA (Grouped Query Attention) models are not compatible with circuit ablation.

**Affected models:** Llama 3.x, Qwen2.5, Moonlight, and other GQA models.

**Solution:** Use standard layer-level ablation instead:
```bash
# Disable circuit ablation for GQA models
bruno --use-circuit-ablation false
```

### CAA Extraction Failing
**Problem:** CAA (Contrastive Activation Addition) extraction fails.

**Cause:** The compliance direction extraction couldn't find a valid direction, often due to the model refusing all prompts or complying with all prompts.

**Behavior (v2.0.0+):** Bruno handles this gracefully with explicit user notification:
1. Prints a warning explaining why CAA extraction failed
2. Continues without CAA (abliteration still works, just without the "add compliance" step)
3. User is explicitly notified - **NOT a silent fallback**

**Manual override:** To disable CAA from the start:
```bash
bruno --use-caa false
```

### Moonlight-16B MoE Memory Requirements
**Problem:** Moonlight-16B-A3B-Instruct causes OOM on A100 80GB despite being "16B" model.

**Root Cause:** Moonlight is a Mixture of Experts (MoE) model. While only 3B parameters are *active* per token, the FULL model weights are ~63GB in BF16.

**Memory breakdown:**
| Component | Size |
|-----------|------|
| Model weights | 63 GB |
| BART-MNLI detector | 1.5 GB |
| Residual extraction | 24 GB |
| **Total needed** | **~88 GB** |

**GPU requirements:**
| GPU | VRAM | Status |
|-----|------|--------|
| A100 40GB | 40GB | Model doesn't fit |
| A100 80GB | 80GB | OOM during residuals |
| **H200 141GB** | 141GB | **Required** |

**Additional Moonlight requirements:**
- `pip install tiktoken` (required dependency)
- `transformers==4.48.2` (specific version required)
- `trust_remote_code=True` (handled in bruno v2.0.0+)

See `docs/ABLITERATION_CHECKLIST.md` for full Moonlight setup guide.

### HuggingFace Inference Endpoint Deployment

**Successfully deployed:** Moonlight-16B-A3B-Instruct-bruno on HuggingFace Inference Endpoints with vLLM.

#### Requirements for Moonlight Deployment

1. **GPU:** A100 80GB minimum (L40S 48GB insufficient - model is 63GB)
2. **Custom code files:** Model repository must contain:
   - `modeling_deepseek.py` (custom architecture)
   - `configuration_deepseek.py` (config class)
   - `tokenization_moonshot.py` (custom tokenizer)

3. **Configuration files properly formatted:**
   - `config.json` with `trust_remote_code: true`
   - `config.json` auto_map using local files (not remote repo references)
   - `tokenizer_config.json` auto_map format: `["tokenization_moonshot.TikTokenTokenizer", None]` (must be list)

4. **vLLM container argument:** `--trust-remote-code`

#### Step-by-Step Deployment Process

**Step 1: Prepare Model Repository**

Copy all custom code files to your abliterated model repo:
```python
from huggingface_hub import HfApi, hf_hub_download

api = HfApi(token=HF_TOKEN)

# Copy modeling files from original to abliterated repo
files_to_copy = [
    'modeling_deepseek.py',
    'configuration_deepseek.py',
    'tokenization_moonshot.py'
]

for filename in files_to_copy:
    file_path = hf_hub_download(
        repo_id='moonshotai/Moonlight-16B-A3B-Instruct',
        filename=filename
    )
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=filename,
        repo_id='rawcell/Moonlight-16B-A3B-Instruct-bruno',
        repo_type='model'
    )
```

**Step 2: Fix config.json auto_map**

Update to use local files instead of remote repo:
```python
import json

# Download, modify, upload config.json
config_path = api.hf_hub_download(
    repo_id='rawcell/Moonlight-16B-A3B-Instruct-bruno',
    filename='config.json'
)

with open(config_path) as f:
    config = json.load(f)

# Update auto_map to local files
config['auto_map'] = {
    'AutoConfig': 'configuration_deepseek.DeepseekV3Config',
    'AutoModel': 'modeling_deepseek.DeepseekV3Model',
    'AutoModelForCausalLM': 'modeling_deepseek.DeepseekV3ForCausalLM'
}
config['trust_remote_code'] = True

# Upload back
with open('temp_config.json', 'w') as f:
    json.dump(config, f, indent=2)

api.upload_file(
    path_or_fileobj='temp_config.json',
    path_in_repo='config.json',
    repo_id='rawcell/Moonlight-16B-A3B-Instruct-bruno',
    repo_type='model'
)
```

**Step 3: Fix tokenizer_config.json auto_map**

Must be a list format:
```python
config_path = api.hf_hub_download(
    repo_id='rawcell/Moonlight-16B-A3B-Instruct-bruno',
    filename='tokenizer_config.json'
)

with open(config_path) as f:
    tokenizer_config = json.load(f)

# CRITICAL: auto_map must be a list, not string
tokenizer_config['auto_map'] = {
    'AutoTokenizer': ['tokenization_moonshot.TikTokenTokenizer', None]
}

# Upload back
with open('temp_tokenizer_config.json', 'w') as f:
    json.dump(tokenizer_config, f, indent=2)

api.upload_file(
    path_or_fileobj='temp_tokenizer_config.json',
    path_in_repo='tokenizer_config.json',
    repo_id='rawcell/Moonlight-16B-A3B-Instruct-bruno',
    repo_type='model'
)
```

**Step 4: Create Inference Endpoint**

On HuggingFace Inference Endpoints UI:
1. Select model: `rawcell/Moonlight-16B-A3B-Instruct-bruno`
2. Hardware: **Nvidia A100 80GB** (minimum)
3. Region: Any AWS region
4. **Advanced Configuration** → **Container Arguments:** `--trust-remote-code`
5. Click "Create Endpoint"

**Step 5: Test Endpoint**

Use the provided chat script:
```bash
python examples/chat_endpoint.py https://YOUR-ENDPOINT-URL.endpoints.huggingface.cloud
```

**Cost:** ~$2.50/hour while running, auto-scales to zero after 1 hour of inactivity.

**Common Errors:**

| Error | Cause | Fix |
|-------|-------|-----|
| `ValueError: trust_remote_code=True` | Container arg not set | Add `--trust-remote-code` to Container Arguments |
| `ValueError: not enough values to unpack` | auto_map wrong format | Use list format: `["module.Class", None]` |
| `500 Internal Server Error` | GPU too small | Use A100 80GB minimum (not L40S 48GB) |
| Model loads wrong repo | auto_map points to original | Update auto_map to use local files |

### Ensemble Probe Failing
**Problem:** `SupervisedProbeError` about class imbalance or low accuracy.

**Cause:** Supervised probing requires balanced data (≥10 samples per class) and reasonable probe accuracy (≥0.65 by default).

**Behavior (v2.0.0+):** Bruno handles this gracefully with explicit user notification:
1. Prints a warning explaining why supervised probing failed
2. Continues with PCA-only extraction automatically
3. User is explicitly notified - **NOT a silent fallback**

**Example output:**
```
[yellow] * Ensemble extraction failed: Class imbalance prevents supervised probing: refusals=397, comply=3...[/yellow]
[yellow] * Continuing with PCA-only extraction[/yellow]
```

**Manual override:** To force PCA-only from the start:
```bash
bruno --ensemble-probe-pca false
```

### Concept Cones Failing
**Problem:** `ConceptConeError` about poor clustering quality or no valid cones.

**Cause:** The harmful prompts don't form distinct categories, or the model uses similar refusal patterns for all categories.

**Behavior (v2.0.0+):** Bruno handles this gracefully with explicit user notification:
1. Prints a warning explaining why concept cones failed
2. Continues with PCA-only extraction automatically
3. User is explicitly notified - **NOT a silent fallback**

**Manual override:** To disable concept cones from the start:
```bash
bruno --use-concept-cones false
```

### Warm-Start Failing
**Problem:** No warm-start profile found for the model family.

**Cause:** The model doesn't match any known family (llama, qwen, mistral, gemma, phi). Common with newer or less common models like Moonlight.

**Behavior (v2.0.0+):** Bruno handles this gracefully with explicit user notification:
1. Prints a warning explaining no profile was found
2. Continues with random initialization instead of crashing
3. User is explicitly notified - **NOT a silent fallback**

**Example output:**
```
[yellow]  * Warm-start failed: No profile found for model family 'unknown' (model: moonshotai/Moonlight-16B-A3B-Instruct)[/yellow]
[yellow]  * Continuing with random initialization (set --model-family to use warm-start)[/yellow]
```

**Manual override:** To specify a model family explicitly:
```bash
bruno --model-family qwen  # Use qwen profile for similar architectures
```

### pip install --force-reinstall Breaks Environment
**Problem:** After running `pip install --force-reinstall`, bruno fails with "Could not import module 'Qwen2ForCausalLM'".

**Solution:** `pip install transformers>=4.55.2`

**Prevention:** Never use `--force-reinstall`. Use `--upgrade` instead.

### PowerShell Scripts Fail from Bash/Codebuff
**Problem:** Running `.ps1` scripts from bash often fails.

**Solution:** Use direct SSH commands or `bruno-vast` CLI instead.

### Instance Stop vs Process Kill
- **Stopping a Vast.ai instance KILLS running processes** - no graceful save
- Check `bruno-vast progress` for "Models Saved" before stopping

### C4 Dataset Configuration Error
**Problem:** `ValueError: Config name is missing` when loading C4 dataset.

**Cause:** C4 and other multi-variant datasets require an explicit config parameter (e.g., "en", "realnewslike").

**Solution:** Pass ALL unhelpfulness_prompts fields via CLI:
```bash
# Via CLI - must pass ALL four fields together
bruno --model MODEL \
  --unhelpfulness-prompts.dataset allenai/c4 \
  --unhelpfulness-prompts.config en \
  --unhelpfulness-prompts.split "train[:200]" \
  --unhelpfulness-prompts.column text

# Via TOML (alternative - no CLI args for unhelpfulness_prompts)
[unhelpfulness_prompts]
dataset = "allenai/c4"
config = "en"
split = "train[:200]"
column = "text"
```

**Important:** CLI arguments ALWAYS override TOML config. If you pass ANY `--unhelpfulness-prompts.*` field on CLI, you must pass ALL of them or validation fails.

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
module-name = "bruno"
data-files = { "bruno" = ["config.default.toml"] }
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
pkill -f 'python.*bruno'
```

### Pydantic Settings Priority (CRITICAL)
**Priority order** (highest to lowest):
1. **CLI arguments** (e.g., `--model foo`)
2. Environment variables (e.g., `BRUNO_MODEL=foo`)
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
bruno --model MODEL

# This WORKS - CLI explicitly sets config
bruno --model MODEL --unhelpfulness-prompts.config en

# This WORKS - Don't pass ANY unhelpfulness args via CLI
# (but only if workspace doesn't have config.toml)
cd /workspace/empty_dir && bruno --model MODEL
```

**Best Practice:** Either pass ALL settings via CLI, OR ensure no CLI defaults conflict with TOML.

### unhelpfulness_prompts CLI Fields (CRITICAL)
**Problem:** Passing ANY unhelpfulness_prompts field via CLI requires ALL fields.

**Error:** `validation error for DatasetSpecification - Field required`

**Solution:** When using CLI for C4 dataset, pass ALL four fields:
```bash
# This FAILS - missing fields
bruno --model MODEL --unhelpfulness-prompts.config en

# This WORKS - all fields specified
bruno --model MODEL \
  --unhelpfulness-prompts.dataset allenai/c4 \
  --unhelpfulness-prompts.config en \
  --unhelpfulness-prompts.split "train[:200]" \
  --unhelpfulness-prompts.column text
```

---

## ⛔ DO NOT DO THESE THINGS ⛔

### NEVER judge instance status by uptime
- Vast.ai shared GPUs show ACCUMULATED uptime from all users
- **ONLY check if bruno process is running, NOT the uptime**

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

---

## Stopping Vast.ai Instances

Verify you're stopping the correct instance:

1. List ALL instances with full details:
```bash
vastai show instances --raw | python -c "import sys,json; data=json.load(sys.stdin); [print(f'ID: {i[\"id\"]}, Status: {i.get(\"actual_status\")}, SSH: {i.get(\"ssh_port\")}') for i in data]"
```

2. **Match the SSH port** to the one you've been using (e.g., port 15168)
3. Stop the CORRECT instance by ID:
```bash
vastai stop instance <CORRECT_ID>
```

4. **Verify it stopped:**
```bash
vastai show instances
```

**DO NOT blindly stop the first instance in the list.** Multiple instances may exist. The wrong one may already be stopped.

---

## Checking Training Progress (CORRECT METHOD)

**DO NOT trust Optuna values directly** - they may be normalized (0-1) not raw counts.

**Step 1: Find the active log file and database**
```bash
ssh -p PORT root@HOST "ls -la /workspace/*.log /workspace/*.db; ps aux | grep bruno"
```

**Step 2: Get actual refusal counts from logs (MOST RELIABLE)**
```bash
ssh -p PORT root@HOST "grep -E 'Refusals:' /workspace/ACTIVE_LOG.log | tail -30"
```
This shows raw counts like "Refusals: 79/104" - use these numbers.

**Step 3: Get trial count and progress**
```bash
ssh -p PORT root@HOST "tail -50 /workspace/ACTIVE_LOG.log | grep -E 'Trial|Elapsed|remaining'"
```

**WRONG: Trusting Optuna values[1] directly**
- Optuna stores normalized values (0.0-1.0), not raw counts
- values[1]=0.8125 means 84/104 refusals, NOT 0 or 1 refusal
- The commander agent may misinterpret these

**CORRECT: Always grep the log for "Refusals: X/Y" format**
- This gives actual counts
- No interpretation needed
- User can verify directly
