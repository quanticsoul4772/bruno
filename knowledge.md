# Project Knowledge

This file gives Codebuff context about the Heretic project: goals, commands, conventions, and gotchas.

## Quickstart

- **Setup:** `uv sync --all-extras --dev`
- **Dev:** `uv run heretic <model-name>`
- **Test:** `uv run ruff check .` (linting), `uv run ruff format .` (formatting)
- **Build:** `uv build`

## Architecture

### Key Directories
- `src/heretic/` - Core abliteration logic
  - `main.py` - Entry point and Optuna optimization loop
  - `model.py` - Model loading, abliteration methods, and all Phase 1-7 improvements
  - `config.py` - Pydantic-based configuration with 50+ settings
  - `evaluator.py` - KL divergence and refusal detection
  - `transfer.py` - Phase 7: Cross-model hyperparameter transfer profiles
  - `validation.py` - Validation framework for measuring abliteration effectiveness
  - `vast.py` - Vast.ai cloud CLI
- `experiments/` - Behavioral direction experiments (verbosity, hedging)
- `models/` - Local modified models (gitignored)

### Data Flow
1. Load model and contrastive prompt datasets (good/bad)
2. Extract per-layer residual activations
3. Compute refusal directions (PCA, supervised probe, or ensemble)
4. Apply optional enhancements (orthogonalization, calibration, concept cones)
5. Run Optuna optimization to find optimal abliteration parameters
6. Present Pareto-optimal trials for selection
7. Save/upload modified model

## Phase 1-7 Advanced Improvements

### Phase 1: Neural Refusal Detection
- `use_neural_refusal_detection` (default: True) - Zero-shot NLI for detecting soft refusals
- `neural_detection_for_optuna` (default: True) - Use neural detection during optimization

### Phase 2: Supervised Probing
- `use_supervised_probing` - Train linear classifiers to find refusal directions
- `ensemble_probe_pca` (default: True) - Combine probe and PCA directions
- `min_probe_accuracy` - Minimum cross-validation accuracy threshold

### Phase 3: Activation Calibration
- `use_activation_calibration` (default: True) - Scale weights based on activation strength
- Computes RefusalActivationStats for adaptive weight scaling

### Phase 4: Concept Cones
- `use_concept_cones` - Cluster harmful prompts by category (violence, fraud, etc.)
- Extracts category-specific refusal directions

### Phase 5: Contrastive Activation Addition (CAA)
- `use_caa` - Add compliance direction alongside refusal removal
- Orthogonality checking to prevent undoing ablation

### Phase 6: Circuit-Level Ablation
- `use_circuit_ablation` - Target specific attention heads instead of layers
- WARNING: Not compatible with GQA models (Llama 3.x, Qwen2.5)

### Phase 7: Warm-Start Transfer
- `use_warm_start_params` (default: True) - Use model family profiles for faster Optuna convergence
- Profiles for: llama, qwen, mistral, gemma, phi

## Conventions

### Formatting/Linting
- Use `ruff format .` for formatting
- Use `ruff check --extend-select I .` for linting with import sorting
- Follow existing code style in each file

### Patterns to Follow
- Comprehensive type hints on all functions/methods
- Dataclasses for structured data (AbliterationParameters, RefusalActivationStats, etc.)
- Explicit error handling with descriptive RuntimeError messages
- Use `empty_cache()` after large tensor operations

### Things to Avoid
- No type casting to `any` - keep proper types
- No silent failures - raise RuntimeError with helpful messages
- No print statements in library code - use the custom `print` from utils
- Don't assume packages are available - check config/imports first
- Never use `pip install --force-reinstall` - breaks dependencies

## Key Gotchas

1. **GQA models** (Llama 3.x, Qwen2.5) are not compatible with circuit-level ablation
2. **32B+ models** need `cache_weights=false` to avoid OOM
3. **Multi-GPU setups** need `device_map="balanced"` not "auto"
4. **Supervised probing** requires minimum 10 samples per class
5. **Concept cones** require good silhouette score (>0.1) or fall back to PCA
6. **CAA** requires at least 10 refusals and 10 compliant responses
