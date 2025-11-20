# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Heretic is a tool for automatic censorship removal (abliteration) from language models using Optuna-based hyperparameter optimization. It computes refusal directions from residual activations and orthogonalizes model weight matrices to reduce refusal behavior while minimizing KL divergence from the original model.

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
1. Load model and tokenizer via `Model` class
2. Load good/bad prompt datasets for direction calculation
3. Auto-detect optimal batch size if `batch_size=0`
4. Compute per-layer refusal directions from residual activations
5. Run Optuna optimization loop with TPESampler
6. Present Pareto-optimal trials for user selection
7. Allow saving/uploading/chatting with abliterated model

### Key Components

**`Model` (`src/heretic/model.py`)**
- Wraps HuggingFace model loading with dtype fallback chain
- `get_layers()`: Handles both text-only and multimodal model architectures
- `get_layer_matrices()`: Extracts abliterable weight matrices (attn.o_proj, mlp.down_proj) supporting dense and MoE variants
- `abliterate()`: Applies orthogonalization with per-layer weight interpolation
- `get_residuals_batched()`: Extracts hidden states for refusal direction computation

**`Evaluator` (`src/heretic/evaluator.py`)**
- Computes KL divergence from first-token probability distributions
- Counts refusals using configurable marker strings
- Returns multi-objective score tuple `(kl_divergence, refusals)`

**`Settings` (`src/heretic/config.py`)**
- Pydantic-based configuration with layered sources: CLI args > env vars > config.toml
- Key settings: `n_trials`, `batch_size`, `dtypes`, `refusal_markers`

### Optimization Strategy
- Multivariate TPE sampler with `n_startup_trials` random exploration
- Parameters per component: `max_weight`, `max_weight_position`, `min_weight`, `min_weight_distance`
- Direction scope: global (single layer index) or per-layer
- Dual objective: minimize both KL divergence and refusal count

## Configuration

Settings cascade: CLI > environment (HERETIC_ prefix) > config.toml > defaults

Key parameters in `config.toml`:
- `model`: HuggingFace model ID or local path
- `n_trials`: Optimization iterations (default 200)
- `batch_size`: 0 for auto-detection
- `max_batch_size`: Upper limit for auto-detection
- `dtypes`: Fallback chain for model loading precision

## Testing

No test suite currently exists. CI runs formatting, linting, and build verification only.
