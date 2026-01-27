# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Heretic is a tool for **neural behavior modification** in language models using activation direction analysis and Optuna-based optimization. While best known for abliteration (refusal removal), the technique is general and can extract/modify any behavioral direction encoded in model weights.

**Vision:** A personal neural engineering workbench for understanding and reshaping LLM behavior at the weight level. See [ROADMAP.md](ROADMAP.md) for full vision and research directions.

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
2. Load contrastive prompt datasets (good/bad) for direction calculation
3. Auto-detect optimal batch size if `batch_size=0`
4. Compute per-layer behavioral directions from residual activations
5. Run Optuna optimization loop with TPESampler
6. Present Pareto-optimal trials for user selection
7. Allow saving/uploading/chatting with modified model

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

## Chat Interface

**`chat_app.py`** - Gradio-based chat UI for abliterated models

### Architecture
- `ModelManager` class: Handles model loading, caching, and streaming text generation
- Custom exception hierarchy: `HereticError` base class with specific exceptions:
  - `ModelNotFoundError`, `ModelValidationError`, `ModelLoadError`
  - `CUDAOutOfMemoryError`, `TokenizationError`, `GenerationError`
- Structured logging via Python's `logging` module (logger: `heretic_chat`)

### Key Features
- Auto-discovers models in `models/` directory with validation
- Validates model files before loading (config.json, weights, tokenizer)
- Streaming token generation via `TextIteratorStreamer`
- Real-time GPU memory monitoring display
- Chat history persistence to `chat_history/` as JSON
- Model caching to avoid reloading between messages
- Cross-model tokenization compatibility (handles Llama, Qwen, etc.)
- Clean monochrome theme using Gradio CSS variables

### Configuration
- `MODELS_DIR`: Path to models directory (default: `./models`)
- `CHAT_HISTORY_DIR`: Path to chat history (default: `./chat_history`)
- Server runs on `0.0.0.0:7860` by default

Run with: `python chat_app.py`

## Vast.ai CLI (`src/heretic/vast.py`)

Dedicated CLI for Vast.ai GPU cloud management:

```bash
heretic-vast create A100_80GB 2   # Create 2x A100 instance
heretic-vast setup                 # Install heretic
heretic-vast run MODEL             # Run abliteration
heretic-vast watch                 # Live dashboard
heretic-vast stop                  # Stop billing
```

### Key Components
- `VastConfig`: Configuration from env vars / .env file
- `GPU_TIERS`: Preset configurations for different GPU types
- `get_connection()`: Fabric SSH connection management
- `watch_dashboard()`: Rich live terminal dashboard

### Dependencies
- `fabric`: SSH connections
- `rich`: Terminal UI (tables, panels, live display)
- `click`: CLI framework

## Experiments Framework

Experiments for testing new behavioral directions live in `experiments/`:

```
experiments/
└── verbosity/              # Verbosity direction spike
    ├── README.md           # Experiment documentation
    ├── concise_prompts.json    # Factual/closed questions (200)
    ├── verbose_prompts.json    # Open-ended questions (200)
    ├── config.verbosity.toml   # Heretic config for experiment
    ├── load_local_dataset.py   # Convert JSON to HF Dataset
    ├── eval_verbosity.py       # Measure response length changes
    └── run_spike.ps1           # Windows automation script
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

## Testing

No test suite currently exists. CI runs formatting, linting, and build verification only.
