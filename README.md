# Heretic

[![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/badge/pypi-heretic--llm-blue.svg)](https://pypi.org/project/heretic-llm/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Tool for modifying language model behaviors through activation direction analysis and Optuna-based optimization.

## What It Does

Heretic identifies and modifies behavioral directions encoded in language model weights. It uses contrastive activation analysis to extract direction vectors, then applies orthogonalization to remove unwanted behaviors while preserving model capabilities.

The technique works for any behavior with distinguishable activation patterns:
- **Refusal behaviors** (most common use case - "abliteration")
- **Verbosity/padding** (tested and validated)
- **Hedging language** (framework ready)
- **Sycophancy** (future research)
- **Other behavioral patterns** (experimental)

## Features

**Core Capabilities:**
- Automatic hyperparameter optimization using Optuna TPE sampler
- Multi-objective optimization balancing behavior removal and capability preservation
- Resume support via SQLite storage (can resume interrupted experiments)
- Pareto-optimal trial selection from optimization results
- Direct HuggingFace Hub upload of modified models
- Interactive chat interface for testing modifications

**Advanced Techniques:**
- Neural refusal detection using zero-shot NLI (catches soft refusals)
- Supervised probing + ensemble extraction for robust direction vectors
- Activation-based calibration for adaptive weight scaling
- Concept cone extraction for category-specific ablation
- Contrastive Activation Addition (CAA) for combined removal + addition
- Circuit-level ablation targeting specific attention heads
- Warm-start parameter transfer using model family profiles
- **Sacred Direction Preservation** - Orthogonalize against capability directions (MMLU) to protect model abilities

**Performance Optimizations:**
- GPU-accelerated PCA (15-20x faster than CPU, ~5 min for 32B models)
- In-memory weight caching (5-10x faster trial reset)
- torch.compile() support (1.5-2x inference speedup)
- Early stopping for refusal detection (40-60% faster evaluation)
- Parallel evaluation (KL divergence and refusal counting)

**Cloud GPU Tools:**
- Dedicated `heretic-vast` CLI for Vast.ai management
- GPU tier presets (RTX_4090, A6000, A100, H100)
- Live terminal dashboard with real-time monitoring
- Docker image for RunPod/Vast.ai deployment

## Installation

### From PyPI

```bash
pip install heretic-llm
```

### From Source

```bash
git clone https://github.com/p-e-w/heretic
cd heretic
uv sync --all-extras --dev
```

### Using Docker

```bash
docker pull quanticsoul4772/heretic
docker run --gpus all -it quanticsoul4772/heretic heretic --help
```

## Quick Start

### Basic Usage

```bash
# Interactive mode (select trial from Pareto frontier)
heretic Qwen/Qwen2.5-7B-Instruct

# Fully automated (headless operation)
heretic Qwen/Qwen2.5-7B-Instruct --auto-select --hf-upload username/model-heretic

# Quick test with fewer trials
heretic Qwen/Qwen2.5-7B-Instruct --n-trials 50
```

### Cloud GPU Deployment

#### Option 1: Vast.ai CLI (Recommended)

```bash
# Setup
export VAST_API_KEY='your-api-key'
pip install heretic-llm fabric rich

# Run abliteration on cloud
heretic-vast create A100_80GB 1          # Rent GPU instance
heretic-vast setup                        # Install heretic
heretic-vast run Qwen/Qwen2.5-32B-Instruct
heretic-vast watch                        # Monitor progress
heretic-vast download MODEL_NAME          # Download results
heretic-vast stop                         # Stop billing
```

See [WORKFLOW.md](WORKFLOW.md) for detailed cloud GPU instructions.

#### Option 2: Docker on RunPod/Vast.ai

```bash
docker run --gpus all -e HF_TOKEN=your_token -it quanticsoul4772/heretic \
    heretic Qwen/Qwen2.5-7B-Instruct --auto-select --hf-upload user/model-heretic
```

## Configuration

### Command-Line Flags

**Essential Flags:**
```bash
--model MODEL              # HuggingFace model ID or local path
--n-trials N               # Number of optimization trials (default: 200)
--auto-select              # Auto-select and save best trial (headless mode)
--hf-upload REPO           # Upload to HuggingFace Hub
--storage sqlite:///file   # Resume support (recommended)
```

**Performance Flags:**
```bash
--compile                  # Enable torch.compile() (1.5-2x faster)
--batch-size N             # Batch size (0 = auto-detect)
--cache-weights BOOL       # In-memory caching (default: true, set false for 32B+)
```

**Advanced Features:**
```bash
--use-neural-refusal-detection    # Zero-shot NLI detection (default: true)
--ensemble-probe-pca              # Supervised + PCA ensemble (default: true)
--use-activation-calibration      # Adaptive scaling (default: true)
--use-concept-cones               # Category-specific ablation (default: false)
--use-caa                         # Contrastive Activation Addition (default: false)
--use-circuit-ablation            # Attention head targeting (default: false, no GQA)
--use-warm-start-params           # Model family warm-start (default: true)
--use-sacred-directions           # Preserve capabilities via MMLU directions (default: false)
```

### Configuration File

Create `config.toml` for complex setups:

```toml
model = "Qwen/Qwen2.5-32B-Instruct"
n_trials = 200
batch_size = 8
cache_weights = false  # Required for 32B+ on multi-GPU

# Resume support
storage = "sqlite:///heretic_study.db"
study_name = "qwen32b-abliteration"

# Auto-save
auto_select = true
auto_select_path = "./models/Qwen2.5-32B-Instruct-heretic"

# Dataset configuration
[bad_prompts]
dataset = "p-e-w/refusal_direction"
split = "train"
column = "rejected"

[unhelpfulness_prompts]
dataset = "allenai/c4"
config = "en"  # Required for C4
split = "train[:200]"
column = "text"
```

See [configs/](configs/) for example configurations.

## Model Size Guidelines

| Size | GPU VRAM | Disk Space | cache_weights | C4 config |
|------|----------|------------|---------------|-----------|
| 7B   | 24GB     | 100GB      | true          | not needed |
| 13B  | 24GB     | 150GB      | false         | not needed |
| 32B  | 80GB     | 200GB      | false         | required (`en`) |
| 70B  | 80GB+    | 300GB      | false         | required (`en`) |

**Note:** v1.1.0+ streams C4 dataset on-demand (no disk overhead). Network required during loading.

## Advanced Usage

### Custom Behavioral Directions

Heretic can extract any behavioral direction, not just refusals:

```bash
# Verbosity modification (tested)
cp experiments/verbosity/config.verbosity.toml config.toml
heretic --model meta-llama/Llama-3.1-8B-Instruct

# Hedging language removal (framework ready)
cp experiments/hedging/config.hedging.toml config.toml
heretic --model meta-llama/Llama-3.1-8B-Instruct
```

See [experiments/](experiments/) for experimental behavioral directions.

### Sacred Direction Preservation (NEW)

Protect model capabilities during abliteration by orthogonalizing against capability-encoding directions:

```bash
heretic <model> --use-sacred-directions true --n-sacred-directions 5
```

**How it works:**
1. Extract "sacred" directions from MMLU questions (encode reasoning capabilities)
2. Measure overlap between refusal direction and sacred directions
3. Orthogonalize refusal direction to remove any capability-damaging components
4. Abliterate with the safe, orthogonalized direction

**Expected impact:** +20-30% capability preservation (reduced MMLU accuracy drop)

### Validation Framework

Enable validation to measure abliteration effectiveness:

```bash
heretic <model> --enable-validation --run-mmlu-validation
```

Measures:
- Refusal rate reduction
- KL divergence (capability preservation)
- MMLU accuracy (optional, comprehensive capability test)

### Resume Support

```bash
# First run
heretic <model> --storage sqlite:///study.db --study-name experiment1

# Resume after interruption
heretic <model> --storage sqlite:///study.db --study-name experiment1
```

Optuna automatically resumes from last completed trial.

## Architecture

**Core Components:**
- `Model` - HuggingFace model wrapper with abliteration operations
- `Evaluator` - Multi-objective scoring (KL divergence + refusal counting)
- `Settings` - Pydantic configuration with CLI/env/file support + validators
- `heretic-vast` - Cloud GPU management CLI for Vast.ai
- `phases/` - Modular pipeline components (dataset loading, direction extraction, optimization, saving)
- `constants` - Centralized thresholds and magic numbers

**Abliteration Pipeline (Modular Phases):**
1. **Dataset Loading** (`phases.dataset_loading`) - Load good/bad/sacred prompts
2. **Direction Extraction** (`phases.direction_extraction`) - Extract refusal directions, apply orthogonalization
3. **Optimization** (`phases.optimization`) - Optuna TPE optimization with warm-start
4. **Model Saving** (`phases.model_saving`) - Save locally or upload to HuggingFace

## Error Handling

Heretic includes comprehensive error handling with:
- Custom exception hierarchy (22 exception types)
- Specific error messages with 2-3 actionable solutions
- Input validation and security hardening
- Enhanced logging with file rotation
- Network retry logic with exponential backoff

All errors provide clear guidance for resolution.

## Troubleshooting

**Common Issues:**

**GPU Out of Memory:**
```bash
# Use smaller batch size
heretic <model> --batch-size 4 --max-batch-size 16

# Disable weight caching for large models
heretic <model> --cache-weights false

# Use smaller model or quantization
```

**C4 Dataset Errors:**
```bash
# C4 requires config parameter
heretic <model> --unhelpfulness-prompts.config en

# C4 requires sample count in split
# Use: train[:200] not just train
```

**Model Loading Failures:**
```bash
# Gated models (Llama, etc.) require authentication
huggingface-cli login

# Or set token
export HF_TOKEN='your-token'
```

See [LESSONS_LEARNED.md](LESSONS_LEARNED.md) for comprehensive troubleshooting.

## Performance Benchmarks

**H200 GPU (Qwen2.5-Coder-32B):**
- Model download: ~5 min (one-time, cached)
- PCA extraction: ~5 min (GPU-accelerated, was 4-6 hrs on CPU)
- Single trial: ~3 min
- 200 trials: ~10 hrs (total cost: ~$22)

**4x RTX 4090 (Qwen2.5-Coder-32B):**
- 200 trials: ~30-35 hrs (total cost: ~$47-55)
- Requires `device_map="balanced"` and `cache_weights=false`

## Documentation

**User Guides:**
- [README.md](README.md) - This file
- [WORKFLOW.md](WORKFLOW.md) - Cloud GPU comprehensive guide
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Command cheatsheet
- [LESSONS_LEARNED.md](LESSONS_LEARNED.md) - Troubleshooting

**Development:**
- [CLAUDE.md](CLAUDE.md) - AI assistant guide
- [ROADMAP.md](ROADMAP.md) - Future research directions
- [docs/](docs/) - Planning and implementation tracking
- [claudedocs/](claudedocs/) - Technical analyses

**Examples & Configs:**
- [examples/](examples/) - Example applications (chat interface)
- [configs/](configs/) - Example configuration files
- [scripts/](scripts/) - Utility scripts

## Development

```bash
# Install development dependencies
uv sync --all-extras --dev

# Format code
uv run ruff format .

# Lint
uv run ruff check --extend-select I .

# Run tests
uv run pytest

# Build package
uv build
```

## Contributing

Contributions welcome. This fork focuses on:
- Error handling improvements
- Performance optimizations
- Cloud GPU workflow automation
- Advanced abliteration techniques

See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for planned improvements.

## Resources

**Research:**
- [Paper: Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
- [Abliterated Models Collection](https://huggingface.co/collections/p-e-w/the-bestiary)

**Repositories:**
- [Original Project](https://github.com/p-e-w/heretic) by Philipp Emanuel Weidmann
- [This Fork](https://github.com/quanticsoul4772/abliteration-workflow) - Extended features

## License

AGPL-3.0-or-later

Copyright (C) 2025 Philipp Emanuel Weidmann <pew@worldwidemann.com>

See [LICENSE](LICENSE) for full license text.
