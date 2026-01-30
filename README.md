# Heretic - Neural Behavior Modification for LLMs

Heretic is a tool for surgical behavior modification in language models using activation direction analysis and Optuna-based optimization.

While best known for **abliteration** (removing refusal behaviors), heretic's technique is general: it can extract and modify *any* behavioral direction encoded in model weights - verbosity, hedging, sycophancy, and more.

> **Vision:** Build a personal neural engineering workbench for understanding and reshaping how language models behave at the weight level. See [ROADMAP.md](ROADMAP.md) for the full vision.

## Quick Start

```bash
# Install
pip install heretic-llm

# Run abliteration (interactive)
heretic Qwen/Qwen3-4B-Instruct-2507

# Run abliteration (fully automated)
heretic Qwen/Qwen3-4B-Instruct-2507 --auto-select --hf-upload username/model-heretic
```

## Cloud GPU Deployment

For users without local GPU access, use our automation scripts or Docker image.

### Option 1: Docker Image (Recommended)

Pre-built Docker image works on RunPod, Vast.ai, and local GPU setups:

```bash
# Pull and run
docker run --gpus all -it quanticsoul4772/heretic heretic Qwen/Qwen3-4B-Instruct-2507

# With HuggingFace token for gated models
docker run --gpus all -e HF_TOKEN=your_token -it quanticsoul4772/heretic \
    heretic meta-llama/Llama-3.1-8B-Instruct --auto-select

# With persistent cache
docker run --gpus all -v heretic-cache:/workspace/.cache -it quanticsoul4772/heretic \
    heretic Qwen/Qwen3-4B-Instruct-2507 --auto-select --hf-upload user/model-heretic
```

**On RunPod:** Use `quanticsoul4772/heretic` as your Docker image when creating a pod.

**On Vast.ai:** Select `quanticsoul4772/heretic` as your Docker image when renting a GPU.

### Option 2: RunPod Automation Script (Windows)

```powershell
# Initial setup (one-time)
.\runpod.ps1 install-runpodctl  # Download CLI tool
.\runpod.ps1 check-tools        # Verify prerequisites

# Run abliteration
.\runpod.ps1 create-pod         # Create RTX 4090 pod (~$0.34/hr)
.\runpod.ps1 setup              # Install heretic
.\runpod.ps1 run Qwen/Qwen3-4B-Instruct-2507
.\runpod.ps1 stop-pod           # Stop billing when done
```

### Option 3: Vast.ai CLI (Recommended for Large Models)

Heretic includes a dedicated Python CLI for Vast.ai with a rich terminal dashboard:

```bash
# Install the CLI
pip install heretic-llm

# Or install with Vast.ai support
pip install heretic-llm fabric rich

# Quick start
heretic-vast create A100_80GB 2    # Create 2x A100 80GB instance
heretic-vast setup                  # Install heretic on instance
heretic-vast run Qwen/Qwen2.5-72B-Instruct
heretic-vast watch                  # Live monitoring dashboard
heretic-vast stop                   # Stop when done
```

**Key Features:**
- GPU tier presets: RTX_4090, A6000, A100_40GB, A100_80GB, H100
- Live dashboard with GPU utilization, memory, and progress
- Automatic SSH key handling via Fabric
- Model download from remote instances

See `heretic-vast --help` for all commands.

See [WORKFLOW.md](WORKFLOW.md) for detailed instructions.

## Features

- Automatic optimization using Optuna TPE sampler for multi-objective hyperparameter tuning
- Resume support via persistent SQLite storage
- Pareto-optimal trial selection balancing capability preservation and behavior modification
- Multi-GPU support with automatic or manual device mapping
- HuggingFace integration for direct model upload
- Fully automated mode for headless operation
- Gradio chat interface for testing modified models
- Cloud CLI for Vast.ai GPU management
- Experiments framework for testing new behavioral directions

### Advanced Abliteration

Heretic includes state-of-the-art abliteration improvements:

| Feature | Default | Description |
|---------|---------|-------------|
| **Neural Refusal Detection** | ✅ ON | Zero-shot NLI catches soft refusals and evasive responses |
| **Supervised Probing + Ensemble** | ✅ ON | Linear probes combined with PCA for better direction extraction |
| **Activation Calibration** | ✅ ON | Adaptive weight scaling based on refusal activation strength |
| **Concept Cones** | OFF | Category-specific directions (violence, fraud, self-harm, etc.) |
| **Contrastive Activation Addition** | OFF | Add compliance direction alongside refusal removal |
| **Circuit-Level Ablation** | OFF | Target specific attention heads (⚠️ Not for GQA models) |
| **Warm-Start Transfer** | ✅ ON | Model family profiles for 2x faster Optuna convergence |

### Performance Optimizations

- In-memory weight caching (5-10x faster trial reset vs disk reload)
- Optional torch.compile() support (1.5-2x inference speedup)
- Early stopping for refusal detection (40-60% faster evaluation)
- Parallel KL divergence and refusal counting
- Model family warm-start for faster optimization convergence

## Requirements

- Python >= 3.10
- CUDA-capable GPU (16GB+ VRAM recommended)
- Or use Docker with `--gpus all`
- Or use RunPod/Vast.ai for cloud GPU access

## Important Notes

### Large Model Support (32B+)

For models with >14B parameters on multi-GPU systems, special configuration is required:

- **Use `device_map="balanced"`** for even GPU distribution (not "auto")
- **Set `cache_weights=false`** (required for multi-GPU, incompatible with weight caching)
- **Set `iterative_rounds=0`** to reduce memory pressure
- **Use `batch_size=4`** for stable performance

See `LESSONS_LEARNED.md` for detailed troubleshooting and configuration guidance.

**Expected runtime for 32B models:**
- 4x RTX 4090: ~30-35 hours for 200 trials
- 1x A100 80GB: ~12-15 hours for 200 trials (faster with weight caching enabled)

### GQA Model Limitations

Models using Grouped Query Attention (GQA) have limited feature support:

- **Affected models:** Llama 3.x, Qwen 2.5, and others with `num_kv_heads != num_attention_heads`
- **Circuit-level ablation:** ❌ Not supported (use standard layer-level ablation)
- **All other features:** ✅ Fully supported

Heretic will automatically detect GQA models and raise a clear error if incompatible features are enabled.

## Installation

```bash
# From PyPI
pip install heretic-llm

# From source
git clone https://github.com/p-e-w/heretic
cd heretic
uv sync --all-extras --dev
uv run heretic <model-name>

# Using Docker
docker pull quanticsoul4772/heretic
docker run --gpus all -it quanticsoul4772/heretic heretic --help
```

## Usage

```bash
# Basic usage (interactive)
heretic <model-name>

# Quick test with fewer trials
heretic <model-name> --n-trials 50

# Fully automated (for cloud/CI)
heretic <model-name> --auto-select --hf-upload username/model-heretic

# With custom config
heretic --model <model-name> --config config.toml

# Examples
heretic Qwen/Qwen3-4B-Instruct-2507
heretic meta-llama/Llama-3.1-8B-Instruct --n-trials 100
heretic mistralai/Mistral-7B-Instruct-v0.3 --auto-select --auto-select-path ./output
```

### Automation Flags

| Flag | Description |
|------|-------------|
| `--auto-select` | Auto-select best trial and save without prompts |
| `--auto-select-path PATH` | Custom output path (default: `./<model>-heretic`) |
| `--hf-upload REPO` | Upload to HuggingFace (e.g., `user/model-heretic`) |
| `--hf-private` | Make HuggingFace repo private |
| `--n-trials N` | Number of trials (default: 200) |
| `--compile` | Enable torch.compile() for faster inference |
| `--storage URL` | Optuna storage URL for resume support (e.g., `sqlite:///study.db`) |
| `--study-name NAME` | Optuna study name (default: `heretic_study`) |
| `--refusal-check-tokens N` | Tokens for refusal detection (default: 30) |

### Advanced Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--use-neural-refusal-detection` | true | Zero-shot NLI refusal detection |
| `--ensemble-probe-pca` | true | Combine supervised probe + PCA |
| `--use-activation-calibration` | true | Adaptive weight scaling |
| `--use-warm-start-params` | true | Model family warm-start |
| `--use-concept-cones` | false | Category-specific directions |
| `--use-caa` | false | Contrastive Activation Addition |
| `--use-circuit-ablation` | false | Attention head targeting (⚠️ No GQA) |
| `--model-family` | auto | Override detected family (llama/qwen/mistral/gemma/phi) |

## Configuration

Copy `config.default.toml` to `config.toml` and customize:

```toml
# Model settings
model = "Qwen/Qwen3-4B-Instruct-2507"
dtype = "auto"           # float16, bfloat16, or auto
batch_size = 0           # 0 = auto-detect
compile = false          # Enable torch.compile() for faster inference

# Optimization settings
n_trials = 100           # Number of Optuna trials
storage = "sqlite:///heretic_study.db"  # Resume support
study_name = "heretic_study"

# Evaluation settings
refusal_check_tokens = 30  # Fewer tokens = faster evaluation

# Advanced features (all enabled by default)
use_neural_refusal_detection = true   # Zero-shot NLI detection
ensemble_probe_pca = true             # Probe + PCA ensemble
use_activation_calibration = true     # Adaptive weight scaling
use_warm_start_params = true          # Model family warm-start
enable_validation = true              # Validation framework

# Optional advanced features (disabled by default)
use_concept_cones = false             # Category clustering
use_caa = false                       # Compliance direction addition
use_circuit_ablation = false          # Attention head targeting
```

### Model Compatibility Notes

| Model Family | Neural Detection | Supervised Probing | Circuit Ablation | Warm-Start |
|--------------|------------------|-------------------|------------------|------------|
| Llama 3.x | ✅ | ✅ | ❌ (GQA) | ✅ |
| Qwen 2.5 | ✅ | ✅ | ❌ (GQA) | ✅ |
| Mistral | ✅ | ✅ | ✅ | ✅ |
| Gemma | ✅ | ✅ | ✅ | ✅ |
| Phi | ✅ | ✅ | ✅ | ✅ |

## How It Works

1. **Load model** - Loads the target model with automatic dtype selection and multi-GPU distribution
2. **Extract behavioral directions** - Computes per-layer activation patterns from contrastive prompt pairs
3. **Apply enhancements** - Neural detection, supervised probing, activation calibration, warm-start
4. **Optimize modification** - Uses Optuna to find optimal parameters that modify behavior while preserving capabilities (measured by KL divergence)
5. **Validate results** - Measure refusal reduction and capability preservation
6. **Select result** - Presents Pareto-optimal trials for user selection
7. **Export** - Save locally or upload to HuggingFace

### The Core Technique

Heretic uses **activation direction analysis** to find and remove behavioral tendencies:

```
1. FIND direction    → Compare activations on contrastive prompts (PCA + supervised probing)
2. CALIBRATE         → Scale weights based on activation strength
3. PROJECT it out    → Orthogonalize weight matrices against that direction
4. OPTIMIZE          → Find intensity that modifies behavior without destroying capability
5. VALIDATE          → Measure improvement with neural refusal detection
```

This technique is general - it works for any behavior encoded as a direction in activation space.

### Advanced Features

**Neural Refusal Detection:** Uses zero-shot NLI to catch soft refusals, evasive responses, and novel refusal patterns that string matching misses.

**Ensemble Direction Extraction:** Combines supervised linear probes with contrastive PCA for more accurate refusal direction identification.

**Activation Calibration:** Measures how strongly prompts activate the refusal direction and scales ablation weights accordingly - stronger activations get stronger ablation.

**Warm-Start Transfer:** Uses pre-computed hyperparameter profiles for model families (Llama, Qwen, Mistral, Gemma, Phi) to initialize Optuna, achieving ~2x faster convergence.

## Web Search Feature

The chat interface includes automatic web search capabilities to augment responses with current information.

### How It Works

The WebSearcher automatically detects when your question needs current information and searches the web. Search results are injected into the conversation context before the model generates its response.

### Automatic vs Explicit Search

**Automatic Search** triggers when your message contains:
- Time-sensitive words: "today", "current", "latest", "recent"
- Information requests: "news", "update", "trending"
- Questions about dynamic data: "price", "weather", "score"
- Direct requests: "search for", "look up", "find out"

**Explicit Search** using the `/search` command:
```
/search latest AI news
/search python 3.13 release date
/search weather in Tokyo
```

### Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| Backend | DuckDuckGo | No API key required |
| Region | Worldwide (`wt-wt`) | No regional bias |
| Max Results | 5 | Results per search |
| Enable/Disable | Checkbox in UI | Toggle in Advanced Settings |

### Failure Handling

Web search is designed to fail gracefully:

- **Search fails:** Model answers without web context, shows warning
- **No results:** Model acknowledges search was attempted but found nothing
- **Network issues:** Treated as search failure, continues without results

The chat will never crash due to search issues - it simply continues without web augmentation.

### Disabling Web Search

In the chat interface, expand "Advanced Settings" and uncheck "Enable Web Search" to disable automatic searches. You can still use `/search` for explicit searches when needed.

---

## Chat Interface

Heretic includes a Gradio-based chat interface for testing abliterated models.

### Quick Start

```bash
# Install dependencies
pip install gradio transformers torch accelerate

# Run the chat app
python chat_app.py
```

Opens web UI at `http://localhost:7860` with model selection, streaming responses, GPU memory monitoring, and chat history persistence.

### Features

- Type-safe implementation with comprehensive type hints
- Custom exception handling (CUDA OOM, model validation, tokenization errors)
- Structured logging with configurable log levels
- Cross-model compatibility (Llama, Qwen, and other architectures)
- Model file validation before loading

### Using Your Models

Place abliterated models in the `models/` directory:

```
models/
  llama-3.2-3b-heretic/
    config.json
    model.safetensors
    tokenizer.json
    ...
  qwen2.5-3b-heretic/
    ...
```

Or download from HuggingFace:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="rawcell/Llama-3.2-3B-Instruct-heretic",
    local_dir="models/llama-3.2-3b-heretic"
)
```

### Programmatic Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load from local path
model = AutoModelForCausalLM.from_pretrained(
    "models/llama-3.2-3b-heretic",
    dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("models/llama-3.2-3b-heretic")

# Generate response
messages = [{"role": "user", "content": "Hello!"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Citation

```bibtex
@misc{arditi2024refusallanguagemodelsmediated,
      title={Refusal in Language Models Is Mediated by a Single Direction},
      author={Andy Arditi and Oscar Obeso and Aaquib Syed and Daniel Paleka and Nina Rimsky and Wes Gurnee and Neel Nanda},
      year={2024},
      eprint={2406.11717},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.11717},
}
```

## Experiments

Heretic includes an experiments framework for testing new behavioral directions:

```
experiments/
└── verbosity/          # Test extraction of "verbosity direction"
    ├── README.md       # Experiment documentation
    ├── concise_prompts.json
    ├── verbose_prompts.json
    ├── config.verbosity.toml
    └── eval_verbosity.py
```

See [ROADMAP.md](ROADMAP.md) for research directions and future plans.

## License

AGPL-3.0-or-later

## Resources

- [Paper: Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
- [Abliterated Models Collection](https://huggingface.co/collections/p-e-w/the-bestiary)
- [Project Roadmap](ROADMAP.md)
- [Vast.ai Workflow](WORKFLOW.md)
