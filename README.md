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

- **Automatic optimization** - Uses Optuna for multi-objective hyperparameter tuning
- **Early stopping** - MedianPruner stops unpromising trials early (30-40% time savings)
- **Pareto-optimal results** - Presents best trade-offs between capability preservation and behavior modification
- **Multi-GPU support** - Scales across available GPUs with `device_map="auto"`
- **HuggingFace integration** - Direct upload to your HF account
- **Fully automated mode** - `--auto-select` + `--hf-upload` for headless operation
- **Chat interface** - Gradio-based UI for interacting with modified models
- **Cloud CLI** - `heretic-vast` for Vast.ai GPU management with live dashboard
- **Experiments framework** - Infrastructure for testing new behavioral directions

## Requirements

- Python >= 3.10
- CUDA-capable GPU (16GB+ VRAM recommended)
- Or use Docker with `--gpus all`
- Or use RunPod/Vast.ai for cloud GPU access

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
| `--prune-trials/--no-prune-trials` | Enable/disable early stopping (default: enabled) |

## Configuration

Copy `config.default.toml` to `config.toml` and customize:

```toml
[model]
dtype = "auto"          # float16, bfloat16, or auto
batch_size = 0          # 0 = auto-detect

[optimization]
n_trials = 100          # Number of Optuna trials
kl_threshold = 1.0      # Max acceptable KL divergence
```

## How It Works

1. **Load model** - Loads the target model with automatic dtype selection and multi-GPU distribution
2. **Extract behavioral directions** - Computes per-layer activation patterns from contrastive prompt pairs
3. **Optimize modification** - Uses Optuna to find optimal parameters that modify behavior while preserving capabilities (measured by KL divergence)
4. **Select result** - Presents Pareto-optimal trials for user selection
5. **Export** - Save locally or upload to HuggingFace

### The Core Technique

Heretic uses **activation direction analysis** to find and remove behavioral tendencies:

```
1. FIND direction    → Compare activations on contrastive prompts
2. PROJECT it out    → Orthogonalize weight matrices against that direction  
3. OPTIMIZE          → Find intensity that modifies behavior without destroying capability
```

This technique is general - it works for any behavior encoded as a direction in activation space.

## Chat Interface

Heretic includes a sophisticated chat interface for interacting with your abliterated models.

### Quick Start

```bash
# Install dependencies
pip install gradio transformers torch accelerate

# Run the chat app
python chat_app.py
```

This opens a web UI at `http://localhost:7860` with:

- **Model Selection** - Switch between abliterated models in the `models/` directory
- **Streaming Responses** - See tokens appear in real-time
- **GPU Memory Monitoring** - Real-time display of GPU usage (e.g., "GPU: 4.2/8.0 GB (52%)")
- **Chat History** - Save and load conversations as JSON files
- **Advanced Settings** - Adjustable temperature and max tokens
- **Model Validation** - Automatic validation of model files before loading
- **Clean Minimal UI** - Monochrome theme with responsive design

### Features

- **Type-safe implementation** with comprehensive type hints
- **Custom exception handling** for better error messages (CUDA OOM, model validation, etc.)
- **Structured logging** with configurable log levels
- **Cross-model compatibility** - Works with Llama, Qwen, and other architectures

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
