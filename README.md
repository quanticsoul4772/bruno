# Heretic - Automatic Censorship Removal for LLMs

Heretic is a tool for automatic censorship removal (abliteration) from language models using Optuna-based hyperparameter optimization.

## Quick Start

```bash
# Install
pip install heretic-llm

# Run abliteration
heretic Qwen/Qwen3-4B-Instruct-2507
```

## RunPod Deployment (Recommended for GPU)

For users without local GPU access, use our RunPod automation script:

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

See [WORKFLOW.md](WORKFLOW.md) for detailed instructions.

## Features

- **Automatic optimization** - Uses Optuna for multi-objective hyperparameter tuning
- **Pareto-optimal results** - Presents best trade-offs between capability preservation and censorship removal
- **Multi-GPU support** - Scales across available GPUs
- **HuggingFace integration** - Direct upload to your HF account

## Requirements

- Python >= 3.10
- CUDA-capable GPU (16GB+ VRAM recommended)
- Or use RunPod for cloud GPU access

## Installation

```bash
# From PyPI
pip install heretic-llm

# From source
git clone https://github.com/p-e-w/heretic
cd heretic
uv sync --all-extras --dev
uv run heretic <model-name>
```

## Usage

```bash
# Basic usage
heretic <model-name>

# With options
heretic --model <model-name> --config config.toml

# Examples
heretic Qwen/Qwen3-4B-Instruct-2507
heretic meta-llama/Llama-3.1-8B-Instruct
heretic mistralai/Mistral-7B-Instruct-v0.3
```

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

1. **Load model** - Loads the target model with automatic dtype selection
2. **Extract refusal directions** - Computes per-layer activation patterns from harmful vs. benign prompts
3. **Optimize ablation** - Uses Optuna to find optimal ablation parameters that minimize refusals while preserving model capabilities (measured by KL divergence)
4. **Select result** - Presents Pareto-optimal trials for user selection
5. **Export** - Save locally or upload to HuggingFace

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

## License

AGPL-3.0-or-later

## Resources

- [Paper](https://arxiv.org/abs/2406.11717)
- [Abliterated Models](https://huggingface.co/collections/p-e-w/the-bestiary)
- [Documentation](knowledge.md)
