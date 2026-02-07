# Bruno

[![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/badge/pypi-bruno--ai-blue.svg)](https://pypi.org/project/bruno-ai/)

Neural behavior modification for language models using contrastive activation analysis and orthogonalization.

Named after Giordano Bruno (1548-1600).

## What It Does

Bruno extracts behavioral direction vectors from model activations and removes them via orthogonalization. It optimizes the process automatically using Optuna, balancing behavior removal against capability preservation. Works for refusal removal ("abliteration"), verbosity reduction, hedging removal, and other behavioral patterns.

## Features

- Multi-objective Optuna optimization with TPE sampler and Pareto-optimal trial selection
- Resume support via SQLite storage
- Neural refusal detection using zero-shot NLI
- Supervised probing and PCA ensemble for direction extraction
- Activation-based calibration for adaptive weight scaling
- Concept cone extraction for category-specific ablation
- Contrastive Activation Addition (CAA)
- Circuit-level ablation targeting specific attention heads (non-GQA models)
- Warm-start parameter transfer using model family profiles
- Sacred direction preservation via MMLU orthogonalization
- Multi-Prompt Orthogonal Ablation (MPOA)
- MoE architecture support with shared expert and router-aware targeting
- GPU-accelerated PCA extraction
- GPU-accelerated probe training (batched PyTorch LBFGS across all layers simultaneously)
- Bounded residual extraction (`residual_max_tokens`) for 14B+ models
- GGUF conversion via llama.cpp for Ollama deployment
- Layer-wise weight caching
- torch.compile() support
- Early stopping for refusal detection
- Parallel KL divergence and refusal evaluation
- C4 dataset streaming (no disk overhead)
- HuggingFace Hub upload
- Interactive chat for testing modifications
- `bruno-vast` CLI for Vast.ai GPU management with live dashboard
- Docker image for cloud deployment
- Configuration verification (`bruno show-config`)
- Custom exception hierarchy with actionable error messages
- Multi-agent swarm support with CrewAI (see [examples/seven_agent_swarm.py](examples/seven_agent_swarm.py))

## Installation

```bash
# From PyPI
pip install bruno-ai

# From source
git clone https://github.com/p-e-w/bruno
cd bruno
uv sync --all-extras --dev

# Docker
docker run --gpus all -it quanticsoul4772/bruno bruno --help
```

## Quick Start

```bash
# Interactive mode
bruno Qwen/Qwen2.5-7B-Instruct

# Automated with upload
bruno Qwen/Qwen2.5-7B-Instruct --auto-select --hf-upload username/model-bruno

# Cloud GPU
bruno-vast create A100_80GB 1
bruno-vast setup
bruno-vast run Qwen/Qwen2.5-32B-Instruct
bruno-vast watch
bruno-vast stop
```

## Configuration

Priority: CLI arguments > environment variables (`BRUNO_` prefix) > `config.toml` > defaults.

```bash
bruno <model> \
  --n-trials 200 \
  --auto-select \
  --storage sqlite:///study.db \
  --compile
```

See [configs/](configs/) for example configuration files. Use `bruno show-config` to verify effective settings.

## Model Size Guidelines

| Size | GPU VRAM | Disk  | Notes |
|------|----------|-------|-------|
| 3B   | 8GB      | 50GB  | Specialist agents, fast abliteration |
| 7B   | 24GB     | 100GB | General purpose |
| 14B  | 80GB     | 150GB | Requires `residual_max_tokens=512` and batch size cap |
| 32B  | 80GB+    | 200GB | C4 config required (`--unhelpfulness-prompts.config en`) |
| 70B  | 140GB+   | 400GB | C4 config required |

14B+ models generate ~33GB of hidden states during residual extraction (`output_hidden_states=True` stores all transformer layers). Bruno automatically caps residual batch size to 16 and truncates inputs to `residual_max_tokens` (default 512) to bound memory.

## Documentation

- [WORKFLOW.md](WORKFLOW.md) - Cloud GPU deployment
- [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) - Command cheatsheet
- [LESSONS_LEARNED.md](LESSONS_LEARNED.md) - Troubleshooting
- [ROADMAP.md](ROADMAP.md) - Future directions
- [AI_AGENT_SWARM_RESEARCH.md](docs/AI_AGENT_SWARM_RESEARCH.md) - Multi-agent swarm architecture
- [FEATURE_ROADMAP.md](docs/FEATURE_ROADMAP.md) - Feature status and plans
- [experiments/](experiments/) - Custom behavioral directions
- [examples/](examples/) - Example applications and multi-agent swarm
- [scripts/](scripts/) - Utility scripts

## Multi-Agent Swarm

Bruno can abliterate multiple specialist models that work together as a development team via CrewAI and Ollama. The architecture uses a 14B orchestrator directing 6x 3B specialists:

| Agent | Model | Role |
|-------|-------|------|
| Orchestrator | Qwen2.5-Coder-14B | Task decomposition and coordination |
| Frontend | Qwen2.5-Coder-3B | UI/UX implementation |
| Backend | Qwen2.5-Coder-3B | API and server logic |
| Test | Qwen2.5-Coder-3B | Testing and QA |
| Security | Qwen2.5-Coder-3B | Security review and hardening |
| Docs | Qwen2.5-Coder-3B | Documentation generation |
| DevOps | Qwen2.5-Coder-3B | CI/CD and deployment |

```bash
# Convert abliterated models to GGUF for Ollama
python -m llama_cpp.convert_hf_to_gguf models/Frontend-3B/ --outtype f16

# Deploy to Ollama
ollama create frontend-3b -f Modelfile.frontend

# Run the swarm
python examples/seven_agent_swarm.py --task "Build a REST API"

# Flat mode (no orchestrator, 3B agents only)
python examples/seven_agent_swarm.py --task "Build a REST API" --flat
```

See [docs/AI_AGENT_SWARM_RESEARCH.md](docs/AI_AGENT_SWARM_RESEARCH.md) for architecture details.

## Development

```bash
uv sync --all-extras --dev
uv run ruff format .
uv run ruff check --extend-select I .
uv run pytest
uv build
```

## Published Models

- [rawcell/Moonlight-16B-A3B-Instruct-bruno](https://huggingface.co/rawcell/Moonlight-16B-A3B-Instruct-bruno) - MoE abliteration (MMLU 48.7%, HellaSwag 58.0%, GSM8K 55.0%)
- [rawcell/Qwen2.5-Coder-7B-Instruct-bruno](https://huggingface.co/rawcell/Qwen2.5-Coder-7B-Instruct-bruno) - Code generation, abliterated
- 6x Qwen2.5-Coder-3B specialist agents (Frontend, Backend, Test, Security, Docs, DevOps) - GGUF format for Ollama

## Resources

- [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
- [Abliterated Models Collection](https://huggingface.co/collections/p-e-w/the-bestiary)
- [Original Project](https://github.com/p-e-w/bruno) by Philipp Emanuel Weidmann
- [This Fork](https://github.com/quanticsoul4772/abliteration-workflow)

## License

AGPL-3.0-or-later. Copyright (C) 2024-2026 Philipp Emanuel Weidmann.
