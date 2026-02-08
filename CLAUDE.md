# CLAUDE.md

Guidance for Claude Code when working with this repository.

---

## Behavior Rules

### Rule 1: No destructive commands without confirmation
Destructive commands: `pkill`, `kill`, `rm -rf`, `git reset --hard`, `DROP TABLE`, stopping instances.
Before running any: state what it does, state it is irreversible, ask "Should I proceed?", wait for explicit "yes".

### Rule 2: Ambiguous requests -- ask first
"Complete the training" / "Finish the job" / "End the process" could mean STOP or LET IT FINISH. Always clarify.

### Rule 3: Trust user observations over system data
When user says "I see X" and your data says "Y", investigate the discrepancy. Never say "it's fine".

### Rule 4: Repeated concerns = stop and investigate
If user raises the same concern twice, stop, do not repeat "it's fine", investigate from their perspective.

### Rule 5: Check disk space before cloud instances
- 32B models: minimum 200GB disk
- 70B models: minimum 300GB disk
- Formula: `(model_size * 2) + 20GB`

### Rule 6: Never create/stop/terminate instances without permission

### Rule 7: When user says STOP, stop immediately

---

## Pre-Action Checklist

```
1. What does the user ACTUALLY want?
2. What do I ALREADY have? (conversation, local files, downloaded models)
3. Do existing tools handle this? (bruno-vast CLI)
4. Can local hardware do this? (RTX 4070, 8GB VRAM - use 4-bit quantization)
5. Is cloud NECESSARY? (Default answer: NO)
6. What's the SIMPLEST approach?
```

| Trigger | Wrong | Right |
|---------|-------|-------|
| "Test model" | Start Vast.ai | Check if results exist locally |
| "Need GPU" | Rent cloud GPU | Check local 4070 first |
| "Error occurred" | Try workaround | Fix the actual error |
| "Compare models" | Load both at once | Sequential with memory clearing |
| User instruction | Treat as suggestion | Treat as hard constraint |

---

## Working Commands

Use the Python CLI, not PowerShell scripts:

```bash
uv run bruno-vast list
uv run bruno-vast create RTX_4090 4
uv run bruno-vast setup
uv run bruno-vast watch
uv run bruno-vast stop
```

Fallback if `bruno-vast create` finds no GPUs:
```bash
vastai search offers "gpu_name=H200 disk_space>=200 rentable=true" --order dph_total
vastai create instance <OFFER_ID> --disk 200 --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
```

Git: always push to `fork`:
```bash
git push fork master
```

---

## Project Overview

Neural behavior modification for language models using contrastive activation analysis and orthogonalization. Extracts behavioral direction vectors and removes them via Optuna-optimized orthogonalization. See [README.md](README.md) for features and usage.

## Dependencies

- PyTorch >= 2.8.0 (CUDA 12.8, backward compatible with 12.1+). Patches known RCE and DoS vulnerabilities in earlier versions.
- Python 3.10-3.12 (not 3.13 - torch wheel availability)
- transformers >= 4.48.2 (Moonlight requires 4.48.2 exactly)

```bash
uv sync --all-extras --dev
```

## Build Commands

```bash
uv run bruno <model-name>          # Run
uv run ruff format .                # Format
uv run ruff check --extend-select I . # Lint
uv build                            # Build wheel
```

---

## Architecture

### Core Flow (`src/bruno/main.py`)

Modular pipeline in `src/bruno/phases/`:
1. **Dataset Loading** (`phases.dataset_loading`) - Load good/bad prompts, returns `DatasetBundle`
2. **Direction Extraction** (`phases.direction_extraction`) - Extract refusal/sacred directions, apply orthogonalization
3. **Optimization** (`phases.optimization`) - Optuna study with TPE sampler, warm-start, Pareto selection
4. **Model Saving** (`phases.model_saving`) - Local save and HuggingFace upload

### Key Components

**`Model` (`src/bruno/model.py`)**
- HuggingFace model wrapper with dtype fallback chain
- `get_layers()`: Handles text-only and multimodal architectures
- `get_layer_matrices()`: Extracts abliterable weights (attn.o_proj, mlp.down_proj), supports dense and MoE
- `abliterate()`: Orthogonalization with per-layer weight interpolation
- `get_residuals_batched()`: Hidden state extraction with CPU offload and batch size capping
- `generate()`: Supports `max_input_length` for selective input truncation
- Phase 1-7 methods: PCA, supervised probes, ensemble, calibration, concept cones, CAA, circuits
- GPU-accelerated probe training: batched PyTorch LBFGS (all layers simultaneously)
- Layer-wise weight caching: selective caching of abliterable components only

**`Evaluator` (`src/bruno/evaluator.py`)**
- KL divergence from first-token distributions
- Refusal counting with pre-compiled regex
- Returns `(kl_divergence, refusals)` tuple
- Early stopping (30 tokens for refusal detection), parallel evaluation

**`Settings` (`src/bruno/config.py`)**
- Pydantic-based: CLI args > env vars (`BRUNO_` prefix) > config.toml > defaults
- 50+ settings for phases 1-7, sacred directions, validation

**`transfer.py` (`src/bruno/transfer.py`)**
- Model family profiles (llama, qwen, mistral, gemma, phi) for warm-start

### Key Directories

- `src/bruno/` - Core logic
- `src/bruno/phases/` - Pipeline phases (dataset_loading, direction_extraction, optimization, model_saving)
- `src/bruno/constants.py` - Centralized thresholds and magic numbers
- `src/bruno/exceptions.py` - 22+ custom exception types
- `src/bruno/vast.py` - Vast.ai cloud CLI
- `experiments/` - Custom behavioral direction experiments
- `models/` - Local modified models (gitignored)

---

## Configuration

Priority: CLI > environment (`BRUNO_` prefix) > config.toml > defaults

Key settings: `model`, `n_trials` (200), `batch_size` (0=auto), `dtypes`, `compile`, `storage`, `study_name`, `refusal_check_tokens` (30), `cache_weights` (true), `residual_max_tokens` (512).

Phase features (all enabled by default except `use_circuit_ablation`): neural refusal detection, ensemble probes, activation calibration, concept cones, CAA, MPOA, warm-start, validation.

Sacred direction preservation (disabled by default): `use_sacred_directions`, `n_sacred_directions`, `sacred_prompts`.

See [configs/](configs/) for examples. Use `bruno show-config` to verify.

## CLI Flags

| Flag | Notes |
|------|-------|
| `--auto-select` | Requires boolean value: `--auto-select true` |
| `--compile` | torch.compile() for inference speedup |
| `--storage` | Optuna storage URL for resume |
| `--study-name` | Optuna study name |
| `--refusal-check-tokens` | Tokens for refusal detection (default: 30) |
| `--cache-weights` | In-memory weight caching (default: true) |
| `--unhelpfulness-prompts.config` | C4 dataset config (required: `"en"`). CLI overrides TOML |
| `--orthogonalize-directions` | Helpfulness direction orthogonalization (default: true) |

---

## Conventions

- Format/lint: `ruff format .` and `ruff check --extend-select I .`
- Use comprehensive type hints
- Use custom exception classes, not bare exceptions
- Use `logging` module, not `print`
- Use `dtype`, not `torch_dtype`
- No type casting to `any`
- No emojis or unicode symbols in output
- CSS: use Gradio CSS variables for theme compatibility

---

## Chat Interface (`chat_app.py`)

- `ModelManager` class: model loading, caching, streaming generation
- Auto-discovers models in `models/` with validation (config.json, weights, tokenizer)
- Streaming via `TextIteratorStreamer`
- GPU memory monitoring, chat history persistence
- Use `apply_chat_template(tokenize=True, return_tensors="pt")` for cross-model compatibility
- Ensure message content is string (not None) for Qwen compatibility

---

## Gotchas

### Local Hardware
User has RTX 4070 with 8GB VRAM. 7B models need 4-bit quantization. Cannot load two 7B models simultaneously.

### PyTorch CUDA with uv
`uv run` may revert to CPU-only torch via lockfile. Fix: use system Python directly, or `uv pip install --reinstall torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121`.

### Tokenizer Issues
Bruno saves `extra_special_tokens` as list instead of dict. Fix:
```python
import json; p='models/MODEL/tokenizer_config.json'; c=json.load(open(p,encoding='utf-8')); c['extra_special_tokens']={}; json.dump(c,open(p,'w',encoding='utf-8'),indent=2)
```

### Dataset Loading
Bruno uses `load_dataset()`, not `load_from_disk()`. `DatasetSpecification` supports: `dataset`, `split`, `column` -- no `data_files`.

### Gated Models
Llama models are gated (require `HF_TOKEN`). Qwen models are not.

### GPU OOM on 32B+ Models
Layer-wise caching (v1.2.0+) reduces cache memory by 55-75%. Works on H200 141GB (model 64GB + cache 28GB = 92GB). Does NOT fit on H100 80GB with caching -- use `--cache-weights false` on H100.

### GPU OOM on 14B+ Models During Residual Extraction
`output_hidden_states=True` stores O(n_layers * batch * seq_len * hidden_dim) memory. For 14B (50 layers, 5120 hidden_dim), this can be 33-67GB per batch with long inputs. Fixed by:
- `residual_max_tokens` setting (default 512) truncates inputs during residual extraction
- Residual batch size capped at min(batch_size, 16) regardless of auto-detected value
- Immediate `del outputs` + `.clone()` to break tensor views + `.cpu()` offload between batches
- Auto batch_size calibrates on `get_responses()` which does NOT use hidden states -- the auto-detected value is too high for `get_residuals()`. This is handled automatically now.

### Probe Training Performance
Supervised probe training uses GPU-accelerated batched PyTorch LBFGS when CUDA is available. Trains all layer probes simultaneously. For 14B (49 layers, 5120 hidden_dim): ~seconds on GPU vs ~40 minutes on CPU with sklearn. Falls back to sklearn + joblib on CPU-only systems.

### SCP Through Vast.ai
SCP can silently fail through Vast.ai's SSH proxy. Use pipe-through-SSH instead:
```bash
cat local_file.py | ssh -p PORT root@HOST "cat > /path/to/dest.py"
```
Verify with `grep -c 'pattern' /path/to/dest.py` after transfer.

### Circuit Ablation
GQA models (Llama 3.x, Qwen2.5, Moonlight) are not compatible. Use `--use-circuit-ablation false`.

### Feature Fallback Behavior
CAA extraction, supervised probing, concept cones, and warm-start may fail for certain models. Bruno handles each gracefully: prints a warning, continues without the feature. Manual override flags: `--use-caa false`, `--ensemble-probe-pca false`, `--use-concept-cones false`, `--model-family qwen`.

### Moonlight MoE Memory
Moonlight-16B is 63GB in BF16 despite "16B" name (MoE with 3B active). Needs ~88GB total. Requires H200 141GB. Also requires `tiktoken` and `transformers==4.48.2`.

See [docs/INFERENCE_DEPLOYMENT.md](docs/INFERENCE_DEPLOYMENT.md) for HuggingFace endpoint deployment.

### C4 Dataset Config
`ValueError: Config name is missing` -- C4 requires explicit config. When using CLI for unhelpfulness_prompts, pass ALL four fields:
```bash
bruno --model MODEL \
  --unhelpfulness-prompts.dataset allenai/c4 \
  --unhelpfulness-prompts.config en \
  --unhelpfulness-prompts.split "train[:200]" \
  --unhelpfulness-prompts.column text
```
C4 streams automatically since v1.1.0 (no disk overhead).

### Pydantic Settings Priority
CLI default values (including `None`) override TOML config. Either pass ALL settings via CLI, or ensure no CLI defaults conflict with TOML.

### unhelpfulness_prompts CLI Fields
Passing ANY `--unhelpfulness-prompts.*` field via CLI requires ALL four fields (`dataset`, `config`, `split`, `column`).

### Instance Stop vs Process Kill
Stopping a Vast.ai instance kills running processes. Check `bruno-vast progress` for "Models Saved" before stopping.

### Package Data in Wheel
Config files must be in `src/<package>/` directory. Verify: `unzip -l dist/*.whl | grep toml`.

### pip install --force-reinstall
Breaks environment. Use `--upgrade` instead. If broken: `pip install transformers>=4.55.2`.

### CrewAI Multi-Agent Swarm
See `examples/seven_agent_swarm.py` for reference implementation.
- Ollama safetensors import is broken -- always convert to GGUF first via `tools/llama.cpp/convert_hf_to_gguf.py`
- Modelfiles need `num_ctx 8192` (4096 too small for CrewAI system prompts), `num_predict 2048` (3B models repeat without cap)
- `timeout=1200` in LLM config (model loading + generation can exceed 600s)
- `max_iter=10` on agents (14B orchestrator may retry delegation format multiple times)
- Set `CREWAI_TRACING_ENABLED=false` to avoid interactive tracing prompt in non-interactive mode
- Ollama multi-model: set `OLLAMA_MAX_LOADED_MODELS=3`, `OLLAMA_KEEP_ALIVE=30m` before starting Ollama
- CrewAI hierarchical mode: `manager_agent` must NOT be in the `agents` list (ValidationError)
- 14B orchestrator sometimes batch-delegates as JSON array instead of single tool call -- `max_iter=10` tolerates retries

---

## Stopping Vast.ai Instances

Verify you're stopping the correct instance:
```bash
vastai show instances --raw | python -c "import sys,json; data=json.load(sys.stdin); [print(f'ID: {i[\"id\"]}, Status: {i.get(\"actual_status\")}, SSH: {i.get(\"ssh_port\")}') for i in data]"
vastai stop instance <CORRECT_ID>
vastai show instances
```
Do not blindly stop the first instance. Multiple instances may exist. Do not judge instance status by uptime -- Vast.ai shows accumulated uptime from all users.

## Checking Training Progress

Do not trust Optuna values directly -- they may be normalized (0-1), not raw counts.

Get actual refusal counts from logs:
```bash
ssh -p PORT root@HOST "grep -E 'Refusals:' /workspace/ACTIVE_LOG.log | tail -30"
```
This shows raw counts like "Refusals: 79/104". Always use this format, not Optuna values.
