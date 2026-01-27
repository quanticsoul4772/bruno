# Project knowledge

This file gives Codebuff context about your project: goals, commands, conventions, and gotchas.

## Vision

Heretic is a **neural behavior modification workbench**, not just an "uncensoring tool." The abliteration technique is general and can extract/modify any behavioral direction encoded in model weights - verbosity, hedging, sycophancy, and more. See [ROADMAP.md](ROADMAP.md) for full vision.

## Quickstart
- Setup: `uv sync --all-extras --dev`
- Dev: `uv run heretic <model-name>`
- Test: No test suite - CI runs `ruff format` and `ruff check`
- Chat UI: `python chat_app.py`
- Cloud (Vast.ai): `heretic-vast create A100_80GB 2 && heretic-vast setup && heretic-vast run MODEL`

## Architecture
- Key directories:
  - `src/heretic/` - Core abliteration logic
  - `src/heretic/vast.py` - Vast.ai cloud CLI
  - `experiments/` - Behavioral direction experiments
  - `models/` - Local modified models (gitignored)
  - `chat_history/` - Saved chat sessions (gitignored)
- Data flow: Load model -> Extract behavioral directions -> Optuna optimization -> Save/upload

## Key Components
- `heretic` - Main CLI for abliteration
- `heretic-vast` - Vast.ai GPU cloud management CLI (Rich dashboard, Fabric SSH)
- `chat_app.py` - Gradio chat interface for testing models
- `experiments/verbosity/` - Spike experiment for verbosity direction extraction

## Conventions
- Formatting/linting: `ruff format .` and `ruff check --extend-select I .`
- Patterns to follow: Use existing Model/Evaluator/Settings classes
- Things to avoid:
  - No emojis or unicode symbols in output
  - No `torch_dtype` (use `dtype`)
  - No type casting to `any` - keep proper types
  - No print statements - use `logging` module instead
- Type hints: Use comprehensive type hints for all functions/methods
- Error handling: Use custom exception classes with descriptive messages
- CSS: Use Gradio CSS variables (e.g., `--body-text-color`) for theme compatibility

## Key Files
- `chat_app.py` - Gradio chat interface for modified models
  - Uses `ModelManager` class for model loading/caching
  - Custom exceptions: `HereticError`, `ModelNotFoundError`, `CUDAOutOfMemoryError`, etc.
  - Logger: `heretic_chat`
- `src/heretic/vast.py` - Vast.ai CLI
  - `VastConfig`: Configuration from env/.env
  - `GPU_TIERS`: Presets for RTX_4090, A6000, A100_40GB, A100_80GB, H100
  - Uses Fabric for SSH, Rich for terminal UI
- `runpod.ps1` - PowerShell automation for RunPod deployment
- `config.default.toml` - Default configuration template

## Current Experiments
- `experiments/verbosity/` - Test extraction of "verbosity direction"
  - Uses naturally verbose vs concise prompts (not explicit instructions)
  - Goal: Validate that behavioral directions beyond refusal can be extracted
  - **STATUS: COMPLETE** - Spike experiment finished successfully
  - Run with: `python experiments/verbosity/load_local_dataset.py` then `heretic --model MODEL --auto-select true`

### Verbosity Spike Results (Qwen 7B)

| Metric | Value |
|--------|-------|
| Model | Qwen/Qwen2.5-7B-Instruct |
| Trials | 50 Optuna optimization trials |
| Datasets | 200 concise + 200 verbose prompts |
| Model Saved | `./models/Qwen2.5-7B-Instruct-heretic` (15.2 GB) |

**Test Results (RTX 4070, 4-bit quantization):**

| Prompt Type | Example | Words |
|-------------|---------|-------|
| Factual | "What is the capital of France?" | 6 |
| Factual | "What is 24 multiplied by 7?" | 8 |
| Factual | "How many continents are there?" | 30 |
| Open-ended | "What do you think about AI?" | 203 |
| Open-ended | "Should I learn Python or JavaScript?" | 195 |

**Conclusions:**
- ✅ **Factual questions**: Concise responses (6-30 words)
- ⚠️ **Open-ended questions**: Still verbose (195-203 words)
- The ablation appears to reduce verbosity for factual Q&A but doesn't affect elaboration on open-ended topics
- This is expected: open-ended questions naturally invite longer responses

## Chat App Patterns
- Model validation: Check for config.json, model weights, and tokenizer files
- GPU monitoring: Use `torch.cuda.memory_reserved()` for accurate usage
- Tokenization: Use `apply_chat_template(tokenize=True, return_tensors="pt")` for cross-model compatibility
- Message validation: Ensure all message content is string (not None) for Qwen compatibility

## Cloud Deployment
- **Vast.ai** (preferred for large models): Use `heretic-vast` CLI
- **RunPod**: Use `runpod.ps1` PowerShell script
- Model size guidelines:
  - 7B-8B: 1x RTX_4090 (24GB)
  - 14B-32B: 1x A100_40GB/80GB
  - 70B+: 2x A100_80GB or H100

### Downloading Models from Vast.ai
**TWO TOOLS AVAILABLE - USE THE RIGHT ONE:**

1. **PowerShell script (RECOMMENDED)**: `.\runpod.ps1 vast-download-model`
   - More robust SSH handling with multiple URL format support
   - Better progress display with rsync
   - Works reliably on Windows

2. **Python CLI**: `heretic-vast download`
   - May have SSH connection issues on Windows/WSL
   - If it fails, use PowerShell script instead
   - Now shows debugging info when SSH fails

## CRITICAL GOTCHAS (Lessons Learned)

### BEFORE TAKING ANY ACTION
1. **READ THIS FILE FIRST** - Check knowledge.md before every task
2. **Check what you already have** - Don't redo work that's already done
3. **Use existing tools** - We have scripts (runpod.ps1) and CLIs (heretic-vast) for a reason
4. **Don't start cloud instances unnecessarily** - Check if local resources can do the job

### Local Hardware
- **User has RTX 4070 with 8GB VRAM** - Use this for testing when possible
- **7B models need 4-bit quantization** to fit in 8GB (use BitsAndBytesConfig)
- **Cannot load two 7B models simultaneously** - Run comparisons sequentially with memory clearing
- **PyTorch CUDA issue with uv**: `uv run` uses lockfile which may revert to CPU-only torch
  - Solution 1: Use system Python directly: `python script.py` instead of `uv run python script.py`
  - Solution 2: Force reinstall: `uv pip install --reinstall torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121`

### Tokenizer Issues
- **Heretic saves tokenizer_config.json with `extra_special_tokens` as a list** instead of dict
- This causes transformers to fail when loading the model
- **Fix**: Convert to empty dict: `c['extra_special_tokens'] = {}`
- Python one-liner fix:
  ```python
  import json; p='models/MODEL/tokenizer_config.json'; c=json.load(open(p,encoding='utf-8')); c['extra_special_tokens']={}; json.dump(c,open(p,'w',encoding='utf-8'),indent=2)
  ```

### Heretic CLI Flags
- **`--auto-select` REQUIRES a boolean value**: Use `--auto-select true`, NOT `--auto-select`
- Example: `heretic --model Qwen/Qwen2.5-7B-Instruct --auto-select true --n-trials 20`

### Dataset Loading
- **heretic uses `load_dataset()` from HuggingFace**, NOT `load_from_disk()`
- `DatasetSpecification` only supports: `dataset`, `split`, `column` - NO `data_files` field
- For local datasets, you must either:
  1. Use HuggingFace Hub datasets (e.g., `mlabonne/harmless_alpaca`)
  2. Push local datasets to HuggingFace Hub first
  3. Modify heretic source to support `load_from_disk()` for local paths

### Gated Models
- **Llama models are gated** - require HuggingFace authentication
- Use `huggingface-cli login` or set `HF_TOKEN` environment variable
- **Qwen models are NOT gated** - use these for quick testing

### Vast.ai exec Command
- Complex shell commands with quotes/escaping often fail through `heretic-vast exec`
- Use **base64 encoding** for complex scripts:
  ```bash
  heretic-vast exec 'echo BASE64_ENCODED_SCRIPT | base64 -d > script.py && python script.py'
  ```
- Single quotes in the command get parsed incorrectly - avoid nested quotes

### Instance Stop vs Process Kill
- **Stopping a Vast.ai instance KILLS running processes** - no graceful save
- If abliteration hasn't saved the model yet, you lose ALL progress
- Check `heretic-vast progress` for "Models Saved" before stopping
