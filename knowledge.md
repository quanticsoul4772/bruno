# Project knowledge

This file gives Codebuff context about your project: goals, commands, conventions, and gotchas.

## Quickstart
- Setup: `uv sync --all-extras --dev`
- Dev: `uv run heretic <model-name>`
- Test: No test suite - CI runs `ruff format` and `ruff check`
- Chat UI: `python chat_app.py`

## Architecture
- Key directories:
  - `src/heretic/` - Core abliteration logic
  - `models/` - Local abliterated models (gitignored)
  - `chat_history/` - Saved chat sessions (gitignored)
- Data flow: Load model -> Extract refusal directions -> Optuna optimization -> Save/upload

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
- `chat_app.py` - Gradio chat interface for abliterated models
  - Uses `ModelManager` class for model loading/caching
  - Custom exceptions: `HereticError`, `ModelNotFoundError`, `CUDAOutOfMemoryError`, etc.
  - Logger: `heretic_chat`
- `runpod.ps1` - PowerShell automation for cloud GPU deployment
- `config.default.toml` - Default configuration template

## Chat App Patterns
- Model validation: Check for config.json, model weights, and tokenizer files
- GPU monitoring: Use `torch.cuda.memory_reserved()` for accurate usage
- Tokenization: Use `apply_chat_template(tokenize=True, return_tensors="pt")` for cross-model compatibility
- Message validation: Ensure all message content is string (not None) for Qwen compatibility
