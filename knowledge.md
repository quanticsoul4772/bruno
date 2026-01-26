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
- Things to avoid: No emojis or unicode symbols in output, no `torch_dtype` (use `dtype`)

## Key Files
- `chat_app.py` - Gradio chat interface for abliterated models
- `runpod.ps1` - PowerShell automation for cloud GPU deployment
- `config.default.toml` - Default configuration template
