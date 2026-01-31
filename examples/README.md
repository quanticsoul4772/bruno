# Heretic Examples

Example applications and utilities built with heretic.

## Available Examples

### chat_app.py

Interactive chat interface using Gradio for testing abliterated models.

**Features:**
- Auto-discovers models in `models/` directory
- Validates model files before loading (config.json, weights, tokenizer)
- Streaming token generation
- Real-time GPU memory monitoring
- Chat history persistence to `chat_history/` as JSON

**Usage:**

```bash
# Run locally (requires model in models/ directory)
uv run python examples/chat_app.py

# Or via Gradio interface
uv run python examples/chat_app.py --share
```

**Requirements:**
- Model files in `models/` directory
- GPU with sufficient VRAM (8GB for 7B with 4-bit quantization)
- transformers, torch, gradio

**Architecture:**
- `ModelManager` class handles model loading and caching
- Custom exception hierarchy for error handling
- Structured logging via Python's `logging` module

See chat_app.py source code for detailed implementation notes.
